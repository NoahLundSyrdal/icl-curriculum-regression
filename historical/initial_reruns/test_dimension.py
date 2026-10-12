"""Scientific validity checks for exact-batch curriculum ordering."""
import dataclasses
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import torch
import run_experiment as base
import dimension_experiment as exp
import summarize_dimension as summary


def equal(test,a,b):
    if isinstance(a,torch.Tensor):test.assertTrue(torch.equal(a.cpu(),b.cpu()))
    elif isinstance(a,dict):
        test.assertEqual(set(a),set(b))
        for k in a:equal(test,a[k],b[k])
    elif isinstance(a,(list,tuple)):
        test.assertEqual(len(a),len(b))
        for x,y in zip(a,b):equal(test,x,y)
    else:test.assertEqual(a,b)


class DimensionChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.cfg=base.preset_config("smoke",mode="clean",seed=8,dims=3,points=7,
            embedding=16,layers=1,heads=2,steps=8,batch_size=4,microbatch=2,
            eval_episodes=4,eval_microbatch=2,bayes_reference=False)
        self.study=exp.Study(initial_dims=1,easy_steps=1,prefix_steps=4,
            evaluation_steps=(0,2,4,6,8),checkpoint_every=2,log_every=2)

    def tearDown(self):self.temp.cleanup()

    def run_arm(self,name,arm="curriculum",**kw):
        study=dataclasses.replace(self.study,arm=arm)
        return exp.train(self.cfg,study,self.root/name,torch.device("cpu"),**kw)

    def state(self,name):return torch.load(self.root/name/"checkpoint.pt",weights_only=True)

    def states_match(self,a,b):
        for key in ("model","optimizer","torch_rng","cuda_rng","python_rng"):
            equal(self,a[key],b[key])

    def test_exact_same_batches_under_permutation_and_common_ending(self):
        c,s=self.study,dataclasses.replace(self.study,arm="shuffled")
        left,right=exp.batch_order(self.cfg,c),exp.batch_order(self.cfg,s)
        self.assertTrue(torch.equal(left[:4].sort().values,right[:4].sort().values))
        self.assertTrue(torch.equal(left[4:],right[4:]))
        for batch_id in left.tolist():
            equal(self,exp.batch(self.cfg,c,batch_id),exp.batch(self.cfg,s,batch_id))
            if batch_id>=4:
                equal(self,exp.batch(self.cfg,c,batch_id),exp.batch(self.cfg,dataclasses.replace(c,arm="hard"),batch_id))
        expected=[1,1,2,3,3,3,3,3]
        self.assertEqual([exp.canonical_dims(self.cfg,c,i) for i in left.tolist()],expected)

    def test_mask_and_target_respect_active_dimension_and_fixed_physical_shape(self):
        xs,y,d,p=exp.batch(self.cfg,self.study,0)
        self.assertEqual(tuple(xs.shape),(4,7,3));self.assertEqual((d,p),(1,3))
        self.assertEqual(float(xs[:,:,1:].abs().sum()),0)
        _,_,full,positions=exp.batch(self.cfg,dataclasses.replace(self.study,arm="hard"),0)
        self.assertEqual((full,positions),(3,7))

    def test_future_inputs_and_labels_cannot_affect_eligible_predictions(self):
        base.seed_model(self.cfg.seed);model=base.RegressionTransformer(self.cfg).eval()
        xs,y,d,p=exp.batch(self.cfg,self.study,0)
        with torch.no_grad():before=model(xs,y)
        xs[:,p:]+=1000;y[:,p:]-=1000
        with torch.no_grad():after=model(xs,y)
        self.assertTrue(torch.equal(before[:,:p],after[:,:p]))

    def test_eval_full_dimension_and_rng_isolation(self):
        self.run_arm("partial",max_steps_this_run=1)
        saved=self.state("partial");model=base.RegressionTransformer(self.cfg)
        model.load_state_dict(saved["model"]);model.train()
        before=exp.capture_rng();original=base.clean_target_losses
        def noisy(*a,**kw):
            torch.rand(2);random.random();return original(*a,**kw)
        with mock.patch.object(base,"clean_target_losses",side_effect=noisy):
            metric=exp.evaluate(self.cfg,self.study,saved["identity"],model,1,
                saved["training_seconds"],self.root/"partial",torch.device("cpu"))
        equal(self,before,exp.capture_rng());self.assertTrue(model.training)
        self.assertEqual((metric["evaluation_dimension"],metric["evaluation_positions"]),(3,7))
        self.assertEqual((metric["k_start"],metric["k_end"]),(3,6))
        self.assertEqual(len(metric["conditions"]["gaussian_sd0"]["mean_nmse_by_k"]),7)
        xs,clean,eps=exp.evaluation_bank(self.cfg)
        self.assertTrue(bool((xs[:,:,1:]!=0).any()))

    def test_resume_bitwise_equal_all_three_arms(self):
        for arm in exp.ARMS:
            self.assertEqual(self.run_arm(arm+"whole",arm),8)
            self.assertEqual(self.run_arm(arm+"split",arm,max_steps_this_run=3),3)
            self.assertEqual(self.run_arm(arm+"split",arm,resume=True),8)
            self.states_match(self.state(arm+"whole"),self.state(arm+"split"))

    def test_recover_boundary_snapshot_and_interrupted_eval_without_new_updates(self):
        self.run_arm("boundary",max_steps_this_run=4)
        state=self.state("boundary")
        (self.root/"boundary/checkpoint_step_000004.pt").unlink()
        (self.root/"boundary/evaluations/step_000004/metrics.json").unlink()
        self.run_arm("boundary",resume=True,max_steps_this_run=1)
        self.assertTrue((self.root/"boundary/checkpoint_step_000004.pt").exists())
        metric=json.loads((self.root/"boundary/evaluations/step_000004/metrics.json").read_text())
        self.assertEqual(metric["checkpoint_sha256"],exp.sha(self.root/"boundary/checkpoint_step_000004.pt"))
        self.assertEqual(metric["completed_steps"],4)

    def test_initialization_identical_and_intermediate_evals_do_not_change_training(self):
        fingerprints=[]
        for arm in exp.ARMS:
            self.run_arm(arm,arm,max_steps_this_run=1)
            fingerprints.append(self.state(arm)["identity"]["initial_model_sha256"])
        self.assertEqual(len(set(fingerprints)),1)
        self.run_arm("dense")
        sparse=dataclasses.replace(self.study,evaluation_steps=(0,4,8))
        exp.train(self.cfg,sparse,self.root/"sparse",torch.device("cpu"))
        self.states_match(self.state("dense"),self.state("sparse"))

    def test_changed_config_or_arm_rejected_on_resume(self):
        self.run_arm("partial",max_steps_this_run=2)
        with self.assertRaisesRegex(ValueError,"identity"):
            self.run_arm("partial","shuffled",resume=True)
        with self.assertRaisesRegex(ValueError,"identity"):
            exp.train(dataclasses.replace(self.cfg,learning_rate=.1),self.study,self.root/"partial",torch.device("cpu"),resume=True)


class SummaryChecks(unittest.TestCase):
    def test_auc_constant_linear_and_nonuniform_times(self):
        self.assertEqual(summary.trapezoidal_mean([0,1,4],[3,3,3]),3)
        self.assertEqual(summary.trapezoidal_mean([0,1,4],[0,1,4]),2)
        with self.assertRaises(ValueError):summary.trapezoidal_mean([1,0],[1,2])
        with self.assertRaises(ValueError):summary.trapezoidal_mean([0,1],[float("nan"),2])


if __name__=="__main__":unittest.main()

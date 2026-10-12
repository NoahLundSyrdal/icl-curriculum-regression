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
import dimension_long_experiment as exp
import summarize_dimension_long as summary


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




class LongStudyChecks(unittest.TestCase):
    def test_fixed_budget_fresh_seeds_and_complete_common_tail(self):
        torch.set_num_threads(1)
        self.assertEqual(exp.SEEDS,(200,201,202,203,204))
        for seed in exp.SEEDS:
            cfg=exp.long_config(seed);c=exp.Study();s=dataclasses.replace(c,arm="shuffled")
            self.assertEqual((cfg.steps,cfg.batch_size,cfg.eval_seed,cfg.eval_episodes),(125000,256,273819,4096))
            self.assertEqual((c.easy_steps,c.prefix_steps),(1000,25000));exp.validate(cfg,c)
            left,right=exp.batch_order(cfg,c),exp.batch_order(cfg,s)
            self.assertTrue(torch.equal(left[:25000],right[:25000].sort().values))
            self.assertTrue(torch.equal(left[25000:],right[25000:]))
            self.assertEqual(exp.canonical_dims(cfg,c,999),5)
            self.assertEqual(exp.canonical_dims(cfg,c,24999),50)
            self.assertEqual(exp.canonical_dims(cfg,c,25000),50)
            self.assertEqual(exp.canonical_dims(cfg,c,124999),50)
    def test_paired_uncertainty_uses_training_seeds(self):
        c=summary.paired_contrast([0,0,0,0,0],[1,2,3,4,5])
        self.assertEqual(c["mean"],-3);self.assertEqual(c["curriculum_wins"],5)
        self.assertAlmostEqual(c["standard_error_across_training_seeds"],(2.5/5)**.5)
        self.assertAlmostEqual(c["ci95_t_df4"][1],-3+2.7764451051977987*(2.5/5)**.5)
        with self.assertRaises(ValueError):summary.paired_contrast([0,0],[1,1])
    def test_missing_rolling_checkpoint_rejected_on_resume(self):
        cfg=base.preset_config("smoke",mode="clean",seed=8,dims=3,points=7,embedding=16,layers=1,heads=2,steps=8,batch_size=4,microbatch=2,eval_episodes=4,eval_microbatch=2,bayes_reference=False)
        study=exp.Study(initial_dims=1,easy_steps=1,prefix_steps=4,evaluation_steps=(0,4,8),checkpoint_every=2,log_every=2)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);exp.train(cfg,study,root,torch.device("cpu"),max_steps_this_run=2)
            (root/"checkpoint.pt").unlink()
            with self.assertRaises(FileNotFoundError):exp.train(cfg,study,root,torch.device("cpu"),resume=True)

    def test_retained_all_scheduled_weights_and_holdout_bank_is_separate(self):
        self.assertEqual(len(exp.Study().evaluation_steps),12)
        self.assertNotEqual(exp.FINAL_HOLDOUT_SEED,exp.long_config(200).eval_seed)
        self.assertEqual(exp.PRIMARY_TIMES[0],exp.Study().prefix_steps)
        self.assertEqual(exp.PRIMARY_TIMES[-1],exp.long_config(200).steps)



class CohortSummaryChecks(unittest.TestCase):
    def test_complete_cohort_summary_and_missing_record_rejection(self):
        import os
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as td:
            previous=Path.cwd();os.chdir(td)
            try:
                with mock.patch.object(exp,"sha",return_value="fixture_hash"):
                    for seed in exp.SEEDS:
                        for arm in exp.ARMS:
                            cfg=exp.long_config(seed);study=exp.Study(arm=arm)
                            d=summary.ROOT/f"seed_{seed}"/arm;d.mkdir(parents=True)
                            manifest=exp.identity(cfg,study);manifest["initial_model_sha256"]=str(seed)
                            (d/"manifest.json").write_text(json.dumps(manifest))
                            for split,times in (("primary",study.evaluation_steps),("final_holdout",(cfg.steps,))):
                                for t in times:
                                    value=1.0 if t==0 else {"curriculum":.10,"shuffled":.20,"hard":.30}[arm]+(seed-200)*.001
                                    m={"identity":manifest,"completed_steps":t,"evaluation_dimension":50,"evaluation_positions":101,"eval_seed":cfg.eval_seed if split=="primary" else exp.FINAL_HOLDOUT_SEED,"eval_episodes":4096,"evaluation_split":split,"zero_predictor_late_nmse":1.,"checkpoint_sha256":"fixture_hash","training_seconds":float(t),"conditions":{c:{"mean_nmse_by_k":[value]*101,"late_context_nmse":value} for c in ("gaussian_sd0","gaussian_sd1","gaussian_sd2")}}
                                    p=d/("evaluations" if split=="primary" else "final_holdout")/f"step_{t:06d}";p.mkdir(parents=True)
                                    (p/"metrics.json").write_text(json.dumps(m))
                    def load_fixture(path,**kwargs):
                        path=Path(path);seed=int(path.parts[-3].split("_")[1]);arm=path.parts[-2];t=int(path.stem.split("_")[-1]);cfg=exp.long_config(seed);study=exp.Study(arm=arm)
                        manifest=json.loads((path.parent/"manifest.json").read_text())
                        return {"identity":manifest,"completed_steps":t,"batch_ids":exp.batch_order(cfg,study),"optimizer":{"state":{} if t==0 else {0:{"step":t}}}}
                    with mock.patch.object(summary.torch,"load",side_effect=load_fixture):summary.main()
                    result=json.loads((summary.ROOT/"summary.json").read_text())
                    self.assertEqual(result["evaluation_records"],195);self.assertEqual(result["frozen_checkpoints_verified"],180)
                    self.assertAlmostEqual(result["primary_contrast"]["mean"],-.1)
                    self.assertTrue(result["all_support_criteria_met"])
                    (summary.ROOT/"summary.json").unlink()
                    (summary.ROOT/"seed_204/hard/evaluations/step_125000/metrics.json").unlink()
                    with mock.patch.object(summary.torch,"load",side_effect=load_fixture):
                        with self.assertRaises(FileNotFoundError):summary.main()
                    self.assertFalse((summary.ROOT/"summary.json").exists())
            finally:os.chdir(previous)

if __name__=="__main__":unittest.main()

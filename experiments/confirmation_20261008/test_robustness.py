"""Scientific treatment, pairing, evaluation and restart invariants."""
import dataclasses
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock
import torch
import robustness_experiment as exp
import summarize_robustness as summary


def equal(test,a,b):
    if isinstance(a,torch.Tensor):test.assertTrue(torch.equal(a.cpu(),b.cpu()))
    elif isinstance(a,dict):
        test.assertEqual(set(a),set(b))
        for k in a:equal(test,a[k],b[k])
    elif isinstance(a,(list,tuple)):
        test.assertEqual(len(a),len(b))
        for x,y in zip(a,b):equal(test,x,y)
    else:test.assertEqual(a,b)


class ScientificChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.cfg=exp.base.preset_config('smoke',mode='clean',seed=300,dims=3,points=7,embedding=16,layers=1,heads=2,steps=8,batch_size=4,microbatch=2,eval_episodes=4,eval_microbatch=2,bayes_reference=False)
        self.study=exp.Study(initial_dims=1,easy_steps=1,prefix_steps=4,evaluation_steps=(0,2,4,6,8),checkpoint_every=2,log_every=2)
    def tearDown(self):self.temp.cleanup()
    def read(self,name):return torch.load(self.root/name/'checkpoint.pt',weights_only=True)
    def run_tiny(self,name,study=None,**kw):return exp.train(self.cfg,study or self.study,self.root/name,torch.device('cpu'),**kw)
    def test_fixed_plan_fresh_seeds_and_identical_optimizer_grid_for_each_arm(self):
        exp.verify_plan();tasks=exp.planned_tasks();self.assertEqual(len(tasks),44);self.assertEqual(exp.SEEDS,(300,301))
        self.assertEqual(len([t for t in tasks if t['cohort']=='optimizer_screen']),30)
        for seed in exp.SEEDS:
            for arm in exp.ARMS:
                subset=[t for t in tasks if t['cohort']=='optimizer_screen' and t['seed']==seed and t['arm']==arm]
                self.assertEqual([(t['learning_rate'],t['precision'],t['objective']) for t in subset],[(lr,p,o) for _,lr,p,o in exp.OPTIMIZERS])
        self.assertEqual(len({str(exp.task_directory(t)) for t in tasks}),44)
    def test_production_masks_boundaries_and_complete_shared_ending(self):
        for task in exp.planned_tasks():
            cfg,study=exp.task_config(task);ids=exp.batch_order(cfg,study)
            self.assertTrue(torch.equal(ids[:25000].sort().values,torch.arange(25000)))
            self.assertTrue(torch.equal(ids[25000:],torch.arange(25000,cfg.steps)))
            self.assertEqual(exp.canonical_dims(cfg,study,999),5);self.assertEqual(exp.canonical_dims(cfg,study,24999),50)
            for bid in (25000,cfg.steps-1):self.assertEqual(exp.treatment(cfg,study,bid),(50,101,1.0))
            self.assertEqual(cfg.eval_seed,315831);self.assertNotEqual(cfg.eval_seed,exp.FINAL_HOLDOUT_SEED)
    def test_component_factorial_and_target_variance_control(self):
        for component,expected in [('both',(1,3)),('dimension',(1,7)),('context',(3,3)),('none',(3,7))]:
            study=dataclasses.replace(self.study,component=component,arm='hard' if component=='none' else 'curriculum')
            xs,y,active,p=exp.batch(self.cfg,study,0);self.assertEqual((active,p),expected);self.assertEqual(tuple(xs.shape),(4,7,3))
            self.assertAlmostEqual(exp.treatment(self.cfg,study,0)[2]**2*active,3.)
            raw=dataclasses.replace(study,target_variance='raw');xr,yr,_,_=exp.batch(self.cfg,raw,0)
            equal(self,xs,xr);self.assertTrue(torch.allclose(y,yr*(3/active)**.5))
    def test_constant_target_variance_empirically_across_dimensions(self):
        cfg=dataclasses.replace(self.cfg,dims=50,points=101,batch_size=4096,microbatch=128,steps=50000)
        study=exp.Study();low=exp.batch(cfg,study,0)[1];high=exp.batch(cfg,study,25000)[1]
        self.assertLess(abs(float(low.square().mean())-50),4);self.assertLess(abs(float(high.square().mean())-50),4)
    def test_exact_realized_batches_labels_and_weights_under_permutation(self):
        for component in ('both','dimension','context'):
            c=dataclasses.replace(self.study,component=component);s=dataclasses.replace(c,arm='shuffled')
            ids=exp.batch_order(self.cfg,s);self.assertTrue(torch.equal(ids[:4].sort().values,torch.arange(4)))
            for bid in range(8):equal(self,exp.batch(self.cfg,c,bid),exp.batch(self.cfg,s,bid))
    def test_exact_shared_full_batches_across_all_treatments(self):
        ref=exp.batch(self.cfg,self.study,4)
        for component in ('both','dimension','context','none'):
            for scale in ('raw','constant'):
                study=dataclasses.replace(self.study,component=component,target_variance=scale,arm='hard' if component=='none' else 'curriculum')
                equal(self,ref,exp.batch(self.cfg,study,4))
    def test_causal_predictions_ignore_own_and_future_labels(self):
        exp.base.seed_model(300);model=exp.base.RegressionTransformer(self.cfg).eval();x,y,_,_=exp.batch(self.cfg,self.study,0)
        with torch.no_grad():p=model(x,y)
        for k in (0,2,6):
            xm=x.clone();ym=y.clone();xm[:,k+1:]+=1000;ym[:,k:]+=10000
            with torch.no_grad():q=model(xm,ym)
            equal(self,p[:,:k+1],q[:,:k+1])
    def test_loss_rescaling_changes_gradients_but_preserves_raw_error(self):
        p=torch.tensor([[1.,3.,9.]],requires_grad=True);y=torch.tensor([[0.,1.,0.]])
        raw,obj=exp.training_objective(p,y,2,3,self.study);self.assertEqual(float(raw),2.5);obj.backward();g=p.grad.clone();p.grad=None
        r,n=exp.training_objective(p,y,2,3,dataclasses.replace(self.study,objective='nmse'));self.assertEqual(float(r),2.5);n.backward()
        self.assertTrue(torch.allclose(p.grad,g/3));self.assertEqual(float(p.grad[0,2]),0.)
    def test_resume_bitwise_preserves_parameters_adam_rng_for_all_treatments(self):
        cases=[dataclasses.replace(self.study,arm=arm,component=component) for _,component,arm in exp.COMPONENTS]
        cases.append(dataclasses.replace(self.study,objective='nmse'))
        for i,study in enumerate(cases):
            self.run_tiny(f'w{i}',study);self.run_tiny(f's{i}',study,max_steps_this_run=3);self.run_tiny(f's{i}',study,resume=True)
            a,b=self.read(f'w{i}'),self.read(f's{i}')
            for key in ('model','optimizer','torch_rng','cuda_rng','python_rng'):equal(self,a[key],b[key])
    def test_evaluation_is_full_dim_and_rng_isolated(self):
        self.run_tiny('partial',max_steps_this_run=1);saved=self.read('partial');model=exp.base.RegressionTransformer(self.cfg);model.load_state_dict(saved['model']);model.train();rng=exp.capture_rng()
        original=exp.base.clean_target_losses
        def noisy(*a,**kw):torch.rand(2);random.random();return original(*a,**kw)
        with mock.patch.object(exp.base,'clean_target_losses',side_effect=noisy):m=exp.evaluate(self.cfg,self.study,saved['identity'],model,1,0,self.root/'partial',torch.device('cpu'))
        equal(self,rng,exp.capture_rng());self.assertTrue(model.training);self.assertEqual((m['evaluation_dimension'],m['evaluation_positions'],m['k_start'],m['k_end']),(3,7,3,6))
        self.assertTrue(bool((exp.evaluation_bank(self.cfg)[0][:,:,1:]!=0).any()))
    def test_changed_treatment_or_config_rejected_and_missing_checkpoint_rejected(self):
        self.run_tiny('part',max_steps_this_run=2)
        for study in (dataclasses.replace(self.study,target_variance='raw'),dataclasses.replace(self.study,component='dimension'),dataclasses.replace(self.study,objective='nmse')):
            with self.assertRaises(ValueError):self.run_tiny('part',study,resume=True)
        with self.assertRaises(ValueError):exp.train(dataclasses.replace(self.cfg,learning_rate=.1),self.study,self.root/'part',torch.device('cpu'),resume=True)
        (self.root/'part/checkpoint.pt').unlink()
        with self.assertRaises(FileNotFoundError):self.run_tiny('part',resume=True)
    def test_evaluation_frequency_does_not_change_weights(self):
        self.run_tiny('dense');self.run_tiny('sparse',dataclasses.replace(self.study,evaluation_steps=(0,4,8)))
        a,b=self.read('dense'),self.read('sparse')
        for key in ('model','optimizer','torch_rng','cuda_rng','python_rng'):equal(self,a[key],b[key])
    def test_retained_snapshots_and_schedule_track_eligible_positions(self):
        self.run_tiny('full');d=self.root/'full'
        for t in self.study.evaluation_steps:self.assertTrue((d/f'checkpoint_step_{t:06d}.pt').is_file())
        s=torch.load(d/'schedule.pt',weights_only=True);self.assertEqual(s['eligible_positions'].tolist(),[3,3,5,7,7,7,7,7])


class AnalysisChecks(unittest.TestCase):
    def test_trapezoid_and_descriptive_pairing(self):
        self.assertEqual(summary.trapezoidal_mean([0,1,4],[0,1,4]),2)
        self.assertEqual(summary.paired([.1,.2],[.5,.4])['seed_differences'],[-.4,-.2])
        self.assertNotIn('p_value',summary.paired([.1,.2],[.5,.4]))
        with self.assertRaises(ValueError):summary.trapezoidal_mean([0,1],[float('nan'),1])
        with self.assertRaises(ValueError):summary.paired([1],[2])
    def test_fixed_ranking_factorial_and_equal_budget_scale_comparison(self):
        data={};held={}
        for task in exp.planned_tasks():
            cfg,study=exp.task_config(task)
            value={'both_ordered':.1,'dimension_ordered':.3,'context_ordered':.4,'hard':.8}.get(task['variant'],.5)
            if task['cohort']=='optimizer_screen':value=.6 if task['variant'].startswith('baseline') else .5
            for t in study.evaluation_steps:data[task['index'],t]={'conditions':{c:{'late_context_nmse':value} for c in ('gaussian_sd0','gaussian_sd1','gaussian_sd2')}}
            held[task['index']]=data[task['index'],cfg.steps]
        r=summary.aggregate(data,held)
        for arm in exp.ARMS:self.assertEqual(r['optimizer_candidate_ranking'][arm]['recommended_setting_for_fresh_long_confirmation'],'lr_low')
        self.assertAlmostEqual(r['constant_scale_factorial_clean_auc_effects']['dimension_main_effect'][0],-.4)
        self.assertAlmostEqual(r['constant_scale_factorial_clean_auc_effects']['context_main_effect'][0],-.3)
        self.assertEqual(r['constant_vs_raw_matched_50000_update_comparison']['curriculum']['common_budget_steps'],50000)
        del data[0,50000]
        with self.assertRaises(KeyError):summary.aggregate(data,held)
    def test_metric_rejects_wrong_query_bank_and_curve(self):
        cfg,study=exp.task_config(exp.planned_tasks()[0]);manifest={'id':'fixture'}
        m=dict(identity=manifest,completed_steps=50000,evaluation_dimension=50,evaluation_positions=101,eval_seed=315831,eval_episodes=4096,evaluation_split='primary',k_start=50,k_end=100,zero_predictor_late_nmse=1.,conditions={c:dict(mean_nmse_by_k=[.1]*101,late_context_nmse=.1) for c in ('gaussian_sd0','gaussian_sd1','gaussian_sd2')})
        summary.check_metric(m,manifest,cfg,50000,'primary')
        with self.assertRaises(ValueError):summary.check_metric(dict(m,eval_seed=315833),manifest,cfg,50000,'primary')
        m['conditions']['gaussian_sd0']['late_context_nmse']=.2
        with self.assertRaises(ValueError):summary.check_metric(m,manifest,cfg,50000,'primary')

if __name__=='__main__':unittest.main()

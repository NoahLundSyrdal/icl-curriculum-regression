"""Scientific pairing, inference, source isolation and recovery checks."""
import dataclasses, json, math, tempfile, unittest
from pathlib import Path
from unittest import mock
import torch
import followup_plan as plan
import followup_experiment as exp
import followup_analysis as analysis

def equal(test,a,b):
    if isinstance(a,torch.Tensor):test.assertTrue(torch.equal(a.cpu(),b.cpu()))
    elif isinstance(a,dict):
        test.assertEqual(set(a),set(b))
        for k in a:equal(test,a[k],b[k])
    elif isinstance(a,(list,tuple)):
        test.assertEqual(len(a),len(b))
        for x,y in zip(a,b):equal(test,x,y)
    else:test.assertEqual(a,b)

class Design(unittest.TestCase):
    def test_equal_full_horizon_tuning_and_disjoint_seeds(self):
        tasks=plan.development_tasks();self.assertEqual(len(tasks),150)
        self.assertEqual(len(plan.templates()),360)
        self.assertFalse(set(plan.DEV_SEEDS)&set(plan.CONFIRM_SEEDS))
        for seed in plan.DEV_SEEDS:
            for x in plan.strategies():
                sub=[t for t in tasks if t['seed']==seed and t['strategy']==x['strategy']]
                self.assertEqual([t['learning_rate'] for t in sub],list(plan.LRS))
                self.assertTrue(all(t['steps']==125000 for t in sub))
    def scores(self):return {(s,x['strategy'],lr):1. for s in plan.DEV_SEEDS for x in plan.strategies() for lr in plan.LRS}
    def test_each_strategy_gets_own_best_and_symmetric_shared_setting(self):
        scores=self.scores()
        for seed in plan.DEV_SEEDS:
            scores[seed,'joint_d20_curriculum',1e-3]=.01
            scores[seed,'joint_d20_shuffled',3e-4]=.1
            scores[seed,'hard_d20',1e-5]=.05
        ranks=plan.rank_development(scores);resolved=plan.resolve(ranks)
        self.assertEqual(ranks['tuned']['joint_d20_curriculum'][0]['learning_rate'],1e-3)
        self.assertEqual(ranks['tuned']['hard_d20'][0]['learning_rate'],1e-5)
        self.assertEqual(ranks['shared']['joint_d20'][0]['learning_rate'],1e-3)
        for t in resolved:
            if t['group']=='joint_d20' and t['panel']=='shared':self.assertEqual(t['learning_rate'],1e-3)
        # Swapping curriculum/shuffled scores cannot change common setting selection.
        swapped={(s,(name.replace('curriculum','TEMP').replace('shuffled','curriculum').replace('TEMP','shuffled')),lr):v for (s,name,lr),v in scores.items()}
        self.assertEqual(plan.rank_development(swapped)['shared'],ranks['shared'])
    def test_missing_nonfinite_and_extra_development_data_rejected(self):
        scores=self.scores();scores.pop(next(iter(scores)))
        with self.assertRaises(ValueError):plan.rank_development(scores)
        scores=self.scores();scores[next(iter(scores))]=float('nan')
        with self.assertRaises(ValueError):plan.rank_development(scores)
        scores=self.scores();scores[2000,'hard_d20',1e-5]=.1
        with self.assertRaises(ValueError):plan.rank_development(scores)
    def test_aliases_reuse_only_exact_training_config_and_do_not_add_replication(self):
        ranks=plan.rank_development(self.scores());resolved=plan.resolve(ranks)
        self.assertEqual(sum(t['alias_of'] is None for t in resolved),200)
        for t in resolved:
            if t['alias_of'] is not None:
                source=resolved[t['alias_of']];self.assertIsNone(source['alias_of'])
                for key in ('seed','dims','arm','component','learning_rate'):self.assertEqual(t[key],source[key])
        self.assertEqual(len({str(exp.task_directory(t)) for t in plan.development_tasks()}),150)
    def test_sign_and_holm_multiple_comparison_controls(self):
        self.assertEqual(plan.sign_test([-1]*20)['p_two_sided'],2/2**20)
        self.assertAlmostEqual(plan.sign_test([-1]*16+[1]*4)['p_two_sided'],.01181793212890625)
        self.assertEqual(plan.sign_test([0]*20)['p_two_sided'],1.)
        self.assertEqual(plan.holm({'a':.01,'b':.03,'c':.2}),{'a':.03,'b':.06,'c':.2})
        with self.assertRaises(ValueError):plan.holm({'a':float('nan')})
    def test_primary_failure_blocks_all_secondary_claims_and_practical_gate(self):
        def fixture(primary_left=.01,primary_right=.02):
            result={group+'_'+suffix:analysis.paired([.01]*20,[.02]*20) for group,_,_ in plan.GROUPS for suffix in ('ordering','policy')}
            result['joint_d20_ordering']=analysis.paired([primary_left]*20,[primary_right]*20)
            return result
        result=fixture(.02,.01);self.assertFalse(analysis.classify(result))
        self.assertFalse(any(v['confirmatory_supported'] for v in result.values()))
        result=fixture();self.assertTrue(analysis.classify(result));self.assertTrue(all(v['confirmatory_supported'] for v in result.values()))
        result=fixture(.019,.02);self.assertFalse(analysis.classify(result))
        result=fixture(.3,.6);self.assertFalse(analysis.classify(result))
    def test_selection_uses_only_complete_development_primary_clean_final_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            selected=Path(directory)/'selection.json';resolved=Path(directory)/'resolved.json'
            def fake(t):
                score=.01 if t['learning_rate']==3e-5 else .1
                return ({'initial_model_sha256':str((t['seed'],t['dims']))},
                    {125000:{'conditions':{'gaussian_sd0':{'late_context_nmse':score},'gaussian_sd2':{'late_context_nmse':1000-score}}}},
                    {'conditions':{'gaussian_sd0':{'late_context_nmse':1000-score}}})
            with mock.patch.object(plan,'verify_sources'),mock.patch.object(plan,'digest',return_value='fixture'),mock.patch.object(plan,'SELECT',selected),mock.patch.object(plan,'RESOLVED',resolved),mock.patch.object(analysis,'read_task',side_effect=fake):
                analysis.select();value=json.loads(selected.read_text())
                self.assertTrue(all(r[0]['learning_rate']==3e-5 for r in value['ranking']['tuned'].values()))
                self.assertEqual(value['unique_confirmation_models'],200)
                with self.assertRaises(FileExistsError):analysis.select()
    def test_geometric_ratio_not_driven_by_a_single_arithmetic_outlier(self):
        c=[.01]*20;s=[.02]*20;s[-1]=1.
        r=analysis.paired(c,s);self.assertLess(r['geometric_mean_ratio'],.5)
        self.assertAlmostEqual(r['leave_one_out_geometric_ratios'][-1],.5)
        with self.assertRaises(ValueError):analysis.paired([.01]*19,[.02]*19)

class Scientific(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.cfg=exp.training.base.preset_config('smoke',mode='clean',seed=1600,dims=3,points=7,
            embedding=16,layers=1,heads=2,steps=8,batch_size=4,microbatch=2,eval_episodes=4,
            eval_microbatch=2,bayes_reference=False)
        self.study=exp.training.Study(initial_dims=1,easy_steps=1,prefix_steps=4,evaluation_steps=(0,2,4,6,8),checkpoint_every=2,log_every=2,cohort='gpu_smoke')
    def tearDown(self):self.tmp.cleanup()
    def train(self,name,study,**kw):return exp.training.train(self.cfg,study,self.root/name,torch.device('cpu'),**kw)
    def test_exact_batches_components_shared_terminal_phase_and_variance(self):
        for component in ('both','dimension','context'):
            c=dataclasses.replace(self.study,component=component);s=dataclasses.replace(c,arm='shuffled')
            ids=exp.training.batch_order(self.cfg,s)
            equal(self,ids[:4].sort().values,torch.arange(4));equal(self,ids[4:],torch.arange(4,8))
            for bid in range(8):equal(self,exp.training.batch(self.cfg,c,bid),exp.training.batch(self.cfg,s,bid))
        reference=exp.training.batch(self.cfg,self.study,4)
        for component,active,eligible in (('both',1,3),('dimension',1,7),('context',3,3),('none',3,7)):
            study=dataclasses.replace(self.study,component=component,arm='hard' if component=='none' else 'curriculum')
            treatment=exp.training.treatment(self.cfg,study,0);self.assertEqual(treatment[:2],(active,eligible))
            self.assertAlmostEqual(treatment[0]*treatment[2]**2,3)
            equal(self,reference,exp.training.batch(self.cfg,study,4))
    def test_production_shapes_and_masks_in_both_dimensions(self):
        for task in plan.development_tasks()[:10]:
            cfg,study=exp.task_config(task)
            self.assertEqual((cfg.points,cfg.dims),(2*task['dims']+1,task['dims']))
            for bid in (0,999,1000,24999,25000,124999):
                active,eligible,mult=exp.training.treatment(cfg,study,bid)
                self.assertAlmostEqual(active*mult**2,cfg.dims)
                if bid>=25000:self.assertEqual((active,eligible,mult),(cfg.dims,cfg.points,1.))
    def test_own_and_future_label_leakage_and_independent_query(self):
        exp.training.base.seed_model(self.cfg.seed);model=exp.training.base.RegressionTransformer(self.cfg).eval()
        x,y,_,_=exp.training.batch(self.cfg,self.study,0)
        with torch.no_grad():full=model(x,y)
        for k in (0,2,6):
            xm=x.clone();ym=y.clone();xm[:,k+1:]+=1000;ym[:,k:]+=10000
            labels=torch.zeros_like(x[:,:k]);labels[:,:,0]=y[:,:k]
            tokens=torch.cat((torch.stack((x[:,:k],labels),dim=2).flatten(1,2),x[:,k:k+1]),dim=1)
            with torch.no_grad():
                changed=model(xm,ym);query=model.output_projection(model.backbone(inputs_embeds=model.input_projection(tokens)).last_hidden_state[:,-1]).squeeze(-1)
            equal(self,full[:,:k+1],changed[:,:k+1]);self.assertTrue(torch.allclose(full[:,k],query,atol=1e-6,rtol=1e-6))
    def test_bitwise_resume_all_components_and_optimizers(self):
        for i,component in enumerate(('both','dimension','context','none')):
            study=dataclasses.replace(self.study,component=component,arm='hard' if component=='none' else 'shuffled')
            self.train(f'w{i}',study);self.train(f's{i}',study,max_steps_this_run=3);self.train(f's{i}',study,resume=True)
            a=torch.load(self.root/f'w{i}/checkpoint.pt',weights_only=True);b=torch.load(self.root/f's{i}/checkpoint.pt',weights_only=True)
            for key in ('model','optimizer','torch_rng','cuda_rng','python_rng'):equal(self,a[key],b[key])
        with self.assertRaises(ValueError):self.train('s0',dataclasses.replace(self.study,component='context'),resume=True)
    def test_sparse_retention_preserves_required_checkpoints_and_rng_isolation(self):
        self.train('a',self.study);root=self.root/'a'
        for t in (0,4,8):self.assertTrue((root/f'checkpoint_step_{t:06d}.pt').exists())
        self.assertFalse((root/'checkpoint_step_000002.pt').exists())
        self.train('b',dataclasses.replace(self.study,evaluation_steps=(0,4,8)))
        a=torch.load(root/'checkpoint.pt',weights_only=True);b=torch.load(self.root/'b/checkpoint.pt',weights_only=True)
        for key in ('model','optimizer','torch_rng','cuda_rng','python_rng'):equal(self,a[key],b[key])
    def test_metric_checks_catch_wrong_bank_target_scale_and_scalar(self):
        task=plan.development_tasks()[0];cfg,study=exp.task_config(task)
        manifest={'study':{'cohort':'development'}}
        m=dict(identity=manifest,completed_steps=125000,evaluation_dimension=20,evaluation_positions=41,
            eval_seed=plan.BANKS['development'][0],eval_episodes=4096,evaluation_split='primary',
            k_start=20,k_end=40,zero_predictor_late_nmse=1.,conditions={c:dict(mean_nmse_by_k=[.1]*41,late_context_nmse=.1) for c in ('gaussian_sd0','gaussian_sd1','gaussian_sd2')})
        analysis.check_metric(m,manifest,cfg,125000,'primary')
        with self.assertRaises(ValueError):analysis.check_metric(m|{'eval_seed':0},manifest,cfg,125000,'primary')
        m['conditions']['gaussian_sd0']['late_context_nmse']=.2
        with self.assertRaises(ValueError):analysis.check_metric(m,manifest,cfg,125000,'primary')
if __name__=='__main__':unittest.main()

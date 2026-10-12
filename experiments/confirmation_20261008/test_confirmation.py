"""Scientific controls and selection/inference isolation, using synthetic fixtures."""
import dataclasses
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import torch
import confirmation_plan as plan
import confirmation_experiment as exp


class PlanAndInference(unittest.TestCase):
    def test_fresh_disjoint_seeds_banks_and_full_grid(self):
        self.assertFalse(set(plan.DEV_SEEDS)&set(plan.CONFIRM_SEEDS))
        self.assertFalse((set(plan.DEV_SEEDS)|set(plan.CONFIRM_SEEDS))&{0,1,2,3,4,100,101,200,201,202,203,204,300,301})
        self.assertEqual(len({plan.DEV_BANK,plan.DEV_HOLDOUT,plan.CONFIRM_BANK,plan.CONFIRM_HOLDOUT}),4)
        self.assertEqual(len(plan.development_tasks()),36);self.assertEqual(len(plan.confirmation_templates()),60)
        for seed in plan.DEV_SEEDS:
            for arm in plan.ARMS:
                subset=[t for t in plan.development_tasks() if t['seed']==seed and t['arm']==arm]
                self.assertEqual([t['setting'] for t in subset],[o[0] for o in plan.OPTIMIZERS])
                self.assertTrue(all(t['target_variance']=='constant' and t['steps']==50000 for t in subset))

    def test_selection_protects_strong_baselines_and_tie_order(self):
        scores={(s,name,arm):1. for s in plan.DEV_SEEDS for name,*_ in plan.OPTIMIZERS for arm in plan.ARMS}
        for seed in plan.DEV_SEEDS:
            scores[seed,'lr_high','curriculum']=.01
            scores[seed,'baseline','shuffled']=.1
            scores[seed,'loss_nmse','hard']=.2
        ranks=plan.rank_development(scores);tasks=plan.resolve_templates(ranks)
        self.assertTrue(all(t['setting']==('loss_nmse' if t['arm']=='hard' else 'baseline') for t in tasks))
        for seed in plan.DEV_SEEDS:scores[seed,'lr_low','shuffled']=.1
        self.assertEqual(plan.rank_development(scores)['shuffled'][0]['setting'],'lr_low')
        del scores[plan.DEV_SEEDS[0],'lr_low','hard']
        with self.assertRaises(KeyError):plan.rank_development(scores)

    def test_trapezoidal_auc_and_invalid_inputs(self):
        self.assertEqual(plan.trapezoidal_mean([0,1,4],[0,1,4]),2.)
        for t,y in [([0,0],[1,1]),([0,1],[float('nan'),1]),([0,1],[-1,1])]:
            with self.assertRaises(ValueError):plan.trapezoidal_mean(t,y)

    def test_exact_sign_test_ties_direction_and_sample_size(self):
        self.assertEqual(plan.exact_sign_test([-1]*5)['two_sided_p'],.0625)
        self.assertAlmostEqual(plan.exact_sign_test([-1]*15+[1]*5)['two_sided_p'],.04138946533203125)
        tied=plan.exact_sign_test([-1,1,0,1e-9]);self.assertEqual((tied['wins'],tied['losses'],tied['ties'],tied['two_sided_p']),(1,1,2,1.))
        self.assertEqual(plan.exact_sign_test([1]*20)['two_sided_p'],plan.exact_sign_test([-1]*20)['two_sided_p'])
        self.assertEqual(plan.exact_sign_test([0]*20)['two_sided_p'],1.)
        with self.assertRaises(ValueError):plan.paired_statistics([1],[2])

    def test_fixed_sequence_cannot_replace_failed_primary_with_ordering(self):
        auc={(s,arm):.1 for s in plan.CONFIRM_SEEDS for arm in plan.ARMS}
        finals={(s,arm):.1 for s in plan.CONFIRM_SEEDS for arm in plan.ARMS}
        for seed in plan.CONFIRM_SEEDS:auc[seed,'shuffled']=.9
        result=plan.analyze_scores(auc,finals)
        self.assertFalse(result['primary_supported']);self.assertFalse(result['ordering_test_eligible']);self.assertFalse(result['ordering_supported'])
        for seed in plan.CONFIRM_SEEDS:auc[seed,'hard']=.9
        result=plan.analyze_scores(auc,finals);self.assertTrue(result['primary_supported']);self.assertTrue(result['ordering_supported'])
        for seed in plan.CONFIRM_SEEDS:finals[seed,'curriculum']=.3
        self.assertFalse(plan.analyze_scores(auc,finals)['primary_supported'])

    def test_bootstrap_is_paired_and_reproducible(self):
        left=[.1]*20;right=[.2]*20
        a=plan.paired_statistics(left,right);b=plan.paired_statistics(left,right)
        self.assertEqual(a,b);self.assertEqual(a['descriptive_bootstrap_mean_ci95'],[-.1,-.1])

    def test_missing_selection_refuses_confirmation_before_development(self):
        with mock.patch.object(plan,'verify_sources'),mock.patch.object(plan,'SELECTION',Path('/tmp/no-selection-confirmation-fixture-does-not-exist')):
            with self.assertRaises(FileNotFoundError):exp.tasks('confirmation')


class TrainerControls(unittest.TestCase):
    def test_all_production_tasks_use_fixed_shape_and_constant_scale(self):
        torch.set_num_threads(1)
        for task in plan.development_tasks():
            cfg,study=exp.task_config(task)
            self.assertEqual((cfg.dims,cfg.points,cfg.batch_size,cfg.microbatch),(50,101,256,128))
            self.assertEqual(cfg.eval_seed,plan.DEV_BANK)
            for bid in (0,999,1000,24999,25000,49999):
                active,eligible,multiplier=exp.training.treatment(cfg,study,bid)
                self.assertAlmostEqual(active*multiplier**2,50)
                if bid>=25000:self.assertEqual((active,eligible,multiplier),(50,101,1.))

    def test_same_initialization_and_batch_multiset_at_every_optimizer(self):
        cfg,study=exp.task_config(plan.development_tasks()[0])
        shuffled=dataclasses.replace(study,arm='shuffled')
        ids=exp.training.batch_order(cfg,shuffled)
        self.assertTrue(torch.equal(ids[:25000].sort().values,torch.arange(25000)))
        self.assertTrue(torch.equal(ids[25000:],torch.arange(25000,cfg.steps)))
        for bid in (0,12345,24999,25000,49999):
            a=exp.training.batch(cfg,study,bid);b=exp.training.batch(cfg,shuffled,bid)
            for x,y in zip(a,b):
                if isinstance(x,torch.Tensor):self.assertTrue(torch.equal(x,y))
                else:self.assertEqual(x,y)

    def test_query_target_absent_independent_causal_forward(self):
        cfg=exp.training.base.preset_config('smoke',mode='clean',seed=600,dims=3,points=7,embedding=16,layers=1,heads=2)
        exp.training.base.seed_model(cfg.seed);model=exp.training.base.RegressionTransformer(cfg).eval()
        x,y,_=exp.training.base.EpisodeStream(731839).draw(4,7,3)
        with torch.no_grad():full=model(x,y)
        for k in (0,2,6):
            labels=torch.zeros_like(x[:,:k]);labels[:,:,0]=y[:,:k]
            tokens=torch.stack((x[:,:k],labels),dim=2).flatten(1,2)
            query_tokens=torch.cat((tokens,x[:,k:k+1]),dim=1)
            self.assertEqual(query_tokens.shape[1],2*k+1)
            with torch.no_grad():
                h=model.backbone(inputs_embeds=model.input_projection(query_tokens)).last_hidden_state
                prediction=model.output_projection(h[:,-1]).squeeze(-1)
            self.assertTrue(torch.allclose(prediction,full[:,k],atol=1e-6,rtol=1e-6))


if __name__=='__main__':unittest.main()

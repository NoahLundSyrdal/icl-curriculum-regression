"""Independent scientific preflight before the development/confirmation pipeline."""
import argparse, dataclasses, json, shutil, subprocess, sys, time, unittest
from pathlib import Path
import torch
import followup_plan as plan
import followup_experiment as exp
from test_followup import equal

def fresh_audit():
    seen=set();banks=set();count=0
    # Read actual prior run records, not source task templates or archived ZIPs.
    parent=Path.cwd().parent
    roots=[parent/'results']+[x/'results' for x in parent.iterdir() if x.is_dir() and x!=Path.cwd() and (x/'results').is_dir()]
    for root in roots:
        for path in list(root.rglob('manifest.json'))+list(root.rglob('metrics.json')):
            value=json.loads(path.read_text());cfg=value.get('config',value.get('identity',{}).get('config',{}))
            if isinstance(cfg.get('seed'),int):seen.add(cfg['seed'])
            if isinstance(cfg.get('eval_seed'),int):banks.add(cfg['eval_seed'])
            if isinstance(value.get('eval_seed'),int):banks.add(value['eval_seed'])
            count+=1
    if seen&set(plan.DEV_SEEDS+plan.CONFIRM_SEEDS):raise ValueError('New training seeds already used')
    if banks&{b for pair in plan.BANKS.values() for b in pair}:raise ValueError('Evaluation seed already used')
    return dict(prior_records=count,prior_seeds=sorted(seen),prior_evaluation_banks=sorted(banks),
        training_seeds_disjoint=True,evaluation_banks_disjoint=True,
        note='Audits prior manifests and primary/holdout metric records before any production training.')

def cpu():
    plan.verify_sources();torch.set_num_threads(2)
    subprocess.run([sys.executable,'-m','unittest','-v','test_followup'],check=True)
    fresh=fresh_audit();probes=[]
    for task in plan.development_tasks():
        cfg,study=exp.task_config(task);ids=exp.training.batch_order(cfg,study)
        if not torch.equal(ids[:25000].sort().values,torch.arange(25000)) or not torch.equal(ids[25000:],torch.arange(25000,125000)):raise ValueError('Wrong multiset/tail')
        for bid in (0,999,1000,12345,24999,25000,124999):
            x,y,active,eligible=exp.training.batch(cfg,study,bid)
            if x.shape!=(256,cfg.points,cfg.dims) or y.shape!=(256,cfg.points) or not torch.isfinite(x).all() or not torch.isfinite(y).all():raise ValueError('Invalid production batch')
            if x[:,:,active:].count_nonzero()!=0:raise ValueError('Invalid mask')
            if abs(active*exp.training.treatment(cfg,study,bid)[2]**2-cfg.dims)>1e-10:raise ValueError('Wrong variance')
            if bid>=25000 and (active,eligible)!=(cfg.dims,cfg.points):raise ValueError('Wrong terminal treatment')
        probes.append(dict(index=task['index'],dims=cfg.dims,strategy=task['strategy'],lr=cfg.learning_rate))
    plan.atomic_json('submissions/cpu-preflight.json',dict(kind=plan.KIND,tests_passed=True,
        all_150_production_batch_probes_passed=True,fresh_seed_audit=fresh,freeze_sha256=plan.digest(plan.FREEZE),
        probes=probes,python=sys.version,torch=str(torch.__version__),transformers=exp.training.base.transformers.__version__))
    with open('submissions/environment.txt','w') as f:subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
    print('CPU scientific tests and all 150 production-task probes passed.',flush=True)

def gpu():
    plan.verify_sources();torch.set_num_threads(2)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 required')
    records=[];initial={};root=Path('results/gpu_smoke')
    # Every LR at both physical dimensions and every treatment component is exercised.
    for task in [t for t in plan.development_tasks() if t['seed']==plan.DEV_SEEDS[0]]:
        cfg,study=exp.task_config(task,smoke=True);whole=root/f"{task['index']}_whole";split=root/f"{task['index']}_split"
        started=time.monotonic()
        if exp.training.train(cfg,study,whole,torch.device('cuda'),max_steps_this_run=4)!=4:raise ValueError('Wrong whole progress')
        if exp.training.train(cfg,study,split,torch.device('cuda'),max_steps_this_run=2)!=2:raise ValueError('Wrong split progress')
        if exp.training.train(cfg,study,split,torch.device('cuda'),resume=True,max_steps_this_run=2)!=4:raise ValueError('Wrong resume progress')
        a=torch.load(whole/'checkpoint.pt',map_location='cpu',weights_only=True);b=torch.load(split/'checkpoint.pt',map_location='cpu',weights_only=True)
        for key in ('model','optimizer','torch_rng','cuda_rng','python_rng'):equal(unittest.TestCase(),a[key],b[key])
        key=(cfg.seed,cfg.dims);fp=a['identity']['initial_model_sha256']
        if key in initial and initial[key]!=fp:raise ValueError('Initialization differs')
        initial[key]=fp
        model=exp.training.base.RegressionTransformer(cfg).to('cuda').eval();model.load_state_dict(a['model'])
        x,y,_=exp.training.base.EpisodeStream(cfg.eval_seed).draw(4,cfg.points,cfg.dims);x=x.to('cuda');y=y.to('cuda')
        deltas=[]
        with torch.no_grad(),exp.training.base.autocast_context(torch.device('cuda'),'bf16'):full=model(x,y)
        for k in (0,cfg.dims,cfg.points-1):
            xm=x.clone();ym=y.clone();xm[:,k+1:]+=1000;ym[:,k:]+=10000
            labels=torch.zeros_like(x[:,:k]);labels[:,:,0]=y[:,:k]
            tokens=torch.cat((torch.stack((x[:,:k],labels),dim=2).flatten(1,2),x[:,k:k+1]),dim=1)
            with torch.no_grad(),exp.training.base.autocast_context(torch.device('cuda'),'bf16'):
                changed=model(xm,ym);query=model.output_projection(model.backbone(inputs_embeds=model.input_projection(tokens)).last_hidden_state[:,-1]).squeeze(-1)
            if not torch.equal(full[:,:k+1],changed[:,:k+1]):raise ValueError('Causal label leakage')
            delta=float((full[:,k].float()-query.float()).abs().max());deltas.append(delta)
            if not torch.allclose(full[:,k].float(),query.float(),atol=.01,rtol=.02):raise ValueError('Query-target-absent mismatch')
        records.append(dict(index=task['index'],strategy=task['strategy'],lr=cfg.learning_rate,dims=cfg.dims,
            bitwise_resume=True,causal_mutation_bitwise=True,query_target_absent_max_difference=max(deltas),
            training_seconds_4_updates=a['training_seconds'],total_probe_seconds=time.monotonic()-started))
        del a,b,model;torch.cuda.empty_cache()
        # Smoke binaries are temporary; report and metrics remain, production binaries are retained.
        for directory in (whole,split):
            for checkpoint in directory.glob('*.pt'):checkpoint.unlink()
    plan.atomic_json('submissions/gpu-smoke.json',dict(kind=plan.KIND,all_50_shapes_treatments_rates_passed=True,
        tests_passed=True,records=records,freeze_sha256=plan.digest(plan.FREEZE),
        gpu=torch.cuda.get_device_name(0),torch=str(torch.__version__)))
    print('GPU shapes, five LRs, all treatments, bitwise resume and causal checks passed.',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',action='store_true');a=p.parse_args();gpu() if a.gpu else cpu()

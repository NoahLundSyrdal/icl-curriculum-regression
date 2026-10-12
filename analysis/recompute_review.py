"""Exploratory review analyses. Run: python recompute_review.py [verified_results.zip] [REVIEW_AUDIT.json]
Uses Python standard library only; the frozen analysis remains in frozen_source/.
"""
from pathlib import Path
import sys,zipfile,json,math,statistics
z=zipfile.ZipFile(sys.argv[1] if len(sys.argv)>1 else 'verified_results.zip')
r=json.loads(z.read('results/confirmation/summary.json'))['individual_runs']; records={(v['seed'],v['arm']):v for v in r};seeds=sorted({s for s,a in records})
# Read raw scalar metrics for every endpoint; verify the existing summary separately.
primary={};holdout={}
for name in z.namelist():
    if not name.endswith('/metrics.json'):continue
    m=json.loads(z.read(name));seed=m['identity']['config']['seed'];arm=m['identity']['study']['arm']
    if m['eval_seed']==731841:primary[seed,arm,m['completed_steps']]=m
    elif m['eval_seed']==731842:holdout[seed,arm]=m
for seed in seeds:
    for arm in ['curriculum','shuffled','hard']:
        times=sorted(t for s,a,t in primary if s==seed and a==arm and t>=25000)
        for cond in ['gaussian_sd0','gaussian_sd1','gaussian_sd2']:
            values=[primary[seed,arm,t]['conditions'][cond]['late_context_nmse'] for t in times]
            raw={'auc':sum((b-a)*(u+v)/2 for a,b,u,v in zip(times,times[1:],values,values[1:]))/100000,'final':values[-1],'holdout':holdout[seed,arm]['conditions'][cond]['late_context_nmse']}
            for metric,value in raw.items():
                assert math.isclose(value,records[seed,arm]['outcomes'][cond][metric],abs_tol=1e-12)
                records[seed,arm]['outcomes'][cond][metric]=value
def signed_rank(ds):
    nonzero=[d for d in ds if d!=0]; vals=sorted(nonzero,key=abs)
    assert len({abs(v) for v in vals})==len(vals),'Exact no-tie implementation requires distinct absolute differences'
    n=len(vals); total=n*(n+1)//2; wp=sum(i for i,d in enumerate(vals,1) if d>0); counts=[0]*(total+1);counts[0]=1
    for rank in range(1,n+1):
        for v in range(total,rank-1,-1):counts[v]+=counts[v-rank]
    p=min(1,2*sum(counts[:min(wp,total-wp)+1])/2**n)
    return {'n':n,'W_plus':wp,'W_minus':total-wp,'two_sided_exact_p':p}
audit={'interpretation':'All 20 seeds retained in frozen analysis. Ratios, Wilcoxon signed-rank tests, bump comparisons and exclusion sensitivity analyses are exploratory and post hoc. Exact signed-rank tests on paired differences have no zero differences or ties and assume symmetry under the null.','ratios':{},'noise':{},'hard_groups':{},'bump_ratio':{}}
for endpoint in ['auc','final','holdout']:
    for excluded in [None,1017]:
        ss=[s for s in seeds if s!=excluded]; a=[records[s,'curriculum']['outcomes']['gaussian_sd0'][endpoint] for s in ss];b=[records[s,'shuffled']['outcomes']['gaussian_sd0'][endpoint] for s in ss]
        audit['ratios'][f'{endpoint}_excluded_{excluded}']={'n':len(ss),'ratio_of_means':statistics.mean(a)/statistics.mean(b),'median_paired_ratio':statistics.median(x/y for x,y in zip(a,b)),'geometric_mean_paired_ratio':math.exp(statistics.mean(math.log(x/y) for x,y in zip(a,b))),'median_difference':statistics.median(x-y for x,y in zip(a,b)),'signed_rank_difference':signed_rank([x-y for x,y in zip(a,b)]),'signed_rank_log_ratio':signed_rank([math.log(x/y) for x,y in zip(a,b)])}
for cond in ['gaussian_sd0','gaussian_sd1','gaussian_sd2']:
    audit['noise'][cond]={}
    for endpoint in ['auc','final','holdout']:
        ds=[records[s,'curriculum']['outcomes'][cond][endpoint]-records[s,'shuffled']['outcomes'][cond][endpoint] for s in seeds]
        audit['noise'][cond][endpoint]={'curriculum_better':sum(d<0 for d in ds),'curriculum_worse':sum(d>0 for d in ds),'mean_difference':statistics.mean(ds),'median_difference':statistics.median(ds)}
for group,learned in [('learned',True),('did_not_learn',False)]:
    ss=[s for s in seeds if (records[s,'hard']['outcomes']['gaussian_sd0']['final']<=.25)==learned]; finals=[records[s,'hard']['outcomes']['gaussian_sd0']['final'] for s in ss]
    audit['hard_groups'][group]={'n':len(ss),'seeds':ss,'final_mean':statistics.mean(finals),'final_median':statistics.median(finals),'final_range':[min(finals),max(finals)],'auc_mean':statistics.mean(records[s,'hard']['outcomes']['gaussian_sd0']['auc'] for s in ss)}
curves={}
for name in z.namelist():
    if not name.endswith('/metrics.json'):continue
    m=json.loads(z.read(name))
    if m['completed_steps']!=125000 or m['eval_seed']!=731841:continue
    cfg=m['identity']['config'];arm=m['identity']['study']['arm']
    for cond in ['gaussian_sd1','gaussian_sd2']:
        curves[cfg['seed'],arm,cond]=m['conditions'][cond]['mean_nmse_by_k']
for cond in ['gaussian_sd1','gaussian_sd2']:
    paired=[]
    for s in seeds:
        row={'seed':s}
        for arm in ['curriculum','shuffled']:
            v=curves[s,arm,cond];row[arm]=statistics.mean(v[48:53])/statistics.mean(v[38:43])
        row['difference']=row['curriculum']-row['shuffled'];paired.append(row)
    ds=[v['difference'] for v in paired]
    audit['bump_ratio'][cond]={'definition':'mean NMSE(k=48..52) / mean NMSE(k=38..42), final checkpoint, evaluation bank 731841','curriculum_larger':sum(d>0 for d in ds),'shuffled_larger':sum(d<0 for d in ds),'curriculum_median':statistics.median(v['curriculum'] for v in paired),'shuffled_median':statistics.median(v['shuffled'] for v in paired),'signed_rank_difference':signed_rank(ds),'paired_seeds':paired}
output=Path(sys.argv[2] if len(sys.argv)>2 else 'REVIEW_AUDIT.json');output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit['bump_ratio'],indent=2))

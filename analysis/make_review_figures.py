from pathlib import Path
import csv, io, json, sys, zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
z=zipfile.ZipFile(sys.argv[1] if len(sys.argv)>1 else 'verified_results.zip')
OUT=Path(sys.argv[2] if len(sys.argv)>2 else 'regenerated_figures');OUT.mkdir(parents=True,exist_ok=True)
s=json.loads(z.read('results/confirmation/summary.json'));seeds=s['seeds'];r={(v['seed'],v['arm']):v for v in s['individual_runs']};conditions=['gaussian_sd0','gaussian_sd1','gaussian_sd2']
rows=list(csv.DictReader(io.StringIO(z.read('results/confirmation/learning_curves.csv').decode())))
clean={(int(v['seed']),v['arm'],int(v['step'])):float(v['late_context_nmse']) for v in rows if v['condition']=='gaussian_sd0'}
times=sorted({int(v['step']) for v in rows});curves={}
for name in z.namelist():
    if not name.endswith('/metrics.json'):continue
    m=json.loads(z.read(name))
    if m['completed_steps']!=125000 or m['eval_seed']!=731841:continue
    seed=m['identity']['config']['seed'];arm=m['identity']['study']['arm']
    for cond in conditions:curves[seed,arm,cond]=m['conditions'][cond]['mean_nmse_by_k']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','savefig.dpi':220})
palette={'curriculum':'#14675c','shuffled':'#b16a20','hard':'#65758b'}
labels={'curriculum':'Curriculum','shuffled':'Same batches shuffled','hard':'Full difficulty'}
fig,axs=plt.subplots(2,2,figsize=(11.4,7.1)); ax,bx,cx,dx=axs.flat
for arm in ['curriculum','shuffled']:
    vals=np.array([[clean[s,arm,t] for t in times] for s in seeds]); low,med,high=np.quantile(vals,[.25,.5,.75],axis=0)
    ax.fill_between(np.array(times)/1000,low,high,color=palette[arm],alpha=.16)
    ax.plot(np.array(times)/1000,med,color=palette[arm],lw=2,label=labels[arm])
ax.set_title('A. Clean learning: medians and middle 50%',loc='left',fontsize=10)
for s in seeds:
    bx.plot(np.array(times)/1000,[clean[s,'hard',t] for t in times],color=palette['hard'],alpha=.32,lw=.9)
bx.plot([],[],color=palette['hard'],lw=1,label='Full difficulty: each of 20 runs')
bx.plot(np.array(times)/1000,np.median([[clean[s,'curriculum',t] for t in times] for s in seeds],axis=0),color=palette['curriculum'],lw=2,label='Curriculum median')
bx.set_title('B. Full difficulty: two groups, no mean curve',loc='left',fontsize=10)
for a in [ax,bx]:
    a.axvspan(0,25,color='#e7edef',alpha=.5,zorder=-1); a.set_yscale('log');a.set_xlim(0,125);a.set_ylim(.003,4.1)
    a.set_xlabel('Optimizer updates (thousands)');a.set_ylabel('Clean NMSE (contexts 50-100)');a.grid(axis='y',alpha=.15);a.legend(frameon=False,fontsize=8)
for idx,cond in enumerate(conditions):
    ratios=np.array([r[s,'curriculum']['outcomes'][cond]['final']/r[s,'shuffled']['outcomes'][cond]['final'] for s in seeds])
    cx.scatter(idx+np.linspace(-.15,.15,20),ratios,color=palette['curriculum'],s=17,alpha=.6)
    gm=float(np.exp(np.log(ratios).mean()));cx.scatter(idx,gm,marker='D',s=50,color='#263b44',zorder=4)
    cx.text(idx,5.1,['17/20 better','19/20 worse','16/20 worse'][idx],ha='center',fontsize=8)
cx.axhline(1,color='#b16a20',lw=1);cx.set_yscale('log');cx.set_ylim(.1,6);cx.set_xticks([0,1,2],['Clean','Noise SD1','Noise SD2'])
cx.set_yticks([.1,.25,.5,1,2,4],['0.1','0.25','0.5','1','2','4']);cx.tick_params(axis='y',which='minor',labelleft=False)
cx.set_ylabel('Final error ratio: curriculum / shuffled');cx.set_title('C. Final paired ratios: dots = all seeds',loc='left',fontsize=10)
cx.text(.03,.03,'Diamonds: geometric mean; below 1 favors curriculum',transform=cx.transAxes,fontsize=8);cx.grid(axis='y',alpha=.15)
for arm in ['curriculum','shuffled']:
    vals=np.array([curves[s,arm,'gaussian_sd2'] for s in seeds]); low,med,high=np.quantile(vals,[.25,.5,.75],axis=0)
    dx.fill_between(np.arange(101),low,high,color=palette[arm],alpha=.15);dx.plot(med,color=palette[arm],lw=2,label=labels[arm])
dx.axvline(50,color='#65758b',lw=.8,ls='--');dx.set_xlim(0,100);dx.set_xlabel('Number of demonstrations (k)');dx.set_ylabel('Final clean-target NMSE; label noise SD2')
dx.set_title('D. Noise response: medians and middle 50%',loc='left',fontsize=10);dx.legend(frameon=False,fontsize=8);dx.grid(axis='y',alpha=.15)
fig.tight_layout(h_pad=2,w_pad=2.5)
fig.savefig(OUT/'icl_confirmation_figures.png',bbox_inches='tight');fig.savefig(OUT/'icl_confirmation_figures.svg',bbox_inches='tight');plt.close(fig)

"""Uncertainty summaries and additional empirical figures, using saved outputs."""
from run_analysis import *
from scipy.stats import spearmanr

pred=pd.read_csv(RES/'misbot_test_predictions.csv');rng=np.random.default_rng(SEED)
intervals=[]
for name in ['Text-style LR','Metadata LR','Metadata RF']:
    auc=[]
    for _ in range(300):
        ii=rng.integers(0,len(pred),len(pred));auc.append(roc_auc_score(pred.label.iloc[ii],pred[name].iloc[ii]))
    intervals.append({'model':name,'bootstrap_replicates':300,'auc_q025':np.quantile(auc,.025),'auc_q975':np.quantile(auc,.975)})
pd.DataFrame(intervals).to_csv(RES/'misbot_auc_intervals.csv',index=False)

f=pd.read_csv(RES/'topic_features.csv');seq=pd.read_csv(RES/'temporal_motifs.csv')
seq=seq.merge(f[['topic','cluster']],on='topic',validate='many_to_one')
counts=pd.crosstab(seq.cluster,seq.motif);prop=counts.div(counts.sum(axis=1),axis=0)
prop.to_csv(RES/'motif_by_cluster.csv')
top=seq.motif.value_counts().head(6).index
fig,ax=plt.subplots(figsize=(7.8,4.4));a=prop[top];im=ax.pcolormesh(np.arange(len(top)+1)-.5,np.arange(len(a)+1)-.5,a,cmap='YlGnBu',vmin=0,vmax=.6)
ax.invert_yaxis()
ax.set_xticks(range(len(top)),top);ax.set_yticks(range(len(a)),[f'C{i+1}' for i in a.index]);ax.set(xlabel='Six-hour activity motif',ylabel='Topic cluster')
for i in range(len(a)):
    for j in range(len(top)):ax.text(j,i,f'{a.iloc[i,j]:.0%}',ha='center',va='center',fontsize=11,color='white' if a.iloc[i,j]>.35 else '#24364D')
cb=fig.colorbar(im,ax=ax,fraction=.035,pad=.04);cb.solids.set_rasterized(False);save(fig,'08_motifs_by_cluster')

d=load_weibo();counts=d.groupby(d.time.dt.to_period('M')).size();fig,ax=plt.subplots(figsize=(8,3.6))
ax.bar(range(len(counts)),counts,color=COLORS[0],width=.75)
ax.set_xticks(list(range(0,len(counts),6)),[str(counts.index[i]) for i in range(0,len(counts),6)])
ax.set(xlabel='Month',ylabel='Posts in the public sample');save(fig,'09_data_coverage')

nd=pd.read_csv(RES/'synchrony_null_test.csv');fig,ax=plt.subplots(figsize=(7.8,4.2))
ax.scatter(nd.null_mean,nd.observed,c=[COLORS[3] if q<.05 else COLORS[0] for q in nd.q_bh],s=35)
v=max(nd.null_mean.max(),nd.observed.max())*1.08;ax.plot([0,v],[0,v],ls='--',c='#9AABBB',lw=1)
ax.set(xlabel='Mean same-minute fraction under within-day null',ylabel='Observed same-minute fraction',xlim=(0,v),ylim=(0,v))
ax.set_title(f'{sum(nd.q_bh<.05)}/{len(nd)} topics at BH q < 0.05 (999 draws)',loc='left',weight='bold',pad=13)
save(fig,'10_synchrony_null')
dump('additional_summary.json',{'hmi_style_spearman':spearmanr(f.hmi,f.bot_style).statistic,
    'null_test_topics':len(nd),'null_draws':999,'significant_bh_005':int(sum(nd.q_bh<.05)),
    'unadjusted_p_005':int(sum(nd.p_upper<.05)),'min_q':nd.q_bh.min(),
    'motif_cluster_window_counts':counts.to_dict() if False else seq.groupby('cluster').size().to_dict()})
print('Additional analysis complete')

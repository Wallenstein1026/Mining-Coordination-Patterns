"""Proposal-aligned hotspot growth, CIB-informed ablations and HMI pattern contexts.

This is retrospective sample analysis, not a ground-truth CIB or early-warning test.
Run after run_analysis.py. No raw posts or user identifiers are exported here.
"""
from run_analysis import *


def main():
    f=pd.read_csv(RES/'topic_features.csv')
    d=load_weibo()
    # Local model is created by run_analysis.py; never load an untrusted pickle.
    with (RES/'text_style_model.pkl').open('rb') as handle:model=pickle.load(handle)
    d['bot_style']=model.predict_proba(np.array([style(s) for s in d.content]))[:,1]
    calendar=pd.date_range(d.time.min().floor('D'),d.time.max().floor('D'),freq='D')
    totals=d.groupby(d.time.dt.floor('D')).size().reindex(calendar,fill_value=0)
    prior_totals=totals.shift(1).rolling(7,min_periods=7).sum()
    rows=[];series={};user_scores={}
    for r in f.itertuples():
        g=d[d.tags.map(lambda tags:r.topic in tags)]
        user_scores[r.topic]=user_balanced_style(g)
        counts=g.groupby(g.time.dt.floor('D')).size().reindex(calendar,fill_value=0)
        users=g.groupby(g.time.dt.floor('D')).user_id.nunique().reindex(calendar,fill_value=0)
        prior=counts.shift(1).rolling(7,min_periods=7).sum()
        # +1 smoothing in both sample-share estimates. Past-day reference only.
        ratio=((counts+1)/(totals+1))/((prior+1)/(prior_totals+1))
        expected=totals*(prior+1)/(prior_totals+1)
        series[r.topic]=(counts,expected)
        for day in calendar[counts>0]:
            eligible=bool(pd.notna(prior[day]) and prior_totals[day]>=20)
            hotspot=eligible and counts[day]>=4 and users[day]>=3 and ratio[day]>=2
            rows.append({'topic':r.topic,'date':str(day.date()),'posts':int(counts[day]),
                'users':int(users[day]),'sample_posts_on_day':int(totals[day]),
                'previous_7d_posts':float(prior[day]),'previous_7d_sample_posts':float(prior_totals[day]),
                'share_growth_ratio':float(ratio[day]),'expected_posts_from_past_share':float(expected[day]),
                'history_eligible':eligible,'hotspot':bool(hotspot)})
    daily=pd.DataFrame(rows)
    daily.to_csv(RES/'topic_day_growth.csv',index=False)
    hot=daily[daily.hotspot].copy()
    hot.to_csv(RES/'hotspot_windows.csv',index=False)
    summary=daily.groupby('topic').agg(hotspot_days=('hotspot','sum'),peak_day_posts=('posts','max'),
        maximum_share_growth=('share_growth_ratio','max')).reset_index()
    summary.to_csv(RES/'topic_hotspot_summary.csv',index=False)
    f=f.merge(summary,on='topic',validate='one_to_one')
    f['user_balanced_style']=f.topic.map(user_scores)

    # Two deterministic explanatory case studies; source names retained separately.
    aliases={'广州 疫情':'Guangzhou COVID updates','如果阳了怎么办':'After testing positive'}
    cases=[]
    fig,axes=plt.subplots(1,2,figsize=(8.8,3.7),layout='constrained')
    for ax,(topic,alias) in zip(axes,aliases.items()):
        event=hot[hot.topic==topic].sort_values(['posts','date'],ascending=[False,True]).iloc[0]
        peak=pd.Timestamp(event.date);days=pd.date_range(peak-pd.Timedelta(days=7),peak+pd.Timedelta(days=7))
        counts,expected=series[topic]
        ax.bar(np.arange(-7,8),counts.reindex(days,fill_value=0),color=COLORS[0],width=.75)
        ax.plot(np.arange(-7,8),expected.reindex(days),ls='--',color=COLORS[1],lw=1.7)
        ax.axvline(0,color='#B5C2CD',lw=.8)
        rr=f.loc[f.topic==topic].iloc[0]
        ax.set_title(f'{alias}\n{peak.date()} | HMI {rr.hmi:.2f}',loc='left',fontsize=13,weight='bold')
        ax.set_ylabel('Posts',fontsize=12);ax.tick_params(labelsize=11)
        ax.set_xlabel('Days from hotspot',fontsize=12);ax.set_xticks([-7,0,7])
        cases.append({'topic':topic,'english':alias,'date':str(peak.date()),'posts':int(event.posts),
            'users':int(event.users),'share_growth_ratio':float(event.share_growth_ratio),'hmi':float(rr.hmi),
            'style_score':float(rr.bot_style)})
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=COLORS[0],label='Observed sample posts'),
        Line2D([0],[0],color=COLORS[1],ls='--',label='Expected from preceding 7-day share')],
        frameon=False,fontsize=10,loc='outside lower center',ncol=2)
    save(fig,'11_hotspot_growth')
    dump('hotspot_cases.json',cases)

    # Explicitly incorporate HMI plus bot-style evidence, with transparent ablations.
    X=np.load(RES/'topic_representations.npz')['fused']
    base_labels=f.cluster.to_numpy();z=StandardScaler().fit_transform(f[['hmi','user_balanced_style']])
    k=int(load_json('selected_clustering.json')['k'])
    ablations=[]
    for name,idx,weight in [('Fused baseline',[],0),('Fused + HMI',[0],.5),
                            ('Fused + user-balanced style',[1],.5),('CIB-informed w=0.25',[0,1],.25),
                            ('CIB-informed w=0.50',[0,1],.5),('CIB-informed w=1.00',[0,1],1.)]:
        xx=X if not idx else np.c_[X,weight*z[:,idx]/np.sqrt(len(idx))]
        labels=base_labels if not idx else KMeans(k,n_init=20,random_state=SEED).fit_predict(xx)
        ablations.append({'representation':name,'auxiliary_weight':weight,'k':k,
            'common_fused_silhouette':silhouette_score(X,labels),
            'native_silhouette':silhouette_score(xx,labels),'ari_vs_baseline':adjusted_rand_score(base_labels,labels),
            'minimum_cluster_size':int(np.bincount(labels).min())})
        if name=='CIB-informed w=0.50':f['cib_informed_cluster']=labels
    pd.DataFrame(ablations).to_csv(RES/'cib_informed_ablation.csv',index=False)
    # This augmented partition is a proposal-aligned exploratory view, not selected as best.
    cols=[c+'_rank' for c in ['duplicate','user_hhi','peak_day','same_minute']]+['hmi','bot_style','user_balanced_style']
    agg=f.groupby('cib_informed_cluster')[cols].mean()
    agg.to_csv(RES/'cib_informed_cluster_summary.csv')
    fig,ax=plt.subplots(figsize=(8.5,4.4),layout='constrained')
    im=ax.pcolormesh(np.arange(8)-.5,np.arange(len(agg)+1)-.5,agg.to_numpy(),cmap='YlGnBu',vmin=0,vmax=1)
    ax.invert_yaxis()
    ax.set_xticks(range(7),['Duplicate\nrank','User HHI\nrank','Peak-day\nrank','Same-minute\nrank','HMI','Post-style\nscore','User-balanced\nstyle'],fontsize=11)
    ax.set_yticks(range(len(agg)),[f'A{x+1}' for x in agg.index],fontsize=13)
    for i in range(len(agg)):
        for j in range(7):ax.text(j,i,f'{agg.iloc[i,j]:.2f}',ha='center',va='center',fontsize=14,
            color='white' if agg.iloc[i,j]>.6 else '#24364D')
    cb=fig.colorbar(im,ax=ax,fraction=.035,pad=.03);cb.solids.set_rasterized(False)
    save(fig,'12_cib_informed_profiles')
    f.to_csv(RES/'proposal_topic_profiles.csv',index=False)

    # Contextual pattern mining: topic-weighted motif shares in HMI strata and growth contexts.
    seq=pd.read_csv(RES/'temporal_motifs.csv').merge(f[['topic','hmi']],on='topic',validate='many_to_one')
    q1,q3=f.hmi.quantile([.25,.75]);seq['hmi_group']=np.where(seq.hmi<=q1,'Low HMI',np.where(seq.hmi>=q3,'High HMI','Middle'))
    seq=seq.merge(daily[['topic','date','hotspot']],on=['topic','date'],validate='one_to_one')
    seq.to_csv(RES/'contextual_motif_windows.csv',index=False)
    motif_counts=pd.crosstab(seq.topic,seq.motif)
    shares=motif_counts.div(motif_counts.sum(axis=1),axis=0)
    meta=f.set_index('topic')
    top=seq.motif.value_counts().head(4).index.tolist();contexts=[]
    for group,mask in [('Low HMI',meta.hmi<=q1),('High HMI',meta.hmi>=q3)]:
        names=meta.index[mask].intersection(shares.index)
        for motif in shares.columns:
            sub=seq[(seq.topic.isin(names)) & (seq.motif==motif)]
            contexts.append({'context':group,'motif':motif,'topic_weighted_share':float(shares.loc[names,motif].mean()),
                'topics_with_eligible_windows':len(names),'motif_windows':len(sub),
                'eligible_windows':int(seq.topic.isin(names).sum())})
    for name,mask in [('Hotspot day',seq.hotspot),('Other active day',~seq.hotspot)]:
        sub=seq[mask];counts=pd.crosstab(sub.topic,sub.motif).reindex(columns=shares.columns,fill_value=0)
        props=counts.div(counts.sum(axis=1),axis=0)
        for motif in props.columns:
            contexts.append({'context':name,'motif':motif,'topic_weighted_share':float(props[motif].mean()),
                'topics_with_eligible_windows':len(props),'motif_windows':int((sub.motif==motif).sum()),'eligible_windows':len(sub)})
    contexts=pd.DataFrame(contexts);contexts.to_csv(RES/'contextual_motif_comparison.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.4),layout='constrained',sharey=True)
    for ax,groups,title in zip(axes,[['Low HMI','High HMI'],['Other active day','Hotspot day']],['HMI strata','Sample growth context']):
        for j,group in enumerate(groups):
            vals=contexts[contexts.context==group].set_index('motif').topic_weighted_share.reindex(top)
            ax.bar(np.arange(len(top))+(j-.5)*.36,vals,width=.34,color=COLORS[j],label=group)
        ax.set_xticks(range(len(top)),top,fontsize=11);ax.set_title(title,loc='left',weight='bold',fontsize=13)
        ax.legend(frameon=False,fontsize=10);ax.set_ylim(0,.65);ax.tick_params(axis='y',labelsize=11)
    axes[0].set_ylabel('Mean within-topic motif share',fontsize=12)
    save(fig,'13_hmi_context_patterns')

    # Recurring tag associations are also compared by context, with explicit overlap.
    eligible=set(f.topic);group_sets={'Low HMI':set(f.loc[f.hmi<=q1,'topic']),
                                    'High HMI':set(f.loc[f.hmi>=q3,'topic'])}
    pattern_rows=[];group_sizes={};post_sets={}
    for name,tags in group_sets.items():
        idx=[i for i,tt in enumerate(d.tags) if set(tt)&tags];post_sets[name]=set(idx)
        trans=[sorted(set(d.iloc[i].tags)&eligible) for i in idx]
        te=TransactionEncoder();mat=te.fit(trans).transform(trans)
        itemsets=fpgrowth(pd.DataFrame(mat,columns=te.columns_),min_support=.01,use_colnames=True,max_len=2)
        pairs=itemsets[itemsets.itemsets.map(len)==2].sort_values('support',ascending=False)
        group_sizes[name]=len(idx)
        for row in pairs.head(10).itertuples():
            pattern_rows.append({'context':name,'itemsets':' + '.join(sorted(row.itemsets)),
                'count':round(row.support*len(idx)),'support_in_context':row.support,'context_posts':len(idx)})
    pd.DataFrame(pattern_rows).to_csv(RES/'hmi_context_itemsets.csv',index=False)
    aug=next(r for r in ablations if r['representation']=='CIB-informed w=0.50')
    dump('proposal_summary.json',{'hotspot_windows':len(hot),'hotspot_topics':hot.topic.nunique(),
        'active_topic_days':len(daily),'hotspot_min_posts':4,'hotspot_min_users':3,'hotspot_min_growth':2,
        'growth_definition':'[(n_td+1)/(N_d+1)] / [(sum preceding 7d n_t+1)/(sum preceding 7d N+1)]',
        'growth_history_min_sample_posts':20,'hmi_q25':q1,'hmi_q75':q3,
        'cib_informed_weight':.5,'cib_informed_common_silhouette':aug['common_fused_silhouette'],
        'cib_informed_ari_vs_baseline':aug['ari_vs_baseline'],
        'auxiliary_aggregation':'Mean post scores per user within topic, then equal-weight mean over users; post-weighted score also saved.',
        'hmi_itemset_context_posts':group_sizes,'hmi_context_shared_posts':len(post_sets['Low HMI']&post_sets['High HMI']),
        'hmi_style_spearman':float(f.hmi.corr(f.bot_style,method='spearman')),
        'caveats':['Hotspot eligibility and HMI use the full observed period: retrospective, not validated online detection.',
                   'Day-level growth uses only the preceding 7 days, but topic screening uses all dates.',
                   'HMI strata share features with clustering and can overlap in member posts; no independent CIB validation.',
                   'CIB-informed features contain bot-style auxiliary evidence, not ground-truth CIB confidence.',
                   'This analysis does not reconstruct propagation edges; relationship semantics and coverage require a separate audit.']})
    print('Proposal-aligned analysis complete:',len(hot),'hotspot windows;',hot.topic.nunique(),'topics',flush=True)


def load_json(name):
    return json.loads((RES/name).read_text())


if __name__=='__main__':main()

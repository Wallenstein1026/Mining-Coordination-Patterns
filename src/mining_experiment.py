"""Retrospective clustering, HMI ablations and pattern mining for Chang's experiment."""
import collections
import itertools
import json
import re
import time

from run_analysis import CJK, COLORS, SEED, plt, style
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from scipy.stats import rankdata
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler, normalize


def build_topics(d, output, model=None):
    print('2. Topic features',flush=True)
    if model is not None:
        d['bot_style']=model.predict_proba(np.array([style(s) for s in d.content]))[:,1]
    # Broad collection tags are omitted from the unit definition, but all raw posts remain in mining denominators.
    broad={'疫情','新冠','新冠肺炎','新型冠状病毒','新型冠状病毒感染肺炎','肺炎[超话]','疫情[超话]','covid-19','新冠病毒'}
    membership=collections.defaultdict(list)
    for i,tags in enumerate(d.tags):
        for tag in tags:
            if tag not in broad:membership[tag].append(i)
    membership={t:ii for t,ii in sorted(membership.items()) if len(ii)>=30 and d.iloc[ii].user_id.nunique()>=5}
    rows=[];docs=[];profiles=[]
    for tag,ii in membership.items():
        g=d.iloc[ii];n=len(g);days=g.time.dt.floor('D').value_counts();mins=g.time.dt.floor('min').value_counts();users=g.user_id.value_counts()
        duplicate=1-g.norm.nunique()/n;concentration=float(np.sum((users/n)**2));burst=float(days.max()/n)
        # Fraction of posts sharing a clock-minute with at least one other post, not a causal coordination test.
        sync=float(mins[mins>=2].sum()/n)
        hist=np.bincount(g.time.dt.hour,minlength=24)/n
        elapsed=(g.time-g.time.min()).dt.total_seconds().to_numpy();maxe=max(float(elapsed.max()),1)
        life=np.histogram(elapsed/maxe,bins=np.linspace(0,1,11))[0]/n
        row={'topic':tag,'posts':n,'users':len(users),'days':len(days),'span_days':maxe/86400,
            'duplicate':duplicate,'user_hhi':concentration,'peak_day':burst,'same_minute':sync,
            'night_share':float(hist[:6].sum()),'repost_mark':float(g.content.str.contains('//@|转发微博',regex=True).mean()),
            'hour_entropy':float(-np.sum(hist[hist>0]*np.log(hist[hist>0]))/np.log(24))}
        if model is not None:
            row['bot_style']=float(g.bot_style.mean())
            row['user_balanced_style']=float(g.groupby('user_id').bot_style.mean().mean())
        rows.append(row);profiles.append(np.r_[hist,life])
        # Remove topic hashtag strings from text so the exact topic label does not dominate text similarity.
        docs.append(' '.join(re.sub(r'#[^#\n]{2,70}#',' ',str(s)) for s in g.content))
    f=pd.DataFrame(rows)
    if len(f)<4:
        raise ValueError('Need at least four topics with 30 posts and five users to compare clusters.')
    for col in ['duplicate','user_hhi','peak_day','same_minute']:
        # Treat floating-point-equivalent fractions as ties, including CSV round trips.
        f[col+'_rank']=rankdata(f[col].round(12),method='average')/len(f)
    rankcols=[c+'_rank' for c in ['duplicate','user_hhi','peak_day','same_minute']]
    f['hmi']=f[rankcols].mean(axis=1)
    tfidf=TfidfVectorizer(analyzer='char',ngram_range=(2,3),min_df=3,max_df=.9,max_features=16000,sublinear_tf=True)
    tx=tfidf.fit_transform(docs);svd=TruncatedSVD(n_components=16,random_state=SEED)
    text=normalize(svd.fit_transform(tx))
    behaviorcols=['duplicate','user_hhi','peak_day','same_minute','night_share','repost_mark','hour_entropy']
    beh=normalize(StandardScaler().fit_transform(np.c_[f[behaviorcols],np.array(profiles)]))
    # Auxiliary score is kept out of HMI and the primary clustering representation.
    fused=np.c_[text/np.sqrt(2),beh/np.sqrt(2)]
    f.to_csv(output.results/'topic_features.csv',index=False)
    np.savez_compressed(output.results/'topic_representations.npz',text=text,behavior=beh,fused=fused,profiles=np.array(profiles))
    output.write_json('topic_design.json',{'n_topics':len(f),'min_posts':30,'min_users':5,'excluded_broad_tags':sorted(broad),
        'memberships':sum(len(ii) for ii in membership.values()),'distinct_posts':len(set(itertools.chain.from_iterable(membership.values()))),
        'svd_explained_variance':float(svd.explained_variance_ratio_.sum()),'behavior_columns':behaviorcols,
        'hmi':'Mean within-sample percentile ranks of duplicate, user_hhi, peak_day, same_minute. Not a manipulation probability.'})
    return f,{'Text':text,'Behavior':beh,'Fused':fused},membership,rankcols

def cluster_labels(features, method, k, seed=SEED):
    if method=='KMeans':
        return KMeans(k,n_init=20,random_state=seed).fit_predict(features)
    return AgglomerativeClustering(k,linkage='ward').fit_predict(features)


def subsample_stability(features, labels, method, k):
    rng=np.random.default_rng(SEED)
    n=max(k,int(.8*len(labels)))
    stability=[]
    for run in range(20):
        indices=np.sort(rng.choice(len(labels),n,replace=False))
        refit=cluster_labels(features[indices],method,k,SEED+run)
        stability.append(adjusted_rand_score(labels[indices],refit))
    return stability


def cluster_topics(f, reps, output):
    print('3. Clustering comparisons',flush=True)
    rows=[];partitions={}
    for rep,X in reps.items():
        for method in ['KMeans','Ward']:
            for k in range(3,min(9,len(f))):
                for seed in ([28,29,30,31,32] if method=='KMeans' else [28]):
                    t=time.perf_counter()
                    labels=cluster_labels(X,method,k,seed)
                    key=(rep,method,k,seed);partitions[key]=labels
                    rows.append(dict(representation=rep,method=method,k=k,seed=seed,
                        native_silhouette=silhouette_score(X,labels),common_fused_silhouette=silhouette_score(reps['Fused'],labels),
                        common_davies_bouldin=davies_bouldin_score(reps['Fused'],labels),seconds=time.perf_counter()-t,
                        min_cluster_size=int(np.bincount(labels).min())))
    metrics_df=pd.DataFrame(rows);metrics_df.to_csv(output.results/'clustering_grid.csv',index=False)
    # Select among fused candidates only; no CIB or HMI labels participate in the selection.
    candidate=metrics_df[(metrics_df.representation=='Fused') & (metrics_df.seed==SEED) & (metrics_df.min_cluster_size>=10)]
    if candidate.empty:
        raise ValueError('No fused clustering meets the minimum size of ten topics per cluster.')
    best=candidate.sort_values('native_silhouette',ascending=False).iloc[0];key=('Fused',best.method,int(best.k),SEED)
    labels=partitions[key];f['cluster']=labels
    X=reps['Fused']
    stability_rows=[]
    for (rep,method,k,seed),partition in partitions.items():
        if k!=best.k or seed!=SEED:
            continue
        values=subsample_stability(reps[rep],partition,method,k)
        if (rep,method,k,seed)==key:
            stability=values
        stability_rows.append({'representation':rep,'method':method,'k':k,'runs':len(values),
                               'ari_mean':np.mean(values),'ari_std':np.std(values),'ari_min':min(values)})
    pd.DataFrame(stability_rows).to_csv(output.results/'cluster_stability_comparison.csv',index=False)
    # Size threshold sensitivity, refitting the selected model on higher-volume subsets.
    sens=[]
    for n in [40,50,75]:
        ii=np.flatnonzero(f.posts>=n)
        if len(ii)>int(best.k)*3:
            ll=cluster_labels(X[ii],best.method,int(best.k))
            sens.append({'min_posts':n,'n_topics':len(ii),'ari_vs_full':adjusted_rand_score(labels[ii],ll)})
    f.to_csv(output.results/'topic_features.csv',index=False)
    pd.DataFrame({'run':range(20),'ari':stability}).to_csv(output.results/'cluster_stability.csv',index=False)
    output.write_json('selected_clustering.json',{'method':best.method,'k':int(best.k),'silhouette':best.native_silhouette,
        'cluster_sizes':dict(collections.Counter(map(str,labels))),'stability_mean':np.mean(stability),'stability_std':np.std(stability),
        'stability_min':min(stability),'size_sensitivity':sens,'selection':'Best seed-28 fused silhouette over k=3..8 and KMeans/Ward, with min cluster size >=10',
        'caveat':'Descriptive transductive clustering; silhouette measures compactness, not semantic or CIB validity.'})
    coords=PCA(n_components=2,random_state=SEED).fit_transform(X)
    fig,ax=plt.subplots(figsize=(7.8,4.4))
    for c in sorted(set(labels)):
        m=labels==c;ax.scatter(coords[m,0],coords[m,1],s=18+20*np.log1p(f.posts[m]/30),c=COLORS[c%len(COLORS)],alpha=.75,edgecolors='white',linewidths=.3,label=f'C{c+1}  (n={m.sum()})')
    ax.set(xlabel='Principal component 1',ylabel='Principal component 2');ax.legend(frameon=False,fontsize=10)
    ax.set_title('Topic clusters in a 2D PCA projection',loc='left',weight='bold',pad=14)
    output.save_figure(fig,'02_cluster_map')
    fig,ax=plt.subplots(figsize=(7.8,4.4))
    for j,((rep,method),g) in enumerate(metrics_df.groupby(['representation','method'])):
        a=g.groupby('k').common_fused_silhouette.agg(['mean','std']).fillna(0)
        ax.plot(a.index,a['mean'],marker='o' if method=='KMeans' else 's',ls='-' if method=='KMeans' else '--',lw=2,color=COLORS[j],label=f'{rep}, {method}')
    ax.set(xlabel='Number of clusters k',ylabel='Silhouette in common fused space',xticks=list(range(3,9)))
    ax.legend(frameon=False,ncol=2,fontsize=9);ax.axhline(0,color='#CCD4DD',lw=1)
    output.save_figure(fig,'03_method_comparison')
    return labels

def transaction_frame(transactions):
    encoder=TransactionEncoder().fit(transactions)
    return pd.DataFrame.sparse.from_spmatrix(encoder.transform(transactions,sparse=True),columns=encoder.columns_)


def mine_patterns(d, f, membership, output):
    print('4. Frequent itemsets and time motifs',flush=True)
    eligible=set(f.topic);trans=[sorted(set(tags)&eligible) for tags in d.tags]
    transactions=transaction_frame(trans)
    bench=[];allsets=None
    for support in [.003,.005,.01]:
        sets=[]
        for name,fn in [('Apriori',apriori),('FP-growth',fpgrowth)]:
            t=time.perf_counter()
            options={'low_memory':True} if name=='Apriori' else {}
            itemsets=fn(transactions,min_support=support,use_colnames=True,max_len=3,**options)
            bench.append({'method':name,'min_support':support,'itemsets':len(itemsets),'seconds':time.perf_counter()-t})
            sets.append(set(itemsets.itemsets))
            if name=='FP-growth' and support==.003:allsets=itemsets
        assert sets[0]==sets[1], 'Frequent-itemset methods disagree'
    pd.DataFrame(bench).to_csv(output.results/'pattern_method_comparison.csv',index=False)
    supports={frozenset(r.itemsets):r.support for r in allsets.itertuples()};rules=[]
    for itemset,support in supports.items():
        if len(itemset)<2:continue
        for k in range(1,len(itemset)):
            for lhs in itertools.combinations(sorted(itemset),k):
                lhs=frozenset(lhs);rhs=itemset-lhs;confidence=support/supports[lhs];lift=confidence/supports[rhs]
                if confidence>=.5 and lift>=1.5:
                    rules.append({'antecedent':' + '.join(sorted(lhs)),'consequent':' + '.join(sorted(rhs)),
                        'support':support,'count':int(round(support*len(d))),'confidence':confidence,'lift':lift})
    rules=pd.DataFrame(rules,columns=['antecedent','consequent','support','count','confidence','lift']).sort_values(['support','lift'],ascending=False)
    rules.to_csv(output.results/'association_rules.csv',index=False)
    out=allsets.copy();out['itemsets']=out.itemsets.map(lambda x:' + '.join(sorted(x)));out.to_csv(output.results/'frequent_itemsets.csv',index=False)
    # 24-hour topic windows: require >=4 posts, discretize 4 six-hour counts at each window's median.
    sequences=[];curves={};topic_motif=collections.defaultdict(collections.Counter)
    for tag,ii in membership.items():
        g=d.iloc[ii]
        for day,dayg in g.groupby(g.time.dt.floor('D')):
            if len(dayg)<4:continue
            bins=np.bincount((dayg.time.dt.hour//6).astype(int),minlength=4)
            pattern=''.join('H' if v>np.median(bins) else 'L' for v in bins)
            sequences.append({'topic':tag,'date':str(day.date()),'n':len(dayg),'motif':pattern})
            topic_motif[pattern][tag]+=1
            curves.setdefault(pattern,[]).append(bins/bins.sum())
    seq=pd.DataFrame(sequences,columns=['topic','date','n','motif']);seq.to_csv(output.results/'temporal_motifs.csv',index=False)
    motif_counts=seq.motif.value_counts()
    motifrows=[{'motif':m,'windows':int(n),'support':n/len(seq),'distinct_topics':len(topic_motif[m])} for m,n in motif_counts.items()]
    pd.DataFrame(motifrows,columns=['motif','windows','support','distinct_topics']).to_csv(output.results/'temporal_motif_summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(7.8,4.4))
    for j,m in enumerate(motif_counts.head(4).index):
        a=np.array(curves[m]);mu=a.mean(axis=0)
        ax.plot([3,9,15,21],mu,marker='o',lw=2.5,c=COLORS[j],label=f'{m}: {len(a)} windows')
    ax.set(xlabel='Clock hour (Beijing time)',ylabel='Mean fraction of daily topic posts',xticks=[3,9,15,21],ylim=(0,1))
    if curves:
        ax.legend(frameon=False,fontsize=10)
    ax.set_title('Recurring six-hour activity motifs',loc='left',weight='bold',pad=14)
    output.save_figure(fig,'05_temporal_motifs')
    # Draw the most frequent pair itemsets as a weighted co-occurrence graph.
    pairs=[(tuple(sorted(k)),v) for k,v in supports.items() if len(k)==2]
    pairs=sorted(pairs,key=lambda x:x[1],reverse=True)[:18]
    import networkx as nx
    G=nx.Graph()
    for (a,b),s in pairs:G.add_edge(a,b,weight=s*len(d))
    fig,axes=plt.subplots(1,2,figsize=(8.5,4.3),gridspec_kw={'width_ratios':[1,1.8]})
    components=sorted(nx.connected_components(G),key=len,reverse=True)
    for ax in axes:
        ax.axis('off')
    if not components:
        axes[0].text(.5,.5,'No pairs at this support threshold',transform=axes[0].transAxes,ha='center')
    for j,(ax,nodes) in enumerate(zip(axes,components)):
        sub=G.subgraph(nodes);pos=nx.circular_layout(sub)
        nx.draw_networkx_edges(sub,pos,ax=ax,width=[1+sub[u][v]['weight']/130 for u,v in sub.edges],edge_color='#B2C7D5',alpha=.65)
        nx.draw_networkx_nodes(sub,pos,ax=ax,node_color=COLORS[j],node_size=270,alpha=.9)
        for index,(n,(x,y)) in enumerate(pos.items(),1):
            ax.text(x*1.28,y*1.28,n if CJK is not None else f'Topic {index}',fontproperties=CJK,fontsize=8.5,ha='center',va='center',
                bbox={'facecolor':'white','alpha':.9,'edgecolor':'none','pad':1})
        ax.set_xlim(-1.9,1.9);ax.set_ylim(-1.55,1.55);ax.axis('off')
        ax.set_title(f'Co-occurrence component {j+1}',fontsize=12,weight='bold')
    output.save_figure(fig,'06_hashtag_network')
    output.write_json('pattern_summary.json',{'transaction_denominator':len(d),'eligible_hashtags':len(eligible),'min_support':.003,
        'itemsets':len(allsets),'rules':len(rules),'rules_min_confidence':.5,'rules_min_lift':1.5,
        'temporal_windows':len(seq),'top_motifs':motifrows[:5],
        'caveat':'Hashtag co-occurrence is an association pattern. This co-occurrence analysis does not reconstruct repost edges; relationship fields need separate auditing.'})
    return rules

def describe_hotspots(d, membership, output):
    f=pd.read_csv(output.results/'topic_features.csv',dtype={'topic':str},keep_default_na=False)
    calendar=pd.date_range(d.time.min().floor('D'),d.time.max().floor('D'),freq='D')
    totals=d.groupby(d.time.dt.floor('D')).size().reindex(calendar,fill_value=0)
    prior_totals=totals.shift(1).rolling(7,min_periods=7).sum()
    rows=[];series={}
    for r in f.itertuples():
        g=d.iloc[membership[r.topic]]
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
    daily.to_csv(output.results/'topic_day_growth.csv',index=False)
    hot=daily[daily.hotspot].copy()
    hot.to_csv(output.results/'hotspot_windows.csv',index=False)
    summary=daily.groupby('topic').agg(hotspot_days=('hotspot','sum'),peak_day_posts=('posts','max'),
        maximum_share_growth=('share_growth_ratio','max')).reset_index()
    summary.to_csv(output.results/'topic_hotspot_summary.csv',index=False)
    f=f.merge(summary,on='topic',validate='one_to_one')
    has_style='user_balanced_style' in f

    # Select disjoint groups with deterministic tie breaking.
    ranked=f.sort_values(['hmi','topic'])
    if len(ranked)<6:
        raise ValueError('Need six topics to review three distinct cases from each HMI tail.')
    tails=pd.concat([ranked.head(3).assign(hmi_group='Low HMI'),
                     ranked.tail(3).iloc[::-1].assign(hmi_group='High HMI')])
    cases=[]
    fig,axes=plt.subplots(2,3,figsize=(11,5.4),layout='constrained')
    for index,(ax,case) in enumerate(zip(axes.flat,tails.itertuples())):
        topic=case.topic
        event=daily[daily.topic==topic].sort_values(
            ['hotspot','posts','date'],ascending=[False,False,True]).iloc[0]
        peak=pd.Timestamp(event.date);days=pd.date_range(peak-pd.Timedelta(days=7),peak+pd.Timedelta(days=7))
        counts,expected=series[topic]
        ax.bar(np.arange(-7,8),counts.reindex(days),color=COLORS[0],width=.75)
        ax.plot(np.arange(-7,8),expected.reindex(days),ls='--',color=COLORS[1],lw=1.7)
        ax.axvline(0,color='#B5C2CD',lw=.8)
        alias=f'{case.hmi_group} {index%3+1}'
        ax.set_title(f'{alias}\n{peak.date()} | HMI {case.hmi:.2f}',loc='left',fontsize=10,weight='bold')
        ax.set_ylabel('Posts',fontsize=12);ax.tick_params(labelsize=11)
        ax.set_xlabel('Days from selected event',fontsize=9);ax.set_xticks([-7,0,7])
        cases.append({'topic':topic,'case':alias,'hmi_group':case.hmi_group,'date':str(peak.date()),
            'posts':int(event.posts),'users':int(event.users),'hotspot':bool(event.hotspot),
            'share_growth_ratio':float(event.share_growth_ratio) if event.history_eligible else None,
            'hmi':float(case.hmi),'duplicate':case.duplicate,'user_hhi':case.user_hhi,
            'peak_day':case.peak_day,'same_minute':case.same_minute,
            'user_balanced_style':case.user_balanced_style if has_style else None})
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=COLORS[0],label='Observed sample posts'),
        Line2D([0],[0],color=COLORS[1],ls='--',label='Expected from preceding 7-day share')],
        frameon=False,fontsize=10,loc='outside lower center',ncol=2)
    output.save_figure(fig,'11_hotspot_growth')
    output.write_json('hotspot_cases.json',cases)

    # Explicitly incorporate HMI plus bot-style evidence, with transparent ablations.
    X=np.load(output.results/'topic_representations.npz')['fused']
    base_labels=f.cluster.to_numpy()
    config=json.loads((output.results/'selected_clustering.json').read_text(encoding='utf-8'))
    method,k=config['method'],int(config['k'])
    signals=['hmi','user_balanced_style'] if has_style else ['hmi']
    z=StandardScaler().fit_transform(f[signals])
    specs=[('Fused baseline',[],0),('Fused + HMI',[0],.5)]
    if has_style:
        specs += [('Fused + user-balanced style',[1],.5), *[
            (f'CIB-informed w={weight:.2f}',[0,1],weight) for weight in (.25,.5,1.)]]
    profile='CIB-informed w=0.50' if has_style else 'Fused + HMI'
    ablations=[]
    for name,idx,weight in specs:
        xx=X if not idx else np.c_[X,weight*z[:,idx]/np.sqrt(len(idx))]
        labels=base_labels if not idx else cluster_labels(xx,method,k)
        stability=subsample_stability(xx,labels,method,k)
        ablations.append({'representation':name,'auxiliary_weight':weight,'method':method,'k':k,
            'common_fused_silhouette':silhouette_score(X,labels),
            'native_silhouette':silhouette_score(xx,labels),'ari_vs_baseline':adjusted_rand_score(base_labels,labels),
            'subsample_ari_mean':np.mean(stability),'subsample_ari_std':np.std(stability),
            'minimum_cluster_size':int(np.bincount(labels).min())})
        if name==profile:f['cib_informed_cluster']=labels
    pd.DataFrame(ablations).to_csv(output.results/'cib_informed_ablation.csv',index=False)
    # This augmented partition is a proposal-aligned exploratory view, not selected as best.
    cols=[c+'_rank' for c in ['duplicate','user_hhi','peak_day','same_minute']]+['hmi']
    labels=['Duplicate\nrank','User HHI\nrank','Peak-day\nrank','Same-minute\nrank','HMI']
    if has_style:
        cols+=['bot_style','user_balanced_style']
        labels+=['Post-style\nscore','User-balanced\nstyle']
    agg=f.groupby('cib_informed_cluster')[cols].mean()
    agg.to_csv(output.results/'cib_informed_cluster_summary.csv')
    fig,ax=plt.subplots(figsize=(8.5,4.4),layout='constrained')
    im=ax.pcolormesh(np.arange(len(cols)+1)-.5,np.arange(len(agg)+1)-.5,agg.to_numpy(),cmap='YlGnBu',vmin=0,vmax=1)
    ax.invert_yaxis()
    ax.set_xticks(range(len(cols)),labels,fontsize=9)
    ax.set_yticks(range(len(agg)),[f'A{x+1}' for x in agg.index],fontsize=13)
    for i in range(len(agg)):
        for j in range(len(cols)):ax.text(j,i,f'{agg.iloc[i,j]:.2f}',ha='center',va='center',fontsize=14,
            color='white' if agg.iloc[i,j]>.6 else '#24364D')
    cb=fig.colorbar(im,ax=ax,fraction=.035,pad=.03);cb.solids.set_rasterized(False)
    output.save_figure(fig,'12_cib_informed_profiles')
    f.to_csv(output.results/'proposal_topic_profiles.csv',index=False)

    # Contextual pattern mining: topic-weighted motif shares in HMI strata and growth contexts.
    seq=pd.read_csv(output.results/'temporal_motifs.csv',dtype={'topic':str},keep_default_na=False).merge(f[['topic','hmi']],on='topic',validate='many_to_one')
    q1,q3=f.hmi.quantile([.25,.75]);seq['hmi_group']=np.where(seq.hmi<=q1,'Low HMI',np.where(seq.hmi>=q3,'High HMI','Middle'))
    seq=seq.merge(daily[['topic','date','hotspot']],on=['topic','date'],validate='one_to_one')
    seq.to_csv(output.results/'contextual_motif_windows.csv',index=False)
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
    contexts=pd.DataFrame(contexts);contexts.to_csv(output.results/'contextual_motif_comparison.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.4),layout='constrained',sharey=True)
    for ax,groups,title in zip(axes,[['Low HMI','High HMI'],['Other active day','Hotspot day']],['HMI strata','Sample growth context']):
        for j,group in enumerate(groups):
            vals=contexts[contexts.context==group].set_index('motif').topic_weighted_share.reindex(top)
            ax.bar(np.arange(len(top))+(j-.5)*.36,vals,width=.34,color=COLORS[j],label=group)
        ax.set_xticks(range(len(top)),top,fontsize=11);ax.set_title(title,loc='left',weight='bold',fontsize=13)
        ax.legend(frameon=False,fontsize=10);ax.set_ylim(0,.65);ax.tick_params(axis='y',labelsize=11)
    axes[0].set_ylabel('Mean within-topic motif share',fontsize=12)
    output.save_figure(fig,'13_hmi_context_patterns')

    # Recurring tag associations are also compared by context, with explicit overlap.
    eligible=set(f.topic);group_sets={'Low HMI':set(f.loc[f.hmi<=q1,'topic']),
                                    'High HMI':set(f.loc[f.hmi>=q3,'topic'])}
    pattern_rows=[];group_sizes={};post_sets={}
    for name,tags in group_sets.items():
        idx=[i for i,tt in enumerate(d.tags) if set(tt)&tags];post_sets[name]=set(idx)
        trans=[sorted(set(tags)&eligible) for tags in d.tags.iloc[idx]]
        itemsets=fpgrowth(transaction_frame(trans),min_support=.01,use_colnames=True,max_len=2)
        pairs=itemsets[itemsets.itemsets.map(len)==2].sort_values('support',ascending=False)
        group_sizes[name]=len(idx)
        for row in pairs.head(10).itertuples():
            pattern_rows.append({'context':name,'itemsets':' + '.join(sorted(row.itemsets)),
                'count':round(row.support*len(idx)),'support_in_context':row.support,'context_posts':len(idx)})
    pd.DataFrame(pattern_rows,columns=['context','itemsets','count','support_in_context','context_posts']).to_csv(output.results/'hmi_context_itemsets.csv',index=False)
    aug=next(r for r in ablations if r['representation']==profile)
    output.write_json('proposal_summary.json',{'hotspot_windows':len(hot),'hotspot_topics':hot.topic.nunique(),
        'active_topic_days':len(daily),'hotspot_min_posts':4,'hotspot_min_users':3,'hotspot_min_growth':2,
        'growth_definition':'[(n_td+1)/(N_d+1)] / [(sum preceding 7d n_t+1)/(sum preceding 7d N+1)]',
        'growth_history_min_sample_posts':20,'hmi_q25':q1,'hmi_q75':q3,
        'style_available':has_style,'profile_ablation':profile,
        'cib_informed_weight':.5,'cib_informed_common_silhouette':aug['common_fused_silhouette'],
        'cib_informed_ari_vs_baseline':aug['ari_vs_baseline'],
        'auxiliary_aggregation':'Mean post scores per user within topic, then equal weight over users.' if has_style else None,
        'hmi_itemset_context_posts':group_sizes,'hmi_context_shared_posts':len(post_sets['Low HMI']&post_sets['High HMI']),
        'hmi_style_spearman':float(f.hmi.corr(f.bot_style,method='spearman')) if has_style else None,
        'caveats':['Hotspot eligibility and HMI use the full observed period: retrospective, not validated online detection.',
                   'Day-level growth uses only the preceding 7 days, but topic screening uses all dates.',
                   'HMI strata share features with clustering and can overlap in member posts; no independent CIB validation.',
                   'Adding HMI reweights the four coordination components already present in the behavioral representation.',
                   'Any available bot-style feature is auxiliary evidence without CIB ground truth.',
                   'This analysis does not reconstruct propagation edges; relationship semantics and coverage require a separate audit.']})
    print('Proposal-aligned analysis complete:',len(hot),'hotspot windows;',hot.topic.nunique(),'topics',flush=True)

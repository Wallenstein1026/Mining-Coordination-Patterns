"""Reproducible descriptive clustering, pattern mining and auxiliary bot benchmark.

Run from any directory: python src/run_analysis.py. No synthetic data are used.
"""
from pathlib import Path
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'group28_mpl'))
os.environ.setdefault('OMP_NUM_THREADS', '2')
import json, re, hashlib, time, pickle, itertools, collections, platform
import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (silhouette_score, davies_bouldin_score, adjusted_rand_score,
    roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score,
    f1_score, confusion_matrix, roc_curve)
from mlxtend.frequent_patterns import apriori, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT/'results', ROOT/'figures'
RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
SEED=28
COLORS=['#185C85','#D4A63E','#219E8F','#C65A51','#786AA4','#689BC2','#C1789D','#65745B']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
    'axes.spines.right':False,'axes.labelcolor':'#24364D','text.color':'#24364D',
    'axes.edgecolor':'#B5C2CD','figure.facecolor':'white','savefig.facecolor':'white',
    'pdf.fonttype':42,'svg.fonttype':'none'})
CJK=None
for path in filter(None,[os.environ.get('GROUP28_CJK_FONT'),'/System/Library/Fonts/PingFang.ttc','/System/Library/Fonts/STHeiti Light.ttc','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc','C:/Windows/Fonts/msyh.ttc']):
    if Path(path).exists(): CJK=font_manager.FontProperties(fname=path); break

def dump(name,obj):
    (RES/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))

def save(fig,name):
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(FIG/(name+'.png'),dpi=200,bbox_inches='tight')
    plt.close(fig)

def clean_text(s):
    s=re.sub(r'https?://\S+|@[^\s:：/@]+',' ',str(s))
    return re.sub(r'\s+','',s).lower()

STYLE_NAMES=['log_length','url_fraction','mention_fraction','hashtag_fraction','repost_mark',
    'punctuation_fraction','digit_fraction','latin_fraction','unique_char_ratio','emoji_brackets',
    'exclamation_fraction','question_fraction']
def style(s):
    s=str(s); n=max(len(s),1)
    return [np.log1p(n),len(re.findall(r'https?://\S+',s))/max(n/100,1),s.count('@')/max(n/100,1),
        s.count('#')/max(n/100,1),float('//@' in s or '转发微博' in s),
        sum(c in '，。！？；：,.!?;:' for c in s)/n,sum(c.isdigit() for c in s)/n,
        sum(c.isascii() and c.isalpha() for c in s)/n,len(set(s))/n,
        len(re.findall(r'\[[^\]]{1,10}\]',s))/max(n/100,1),
        (s.count('!')+s.count('！'))/n,(s.count('?')+s.count('？'))/n]

def metrics(y,score,thr):
    pred=score>=thr
    return dict(roc_auc=roc_auc_score(y,score),average_precision=average_precision_score(y,score),
        accuracy=accuracy_score(y,pred),balanced_accuracy=balanced_accuracy_score(y,pred),
        f1=f1_score(y,pred),threshold=thr,confusion_matrix=confusion_matrix(y,pred).tolist())

def bot_benchmark():
    print('1. MisBot benchmark',flush=True)
    path, manifest = dataset_input('misbot')
    records=[json.loads(l) for l in path.open()]
    dump('misbot_input_provenance.json', manifest)
    ids=np.array([r['uid'] for r in records]); y=np.array([int(r['label']) for r in records])
    assert len(set(ids))==len(ids)
    idx=np.arange(len(y)); train,rest=train_test_split(idx,test_size=.4,stratify=y,random_state=SEED)
    val,test=train_test_split(rest,test_size=.5,stratify=y[rest],random_state=SEED)
    sets={i:s for s,ii in [('train',train),('validation',val),('test',test)] for i in ii}
    pd.DataFrame({'uid':ids,'label':y,'split':[sets[i] for i in idx]}).to_csv(RES/'misbot_account_splits.csv',index=False)
    metadata=np.array([np.log1p(np.maximum(np.asarray(r['numerical'],float),0)).tolist()+list(r['categorical']) for r in records])
    features=[]; targets=[]; owner=[]; split=[]; normtexts=[]
    for i,r in enumerate(records):
        for s in r['tweet']:
            if str(s).strip():
                features.append(style(s));targets.append(y[i]);owner.append(i);split.append(sets[i]);normtexts.append(clean_text(s))
    features=np.array(features);targets=np.array(targets);owner=np.array(owner);split=np.array(split)
    tr=split=='train'; va=split=='validation'; te=split=='test'
    train_texts={s for s,m in zip(normtexts,tr) if m and s}
    unique_test=np.array([bool(s) and s not in train_texts for s in normtexts]) & te
    style_model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=SEED))
    style_model.fit(features[tr],targets[tr])
    score_post=style_model.predict_proba(features)[:,1]
    account_score=pd.Series(score_post).groupby(owner).mean().reindex(idx).fillna(y[train].mean()).to_numpy()
    models={'Metadata LR':make_pipeline(StandardScaler(),LogisticRegression(max_iter=1500,random_state=SEED)),
        'Metadata RF':RandomForestClassifier(n_estimators=200,min_samples_leaf=5,max_features='sqrt',n_jobs=2,random_state=SEED)}
    scores={'Text-style LR':account_score}
    for name,m in models.items():
        m.fit(metadata[train],y[train]);scores[name]=m.predict_proba(metadata)[:,1]
    report={};fig,ax=plt.subplots(figsize=(7.8,4.4))
    predictions={'uid':ids[test],'label':y[test]}
    for j,(name,sc) in enumerate(scores.items()):
        thresholds=np.linspace(.1,.9,81)
        thr=float(max(thresholds,key=lambda t:balanced_accuracy_score(y[val],sc[val]>=t)))
        report[name]=metrics(y[test],sc[test],thr)
        a,b,_=roc_curve(y[test],sc[test]);ax.plot(a,b,lw=2.5,color=COLORS[j],label=f'{name}  AUC {report[name]["roc_auc"]:.3f}')
        predictions[name]=sc[test]
    ax.plot([0,1],[0,1],color='#AAB6C3',ls='--',lw=1)
    ax.set(xlabel='False positive rate',ylabel='True positive rate',xlim=(0,1),ylim=(0,1.02))
    ax.legend(loc='lower right',frameon=False,fontsize=10)
    ax.set_title(f'MisBot: {len(test):,} held-out accounts',loc='left',weight='bold',pad=14)
    save(fig,'01_bot_roc')
    pd.DataFrame(predictions).to_csv(RES/'misbot_test_predictions.csv',index=False)
    pd.DataFrame(report).T.drop(columns='confusion_matrix').to_csv(RES/'misbot_metrics.csv')
    dump('misbot_benchmark.json',{'n_accounts':len(y),'n_posts':len(features),'labels':dict(collections.Counter(map(str,y))),
        'split_counts':{s:sum(v==s for v in sets.values()) for s in ['train','validation','test']},
        'models':report,'style_post_auc_all_test':roc_auc_score(targets[te],score_post[te]),
        'style_post_auc_without_train_duplicates':roc_auc_score(targets[unique_test],score_post[unique_test]),
        'test_posts':int(te.sum()),'test_posts_after_exact_train_overlap_removal':int(unique_test.sum()),
        'threshold_selection':'max validation account balanced accuracy, grid 0.10:0.01:0.90',
        'transfer_caveat':'Text-style scores on COVID posts are uncalibrated cross-domain auxiliary scores, not CIB probabilities.'})
    with (RES/'text_style_model.pkl').open('wb') as f:pickle.dump(style_model,f)
    return style_model

def dataset_input(kind):
    """Require explicit input provenance; never silently substitute another corpus."""
    path = ROOT/'data'/(kind+'_manifest.json')
    if not path.exists():
        raise FileNotFoundError(f'Missing {path.name}: obtain and audit the dataset using data/README.md before running.')
    manifest=json.loads(path.read_text())
    if kind=='weibo':
        if manifest.get('dataset')!='nghuyong/Weibo-COV' or manifest.get('timezone')!='Asia/Shanghai':
            raise ValueError('Weibo manifest must name the official corpus and audited timezone.')
        key,hashkey='normalized_file','normalized_sha256'
    else:
        if manifest.get('release_kind') not in {'official','mirror'}:
            raise ValueError('MisBot manifest must distinguish official release from mirror.')
        key,hashkey='file','sha256'
    value=Path(manifest[key])
    if value.is_absolute() or not value.parts or value.parts[0]!='data' or '..' in value.parts:
        raise ValueError('Dataset input must be a repository-relative path below data/.')
    data=(ROOT/value).resolve()
    if not data.is_relative_to((ROOT/'data').resolve()):
        raise ValueError('Dataset path escapes the local data directory.')
    if not data.exists():raise FileNotFoundError(f'Missing local data: {value}')
    digest=hashlib.sha256()
    with data.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
    if digest.hexdigest()!=manifest[hashkey]:raise ValueError('Dataset SHA-256 mismatch: '+str(value))
    return data,manifest


def load_weibo():
    path,manifest=dataset_input('weibo')
    d=pd.read_csv(path,dtype=str,keep_default_na=False)
    required=['mid','user_id','created_at','content']
    if not set(required).issubset(d.columns):raise ValueError('Normalized Weibo CSV requires '+','.join(required))
    raw_n=len(d)
    valid=d[required].apply(lambda col:col.str.strip().ne('')).all(axis=1)
    d=d.loc[valid].drop_duplicates('mid').copy()
    d['time']=pd.to_datetime(d['created_at'],errors='coerce')
    if d.time.dt.tz is not None:raise ValueError('Normalize created_at to naive Asia/Shanghai local time first.')
    d=d[d.time.notna()].reset_index(drop=True)
    if d.empty:raise ValueError('No valid Weibo records after input checks.')
    d['norm']=d.content.map(clean_text)
    d['tags']=d.content.map(lambda v:sorted(set(t.strip().lower() for t in re.findall(r'#([^#\n]{2,70})#',v) if t.strip())))
    dump('weibo_audit.json',{'raw_posts':raw_n,'clean_posts':len(d),'unique_users':d.user_id.nunique(),
        'time_min':str(d.time.min()),'time_max':str(d.time.max()),'hashtag_posts':int(d.tags.map(bool).sum()),
        'source':manifest,'available_fields':list(d.columns),'cib_truth':'not established by this import',
        'propagation':'Additional relationship columns require separate semantic and coverage audit.'})
    return d


def build_topics(d,model):
    print('2. Topic features',flush=True)
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
            'bot_style':float(g.bot_style.mean()),'hour_entropy':float(-np.sum(hist[hist>0]*np.log(hist[hist>0]))/np.log(24))}
        rows.append(row);profiles.append(np.r_[hist,life])
        # Remove topic hashtag strings from text so the exact topic label does not dominate text similarity.
        docs.append(' '.join(re.sub(r'#[^#\n]{2,70}#',' ',str(s)) for s in g.content))
    f=pd.DataFrame(rows)
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
    f.to_csv(RES/'topic_features.csv',index=False)
    np.savez_compressed(RES/'topic_representations.npz',text=text,behavior=beh,fused=fused,profiles=np.array(profiles))
    dump('topic_design.json',{'n_topics':len(f),'min_posts':30,'min_users':5,'excluded_broad_tags':sorted(broad),
        'memberships':sum(len(ii) for ii in membership.values()),'distinct_posts':len(set(itertools.chain.from_iterable(membership.values()))),
        'svd_explained_variance':float(svd.explained_variance_ratio_.sum()),'behavior_columns':behaviorcols,
        'hmi':'Mean within-sample percentile ranks of duplicate, user_hhi, peak_day, same_minute. Not a manipulation probability.'})
    return f,{'Text':text,'Behavior':beh,'Fused':fused},membership,rankcols

def cluster_topics(f,reps):
    print('3. Clustering comparisons',flush=True)
    rows=[];partitions={}
    for rep,X in reps.items():
        for method in ['KMeans','Ward']:
            for k in range(3,9):
                for seed in ([28,29,30,31,32] if method=='KMeans' else [28]):
                    t=time.perf_counter()
                    labels=KMeans(k,n_init=20,random_state=seed).fit_predict(X) if method=='KMeans' else AgglomerativeClustering(k,linkage='ward').fit_predict(X)
                    key=(rep,method,k,seed);partitions[key]=labels
                    rows.append(dict(representation=rep,method=method,k=k,seed=seed,
                        native_silhouette=silhouette_score(X,labels),common_fused_silhouette=silhouette_score(reps['Fused'],labels),
                        common_davies_bouldin=davies_bouldin_score(reps['Fused'],labels),seconds=time.perf_counter()-t,
                        min_cluster_size=int(np.bincount(labels).min())))
    metrics_df=pd.DataFrame(rows);metrics_df.to_csv(RES/'clustering_grid.csv',index=False)
    # Select among fused candidates only; no CIB or HMI labels participate in the selection.
    candidate=metrics_df[(metrics_df.representation=='Fused') & (metrics_df.seed==SEED) & (metrics_df.min_cluster_size>=10)]
    best=candidate.sort_values('native_silhouette',ascending=False).iloc[0];key=('Fused',best.method,int(best.k),SEED)
    labels=partitions[key];f['cluster']=labels
    stability=[];rng=np.random.default_rng(SEED);X=reps['Fused']
    for run in range(20):
        ii=np.sort(rng.choice(len(f),int(.8*len(f)),replace=False))
        ll=KMeans(int(best.k),n_init=20,random_state=SEED+run).fit_predict(X[ii]) if best.method=='KMeans' else AgglomerativeClustering(int(best.k)).fit_predict(X[ii])
        stability.append(adjusted_rand_score(labels[ii],ll))
    # Size threshold sensitivity, refitting the selected model on higher-volume subsets.
    sens=[]
    for n in [40,50,75]:
        ii=np.flatnonzero(f.posts>=n)
        if len(ii)>int(best.k)*3:
            ll=KMeans(int(best.k),n_init=20,random_state=SEED).fit_predict(X[ii]) if best.method=='KMeans' else AgglomerativeClustering(int(best.k)).fit_predict(X[ii])
            sens.append({'min_posts':n,'n_topics':len(ii),'ari_vs_full':adjusted_rand_score(labels[ii],ll)})
    f.to_csv(RES/'topic_features.csv',index=False)
    pd.DataFrame({'run':range(20),'ari':stability}).to_csv(RES/'cluster_stability.csv',index=False)
    dump('selected_clustering.json',{'method':best.method,'k':int(best.k),'silhouette':best.native_silhouette,
        'cluster_sizes':dict(collections.Counter(map(str,labels))),'stability_mean':np.mean(stability),'stability_std':np.std(stability),
        'stability_min':min(stability),'size_sensitivity':sens,'selection':'Best seed-28 fused silhouette over k=3..8 and KMeans/Ward, with min cluster size >=10',
        'caveat':'Descriptive transductive clustering; silhouette measures compactness, not semantic or CIB validity.'})
    coords=PCA(n_components=2,random_state=SEED).fit_transform(X)
    fig,ax=plt.subplots(figsize=(7.8,4.4))
    for c in sorted(set(labels)):
        m=labels==c;ax.scatter(coords[m,0],coords[m,1],s=18+20*np.log1p(f.posts[m]/30),c=COLORS[c],alpha=.75,edgecolors='white',linewidths=.3,label=f'C{c+1}  (n={m.sum()})')
    ax.set(xlabel='Principal component 1',ylabel='Principal component 2');ax.legend(frameon=False,fontsize=10)
    ax.set_title('Topic clusters in a 2D PCA projection',loc='left',weight='bold',pad=14)
    save(fig,'02_cluster_map')
    fig,ax=plt.subplots(figsize=(7.8,4.4))
    for j,((rep,method),g) in enumerate(metrics_df.groupby(['representation','method'])):
        a=g.groupby('k').common_fused_silhouette.agg(['mean','std']).fillna(0)
        ax.plot(a.index,a['mean'],marker='o' if method=='KMeans' else 's',ls='-' if method=='KMeans' else '--',lw=2,color=COLORS[j],label=f'{rep}, {method}')
    ax.set(xlabel='Number of clusters k',ylabel='Silhouette in common fused space',xticks=list(range(3,9)))
    ax.legend(frameon=False,ncol=2,fontsize=9);ax.axhline(0,color='#CCD4DD',lw=1)
    save(fig,'03_method_comparison')
    return labels

def mine_patterns(d,f,membership):
    print('4. Frequent itemsets and time motifs',flush=True)
    eligible=set(f.topic);trans=[sorted(set(tags)&eligible) for tags in d.tags]
    te=TransactionEncoder();mat=te.fit(trans).transform(trans);transactions=pd.DataFrame(mat,columns=te.columns_)
    bench=[];allsets=None
    for support in [.003,.005,.01]:
        sets=[]
        for name,fn in [('Apriori',apriori),('FP-growth',fpgrowth)]:
            t=time.perf_counter();itemsets=fn(transactions,min_support=support,use_colnames=True,max_len=3)
            bench.append({'method':name,'min_support':support,'itemsets':len(itemsets),'seconds':time.perf_counter()-t})
            sets.append(set(itemsets.itemsets))
            if name=='FP-growth' and support==.003:allsets=itemsets
        assert sets[0]==sets[1], 'Frequent-itemset methods disagree'
    pd.DataFrame(bench).to_csv(RES/'pattern_method_comparison.csv',index=False)
    supports={frozenset(r.itemsets):r.support for r in allsets.itertuples()};rules=[]
    for itemset,support in supports.items():
        if len(itemset)<2:continue
        for k in range(1,len(itemset)):
            for lhs in itertools.combinations(sorted(itemset),k):
                lhs=frozenset(lhs);rhs=itemset-lhs;confidence=support/supports[lhs];lift=confidence/supports[rhs]
                if confidence>=.5 and lift>=1.5:
                    rules.append({'antecedent':' + '.join(sorted(lhs)),'consequent':' + '.join(sorted(rhs)),
                        'support':support,'count':int(round(support*len(d))),'confidence':confidence,'lift':lift})
    rules=pd.DataFrame(rules).sort_values(['support','lift'],ascending=False)
    rules.to_csv(RES/'association_rules.csv',index=False)
    out=allsets.copy();out['itemsets']=out.itemsets.map(lambda x:' + '.join(sorted(x)));out.to_csv(RES/'frequent_itemsets.csv',index=False)
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
    seq=pd.DataFrame(sequences);seq.to_csv(RES/'temporal_motifs.csv',index=False)
    motif_counts=seq.motif.value_counts()
    motifrows=[{'motif':m,'windows':int(n),'support':n/len(seq),'distinct_topics':len(topic_motif[m])} for m,n in motif_counts.items()]
    pd.DataFrame(motifrows).to_csv(RES/'temporal_motif_summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(7.8,4.4))
    for j,m in enumerate(motif_counts.head(4).index):
        a=np.array(curves[m]);mu=a.mean(axis=0)
        ax.plot([3,9,15,21],mu,marker='o',lw=2.5,c=COLORS[j],label=f'{m}: {len(a)} windows')
    ax.set(xlabel='Clock hour (Beijing time)',ylabel='Mean fraction of daily topic posts',xticks=[3,9,15,21],ylim=(0,1))
    ax.legend(frameon=False,fontsize=10);ax.set_title('Recurring six-hour activity motifs',loc='left',weight='bold',pad=14)
    save(fig,'05_temporal_motifs')
    # Draw the most frequent pair itemsets as a weighted co-occurrence graph.
    pairs=[(tuple(sorted(k)),v) for k,v in supports.items() if len(k)==2]
    pairs=sorted(pairs,key=lambda x:x[1],reverse=True)[:18]
    import networkx as nx
    G=nx.Graph()
    for (a,b),s in pairs:G.add_edge(a,b,weight=s*len(d))
    english={'兰州爆料':'Lanzhou updates','抗击疫情 甘肃在行动':'Gansu response',
        '恶心':'Nausea','出汗':'Sweating','腹痛':'Abdominal pain','面色苍白':'Pallor',
        '昏昏欲睡':'Drowsiness','咳嗽':'Cough','呕吐':'Vomiting'}
    fig,axes=plt.subplots(1,2,figsize=(8.5,4.3),gridspec_kw={'width_ratios':[1,1.8]})
    components=sorted(nx.connected_components(G),key=len)
    for j,(ax,nodes) in enumerate(zip(axes,components)):
        sub=G.subgraph(nodes);pos=nx.circular_layout(sub)
        nx.draw_networkx_edges(sub,pos,ax=ax,width=[1+sub[u][v]['weight']/130 for u,v in sub.edges],edge_color='#B2C7D5',alpha=.65)
        nx.draw_networkx_nodes(sub,pos,ax=ax,node_color=COLORS[j],node_size=270,alpha=.9)
        for n,(x,y) in pos.items():
            ax.text(x*1.28,y*1.28,english.get(n,n),fontsize=8.5,ha='center',va='center',
                bbox={'facecolor':'white','alpha':.9,'edgecolor':'none','pad':1})
        ax.set_xlim(-1.9,1.9);ax.set_ylim(-1.55,1.55);ax.axis('off')
        ax.set_title(['Regional tags','Symptom tags'][j],fontsize=12,weight='bold')
    save(fig,'06_hashtag_network')
    dump('pattern_summary.json',{'transaction_denominator':len(d),'eligible_hashtags':len(eligible),'min_support':.003,
        'itemsets':len(allsets),'rules':len(rules),'rules_min_confidence':.5,'rules_min_lift':1.5,
        'temporal_windows':len(seq),'top_motifs':motifrows[:5],
        'caveat':'Hashtag co-occurrence is an association pattern. This co-occurrence analysis does not reconstruct repost edges; relationship fields need separate auditing.'})
    return rules

def plot_signal_heatmap(f,rankcols):
    cols=rankcols+['hmi','bot_style'];agg=f.groupby('cluster')[cols].mean()
    fig,ax=plt.subplots(figsize=(8,4.4));im=ax.pcolormesh(np.arange(7)-.5,np.arange(len(agg)+1)-.5,agg.to_numpy(),cmap='YlGnBu',vmin=0,vmax=1)
    ax.invert_yaxis()
    ax.set_xticks(range(6),['Duplicate\nrank','User HHI\nrank','Peak-day\nrank','Same-minute\nrank','HMI','Bot-style\nscore'],fontsize=9)
    ax.set_yticks(range(len(agg)),[f'C{x+1}' for x in agg.index])
    for i in range(len(agg)):
        for j in range(6):ax.text(j,i,f'{agg.iloc[i,j]:.2f}',ha='center',va='center',color='white' if agg.iloc[i,j]>.6 else '#17385C',fontsize=12,weight='bold')
    cb=fig.colorbar(im,ax=ax,fraction=.035,pad=.04)
    cb.solids.set_rasterized(False)
    save(fig,'04_cluster_signals')

def signal_figures(d,f,rankcols):
    print('5. Figures and robustness',flush=True)
    plot_signal_heatmap(f,rankcols)
    # Two contrasting observed topics, show daily sample counts; selection is deterministic by HMI tails.
    eligible=f[f.posts>=100]
    selected=[eligible.sort_values('hmi').iloc[0],eligible.sort_values('hmi',ascending=False).iloc[0]]
    fig,axes=plt.subplots(2,1,figsize=(8.5,4.6),constrained_layout=True)
    example=[]
    for j,(ax,r) in enumerate(zip(axes,selected)):
        ii=[i for i,tt in enumerate(d.tags) if r.topic in tt];g=d.iloc[ii];daily=g.groupby(g.time.dt.floor('D')).size()
        full=daily.reindex(pd.date_range(daily.index.min(),daily.index.max(),freq='D'),fill_value=0)
        ax.plot(full.index,full.values,color=COLORS[j],lw=1.5)
        title=r.topic if CJK is not None else ['Lower-HMI topic (source tag in results/timeline_examples.json)','Higher-HMI topic (source tag in results/timeline_examples.json)'][j]
        ax.set_title(title,fontproperties=CJK,loc='left',fontsize=11)
        ax.text(.99,.85,f'n={int(r.posts)}  HMI={r.hmi:.2f}',transform=ax.transAxes,ha='right',fontsize=10)
        ax.set_ylabel('Posts / day');ax.tick_params(axis='x',labelsize=8)
        example.append({'topic':r.topic,'posts':int(r.posts),'hmi':r.hmi})
    save(fig,'07_observed_timelines');dump('timeline_examples.json',example)
    # HMI sensitivity to dropping any one component. Correlation is rank concordance, not external validation.
    from scipy.stats import spearmanr
    leave=[]
    for col in rankcols:
        alt=f[[c for c in rankcols if c!=col]].mean(axis=1)
        leave.append({'dropped':col,'spearman_vs_hmi':spearmanr(f.hmi,alt).statistic})
    pd.DataFrame(leave).to_csv(RES/'hmi_sensitivity.csv',index=False)
    # Conditional within-day null: shuffle topic timestamps among all sample posts on the same date.
    rng=np.random.default_rng(SEED);daypools={str(day.date()):g.time.dt.floor('min').to_numpy() for day,g in d.groupby(d.time.dt.floor('D'))}
    eligible=f.nlargest(50,'posts');null=[]
    for r in eligible.itertuples():
        g=d.iloc[[i for i,tags in enumerate(d.tags) if r.topic in tags]]
        counts=g.groupby(g.time.dt.floor('D')).size();sim=[]
        for _ in range(999):
            total=0
            for day,n in counts.items():
                sample=rng.choice(daypools[str(day.date())],int(n),replace=False)
                vals=np.unique(sample,return_counts=True)[1];total+=vals[vals>=2].sum()
            sim.append(total/len(g))
        null.append({'topic':r.topic,'posts':r.posts,'observed':r.same_minute,'null_mean':np.mean(sim),
            'null_q95':np.quantile(sim,.95),'p_upper':(1+sum(v>=r.same_minute for v in sim))/1000})
    # BH adjusted p values over the 50 preselected topics, preserving the coarse Monte Carlo resolution.
    nd=pd.DataFrame(null);p=nd.p_upper.to_numpy();order=np.argsort(p);adj=np.empty(len(p))
    adj[order]=np.minimum(1,np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]);nd['q_bh']=adj
    nd.to_csv(RES/'synchrony_null_test.csv',index=False)
    f.groupby('cluster').agg(topics=('topic','size'),median_posts=('posts','median'),hmi_mean=('hmi','mean'),style_mean=('bot_style','mean')).to_csv(RES/'cluster_summary.csv')
    return nd

def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['all','misbot'],default='all')
    args=parser.parse_args()
    start=time.time()
    if args.stage=='misbot':
        bot_benchmark()
    else:
        d=load_weibo()  # Fail before model training if official inputs are missing.
        model=bot_benchmark()
        f,reps,membership,rankcols=build_topics(d,model)
        cluster_topics(f,reps);mine_patterns(d,f,membership);signal_figures(d,f,rankcols)
    from importlib.metadata import version
    dump('environment.json',{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
        'sklearn':sklearn_version,'scipy':scipy.__version__,'seed':SEED,'runtime_seconds':time.time()-start,
        'packages':{p:version(p) for p in ['matplotlib','mlxtend','networkx','reportlab','pdfrw']}})
    print('DONE',time.time()-start,flush=True)

if __name__=='__main__':main()

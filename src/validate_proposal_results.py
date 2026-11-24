"""Recount proposal-aligned results against raw records and saved topic features."""
from pathlib import Path
import json,re,pickle
from run_analysis import load_weibo
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,adjusted_rand_score

ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results';checks=[]
def check(name,condition):
    if not condition:raise AssertionError(name)
    checks.append(name)

raw=load_weibo()
raw['date']=pd.to_datetime(raw.created_at).dt.floor('D')
raw['tags']=raw.content.map(lambda s:set(t.strip().lower() for t in re.findall(r'#([^#\n]{2,70})#',s) if t.strip()))
f=pd.read_csv(R/'proposal_topic_profiles.csv');daily=pd.read_csv(R/'topic_day_growth.csv')
hot=pd.read_csv(R/'hotspot_windows.csv');summary=json.loads((R/'proposal_summary.json').read_text())
check('Topic-date growth rows are unique',not daily.duplicated(['topic','date']).any())
check('Hotspot count and topic count reconcile',len(hot)==summary['hotspot_windows'] and hot.topic.nunique()==summary['hotspot_topics'])
flags=daily.history_eligible & daily.posts.ge(4) & daily.users.ge(3) & daily.share_growth_ratio.ge(2)
check('All hotspot thresholds and stored rows reconcile',np.array_equal(flags,daily.hotspot) and set(zip(hot.topic,hot.date))==set(zip(daily.loc[flags,'topic'],daily.loc[flags,'date'])))
calendar=pd.date_range(raw.date.min(),raw.date.max(),freq='D')
totals=raw.groupby('date').size().reindex(calendar,fill_value=0)
cases=json.loads((R/'hotspot_cases.json').read_text())
samples=hot.sort_values('posts',ascending=False).drop_duplicates('topic').head(5)[['topic','date']].to_dict('records')+cases
for case in samples:
    topic=case['topic'];day=pd.Timestamp(case['date']);g=raw[raw.tags.map(lambda tags:topic in tags)]
    today=g[g.date==day];start=day-pd.Timedelta(days=7)
    prior=g[(g.date>=start)&(g.date<day)];ref=raw[(raw.date>=start)&(raw.date<day)]
    score=((len(today)+1)/(totals.loc[day]+1))/((len(prior)+1)/(len(ref)+1))
    saved=hot[(hot.topic==topic)&(hot.date==case['date'])].iloc[0]
    check('Raw hotspot recount: '+topic,len(today)==saved.posts and today.user_id.nunique()==saved.users and len(prior)==saved.previous_7d_posts and len(ref)==saved.previous_7d_sample_posts and abs(score-saved.share_growth_ratio)<1e-9)

rankcols=['duplicate','user_hhi','peak_day','same_minute']
independent_hmi=sum(f[c].round(12).rank(method='average',pct=True) for c in rankcols)/4
check('HMI percentile formula independently recomputes',np.allclose(independent_hmi,f.hmi))
from run_analysis import style
with (R/'text_style_model.pkl').open('rb') as handle:model=pickle.load(handle)
for topic in list(dict.fromkeys([c['topic'] for c in cases]+f.topic.head(3).tolist()))[:3]:
    g=raw[raw.tags.map(lambda tags:topic in tags)].copy()
    g['score']=model.predict_proba(np.asarray([style(s) for s in g.content]))[:,1]
    scores=g.groupby('user_id').score.mean()
    check('Equal-user style aggregation: '+topic,abs(scores.mean()-f.loc[f.topic==topic,'user_balanced_style'].iloc[0])<1e-12)

X=np.load(R/'topic_representations.npz')['fused'];z=StandardScaler().fit_transform(f[['hmi','user_balanced_style']])
aug=np.c_[X,.5*z/np.sqrt(2)]
k=int(json.loads((R/'selected_clustering.json').read_text())['k'])
labels=KMeans(k,n_init=20,random_state=28).fit_predict(aug)
check('Augmented clustering independently refits',np.array_equal(labels,f.cib_informed_cluster))
check('Augmented common silhouette recomputes',abs(silhouette_score(X,labels)-summary['cib_informed_common_silhouette'])<1e-12)
check('Augmented partition ARI recomputes',abs(adjusted_rand_score(f.cluster,labels)-summary['cib_informed_ari_vs_baseline'])<1e-12)
seq=pd.read_csv(R/'contextual_motif_windows.csv');original=pd.read_csv(R/'temporal_motifs.csv')
check('Context annotation preserves all motif windows',len(seq)==len(original) and not seq.duplicated(['topic','date']).any())
check('Context hotspot flags reconcile',set(zip(seq.loc[seq.hotspot,'topic'],seq.loc[seq.hotspot,'date']))==set(zip(hot.topic,hot.date)))
contexts=pd.read_csv(R/'contextual_motif_comparison.csv')
q1,q3=f.hmi.quantile([.25,.75])
for name,part in [('Low HMI',seq[seq.hmi<=q1]),('High HMI',seq[seq.hmi>=q3]),('Hotspot day',seq[seq.hotspot]),('Other active day',seq[~seq.hotspot])]:
    topic_props=part.groupby(['topic','motif']).size().unstack(fill_value=0)
    topic_props=topic_props.div(topic_props.sum(axis=1),axis=0)
    saved=contexts[contexts.context==name].set_index('motif')
    actual=topic_props.mean().reindex(saved.index,fill_value=0)
    check('Topic-weighted motif proportions: '+name,np.allclose(actual,saved.topic_weighted_share) and saved.eligible_windows.eq(len(part)).all() and saved.topics_with_eligible_windows.eq(part.topic.nunique()).all())

(R/'proposal_validation.json').write_text(json.dumps({'passed':len(checks),'checks':checks},ensure_ascii=False,indent=2))
print(f'{len(checks)} proposal-aligned evidence checks passed')

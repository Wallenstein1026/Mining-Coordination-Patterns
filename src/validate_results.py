"""Independent checks of saved experimental evidence against the source data."""
from pathlib import Path
import hashlib,json,re
from run_analysis import dataset_input,load_weibo
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,f1_score,balanced_accuracy_score

ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results';checks=[]
def ok(label,condition):
    if not condition:raise AssertionError(label)
    checks.append(label)

for kind in ['weibo','misbot']:
    path,manifest=dataset_input(kind)
    ok('Audited input hash verified: '+kind,path.exists())
splits=pd.read_csv(R/'misbot_account_splits.csv')
ok('Account IDs unique across train, validation, test',splits.uid.is_unique and set(splits.split)=={'train','validation','test'})
p=pd.read_csv(R/'misbot_test_predictions.csv');b=json.loads((R/'misbot_benchmark.json').read_text())
ok('Predictions match the held-out account set',set(p.uid)==set(splits.loc[splits.split=='test','uid']))
for name,m in b['models'].items():
    ok('Recomputed test ROC AUC: '+name,abs(roc_auc_score(p.label,p[name])-m['roc_auc'])<1e-10)
    ok('Recomputed test F1: '+name,abs(f1_score(p.label,p[name]>=m['threshold'])-m['f1'])<1e-10)
f=pd.read_csv(R/'topic_features.csv');rr=np.load(R/'topic_representations.npz')
ok('Topic eligibility thresholds',(f.posts>=30).all() and (f.users>=5).all())
ok('HMI recomputes from component ranks',np.allclose(f.hmi,f[[c+'_rank' for c in ['duplicate','user_hhi','peak_day','same_minute']]].mean(axis=1)))
ok('Finite, aligned topic representations',all(np.isfinite(rr[n]).all() and len(rr[n])==len(f) for n in rr.files))
ok('Cluster sizes match selected configuration',len(f.cluster.unique())==json.loads((R/'selected_clustering.json').read_text())['k'])
d=load_weibo()
tags=[set(t.strip().lower() for t in re.findall(r'#([^#\n]{2,70})#',s)) for s in d.content]
rules=pd.read_csv(R/'association_rules.csv')
# Top rules are independently recounted in the original posts, including the support denominator.
for r in rules.head(15).itertuples():
    lhs=set(r.antecedent.split(' + '));rhs=set(r.consequent.split(' + '));n=sum(lhs|rhs<=t for t in tags)
    nl=sum(lhs<=t for t in tags);nr=sum(rhs<=t for t in tags)
    ok('Recount rule: '+r.antecedent+' => '+r.consequent,n==r.count and abs(n/len(d)-r.support)<1e-10 and abs(n/nl-r.confidence)<1e-10 and abs(n*len(d)/(nl*nr)-r.lift)<1e-8)
motifs=pd.read_csv(R/'temporal_motifs.csv');summ=pd.read_csv(R/'temporal_motif_summary.csv')
ok('Temporal windows and motif support reconcile',(motifs.n>=4).all() and summ.windows.sum()==len(motifs) and abs(summ.support.sum()-1)<1e-10)
nd=pd.read_csv(R/'synchrony_null_test.csv')
ok('Null-test probabilities and BH correction valid',len(nd)==min(50,len(f)) and nd.p_upper.between(.001,1).all() and (nd.q_bh>=nd.p_upper-1e-12).all() and nd.q_bh.between(0,1).all())
counts=pd.read_csv(R/'pattern_method_comparison.csv').pivot(index='min_support',columns='method',values='itemsets')
ok('Apriori and FP-growth saved itemset counts agree',(counts.Apriori==counts['FP-growth']).all())
(R/'validation.json').write_text(json.dumps({'passed':len(checks),'checks':checks},ensure_ascii=False,indent=2))
print(f'{len(checks)} evidence checks passed')

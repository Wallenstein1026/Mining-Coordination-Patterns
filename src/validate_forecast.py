"""Check temporal separation, prefix invariance and forward-test metrics."""
from predict_hotspots import *

cfg=load_config()
checks=[]
def check(name,condition):
    if not condition:raise AssertionError(name)
    checks.append(name)

f=pd.read_csv(RES/'forecast_features.csv',float_precision='round_trip');p=pd.read_csv(RES/'forecast_predictions.csv',float_precision='round_trip')
met=pd.read_csv(RES/'forecast_metrics.csv');summary=json.loads((RES/'forecast_summary.json').read_text())
check('One record per past-visible topic and forecast date',not f.duplicated(['topic','as_of']).any())
check('No feature date reaches the target day',(pd.to_datetime(f.feature_end)<=pd.to_datetime(f.as_of)).all() and
      (pd.to_datetime(f.target_date)-pd.to_datetime(f.as_of)).eq(pd.Timedelta(days=1)).all())
check('Past-only candidate requirements',(f.history_posts>=5).all() and (f.history_users>=3).all() and (f.posts_7>=1).all() and (f.sample_posts_7>=20).all())
check('Outcome rule reconciles with labels',np.array_equal(f.target,((f.nextday_posts>=4)&(f.nextday_users>=3)&(f.target_growth>=2)).astype(int)))
check('Calendar splits are separated',f.loc[f.split=='train','target_date'].max()<f.loc[f.split=='validation','as_of'].min() and f.loc[f.split=='validation','target_date'].max()<f.loc[f.split=='test','as_of'].min())
check('Saved predictions align with features',p[['topic','as_of','split','target']].equals(f[['topic','as_of','split','target']]))
# Truncate raw input and rebuild: future rows must not change existing features.
prefix_day=(pd.Timestamp(cfg['validation_start'])+(pd.Timestamp(cfg['validation_end'])-pd.Timestamp(cfg['validation_start']))//2).floor('D')
prefix,_,_=make_rows(cutoff=prefix_day,write_results=False)
keys=['topic','as_of'];cols=ACTIVITY+COORD+[c for c in f if c.startswith('motif_')]
expected=f[pd.to_datetime(f.as_of)<prefix_day].set_index(keys).sort_index()
actual=prefix.set_index(keys).sort_index()
check('Future-data truncation preserves the historical candidate set',expected.index.equals(actual.index))
check('Future-data truncation preserves every historical predictor',np.allclose(expected[cols],actual[cols],equal_nan=True))
check('Rolling HMI percentiles recompute',np.allclose(f[['duplicate_28','hhi_28','peak_28','sync_28']].round(12).groupby(f.as_of).rank(pct=True).mean(axis=1),f.rolling_hmi))
for split in ['validation','test']:
    part=p[p.split==split]
    for row in met[met.split==split].itertuples():
        check('Recompute AP and daily-review count: '+split+' / '+row.model,
              abs(average_precision_score(part.target,part[row.model])-row.average_precision)<1e-12 and
              int(part.groupby('as_of').size().clip(upper=5).sum())==row.reviewed_candidates)
vm=met[met.split=='validation'].set_index('model').average_precision
check('Model family chosen only by validation AP',summary['validation_selected_family']==('RF' if vm['Activity RF']>vm['Activity LR'] else 'LR'))
selected=met[met.split=='test'].set_index('model').average_precision
check('Selected-family incremental AP reconciles',abs(summary['test_ap_delta_full_minus_base']-(selected[summary['full_model']]-selected[summary['base_model']]))<1e-12)
check('Bootstrap interval is finite and ordered',np.isfinite([summary['ap_delta_ci025'],summary['ap_delta_ci975']]).all() and summary['ap_delta_ci025']<=summary['ap_delta_ci975'])
all_hot=pd.read_csv(RES/'forecast_all_hotspot_outcomes.csv')
test_hot=all_hot[pd.to_datetime(all_hot.target_date).between(pd.Timestamp(cfg['test_start'])+pd.Timedelta(days=1),pd.Timestamp(cfg['test_end'])+pd.Timedelta(days=1))]
check('Coverage denominator includes unforecastable new topics',len(test_hot)==summary['all_test_hotspot_topicdays'] and int(p[p.split=='test'].target.sum())==summary['candidate_test_hotspot_topicdays'])
(RES/'forecast_validation.json').write_text(json.dumps({'passed':len(checks),'checks':checks},ensure_ascii=False,indent=2))
print(len(checks),'forward-forecast checks passed, including future-data truncation')

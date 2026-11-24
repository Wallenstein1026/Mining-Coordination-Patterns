"""Forward next-day sample-hotspot test with past-only topic eligibility and features.

No full-period topic lists, HMI, representations or MisBot model are used.
The task is forecasting growth of already observed topics, not unseen topics/CIB.
"""
from run_analysis import *
from forecast_config import load_config
from sklearn.metrics import precision_recall_curve, brier_score_loss
from collections import Counter

ACTIVITY=['log_posts_1','log_posts_3','log_posts_7','log_posts_28','log_users_7',
          'log_topic_age','days_since_post','active_days_7','share_today','growth_today']
COORD=['duplicate_28','hhi_28','peak_28','sync_28','rolling_hmi']


def make_rows(cutoff=None,write_results=True):
    cfg=load_config()
    d=load_weibo().sort_values('time').reset_index(drop=True)
    if cutoff is not None:d=d[d.time.dt.floor('D')<=pd.Timestamp(cutoff)].reset_index(drop=True)
    calendar=pd.date_range(d.time.min().floor('D'),d.time.max().floor('D'),freq='D')
    totals=d.groupby(d.time.dt.floor('D')).size().reindex(calendar,fill_value=0).to_numpy()
    broad={'疫情','新冠','新冠肺炎','新型冠状病毒','新型冠状病毒感染肺炎','肺炎[超话]','疫情[超话]','covid-19','新冠病毒'}
    groups=collections.defaultdict(list)
    for i,tags in enumerate(d.tags):
        for tag in tags:
            if tag not in broad:groups[tag].append(i)
    # A tag with <4 total records cannot ever meet the outcome's >=4/day rule
    # or the past-only >=5 cumulative-post candidate rule. Pruning is equivalent.
    rows=[];all_hot=[];motif_counter=Counter()
    for tag,ii in groups.items():
        if len(ii)<4:continue
        g=d.iloc[ii];di=((g.time.dt.floor('D')-calendar[0]).dt.days).to_numpy()
        n=np.bincount(di,minlength=len(calendar))
        users=g.user_id.to_numpy();norm=g.norm.to_numpy()
        mins=g.time.dt.floor('min').astype('int64').to_numpy()
        count_cum=np.cumsum(n)
        byday_users={int(day):len(set(users[di==day])) for day in np.unique(di)}
        c7=pd.Series(n).rolling(7,min_periods=7).sum().to_numpy()
        n7=pd.Series(totals).rolling(7,min_periods=7).sum().to_numpy()
        day_motifs={}
        hours=g.time.dt.hour.to_numpy()
        for day in np.unique(di):
            if n[day]>=4:
                bins=np.bincount(hours[di==day]//6,minlength=4)
                pattern=''.join('H' if value>np.median(bins) else 'L' for value in bins)
                day_motifs[int(day)]=pattern
        # Enumerate all outcome-positive topic-days, including new/unforecastable tags.
        for day in np.flatnonzero(n>=4):
            if day<7 or byday_users[day]<3 or n7[day-1]<20:continue
            growth=((n[day]+1)/(totals[day]+1))/((c7[day-1]+1)/(n7[day-1]+1))
            if growth>=2:all_hot.append({'topic':tag,'target_date':str(calendar[day].date()),'posts':int(n[day])})
        if len(ii)<5:continue
        current_users=set();seen_end=0
        for day in np.flatnonzero(np.nan_to_num(c7)>0):
            if day<7 or day>=len(calendar)-1:continue
            stop=np.searchsorted(di,day,side='right')
            current_users.update(users[seen_end:stop]);seen_end=stop
            if count_cum[day]<5 or len(current_users)<3 or n7[day]<20:continue
            lo=np.searchsorted(di,max(0,day-27));lo7=np.searchsorted(di,day-6)
            window=norm[lo:stop];u=users[lo:stop];m=mins[lo:stop];nw=len(window)
            uc=np.asarray(list(Counter(u).values()),float)
            mc=np.asarray(list(Counter(m).values()))
            previous7=int(n[max(0,day-7):day].sum())
            previousN=int(totals[max(0,day-7):day].sum())
            growth_today=((n[day]+1)/(totals[day]+1))/((previous7+1)/(previousN+1))
            nextday=day+1
            target_growth=((n[nextday]+1)/(totals[nextday]+1))/((c7[day]+1)/(n7[day]+1))
            target=int(n[nextday]>=4 and byday_users.get(nextday,0)>=3 and target_growth>=2)
            motifs=Counter(value for date,value in day_motifs.items() if day-27<=date<=day)
            row={'topic':tag,'as_of':str(calendar[day].date()),'target_date':str(calendar[nextday].date()),
                'feature_start':str(calendar[max(0,day-27)].date()),'feature_end':str(calendar[day].date()),
                'target':target,'nextday_posts':int(n[nextday]),'nextday_users':byday_users.get(nextday,0),
                'target_growth':float(target_growth),'history_posts':int(count_cum[day]),'history_users':len(current_users),
                'posts_7':int(c7[day]),'sample_posts_7':int(n7[day]),
                'log_posts_1':np.log1p(n[day]),'log_posts_3':np.log1p(n[max(0,day-2):day+1].sum()),
                'log_posts_7':np.log1p(c7[day]),'log_posts_28':np.log1p(nw),
                'log_users_7':np.log1p(len(set(users[lo7:stop]))),
                'log_topic_age':np.log1p(day-int(di[0])+1),'days_since_post':int(day-di[stop-1]),
                'active_days_7':int((n[day-6:day+1]>0).sum()),'share_today':(n[day]+1)/(totals[day]+1),
                'growth_today':np.log1p(growth_today),'duplicate_28':1-len(set(window))/nw,
                'hhi_28':float(np.sum((uc/nw)**2)),'peak_28':n[max(0,day-27):day+1].max()/nw,
                'sync_28':float(mc[mc>=2].sum()/nw),'last_motif':day_motifs.get(day,'SPARSE'),
                'eligible_motif_days_28':sum(motifs.values())}
            for motif in ['HHLL','HLHL','HLLH','HLLL','LHHL','LHLH','LHLL','LLHH','LLHL','LLLH','LLLL']:
                row['motif_'+motif]=motifs[motif]/max(sum(motifs.values()),1)
            rows.append(row)
            if calendar[day]<=pd.Timestamp(cfg['train_end']):motif_counter.update([row['last_motif']])
    f=pd.DataFrame(rows).sort_values(['as_of','topic']).reset_index(drop=True)
    # Percentiles use the risk set visible on each prediction date, never all dates.
    components=['duplicate_28','hhi_28','peak_28','sync_28']
    rounded=f[components].round(12)
    f['rolling_hmi']=rounded.groupby(f.as_of).rank(pct=True).mean(axis=1)
    asof=pd.to_datetime(f.as_of)
    f['split']=np.select([asof<=pd.Timestamp(cfg['train_end']),
        asof.between(cfg['validation_start'],cfg['validation_end']),asof.between(cfg['test_start'],cfg['test_end'])],
        ['train','validation','test'],default='embargo')
    if write_results:
        f.to_csv(RES/'forecast_features.csv',index=False)
        pd.DataFrame(all_hot).to_csv(RES/'forecast_all_hotspot_outcomes.csv',index=False)
    top=[m for m,n in motif_counter.most_common() if m!='SPARSE'][:4]
    return f,top,pd.DataFrame(all_hot)


def ranking_metrics(frame,score):
    y=frame.target.to_numpy();pred=frame[['as_of','topic','target']].copy();pred['score']=score
    ranked=pred.sort_values(['as_of','score','topic'],ascending=[True,False,True])
    reviewed=ranked.groupby('as_of').head(5)
    return {'average_precision':average_precision_score(y,score),'roc_auc':roc_auc_score(y,score),
        'precision_top5_per_day':float(reviewed.target.mean()),
        'recall_top5_per_day':float(reviewed.target.sum()/y.sum()),
        'reviewed_candidates':len(reviewed),'reviewed_positives':int(reviewed.target.sum())}


def main():
    cfg=load_config()
    start=time.time();f,top,all_hot=make_rows()
    train=f.split.eq('train');val=f.split.eq('validation');test=f.split.eq('test')
    for name,mask in [('train',train),('validation',val),('test',test)]:
        if f.loc[mask,'target'].nunique()!=2:
            raise ValueError(name+' must contain positives and negatives; review support without tuning to test performance.')
    print('Forecast risk sets:',f.groupby('split').target.agg(['size','sum']).to_dict(),flush=True)
    # Normalize and cluster exclusively on training dates; future rows are assigned.
    scaler=StandardScaler().fit(f.loc[train,ACTIVITY+COORD])
    reps=scaler.transform(f[ACTIVITY+COORD])
    km=KMeans(6,n_init=20,random_state=28).fit(reps[train])
    cluster=km.predict(reps)
    features=f.copy()
    for c in range(6):features['past_cluster_'+str(c)]=(cluster==c).astype(int)
    pattern_cols=['motif_'+m for m in top]+['eligible_motif_days_28']+['past_cluster_'+str(c) for c in range(6)]
    specs={'Activity LR':ACTIVITY,'Activity + HMI LR':ACTIVITY+COORD,
        'Activity + HMI + patterns/clusters LR':ACTIVITY+COORD+pattern_cols}
    scores={'Recent volume':f.log_posts_1.to_numpy(),'Current growth':f.growth_today.to_numpy()}
    for name,cols in specs.items():
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,class_weight='balanced',random_state=28))
        model.fit(features.loc[train,cols],f.loc[train,'target']);scores[name]=model.predict_proba(features[cols])[:,1]
    for name,cols in [('Activity RF',ACTIVITY),('Activity + HMI + patterns/clusters RF',ACTIVITY+COORD+pattern_cols)]:
        model=RandomForestClassifier(n_estimators=250,min_samples_leaf=15,max_features='sqrt',
            class_weight='balanced_subsample',n_jobs=2,random_state=28)
        model.fit(features.loc[train,cols],f.loc[train,'target']);scores[name]=model.predict_proba(features[cols])[:,1]
    metrics=[];prediction=f[['topic','as_of','target_date','split','target']].copy()
    for name,score in scores.items():
        prediction[name]=score
        for split,mask in [('validation',val),('test',test)]:
            metrics.append({'model':name,'split':split,'n':int(mask.sum()),'positives':int(f.loc[mask,'target'].sum()),
                **ranking_metrics(f.loc[mask],score[mask])})
    prediction.to_csv(RES/'forecast_predictions.csv',index=False)
    met=pd.DataFrame(metrics);met.to_csv(RES/'forecast_metrics.csv',index=False)
    # Pick the model family by validation AP only, then compare nested features.
    vm=met[met.split=='validation'].set_index('model').average_precision
    family='RF' if vm['Activity RF']>vm['Activity LR'] else 'LR'
    base='Activity '+family;full='Activity + HMI + patterns/clusters '+family
    sel=met[met.split=='test'].set_index('model')
    tested=f.loc[test].reset_index(drop=True);y=tested.target.to_numpy()
    weeks=pd.to_datetime(tested.as_of).dt.to_period('W').astype(str)
    units=[np.flatnonzero(weeks.eq(w).to_numpy()) for w in weeks.unique()]
    rng=np.random.default_rng(28);deltas=[]
    bs=scores[base][test];fs=scores[full][test]
    for _ in range(400):
        idx=np.concatenate([units[i] for i in rng.integers(0,len(units),len(units))])
        if len(np.unique(y[idx]))<2:continue
        deltas.append(average_precision_score(y[idx],fs[idx])-average_precision_score(y[idx],bs[idx]))
    outcome_dates=pd.to_datetime(all_hot.target_date)
    all_test=all_hot[outcome_dates.between(pd.Timestamp(cfg['test_start'])+pd.Timedelta(days=1),pd.Timestamp(cfg['test_end'])+pd.Timedelta(days=1))]
    # Outcome census follows the configured next-day target range.
    prevalence=float(y.mean());ap_delta=float(sel.loc[full,'average_precision']-sel.loc[base,'average_precision'])
    dump('forecast_summary.json',{'task':'At day end predict next-day sample-relative growth for previously observed topics.',
        'candidate_rule':'By as-of: >=5 cumulative posts, >=3 cumulative users, >=1 post in past7d; >=20 sample posts in past7d.',
        'target_rule':'>=4 posts, >=3 users, >=2x smoothed share versus preceding7d on next day.',
        'splits':f.groupby('split').target.agg(['size','sum']).to_dict(orient='index'),
        'train_end':cfg['train_end'],'validation_start':cfg['validation_start'],'validation_end':cfg['validation_end'],'test_start':cfg['test_start'],'date_config':cfg,
        'test_end':tested.as_of.max(),'test_prevalence':prevalence,'test_topics':tested.topic.nunique(),
        'test_forecast_days':tested.as_of.nunique(),'train_frequent_motifs':top,
        'validation_selected_family':family,'base_model':base,'full_model':full,
        'test_ap_delta_full_minus_base':ap_delta,'weekly_bootstrap_replicates':len(deltas),
        'ap_delta_ci025':float(np.quantile(deltas,.025)),'ap_delta_ci975':float(np.quantile(deltas,.975)),
        'all_test_hotspot_topicdays':len(all_test),'candidate_test_hotspot_topicdays':int(y.sum()),
        'candidate_hotspot_coverage':float(y.sum()/len(all_test)),
        'elapsed_seconds':time.time()-start,
        'limits':['One audited research subset; outcomes are sample-defined, not official Weibo trend labels.',
                  'Already-observed-topic prediction excludes unseen new topics; coverage is explicitly reported.',
                  'Rolling HMI uses only 28 past/current days; it differs from full-period retrospective HMI.',
                  'No future full-period topic screening, HMI or source representations; clusters/motif selection/scalers fit only on configured training dates.',
                  'No MisBot future-trained score is used in the historical prediction experiment.',
                  'Forecast performance cannot establish CIB truth or inauthentic intent.']})
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
    names=['Recent volume','Current growth',base,full]
    for j,name in enumerate(names):
        prec,rec,_=precision_recall_curve(y,scores[name][test])
        axes[0].plot(rec,prec,color=COLORS[j],lw=1.8,label=f'{name.replace("Activity + HMI + patterns/clusters ","Full ")} (AP {sel.loc[name,"average_precision"]:.3f})')
    axes[0].axhline(prevalence,ls='--',color='#AAB6C3',label=f'Prevalence {prevalence:.3f}')
    axes[0].set(xlabel='Recall',ylabel='Precision',title='Held-out future dates');axes[0].legend(fontsize=7,frameon=False)
    axes[1].barh(range(4),[sel.loc[n,'average_precision'] for n in names],color=COLORS[:4])
    axes[1].set_yticks(range(4),['Recent volume','Current growth',base,'Full '+family],fontsize=9)
    axes[1].invert_yaxis();axes[1].axvline(prevalence,color='#AAB6C3',ls='--')
    axes[1].set(xlabel='Average precision (AP)',title='Do HMI / mined patterns add value?')
    save(fig,'14_nextday_prediction')
    print(met[met.split=='test'].to_string(index=False),flush=True)
    print('Validation-selected family',family,'AP difference',ap_delta,'95% weekly-bootstrap CI',np.quantile(deltas,[.025,.975]),flush=True)


if __name__=='__main__':main()

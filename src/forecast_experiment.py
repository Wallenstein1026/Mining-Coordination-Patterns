"""Forward next-day sample-hotspot test with past-only topic eligibility and features.

No full-period topic lists, HMI, representations or MisBot model are used.
The task is forecasting growth of already observed topics, not unseen topics/CIB.
"""
from collections import Counter, defaultdict
import time
import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiment_io import load_posts
from forecast_config import load_config
from run_analysis import COLORS, SEED, plt

ACTIVITY=['log_posts_1','log_posts_3','log_posts_7','log_posts_28','log_users_7',
          'log_topic_age','days_since_post','active_days_7','share_today','growth_today']
COORD=['duplicate_28','hhi_28','peak_28','sync_28','rolling_hmi']


def make_rows(cfg, cutoff=None):
    d=load_posts(cutoff=cutoff).sort_values('time').reset_index(drop=True)
    calendar=pd.date_range(d.time.min().floor('D'),d.time.max().floor('D'),freq='D')
    if cutoff is not None and calendar[-1]!=pd.Timestamp(cutoff).normalize():
        raise ValueError('Observations do not cover the requested next-day label boundary.')
    totals=d.groupby(d.time.dt.floor('D')).size().reindex(calendar,fill_value=0).to_numpy()
    broad={'疫情','新冠','新冠肺炎','新型冠状病毒','新型冠状病毒感染肺炎','肺炎[超话]','疫情[超话]','covid-19','新冠病毒'}
    groups=defaultdict(list)
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
    if not rows:
        raise ValueError('No eligible prediction rows in the requested observation window.')
    f=pd.DataFrame(rows).sort_values(['as_of','topic']).reset_index(drop=True)
    # Percentiles use the risk set visible on each prediction date, never all dates.
    components=['duplicate_28','hhi_28','peak_28','sync_28']
    rounded=f[components].round(12)
    f['rolling_hmi']=rounded.groupby(f.as_of).rank(pct=True).mean(axis=1)
    asof=pd.to_datetime(f.as_of)
    f['split']=np.select([asof<=pd.Timestamp(cfg['train_end']),
        asof.between(cfg['validation_start'],cfg['validation_end']),asof.between(cfg['test_start'],cfg['test_end'])],
        ['train','validation','test'],default='embargo')
    top=[m for m,n in motif_counter.most_common() if m!='SPARSE'][:4]
    return f,top,pd.DataFrame(all_hot,columns=['topic','target_date','posts'])


def ranking_metrics(frame, score, all_hotspots):
    labels = frame.target.to_numpy()
    ranked = frame[["as_of", "topic", "target"]].assign(score=score).sort_values(
        ["as_of", "score", "topic"], ascending=[True, False, True])
    reviewed = ranked.groupby("as_of").head(5)
    hits = int(reviewed.target.sum())
    positives = int(labels.sum())
    return {
        "average_precision": average_precision_score(labels, score),
        "roc_auc": roc_auc_score(labels, score),
        "precision_top5_per_day": hits / len(reviewed),
        "recall_top5_per_day": hits / positives,
        "end_to_end_recall_top5_per_day": hits / all_hotspots,
        "candidate_hotspot_coverage": positives / all_hotspots,
        "all_hotspot_topicdays": all_hotspots,
        "reviewed_candidates": len(reviewed),
        "reviewed_positives": hits,
    }


def family_models(family):
    return (
        f"Activity {family}",
        f"Activity + HMI {family}",
        f"Activity + HMI + patterns/clusters {family}",
    )


def fit_scores(frame, motifs):
    train = frame.split.eq("train")
    scaler = StandardScaler().fit(frame.loc[train, ACTIVITY + COORD])
    representations = scaler.transform(frame[ACTIVITY + COORD])
    clustering = KMeans(6, n_init=20, random_state=SEED).fit(representations[train])
    assignments = clustering.predict(representations)
    features = frame.copy()
    cluster_cols = [f"past_cluster_{index}" for index in range(6)]
    for index, column in enumerate(cluster_cols):
        features[column] = assignments == index

    pattern_cols = [f"motif_{motif}" for motif in motifs] + ["eligible_motif_days_28"] + cluster_cols
    ablations = [ACTIVITY, ACTIVITY + COORD, ACTIVITY + COORD + pattern_cols]
    scores = {"Recent volume": frame.log_posts_1.to_numpy(), "Current growth": frame.growth_today.to_numpy()}
    for family in ("LR", "RF"):
        for name, columns in zip(family_models(family), ablations):
            if family == "LR":
                model = make_pipeline(StandardScaler(), LogisticRegression(
                    C=1, max_iter=2000, class_weight="balanced", random_state=SEED))
            else:
                model = RandomForestClassifier(
                    n_estimators=250, min_samples_leaf=15, max_features="sqrt",
                    class_weight="balanced_subsample", n_jobs=2, random_state=SEED)
            model.fit(features.loc[train, columns], frame.loc[train, "target"])
            scores[name] = model.predict_proba(features[columns])[:, 1]
    return scores


def outcome_count(outcomes, first_as_of, last_as_of):
    first = pd.Timestamp(first_as_of) + pd.Timedelta(days=1)
    last = pd.Timestamp(last_as_of) + pd.Timedelta(days=1)
    return int(pd.to_datetime(outcomes.target_date).between(first, last).sum())


def ap_increments(average_precision, model_names):
    base, hmi, full = [average_precision[name] for name in model_names]
    return {"hmi_minus_activity": hmi - base, "patterns_minus_hmi": full - hmi, "full_minus_activity": full - base}


def weekly_ap(predictions, model_names):
    weeks = pd.to_datetime(predictions.as_of).dt.to_period("W-SUN")
    rows = []
    for week, group in predictions.groupby(weeks):
        record = {"week_start": str(week.start_time.date()), "n": len(group), "positives": int(group.target.sum())}
        average_precision = {
            name: average_precision_score(group.target, group[name]) if group.target.nunique() == 2 else np.nan
            for name in model_names
        }
        rows.append({**record, **ap_increments(average_precision, model_names)})
    return pd.DataFrame(rows)


def paired_bootstrap(predictions, base, full, replicates=400):
    weeks = pd.to_datetime(predictions.as_of).dt.to_period("W-SUN")
    blocks = list(predictions.groupby(weeks).indices.values())
    labels = predictions.target.to_numpy()
    base_scores, full_scores = predictions[base].to_numpy(), predictions[full].to_numpy()
    rng = np.random.default_rng(SEED)
    deltas = []
    if len(blocks) >= 2:
        for _ in range(replicates):
            selected = rng.integers(0, len(blocks), len(blocks))
            indices = np.concatenate([blocks[index] for index in selected])
            y = labels[indices]
            if len(np.unique(y)) == 2:
                deltas.append(
                    average_precision_score(y, full_scores[indices]) - average_precision_score(y, base_scores[indices]))
    interval = np.quantile(deltas, [.025, .975]).tolist() if deltas else [None, None]
    return interval, len(deltas), len(blocks)


def forecast_dates():
    return load_config()


def report_figure(predictions, metrics, model_names, report_split, output):
    names = ["Recent volume", "Current growth", *model_names]
    labels = ["Recent volume", "Current growth", "Activity", "Activity + HMI", "Activity + HMI + patterns/clusters"]
    y = predictions.target
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    for idx, name in enumerate(names):
        precision, recall, _ = precision_recall_curve(y, predictions[name])
        axes[0].plot(recall, precision, color=COLORS[idx], lw=1.8,
                     label=f"{labels[idx]} (AP {metrics.loc[name, 'average_precision']:.3f})")
    axes[0].axhline(y.mean(), ls="--", color="#AAB6C3", label=f"Prevalence {y.mean():.3f}")
    axes[0].set(xlabel="Recall", ylabel="Precision", title=f"{report_split.title()} precision–recall")
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].barh(range(len(names)), metrics.loc[names, "average_precision"], color=COLORS[:len(names)])
    axes[1].set_yticks(range(len(names)), labels, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].axvline(y.mean(), ls="--", color="#AAB6C3")
    axes[1].set(xlabel="Average precision (AP)", title="Feature ablation")
    output.save_figure(fig, "14_nextday_prediction")


def run_forecast(output, held_out=False, cfg=None):
    cfg = forecast_dates() if cfg is None else cfg
    start = time.perf_counter()
    report_split = "validation" if not held_out else "test"
    report_end = pd.Timestamp(cfg[f"{report_split}_end"])
    frame, motifs, outcomes = make_rows(cfg, cutoff=report_end + pd.Timedelta(days=1))
    frame.to_csv(output.results / "forecast_features.csv", index=False)
    outcomes.to_csv(output.results / "forecast_all_hotspot_outcomes.csv", index=False)
    splits = ["train", "validation"] if not held_out else ["train", "validation", "test"]
    for split in splits:
        section = frame[frame.split.eq(split)]
        if section.target.nunique() != 2:
            raise ValueError(f"{split} must contain positives and negatives to evaluate ranking performance.")
    counts = frame.groupby("split").target.agg(["size", "sum"])
    print("Forecast risk sets:", counts.to_dict(orient="index"), flush=True)

    scores = fit_scores(frame, motifs)
    predictions = frame[["topic", "as_of", "target_date", "split", "target"]].assign(**scores)
    records = []
    for split in splits[1:]:
        subset = predictions[predictions.split.eq(split)]
        total = outcome_count(outcomes, cfg[f"{split}_start"], cfg[f"{split}_end"])
        for name in scores:
            records.append({"model": name, "split": split, "n": len(subset),
                            "positives": int(subset.target.sum()), **ranking_metrics(subset, subset[name], total)})
    metrics = pd.DataFrame(records)
    predictions.to_csv(output.results / "forecast_predictions.csv", index=False)
    metrics.to_csv(output.results / "forecast_metrics.csv", index=False)

    val_ap = metrics[metrics.split.eq("validation")].set_index("model").average_precision
    family = "RF" if val_ap["Activity RF"] > val_ap["Activity LR"] else "LR"
    model_names = family_models(family)
    selected = metrics[metrics.split.eq(report_split)].set_index("model")
    evaluated = predictions[predictions.split.eq(report_split)]
    increments = ap_increments(selected.average_precision, model_names)
    weekly = weekly_ap(evaluated, model_names)
    weekly.to_csv(output.results / "forecast_weekly_ap_deltas.csv", index=False)
    bootstrap = None
    if held_out:
        interval, replicates, blocks = paired_bootstrap(evaluated, model_names[0], model_names[-1])
        bootstrap = {"interval_95": interval, "valid_replicates": replicates, "week_blocks": blocks}

    report_figure(evaluated, selected, model_names, report_split, output)
    output.write_json("forecast_summary.json", {
        "task": "Rank previously observed topics for next-day sample-relative hotspots.",
        "candidate_rule": ">=5 cumulative posts, >=3 cumulative users, >=1 post in past 7 days, >=20 sample posts in past 7 days.",
        "target_rule": ">=4 posts, >=3 users, >=2x smoothed share relative to previous 7 days.",
        "date_config": cfg, "reported_split": report_split, "splits": counts.to_dict(orient="index"),
        "forecast_days": evaluated.as_of.nunique(), "prevalence": float(evaluated.target.mean()),
        "train_frequent_motifs": motifs, "validation_selected_family": family,
        "selected_models": model_names, "ap_deltas": increments, "paired_weekly_bootstrap": bootstrap,
        "all_hotspot_topicdays": int(selected.all_hotspot_topicdays.iloc[0]),
        "candidate_hotspot_topicdays": int(evaluated.target.sum()),
        "candidate_hotspot_coverage": float(selected.candidate_hotspot_coverage.iloc[0]),
        "elapsed_seconds": time.perf_counter() - start,
        "limits": [
            "Sample-defined hotspots in one audited research subset.",
            "Candidate coverage includes unseen or otherwise ineligible hotspots in the denominator.",
            "The historical model uses only past-visible candidates and rolling HMI.",
            "Standardization, clusters and motif selection use training dates; model family uses validation AP.",
            "The forecast excludes retrospective MisBot scores.",
            "A small number of weekly blocks limits precision of the interval.",
        ],
    })
    print(f"{report_split.title()} metrics and AP differences saved to {output.results}", flush=True)


def validate_forecast(output, held_out=False, cfg=None):
    cfg = forecast_dates() if cfg is None else cfg
    checks = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    read_options = {"float_precision": "round_trip", "dtype": {"topic": str}, "keep_default_na": False}
    f = pd.read_csv(output.results / "forecast_features.csv", **read_options)
    p = pd.read_csv(output.results / "forecast_predictions.csv", **read_options)
    met = pd.read_csv(output.results / "forecast_metrics.csv")
    all_hot = pd.read_csv(output.results / "forecast_all_hotspot_outcomes.csv", **read_options)
    summary = json.loads((output.results / "forecast_summary.json").read_text(encoding="utf-8"))
    report_split = "validation" if not held_out else "test"
    dates = pd.to_datetime(f.as_of)
    target_dates = pd.to_datetime(f.target_date)

    check("Saved results use the frozen configuration and requested evaluation split",
          summary["date_config"] == cfg and summary["reported_split"] == report_split)
    check("One candidate per topic and date", not f.duplicated(["topic", "as_of"]).any())
    check("Outcomes are unique by topic and target date", not all_hot.duplicated(["topic", "target_date"]).any())
    check("Features end before the next-day label",
          (pd.to_datetime(f.feature_end) <= dates).all() and (target_dates - dates).eq(pd.Timedelta(days=1)).all())
    check("Past-only candidate requirements",
          (f.history_posts.ge(5) & f.history_users.ge(3) & f.posts_7.ge(1) & f.sample_posts_7.ge(20)).all())
    check("Outcome rule reconciles with labels",
          f.target.eq((f.nextday_posts.ge(4) & f.nextday_users.ge(3) & f.target_growth.ge(2)).astype(int)).all())
    for split in ("validation", "test"):
        expected = dates.between(cfg[f"{split}_start"], cfg[f"{split}_end"])
        check(f"Exact {split} split membership", f.split.eq(split).equals(expected))
    check("Exact training split membership", f.split.eq("train").equals(dates.le(cfg["train_end"])))
    check("Calendar splits are separated from preceding labels",
          target_dates[f.split.eq("train")].max() < dates[f.split.eq("validation")].min() and
          (not held_out or target_dates[f.split.eq("validation")].max() < dates[f.split.eq("test")].min()))
    keys = ["topic", "as_of"]
    check("Predictions align with features",
          p[keys + ["target_date", "split", "target"]].equals(f[keys + ["target_date", "split", "target"]]))

    cutoff = pd.Timestamp(cfg["validation_start"]) + (
        pd.Timestamp(cfg["validation_end"]) - pd.Timestamp(cfg["validation_start"])) // 2
    prefix, motifs, _ = make_rows(cfg, cutoff=cutoff.floor("D"))
    predictors = ACTIVITY + COORD + ["eligible_motif_days_28"] + [c for c in f if c.startswith("motif_")]
    historical = f[dates < cutoff.floor("D")].set_index(keys).sort_index()
    rebuilt = prefix.set_index(keys).sort_index()
    check("Future truncation preserves historical candidates", historical.index.equals(rebuilt.index))
    check("Future truncation preserves every historical predictor",
          np.allclose(historical[predictors], rebuilt[predictors], equal_nan=True))
    check("Training motif selection is invariant to later data", motifs == summary["train_frequent_motifs"])
    ranks = f[["duplicate_28", "hhi_28", "peak_28", "sync_28"]].round(12).groupby(f.as_of).rank(pct=True)
    check("Rolling HMI recomputes", np.allclose(ranks.mean(axis=1), f.rolling_hmi))

    splits = ["validation"] if not held_out else ["validation", "test"]
    names = ["Recent volume", "Current growth", *family_models("LR"), *family_models("RF")]
    check("All development and evaluation comparators are present exactly once",
          set(zip(met.split, met.model)) == {(split, name) for split in splits for name in names} and
          len(met) == len(splits) * len(names))
    for split in splits:
        part = p[p.split.eq(split)]
        total = outcome_count(all_hot, cfg[f"{split}_start"], cfg[f"{split}_end"])
        positive_keys = set(zip(part.loc[part.target.eq(1), "topic"], part.loc[part.target.eq(1), "target_date"]))
        outcome_keys = set(zip(all_hot.topic, all_hot.target_date))
        check(f"{split} positives occur in the full hotspot census", positive_keys <= outcome_keys)
        for row in met[met.split.eq(split)].itertuples():
            actual = ranking_metrics(part, part[row.model], total)
            check(f"Recompute {split} metrics for {row.model}",
                  row.n == len(part) and row.positives == part.target.sum() and
                  all(np.isclose(getattr(row, key), value, atol=1e-12, rtol=0) for key, value in actual.items()))

    val_ap = met[met.split.eq("validation")].set_index("model").average_precision
    family = "RF" if val_ap["Activity RF"] > val_ap["Activity LR"] else "LR"
    selected = family_models(family)
    check("Model family selected on validation only",
          summary["validation_selected_family"] == family and summary["selected_models"] == list(selected))
    scores = met[met.split.eq(report_split)].set_index("model")
    evaluated = p[p.split.eq(report_split)]
    actual = ap_increments(scores.average_precision, selected)
    check("Selected-family AP increments recompute",
          all(np.isclose(value, summary["ap_deltas"][key], atol=1e-12, rtol=0) for key, value in actual.items()))
    weekly = pd.read_csv(output.results / "forecast_weekly_ap_deltas.csv")
    pd.testing.assert_frame_equal(weekly, weekly_ap(evaluated, selected), atol=1e-12, rtol=0)
    check("Weekly AP differences and supports recompute", True)
    if held_out:
        interval, replicates, weeks = paired_bootstrap(evaluated, selected[0], selected[-1])
        saved = summary["paired_weekly_bootstrap"]
        check("Weekly bootstrap interval and support recompute",
              saved == {"interval_95": interval, "valid_replicates": replicates, "week_blocks": weeks})
    else:
        check("Development run does not include held-out predictions or a test interval",
              not p.split.eq("test").any() and summary["paired_weekly_bootstrap"] is None)

    output.write_json("forecast_validation.json", {"passed": len(checks), "checks": checks})
    print(f"{len(checks)} forward-forecast checks passed ({report_split}).")

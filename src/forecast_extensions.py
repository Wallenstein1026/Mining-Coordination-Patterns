"""Isolate forecast feature groups and compare past-only candidate rules."""
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from experiment_io import EXPERIMENT_DIR, ExperimentOutput, load_posts
from forecast_experiment import ACTIVITY, COORD, forecast_dates, outcome_count, ranking_metrics
from run_analysis import SEED, plt


ORIGINAL = (5, 3, 7)
RULES = (ORIGINAL, (3, 2, 7), (2, 2, 7), (5, 3, 14), (3, 2, 14))
COMPONENTS = COORD[:4]
EXCLUDED = {'疫情', '新冠', '新冠肺炎', '新型冠状病毒', '新型冠状病毒感染肺炎',
            '肺炎[超话]', '疫情[超话]', 'covid-19', '新冠病毒'}
MOTIFS = ('HHLL', 'HLHL', 'HLLH', 'HLLL', 'LHHL', 'LHLH', 'LHLL', 'LLHH', 'LLHL', 'LLLH', 'LLLL')
FULL_FOREST = "Activity + HMI + patterns/clusters RF"


def read_original(name):
    return pd.read_csv(EXPERIMENT_DIR / "forecast" / "results" / name,
                       dtype={"topic": str}, keep_default_na=False, float_precision="round_trip")


def expanded_rows(posts, cfg):
    """Compute numeric history once for the union of the prespecified candidate rules."""
    posts = posts.sort_values("time").reset_index(drop=True)
    dates = pd.date_range(posts.time.min().floor("D"), posts.time.max().floor("D"), freq="D")
    day_index = ((posts.time.dt.floor("D") - dates[0]).dt.days).to_numpy()
    minutes = posts.time.dt.floor("min").astype("int64").to_numpy()
    hours = posts.time.dt.hour.to_numpy()
    users = posts.user_id.to_numpy()
    text = posts.norm.to_numpy()
    daily_total = np.bincount(day_index, minlength=len(dates))
    weekly_total = np.convolve(daily_total, np.ones(7, dtype=int))[:len(dates)]
    dates_text = dates.strftime("%Y-%m-%d").to_numpy()
    membership = defaultdict(list)
    for index, tags in enumerate(posts.tags):
        for tag in tags:
            if tag not in EXCLUDED:
                membership[tag].append(index)

    rows = []
    for topic, indices in membership.items():
        if len(indices) < 2:
            continue
        days, people, words, clocks = day_index[indices], users[indices], text[indices], minutes[indices]
        daily = np.bincount(days, minlength=len(dates))
        weekly = np.convolve(daily, np.ones(7, dtype=int))[:len(dates)]
        observed = np.unique(days)
        starts = np.searchsorted(days, observed)
        stops = np.searchsorted(days, observed, side="right")
        daily_users = {day: len(set(people[start:stop])) for day, start, stop in zip(observed, starts, stops)}
        motifs = {}
        for day, start, stop in zip(observed, starts, stops):
            if stop - start >= 4:
                bins = np.bincount(hours[indices[start:stop]] // 6, minlength=4)
                motifs[day] = "".join("H" if value > np.median(bins) else "L" for value in bins)
        seen = set()
        last_stop = 0
        for day in range(max(7, days[0]), min(len(dates) - 1, days[-1] + 14)):
            stop = np.searchsorted(days, day, side="right")
            seen.update(people[last_stop:stop])
            last_stop = stop
            age = day - days[stop - 1]
            if stop < 2 or len(seen) < 2 or age >= 14 or weekly_total[day] < 20:
                continue
            start = np.searchsorted(days, day - 27)
            recent = np.searchsorted(days, day - 6)
            window_count = stop - start
            user_counts = np.asarray(list(Counter(people[start:stop]).values()))
            minute_counts = np.asarray(list(Counter(clocks[start:stop]).values()))
            previous_share = (weekly[day - 1] + 1) / (weekly_total[day - 1] + 1)
            today_share = (daily[day] + 1) / (daily_total[day] + 1)
            next_share = (daily[day + 1] + 1) / (daily_total[day + 1] + 1)
            reference = (weekly[day] + 1) / (weekly_total[day] + 1)
            past_motifs = Counter(pattern for when, pattern in motifs.items() if day - 27 <= when <= day)
            motif_days = sum(past_motifs.values())
            row = {
                "topic": topic, "as_of": dates_text[day], "target_date": dates_text[day + 1],
                "target": int(daily[day + 1] >= 4 and daily_users.get(day + 1, 0) >= 3 and next_share / reference >= 2),
                "history_posts": stop, "history_users": len(seen), "days_since_post": age,
                "log_posts_1": np.log1p(daily[day]), "log_posts_3": np.log1p(daily[day - 2:day + 1].sum()),
                "log_posts_7": np.log1p(weekly[day]), "log_posts_28": np.log1p(window_count),
                "log_users_7": np.log1p(len(set(people[recent:stop]))), "log_topic_age": np.log1p(day - days[0] + 1),
                "active_days_7": int((daily[day - 6:day + 1] > 0).sum()), "share_today": today_share,
                "growth_today": np.log1p(today_share / previous_share),
                "duplicate_28": 1 - len(set(words[start:stop])) / window_count,
                "hhi_28": np.square(user_counts / window_count).sum(),
                "peak_28": daily[max(0, day - 27):day + 1].max() / window_count,
                "sync_28": minute_counts[minute_counts >= 2].sum() / window_count,
                "last_motif": motifs.get(day, "SPARSE"), "eligible_motif_days_28": motif_days,
            }
            row.update({f"motif_{pattern}": past_motifs[pattern] / max(motif_days, 1) for pattern in MOTIFS})
            rows.append(row)
    frame = pd.DataFrame(rows)
    # Restrict the candidate study to the original observation calendar.
    return frame[frame.as_of.le(cfg["test_end"])]


def candidates(frame, rule):
    posts, users, days = rule
    result = frame[frame.history_posts.ge(posts) & frame.history_users.ge(users)
                   & frame.days_since_post.lt(days)].sort_values(["as_of", "topic"]).reset_index(drop=True)
    # Recompute each day's ranks on that day's candidate list.
    result["rolling_hmi"] = result[COMPONENTS].round(12).groupby(result.as_of).rank(pct=True).mean(axis=1)
    return result


def feature_groups(frame, train):
    numeric = ACTIVITY + COORD
    scaler = StandardScaler().fit(frame.loc[train, numeric])
    vectors = scaler.transform(frame[numeric])
    groups = KMeans(6, n_init=20, random_state=SEED).fit(vectors[train])
    assignments = groups.predict(vectors)
    cluster = [f"cluster_{index}" for index in range(6)]
    result = frame.assign(**{column: assignments == i for i, column in enumerate(cluster)})
    counts = Counter(frame.loc[train & frame.last_motif.ne("SPARSE"), "last_motif"])
    top = sorted(counts, key=lambda key: (-counts[key], key))[:4]
    modal = [f"motif_{pattern}" for pattern in top]
    return result, modal, ["eligible_motif_days_28"], cluster


def forest_scores(frame, columns, train, evaluate):
    forest = RandomForestClassifier(n_estimators=250, min_samples_leaf=15, max_features="sqrt",
                                    class_weight="balanced_subsample", n_jobs=2, random_state=SEED)
    forest.fit(frame.loc[train, columns], frame.loc[train, "target"])
    return forest.predict_proba(frame.loc[evaluate, columns])[:, 1]


def evaluate_split(frame, columns, train, mask, total):
    part = frame.loc[mask]
    score = forest_scores(frame, columns, train, mask)
    metrics = ranking_metrics(part, score, total)
    return {"n": len(part), "positives": int(part.target.sum()), **metrics}


def ablate(original, predictions, outcomes, cfg, output):
    train = original.split.eq("train")
    test = original.split.eq("test")
    total = outcome_count(outcomes, cfg["test_start"], cfg["test_end"])
    test_predictions = predictions.loc[predictions.split.eq("test")]
    frame, motif, days, cluster = feature_groups(original, train)
    baseline = ACTIVITY + COORD
    added = {"motif_only": motif, "days_only": days, "cluster_only": cluster,
             "motif_and_days": motif + days, "motif_and_cluster": motif + cluster,
             "days_and_cluster": days + cluster}
    rows = []
    for label, name in [("activity", "Activity RF"), ("activity_hmi", "Activity + HMI RF"),
                        ("all_three", FULL_FOREST)]:
        rows.append({"variant": label, **ranking_metrics(test_predictions, test_predictions[name], total)})
    for name, columns in added.items():
        rows.append({"variant": name, **evaluate_split(frame, baseline + columns, train, test, total)})
    measures = pd.DataFrame(rows)
    measures["ap_delta_vs_activity_hmi"] = measures.average_precision - measures.average_precision.iloc[1]
    measures.to_csv(output.results / "feature_ablation.csv", index=False)
    leave_rows = []
    weeks = pd.to_datetime(test_predictions.as_of).dt.to_period("W-SUN")
    for week in sorted(weeks.unique()):
        kept = test_predictions[weeks.ne(week)]
        delta = average_precision_score(kept.target, kept[FULL_FOREST]) - average_precision_score(kept.target, kept["Activity RF"])
        leave_rows.append({"omitted_week": week.start_time.date(), "positives": int(kept.target.sum()), "ap_delta": delta})
    pd.DataFrame(leave_rows).to_csv(output.results / "leave_one_week_out.csv", index=False)
    return measures


def compare_candidates(expanded, original, original_predictions, outcomes, cfg, output):
    outcome_keys = pd.MultiIndex.from_frame(outcomes[["topic", "target_date"]])
    expanded_keys = pd.MultiIndex.from_frame(expanded[["topic", "target_date"]])
    assert np.array_equal(expanded_keys.isin(outcome_keys), expanded.target.eq(1))
    round_trip = candidates(expanded, ORIGINAL)
    keys = ["topic", "as_of", "target_date", "target"]
    assert round_trip[keys].equals(original[keys]), "Original eligibility or next-day label changed."
    assert np.allclose(round_trip[ACTIVITY + COORD], original[ACTIVITY + COORD], atol=1e-12, rtol=0)
    validation_total = outcome_count(outcomes, cfg["validation_start"], cfg["validation_end"])
    test_total = outcome_count(outcomes, cfg["test_start"], cfg["test_end"])
    base = {split: original_predictions[original_predictions.split.eq(split)] for split in ("validation", "test")}
    variant_rows = []
    for rule in RULES:
        print(f"Candidate rule: {rule}", flush=True)
        if rule == ORIGINAL:
            part = base["validation"]
            metrics = {"n": len(part), "positives": int(part.target.sum()),
                       **ranking_metrics(part, part[FULL_FOREST], validation_total)}
        else:
            section = expanded[expanded.as_of.le(cfg["validation_end"])]
            frame = candidates(section, rule)
            train = frame.as_of.le(cfg["train_end"])
            val = frame.as_of.between(cfg["validation_start"], cfg["validation_end"])
            frame, motif, days, cluster = feature_groups(frame, train)
            metrics = evaluate_split(frame, ACTIVITY + COORD + motif + days + cluster, train, val, validation_total)
        variant_rows.append({"rule": rule, **metrics})
    # The single winning rule is selected before any expanded-candidate test metric is computed.
    selected = max(variant_rows, key=lambda row: (row["reviewed_positives"], row["candidate_hotspot_coverage"], -row["n"]))
    rule = selected["rule"]
    pd.DataFrame(variant_rows).to_csv(output.results / "candidate_validation.csv", index=False)
    part = base["test"]
    baseline = {"n": len(part), "positives": int(part.target.sum()),
                **ranking_metrics(part, part[FULL_FOREST], test_total)}
    if rule == ORIGINAL:
        result = baseline
    else:
        frame = candidates(expanded, rule)
        train = frame.as_of.le(cfg["train_end"])
        mask = frame.as_of.between(cfg["test_start"], cfg["test_end"])
        frame, motif, days, cluster = feature_groups(frame, train)
        result = evaluate_split(frame, ACTIVITY + COORD + motif + days + cluster, train, mask, test_total)
    testing = pd.DataFrame([{"variant": "original", "rule": ORIGINAL, **baseline},
                            {"variant": "validation_selected", "rule": rule, **result}])
    testing.to_csv(output.results / "candidate_test.csv", index=False)
    output.write_json("candidate_selection.json", {
        "eligible_rules": RULES, "selected_rule": rule, "tie_break": "validation top-five hits, coverage, then fewer candidates",
        "baseline_keys_labels_and_features_match": True,
    })
    return pd.DataFrame(variant_rows), testing


def rolling_validation(original, outcomes, output):
    rows = []
    for first in pd.date_range("2020-09-27", "2020-10-18", freq="7D"):
        last = first + pd.Timedelta(days=6)
        print(f"Rolling check: {first.date()}", flush=True)
        train_end = first - pd.Timedelta(days=2)
        section = original[original.as_of.le(str(last.date()))].reset_index(drop=True)
        train = section.as_of.le(str(train_end.date()))
        evaluate = section.as_of.between(str(first.date()), str(last.date()))
        frame, motif, days, cluster = feature_groups(section, train)
        total = outcome_count(outcomes, first, last)
        base = evaluate_split(frame, ACTIVITY, train, evaluate, total)
        full = evaluate_split(frame, ACTIVITY + COORD + motif + days + cluster, train, evaluate, total)
        rows.append({"validation_start": str(first.date()), "validation_end": str(last.date()), "positives": full["positives"],
                     "base_ap": base["average_precision"], "full_ap": full["average_precision"],
                     "ap_delta": full["average_precision"] - base["average_precision"],
                     "base_hits": base["reviewed_positives"], "full_hits": full["reviewed_positives"]})
    pd.DataFrame(rows).to_csv(output.results / "rolling_validation.csv", index=False)


def plot_results(ablation, validation, output):
    names = ["activity_hmi", "motif_only", "days_only", "cluster_only", "all_three"]
    picked = ablation.set_index("variant").loc[names]
    labels = ["Activity + HMI", "+ Time patterns", "+ Pattern-day count", "+ Clusters", "+ All three"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    a.barh(labels, picked.average_precision, color=["#A8BAC6"] + ["#216C8B"] * 4)
    a.invert_yaxis()
    a.set_xlim(0, max(picked.average_precision)*1.1)
    a.set(xlabel="Average precision on original test dates", title="(a) Feature-group ablation")
    for index, row in validation.iterrows():
        rule_label = "/".join(map(str, row.rule))
        b.scatter(row.candidate_hotspot_coverage, row.reviewed_positives, s=30, color="#217B72")
        b.annotate(rule_label, (row.candidate_hotspot_coverage, row.reviewed_positives),
                   textcoords="offset points", xytext=((-33, 6) if index == len(validation) - 1 else (3, 6)), fontsize=8)
    b.margins(x=.17, y=.2)
    b.set(xlabel="Hotspot coverage on validation dates", ylabel="Daily top-five hits, summed",
          title="(b) Candidate rule: posts / authors / days")
    output.save_figure(fig, "17_forecast_extensions")


def main():
    cfg = forecast_dates()
    output = ExperimentOutput(EXPERIMENT_DIR / "extensions" / "forecast")
    original = read_original("forecast_features.csv")
    predictions = read_original("forecast_predictions.csv")
    outcomes = read_original("forecast_all_hotspot_outcomes.csv")
    print("Building past-only candidate alternatives", flush=True)
    expanded = expanded_rows(load_posts(), cfg)
    validation, _ = compare_candidates(expanded, original, predictions, outcomes, cfg, output)
    del expanded
    print("Separating forecast feature groups", flush=True)
    ablation = ablate(original, predictions, outcomes, cfg, output)
    rolling_validation(original, outcomes, output)
    plot_results(ablation, validation, output)


if __name__ == "__main__":
    main()

"""Recount experimental evidence from the standalone descriptive output."""
import json

from run_analysis import style
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from mining_experiment import cluster_labels


def validate_mining(output, posts):
    results = output.results
    checks = []

    def check(label, condition):
        if not condition:
            raise AssertionError(label)
        checks.append(label)

    csv_options = {"dtype": {"topic": str}, "keep_default_na": False}
    topics = pd.read_csv(results / "topic_features.csv", **csv_options)
    check("Topics are unique and meet eligibility thresholds",
          topics.topic.is_unique and (topics.posts.ge(30) & topics.users.ge(5)).all())
    ranks = topics[["duplicate", "user_hhi", "peak_day", "same_minute"]].round(12).rank(pct=True)
    check("HMI recomputes from component ranks", np.allclose(topics.hmi, ranks.mean(axis=1)))
    selected = json.loads((results / "selected_clustering.json").read_text(encoding="utf-8"))
    with np.load(results / "topic_representations.npz") as reps:
        check("Topic representations are finite and aligned",
              all(np.isfinite(reps[name]).all() and len(reps[name]) == len(topics) for name in reps.files))
        check("Selected clustering and silhouette reconcile",
              topics.cluster.nunique() == selected["k"] and
              np.isclose(silhouette_score(reps["fused"], topics.cluster), selected["silhouette"], atol=1e-12, rtol=0))
    stability = pd.read_csv(results / "cluster_stability_comparison.csv")
    check("All six representation/method pairs have conditional stability comparisons",
          len(stability) == 6 and stability.k.eq(selected["k"]).all() and stability.runs.eq(20).all()
          and stability.ari_mean.between(-1, 1).all())


    tags = [set(values) for values in posts.tags]
    rules = pd.read_csv(results / "association_rules.csv", keep_default_na=False)
    for rule in rules.head(15).itertuples():
        lhs, rhs = set(rule.antecedent.split(" + ")), set(rule.consequent.split(" + "))
        joint = sum(lhs | rhs <= transaction for transaction in tags)
        left = sum(lhs <= transaction for transaction in tags)
        right = sum(rhs <= transaction for transaction in tags)
        check(f"Recount rule {rule.Index + 1}",
              joint == rule.count and np.allclose(
                  [joint / len(tags), joint / left, joint * len(tags) / (left * right)],
                  [rule.support, rule.confidence, rule.lift], atol=1e-10, rtol=0))

    motifs = pd.read_csv(results / "temporal_motifs.csv", **csv_options)
    summary = pd.read_csv(results / "temporal_motif_summary.csv")
    check("Temporal windows and motif support reconcile",
          motifs.n.ge(4).all() and not motifs.duplicated(["topic", "date"]).any()
          and summary.windows.sum() == len(motifs)
          and np.isclose(summary.support.sum(), int(not motifs.empty), atol=1e-10, rtol=0))
    counts = pd.read_csv(results / "pattern_method_comparison.csv").pivot(
        index="min_support", columns="method", values="itemsets")
    check("Apriori and FP-growth saved itemset counts match at each threshold",
          counts.Apriori.eq(counts["FP-growth"]).all())
    output.write_json("validation.json", {"passed": len(checks), "checks": checks})
    print(f"{len(checks)} mining evidence checks passed.")


def validate_retrospective(output, posts, style_model=None):
    results = output.results
    checks = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    def read_csv(name):
        return pd.read_csv(results / name, dtype={"topic": str})

    dates = posts.time.dt.floor("D")
    profiles = read_csv("proposal_topic_profiles.csv")
    daily = read_csv("topic_day_growth.csv")
    hot = read_csv("hotspot_windows.csv")
    summary = json.loads((results / "proposal_summary.json").read_text(encoding="utf-8"))
    check("Topic-date growth rows are unique", not daily.duplicated(["topic", "date"]).any())
    check("Hotspot counts reconcile",
          len(hot) == summary["hotspot_windows"] and hot.topic.nunique() == summary["hotspot_topics"])
    flags = daily.history_eligible & daily.posts.ge(4) & daily.users.ge(3) & daily.share_growth_ratio.ge(2)
    check("Hotspots reconcile with the configured thresholds",
          daily.hotspot.equals(flags) and set(zip(hot.topic, hot.date)) == set(zip(daily.loc[flags, "topic"], daily.loc[flags, "date"])))

    cases = json.loads((results / "hotspot_cases.json").read_text(encoding="utf-8"))
    check("Three distinct cases per HMI tail",
              len({case["topic"] for case in cases}) == 6 and
              all(sum(case["hmi_group"] == name for case in cases) == 3 for name in ["Low HMI", "High HMI"]))
    sample = hot.nlargest(30, "posts").drop_duplicates("topic").head(5)[["topic", "date"]].to_dict("records")
    totals = dates.value_counts()
    for index, case in enumerate(sample + cases, 1):
        topic, day = case["topic"], pd.Timestamp(case["date"])
        topic_mask = posts.tags.map(lambda tags: topic in tags)
        current = topic_mask & dates.eq(day)
        previous = dates.ge(day - pd.Timedelta(days=7)) & dates.lt(day)
        saved = daily.loc[daily.topic.eq(topic) & daily.date.eq(case["date"])].iloc[0]
        check(f"Source-level post/user recount for event {index}",
              current.sum() == saved.posts and posts.loc[current, "user_id"].nunique() == saved.users)
        if saved.history_eligible:
            prior = (topic_mask & previous).sum()
            ratio = ((current.sum() + 1) / (totals.loc[day] + 1)) / ((prior + 1) / (previous.sum() + 1))
            check(f"Source-level share-growth recount for event {index}",
                  prior == saved.previous_7d_posts and previous.sum() == saved.previous_7d_sample_posts
                  and np.isclose(ratio, saved.share_growth_ratio, atol=1e-9, rtol=0))

    ranks = profiles[["duplicate", "user_hhi", "peak_day", "same_minute"]].round(12).rank(pct=True)
    check("HMI recomputes from component ranks", np.allclose(ranks.mean(axis=1), profiles.hmi))
    has_style = summary.get("style_available", "user_balanced_style" in profiles)
    signals = ["hmi"]
    if has_style:
        signals.append("user_balanced_style")
        for index, topic in enumerate(profiles.topic.head(3), 1):
            sample_posts = posts.loc[posts.tags.map(lambda tags: topic in tags)]
            scores = style_model.predict_proba(np.asarray([style(s) for s in sample_posts.content]))[:, 1]
            user_mean = sample_posts.assign(score=scores).groupby("user_id").score.mean().mean()
            saved = profiles.loc[profiles.topic.eq(topic), "user_balanced_style"].iloc[0]
            check(f"Equal-user style aggregation for sample topic {index}", np.isclose(user_mean, saved, atol=1e-12, rtol=0))

    with np.load(results / "topic_representations.npz") as reps:
        fused = reps["fused"]
    standardized = StandardScaler().fit_transform(profiles[signals])
    selected = json.loads((results / "selected_clustering.json").read_text(encoding="utf-8"))
    augmented = np.c_[fused, .5 * standardized / np.sqrt(len(signals))]
    labels = cluster_labels(augmented, selected["method"], selected["k"])
    saved_labels = profiles.cib_informed_cluster
    check("Augmented partition independently refits", adjusted_rand_score(labels, saved_labels) == 1)
    check("Augmented common silhouette independently recomputes",
          np.isclose(silhouette_score(fused, labels), summary["cib_informed_common_silhouette"], atol=1e-12, rtol=0))
    check("Augmented partition ARI independently recomputes",
          np.isclose(adjusted_rand_score(profiles.cluster, labels), summary["cib_informed_ari_vs_baseline"], atol=1e-12, rtol=0))

    motifs = read_csv("contextual_motif_windows.csv")
    original = read_csv("temporal_motifs.csv")
    check("Context annotation preserves the original motif windows",
          motifs[["topic", "date", "n", "motif"]].equals(original[["topic", "date", "n", "motif"]]))
    check("Motif hotspot flags reconcile",
          set(zip(motifs.loc[motifs.hotspot, "topic"], motifs.loc[motifs.hotspot, "date"])) == set(zip(hot.topic, hot.date)))
    contexts = read_csv("contextual_motif_comparison.csv")
    q1, q3 = profiles.hmi.quantile([.25, .75])
    for name, part in [
        ("Low HMI", motifs[motifs.hmi.le(q1)]), ("High HMI", motifs[motifs.hmi.ge(q3)]),
        ("Hotspot day", motifs[motifs.hotspot]), ("Other active day", motifs[~motifs.hotspot]),
    ]:
        frequencies = part.groupby(["topic", "motif"]).size().unstack(fill_value=0)
        within_topic = frequencies.div(frequencies.sum(axis=1), axis=0)
        saved = contexts[contexts.context.eq(name)].set_index("motif")
        check(f"Topic-weighted motif proportions and supports: {name}",
              np.allclose(within_topic.mean().reindex(saved.index, fill_value=0), saved.topic_weighted_share)
              and saved.eligible_windows.eq(len(part)).all()
              and saved.topics_with_eligible_windows.eq(part.topic.nunique()).all())

    output.write_json("proposal_validation.json", {"passed": len(checks), "checks": checks})
    print(f"{len(checks)} retrospective evidence checks passed.")

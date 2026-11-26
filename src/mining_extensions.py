"""Sensitivity checks for the retrospective Weibo experiments."""
from collections import defaultdict
from itertools import product
import re
import time

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler, normalize

from experiment_io import EXPERIMENT_DIR, ExperimentOutput, load_posts
from mining_experiment import cluster_labels, transaction_frame
from run_analysis import SEED, plt


ORIGINAL_RESULTS = EXPERIMENT_DIR / "descriptive" / "results"
COORD = ["duplicate", "user_hhi", "peak_day", "same_minute"]
OTHER_BEHAVIOR = ["night_share", "repost_mark", "hour_entropy"]
FIGURE_BLUE, FIGURE_GOLD = "#216C8B", "#CE9B39"


def read_original(name):
    return pd.read_csv(ORIGINAL_RESULTS / name, dtype={"topic": str}, keep_default_na=False,
                       float_precision="round_trip")


def behavior_vector(topics, profiles, columns):
    return normalize(StandardScaler().fit_transform(np.c_[topics[columns], profiles]))


def text_matrix(documents):
    return TfidfVectorizer(analyzer="char", ngram_range=(2, 3), min_df=3, max_df=.9,
                           max_features=16000, sublinear_tf=True).fit_transform(documents)


def reduced_text(matrix, dimensions):
    transform = TruncatedSVD(n_components=dimensions, random_state=SEED)
    vector = normalize(transform.fit_transform(matrix))
    return vector, float(transform.explained_variance_ratio_.sum())


def fuse(text, behavior, weight=.5):
    return np.c_[text * np.sqrt(weight), behavior * np.sqrt(1 - weight)]


def describe_clustering(posts, topics, membership, output):
    with np.load(ORIGINAL_RESULTS / "topic_representations.npz") as data:
        profiles, reference, text16, behavior = (data[name] for name in ("profiles", "fused", "text", "behavior"))
    documents = np.asarray([
        " ".join(re.sub(r"#[^#\n]{2,70}#", " ", body) for body in posts.content.iloc[membership[topic]])
        for topic in topics.topic
    ], dtype=object)
    matrix = text_matrix(documents)
    stability, sensitivity = [], []
    for dimension in (16, 32, 64):
        print(f"Text dimensions: {dimension}", flush=True)
        text, explained = reduced_text(matrix, dimension)
        if dimension == 16:
            assert np.allclose(text, text16, atol=1e-9, rtol=0)
        for weight in (.25, .5, .75):
            representation = fuse(text, behavior, weight)
            group = cluster_labels(representation, "KMeans", 7)
            sensitivity.append({"dimensions": dimension, "text_weight": weight, "tfidf_variance_share": explained,
                                "common_silhouette": silhouette_score(reference, group),
                                "ari_vs_original": adjusted_rand_score(topics.cluster, group)})
    grid = pd.DataFrame(sensitivity)
    grid.to_csv(output.results / "cluster_sensitivity.csv", index=False)
    rng = np.random.default_rng(SEED)
    for index in range(6):
        print(f"End-to-end clustering sample {index + 1}/6", flush=True)
        pick = np.sort(rng.choice(len(topics), int(.8*len(topics)), replace=False))
        local_text, _ = reduced_text(text_matrix(documents[pick]), 16)
        local_behavior = behavior_vector(topics.iloc[pick], profiles[pick], COORD + OTHER_BEHAVIOR)
        initial = topics.cluster.iloc[pick]
        fixed = cluster_labels(reference[pick], "KMeans", 7, SEED + index)
        refitted = cluster_labels(fuse(local_text, local_behavior), "KMeans", 7, SEED + index)
        stability.append({"repeat": index + 1, "topics": len(pick),
                          "ari_fixed_representation": adjusted_rand_score(initial, fixed),
                          "ari_refitted_representation": adjusted_rand_score(initial, refitted)})
    resamples = pd.DataFrame(stability)
    resamples.to_csv(output.results / "cluster_end_to_end.csv", index=False)
    return resamples, grid, profiles, reference, text16


def compare_hmi(topics, profiles, reference, text, output):
    reduced = fuse(text, behavior_vector(topics, profiles, OTHER_BEHAVIOR))
    packed = topics[[component + "_rank" for component in COORD]]
    settings = [
        ("no_coordination", None),
        ("four_raw_components", topics[COORD]),
        ("hmi_only", topics[["hmi"]]),
    ]
    for component in COORD:
        score = packed.drop(columns=component + "_rank").mean(axis=1)
        settings.append((f"hmi_without_{component}", score.to_frame("hmi")))
    rows = []
    for label, extra in settings:
        data = reduced
        if extra is not None:
            data = np.c_[reduced, .5 * StandardScaler().fit_transform(extra) / np.sqrt(extra.shape[1])]
        assignment = cluster_labels(data, "KMeans", 7)
        rows.append({"variant": label, "common_silhouette": silhouette_score(reference, assignment),
                     "ari_vs_original": adjusted_rand_score(topics.cluster, assignment)})
    result = pd.DataFrame(rows)
    result.to_csv(output.results / "hmi_components.csv", index=False)
    return result


def prepare_blinded_pairs(topics, vectors, output):
    """Prepare title-only prompts and a separate grouping key for human or AI review."""
    rng = np.random.default_rng(SEED + 1)
    groups = topics.cluster.to_numpy()
    labels = topics.topic.to_numpy()
    selected = []
    anchors = rng.choice(len(topics), 12, replace=False)
    for index in anchors:
        similarities = vectors @ vectors[index]
        same = np.flatnonzero((groups == groups[index]) & (np.arange(len(groups)) != index))
        near = same[np.argsort(similarities[same])[-12:]]
        inside = int(rng.choice(near))
        different = np.flatnonzero(groups != groups[index])
        outside = different[np.argmin(np.abs(similarities[different] - similarities[inside]))]
        selected.extend([
            (labels[index], labels[inside], True, float(similarities[inside])),
            (labels[index], labels[outside], False, float(similarities[outside])),
        ])
    prompts, key = [], []
    for number, position in enumerate(rng.permutation(len(selected)), 1):
        first, second, same_group, similarity = selected[position]
        if rng.integers(0, 2):
            first, second = second, first
        pair = f"P{number:02}"
        prompts.append({"pair": pair, "first_title": first, "second_title": second})
        key.append({"pair": pair, "same_cluster": same_group, "text_cosine": similarity})
    pd.DataFrame(prompts).to_csv(output.results / "cluster_review_prompts_local.csv", index=False)
    pd.DataFrame(key).to_csv(output.results / "cluster_review_key_local.csv", index=False)


def compare_itemsets(posts, topics, output):
    tags = [frozenset(row) for row in posts.tags]
    eligible = set(topics.topic)
    transactions = transaction_frame([sorted(row & eligible) for row in tags])
    benchmark = []
    for threshold in (.003, .005, .01):
        truth = None
        methods = [("Apriori", apriori, {"low_memory": True}), ("FP-growth", fpgrowth, {})]
        for repeat in range(5):
            ordered = methods if repeat % 2 == 0 else reversed(methods)
            for name, method, options in ordered:
                start = time.perf_counter()
                items = method(transactions, min_support=threshold, use_colnames=True, max_len=3, **options)
                elapsed = time.perf_counter() - start
                found = set(items.itemsets)
                truth = found if truth is None else truth
                assert found == truth
                benchmark.append({"method": name, "support_threshold": threshold, "repeat": repeat + 1,
                                  "itemsets": len(items), "seconds": elapsed})
    measurements = pd.DataFrame(benchmark)
    measurements.to_csv(output.results / "itemset_repeated_runtime.csv", index=False)
    rules = read_original("association_rules.csv")
    unique_body = ~posts.norm.duplicated().to_numpy()
    user_ids, unique_users = pd.factorize(posts.user_id)
    weeks = posts.time.dt.to_period("W-SUN")
    weekly = [(str(week.start_time.date()), weeks.eq(week).to_numpy()) for week in weeks.unique()]
    match_cache = {}
    def match(labels):
        if labels not in match_cache:
            match_cache[labels] = np.asarray([labels <= values for values in tags])
        return match_cache[labels]

    def evaluate(label, scope, total, lhs, rhs, both):
        support = both / total
        confidence = both / lhs if lhs else 0.0
        lift = confidence * total / rhs if rhs else 0.0
        return {"rule": label, "scope": scope, "observations": total, "matches": both,
                "support": support, "confidence": confidence, "lift": lift,
                "kept": support >= .003 and confidence >= .5 and lift >= 1.5}

    rows = []
    for index, rule in enumerate(rules.itertuples(), 1):
        label = f"rule_{index:02}"
        a, b = map(lambda value: frozenset(value.split(" + ")), (rule.antecedent, rule.consequent))
        lhs, rhs, both = match(a), match(b), match(a | b)
        rows.append(evaluate(label, "all_posts", len(posts), lhs.sum(), rhs.sum(), both.sum()))
        for scope, keep in [("deduplicated_text", unique_body), *weekly]:
            rows.append(evaluate(label, scope, keep.sum(), (lhs & keep).sum(), (rhs & keep).sum(), (both & keep).sum()))
        counts = [len(np.unique(user_ids[flags])) for flags in (lhs, rhs, both)]
        rows.append(evaluate(label, "distinct_users_same_post", len(unique_users), *counts))
    rules_result = pd.DataFrame(rows)
    assert rules_result.loc[rules_result.scope.eq("all_posts"), "kept"].all()
    rules_result.to_csv(output.results / "association_rule_sensitivity.csv", index=False)
    return measurements, rules_result


def temporal_checks(posts, topics, output):
    bins = defaultdict(lambda: [0, 0, 0, 0])
    allowed = set(topics.topic)
    for labels, day, hour in zip(posts.tags, posts.time.dt.strftime("%Y-%m-%d"), posts.time.dt.hour):
        for tag in labels:
            if tag in allowed:
                bins[(tag, day)][hour // 6] += 1
    rows = [(tag, day, *counts) for (tag, day), counts in bins.items()]
    slot_columns = [f"slot_{index + 1}" for index in range(4)]
    timing = pd.DataFrame(rows, columns=["topic", "date", *slot_columns])
    daily = read_original("topic_day_growth.csv")
    timing = daily.merge(timing, on=["topic", "date"], validate="one_to_one")
    assert timing[slot_columns].sum(axis=1).equals(timing.posts)
    timing = timing[timing.history_eligible].copy()
    timing["pattern"] = ["".join("H" if value > np.median(row) else "L" for value in row)
                         for row in timing[slot_columns].to_numpy()]
    timing[slot_columns] = timing[slot_columns].div(timing.posts, axis=0)
    q1, q3 = topics.hmi.quantile([.25, .75])
    scores = topics.set_index("topic").hmi
    timing["hmi"] = timing.topic.map(scores)
    metrics = [*slot_columns, "LLHH", "LHHL"]
    timing["LLHH"] = timing.pattern.eq("LLHH").astype(float)
    timing["LHHL"] = timing.pattern.eq("LHHL").astype(float)
    selected, paired = [], []
    for threshold in (4, 6, 8):
        part = timing[timing.posts.ge(threshold)]
        for name, group in [("low_hmi", part[part.hmi.le(q1)]), ("high_hmi", part[part.hmi.ge(q3)])]:
            values = group.groupby("topic")[metrics].mean().mean()
            selected.append({"minimum_posts": threshold, "group": name, "topics": group.topic.nunique(),
                             "topic_days": len(group), **values.to_dict()})
        hot = part[part.hotspot].groupby("topic")[metrics].mean()
        usual = part[~part.hotspot].groupby("topic")[metrics].mean()
        ids = hot.index.intersection(usual.index)
        diffs = hot.loc[ids] - usual.loc[ids]
        for metric in metrics:
            values = diffs[metric].to_numpy()
            rng = np.random.default_rng(SEED)
            boot = rng.choice(values, size=(400, len(values)), replace=True).mean(axis=1)
            lwr, upr = np.quantile(boot, [.025, .975])
            paired.append({"minimum_posts": threshold, "metric": metric, "matched_topics": len(ids),
                           "hot_minus_other": values.mean(), "lower_95": lwr, "upper_95": upr})
    levels, within = pd.DataFrame(selected), pd.DataFrame(paired)
    levels.to_csv(output.results / "motif_threshold_sensitivity.csv", index=False)
    within.to_csv(output.results / "matched_topic_motifs.csv", index=False)
    return levels, within


def matched_cases(posts, topics, membership, output):
    daily = read_original("topic_day_growth.csv")
    start, end = posts.time.min().normalize(), posts.time.max().normalize()
    interior = pd.to_datetime(daily.date).between(start + pd.Timedelta(days=7), end - pd.Timedelta(days=7))
    events = daily[daily.hotspot & interior].sort_values(["posts", "date"], ascending=[False, True]).drop_duplicates("topic")
    available = topics.merge(events[["topic", "date"]], on="topic")
    low, high = topics.hmi.quantile([.25, .75])
    lows = available[available.hmi.le(low)]
    highs = available[available.hmi.ge(high)]
    matches = []
    with np.load(ORIGINAL_RESULTS / "topic_representations.npz") as data:
        text = {topic: vector for topic, vector in zip(topics.topic, data["text"])}
    for group in sorted(set(lows.cluster) & set(highs.cluster)):
        for a, b in product(lows[lows.cluster.eq(group)].itertuples(), highs[highs.cluster.eq(group)].itertuples()):
            difference = abs(np.log(a.posts/b.posts)) + abs(np.log(a.days/b.days))
            distance = 1 - np.dot(text[a.topic], text[b.topic])
            matches.append((difference + .35 * distance, a.topic, b.topic, a, b))
    chosen, used = [], set()
    for _, left, right, a, b in sorted(matches):
        if left in used or right in used:
            continue
        chosen.append((a, b))
        used.update([left, right])
        if len(chosen) == 3:
            break
    assert len(chosen) == 3
    fig, axes = plt.subplots(2, 3, figsize=(10, 5), sharex=True, layout="constrained")
    rows = []
    for index, pair in enumerate(chosen, 1):
        axes[0, index-1].sharey(axes[1, index-1])
        for offset, (name, group) in enumerate(zip(("Low", "High"), pair)):
            ax = axes[offset, index-1]
            g = posts.iloc[membership[group.topic]]
            date = pd.Timestamp(group.date)
            axis = pd.date_range(date - pd.Timedelta(days=7), date + pd.Timedelta(days=7))
            counts = g.time.dt.normalize().value_counts().reindex(axis, fill_value=0)
            day = g[g.time.dt.normalize().eq(date)]
            raw_repost = day.content.str.contains(r"//@|转发微博", regex=True).mean()
            rows.append({"pair": index, "group": name, "topic": group.topic, "date": group.date,
                         "cluster": group.cluster, "hmi": group.hmi, "all_posts": group.posts, "active_days": group.days,
                         "event_posts": len(day), "event_users": day.user_id.nunique(),
                         "event_duplicate_share": 1 - day.norm.nunique()/len(day),
                         "event_repost_marker_share": raw_repost, "largest_daily_share": group.peak_day})
            ax.bar(range(-7, 8), counts, color=FIGURE_BLUE if offset == 0 else FIGURE_GOLD)
            ax.axvline(0, color="#586C78", lw=.8)
            ax.set_title(f"Pair {index} · {name} HMI", fontsize=10)
            if index == 1:
                ax.set_ylabel("Posts")
            if offset == 1:
                ax.set_xlabel("Days from hotspot")
    output.save_figure(fig, "18_matched_cases")
    pd.DataFrame(rows).to_csv(output.results / "matched_cases_local.csv", index=False)


def root_reference_check(posts, output):
    references = posts.loc[posts.origin_weibo.ne("")].copy()
    references["root_found"] = references.origin_weibo.isin(posts.mid)
    references["month"] = references.time.dt.strftime("%Y-%m")
    references["tagged"] = references.tags.map(bool)
    references["duplicate"] = references.norm.duplicated(keep=False)
    references["reposts"] = pd.to_numeric(references.repost_num, errors="coerce")
    summary = references.groupby(["month", "root_found"]).agg(
        posts=("mid", "size"), unique_authors=("user_id", "nunique"), tagged_share=("tagged", "mean"),
        repeated_text_share=("duplicate", "mean"), observed_repost_count_median=("reposts", "median"),
        observed_repost_count_records=("reposts", "count"),
    ).reset_index()
    assert summary.posts.sum() == len(references) and summary.posts[summary.root_found].sum() == int(references.root_found.sum())
    summary.to_csv(output.results / "reference_match_comparison.csv", index=False)


def plot_extensions(resamples, hmi, paired, output):
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for row in resamples.itertuples():
        a.plot([0, 1], [row.ari_fixed_representation, row.ari_refitted_representation], marker="o",
               color=FIGURE_BLUE, alpha=.6)
    a.set_xticks([0, 1], ["Fixed features", "Refit features"])
    a.set(xlim=(-.2, 1.2), ylim=(0, 1), ylabel="Adjusted Rand index", title="(a) Six paired resamples")
    order = hmi.set_index("variant").loc[["no_coordination", "four_raw_components", "hmi_only"]]
    b.barh(["No coordination", "Four components", "HMI only"], order.common_silhouette,
           color=["#A9B9C4", FIGURE_BLUE, FIGURE_GOLD])
    b.invert_yaxis()
    b.set(xlabel="Silhouette in reference space", title="(b) Coordination input")
    for label, mark, color in [("LLHH", "o", FIGURE_BLUE), ("LHHL", "s", FIGURE_GOLD)]:
        part = paired[paired.metric.eq(label)]
        c.errorbar(part.minimum_posts, part.hot_minus_other,
                   yerr=[part.hot_minus_other-part.lower_95, part.upper_95-part.hot_minus_other],
                   marker=mark, capsize=3, color=color, label=label)
    c.axhline(0, color="#8192A0", lw=1)
    c.set_xticks([4, 6, 8])
    c.set(xlabel="Minimum daily posts", ylabel="Hotspot minus other active days",
          title="(c) Paired by topic")
    c.legend(frameon=False)
    output.save_figure(fig, "16_mining_extensions")


def main():
    output = ExperimentOutput(EXPERIMENT_DIR / "extensions" / "descriptive")
    posts = load_posts()
    topics = read_original("topic_features.csv")
    membership = {name: [] for name in topics.topic}
    for index, tags in enumerate(posts.tags):
        for tag in tags:
            if tag in membership:
                membership[tag].append(index)
    # Run smaller checks first so their results are available during the clustering refits.
    print("Checking matched examples and observed root references", flush=True)
    matched_cases(posts, topics, membership, output)
    root_reference_check(posts, output)
    print("Checking co-occurrence and time patterns", flush=True)
    compare_itemsets(posts, topics, output)
    _, paired = temporal_checks(posts, topics, output)
    resamples, _, profiles, reference, text = describe_clustering(posts, topics, membership, output)
    prepare_blinded_pairs(topics, text, output)
    hmi = compare_hmi(topics, profiles, reference, text, output)
    plot_extensions(resamples, hmi, paired, output)


if __name__ == "__main__":
    main()

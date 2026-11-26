"""Compare activity, HMI and seven past-only text/behavior clusters on validation dates."""
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler, normalize

from experiment_io import EXPERIMENT_DIR, ExperimentOutput, load_posts
from forecast_experiment import ACTIVITY, COORD, forecast_dates, make_rows, outcome_count, ranking_metrics
from forecast_extensions import forest_scores
from run_analysis import SEED, plt


GROUPS = 7
WINDOW_DAYS = 28
TEXT_DIMENSIONS = 16
BEHAVIOR_DIMENSIONS = 41
LIFE_BOUNDARIES = np.linspace(0, 1, 11)


def rolling_post_indices(frame, posts):
    """Return the source posts in each topic's window, with the same order as frame."""
    day = posts.time.to_numpy(dtype="datetime64[D]")
    topics = set(frame.topic)
    membership = defaultdict(list)
    for position, tags in enumerate(posts.tags):
        for tag in tags:
            if tag in topics:
                membership[tag].append(position)

    as_of = frame.as_of.to_numpy(dtype="datetime64[D]")
    start = as_of - np.timedelta64(WINDOW_DAYS - 1, "D")
    windows = [None] * len(frame)
    for tag, rows in frame.groupby("topic", sort=False).indices.items():
        positions = np.asarray(membership[tag], dtype=np.intp)
        lo = np.searchsorted(day[positions], start[rows])
        hi = np.searchsorted(day[positions], as_of[rows], side="right")
        for row, left, right in zip(rows, lo, hi):
            windows[row] = positions[left:right]
    expected_counts = np.rint(np.expm1(frame.log_posts_28)).astype(int)
    np.testing.assert_array_equal(np.fromiter(map(len, windows), dtype=int), expected_counts)
    window_first = np.fromiter((positions[0] for positions in windows), dtype=int)
    window_last = np.fromiter((positions[-1] for positions in windows), dtype=int)
    assert (day[window_first] >= start).all() and (day[window_last] <= as_of).all()
    return windows


def behavior_features(frame, posts, windows):
    hours = posts.time.dt.hour.to_numpy()
    time_ns = posts.time.astype("int64").to_numpy()
    repost = posts.content.str.contains(r"//@|转发微博", regex=True).to_numpy()
    shares = np.empty((len(frame), 37))
    for row, positions in enumerate(windows):
        hourly = np.bincount(hours[positions], minlength=24) / len(positions)
        observed = hourly[hourly > 0]
        entropy = -np.sum(observed * np.log(observed)) / np.log(24)
        elapsed = (time_ns[positions] - time_ns[positions[0]]) / 1e9
        life = np.histogram(elapsed / max(elapsed[-1], 1), bins=LIFE_BOUNDARIES)[0] / len(positions)
        shares[row] = np.r_[hourly[:6].sum(), repost[positions].mean(), entropy, hourly, life]
    # The forecast already computes these four measures over the same 28-day windows.
    features = np.c_[frame[COORD[:4]], shares]
    assert features.shape[1] == BEHAVIOR_DIMENSIONS
    return features


def unique_documents(bodies, windows):
    """Tokenize identical windows once while tracking how many training rows they represent."""
    keys = {}
    unique = []
    inverse = []
    for window in windows:
        key = window.tobytes()
        if key not in keys:
            keys[key] = len(unique)
            unique.append(window)
        inverse.append(keys[key])
    return (" ".join(bodies[window]) for window in unique), np.asarray(inverse), len(unique)


def rolling_tfidf(bodies, train_windows, validation_windows):
    # Match the descriptive char ranges, frequency bounds, feature cap and sublinear TF.
    train_documents, reverse, unique_count = unique_documents(bodies, train_windows)
    print(f"Counting character fragments in {unique_count:,} distinct training windows", flush=True)
    counter = CountVectorizer(analyzer="char", ngram_range=(2, 3), dtype=np.int32)
    counts = counter.fit_transform(train_documents)
    weights = np.bincount(reverse)
    # Multiplicity preserves the document and token counts of the full topic-day training set.
    document_counts = counts.sign().T @ weights
    token_counts = counts.T @ weights
    retained = np.flatnonzero((document_counts >= 3) & (document_counts <= .9 * len(train_windows)))
    if len(retained) > 16000:
        retained = np.sort(retained[np.argsort(-token_counts[retained])[:16000]])

    vocabulary = {word: i for i, word in enumerate(counter.get_feature_names_out()[retained])}
    train_counts = counts[:, retained][reverse]
    del counts, counter
    tfidf = TfidfTransformer(sublinear_tf=True)
    train_text = tfidf.fit_transform(train_counts)
    del train_counts
    validation_docs, validation_reverse, _ = unique_documents(bodies, validation_windows)
    validation_counter = CountVectorizer(analyzer="char", ngram_range=(2, 3), vocabulary=vocabulary, dtype=np.int32)
    validation_text = tfidf.transform(validation_counter.transform(validation_docs))[validation_reverse]
    return train_text, validation_text


def cluster_assignments(frame, posts, train, validation, windows):
    bodies = posts.content.str.replace(r"#[^#\n]{2,70}#", " ", regex=True).to_numpy()
    train_windows = [windows[i] for i in np.flatnonzero(train)]
    validation_windows = [windows[i] for i in np.flatnonzero(validation)]
    train_text, validation_text = rolling_tfidf(bodies, train_windows, validation_windows)
    vocabulary_size = train_text.shape[1]
    svd = TruncatedSVD(n_components=TEXT_DIMENSIONS, random_state=SEED)
    print("Fitting the 16-dimensional text representation and the 41-dimensional behavior representation", flush=True)
    text_train = normalize(svd.fit_transform(train_text))
    text_validation = normalize(svd.transform(validation_text))
    del train_text, validation_text
    behavior = behavior_features(frame, posts, windows)
    scaler = StandardScaler().fit(behavior[train])
    behavior_train = normalize(scaler.transform(behavior[train]))
    behavior_validation = normalize(scaler.transform(behavior[validation]))
    fused_train = np.c_[text_train, behavior_train] / np.sqrt(2)
    fused_validation = np.c_[text_validation, behavior_validation] / np.sqrt(2)
    assert fused_train.shape[1] == fused_validation.shape[1] == 57
    model = KMeans(GROUPS, n_init=20, random_state=SEED).fit(fused_train)
    assignments = np.empty(len(frame), dtype=int)
    assignments[train] = model.labels_
    assignments[validation] = model.predict(fused_validation)
    assert len(np.unique(assignments[train])) == GROUPS
    return assignments, vocabulary_size, float(svd.explained_variance_ratio_.sum())


def compare_forests(frame, train, validation, cluster_columns, total_hotspots):
    variants = [
        (1, "Activity", ACTIVITY),
        (4, "Activity + HMI", ACTIVITY + COORD),
        (5, "Activity + seven clusters", ACTIVITY + cluster_columns),
        (7, "Activity + HMI + seven clusters", ACTIVITY + COORD + cluster_columns),
    ]
    scored = frame.loc[validation, ["topic", "as_of", "target"]].copy()
    rows = []
    for group, name, columns in variants:
        print(f"Training group {group}: {name}", flush=True)
        score = forest_scores(frame, columns, train, validation)
        scored[str(group)] = score
        rows.append({"group": group, "inputs": name, "input_columns": len(columns),
                     **ranking_metrics(scored, score, total_hotspots)})
    metrics = pd.DataFrame(rows)
    metrics["ap_gain_over_activity"] = metrics.average_precision - metrics.average_precision.iloc[0]
    date = pd.to_datetime(scored.as_of)
    blocks = (date - date.min()).dt.days // 7
    weekly = []
    for _, part in scored.assign(date=date).groupby(blocks):
        for group, _, _ in variants:
            top = part.sort_values(["as_of", str(group), "topic"], ascending=[True, False, True]).groupby("as_of").head(5)
            weekly.append({"start": part.date.min().date().isoformat(), "end": part.date.max().date().isoformat(),
                           "group": group, "candidate_positives": int(part.target.sum()),
                           "average_precision": average_precision_score(part.target, part[str(group)]),
                           "top5_hits": int(top.target.sum())})
    return metrics, pd.DataFrame(weekly)


def plot(metrics, output):
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 3.6), layout="constrained")
    labels = ["1  Activity", "4  Activity + HMI", "5  Activity + clusters", "7  Activity + HMI + clusters"]
    colors = ["#90A9BB", "#327EA3", "#478A74", "#BD642C"]
    for ax, column, title, xlabel, formatting in [
        (axes[0], "average_precision", "(a) Overall ranking", "Average precision (AP)", "%.4f"),
        (axes[1], "reviewed_positives", "(b) Daily top-five results", "Total hits across validation dates", "%.0f"),
    ]:
        values = metrics[column]
        bars = ax.barh(range(len(labels)), values, color=colors, height=.62)
        ax.bar_label(bars, fmt=formatting, padding=4)
        ax.set_yticks(range(len(labels)), labels if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.set(xlabel=xlabel, title=title, xlim=(0, values.max() * 1.22))
    output.save_figure(fig, "23_forecast_cluster7_ablation")


def main():
    config = forecast_dates()
    cutoff = pd.Timestamp(config["validation_end"]) + pd.Timedelta(days=1)
    print("Rebuilding the original candidate rows up to the final validation outcome", flush=True)
    frame, _, outcomes = make_rows(config, cutoff=cutoff)
    frame = frame[frame.split.isin(["train", "validation"])].reset_index(drop=True)
    train = frame.split.eq("train")
    validation = frame.split.eq("validation")
    assert (pd.to_datetime(frame.target_date) - pd.to_datetime(frame.as_of)).eq(pd.Timedelta(days=1)).all()
    assert frame.loc[train, "target_date"].max() < frame.loc[validation, "as_of"].min()
    assert frame.as_of.max() == config["validation_end"]
    assert not frame.duplicated(["topic", "as_of"]).any()
    posts = load_posts(cutoff=config["validation_end"]).sort_values("time", kind="stable").reset_index(drop=True)
    windows = rolling_post_indices(frame, posts)

    # Verify that rebuilding from a training-only prefix leaves every training window unchanged.
    prefix = posts[posts.time.dt.floor("D").le(pd.Timestamp(config["train_end"]))]
    truncated = rolling_post_indices(frame.loc[train].reset_index(drop=True), prefix)
    assert all(np.array_equal(a, b) for a, b in zip(truncated, (windows[i] for i in np.flatnonzero(train)), strict=True))
    assignments, vocabulary_size, explained_variance = cluster_assignments(frame, posts, train, validation, windows)
    clusters = [f"fused_cluster_{i + 1}" for i in range(GROUPS)]
    frame[clusters] = np.equal.outer(assignments, np.arange(GROUPS))
    assert frame[clusters].sum(axis=1).eq(1).all()

    total = outcome_count(outcomes, config["validation_start"], config["validation_end"])
    metrics, weekly = compare_forests(frame, train, validation, clusters, total)
    output = ExperimentOutput(EXPERIMENT_DIR / "forecast-fused-cluster7")
    metrics.to_csv(output.results / "forecast_cluster7_validation_metrics.csv", index=False, lineterminator="\n")
    weekly.to_csv(output.results / "forecast_cluster7_validation_weeks.csv", index=False, lineterminator="\n")
    output.write_json("validation.json", {
        "date_config": config, "split_reported": "validation", "training_rows": int(train.sum()),
        "validation_rows": int(validation.sum()), "validation_positives": int(frame.loc[validation, "target"].sum()),
        "all_validation_hotspots": total, "window_days": WINDOW_DAYS, "vocabulary_size": vocabulary_size,
        "text_variance_explained": explained_variance, "fused_dimensions": 57,
        "training_cluster_sizes": np.bincount(assignments[train], minlength=GROUPS).tolist(),
        "validation_cluster_sizes": np.bincount(assignments[validation], minlength=GROUPS).tolist(),
        "checks": ["Original candidate windows reconcile to the raw post counts", "All windows end by the as-of date",
                   "Training windows are identical after removing later posts", "Groups are encoded by exactly one active column"],
    })
    plot(metrics, output)
    print(metrics[["group", "average_precision", "reviewed_positives", "ap_gain_over_activity"]].to_string(index=False))
    print(weekly.to_string(index=False))


if __name__ == "__main__":
    main()

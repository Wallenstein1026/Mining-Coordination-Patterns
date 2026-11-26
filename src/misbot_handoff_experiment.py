"""Verify the private MisBot score handoff and run the descriptive ablations."""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from experiment_io import EXPERIMENT_DIR, ExperimentOutput, load_posts
from mining_experiment import cluster_labels, subsample_stability
from run_analysis import ROOT, plt


DEFAULT_ARCHIVE = ROOT / "misbot_weibo_2020-09_11_570a297f_private.tar.gz"
BASELINE = EXPERIMENT_DIR / "descriptive" / "results"
POST_TABLE = "post_style_scores.csv"
TOPIC_TABLE = "topic_style_scores.csv"
STYLE = "user_balanced_style"
SCORE_TOLERANCE = 1e-8


def json_file(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_archive(path, check):
    expected = {"receipt.json", POST_TABLE, TOPIC_TABLE}
    with tarfile.open(path, "r:gz") as archive:
        entries = archive.getmembers()
        check("Archive contains exactly the expected three regular files",
              len(entries) == len(expected) and {entry.name for entry in entries} == expected
              and all(entry.isfile() for entry in entries))
        with archive.extractfile("receipt.json") as stream:
            receipt = json.load(stream)
        frames = []
        for name in (POST_TABLE, TOPIC_TABLE):
            with archive.extractfile(name) as stream:
                payload = stream.read()
            check(f"SHA-256 matches the handoff receipt: {name}",
                  hashlib.sha256(payload).hexdigest() == receipt["output_sha256"][name])
            frames.append(pd.read_csv(BytesIO(payload), dtype={"mid": str, "topic": str},
                                      keep_default_na=False, float_precision="round_trip"))
    return receipt, *frames


def audit_scores(post_scores, topic_scores, receipt, check):
    data_manifest = json_file(ROOT / "data" / "weibo_manifest.json")
    misbot_manifest = json_file(ROOT / "data" / "misbot_manifest.json")
    design = json_file(BASELINE / "topic_design.json")
    check("Weibo SHA-256 matches the local manifest; the loader verifies the actual file hash",
          receipt["weibo_normalized_sha256"] == data_manifest["normalized_sha256"])
    posts = load_posts()
    check("Receipt matches local Weibo dates, post count and author count",
          len(posts) == receipt["scored_posts"] and posts.user_id.nunique() == receipt["weibo_users"]
          and posts.time.min() == pd.Timestamp(receipt["weibo_coverage_start"])
          and posts.time.max() == pd.Timestamp(receipt["weibo_coverage_end"]))
    check("Score tables have the expected schemas",
          list(post_scores.columns) == ["mid", "bot_style"]
          and list(topic_scores.columns) == ["topic", "posts", "users", "bot_style", STYLE])
    check("Post scores are finite and lie in the unit interval", post_scores.bot_style.between(0, 1).all())
    check("Topic scores are finite and lie in the unit interval",
          topic_scores.bot_style.between(0, 1).all() and topic_scores[STYLE].between(0, 1).all())
    check("Post IDs match one-to-one with full coverage",
          post_scores.mid.is_unique and len(post_scores) == len(posts) and set(post_scores.mid) == set(posts.mid))
    check("Receipt thresholds match the original descriptive experiment",
          receipt["topic_min_posts"] == design["min_posts"] and receipt["topic_min_users"] == design["min_users"])

    # Recompute topic-level scores independently from each post score and the audited Weibo sample.
    joined = posts[["mid", "user_id", "tags"]].merge(post_scores, on="mid", validate="one_to_one")
    membership = joined.explode("tags").dropna(subset=["tags"]).rename(columns={"tags": "topic"})
    membership = membership[~membership.topic.isin(design["excluded_broad_tags"])]
    counts = membership.groupby("topic").agg(
        posts=("mid", "size"), users=("user_id", "nunique"), bot_style=("bot_style", "mean"))
    counts = counts[counts.posts.ge(design["min_posts"]) & counts.users.ge(design["min_users"])]
    authors = membership[membership.topic.isin(counts.index)].groupby(["topic", "user_id"]).bot_style.mean()
    counts[STYLE] = authors.groupby(level="topic").mean()
    supplied = topic_scores.set_index("topic").sort_index()
    check("Topic coverage agrees with the receipt and all independently eligible topics",
          supplied.index.is_unique and len(counts) == len(supplied) == receipt["eligible_topics"]
          and counts.index.equals(supplied.index))
    check("All per-topic post and unique-author counts agree",
          counts[["posts", "users"]].equals(supplied[["posts", "users"]]))
    errors = {}
    for name in ("bot_style", STYLE):
        errors[name] = float((counts[name] - supplied[name]).abs().max())
        check(f"All topic aggregates independently recompute: {name}", errors[name] <= SCORE_TOLERANCE)
    original = pd.read_csv(BASELINE / "topic_features.csv", dtype={"topic": str},
                           keep_default_na=False, float_precision="round_trip")
    check("The original clustering contains the same topics and counts",
          original.topic.is_unique and original.set_index("topic")[["posts", "users"]].sort_index()
          .equals(counts[["posts", "users"]]))
    topics = original.merge(counts[["bot_style", STYLE]], on="topic", validate="one_to_one", sort=False)
    check("The original feature-matrix topic order is preserved", original.topic.equals(topics.topic))
    identical_input = receipt["misbot_input_sha256"] == misbot_manifest.get("sha256")
    metadata = {
        "scored_posts": len(posts), "weibo_users": int(posts.user_id.nunique()), "eligible_topics": len(counts),
        "weibo_normalized_sha256": receipt["weibo_normalized_sha256"],
        "misbot_input_sha256": receipt["misbot_input_sha256"],
        "misbot_declared_model_sha256": receipt["model_sha256"],
        "handoff_release_kind": receipt["misbot_release_kind"],
        "repository_release_kind": misbot_manifest["release_kind"],
        "misbot_input_hash_matches_repository_manifest": identical_input,
        "misbot_release_kind_conflict": identical_input and receipt["misbot_release_kind"] != misbot_manifest["release_kind"],
        "max_topic_aggregate_difference": errors, "topic_aggregate_tolerance": SCORE_TOLERANCE,
    }
    return topics, metadata


def style_ablation(topics, check):
    selected = json_file(BASELINE / "selected_clustering.json")
    method, k = selected["method"], selected["k"]
    with np.load(BASELINE / "topic_representations.npz") as matrices:
        fused = matrices["fused"]
    baseline = topics.cluster.to_numpy()
    check("The saved fused feature matrix has the verified number of topics", len(fused) == len(topics))
    check("The saved baseline grouping independently refits",
          adjusted_rand_score(baseline, cluster_labels(fused, method, k)) == 1)
    designs = [
        ("Fused baseline", [], 0.0),
        ("Fused + HMI", ["hmi"], .5),
        ("Fused + user-balanced style", [STYLE], .5),
        ("CIB-informed w=0.25", ["hmi", STYLE], .25),
        ("CIB-informed w=0.50", ["hmi", STYLE], .5),
        ("CIB-informed w=1.00", ["hmi", STYLE], 1.0),
        ("Fused + post-weighted style", ["bot_style"], .5),
    ]
    rows = []
    for name, inputs, weight in designs:
        coordinates = fused
        if inputs:
            signals = StandardScaler().fit_transform(topics[inputs])
            coordinates = np.c_[fused, weight * signals / np.sqrt(len(inputs))]
        labels = cluster_labels(coordinates, method, k) if inputs else baseline
        stable = subsample_stability(coordinates, labels, method, k)
        rows.append({"representation": name, "auxiliary_weight": weight, "method": method, "k": k,
                     "common_fused_silhouette": silhouette_score(fused, labels),
                     "native_silhouette": silhouette_score(coordinates, labels),
                     "ari_vs_baseline": adjusted_rand_score(baseline, labels),
                     "subsample_ari_mean": np.mean(stable), "subsample_ari_std": np.std(stable),
                     "minimum_cluster_size": int(np.bincount(labels).min())})
        if name == "CIB-informed w=0.50":
            topics["joint_cluster"] = labels
    ablation = pd.DataFrame(rows)
    earlier = pd.read_csv(BASELINE / "cib_informed_ablation.csv").set_index("representation")
    metrics = ["common_fused_silhouette", "ari_vs_baseline", "subsample_ari_mean"]
    check("The unchanged fused and HMI-only controls reproduce the previous metrics",
          np.allclose(ablation.set_index("representation").loc[earlier.index, metrics],
                      earlier[metrics], atol=1e-10, rtol=0))
    profiles = topics.groupby("joint_cluster").agg(
        n=("topic", "size"), average_hmi=("hmi", "mean"),
        post_weighted_style=("bot_style", "mean"), user_balanced_style=(STYLE, "mean"),
    ).reset_index()
    check("Joint cluster profiles account for all verified topics", profiles.n.sum() == len(topics))
    return ablation, profiles


def make_figures(ablation, topics, output):
    descriptions = ["Fused", "Add HMI", "Add user-balanced style",
                    "Add both (0.25)", "Add both (0.50)", "Add both (1.00)", "Add post-weighted style"]
    positions = np.arange(len(descriptions))
    colors = ["#A3B7C5", "#256A8B", "#1B8278", "#D9C076", "#BC8A33", "#A76C17", "#83A995"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, layout="constrained")
    left.barh(positions, ablation.common_fused_silhouette, color=colors)
    left.set_yticks(positions, descriptions)
    left.invert_yaxis()
    left.set(xlim=(0, .18), xlabel="Silhouette in common fused space", title="(a) Separation")
    right.barh(positions, ablation.subsample_ari_mean, xerr=ablation.subsample_ari_std,
               color=colors, capsize=3, error_kw={"ecolor": "#24364D", "lw": 1})
    right.set(xlim=(0, 1.05), xlabel="Adjusted Rand index", title="(b) Stability (mean ± standard deviation)")
    output.save_figure(fig, "19_misbot_style_ablation")

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    left.scatter(topics.hmi, topics[STYLE], s=8, alpha=.34, c="#226B8A", edgecolors="none")
    left.set(xlabel="Full-period HMI", ylabel="User-balanced style score", title="(a) Each point is one topic")
    delta = topics[STYLE] - topics.bot_style
    right.hist(delta, bins=32, color="#1B8278", edgecolor="white", linewidth=.5)
    right.axvline(0, color="#536879", lw=1, ls="--")
    right.set(xlabel="User-balanced score minus post-weighted score", ylabel="Topics",
              title="(b) Effect of aggregating by user")
    output.save_figure(fig, "20_misbot_style_diagnostics")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", nargs="?", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    checks = []

    def check(name, condition):
        if not condition:
            raise ValueError(name)
        checks.append(name)

    output = ExperimentOutput(EXPERIMENT_DIR / "misbot-handoff")
    receipt, post_scores, topic_scores = read_archive(args.archive, check)
    topics, audit = audit_scores(post_scores, topic_scores, receipt, check)
    print(f"Verified {audit['scored_posts']:,} post scores and recomputed {audit['eligible_topics']:,} topic scores.",
          flush=True)
    ablation, profiles = style_ablation(topics, check)
    ablation.to_csv(output.results / "cib_informed_ablation.csv", index=False)
    profiles.to_csv(output.results / "joint_cluster_profiles.csv", index=False)
    delta = (topics[STYLE] - topics.bot_style).abs()
    audit.update({
        "passed_checks": checks, "style_mean": float(topics[STYLE].mean()),
        "hmi_style_spearman": float(topics.hmi.corr(topics[STYLE], method="spearman")),
        "topic_style_spearman_by_aggregation": float(topics.bot_style.corr(topics[STYLE], method="spearman")),
        "median_absolute_user_vs_post_weighted_difference": float(delta.median()),
        "max_absolute_user_vs_post_weighted_difference": float(delta.max()),
    })
    output.write_json("handoff_validation.json", audit)
    make_figures(ablation, topics, output)
    print(f"{len(checks)} checks passed. Results: {output.results}")


if __name__ == "__main__":
    main()

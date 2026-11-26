"""Recompute topic style scores after removing the most active authors."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_io import EXPERIMENT_DIR, ExperimentOutput, load_posts
from misbot_handoff_experiment import DEFAULT_ARCHIVE, SCORE_TOLERANCE, STYLE, audit_scores, read_archive
from run_analysis import plt


SMALL_GAP = 0.001
LARGE_GAP_QUANTILE = 0.9
AUTHOR_LIMITS = (1, 5)
COMPARE_EPSILON = 1e-12


def author_totals(post_scores, topics):
    posts = load_posts()[["mid", "user_id", "tags"]]
    membership = posts.merge(post_scores, on="mid", validate="one_to_one").explode("tags")
    membership = membership.loc[membership.tags.isin(topics.index)].rename(columns={"tags": "topic"})
    return membership.groupby(["topic", "user_id"], as_index=False).agg(
        posts=("mid", "size"), score_sum=("bot_style", "sum"))


def recompute_after_removal(topics, authors):
    scores = topics[["posts", "users", "bot_style", STYLE]].copy()
    scores["gap"] = (scores[STYLE] - scores.bot_style).abs()
    authors = authors.sort_values(["topic", "posts", "user_id"], ascending=[True, False, True])
    totals = authors.groupby("topic").agg(posts=("posts", "sum"), users=("user_id", "size"),
                                         score_sum=("score_sum", "sum"))
    assert totals[["posts", "users"]].equals(scores[["posts", "users"]].sort_index())
    assert np.allclose(totals.score_sum / totals.posts, scores.bot_style.sort_index(), atol=SCORE_TOLERANCE, rtol=0)

    for limit in AUTHOR_LIMITS:
        removed = authors.groupby("topic").head(limit).groupby("topic")[["posts", "score_sum"]].sum()
        remaining = totals.posts - removed.posts
        after = (totals.score_sum - removed.score_sum) / remaining.where(remaining.gt(0))
        scores[f"top{limit}_share"] = removed.posts / totals.posts
        scores[f"after_top{limit}"] = after
        scores[f"change_top{limit}"] = (after - scores.bot_style).abs()
        scores[f"gap_after_top{limit}"] = (after - scores[STYLE]).abs()
        assert after.isna().equals(remaining.eq(0))
        assert after.dropna().between(-COMPARE_EPSILON, 1 + COMPARE_EPSILON).all()
    return scores


def summarize(scores):
    cutoff = scores.gap.quantile(LARGE_GAP_QUANTILE)
    assert cutoff > SMALL_GAP, "Small- and large-gap groups must be disjoint."
    cohorts = {"small_gap": scores[scores.gap.lt(SMALL_GAP)], "large_gap": scores[scores.gap.ge(cutoff)]}
    rows = []
    for cohort, subset in cohorts.items():
        for limit in AUTHOR_LIMITS:
            available = subset[subset[f"after_top{limit}"].notna()]
            residual_gap = available[f"gap_after_top{limit}"]
            rows.append({
                "cohort": cohort, "top_authors_removed": limit, "topics": len(subset),
                "score_available": len(available), "no_posts_remaining": len(subset) - len(available),
                "median_post_share": subset[f"top{limit}_share"].median(),
                "median_absolute_score_change": available[f"change_top{limit}"].median(),
                "median_gap_before_removal": available.gap.median(),
                "median_gap_after_removal": residual_gap.median(),
                "topics_with_narrower_gap": int(residual_gap.lt(available.gap - COMPARE_EPSILON).sum()),
                "topics_with_gap_at_least_halved": int(
                    (available.gap.gt(COMPARE_EPSILON) & residual_gap.le(available.gap / 2 + COMPARE_EPSILON)).sum()),
            })
    return pd.DataFrame(rows), cutoff


def plot(summary, output):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), layout="constrained")
    positions = np.arange(len(AUTHOR_LIMITS))
    cohorts = [("small_gap", "Gap < 0.001", "#90A9BB", -.18),
               ("large_gap", "Largest 10% of gaps", "#BD642C", .18)]
    for ax, column, scale, title, xlabel, formatter in [
        (axes[0], "median_post_share", 100, "(a) Posting concentration", "Median share of posts (%)", "%.1f%%"),
        (axes[1], "median_absolute_score_change", 1, "(b) Change after removing authors",
         "Median absolute score change (0 to 1 scale)", "%.4f"),
    ]:
        for key, label, color, offset in cohorts:
            values = summary.loc[summary.cohort.eq(key), column] * scale
            bars = ax.barh(positions + offset, values, height=.30, color=color, label=label)
            ax.bar_label(bars, fmt=formatter, padding=4, fontsize=9)
        ax.set_yticks(positions, ["Top 1 author", "Top 5 authors"])
        ax.invert_yaxis()
        ax.set(xlabel=xlabel, title=title)
        ax.set_xlim(0, summary[column].max() * scale * 1.25)
    axes[1].legend(frameon=False, fontsize=9, loc="upper right")
    output.save_figure(fig, "22_misbot_author_influence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", nargs="?", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    checks = []

    def check(name, condition):
        if not condition:
            raise ValueError(name)
        checks.append(name)

    receipt, post_scores, topic_scores = read_archive(args.archive, check)
    topics, _ = audit_scores(post_scores, topic_scores, receipt, check)
    topics = topics.set_index("topic")
    results = recompute_after_removal(topics, author_totals(post_scores, topics))
    summary, cutoff = summarize(results)
    output = ExperimentOutput(EXPERIMENT_DIR / "misbot-author-influence")
    summary.to_csv(output.results / "author_influence_summary.csv", index=False)
    output.write_json("author_influence_validation.json", {
        "topic_count": len(topics), "small_gap_exclusive_upper_bound": SMALL_GAP,
        "large_gap_quantile": LARGE_GAP_QUANTILE, "large_gap_inclusive_lower_bound": cutoff,
        "ranking_rule": "Post count descending, then author identifier ascending within each topic",
        "gap_after_removal_reference": "Original user-balanced topic score, held fixed",
        "handoff_checks": checks,
    })
    plot(summary, output)
    print(f"Large-gap threshold: {cutoff:.8f}. Summarized {len(results):,} verified topics.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

"""Run Chang's standalone Weibo mining and next-day forecasting experiments."""
import argparse
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import pickle
from pathlib import Path

import pandas as pd
import run_analysis as legacy

from experiment_io import EXPERIMENT_DIR, ExperimentOutput, load_posts
from forecast_experiment import run_forecast, validate_forecast
from mining_experiment import build_topics, cluster_topics, describe_hotspots, mine_patterns
from mining_experiment_validation import validate_mining, validate_retrospective


def root_reference_summary(posts):
    if "origin_weibo" not in posts:
        return {"status": "field_unavailable", "posts": len(posts)}
    refs = posts.loc[posts.origin_weibo.ne(""), ["mid", "origin_weibo"]]
    matched = refs.origin_weibo.isin(posts.mid)
    self_refs = refs.origin_weibo.eq(refs.mid)
    return {
        "status": "observed_sample_root_references",
        "posts": len(posts),
        "nonempty_references": len(refs),
        "matched_including_self": int(matched.sum()),
        "self_references": int(self_refs.sum()),
        "matched_other": int((matched & ~self_refs).sum()),
        "matched_fraction_of_nonempty": float(matched.mean()) if len(refs) else None,
        "interpretation": "Matched roots in this subset; no intermediate parent edges are observed.",
    }


def fit_misbot(output):
    spec = spec_from_file_location("_chang_misbot_benchmark", legacy.__file__)
    benchmark = module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    benchmark.RES = output.results
    benchmark.FIG = output.figures
    return benchmark.bot_benchmark()


def descriptive_summary_figure(output):
    stability = pd.read_csv(output.results / "cluster_stability_comparison.csv")
    ablation = pd.read_csv(output.results / "cib_informed_ablation.csv").set_index("representation")
    timing = pd.read_csv(output.results / "pattern_method_comparison.csv").pivot(
        index="min_support", columns="method", values="seconds")
    fig, (left, middle, right) = legacy.plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    blue, gold, green = "#185C85", "#BB7C18", "#1B8278"

    left.barh(stability.representation.str.title() + " / " + stability.method, stability.ari_mean,
              xerr=stability.ari_std, color=stability.method.map({"KMeans": blue, "Ward": "#A7B5C3"}),
              error_kw={"ecolor": "#24364D", "capsize": 3})
    left.invert_yaxis()
    left.set(xlim=(0, 1.04), xlabel="Mean subsample ARI (±1 standard deviation)",
             title="(a) Stability with seven clusters")

    bars = middle.barh(["Fused", "Fused + HMI"], ablation.loc[
        ["Fused baseline", "Fused + HMI"], "common_fused_silhouette"], color=[blue, green], height=.5)
    middle.bar_label(bars, fmt="%.3f", padding=4)
    middle.invert_yaxis()
    middle.set(xlim=(0, .19), xlabel="Silhouette in common fused space", title="(b) Effect of adding HMI")

    positions = range(len(timing))
    for offset, method, color in [(-.18, "Apriori", blue), (.18, "FP-growth", gold)]:
        right.bar([position + offset for position in positions], timing[method], width=.34,
                  label=method, color=color)
    right.set_xticks(list(positions), [f"{support:.1%}" for support in timing.index])
    right.set(xlabel="Minimum support", ylabel="Elapsed seconds", title="(c) Itemset mining runtime")
    right.set_ylim(0, timing.max().max() * 1.25)
    right.legend(loc="upper center", ncol=2, fontsize=9, frameon=False)
    output.save_figure(fig, "15_descriptive_comparisons")


def run_descriptive(args):
    variant = "misbot" if args.with_misbot else "handed-off-style" if args.style_model else "descriptive"
    output = ExperimentOutput(EXPERIMENT_DIR / variant)

    posts = load_posts()
    model = None
    if args.with_misbot:
        model = fit_misbot(output)
    elif args.style_model:
        with args.style_model.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
            handle.seek(0)
            model = pickle.load(handle)
        output.write_json("style_model_provenance.json", {"source": "trusted local team handoff", "sha256": digest})

    topics, representations, membership, _ = build_topics(posts, output, model)
    cluster_topics(topics, representations, output)
    mine_patterns(posts, topics, membership, output)
    describe_hotspots(posts, membership, output)
    output.write_json("observed_root_references.json", root_reference_summary(posts))
    validate_mining(output, posts)
    validate_retrospective(output, posts, style_model=model)
    descriptive_summary_figure(output)
    print(f"Descriptive outputs and validation records saved to {output.results}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="experiment", required=True)
    descriptive = subcommands.add_parser("describe", help="Run retrospective mining and verify the saved results.")
    style = descriptive.add_mutually_exclusive_group()
    style.add_argument("--with-misbot", action="store_true", help="Fit the style model from the audited local MisBot input.")
    style.add_argument("--style-model", type=Path, help="Use a trusted local style model from the team handoff.")
    forecast = subcommands.add_parser("forecast", help="Run and verify the time-ordered forecast.")
    forecast.add_argument("--held-out", action="store_true", help="Score the frozen test window; defaults to validation.")
    args = parser.parse_args()
    if args.experiment == "describe":
        run_descriptive(args)
    else:
        variant = "forecast" if args.held_out else "forecast-development"
        output = ExperimentOutput(EXPERIMENT_DIR / variant)
        run_forecast(output, held_out=args.held_out)
        validate_forecast(output, held_out=args.held_out)


if __name__ == "__main__":
    main()

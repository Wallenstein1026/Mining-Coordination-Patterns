"""Input and private output helpers for Chang's standalone experiments."""
import json
from pathlib import Path
import re

import run_analysis as legacy
import numpy as np
import pandas as pd


class ExperimentOutput:
    def __init__(self, directory):
        self.results = directory / "results"
        self.figures = directory / "figures"
        self.results.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)

    def write_json(self, name, value):
        def encode_scalar(scalar):
            if isinstance(scalar, np.generic):
                return scalar.item()
            raise TypeError(f"Cannot encode {type(scalar).__name__} as JSON")

        (self.results / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=encode_scalar), encoding="utf-8", newline="\n")

    def save_figure(self, figure, name):
        figure.savefig(self.figures / f"{name}.pdf", bbox_inches="tight")
        figure.savefig(self.figures / f"{name}.png", dpi=200, bbox_inches="tight")
        legacy.plt.close(figure)


EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / ".local" / "experiments" / "standalone"


def load_posts(cutoff=None):
    path, _ = legacy.dataset_input("weibo")
    posts = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = ["mid", "user_id", "created_at", "content"]
    if not set(required).issubset(posts):
        raise ValueError("Normalized Weibo CSV requires " + ", ".join(required))
    valid = posts[required].apply(lambda column: column.str.strip().ne("")).all(axis=1)
    posts = posts.loc[valid].drop_duplicates("mid").copy()
    posts["time"] = pd.to_datetime(posts.created_at, errors="coerce")
    if posts.time.dt.tz is not None:
        raise ValueError("Normalize timestamps to naive Asia/Shanghai local time first.")
    posts = posts[posts.time.notna()].reset_index(drop=True)
    if cutoff is not None:
        posts = posts[posts.time.dt.floor("D").le(pd.Timestamp(cutoff))].reset_index(drop=True)
    if posts.empty:
        raise ValueError("No valid Weibo posts in the requested observation window.")
    posts["norm"] = posts.content.map(legacy.clean_text)
    posts["tags"] = posts.content.map(
        lambda text: sorted({tag.strip().lower() for tag in re.findall(r"#([^#\n]{2,70})#", text) if tag.strip()}))
    return posts

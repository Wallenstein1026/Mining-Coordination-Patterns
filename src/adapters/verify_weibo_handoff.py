"""Check the published input through Chang's actual reader; print aggregates only."""
import argparse
from collections import Counter
from pathlib import Path
import sys

from weibo_cov import AUDIT, MANIFEST, PLAN, RAW, ROOT, inventory, read_json, sha256

sys.path.insert(0, str(ROOT / "src"))
import run_analysis


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root, check_raw=False):
    audit = read_json(root / AUDIT)
    require(audit["status"] == "monthly_import_verified", "Waiting for the monthly import")
    manifest = read_json(root / MANIFEST)
    plan = read_json(root / PLAN)
    require(plan == manifest["selection"] == audit["selection"], "Selection differs")
    require(sha256(root / PLAN) == audit["selection_config_sha256"] == manifest["selection_config_sha256"],
            "Selection file hash differs")
    require(manifest["normalized_sha256"] == audit["normalized_sha256"], "Audit and manifest disagree")
    require(manifest["raw_files"] == audit["raw_files"], "Raw inventories differ")
    if check_raw:
        listed = {item["filename"]: item for item in inventory(root, plan)}
        require(set(listed) == {f["filename"] for f in manifest["raw_files"]}, "Monthly names differ")
        for item in manifest["raw_files"]:
            name = item["filename"]
            require(listed[name]["state"] == "size_matches" and item["bytes"] == listed[name]["expected_bytes"],
                    f"{name}: byte count differs")
            require(sha256(root / RAW / name) == item["sha256"], f"{name}: raw SHA-256 differs")

    # load_weibo() checks the actual input SHA-256 before parsing and writes its own aggregate audit.
    d = run_analysis.load_weibo()
    local = read_json(root / "results/weibo_audit.json")
    n = len(d)
    require(n == local["raw_posts"] == audit["flow"]["retained"], "Normalized count or row validity differs")
    require(d.mid.is_unique and d.user_id.nunique() == audit["unique_users"], "Key or user count differs")
    require(str(d.time.min()) == str(run_analysis.pd.Timestamp(audit["first"])) and
            str(d.time.max()) == str(run_analysis.pd.Timestamp(audit["last"])), "Date bounds differ")
    require(Counter(d.time.dt.strftime("%Y-%m-%d")) ==
            Counter({day: count for day, count in audit["retained_daily_posts"].items() if count}),
            "Daily coverage differs")
    refs = d.origin_weibo[d.origin_weibo.ne("")]
    rel = audit["references"]
    require(len(refs) == rel["nonempty"] and int(refs.isin(d.mid).sum()) == rel["matched_including_self"]
            and int(refs.eq(d.mid.loc[refs.index]).sum()) == rel["self_references"], "Root coverage differs")
    for field, checks in audit["retained_field_checks"].items():
        missing = d[field].eq("")
        require(d[field + "_missing"].eq(missing.astype(int).astype(str)).all()
                and int(missing.sum()) == checks["unavailable"], f"{field}: missing flags differ")
    return dict(rows=n, users=audit["unique_users"], first=audit["first"], last=audit["last"],
                normalized_sha256=manifest["normalized_sha256"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-raw", action="store_true", help="Also re-hash each original monthly file")
    args = parser.parse_args()
    try:
        print(verify(ROOT, args.check_raw))
    except (OSError, KeyError, ValueError) as exc:
        sys.exit(str(exc))

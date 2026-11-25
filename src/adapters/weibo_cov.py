"""Build Group 28's fixed Weibo-COV 2.0 subset using only the Python standard library."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
from collections import Counter
from contextlib import closing
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = Path("data/raw/original_weibo_cov")
NORMAL = Path("data/processed/weibo.csv")
MANIFEST = Path("data/weibo_manifest.json")
AUDIT = Path("results/weibo_data_audit.json")
PLAN = Path("configs/weibo_subset.json")
ORIGIN = "https://github.com/nghuyong/weibo-cov"
COUNTS = ("like_num", "repost_num", "comment_num")
REQUIRED = ("_id", "user_id", "created_at", "content")
FIELDS = ("mid", "user_id", "created_at", "content", "origin_weibo", *COUNTS,
          *(f"{field}_missing" for field in ("origin_weibo", *COUNTS)))
FLOW = ("raw_rows", "bad_width", "bad_timestamp", "outside_window", "missing_mid",
        "outside_hash_bucket", "selected", "invalid_required", "valid_before_dedup",
        "duplicates", "retained")
MISSING = {"", "null", "none", "nan", "na", "n/a", r"\n"}
ZONE = "Asia/Shanghai"


def blank(value):
    return value.strip().lower() in MISSING


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def day_range(plan):
    day = date.fromisoformat(plan["start"])
    end = date.fromisoformat(plan["end"])
    if end < day or plan["timezone"] != ZONE:
        raise ValueError("Invalid window or time zone")
    if not 0 <= plan["mid_hash_remainder"] < plan["mid_hash_modulus"]:
        raise ValueError("Invalid sampling bucket")
    while day <= end:
        yield day.isoformat()
        day += timedelta(days=1)


def inventory(root, plan):
    listing = read_json(root / "data/official_access.json")["listing"]
    sizes = {item["filename"]: item["bytes"] for item in listing}
    names = sorted({day[:7] + ".csv" for day in day_range(plan)})
    items = []
    for name in names:
        path = root / RAW / name
        actual = path.stat().st_size if path.is_file() else None
        expected = sizes[name]
        state = "size_matches" if actual == expected else "missing" if actual is None else "size_mismatch"
        items.append(dict(filename=name, expected_bytes=expected, local_bytes=actual, state=state))
    return items


@lru_cache(maxsize=8192)
def normalized_time(value):
    # Keep the same naive Beijing-time grammar under Python 3.10 and 3.12.
    value = value.strip()
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}[ T][0-9]{2}:[0-9]{2}(?::[0-9]{2}(?:[.][0-9]{1,6})?)?", value):
        return None
    if "." in value:
        whole, fraction = value.split(".")
        value = whole + "." + fraction.ljust(6, "0")
    try:
        return datetime.fromisoformat(value).isoformat(sep=" ", timespec="microseconds")
    except ValueError:
        return None


def normalize_metrics(row, indices, audit):
    values, flags = [], []
    for field in ("origin_weibo", *COUNTS):
        value = row[indices[field]].strip()
        absent = blank(value)
        if field in COUNTS and not absent:
            if re.fullmatch(r"[0-9]+", value):
                value = str(int(value))
            else:
                audit[field + "_invalid"] += 1
                absent = True
        if absent:
            value = ""
            audit[field + "_unavailable"] += 1
        values.append(value)
        flags.append(str(int(absent)))
    return [*values, *flags]


def scan_file(path, plan, db):
    flow = Counter(dict.fromkeys(FLOW, 0))
    missing = Counter(dict.fromkeys((*REQUIRED, "origin_weibo", *COUNTS), 0))
    raw_days, quality = Counter(), Counter()
    unexpected_month = 0
    field_set = set(missing)
    with path.open(encoding="utf-8-sig", newline="") as src:
        reader = csv.reader(src, strict=True)
        header = next(reader, [])
        if len(set(header)) != len(header) or not field_set.issubset(header):
            raise ValueError(f"{path.name}: missing or duplicate column headers")
        cols = {field: header.index(field) for field in field_set}
        for row in reader:
            flow["raw_rows"] += 1
            if len(row) != len(header):
                flow["bad_width"] += 1
                continue
            for field, i in cols.items():
                is_missing = not row[i].strip() if field == "content" else blank(row[i])
                if is_missing:
                    missing[field] += 1
            stamp = normalized_time(row[cols["created_at"]])
            if stamp is None:
                flow["bad_timestamp"] += 1
                continue
            day = stamp[:10]
            raw_days[day] += 1
            unexpected_month += day[:7] != path.stem
            if not plan["start"] <= day <= plan["end"]:
                flow["outside_window"] += 1
                continue
            mid = row[cols["_id"]].strip()
            if blank(mid):
                flow["missing_mid"] += 1
                continue
            bucket = int.from_bytes(hashlib.sha256(mid.encode("utf-8")).digest()[:8], "big")
            if bucket % plan["mid_hash_modulus"] != plan["mid_hash_remainder"]:
                flow["outside_hash_bucket"] += 1
                continue
            flow["selected"] += 1
            uid, content = row[cols["user_id"]].strip(), row[cols["content"]]
            if blank(uid) or not content.strip():
                flow["invalid_required"] += 1
                continue
            flow["valid_before_dedup"] += 1
            row_quality = Counter()
            payload = (mid, uid, stamp, content, *normalize_metrics(row, cols, row_quality))
            ins = db.execute(INSERT, payload)
            if ins.rowcount:
                flow["retained"] += 1
                quality.update(row_quality)
            else:
                flow["duplicates"] += 1
    if (flow["raw_rows"] != sum(flow[k] for k in FLOW[1:7])
            or flow["selected"] != flow["invalid_required"] + flow["valid_before_dedup"]
            or flow["valid_before_dedup"] != flow["retained"] + flow["duplicates"]):
        raise ValueError("Row accounting failed")
    return dict(flow=dict(flow), raw_field_missing=dict(missing),
                parseable_rows_by_day=dict(raw_days), rows_outside_filename_month=unexpected_month,
                retained_field_checks=dict(quality))


INSERT = "INSERT OR IGNORE INTO posts VALUES (" + ",".join("?" for _ in FIELDS) + ")"


def build(paths, plan, workspace):
    csv.field_size_limit(64 * 1024 * 1024)
    raw_files, file_reports = [], {}
    with closing(sqlite3.connect(workspace / "rows.sqlite")) as db:
        definitions = ", ".join(f + " TEXT" for f in FIELDS)
        db.execute("CREATE TABLE posts (" + definitions + ", PRIMARY KEY(mid)) WITHOUT ROWID")
        for path in sorted(paths):
            before = path.stat()
            digest = sha256(path)
            with db:
                report = scan_file(path, plan, db)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                raise ValueError(f"{path.name} changed during the import")
            raw_files.append(dict(filename=path.name, bytes=after.st_size, sha256=digest))
            file_reports[path.name] = report
        total, users, lo, hi = db.execute(
            "SELECT COUNT(*), COUNT(DISTINCT user_id), MIN(created_at), MAX(created_at) FROM posts"
        ).fetchone()
        if not total:
            raise ValueError("No valid retained rows; check raw files and fixed selection")
        daily = dict.fromkeys(day_range(plan), 0)
        daily.update(db.execute("SELECT substr(created_at,1,10), count(*) FROM posts GROUP BY 1"))
        refs, matched, self_refs = db.execute("""
            SELECT COUNT(*), COUNT(r.mid), COUNT(CASE WHEN p.mid=p.origin_weibo THEN 1 END)
            FROM posts p LEFT JOIN posts r ON r.mid=p.origin_weibo WHERE p.origin_weibo!=''
        """).fetchone()
        with (workspace / "weibo.csv").open("w", newline="", encoding="utf-8") as out:
            writer = csv.writer(out, lineterminator="\n", quoting=csv.QUOTE_ALL)
            writer.writerow(FIELDS)
            writer.writerows(db.execute("SELECT * FROM posts ORDER BY created_at, mid"))
    flow = {key: sum(rep["flow"][key] for rep in file_reports.values()) for key in FLOW}
    if total != flow["retained"] or total != sum(daily.values()):
        raise ValueError("Export accounting failed")
    quality = {f: {kind: sum(rep["retained_field_checks"].get(f + "_" + kind, 0)
                             for rep in file_reports.values())
                   for kind in (("unavailable",) if f == "origin_weibo" else ("invalid", "unavailable"))}
               for f in ("origin_weibo", *COUNTS)}
    normalized_hash = sha256(workspace / "weibo.csv")
    return dict(files=file_reports, flow=flow, unique_users=users, first=lo, last=hi,
                retained_daily_posts=daily, days_without_retained_posts=[d for d, n in daily.items() if n == 0],
                retained_field_checks=quality,
                references=dict(nonempty=refs, nonempty_over_posts=refs/total,
                                matched_including_self=matched, self_references=self_refs,
                                matched_over_nonempty=matched/refs if refs else None,
                                matched_other_over_nonempty=(matched-self_refs)/refs if refs else None),
                raw_files=raw_files, normalized_sha256=normalized_hash, selection=plan)


def publish(root, plan, items):
    incomplete = [item["filename"] + ": " + item["state"] for item in items if item["state"] != "size_matches"]
    if incomplete:
        raise ValueError("Monthly inputs not ready: " + ", ".join(incomplete))
    private = root / ".local"
    private.mkdir(mode=0o700, exist_ok=True)
    with (private / "weibo-import.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with tempfile.TemporaryDirectory(dir=private, prefix="weibo-build-") as directory:
            tmp = Path(directory)
            report = build([root / RAW / item["filename"] for item in items], plan, tmp)
            if {f["filename"]: f["bytes"] for f in report["raw_files"]} != {
                item["filename"]: item["expected_bytes"] for item in items
            }:
                raise ValueError("Monthly file size changed after inventory")
            if read_json(root / PLAN) != plan:
                raise ValueError("Selection changed during import")
            report["status"] = "monthly_import_verified"
            report["selection_config_sha256"] = sha256(root / PLAN)
            manifest = dict(dataset="nghuyong/Weibo-COV", version="2.0", source_url=ORIGIN,
                            timezone=ZONE, normalized_file=NORMAL.as_posix(),
                            normalized_sha256=report["normalized_sha256"], raw_files=report["raw_files"],
                            selection=plan, selection_config_sha256=report["selection_config_sha256"])
            save_json(tmp / "manifest.json", manifest)
            save_json(tmp / "audit.json", report)
            if (root / MANIFEST).exists() and read_json(root / MANIFEST) != manifest:
                raise ValueError("Existing manifest differs. Preserve the prior release and review input drift.")
            if (root / NORMAL).exists() and sha256(root / NORMAL) != report["normalized_sha256"]:
                raise ValueError("Existing normalized file differs; review it before publishing.")
            (root / NORMAL).parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            # Publish the manifest last. The downstream reader verifies its data hash.
            for src, dest in [("weibo.csv", NORMAL), ("audit.json", AUDIT), ("manifest.json", MANIFEST)]:
                os.replace(tmp / src, root / dest)
    print("Rows:", report["flow"]["retained"], "dates:", report["first"], report["last"])
    print("SHA-256:", report["normalized_sha256"])


def audit_available(root, plan, items):
    ready = [item["filename"] for item in items if item["state"] == "size_matches"]
    if not ready:
        raise ValueError("No complete month available for a provisional audit")
    private = root / ".local"
    private.mkdir(mode=0o700, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=private, prefix="weibo-preview-") as directory:
        report = build([root / RAW / name for name in ready], plan, Path(directory))
    months = {name[:7] for name in ready}
    covered = {day: count for day, count in report["retained_daily_posts"].items() if day[:7] in months}
    report.update(status="provisional_received_months_only", inventory=items,
                  retained_daily_posts=covered, days_without_retained_posts=[d for d,n in covered.items() if n == 0])
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--audit-available", action="store_true", help="Print aggregate QA for complete files received to date")
    options.add_argument("--inventory", action="store_true", help="List monthly input sizes and availability")
    args = parser.parse_args()
    os.umask(0o077)
    plan = read_json(ROOT / PLAN)
    items = inventory(ROOT, plan)
    if args.inventory:
        print(json.dumps(items, indent=2))
        return
    if args.audit_available:
        audit_available(ROOT, plan, items)
        return
    publish(ROOT, plan, items)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, csv.Error, sqlite3.Error) as exc:
        sys.exit(str(exc))

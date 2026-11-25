"""Compare root-reference coverage for the same sample, with two endpoint search ranges."""
import csv
import json
from collections import Counter
from weibo_cov import AUDIT, MANIFEST, NORMAL, RAW, ROOT, read_json, sha256


def count_matches():
    audit = read_json(ROOT / AUDIT)
    manifest = read_json(ROOT / MANIFEST)
    if sha256(ROOT / NORMAL) != manifest["normalized_sha256"] or manifest["raw_files"] != audit["raw_files"]:
        raise ValueError("Inputs differ from the published audit")
    csv.field_size_limit(64 * 1024 * 1024)
    references = Counter()
    sample_ids = set()
    with (ROOT / NORMAL).open(encoding="utf-8", newline="") as src:
        for row in csv.DictReader(src):
            sample_ids.add(row["mid"])
            if row["origin_weibo"]:
                references[row["origin_weibo"]] += 1
    in_sample = references.keys() & sample_ids
    in_raw = set()
    for item in manifest["raw_files"]:
        path = ROOT / RAW / item["filename"]
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Raw SHA-256 differs: {path.name}")
        with path.open(encoding="utf-8-sig", newline="") as src:
            reader = csv.reader(src, strict=True)
            header = next(reader)
            idx = header.index("_id")
            n = 0
            for record in reader:
                n += 1
                if len(record) == len(header):
                    mid = record[idx].strip()
                    if mid in references:
                        in_raw.add(mid)
            if n != audit["files"][path.name]["flow"]["raw_rows"]:
                raise ValueError(f"Raw count differs: {path.name}")
    ref_rows = sum(references.values())
    sample_rows = sum(references[x] for x in in_sample)
    raw_rows = sum(references[x] for x in in_raw)
    if (not in_sample.issubset(in_raw) or ref_rows != audit["references"]["nonempty"]
            or sample_rows != audit["references"]["matched_including_self"]):
        raise ValueError("Sample reference counts differ")
    return dict(normalized_sha256=manifest["normalized_sha256"], referenced_posts=ref_rows,
                distinct_referenced_ids=len(references), posts_with_root_in_subset=sample_rows,
                distinct_roots_in_subset=len(in_sample), posts_with_root_in_raw_months=raw_rows,
                distinct_roots_in_raw_months=len(in_raw), extra_resolved_posts=raw_rows-sample_rows,
                denominator="Retained posts with a nonempty origin_weibo; repeated references count per post.",
                raw_scope="All correctly shaped records in the three input CSVs, matched by _id.")


if __name__ == "__main__":
    print(json.dumps(count_matches(), ensure_ascii=False, indent=2))

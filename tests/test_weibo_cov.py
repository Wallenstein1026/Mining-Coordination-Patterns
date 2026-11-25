"""Focused regressions for ingestion. Optional example: extract the pinned upstream v2 README CSV."""
import csv
import io
import json
from contextlib import redirect_stdout
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src/adapters"))
import weibo_cov as w

HEADER = ["_id", "user_id", "created_at", "content", "origin_weibo", *w.COUNTS]


class ImportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=os.getenv("TMPDIR"))
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        for d in [w.RAW, w.PLAN.parent, w.AUDIT.parent]:
            (self.root / d).mkdir(parents=True, exist_ok=True)
        self.plan = dict(start="2020-09-01", end="2020-10-31", timezone=w.ZONE,
                         mid_hash_modulus=1, mid_hash_remainder=0)

    def raw(self, name, rows, header=HEADER):
        path = self.root / w.RAW / name
        with path.open("w", newline="", encoding="utf-8-sig") as dest:
            writer = csv.writer(dest)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def register(self):
        w.save_json(self.root / w.PLAN, self.plan)
        w.save_json(self.root / "data/official_access.json", dict(listing=[
            dict(filename=p.name, bytes=p.stat().st_size) for p in sorted((self.root / w.RAW).glob("*.csv"))
        ]))

    def publish(self):
        w.publish(self.root, self.plan, w.inventory(self.root, self.plan))

    def test_rows_metrics_relations_and_reproducibility(self):
        a = ["001", "u1", "2020-09-01 08:03", 'a,"quote"\r\nb', "", "0", "1", "2"]
        b = ["002", "u2", "2020-09-02 00:00:00.5", " NA ", "001", "", "-1", "oops"]
        self.raw("2020-09.csv", [a, b, b, [*a[:3], "  ", *a[4:]], a[:-1],
                                 [a[0],a[1],"bad-time",*a[3:]], ["", *a[1:]],
                                 [a[0],a[1],"2019-01-01 01:00",*a[3:]]])
        self.raw("2020-10.csv", [a, ["003","u1","2020-10-01 12:00","x","elsewhere","0","0","0"],
                                 ["004","u3","2020-10-02 12:00","x","004","0","0","0"]])
        self.register()
        self.publish()
        info = w.read_json(self.root / w.AUDIT)
        self.assertEqual(info["flow"], dict(raw_rows=11, bad_width=1, bad_timestamp=1, outside_window=1,
            missing_mid=1, outside_hash_bucket=0, selected=7, invalid_required=1, valid_before_dedup=6,
            duplicates=2, retained=4))
        self.assertEqual(info["references"], dict(nonempty=3,nonempty_over_posts=.75,matched_including_self=2,
            self_references=1,matched_over_nonempty=2/3,matched_other_over_nonempty=1/3))
        self.assertEqual(info["retained_field_checks"]["repost_num"], {"unavailable":1,"invalid":1})
        self.assertEqual(info["retained_field_checks"]["like_num"], {"unavailable":1,"invalid":0})
        with (self.root / w.NORMAL).open(newline="") as src:
            rows = list(csv.DictReader(src))
        self.assertEqual(rows[0]["mid"],"001")
        self.assertEqual(rows[0]["content"],'a,"quote"\r\nb')
        self.assertEqual(rows[1]["content"]," NA ")
        self.assertEqual(len(info["retained_daily_posts"]),61)
        old = [(self.root / p).read_bytes() for p in (w.NORMAL,w.AUDIT,w.MANIFEST)]
        self.publish()
        self.assertEqual(old, [(self.root / p).read_bytes() for p in (w.NORMAL,w.AUDIT,w.MANIFEST)])
        with (self.root / w.RAW / "2020-09.csv").open("ab") as f:
            f.write(b"partial")
        with self.assertRaisesRegex(ValueError,"size_mismatch"):
            self.publish()
        self.assertEqual(old, [(self.root / p).read_bytes() for p in (w.NORMAL,w.AUDIT,w.MANIFEST)])

    def test_hash_bucket_and_malformed_input(self):
        self.plan.update(end="2020-09-30", mid_hash_modulus=32)
        # Use independent full-digest hex prefix and modulo for the expected sample.
        rows = [[str(i),str(i),"2020-09-02 00:00","x","","0","0","0"] for i in range(256)]
        self.raw("2020-09.csv", rows)
        self.register()
        self.publish()
        import hashlib
        expected = {r[0] for r in rows if int(hashlib.sha256(r[0].encode()).hexdigest()[:16],16)%32 == 0}
        with (self.root / w.NORMAL).open() as f:
            self.assertEqual({r["mid"] for r in csv.DictReader(f)},expected)
        p = self.raw("2020-09.csv",[],["wrong"])
        self.register()
        with self.assertRaisesRegex(ValueError,"column headers"):
            self.publish()
        p.unlink()
        with self.assertRaisesRegex(ValueError,"missing"):
            self.publish()

    def test_early_coverage_does_not_publish_partial_data(self):
        path = self.raw("2020-09.csv", [["01","u","2020-09-02 00:00","x","","1","0","0"]])
        w.save_json(self.root / "data/official_access.json", dict(listing=[
            dict(filename=path.name, bytes=path.stat().st_size), dict(filename="2020-10.csv", bytes=100)]))
        out = io.StringIO()
        with redirect_stdout(out):
            w.audit_available(self.root, self.plan, w.inventory(self.root, self.plan))
        report = json.loads(out.getvalue())
        self.assertEqual(report["status"], "provisional_received_months_only")
        self.assertEqual(len(report["retained_daily_posts"]),30)
        self.assertEqual(report["flow"]["retained"],1)
        self.assertFalse((self.root / w.MANIFEST).exists())
        self.assertFalse((self.root / w.AUDIT).exists())
        with self.assertRaisesRegex(ValueError,"missing"):
            self.publish()

    @unittest.skipUnless(os.getenv("WEIBO_PUBLISHER_EXAMPLE"), "set WEIBO_PUBLISHER_EXAMPLE for pinned official example")
    def test_official_documentation_example_with_changs_reader(self):
        import verify_weibo_handoff as v
        original = Path(os.environ["WEIBO_PUBLISHER_EXAMPLE"])
        self.assertEqual(w.sha256(original),"0238359f1001aab6ccda423f68a43a5fcdeb9837bed122daf5642aafca5adce0")
        self.plan.update(start="2020-12-01",end="2020-12-01")
        shutil.copyfile(original,self.root / w.RAW / "2020-12.csv")
        self.register()
        self.publish()
        with patch.object(v.run_analysis,"ROOT",self.root), patch.object(v.run_analysis,"RES",self.root/"results"):
            verified = v.verify(self.root,check_raw=True)
            self.assertEqual(verified["rows"],6)
            self.assertEqual(verified["users"],6)
            self.assertEqual(verified["first"],"2020-12-01 00:00:00.000000")
            result = w.read_json(self.root / w.AUDIT)
            print("Publisher README example only:", {**verified, "references": result["references"]})
            self.publish()
            self.assertEqual(verified, v.verify(self.root))
            with (self.root / w.NORMAL).open("ab") as f:
                f.write(b"\n")
            with self.assertRaisesRegex(ValueError,"SHA-256 mismatch"):
                v.verify(self.root)


if __name__ == "__main__":
    unittest.main()

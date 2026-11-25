"""Unit tests for dynamic degree handling in tools/build_atlas.py."""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import build_atlas


class BuildAtlasTests(unittest.TestCase):
    def test_max_degree_defaults_to_min_maxn(self):
        self.assertEqual(build_atlas.max_degree([]), build_atlas.MIN_MAXN)

    def test_max_degree_extends_for_rules_above_min_maxn(self):
        fams = [{"rules": [{"degree": 12}, {"degree": 51}]}]
        self.assertEqual(build_atlas.max_degree(fams), 51)

    def test_max_degree_extends_for_literature_counts(self):
        lc = {"cube": [{"k": "pk", "arr": [None] * 54 + [7000]}]}
        self.assertEqual(build_atlas.max_degree([], lc), 55)

    def test_gj_tables_cover_highest_degree(self):
        tables = build_atlas.gj_tables(51)
        self.assertIn("26", tables["L"])
        self.assertIn("26", tables["J1"])
        self.assertIn("26", tables["J2"])

    def test_id_regex_accepts_two_and_three_digit_degrees(self):
        self.assertIsNotNone(build_atlas.ID_RE.match("CA-C3-51-6303-PI-MPC"))
        self.assertIsNotNone(build_atlas.ID_RE.match("CA-C3-100-50000-PI-MPC"))
        self.assertIsNone(build_atlas.ID_RE.match("CA-C3-5-13-PI-MPC"))

    def test_domain_pts_and_lazy_payload(self):
        import json
        fams = [{
            "id": "test_pi", "tag": "T", "role": "pi",
            "name": {"en": "Test", "de": "Test"},
            "short": {"en": "T", "de": "T"},
            "rules": [{
                "id": "CA-C2-01-1-PI-MPC", "family": "test_pi", "domain": "square",
                "degree": 1, "points": 1, "_q": "PI", "_res": 0.0,
                "nodes": [[0.0, 0.0]], "weights": [4.0],
            }],
        }]
        chunks, hashes = build_atlas.domain_pts(fams)
        self.assertEqual(set(chunks.keys()), set(build_atlas.DOMAINS))
        self.assertEqual(json.loads(chunks["square"]), {"CA-C2-01-1-PI-MPC": [[0.0, 0.0, 4.0]]})
        self.assertEqual(len(hashes["square"]), 8)
        lazy = build_atlas.payload(fams, {}, {}, "", "", False, "de", include_pts=False, pts_hashes=hashes)
        r_lazy = lazy["families"][0]["rules"]["square"][0]
        self.assertNotIn("pts", r_lazy)
        self.assertEqual(r_lazy["n"], 1)
        self.assertEqual(lazy["links"]["ptsBase"], "../data/")
        self.assertEqual(lazy["ptsHash"]["square"], hashes["square"])
        full = build_atlas.payload(fams, {}, {}, "", "", False, "en", include_pts=True)
        r_full = full["families"][0]["rules"]["square"][0]
        self.assertEqual(r_full["pts"], [[0.0, 0.0, 4.0]])
        self.assertNotIn("ptsBase", full["links"])

    def test_blas_threads_forced_to_one(self):
        for var in build_atlas.BLAS_THREAD_VARS:
            self.assertEqual(os.environ.get(var), "1", f"{var} should be forced to 1")

    def test_parallel_and_sequential_rule_checking_agree(self):
        fams1 = build_atlas.load_families()
        fams2 = build_atlas.load_families()
        # Check a small deterministic subset (6 real rules across domains) sequentially vs in parallel
        only = set()
        for f in fams1:
            fdir = os.path.join(build_atlas.RULES, f["id"])
            for dom in sorted(os.listdir(fdir)):
                ddir = os.path.join(fdir, dom)
                if not os.path.isdir(ddir):
                    continue
                for fn in sorted(os.listdir(ddir)):
                    if fn.endswith(".json"):
                        only.add(os.path.abspath(os.path.join(ddir, fn)))
                        break
                if len(only) >= 6:
                    break
            if len(only) >= 6:
                break
        build_atlas.errors.clear()
        build_atlas.load_all_rules(fams1, only=only, jobs=1)
        errs1 = list(build_atlas.errors)
        build_atlas.errors.clear()
        build_atlas.load_all_rules(fams2, only=only, jobs=2)
        errs2 = list(build_atlas.errors)
        self.assertEqual(errs1, [])
        self.assertEqual(errs2, [])
        ids1 = [(r["id"], r.get("_res"), r.get("_q")) for f in fams1 for r in f["rules"]]
        ids2 = [(r["id"], r.get("_res"), r.get("_q")) for f in fams2 for r in f["rules"]]
        self.assertEqual(ids1, ids2)


if __name__ == "__main__":
    unittest.main()

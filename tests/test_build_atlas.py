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


if __name__ == "__main__":
    unittest.main()

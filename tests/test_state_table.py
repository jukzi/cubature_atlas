"""Unit tests for the atlas state-table coverage."""

import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from state_table import load_rules, state_rows


class StateTableTests(unittest.TestCase):
    def test_missing_cells_are_listed_in_each_domain(self):
        rows = state_rows({}, min_degree=29)
        self.assertEqual(len(rows), 10)
        self.assertEqual(
            [(row[0], row[1]) for row in rows],
            [(domain, degree) for domain in sorted(
                ("interval", "triangle", "square", "tetra", "cube")
            ) for degree in (29, 30)],
        )
        self.assertEqual(rows[-1][0:2], ["triangle", 30])

    def test_domain_range_extends_to_highest_stored_degree(self):
        best = {("tetra", 35, "PI"): (900, "rule-35")}
        rows = state_rows(best, domain_filter="tetra")
        self.assertEqual(len(rows), 35)
        self.assertEqual(rows[-1][0:2], ["tetra", 35])
        self.assertEqual(rows[-1][3], 900)

    def test_pi_po_gaps_and_filtering(self):
        best = {
            ("triangle", 2, "PI"): (3, "pi-rule"),
            ("triangle", 2, "PO"): (2, "po-rule"),
        }
        rows = state_rows(best, domain_filter="triangle", min_degree=2)
        degree_two = rows[0]
        self.assertEqual(degree_two, ["triangle", 2, 2, 3, 1, 2, 0, "yes"])

    def test_load_rules_keeps_smallest_count_per_quality(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(
                root, "rules", "family", "triangle", "rule.json"
            )
            os.makedirs(os.path.dirname(path))
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({
                    "domain": "triangle",
                    "degree": 4,
                    "quality": "PI",
                    "points": 11,
                    "id": "CA-T2-04-11-PI-X",
                }, handle)
            best = load_rules(root)
        self.assertEqual(best[("triangle", 4, "PI")], (11, "CA-T2-04-11-PI-X"))


if __name__ == "__main__":
    unittest.main()

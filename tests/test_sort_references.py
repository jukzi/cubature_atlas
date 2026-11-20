"""Unit tests for stable bibliography sorting and citation renumbering."""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from sort_references import (
    build_mapping,
    is_sorted,
    remap_brackets,
    remap_list_text,
    year_key,
)


class ReferenceSortingTests(unittest.TestCase):
    def test_years_are_sorted_numerically_and_undated_entries_last(self):
        refs = [
            {"n": 1, "y": "2006"},
            {"n": 2, "y": "1958"},
            {"n": 3, "y": ""},
            {"n": 4, "y": "not-a-year"},
        ]

        self.assertEqual(
            sorted(refs, key=year_key),
            [refs[1], refs[0], refs[2], refs[3]],
        )

    def test_same_year_entries_keep_their_original_order(self):
        refs = [
            {"n": 8, "y": "2020"},
            {"n": 3, "y": "1958"},
            {"n": 6, "y": "2020"},
            {"n": 4, "y": ""},
        ]

        mapping, ordered = build_mapping(refs)

        self.assertEqual([entry["n"] for entry in ordered], [3, 8, 6, 4])
        self.assertEqual(mapping, {3: 1, 8: 2, 6: 3, 4: 4})

    def test_is_sorted_accepts_equal_years_and_undated_tail(self):
        self.assertTrue(is_sorted([
            {"y": "1958"},
            {"y": "2020"},
            {"y": "2020"},
            {"y": ""},
        ]))
        self.assertFalse(is_sorted([
            {"y": "2020"},
            {"y": "1958"},
        ]))
        self.assertFalse(is_sorted([
            {"y": ""},
            {"y": "2020"},
        ]))

    def test_remap_list_text_changes_only_the_named_json_arrays(self):
        source = '{"references": [8, 3], "other": [8], "references_extra": [3]}'

        result = remap_list_text(source, "references", {8: 2, 3: 1})

        self.assertEqual(
            result,
            '{"references": [2, 1], "other": [8], "references_extra": [3]}',
        )

    def test_remap_brackets_preserves_unknown_and_non_numeric_brackets(self):
        source = "see [8], [3], [unknown], and [8, 3]"

        result = remap_brackets(source, {8: 2, 3: 1})

        self.assertEqual(result, "see [2], [1], [unknown], and [8, 3]")


if __name__ == "__main__":
    unittest.main()

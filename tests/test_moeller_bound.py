"""Regression tests for the modular Moeller rank arithmetic."""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
import moeller_bound as MB
from triangle_moeller import moeller_rank


class ModularMoellerTests(unittest.TestCase):
    def test_simplex_and_box_moments_have_correct_normalization(self):
        self.assertEqual(MB.moment_raw("triangle", (0, 0)), MB.inv(2))
        self.assertEqual(MB.moment_raw("tetra", (0, 0, 0)), MB.inv(6))
        self.assertEqual(MB.moment_raw("square", (0, 0)), 4)
        self.assertEqual(MB.moment_raw("cube", (0, 0, 0)), 8)
        self.assertEqual(MB.moment_raw("square", (1, 0)), 0)
        self.assertEqual(MB.moment_raw("square", (2, 0)), 4 * MB.inv(3) % MB.PRIME)

    def test_split_modular_matmul_avoids_large_integer_products(self):
        p = MB.PRIME
        a = np.array([[p - 2, p - 5, p - 7], [p - 11, 13, p - 17]], dtype=np.int64)
        b = np.array([[p - 19, 23], [p - 29, p - 31], [37, p - 41]], dtype=np.int64)

        actual = MB.matmul(a, b)
        expected = np.array([
            [sum(int(a[i, k]) * int(b[k, j]) for k in range(a.shape[1])) % p
             for j in range(b.shape[1])]
            for i in range(a.shape[0])
        ], dtype=np.int64)

        np.testing.assert_array_equal(actual, expected)

    def test_modular_triangle_rank_matches_independent_exact_fraction_rank(self):
        ranks, total, r_k = MB.structural_ranks("triangle", 4)

        self.assertIsNone(total)
        self.assertEqual(r_k, 5)
        self.assertEqual(ranks[(0, 1)], moeller_rank(4))
        self.assertEqual(ranks[(0, 1)], 4)

    def test_tetrahedron_structural_rank_shapes_are_well_formed(self):
        ranks, total, r_k = MB.structural_ranks("tetra", 1)

        self.assertEqual(set(ranks), {(0, 1), (0, 2), (1, 2)})
        self.assertIsInstance(total, int)
        self.assertEqual(r_k, 3)
        self.assertTrue(all(value % 2 == 0 for value in ranks.values()))
        self.assertTrue(all(0 <= value <= r_k for value in ranks.values()))
        self.assertTrue(0 <= total <= 3 * r_k)


if __name__ == "__main__":
    unittest.main()

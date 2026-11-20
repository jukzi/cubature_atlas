"""Exact-arithmetic regression tests for the triangle Moeller bound."""

from fractions import Fraction
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
from triangle_moeller import moment, monomials, moeller_rank, rank, solve


class TriangleMoellerTests(unittest.TestCase):
    def test_reference_triangle_moments_are_exact_fractions(self):
        self.assertEqual(moment(0, 0), Fraction(1, 2))
        self.assertEqual(moment(2, 3), Fraction(1, 420))
        self.assertIsInstance(moment(2, 3), Fraction)

    def test_monomial_basis_is_ordered_by_total_degree(self):
        self.assertEqual(
            monomials(2),
            [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)],
        )

    def test_exact_rank_handles_full_and_deficient_fraction_matrices(self):
        self.assertEqual(
            rank([
                [Fraction(1), Fraction(2), Fraction(3)],
                [Fraction(0), Fraction(1), Fraction(4)],
                [Fraction(0), Fraction(0), Fraction(1)],
            ]),
            3,
        )
        self.assertEqual(
            rank([
                [Fraction(1), Fraction(2), Fraction(3)],
                [Fraction(2), Fraction(4), Fraction(6)],
                [Fraction(0), Fraction(1), Fraction(1)],
            ]),
            2,
        )
        self.assertEqual(
            rank([[Fraction(0), Fraction(0)], [Fraction(0), Fraction(0)]]),
            0,
        )

    def test_exact_linear_solver_uses_column_rhs_convention(self):
        self.assertEqual(
            solve(
                [[Fraction(2), Fraction(0)], [Fraction(0), Fraction(3)]],
                [[Fraction(4), Fraction(9)], [Fraction(6), Fraction(12)]],
            ),
            [[Fraction(2), Fraction(3)], [Fraction(3), Fraction(4)]],
        )

    def test_degree_nine_moeller_rank_gives_exact_node_lower_bound(self):
        # For degree 9, k = 4, dim(P_4) = 15 and rank(M_4) = 4.
        # Moeller's bound is therefore N >= 15 + 4/2 = 17.
        k = 4
        dim_pk = (k + 1) * (k + 2) // 2
        r = moeller_rank(k)
        self.assertEqual(dim_pk, 15)
        self.assertEqual(r, 4)
        self.assertEqual(dim_pk + r // 2, 17)
        self.assertEqual(r % 2, 0)


if __name__ == "__main__":
    unittest.main()

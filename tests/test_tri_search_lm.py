"""Regression tests for the adaptive primal/dual triangle LM solver."""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
from tri_search import lm_bounded


class AdaptiveLMSolverTests(unittest.TestCase):
    def solve_linear(self, matrix, target, lower, upper):
        def fun_jac(x):
            return matrix @ x - target, matrix

        start = np.zeros(matrix.shape[1], dtype=float)
        return lm_bounded(fun_jac, start, lower, upper, max_nfev=300)

    def test_wide_system_uses_dual_sized_problem_and_solves(self):
        rng = np.random.default_rng(27201)
        matrix = rng.normal(size=(5, 14))
        expected = rng.uniform(-0.5, 0.5, size=14)
        target = matrix @ expected
        lower = np.full(14, -2.0)
        upper = np.full(14, 2.0)

        solution, residual = self.solve_linear(matrix, target, lower, upper)

        self.assertLess(residual, 1e-11)
        np.testing.assert_allclose(matrix @ solution, target, atol=1e-11)

    def test_tall_system_uses_primal_sized_problem_and_solves(self):
        rng = np.random.default_rng(27202)
        matrix = rng.normal(size=(14, 5))
        expected = rng.uniform(-0.5, 0.5, size=5)
        target = matrix @ expected
        lower = np.full(5, -2.0)
        upper = np.full(5, 2.0)

        solution, residual = self.solve_linear(matrix, target, lower, upper)

        self.assertLess(residual, 1e-11)
        np.testing.assert_allclose(matrix @ solution, target, atol=1e-11)

    def test_box_bounds_are_respected_at_both_sides(self):
        matrix = np.eye(2)
        target = np.array([-1.0, 1.5])
        lower = np.zeros(2)
        upper = np.ones(2)

        solution, _ = self.solve_linear(matrix, target, lower, upper)

        np.testing.assert_allclose(solution, [0.0, 1.0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()

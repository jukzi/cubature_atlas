"""Regression tests for the geodesic-accelerated square C4 LM solver."""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
import square_cheb_c4 as sq


class SquareChebC4Tests(unittest.TestCase):
    def test_legendre_eval2_matches_values_and_second_derivative(self):
        x = np.array([-0.73, -0.19, 0.0, 0.41, 0.87])
        P_val = sq.legendre_values(15, x)
        P, dP, ddP = sq.legendre_eval2(15, x)
        np.testing.assert_allclose(P_val, P, atol=1e-14, rtol=1e-14)
        h = 1e-6
        _, dP_plus = sq.legendre_eval(15, x + h)
        _, dP_minus = sq.legendre_eval(15, x - h)
        ddP_fd = (dP_plus - dP_minus) / (2.0 * h)
        np.testing.assert_allclose(ddP, ddP_fd, atol=1e-7, rtol=1e-7)

    def test_solve_c4_lm_converges_with_and_without_geodesic_acceleration(self):
        B = sq.SquareC4Basis(15)
        x0, y0, w0, wc0 = sq.rotated_cheb_start(15, 12, False, 49.0, 0.6)
        _, res_geo, it_geo, xg, yg, wg, wcg = sq.solve_c4_lm(
            B, x0, y0, w0, False, wc0, 500, 35, 1e-2, geo=True
        )
        _, res_plain, _, xp, yp, wp, wcp = sq.solve_c4_lm(
            B, x0, y0, w0, False, wc0, 500, 35, 1e-2, geo=False
        )
        self.assertLess(res_geo, 1e-13)
        self.assertLess(res_plain, 1e-13)
        self.assertLess(it_geo, 200)
        nodes, weights = sq.expand_c4(xg, yg, wg, False, wcg)
        self.assertLess(sq.monomial_residual(15, nodes, weights), 1e-14)
        self.assertGreater(float(weights.min()), 0.0)
        self.assertLess(float(np.abs(nodes).max()), 1.0)

    def test_solve_c4_lm_with_centre_node(self):
        B = sq.SquareC4Basis(13)
        x0, y0, w0, wc0 = sq.rotated_cheb_start(13, 9, True, 49.0, 0.6)
        _, res, _, x, y, w, wc = sq.solve_c4_lm(
            B, x0, y0, w0, True, wc0, 500, 35, 1e-2, geo=True, geo_tau=0.0
        )
        self.assertLess(res, 1e-13)
        nodes, weights = sq.expand_c4(x, y, w, True, wc)
        self.assertEqual(len(nodes), 37)
        self.assertLess(sq.monomial_residual(13, nodes, weights), 1e-14)
        self.assertGreater(float(weights.min()), 0.0)


if __name__ == "__main__":
    unittest.main()

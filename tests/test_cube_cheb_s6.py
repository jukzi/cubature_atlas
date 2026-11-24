"""Regression tests for the geodesic-accelerated cube S6 LM solver."""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
import cube_cheb_s6 as cb


class CubeChebS6Tests(unittest.TestCase):
    def test_legendre_eval2_matches_values_and_second_derivative(self):
        x = np.array([-0.73, -0.19, 0.0, 0.41, 0.87])
        P_val = cb.legendre_values(15, x)
        P, dP, ddP = cb.legendre_eval2(15, x)
        np.testing.assert_allclose(P_val, P, atol=1e-14, rtol=1e-14)
        h = 1e-6
        _, dP_plus = cb.legendre_eval(15, x + h)
        _, dP_minus = cb.legendre_eval(15, x - h)
        ddP_fd = (dP_plus - dP_minus) / (2.0 * h)
        np.testing.assert_allclose(ddP, ddP_fd, atol=1e-7, rtol=1e-7)

    def test_diag_jet_matches_finite_differences(self):
        B = cb.CubeS6Basis(11)
        t = np.array([0.17, 0.53, 0.82])
        Vd, Vdt, Vdtt = B.eval_diag_jet(t)
        h = 1e-6
        Vp, Vdt_p = B.eval_diag_grad(t + h)
        Vm, Vdt_m = B.eval_diag_grad(t - h)
        Vdt_fd = (Vp - Vm) / (2.0 * h)
        Vdtt_fd = (Vdt_p - Vdt_m) / (2.0 * h)
        np.testing.assert_allclose(Vdt, Vdt_fd, atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(Vdtt, Vdtt_fd, atol=1e-6, rtol=1e-6)

    def test_bcc_cheb_start_exact_node_counts_all_mod6(self):
        for target_n in range(186, 192):
            xg, yg, zg, wg, td, wd, hc, wc = cb.bcc_cheb_start(15, target_n, 4.0, 0.55)
            nodes, weights = cb.expand_s6(xg, yg, zg, wg, td, wd, hc, wc)
            self.assertEqual(len(nodes), target_n)
            self.assertEqual(len(weights), target_n)
            self.assertAlmostEqual(float(weights.sum()), 8.0, places=12)
            self.assertGreater(float(weights.min()), 0.0)
            self.assertLess(float(np.abs(nodes).max()), 1.0)

    def test_solve_s6_lm_without_centre_geo_and_plain(self):
        B = cb.CubeS6Basis(17)
        xg0, yg0, zg0, wg0, td0, wd0, hc, wc0 = cb.bcc_cheb_start(17, 262, 8.0, 0.30)
        self.assertFalse(hc)
        _, res_geo, it_geo, xg, yg, zg, wg, td, wd, wc = cb.solve_s6_lm(
            B, xg0, yg0, zg0, wg0, td0, wd0, hc, wc0, 200, 35, 1e-2, geo=True, geo_tau=0.0
        )
        _, res_plain, _, xp, yp, zp, wp, tp, wdp, wcp = cb.solve_s6_lm(
            B, xg0, yg0, zg0, wg0, td0, wd0, hc, wc0, 200, 35, 1e-2, geo=False
        )
        self.assertLess(res_geo, 1e-13)
        self.assertLess(res_plain, 1e-13)
        self.assertLess(it_geo, 100)
        nodes, weights = cb.expand_s6(xg, yg, zg, wg, td, wd, hc, wc)
        self.assertEqual(len(nodes), 262)
        self.assertLess(cb.monomial_residual(17, nodes, weights), 1e-14)
        self.assertGreater(float(weights.min()), 0.0)
        self.assertLess(float(np.abs(nodes).max()), 1.0)

    def test_solve_s6_lm_with_centre_node(self):
        B = cb.CubeS6Basis(15)
        xg0, yg0, zg0, wg0, td0, wd0, hc, wc0 = cb.bcc_cheb_start(15, 189, 4.0, 0.55)
        self.assertTrue(hc)
        _, res, _, xg, yg, zg, wg, td, wd, wc = cb.solve_s6_lm(
            B, xg0, yg0, zg0, wg0, td0, wd0, hc, wc0, 200, 35, 1e-2, geo=True, geo_tau=0.0
        )
        self.assertLess(res, 1e-13)
        nodes, weights = cb.expand_s6(xg, yg, zg, wg, td, wd, hc, wc)
        self.assertEqual(len(nodes), 189)
        self.assertLess(cb.monomial_residual(15, nodes, weights), 1e-14)
        self.assertGreater(float(weights.min()), 0.0)
        self.assertLess(float(np.abs(nodes).max()), 1.0)

    def test_solve_s6_plain_and_hinge_pull(self):
        B = cb.CubeS6Basis(17)
        xg0, yg0, zg0, wg0, td0, wd0, hc, wc0 = cb.bcc_cheb_start(17, 262, 8.0, 0.30)
        _, res1, _, xg, yg, zg, wg, td, wd, wc = cb.solve_s6_plain(
            B, xg0, yg0, zg0, wg0, td0, wd0, hc, wc0, 120, 35, 0.5, geo=True, geo_tau=0.05
        )
        self.assertLess(res1, 1e-13)
        _, res2, _, xg, yg, zg, wg, td, wd, wc = cb.solve_s6_plain(
            B, xg, yg, zg, wg, td, wd, hc, wc, 80, 35, 1e-3, geo=True, geo_tau=0.0,
            pull_b=0.995, pull_wt=1.0
        )
        self.assertLess(res2, 1e-12)
        nodes, weights = cb.expand_s6(xg, yg, zg, wg, td, wd, hc, wc)
        self.assertEqual(len(nodes), 262)
        self.assertLess(cb.monomial_residual(17, nodes, weights), 1e-13)
        self.assertGreater(float(weights.min()), 0.0)
        self.assertLessEqual(float(np.abs(nodes).max()), 0.995 + 1e-10)


if __name__ == "__main__":
    unittest.main()

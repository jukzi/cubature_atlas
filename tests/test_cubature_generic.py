import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
import cubature_generic as cg


class GenericCubatureTests(unittest.TestCase):
    def test_basis_derivatives_and_jets_1d_to_4d(self):
        rng = np.random.default_rng(123)
        for domain in ("cube", "simplex"):
            for dim in (1, 2, 3, 4):
                deg = 4 if dim <= 3 else 3
                B = cg.CubeBasis(dim, deg) if domain == "cube" else cg.SimplexBasis(dim, deg)
                if domain == "cube":
                    X = rng.uniform(-0.7, 0.7, (4, dim))
                else:
                    X = rng.uniform(0.04, 0.18, (4, dim))
                dX = rng.normal(0.0, 1.0, (4, dim))
                V, G = B.eval_val_grad(X)
                F0, F1, F2 = B.eval_jet2(X, dX)
                h = 1e-4
                Vp = B.eval_val(X + h * dX)
                Vm = B.eval_val(X - h * dX)
                fd1 = (Vp - Vm) / (2.0 * h)
                fd2 = (Vp - 2.0 * V + Vm) / (h * h)
                g_dir = sum(G[j] * dX[:, j][None, :] for j in range(dim))
                self.assertLess(float(np.abs(F0 - V).max()), 1e-13)
                self.assertLess(float(np.abs(g_dir - F1).max()), 1e-12)
                self.assertLess(
                    float(np.abs(F1 - fd1).max()) / max(1.0, float(np.abs(F1).max())),
                    1e-5,
                )
                self.assertLess(
                    float(np.abs(F2 - fd2).max()) / max(1.0, float(np.abs(F2).max())),
                    1e-4,
                )

    def test_simplex_orthonormality_1d_2d(self):
        # 1D Gauss-Legendre on [0, 1]
        x1, w1 = np.polynomial.legendre.leggauss(6)
        B1 = cg.SimplexBasis(1, 5)
        V1 = B1.eval_val(((x1 + 1.0) * 0.5)[:, None])
        G1 = (V1 * (w1 * 0.5)[None, :]) @ V1.T
        self.assertLess(float(np.abs(G1 - np.eye(B1.M)).max()), 1e-13)
        # 2D tensor Duffy product on Delta^2
        t, wt = np.polynomial.legendre.leggauss(6)
        u = 0.5 * (t + 1.0)
        wu = 0.5 * wt
        pts = []
        ws = []
        for a, wa in zip(u, wu):
            for b, wb in zip(u, wu):
                pts.append([a, (1.0 - a) * b])
                ws.append(wa * wb * (1.0 - a) * 2.0)
        B2 = cg.SimplexBasis(2, 4)
        V2 = B2.eval_val(np.asarray(pts))
        G2 = (V2 * np.asarray(ws)[None, :]) @ V2.T
        self.assertLess(float(np.abs(G2 - np.eye(B2.M)).max()), 1e-13)

    def test_cli_defaults_and_timestamp_logging(self):
        script = os.path.join(os.path.dirname(__file__), "..", "code", "cubature_generic.py")
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [sys.executable, script],
                cwd=tmp,
                capture_output=True,
                text=True,
                check=True,
            )
            lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
            self.assertGreater(len(lines), 0)
            ts_pat = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| ")
            for ln in lines:
                self.assertRegex(ln, ts_pat)
            out_file = os.path.join(tmp, "CA-T2-07-12-PI-MPC.json")
            self.assertTrue(os.path.exists(out_file))
            with open(out_file, encoding="utf-8") as fh:
                rec = json.load(fh)
            self.assertEqual(rec["domain"], "triangle")
            self.assertEqual(rec["degree"], 7)
            self.assertEqual(rec["points"], 12)
            self.assertEqual(rec["quality"], "PI")
            self.assertLess(rec["residual_reported"], 1e-13)

    def test_4d_cube_and_simplex_pi(self):
        for dom, deg, sym in (("simplex", 3, "chiral"), ("cube", 3, "chiral")):
            base = cg.CubeBasis(4, deg) if dom == "cube" else cg.SimplexBasis(4, deg)
            out = cg.solve_for_symmetry(base, sym, 0, seed=None, verbose=False)
            self.assertIsNotNone(out)
            res, _, _ = out
            self.assertEqual(res["quality"], "PI")
            merr = cg.monomial_residual(dom, 4, deg, res["nodes"], res["weights"] * base.vol)
            self.assertLess(merr, 1e-13)


if __name__ == "__main__":
    unittest.main()

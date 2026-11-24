import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
import cubature_generic


class GenericCubatureTests(unittest.TestCase):
    def test_basis_derivatives_and_jets_1d_to_4d(self):
        rng = np.random.default_rng(123)
        for domain in ("cube", "simplex"):
            for dimension in (1, 2, 3, 4):
                degree = 4 if dimension <= 3 else 3
                basis = (
                    cubature_generic.CubeBasis(dimension, degree)
                    if domain == "cube"
                    else cubature_generic.SimplexBasis(dimension, degree)
                )
                if domain == "cube":
                    points = rng.uniform(-0.7, 0.7, (4, dimension))
                else:
                    points = rng.uniform(0.04, 0.18, (4, dimension))
                directions = rng.normal(0.0, 1.0, (4, dimension))

                values, gradients = basis.evaluate_values_and_gradients(points)
                jet_val, jet_first, jet_second = basis.evaluate_second_order_jet(
                    points, directions
                )

                step_h = 1e-4
                values_plus = basis.evaluate_values(points + step_h * directions)
                values_minus = basis.evaluate_values(points - step_h * directions)
                fd_first = (values_plus - values_minus) / (2.0 * step_h)
                fd_second = (values_plus - 2.0 * values + values_minus) / (
                    step_h * step_h
                )
                grad_directional = sum(
                    gradients[axis] * directions[:, axis][None, :]
                    for axis in range(dimension)
                )

                self.assertLess(float(np.abs(jet_val - values).max()), 1e-13)
                self.assertLess(
                    float(np.abs(grad_directional - jet_first).max()), 1e-12
                )
                self.assertLess(
                    float(np.abs(jet_first - fd_first).max())
                    / max(1.0, float(np.abs(jet_first).max())),
                    1e-5,
                )
                self.assertLess(
                    float(np.abs(jet_second - fd_second).max())
                    / max(1.0, float(np.abs(jet_second).max())),
                    1e-4,
                )

    def test_simplex_orthonormality_1d_2d(self):
        # 1D Gauss-Legendre quadrature on [0, 1]
        gauss_x, gauss_w = np.polynomial.legendre.leggauss(6)
        basis_1d = cubature_generic.SimplexBasis(1, 5)
        values_1d = basis_1d.evaluate_values(((gauss_x + 1.0) * 0.5)[:, None])
        gram_1d = (values_1d * (gauss_w * 0.5)[None, :]) @ values_1d.T
        self.assertLess(
            float(np.abs(gram_1d - np.eye(basis_1d.num_basis)).max()), 1e-13
        )

        # 2D tensor Duffy product quadrature on Delta^2
        unit_nodes = 0.5 * (gauss_x + 1.0)
        unit_weights = 0.5 * gauss_w
        points_2d = []
        weights_2d = []
        for coord_a, weight_a in zip(unit_nodes, unit_weights):
            for coord_b, weight_b in zip(unit_nodes, unit_weights):
                points_2d.append([coord_a, (1.0 - coord_a) * coord_b])
                weights_2d.append(weight_a * weight_b * (1.0 - coord_a) * 2.0)
        basis_2d = cubature_generic.SimplexBasis(2, 4)
        values_2d = basis_2d.evaluate_values(np.asarray(points_2d))
        gram_2d = (values_2d * np.asarray(weights_2d)[None, :]) @ values_2d.T
        self.assertLess(
            float(np.abs(gram_2d - np.eye(basis_2d.num_basis)).max()), 1e-13
        )

    def test_cli_defaults_and_timestamp_logging(self):
        script_path = os.path.join(
            os.path.dirname(__file__), "..", "code", "cubature_generic.py"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            completed = subprocess.run(
                [sys.executable, script_path],
                cwd=tmp_dir,
                capture_output=True,
                text=True,
                check=True,
            )
            log_lines = [
                line for line in completed.stdout.splitlines() if line.strip()
            ]
            self.assertGreater(len(log_lines), 0)
            timestamp_pattern = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| ")
            for line in log_lines:
                self.assertRegex(line, timestamp_pattern)

            output_file = os.path.join(tmp_dir, "CA-T2-07-12-PI-MPC.json")
            self.assertTrue(os.path.exists(output_file))
            with open(output_file, encoding="utf-8") as file_handle:
                record = json.load(file_handle)
            self.assertEqual(record["domain"], "triangle")
            self.assertEqual(record["degree"], 7)
            self.assertEqual(record["points"], 12)
            self.assertEqual(record["quality"], "PI")
            self.assertLess(record["residual_reported"], 1e-13)

    def test_4d_cube_and_simplex_pi(self):
        for domain, degree, symmetry in (
            ("simplex", 3, "chiral"),
            ("cube", 3, "chiral"),
        ):
            base_basis = (
                cubature_generic.CubeBasis(4, degree)
                if domain == "cube"
                else cubature_generic.SimplexBasis(4, degree)
            )
            outcome = cubature_generic.solve_for_symmetry(
                base_basis, symmetry, 0, seed=None, verbose=False
            )
            self.assertIsNotNone(outcome)
            solution, _, _ = outcome
            self.assertEqual(solution.quality, "PI")
            monomial_err = cubature_generic.compute_monomial_residual(
                domain,
                4,
                degree,
                solution.nodes,
                solution.weights * base_basis.volume,
            )
            self.assertLess(monomial_err, 1e-13)

    def test_demi_and_simplex_central_symmetries(self):
        # 3D cube degree 10 with 71 nodes (< Z = 72) via demihypercube (Td) symmetry
        cube_basis = cubature_generic.CubeBasis(3, 10)
        cube_out = cubature_generic.solve_for_symmetry(
            cube_basis, "demi", 71, seed=None, verbose=False
        )
        self.assertIsNotNone(cube_out)
        cube_sol, _, _ = cube_out
        self.assertEqual(cube_sol.quality, "PI")
        self.assertEqual(len(cube_sol.weights), 71)
        self.assertLess(
            cubature_generic.compute_monomial_residual(
                "cube", 3, 10, cube_sol.nodes, cube_sol.weights * cube_basis.volume
            ),
            1e-13,
        )

        # 3D simplex degree 4 with 11 nodes via order-2 vertex-reversal (central) symmetry
        tetra_basis = cubature_generic.SimplexBasis(3, 4)
        tetra_out = cubature_generic.solve_for_symmetry(
            tetra_basis, "central", 11, seed=None, verbose=False
        )
        self.assertIsNotNone(tetra_out)
        tetra_sol, _, _ = tetra_out
        self.assertEqual(tetra_sol.quality, "PI")
        self.assertEqual(len(tetra_sol.weights), 11)
        self.assertLess(
            cubature_generic.compute_monomial_residual(
                "simplex",
                3,
                4,
                tetra_sol.nodes,
                tetra_sol.weights * tetra_basis.volume,
            ),
            1e-13,
        )


if __name__ == "__main__":
    unittest.main()

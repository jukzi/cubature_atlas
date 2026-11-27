import json
import math
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

    def test_stratum_test_rejects_starts_on_the_mirror_axis(self):
        # Triangle degree 35 with the mirror group: at N = 200 the start needs
        # many nodes on the axis to reach M_G unknowns, which cannot work.
        base_basis = cubature_generic.SimplexBasis(2, 35)
        group_transforms, sym_basis = cubature_generic.prepare_symmetry(
            base_basis, "central"
        )
        for num_points, expected in ((200, False), (228, True)):
            orbits = cubature_generic.build_simplex_equilibrium_start(
                2,
                35,
                num_points,
                "central",
                group_transforms,
                min_unknowns=sym_basis.num_basis,
            )
            self.assertIsNotNone(orbits)
            self.assertEqual(
                cubature_generic.stratum_unknowns_suffice(sym_basis, orbits),
                expected,
            )
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
            self.assertIn("seed", record["reproduce"])
            self.assertIsNone(record["reproduce"]["seed"])

    def test_cli_out_directory(self):
        script_path = os.path.join(
            os.path.dirname(__file__), "..", "code", "cubature_generic.py"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = os.path.join(tmp_dir, "out")
            os.makedirs(out_dir)
            subprocess.run(
                [
                    sys.executable,
                    script_path,
                    "--domain",
                    "cube",
                    "--degree",
                    "5",
                    "--out",
                    out_dir,
                ],
                cwd=tmp_dir,
                capture_output=True,
                text=True,
                check=True,
            )
            written = sorted(os.listdir(out_dir))
            # the rule and the file with the tried starts (starts-<prefix>-<degree>.json)
            self.assertEqual(len(written), 2)
            self.assertRegex(written[0], r"^CA-C2-05-\d+-PI-MPC\.json$")
            self.assertEqual(written[1], "starts-C2-05.json")
            self.assertEqual(os.listdir(tmp_dir), ["out"])

    def test_inside_tolerance_matches_atlas(self):
        # Boundary counts as inside (build_atlas.py uses 1e-12 slack).
        self.assertTrue(cubature_generic.is_inside(0.0))
        self.assertTrue(cubature_generic.is_inside(-5e-13))
        self.assertFalse(cubature_generic.is_inside(-1e-9))

    def test_collapsed_orbit_is_rejected(self):
        # A size-3 C3 orbit whose representative drifted to the centroid (1/3, 1/3)
        # expands to 1 node instead of 3 and must be rejected.
        base_basis = cubature_generic.SimplexBasis(2, 2)
        sym_basis = cubature_generic.SymmetricBasis(
            base_basis,
            "chiral",
            cubature_generic.build_symmetry_group("simplex", 2, "chiral"),
        )
        centroid = np.array([1.0 / 3.0, 1.0 / 3.0])
        collapsed_orbit = cubature_generic.SymmetryOrbit(
            size=3,
            subspace_origin=np.zeros(2),
            subspace_basis=np.eye(2),
            subspace_coords=centroid,
            weight_sqrt=math.sqrt(1.0 / 3.0),
        )
        params = cubature_generic.pack_orbit_parameters([collapsed_orbit])
        self.assertIsNone(
            cubature_generic._build_solution_if_positive(
                sym_basis, [collapsed_orbit], params, 0.0, 1.0 / 3.0, "PI"
            )
        )

    def test_save_rejects_high_residual(self):
        bad_solution = cubature_generic.CubatureSolution(
            quality="PI",
            moment_residual=1e-3,
            boundary_distance=0.2,
            nodes=np.array([[0.2, 0.2]]),
            weights=np.array([1.0]),
            orbits=[],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path, rule_id, monomial_err = cubature_generic.save_atlas_rule_json(
                tmp_dir,
                "simplex",
                2,
                3,
                bad_solution,
                "none",
                "python code/cubature_generic.py",
                seed=None,
            )
            self.assertIsNone(output_path)
            self.assertEqual(rule_id, "CA-T2-03-1-PI-MPC")
            self.assertGreater(monomial_err, cubature_generic.RES_TOL)
            self.assertEqual(os.listdir(tmp_dir), [])

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
            solution, found_symmetry, _, _ = (
                cubature_generic.solve_point_count_ladder(
                    base_basis, [symmetry], None, None, start_rounds=1
                )
            )
            self.assertIsNotNone(solution)
            self.assertEqual(found_symmetry, symmetry)
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

    def test_sparse_symmetric_basis_invariance_and_orthonormality(self):
        rng = np.random.default_rng(77)
        for domain, dim, deg, syms in (
            ("cube", 2, 6, ("central", "mirror", "chiral", "demi", "full")),
            ("cube", 3, 5, ("central", "mirror", "chiral", "demi", "full")),
            ("simplex", 2, 6, ("central", "mirror", "chiral", "full")),
            ("simplex", 3, 5, ("central", "mirror", "chiral", "full")),
        ):
            base = (
                cubature_generic.CubeBasis(dim, deg)
                if domain == "cube"
                else cubature_generic.SimplexBasis(dim, deg)
            )
            pts = (
                rng.uniform(-0.8, 0.8, (5, dim))
                if domain == "cube"
                else rng.uniform(0.05, 0.2, (5, dim))
            )
            for sym in syms:
                group = cubature_generic.build_symmetry_group(domain, dim, sym)
                sb = cubature_generic.SymmetricBasis(base, sym, group)
                gram = sb.projection_matrix @ sb.projection_matrix.T
                self.assertLess(
                    float(np.abs(gram - np.eye(sb.num_basis)).max()), 1e-12
                )
                vals_0 = sb.evaluate_values(pts)
                for transform in group:
                    t_pts = pts @ transform.linear.T + transform.translation[None, :]
                    vals_g = sb.evaluate_values(t_pts)
                    self.assertLess(float(np.abs(vals_g - vals_0).max()), 1e-12)

    def test_mirror_symmetry_and_riesz_crystals(self):
        # 2D square degree 8 with 16 nodes via hyperplane mirror symmetry x_0 <-> x_1
        sq_basis = cubature_generic.CubeBasis(2, 8)
        sq_out = cubature_generic.solve_for_symmetry(
            sq_basis, "mirror", 16, seed=None, verbose=False
        )
        self.assertIsNotNone(sq_out)
        sq_sol, _, _ = sq_out
        self.assertEqual(sq_sol.quality, "PI")
        self.assertEqual(len(sq_sol.weights), 16)
        self.assertLess(
            cubature_generic.compute_monomial_residual(
                "cube", 2, 8, sq_sol.nodes, sq_sol.weights * sq_basis.volume
            ),
            1e-13,
        )

        # 2D triangle degree 13 with 35 nodes via central/mirror + G-equivariant Riesz
        tri_basis = cubature_generic.SimplexBasis(2, 13)
        tri_out = cubature_generic.solve_for_symmetry(
            tri_basis, "central", 35, seed=None, verbose=False
        )
        self.assertIsNotNone(tri_out)
        tri_sol, _, _ = tri_out
        self.assertEqual(tri_sol.quality, "PI")
        self.assertEqual(len(tri_sol.weights), 35)
        self.assertLess(
            cubature_generic.compute_monomial_residual(
                "simplex", 2, 13, tri_sol.nodes, tri_sol.weights * tri_basis.volume
            ),
            1e-13,
        )

    def test_select_orbits_weights_rank_by_orbit_size(self):
        # 3 inner size-2 orbits (indices 0, 1, 2 -> total cost 2*(0+1+2)=6) must
        # beat 1 outer size-6 orbit (index 5 -> cost 6*5=30) for target_points=6.
        def make_orbit(size, dof):
            return cubature_generic.SymmetryOrbit(
                size=size,
                subspace_origin=np.zeros(3),
                subspace_basis=np.eye(3)[:, :dof],
                subspace_coords=np.zeros(dof),
                weight_sqrt=1.0,
            )

        candidates = [
            make_orbit(2, 1),
            make_orbit(2, 1),
            make_orbit(2, 1),
            make_orbit(4, 2),
            make_orbit(4, 2),
            make_orbit(6, 3),
        ]
        selected = cubature_generic.select_orbits_for_point_count(
            candidates, target_points=6, min_unknowns=4
        )
        self.assertIsNotNone(selected)
        self.assertEqual([orb.size for orb in selected], [2, 2, 2])

    def test_3d_simplex_chiral_start_exact_count_and_distinct_orbits(self):
        group = cubature_generic.build_symmetry_group("simplex", 3, "chiral")
        orbits = cubature_generic.build_simplex_equilibrium_start(
            dimension=3,
            degree=14,
            target_points=196,
            symmetry_name="chiral",
            group_transforms=group,
            min_unknowns=60,
        )
        self.assertIsNotNone(orbits)
        self.assertEqual(sum(orb.size for orb in orbits), 196)
        nodes, _ = cubature_generic.expand_orbits_to_nodes(orbits, group)
        self.assertEqual(len(nodes), 196)


if __name__ == "__main__":
    unittest.main()

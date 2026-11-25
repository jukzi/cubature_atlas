"""Generic deterministic high-order cubature solver for d-cubes and d-simplices.

Unifies the mathematical and numerical techniques of the atlas solvers into a
single dimension-independent implementation without configuration-specific
special cases:

- Exact orthonormal polynomial bases for any dimension d >= 1:
  * Cube [-1, 1]^d: tensor-product Legendre basis with 3-term recurrence.
  * Simplex Delta^d: division-free homogeneous Proriol-Koornwinder-Dubiner
    basis in barycentric coordinates (works identically inside and outside).
- Exact analytic 1st derivatives and exact directional 2nd derivatives via
  2nd-order Taylor-jet arithmetic (O(num_basis * num_points) cost, zero
  finite-difference loss).
- Generic finite symmetry engine (exact combinatorial Reynolds projection for
  cubes, degree-block Reynolds+SVD for simplices, sparse invariant projection,
  and stabilizer fixed-subspace orbit parameterization) supporting none,
  central, mirror, chiral, demi, and full symmetry groups in arbitrary
  dimension d.
- Deterministic equilibrium starts (rotated D_d* lattice / G-equivariant Riesz
  crystals in Chebyshev angle space for cubes; concentric geodesic S^d_+
  shells / G-equivariant Riesz-S^d_+ crystals for simplices) with exact
  Pareto-pruned min-rank DP orbit selection for exact target point count N.
- Fold-free (x, sqrt(w)^2) geodesic-accelerated Cholesky Levenberg-Marquardt
  (Transtrum & Sethna 2012) with a PI-preferred / PO-fallback pipeline:
  Stage A smooth sliding-wall solve -> Stage B free solve -> exact face-normal
  boundary hinge pull -> interior pure-moment polish + automatic minimum-norm
  Gauss-Newton end-stage. Logging and stopping criteria strictly report the
  pure moment residual (excluding the boundary penalty).
- Quality and exactness follow tools/build_atlas.py (PI: positive weights and
  no node more than 1e-12 outside; a rule is written only if its monomial
  residual is at most 1e-11).
"""

import argparse
from dataclasses import dataclass
import itertools
import json
import math
import os
import shlex
import sys
import time

# Small-to-medium dense Cholesky solves (M_G <= 500) run >2x faster and more
# reproducibly without BLAS thread-pool spin-up overhead.
for _blas_env_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_blas_env_var, "1")

import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.special import betaincinv

# Same acceptance thresholds as tools/build_atlas.py: a node counts as inside if
# it lies at most INSIDE_TOL outside the domain (the boundary counts as inside),
# and a rule is exact if its max monomial error / volume is at most RES_TOL.
INSIDE_TOL = 1e-12
RES_TOL = 1e-11


def is_inside(boundary_distance):
    """Atlas inside test: True if no node is more than INSIDE_TOL outside the domain."""
    return boundary_distance >= -INSIDE_TOL


# ---------------------------------------------------------------------------
# Logging and deterministic combinatorics helpers
# ---------------------------------------------------------------------------


def log_message(message):
    """Print a timestamped log line with second-level resolution."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} | {message}", flush=True)


def generate_multi_indices(dimension, max_degree):
    """Return all d-tuples of non-negative integers with sum <= max_degree."""
    indices = []

    def append_recursive(prefix, remaining_dims, remaining_degree):
        if remaining_dims == 1:
            for exponent in range(remaining_degree + 1):
                indices.append(prefix + (exponent,))
            return
        for exponent in range(remaining_degree + 1):
            append_recursive(
                prefix + (exponent,),
                remaining_dims - 1,
                remaining_degree - exponent,
            )

    append_recursive((), dimension, max_degree)
    return indices


def generate_halton_points(num_points, dimension, skip=0):
    """Return deterministic Halton low-discrepancy points in (0, 1)^dimension."""
    primes = []
    candidate = 2
    while len(primes) < dimension:
        is_prime = True
        for prime in primes:
            if prime * prime > candidate:
                break
            if candidate % prime == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1

    points = np.zeros((num_points, dimension), dtype=float)
    for axis, base in enumerate(primes):
        for point_index in range(num_points):
            index = point_index + skip + 1
            radix_fraction = 1.0
            radical_inverse = 0.0
            while index > 0:
                radix_fraction /= base
                radical_inverse += radix_fraction * (index % base)
                index //= base
            points[point_index, axis] = radical_inverse
    return points


# ---------------------------------------------------------------------------
# Orthonormal polynomial bases with exact 1st and directional 2nd derivatives
# ---------------------------------------------------------------------------


class CubeBasis:
    """Orthonormal tensor-Legendre basis on [-1, 1]^d (probability measure)."""

    def __init__(self, dimension, degree):
        self.domain = "cube"
        self.dimension = dimension
        self.degree = degree
        self.volume = float(2.0 ** dimension)
        self.multi_indices = generate_multi_indices(dimension, degree)
        self.num_basis = len(self.multi_indices)
        self.multi_index_array = np.asarray(self.multi_indices, dtype=np.int32)
        self.legendre_norms_1d = np.sqrt(
            2.0 * np.arange(degree + 1, dtype=float) + 1.0
        )
        self.exact_moments = np.zeros(self.num_basis, dtype=float)
        self.exact_moments[0] = 1.0

    def _legendre_1d_values(self, coords):
        """Evaluate orthonormal 1D Legendre polynomials P_0..P_degree at coords."""
        max_deg = self.degree
        values = np.empty((max_deg + 1, len(coords)), dtype=float)
        values[0] = 1.0
        if max_deg >= 1:
            values[1] = coords
        for deg in range(2, max_deg + 1):
            values[deg] = (
                (2 * deg - 1) * coords * values[deg - 1]
                - (deg - 1) * values[deg - 2]
            ) / deg
        return values * self.legendre_norms_1d[:, None]

    def _legendre_1d_values_and_derivatives(self, coords):
        """Evaluate orthonormal 1D Legendre polynomials and their 1st derivatives."""
        max_deg = self.degree
        values = np.empty((max_deg + 1, len(coords)), dtype=float)
        derivatives = np.empty((max_deg + 1, len(coords)), dtype=float)
        values[0] = 1.0
        derivatives[0] = 0.0
        if max_deg >= 1:
            values[1] = coords
            derivatives[1] = 1.0
        for deg in range(2, max_deg + 1):
            values[deg] = (
                (2 * deg - 1) * coords * values[deg - 1]
                - (deg - 1) * values[deg - 2]
            ) / deg
            derivatives[deg] = (
                (2 * deg - 1) * (values[deg - 1] + coords * derivatives[deg - 1])
                - (deg - 1) * derivatives[deg - 2]
            ) / deg
        scale = self.legendre_norms_1d[:, None]
        return values * scale, derivatives * scale

    def _legendre_1d_second_order_jet(self, coords, coord_directions):
        """Evaluate 2nd-order Taylor jet (f, f', f'') of 1D Legendre polynomials."""
        max_deg = self.degree
        values = np.empty((max_deg + 1, len(coords)), dtype=float)
        first_deriv = np.empty((max_deg + 1, len(coords)), dtype=float)
        second_deriv = np.empty((max_deg + 1, len(coords)), dtype=float)
        values[0] = 1.0
        first_deriv[0] = 0.0
        second_deriv[0] = 0.0
        if max_deg >= 1:
            values[1] = coords
            first_deriv[1] = coord_directions
            second_deriv[1] = 0.0
        for deg in range(2, max_deg + 1):
            coeff_curr = (2 * deg - 1) / deg
            coeff_prev = (deg - 1) / deg
            values[deg] = (
                coeff_curr * coords * values[deg - 1]
                - coeff_prev * values[deg - 2]
            )
            first_deriv[deg] = (
                coeff_curr
                * (coord_directions * values[deg - 1] + coords * first_deriv[deg - 1])
                - coeff_prev * first_deriv[deg - 2]
            )
            second_deriv[deg] = (
                coeff_curr
                * (
                    2.0 * coord_directions * first_deriv[deg - 1]
                    + coords * second_deriv[deg - 1]
                )
                - coeff_prev * second_deriv[deg - 2]
            )
        scale = self.legendre_norms_1d[:, None]
        return values * scale, first_deriv * scale, second_deriv * scale

    def evaluate_values(self, points, active_indices=None):
        """Evaluate all (or active) basis polynomials at points, returning (num_sel, num_points)."""
        num_points = len(points)
        idx_arr = (
            self.multi_index_array
            if active_indices is None
            else self.multi_index_array[active_indices]
        )
        tables_by_axis = [
            self._legendre_1d_values(points[:, axis])
            for axis in range(self.dimension)
        ]
        values = np.ones((len(idx_arr), num_points), dtype=float)
        for axis in range(self.dimension):
            values *= tables_by_axis[axis][idx_arr[:, axis]]
        return values

    def evaluate_values_and_gradients(self, points, active_indices=None):
        """Evaluate basis values and coordinate gradients at points."""
        num_points = len(points)
        idx_arr = (
            self.multi_index_array
            if active_indices is None
            else self.multi_index_array[active_indices]
        )
        num_selected = len(idx_arr)
        factors_by_axis = []
        derivs_by_axis = []
        for axis in range(self.dimension):
            axis_vals, axis_derivs = self._legendre_1d_values_and_derivatives(
                points[:, axis]
            )
            exponents = idx_arr[:, axis]
            factors_by_axis.append(axis_vals[exponents])
            derivs_by_axis.append(axis_derivs[exponents])

        values = np.ones((num_selected, num_points), dtype=float)
        for axis in range(self.dimension):
            values *= factors_by_axis[axis]

        gradients = np.empty(
            (self.dimension, num_selected, num_points), dtype=float
        )
        for axis in range(self.dimension):
            axis_grad = derivs_by_axis[axis].copy()
            for other_axis in range(self.dimension):
                if other_axis != axis:
                    axis_grad *= factors_by_axis[other_axis]
            gradients[axis] = axis_grad
        return values, gradients

    def evaluate_second_order_jet(self, points, directions, active_indices=None):
        """Evaluate (value, 1st directional deriv, 2nd directional deriv) along directions."""
        num_points = len(points)
        idx_arr = (
            self.multi_index_array
            if active_indices is None
            else self.multi_index_array[active_indices]
        )
        num_selected = len(idx_arr)
        prod_val = np.ones((num_selected, num_points), dtype=float)
        prod_first = np.zeros((num_selected, num_points), dtype=float)
        prod_second = np.zeros((num_selected, num_points), dtype=float)
        for axis in range(self.dimension):
            axis_val, axis_first, axis_second = self._legendre_1d_second_order_jet(
                points[:, axis], directions[:, axis]
            )
            exponents = idx_arr[:, axis]
            factor_val = axis_val[exponents]
            factor_first = axis_first[exponents]
            factor_second = axis_second[exponents]
            # Leibniz product rule for 2nd-order Taylor jets
            next_val = prod_val * factor_val
            next_first = prod_val * factor_first + prod_first * factor_val
            next_second = (
                prod_val * factor_second
                + 2.0 * prod_first * factor_first
                + prod_second * factor_val
            )
            prod_val = next_val
            prod_first = next_first
            prod_second = next_second
        return prod_val, prod_first, prod_second

    def boundary_distance(self, points):
        """Minimum distance to boundary (> 0 strictly interior, <= 0 on/outside)."""
        return float(1.0 - np.abs(points).max())

    def boundary_hinge_penalty(self, points, target_margin):
        """Return (violations, gradients, exact_hessians) for |x_j| > 1 - target_margin."""
        inner_limit = 1.0 - target_margin
        abs_coords = np.abs(points)
        active = abs_coords > inner_limit
        violations = np.maximum(0.0, abs_coords - inner_limit)
        gradients = np.where(active, np.sign(points) * violations, 0.0)
        hessians = np.zeros((len(points), self.dimension, self.dimension), dtype=float)
        diag_axes = np.arange(self.dimension)
        hessians[:, diag_axes, diag_axes] = active.astype(float)
        return violations, gradients, hessians


class SimplexBasis:
    """Orthonormal division-free homogeneous Dubiner basis on Delta^d (prob. measure)."""

    def __init__(self, dimension, degree):
        self.domain = "simplex"
        self.dimension = dimension
        self.degree = degree
        self.volume = 1.0 / float(math.factorial(dimension))
        self.multi_indices = generate_multi_indices(dimension, degree)
        self.num_basis = len(self.multi_indices)
        self.multi_index_array = np.asarray(self.multi_indices, dtype=np.int32)

        # Precompute Jacobi parameter alpha_m = 2 * sum_{r > m} k_r + (d - 1 - m)
        self.jacobi_alpha_params = np.zeros(
            (self.num_basis, dimension), dtype=np.int32
        )
        self.normalization_factors = np.ones(self.num_basis, dtype=float)
        dim_factorial = float(math.factorial(dimension))
        for basis_idx, exponents in enumerate(self.multi_indices):
            trailing_degree_sum = 0
            norm_product = 1.0
            for axis in range(dimension - 1, -1, -1):
                alpha_param = 2 * trailing_degree_sum + (dimension - 1 - axis)
                self.jacobi_alpha_params[basis_idx, axis] = alpha_param
                norm_product *= float(2 * exponents[axis] + alpha_param + 1)
                trailing_degree_sum += exponents[axis]
            self.normalization_factors[basis_idx] = math.sqrt(
                norm_product / dim_factorial
            )

        # Group unique (axis, alpha_param) pairs and precompute vectorized recurrence tables
        self.unique_alphas_by_axis = []
        self.alpha_slot_by_axis = []
        self.jacobi_tables_by_axis = []
        for axis in range(dimension):
            unique_alphas = sorted(
                set(int(val) for val in self.jacobi_alpha_params[:, axis])
            )
            self.unique_alphas_by_axis.append(unique_alphas)
            alpha_to_slot = {
                alpha_val: slot for slot, alpha_val in enumerate(unique_alphas)
            }
            self.alpha_slot_by_axis.append(
                np.array(
                    [
                        alpha_to_slot[int(val)]
                        for val in self.jacobi_alpha_params[:, axis]
                    ],
                    dtype=np.int32,
                )
            )
            alpha_vec = np.asarray(unique_alphas, dtype=float)
            coeff_shifted = np.zeros((degree + 1, len(unique_alphas)), dtype=float)
            coeff_tail = np.zeros((degree + 1, len(unique_alphas)), dtype=float)
            coeff_prev = np.zeros((degree + 1, len(unique_alphas)), dtype=float)
            for deg in range(2, degree + 1):
                k_val = float(deg)
                denom = 2.0 * k_val * (k_val + alpha_vec) * (2.0 * k_val + alpha_vec - 2.0)
                coeff_shifted[deg] = (
                    (2.0 * k_val + alpha_vec - 1.0)
                    * (2.0 * k_val + alpha_vec)
                    * (2.0 * k_val + alpha_vec - 2.0)
                    / denom
                )
                coeff_tail[deg] = (
                    (2.0 * k_val + alpha_vec - 1.0) * (alpha_vec * alpha_vec) / denom
                )
                coeff_prev[deg] = (
                    2.0
                    * (k_val + alpha_vec - 1.0)
                    * (k_val - 1.0)
                    * (2.0 * k_val + alpha_vec)
                    / denom
                )
            self.jacobi_tables_by_axis.append(
                (alpha_vec, coeff_shifted, coeff_tail, coeff_prev)
            )

        self.exact_moments = np.zeros(self.num_basis, dtype=float)
        self.exact_moments[0] = 1.0

    def _barycentric_and_tail_sums(self, points):
        """Compute barycentric coordinates u_0..u_d and suffix sums s_m = sum_{r>=m} u_r."""
        num_points = len(points)
        barycentric = np.empty((self.dimension + 1, num_points), dtype=float)
        barycentric[0] = 1.0 - np.sum(points, axis=1)
        for axis in range(self.dimension):
            barycentric[axis + 1] = points[:, axis]

        tail_sums = np.empty((self.dimension + 1, num_points), dtype=float)
        tail_sums[self.dimension] = barycentric[self.dimension]
        for axis in range(self.dimension - 1, -1, -1):
            tail_sums[axis] = barycentric[axis] + tail_sums[axis + 1]
        return barycentric, tail_sums

    def _axis_jacobi_values(self, axis, coord, tail_sum):
        """Vectorized evaluation of homogeneous Jacobi polynomials for all alphas on axis."""
        alpha_vec, coeff_shifted, coeff_tail, coeff_prev = (
            self.jacobi_tables_by_axis[axis]
        )
        max_deg = self.degree
        num_alphas = len(alpha_vec)
        num_points = len(coord)
        values = np.empty((max_deg + 1, num_alphas, num_points), dtype=float)
        values[0] = 1.0
        if max_deg >= 1:
            values[1] = (alpha_vec[:, None] + 2.0) * coord[None, :] - tail_sum[None, :]

        tail_squared = (tail_sum * tail_sum)[None, :]
        shifted_coord = (2.0 * coord - tail_sum)[None, :]
        tail_row = tail_sum[None, :]
        for deg in range(2, max_deg + 1):
            linear_factor = (
                coeff_shifted[deg][:, None] * shifted_coord
                + coeff_tail[deg][:, None] * tail_row
            )
            values[deg] = (
                linear_factor * values[deg - 1]
                - coeff_prev[deg][:, None] * tail_squared * values[deg - 2]
            )
        return values

    def _axis_jacobi_values_and_partials(self, axis, coord, tail_sum):
        """Single-pass evaluation of (value, d/dcoord, d/dtail_sum) for all alphas on axis."""
        alpha_vec, coeff_shifted, coeff_tail, coeff_prev = (
            self.jacobi_tables_by_axis[axis]
        )
        max_deg = self.degree
        num_alphas = len(alpha_vec)
        num_points = len(coord)
        values = np.empty((max_deg + 1, num_alphas, num_points), dtype=float)
        dcoord = np.empty((max_deg + 1, num_alphas, num_points), dtype=float)
        dtail = np.empty((max_deg + 1, num_alphas, num_points), dtype=float)

        values[0] = 1.0
        dcoord[0] = 0.0
        dtail[0] = 0.0
        if max_deg >= 1:
            values[1] = (alpha_vec[:, None] + 2.0) * coord[None, :] - tail_sum[None, :]
            dcoord[1] = alpha_vec[:, None] + 2.0
            dtail[1] = -1.0

        tail_squared = (tail_sum * tail_sum)[None, :]
        two_tail = (2.0 * tail_sum)[None, :]
        shifted_coord = (2.0 * coord - tail_sum)[None, :]
        tail_row = tail_sum[None, :]
        for deg in range(2, max_deg + 1):
            c_shift = coeff_shifted[deg][:, None]
            c_tail = coeff_tail[deg][:, None]
            c_prev = coeff_prev[deg][:, None]
            linear_factor = c_shift * shifted_coord + c_tail * tail_row
            dlinear_dcoord = 2.0 * c_shift
            dlinear_dtail = c_tail - c_shift
            prev_val = values[deg - 1]
            prev2_val = values[deg - 2]
            values[deg] = linear_factor * prev_val - c_prev * tail_squared * prev2_val
            dcoord[deg] = (
                dlinear_dcoord * prev_val
                + linear_factor * dcoord[deg - 1]
                - c_prev * tail_squared * dcoord[deg - 2]
            )
            dtail[deg] = (
                dlinear_dtail * prev_val
                + linear_factor * dtail[deg - 1]
                - c_prev * (two_tail * prev2_val + tail_squared * dtail[deg - 2])
            )
        return values, dcoord, dtail

    def _axis_jacobi_second_order_jet(
        self, axis, coord, coord_dir, tail_sum, tail_sum_dir
    ):
        """Vectorized 2nd-order Taylor jet (f, f', f'') for all alphas on axis."""
        alpha_vec, coeff_shifted, coeff_tail, coeff_prev = (
            self.jacobi_tables_by_axis[axis]
        )
        max_deg = self.degree
        num_alphas = len(alpha_vec)
        num_points = len(coord)
        values = np.empty((max_deg + 1, num_alphas, num_points), dtype=float)
        first_deriv = np.empty((max_deg + 1, num_alphas, num_points), dtype=float)
        second_deriv = np.empty((max_deg + 1, num_alphas, num_points), dtype=float)

        values[0] = 1.0
        first_deriv[0] = 0.0
        second_deriv[0] = 0.0
        if max_deg >= 1:
            alpha_col = alpha_vec[:, None] + 2.0
            values[1] = alpha_col * coord[None, :] - tail_sum[None, :]
            first_deriv[1] = alpha_col * coord_dir[None, :] - tail_sum_dir[None, :]
            second_deriv[1] = 0.0

        tail_squared = (tail_sum * tail_sum)[None, :]
        tail_squared_first_deriv = (2.0 * tail_sum * tail_sum_dir)[None, :]
        tail_squared_second_deriv = (2.0 * tail_sum_dir * tail_sum_dir)[None, :]
        shifted_coord = (2.0 * coord - tail_sum)[None, :]
        shifted_coord_dir = (2.0 * coord_dir - tail_sum_dir)[None, :]
        tail_row = tail_sum[None, :]
        tail_dir_row = tail_sum_dir[None, :]

        for deg in range(2, max_deg + 1):
            c_shift = coeff_shifted[deg][:, None]
            c_tail = coeff_tail[deg][:, None]
            c_prev = coeff_prev[deg][:, None]
            linear_factor = c_shift * shifted_coord + c_tail * tail_row
            linear_factor_dir = c_shift * shifted_coord_dir + c_tail * tail_dir_row

            values[deg] = (
                linear_factor * values[deg - 1]
                - c_prev * tail_squared * values[deg - 2]
            )
            first_deriv[deg] = (
                linear_factor_dir * values[deg - 1]
                + linear_factor * first_deriv[deg - 1]
                - c_prev
                * (
                    tail_squared_first_deriv * values[deg - 2]
                    + tail_squared * first_deriv[deg - 2]
                )
            )
            second_deriv[deg] = (
                2.0 * linear_factor_dir * first_deriv[deg - 1]
                + linear_factor * second_deriv[deg - 1]
                - c_prev
                * (
                    tail_squared_second_deriv * values[deg - 2]
                    + 2.0 * tail_squared_first_deriv * first_deriv[deg - 2]
                    + tail_squared * second_deriv[deg - 2]
                )
            )
        return values, first_deriv, second_deriv

    def evaluate_values(self, points):
        """Evaluate all Dubiner basis polynomials at points."""
        num_points = len(points)
        barycentric, tail_sums = self._barycentric_and_tail_sums(points)
        values = np.repeat(
            self.normalization_factors[:, None], num_points, axis=1
        )
        for axis in range(self.dimension):
            table = self._axis_jacobi_values(
                axis, barycentric[axis], tail_sums[axis]
            )
            values *= table[
                self.multi_index_array[:, axis], self.alpha_slot_by_axis[axis]
            ]
        return values

    def evaluate_values_and_gradients(self, points):
        """Evaluate Dubiner basis values and Cartesian coordinate gradients at points."""
        num_points = len(points)
        barycentric, tail_sums = self._barycentric_and_tail_sums(points)

        # Partial derivatives wrt barycentric u_m and tail sum s_m, then chain rule:
        # du_0/dx_j = -1, du_{j+1}/dx_j = +1; ds_0/dx_j = 0, ds_m/dx_j = 1 if m <= j+1
        factor_values_by_axis = []
        factor_grads = [[None] * self.dimension for _ in range(self.dimension)]

        for axis in range(self.dimension):
            slots = self.alpha_slot_by_axis[axis]
            exponents = self.multi_index_array[:, axis]
            val_tab, dcoord_tab, dtail_tab = self._axis_jacobi_values_and_partials(
                axis, barycentric[axis], tail_sums[axis]
            )
            factor_val = val_tab[exponents, slots]
            factor_dcoord = dcoord_tab[exponents, slots]
            factor_dtail = dtail_tab[exponents, slots]
            factor_values_by_axis.append(factor_val)

            for coord_idx in range(self.dimension):
                du_dx = (
                    -1.0
                    if axis == 0
                    else (1.0 if axis == coord_idx + 1 else 0.0)
                )
                ds_dx = (
                    0.0
                    if axis == 0
                    else (1.0 if axis <= coord_idx + 1 else 0.0)
                )
                if du_dx != 0.0 or ds_dx != 0.0:
                    factor_grads[coord_idx][axis] = (
                        du_dx * factor_dcoord + ds_dx * factor_dtail
                    )

        values = np.repeat(
            self.normalization_factors[:, None], num_points, axis=1
        )
        for axis in range(self.dimension):
            values *= factor_values_by_axis[axis]

        gradients = np.zeros(
            (self.dimension, self.num_basis, num_points), dtype=float
        )
        for coord_idx in range(self.dimension):
            for axis in range(self.dimension):
                axis_deriv = factor_grads[coord_idx][axis]
                if axis_deriv is None:
                    continue
                term = self.normalization_factors[:, None] * axis_deriv
                for other_axis in range(self.dimension):
                    if other_axis != axis:
                        term *= factor_values_by_axis[other_axis]
                gradients[coord_idx] += term
        return values, gradients

    def evaluate_second_order_jet(self, points, directions):
        """Evaluate (value, 1st directional deriv, 2nd directional deriv) along directions."""
        num_points = len(points)
        barycentric, tail_sums = self._barycentric_and_tail_sums(points)

        barycentric_dir = np.empty((self.dimension + 1, num_points), dtype=float)
        barycentric_dir[0] = -np.sum(directions, axis=1)
        for axis in range(self.dimension):
            barycentric_dir[axis + 1] = directions[:, axis]

        tail_sums_dir = np.empty((self.dimension + 1, num_points), dtype=float)
        tail_sums_dir[self.dimension] = barycentric_dir[self.dimension]
        for axis in range(self.dimension - 1, -1, -1):
            tail_sums_dir[axis] = (
                barycentric_dir[axis] + tail_sums_dir[axis + 1]
            )

        prod_val = np.repeat(
            self.normalization_factors[:, None], num_points, axis=1
        )
        prod_first = np.zeros((self.num_basis, num_points), dtype=float)
        prod_second = np.zeros((self.num_basis, num_points), dtype=float)

        for axis in range(self.dimension):
            slots = self.alpha_slot_by_axis[axis]
            exponents = self.multi_index_array[:, axis]
            val_tab, first_tab, second_tab = self._axis_jacobi_second_order_jet(
                axis,
                barycentric[axis],
                barycentric_dir[axis],
                tail_sums[axis],
                tail_sums_dir[axis],
            )
            factor_val = val_tab[exponents, slots]
            factor_first = first_tab[exponents, slots]
            factor_second = second_tab[exponents, slots]

            next_val = prod_val * factor_val
            next_first = prod_val * factor_first + prod_first * factor_val
            next_second = (
                prod_val * factor_second
                + 2.0 * prod_first * factor_first
                + prod_second * factor_val
            )
            prod_val = next_val
            prod_first = next_first
            prod_second = next_second
        return prod_val, prod_first, prod_second

    def boundary_distance(self, points):
        """Minimum barycentric coordinate (> 0 strictly interior, <= 0 on/outside)."""
        first_bary = 1.0 - np.sum(points, axis=1)
        return float(min(float(first_bary.min()), float(points.min())))

    def boundary_hinge_penalty(self, points, target_margin):
        """Return (violations, gradients, exact_hessians) for u_j < target_margin."""
        first_bary = 1.0 - np.sum(points, axis=1)
        first_violation = np.maximum(0.0, target_margin - first_bary)
        coord_violations = np.maximum(0.0, target_margin - points)
        # Penalty = 0.5 * (sum v0^2 + sum vx^2) -> d/dX_j = +v0 - vx_j
        gradients = first_violation[:, None] - coord_violations
        violations = np.concatenate(
            [first_violation[:, None], coord_violations], axis=1
        )
        hessians = np.zeros((len(points), self.dimension, self.dimension), dtype=float)
        # Active u_0 face has outward normal (1, ..., 1), so Hessian += 1 1^T
        first_active = first_violation > 0.0
        if np.any(first_active):
            hessians[first_active] += 1.0
        # Active u_{j+1} = x_j face has outward normal -e_j, so Hessian[j, j] += 1
        diag_axes = np.arange(self.dimension)
        hessians[:, diag_axes, diag_axes] += (coord_violations > 0.0).astype(float)
        return violations, gradients, hessians


# ---------------------------------------------------------------------------
# Finite symmetry groups and Reynolds-projected invariant polynomial basis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AffineTransform:
    """Affine symmetry operation g(x) = linear @ x + translation."""

    linear: np.ndarray
    translation: np.ndarray


def build_symmetry_group(domain, dimension, symmetry_name):
    """Build affine symmetry group operations for (domain, dimension, symmetry_name)."""
    identity = np.eye(dimension, dtype=float)
    zero_shift = np.zeros(dimension, dtype=float)

    if symmetry_name == "none":
        return [AffineTransform(identity, zero_shift)]

    if domain == "cube":
        if symmetry_name == "central":
            return [
                AffineTransform(identity, zero_shift),
                AffineTransform(-identity, zero_shift),
            ]
        if symmetry_name == "mirror":
            if dimension < 2:
                return None
            # Hyperplane reflection swapping axes 0 <-> 1 (codim-1 fixed plane x_0 = x_1)
            swap = identity.copy()
            swap[0, 0] = 0.0
            swap[1, 1] = 0.0
            swap[0, 1] = 1.0
            swap[1, 0] = 1.0
            return [
                AffineTransform(identity, zero_shift),
                AffineTransform(swap, zero_shift),
            ]
        if symmetry_name == "chiral":
            if dimension == 1:
                return [
                    AffineTransform(identity, zero_shift),
                    AffineTransform(-identity, zero_shift),
                ]
            if dimension == 2:
                # C4: 90-degree rotations of the square
                transforms = []
                rot_90 = np.array([[0.0, -1.0], [1.0, 0.0]])
                current = identity.copy()
                for _ in range(4):
                    transforms.append(AffineTransform(current.copy(), zero_shift))
                    current = current @ rot_90
                return transforms
            # d >= 3: cyclic coordinate shifts x {+I, -I} (order 2d; S6 in 3D)
            transforms = []
            for shift in range(dimension):
                perm_matrix = np.roll(identity, shift, axis=0)
                transforms.append(AffineTransform(perm_matrix, zero_shift))
                transforms.append(AffineTransform(-perm_matrix, zero_shift))
            return transforms
        if symmetry_name == "demi":
            # Demihypercube group D_d: all permutations with prod(signs) == +1
            # (D_2 of order 4 in 2D, tetrahedral Td of order 24 in 3D, D_4 in 4D)
            if dimension < 2 or dimension > 4:
                return None
            transforms = []
            for perm in itertools.permutations(range(dimension)):
                perm_matrix = np.zeros((dimension, dimension), dtype=float)
                for row, col in enumerate(perm):
                    perm_matrix[row, col] = 1.0
                for signs in itertools.product((-1.0, 1.0), repeat=dimension):
                    if math.prod(signs) > 0.0:
                        signed_matrix = perm_matrix * np.asarray(signs)[:, None]
                        transforms.append(AffineTransform(signed_matrix, zero_shift))
            return transforms
        if symmetry_name == "full":
            if dimension > 3:
                return None
            transforms = []
            for perm in itertools.permutations(range(dimension)):
                perm_matrix = np.zeros((dimension, dimension), dtype=float)
                for row, col in enumerate(perm):
                    perm_matrix[row, col] = 1.0
                for signs in itertools.product((-1.0, 1.0), repeat=dimension):
                    signed_matrix = perm_matrix * np.asarray(signs)[:, None]
                    transforms.append(AffineTransform(signed_matrix, zero_shift))
            return transforms
        raise ValueError(f"unknown symmetry '{symmetry_name}' for cube")

    # Simplex Delta^d: vertex permutations of v_0 = 0, v_1 = e_1, ..., v_d = e_d
    if symmetry_name == "demi":
        return None

    vertices = np.zeros((dimension + 1, dimension), dtype=float)
    for axis in range(dimension):
        vertices[axis + 1, axis] = 1.0

    def vertex_perm_to_transform(perm):
        # x = sum_{j=1}^d x_j v_j + (1 - sum x_j) v_0 -> map v_k -> v_{perm[k]}
        translation = vertices[perm[0]].copy()
        linear = np.zeros((dimension, dimension), dtype=float)
        for axis in range(dimension):
            linear[:, axis] = vertices[perm[axis + 1]] - translation
        return AffineTransform(linear, translation)

    if symmetry_name == "mirror":
        if dimension < 2:
            return None
        # Hyperplane reflection swapping vertices v_1 <-> v_2 (x_0 <-> x_1)
        identity_perm = tuple(range(dimension + 1))
        swap_perm = (0, 2, 1) + tuple(range(3, dimension + 1))
        return [
            vertex_perm_to_transform(identity_perm),
            vertex_perm_to_transform(swap_perm),
        ]

    if symmetry_name == "central":
        # Order-2 vertex reversal (0, ..., d) -> (d, ..., 0):
        # 1D interval reflection, 2D altitude mirror, 3D 180-deg C2 rotation, ...
        identity_perm = tuple(range(dimension + 1))
        reversed_perm = tuple(range(dimension, -1, -1))
        return [
            vertex_perm_to_transform(identity_perm),
            vertex_perm_to_transform(reversed_perm),
        ]

    if symmetry_name == "chiral":
        if dimension == 1:
            perms = [(0, 1), (1, 0)]
        elif dimension == 3:
            # A4: all even permutations of the 4 tetrahedron vertices (order 12)
            perms = []
            for perm in itertools.permutations(range(4)):
                inversions = sum(
                    1
                    for i in range(4)
                    for j in range(i + 1, 4)
                    if perm[i] > perm[j]
                )
                if inversions % 2 == 0:
                    perms.append(perm)
        else:
            # Cyclic group C_{d+1} (C3 in 2D, C5 in 4D, ...)
            perms = [
                tuple((i + shift) % (dimension + 1) for i in range(dimension + 1))
                for shift in range(dimension + 1)
            ]
        return [vertex_perm_to_transform(perm) for perm in perms]

    if symmetry_name == "full":
        if dimension > 3:
            return None
        return [
            vertex_perm_to_transform(perm)
            for perm in itertools.permutations(range(dimension + 1))
        ]

    raise ValueError(f"unknown symmetry '{symmetry_name}' for simplex")


class SymmetricBasis:
    """Symmetry-reduced orthonormal polynomial basis via Reynolds projection.

    For cubes, every group element is a signed coordinate permutation so the
    Reynolds projection acts combinatorially on multi-indices in O(|G| * M)
    with zero SVD overhead and ultra-sparse rows.
    For simplices, isometric symmetries preserve each total-degree orthogonal
    block P_k \\cap P_{k-1}^perp, so Reynolds + SVD runs block-by-block per
    total degree and stores a sparse block-diagonal projection matrix.
    """

    def __init__(self, base_basis, symmetry_name, group_transforms):
        self.base_basis = base_basis
        self.domain = base_basis.domain
        self.dimension = base_basis.dimension
        self.degree = base_basis.degree
        self.symmetry_name = symmetry_name
        self.group_transforms = group_transforms
        self.group_order = len(group_transforms)
        self._active_indices = None
        self._use_sparse = False
        self._sparse_proj = None

        if self.group_order == 1:
            self.projection_matrix = np.eye(base_basis.num_basis, dtype=float)
            self.num_basis = base_basis.num_basis
            self.exact_moments = base_basis.exact_moments.copy()
            return

        if base_basis.domain == "cube":
            self._init_cube_combinatorial(base_basis, group_transforms)
        else:
            self._init_simplex_degree_blocks(base_basis, group_transforms)

    def _init_cube_combinatorial(self, base_basis, group_transforms):
        """Build exact sparse invariant basis on [-1, 1]^d from signed permutations."""
        dimension = base_basis.dimension
        index_lookup = {mi: idx for idx, mi in enumerate(base_basis.multi_indices)}

        # Extract (col_perm, row_signs) for each transform: (g x)_i = signs[i] * x_{perm[i]}
        signed_perms = []
        for transform in group_transforms:
            mat = transform.linear
            perm = []
            signs = []
            for row in range(dimension):
                col = int(np.argmax(np.abs(mat[row])))
                perm.append(col)
                signs.append(1 if mat[row, col] > 0.0 else -1)
            signed_perms.append((tuple(perm), tuple(signs)))

        visited = set()
        rows = []
        cols = []
        vals = []
        row_count = 0

        for mi in base_basis.multi_indices:
            if mi in visited:
                continue
            orbit_coeffs = {}
            for perm, signs in signed_perms:
                beta = [0] * dimension
                sign_prod = 1
                for axis in range(dimension):
                    exp = mi[axis]
                    beta[perm[axis]] = exp
                    if (exp & 1) and signs[axis] < 0:
                        sign_prod = -sign_prod
                beta_tuple = tuple(beta)
                orbit_coeffs[beta_tuple] = orbit_coeffs.get(beta_tuple, 0) + sign_prod

            for beta_tuple in orbit_coeffs:
                visited.add(beta_tuple)

            nonzero = [(b, c) for b, c in orbit_coeffs.items() if c != 0]
            if not nonzero:
                continue
            norm = math.sqrt(sum(c * c for _, c in nonzero))
            for beta_tuple, coeff in nonzero:
                rows.append(row_count)
                cols.append(index_lookup[beta_tuple])
                vals.append(coeff / norm)
            row_count += 1

        full_sparse = sparse.csr_matrix(
            (vals, (rows, cols)),
            shape=(row_count, base_basis.num_basis),
            dtype=float,
        )
        self.num_basis = row_count
        self.projection_matrix = full_sparse.toarray()
        self.exact_moments = full_sparse @ base_basis.exact_moments

        active_cols = np.flatnonzero(np.diff(full_sparse.tocsc().indptr) > 0)
        if len(active_cols) < base_basis.num_basis:
            self._active_indices = active_cols
            self._sparse_proj = full_sparse[:, active_cols].tocsr()
        else:
            self._sparse_proj = full_sparse
        self._use_sparse = True

    def _init_simplex_degree_blocks(self, base_basis, group_transforms):
        """Build degree-block orthonormal invariant basis on Delta^d via Reynolds + SVD."""
        blocks = [[] for _ in range(base_basis.degree + 1)]
        for idx, mi in enumerate(base_basis.multi_indices):
            blocks[sum(mi)].append(idx)
        max_block = max(len(block) for block in blocks)

        num_probes = max(2 * max_block + 16, 40)
        halton = generate_halton_points(num_probes, base_basis.dimension)
        # Map (0, 1)^d smoothly into interior of Delta^d via stick-breaking
        probe_points = np.zeros_like(halton)
        remaining = np.ones(num_probes, dtype=float)
        for axis in range(base_basis.dimension):
            fraction = 0.05 + 0.85 * halton[:, axis]
            probe_points[:, axis] = (
                remaining * fraction / float(base_basis.dimension - axis + 1)
            )
            remaining -= probe_points[:, axis]

        reynolds_avg = np.zeros((base_basis.num_basis, num_probes), dtype=float)
        for transform in group_transforms:
            transformed = probe_points @ transform.linear.T + transform.translation[None, :]
            reynolds_avg += base_basis.evaluate_values(transformed)
        reynolds_avg /= float(self.group_order)

        row_indices = []
        col_indices = []
        data_values = []
        row_cursor = 0
        for block_cols in blocks:
            if not block_cols:
                continue
            sub_avg = reynolds_avg[block_cols, :]
            left_singular, singular_values, _ = np.linalg.svd(
                sub_avg, full_matrices=False
            )
            if len(singular_values) == 0 or singular_values[0] <= 1e-12:
                continue
            rank_tol = 1e-10 * float(singular_values[0])
            block_rank = int(np.sum(singular_values > rank_tol))
            block_basis = left_singular[:, :block_rank].T
            for local_row in range(block_rank):
                vec = block_basis[local_row]
                for local_col, val in enumerate(vec):
                    if abs(val) > 1e-13:
                        row_indices.append(row_cursor)
                        col_indices.append(block_cols[local_col])
                        data_values.append(float(val))
                row_cursor += 1

        full_sparse = sparse.csr_matrix(
            (data_values, (row_indices, col_indices)),
            shape=(row_cursor, base_basis.num_basis),
            dtype=float,
        )
        self.num_basis = row_cursor
        self.projection_matrix = full_sparse.toarray()
        self.exact_moments = full_sparse @ base_basis.exact_moments
        density = full_sparse.nnz / float(max(1, row_cursor * base_basis.num_basis))
        if density < 0.35:
            self._sparse_proj = full_sparse
            self._use_sparse = True

    def _project(self, matrix):
        if self._use_sparse:
            return self._sparse_proj @ matrix
        return self.projection_matrix @ matrix

    def evaluate_values(self, points):
        if self.group_order == 1:
            return self.base_basis.evaluate_values(points)
        if self._active_indices is not None:
            return self._project(
                self.base_basis.evaluate_values(
                    points, active_indices=self._active_indices
                )
            )
        return self._project(self.base_basis.evaluate_values(points))

    def evaluate_values_and_gradients(self, points):
        if self.group_order == 1:
            return self.base_basis.evaluate_values_and_gradients(points)
        if self._active_indices is not None:
            base_vals, base_grads = self.base_basis.evaluate_values_and_gradients(
                points, active_indices=self._active_indices
            )
        else:
            base_vals, base_grads = self.base_basis.evaluate_values_and_gradients(
                points
            )
        values = self._project(base_vals)
        gradients = np.empty((self.dimension, self.num_basis, len(points)), dtype=float)
        for axis in range(self.dimension):
            gradients[axis] = self._project(base_grads[axis])
        return values, gradients

    def evaluate_second_order_jet(self, points, directions):
        if self.group_order == 1:
            return self.base_basis.evaluate_second_order_jet(points, directions)
        if self._active_indices is not None:
            base_val, base_first, base_second = (
                self.base_basis.evaluate_second_order_jet(
                    points, directions, active_indices=self._active_indices
                )
            )
        else:
            base_val, base_first, base_second = (
                self.base_basis.evaluate_second_order_jet(points, directions)
            )
        return (
            self._project(base_val),
            self._project(base_first),
            self._project(base_second),
        )


# ---------------------------------------------------------------------------
# Orbit parameterization and deterministic equilibrium initializations
# ---------------------------------------------------------------------------


@dataclass
class SymmetryOrbit:
    """Single symmetry orbit parameterized on its stabilizer fixed subspace."""

    size: int
    subspace_origin: np.ndarray
    subspace_basis: np.ndarray
    subspace_coords: np.ndarray
    weight_sqrt: float
    initial_point: np.ndarray = None

    @property
    def spatial_dof(self):
        return int(self.subspace_basis.shape[1])

    @property
    def num_unknowns(self):
        return self.spatial_dof + 1

    def representative_point(self):
        if self.spatial_dof > 0:
            return self.subspace_origin + self.subspace_basis @ self.subspace_coords
        return self.subspace_origin.copy()


@dataclass
class CubatureSolution:
    """Converged cubature rule with quality tag and diagnostic metrics."""

    quality: str
    moment_residual: float
    boundary_distance: float
    nodes: np.ndarray
    weights: np.ndarray
    orbits: list


def compute_orbit_subspace(point, group_transforms, tol=1e-7):
    """Compute orbit size, stabilizer fixed subspace (origin, basis), and orbit images."""
    dimension = len(point)
    images = []
    stabilizer = []
    for transform in group_transforms:
        image = transform.linear @ point + transform.translation
        if float(np.linalg.norm(image - point)) < tol:
            stabilizer.append(transform)
        if all(float(np.linalg.norm(image - prev)) >= tol for prev in images):
            images.append(image)

    orbit_size = len(images)
    if len(stabilizer) == 1:
        return (
            orbit_size,
            np.zeros(dimension, dtype=float),
            np.eye(dimension, dtype=float),
            np.asarray(images),
        )

    # Fixed subspace: (A - I) x = -t for all (A, t) in stabilizer.
    # Since point is fixed, x = point + U @ xi with (A - I) @ U = 0.
    identity = np.eye(dimension, dtype=float)
    constraint_matrix = np.vstack(
        [transform.linear - identity for transform in stabilizer]
    )
    _, singular_values, right_singular_t = np.linalg.svd(
        constraint_matrix, full_matrices=True
    )
    constraint_rank = int(np.sum(singular_values > 1e-9))
    subspace_basis = right_singular_t[constraint_rank:].T.copy()

    if subspace_basis.shape[1] > 0:
        subspace_origin = point - subspace_basis @ (subspace_basis.T @ point)
    else:
        subspace_origin = point.copy()
    return orbit_size, subspace_origin, subspace_basis, np.asarray(images)


def expand_orbits_to_nodes(orbits, group_transforms):
    """Expand symmetry orbits into full (nodes, probability_weights) arrays."""
    all_nodes = []
    all_weights = []
    for orbit in orbits:
        rep_point = orbit.representative_point()
        orbit_weight = float(orbit.weight_sqrt * orbit.weight_sqrt)
        images = []
        for transform in group_transforms:
            image = transform.linear @ rep_point + transform.translation
            if all(float(np.linalg.norm(image - prev)) >= 1e-7 for prev in images):
                images.append(image)
        for image in images:
            all_nodes.append(image)
            all_weights.append(orbit_weight)
    return np.asarray(all_nodes, dtype=float), np.asarray(all_weights, dtype=float)


def cluster_points_into_orbits(points, group_transforms, tol=1e-6):
    """Cluster a symmetric point cloud into distinct G-orbits with fixed subspaces."""
    num_points = len(points)
    visited = np.zeros(num_points, dtype=bool)
    orbits = []
    for idx in range(num_points):
        if visited[idx]:
            continue
        point = points[idx]
        orbit_size, origin, basis, images = compute_orbit_subspace(
            point, group_transforms, tol=tol
        )
        for image in images:
            distances = np.linalg.norm(points - image[None, :], axis=1)
            visited[distances < tol] = True
        subspace_coords = (
            basis.T @ (point - origin)
            if basis.shape[1] > 0
            else np.zeros(0, dtype=float)
        )
        orbits.append(
            SymmetryOrbit(
                size=orbit_size,
                subspace_origin=origin,
                subspace_basis=basis,
                subspace_coords=subspace_coords,
                weight_sqrt=1.0,
                initial_point=point.copy(),
            )
        )
    return orbits


def select_orbits_for_point_count(candidate_orbits, target_points, min_unknowns):
    """Exact Pareto-pruned DP minimizing total orbit rank for exact target_points.

    Finds a subset of candidate_orbits with sum(size) == target_points and
    sum(num_unknowns) >= min_unknowns that minimizes the sum of candidate rank
    indices (selecting the innermost natural equilibrium lattice shells).
    """
    candidate_pool = candidate_orbits[: max(target_points + 64, 160)]
    # State: pareto_by_count[point_count] = {capped_unknowns: min_rank_cost}
    pareto_by_count = {0: {0: 0.0}}
    parent_pointers = []

    for orbit_idx, orbit in enumerate(candidate_pool):
        orbit_size = orbit.size
        orbit_unknowns = orbit.num_unknowns
        rank_cost = float(orbit_idx)
        next_by_count = {
            cnt: dict(unk_map) for cnt, unk_map in pareto_by_count.items()
        }
        chosen_prev = {}

        for point_count, unk_map in pareto_by_count.items():
            next_count = point_count + orbit_size
            if next_count > target_points:
                continue
            target_map = next_by_count.setdefault(next_count, {})
            for unk_sum, cur_cost in unk_map.items():
                next_unknowns = min(min_unknowns, unk_sum + orbit_unknowns)
                next_cost = cur_cost + rank_cost
                if next_unknowns not in target_map or next_cost < target_map[next_unknowns]:
                    target_map[next_unknowns] = next_cost
                    chosen_prev[(next_count, next_unknowns)] = (point_count, unk_sum)

        # Prune strictly dominated states: (u1 >= u2 and cost1 <= cost2) drops (u2, cost2)
        for point_count, unk_map in next_by_count.items():
            if len(unk_map) <= 1:
                continue
            kept = {}
            best_cost_so_far = float("inf")
            for unk_val, cost_val in sorted(unk_map.items(), key=lambda kv: -kv[0]):
                if cost_val < best_cost_so_far:
                    kept[unk_val] = cost_val
                    best_cost_so_far = cost_val
            next_by_count[point_count] = kept

        pareto_by_count = next_by_count
        parent_pointers.append(chosen_prev)

    if (
        target_points not in pareto_by_count
        or min_unknowns not in pareto_by_count[target_points]
    ):
        return None

    selected = []
    current_state = (target_points, min_unknowns)
    for orbit_idx in range(len(candidate_pool) - 1, -1, -1):
        if current_state in parent_pointers[orbit_idx]:
            selected.append(candidate_pool[orbit_idx])
            current_state = parent_pointers[orbit_idx][current_state]
    selected.reverse()
    return selected


def rotation_around_main_diagonal(dimension, angle_degrees):
    """Deterministic chiral orthogonal rotation fixing (1, ..., 1) for dimension >= 2."""
    if dimension < 2 or abs(angle_degrees) < 1e-12:
        return np.eye(dimension, dtype=float)
    angle_rad = math.radians(angle_degrees)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    if dimension == 2:
        return np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=float)

    # Orthonormal basis of (1, ..., 1)^perp; apply 2D rotations in orthogonal pairs
    diagonal_unit = np.ones((dimension, 1), dtype=float) / math.sqrt(dimension)
    ortho_basis, _ = np.linalg.qr(
        np.hstack([diagonal_unit, np.eye(dimension, dtype=float)]),
        mode="complete",
    )
    subspace_rot = np.eye(dimension, dtype=float)
    idx = 1
    while idx + 1 < dimension:
        subspace_rot[idx, idx] = cos_a
        subspace_rot[idx, idx + 1] = -sin_a
        subspace_rot[idx + 1, idx] = sin_a
        subspace_rot[idx + 1, idx + 1] = cos_a
        idx += 2
    return ortho_basis @ subspace_rot @ ortho_basis.T


def _initialize_cube_orbits_from_points(points, group_transforms):
    """Project cube representative points onto stabilizer subspaces with Chebyshev weights."""
    initialized_orbits = []
    raw_weights = []
    total_weight = 0.0
    for coords in points:
        orbit_size, origin, basis, _ = compute_orbit_subspace(
            coords, group_transforms, tol=1e-6
        )
        subspace_coords = (
            basis.T @ (coords - origin)
            if basis.shape[1] > 0
            else np.zeros(0, dtype=float)
        )
        projected = (
            origin + basis @ subspace_coords if basis.shape[1] > 0 else origin
        )
        equilibrium_weight = float(
            np.prod(np.sqrt(np.maximum(1.0 - projected * projected, 1e-4)))
        )
        total_weight += orbit_size * equilibrium_weight
        raw_weights.append(equilibrium_weight)
        initialized_orbits.append(
            SymmetryOrbit(
                size=orbit_size,
                subspace_origin=origin,
                subspace_basis=basis,
                subspace_coords=subspace_coords,
                weight_sqrt=0.0,
            )
        )
    weight_norm = max(total_weight, 1e-30)
    for orb, raw_w in zip(initialized_orbits, raw_weights):
        orb.weight_sqrt = math.sqrt(raw_w / weight_norm)
    return initialized_orbits


def _initialize_simplex_orbits_from_points(points, group_transforms):
    """Project simplex representative points onto stabilizer subspaces with equilibrium weights."""
    initialized_orbits = []
    raw_weights = []
    total_weight = 0.0
    for coords in points:
        orbit_size, origin, basis, _ = compute_orbit_subspace(
            coords, group_transforms, tol=1e-6
        )
        subspace_coords = (
            basis.T @ (coords - origin)
            if basis.shape[1] > 0
            else np.zeros(0, dtype=float)
        )
        projected = (
            origin + basis @ subspace_coords if basis.shape[1] > 0 else origin
        )
        bary_vec = np.concatenate([[1.0 - float(np.sum(projected))], projected])
        equilibrium_weight = float(np.prod(np.sqrt(np.maximum(bary_vec, 1e-4))))
        total_weight += orbit_size * equilibrium_weight
        raw_weights.append(equilibrium_weight)
        initialized_orbits.append(
            SymmetryOrbit(
                size=orbit_size,
                subspace_origin=origin,
                subspace_basis=basis,
                subspace_coords=subspace_coords,
                weight_sqrt=0.0,
            )
        )
    weight_norm = max(total_weight, 1e-30)
    for orb, raw_w in zip(initialized_orbits, raw_weights):
        orb.weight_sqrt = math.sqrt(raw_w / weight_norm)
    return initialized_orbits


def _relax_orbits_riesz_cube(
    orbits,
    group_transforms,
    margin_eps,
    num_steps=60,
    riesz_excess=2.0,
    wall_strength=1.0,
):
    """Relax a G-symmetric orbit set into a Chebyshev-angle Riesz crystal on [-pi/2, pi/2]^d."""
    num_orbits = len(orbits)
    total_points = sum(orb.size for orb in orbits)
    if num_orbits == 0 or total_points <= 1 or num_steps <= 0 or total_points > 2000:
        return orbits

    dimension = len(orbits[0].subspace_origin)
    reps = np.vstack([orb.representative_point() for orb in orbits])
    angles = np.arcsin(np.clip(reps, -0.9999, 0.9999))
    half_pi = 0.5 * math.pi
    max_angle = half_pi * (1.0 - max(0.005, min(0.25, margin_eps)))

    # Precompute each orbit's stabilizer matrices and distinct image matrices
    stab_mats = []
    image_mats = []
    free_mask = np.array([orb.spatial_dof > 0 for orb in orbits], dtype=bool)
    for idx, orb in enumerate(orbits):
        rep = reps[idx]
        stab = []
        distinct_pts = []
        distinct_mats = []
        for transform in group_transforms:
            img = transform.linear @ rep + transform.translation
            if float(np.linalg.norm(img - rep)) < 1e-5:
                stab.append(transform.linear)
            if not any(float(np.linalg.norm(img - prev)) < 1e-5 for prev in distinct_pts):
                distinct_pts.append(img)
                distinct_mats.append(transform.linear)
        if len(distinct_mats) != orb.size:
            return orbits
        stab_mats.append(stab)
        image_mats.append(np.stack(distinct_mats, axis=0))

    spacing = math.pi / (float(total_points) ** (1.0 / float(dimension)))
    step_size = 0.10 * spacing
    softening_sq = (0.06 * spacing) ** 2
    riesz_power = -0.5 * float(dimension + riesz_excess)

    for _ in range(num_steps):
        all_chunks = [
            np.einsum("gij,j->gi", image_mats[idx], angles[idx])
            for idx in range(num_orbits)
        ]
        all_angles = np.vstack(all_chunks)

        diff = angles[:, None, :] - all_angles[None, :, :]
        raw_d2 = np.sum(diff * diff, axis=2)
        dist_sq = np.where(raw_d2 < 1e-12, 1e30, raw_d2 + softening_sq)
        force = np.sum(diff * (dist_sq ** riesz_power)[:, :, None], axis=1)

        # Mirror-image repulsion from all 2d boundary walls theta_j = +/- pi/2
        dist_pos = np.maximum(half_pi - angles, 0.5 * (half_pi - max_angle))
        dist_neg = np.maximum(half_pi + angles, 0.5 * (half_pi - max_angle))
        force -= (
            wall_strength
            * (2.0 * dist_pos)
            * (((2.0 * dist_pos) ** 2 + softening_sq) ** riesz_power)
        )
        force += (
            wall_strength
            * (2.0 * dist_neg)
            * (((2.0 * dist_neg) ** 2 + softening_sq) ** riesz_power)
        )

        # Average force over each orbit's stabilizer so every subspace stays exact
        projected_force = np.zeros_like(force)
        for idx in range(num_orbits):
            if not free_mask[idx]:
                continue
            avg_f = np.zeros(dimension, dtype=float)
            for mat in stab_mats[idx]:
                avg_f += mat @ force[idx]
            projected_force[idx] = avg_f / float(len(stab_mats[idx]))

        force_norm = np.linalg.norm(projected_force, axis=1, keepdims=True)
        step_scale = np.minimum(1.0, step_size / np.maximum(force_norm, 1e-12))
        trial_angles = np.clip(
            angles + projected_force * step_scale, -max_angle, max_angle
        )

        # Guard against collapsing an orbit onto a higher-symmetry fixed subspace
        for idx in range(num_orbits):
            if not free_mask[idx] or orbits[idx].size <= 1:
                continue
            imgs = np.einsum("gij,j->gi", image_mats[idx], trial_angles[idx])
            d2 = np.sum((imgs[:, None, :] - imgs[None, :, :]) ** 2, axis=2)
            np.fill_diagonal(d2, 1e30)
            if float(np.min(d2)) < 1e-6:
                trial_angles[idx] = angles[idx]
        angles = trial_angles
        step_size *= 0.96

    relaxed_points = np.sin(angles)
    return _initialize_cube_orbits_from_points(relaxed_points, group_transforms)


def _relax_orbits_riesz_simplex(
    orbits,
    group_transforms,
    margin_eps,
    num_steps=60,
    riesz_excess=2.0,
    wall_strength=1.0,
):
    """Relax a G-symmetric orbit set into a geodesic Riesz crystal on S^d_+."""
    num_orbits = len(orbits)
    total_points = sum(orb.size for orb in orbits)
    if num_orbits == 0 or total_points <= 1 or num_steps <= 0 or total_points > 2000:
        return orbits

    dimension = len(orbits[0].subspace_origin)
    num_bary = dimension + 1

    # Extract barycentric index permutation inv_perm for each group element:
    # b(g(x))[m] = b(x)[inv_perm[m]]
    vertices = np.zeros((num_bary, dimension), dtype=float)
    for axis in range(dimension):
        vertices[axis + 1, axis] = 1.0
    inv_perms = []
    for transform in group_transforms:
        mapped_v = vertices @ transform.linear.T + transform.translation[None, :]
        mapped_bary = np.hstack(
            [1.0 - np.sum(mapped_v, axis=1, keepdims=True), mapped_v]
        )
        target = [int(np.argmax(mapped_bary[k])) for k in range(num_bary)]
        inv_perm = [0] * num_bary
        for k, dst in enumerate(target):
            inv_perm[dst] = k
        inv_perms.append(np.asarray(inv_perm, dtype=np.int32))

    reps = np.vstack([orb.representative_point() for orb in orbits])
    bary = np.hstack([1.0 - np.sum(reps, axis=1, keepdims=True), reps])
    sphere = np.sqrt(np.maximum(bary, 1e-8))
    sphere /= np.linalg.norm(sphere, axis=1, keepdims=True)

    stab_perms = []
    image_perms = []
    free_mask = np.array([orb.spatial_dof > 0 for orb in orbits], dtype=bool)
    for idx, orb in enumerate(orbits):
        s_vec = sphere[idx]
        stab = []
        distinct_s = []
        distinct_p = []
        for perm in inv_perms:
            img = s_vec[perm]
            if float(np.linalg.norm(img - s_vec)) < 1e-5:
                stab.append(perm)
            if not any(float(np.linalg.norm(img - prev)) < 1e-5 for prev in distinct_s):
                distinct_s.append(img)
                distinct_p.append(perm)
        if len(distinct_p) != orb.size:
            return orbits
        stab_perms.append(stab)
        image_perms.append(np.stack(distinct_p, axis=0))

    min_coord = math.sqrt(max(1e-4, 0.25 * margin_eps))
    spacing = (1.0 / float(max(total_points, 1))) ** (1.0 / float(dimension))
    step_size = 0.09 * spacing
    softening_sq = (0.05 * spacing) ** 2
    riesz_power = -0.5 * float(dimension + riesz_excess)

    for _ in range(num_steps):
        all_chunks = [sphere[idx][image_perms[idx]] for idx in range(num_orbits)]
        all_sphere = np.vstack(all_chunks)

        diff = sphere[:, None, :] - all_sphere[None, :, :]
        raw_d2 = np.sum(diff * diff, axis=2)
        dist_sq = np.where(raw_d2 < 1e-12, 1e30, raw_d2 + softening_sq)
        force = np.sum(diff * (dist_sq ** riesz_power)[:, :, None], axis=1)

        # Mirror-image repulsion across each boundary wall s_m = 0
        wall_dist_sq = (2.0 * np.maximum(sphere, 0.5 * min_coord)) ** 2 + softening_sq
        force += wall_strength * (2.0 * sphere) * (wall_dist_sq ** riesz_power)

        # Average force over each orbit's stabilizer permutations
        projected_force = np.zeros_like(force)
        for idx in range(num_orbits):
            if not free_mask[idx]:
                continue
            avg_f = np.zeros(num_bary, dtype=float)
            for perm in stab_perms[idx]:
                avg_f += force[idx][perm]
            projected_force[idx] = avg_f / float(len(stab_perms[idx]))

        # Project tangent to unit sphere S^d_+
        projected_force -= (
            np.sum(projected_force * sphere, axis=1, keepdims=True) * sphere
        )
        force_norm = np.linalg.norm(projected_force, axis=1, keepdims=True)
        step_scale = np.minimum(1.0, step_size / np.maximum(force_norm, 1e-12))
        trial_sphere = np.maximum(sphere + projected_force * step_scale, min_coord)
        trial_sphere /= np.linalg.norm(trial_sphere, axis=1, keepdims=True)

        # Guard against collapsing an orbit onto a higher-symmetry fixed subspace
        for idx in range(num_orbits):
            if not free_mask[idx] or orbits[idx].size <= 1:
                continue
            imgs = trial_sphere[idx][image_perms[idx]]
            d2 = np.sum((imgs[:, None, :] - imgs[None, :, :]) ** 2, axis=2)
            np.fill_diagonal(d2, 1e30)
            if float(np.min(d2)) < 1e-6:
                trial_sphere[idx] = sphere[idx]
        sphere = trial_sphere
        step_size *= 0.96

    relaxed_bary = sphere * sphere
    return _initialize_simplex_orbits_from_points(
        relaxed_bary[:, 1:], group_transforms
    )


def build_cube_equilibrium_start(
    dimension,
    degree,
    target_points,
    symmetry_name,
    group_transforms,
    boundary_inset=0.35,
    rotation_degrees=6.0,
    norm_order=12.0,
    min_unknowns=1,
    halton_skip=0,
    riesz_steps=0,
    riesz_excess=2.0,
    wall_strength=1.0,
    extra_fixed_pairs=0,
    use_lattice_for_central=False,
    halton_generic=False,
):
    """Deterministic equilibrium start on [-1, 1]^d (Halton/Riesz crystal or rotated D_d*)."""
    eps = max(0.015, min(0.18, 0.5 * boundary_inset / float(max(degree, 2))))

    # For none / mirror (and central unless rotated D_d* lattice is requested) in d >= 2,
    # use exact arcsine-mapped Halton equilibrium clouds (optionally Riesz-relaxed).
    if dimension >= 2 and (
        symmetry_name in ("none", "mirror")
        or (symmetry_name == "central" and not use_lattice_for_central)
    ):
        if symmetry_name == "none":
            if target_points * (dimension + 1) < min_unknowns:
                return None
            halton = generate_halton_points(
                target_points, dimension, skip=halton_skip
            )
            angles = math.pi * (eps + (1.0 - 2.0 * eps) * halton)
            coords_list = -np.cos(angles)
            orbits = _initialize_cube_orbits_from_points(coords_list, group_transforms)
        elif symmetry_name == "central":
            # central symmetry {+I, -I}: draw pairs in half-space x_0 > 0 (+ origin if odd N)
            num_pairs = target_points // 2
            has_origin = (target_points % 2) == 1
            total_unk = num_pairs * (dimension + 1) + (1 if has_origin else 0)
            if total_unk < min_unknowns:
                return None
            coords_list = []
            if has_origin:
                coords_list.append(np.zeros(dimension, dtype=float))
            if num_pairs > 0:
                halton = generate_halton_points(
                    num_pairs, dimension, skip=halton_skip
                )
                angles = math.pi * (eps + (1.0 - 2.0 * eps) * halton)
                # Restrict first coordinate to (0, 1) so each antipodal pair is distinct
                angles[:, 0] = math.pi * (
                    0.5 + 0.5 * eps + (0.5 - 1.5 * eps) * halton[:, 0]
                )
                for row in -np.cos(angles):
                    coords_list.append(row)
            orbits = _initialize_cube_orbits_from_points(coords_list, group_transforms)
        else:
            # mirror symmetry x_0 <-> x_1: fixed plane has dim (d - 1) => d unknowns/pt
            num_fixed = None
            for cand_fixed in range(target_points % 2, target_points + 1, 2):
                cand_pairs = (target_points - cand_fixed) // 2
                cand_unk = cand_pairs * (dimension + 1) + cand_fixed * dimension
                if cand_unk >= min_unknowns:
                    num_fixed = cand_fixed
                    break
            if num_fixed is None:
                return None
            if extra_fixed_pairs > 0 and num_fixed + 2 * extra_fixed_pairs <= target_points:
                cand_fixed = num_fixed + 2 * extra_fixed_pairs
                cand_pairs = (target_points - cand_fixed) // 2
                if cand_pairs * (dimension + 1) + cand_fixed * dimension >= min_unknowns:
                    num_fixed = cand_fixed
            num_pairs = (target_points - num_fixed) // 2
            coords_list = []
            if num_fixed > 0:
                halton_fix = generate_halton_points(
                    num_fixed, dimension - 1, skip=halton_skip
                )
                ang_fix = math.pi * (eps + (1.0 - 2.0 * eps) * halton_fix)
                pts_fix = -np.cos(ang_fix)
                for row in pts_fix:
                    coords_list.append(np.concatenate([[row[0], row[0]], row[1:]]))
            if num_pairs > 0:
                halton_pair = generate_halton_points(
                    num_pairs, dimension, skip=halton_skip + num_fixed + 7
                )
                ang_pair = math.pi * (eps + (1.0 - 2.0 * eps) * halton_pair)
                pts_pair = -np.cos(ang_pair)
                for row in pts_pair:
                    if row[0] < row[1]:
                        row = row.copy()
                        row[0], row[1] = row[1], row[0]
                    if abs(row[0] - row[1]) < 0.03:
                        row = row.copy()
                        row[0] = min(0.96, row[0] + 0.03)
                        row[1] = max(-0.96, row[1] - 0.03)
                    coords_list.append(row)
            orbits = _initialize_cube_orbits_from_points(coords_list, group_transforms)

        if riesz_steps > 0:
            orbits = _relax_orbits_riesz_cube(
                orbits,
                group_transforms,
                eps,
                num_steps=riesz_steps,
                riesz_excess=riesz_excess,
                wall_strength=wall_strength,
            )
        return orbits

    # Parity shift: odd N includes origin (0-shift); even N uses (1/4)-shift
    lattice_shift = 0.0 if (target_points % 2 == 1) else 0.25
    points_per_cell = 2.0 if dimension >= 2 else 1.0
    estimated_side = max(
        1.5, ((target_points * 1.8) / points_per_cell) ** (1.0 / dimension)
    )
    half_width = int(math.ceil(estimated_side / 2.0)) + 2
    while (2 * half_width + 1) ** dimension > 25000 and half_width > 1:
        half_width -= 1

    axis_grid = np.arange(-half_width, half_width + 1, dtype=float)
    mesh = np.meshgrid(*([axis_grid] * dimension), indexing="ij")
    integer_grid = np.stack([coord_grid.ravel() for coord_grid in mesh], axis=1)

    if dimension >= 2:
        if symmetry_name in ("demi", "full"):
            # Standard D_d* lattice Z^d U (Z + 1/2)^d covers every stabilizer orbit type
            raw_points = np.vstack([integer_grid, integer_grid + 0.5])
        else:
            shifted_a = integer_grid + lattice_shift
            shifted_b = integer_grid + (lattice_shift + 0.5)
            raw_points = np.vstack([shifted_a, shifted_b])
    else:
        raw_points = integer_grid + (0.0 if target_points % 2 == 1 else 0.5)

    # Apply chiral rotation for mirror-free groups while preserving G-symmetry
    if symmetry_name in ("chiral", "central") and dimension >= 2:
        rot_matrix = rotation_around_main_diagonal(dimension, rotation_degrees)
        raw_points = raw_points @ rot_matrix.T

    max_abs_coord = np.max(np.abs(raw_points), axis=1)
    raw_points = raw_points[max_abs_coord <= float(half_width) + 0.6]

    # Close under G to guarantee exact symmetry of the candidate pool, then deduplicate
    # so large symmetry groups (e.g. demi/Td order 24 or full/Oh order 48) do not fill
    # the trimmed pool with redundant copies of inner points.
    closed_clouds = [raw_points]
    for transform in group_transforms[1:]:
        closed_clouds.append(
            raw_points @ transform.linear.T + transform.translation[None, :]
        )
    raw_points = np.vstack(closed_clouds)
    _, unique_indices = np.unique(
        np.round(raw_points, decimals=6), axis=0, return_index=True
    )
    raw_points = raw_points[unique_indices]

    if norm_order > 0.0:
        point_radii = np.sum(np.abs(raw_points) ** norm_order, axis=1) ** (
            1.0 / norm_order
        )
    else:
        point_radii = np.max(np.abs(raw_points), axis=1)
    raw_points = raw_points[np.argsort(point_radii)]
    raw_points = raw_points[: max(target_points * 8, 500)]

    candidate_orbits = cluster_points_into_orbits(
        raw_points, group_transforms, tol=1e-5
    )
    if norm_order > 0.0:
        candidate_orbits.sort(
            key=lambda orb: float(
                np.sum(np.abs(orb.initial_point) ** norm_order) ** (1.0 / norm_order)
            )
        )
    else:
        candidate_orbits.sort(
            key=lambda orb: float(np.max(np.abs(orb.initial_point)))
        )

    selected = select_orbits_for_point_count(
        candidate_orbits, target_points, min_unknowns
    )
    if selected is None:
        return None

    # Map lattice coordinates to Chebyshev angles in [-target_angle, target_angle]^d
    selected_raw = np.vstack([orb.initial_point for orb in selected])
    max_angle = float(np.abs(selected_raw).max())
    target_angle = (
        0.5 * math.pi * max(0.35, 1.0 - 2.0 * boundary_inset / max(degree, 1))
    )
    angle_scale = (target_angle / max_angle) if max_angle > 1e-12 else 0.0
    mapped_points = [np.sin(orb.initial_point * angle_scale) for orb in selected]
    orbits = _initialize_cube_orbits_from_points(mapped_points, group_transforms)

    if halton_generic and dimension >= 2:
        # Break artificial lattice mirror symmetries by seeding full-dim orbits
        # from arcsine-mapped Halton points before G-equivariant Riesz relaxation.
        num_gen = sum(1 for orb in orbits if orb.spatial_dof == dimension)
        if num_gen > 0:
            halton = generate_halton_points(num_gen, dimension, skip=halton_skip)
            angles = math.pi * (eps + (1.0 - 2.0 * eps) * halton)
            halton_pts = -np.cos(angles)
            mixed_pts = []
            gen_idx = 0
            for orb in orbits:
                if orb.spatial_dof == dimension:
                    mixed_pts.append(halton_pts[gen_idx])
                    gen_idx += 1
                else:
                    mixed_pts.append(orb.representative_point())
            trial_orbits = _initialize_cube_orbits_from_points(
                mixed_pts, group_transforms
            )
            if sum(orb.size for orb in trial_orbits) == target_points:
                orbits = trial_orbits

    if riesz_steps > 0 and dimension >= 2:
        orbits = _relax_orbits_riesz_cube(
            orbits,
            group_transforms,
            eps,
            num_steps=riesz_steps,
            riesz_excess=riesz_excess,
            wall_strength=wall_strength,
        )
    return orbits


def generate_simplex_barycentric_lattice(dimension, level):
    """Return all barycentric grid points m / level with m in Z_>=0^{d+1}, sum(m) == level."""
    lattice_points = []

    def append_recursive(prefix, remaining_slots, remaining_sum):
        if remaining_slots == 1:
            lattice_points.append(prefix + (remaining_sum,))
            return
        for count in range(remaining_sum + 1):
            append_recursive(
                prefix + (count,), remaining_slots - 1, remaining_sum - count
            )

    append_recursive((), dimension + 1, level)
    return np.asarray(lattice_points, dtype=float) / float(max(level, 1))


def _sample_dirichlet_half_halton(num_points, dimension, margin_eps, skip=0):
    """Deterministic Halton points mapped to exact Dirichlet(1/2, ..., 1/2) equilibrium measure."""
    halton = generate_halton_points(num_points, dimension, skip=skip)
    clipped = margin_eps + (1.0 - 2.0 * margin_eps) * halton
    bary = np.empty((num_points, dimension + 1), dtype=float)
    remaining = np.ones(num_points, dtype=float)
    for axis in range(dimension):
        beta_param = 0.5 * float(dimension - axis)
        fraction = betaincinv(0.5, beta_param, clipped[:, axis])
        bary[:, axis] = remaining * fraction
        remaining = remaining * (1.0 - fraction)
    bary[:, dimension] = remaining
    return bary


def _riesz_relax_on_positive_sphere(
    bary_points, margin_eps, num_steps=50, riesz_excess=2.0, wall_strength=1.0
):
    """Relax barycentric points into a well-separated crystal on S^d_+ via Riesz repulsion."""
    num_points, num_bary = bary_points.shape
    dimension = num_bary - 1
    if num_points <= 1 or num_steps <= 0 or num_points > 2000:
        return bary_points

    sphere = np.sqrt(np.maximum(bary_points, 1e-8))
    sphere /= np.linalg.norm(sphere, axis=1, keepdims=True)
    min_coord = math.sqrt(max(1e-4, 0.25 * margin_eps))
    spacing = (1.0 / float(max(num_points, 1))) ** (1.0 / float(dimension))
    step_size = 0.08 * spacing
    softening_sq = (0.05 * spacing) ** 2
    riesz_power = -0.5 * float(dimension + riesz_excess)

    for _ in range(num_steps):
        diff = sphere[:, None, :] - sphere[None, :, :]
        dist_sq = np.sum(diff * diff, axis=2) + softening_sq
        np.fill_diagonal(dist_sq, 1e30)
        force = np.sum(diff * (dist_sq ** riesz_power)[:, :, None], axis=1)
        # Mirror-image repulsion across each boundary wall s_m = 0
        wall_dist_sq = (2.0 * np.maximum(sphere, 0.5 * min_coord)) ** 2 + softening_sq
        force += wall_strength * (2.0 * sphere) * (wall_dist_sq ** riesz_power)
        # Project force onto tangent space of unit sphere S^d_+
        force -= np.sum(force * sphere, axis=1, keepdims=True) * sphere
        force_norm = np.linalg.norm(force, axis=1, keepdims=True)
        step_scale = np.minimum(1.0, step_size / np.maximum(force_norm, 1e-12))
        sphere = np.maximum(sphere + force * step_scale, min_coord)
        sphere /= np.linalg.norm(sphere, axis=1, keepdims=True)
        step_size *= 0.96

    return sphere * sphere


def build_simplex_equilibrium_start(
    dimension,
    degree,
    target_points,
    symmetry_name,
    group_transforms,
    boundary_inset=0.45,
    chiral_twist=0.06,
    min_unknowns=1,
    halton_skip=0,
    riesz_steps=0,
    riesz_excess=2.0,
    wall_strength=1.0,
    extra_fixed_pairs=0,
    halton_generic=False,
):
    """Deterministic equilibrium start on Delta^d (Riesz-S^d_+/Halton or S^d_+ shells)."""
    eps = max(0.015, min(0.18, 0.5 * boundary_inset / float(max(degree, 2))))

    if dimension >= 2 and symmetry_name == "none":
        if target_points * (dimension + 1) < min_unknowns:
            return None
        bary = _sample_dirichlet_half_halton(
            target_points, dimension, eps, skip=halton_skip
        )
        if riesz_steps > 0:
            bary = _riesz_relax_on_positive_sphere(
                bary,
                eps,
                num_steps=riesz_steps,
                riesz_excess=riesz_excess,
                wall_strength=wall_strength,
            )
        return _initialize_simplex_orbits_from_points(bary[:, 1:], group_transforms)

    if dimension >= 2 and symmetry_name in ("central", "mirror"):
        # Order-2 vertex permutation:
        #   central: (u_0, ..., u_d) -> (u_d, ..., u_0), fixed_dof = floor(d / 2)
        #   mirror:  u_1 <-> u_2 (x_0 <-> x_1), fixed_dof = d - 1
        fixed_dof = (dimension // 2) if symmetry_name == "central" else (dimension - 1)
        num_fixed = None
        for cand_fixed in range(target_points % 2, target_points + 1, 2):
            cand_pairs = (target_points - cand_fixed) // 2
            cand_unk = cand_pairs * (dimension + 1) + cand_fixed * (fixed_dof + 1)
            if cand_unk >= min_unknowns:
                num_fixed = cand_fixed
                break
        if num_fixed is None:
            return None
        if extra_fixed_pairs > 0 and num_fixed + 2 * extra_fixed_pairs <= target_points:
            cand_fixed = num_fixed + 2 * extra_fixed_pairs
            cand_pairs = (target_points - cand_fixed) // 2
            if cand_pairs * (dimension + 1) + cand_fixed * (fixed_dof + 1) >= min_unknowns:
                num_fixed = cand_fixed

        num_pairs = (target_points - num_fixed) // 2
        coords_list = []
        if num_fixed > 0:
            # Sample fixed points directly in the fixed barycentric simplex of dim fixed_dof
            fixed_bary_sub = _sample_dirichlet_half_halton(
                num_fixed, fixed_dof, eps, skip=halton_skip
            )
            for row in fixed_bary_sub:
                full_bary = np.zeros(dimension + 1, dtype=float)
                if symmetry_name == "mirror":
                    # u_1 == u_2; remaining d - 1 entries are u_0, u_3, ..., u_d
                    full_bary[0] = row[0]
                    full_bary[1] = 0.5 * row[1]
                    full_bary[2] = 0.5 * row[1]
                    if dimension > 2:
                        full_bary[3:] = row[2:]
                else:
                    # Palindromic barycentric coordinates u_k == u_{d - k}
                    num_sym_pairs = (dimension + 1) // 2
                    for k in range(num_sym_pairs):
                        full_bary[k] = 0.5 * row[k]
                        full_bary[dimension - k] = 0.5 * row[k]
                    if (dimension + 1) % 2 == 1:
                        full_bary[num_sym_pairs] = row[num_sym_pairs]
                coords_list.append(full_bary[1:])
        if num_pairs > 0:
            bary = _sample_dirichlet_half_halton(
                num_pairs, dimension, eps, skip=halton_skip + num_fixed + 7
            )
            for row in bary:
                row = row.copy()
                if symmetry_name == "mirror":
                    if row[1] < row[2]:
                        row[1], row[2] = row[2], row[1]
                    if abs(row[1] - row[2]) < 0.015:
                        delta = min(0.015, 0.4 * row[2])
                        row[1] += delta
                        row[2] -= delta
                else:
                    if row[0] < row[dimension]:
                        row = row[::-1]
                    if abs(row[0] - row[dimension]) < 0.015:
                        delta = min(0.015, 0.4 * row[dimension])
                        row[0] += delta
                        row[dimension] -= delta
                coords_list.append(row[1:])
        orbits = _initialize_simplex_orbits_from_points(coords_list, group_transforms)
        if riesz_steps > 0:
            orbits = _relax_orbits_riesz_simplex(
                orbits,
                group_transforms,
                eps,
                num_steps=riesz_steps,
                riesz_excess=riesz_excess,
                wall_strength=wall_strength,
            )
        return orbits

    # Choose primary barycentric level L whose simplex lattice first covers target_points
    primary_level = 1
    while math.comb(primary_level + dimension, dimension) < target_points:
        primary_level += 1
        if math.comb(primary_level + dimension, dimension) > 15000:
            break
    max_level = primary_level
    while math.comb(max_level + dimension, dimension) < max(
        int(target_points * 1.4), target_points + 12
    ):
        if math.comb(max_level + 1 + dimension, dimension) > 15000:
            break
        max_level += 1

    inset_eps = max(0.03, min(0.60, boundary_inset / float(max(degree, 2))))

    def map_level_to_equilibrium(level):
        bary_grid = generate_simplex_barycentric_lattice(dimension, level)
        bary_interior = (1.0 - inset_eps) * bary_grid + inset_eps / float(
            dimension + 1
        )
        # Map to S^d_+ equilibrium density via sin(pi/2 * bary) -> normalize -> u = s^2
        sphere_coords = np.sin(0.5 * math.pi * bary_interior)
        sphere_coords /= np.linalg.norm(sphere_coords, axis=1, keepdims=True)
        bary_mapped = sphere_coords * sphere_coords
        if symmetry_name == "chiral" and chiral_twist > 0.0 and dimension >= 2:
            # Keep lower-dimensional fixed axes intact while twisting generic shells
            sorted_bary = np.sort(bary_mapped, axis=1)
            is_generic = np.all(np.diff(sorted_bary, axis=1) > 1e-6, axis=1)
            twisted = bary_mapped.copy()
            twisted[is_generic] = (
                (1.0 - chiral_twist) * bary_mapped[is_generic]
                + chiral_twist * np.roll(bary_mapped[is_generic], 1, axis=1)
            )
            bary_mapped = twisted
        return bary_mapped[:, 1:]

    # Build primary-level orbits first (so full-simplex equilibrium shells are preferred),
    # then auxiliary levels 1..max_level + centroid so any exact N decomposition is feasible.
    centroid_coord = 1.0 / float(dimension + 1)

    def centroid_distance(point):
        bary_vec = np.concatenate([[1.0 - float(np.sum(point))], point])
        return float(np.linalg.norm(bary_vec - centroid_coord))

    def collect_orbits_for_cloud(cloud):
        closed_clouds = [cloud]
        for transform in group_transforms[1:]:
            closed_clouds.append(
                cloud @ transform.linear.T + transform.translation[None, :]
            )
        full_cloud = np.vstack(closed_clouds)
        orbs = cluster_points_into_orbits(full_cloud, group_transforms, tol=1e-4)
        orbs.sort(key=lambda orb: centroid_distance(orb.initial_point))
        return orbs

    primary_orbits = collect_orbits_for_cloud(map_level_to_equilibrium(primary_level))

    auxiliary_clouds = [
        np.full((1, dimension), centroid_coord, dtype=float)
    ]
    for level in range(1, max_level + 1):
        if level != primary_level:
            auxiliary_clouds.append(map_level_to_equilibrium(level))
    aux_raw = np.vstack(auxiliary_clouds)
    aux_dists = np.array([centroid_distance(pt) for pt in aux_raw], dtype=float)
    aux_raw = aux_raw[np.argsort(aux_dists)][: max(target_points * 5, 300)]
    auxiliary_orbits = collect_orbits_for_cloud(aux_raw)

    # Combine primary orbits first, then non-duplicate auxiliary orbits
    candidate_orbits = list(primary_orbits)
    for aux_orb in auxiliary_orbits:
        is_duplicate = False
        for existing in candidate_orbits:
            if existing.size != aux_orb.size:
                continue
            min_dist = min(
                float(
                    np.linalg.norm(
                        existing.initial_point
                        - (transform.linear @ aux_orb.initial_point + transform.translation)
                    )
                )
                for transform in group_transforms
            )
            if min_dist < 1e-3:
                is_duplicate = True
                break
        if not is_duplicate:
            candidate_orbits.append(aux_orb)

    selected = select_orbits_for_point_count(
        candidate_orbits, target_points, min_unknowns
    )
    if selected is None:
        return None
    mapped_points = [orb.initial_point for orb in selected]
    orbits = _initialize_simplex_orbits_from_points(mapped_points, group_transforms)

    if halton_generic and dimension >= 2:
        # Break artificial mirror symmetry of the barycentric lattice by seeding
        # generic (full-dim) orbit representatives from Dirichlet(1/2) Halton
        # while keeping lower-dimensional stabilizer orbits intact.
        num_gen = sum(1 for orb in orbits if orb.spatial_dof == dimension)
        if num_gen > 0:
            halton_bary = _sample_dirichlet_half_halton(
                num_gen, dimension, eps, skip=halton_skip
            )
            mixed_pts = []
            gen_idx = 0
            for orb in orbits:
                if orb.spatial_dof == dimension:
                    mixed_pts.append(halton_bary[gen_idx, 1:])
                    gen_idx += 1
                else:
                    mixed_pts.append(orb.representative_point())
            trial_orbits = _initialize_simplex_orbits_from_points(
                mixed_pts, group_transforms
            )
            if sum(orb.size for orb in trial_orbits) == target_points:
                orbits = trial_orbits

    if riesz_steps > 0 and dimension >= 2:
        orbits = _relax_orbits_riesz_simplex(
            orbits,
            group_transforms,
            eps,
            num_steps=riesz_steps,
            riesz_excess=riesz_excess,
            wall_strength=wall_strength,
        )
    return orbits


# ---------------------------------------------------------------------------
# Flat parameter packing, moment residuals, exact Jacobians, and penalties
# ---------------------------------------------------------------------------


def pack_orbit_parameters(orbits):
    """Pack all orbit (subspace_coords, weight_sqrt) into a single 1D vector."""
    chunks = []
    for orbit in orbits:
        if orbit.spatial_dof > 0:
            chunks.append(orbit.subspace_coords)
        chunks.append(np.array([orbit.weight_sqrt], dtype=float))
    return np.concatenate(chunks)


def unpack_orbit_parameters(orbits, param_vector):
    """Return copies of orbits populated from a flat parameter vector."""
    unpacked = []
    offset = 0
    for orbit in orbits:
        dof = orbit.spatial_dof
        if dof > 0:
            subspace_coords = param_vector[offset : offset + dof].copy()
        else:
            subspace_coords = np.zeros(0, dtype=float)
        offset += dof
        weight_sqrt = float(param_vector[offset])
        offset += 1
        unpacked.append(
            SymmetryOrbit(
                size=orbit.size,
                subspace_origin=orbit.subspace_origin,
                subspace_basis=orbit.subspace_basis,
                subspace_coords=subspace_coords,
                weight_sqrt=weight_sqrt,
            )
        )
    return unpacked


def decode_orbit_representatives(orbits, param_vector):
    """Decode flat parameter vector into (representatives, weights, weight_sqrts, sizes)."""
    num_orbits = len(orbits)
    dimension = len(orbits[0].subspace_origin)
    representatives = np.empty((num_orbits, dimension), dtype=float)
    orbit_weights = np.empty(num_orbits, dtype=float)
    weight_sqrts = np.empty(num_orbits, dtype=float)
    orbit_sizes = np.empty(num_orbits, dtype=float)

    offset = 0
    for idx, orbit in enumerate(orbits):
        dof = orbit.spatial_dof
        if dof > 0:
            subspace_coords = param_vector[offset : offset + dof]
            representatives[idx] = (
                orbit.subspace_origin + orbit.subspace_basis @ subspace_coords
            )
        else:
            representatives[idx] = orbit.subspace_origin
        offset += dof
        weight_sqrt = param_vector[offset]
        offset += 1
        weight_sqrts[idx] = weight_sqrt
        orbit_weights[idx] = weight_sqrt * weight_sqrt
        orbit_sizes[idx] = float(orbit.size)
    return representatives, orbit_weights, weight_sqrts, orbit_sizes


def evaluate_moment_residual(sym_basis, orbits, param_vector):
    """Evaluate pure moment residual vector r (excluding any boundary penalty)."""
    representatives, orbit_weights, _, orbit_sizes = decode_orbit_representatives(
        orbits, param_vector
    )
    basis_values = sym_basis.evaluate_values(representatives)
    return basis_values @ (orbit_sizes * orbit_weights) - sym_basis.exact_moments


def evaluate_residual_and_jacobian(sym_basis, orbits, param_vector):
    """Evaluate pure moment residual r, exact Jacobian J, and orbit representatives."""
    representatives, orbit_weights, weight_sqrts, orbit_sizes = (
        decode_orbit_representatives(orbits, param_vector)
    )
    basis_values, basis_grads = sym_basis.evaluate_values_and_gradients(
        representatives
    )
    total_orbit_weights = orbit_sizes * orbit_weights
    residual = basis_values @ total_orbit_weights - sym_basis.exact_moments

    num_unknowns = len(param_vector)
    jacobian = np.empty((sym_basis.num_basis, num_unknowns), dtype=float)
    offset = 0
    for idx, orbit in enumerate(orbits):
        dof = orbit.spatial_dof
        if dof > 0:
            # basis_grads[:, :, idx] has shape (dim, num_basis); basis has (dim, dof)
            jacobian[:, offset : offset + dof] = total_orbit_weights[idx] * (
                basis_grads[:, :, idx].T @ orbit.subspace_basis
            )
            offset += dof
        jacobian[:, offset] = (
            2.0 * orbit_sizes[idx] * weight_sqrts[idx]
        ) * basis_values[:, idx]
        offset += 1
    return residual, jacobian, representatives


def evaluate_directional_second_derivative(sym_basis, orbits, param_vector, direction):
    """Compute exact directional 2nd derivative D^2 r(q)[dq, dq] via 2nd-order Taylor jets."""
    num_orbits = len(orbits)
    dimension = sym_basis.dimension
    representatives = np.empty((num_orbits, dimension), dtype=float)
    rep_directions = np.empty((num_orbits, dimension), dtype=float)
    weight_val = np.empty(num_orbits, dtype=float)
    weight_first_deriv = np.empty(num_orbits, dtype=float)
    weight_second_deriv = np.empty(num_orbits, dtype=float)

    offset = 0
    for idx, orbit in enumerate(orbits):
        dof = orbit.spatial_dof
        if dof > 0:
            coords = param_vector[offset : offset + dof]
            coords_dir = direction[offset : offset + dof]
            representatives[idx] = (
                orbit.subspace_origin + orbit.subspace_basis @ coords
            )
            rep_directions[idx] = orbit.subspace_basis @ coords_dir
        else:
            representatives[idx] = orbit.subspace_origin
            rep_directions[idx] = 0.0
        offset += dof

        weight_sqrt = param_vector[offset]
        weight_sqrt_dir = direction[offset]
        offset += 1
        orbit_size = float(orbit.size)
        weight_val[idx] = orbit_size * (weight_sqrt * weight_sqrt)
        weight_first_deriv[idx] = orbit_size * (2.0 * weight_sqrt * weight_sqrt_dir)
        weight_second_deriv[idx] = orbit_size * (
            2.0 * weight_sqrt_dir * weight_sqrt_dir
        )

    jet_val, jet_first, jet_second = sym_basis.evaluate_second_order_jet(
        representatives, rep_directions
    )
    return (
        jet_second @ weight_val
        + 2.0 * (jet_first @ weight_first_deriv)
        + jet_val @ weight_second_deriv
    )


def evaluate_boundary_penalty(
    sym_basis, orbits, param_vector, target_margin, penalty_weight
):
    """Compute smooth boundary hinge cost, gradient, and exact face-normal Hessian blocks."""
    if penalty_weight <= 0.0 or target_margin is None:
        return 0.0, None, None
    representatives, _, _, orbit_sizes = decode_orbit_representatives(
        orbits, param_vector
    )
    violations, rep_grads, rep_hessians = (
        sym_basis.base_basis.boundary_hinge_penalty(representatives, target_margin)
    )
    if float(violations.max()) <= 0.0:
        zeros = np.zeros(len(param_vector), dtype=float)
        return 0.0, zeros, []

    # Weight each orbit's boundary penalty by orbit size so every node weighs equally
    penalty_cost = (
        0.5
        * penalty_weight
        * float(np.sum((orbit_sizes[:, None] * violations) * violations))
    )
    penalty_grad = np.zeros(len(param_vector), dtype=float)
    penalty_hessian_blocks = []
    has_violation = np.any(violations > 0.0, axis=1)

    offset = 0
    for idx, orbit in enumerate(orbits):
        dof = orbit.spatial_dof
        if dof > 0 and has_violation[idx]:
            # Chain rule onto fixed subspace U: grad = U^T g_x, hess = U^T H_x U
            orbit_scale = penalty_weight * orbit_sizes[idx]
            basis = orbit.subspace_basis
            penalty_grad[offset : offset + dof] = orbit_scale * (
                basis.T @ rep_grads[idx]
            )
            hess_block = orbit_scale * (basis.T @ rep_hessians[idx] @ basis)
            penalty_hessian_blocks.append((offset, dof, hess_block))
        offset += dof + 1
    return penalty_cost, penalty_grad, penalty_hessian_blocks


# ---------------------------------------------------------------------------
# Nonlinear solvers: Geodesic-accelerated Cholesky LM & Minimum-Norm GN
# ---------------------------------------------------------------------------


def polish_minimum_norm_gauss_newton(
    sym_basis, orbits, param_vector, max_steps=5, require_interior=False
):
    """Minimum-norm Gauss-Newton polish in (subspace_coords, weight) to machine floor."""
    residual = evaluate_moment_residual(sym_basis, orbits, param_vector)
    current_norm = float(np.linalg.norm(residual))
    if current_norm < 1e-15:
        return current_norm, param_vector

    params = param_vector
    for _ in range(max_steps):
        representatives, orbit_weights, weight_sqrts, orbit_sizes = (
            decode_orbit_representatives(orbits, params)
        )
        basis_values, basis_grads = sym_basis.evaluate_values_and_gradients(
            representatives
        )
        total_orbit_weights = orbit_sizes * orbit_weights
        num_unknowns = len(params)

        # Build Jacobian wrt (subspace_coords, weight) directly so tiny weights stay well-scaled
        jacobian = np.empty((sym_basis.num_basis, num_unknowns), dtype=float)
        offset = 0
        for idx, orbit in enumerate(orbits):
            dof = orbit.spatial_dof
            if dof > 0:
                jacobian[:, offset : offset + dof] = total_orbit_weights[idx] * (
                    basis_grads[:, :, idx].T @ orbit.subspace_basis
                )
                offset += dof
            jacobian[:, offset] = orbit_sizes[idx] * basis_values[:, idx]
            offset += 1

        gram_matrix = jacobian @ jacobian.T
        try:
            cholesky_factor = cho_factor(gram_matrix, lower=True, check_finite=False)
            weight_space_step = -(
                jacobian.T @ cho_solve(cholesky_factor, residual, check_finite=False)
            )
        except np.linalg.LinAlgError:
            break

        # Apply step to (subspace_coords, weight) and map weight back to sqrt(weight)
        trial_params = params.copy()
        offset = 0
        step_valid = True
        for idx, orbit in enumerate(orbits):
            dof = orbit.spatial_dof
            if dof > 0:
                trial_params[offset : offset + dof] = (
                    params[offset : offset + dof]
                    + weight_space_step[offset : offset + dof]
                )
                offset += dof
            updated_weight = orbit_weights[idx] + float(weight_space_step[offset])
            if updated_weight <= 0.0:
                step_valid = False
                break
            sign_ref = weight_sqrts[idx] if weight_sqrts[idx] != 0.0 else 1.0
            trial_params[offset] = math.copysign(math.sqrt(updated_weight), sign_ref)
            offset += 1

        if not step_valid:
            break

        trial_reps, _, _, _ = decode_orbit_representatives(orbits, trial_params)
        if require_interior and sym_basis.base_basis.boundary_distance(trial_reps) <= 1e-8:
            break

        trial_residual = evaluate_moment_residual(sym_basis, orbits, trial_params)
        trial_norm = float(np.linalg.norm(trial_residual))
        if not math.isfinite(trial_norm) or trial_norm >= current_norm:
            break

        params = trial_params
        residual = trial_residual
        current_norm = trial_norm
        if current_norm < 1e-15:
            break

    return current_norm, params


def solve_geodesic_levenberg_marquardt(
    sym_basis,
    orbits,
    initial_params,
    max_steps=150,
    stagnation_window=25,
    initial_damping=0.2,
    geodesic_threshold=0.05,
    penalty_margin=None,
    penalty_weight=0.0,
    require_interior=False,
    phase_label="P1",
    verbose=False,
):
    """Analytic geodesic-accelerated Cholesky LM with delayed gratification and bold uphill.

    Stopping criteria and logged residuals strictly use the pure moment residual
    (never including the boundary hinge penalty).
    """
    params = initial_params.copy()
    residual, jacobian, representatives = evaluate_residual_and_jacobian(
        sym_basis, orbits, params
    )
    moment_norm = float(np.linalg.norm(residual))
    boundary_dist = sym_basis.base_basis.boundary_distance(representatives)
    penalty_cost, penalty_grad, penalty_hessian_blocks = evaluate_boundary_penalty(
        sym_basis, orbits, params, penalty_margin, penalty_weight
    )
    total_cost = 0.5 * (moment_norm * moment_norm) + penalty_cost

    best_moment_norm = moment_norm
    best_total_cost = total_cost
    best_params = params.copy()

    damping = initial_damping
    diag_indices = np.arange(len(params))
    residual_history = []
    steps_taken = 0
    prev_step = None
    geodesic_active = (
        (moment_norm < geodesic_threshold) if geodesic_threshold > 0.0 else True
    )

    for iteration in range(1, max_steps + 1):
        steps_taken = iteration
        # Pure moment stopping criterion (ignores boundary penalty!)
        if moment_norm < 1e-13 and (penalty_weight <= 0.0 or boundary_dist > 0.0):
            break

        normal_matrix = jacobian.T @ jacobian
        gradient = jacobian.T @ residual
        if penalty_grad is not None:
            gradient = gradient + penalty_grad
            for blk_off, blk_dof, hess_block in penalty_hessian_blocks:
                normal_matrix[
                    blk_off : blk_off + blk_dof, blk_off : blk_off + blk_dof
                ] += hess_block
        diag_scale = np.maximum(np.diag(normal_matrix), 1e-12)

        step_accepted = False
        for _ in range(12):
            damped_matrix = normal_matrix.copy()
            damped_matrix[diag_indices, diag_indices] += damping * diag_scale
            try:
                cholesky_factor = cho_factor(
                    damped_matrix, lower=True, check_finite=False
                )
                velocity_step = cho_solve(
                    cholesky_factor, -gradient, check_finite=False
                )
            except np.linalg.LinAlgError:
                damping = min(damping * 5.0, 1e10)
                continue

            trial_step = velocity_step
            if geodesic_active:
                directional_d2 = evaluate_directional_second_derivative(
                    sym_basis, orbits, params, velocity_step
                )
                acceleration_step = cho_solve(
                    cholesky_factor, -(jacobian.T @ directional_d2), check_finite=False
                )
                velocity_norm = float(np.linalg.norm(velocity_step))
                accel_norm = float(np.linalg.norm(acceleration_step))
                if velocity_norm > 0.0 and accel_norm <= 0.75 * velocity_norm:
                    trial_step = velocity_step + 0.5 * acceleration_step

            trial_params = params + trial_step
            trial_reps, _, _, _ = decode_orbit_representatives(orbits, trial_params)
            if not np.all(np.isfinite(trial_reps)):
                damping = min(damping * 5.0, 1e10)
                continue

            trial_boundary_dist = sym_basis.base_basis.boundary_distance(trial_reps)
            if require_interior and trial_boundary_dist <= 1e-7:
                # Gently backtrack along the same step before escalating damping
                backtracked = False
                for _ in range(3):
                    trial_step = 0.5 * trial_step
                    trial_params = params + trial_step
                    trial_reps, _, _, _ = decode_orbit_representatives(
                        orbits, trial_params
                    )
                    if np.all(np.isfinite(trial_reps)):
                        trial_boundary_dist = sym_basis.base_basis.boundary_distance(
                            trial_reps
                        )
                        if trial_boundary_dist > 1e-7:
                            backtracked = True
                            break
                if not backtracked:
                    damping = min(damping * 5.0, 1e10)
                    continue
            # Escape cap prevents runaway far outside the domain
            if trial_boundary_dist < -0.35:
                damping = min(damping * 5.0, 1e10)
                continue

            trial_residual = evaluate_moment_residual(sym_basis, orbits, trial_params)
            trial_moment_norm = float(np.linalg.norm(trial_residual))
            if not math.isfinite(trial_moment_norm):
                damping = min(damping * 5.0, 1e10)
                continue

            trial_penalty_cost, _, _ = evaluate_boundary_penalty(
                sym_basis, orbits, trial_params, penalty_margin, penalty_weight
            )
            trial_total_cost = (
                0.5 * (trial_moment_norm * trial_moment_norm) + trial_penalty_cost
            )

            is_downhill = trial_total_cost < total_cost
            is_bold_uphill_ok = False
            if (not is_downhill) and geodesic_active and prev_step is not None:
                step_norm = float(np.linalg.norm(trial_step))
                prev_norm = float(np.linalg.norm(prev_step))
                if step_norm > 0.0 and prev_norm > 0.0:
                    cos_angle = float(np.dot(trial_step, prev_step)) / (
                        step_norm * prev_norm
                    )
                    if (
                        cos_angle > 0.0
                        and ((1.0 - cos_angle) ** 2) * trial_total_cost
                        <= best_total_cost
                    ):
                        is_bold_uphill_ok = True

            if is_downhill or is_bold_uphill_ok:
                params = trial_params
                residual, jacobian, representatives = evaluate_residual_and_jacobian(
                    sym_basis, orbits, params
                )
                moment_norm = float(np.linalg.norm(residual))
                boundary_dist = sym_basis.base_basis.boundary_distance(representatives)
                penalty_cost, penalty_grad, penalty_hessian_blocks = (
                    evaluate_boundary_penalty(
                        sym_basis, orbits, params, penalty_margin, penalty_weight
                    )
                )
                total_cost = 0.5 * (moment_norm * moment_norm) + penalty_cost
                prev_step = trial_step

                if total_cost < best_total_cost:
                    best_total_cost = total_cost
                    best_moment_norm = moment_norm
                    best_params = params.copy()
                if is_downhill:
                    damping = max(damping / (1.5 if geodesic_active else 3.0), 1e-13)
                if (not geodesic_active) and moment_norm < geodesic_threshold:
                    geodesic_active = True
                step_accepted = True
                break

            damping = min(damping * 5.0, 1e10)

        if not step_accepted:
            break

        residual_history.append(best_moment_norm)
        if verbose and (iteration <= 5 or iteration % 10 == 0 or moment_norm < 1e-13):
            log_message(
                f"{phase_label} it={iteration:3d} mom_res={moment_norm:.3e} "
                f"bdist={boundary_dist:+.3e} lam={damping:.1e} geo={int(geodesic_active)}"
            )
        if moment_norm < 1e-13 and (penalty_weight <= 0.0 or boundary_dist > 0.0):
            break
        if stagnation_window > 0 and len(residual_history) >= stagnation_window:
            if (
                residual_history[-1] > 0.99 * residual_history[-stagnation_window]
                and damping > 1e-2
            ):
                break

    best_reps, _, _, _ = decode_orbit_representatives(orbits, best_params)
    best_boundary_dist = sym_basis.base_basis.boundary_distance(best_reps)
    return best_moment_norm, best_boundary_dist, steps_taken, best_params


def _build_solution_if_positive(
    sym_basis, orbits, params, moment_residual, boundary_distance, quality
):
    """Unpack orbits and construct a CubatureSolution if valid and positive."""
    solved_orbits = unpack_orbit_parameters(orbits, params)
    nodes, weights = expand_orbits_to_nodes(solved_orbits, sym_basis.group_transforms)
    # A representative that drifted onto a point with a larger stabilizer has
    # fewer distinct images than its orbit size; the moment equations counted
    # the full size, so the expanded rule would not be exact.
    if len(weights) != sum(orbit.size for orbit in solved_orbits):
        return None
    if float(weights.min()) <= 0.0:
        return None
    return CubatureSolution(
        quality=quality,
        moment_residual=moment_residual,
        boundary_distance=boundary_distance,
        nodes=nodes,
        weights=weights,
        orbits=solved_orbits,
    )


def _pull_and_polish_interior(sym_basis, orbits, start_params, start_bdist, verbose):
    """Pull a near-solution inside with exact face-normal hinge, then polish penalty-free."""
    target_margin = 5e-4 if sym_basis.degree >= 12 else 2e-3
    pulled_params = start_params
    pulled_bdist = start_bdist
    if pulled_bdist <= 1e-5:
        for pull_weight in (10.0, 100.0):
            _, pulled_bdist, _, pulled_params = solve_geodesic_levenberg_marquardt(
                sym_basis,
                orbits,
                pulled_params,
                max_steps=140,
                stagnation_window=25,
                initial_damping=1e-2,
                geodesic_threshold=0.0,
                penalty_margin=target_margin,
                penalty_weight=pull_weight,
                require_interior=False,
                phase_label="P2",
                verbose=verbose,
            )
            if pulled_bdist > 1e-5:
                break

    if pulled_bdist <= 1e-6:
        return None

    polish_res, polish_bdist, _, polish_params = solve_geodesic_levenberg_marquardt(
        sym_basis,
        orbits,
        pulled_params,
        max_steps=150,
        stagnation_window=25,
        initial_damping=1e-3,
        geodesic_threshold=0.0,
        penalty_margin=None,
        penalty_weight=0.0,
        require_interior=True,
        phase_label="P3",
        verbose=verbose,
    )
    if polish_res < 1e-9:
        polish_res, polish_params = polish_minimum_norm_gauss_newton(
            sym_basis, orbits, polish_params, max_steps=5, require_interior=True
        )
        polish_reps, _, _, _ = decode_orbit_representatives(orbits, polish_params)
        polish_bdist = sym_basis.base_basis.boundary_distance(polish_reps)
    if polish_res < 1e-12 and is_inside(polish_bdist):
        return _build_solution_if_positive(
            sym_basis, orbits, polish_params, polish_res, polish_bdist, "PI"
        )
    return None


def solve_from_single_start(sym_basis, orbits, verbose=False):
    """Run PI-preferred / PO-fallback solve on one orbit initialization.

    Stage A: smooth sliding-wall geodesic LM from the interior start (zero cost
    strictly inside; exact rank-1 face-normal wall outside so nodes glide along
    faces without slamming into a hard wall or escaping into a deep PO basin).
    Stage B (if Stage A did not reach PI): free fold-free geodesic LM (records
    any converged PO rule as fallback) followed by exact face-normal boundary
    hinge pull and penalty-free interior polish + minimum-norm Gauss-Newton.
    """
    initial_params = pack_orbit_parameters(orbits)
    boundary_pi_fallback = None

    # Stage A: smooth sliding-wall geodesic LM (zero penalty inside, rank-1 face wall outside)
    wall_res, wall_bdist, _, wall_params = solve_geodesic_levenberg_marquardt(
        sym_basis,
        orbits,
        initial_params,
        max_steps=250,
        stagnation_window=30,
        initial_damping=0.25,
        geodesic_threshold=0.05,
        penalty_margin=0.0,
        penalty_weight=25.0,
        require_interior=False,
        phase_label="PI",
        verbose=verbose,
    )
    if wall_res < 1e-9 and is_inside(wall_bdist):
        wall_res, wall_params = polish_minimum_norm_gauss_newton(
            sym_basis,
            orbits,
            wall_params,
            max_steps=5,
            require_interior=(wall_bdist > 1e-6),
        )
        wall_reps, _, _, _ = decode_orbit_representatives(orbits, wall_params)
        wall_bdist = sym_basis.base_basis.boundary_distance(wall_reps)
    if wall_res < 1e-12 and is_inside(wall_bdist):
        solution = _build_solution_if_positive(
            sym_basis, orbits, wall_params, wall_res, wall_bdist, "PI"
        )
        if solution is not None:
            if wall_bdist > 1e-6:
                return solution
            # Boundary-grazing PI (valid per Atlas convention); still try to pull
            # deeper inside for easier rigorous certification, keeping this as fallback.
            boundary_pi_fallback = solution
    if wall_res < 1e-2:
        pulled_solution = _pull_and_polish_interior(
            sym_basis, orbits, wall_params, wall_bdist, verbose
        )
        if pulled_solution is not None:
            return pulled_solution
    if boundary_pi_fallback is not None:
        return boundary_pi_fallback

    # Stage B / Phase 1: free fold-free geodesic LM (allowed to step outside)
    free_res, free_bdist, _, free_params = solve_geodesic_levenberg_marquardt(
        sym_basis,
        orbits,
        initial_params,
        max_steps=250,
        stagnation_window=30,
        initial_damping=0.25,
        geodesic_threshold=0.05,
        penalty_margin=None,
        penalty_weight=0.0,
        require_interior=False,
        phase_label="P1",
        verbose=verbose,
    )
    if free_res < 1e-9:
        free_res, free_params = polish_minimum_norm_gauss_newton(
            sym_basis, orbits, free_params, max_steps=5, require_interior=False
        )
        free_reps, _, _, _ = decode_orbit_representatives(orbits, free_params)
        free_bdist = sym_basis.base_basis.boundary_distance(free_reps)

    po_fallback = None
    if free_res < 1e-12:
        quality = "PI" if is_inside(free_bdist) else "PO"
        solution = _build_solution_if_positive(
            sym_basis, orbits, free_params, free_res, free_bdist, quality
        )
        if solution is not None:
            if quality == "PI":
                if free_bdist > 1e-6:
                    return solution
                boundary_pi_fallback = solution
            else:
                po_fallback = solution

    # Stage B / Phase 2 & 3: exact face-normal hinge pull + pure interior polish
    if free_res < 1e-2:
        pulled_solution = _pull_and_polish_interior(
            sym_basis, orbits, free_params, free_bdist, verbose
        )
        if pulled_solution is not None:
            return pulled_solution

    return boundary_pi_fallback if boundary_pi_fallback is not None else po_fallback


# ---------------------------------------------------------------------------
# Verification, default point-count inference, orchestration, and Atlas I/O
# ---------------------------------------------------------------------------


def compute_monomial_residual(domain, dimension, degree, nodes, scaled_weights):
    """Compute exact max normalized monomial residual over all |alpha| <= degree."""
    volume = (
        float(2.0 ** dimension)
        if domain == "cube"
        else (1.0 / float(math.factorial(dimension)))
    )
    all_exponents = generate_multi_indices(dimension, degree)

    coord_powers = []
    for axis in range(dimension):
        axis_coords = nodes[:, axis]
        powers = np.ones((degree + 1, len(axis_coords)), dtype=float)
        for exp in range(1, degree + 1):
            powers[exp] = powers[exp - 1] * axis_coords
        coord_powers.append(powers)

    max_error = 0.0
    for exponents in all_exponents:
        weighted_monomial = scaled_weights.copy()
        for axis in range(dimension):
            if exponents[axis] > 0:
                weighted_monomial *= coord_powers[axis][exponents[axis]]
        numerical_integral = float(np.sum(weighted_monomial))

        if domain == "cube":
            if all(exp % 2 == 0 for exp in exponents):
                exact_integral = 1.0
                for exp in exponents:
                    exact_integral *= 2.0 / float(exp + 1)
            else:
                exact_integral = 0.0
        else:
            numerator_factorial = 1.0
            for exp in exponents:
                numerator_factorial *= float(math.factorial(exp))
            exact_integral = numerator_factorial / float(
                math.factorial(sum(exponents) + dimension)
            )

        error = abs(numerical_integral - exact_integral) / volume
        if error > max_error:
            max_error = error
    return max_error


def infer_default_point_count(domain, dimension, degree, symmetry_name, sym_basis):
    """Find smallest natural target N near Z respecting Stroud bound and full-dim span."""
    num_full_moments = math.comb(degree + dimension, dimension)
    degree_of_freedom_target = int(
        math.ceil(num_full_moments / float(dimension + 1))
    )
    # Classical Stroud lower bound: N >= comb(floor(degree / 2) + dim, dim)
    stroud_lower_bound = math.comb(degree // 2 + dimension, dimension)
    if domain == "cube" and degree >= 3:
        stroud_lower_bound = max(stroud_lower_bound, 2 * dimension)

    if symmetry_name == "none":
        return max(degree_of_freedom_target, stroud_lower_bound)

    if (
        domain == "cube"
        and degree % 2 == 1
        and symmetry_name in ("central", "chiral")
    ):
        search_start = max(
            stroud_lower_bound,
            int(
                math.ceil(
                    sym_basis.num_basis
                    * (sym_basis.group_order / float(dimension + 1))
                )
            )
            - dimension * 2,
        )
    else:
        simplex_chiral_offset = (
            dimension
            if (domain == "simplex" and degree % 2 == 1 and symmetry_name == "chiral")
            else 0
        )
        search_start = max(
            stroud_lower_bound, degree_of_freedom_target - simplex_chiral_offset
        )

    require_full_dim_orbit = degree >= 2 and dimension >= 2
    search_end = max(degree_of_freedom_target, stroud_lower_bound) + max(
        32, 3 * sym_basis.group_order
    )
    for candidate_n in range(search_start, search_end + 1):
        if domain == "cube":
            orbits = build_cube_equilibrium_start(
                dimension,
                degree,
                candidate_n,
                symmetry_name,
                sym_basis.group_transforms,
                min_unknowns=sym_basis.num_basis,
            )
        else:
            orbits = build_simplex_equilibrium_start(
                dimension,
                degree,
                candidate_n,
                symmetry_name,
                sym_basis.group_transforms,
                min_unknowns=sym_basis.num_basis,
            )
        if orbits is not None:
            if (not require_full_dim_orbit) or any(
                orb.spatial_dof == dimension for orb in orbits
            ):
                return candidate_n
    return max(degree_of_freedom_target, stroud_lower_bound)


def get_atlas_domain_info(domain, dimension):
    """Return (atlas_domain_name, filename_prefix) for a domain and dimension."""
    if domain == "cube":
        cube_names = {1: "interval", 2: "square", 3: "cube"}
        return cube_names.get(dimension, f"cube{dimension}"), f"C{dimension}"
    simplex_names = {1: "simplex1", 2: "triangle", 3: "tetra"}
    return simplex_names.get(dimension, f"simplex{dimension}"), f"T{dimension}"


def solve_for_symmetry(
    base_basis, symmetry_name, target_points, seed=None, verbose=False
):
    """Run deterministic mini-ladder (plus optional seeded jitter) for one symmetry group."""
    group_transforms = build_symmetry_group(
        base_basis.domain, base_basis.dimension, symmetry_name
    )
    if group_transforms is None:
        return None

    sym_basis = SymmetricBasis(base_basis, symmetry_name, group_transforms)
    num_points = (
        target_points
        if target_points > 0
        else infer_default_point_count(
            base_basis.domain,
            base_basis.dimension,
            base_basis.degree,
            symmetry_name,
            sym_basis,
        )
    )
    log_message(
        f"Trying symmetry={symmetry_name} "
        f"(|G|={sym_basis.group_order}, M_G={sym_basis.num_basis}, N={num_points})"
    )

    # Deterministic generic mini-ladder (no per-configuration special cases).
    # Easy cases converge on step 0-1 (pure lattice starts); harder cases engage
    # Halton-seeded G-equivariant Riesz crystals and alternative orbit DOF counts.
    if base_basis.domain == "cube":
        # (inset, rot_deg, norm_order, riesz_steps, riesz_excess, wall_strength, extra_dof, halton_gen)
        lattice_ladder = [
            (0.30, 6.0, 12.0, 0, 2.0, 1.0, 0, False),
            (0.45, 8.0, 12.0, 0, 2.0, 1.0, 0, False),
            (0.25, 4.0, 0.0, 80, 2.0, 1.0, 0, False),
            (0.55, 11.0, 8.0, 100, 2.0, 1.2, 0, True),
            (0.25, 4.0, 0.0, 0, 2.0, 1.0, 2, False),
            (0.40, 11.0, 8.0, 100, 3.0, 0.8, 0, True),
            (0.35, 7.0, 12.0, 80, 1.5, 1.5, 1, False),
            (0.45, 5.0, 10.0, 100, 2.0, 1.4, 2, True),
            (0.30, 13.0, 0.0, 100, 2.5, 1.0, 2, False),
            (0.50, 10.0, 12.0, 100, 2.0, 1.8, 0, True),
        ]
    else:
        # (inset, chiral_twist, unused, riesz_steps, riesz_excess, wall_strength, extra_dof, halton_gen)
        lattice_ladder = [
            (0.50, 0.06, 0.0, 0, 2.0, 1.0, 0, False),
            (1.10, 0.12, 0.0, 0, 2.0, 1.0, 0, False),
            (0.60, 0.08, 0.0, 60, 2.0, 1.0, 0, False),
            (0.50, 0.05, 0.0, 80, 3.0, 1.3, 0, False),
            (0.60, 0.08, 0.0, 100, 2.0, 1.0, 0, True),
            (0.80, 0.14, 0.0, 100, 3.0, 0.7, 0, True),
            (0.50, 0.05, 0.0, 100, 1.5, 1.4, 0, True),
            (1.80, 0.04, 0.0, 0, 2.0, 1.0, 0, False),
            (0.70, 0.10, 0.0, 100, 2.0, 1.0, 0, True),
            (0.90, 0.16, 0.0, 100, 2.5, 1.4, 0, True),
            (0.40, 0.20, 0.0, 100, 3.0, 0.7, 1, True),
            (0.60, 0.18, 0.0, 100, 2.0, 1.6, 2, True),
        ]

    is_low_order_cloud = base_basis.dimension >= 2 and symmetry_name in (
        "none",
        "central",
        "mirror",
    )
    if is_low_order_cloud:
        num_ladder_steps = 16
    elif base_basis.dimension == 1:
        num_ladder_steps = 4
    else:
        num_ladder_steps = len(lattice_ladder)

    best_po_solution = None
    first_valid_orbits = None
    halton_step_counter = 0
    for step_idx in range(num_ladder_steps):
        (
            inset,
            twist_or_angle,
            norm_order,
            riesz_steps,
            riesz_excess,
            wall_strength,
            extra_dof,
            halton_gen,
        ) = lattice_ladder[step_idx % len(lattice_ladder)]
        use_lattice_central = False
        if is_low_order_cloud:
            # Alternate G-equivariant Riesz crystals (even steps) and raw Halton
            # clouds (odd steps); both sweep consecutive skips 0, 1, 2, ...
            riesz_idx = step_idx // 2
            halton_skip = riesz_idx * max(num_points, 19)
            riesz_excess = (2.0, 3.0, 1.5, 2.5)[riesz_idx % 4]
            wall_strength = (1.0, 1.4, 0.7, 1.8)[riesz_idx % 4]
            if step_idx % 2 == 0:
                riesz_steps = 100
                if riesz_idx < 4:
                    inset = 0.80
                    extra_fixed_pairs = 0
                else:
                    extra_fixed_pairs = (riesz_idx - 4) // 2 + 1
            else:
                inset = (0.30, 0.45, 0.25, 0.55)[riesz_idx % 4]
                riesz_steps = 0
                extra_fixed_pairs = riesz_idx // 4
            min_unk_target = sym_basis.num_basis
            halton_gen = False
            if (
                base_basis.domain == "cube"
                and symmetry_name == "central"
                and base_basis.dimension >= 3
                and step_idx in (0, 2, 4, 6)
            ):
                use_lattice_central = True
                riesz_steps = 60 if step_idx >= 4 else 0
        else:
            extra_fixed_pairs = 0
            min_unk_target = sym_basis.num_basis + extra_dof
            if halton_gen:
                halton_skip = halton_step_counter * max(num_points, 19)
                halton_step_counter += 1
            else:
                halton_skip = 0

        if base_basis.domain == "cube":
            orbits = build_cube_equilibrium_start(
                base_basis.dimension,
                base_basis.degree,
                num_points,
                symmetry_name,
                group_transforms,
                boundary_inset=inset,
                rotation_degrees=twist_or_angle,
                norm_order=norm_order,
                min_unknowns=min_unk_target,
                halton_skip=halton_skip,
                riesz_steps=riesz_steps,
                riesz_excess=riesz_excess,
                wall_strength=wall_strength,
                extra_fixed_pairs=extra_fixed_pairs,
                use_lattice_for_central=use_lattice_central,
                halton_generic=halton_gen,
            )
        else:
            orbits = build_simplex_equilibrium_start(
                base_basis.dimension,
                base_basis.degree,
                num_points,
                symmetry_name,
                group_transforms,
                boundary_inset=inset,
                chiral_twist=twist_or_angle,
                min_unknowns=min_unk_target,
                halton_skip=halton_skip,
                riesz_steps=riesz_steps,
                riesz_excess=riesz_excess,
                wall_strength=wall_strength,
                extra_fixed_pairs=extra_fixed_pairs,
                halton_generic=halton_gen,
            )

        if orbits is None:
            if step_idx == 0:
                log_message(
                    f"symmetry={symmetry_name}: N={num_points} infeasible "
                    f"for orbit/unknown count (M_G={sym_basis.num_basis})"
                )
                return None
            continue

        if first_valid_orbits is None:
            first_valid_orbits = orbits
        total_unknowns = sum(orb.num_unknowns for orb in orbits)
        log_message(
            f"symmetry={symmetry_name} ladder[{step_idx}] "
            f"unk={total_unknowns}/{sym_basis.num_basis} (orbits={len(orbits)})"
        )

        solution = solve_from_single_start(sym_basis, orbits, verbose=verbose)
        if solution is not None:
            log_message(
                f"symmetry={symmetry_name} ladder[{step_idx}] -> {solution.quality} "
                f"mom_res={solution.moment_residual:.2e} "
                f"bdist={solution.boundary_distance:+.3e}"
            )
            if solution.quality == "PI":
                return solution, sym_basis, num_points
            if (
                best_po_solution is None
                or solution.boundary_distance > best_po_solution.boundary_distance
            ):
                best_po_solution = solution

    # Optional seeded perturbation stage when --seed is supplied
    if seed is not None and first_valid_orbits is not None:
        rng = np.random.default_rng(seed)
        for trial in range(8):
            jitter_scale = 0.03 if trial < 4 else 0.08
            jittered_orbits = []
            for orb in first_valid_orbits:
                coords = (
                    orb.subspace_coords
                    + rng.normal(0.0, jitter_scale, size=orb.spatial_dof)
                    if orb.spatial_dof > 0
                    else orb.subspace_coords.copy()
                )
                weight_sqrt = orb.weight_sqrt * math.exp(
                    float(rng.normal(0.0, 0.05))
                )
                jittered_orbits.append(
                    SymmetryOrbit(
                        size=orb.size,
                        subspace_origin=orb.subspace_origin,
                        subspace_basis=orb.subspace_basis,
                        subspace_coords=coords,
                        weight_sqrt=weight_sqrt,
                    )
                )

            solution = solve_from_single_start(
                sym_basis, jittered_orbits, verbose=verbose
            )
            if solution is not None:
                log_message(
                    f"symmetry={symmetry_name} seed={seed} trial={trial} -> "
                    f"{solution.quality} mom_res={solution.moment_residual:.2e} "
                    f"bdist={solution.boundary_distance:+.3e}"
                )
                if solution.quality == "PI":
                    return solution, sym_basis, num_points
                if (
                    best_po_solution is None
                    or solution.boundary_distance
                    > best_po_solution.boundary_distance
                ):
                    best_po_solution = solution

    if best_po_solution is not None:
        return best_po_solution, sym_basis, num_points
    return None


def get_symmetry_schedule(domain, dimension, degree):
    """Generic symmetry trial order when --symmetry is omitted."""
    if dimension == 1:
        return ["full", "none"]
    if domain == "cube":
        if degree % 2 == 1:
            return ["chiral", "central", "mirror", "full", "none"]
        return ["demi", "chiral", "mirror", "central", "full", "none"]
    if dimension == 2:
        # In 2D simplex, central (v_0 <-> v_2) is already a reflection symmetry
        return ["chiral", "full", "central", "none"]
    return ["chiral", "full", "mirror", "central", "none"]


def save_atlas_rule_json(
    output_path,
    domain,
    dimension,
    degree,
    solution,
    symmetry_name,
    reproduce_command,
    seed=None,
):
    """Write a converged cubature rule in Atlas-standard JSON format.

    output_path may be None (write <rule id>.json in the current directory),
    an existing directory (write <rule id>.json there) or a file path.
    Returns (output_path, rule_id, monomial_err); output_path is None and
    nothing is written if the monomial residual exceeds RES_TOL.
    """
    atlas_domain, prefix = get_atlas_domain_info(domain, dimension)
    volume = (
        float(2.0 ** dimension)
        if domain == "cube"
        else (1.0 / float(math.factorial(dimension)))
    )
    nodes = solution.nodes
    scaled_weights = solution.weights * volume
    num_points = len(scaled_weights)
    quality = solution.quality
    rule_id = f"CA-{prefix}-{degree:02d}-{num_points}-{quality}-MPC"
    monomial_err = compute_monomial_residual(
        domain, dimension, degree, nodes, scaled_weights
    )
    if not monomial_err <= RES_TOL:
        return None, rule_id, monomial_err
    family = (
        "minimal_pointcount_for_degree_n_PI"
        if quality == "PI"
        else "minimal_pointcount_for_degree_n_PO"
    )
    record = {
        "id": rule_id,
        "family": family,
        "domain": atlas_domain,
        "dimension": dimension,
        "degree": degree,
        "points": int(num_points),
        "quality": quality,
        "symmetry": symmetry_name,
        "origin": "found",
        "method": (
            f"Generic deterministic equilibrium-start solver (domain={domain}, dim={dimension}, "
            f"symmetry={symmetry_name}) with fold-free geodesic-accelerated Cholesky LM, "
            f"smooth boundary hinge pull, and minimum-norm Gauss-Newton end-stage"
        ),
        "author": "Antigravity (AI-assisted)",
        "reproduce": {
            "script": "code/cubature_generic.py",
            "seed": seed,
            "command": reproduce_command,
        },
        "status": "numerical",
        "residual_reported": float(monomial_err),
        "nodes": [[float(coord) for coord in row] for row in nodes],
        "weights": [float(weight) for weight in scaled_weights],
    }
    if output_path is None:
        output_path = f"{rule_id}.json"
    elif os.path.isdir(output_path):
        output_path = os.path.join(output_path, f"{rule_id}.json")
    parent_dir = os.path.dirname(os.path.abspath(output_path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_handle:
        json.dump(record, file_handle, indent=2)
        file_handle.write("\n")
    return output_path, rule_id, monomial_err


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dim", type=int, default=2, help="spatial dimension d >= 1 (default: 2)"
    )
    parser.add_argument(
        "--degree", type=int, default=7, help="polynomial degree n >= 1 (default: 7)"
    )
    parser.add_argument(
        "--domain",
        choices=("simplex", "cube"),
        default="simplex",
        help="reference domain: simplex or cube (default: simplex)",
    )
    parser.add_argument(
        "--symmetry",
        choices=("auto", "none", "central", "mirror", "chiral", "demi", "full"),
        default="auto",
        help="symmetry group (default: auto = try all applicable symmetries)",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=0,
        help="target number of nodes N (default: 0 = automatic near Z)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="optional RNG seed for jittered fallback starts after deterministic mini-ladder",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output JSON file or existing directory (default: <rule id>.json in the current directory)",
    )
    args = parser.parse_args()
    if args.dim < 1:
        raise ValueError("dim must be >= 1")
    if args.degree < 1:
        raise ValueError("degree must be >= 1")

    start_time = time.time()
    base_basis = (
        CubeBasis(args.dim, args.degree)
        if args.domain == "cube"
        else SimplexBasis(args.dim, args.degree)
    )
    dof_target_z = int(math.ceil(base_basis.num_basis / float(args.dim + 1)))
    log_message(
        f"Start generic solver: domain={args.domain} dim={args.dim} "
        f"degree={args.degree} M={base_basis.num_basis} Z={dof_target_z} "
        f"symmetry={args.symmetry} points={args.points} seed={args.seed}"
    )

    symmetries = (
        get_symmetry_schedule(args.domain, args.dim, args.degree)
        if args.symmetry == "auto"
        else [args.symmetry]
    )
    best_solution = None
    best_symmetry = None

    for symmetry_name in symmetries:
        if (
            args.points == 0
            and symmetry_name == "none"
            and best_solution is not None
            and best_solution.quality == "PI"
            and len(best_solution.weights) <= dof_target_z
        ):
            continue
        outcome = solve_for_symmetry(
            base_basis, symmetry_name, args.points, seed=args.seed, verbose=False
        )
        if outcome is None:
            continue
        solution, _, _ = outcome
        if solution.quality == "PI":
            if (
                best_solution is None
                or best_solution.quality != "PI"
                or len(solution.weights) < len(best_solution.weights)
            ):
                best_solution = solution
                best_symmetry = symmetry_name
            if args.points > 0:
                break
        elif best_solution is None or (
            best_solution.quality != "PI"
            and solution.boundary_distance > best_solution.boundary_distance
        ):
            best_solution = solution
            best_symmetry = symmetry_name

    elapsed = time.time() - start_time
    if best_solution is None:
        log_message(f"No solution found in {elapsed:.2f}s")
        sys.exit(1)

    command_args = [
        "--dim",
        str(args.dim),
        "--degree",
        str(args.degree),
        "--domain",
        args.domain,
        "--symmetry",
        best_symmetry,
        "--points",
        str(len(best_solution.weights)),
    ]
    if args.seed is not None:
        command_args.extend(["--seed", str(args.seed)])
    reproduce_command = "python code/cubature_generic.py " + shlex.join(command_args)

    output_path, rule_id, monomial_err = save_atlas_rule_json(
        args.out,
        args.domain,
        args.dim,
        args.degree,
        best_solution,
        best_symmetry,
        reproduce_command,
        seed=args.seed,
    )
    if output_path is None:
        log_message(
            f"REJECTED {rule_id}: monomial residual {monomial_err:.2e} exceeds "
            f"{RES_TOL:.0e} ({elapsed:.2f}s)"
        )
        sys.exit(1)
    log_message(
        f"FOUND {rule_id} ({best_solution.quality}, sym={best_symmetry}) "
        f"N={len(best_solution.weights)} mom_res={best_solution.moment_residual:.2e} "
        f"monom={monomial_err:.2e} "
        f"min_w={float(best_solution.weights.min() * base_basis.volume):.3e} "
        f"bdist={best_solution.boundary_distance:+.3e} "
        f"wrote={output_path} ({elapsed:.2f}s)"
    )


if __name__ == "__main__":
    main()

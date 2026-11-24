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
- Generic finite symmetry engine (Reynolds projection + SVD invariant basis +
  stabilizer fixed-subspace orbit parameterization) supporting none, central,
  chiral, and full symmetry groups in arbitrary dimension d.
- Deterministic equilibrium starts (rotated D_d* lattice in Chebyshev angle
  space for cubes; concentric geodesic S^d_+ shells for simplices) with
  dynamic-programming orbit selection for exact target point count N.
- Fold-free (x, sqrt(w)^2) geodesic-accelerated Cholesky Levenberg-Marquardt
  (Transtrum & Sethna 2012) with a PI-preferred / PO-fallback pipeline:
  Stage A strictly interior solve -> Stage B free solve -> smooth diagonal
  boundary hinge pull -> interior pure-moment polish + automatic minimum-norm
  Gauss-Newton end-stage. Logging and stopping criteria strictly report the
  pure moment residual (excluding the boundary penalty).
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

import numpy as np
from scipy.linalg import cho_factor, cho_solve


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


def generate_halton_points(num_points, dimension):
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
            index = point_index + 1
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

    def evaluate_values(self, points):
        """Evaluate all basis polynomials at points, returning (num_basis, num_points)."""
        num_points = len(points)
        tables_by_axis = [
            self._legendre_1d_values(points[:, axis])
            for axis in range(self.dimension)
        ]
        values = np.ones((self.num_basis, num_points), dtype=float)
        for axis in range(self.dimension):
            values *= tables_by_axis[axis][self.multi_index_array[:, axis]]
        return values

    def evaluate_values_and_gradients(self, points):
        """Evaluate basis values and coordinate gradients at points."""
        num_points = len(points)
        factors_by_axis = []
        derivs_by_axis = []
        for axis in range(self.dimension):
            axis_vals, axis_derivs = self._legendre_1d_values_and_derivatives(
                points[:, axis]
            )
            exponents = self.multi_index_array[:, axis]
            factors_by_axis.append(axis_vals[exponents])
            derivs_by_axis.append(axis_derivs[exponents])

        values = np.ones((self.num_basis, num_points), dtype=float)
        for axis in range(self.dimension):
            values *= factors_by_axis[axis]

        gradients = np.empty(
            (self.dimension, self.num_basis, num_points), dtype=float
        )
        for axis in range(self.dimension):
            axis_grad = derivs_by_axis[axis].copy()
            for other_axis in range(self.dimension):
                if other_axis != axis:
                    axis_grad *= factors_by_axis[other_axis]
            gradients[axis] = axis_grad
        return values, gradients

    def evaluate_second_order_jet(self, points, directions):
        """Evaluate (value, 1st directional deriv, 2nd directional deriv) along directions."""
        num_points = len(points)
        prod_val = np.ones((self.num_basis, num_points), dtype=float)
        prod_first = np.zeros((self.num_basis, num_points), dtype=float)
        prod_second = np.zeros((self.num_basis, num_points), dtype=float)
        for axis in range(self.dimension):
            axis_val, axis_first, axis_second = self._legendre_1d_second_order_jet(
                points[:, axis], directions[:, axis]
            )
            exponents = self.multi_index_array[:, axis]
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
        """Return hinge violations and gradients d(hinge)/dX for |x_j| > 1 - target_margin."""
        inner_limit = 1.0 - target_margin
        abs_coords = np.abs(points)
        violations = np.maximum(0.0, abs_coords - inner_limit)
        gradients = np.where(
            abs_coords > inner_limit, np.sign(points) * violations, 0.0
        )
        return violations, gradients


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

        # Group unique (axis, alpha_param) pairs so each Jacobi recurrence runs once
        self.unique_alphas_by_axis = []
        self.alpha_slot_by_axis = []
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

        self.exact_moments = np.zeros(self.num_basis, dtype=float)
        self.exact_moments[0] = 1.0

    @staticmethod
    def _homogeneous_jacobi_recurrence_coeffs(degree_k, alpha_param):
        """3-term recurrence coefficients for homogeneous Jacobi polynomials P_k^{(a, 0)}."""
        k_val = float(degree_k)
        a_val = float(alpha_param)
        denominator = 2.0 * k_val * (k_val + a_val) * (2.0 * k_val + a_val - 2.0)
        coeff_shifted = (
            (2.0 * k_val + a_val - 1.0)
            * (2.0 * k_val + a_val)
            * (2.0 * k_val + a_val - 2.0)
            / denominator
        )
        coeff_tail = (2.0 * k_val + a_val - 1.0) * (a_val * a_val) / denominator
        coeff_prev = (
            2.0
            * (k_val + a_val - 1.0)
            * (k_val - 1.0)
            * (2.0 * k_val + a_val)
            / denominator
        )
        return coeff_shifted, coeff_tail, coeff_prev

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

    def _homogeneous_jacobi_values(self, coord, tail_sum, alpha_param, max_degree):
        """Evaluate s^k * P_k^{(alpha, 0)}(2*coord/tail_sum - 1) without division."""
        values = np.empty((max_degree + 1, len(coord)), dtype=float)
        values[0] = 1.0
        if max_degree >= 1:
            values[1] = (alpha_param + 2.0) * coord - tail_sum

        tail_squared = tail_sum * tail_sum
        shifted_coord = 2.0 * coord - tail_sum
        for deg in range(2, max_degree + 1):
            coeff_shifted, coeff_tail, coeff_prev = (
                self._homogeneous_jacobi_recurrence_coeffs(deg, alpha_param)
            )
            linear_factor = coeff_shifted * shifted_coord + coeff_tail * tail_sum
            values[deg] = (
                linear_factor * values[deg - 1]
                - coeff_prev * tail_squared * values[deg - 2]
            )
        return values

    def _homogeneous_jacobi_first_order_jet(
        self, coord, coord_dir, tail_sum, tail_sum_dir, alpha_param, max_degree
    ):
        """Evaluate (value, 1st directional deriv) of homogeneous Jacobi polynomials."""
        values = np.empty((max_degree + 1, len(coord)), dtype=float)
        first_deriv = np.empty((max_degree + 1, len(coord)), dtype=float)
        values[0] = 1.0
        first_deriv[0] = 0.0
        if max_degree >= 1:
            values[1] = (alpha_param + 2.0) * coord - tail_sum
            first_deriv[1] = (alpha_param + 2.0) * coord_dir - tail_sum_dir

        tail_squared = tail_sum * tail_sum
        tail_squared_deriv = 2.0 * tail_sum * tail_sum_dir
        shifted_coord = 2.0 * coord - tail_sum
        shifted_coord_dir = 2.0 * coord_dir - tail_sum_dir

        for deg in range(2, max_degree + 1):
            coeff_shifted, coeff_tail, coeff_prev = (
                self._homogeneous_jacobi_recurrence_coeffs(deg, alpha_param)
            )
            linear_factor = coeff_shifted * shifted_coord + coeff_tail * tail_sum
            linear_factor_dir = (
                coeff_shifted * shifted_coord_dir + coeff_tail * tail_sum_dir
            )
            values[deg] = (
                linear_factor * values[deg - 1]
                - coeff_prev * tail_squared * values[deg - 2]
            )
            first_deriv[deg] = (
                linear_factor_dir * values[deg - 1]
                + linear_factor * first_deriv[deg - 1]
                - coeff_prev
                * (
                    tail_squared_deriv * values[deg - 2]
                    + tail_squared * first_deriv[deg - 2]
                )
            )
        return values, first_deriv

    def _homogeneous_jacobi_second_order_jet(
        self, coord, coord_dir, tail_sum, tail_sum_dir, alpha_param, max_degree
    ):
        """Evaluate (value, 1st, 2nd directional derivs) of homogeneous Jacobi polynomials."""
        values = np.empty((max_degree + 1, len(coord)), dtype=float)
        first_deriv = np.empty((max_degree + 1, len(coord)), dtype=float)
        second_deriv = np.empty((max_degree + 1, len(coord)), dtype=float)
        values[0] = 1.0
        first_deriv[0] = 0.0
        second_deriv[0] = 0.0
        if max_degree >= 1:
            values[1] = (alpha_param + 2.0) * coord - tail_sum
            first_deriv[1] = (alpha_param + 2.0) * coord_dir - tail_sum_dir
            second_deriv[1] = 0.0

        tail_squared = tail_sum * tail_sum
        tail_squared_first_deriv = 2.0 * tail_sum * tail_sum_dir
        tail_squared_second_deriv = 2.0 * tail_sum_dir * tail_sum_dir
        shifted_coord = 2.0 * coord - tail_sum
        shifted_coord_dir = 2.0 * coord_dir - tail_sum_dir

        for deg in range(2, max_degree + 1):
            coeff_shifted, coeff_tail, coeff_prev = (
                self._homogeneous_jacobi_recurrence_coeffs(deg, alpha_param)
            )
            linear_factor = coeff_shifted * shifted_coord + coeff_tail * tail_sum
            linear_factor_dir = (
                coeff_shifted * shifted_coord_dir + coeff_tail * tail_sum_dir
            )
            values[deg] = (
                linear_factor * values[deg - 1]
                - coeff_prev * tail_squared * values[deg - 2]
            )
            first_deriv[deg] = (
                linear_factor_dir * values[deg - 1]
                + linear_factor * first_deriv[deg - 1]
                - coeff_prev
                * (
                    tail_squared_first_deriv * values[deg - 2]
                    + tail_squared * first_deriv[deg - 2]
                )
            )
            second_deriv[deg] = (
                2.0 * linear_factor_dir * first_deriv[deg - 1]
                + linear_factor * second_deriv[deg - 1]
                - coeff_prev
                * (
                    tail_squared_second_deriv * values[deg - 2]
                    + 2.0 * tail_squared_first_deriv * first_deriv[deg - 2]
                    + tail_squared * second_deriv[deg - 2]
                )
            )
        return values, first_deriv, second_deriv

    def _max_axis_degree(self, axis, alpha_param):
        """Maximum polynomial degree needed along axis for a given Jacobi alpha."""
        trailing_sum = max(0, (alpha_param - (self.dimension - 1 - axis)) // 2)
        return self.degree - trailing_sum

    def evaluate_values(self, points):
        """Evaluate all Dubiner basis polynomials at points."""
        num_points = len(points)
        barycentric, tail_sums = self._barycentric_and_tail_sums(points)
        values = np.repeat(
            self.normalization_factors[:, None], num_points, axis=1
        )
        for axis in range(self.dimension):
            unique_alphas = self.unique_alphas_by_axis[axis]
            tables = [
                self._homogeneous_jacobi_values(
                    barycentric[axis],
                    tail_sums[axis],
                    alpha_val,
                    self._max_axis_degree(axis, alpha_val),
                )
                for alpha_val in unique_alphas
            ]
            axis_factor = np.empty((self.num_basis, num_points), dtype=float)
            slots = self.alpha_slot_by_axis[axis]
            exponents = self.multi_index_array[:, axis]
            for slot, table in enumerate(tables):
                mask = slots == slot
                axis_factor[mask] = table[exponents[mask]]
            values *= axis_factor
        return values

    def evaluate_values_and_gradients(self, points):
        """Evaluate Dubiner basis values and Cartesian coordinate gradients at points."""
        num_points = len(points)
        barycentric, tail_sums = self._barycentric_and_tail_sums(points)
        unit_direction = np.ones(num_points, dtype=float)
        zero_direction = np.zeros(num_points, dtype=float)

        # Partial derivatives wrt barycentric u_m and tail sum s_m, then chain rule:
        # du_0/dx_j = -1, du_{j+1}/dx_j = +1; ds_0/dx_j = 0, ds_m/dx_j = 1 if m <= j+1
        factor_values_by_axis = []
        factor_grads = [[None] * self.dimension for _ in range(self.dimension)]

        for axis in range(self.dimension):
            unique_alphas = self.unique_alphas_by_axis[axis]
            slots = self.alpha_slot_by_axis[axis]
            exponents = self.multi_index_array[:, axis]

            tables_val = []
            tables_dcoord = []
            tables_dtail = []
            for alpha_val in unique_alphas:
                max_deg = self._max_axis_degree(axis, alpha_val)
                val_tab, dcoord_tab = self._homogeneous_jacobi_first_order_jet(
                    barycentric[axis],
                    unit_direction,
                    tail_sums[axis],
                    zero_direction,
                    alpha_val,
                    max_deg,
                )
                _, dtail_tab = self._homogeneous_jacobi_first_order_jet(
                    barycentric[axis],
                    zero_direction,
                    tail_sums[axis],
                    unit_direction,
                    alpha_val,
                    max_deg,
                )
                tables_val.append(val_tab)
                tables_dcoord.append(dcoord_tab)
                tables_dtail.append(dtail_tab)

            factor_val = np.empty((self.num_basis, num_points), dtype=float)
            factor_dcoord = np.empty((self.num_basis, num_points), dtype=float)
            factor_dtail = np.empty((self.num_basis, num_points), dtype=float)
            for slot in range(len(unique_alphas)):
                mask = slots == slot
                factor_val[mask] = tables_val[slot][exponents[mask]]
                factor_dcoord[mask] = tables_dcoord[slot][exponents[mask]]
                factor_dtail[mask] = tables_dtail[slot][exponents[mask]]
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
            unique_alphas = self.unique_alphas_by_axis[axis]
            slots = self.alpha_slot_by_axis[axis]
            exponents = self.multi_index_array[:, axis]
            factor_val = np.empty((self.num_basis, num_points), dtype=float)
            factor_first = np.empty((self.num_basis, num_points), dtype=float)
            factor_second = np.empty((self.num_basis, num_points), dtype=float)

            for slot, alpha_val in enumerate(unique_alphas):
                max_deg = self._max_axis_degree(axis, alpha_val)
                val_tab, first_tab, second_tab = (
                    self._homogeneous_jacobi_second_order_jet(
                        barycentric[axis],
                        barycentric_dir[axis],
                        tail_sums[axis],
                        tail_sums_dir[axis],
                        alpha_val,
                        max_deg,
                    )
                )
                mask = slots == slot
                factor_val[mask] = val_tab[exponents[mask]]
                factor_first[mask] = first_tab[exponents[mask]]
                factor_second[mask] = second_tab[exponents[mask]]

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
        """Return hinge violations and gradients d(hinge)/dX for u_j < target_margin."""
        first_bary = 1.0 - np.sum(points, axis=1)
        first_violation = np.maximum(0.0, target_margin - first_bary)
        coord_violations = np.maximum(0.0, target_margin - points)
        # Penalty = 0.5 * (sum v0^2 + sum vx^2) -> d/dX_j = +v0 - vx_j
        gradients = first_violation[:, None] - coord_violations
        violations = np.concatenate(
            [first_violation[:, None], coord_violations], axis=1
        )
        return violations, gradients


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
    if symmetry_name == "central":
        if dimension == 1:
            symmetry_name = "full"
        else:
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
    """Symmetry-reduced orthonormal polynomial basis via Reynolds projection + SVD."""

    def __init__(self, base_basis, symmetry_name, group_transforms):
        self.base_basis = base_basis
        self.domain = base_basis.domain
        self.dimension = base_basis.dimension
        self.degree = base_basis.degree
        self.symmetry_name = symmetry_name
        self.group_transforms = group_transforms
        self.group_order = len(group_transforms)

        if self.group_order == 1:
            self.projection_matrix = np.eye(base_basis.num_basis, dtype=float)
            self.num_basis = base_basis.num_basis
            self.exact_moments = base_basis.exact_moments.copy()
            return

        # Deterministic probe points strictly inside the domain
        num_probes = max(2 * base_basis.num_basis + 16, 40)
        halton = generate_halton_points(num_probes, base_basis.dimension)
        if base_basis.domain == "cube":
            probe_points = 1.9 * halton - 0.95
        else:
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

        left_singular, singular_values, _ = np.linalg.svd(
            reynolds_avg, full_matrices=False
        )
        rank_tol = (
            1e-10 * float(singular_values[0])
            if len(singular_values) > 0
            else 1e-10
        )
        invariant_rank = int(np.sum(singular_values > rank_tol))
        self.projection_matrix = left_singular[:, :invariant_rank].T.copy()
        self.num_basis = invariant_rank
        self.exact_moments = self.projection_matrix @ base_basis.exact_moments

    def evaluate_values(self, points):
        return self.projection_matrix @ self.base_basis.evaluate_values(points)

    def evaluate_values_and_gradients(self, points):
        base_vals, base_grads = self.base_basis.evaluate_values_and_gradients(points)
        values = self.projection_matrix @ base_vals
        gradients = np.empty((self.dimension, self.num_basis, len(points)), dtype=float)
        for axis in range(self.dimension):
            gradients[axis] = self.projection_matrix @ base_grads[axis]
        return values, gradients

    def evaluate_second_order_jet(self, points, directions):
        base_val, base_first, base_second = (
            self.base_basis.evaluate_second_order_jet(points, directions)
        )
        return (
            self.projection_matrix @ base_val,
            self.projection_matrix @ base_first,
            self.projection_matrix @ base_second,
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
    """Deterministic DP selecting orbits with sum(size) == target_points and sum(unk) >= min_unknowns.

    Prefers inner orbits (lower candidate index) and higher unknown count.
    """
    candidate_pool = candidate_orbits[: max(target_points + 64, 160)]
    # DP state: best_by_count[point_count] = (score, total_unknowns, orbit_idx, prev_count)
    best_by_count = {0: (0.0, 0, -1, -1)}
    parent_pointers = []

    for orbit_idx, orbit in enumerate(candidate_pool):
        orbit_size = orbit.size
        orbit_unknowns = orbit.num_unknowns
        rank_penalty = float(orbit_idx)
        next_by_count = dict(best_by_count)
        chosen_prev = {}

        for point_count, (_, unk_sum, _, _) in best_by_count.items():
            next_count = point_count + orbit_size
            if next_count > target_points:
                continue
            next_unknowns = unk_sum + orbit_unknowns
            # Strongly reward reaching min_unknowns, then prefer inner orbits
            next_score = (
                float(min(next_unknowns, min_unknowns + 4)) * 1e5
                + float(next_unknowns) * 10.0
                - rank_penalty
            )
            existing = next_by_count.get(next_count)
            if existing is None or next_score > existing[0]:
                next_by_count[next_count] = (
                    next_score,
                    next_unknowns,
                    orbit_idx,
                    point_count,
                )
                chosen_prev[next_count] = point_count

        best_by_count = next_by_count
        parent_pointers.append(chosen_prev)

    if target_points not in best_by_count:
        return None
    _, total_unknowns, _, _ = best_by_count[target_points]
    if total_unknowns < min_unknowns:
        return None

    selected = []
    remaining_count = target_points
    for orbit_idx in range(len(candidate_pool) - 1, -1, -1):
        if remaining_count in parent_pointers[orbit_idx]:
            selected.append(candidate_pool[orbit_idx])
            remaining_count = parent_pointers[orbit_idx][remaining_count]
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
):
    """Deterministic rotated D_d* Chebyshev-angle lattice start on [-1, 1]^d."""
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
        shifted_a = integer_grid + lattice_shift
        shifted_b = integer_grid + (lattice_shift + 0.5)
        if symmetry_name == "full":
            raw_points = np.vstack([integer_grid, shifted_a, shifted_b])
        else:
            raw_points = np.vstack([shifted_a, shifted_b])
    else:
        raw_points = integer_grid + (0.0 if target_points % 2 == 1 else 0.5)

    # Apply chiral rotation for mirror-free groups while preserving G-symmetry
    if symmetry_name in ("chiral", "central", "none") and dimension >= 2:
        rot_matrix = rotation_around_main_diagonal(dimension, rotation_degrees)
        raw_points = raw_points @ rot_matrix.T

    max_abs_coord = np.max(np.abs(raw_points), axis=1)
    raw_points = raw_points[max_abs_coord <= float(half_width) + 0.6]

    # Close under G to guarantee exact symmetry of the candidate pool
    closed_clouds = [raw_points]
    for transform in group_transforms[1:]:
        closed_clouds.append(
            raw_points @ transform.linear.T + transform.translation[None, :]
        )
    raw_points = np.vstack(closed_clouds)

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

    initialized_orbits = []
    raw_weights = []
    total_weight = 0.0
    for orb in selected:
        coords = np.sin(orb.initial_point * angle_scale)
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


def build_simplex_equilibrium_start(
    dimension,
    degree,
    target_points,
    symmetry_name,
    group_transforms,
    boundary_inset=0.45,
    chiral_twist=0.06,
    min_unknowns=1,
):
    """Deterministic S^d_+ concentric-shell equilibrium start on standard simplex Delta^d."""
    max_level = 1
    while math.comb(max_level + dimension, dimension) < max(
        int(target_points * 1.6), target_points + 16
    ):
        max_level += 1
        if math.comb(max_level + dimension, dimension) > 15000:
            break

    # Include all concentric levels 1..max_level so every stabilizer orbit type exists
    barycentric_shells = []
    inset_eps = max(0.03, min(0.60, boundary_inset / float(max(degree, 2))))
    for level in range(1, max_level + 1):
        bary_grid = generate_simplex_barycentric_lattice(dimension, level)
        bary_interior = (1.0 - inset_eps) * bary_grid + inset_eps / float(
            dimension + 1
        )
        # Map to S^d_+ equilibrium density via sin(pi/2 * bary) -> normalize -> u = s^2
        sphere_coords = np.sin(0.5 * math.pi * bary_interior)
        sphere_coords /= np.linalg.norm(sphere_coords, axis=1, keepdims=True)
        bary_mapped = sphere_coords * sphere_coords

        if symmetry_name in ("chiral", "none") and chiral_twist > 0.0 and dimension >= 2:
            # Keep lower-dimensional fixed axes intact while twisting generic shells
            sorted_bary = np.sort(bary_mapped, axis=1)
            is_generic = np.all(np.diff(sorted_bary, axis=1) > 1e-6, axis=1)
            twisted = bary_mapped.copy()
            twisted[is_generic] = (
                (1.0 - chiral_twist) * bary_mapped[is_generic]
                + chiral_twist * np.roll(bary_mapped[is_generic], 1, axis=1)
            )
            bary_mapped = twisted
        barycentric_shells.append(bary_mapped)

    # Always include the exact centroid explicitly
    centroid_bary = np.full((1, dimension + 1), 1.0 / float(dimension + 1), dtype=float)
    barycentric_shells.append(centroid_bary)

    raw_points = np.vstack(barycentric_shells)[:, 1:]
    closed_clouds = [raw_points]
    for transform in group_transforms[1:]:
        closed_clouds.append(
            raw_points @ transform.linear.T + transform.translation[None, :]
        )
    raw_points = np.vstack(closed_clouds)

    # Sort by distance from centroid so inner shells are preferred
    first_bary = 1.0 - np.sum(raw_points, axis=1, keepdims=True)
    full_bary = np.hstack([first_bary, raw_points])
    centroid_coord = 1.0 / float(dimension + 1)
    centroid_dist = np.linalg.norm(full_bary - centroid_coord, axis=1)
    raw_points = raw_points[np.argsort(centroid_dist)]
    raw_points = raw_points[: max(target_points * 6, 400)]

    candidate_orbits = cluster_points_into_orbits(
        raw_points, group_transforms, tol=1e-4
    )
    candidate_orbits.sort(
        key=lambda orb: float(
            np.linalg.norm(
                np.concatenate(
                    [[1.0 - np.sum(orb.initial_point)], orb.initial_point]
                )
                - centroid_coord
            )
        )
    )

    selected = select_orbits_for_point_count(
        candidate_orbits, target_points, min_unknowns
    )
    if selected is None:
        return None

    initialized_orbits = []
    raw_weights = []
    total_weight = 0.0
    for orb in selected:
        coords = orb.initial_point
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
    """Compute smooth diagonal boundary hinge penalty cost, gradient, and diagonal Hessian."""
    if penalty_weight <= 0.0 or target_margin is None:
        return 0.0, None, None
    representatives, _, _, orbit_sizes = decode_orbit_representatives(
        orbits, param_vector
    )
    violations, rep_grads = sym_basis.base_basis.boundary_hinge_penalty(
        representatives, target_margin
    )
    if float(violations.max()) <= 0.0:
        zeros = np.zeros(len(param_vector), dtype=float)
        return 0.0, zeros, zeros.copy()

    # Weight each orbit's boundary penalty by orbit size so every node weighs equally
    penalty_cost = (
        0.5 * penalty_weight * float(np.sum((orbit_sizes[:, None] * violations) * violations))
    )
    penalty_grad = np.zeros(len(param_vector), dtype=float)
    penalty_diag_hessian = np.zeros(len(param_vector), dtype=float)
    has_violation = np.any(violations > 0.0, axis=1)

    offset = 0
    for idx, orbit in enumerate(orbits):
        dof = orbit.spatial_dof
        if dof > 0 and has_violation[idx]:
            # Chain rule onto fixed subspace: d/dxi = subspace_basis^T @ rep_grads[idx]
            orbit_scale = penalty_weight * orbit_sizes[idx]
            penalty_grad[offset : offset + dof] = orbit_scale * (
                orbit.subspace_basis.T @ rep_grads[idx]
            )
            penalty_diag_hessian[offset : offset + dof] = orbit_scale
        offset += dof + 1
    return penalty_cost, penalty_grad, penalty_diag_hessian


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
    penalty_cost, penalty_grad, penalty_diag_hessian = evaluate_boundary_penalty(
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
            normal_matrix[diag_indices, diag_indices] += penalty_diag_hessian
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
            # Escape cap prevents runaway far outside the domain
            if trial_boundary_dist < -0.35:
                damping = min(damping * 5.0, 1e10)
                continue
            if require_interior and trial_boundary_dist <= 1e-7:
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
                penalty_cost, penalty_grad, penalty_diag_hessian = (
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
    """Unpack orbits and construct a CubatureSolution if all weights are positive."""
    solved_orbits = unpack_orbit_parameters(orbits, params)
    nodes, weights = expand_orbits_to_nodes(solved_orbits, sym_basis.group_transforms)
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


def solve_from_single_start(sym_basis, orbits, verbose=False):
    """Run PI-preferred / PO-fallback solve on one orbit initialization.

    Stage A: strictly-interior geodesic LM from the interior start (prevents
    jumping across the boundary into a PO basin when a nearby PI basin exists).
    Stage B (if Stage A did not reach PI): free fold-free geodesic LM (records
    any converged PO rule as fallback) followed by smooth diagonal boundary
    hinge pull (Phase 2) and penalty-free interior polish (Phase 3) with
    automatic minimum-norm Gauss-Newton end-stage.
    """
    initial_params = pack_orbit_parameters(orbits)

    # Stage A: strictly interior geodesic LM directly from interior start
    interior_res, interior_bdist, _, interior_params = (
        solve_geodesic_levenberg_marquardt(
            sym_basis,
            orbits,
            initial_params,
            max_steps=160,
            stagnation_window=30,
            initial_damping=0.25,
            geodesic_threshold=0.05,
            penalty_margin=None,
            penalty_weight=0.0,
            require_interior=True,
            phase_label="PI",
            verbose=verbose,
        )
    )
    if interior_res < 1e-9:
        interior_res, interior_params = polish_minimum_norm_gauss_newton(
            sym_basis, orbits, interior_params, max_steps=5, require_interior=True
        )
        interior_reps, _, _, _ = decode_orbit_representatives(
            orbits, interior_params
        )
        interior_bdist = sym_basis.base_basis.boundary_distance(interior_reps)
    if interior_res < 1e-12 and interior_bdist > 0.0:
        solution = _build_solution_if_positive(
            sym_basis, orbits, interior_params, interior_res, interior_bdist, "PI"
        )
        if solution is not None:
            return solution

    # Stage B / Phase 1: free fold-free geodesic LM (allowed to step outside)
    free_res, free_bdist, _, free_params = solve_geodesic_levenberg_marquardt(
        sym_basis,
        orbits,
        initial_params,
        max_steps=160,
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
        quality = "PI" if free_bdist > 0.0 else "PO"
        solution = _build_solution_if_positive(
            sym_basis, orbits, free_params, free_res, free_bdist, quality
        )
        if solution is not None:
            if quality == "PI":
                return solution
            po_fallback = solution

    # Stage B / Phase 2: smooth diagonal boundary hinge pull back inside
    if free_res < 1e-2:
        target_margin = 5e-4 if sym_basis.degree >= 12 else 2e-3
        _, pulled_bdist, _, pulled_params = solve_geodesic_levenberg_marquardt(
            sym_basis,
            orbits,
            free_params,
            max_steps=120,
            stagnation_window=25,
            initial_damping=1e-2,
            geodesic_threshold=0.0,
            penalty_margin=target_margin,
            penalty_weight=10.0,
            require_interior=False,
            phase_label="P2",
            verbose=verbose,
        )
        # Stage B / Phase 3: turn penalty OFF once inside; pure-moment polish + GN
        if pulled_bdist > 0.0:
            polish_res, polish_bdist, _, polish_params = (
                solve_geodesic_levenberg_marquardt(
                    sym_basis,
                    orbits,
                    pulled_params,
                    max_steps=120,
                    stagnation_window=25,
                    initial_damping=1e-3,
                    geodesic_threshold=0.0,
                    penalty_margin=None,
                    penalty_weight=0.0,
                    require_interior=True,
                    phase_label="P3",
                    verbose=verbose,
                )
            )
            if polish_res < 1e-9:
                polish_res, polish_params = polish_minimum_norm_gauss_newton(
                    sym_basis, orbits, polish_params, max_steps=5, require_interior=True
                )
                polish_reps, _, _, _ = decode_orbit_representatives(
                    orbits, polish_params
                )
                polish_bdist = sym_basis.base_basis.boundary_distance(polish_reps)
            if polish_res < 1e-12 and polish_bdist > 0.0:
                solution = _build_solution_if_positive(
                    sym_basis, orbits, polish_params, polish_res, polish_bdist, "PI"
                )
                if solution is not None:
                    return solution

    return po_fallback


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

    # Deterministic generic mini-ladder (no per-configuration special cases)
    if base_basis.domain == "cube":
        start_ladder = [
            (0.30, 6.0, 12.0),
            (0.45, 8.0, 12.0),
            (0.25, 4.0, 0.0),
            (0.55, 11.0, 8.0),
        ]
    else:
        start_ladder = [
            (0.50, 0.06, 0.0),
            (1.10, 0.12, 0.0),
            (1.80, 0.04, 0.0),
            (0.30, 0.18, 0.0),
        ]

    best_po_solution = None
    first_valid_orbits = None
    for step_idx, (inset, twist_or_angle, norm_order) in enumerate(start_ladder):
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
                min_unknowns=sym_basis.num_basis,
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
                min_unknowns=sym_basis.num_basis,
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
            return ["chiral", "central", "full", "none"]
        return ["none", "central", "chiral", "full"]
    return ["chiral", "full", "none"]


def save_atlas_rule_json(
    output_path, domain, dimension, degree, solution, symmetry_name, reproduce_command
):
    """Write a converged cubature rule in Atlas-standard JSON format."""
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
            "command": reproduce_command,
        },
        "status": "numerical",
        "residual_reported": float(monomial_err),
        "nodes": [[float(coord) for coord in row] for row in nodes],
        "weights": [float(weight) for weight in scaled_weights],
    }
    if output_path is None:
        output_path = f"{rule_id}.json"
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
        choices=("auto", "none", "central", "chiral", "full"),
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
        None,
        args.domain,
        args.dim,
        args.degree,
        best_solution,
        best_symmetry,
        reproduce_command,
    )
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

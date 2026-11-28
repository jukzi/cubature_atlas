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
  Stage A sliding-wall solve (exponential boundary penalty) -> Stage B free solve ->
  exponential-penalty pull inside -> interior pure-moment polish + automatic minimum-norm
  Gauss-Newton end-stage. Logging and stopping criteria strictly report the
  pure moment residual (excluding the boundary penalty).
- Automatic point count: a generic N ladder from the classical lower bound
  dim P_{floor(n/2)} upward; at every N all applicable symmetry groups whose
  orbits admit N nodes with at least M_G unknowns are tried (largest group
  first) and the first PI rule ends the ladder. Then N - 1, N - 2, ... are
  retried with further rounds of fresh Halton starts while they succeed.
  A PO rule with fewer nodes than the PI rule is written as well.
- Design rules (see code/README.md): rules may depend on domain, dimension
  and degree modulo a small number, never on individual degrees or node
  counts; starts must converge directly (no node removal, no continuation
  from larger rules, no long trial-and-error); the main goal is a rule with
  at most Z nodes in the common configurations, fast. Z = ceil(M / (d + 1));
  on the cube it is the smaller of this and ceil(2 E / (d + 1)), E the
  number of moments of even degree (the count of a centrally symmetric rule),
  see counting_value.
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
from scipy.linalg.blas import dsyrk
from scipy.spatial import cKDTree
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


# Every start tried by solve_for_symmetry is appended here when it is a
# list (main sets it), so that starts can later be compared with the rules.
STARTS_RECORD = None


def _record_start(sym_basis, symmetry_name, num_points, step_idx, orbits):
    """Append one tried start (nodes and weights scaled to the volume) to STARTS_RECORD."""
    if STARTS_RECORD is None:
        return None
    volume = sym_basis.base_basis.volume
    nodes, weights = expand_orbits_to_nodes(orbits, sym_basis.group_transforms)
    entry = {
        "symmetry": symmetry_name,
        "points": int(num_points),
        "ladder_step": int(step_idx),
        "orbits": [[int(orb.size), int(orb.num_unknowns)] for orb in orbits],
        "start_nodes": [[float(c) for c in row] for row in nodes],
        "start_weights": [float(w * volume) for w in weights],
        "result": None,
    }
    STARTS_RECORD.append(entry)
    return entry


def _record_result(entry, sym_basis, solution):
    """Store the rule a recorded start converged to (or leave None)."""
    if entry is None or solution is None:
        return
    volume = sym_basis.base_basis.volume
    entry["result"] = {
        "quality": solution.quality,
        "moment_residual": float(solution.moment_residual),
        "boundary_distance": float(solution.boundary_distance),
        "nodes": [[float(c) for c in row] for row in solution.nodes],
        "weights": [float(w * volume) for w in solution.weights],
    }


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
        self._val_workspace = None
        self._vg_workspace = None
        self._jet_workspace = None

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

    def _evaluate_values_buffered(self, points, active_indices=None):
        """Cache-friendly chunked evaluation reusing output buffers (caller must not retain alias)."""
        num_points = len(points)
        idx_arr = (
            self.multi_index_array
            if active_indices is None
            else self.multi_index_array[active_indices]
        )
        num_selected = len(idx_arr)
        tables_by_axis = [
            self._legendre_1d_values(points[:, axis])
            for axis in range(self.dimension)
        ]
        chunk = min(256, max(1, num_selected))
        ws = self._val_workspace
        if ws is None or ws[0] != (num_selected, num_points, chunk):
            vals_out = np.empty((num_selected, num_points), dtype=float)
            f_buf = np.empty((chunk, num_points), dtype=float)
            ws = ((num_selected, num_points, chunk), vals_out, f_buf)
            self._val_workspace = ws
        else:
            _, vals_out, f_buf = ws

        for start in range(0, num_selected, chunk):
            end = min(start + chunk, num_selected)
            size = end - start
            np.take(
                tables_by_axis[0],
                idx_arr[start:end, 0],
                axis=0,
                out=vals_out[start:end],
            )
            for axis in range(1, self.dimension):
                np.take(
                    tables_by_axis[axis],
                    idx_arr[start:end, axis],
                    axis=0,
                    out=f_buf[:size],
                )
                vals_out[start:end] *= f_buf[:size]
        return vals_out

    def evaluate_values(self, points, active_indices=None):
        """Evaluate all (or active) basis polynomials at points, returning (num_sel, num_points)."""
        return self._evaluate_values_buffered(points, active_indices).copy()

    def _evaluate_values_and_gradients_buffered(self, points, active_indices=None):
        """Cache-friendly chunked value+gradient evaluation reusing output buffers."""
        num_points = len(points)
        idx_arr = (
            self.multi_index_array
            if active_indices is None
            else self.multi_index_array[active_indices]
        )
        num_selected = len(idx_arr)
        axis_vals = []
        axis_derivs = []
        for axis in range(self.dimension):
            vals_1d, derivs_1d = self._legendre_1d_values_and_derivatives(
                points[:, axis]
            )
            axis_vals.append(vals_1d)
            axis_derivs.append(derivs_1d)

        chunk = min(256, max(1, num_selected))
        ws = self._vg_workspace
        if ws is None or ws[0] != (num_selected, num_points, chunk):
            vals_out = np.empty((num_selected, num_points), dtype=float)
            grads_out = np.empty(
                (self.dimension, num_selected, num_points), dtype=float
            )
            f_bufs = [
                np.empty((chunk, num_points), dtype=float)
                for _ in range(self.dimension)
            ]
            d_bufs = [
                np.empty((chunk, num_points), dtype=float)
                for _ in range(self.dimension)
            ]
            ws = ((num_selected, num_points, chunk), vals_out, grads_out, f_bufs, d_bufs)
            self._vg_workspace = ws
        else:
            _, vals_out, grads_out, f_bufs, d_bufs = ws

        dim = self.dimension
        for start in range(0, num_selected, chunk):
            end = min(start + chunk, num_selected)
            size = end - start
            for axis in range(dim):
                exponents = idx_arr[start:end, axis]
                np.take(axis_vals[axis], exponents, axis=0, out=f_bufs[axis][:size])
                np.take(axis_derivs[axis], exponents, axis=0, out=d_bufs[axis][:size])

            np.copyto(vals_out[start:end], f_bufs[0][:size])
            for axis in range(1, dim):
                vals_out[start:end] *= f_bufs[axis][:size]

            for axis in range(dim):
                dest = grads_out[axis, start:end]
                np.copyto(dest, d_bufs[axis][:size])
                for other_axis in range(dim):
                    if other_axis != axis:
                        dest *= f_bufs[other_axis][:size]
        return vals_out, grads_out

    def evaluate_values_and_gradients(self, points, active_indices=None):
        """Evaluate basis values and coordinate gradients at points."""
        vals, grads = self._evaluate_values_and_gradients_buffered(
            points, active_indices
        )
        return vals.copy(), grads.copy()

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

    def evaluate_weighted_second_order_jet(
        self,
        points,
        directions,
        weight_val,
        weight_first,
        weight_second,
        active_indices=None,
    ):
        """Chunked weighted contraction jet_second @ w0 + 2 * jet_first @ w1 + jet_val @ w2."""
        num_points = len(points)
        idx_arr = (
            self.multi_index_array
            if active_indices is None
            else self.multi_index_array[active_indices]
        )
        num_selected = len(idx_arr)
        axis_jets = [
            self._legendre_1d_second_order_jet(points[:, axis], directions[:, axis])
            for axis in range(self.dimension)
        ]

        chunk = min(256, max(1, num_selected))
        ws = self._jet_workspace
        if ws is None or ws[0] != (num_points, chunk):
            pv = np.empty((chunk, num_points), dtype=float)
            pf = np.empty((chunk, num_points), dtype=float)
            ps = np.empty((chunk, num_points), dtype=float)
            fv = np.empty((chunk, num_points), dtype=float)
            ff = np.empty((chunk, num_points), dtype=float)
            fs = np.empty((chunk, num_points), dtype=float)
            tmp = np.empty((chunk, num_points), dtype=float)
            ws = ((num_points, chunk), pv, pf, ps, fv, ff, fs, tmp)
            self._jet_workspace = ws
        else:
            _, pv, pf, ps, fv, ff, fs, tmp = ws

        result = np.empty(num_selected, dtype=float)
        dim = self.dimension
        for start in range(0, num_selected, chunk):
            end = min(start + chunk, num_selected)
            size = end - start
            ex0 = idx_arr[start:end, 0]
            np.take(axis_jets[0][0], ex0, axis=0, out=pv[:size])
            np.take(axis_jets[0][1], ex0, axis=0, out=pf[:size])
            np.take(axis_jets[0][2], ex0, axis=0, out=ps[:size])
            for axis in range(1, dim):
                ex = idx_arr[start:end, axis]
                np.take(axis_jets[axis][0], ex, axis=0, out=fv[:size])
                np.take(axis_jets[axis][1], ex, axis=0, out=ff[:size])
                np.take(axis_jets[axis][2], ex, axis=0, out=fs[:size])
                ps[:size] *= fv[:size]
                np.multiply(pf[:size], ff[:size], out=tmp[:size])
                ps[:size] += 2.0 * tmp[:size]
                np.multiply(pv[:size], fs[:size], out=tmp[:size])
                ps[:size] += tmp[:size]
                pf[:size] *= fv[:size]
                np.multiply(pv[:size], ff[:size], out=tmp[:size])
                pf[:size] += tmp[:size]
                pv[:size] *= fv[:size]
            result[start:end] = (
                ps[:size] @ weight_val
                + 2.0 * (pf[:size] @ weight_first)
                + pv[:size] @ weight_second
            )
        return result

    def boundary_distance(self, points):
        """Minimum distance to boundary (> 0 strictly interior, <= 0 on/outside)."""
        return float(1.0 - np.abs(points).max())

    def boundary_penalty_vanishes(self, points, target_margin):
        """True if no node is beyond a face moved inward by target_margin.

        Then every outside distance in boundary_exponential_penalty is 0 and
        all its values are exactly 0.
        """
        return bool(float(np.max(np.abs(points) - (1.0 - target_margin))) <= 0.0)

    def boundary_exponential_penalty(self, points, target_margin, tau):
        """Return (values, gradients, exact_hessians) of sum over faces exp(a / tau) - 1 - a / tau, a = |x_j| - (1 - margin) > 0."""
        outside = np.maximum(0.0, np.abs(points) - (1.0 - target_margin))
        scaled = np.minimum(outside / tau, 600.0)
        growth = np.exp(scaled)
        values = np.sum(growth - 1.0 - scaled, axis=1)
        slopes = np.where(outside > 0.0, (growth - 1.0) / tau, 0.0)
        gradients = np.sign(points) * slopes
        curvature = np.where(outside > 0.0, growth / (tau * tau), 0.0)
        hessians = np.zeros((len(points), self.dimension, self.dimension), dtype=float)
        diag_axes = np.arange(self.dimension)
        hessians[:, diag_axes, diag_axes] = curvature
        return values, gradients, hessians


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

        # All per-axis recurrence tables stacked into one: slot s of the
        # stacked table belongs to axis stacked_axis[s] and has the Jacobi
        # parameter stacked_alpha[s]. One recurrence then runs for all axes
        # (every number is computed by the same operations as with one table
        # per axis, only with fewer, larger array operations).
        self.stacked_offsets = []
        offset = 0
        for axis in range(dimension):
            self.stacked_offsets.append(offset)
            offset += len(self.unique_alphas_by_axis[axis])
        num_stacked = offset
        self.num_stacked = num_stacked
        self.stacked_axis = np.concatenate(
            [
                np.full(len(self.unique_alphas_by_axis[axis]), axis, dtype=np.intp)
                for axis in range(dimension)
            ]
        )
        self.stacked_alpha = np.concatenate(
            [table[0] for table in self.jacobi_tables_by_axis]
        )
        # coefficient rows of the degrees 2..n, shape (n - 1, slots, 1)
        self.stacked_shift = np.concatenate(
            [table[1] for table in self.jacobi_tables_by_axis], axis=1
        )[2:, :, None]
        self.stacked_tail = np.concatenate(
            [table[2] for table in self.jacobi_tables_by_axis], axis=1
        )[2:, :, None]
        self.stacked_prev = np.concatenate(
            [table[3] for table in self.jacobi_tables_by_axis], axis=1
        )[2:, :, None]
        self.stacked_dlinear_dcoord = 2.0 * self.stacked_shift
        self.stacked_dlinear_dtail = self.stacked_tail - self.stacked_shift
        # Row of basis function i in the flattened stacked table
        # (degree, slot, point) -> (degree * num_stacked + slot, point)
        self.jacobi_rows_by_axis = [
            self.multi_index_array[:, axis].astype(np.intp) * num_stacked
            + self.stacked_offsets[axis]
            + self.alpha_slot_by_axis[axis]
            for axis in range(dimension)
        ]

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

    def _stacked_linear_factors(self, coord, tail_sum):
        """Per-degree factors of the stacked recurrence, all degrees at once.

        coord and tail_sum hold one row per stacked slot. Returns
        linear_factor[k] = c_shift * (2 coord - tail) + c_tail * tail and
        prev_tail[k] = c_prev * tail^2 for the degrees k + 2, and tail^2.
        """
        shifted_coord = 2.0 * coord - tail_sum
        tail_squared = tail_sum * tail_sum
        linear_factor = self.stacked_shift * shifted_coord[None, :, :]
        linear_factor += self.stacked_tail * tail_sum[None, :, :]
        prev_tail = self.stacked_prev * tail_squared[None, :, :]
        return linear_factor, prev_tail, tail_squared

    def _stacked_jacobi_values(self, barycentric, tail_sums):
        """Homogeneous Jacobi values of all axes and alphas: (n + 1, slots, points).

        values[k] = linear_factor * values[k - 1] - (c_prev * tail^2) * values[k - 2]
        """
        coord = barycentric.take(self.stacked_axis, axis=0)
        tail_sum = tail_sums.take(self.stacked_axis, axis=0)
        max_deg = self.degree
        values = np.empty((max_deg + 1,) + coord.shape, dtype=float)
        values[0] = 1.0
        if max_deg >= 1:
            values[1] = (self.stacked_alpha[:, None] + 2.0) * coord - tail_sum
        if max_deg < 2:
            return values
        linear_factor, prev_tail, _ = self._stacked_linear_factors(coord, tail_sum)
        scratch = np.empty(coord.shape, dtype=float)
        for deg in range(2, max_deg + 1):
            out = values[deg]
            np.multiply(linear_factor[deg - 2], values[deg - 1], out=out)
            np.multiply(prev_tail[deg - 2], values[deg - 2], out=scratch)
            out -= scratch
        return values

    def _stacked_jacobi_values_and_partials(self, barycentric, tail_sums):
        """Stacked (value, d/dcoord, d/dtail_sum) tables, each (n + 1, slots, points).

        dcoord[k] = (dlin_dcoord * v[k-1] + lin * dcoord[k-1]) - (c_prev * tail^2) * dcoord[k-2]
        dtail[k] = (dlin_dtail * v[k-1] + lin * dtail[k-1])
                   - c_prev * ((2 tail) * v[k-2] + tail^2 * dtail[k-2])
        """
        coord = barycentric.take(self.stacked_axis, axis=0)
        tail_sum = tail_sums.take(self.stacked_axis, axis=0)
        max_deg = self.degree
        shape = (max_deg + 1,) + coord.shape
        values = np.empty(shape, dtype=float)
        dcoord = np.empty(shape, dtype=float)
        dtail = np.empty(shape, dtype=float)
        values[0] = 1.0
        dcoord[0] = 0.0
        dtail[0] = 0.0
        if max_deg >= 1:
            alpha_col = self.stacked_alpha[:, None] + 2.0
            values[1] = alpha_col * coord - tail_sum
            dcoord[1] = alpha_col
            dtail[1] = -1.0
        if max_deg < 2:
            return values, dcoord, dtail
        linear_factor, prev_tail, tail_squared = self._stacked_linear_factors(
            coord, tail_sum
        )
        two_tail = 2.0 * tail_sum
        scratch = np.empty(coord.shape, dtype=float)
        inner = np.empty(coord.shape, dtype=float)
        for deg in range(2, max_deg + 1):
            k = deg - 2
            lin = linear_factor[k]
            prev_val = values[deg - 1]
            out = values[deg]
            np.multiply(lin, prev_val, out=out)
            np.multiply(prev_tail[k], values[deg - 2], out=scratch)
            out -= scratch
            out = dcoord[deg]
            np.multiply(self.stacked_dlinear_dcoord[k], prev_val, out=out)
            np.multiply(lin, dcoord[deg - 1], out=scratch)
            out += scratch
            np.multiply(prev_tail[k], dcoord[deg - 2], out=scratch)
            out -= scratch
            out = dtail[deg]
            np.multiply(self.stacked_dlinear_dtail[k], prev_val, out=out)
            np.multiply(lin, dtail[deg - 1], out=scratch)
            out += scratch
            np.multiply(two_tail, values[deg - 2], out=inner)
            np.multiply(tail_squared, dtail[deg - 2], out=scratch)
            inner += scratch
            np.multiply(self.stacked_prev[k], inner, out=scratch)
            out -= scratch
        return values, dcoord, dtail

    def _stacked_jacobi_second_order_jet(
        self, barycentric, barycentric_dir, tail_sums, tail_sums_dir
    ):
        """Stacked 2nd-order Taylor jet (f, f', f'') tables along a direction.

        first[k] = (lin' * v[k-1] + lin * f[k-1])
                   - c_prev * ((tail^2)' * v[k-2] + tail^2 * f[k-2])
        second[k] = ((2 lin') * f[k-1] + lin * s[k-1])
                    - c_prev * (((tail^2)'' * v[k-2] + (2 (tail^2)') * f[k-2])
                                + tail^2 * s[k-2])
        with (tail^2)' = (2 tail) * tail', (tail^2)'' = (2 tail') * tail'.
        """
        coord = barycentric.take(self.stacked_axis, axis=0)
        coord_dir = barycentric_dir.take(self.stacked_axis, axis=0)
        tail_sum = tail_sums.take(self.stacked_axis, axis=0)
        tail_sum_dir = tail_sums_dir.take(self.stacked_axis, axis=0)
        max_deg = self.degree
        shape = (max_deg + 1,) + coord.shape
        values = np.empty(shape, dtype=float)
        first_deriv = np.empty(shape, dtype=float)
        second_deriv = np.empty(shape, dtype=float)
        values[0] = 1.0
        first_deriv[0] = 0.0
        second_deriv[0] = 0.0
        if max_deg >= 1:
            alpha_col = self.stacked_alpha[:, None] + 2.0
            values[1] = alpha_col * coord - tail_sum
            first_deriv[1] = alpha_col * coord_dir - tail_sum_dir
            second_deriv[1] = 0.0
        if max_deg < 2:
            return values, first_deriv, second_deriv
        linear_factor, prev_tail, tail_squared = self._stacked_linear_factors(
            coord, tail_sum
        )
        shifted_coord_dir = 2.0 * coord_dir - tail_sum_dir
        linear_factor_dir = self.stacked_shift * shifted_coord_dir[None, :, :]
        linear_factor_dir += self.stacked_tail * tail_sum_dir[None, :, :]
        two_linear_factor_dir = 2.0 * linear_factor_dir
        tail_squared_first_deriv = 2.0 * tail_sum * tail_sum_dir
        two_tail_squared_first_deriv = 2.0 * tail_squared_first_deriv
        tail_squared_second_deriv = 2.0 * tail_sum_dir * tail_sum_dir
        scratch = np.empty(coord.shape, dtype=float)
        inner = np.empty(coord.shape, dtype=float)
        for deg in range(2, max_deg + 1):
            k = deg - 2
            lin = linear_factor[k]
            prev_val = values[deg - 1]
            prev2_val = values[deg - 2]
            prev_first = first_deriv[deg - 1]
            prev2_first = first_deriv[deg - 2]
            c_prev = self.stacked_prev[k]

            out = values[deg]
            np.multiply(lin, prev_val, out=out)
            np.multiply(prev_tail[k], prev2_val, out=scratch)
            out -= scratch

            out = first_deriv[deg]
            np.multiply(linear_factor_dir[k], prev_val, out=out)
            np.multiply(lin, prev_first, out=scratch)
            out += scratch
            np.multiply(tail_squared_first_deriv, prev2_val, out=inner)
            np.multiply(tail_squared, prev2_first, out=scratch)
            inner += scratch
            np.multiply(c_prev, inner, out=scratch)
            out -= scratch

            out = second_deriv[deg]
            np.multiply(two_linear_factor_dir[k], prev_first, out=out)
            np.multiply(lin, second_deriv[deg - 1], out=scratch)
            out += scratch
            np.multiply(tail_squared_second_deriv, prev2_val, out=inner)
            np.multiply(two_tail_squared_first_deriv, prev2_first, out=scratch)
            inner += scratch
            np.multiply(tail_squared, second_deriv[deg - 2], out=scratch)
            inner += scratch
            np.multiply(c_prev, inner, out=scratch)
            out -= scratch
        return values, first_deriv, second_deriv

    def evaluate_values(self, points):
        """Evaluate all Dubiner basis polynomials at points."""
        num_points = len(points)
        barycentric, tail_sums = self._barycentric_and_tail_sums(points)
        table = self._stacked_jacobi_values(barycentric, tail_sums).reshape(-1, num_points)
        values = np.repeat(
            self.normalization_factors[:, None], num_points, axis=1
        )
        for axis in range(self.dimension):
            values *= table.take(self.jacobi_rows_by_axis[axis], axis=0)
        return values

    def evaluate_values_and_gradients(self, points):
        """Evaluate Dubiner basis values and Cartesian coordinate gradients at points."""
        num_points = len(points)
        barycentric, tail_sums = self._barycentric_and_tail_sums(points)

        # Partial derivatives wrt barycentric u_m and tail sum s_m, then chain rule:
        # du_0/dx_j = -1, du_{j+1}/dx_j = +1; ds_0/dx_j = 0, ds_m/dx_j = 1 if m <= j+1
        factor_values_by_axis = []
        factor_dcoord_by_axis = []
        factor_dtail_by_axis = []
        val_tab, dcoord_tab, dtail_tab = self._stacked_jacobi_values_and_partials(
            barycentric, tail_sums
        )
        val_tab = val_tab.reshape(-1, num_points)
        dcoord_tab = dcoord_tab.reshape(-1, num_points)
        dtail_tab = dtail_tab.reshape(-1, num_points)
        for axis in range(self.dimension):
            rows = self.jacobi_rows_by_axis[axis]
            factor_values_by_axis.append(val_tab.take(rows, axis=0))
            factor_dcoord_by_axis.append(dcoord_tab.take(rows, axis=0))
            factor_dtail_by_axis.append(dtail_tab.take(rows, axis=0))

        values = np.repeat(
            self.normalization_factors[:, None], num_points, axis=1
        )
        for axis in range(self.dimension):
            values *= factor_values_by_axis[axis]

        # The chain-rule factors are 0 or +-1, so du * dcoord + ds * dtail is
        # one of -dcoord, dcoord + dtail, dtail (exact, no rounding), and the
        # product term of one (axis, du, ds) is the same for every coordinate
        # that has it: it is computed once (-dcoord as the negated product,
        # which rounds identically).
        term_cache = {}
        gradients = np.empty(
            (self.dimension, self.num_basis, num_points), dtype=float
        )
        for coord_idx in range(self.dimension):
            # The first term is stored, not added to zero (same values; only
            # the sign of an exact zero can differ).
            first_term = True
            for axis in range(self.dimension):
                du_dx = (
                    -1
                    if axis == 0
                    else (1 if axis == coord_idx + 1 else 0)
                )
                ds_dx = (
                    0
                    if axis == 0
                    else (1 if axis <= coord_idx + 1 else 0)
                )
                if du_dx == 0 and ds_dx == 0:
                    continue
                key = (axis, abs(du_dx), ds_dx)
                term = term_cache.get(key)
                if term is None:
                    if du_dx != 0 and ds_dx != 0:
                        axis_deriv = factor_dcoord_by_axis[axis] + factor_dtail_by_axis[axis]
                    elif du_dx != 0:
                        axis_deriv = factor_dcoord_by_axis[axis]
                    else:
                        axis_deriv = factor_dtail_by_axis[axis]
                    term = self.normalization_factors[:, None] * axis_deriv
                    for other_axis in range(self.dimension):
                        if other_axis != axis:
                            term *= factor_values_by_axis[other_axis]
                    term_cache[key] = term
                negate = du_dx < 0 and ds_dx == 0
                if first_term:
                    if negate:
                        np.negative(term, out=gradients[coord_idx])
                    else:
                        gradients[coord_idx] = term
                    first_term = False
                elif negate:
                    gradients[coord_idx] -= term
                else:
                    gradients[coord_idx] += term
            if first_term:
                gradients[coord_idx] = 0.0
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

        # Product rule over the axes, starting from val = normalization and
        # first = second = 0:
        #   val <- val * f
        #   first <- val * f' + first * f
        #   second <- (val * f'' + (2 * first) * f') + second * f
        # The buffers are updated in place in exactly this order, so every
        # number rounds as in the plain expressions. The zero terms of the
        # first axis are left out (same values; only the sign of an exact
        # zero can differ).
        norm_col = self.normalization_factors[:, None]
        prod_val = None
        prod_first = None
        prod_second = None
        scratch = None
        val_tab, first_tab, second_tab = self._stacked_jacobi_second_order_jet(
            barycentric, barycentric_dir, tail_sums, tail_sums_dir
        )
        val_tab = val_tab.reshape(-1, num_points)
        first_tab = first_tab.reshape(-1, num_points)
        second_tab = second_tab.reshape(-1, num_points)
        for axis in range(self.dimension):
            rows = self.jacobi_rows_by_axis[axis]
            factor_val = val_tab.take(rows, axis=0)
            factor_first = first_tab.take(rows, axis=0)
            factor_second = second_tab.take(rows, axis=0)
            if prod_val is None:
                prod_val = norm_col * factor_val
                prod_first = norm_col * factor_first
                prod_second = norm_col * factor_second
                scratch = np.empty_like(prod_val)
                continue
            factor_second *= prod_val
            np.multiply(2.0, prod_first, out=scratch)
            scratch *= factor_first
            factor_second += scratch
            np.multiply(prod_second, factor_val, out=scratch)
            factor_second += scratch
            factor_first *= prod_val
            np.multiply(prod_first, factor_val, out=scratch)
            factor_first += scratch
            prod_val *= factor_val
            prod_first = factor_first
            prod_second = factor_second
        return prod_val, prod_first, prod_second

    def boundary_distance(self, points):
        """Minimum barycentric coordinate (> 0 strictly interior, <= 0 on/outside)."""
        first_bary = 1.0 - np.sum(points, axis=1)
        return float(min(float(first_bary.min()), float(points.min())))

    def boundary_penalty_vanishes(self, points, target_margin):
        """True if no barycentric coordinate is below target_margin.

        Then every outside distance in boundary_exponential_penalty is 0 and
        all its values are exactly 0.
        """
        if float(np.max(np.sum(points, axis=1) - (1.0 - target_margin))) > 0.0:
            return False
        return bool(float(np.max(target_margin - points)) <= 0.0)

    def boundary_exponential_penalty(self, points, target_margin, tau):
        """Return (values, gradients, exact_hessians) of sum over faces exp(a / tau) - 1 - a / tau, a = margin - u_j > 0 for the barycentric coordinates u_j."""
        first_outside = np.maximum(0.0, np.sum(points, axis=1) - (1.0 - target_margin))
        coord_outside = np.maximum(0.0, target_margin - points)
        first_scaled = np.minimum(first_outside / tau, 600.0)
        coord_scaled = np.minimum(coord_outside / tau, 600.0)
        first_growth = np.exp(first_scaled)
        coord_growth = np.exp(coord_scaled)
        values = (first_growth - 1.0 - first_scaled) + np.sum(coord_growth - 1.0 - coord_scaled, axis=1)
        first_slope = np.where(first_outside > 0.0, (first_growth - 1.0) / tau, 0.0)
        coord_slope = np.where(coord_outside > 0.0, (coord_growth - 1.0) / tau, 0.0)
        # outward normals: (1, ..., 1) for u_0 and -e_j for u_{j+1} = x_j
        gradients = first_slope[:, None] - coord_slope
        first_curv = np.where(first_outside > 0.0, first_growth / (tau * tau), 0.0)
        coord_curv = np.where(coord_outside > 0.0, coord_growth / (tau * tau), 0.0)
        hessians = np.zeros((len(points), self.dimension, self.dimension), dtype=float)
        hessians += first_curv[:, None, None]
        diag_axes = np.arange(self.dimension)
        hessians[:, diag_axes, diag_axes] += coord_curv
        return values, gradients, hessians


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
        # Projected basis values of the last few point sets (key: the bytes of
        # the points). The LM solver evaluates the values at a trial point and,
        # if the step is accepted, values and gradients at the same point, then
        # the second-order jet there; the projection of the values is done once.
        self._value_cache = []

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

        num_probes = max(max_block + 16, 40)
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
        eval_fn = getattr(
            self.base_basis, "_evaluate_values_buffered", self.base_basis.evaluate_values
        )
        if self._active_indices is not None:
            values = self._project(eval_fn(points, active_indices=self._active_indices))
        else:
            values = self._project(eval_fn(points))
        self._store_values(points, values)
        return values

    def _cached_values(self, points):
        """Projected values at exactly these points if still cached, else None.

        A deterministic function of the points, so a cached array equals a
        fresh evaluation bit for bit. Callers must not modify it.
        """
        key = points.tobytes()
        for cached_shape, cached_key, cached_values in self._value_cache:
            if cached_shape == points.shape and cached_key == key:
                return cached_values
        return None

    def _store_values(self, points, values):
        self._value_cache.insert(0, (points.shape, points.tobytes(), values))
        del self._value_cache[4:]

    def evaluate_values_and_gradients(self, points):
        if self.group_order == 1:
            return self.base_basis.evaluate_values_and_gradients(points)
        eval_fn = getattr(
            self.base_basis,
            "_evaluate_values_and_gradients_buffered",
            self.base_basis.evaluate_values_and_gradients,
        )
        if self._active_indices is not None:
            base_vals, base_grads = eval_fn(points, active_indices=self._active_indices)
        else:
            base_vals, base_grads = eval_fn(points)
        values = self._cached_values(points)
        if values is None:
            values = self._project(base_vals)
            self._store_values(points, values)
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
        # On the simplex the value part of the jet is computed exactly like
        # the basis values (same recurrence and product order), so a cached
        # projection of the values can stand in for it. (The Legendre jet of
        # the cube rounds its values differently.)
        values = self._cached_values(points) if self.domain == "simplex" else None
        if values is None:
            values = self._project(base_val)
        return (
            values,
            self._project(base_first),
            self._project(base_second),
        )

    def evaluate_weighted_second_order_jet(
        self, points, directions, weight_val, weight_first, weight_second
    ):
        """Contract 2nd-order Taylor jet with orbit weights before symmetry projection."""
        weighted_fn = getattr(self.base_basis, "evaluate_weighted_second_order_jet", None)
        if weighted_fn is not None:
            if self._active_indices is not None:
                vec = weighted_fn(
                    points,
                    directions,
                    weight_val,
                    weight_first,
                    weight_second,
                    active_indices=self._active_indices,
                )
            else:
                vec = weighted_fn(
                    points, directions, weight_val, weight_first, weight_second
                )
            return vec if self.group_order == 1 else self._project(vec)
        jet_val, jet_first, jet_second = self.evaluate_second_order_jet(
            points, directions
        )
        return (
            jet_second @ weight_val
            + 2.0 * (jet_first @ weight_first)
            + jet_val @ weight_second
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


def _group_images(point, group_transforms):
    """Images g(x) of one point under all group elements, shape (|G|, d)."""
    linears, translations = _stacked_transforms(group_transforms)
    return np.matmul(linears, point[None, :, None])[:, :, 0] + translations


_STACKED_TRANSFORMS = []


def unique_row_indices(rows):
    """First index of every distinct row, in lexicographic order of the rows.

    The same indices as np.unique(rows, axis=0, return_index=True)[1] (a
    stable sort of the rows compared column by column, then the first row of
    every run of equal rows), but with np.lexsort instead of a sort of
    structured records, which is much faster.
    """
    if len(rows) == 0:
        return np.zeros(0, dtype=np.intp)
    order = np.lexsort(rows.T[::-1])
    sorted_rows = rows[order]
    first = np.empty(len(rows), dtype=bool)
    first[0] = True
    np.any(sorted_rows[1:] != sorted_rows[:-1], axis=1, out=first[1:])
    return order[first]


def _stacked_transforms(group_transforms):
    """Linear parts (|G|, d, d) and translations (|G|, d) of a group, cached."""
    for cached_group, linears, translations in _STACKED_TRANSFORMS:
        if cached_group is group_transforms:
            return linears, translations
    linears = np.stack([transform.linear for transform in group_transforms])
    translations = np.stack([transform.translation for transform in group_transforms])
    _STACKED_TRANSFORMS.insert(0, (group_transforms, linears, translations))
    del _STACKED_TRANSFORMS[8:]
    return linears, translations


def _distinct_images(all_images, tol):
    """The images in order, each kept if it is at least tol from every kept one."""
    differences = all_images[:, None, :] - all_images[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    kept = []
    for idx in range(len(all_images)):
        if all(float(distances[idx, prev]) >= tol for prev in kept):
            kept.append(idx)
    return [all_images[idx] for idx in kept]


def compute_orbit_subspace(point, group_transforms, tol=1e-7):
    """Compute orbit size, stabilizer fixed subspace (origin, basis), and orbit images."""
    dimension = len(point)
    all_images = _group_images(point, group_transforms)
    stabilizer = [
        transform
        for transform, distance in zip(
            group_transforms, np.linalg.norm(all_images - point[None, :], axis=1)
        )
        if float(distance) < tol
    ]
    images = _distinct_images(all_images, tol)

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
        images = _distinct_images(_group_images(rep_point, group_transforms), 1e-7)
        for image in images:
            all_nodes.append(image)
            all_weights.append(orbit_weight)
    return np.asarray(all_nodes, dtype=float), np.asarray(all_weights, dtype=float)


def cluster_points_into_orbits(points, group_transforms, tol=1e-6):
    """Cluster a symmetric point cloud into distinct G-orbits with fixed subspaces."""
    num_points = len(points)
    if num_points == 0:
        return []
    tree = cKDTree(points)
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
            for neighbor in tree.query_ball_point(image, tol):
                visited[neighbor] = True
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
    # Keep innermost orbits covering ~1.5x target_points plus a reserve of each
    # distinct (size, num_unknowns) orbit type so parity/DOF combinations stay
    # feasible without running DP over thousands of redundant outer orbits.
    point_budget = max(int(target_points * 1.5) + 64, 160)
    candidate_pool = []
    cum_points = 0
    type_counts = {}
    for orbit in candidate_orbits:
        key = (orbit.size, orbit.num_unknowns)
        if cum_points < point_budget or type_counts.get(key, 0) < 12:
            candidate_pool.append(orbit)
            cum_points += orbit.size
            type_counts[key] = type_counts.get(key, 0) + 1
        elif len(candidate_pool) >= max(target_points + 64, 160):
            break
    # State: pareto_by_count[point_count] = {capped_unknowns: min_rank_cost}
    pareto_by_count = {0: {0: 0.0}}
    parent_pointers = []

    for orbit_idx, orbit in enumerate(candidate_pool):
        orbit_size = orbit.size
        orbit_unknowns = orbit.num_unknowns
        # Weight by orbit_size so every point in the selected N-point cloud pays
        # its own radial rank (preventing the DP from dropping 3 inner size-2
        # orbits in favor of 1 outer size-6 orbit solely to have fewer terms).
        rank_cost = float(orbit_size) * float(orbit_idx)
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


def _fill_square_gaps(theta, deficit, orbit_size, grid=161):
    """Add deficit points in angle coordinates as whole orbits into the largest gaps.

    The centre is added first if the deficit is odd and the centre is free.
    Each further orbit goes to the grid point farthest from all points, where
    the boundary counts as a mirror (distance 2 * boundary distance).
    """
    points = [tuple(row) for row in theta]
    centre = (0.5 * math.pi, 0.5 * math.pi)
    has_centre = any(abs(p[0] - centre[0]) < 1e-9 and abs(p[1] - centre[1]) < 1e-9 for p in points)
    if orbit_size > 1 and deficit % orbit_size == 1:
        if has_centre:
            return None
        points.append(centre)
        deficit -= 1
    if deficit % orbit_size != 0:
        return None
    axis = np.linspace(0.0, math.pi, grid)[1:-1]
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
    candidates = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    off_centre = np.hypot(candidates[:, 0] - centre[0], candidates[:, 1] - centre[1]) > 1e-9
    candidates = candidates[off_centre]
    wall = 2.0 * np.min(np.minimum(candidates, math.pi - candidates), axis=1)
    # Distance of every candidate to the nearest point so far, updated with
    # the points of each new orbit only (the minimum is exact, so the result
    # is the same as recomputing it from all points).
    nearest = np.full(len(candidates), np.inf)
    num_seen = 0
    for _ in range(deficit // orbit_size):
        added = np.array(points[num_seen:])
        num_seen = len(points)
        nearest = np.minimum(
            nearest,
            np.sqrt(((candidates[:, None, :] - added[None, :, :]) ** 2).sum(axis=2)).min(axis=1),
        )
        distance = np.minimum(nearest, wall)
        best = candidates[int(np.argmax(distance))]
        dx = best[0] - centre[0]
        dy = best[1] - centre[1]
        if orbit_size == 4:
            images = ((dx, dy), (-dy, dx), (-dx, -dy), (dy, -dx))
        elif orbit_size == 2:
            images = ((dx, dy), (-dx, -dy))
        else:
            images = ((dx, dy),)
        for image_x, image_y in images:
            points.append((centre[0] + image_x, centre[1] + image_y))
    return np.array(points)


def _square_shell_rotation(points, angle):
    """Rotate 2D points about the origin along their concentric square (L_inf shell).

    A point with r = max(|x|, |y|) moves along the square of half side r by the arc
    length angle / (2 pi) * 8 r. For small angles near the middle of a side this equals
    a rigid rotation, but every point keeps its L_inf radius, so a row parallel to a side
    stays at the same distance from that side. The map commutes with the rotation by 90
    degrees and with the point reflection, so C4 and C2 symmetry are kept.
    """
    x = points[:, 0]
    y = points[:, 1]
    r = np.maximum(np.abs(x), np.abs(y))
    safe_r = np.where(r > 0.0, r, 1.0)
    # perimeter coordinate p in [0, 8): right side, top, left side, bottom (counterclockwise)
    u = x / safe_r
    v = y / safe_r
    on_right = (u >= 1.0 - 1e-12) & (v > -1.0 + 1e-12)
    on_top = (~on_right) & (v >= 1.0 - 1e-12)
    on_left = (~on_right) & (~on_top) & (u <= -1.0 + 1e-12)
    p = np.where(on_right, v + 1.0,
                 np.where(on_top, 3.0 - u,
                          np.where(on_left, 5.0 - v, 7.0 + u)))
    p = np.mod(p + 4.0 * angle / math.pi, 8.0)
    side = np.floor(p / 2.0)
    t = p - 2.0 * side - 1.0
    new_u = np.select([side == 0, side == 1, side == 2], [np.ones_like(t), -t, -np.ones_like(t)], t)
    new_v = np.select([side == 0, side == 1, side == 2], [t, np.ones_like(t), -t], -np.ones_like(t))
    out = np.stack([new_u * r, new_v * r], axis=1)
    out[r == 0.0] = 0.0
    return out


def build_square_angle_lattice_start(target_points, group_transforms, min_unknowns, orbit_size=4):
    """C4, C2 or unsymmetric start on the square from a rotated lattice in angle coordinates theta = arccos(-x).

    A polynomial of degree n becomes a cosine polynomial in theta with spectrum
    |k_1| + |k_2| <= n; the 45-degree rotated square (checkerboard) lattice
    h * (a, b), a + b even, is the matching lattice. It is used with
    h = pi / sqrt(2 N), centred on the square (a point at the centre if N is
    odd), and rotated by alpha = 2 delta / pi, where delta is the boundary
    distance of the unrotated lattice (the rotation moves one end of every
    outer row by about delta towards the boundary). Surplus points are removed
    as whole orbits (orbit_size = 4 for C4, 2 for C2, 1 without symmetry,
    where the lattice is also shifted by a quarter point spacing) nearest to the corners
    (the centre first if the surplus is odd); missing points are put as whole
    orbits into the largest gaps (the centre first if the deficit is odd).
    Returns None if the counts do not fit.
    """
    if orbit_size > 1 and target_points % orbit_size not in (0, 1):
        return None
    if orbit_size == 1:
        unknowns = 3 * target_points
    else:
        unknowns = 3 * (target_points // orbit_size) + (target_points % orbit_size)
    if unknowns < min_unknowns:
        return None
    spacing = math.pi / math.sqrt(2.0 * target_points)
    centre_point = (target_points % 2) == 1
    reach = int(math.ceil(math.pi / spacing)) + 3
    grid = np.arange(-reach, reach + 1)
    a_index, b_index = np.meshgrid(grid, grid, indexing="ij")
    a_index = a_index.ravel()
    b_index = b_index.ravel()
    parity = 0 if centre_point else 1
    keep = ((a_index + b_index) % 2) == parity
    lattice = spacing * np.stack([a_index[keep], b_index[keep]], axis=1).astype(float)

    # without symmetry the lattice is also shifted by a quarter point spacing at the
    # centre; the shift of each coordinate shrinks in proportion to its distance from
    # the boundary (full at pi / 2, zero at 0 and pi), so outer points barely move
    shift = np.array([0.25 * math.sqrt(2.0) * spacing, 0.0]) if orbit_size == 1 else np.zeros(2)

    def points_inside(angle, with_shift=True):
        # rotation along concentric squares (L_inf shells) about the centre: every point
        # slides along its shell by the fraction angle / (2 pi) of the shell perimeter, so
        # points near a side move parallel to that side and keep their boundary distance
        theta = _square_shell_rotation(lattice, angle) + 0.5 * math.pi
        inside = np.all((theta > 1e-12) & (theta < math.pi - 1e-12), axis=1)
        theta = theta[inside]
        if with_shift:
            boundary_fraction = np.minimum(theta, math.pi - theta) / (0.5 * math.pi)
            theta = theta + boundary_fraction * shift
        return theta

    unrotated = points_inside(0.0, with_shift=False)
    delta = float(np.min(np.minimum(unrotated, math.pi - unrotated)))
    alpha = 2.0 * delta / math.pi
    theta = points_inside(alpha)
    surplus = len(theta) - target_points
    if surplus < 0:
        theta = _fill_square_gaps(theta, -surplus, orbit_size)
        if theta is None:
            return None
        surplus = 0
    centre_offset = theta - 0.5 * math.pi
    is_centre = np.hypot(centre_offset[:, 0], centre_offset[:, 1]) < 1e-9
    if orbit_size > 1 and surplus % orbit_size == 1 and np.any(is_centre):
        theta = theta[~is_centre]
        surplus -= 1
    if surplus % orbit_size != 0:
        return None
    corner_distance = np.min(
        [
            np.hypot(theta[:, 0] - corner_x, theta[:, 1] - corner_y)
            for corner_x in (0.0, math.pi)
            for corner_y in (0.0, math.pi)
        ],
        axis=0,
    )
    order = np.argsort(corner_distance, kind="stable")
    keep_points = np.ones(len(theta), dtype=bool)
    keep_points[order[:surplus]] = False
    theta = theta[keep_points]
    coords = -np.cos(theta)
    clustered = cluster_points_into_orbits(coords, group_transforms, tol=1e-6)
    if sum(orbit.size for orbit in clustered) != target_points:
        return None
    return _initialize_cube_orbits_from_points(
        [orbit.initial_point for orbit in clustered], group_transforms
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
    angle_lattice=True,
):
    """Deterministic equilibrium start on [-1, 1]^d (Halton/Riesz crystal or rotated D_d*)."""
    if angle_lattice and dimension == 2 and symmetry_name in ("chiral", "central", "none"):
        orbits = build_square_angle_lattice_start(
            target_points,
            group_transforms,
            min_unknowns,
            orbit_size={"chiral": 4, "central": 2, "none": 1}[symmetry_name],
        )
        if orbits is not None:
            return orbits
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
        if symmetry_name in ("demi", "full") or (target_points % 2 == 1):
            # Standard D_d* lattice Z^d U (Z + 1/2)^d covers every stabilizer orbit type
            raw_points = np.vstack([integer_grid, integer_grid + 0.5])
        else:
            # Origin-free shifted lattice for even N: use (1/4)-shift when preserved
            # mod D_d* by G (e.g. S_6 in 3D), otherwise the B_d-invariant edge-midpoint
            # union U_k (Z^d + e_k / 2) so closing under G never doubles lattice density.
            quarter = np.full(dimension, 0.25, dtype=float)
            quarter_preserved = all(
                np.allclose((g.linear @ quarter + g.translation) % 1.0, 0.25)
                or np.allclose((g.linear @ quarter + g.translation) % 1.0, 0.75)
                for g in group_transforms
            )
            if quarter_preserved:
                raw_points = np.vstack([integer_grid + 0.25, integer_grid + 0.75])
            else:
                edge_layers = []
                for axis in range(dimension):
                    shift = np.zeros(dimension, dtype=float)
                    shift[axis] = 0.5
                    edge_layers.append(integer_grid + shift)
                raw_points = np.vstack(edge_layers)
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
    unique_indices = unique_row_indices(np.round(raw_points, decimals=6))
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
            key=lambda orb: (
                round(float(np.max(np.abs(orb.initial_point))), 9),
                float(np.linalg.norm(orb.initial_point)),
            )
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
    eps = max(1.0 / float(max(degree, 2)) ** 2, min(0.18, 0.5 * boundary_inset / float(max(degree, 2))))

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

    inset_eps = max(2.0 / float(max(degree, 2)) ** 2, min(0.60, boundary_inset / float(max(degree, 2))))

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
        unique_indices = unique_row_indices(np.round(full_cloud, decimals=5))
        full_cloud = full_cloud[unique_indices]
        orbs = cluster_points_into_orbits(full_cloud, group_transforms, tol=1e-4)
        orbs.sort(key=lambda orb: centroid_distance(orb.initial_point))
        return orbs

    primary_orbits = collect_orbits_for_cloud(map_level_to_equilibrium(primary_level))

    auxiliary_clouds = [
        np.full((1, dimension), centroid_coord, dtype=float)
    ]
    aux_levels = sorted(
        {
            lvl
            for lvl in (1, 2, 3, 4, primary_level - 1, primary_level + 1, max_level)
            if 1 <= lvl <= max_level and lvl != primary_level
        }
    )
    for level in aux_levels:
        auxiliary_clouds.append(map_level_to_equilibrium(level))
    aux_raw = np.vstack(auxiliary_clouds)
    aux_dists = np.array([centroid_distance(pt) for pt in aux_raw], dtype=float)
    aux_raw = aux_raw[np.argsort(aux_dists)][: max(target_points * 3, 300)]
    auxiliary_orbits = collect_orbits_for_cloud(aux_raw)

    # Combine primary orbits first, then non-duplicate auxiliary orbits using a
    # spatial index over all G-images of primary orbits so high-N 3D simplices
    # stay O(K log K) instead of O(K^2 |G|).
    candidate_orbits = list(primary_orbits)
    if primary_orbits:
        primary_images = np.vstack(
            [
                np.vstack(
                    [
                        transform.linear @ orb.initial_point + transform.translation
                        for transform in group_transforms
                    ]
                )
                for orb in primary_orbits
            ]
        )
        primary_tree = cKDTree(primary_images)
        for aux_orb in auxiliary_orbits:
            nearest_dist, _ = primary_tree.query(aux_orb.initial_point, k=1)
            if float(nearest_dist) >= 1e-3:
                candidate_orbits.append(aux_orb)
    else:
        candidate_orbits.extend(auxiliary_orbits)

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


class OrbitLayout:
    """Position of every orbit in the flat parameter vector, as index arrays.

    Orbits with the same number of spatial unknowns form one group, so that
    decoding the representatives and assembling the Jacobian are a few array
    operations instead of a Python loop over all orbits.
    """

    def __init__(self, orbits):
        num_orbits = len(orbits)
        self.num_orbits = num_orbits
        self.origins = np.array([orbit.subspace_origin for orbit in orbits], dtype=float)
        self.sizes = np.array([float(orbit.size) for orbit in orbits], dtype=float)
        self.weight_columns = np.empty(num_orbits, dtype=np.intp)
        grouped = {}
        offset = 0
        for idx, orbit in enumerate(orbits):
            dof = orbit.spatial_dof
            if dof > 0:
                entry = grouped.setdefault(dof, ([], [], []))
                entry[0].append(idx)
                entry[1].append(orbit.subspace_basis)
                entry[2].append(offset)
            offset += dof
            self.weight_columns[idx] = offset
            offset += 1
        self.num_unknowns = offset
        # (dof, orbit indices, stacked bases (k, d, dof), columns (k, dof))
        self.groups = []
        for dof in sorted(grouped):
            indices, bases, starts = grouped[dof]
            columns = np.asarray(starts, dtype=np.intp)[:, None] + np.arange(dof, dtype=np.intp)[None, :]
            self.groups.append(
                (dof, np.asarray(indices, dtype=np.intp), np.stack(bases), columns)
            )


_ORBIT_LAYOUT_CACHE = []


def orbit_layout(orbits):
    """OrbitLayout of an orbit list, cached for the few lists in use.

    The solver never changes an orbit list after building it; the length and
    the first and last orbit are compared as a guard anyway.
    """
    for cached_orbits, first, last, cached_layout in _ORBIT_LAYOUT_CACHE:
        if (
            cached_orbits is orbits
            and len(orbits) == cached_layout.num_orbits
            and orbits[0] is first
            and orbits[-1] is last
        ):
            return cached_layout
    layout = OrbitLayout(orbits)
    _ORBIT_LAYOUT_CACHE.insert(0, (orbits, orbits[0], orbits[-1], layout))
    del _ORBIT_LAYOUT_CACHE[8:]
    return layout


_DECODE_CACHE = []


def decode_orbit_representatives(orbits, param_vector):
    """Decode flat parameter vector into (representatives, weights, weight_sqrts, sizes).

    The LM solver decodes the same trial vector several times (boundary
    distance, moment residual, boundary penalty); the last few results are
    kept (key: the orbit layout and the bytes of the vector). The decoding is
    a deterministic function of these, so a cached result equals a fresh one
    bit for bit. Callers must not modify the returned arrays.
    """
    layout = orbit_layout(orbits)
    key = param_vector.tobytes()
    for cached_layout, cached_key, cached_result in _DECODE_CACHE:
        if cached_layout is layout and cached_key == key:
            return cached_result
    representatives = layout.origins.copy()
    for dof, indices, bases, columns in layout.groups:
        coords = param_vector[columns]
        # Batched matmul rounds like the per-orbit product basis @ coords.
        shift = np.matmul(bases, coords[:, :, None])[:, :, 0]
        representatives[indices] += shift
    weight_sqrts = param_vector[layout.weight_columns].astype(float)
    orbit_weights = weight_sqrts * weight_sqrts
    result = (representatives, orbit_weights, weight_sqrts, layout.sizes.copy())
    _DECODE_CACHE.insert(0, (layout, key, result))
    del _DECODE_CACHE[4:]
    return result


def assemble_orbit_jacobian(orbits, basis_values, basis_grads, spatial_scale, weight_scale):
    """Jacobian with the columns of pack_orbit_parameters.

    The spatial columns of orbit i are spatial_scale[i] * grad^T U_i (U_i the
    basis of its fixed subspace), its weight column is weight_scale[i] times
    the basis values at the representative.
    """
    layout = orbit_layout(orbits)
    num_rows = basis_values.shape[0]
    jacobian = np.empty((num_rows, layout.num_unknowns), dtype=float)
    dimension = basis_grads.shape[0]
    for dof, indices, bases, columns in layout.groups:
        if dof == 1:
            # Same rounding as the per-orbit product grad^T @ basis.
            block = basis_grads[0][:, indices] * bases[:, 0, 0][None, :]
            for axis in range(1, dimension):
                block += basis_grads[axis][:, indices] * bases[:, axis, 0][None, :]
            block *= spatial_scale[indices][None, :]
            jacobian[:, columns[:, 0]] = block
            continue
        # (k, rows, d) @ (k, d, dof): rounds like the per-orbit grad^T @ basis
        grads = np.ascontiguousarray(basis_grads[:, :, indices].transpose(2, 1, 0))
        block = np.matmul(grads, bases)
        block *= spatial_scale[indices][:, None, None]
        # one scatter of all (orbit, col) columns: jacobian[:, columns[i, c]] = block[i, :, c]
        jacobian[:, columns] = block.transpose(1, 0, 2)
    jacobian[:, layout.weight_columns] = basis_values * weight_scale[None, :]
    return jacobian


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

    jacobian = assemble_orbit_jacobian(
        orbits,
        basis_values,
        basis_grads,
        total_orbit_weights,
        2.0 * orbit_sizes * weight_sqrts,
    )
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

    return sym_basis.evaluate_weighted_second_order_jet(
        representatives,
        rep_directions,
        weight_val,
        weight_first_deriv,
        weight_second_deriv,
    )


def evaluate_boundary_penalty(
    sym_basis, orbits, param_vector, target_margin, penalty_weight
):
    """Boundary penalty cost, gradient and exact Hessian blocks.

    Every node pays weight * tau^2 * (exp(a / tau) - 1 - a / tau) per face, a its
    distance beyond the face moved inward by target_margin, tau = 1 / (n + 1)^2
    (the scale of the outer node shell). Value and slope are 0 at the face, nodes
    inside cost nothing, near the face it equals weight * a^2 / 2, and further out
    it grows exponentially.
    """
    if penalty_weight <= 0.0 or target_margin is None:
        return 0.0, None, None
    representatives, _, _, orbit_sizes = decode_orbit_representatives(
        orbits, param_vector
    )
    if sym_basis.base_basis.boundary_penalty_vanishes(representatives, target_margin):
        # every node is inside by the margin: all penalty values are exactly 0
        zeros = np.zeros(len(param_vector), dtype=float)
        return 0.0, zeros, []
    tau = 1.0 / float((sym_basis.degree + 1) ** 2)
    values, rep_grads, rep_hessians = (
        sym_basis.base_basis.boundary_exponential_penalty(representatives, target_margin, tau)
    )
    if float(values.max()) <= 0.0:
        zeros = np.zeros(len(param_vector), dtype=float)
        return 0.0, zeros, []

    # Weight each orbit's boundary penalty by orbit size so every node weighs equally
    scale = penalty_weight * tau * tau
    penalty_cost = scale * float(np.sum(orbit_sizes * values))
    penalty_grad = np.zeros(len(param_vector), dtype=float)
    penalty_hessian_blocks = []
    has_violation = values > 0.0

    offset = 0
    for idx, orbit in enumerate(orbits):
        dof = orbit.spatial_dof
        if dof > 0 and has_violation[idx]:
            # Chain rule onto fixed subspace U: grad = U^T g_x, hess = U^T H_x U
            orbit_scale = scale * orbit_sizes[idx]
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

        # Build Jacobian wrt (subspace_coords, weight) directly so tiny weights stay well-scaled
        jacobian = assemble_orbit_jacobian(
            orbits, basis_values, basis_grads, total_orbit_weights, orbit_sizes
        )

        gram_matrix = lower_gram_matrix(jacobian.T)
        try:
            cholesky_factor = cho_factor(
                gram_matrix, lower=True, overwrite_a=True, check_finite=False
            )
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


def lower_gram_matrix(matrix):
    """Lower triangle of matrix^T matrix (upper triangle left as zero).

    Uses the symmetric rank-k update of BLAS, which needs half the work of a
    general matrix product. Only the lower triangle is filled, which is all
    that cho_factor(..., lower=True) and np.diag read.
    """
    return dsyrk(1.0, np.asfortranarray(matrix.T), trans=0, lower=1)


def weight_square_null_direction(orbits, params, jacobian, gram=None):
    """Unit descent direction of sum(w^2) in the null space of the moment Jacobian.

    Every node of an orbit carries the weight w = s^2 (s the stored square
    root), so sum(w^2) = sum(size * s^4) with gradient g = 4 * size * s^3 in the
    s entries and 0 in the spatial entries. The null-space part of g is
    mu * (J^T J + mu I)^-1 g for a small ridge mu (exact as mu -> 0), computed
    with one Cholesky factorisation; a small step along it keeps the moments to
    first order. gram is the lower triangle of J^T J if the caller already
    has it (see lower_gram_matrix). Returns None if the projection vanishes.
    """
    gradient = np.zeros(len(params), dtype=float)
    offset = 0
    for orbit in orbits:
        offset += orbit.spatial_dof
        weight_sqrt = params[offset]
        gradient[offset] = 4.0 * orbit.size * weight_sqrt * weight_sqrt * weight_sqrt
        offset += 1
    gradient_size = float(np.linalg.norm(gradient))
    if gradient_size <= 1e-300:
        return None
    if gram is None:
        gram = lower_gram_matrix(jacobian)
    ridge = 1e-8 * (float(np.max(np.diag(gram))) + 1e-300)
    ridged = gram.copy(order="F")
    ridged[np.diag_indices_from(ridged)] += ridge
    try:
        factor = cho_factor(ridged, lower=True, overwrite_a=True, check_finite=False)
    except np.linalg.LinAlgError:
        return None
    projected = ridge * cho_solve(factor, gradient, check_finite=False)
    size = float(np.linalg.norm(projected))
    if not math.isfinite(size) or size <= 1e-8 * gradient_size:
        return None
    return -projected / size


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
    weight_norm_direction=False,
):
    """Analytic geodesic-accelerated Cholesky LM with delayed gratification and bold uphill.

    Stopping criteria and logged residuals strictly use the pure moment residual
    (never including the boundary penalty). With weight_norm_direction each step
    first tries an added descent direction of sum(w^2) projected onto the null
    space of the moment Jacobian: it does not change the moments to first order
    and does not enter the cost or the reported residual.
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
    allow_bold_uphill = True
    snapped_back = False
    geodesic_active = (
        (moment_norm < geodesic_threshold) if geodesic_threshold > 0.0 else True
    )

    def _eval_trial_step(candidate_step):
        cand_step = candidate_step
        cand_params = params + cand_step
        cand_reps, _, _, _ = decode_orbit_representatives(orbits, cand_params)
        if not np.all(np.isfinite(cand_reps)):
            return None
        cand_bdist = sym_basis.base_basis.boundary_distance(cand_reps)
        if require_interior and cand_bdist <= 1e-7:
            backtracked = False
            for _ in range(3):
                cand_step = 0.5 * cand_step
                cand_params = params + cand_step
                cand_reps, _, _, _ = decode_orbit_representatives(orbits, cand_params)
                if np.all(np.isfinite(cand_reps)):
                    cand_bdist = sym_basis.base_basis.boundary_distance(cand_reps)
                    if cand_bdist > 1e-7:
                        backtracked = True
                        break
            if not backtracked:
                return None
        # Escape cap prevents runaway far outside the domain
        if cand_bdist < -0.35:
            return None
        cand_res = evaluate_moment_residual(sym_basis, orbits, cand_params)
        cand_mom = float(np.linalg.norm(cand_res))
        if not math.isfinite(cand_mom):
            return None
        cand_pen, _, _ = evaluate_boundary_penalty(
            sym_basis, orbits, cand_params, penalty_margin, penalty_weight
        )
        cand_cost = 0.5 * (cand_mom * cand_mom) + cand_pen
        return cand_cost, cand_mom, cand_bdist, cand_params, cand_step

    for iteration in range(1, max_steps + 1):
        steps_taken = iteration
        # Pure moment stopping criterion (ignores boundary penalty!)
        if moment_norm < 1e-13 and (penalty_weight <= 0.0 or boundary_dist > 0.0):
            break

        normal_matrix = lower_gram_matrix(jacobian)
        null_direction = None
        if weight_norm_direction:
            null_direction = weight_square_null_direction(
                orbits, params, jacobian, gram=normal_matrix
            )
        gradient = jacobian.T @ residual
        if penalty_grad is not None:
            gradient = gradient + penalty_grad
            for blk_off, blk_dof, hess_block in penalty_hessian_blocks:
                normal_matrix[
                    blk_off : blk_off + blk_dof, blk_off : blk_off + blk_dof
                ] += hess_block
        raw_diag = np.diag(normal_matrix)
        diag_max = float(np.max(raw_diag)) + 1e-300
        diag_scale = np.maximum(raw_diag, 0.0) + 1e-12 * diag_max

        step_accepted = False
        damped_matrix = np.empty_like(normal_matrix, order="F")
        for _ in range(12):
            np.copyto(damped_matrix, normal_matrix)
            damped_matrix[diag_indices, diag_indices] += damping * diag_scale
            try:
                cholesky_factor = cho_factor(
                    damped_matrix, lower=True, overwrite_a=True, check_finite=False
                )
                velocity_step = cho_solve(
                    cholesky_factor, -gradient, check_finite=False
                )
            except np.linalg.LinAlgError:
                damping = min(damping * 5.0, 1e10)
                continue

            trial_step = velocity_step
            used_geodesic = False
            if geodesic_active:
                directional_d2 = evaluate_directional_second_derivative(
                    sym_basis, orbits, params, velocity_step
                )
                acceleration_step = cho_solve(
                    cholesky_factor, -(jacobian.T @ directional_d2), check_finite=False
                )
                velocity_norm = math.sqrt(
                    float(np.dot(diag_scale, velocity_step * velocity_step))
                )
                accel_norm = math.sqrt(
                    float(np.dot(diag_scale, acceleration_step * acceleration_step))
                )
                if velocity_norm > 0.0 and accel_norm <= 0.75 * velocity_norm:
                    trial_step = velocity_step + 0.5 * acceleration_step
                    used_geodesic = True

            candidate = None
            took_null_step = False
            if null_direction is not None:
                velocity_size = float(np.linalg.norm(velocity_step))
                combined = _eval_trial_step(
                    trial_step + (0.5 * velocity_size) * null_direction
                )
                if combined is not None and combined[0] < total_cost:
                    candidate = combined
                    took_null_step = True
            if candidate is None:
                candidate = _eval_trial_step(trial_step)
            if used_geodesic and not took_null_step:
                vel_candidate = _eval_trial_step(velocity_step)
                if candidate is None or (
                    vel_candidate is not None and vel_candidate[0] < candidate[0]
                ):
                    candidate = vel_candidate
            if candidate is None:
                damping = min(damping * 5.0, 1e10)
                continue

            trial_total_cost, _, _, trial_params, trial_step = candidate
            is_downhill = trial_total_cost < total_cost
            is_bold_uphill_ok = False
            if (
                (not is_downhill)
                and geodesic_active
                and allow_bold_uphill
                and prev_step is not None
            ):
                step_norm = math.sqrt(
                    float(np.dot(diag_scale, trial_step * trial_step))
                )
                prev_norm = math.sqrt(float(np.dot(diag_scale, prev_step * prev_step)))
                if step_norm > 0.0 and prev_norm > 0.0:
                    cos_angle = max(
                        -1.0,
                        min(
                            1.0,
                            float(np.dot(diag_scale, trial_step * prev_step))
                            / (step_norm * prev_norm),
                        ),
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

        residual_history.append(best_moment_norm)
        if verbose and (iteration <= 5 or iteration % 10 == 0 or moment_norm < 1e-13):
            log_message(
                f"{phase_label} it={iteration:3d} mom_res={moment_norm:.3e} "
                f"bdist={boundary_dist:+.3e} lam={damping:.1e} geo={int(geodesic_active)}"
            )
        if moment_norm < 1e-13 and (penalty_weight <= 0.0 or boundary_dist > 0.0):
            break

        stagnated = not step_accepted
        if (
            not stagnated
            and stagnation_window > 0
            and len(residual_history) >= stagnation_window
        ):
            prev_res = residual_history[-stagnation_window]
            if (residual_history[-1] > 0.99 * prev_res and damping > 1e-2) or (
                residual_history[-1] > 0.998 * prev_res and residual_history[-1] > 1e-10
            ):
                stagnated = True

        if stagnated:
            if (
                geodesic_active
                and (not snapped_back)
                and total_cost > 1.01 * best_total_cost
            ):
                # Snap back once to best_params and finish with monotone descent
                snapped_back = True
                allow_bold_uphill = False
                params = best_params.copy()
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
                damping = min(damping, 1e-4)
                prev_step = None
                residual_history.clear()
                continue
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
    # Outer quadrature shell scales as O(1/(degree + 1)^2) from the boundary.
    target_margin = min(2e-3, 0.25 / float((sym_basis.degree + 1) ** 2))
    pulled_params = start_params
    pulled_bdist = start_bdist
    if pulled_bdist <= 1e-5:
        log_message(f"  P2 pull inside: start bdist={pulled_bdist:+.2e}, margin={target_margin:.2e}")
        for pull_weight in (1.0, 10.0, 100.0):
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
            log_message(f"  P2 weight {pull_weight:g}: bdist={pulled_bdist:+.2e}")
            if pulled_bdist > 1e-5:
                break
    else:
        log_message(f"  P2 skipped: bdist={pulled_bdist:+.2e} already inside")

    if pulled_bdist <= 1e-6:
        log_message("  P2 failed: nodes stay on or outside the boundary")
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
    log_message(f"  P3 polish inside: res={polish_res:.2e} bdist={polish_bdist:+.2e}")
    if polish_res < 1e-12 and is_inside(polish_bdist):
        return _build_solution_if_positive(
            sym_basis, orbits, polish_params, polish_res, polish_bdist, "PI"
        )
    return None


def solve_from_single_start(sym_basis, orbits, verbose=False):
    """Run PI-preferred / PO-fallback solve on one orbit initialization.

    Stage A: geodesic LM from the interior start with the boundary penalty
    (zero cost inside, exponential growth outside, see evaluate_boundary_penalty),
    so nodes glide along faces instead of slamming into a hard wall or escaping
    into a deep PO basin. If A ends near a rule with nodes on or just beyond the
    boundary: pull inside with the same penalty at a small margin, then
    penalty-free interior polish and minimum-norm Gauss-Newton.
    Stage B (if A did not reach PI): free geodesic LM from the same start
    without penalty (a converged PO rule is kept as fallback), then the same
    pull and polish.
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
        weight_norm_direction=True,
    )
    log_message(f"  A: res={wall_res:.2e} bdist={wall_bdist:+.2e}")
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
                log_message("  result: PI from A")
                return solution
            # Boundary-grazing PI (valid per Atlas convention); still try to pull
            # deeper inside for easier rigorous certification, keeping this as fallback.
            boundary_pi_fallback = solution
    if wall_res < 1e-4:
        pulled_solution = _pull_and_polish_interior(
            sym_basis, orbits, wall_params, wall_bdist, verbose
        )
        if pulled_solution is not None:
            log_message("  result: PI from A after P2/P3")
            return pulled_solution
    if boundary_pi_fallback is not None:
        log_message("  result: PI from A with nodes on the boundary (P2/P3 did not move them inside)")
        return boundary_pi_fallback

    # Stage B / Phase 1: free fold-free geodesic LM from the start (allowed to step outside)
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
        weight_norm_direction=True,
    )
    if free_res < 1e-9:
        free_res, free_params = polish_minimum_norm_gauss_newton(
            sym_basis, orbits, free_params, max_steps=5, require_interior=False
        )
        free_reps, _, _, _ = decode_orbit_representatives(orbits, free_params)
        free_bdist = sym_basis.base_basis.boundary_distance(free_reps)
    log_message(f"  B: res={free_res:.2e} bdist={free_bdist:+.2e}")

    po_fallback = None
    if free_res < 1e-12:
        quality = "PI" if is_inside(free_bdist) else "PO"
        solution = _build_solution_if_positive(
            sym_basis, orbits, free_params, free_res, free_bdist, quality
        )
        if solution is not None:
            if quality == "PI":
                if free_bdist > 1e-6:
                    log_message("  result: PI from B")
                    return solution
                boundary_pi_fallback = solution
            else:
                po_fallback = solution

    # Stage B / Phase 2 & 3: pull inside with the exponential penalty + pure interior polish
    if free_res < 1e-4:
        pulled_solution = _pull_and_polish_interior(
            sym_basis, orbits, free_params, free_bdist, verbose
        )
        if pulled_solution is not None:
            log_message("  result: PI from B after P2/P3")
            return pulled_solution

    if boundary_pi_fallback is not None:
        log_message("  result: PI from B with nodes on the boundary")
        return boundary_pi_fallback
    if po_fallback is not None:
        log_message("  result: PO from B")
        return po_fallback
    log_message("  result: none")
    return None


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


def get_atlas_domain_info(domain, dimension):
    """Return (atlas_domain_name, filename_prefix) for a domain and dimension."""
    if domain == "cube":
        cube_names = {1: "interval", 2: "square", 3: "cube"}
        return cube_names.get(dimension, f"cube{dimension}"), f"C{dimension}"
    simplex_names = {1: "simplex1", 2: "triangle", 3: "tetra"}
    return simplex_names.get(dimension, f"simplex{dimension}"), f"T{dimension}"


def prepare_symmetry(base_basis, symmetry_name):
    """Return (group_transforms, sym_basis) for a symmetry, or None if it does not apply."""
    group_transforms = build_symmetry_group(
        base_basis.domain, base_basis.dimension, symmetry_name
    )
    if group_transforms is None:
        return None
    sym_basis = SymmetricBasis(base_basis, symmetry_name, group_transforms)
    return group_transforms, sym_basis


def point_count_lower_bound(domain, dimension, degree):
    """Classical lower bound for the node count of a degree-n rule (any symmetry).

    Stroud: N >= dim P_{floor(n/2)}; for the centrally symmetric cube and odd n
    the Moeller-type bound 2 dim P_{(n-1)/2} - 1 (or the even-part count) would
    be sharper, but the plain bound is valid for every symmetry.
    """
    return math.comb(degree // 2 + dimension, dimension)


def counting_value(domain, dimension, degree):
    """Counting value Z of the atlas (as on the website).

    Z = ceil(M / (d + 1)) with M = C(n + d, d) moments and d + 1 unknowns per
    node. The cube is point-symmetric: a centrally symmetric rule meets the
    odd moments by symmetry and only the E = sum_{j even <= n} C(j + d - 1,
    d - 1) moments of even degree with node pairs, so there Z is the smaller
    of the two counts, min(ceil(M / (d + 1)), ceil(2 E / (d + 1))).
    """
    moments = math.comb(degree + dimension, dimension)
    z = -(-moments // (dimension + 1))
    if domain == "cube":
        even = 0
        for j in range(0, degree + 1, 2):
            even = even + math.comb(j + dimension - 1, dimension - 1)
        z = min(z, -(-2 * even // (dimension + 1)))
    return z


def start_is_feasible(sym_basis, num_points):
    """True if the default start has N nodes, >= M_G unknowns and passes the stratum test.

    The test only needs the orbit types, so it runs on the unrelaxed default
    start (milliseconds) before any start of the ladder is relaxed or solved.
    Later ladder steps move nodes, put more of them on strata, or ask for up
    to 2 more unknowns; the last case is tested as well before (G, N) is
    skipped.
    """
    base_basis = sym_basis.base_basis
    builder = (
        build_cube_equilibrium_start
        if base_basis.domain == "cube"
        else build_simplex_equilibrium_start
    )
    orbits = builder(
        base_basis.dimension,
        base_basis.degree,
        num_points,
        sym_basis.symmetry_name,
        sym_basis.group_transforms,
        min_unknowns=sym_basis.num_basis,
    )
    if orbits is None:
        return False
    if not stratum_unknowns_suffice(sym_basis, orbits):
        # Ladder steps with extra_dof ask for up to 2 more unknowns, which
        # can move nodes off a stratum; skip only if that start fails too.
        wider = builder(
            base_basis.dimension,
            base_basis.degree,
            num_points,
            sym_basis.symmetry_name,
            sym_basis.group_transforms,
            min_unknowns=sym_basis.num_basis + 2,
        )
        if wider is not None and stratum_unknowns_suffice(sym_basis, wider):
            return True
        log_message(
            f"symmetry={sym_basis.symmetry_name} N={num_points}: skipped "
            f"(too few unknowns outside a symmetry stratum)"
        )
        return False
    return True


def _point_in_affine_subspace(point, origin, basis, tol=1e-7):
    """True if point lies in the affine subspace origin + span(basis)."""
    offset = point - origin
    if basis.shape[1] > 0:
        offset = offset - basis @ (basis.T @ offset)
    return float(np.linalg.norm(offset)) < tol


def _orbit_lies_in_stratum(orbit, stratum, group_transforms):
    """True if some group image of the orbit's fixed subspace lies in the stratum."""
    origin, basis = stratum
    probe_points = [orbit.representative_point()]
    if orbit.spatial_dof > 0:
        direction = np.linspace(0.3, 0.7, orbit.spatial_dof)
        probe_points.append(probe_points[0] + 0.01 * (orbit.subspace_basis @ direction))
    for transform in group_transforms:
        if all(
            _point_in_affine_subspace(
                transform.linear @ point + transform.translation, origin, basis
            )
            for point in probe_points
        ):
            return True
    return False


def stratum_unknowns_suffice(sym_basis, orbits):
    """Necessary orbit-count condition for every symmetry stratum.

    A stratum S is the fixed subspace of a lower-dimensional orbit (an axis,
    a mirror plane, the centre). Invariant polynomials that vanish on S see
    only the orbits outside G.S, so these orbits must carry at least
    M_G - r_S unknowns, where r_S is the rank of the invariant basis restricted
    to S (bounded by dim P_n(S)). Starts that put too many nodes on S cannot
    give a regular solution,
    however long the solver runs; this is the same counting argument as
    unknowns >= M_G, applied to every stratum.
    """
    base_basis = sym_basis.base_basis
    dimension = base_basis.dimension
    degree = base_basis.degree
    num_basis = sym_basis.num_basis
    group_transforms = sym_basis.group_transforms
    strata = []
    for orbit in orbits:
        if orbit.spatial_dof >= dimension:
            continue
        if any(_orbit_lies_in_stratum(orbit, stratum, group_transforms)
               and orbit.spatial_dof == stratum[1].shape[1]
               for stratum, _ in strata):
            continue
        strata.append(((orbit.subspace_origin, orbit.subspace_basis), orbit))
    for (origin, basis), orbit in strata:
        # r_S <= dim P_n(S): the invariant basis restricted to a k-dimensional
        # stratum spans at most the polynomials of degree n in k variables.
        # The bound is exact for mirror axes and only weakens the test
        # elsewhere, so no start that could converge is rejected.
        stratum_dim = basis.shape[1]
        restricted_rank = min(math.comb(degree + stratum_dim, stratum_dim), num_basis)
        outside_unknowns = sum(
            other.num_unknowns
            for other in orbits
            if not _orbit_lies_in_stratum(other, (origin, basis), group_transforms)
        )
        if outside_unknowns < num_basis - restricted_rank:
            return False
    return True


def solve_for_symmetry(
    base_basis,
    symmetry_name,
    target_points,
    seed=None,
    verbose=False,
    max_points=None,
    prepared=None,
    rounds=(1, 1),
    max_starts=None,
):
    """Run deterministic mini-ladder (plus optional seeded jitter) for one symmetry group.

    If ``max_points`` is given, the stage is skipped (returns None) when its
    point count is not strictly smaller, i.e. it could not improve on an
    already-found PI rule with ``max_points`` nodes.
    ``prepared`` is an optional (group_transforms, sym_basis) pair from
    prepare_symmetry, so repeated calls for several N reuse the basis.
    ``rounds`` = (first, last) selects the passes over the start ladder; every
    pass after the first uses fresh Halton starts.
    """
    if prepared is None:
        prepared = prepare_symmetry(base_basis, symmetry_name)
    if prepared is None:
        return None
    group_transforms, sym_basis = prepared
    num_points = target_points
    if max_points is not None and num_points >= max_points:
        log_message(
            f"Skipping symmetry={symmetry_name} (N={num_points} >= "
            f"N={max_points} of PI rule already found)"
        )
        return None
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
            (1.00, 5.0, 0.0, 0, 2.0, 1.0, 0, False),
            (1.20, 7.5, 0.0, 0, 2.0, 1.0, 0, False),
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
    first_step = 0
    if base_basis.dimension >= 2:
        first_step = num_ladder_steps * (rounds[0] - 1)
        num_ladder_steps *= rounds[1]
    if max_starts is not None:
        num_ladder_steps = min(num_ladder_steps, first_step + max_starts)

    best_po_solution = None
    first_valid_orbits = None
    halton_step_counter = 0
    num_rejected_starts = 0
    for step_idx in range(first_step, num_ladder_steps):
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
        if step_idx >= len(lattice_ladder):
            # Later rounds repeat the table with fresh Halton generic orbits.
            halton_gen = True
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
                    extra_fixed_pairs = ((riesz_idx - 4) // 2) % 4 + 1
            else:
                inset = (0.30, 0.45, 0.25, 0.55)[riesz_idx % 4]
                riesz_steps = 0
                extra_fixed_pairs = (riesz_idx // 4) % 4
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
                angle_lattice=(step_idx == 0),
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

        if not stratum_unknowns_suffice(sym_basis, orbits):
            num_rejected_starts += 1
            continue
        if first_valid_orbits is None:
            first_valid_orbits = orbits
        total_unknowns = sum(orb.num_unknowns for orb in orbits)
        log_message(
            f"symmetry={symmetry_name} ladder[{step_idx}] "
            f"unk={total_unknowns}/{sym_basis.num_basis} (orbits={len(orbits)})"
        )

        record_entry = _record_start(
            sym_basis, symmetry_name, num_points, step_idx, orbits
        )
        solution = solve_from_single_start(sym_basis, orbits, verbose=verbose)
        _record_result(record_entry, sym_basis, solution)
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

    if num_rejected_starts > 0:
        log_message(
            f"symmetry={symmetry_name} N={num_points}: {num_rejected_starts} starts "
            f"skipped (too few unknowns outside a symmetry stratum)"
        )

    # Optional seeded perturbation stage when --seed is supplied
    if seed is not None and first_valid_orbits is not None and rounds[0] == 1:
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
    """Symmetries tried when --symmetry is omitted (the N ladder orders them by |G|)."""
    if dimension == 1:
        return ["full", "none"]
    if domain == "cube":
        return ["full", "demi", "chiral", "mirror", "central", "none"]
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


def _better_po(candidate, current):
    """True if PO rule candidate beats current (fewer nodes, then less outside)."""
    if current is None:
        return True
    if len(candidate.weights) != len(current.weights):
        return len(candidate.weights) < len(current.weights)
    return candidate.boundary_distance > current.boundary_distance


def solve_fixed_point_count(base_basis, symmetries, num_points, seed):
    """Try the symmetries in order at a fixed N; stop at the first PI rule."""
    best_po = None
    best_po_symmetry = None
    for symmetry_name in symmetries:
        outcome = solve_for_symmetry(
            base_basis, symmetry_name, num_points, seed=seed, verbose=False
        )
        if outcome is None:
            continue
        solution = outcome[0]
        if solution.quality == "PI":
            return solution, symmetry_name, best_po, best_po_symmetry
        if _better_po(solution, best_po):
            best_po = solution
            best_po_symmetry = symmetry_name
    return None, None, best_po, best_po_symmetry


def solve_point_count_ladder(
    base_basis, symmetries, seed, max_extra_points, start_rounds=1
):
    """Generic point-count ladder: N = lower bound, lower bound + 1, ...

    For every N all symmetries whose orbit structure admits N nodes with at
    least M_G unknowns are tried, larger groups first (fewer unknowns, faster).
    The first PI rule ends the first pass. With start_rounds > 1 a second
    pass tries N - 1, N - 2, ... with further rounds of starts until one fails.
    PO rules met on the way are kept if they have fewer nodes.
    """
    domain = base_basis.domain
    dimension = base_basis.dimension
    degree = base_basis.degree
    prepared_list = []
    for symmetry_name in symmetries:
        prepared = prepare_symmetry(base_basis, symmetry_name)
        if prepared is not None:
            prepared_list.append((symmetry_name, prepared))
    prepared_list.sort(key=lambda item: -item[1][1].group_order)
    lower_bound = point_count_lower_bound(domain, dimension, degree)
    # Symmetric rules can have fewer than Z nodes (see counting_value), since
    # small orbits (centre, axes) carry more unknowns per node than generic
    # ones. The ladder therefore starts at the classical lower bound and lets
    # the orbit feasibility test (unknowns >= M_G) skip impossible (G, N).
    start_n = lower_bound
    dof_target_z = counting_value(domain, dimension, degree)
    if max_extra_points is None:
        max_extra_points = max(8, dof_target_z // 4)
    end_n = max(start_n, dof_target_z) + max_extra_points
    log_message(
        f"Point-count ladder N={start_n}..{end_n} "
        f"(lower bound {lower_bound}, symmetries "
        f"{', '.join(name for name, _ in prepared_list)})"
    )
    best_po = [None, None]

    def try_point_count(num_points, rounds):
        """Try all feasible symmetries at N; return (solution, symmetry) of a PI rule."""
        for symmetry_name, prepared in prepared_list:
            sym_basis = prepared[1]
            if not start_is_feasible(sym_basis, num_points):
                continue
            outcome = solve_for_symmetry(
                base_basis,
                symmetry_name,
                num_points,
                seed=seed,
                verbose=False,
                prepared=prepared,
                rounds=rounds,
            )
            if outcome is None:
                continue
            solution = outcome[0]
            if solution.quality == "PI":
                return solution, symmetry_name
            if _better_po(solution, best_po[0]):
                best_po[0] = solution
                best_po[1] = symmetry_name
        return None, None

    # Pass 1: one round of starts per N, N ascending, up to the first PI rule.
    best_solution = None
    best_symmetry = None
    for num_points in range(start_n, end_n + 1):
        best_solution, best_symmetry = try_point_count(num_points, (1, 1))
        if best_solution is not None:
            break
    # Pass 2: with more start rounds, go down from the PI count found while
    # smaller N still give PI rules (cheap N first, expensive N only if needed).
    if best_solution is not None and start_rounds > 1:
        num_points = len(best_solution.weights) - 1
        while num_points >= start_n:
            log_message(f"Refining N={num_points} with start rounds 2..{start_rounds}")
            solution, symmetry_name = try_point_count(num_points, (2, start_rounds))
            if solution is None:
                break
            best_solution = solution
            best_symmetry = symmetry_name
            num_points -= 1
    return best_solution, best_symmetry, best_po[0], best_po[1]


def expected_point_count(sym_basis):
    """Node count suggested by counting: |G| * M_G / (d + 1), rounded up.

    A generic orbit has |G| nodes and d + 1 unknowns, and the rule must meet
    the M_G invariant moment equations. Without symmetry this is Z. For a
    centrally symmetric domain (cube) and odd degree the odd moments vanish
    for every point-symmetric rule, so M_G and the estimate are well below Z.
    """
    dimension = sym_basis.base_basis.dimension
    return int(math.ceil(sym_basis.group_order * sym_basis.num_basis / float(dimension + 1)))


def solve_single_start(base_basis, symmetries, max_extra_points):
    """One targeted start per symmetry group, no ladder of starts.

    Every applicable group G gets the node count N_G = the smallest N >= max(
    lower bound, |G| M_G / (d + 1)) (C4 and C2 on the square: min(that, Z)) whose default
    orbit start exists and passes
    the stratum test. The groups are tried by increasing N_G (larger group
    first on ties), one start each (ladder step 0); the first PI rule ends the
    search. PO rules are kept as a fallback.
    """
    dimension = base_basis.dimension
    dof_target_z = counting_value(base_basis.domain, dimension, base_basis.degree)
    lower_bound = point_count_lower_bound(base_basis.domain, dimension, base_basis.degree)
    if max_extra_points is None:
        max_extra_points = max(8, dof_target_z // 4)
    plan = []
    for symmetry_name in symmetries:
        prepared = prepare_symmetry(base_basis, symmetry_name)
        if prepared is None:
            continue
        sym_basis = prepared[1]
        start_n = max(lower_bound, expected_point_count(sym_basis))
        if (
            base_basis.domain == "cube"
            and dimension == 2
            and symmetry_name in ("chiral", "central", "none")
        ):
            # The angle lattice start of C4 and C2 on the square reaches the
            # counting value Z for many degrees, so these groups start at Z.
            start_n = max(lower_bound, min(start_n, dof_target_z))
        num_points = None
        for candidate in range(start_n, start_n + max_extra_points + 1):
            if start_is_feasible(sym_basis, candidate):
                num_points = candidate
                break
        if num_points is None:
            log_message(
                f"symmetry={symmetry_name}: no feasible N in "
                f"{start_n}..{start_n + max_extra_points}"
            )
            continue
        plan.append((num_points, -sym_basis.group_order, symmetry_name, prepared))
    plan.sort(key=lambda item: (item[0], item[1]))
    log_message(
        "Single-start plan: "
        + ", ".join(f"{name} N={n}" for n, _, name, _ in plan)
    )
    best_po = None
    best_po_symmetry = None
    for num_points, _, symmetry_name, prepared in plan:
        outcome = solve_for_symmetry(
            base_basis,
            symmetry_name,
            num_points,
            seed=None,
            verbose=False,
            prepared=prepared,
            max_starts=1,
        )
        if outcome is None:
            continue
        solution = outcome[0]
        if solution.quality == "PI":
            return solution, symmetry_name, best_po, best_po_symmetry
        if _better_po(solution, best_po):
            best_po = solution
            best_po_symmetry = symmetry_name
    return None, None, best_po, best_po_symmetry


def default_starts_path(out, domain, dimension, degree):
    """File for the tried starts: next to the rule output, named by the cell."""
    _, prefix = get_atlas_domain_info(domain, dimension)
    name = f"starts-{prefix}-{degree:02d}.json"
    if out is None:
        return name
    if os.path.isdir(out):
        return os.path.join(out, name)
    return os.path.join(os.path.dirname(os.path.abspath(out)), name)


def main():
    global STARTS_RECORD
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
        "--start-rounds",
        type=int,
        default=4,
        help="automatic N only: after the first PI rule at N, retry N - 1, "
        "N - 2, ... with start ladder passes 2..R (fresh Halton starts) "
        "while they succeed (default: 4)",
    )
    parser.add_argument(
        "--max-extra-points",
        type=int,
        default=None,
        help="automatic N only: try N up to Z + this many, "
        "Z = counting value, ceil(M / (d + 1)), on the cube at most "
        "ceil(2 E / (d + 1)) (default: max(8, Z/4))",
    )
    parser.add_argument(
        "--single-start",
        action="store_true",
        help="automatic N only: no ladder; one targeted start per symmetry at the "
        "smallest feasible N >= Z",
    )
    parser.add_argument(
        "--starts-out",
        default=None,
        help="JSON file for every tried start and its result (default: "
        "starts-<prefix>-<degree>.json next to --out)",
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
    dof_target_z = counting_value(args.domain, args.dim, args.degree)
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
    STARTS_RECORD = []
    if args.points > 0:
        best_solution, best_symmetry, best_po_solution, best_po_symmetry = (
            solve_fixed_point_count(base_basis, symmetries, args.points, args.seed)
        )
    elif args.single_start:
        best_solution, best_symmetry, best_po_solution, best_po_symmetry = (
            solve_single_start(base_basis, symmetries, args.max_extra_points)
        )
    else:
        best_solution, best_symmetry, best_po_solution, best_po_symmetry = (
            solve_point_count_ladder(
                base_basis,
                symmetries,
                args.seed,
                args.max_extra_points,
                start_rounds=max(1, args.start_rounds),
            )
        )
    if best_solution is None and best_po_solution is not None:
        best_solution = best_po_solution
        best_symmetry = best_po_symmetry
        best_po_solution = None

    write_starts_file(args, base_basis, dof_target_z)
    elapsed = time.time() - start_time
    if best_solution is None:
        log_message(f"No solution found in {elapsed:.2f}s")
        sys.exit(1)

    write_rule_and_log(args, base_basis, best_solution, best_symmetry, elapsed)
    if best_po_solution is not None and len(best_po_solution.weights) < len(
        best_solution.weights
    ):
        write_rule_and_log(
            args, base_basis, best_po_solution, best_po_symmetry, elapsed
        )


def write_starts_file(args, base_basis, dof_target_z):
    """Write every tried start with its result, for a later comparison with the rules."""
    path = args.starts_out or default_starts_path(
        args.out, args.domain, args.dim, args.degree
    )
    parent_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent_dir, exist_ok=True)
    record = {
        "domain": args.domain,
        "dimension": args.dim,
        "degree": args.degree,
        "moments": int(base_basis.num_basis),
        "Z": dof_target_z,
        "command": "python code/cubature_generic.py " + shlex.join(sys.argv[1:]),
        "starts": STARTS_RECORD,
    }
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(record, file_handle)
        file_handle.write("\n")
    log_message(f"wrote {len(STARTS_RECORD)} tried starts to {path}")


def write_rule_and_log(args, base_basis, best_solution, best_symmetry, elapsed):
    """Write one rule in atlas format and log the result line."""
    command_args = [
        "--dim",
        str(args.dim),
        "--degree",
        str(args.degree),
        "--domain",
        args.domain,
    ]
    if args.points > 0:
        command_args.extend(
            ["--symmetry", best_symmetry, "--points", str(len(best_solution.weights))]
        )
    else:
        # The ladder (and its refinement rounds) found the rule: repeat the same run.
        if args.symmetry != "auto":
            command_args.extend(["--symmetry", args.symmetry])
        if args.start_rounds != 4:
            command_args.extend(["--start-rounds", str(args.start_rounds)])
        if args.max_extra_points is not None:
            command_args.extend(["--max-extra-points", str(args.max_extra_points)])
        if args.single_start:
            command_args.append("--single-start")
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
        return
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

"""Generic deterministic high-order cubature solver for d-cubes and d-simplices.

Unifies the mathematical and numerical techniques of the atlas solvers into a
single dimension-independent implementation without configuration-specific
special cases:

- Exact orthonormal polynomial bases for any dimension d >= 1:
  * Cube [-1, 1]^d: tensor-product Legendre basis with 3-term recurrence.
  * Simplex Delta^d: division-free homogeneous Proriol-Koornwinder-Dubiner
    basis in barycentric coordinates (works identically inside and outside).
- Exact analytic 1st derivatives and exact directional 2nd derivatives via
  2nd-order Taylor-jet arithmetic (O(M * K) cost, zero finite-difference loss).
- Generic finite symmetry engine (Reynolds projection + SVD invariant basis +
  stabilizer fixed-subspace orbit parameterization) supporting none, central,
  chiral, and full symmetry groups in arbitrary dimension d.
- Deterministic equilibrium starts (rotated D_d* lattice in Chebyshev angle
  space for cubes; concentric geodesic S^d_+ shells for simplices) with
  dynamic-programming orbit selection for exact target point count N.
- Fold-free (x, v^2) geodesic-accelerated Cholesky Levenberg-Marquardt
  (Transtrum & Sethna 2012) with 3-phase PI-preferred / PO-fallback pipeline:
  Phase 1 free solve -> Phase 2 smooth diagonal boundary hinge pull ->
  Phase 3 interior pure-moment polish + automatic minimum-norm Gauss-Newton
  end-stage. Logging and stopping criteria strictly report the pure moment
  residual (excluding the boundary penalty).
"""

import argparse
import itertools
import json
import math
import os
import shlex
import sys
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def log(msg):
    """Print a timestamped log line with second-level resolution."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} | {msg}", flush=True)


def multi_indices(dim, degree):
    """All d-tuples of non-negative integers with sum <= degree (graded order)."""
    out = []

    def rec(prefix, rem_dim, rem_deg):
        if rem_dim == 1:
            for k in range(rem_deg + 1):
                out.append(prefix + (k,))
            return
        for k in range(rem_deg + 1):
            rec(prefix + (k,), rem_dim - 1, rem_deg - k)

    rec((), dim, degree)
    return out


def halton_points(count, dim):
    """Deterministic Halton low-discrepancy points in (0, 1)^dim."""
    primes = []
    cand = 2
    while len(primes) < dim:
        is_p = True
        for p in primes:
            if p * p > cand:
                break
            if cand % p == 0:
                is_p = False
                break
        if is_p:
            primes.append(cand)
        cand += 1
    pts = np.zeros((count, dim), dtype=float)
    for j, base in enumerate(primes):
        for i in range(count):
            idx = i + 1
            f = 1.0
            r = 0.0
            while idx > 0:
                f /= base
                r += f * (idx % base)
                idx //= base
            pts[i, j] = r
    return pts


class CubeBasis:
    """Orthonormal tensor-Legendre basis on [-1, 1]^d with probability measure."""

    def __init__(self, dim, degree):
        self.domain = "cube"
        self.dim = dim
        self.degree = degree
        self.vol = float(2.0 ** dim)
        self.alphas = multi_indices(dim, degree)
        self.M = len(self.alphas)
        self.alpha_arr = np.asarray(self.alphas, dtype=np.int32)
        self.norm1d = np.sqrt(2.0 * np.arange(degree + 1, dtype=float) + 1.0)
        self.rhs = np.zeros(self.M, dtype=float)
        self.rhs[0] = 1.0

    def _leg1d_val(self, x):
        n = self.degree
        P = np.empty((n + 1, len(x)), dtype=float)
        P[0] = 1.0
        if n >= 1:
            P[1] = x
        for k in range(2, n + 1):
            P[k] = ((2 * k - 1) * x * P[k - 1] - (k - 1) * P[k - 2]) / k
        return P * self.norm1d[:, None]

    def _leg1d_val_grad(self, x):
        n = self.degree
        P = np.empty((n + 1, len(x)), dtype=float)
        dP = np.empty((n + 1, len(x)), dtype=float)
        P[0] = 1.0
        dP[0] = 0.0
        if n >= 1:
            P[1] = x
            dP[1] = 1.0
        for k in range(2, n + 1):
            P[k] = ((2 * k - 1) * x * P[k - 1] - (k - 1) * P[k - 2]) / k
            dP[k] = ((2 * k - 1) * (P[k - 1] + x * dP[k - 1]) - (k - 1) * dP[k - 2]) / k
        return P * self.norm1d[:, None], dP * self.norm1d[:, None]

    def _leg1d_jet2(self, x, dx):
        n = self.degree
        P0 = np.empty((n + 1, len(x)), dtype=float)
        P1 = np.empty((n + 1, len(x)), dtype=float)
        P2 = np.empty((n + 1, len(x)), dtype=float)
        P0[0] = 1.0
        P1[0] = 0.0
        P2[0] = 0.0
        if n >= 1:
            P0[1] = x
            P1[1] = dx
            P2[1] = 0.0
        for k in range(2, n + 1):
            c1 = (2 * k - 1) / k
            c2 = (k - 1) / k
            P0[k] = c1 * x * P0[k - 1] - c2 * P0[k - 2]
            P1[k] = c1 * (dx * P0[k - 1] + x * P1[k - 1]) - c2 * P1[k - 2]
            P2[k] = c1 * (2.0 * dx * P1[k - 1] + x * P2[k - 1]) - c2 * P2[k - 2]
        s = self.norm1d[:, None]
        return P0 * s, P1 * s, P2 * s

    def eval_val(self, X):
        K = len(X)
        mats = [self._leg1d_val(X[:, j]) for j in range(self.dim)]
        V = np.ones((self.M, K), dtype=float)
        for j in range(self.dim):
            V *= mats[j][self.alpha_arr[:, j]]
        return V

    def eval_val_grad(self, X):
        K = len(X)
        P = []
        dP = []
        for j in range(self.dim):
            pj, dpj = self._leg1d_val_grad(X[:, j])
            P.append(pj[self.alpha_arr[:, j]])
            dP.append(dpj[self.alpha_arr[:, j]])
        V = np.ones((self.M, K), dtype=float)
        for j in range(self.dim):
            V *= P[j]
        G = np.empty((self.dim, self.M, K), dtype=float)
        for j in range(self.dim):
            gj = dP[j].copy()
            for r in range(self.dim):
                if r != j:
                    gj *= P[r]
            G[j] = gj
        return V, G

    def eval_jet2(self, X, dX):
        K = len(X)
        F0 = np.ones((self.M, K), dtype=float)
        F1 = np.zeros((self.M, K), dtype=float)
        F2 = np.zeros((self.M, K), dtype=float)
        for j in range(self.dim):
            p0, p1, p2 = self._leg1d_jet2(X[:, j], dX[:, j])
            idx = self.alpha_arr[:, j]
            q0 = p0[idx]
            q1 = p1[idx]
            q2 = p2[idx]
            n0 = F0 * q0
            n1 = F0 * q1 + F1 * q0
            n2 = F0 * q2 + 2.0 * F1 * q1 + F2 * q0
            F0 = n0
            F1 = n1
            F2 = n2
        return F0, F1, F2

    def boundary_margin(self, X):
        """Minimum distance to boundary (> 0 strictly interior, <= 0 outside/on)."""
        return float(1.0 - np.abs(X).max())

    def boundary_hinge(self, X, margin):
        """Return hinge violation values and gradients d(hinge)/dX for |x_j| > 1 - margin."""
        cap = 1.0 - margin
        ax = np.abs(X)
        over = np.maximum(0.0, ax - cap)
        grad = np.where(ax > cap, np.sign(X) * over, 0.0)
        return over, grad


class SimplexBasis:
    """Orthonormal division-free homogeneous Dubiner basis on Delta^d (prob. measure)."""

    def __init__(self, dim, degree):
        self.domain = "simplex"
        self.dim = dim
        self.degree = degree
        self.vol = 1.0 / float(math.factorial(dim))
        self.alphas = multi_indices(dim, degree)
        self.M = len(self.alphas)
        self.alpha_arr = np.asarray(self.alphas, dtype=np.int32)
        # Precompute Jacobi parameter a_m = 2 * sum_{r > m} k_r + (d - 1 - m)
        self.a_params = np.zeros((self.M, dim), dtype=np.int32)
        self.norms = np.ones(self.M, dtype=float)
        dfac = float(math.factorial(dim))
        for idx, alpha in enumerate(self.alphas):
            rem = 0
            prod = 1.0
            for m in range(dim - 1, -1, -1):
                a_m = 2 * rem + (dim - 1 - m)
                self.a_params[idx, m] = a_m
                prod *= float(2 * alpha[m] + a_m + 1)
                rem += alpha[m]
            self.norms[idx] = math.sqrt(prod / dfac)
        # Group unique (m, a_m) pairs to evaluate homogeneous Jacobi recurrences once
        self.unique_a_by_m = []
        self.a_to_slot = []
        for m in range(dim):
            uniq = sorted(set(int(v) for v in self.a_params[:, m]))
            self.unique_a_by_m.append(uniq)
            slot_map = {a_val: s for s, a_val in enumerate(uniq)}
            self.a_to_slot.append(
                np.array([slot_map[int(v)] for v in self.a_params[:, m]], dtype=np.int32)
            )
        self.rhs = np.zeros(self.M, dtype=float)
        self.rhs[0] = 1.0

    def _bary_and_tails(self, X):
        K = len(X)
        u = np.empty((self.dim + 1, K), dtype=float)
        u[0] = 1.0 - np.sum(X, axis=1)
        for j in range(self.dim):
            u[j + 1] = X[:, j]
        s = np.empty((self.dim + 1, K), dtype=float)
        s[self.dim] = u[self.dim]
        for m in range(self.dim - 1, -1, -1):
            s[m] = u[m] + s[m + 1]
        return u, s

    def _hjac_val(self, v, s, a, kmax):
        P = np.empty((kmax + 1, len(v)), dtype=float)
        P[0] = 1.0
        if kmax >= 1:
            P[1] = (a + 2.0) * v - s
        s2 = s * s
        v2_s = 2.0 * v - s
        for k in range(2, kmax + 1):
            kf = float(k)
            af = float(a)
            den = 2.0 * kf * (kf + af) * (2.0 * kf + af - 2.0)
            c1 = (2.0 * kf + af - 1.0) * (2.0 * kf + af) * (2.0 * kf + af - 2.0) / den
            c2 = (2.0 * kf + af - 1.0) * (af * af) / den
            c3 = 2.0 * (kf + af - 1.0) * (kf - 1.0) * (2.0 * kf + af) / den
            P[k] = (c1 * v2_s + c2 * s) * P[k - 1] - c3 * s2 * P[k - 2]
        return P

    def _hjac_jet1(self, v, dv, s, ds, a, kmax):
        P0 = np.empty((kmax + 1, len(v)), dtype=float)
        P1 = np.empty((kmax + 1, len(v)), dtype=float)
        P0[0] = 1.0
        P1[0] = 0.0
        if kmax >= 1:
            P0[1] = (a + 2.0) * v - s
            P1[1] = (a + 2.0) * dv - ds
        s2 = s * s
        ds2 = 2.0 * s * ds
        v2_s = 2.0 * v - s
        dv2_s = 2.0 * dv - ds
        for k in range(2, kmax + 1):
            kf = float(k)
            af = float(a)
            den = 2.0 * kf * (kf + af) * (2.0 * kf + af - 2.0)
            c1 = (2.0 * kf + af - 1.0) * (2.0 * kf + af) * (2.0 * kf + af - 2.0) / den
            c2 = (2.0 * kf + af - 1.0) * (af * af) / den
            c3 = 2.0 * (kf + af - 1.0) * (kf - 1.0) * (2.0 * kf + af) / den
            q0 = c1 * v2_s + c2 * s
            q1 = c1 * dv2_s + c2 * ds
            P0[k] = q0 * P0[k - 1] - c3 * s2 * P0[k - 2]
            P1[k] = q1 * P0[k - 1] + q0 * P1[k - 1] - c3 * (ds2 * P0[k - 2] + s2 * P1[k - 2])
        return P0, P1

    def _hjac_jet2(self, v, dv, s, ds, a, kmax):
        P0 = np.empty((kmax + 1, len(v)), dtype=float)
        P1 = np.empty((kmax + 1, len(v)), dtype=float)
        P2 = np.empty((kmax + 1, len(v)), dtype=float)
        P0[0] = 1.0
        P1[0] = 0.0
        P2[0] = 0.0
        if kmax >= 1:
            P0[1] = (a + 2.0) * v - s
            P1[1] = (a + 2.0) * dv - ds
            P2[1] = 0.0
        s2 = s * s
        ds2 = 2.0 * s * ds
        d2s2 = 2.0 * ds * ds
        v2_s = 2.0 * v - s
        dv2_s = 2.0 * dv - ds
        for k in range(2, kmax + 1):
            kf = float(k)
            af = float(a)
            den = 2.0 * kf * (kf + af) * (2.0 * kf + af - 2.0)
            c1 = (2.0 * kf + af - 1.0) * (2.0 * kf + af) * (2.0 * kf + af - 2.0) / den
            c2 = (2.0 * kf + af - 1.0) * (af * af) / den
            c3 = 2.0 * (kf + af - 1.0) * (kf - 1.0) * (2.0 * kf + af) / den
            q0 = c1 * v2_s + c2 * s
            q1 = c1 * dv2_s + c2 * ds
            P0[k] = q0 * P0[k - 1] - c3 * s2 * P0[k - 2]
            P1[k] = q1 * P0[k - 1] + q0 * P1[k - 1] - c3 * (ds2 * P0[k - 2] + s2 * P1[k - 2])
            P2[k] = (
                2.0 * q1 * P1[k - 1]
                + q0 * P2[k - 1]
                - c3 * (d2s2 * P0[k - 2] + 2.0 * ds2 * P1[k - 2] + s2 * P2[k - 2])
            )
        return P0, P1, P2

    def eval_val(self, X):
        K = len(X)
        u, s = self._bary_and_tails(X)
        V = np.repeat(self.norms[:, None], K, axis=1)
        for m in range(self.dim):
            uniq = self.unique_a_by_m[m]
            tables = []
            for a_val in uniq:
                kmax = self.degree - max(0, (a_val - (self.dim - 1 - m)) // 2)
                tables.append(self._hjac_val(u[m], s[m], a_val, kmax))
            factor = np.empty((self.M, K), dtype=float)
            slots = self.a_to_slot[m]
            ks = self.alpha_arr[:, m]
            for slot, tab in enumerate(tables):
                mask = slots == slot
                factor[mask] = tab[ks[mask]]
            V *= factor
        return V

    def eval_val_grad(self, X):
        K = len(X)
        u, s = self._bary_and_tails(X)
        # Precompute factors and directional derivatives along each coordinate e_j
        # du[0]/dx_j = -1, du[j+1]/dx_j = +1; ds[0]/dx_j = 0, ds[m]/dx_j = 1 if m <= j+1 else 0
        F0_list = []
        dF_list = [[None] * self.dim for _ in range(self.dim)]
        for m in range(self.dim):
            uniq = self.unique_a_by_m[m]
            slots = self.a_to_slot[m]
            ks = self.alpha_arr[:, m]
            # Evaluate partial derivatives wrt v = u[m] and s = s[m] once!
            dv1 = np.ones(K, dtype=float)
            dz0 = np.zeros(K, dtype=float)
            tab_v0 = []
            tab_dv = []
            tab_ds = []
            for a_val in uniq:
                kmax = self.degree - max(0, (a_val - (self.dim - 1 - m)) // 2)
                p0, pv = self._hjac_jet1(u[m], dv1, s[m], dz0, a_val, kmax)
                _, ps = self._hjac_jet1(u[m], dz0, s[m], dv1, a_val, kmax)
                tab_v0.append(p0)
                tab_dv.append(pv)
                tab_ds.append(ps)
            f0 = np.empty((self.M, K), dtype=float)
            fv = np.empty((self.M, K), dtype=float)
            fs = np.empty((self.M, K), dtype=float)
            for slot in range(len(uniq)):
                mask = slots == slot
                f0[mask] = tab_v0[slot][ks[mask]]
                fv[mask] = tab_dv[slot][ks[mask]]
                fs[mask] = tab_ds[slot][ks[mask]]
            F0_list.append(f0)
            for j in range(self.dim):
                du_m = -1.0 if m == 0 else (1.0 if m == j + 1 else 0.0)
                ds_m = 0.0 if m == 0 else (1.0 if m <= j + 1 else 0.0)
                if du_m == 0.0 and ds_m == 0.0:
                    dF_list[j][m] = None
                else:
                    dF_list[j][m] = du_m * fv + ds_m * fs
        V = np.repeat(self.norms[:, None], K, axis=1)
        for m in range(self.dim):
            V *= F0_list[m]
        G = np.zeros((self.dim, self.M, K), dtype=float)
        for j in range(self.dim):
            for m in range(self.dim):
                dfm = dF_list[j][m]
                if dfm is None:
                    continue
                term = self.norms[:, None] * dfm
                for r in range(self.dim):
                    if r != m:
                        term *= F0_list[r]
                G[j] += term
        return V, G

    def eval_jet2(self, X, dX):
        K = len(X)
        u, s = self._bary_and_tails(X)
        du = np.empty((self.dim + 1, K), dtype=float)
        du[0] = -np.sum(dX, axis=1)
        for j in range(self.dim):
            du[j + 1] = dX[:, j]
        ds = np.empty((self.dim + 1, K), dtype=float)
        ds[self.dim] = du[self.dim]
        for m in range(self.dim - 1, -1, -1):
            ds[m] = du[m] + ds[m + 1]
        F0 = np.repeat(self.norms[:, None], K, axis=1)
        F1 = np.zeros((self.M, K), dtype=float)
        F2 = np.zeros((self.M, K), dtype=float)
        for m in range(self.dim):
            uniq = self.unique_a_by_m[m]
            slots = self.a_to_slot[m]
            ks = self.alpha_arr[:, m]
            q0 = np.empty((self.M, K), dtype=float)
            q1 = np.empty((self.M, K), dtype=float)
            q2 = np.empty((self.M, K), dtype=float)
            for slot, a_val in enumerate(uniq):
                kmax = self.degree - max(0, (a_val - (self.dim - 1 - m)) // 2)
                p0, p1, p2 = self._hjac_jet2(u[m], du[m], s[m], ds[m], a_val, kmax)
                mask = slots == slot
                q0[mask] = p0[ks[mask]]
                q1[mask] = p1[ks[mask]]
                q2[mask] = p2[ks[mask]]
            n0 = F0 * q0
            n1 = F0 * q1 + F1 * q0
            n2 = F0 * q2 + 2.0 * F1 * q1 + F2 * q0
            F0 = n0
            F1 = n1
            F2 = n2
        return F0, F1, F2

    def boundary_margin(self, X):
        """Minimum barycentric coordinate (> 0 strictly interior, <= 0 outside/on)."""
        u0 = 1.0 - np.sum(X, axis=1)
        return float(min(float(u0.min()), float(X.min())))

    def boundary_hinge(self, X, margin):
        """Return hinge violation values and gradients d(hinge)/dX for u_j < margin."""
        u0 = 1.0 - np.sum(X, axis=1)
        v0 = np.maximum(0.0, margin - u0)
        vx = np.maximum(0.0, margin - X)
        # Penalty = 0.5 * (sum v0^2 + sum vx^2) -> d/dX_j = +v0 - vx_j
        grad = v0[:, None] - vx
        over = np.concatenate([v0[:, None], vx], axis=1)
        return over, grad


def build_group_ops(domain, dim, sym_name):
    """Return list of affine symmetry ops (A, t) with g(x) = x @ A.T + t."""
    if sym_name == "none":
        return [(np.eye(dim, dtype=float), np.zeros(dim, dtype=float))]
    if domain == "cube":
        if sym_name == "central":
            return [
                (np.eye(dim, dtype=float), np.zeros(dim, dtype=float)),
                (-np.eye(dim, dtype=float), np.zeros(dim, dtype=float)),
            ]
        if sym_name == "chiral":
            if dim == 1:
                return [
                    (np.eye(1, dtype=float), np.zeros(1, dtype=float)),
                    (-np.eye(1, dtype=float), np.zeros(1, dtype=float)),
                ]
            if dim == 2:
                # C4: 90-degree rotations
                ops = []
                R = np.array([[0.0, -1.0], [1.0, 0.0]])
                cur = np.eye(2, dtype=float)
                for _ in range(4):
                    ops.append((cur.copy(), np.zeros(2, dtype=float)))
                    cur = cur @ R
                return ops
            # d >= 3: cyclic coordinate shifts x {+I, -I} (order 2d; S6 in 3D)
            ops = []
            for shift in range(dim):
                P = np.roll(np.eye(dim, dtype=float), shift, axis=0)
                ops.append((P, np.zeros(dim, dtype=float)))
                ops.append((-P, np.zeros(dim, dtype=float)))
            return ops
        if sym_name == "full":
            if dim > 3:
                return None
            ops = []
            for perm in itertools.permutations(range(dim)):
                P = np.zeros((dim, dim), dtype=float)
                for i, p in enumerate(perm):
                    P[i, p] = 1.0
                for signs in itertools.product((-1.0, 1.0), repeat=dim):
                    ops.append((P * np.asarray(signs)[:, None], np.zeros(dim, dtype=float)))
            return ops
        raise ValueError(f"unknown symmetry '{sym_name}' for cube")
    # Simplex Delta^d: vertex permutations of v_0 = 0, v_1 = e_1, ..., v_d = e_d
    if sym_name == "central":
        if dim == 1:
            sym_name = "full"
        else:
            return None
    verts = np.zeros((dim + 1, dim), dtype=float)
    for j in range(dim):
        verts[j + 1, j] = 1.0

    def perm_to_affine(perm):
        # x = sum_{j=1}^d x_j v_j + (1 - sum x_j) v_0 -> map v_k -> v_{perm[k]}
        t = verts[perm[0]].copy()
        A = np.zeros((dim, dim), dtype=float)
        for j in range(dim):
            A[:, j] = verts[perm[j + 1]] - t
        return A, t

    if sym_name == "chiral":
        if dim == 1:
            perms = [(0, 1), (1, 0)]
        elif dim == 3:
            # A4: all even permutations of 4 vertices (order 12)
            perms = []
            for p in itertools.permutations(range(4)):
                inv = sum(1 for i in range(4) for j in range(i + 1, 4) if p[i] > p[j])
                if inv % 2 == 0:
                    perms.append(p)
        else:
            # Cyclic group C_{d+1} (C3 in 2D, C5 in 4D, ...)
            perms = [tuple((i + s) % (dim + 1) for i in range(dim + 1)) for s in range(dim + 1)]
        return [perm_to_affine(p) for p in perms]
    if sym_name == "full":
        if dim > 3:
            return None
        return [perm_to_affine(p) for p in itertools.permutations(range(dim + 1))]
    raise ValueError(f"unknown symmetry '{sym_name}' for simplex")


class SymmetricBasis:
    """Symmetry-reduced orthonormal polynomial basis via Reynolds projection + SVD."""

    def __init__(self, base, sym_name, ops):
        self.base = base
        self.domain = base.domain
        self.dim = base.dim
        self.degree = base.degree
        self.sym_name = sym_name
        self.ops = ops
        self.gsize = len(ops)
        if self.gsize == 1:
            self.C = np.eye(base.M, dtype=float)
            self.M = base.M
            self.rhs = base.rhs.copy()
            return
        # Deterministic probe points strictly inside the domain
        n_probe = max(2 * base.M + 16, 40)
        H = halton_points(n_probe, base.dim)
        if base.domain == "cube":
            Xp = 1.9 * H - 0.95
        else:
            # Map (0,1)^d smoothly into interior of Delta^d via stick-breaking
            Xp = np.zeros_like(H)
            rem = np.ones(n_probe, dtype=float)
            for j in range(base.dim):
                frac = 0.05 + 0.85 * H[:, j]
                Xp[:, j] = rem * frac / float(base.dim - j + 1)
                rem -= Xp[:, j]
        Vbar = np.zeros((base.M, n_probe), dtype=float)
        for A, t in ops:
            Xg = Xp @ A.T + t[None, :]
            Vbar += base.eval_val(Xg)
        Vbar /= float(self.gsize)
        U, sv, _ = np.linalg.svd(Vbar, full_matrices=False)
        tol = 1e-10 * float(sv[0]) if len(sv) > 0 else 1e-10
        rank = int(np.sum(sv > tol))
        self.C = U[:, :rank].T.copy()
        self.M = rank
        self.rhs = self.C @ base.rhs

    def eval_val(self, X):
        return self.C @ self.base.eval_val(X)

    def eval_val_grad(self, X):
        V0, G0 = self.base.eval_val_grad(X)
        V = self.C @ V0
        G = np.empty((self.dim, self.M, len(X)), dtype=float)
        for j in range(self.dim):
            G[j] = self.C @ G0[j]
        return V, G

    def eval_jet2(self, X, dX):
        F0, F1, F2 = self.base.eval_jet2(X, dX)
        return self.C @ F0, self.C @ F1, self.C @ F2


def orbit_subspace(point, ops, tol=1e-7):
    """Compute orbit size, fixed-point subspace (c0, U), and canonical images."""
    dim = len(point)
    imgs = []
    stab = []
    for A, t in ops:
        q = A @ point + t
        if float(np.linalg.norm(q - point)) < tol:
            stab.append((A, t))
        is_new = True
        for prev in imgs:
            if float(np.linalg.norm(q - prev)) < tol:
                is_new = False
                break
        if is_new:
            imgs.append(q)
    size = len(imgs)
    if len(stab) == 1:
        return size, np.zeros(dim, dtype=float), np.eye(dim, dtype=float), np.asarray(imgs)
    # Fixed subspace: (A - I) x = -t for all (A, t) in stab
    # Since point itself is fixed, x = point + U xi with (A - I) U = 0
    rows = [A - np.eye(dim, dtype=float) for A, _ in stab]
    S = np.vstack(rows)
    _, sv, Vt = np.linalg.svd(S, full_matrices=True)
    rank = int(np.sum(sv > 1e-9))
    U = Vt[rank:].T.copy()
    # Particular fixed point c0 = projection of point orthogonal to U
    if U.shape[1] > 0:
        c0 = point - U @ (U.T @ point)
    else:
        c0 = point.copy()
    return size, c0, U, np.asarray(imgs)


def expand_orbits(orbits, ops):
    """Expand list of orbit dicts into full (nodes, weights) arrays."""
    all_pts = []
    all_w = []
    for orb in orbits:
        x = orb["c0"] + orb["U"] @ orb["xi"]
        w = float(orb["v"] * orb["v"])
        imgs = []
        for A, t in ops:
            q = A @ x + t
            is_new = True
            for prev in imgs:
                if float(np.linalg.norm(q - prev)) < 1e-7:
                    is_new = False
                    break
            if is_new:
                imgs.append(q)
        for q in imgs:
            all_pts.append(q)
            all_w.append(w)
    return np.asarray(all_pts, dtype=float), np.asarray(all_w, dtype=float)


def group_points_into_orbits(pts, ops, tol=1e-6):
    """Cluster a symmetric point cloud into distinct G-orbits with fixed subspaces."""
    n_pts = len(pts)
    used = np.zeros(n_pts, dtype=bool)
    orbits = []
    for i in range(n_pts):
        if used[i]:
            continue
        p = pts[i]
        size, c0, U, imgs = orbit_subspace(p, ops, tol=tol)
        # Mark all points in pts that match any orbit image
        for q in imgs:
            d = np.linalg.norm(pts - q[None, :], axis=1)
            used[d < tol] = True
        xi = U.T @ (p - c0) if U.shape[1] > 0 else np.zeros(0, dtype=float)
        orbits.append(
            {
                "point": p.copy(),
                "size": size,
                "c0": c0,
                "U": U,
                "dof": int(U.shape[1]),
                "unk": int(U.shape[1]) + 1,
                "xi": xi,
                "v": 1.0,
            }
        )
    return orbits


def select_orbits_exact_n(candidates, target_n, min_unk):
    """Deterministic DP selecting orbits with sum(size) == target_n and sum(unk) >= min_unk.

    Prefers inner orbits (lower candidate index) and higher unknown count.
    """
    # Limit pool to avoid huge DP tables when candidates >> target_n
    pool = candidates[: max(target_n + 64, 160)]
    # DP state: dp[s] = (score, unk_sum, prev_idx, prev_s)
    # Maximize (unk_capped * 1e6 - radius_rank_sum)
    dp = {0: (0.0, 0, -1, -1)}
    parent = []
    for idx, orb in enumerate(pool):
        sz = orb["size"]
        uk = orb["unk"]
        cost = float(idx) * 1.0
        new_dp = dict(dp)
        chosen_map = {}
        for s, (sc, u_sum, _, _) in dp.items():
            ns = s + sz
            if ns > target_n:
                continue
            nu = u_sum + uk
            # Strongly reward reaching min_unk, then prefer inner orbits (smaller idx sum)
            nsc = float(min(nu, min_unk + 4)) * 1e5 + float(nu) * 10.0 - cost
            prev = new_dp.get(ns)
            if prev is None or nsc > prev[0]:
                new_dp[ns] = (nsc, nu, idx, s)
                chosen_map[ns] = s
        dp = new_dp
        parent.append(chosen_map)
    if target_n not in dp:
        return None
    _, total_unk, _, _ = dp[target_n]
    if total_unk < min_unk:
        return None
    # Backtrack selected orbit indices
    chosen = []
    cur_s = target_n
    for idx in range(len(pool) - 1, -1, -1):
        if cur_s in parent[idx]:
            chosen.append(pool[idx])
            cur_s = parent[idx][cur_s]
    chosen.reverse()
    return chosen


def rotation_around_diag(dim, deg):
    """Deterministic chiral orthogonal rotation fixing (1, ..., 1) for dim >= 2."""
    if dim < 2 or abs(deg) < 1e-12:
        return np.eye(dim, dtype=float)
    th = math.radians(deg)
    if dim == 2:
        c = math.cos(th)
        s = math.sin(th)
        return np.array([[c, -s], [s, c]], dtype=float)
    # Construct orthonormal basis of (1, ..., 1)^perp and apply 2D rotations in pairs
    ones = np.ones((dim, 1), dtype=float) / math.sqrt(dim)
    Q, _ = np.linalg.qr(np.hstack([ones, np.eye(dim, dtype=float)]), mode="complete")
    R_sub = np.eye(dim, dtype=float)
    c = math.cos(th)
    s = math.sin(th)
    # Rotate in plane spanned by Q[:, 1], Q[:, 2] (for d=3: exact Rodrigues around (1,1,1))
    i = 1
    while i + 1 < dim:
        R_sub[i, i] = c
        R_sub[i, i + 1] = -s
        R_sub[i + 1, i] = s
        R_sub[i + 1, i + 1] = c
        i += 2
    return Q @ R_sub @ Q.T


def cube_equilibrium_start(dim, degree, target_n, sym_name, ops, r0=0.35, angle=6.0, pnorm=12.0, min_unk=1):
    """Deterministic rotated D_d* Chebyshev-angle lattice start on [-1, 1]^d."""
    # Estimate half-width L of integer D_d* lattice Z^d U (Z + 1/2)^d (2 pts per unit cell)
    # Parity shift: odd N -> include origin (0-shift), even N -> (1/4)-shift
    shift = 0.0 if (target_n % 2 == 1) else 0.25
    pts_per_cell = 2.0 if dim >= 2 else 1.0
    side = max(1.5, ((target_n * 1.8) / pts_per_cell) ** (1.0 / dim))
    L = int(math.ceil(side / 2.0)) + 2
    # Cap L for high dimensions so grid generation stays fast
    while (2 * L + 1) ** dim > 25000 and L > 1:
        L -= 1
    grid1 = np.arange(-L, L + 1, dtype=float)
    mesh = np.meshgrid(*([grid1] * dim), indexing="ij")
    base_pts = np.stack([m.ravel() for m in mesh], axis=1)
    if dim >= 2:
        pts_a = base_pts + shift
        pts_b = base_pts + (shift + 0.5)
        if sym_name == "full":
            raw = np.vstack([base_pts, pts_a, pts_b])
        else:
            raw = np.vstack([pts_a, pts_b])
    else:
        raw = base_pts + (0.0 if target_n % 2 == 1 else 0.5)
    # Apply chiral rotation for mirror-free groups (preserving G-symmetry)
    if sym_name in ("chiral", "central", "none") and dim >= 2:
        R = rotation_around_diag(dim, angle)
        raw = raw @ R.T
    # Keep symmetric closure under G around inner region
    rad = np.max(np.abs(raw), axis=1)
    keep = rad <= float(L) + 0.6
    raw = raw[keep]
    # Close under G to guarantee exact symmetry of candidate pool
    closed = [raw]
    for A, t in ops[1:]:
        closed.append(raw @ A.T + t[None, :])
    raw = np.vstack(closed)
    # Order by L_p norm before clustering so inner orbits come first
    if pnorm > 0.0:
        norms = np.sum(np.abs(raw) ** pnorm, axis=1) ** (1.0 / pnorm)
    else:
        norms = np.max(np.abs(raw), axis=1)
    order = np.argsort(norms)
    raw = raw[order]
    # Trim raw pool to a generous multiple of target_n before orbit clustering
    raw = raw[: max(target_n * 8, 500)]
    orbits = group_points_into_orbits(raw, ops, tol=1e-5)
    # Sort orbits by L_p norm of representative
    if pnorm > 0.0:
        orbits.sort(key=lambda o: float(np.sum(np.abs(o["point"]) ** pnorm) ** (1.0 / pnorm)))
    else:
        orbits.sort(key=lambda o: float(np.max(np.abs(o["point"]))))
    chosen = select_orbits_exact_n(orbits, target_n, min_unk)
    if chosen is None:
        return None
    # Scale angle coordinates into [-pi/2 * (1 - 2*r0/n), pi/2 * (1 - 2*r0/n)]^d
    all_raw = np.vstack([o["point"] for o in chosen])
    max_ang = float(np.abs(all_raw).max())
    target_ang = 0.5 * math.pi * max(0.35, 1.0 - 2.0 * r0 / max(degree, 1))
    scale = (target_ang / max_ang) if max_ang > 1e-12 else 0.0
    out = []
    w_sum = 0.0
    for o in chosen:
        ang = o["point"] * scale
        x = np.sin(ang)
        # Re-project onto exact fixed subspace of orbit
        size, c0, U, _ = orbit_subspace(x, ops, tol=1e-6)
        xi = U.T @ (x - c0) if U.shape[1] > 0 else np.zeros(0, dtype=float)
        x_proj = c0 + (U @ xi if U.shape[1] > 0 else 0.0)
        eq_w = float(np.prod(np.sqrt(np.maximum(1.0 - x_proj * x_proj, 1e-4))))
        w_sum += size * eq_w
        out.append(
            {
                "size": size,
                "c0": c0,
                "U": U,
                "dof": int(U.shape[1]),
                "unk": int(U.shape[1]) + 1,
                "xi": xi,
                "w": eq_w,
            }
        )
    for o in out:
        o["v"] = math.sqrt(o["w"] / max(w_sum, 1e-30))
        del o["w"]
    return out


def simplex_bary_lattice(dim, level):
    """All barycentric integer points m in Z_>=0^{dim+1} with sum(m) == level."""
    pts = []

    def rec(prefix, rem_slots, rem_sum):
        if rem_slots == 1:
            pts.append(prefix + (rem_sum,))
            return
        for k in range(rem_sum + 1):
            rec(prefix + (k,), rem_slots - 1, rem_sum - k)

    rec((), dim + 1, level)
    return np.asarray(pts, dtype=float) / float(max(level, 1))


def simplex_equilibrium_start(dim, degree, target_n, sym_name, ops, r0=0.45, twist=0.06, min_unk=1):
    """Deterministic S^d_+ concentric-shell equilibrium start on standard simplex Delta^d."""
    # Choose barycentric grid level L so comb(L + d, d) generously covers target_n
    L = 1
    while math.comb(L + dim, dim) < max(int(target_n * 1.6), target_n + 16):
        L += 1
        if math.comb(L + dim, dim) > 15000:
            break
    # Include all concentric shell levels 1..L so every stabilizer orbit type is available
    bary_list = []
    eps = max(0.03, min(0.60, r0 / float(max(degree, 2))))
    for lv in range(1, L + 1):
        b = simplex_bary_lattice(dim, lv)
        b_in = (1.0 - eps) * b + eps / float(dim + 1)
        # Map to S^d_+ equilibrium density via sin(pi/2 * b_in) -> normalize -> u = s^2
        s = np.sin(0.5 * math.pi * b_in)
        s /= np.linalg.norm(s, axis=1, keepdims=True)
        u = s * s
        # Apply gentle chiral cyclic twist for mirror-free groups
        if sym_name in ("chiral", "none") and twist > 0.0 and dim >= 2:
            # Preserve lower-dimensional fixed axes (repeated coordinates) while twisting generic shells
            u_sorted = np.sort(u, axis=1)
            diffs = np.diff(u_sorted, axis=1)
            is_gen = np.all(diffs > 1e-6, axis=1)
            u_tw = u.copy()
            u_tw[is_gen] = (1.0 - twist) * u[is_gen] + twist * np.roll(u[is_gen], 1, axis=1)
            u = u_tw
        bary_list.append(u)
    # Also include exact centroid explicitly
    bary_list.append(np.full((1, dim + 1), 1.0 / float(dim + 1), dtype=float))
    u_all = np.vstack(bary_list)
    raw = u_all[:, 1:]
    # Close under G
    closed = [raw]
    for A, t in ops[1:]:
        closed.append(raw @ A.T + t[None, :])
    raw = np.vstack(closed)
    # Rank by distance from centroid so inner shells come first
    u0 = 1.0 - np.sum(raw, axis=1, keepdims=True)
    ub = np.hstack([u0, raw])
    cent = 1.0 / float(dim + 1)
    dist = np.linalg.norm(ub - cent, axis=1)
    raw = raw[np.argsort(dist)]
    raw = raw[: max(target_n * 6, 400)]
    orbits = group_points_into_orbits(raw, ops, tol=1e-4)
    orbits.sort(key=lambda o: float(np.linalg.norm(
        np.concatenate([[1.0 - np.sum(o["point"])], o["point"]]) - cent
    )))
    chosen = select_orbits_exact_n(orbits, target_n, min_unk)
    if chosen is None:
        return None
    out = []
    w_sum = 0.0
    for o in chosen:
        x = o["point"]
        size, c0, U, _ = orbit_subspace(x, ops, tol=1e-6)
        xi = U.T @ (x - c0) if U.shape[1] > 0 else np.zeros(0, dtype=float)
        x_proj = c0 + (U @ xi if U.shape[1] > 0 else 0.0)
        u_vec = np.concatenate([[1.0 - float(np.sum(x_proj))], x_proj])
        eq_w = float(np.prod(np.sqrt(np.maximum(u_vec, 1e-4))))
        w_sum += size * eq_w
        out.append(
            {
                "size": size,
                "c0": c0,
                "U": U,
                "dof": int(U.shape[1]),
                "unk": int(U.shape[1]) + 1,
                "xi": xi,
                "w": eq_w,
            }
        )
    for o in out:
        o["v"] = math.sqrt(o["w"] / max(w_sum, 1e-30))
        del o["w"]
    return out


def pack_orbits(orbits):
    """Pack orbit (xi, v) parameters into a single flat vector q."""
    parts = []
    for o in orbits:
        if o["dof"] > 0:
            parts.append(o["xi"])
        parts.append(np.array([o["v"]], dtype=float))
    return np.concatenate(parts)


def unpack_orbits(orbits, q):
    """Return list of orbit copies with parameters populated from flat vector q."""
    out = []
    off = 0
    for o in orbits:
        dof = o["dof"]
        xi = q[off:off + dof].copy() if dof > 0 else np.zeros(0, dtype=float)
        off += dof
        v = float(q[off])
        off += 1
        no = dict(o)
        no["xi"] = xi
        no["v"] = v
        out.append(no)
    return out


def orbits_to_reps(orbits, q):
    """Map flat parameter vector q -> representative points X (K, d), weights w (K,), sizes (K,)."""
    K = len(orbits)
    dim = len(orbits[0]["c0"])
    X = np.empty((K, dim), dtype=float)
    w = np.empty(K, dtype=float)
    v_arr = np.empty(K, dtype=float)
    sizes = np.empty(K, dtype=float)
    off = 0
    for k, o in enumerate(orbits):
        dof = o["dof"]
        if dof > 0:
            xi = q[off:off + dof]
            X[k] = o["c0"] + o["U"] @ xi
        else:
            X[k] = o["c0"]
        off += dof
        v = q[off]
        off += 1
        v_arr[k] = v
        w[k] = v * v
        sizes[k] = float(o["size"])
    return X, w, v_arr, sizes


def eval_residual(SB, orbits, q):
    """Pure moment residual vector r (excluding any boundary penalty)."""
    X, w, _, sizes = orbits_to_reps(orbits, q)
    V = SB.eval_val(X)
    return V @ (sizes * w) - SB.rhs


def eval_res_jac(SB, orbits, q):
    """Pure moment residual r and exact Jacobian J wrt flat parameter vector q."""
    X, w, v_arr, sizes = orbits_to_reps(orbits, q)
    V, G = SB.eval_val_grad(X)
    sw = sizes * w
    r = V @ sw - SB.rhs
    n_unk = len(q)
    J = np.empty((SB.M, n_unk), dtype=float)
    off = 0
    for k, o in enumerate(orbits):
        dof = o["dof"]
        if dof > 0:
            # G[:, :, k] has shape (dim, M); U has shape (dim, dof)
            # d(sw_k * phi(x_k))/d(xi_k) = sw_k * (G[:, :, k].T @ U)
            J[:, off:off + dof] = sw[k] * (G[:, :, k].T @ o["U"])
            off += dof
        J[:, off] = (2.0 * sizes[k] * v_arr[k]) * V[:, k]
        off += 1
    return r, J, X


def eval_d2_dir(SB, orbits, q, dq):
    """Exact directional 2nd derivative D^2 r(q)[dq, dq] via 2nd-order Taylor jets."""
    K = len(orbits)
    dim = SB.dim
    X = np.empty((K, dim), dtype=float)
    dX = np.empty((K, dim), dtype=float)
    w0 = np.empty(K, dtype=float)
    w1 = np.empty(K, dtype=float)
    w2 = np.empty(K, dtype=float)
    off = 0
    for k, o in enumerate(orbits):
        dof = o["dof"]
        if dof > 0:
            xi = q[off:off + dof]
            dxi = dq[off:off + dof]
            X[k] = o["c0"] + o["U"] @ xi
            dX[k] = o["U"] @ dxi
        else:
            X[k] = o["c0"]
            dX[k] = 0.0
        off += dof
        v = q[off]
        dv = dq[off]
        off += 1
        sz = float(o["size"])
        w0[k] = sz * (v * v)
        w1[k] = sz * (2.0 * v * dv)
        w2[k] = sz * (2.0 * dv * dv)
    F0, F1, F2 = SB.eval_jet2(X, dX)
    return F2 @ w0 + 2.0 * (F1 @ w1) + F0 @ w2


def penalty_terms(SB, orbits, q, pull_margin, pull_wt):
    """Smooth diagonal boundary hinge penalty value, gradient, and diagonal Hessian."""
    if pull_wt <= 0.0 or pull_margin is None:
        return 0.0, None, None
    X, _, _, sizes = orbits_to_reps(orbits, q)
    over, grad_X = SB.base.boundary_hinge(X, pull_margin)
    if float(over.max()) <= 0.0:
        return 0.0, np.zeros(len(q), dtype=float), np.zeros(len(q), dtype=float)
    # Weight each orbit's boundary penalty by orbit size so all nodes weigh equally
    cost = 0.5 * pull_wt * float(np.sum((sizes[:, None] * over) * over))
    g_pen = np.zeros(len(q), dtype=float)
    h_pen = np.zeros(len(q), dtype=float)
    active_any = np.any(over > 0.0, axis=1)
    off = 0
    for k, o in enumerate(orbits):
        dof = o["dof"]
        if dof > 0 and active_any[k]:
            # Chain rule onto fixed subspace U: d/dxi = U^T @ grad_X[k]
            g_pen[off:off + dof] = (pull_wt * sizes[k]) * (o["U"].T @ grad_X[k])
            h_pen[off:off + dof] = pull_wt * sizes[k]
        off += dof + 1
    return cost, g_pen, h_pen


def gn_polish(SB, orbits, q, steps=5, require_interior=False):
    """Minimum-norm Gauss-Newton polish in (xi, w) driving pure moment residual to machine floor."""
    r = eval_residual(SB, orbits, q)
    cur_norm = float(np.linalg.norm(r))
    if cur_norm < 1e-15:
        return cur_norm, q
    for _ in range(steps):
        X, w, v_arr, sizes = orbits_to_reps(orbits, q)
        V, G = SB.eval_val_grad(X)
        sw = sizes * w
        n_unk = len(q)
        # Build Jacobian wrt (xi, w) directly so small weights stay well-scaled
        J = np.empty((SB.M, n_unk), dtype=float)
        off = 0
        for k, o in enumerate(orbits):
            dof = o["dof"]
            if dof > 0:
                J[:, off:off + dof] = sw[k] * (G[:, :, k].T @ o["U"])
                off += dof
            J[:, off] = sizes[k] * V[:, k]
            off += 1
        GG = J @ J.T
        try:
            cf = cho_factor(GG, lower=True, check_finite=False)
            dq_w = -(J.T @ cho_solve(cf, r, check_finite=False))
        except np.linalg.LinAlgError:
            break
        # Apply step to (xi, w) and convert back to v = sqrt(w)
        q1 = q.copy()
        off = 0
        valid = True
        for k, o in enumerate(orbits):
            dof = o["dof"]
            if dof > 0:
                q1[off:off + dof] = q[off:off + dof] + dq_w[off:off + dof]
                off += dof
            w_new = w[k] + float(dq_w[off])
            if w_new <= 0.0:
                valid = False
                break
            q1[off] = math.copysign(math.sqrt(w_new), v_arr[k] if v_arr[k] != 0.0 else 1.0)
            off += 1
        if not valid:
            break
        X1, _, _, _ = orbits_to_reps(orbits, q1)
        if require_interior and SB.base.boundary_margin(X1) <= 1e-8:
            break
        r1 = eval_residual(SB, orbits, q1)
        n1 = float(np.linalg.norm(r1))
        if not math.isfinite(n1) or n1 >= cur_norm:
            break
        q = q1
        r = r1
        cur_norm = n1
        if cur_norm < 1e-15:
            break
    return cur_norm, q


def solve_geodesic_lm(
    SB,
    orbits,
    q0,
    max_steps=150,
    win=25,
    lam0=0.2,
    geo_tau=0.05,
    pull_margin=None,
    pull_wt=0.0,
    require_interior=False,
    phase_label="P1",
    verbose=False,
):
    """Exact analytic geodesic-accelerated Cholesky LM with delayed gratification & bold uphill.

    Stopping criteria and logged residuals strictly use the pure moment residual
    (never including the boundary hinge penalty).
    """
    q = q0.copy()
    r, J, X = eval_res_jac(SB, orbits, q)
    mom = float(np.linalg.norm(r))
    bdist = SB.base.boundary_margin(X)
    p_cost, g_pen, h_pen = penalty_terms(SB, orbits, q, pull_margin, pull_wt)
    tot_cost = 0.5 * (mom * mom) + p_cost
    best_mom = mom
    best_cost = tot_cost
    best_q = q.copy()
    lam = lam0
    n_unk = len(q)
    diag_idx = np.arange(n_unk)
    hist = []
    used = 0
    v_prev = None
    in_geo = (mom < geo_tau) if geo_tau > 0.0 else True

    for it in range(1, max_steps + 1):
        used = it
        # Pure moment stop condition (ignoring penalty!)
        if mom < 1e-13 and (pull_wt <= 0.0 or bdist > 0.0):
            break
        H = J.T @ J
        g = J.T @ r
        if g_pen is not None:
            g = g + g_pen
            H[diag_idx, diag_idx] += h_pen
        d_scale = np.maximum(np.diag(H), 1e-12)
        accepted = False
        for _ in range(12):
            H_d = H.copy()
            H_d[diag_idx, diag_idx] += lam * d_scale
            try:
                cf = cho_factor(H_d, lower=True, check_finite=False)
                v = cho_solve(cf, -g, check_finite=False)
            except np.linalg.LinAlgError:
                lam = min(lam * 5.0, 1e10)
                continue
            step = v
            if in_geo:
                d2r = eval_d2_dir(SB, orbits, q, v)
                a = cho_solve(cf, -(J.T @ d2r), check_finite=False)
                vn = float(np.linalg.norm(v))
                an = float(np.linalg.norm(a))
                if vn > 0.0 and an <= 0.75 * vn:
                    step = v + 0.5 * a
            q1 = q + step
            X1, _, _, _ = orbits_to_reps(orbits, q1)
            if not np.all(np.isfinite(X1)):
                lam = min(lam * 5.0, 1e10)
                continue
            bd1 = SB.base.boundary_margin(X1)
            # Escape cap prevents runaway outside domain
            if bd1 < -0.35:
                lam = min(lam * 5.0, 1e10)
                continue
            if require_interior and bd1 <= 1e-7:
                lam = min(lam * 5.0, 1e10)
                continue
            r1 = eval_residual(SB, orbits, q1)
            mom1 = float(np.linalg.norm(r1))
            if not math.isfinite(mom1):
                lam = min(lam * 5.0, 1e10)
                continue
            pc1, _, _ = penalty_terms(SB, orbits, q1, pull_margin, pull_wt)
            c1 = 0.5 * (mom1 * mom1) + pc1
            downhill = c1 < tot_cost
            uphill_ok = False
            if (not downhill) and in_geo and v_prev is not None:
                n1 = float(np.linalg.norm(step))
                n0 = float(np.linalg.norm(v_prev))
                if n1 > 0.0 and n0 > 0.0:
                    cos_b = float(np.dot(step, v_prev)) / (n1 * n0)
                    if cos_b > 0.0 and ((1.0 - cos_b) ** 2) * c1 <= best_cost:
                        uphill_ok = True
            if downhill or uphill_ok:
                q = q1
                r, J, X = eval_res_jac(SB, orbits, q)
                mom = float(np.linalg.norm(r))
                bdist = SB.base.boundary_margin(X)
                p_cost, g_pen, h_pen = penalty_terms(SB, orbits, q, pull_margin, pull_wt)
                tot_cost = 0.5 * (mom * mom) + p_cost
                v_prev = step
                if tot_cost < best_cost:
                    best_cost = tot_cost
                    best_mom = mom
                    best_q = q.copy()
                if downhill:
                    lam = max(lam / (1.5 if in_geo else 3.0), 1e-13)
                if (not in_geo) and mom < geo_tau:
                    in_geo = True
                accepted = True
                break
            lam = min(lam * 5.0, 1e10)
        if not accepted:
            break
        hist.append(best_mom)
        if verbose and (it <= 5 or it % 10 == 0 or mom < 1e-13):
            log(f"{phase_label} it={it:3d} mom_res={mom:.3e} bdist={bdist:+.3e} lam={lam:.1e} geo={int(in_geo)}")
        if mom < 1e-13 and (pull_wt <= 0.0 or bdist > 0.0):
            break
        if win > 0 and len(hist) >= win:
            if hist[-1] > 0.99 * hist[-win] and lam > 1e-2:
                break
    # Return best point found
    X_b, _, _, _ = orbits_to_reps(orbits, best_q)
    bd_b = SB.base.boundary_margin(X_b)
    return best_mom, bd_b, used, best_q


def merge_close_orbits(orbits, ops, tol=1e-4):
    """Merge pairs of orbits whose expanded clouds collide within tol."""
    if tol <= 0.0 or len(orbits) <= 1:
        return orbits, 0
    merged = list(orbits)
    n_merged = 0
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            xi = merged[i]["c0"] + merged[i]["U"] @ merged[i]["xi"]
            for j in range(i + 1, len(merged)):
                if merged[i]["size"] != merged[j]["size"]:
                    continue
                xj = merged[j]["c0"] + merged[j]["U"] @ merged[j]["xi"]
                # Check minimum distance across G-images
                dmin = min(float(np.linalg.norm(xi - (A @ xj + t))) for A, t in ops)
                if dmin < tol:
                    wi = merged[i]["v"] ** 2
                    wj = merged[j]["v"] ** 2
                    wsum = wi + wj
                    # Find closest image of xj to xi
                    best_xj = min((A @ xj + t for A, t in ops), key=lambda p: float(np.linalg.norm(xi - p)))
                    xavg = (wi * xi + wj * best_xj) / max(wsum, 1e-30)
                    sz, c0, U, _ = orbit_subspace(xavg, ops, tol=1e-6)
                    xi_new = U.T @ (xavg - c0) if U.shape[1] > 0 else np.zeros(0, dtype=float)
                    merged[i] = {
                        "size": sz,
                        "c0": c0,
                        "U": U,
                        "dof": int(U.shape[1]),
                        "unk": int(U.shape[1]) + 1,
                        "xi": xi_new,
                        "v": math.sqrt(wsum),
                    }
                    del merged[j]
                    n_merged += 1
                    changed = True
                    break
            if changed:
                break
    return merged, n_merged


def solve_single_start(SB, orbits, verbose=False):
    """Run PI-preferred / PO-fallback solve on one orbit initialization.

    Stage A: strictly-interior geodesic LM from the interior start q0 (prevents
    jumping across the boundary into a PO basin when a nearby PI basin exists).
    Stage B (if Stage A did not reach PI): free fold-free geodesic LM (records
    any converged PO rule as fallback) followed by smooth diagonal boundary
    hinge pull (Phase 2) and penalty-free interior polish (Phase 3) with
    automatic minimum-norm Gauss-Newton end-stage.
    """
    q0 = pack_orbits(orbits)
    # Stage A: strictly interior geodesic LM directly from interior start q0
    mom0, bd0, _, q_in = solve_geodesic_lm(
        SB,
        orbits,
        q0,
        max_steps=160,
        win=30,
        lam0=0.25,
        geo_tau=0.05,
        pull_margin=None,
        pull_wt=0.0,
        require_interior=True,
        phase_label="PI",
        verbose=verbose,
    )
    if mom0 < 1e-9:
        mom0, q_in = gn_polish(SB, orbits, q_in, steps=5, require_interior=True)
        X0, _, _, _ = orbits_to_reps(orbits, q_in)
        bd0 = SB.base.boundary_margin(X0)
    if mom0 < 1e-12 and bd0 > 0.0:
        orb0 = unpack_orbits(orbits, q_in)
        nodes0, weights0 = expand_orbits(orb0, SB.ops)
        if float(weights0.min()) > 0.0:
            return {
                "quality": "PI",
                "mom_res": mom0,
                "bdist": bd0,
                "nodes": nodes0,
                "weights": weights0,
                "orbits": orb0,
            }
    # Stage B / Phase 1: free fold-free geodesic LM (can step outside and pull back)
    mom1, bd1, _, q1 = solve_geodesic_lm(
        SB,
        orbits,
        q0,
        max_steps=160,
        win=30,
        lam0=0.25,
        geo_tau=0.05,
        pull_margin=None,
        pull_wt=0.0,
        require_interior=False,
        phase_label="P1",
        verbose=verbose,
    )
    if mom1 < 1e-9:
        mom1, q1 = gn_polish(SB, orbits, q1, steps=5, require_interior=False)
        X1, _, _, _ = orbits_to_reps(orbits, q1)
        bd1 = SB.base.boundary_margin(X1)
    po_backup = None
    if mom1 < 1e-12:
        orb1 = unpack_orbits(orbits, q1)
        nodes1, weights1 = expand_orbits(orb1, SB.ops)
        if float(weights1.min()) > 0.0:
            if bd1 > 0.0:
                return {
                    "quality": "PI",
                    "mom_res": mom1,
                    "bdist": bd1,
                    "nodes": nodes1,
                    "weights": weights1,
                    "orbits": orb1,
                }
            po_backup = {
                "quality": "PO",
                "mom_res": mom1,
                "bdist": bd1,
                "nodes": nodes1,
                "weights": weights1,
                "orbits": orb1,
            }
    # Phase 2: smooth diagonal boundary hinge pull from Phase 1 back inside
    if mom1 < 1e-2:
        target_margin = 5e-4 if SB.degree >= 12 else 2e-3
        _, bd2, _, q2 = solve_geodesic_lm(
            SB,
            orbits,
            q1,
            max_steps=120,
            win=25,
            lam0=1e-2,
            geo_tau=0.0,
            pull_margin=target_margin,
            pull_wt=10.0,
            require_interior=False,
            phase_label="P2",
            verbose=verbose,
        )
        # Phase 3: turn penalty OFF once strictly inside; pure-moment polish + GN end-stage
        if bd2 > 0.0:
            mom3, bd3, _, q3 = solve_geodesic_lm(
                SB,
                orbits,
                q2,
                max_steps=120,
                win=25,
                lam0=1e-3,
                geo_tau=0.0,
                pull_margin=None,
                pull_wt=0.0,
                require_interior=True,
                phase_label="P3",
                verbose=verbose,
            )
            if mom3 < 1e-9:
                mom3, q3 = gn_polish(SB, orbits, q3, steps=5, require_interior=True)
                X3, _, _, _ = orbits_to_reps(orbits, q3)
                bd3 = SB.base.boundary_margin(X3)
            if mom3 < 1e-12 and bd3 > 0.0:
                orb3 = unpack_orbits(orbits, q3)
                nodes3, weights3 = expand_orbits(orb3, SB.ops)
                if float(weights3.min()) > 0.0:
                    return {
                        "quality": "PI",
                        "mom_res": mom3,
                        "bdist": bd3,
                        "nodes": nodes3,
                        "weights": weights3,
                        "orbits": orb3,
                    }
    return po_backup


def monomial_residual(domain, dim, degree, nodes, weights_scaled):
    """Exact max normalized monomial residual over all |alpha| <= degree."""
    vol = float(2.0 ** dim) if domain == "cube" else (1.0 / float(math.factorial(dim)))
    alphas = multi_indices(dim, degree)
    pows = []
    for j in range(dim):
        xj = nodes[:, j]
        pj = np.ones((degree + 1, len(xj)), dtype=float)
        for k in range(1, degree + 1):
            pj[k] = pj[k - 1] * xj
        pows.append(pj)
    max_err = 0.0
    for alpha in alphas:
        val = weights_scaled.copy()
        for j in range(dim):
            if alpha[j] > 0:
                val *= pows[j][alpha[j]]
        num = float(np.sum(val))
        if domain == "cube":
            if all(a % 2 == 0 for a in alpha):
                exact = 1.0
                for a in alpha:
                    exact *= 2.0 / float(a + 1)
            else:
                exact = 0.0
        else:
            num_fac = 1.0
            for a in alpha:
                num_fac *= float(math.factorial(a))
            exact = num_fac / float(math.factorial(sum(alpha) + dim))
        err = abs(num - exact) / vol
        if err > max_err:
            max_err = err
    return max_err


def infer_default_points(domain, dim, degree, sym_name, SB):
    """Compute smallest natural target N near Z respecting Stroud bound and full-dim span."""
    M_full = math.comb(degree + dim, dim)
    Z = int(math.ceil(M_full / float(dim + 1)))
    # Classical Stroud lower bound: N >= comb(floor(degree / 2) + dim, dim)
    stroud_lb = math.comb(degree // 2 + dim, dim)
    if domain == "cube" and degree >= 3:
        stroud_lb = max(stroud_lb, 2 * dim)
    if sym_name == "none":
        return max(Z, stroud_lb)
    if domain == "cube" and degree % 2 == 1 and sym_name in ("central", "chiral"):
        n_start = max(stroud_lb, int(math.ceil(SB.M * (SB.gsize / float(dim + 1)))) - dim * 2)
    else:
        n_start = max(
            stroud_lb,
            Z - (dim if (domain == "simplex" and degree % 2 == 1 and sym_name == "chiral") else 0),
        )
    need_full_dof = (degree >= 2 and dim >= 2)
    for cand_n in range(n_start, max(Z, stroud_lb) + max(32, 3 * SB.gsize) + 1):
        if domain == "cube":
            orbs = cube_equilibrium_start(dim, degree, cand_n, sym_name, SB.ops, min_unk=SB.M)
        else:
            orbs = simplex_equilibrium_start(dim, degree, cand_n, sym_name, SB.ops, min_unk=SB.M)
        if orbs is not None:
            if (not need_full_dof) or any(o["dof"] == dim for o in orbs):
                return cand_n
    return max(Z, stroud_lb)


def atlas_domain_name(domain, dim):
    """Return Atlas-style domain name and filename prefix."""
    if domain == "cube":
        names = {1: "interval", 2: "square", 3: "cube"}
        return names.get(dim, f"cube{dim}"), f"C{dim}"
    names = {1: "simplex1", 2: "triangle", 3: "tetra"}
    return names.get(dim, f"simplex{dim}"), f"T{dim}"


def solve_for_symmetry(base, sym_name, target_n, seed=None, verbose=False):
    """Run deterministic mini-ladder (plus optional seeded jitter) for one symmetry group."""
    ops = build_group_ops(base.domain, base.dim, sym_name)
    if ops is None:
        return None
    SB = SymmetricBasis(base, sym_name, ops)
    n_use = target_n if target_n > 0 else infer_default_points(base.domain, base.dim, base.degree, sym_name, SB)
    log(f"Trying symmetry={sym_name} (|G|={SB.gsize}, M_G={SB.M}, N={n_use})")
    # Deterministic generic mini-ladder (no per-configuration special cases)
    if base.domain == "cube":
        ladder = [
            (0.30, 6.0, 12.0),
            (0.45, 8.0, 12.0),
            (0.25, 4.0, 0.0),
            (0.55, 11.0, 8.0),
        ]
    else:
        ladder = [
            (0.50, 0.06, 0.0),
            (1.10, 0.12, 0.0),
            (1.80, 0.04, 0.0),
            (0.30, 0.18, 0.0),
        ]
    best_po = None
    first_orbs = None
    for idx, (p1, p2, p3) in enumerate(ladder):
        if base.domain == "cube":
            orbs = cube_equilibrium_start(
                base.dim, base.degree, n_use, sym_name, ops, r0=p1, angle=p2, pnorm=p3, min_unk=SB.M
            )
        else:
            orbs = simplex_equilibrium_start(
                base.dim, base.degree, n_use, sym_name, ops, r0=p1, twist=p2, min_unk=SB.M
            )
        if orbs is None:
            if idx == 0:
                log(f"symmetry={sym_name}: N={n_use} infeasible for orbit/unknown count (M_G={SB.M})")
                return None
            continue
        if first_orbs is None:
            first_orbs = orbs
        unk = sum(o["unk"] for o in orbs)
        log(f"symmetry={sym_name} ladder[{idx}] unk={unk}/{SB.M} (orbits={len(orbs)})")
        res = solve_single_start(SB, orbs, verbose=verbose)
        if res is not None:
            log(
                f"symmetry={sym_name} ladder[{idx}] -> {res['quality']} "
                f"mom_res={res['mom_res']:.2e} bdist={res['bdist']:+.3e}"
            )
            if res["quality"] == "PI":
                return res, SB, n_use
            if best_po is None or res["bdist"] > best_po["bdist"]:
                best_po = res
    # Optional seeded perturbation stage when --seed is supplied
    if seed is not None and first_orbs is not None:
        rng = np.random.default_rng(seed)
        for trial in range(8):
            j_orbs = []
            scale = 0.03 if trial < 4 else 0.08
            for o in first_orbs:
                no = dict(o)
                if o["dof"] > 0:
                    no["xi"] = o["xi"] + rng.normal(0.0, scale, size=o["dof"])
                no["v"] = o["v"] * math.exp(float(rng.normal(0.0, 0.05)))
                j_orbs.append(no)
            res = solve_single_start(SB, j_orbs, verbose=verbose)
            if res is not None:
                log(
                    f"symmetry={sym_name} seed={seed} trial={trial} -> {res['quality']} "
                    f"mom_res={res['mom_res']:.2e} bdist={res['bdist']:+.3e}"
                )
                if res["quality"] == "PI":
                    return res, SB, n_use
                if best_po is None or res["bdist"] > best_po["bdist"]:
                    best_po = res
    if best_po is not None:
        return best_po, SB, n_use
    return None


def symmetry_schedule(domain, dim, degree):
    """Generic symmetry trial order when --symmetry is omitted."""
    if dim == 1:
        return ["full", "none"]
    if domain == "cube":
        if degree % 2 == 1:
            return ["chiral", "central", "full", "none"]
        return ["none", "central", "chiral", "full"]
    return ["chiral", "full", "none"]


def save_atlas_json(out_path, domain, dim, degree, res, sym_name, cmd):
    """Write rule in Atlas-standard JSON format."""
    dom_name, prefix = atlas_domain_name(domain, dim)
    vol = float(2.0 ** dim) if domain == "cube" else (1.0 / float(math.factorial(dim)))
    nodes = res["nodes"]
    weights = res["weights"] * vol
    N = len(weights)
    qual = res["quality"]
    rule_id = f"CA-{prefix}-{degree:02d}-{N}-{qual}-MPC"
    merr = monomial_residual(domain, dim, degree, nodes, weights)
    fam = (
        "minimal_pointcount_for_degree_n_PI"
        if qual == "PI"
        else "minimal_pointcount_for_degree_n_PO"
    )
    record = {
        "id": rule_id,
        "family": fam,
        "domain": dom_name,
        "dimension": dim,
        "degree": degree,
        "points": int(N),
        "quality": qual,
        "symmetry": sym_name,
        "origin": "found",
        "method": (
            f"Generic deterministic equilibrium-start solver (domain={domain}, dim={dim}, "
            f"symmetry={sym_name}) with fold-free geodesic-accelerated Cholesky LM, "
            f"smooth boundary hinge pull, and minimum-norm Gauss-Newton end-stage"
        ),
        "author": "Antigravity (AI-assisted)",
        "reproduce": {
            "script": "code/cubature_generic.py",
            "command": cmd,
        },
        "status": "numerical",
        "residual_reported": float(merr),
        "nodes": [[float(c) for c in row] for row in nodes],
        "weights": [float(w) for w in weights],
    }
    if out_path is None:
        out_path = f"{rule_id}.json"
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
        fh.write("\n")
    return out_path, rule_id, merr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dim", type=int, default=2, help="spatial dimension d >= 1 (default: 2)")
    ap.add_argument("--degree", type=int, default=7, help="polynomial degree n >= 1 (default: 7)")
    ap.add_argument(
        "--domain",
        choices=("simplex", "cube"),
        default="simplex",
        help="reference domain: simplex or cube (default: simplex)",
    )
    ap.add_argument(
        "--symmetry",
        choices=("auto", "none", "central", "chiral", "full"),
        default="auto",
        help="symmetry group (default: auto = try all applicable symmetries)",
    )
    ap.add_argument(
        "--points",
        type=int,
        default=0,
        help="target number of nodes N (default: 0 = automatic near Z)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="optional RNG seed for jittered fallback starts after deterministic mini-ladder",
    )
    a = ap.parse_args()
    if a.dim < 1:
        raise ValueError("dim must be >= 1")
    if a.degree < 1:
        raise ValueError("degree must be >= 1")
    t0 = time.time()
    base = CubeBasis(a.dim, a.degree) if a.domain == "cube" else SimplexBasis(a.dim, a.degree)
    Z = int(math.ceil(base.M / float(a.dim + 1)))
    log(
        f"Start generic solver: domain={a.domain} dim={a.dim} degree={a.degree} "
        f"M={base.M} Z={Z} symmetry={a.symmetry} points={a.points} seed={a.seed}"
    )
    syms = symmetry_schedule(a.domain, a.dim, a.degree) if a.symmetry == "auto" else [a.symmetry]
    best = None
    best_sym = None
    for sym in syms:
        if a.points == 0 and sym == "none" and best is not None and best["quality"] == "PI" and len(best["weights"]) <= Z:
            continue
        out = solve_for_symmetry(base, sym, a.points, seed=a.seed, verbose=False)
        if out is None:
            continue
        res, _, _ = out
        if res["quality"] == "PI":
            if best is None or best["quality"] != "PI" or len(res["weights"]) < len(best["weights"]):
                best = res
                best_sym = sym
            if a.points > 0:
                break
        elif best is None or (best["quality"] != "PI" and res["bdist"] > best["bdist"]):
            best = res
            best_sym = sym
    elapsed = time.time() - t0
    if best is None:
        log(f"No solution found in {elapsed:.2f}s")
        sys.exit(1)
    cmd_args = [
        "--dim", str(a.dim),
        "--degree", str(a.degree),
        "--domain", a.domain,
        "--symmetry", best_sym,
        "--points", str(len(best["weights"])),
    ]
    if a.seed is not None:
        cmd_args.extend(["--seed", str(a.seed)])
    cmd = "python code/cubature_generic.py " + shlex.join(cmd_args)
    out_path, rule_id, merr = save_atlas_json(None, a.domain, a.dim, a.degree, best, best_sym, cmd)
    log(
        f"FOUND {rule_id} ({best['quality']}, sym={best_sym}) "
        f"N={len(best['weights'])} mom_res={best['mom_res']:.2e} monom={merr:.2e} "
        f"min_w={float(best['weights'].min() * base.vol):.3e} bdist={best['bdist']:+.3e} "
        f"wrote={out_path} ({elapsed:.2f}s)"
    )


if __name__ == "__main__":
    main()

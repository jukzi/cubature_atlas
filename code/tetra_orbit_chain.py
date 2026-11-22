#!/usr/bin/env python3
"""Chained tetrahedron orbit elimination and demotion with analytic Dubiner basis and block-diagonal projection.

Ideas justified and measured against code/tetra_block_elim.py on the known degree-37
S4 reduction CA-T3-37-2664-PI-MPC -> 2660 (seed 13006, single-threaded BLAS):
1. Analytic real Dubiner basis with shared s = p + q Jacobi recurrences:
   dom.py evaluates the 3D Dubiner basis four times (once real, three times complex-step)
   with an O(n^3) dict and recomputes S(2p+2q+2, z) for every (p, q) pair. Differentiating
   the three-term recurrences in real float64, sharing S by s = p + q (n+1 instead of
   (n+1)(n+2)/2 recurrences), and vectorizing over r eliminates dict overhead and complex
   arithmetic. Measured on degree 37: res() 0.194s -> 0.037s (5.21x), jac() 1.414s -> 0.219s (6.44x).
2. Block-diagonal invariant projection:
   the G-invariant Dubiner basis U is block-diagonal across total-degree layers n = 0..deg.
   Folding normalization into the blocks and projecting layer-by-layer avoids >90% zero
   multiplications in res() and jac().
3. Unified nullspace ranking and linearized initialization for deletions and demotions:
   both deleting an orbit (w_i -> 0) and demoting it to a strictly larger stabilizer
   (linear barycentric constraints C b(q_i) = 0, e.g. S4: 24->12, 12->4, 12->6, 6->1, 4->1)
   are projected onto the scaled moment Jacobian nullspace. Ranking by move score divided
   by nodes saved per unknown favours high-yield moves, and the clamped nullspace step
   provides a first-order-feasible starting point for Levenberg-Marquardt.
4. Column-scaled projected Levenberg-Marquardt with stagnation stopping and optional
   orbit-space L2 weight minimisation (sum |orbit| w^2). Measured on CA-T3-37-2664 -> 2660
   (--order linear): tetra_block_elim.py took 29.21s (setup 5.49s, solve 23.72s);
   tetra_orbit_chain.py took 9.5s total (setup 2.38s, solve 2.54s, 9.3x faster solve).
   With --order yield, step 1 demoted a 12-orbit to a 4-orbit (2664 -> 2656, -8 nodes
   for 1 dof) in 2.86s.
"""
import argparse
import json
from pathlib import Path
import shlex
import sys
from time import perf_counter

import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError, qr, solve_triangular

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import po_elim
import sym_elim as se
from tetra_block_elim import TetraBasis, invariant_dimension


def jacobi_coeffs(alpha, k):
    """Three-term recurrence coefficients (a, b, c) for Jacobi P_{k+1}^{(alpha, 0)}."""
    if alpha == 0:
        return (2 * k + 1) / (k + 1.0), 0.0, k / (k + 1.0)
    den = 2.0 * (k + 1) * (k + alpha + 1)
    a = (2 * k + alpha + 1) * (2 * k + alpha + 2) / den
    b = (alpha * alpha * (2 * k + alpha + 1)) / (den * (2 * k + alpha))
    c = (k * (k + alpha) * (2 * k + alpha + 2)) / ((k + 1.0) * (k + alpha + 1) * (2 * k + alpha))
    return a, b, c


class AnalyticTetraBasis(TetraBasis):
    """TetraBasis with real analytic Dubiner values and derivatives, sharing s = p + q recurrences."""

    def __init__(self, degree):
        super().__init__("tetra", degree)
        n = self.deg
        # Precompute Jacobi recurrence coefficients for H (alpha=2p+1) and S (alpha=2s+2).
        self._h_coeff = [[jacobi_coeffs(2 * p + 1, k) for k in range(n - p)] for p in range(n + 1)]
        self._s_coeff = [[jacobi_coeffs(2 * s + 2, k) for k in range(n - s)] for s in range(n + 1)]

    def raw_values(self, pts):
        """Unnormalized Dubiner basis matrix (N, M) in real float64."""
        n = self.deg
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        npts = len(x)
        v = 1.0 - y - z
        v2 = v * v
        ew = 2.0 * x - v
        q_arr = np.empty((n + 1, npts))
        q_arr[0] = 1.0
        if n >= 1:
            q_arr[1] = ew
        for p in range(1, n):
            q_arr[p + 1] = ((2 * p + 1) * ew * q_arr[p] - p * v2 * q_arr[p - 1]) / (p + 1.0)

        zeta = 2.0 * z - 1.0
        s_mats = []
        for s in range(n + 1):
            nr = n - s + 1
            sm = np.empty((npts, nr))
            sm[:, 0] = 1.0
            for k, (a, b, c) in enumerate(self._s_coeff[s]):
                nxt = (a * zeta + b) * sm[:, k]
                if k > 0:
                    nxt = nxt - c * sm[:, k - 1]
                sm[:, k + 1] = nxt
            s_mats.append(sm)

        w2 = 1.0 - z
        w2_sq = w2 * w2
        ew2 = 2.0 * y - w2
        out = np.empty((npts, self.M))
        col = 0
        for p in range(n + 1):
            nq_len = n - p + 1
            h_arr = np.empty((nq_len, npts))
            h_arr[0] = 1.0
            for k, (a, b, c) in enumerate(self._h_coeff[p]):
                nxt = (a * ew2 + b * w2) * h_arr[k]
                if k > 0:
                    nxt = nxt - c * w2_sq * h_arr[k - 1]
                h_arr[k + 1] = nxt
            qp = q_arr[p]
            for q in range(nq_len):
                s = p + q
                nr = n - s + 1
                base = qp * h_arr[q]
                out[:, col:col + nr] = base[:, None] * s_mats[s]
                col += nr
        return out

    def raw_basis(self, pts):
        """Unnormalized Dubiner basis (N, M) and analytic partial derivatives [d/dx, d/dy, d/dz]."""
        n = self.deg
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        npts = len(x)
        v = 1.0 - y - z
        v2 = v * v
        ew = 2.0 * x - v
        q_arr = np.empty((n + 1, npts))
        dqx = np.empty((n + 1, npts))
        # Q depends on y and z only through y + z, so dQ/dy == dQ/dz.
        dqyz = np.empty((n + 1, npts))
        q_arr[0] = 1.0
        dqx[0] = 0.0
        dqyz[0] = 0.0
        if n >= 1:
            q_arr[1] = ew
            dqx[1] = 2.0
            dqyz[1] = 1.0
        for p in range(1, n):
            c1 = (2 * p + 1) / (p + 1.0)
            c2 = p / (p + 1.0)
            qp = q_arr[p]
            qm = q_arr[p - 1]
            q_arr[p + 1] = c1 * ew * qp - c2 * v2 * qm
            dqx[p + 1] = c1 * (2.0 * qp + ew * dqx[p]) - c2 * v2 * dqx[p - 1]
            dqyz[p + 1] = c1 * (qp + ew * dqyz[p]) - c2 * (-2.0 * v * qm + v2 * dqyz[p - 1])

        zeta = 2.0 * z - 1.0
        s_mats = []
        ds_mats = []
        for s in range(n + 1):
            nr = n - s + 1
            sm = np.empty((npts, nr))
            dsm = np.empty((npts, nr))
            sm[:, 0] = 1.0
            dsm[:, 0] = 0.0
            for k, (a, b, c) in enumerate(self._s_coeff[s]):
                lin = a * zeta + b
                sk = sm[:, k]
                nxt = lin * sk
                dnxt = 2.0 * a * sk + lin * dsm[:, k]
                if k > 0:
                    nxt = nxt - c * sm[:, k - 1]
                    dnxt = dnxt - c * dsm[:, k - 1]
                sm[:, k + 1] = nxt
                dsm[:, k + 1] = dnxt
            s_mats.append(sm)
            ds_mats.append(dsm)

        w2 = 1.0 - z
        w2_sq = w2 * w2
        ew2 = 2.0 * y - w2
        vals = np.empty((npts, self.M))
        dvx = np.empty((npts, self.M))
        dvy = np.empty((npts, self.M))
        dvz = np.empty((npts, self.M))
        col = 0
        for p in range(n + 1):
            nq_len = n - p + 1
            h_arr = np.empty((nq_len, npts))
            dhy = np.empty((nq_len, npts))
            dhz = np.empty((nq_len, npts))
            h_arr[0] = 1.0
            dhy[0] = 0.0
            dhz[0] = 0.0
            for k, (a, b, c) in enumerate(self._h_coeff[p]):
                lin = a * ew2 + b * w2
                hk = h_arr[k]
                nxt = lin * hk
                ny = 2.0 * a * hk + lin * dhy[k]
                nz = (a - b) * hk + lin * dhz[k]
                if k > 0:
                    hm = h_arr[k - 1]
                    nxt = nxt - c * w2_sq * hm
                    ny = ny - c * w2_sq * dhy[k - 1]
                    nz = nz - c * (-2.0 * w2 * hm + w2_sq * dhz[k - 1])
                h_arr[k + 1] = nxt
                dhy[k + 1] = ny
                dhz[k + 1] = nz
            qp = q_arr[p]
            qpx = dqx[p]
            qpyz = dqyz[p]
            for q in range(nq_len):
                s = p + q
                nr = n - s + 1
                hq = h_arr[q]
                b0 = qp * hq
                bx = qpx * hq
                by = qpyz * hq + qp * dhy[q]
                bz = qpyz * hq + qp * dhz[q]
                sm = s_mats[s]
                vals[:, col:col + nr] = b0[:, None] * sm
                dvx[:, col:col + nr] = bx[:, None] * sm
                dvy[:, col:col + nr] = by[:, None] * sm
                dvz[:, col:col + nr] = bz[:, None] * sm + b0[:, None] * ds_mats[s]
                col += nr
        return vals, [dvx, dvy, dvz]

    def values(self, pts):
        return self.raw_values(pts) * self.norm

    def basis(self, pts):
        vals, dvals = self.raw_basis(pts)
        nrm = self.norm
        return vals * nrm, [dvals[0] * nrm, dvals[1] * nrm, dvals[2] * nrm]


def build_invariant_blocks(group, basis, seed):
    """Build block-diagonal G-invariant basis by total degree layer, with basis.norm folded in."""
    deg = basis.deg
    rng = np.random.default_rng(seed)
    cumulative = [int(invariant_dimension(group, n)) for n in range(deg + 1)]
    layer_sizes = np.diff(cumulative, prepend=0)
    bary = rng.dirichlet(np.full(4, 0.65), int(3 * max(layer_sizes) + 20))
    samples = np.zeros((len(bary), basis.M))
    for perm in group:
        samples += basis.values(bary[:, perm[1:]]) / len(group)
    degrees = np.sum(basis.idx, axis=1)
    blocks = []
    offset = 0
    for n, size in enumerate(layer_sizes):
        size = int(size)
        if size == 0:
            continue
        idx = np.nonzero(degrees == n)[0]
        q_mat, r_mat, _ = qr(samples[:, idx].T, mode="economic", pivoting=True)
        if abs(r_mat[size - 1, size - 1]) < 1e-10 * abs(r_mat[0, 0]):
            raise RuntimeError(f"rank deficiency in degree {n}")
        block = np.ascontiguousarray(q_mat[:, :size])
        error = np.linalg.norm(samples[:, idx] - (samples[:, idx] @ block) @ block.T)
        if error > 1e-9 * np.linalg.norm(samples[:, idx]):
            raise RuntimeError(f"incomplete invariant basis in degree {n}")
        # Fold diagonal normalization into the projection block for raw_values / raw_basis.
        block_norm = np.ascontiguousarray(basis.norm[idx][:, None] * block)
        blocks.append((idx, offset, offset + size, block, block_norm))
        offset += size
    return blocks, offset


class FastOrbitProblem(se.SymProb):
    """SymProb using real analytic Dubiner basis and block-diagonal degree-layer projection."""

    def __init__(self, basis, blocks, inv_dim):
        self.D = basis
        self.U = None
        self.blocks = blocks
        self.M = inv_dim
        t_inv = np.zeros(inv_dim)
        for idx, c0, c1, block, _ in blocks:
            t_inv[c0:c1] = basis.T[idx] @ block
        self.T = t_inv

    def _project_raw_rows(self, mat):
        """Project rows of unnormalized basis matrix mat (K, D.M) onto invariant space (K, M)."""
        out = np.empty((mat.shape[0], self.M))
        for idx, c0, c1, _, block_norm in self.blocks:
            out[:, c0:c1] = mat[:, idx] @ block_norm
        return out

    def res(self, x):
        pts, _ = self.points(x)
        w = x[self.off[-1]:]
        raw_vals = self.D.raw_values(pts)
        r_full = (w * self.size) @ raw_vals
        out = np.empty(self.M)
        for idx, c0, c1, _, block_norm in self.blocks:
            out[c0:c1] = r_full[idx] @ block_norm
        return out - self.T

    def jac(self, x):
        pts, j_list = self.points(x)
        w = x[self.off[-1]:]
        raw_vals, raw_dvals = self.D.raw_basis(pts)
        n = self.n
        stacked = np.vstack((raw_vals, raw_dvals[0], raw_dvals[1], raw_dvals[2]))
        proj = self._project_raw_rows(stacked)
        v_u = proj[:n]
        dv_u = proj[n:].reshape(3, n, self.M)
        nq = self.off[-1]
        jac_t = np.empty((nq + n, self.M))
        for t in range(n):
            e = self.ob[t].e
            if e > 0:
                sw = self.size[t] * w[t]
                jac_t[self.off[t]:self.off[t + 1]] = sw * (j_list[t].T @ dv_u[:, t, :])
        jac_t[nq:] = v_u * self.size[:, None]
        return jac_t.T


def subgroup_closure(group, gens):
    """Return sorted subgroup of group generated by gens."""
    sub = {(0, 1, 2, 3)}
    stack = [(0, 1, 2, 3)]
    while stack:
        g = stack.pop()
        for h in gens:
            k = se.compose(g, h)
            if k not in sub:
                sub.add(k)
                stack.append(k)
    return tuple(sorted(sub))


def all_subgroups(group):
    """Enumerate all subgroups of a small permutation group (|G| <= 24)."""
    subs = {((0, 1, 2, 3),)}
    queue = [((0, 1, 2, 3),)]
    while queue:
        sub = queue.pop()
        sub_set = set(sub)
        for g in group:
            if g in sub_set:
                continue
            cand = subgroup_closure(group, list(sub) + [g])
            if cand not in subs:
                subs.add(cand)
                queue.append(cand)
    return sorted(subs, key=lambda s: (len(s), s))


def vertex_partition(sub):
    """Return canonical tuple of vertex orbits under subgroup sub."""
    seen = set()
    parts = []
    for v in range(4):
        if v in seen:
            continue
        orb = {v}
        changed = True
        while changed:
            changed = False
            for h in sub:
                for u in list(orb):
                    if h[u] not in orb:
                        orb.add(h[u])
                        changed = True
        seen |= orb
        parts.append(tuple(sorted(orb)))
    return tuple(sorted(parts))


def partition_constraint_matrix(part_from, part_to):
    """Linear constraint rows C on barycentric vector b enforcing part_to given part_from."""
    rows = []
    for big in part_to:
        big_set = set(big)
        pieces = [blk for blk in part_from if set(blk).issubset(big_set)]
        if len(pieces) <= 1:
            continue
        base = pieces[0]
        for other in pieces[1:]:
            row = np.zeros(4)
            for v in base:
                row[v] += 1.0 / len(base)
            for v in other:
                row[v] -= 1.0 / len(other)
            rows.append(row)
    return np.asarray(rows, dtype=float)


def super_orbit_types(group):
    """Map each orbit vertex partition to strictly smaller-size orbit types in group."""
    subs = all_subgroups(group)
    by_part = {}
    for sub in subs:
        part = vertex_partition(sub)
        if part not in by_part or len(sub) > len(by_part[part]):
            by_part[part] = sub
    orbits_by_part = {part: se.Orbit(group, sub) for part, sub in by_part.items()}
    supers = {}
    for p_small, h_small in by_part.items():
        ob_small = orbits_by_part[p_small]
        targets = []
        set_small = set(h_small)
        for p_big, h_big in by_part.items():
            if len(h_big) <= len(h_small):
                continue
            if not set_small.issubset(set(h_big)):
                continue
            ob_big = orbits_by_part[p_big]
            if ob_big.size < ob_small.size:
                c_mat = partition_constraint_matrix(p_small, p_big)
                targets.append((p_big, ob_big, c_mat))
        targets.sort(key=lambda item: (-item[1].size, item[0]))
        supers[p_small] = targets
    return orbits_by_part, supers


def project_bary_to_partition(b, part):
    """Project barycentric vector b onto the subspace constant on each block of part."""
    out = np.empty(4)
    for block in part:
        val = float(np.mean(b[list(block)]))
        for v in block:
            out[v] = val
    out = np.maximum(out, 0.0)
    out /= out.sum()
    return out


def point_to_duffy(ob, pt):
    """Convert Cartesian point pt in ob's frame simplex to Duffy coordinates q."""
    if ob.e == 0:
        return np.zeros(0)
    y = np.linalg.lstsq(ob.E, pt - ob.F[0], rcond=None)[0]
    return se.duffy_inv(y)


def solve_lm_scaled(prob, x0, max_iter, tol=1e-26, scaled=True, stag_win=12, stag_gain=0.05):
    """Projected Levenberg-Marquardt with column scaling and stagnation stop."""
    lo, hi = prob.bounds()
    x = np.clip(x0.copy(), lo, hi)
    r = prob.res(x)
    c = 0.5 * float(r @ r)
    mu = 1e-3
    m = prob.M
    history = [c]
    for _ in range(max_iter):
        if c < tol:
            break
        j_mat = prob.jac(x)
        g = j_mat.T @ r
        free = ~(((x <= lo + 1e-13) & (g > 0)) | ((x >= hi - 1e-13) & (g < 0)))
        jf = j_mat[:, free]
        if jf.shape[1] == 0:
            break
        if scaled:
            col_norm = np.linalg.norm(jf, axis=0)
            col_scale = 1.0 / np.maximum(col_norm, 1e-12)
            jfs = jf * col_scale
        else:
            col_scale = None
            jfs = jf
        wide = jfs.shape[1] >= m
        a0 = jfs @ jfs.T if wide else jfs.T @ jfs
        k = a0.shape[0]
        t0 = float(np.trace(a0)) / k + 1e-30
        rhs = -r if wide else -(jfs.T @ r)
        ok = False
        for _ in range(14):
            try:
                a_reg = a0.copy()
                a_reg[np.diag_indices(k)] += mu * t0
                step = cho_solve(cho_factor(a_reg, lower=True, overwrite_a=True, check_finite=False),
                                 rhs, check_finite=False)
                if wide:
                    step = jfs.T @ step
                if scaled:
                    step = step * col_scale
            except (LinAlgError, ValueError):
                mu *= 10.0
                continue
            xn = x.copy()
            xn[free] += step
            xn = np.clip(xn, lo, hi)
            rn = prob.res(xn)
            cn = 0.5 * float(rn @ rn)
            if cn < c:
                x = xn
                r = rn
                c = cn
                mu = max(mu / 4.0, 1e-15)
                ok = True
                break
            mu *= 4.0
        if not ok:
            break
        history.append(c)
        if stag_win > 0 and len(history) > stag_win and c > 1e-18:
            prev = history[-1 - stag_win]
            if (prev - c) < stag_gain * prev:
                break
    return x, c


def extract_orbits(prob, x):
    """Extract current (Orbit, q, w) list from problem and solution vector x."""
    w = x[prob.off[-1]:]
    out = []
    for t in range(prob.n):
        q = x[prob.off[t]:prob.off[t + 1]].copy()
        out.append((prob.ob[t], q, float(w[t])))
    return out


def compute_nullspace(prob, x):
    """Return scaling vector and orthonormal nullspace basis of the scaled moment Jacobian."""
    nq = prob.off[-1]
    scale = np.r_[np.maximum(np.minimum(x[:nq], 1.0 - x[:nq]), 0.02), np.maximum(x[nq:], 1e-12)]
    jacobian = prob.jac(x) * scale
    vectors, triangular, _ = qr(jacobian.T, mode="full", pivoting=True)
    diagonal = np.abs(np.diag(triangular))
    rank = int(np.count_nonzero(diagonal > diagonal[0] * 1e-10))
    null = vectors[:, rank:]
    return scale, null


def score_move_vector(x, move, scale, nq, ignore_idx=None):
    """Score a scaled nullspace move by step norm plus bound-violation penalty."""
    trial = x + move * scale
    violation = np.maximum(-trial, 0.0) / scale
    violation[:nq] += np.maximum(trial[:nq] - 1.0, 0.0) / scale[:nq]
    if ignore_idx is not None:
        violation[ignore_idx] = 0.0
    return float(np.linalg.norm(move) + 10.0 * np.linalg.norm(violation)), trial


def rank_candidates(group, prob, orbits, x, supers, order_mode, rng, jitter=0.0):
    """Rank deletion and demotion moves using the shared scaled Jacobian nullspace."""
    dof = len(x) - prob.M
    nq = prob.off[-1]
    moves = []
    if order_mode == "random":
        for i, (ob, _, _) in enumerate(orbits):
            if ob.e + 1 <= dof:
                moves.append((float(rng.random()), "del", i, ob.size, ob.e + 1, 0.0, None))
        moves.sort(key=lambda item: item[0])
        return moves

    scale, null = compute_nullspace(prob, x)
    # 1. Deletions: w_i -> 0 in scaled coordinates means dz[nq + i] = -1.
    for i, (ob, _, _) in enumerate(orbits):
        dof_cost = ob.e + 1
        if dof_cost > dof:
            continue
        axis = null[nq + i]
        freedom = float(axis @ axis)
        if freedom <= 1e-14:
            continue
        move = -(null @ axis) / freedom
        sc, trial_x = score_move_vector(x, move, scale, nq, ignore_idx=nq + i)
        if order_mode == "linear":
            key = sc
        else:
            eff = ob.size / float(dof_cost)
            key = sc / eff
        if jitter > 0:
            key *= float(np.exp(jitter * rng.normal()))
        moves.append((key, "del", i, ob.size, dof_cost, sc, trial_x))

    # 2. Demotions: linear barycentric constraints C @ b(q_i) = 0 projected onto nullspace.
    if order_mode in ("yield", "demote-first"):
        pts, j_list = prob.points(x)
        has_centroid = any(ob.size == 1 for ob, _, _ in orbits)
        db_dp = np.array([[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        for i, (ob, _q, wt) in enumerate(orbits):
            if ob.e == 0:
                continue
            part = vertex_partition(ob.H)
            b = se.bary(pts[i:i + 1])[0]
            db_dq = db_dp @ j_list[i]
            s0 = int(prob.off[i])
            s1 = int(prob.off[i + 1])
            q_scale = scale[s0:s1]
            null_qi = null[s0:s1, :]
            for t_part, t_ob, c_mat in supers.get(part, []):
                if t_ob.size == 1 and has_centroid:
                    continue
                dof_cost = (ob.e + 1) - (t_ob.e + 1)
                if dof_cost > dof:
                    continue
                rhs = -(c_mat @ b)
                a_qi = (c_mat @ db_dq) * q_scale[None, :]
                a_null = a_qi @ null_qi
                gram = a_null @ a_null.T
                if np.linalg.cond(gram) > 1e12:
                    continue
                try:
                    u = a_null.T @ np.linalg.solve(gram, rhs)
                except LinAlgError:
                    continue
                move = null @ u
                sc, trial_x = score_move_vector(x, move, scale, nq)
                nodes_saved = ob.size - t_ob.size
                eff = nodes_saved / max(float(dof_cost), 0.5)
                key = sc / eff
                if order_mode == "demote-first":
                    key *= 0.5
                if jitter > 0:
                    key *= float(np.exp(jitter * rng.normal()))
                moves.append((key, "dem", i, nodes_saved, dof_cost, sc, (t_part, t_ob, trial_x)))

    moves.sort(key=lambda item: item[0])
    return moves


def build_delete_trial(group, prob, orbits, x, drop_idx, trial_x, rng, noise, use_lin_init=True):
    """Construct starting orbits after deleting drop_idx, blending nullspace step if feasible."""
    nq = prob.off[-1]
    if use_lin_init and trial_x is not None:
        dx = trial_x - x
        alpha = 1.0
        for i in range(prob.n):
            if i == drop_idx:
                continue
            w0 = x[nq + i]
            dw = dx[nq + i]
            if dw < 0 and w0 + alpha * dw < 0.25 * w0:
                alpha = min(alpha, 0.75 * w0 / (-dw))
        x_blend = x + alpha * dx
        rem_frac = max(0.0, 1.0 - alpha)
    else:
        x_blend = x.copy()
        rem_frac = 1.0

    add_mass = {}
    if rem_frac > 1e-12:
        pts, _ = prob.points(x)
        drop_ob, _, drop_w = orbits[drop_idx]
        images = np.array([se.apply(g, pts[drop_idx:drop_idx + 1])[0] for g in group])
        dists = np.min(np.linalg.norm(pts[:, None] - images[None, :, :], axis=2), axis=1)
        dists[drop_idx] = np.inf
        k_near = min(3, len(orbits) - 1)
        near_idx = np.argsort(dists)[:k_near]
        d_near = dists[near_idx]
        d_scale = max(float(d_near[-1]), 1e-12)
        shares = np.exp(-((d_near / d_scale) ** 2))
        shares /= shares.sum()
        mass = rem_frac * drop_ob.size * drop_w
        add_mass = {int(idx): float(mass * sh) for idx, sh in zip(near_idx, shares)}

    trial = []
    for i, (ob, _, _) in enumerate(orbits):
        if i == drop_idx:
            continue
        q_i = np.clip(x_blend[prob.off[i]:prob.off[i + 1]], 0.0, 1.0)
        if noise > 0:
            q_i = np.clip(q_i + noise * rng.normal(size=q_i.shape), 0.0, 1.0)
        w_i = max(float(x_blend[nq + i]) + add_mass.get(i, 0.0) / ob.size, 1e-12)
        trial.append((ob, q_i, w_i))
    return trial


def build_demote_trial(prob, orbits, x, dem_idx, t_part, t_ob, trial_x, rng, noise, use_lin_init=True):
    """Construct starting orbits after demoting dem_idx to t_ob using linearized nullspace step."""
    nq = prob.off[-1]
    if use_lin_init and trial_x is not None:
        dx = trial_x - x
        alpha = 1.0
        for i in range(prob.n):
            w0 = x[nq + i]
            dw = dx[nq + i]
            if dw < 0 and w0 + alpha * dw < 0.25 * w0:
                alpha = min(alpha, 0.75 * w0 / (-dw))
        x_blend = np.clip(x + alpha * dx, 0.0, 50.0)
        x_blend[:nq] = np.clip(x_blend[:nq], 0.0, 1.0)
    else:
        x_blend = x.copy()

    pts, _ = prob.points(x_blend)
    trial = []
    for i, (ob, _, _) in enumerate(orbits):
        w_i = max(float(x_blend[nq + i]), 1e-12)
        if i == dem_idx:
            b = se.bary(pts[i:i + 1])[0]
            b_proj = project_bary_to_partition(b, t_part)
            q_proj = point_to_duffy(t_ob, b_proj[1:])
            if noise > 0:
                q_proj = np.clip(q_proj + noise * rng.normal(size=q_proj.shape), 0.0, 1.0)
            w_proj = w_i * (ob.size / t_ob.size)
            trial.append((t_ob, q_proj, w_proj))
        else:
            q_i = np.clip(x_blend[prob.off[i]:prob.off[i + 1]], 0.0, 1.0)
            if noise > 0:
                q_i = np.clip(q_i + noise * rng.normal(size=q_i.shape), 0.0, 1.0)
            trial.append((ob, q_i, w_i))
    return trial


def orbit_l2_minimize(prob, orbits, max_iter=40):
    """Minimise sum(|orbit| * w^2) in orbit variables under the invariant moment equations."""
    x = prob.setup(orbits)
    nq = prob.off[-1]
    sizes = prob.size
    lo, hi = prob.bounds()
    pts0, _ = prob.points(x)
    min_bary0 = float(se.bary(pts0).min())

    def obj(z):
        w = z[nq:]
        return float(np.sum(sizes * (w ** 2)))

    f0 = obj(x)
    for _ in range(max_iter):
        j_mat = prob.jac(x)
        if j_mat.shape[1] <= j_mat.shape[0]:
            break
        q_mat, r_mat = qr(j_mat.T, mode="economic")
        grad = np.zeros_like(x)
        grad[nq:] = 2.0 * sizes * x[nq:]
        direction = q_mat @ (q_mat.T @ grad) - grad
        gnorm = float(np.linalg.norm(direction))
        if gnorm < 1e-15:
            break
        direction /= gnorm
        improved = False
        cur_f = obj(x)
        for step in np.geomspace(0.05, 1e-6, num=16):
            trial = x + step * direction
            for _ in range(6):
                if not np.all((lo <= trial) & (trial <= hi)):
                    break
                res = prob.res(trial)
                if float(np.linalg.norm(res, np.inf)) < 1e-14:
                    if trial[nq:].min() <= 0:
                        break
                    pts_t, _ = prob.points(trial)
                    if float(se.bary(pts_t).min()) < min(min_bary0, 1e-7) - 1e-12:
                        break
                    if obj(trial) < cur_f:
                        x = trial
                        improved = True
                    break
                try:
                    corr = q_mat @ solve_triangular(r_mat.T, res, lower=True, check_finite=False)
                except LinAlgError:
                    break
                trial = trial - corr
            if improved:
                break
        if not improved:
            break
    print(f"orbit L2 sum(w^2): {f0:.10g} -> {obj(x):.10g}", flush=True)
    return extract_orbits(prob, x)


def validate_and_expand(group, prob, x, degree):
    """Expand orbit vector x to full rule and verify PI + monomial error."""
    w = x[prob.off[-1]:]
    if w.min() <= 0:
        return None
    pts, wts = se.expand(group, prob, x)
    b = se.bary(pts)
    if float(b.min()) < -1e-14:
        return None
    b = np.maximum(b, 0.0)
    b /= b.sum(axis=1, keepdims=True)
    pts = b[:, 1:]
    err = po_elim.monomial_error("tetra", degree, pts, wts)
    if err >= 1e-11:
        return None
    return pts, wts, float(err), float(b.min()), float(wts.min())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("degree", type=int)
    parser.add_argument("start", type=Path)
    parser.add_argument("--group", choices=["s4", "a4", "s3", "c3"], default="s4")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--tries", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--seconds", type=float, default=1200.0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--jitter", type=float, default=0.0, help="Log-normal jitter on candidate ranking")
    parser.add_argument("--order", choices=["yield", "linear", "demote-first", "random"], default="yield")
    parser.add_argument("--unscaled", action="store_true", help="Disable column scaling in LM")
    parser.add_argument("--no-stag", action="store_true", help="Disable stagnation stop")
    parser.add_argument("--no-lin-init", action="store_true", help="Disable linearized nullspace start")
    parser.add_argument("--l2", action="store_true", help="Run orbit-space L2 minimisation at end")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = perf_counter()
    if args.start.suffix == ".npy":
        arr = np.load(args.start)
        nodes = arr[:, :3]
        weights = arr[:, 3]
    else:
        rule = json.loads(args.start.read_text())
        nodes = np.asarray(rule["nodes"], dtype=float)
        weights = np.asarray(rule["weights"], dtype=float)

    group = se.make_group(se.GENS[args.group])
    _, supers = super_orbit_types(group)
    basis = AnalyticTetraBasis(args.degree)
    blocks, inv_dim = build_invariant_blocks(group, basis, args.seed)
    prob = FastOrbitProblem(basis, blocks, inv_dim)
    orbits = se.decompose(group, nodes, weights)
    x = prob.setup(orbits)
    start_res = float(np.linalg.norm(prob.res(x), np.inf))
    print(f"start: N={len(weights)} orbits={len(orbits)} eq={prob.M} unk={len(x)} dof={len(x) - prob.M} "
          f"res={start_res:.3g} setup={perf_counter() - t0:.2f}s", flush=True)

    rng = np.random.default_rng(args.seed)
    steps_log = []
    best_orbits = orbits

    for step_idx in range(1, args.max_steps + 1):
        if perf_counter() - t0 >= args.seconds:
            print(f"time budget {args.seconds:.0f}s reached at step {step_idx}", flush=True)
            break
        x = prob.setup(orbits)
        dof = len(x) - prob.M
        if dof <= 0:
            print("no remaining degrees of freedom", flush=True)
            break
        moves = rank_candidates(group, prob, orbits, x, supers, args.order, rng, jitter=args.jitter)
        if not moves:
            print("no feasible moves left", flush=True)
            break
        top_preview = [(m[1], m[2], m[3], m[4], round(m[5], 3)) for m in moves[:6]]
        print(f"step {step_idx}: N={int(prob.size.sum())} dof={dof} top={top_preview}", flush=True)

        accepted = False
        for attempt, (key, kind, idx, nodes_saved, dof_cost, raw_score, extra) in enumerate(moves[:args.tries], 1):
            if perf_counter() - t0 >= args.seconds:
                break
            prob.setup(orbits)
            if kind == "del":
                trial = build_delete_trial(
                    group, prob, orbits, x, idx, extra, rng, args.noise,
                    use_lin_init=not args.no_lin_init,
                )
            else:
                t_part, t_ob, trial_x = extra
                trial = build_demote_trial(
                    prob, orbits, x, idx, t_part, t_ob, trial_x, rng, args.noise,
                    use_lin_init=not args.no_lin_init,
                )
            xt = prob.setup(trial)
            t_solve = perf_counter()
            xs, cost = solve_lm_scaled(
                prob, xt, args.iterations,
                scaled=not args.unscaled,
                stag_win=0 if args.no_stag else 12,
            )
            if 1e-24 <= cost < 1e-10:
                xs, cost = solve_lm_scaled(
                    prob, xs, args.iterations,
                    scaled=not args.unscaled,
                    stag_win=0,
                )
            dt = perf_counter() - t_solve
            print(f"  try {attempt} {kind} orb={idx} -{nodes_saved}n (-{dof_cost}dof) "
                  f"sc={raw_score:.3f} cost={cost:.4g} dt={dt:.2f}s", flush=True)
            if cost >= 1e-24:
                continue
            ws = xs[prob.off[-1]:]
            if ws.min() <= 1e-14 * ws.max():
                live = [t for t in range(prob.n) if ws[t] > 1e-14 * ws.max()]
                pruned = [(prob.ob[t], xs[prob.off[t]:prob.off[t + 1]].copy(), float(ws[t])) for t in live]
                xt2 = prob.setup(pruned)
                xs, cost = solve_lm_scaled(prob, xt2, 40, scaled=not args.unscaled, stag_win=0)
                if cost >= 1e-24:
                    continue
            valid = validate_and_expand(group, prob, xs, args.degree)
            if valid is None:
                continue
            pts, wts, err, min_bary, min_w = valid
            orbits = extract_orbits(prob, xs)
            best_orbits = orbits
            steps_log.append({
                "step": step_idx,
                "attempt": attempt,
                "kind": kind,
                "orbit": int(idx),
                "nodes_saved": int(nodes_saved),
                "nodes_after": int(len(wts)),
                "cost": float(cost),
                "monomial_error": err,
                "min_bary": min_bary,
                "min_w": min_w,
                "elapsed": round(perf_counter() - t0, 2),
            })
            stem = args.out / f"tetra-d{args.degree}-n{len(wts)}-s{args.seed}"
            np.save(stem.with_suffix(".npy"), np.column_stack((pts, wts)))
            info = {
                "seed": args.seed,
                "group": args.group,
                "degree": args.degree,
                "nodes": int(len(wts)),
                "start": str(args.start),
                "command": shlex.join(["python", *sys.argv]),
                "monomial_error": err,
                "min_bary": min_bary,
                "min_w": min_w,
                "steps": steps_log,
            }
            stem.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
            print(f"solution found: N={len(wts)} (saved {nodes_saved} via {kind}) "
                  f"err={err:.3g} min_w={min_w:.3g} min_bary={min_bary:.3g} "
                  f"elapsed={perf_counter() - t0:.1f}s", flush=True)
            accepted = True
            break
        if not accepted:
            print(f"stalled at step {step_idx} with N={int(sum(o.size for o, _, _ in orbits))}", flush=True)
            break

    if args.l2 and steps_log:
        prob.setup(best_orbits)
        best_orbits = orbit_l2_minimize(prob, best_orbits)
        xs = prob.setup(best_orbits)
        valid = validate_and_expand(group, prob, xs, args.degree)
        if valid is not None:
            pts, wts, err, min_bary, min_w = valid
            stem = args.out / f"tetra-d{args.degree}-n{len(wts)}-s{args.seed}-l2"
            np.save(stem.with_suffix(".npy"), np.column_stack((pts, wts)))
            info = {
                "seed": args.seed,
                "group": args.group,
                "degree": args.degree,
                "nodes": int(len(wts)),
                "start": str(args.start),
                "command": shlex.join(["python", *sys.argv]),
                "monomial_error": err,
                "min_bary": min_bary,
                "min_w": min_w,
                "l2_minimized": True,
                "steps": steps_log,
            }
            stem.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
            print(f"l2 saved: {stem}.npy err={err:.3g} min_w={min_w:.3g} min_bary={min_bary:.3g}", flush=True)


if __name__ == "__main__":
    main()

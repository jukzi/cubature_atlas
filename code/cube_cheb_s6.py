#!/usr/bin/env python3
"""Deterministic S6-symmetric cubature search on the cube [-1, 1]^3 for odd degrees.

Constructs starting nodes in Chebyshev equilibrium angle coordinates [-pi/2, pi/2]^3
by taking a body-centred cubic (BCC) lattice (Z^3 U (Z + 1/2)^3 with a centre node
for odd N, or the centrally symmetric shifted BCC lattice (Z + 1/4)^3 U (Z + 3/4)^3
without a centre node for even N). The dual of BCC is FCC, whose shortest nonzero
vectors have L1 norm 2 and avoid axial aliasing inside the L1 octahedron
|a| + |b| + |c| <= n.

Rotating the BCC lattice around the (1, 1, 1) body diagonal by a small chiral angle
breaks O_h -> S6 = C3i = <-R_{(1,1,1), 2*pi/3}> (order 6), the unique maximal
mirror-free subgroup of O_h containing central inversion -I. Every odd-degree moment
vanishes identically under -I, while cyclic coordinate symmetry cuts the even-degree
system by a factor of 3 without the O(n^2) Burnside penalty of mirror planes.
Orbits (generic 6-orbits, diagonal 2-orbits along (1, 1, 1), and optional centre)
are ordered by (L_infinity, L_2), scaled to boundary offset r0_fac * (2*pi/n),
mapped back via x = sin(u), and assigned equilibrium weights proportional to
sqrt((1 - x^2)(1 - y^2)(1 - z^2)).

Solves the exact S6-invariant Legendre moment system with Marquardt-scaled
Levenberg-Marquardt using Cholesky factorization, residual-only trial evaluations,
and exact analytic geodesic acceleration (Transtrum & Sethna 2012, arXiv:1201.5885)
with the exact 3-term differentiated Legendre recurrence for P_k''(x) (free of
boundary cancellation at |x| -> 1). Supports PI-by-construction coordinates
(--coords sin: x = sin(s), w = v^2) and a fold-free two-phase mode (--coords plain:
plain (x, y, z, v^2) geodesic LM followed by a diagonal boundary hinge pull inside
[-1, 1]^3) that removes the dx/ds = 0 boundary fold at high degrees.

Usage:
    python code/cube_cheb_s6.py --degree 17 --nodes 262 --angle 8 --r0 0.3 --out rule.json
    python code/cube_cheb_s6.py --degree 43 --nodes 3685 --angle 6 --r0 0.25 --pnorm 12 --lam0 0.5 --geo-tau 0.05 --coords plain --out rule.json
"""
import argparse
import json
import math
import os
import shlex
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from scipy.linalg import cho_factor
from scipy.linalg import cho_solve


def legendre_values(n, x):
    """Evaluate orthonormal Legendre polynomials P_k(x) on [-1, 1] (residual-only)."""
    count = x.size
    P = np.zeros((n + 1, count))
    P[0] = 1.0
    if n >= 1:
        P[1] = x
    for k in range(2, n + 1):
        c1 = 2 * k - 1
        c2 = k - 1
        P[k] = (c1 * x * P[k - 1] - c2 * P[k - 2]) / k
    sc = np.sqrt((2 * np.arange(n + 1) + 1) / 2.0)[:, None]
    return P * sc


def legendre_eval(n, x):
    """Evaluate orthonormal Legendre polynomials P_k(x) and derivatives on [-1, 1]."""
    count = x.size
    P = np.zeros((n + 1, count))
    dP = np.zeros((n + 1, count))
    P[0] = 1.0
    if n >= 1:
        P[1] = x
        dP[1] = 1.0
    for k in range(2, n + 1):
        c1 = 2 * k - 1
        c2 = k - 1
        P[k] = (c1 * x * P[k - 1] - c2 * P[k - 2]) / k
        dP[k] = (c1 * (P[k - 1] + x * dP[k - 1]) - c2 * dP[k - 2]) / k
    sc = np.sqrt((2 * np.arange(n + 1) + 1) / 2.0)[:, None]
    return P * sc, dP * sc


def legendre_eval2(n, x):
    """Evaluate orthonormal P_k(x), P_k'(x), and exact P_k''(x) on [-1, 1]."""
    count = x.size
    P = np.zeros((n + 1, count))
    dP = np.zeros((n + 1, count))
    ddP = np.zeros((n + 1, count))
    P[0] = 1.0
    if n >= 1:
        P[1] = x
        dP[1] = 1.0
    for k in range(2, n + 1):
        c1 = 2 * k - 1
        c2 = k - 1
        P[k] = (c1 * x * P[k - 1] - c2 * P[k - 2]) / k
        dP[k] = (c1 * (P[k - 1] + x * dP[k - 1]) - c2 * dP[k - 2]) / k
        ddP[k] = (c1 * (2.0 * dP[k - 1] + x * ddP[k - 1]) - c2 * ddP[k - 2]) / k
    sc = np.sqrt((2 * np.arange(n + 1) + 1) / 2.0)[:, None]
    return P * sc, dP * sc, ddP * sc


class CubeS6Basis:
    """Orthonormal S6-invariant polynomial basis on [-1, 1]^3 up to odd degree n."""

    def __init__(self, n):
        self.n = n
        self.triples = []
        for a in range(n + 1):
            for b in range(n + 1 - a):
                for c in range(n + 1 - a - b):
                    if (a + b + c) % 2 != 0:
                        continue
                    cyc_min = min((a, b, c), (b, c, a), (c, a, b))
                    if (a, b, c) == cyc_min:
                        sc = 1.0 / 3.0 if (a == b == c) else 1.0 / math.sqrt(3.0)
                        self.triples.append((a, b, c, sc))
        self.M = len(self.triples)
        self.a_idx = np.array([t[0] for t in self.triples])
        self.b_idx = np.array([t[1] for t in self.triples])
        self.c_idx = np.array([t[2] for t in self.triples])
        self.sc = np.array([t[3] for t in self.triples])[:, None]
        self.rhs = np.zeros(self.M)
        self.rhs[0] = 2.0 * math.sqrt(2.0)

    def eval_values(self, x, y, z):
        """Return S6-invariant basis values V for generic orbit representatives."""
        Px = legendre_values(self.n, x)
        Py = legendre_values(self.n, y)
        Pz = legendre_values(self.n, z)
        a = self.a_idx
        b = self.b_idx
        c = self.c_idx
        return self.sc * (
            Px[a] * Py[b] * Pz[c]
            + Px[b] * Py[c] * Pz[a]
            + Px[c] * Py[a] * Pz[b]
        )

    def eval_diag_values(self, t):
        """Return S6-invariant basis values V for diagonal points (t, t, t)."""
        Pt = legendre_values(self.n, t)
        a = self.a_idx
        b = self.b_idx
        c = self.c_idx
        return (3.0 * self.sc) * (Pt[a] * Pt[b] * Pt[c])

    def eval_and_grad(self, x, y, z):
        """Return V and partial derivatives Vx, Vy, Vz for generic orbits."""
        Px, dPx = legendre_eval(self.n, x)
        Py, dPy = legendre_eval(self.n, y)
        Pz, dPz = legendre_eval(self.n, z)
        a = self.a_idx
        b = self.b_idx
        c = self.c_idx
        V = self.sc * (
            Px[a] * Py[b] * Pz[c]
            + Px[b] * Py[c] * Pz[a]
            + Px[c] * Py[a] * Pz[b]
        )
        Vx = self.sc * (
            dPx[a] * Py[b] * Pz[c]
            + dPx[b] * Py[c] * Pz[a]
            + dPx[c] * Py[a] * Pz[b]
        )
        Vy = self.sc * (
            Px[a] * dPy[b] * Pz[c]
            + Px[b] * dPy[c] * Pz[a]
            + Px[c] * dPy[a] * Pz[b]
        )
        Vz = self.sc * (
            Px[a] * Py[b] * dPz[c]
            + Px[b] * Py[c] * dPz[a]
            + Px[c] * Py[a] * dPz[b]
        )
        return V, Vx, Vy, Vz

    def eval_jet(self, x, y, z):
        """Return V, Vx, Vy, Vz and 1D Legendre jets for exact 2nd derivatives."""
        Px, dPx, ddPx = legendre_eval2(self.n, x)
        Py, dPy, ddPy = legendre_eval2(self.n, y)
        Pz, dPz, ddPz = legendre_eval2(self.n, z)
        a = self.a_idx
        b = self.b_idx
        c = self.c_idx
        V = self.sc * (
            Px[a] * Py[b] * Pz[c]
            + Px[b] * Py[c] * Pz[a]
            + Px[c] * Py[a] * Pz[b]
        )
        Vx = self.sc * (
            dPx[a] * Py[b] * Pz[c]
            + dPx[b] * Py[c] * Pz[a]
            + dPx[c] * Py[a] * Pz[b]
        )
        Vy = self.sc * (
            Px[a] * dPy[b] * Pz[c]
            + Px[b] * dPy[c] * Pz[a]
            + Px[c] * dPy[a] * Pz[b]
        )
        Vz = self.sc * (
            Px[a] * Py[b] * dPz[c]
            + Px[b] * Py[c] * dPz[a]
            + Px[c] * Py[a] * dPz[b]
        )
        return V, Vx, Vy, Vz, Px, dPx, ddPx, Py, dPy, ddPy, Pz, dPz, ddPz

    def eval_diag_grad(self, t):
        """Return V and dV/dt for diagonal points (t, t, t)."""
        Pt, dPt = legendre_eval(self.n, t)
        a = self.a_idx
        b = self.b_idx
        c = self.c_idx
        pa = Pt[a]
        pb = Pt[b]
        pc = Pt[c]
        da = dPt[a]
        db = dPt[b]
        dc = dPt[c]
        s3 = 3.0 * self.sc
        Vd = s3 * (pa * pb * pc)
        Vdt = s3 * (da * pb * pc + pa * db * pc + pa * pb * dc)
        return Vd, Vdt

    def eval_diag_jet(self, t):
        """Return V, dV/dt, and exact d^2V/dt^2 for diagonal points (t, t, t)."""
        Pt, dPt, ddPt = legendre_eval2(self.n, t)
        a = self.a_idx
        b = self.b_idx
        c = self.c_idx
        pa = Pt[a]
        pb = Pt[b]
        pc = Pt[c]
        da = dPt[a]
        db = dPt[b]
        dc = dPt[c]
        fa = ddPt[a]
        fb = ddPt[b]
        fc = ddPt[c]
        s3 = 3.0 * self.sc
        Vd = s3 * (pa * pb * pc)
        Vdt = s3 * (da * pb * pc + pa * db * pc + pa * pb * dc)
        Vdtt = s3 * (
            fa * pb * pc
            + pa * fb * pc
            + pa * pb * fc
            + 2.0 * (da * db * pc + da * pb * dc + pa * db * dc)
        )
        return Vd, Vdt, Vdtt


def rot_111(angle_deg):
    """3x3 rotation matrix around the (1, 1, 1) body diagonal by angle_deg."""
    th = math.radians(angle_deg)
    ca = math.cos(th)
    sa = math.sin(th)
    u = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
    K = np.array(
        [
            [0.0, -u[2], u[1]],
            [u[2], 0.0, -u[0]],
            [-u[1], u[0], 0.0],
        ]
    )
    return ca * np.eye(3) + sa * K + (1.0 - ca) * np.outer(u, u)


def _bcc_sorted_orbits(est_nodes, angle_deg, has_centre, pnorm=0.0):
    """Generate S6-unique BCC orbits sorted by (L_p or L_infinity, L_2) after twist."""
    R = rot_111(angle_deg)
    rad = int(math.ceil(((0.5 * est_nodes) ** (1.0 / 3.0)) * 1.25)) + 3
    g = np.arange(-rad, rad + 1, dtype=float)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    P = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    offs = 0.0 if has_centre else 0.25
    P = np.vstack([P + offs, P + 0.5 + offs])
    P = P[np.sum(P * P, axis=1) > 1e-12]
    U = P @ R.T
    is_diag = (np.abs(U[:, 0] - U[:, 1]) < 1e-8) & (np.abs(U[:, 1] - U[:, 2]) < 1e-8)
    Ud = U[is_diag]
    Ud = Ud[Ud[:, 0] > 1e-8]
    Ug = U[~is_diag]
    imgs = np.stack(
        [
            Ug,
            Ug[:, [1, 2, 0]],
            Ug[:, [2, 0, 1]],
            -Ug,
            -Ug[:, [1, 2, 0]],
            -Ug[:, [2, 0, 1]],
        ],
        axis=1,
    )
    scores = imgs[:, :, 0] * 1e8 + imgs[:, :, 1] * 1e4 + imgs[:, :, 2]
    best_idx = np.argmax(scores, axis=1)
    reps = imgs[np.arange(len(Ug)), best_idx]
    _, uniq_i = np.unique(np.round(reps, 6), axis=0, return_index=True)
    reps = reps[uniq_i]
    if len(Ud) > 0:
        _, ud_i = np.unique(np.round(Ud[:, 0], 6), return_index=True)
        Ud = Ud[ud_i]
    orbs = []
    for r in reps:
        if pnorm > 0.0:
            key1 = round(float(np.sum(np.abs(r) ** pnorm) ** (1.0 / pnorm)), 7)
        else:
            key1 = round(float(np.max(np.abs(r))), 7)
        l2 = float(np.linalg.norm(r))
        orbs.append((key1, l2, 0, r))
    s3 = math.sqrt(3.0)
    diag_fac = (3.0 ** (1.0 / pnorm)) if pnorm > 0.0 else 1.0
    for r in Ud:
        tval = float(r[0])
        orbs.append((round(diag_fac * tval, 7), s3 * tval, 1, r))
    orbs.sort(key=lambda item: (item[0], item[1]))
    return orbs


def _orbits_to_nodes(n, gen_u, diag_u, has_centre, r0_fac):
    """Map selected angle-space orbits into [-1, 1]^3 with equilibrium weights."""
    gu = np.array(gen_u, dtype=float)
    du = np.array(diag_u, dtype=float)
    umax = float(np.abs(gu).max()) if len(gu) > 0 else 0.0
    if len(du) > 0:
        umax = max(umax, float(np.abs(du).max()))
    scale = 2.0 * math.pi / n
    hw = 0.5 * math.pi - r0_fac * scale
    gu = gu * (hw / umax)
    du = du * (hw / umax)
    xg = np.sin(gu[:, 0])
    yg = np.sin(gu[:, 1])
    zg = np.sin(gu[:, 2])
    td = np.sin(du)
    wg_raw = np.sqrt(
        np.maximum(1e-12, (1.0 - xg * xg) * (1.0 - yg * yg) * (1.0 - zg * zg))
    )
    wd_raw = np.sqrt(np.maximum(1e-12, (1.0 - td * td) ** 3))
    wc_raw = 1.0 if has_centre else 0.0
    tot = 6.0 * float(wg_raw.sum()) + 2.0 * float(wd_raw.sum()) + wc_raw
    wg = 8.0 * wg_raw / tot
    wd = 8.0 * wd_raw / tot
    wc = 8.0 * wc_raw / tot
    return xg, yg, zg, wg, td, wd, wc


def bcc_cheb_start(n, nodes, angle_deg, r0_fac, pnorm=0.0):
    """Deterministic rotated BCC Chebyshev start with exact node count N."""
    has_centre = (nodes % 2 == 1)
    orbs = _bcc_sorted_orbits(nodes, angle_deg, has_centre, pnorm=pnorm)
    gen_u = []
    diag_u = []
    cur_n = 1 if has_centre else 0
    idx = 0
    while idx < len(orbs) and cur_n < nodes:
        _, _, isd, rep = orbs[idx]
        idx += 1
        if isd == 0:
            gen_u.append(rep)
            cur_n += 6
        else:
            diag_u.append(float(rep[0]))
            cur_n += 2
    diff = cur_n - nodes
    if diff == 2:
        if len(diag_u) > 0:
            diag_u.pop()
            cur_n -= 2
        else:
            gen_u.pop()
            cur_n -= 6
            while idx < len(orbs) and cur_n < nodes:
                _, _, isd, rep = orbs[idx]
                idx += 1
                if isd == 1:
                    diag_u.append(float(rep[0]))
                    cur_n += 2
    elif diff == 4:
        gen_u.pop()
        cur_n -= 6
        while idx < len(orbs) and cur_n < nodes:
            _, _, isd, rep = orbs[idx]
            idx += 1
            if isd == 1:
                diag_u.append(float(rep[0]))
                cur_n += 2
    if cur_n != nodes:
        raise ValueError(f"could not construct exact BCC S6 start for N={nodes}")
    xg, yg, zg, wg, td, wd, wc = _orbits_to_nodes(n, gen_u, diag_u, has_centre, r0_fac)
    return xg, yg, zg, wg, td, wd, has_centre, wc


def bcc_natural_nodes(n, target_unk, angle_deg, has_centre, pnorm=0.0):
    """Return the natural BCC S6 node count reaching at least target_unk unknowns."""
    est_nodes = int(1.6 * target_unk) + 20
    orbs = _bcc_sorted_orbits(est_nodes, angle_deg, has_centre, pnorm=pnorm)
    unk = 1 if has_centre else 0
    cur_n = 1 if has_centre else 0
    for _, _, isd, _ in orbs:
        if unk >= target_unk:
            break
        if isd == 0:
            unk += 4
            cur_n += 6
        else:
            unk += 2
            cur_n += 2
    return cur_n


def solve_s6_lm(
    B,
    x0,
    y0,
    z0,
    wg0,
    td0,
    wd0,
    has_centre,
    wc0,
    steps,
    win,
    lam0,
    geo=True,
    alpha=0.75,
    bold=2,
    geo_tau=0.0,
    lam_down=1.5,
    lam_up=5.0,
    verbose=False,
):
    """Marquardt-scaled LM with PI-by-construction map and exact geodesic acceleration."""
    kg = len(x0)
    kd = len(td0)
    parts = [
        np.arcsin(np.clip(x0, -0.999999, 0.999999)),
        np.arcsin(np.clip(y0, -0.999999, 0.999999)),
        np.arcsin(np.clip(z0, -0.999999, 0.999999)),
        np.sqrt(np.maximum(wg0, 1e-14)),
    ]
    if kd > 0:
        parts.append(np.arcsin(np.clip(td0, -0.999999, 0.999999)))
        parts.append(np.sqrt(np.maximum(wd0, 1e-14)))
    if has_centre:
        parts.append(np.array([math.sqrt(max(wc0, 1e-14))]))
        V0 = B.eval_values(np.array([0.0]), np.array([0.0]), np.array([0.0]))
        v_origin = V0[:, 0]
    else:
        v_origin = None
    q = np.concatenate(parts)
    a_idx = B.a_idx
    b_idx = B.b_idx
    c_idx = B.c_idx
    sc = B.sc

    def eval_res(vec):
        x = np.sin(vec[:kg])
        y = np.sin(vec[kg:2 * kg])
        z = np.sin(vec[2 * kg:3 * kg])
        vg = vec[3 * kg:4 * kg]
        V = B.eval_values(x, y, z)
        r = 6.0 * (V @ (vg * vg)) - B.rhs
        off = 4 * kg
        if kd > 0:
            td = np.sin(vec[off:off + kd])
            vd = vec[off + kd:off + 2 * kd]
            Vd = B.eval_diag_values(td)
            r = r + 2.0 * (Vd @ (vd * vd))
            off += 2 * kd
        if has_centre:
            vc = vec[off]
            r = r + v_origin * (vc * vc)
        return r

    def eval_full(vec, need_jet):
        sx = vec[:kg]
        sy = vec[kg:2 * kg]
        sz = vec[2 * kg:3 * kg]
        vg = vec[3 * kg:4 * kg]
        x = np.sin(sx)
        y = np.sin(sy)
        z = np.sin(sz)
        cx = np.cos(sx)
        cy = np.cos(sy)
        cz = np.cos(sz)
        wg = vg * vg
        if need_jet:
            V, Vx, Vy, Vz, Px, dPx, ddPx, Py, dPy, ddPy, Pz, dPz, ddPz = B.eval_jet(
                x, y, z
            )
        else:
            V, Vx, Vy, Vz = B.eval_and_grad(x, y, z)
            Px = None
            dPx = None
            ddPx = None
            Py = None
            dPy = None
            ddPy = None
            Pz = None
            dPz = None
            ddPz = None
        r = 6.0 * (V @ wg) - B.rhs
        cols = [
            Vx * (6.0 * wg * cx)[None, :],
            Vy * (6.0 * wg * cy)[None, :],
            Vz * (6.0 * wg * cz)[None, :],
            V * (12.0 * vg)[None, :],
        ]
        off = 4 * kg
        dstate = None
        if kd > 0:
            st = vec[off:off + kd]
            vd = vec[off + kd:off + 2 * kd]
            td = np.sin(st)
            ctd = np.cos(st)
            wd = vd * vd
            if need_jet:
                Vd, Vdt, Vdtt = B.eval_diag_jet(td)
                dstate = (td, ctd, vd, wd, Vd, Vdt, Vdtt)
            else:
                Vd, Vdt = B.eval_diag_grad(td)
            r = r + 2.0 * (Vd @ wd)
            cols.append(Vdt * (2.0 * wd * ctd)[None, :])
            cols.append(Vd * (4.0 * vd)[None, :])
            off += 2 * kd
        if has_centre:
            vc = vec[off]
            r = r + v_origin * (vc * vc)
            cols.append((v_origin * (2.0 * vc))[:, None])
        if need_jet:
            state = (
                x, y, z, cx, cy, cz, vg, wg,
                Px, dPx, ddPx, Py, dPy, ddPy, Pz, dPz, ddPz,
                V, Vx, Vy, Vz, dstate,
            )
        else:
            state = None
        return r, np.hstack(cols), state

    def d2r(state, vel):
        """Exact directional second derivative r''(vel) in (sin s, v^2) coordinates."""
        (
            x, y, z, cx, cy, cz, vg, wg,
            Px, dPx, ddPx, Py, dPy, ddPy, Pz, dPz, ddPz,
            V, Vx, Vy, Vz, dstate,
        ) = state
        dsx = vel[:kg]
        dsy = vel[kg:2 * kg]
        dsz = vel[2 * kg:3 * kg]
        dvg = vel[3 * kg:4 * kg]
        dx = cx * dsx
        dy = cy * dsy
        dz = cz * dsz
        cV = 12.0 * (dvg * dvg)
        cVx = 24.0 * vg * dvg * dx - 6.0 * wg * x * (dsx * dsx)
        cVy = 24.0 * vg * dvg * dy - 6.0 * wg * y * (dsy * dsy)
        cVz = 24.0 * vg * dvg * dz - 6.0 * wg * z * (dsz * dsz)
        out = (V @ cV) + (Vx @ cVx) + (Vy @ cVy) + (Vz @ cVz)
        # Contract each 2nd-derivative block immediately to keep peak RAM small.
        blk = sc * (
            ddPx[a_idx] * Py[b_idx] * Pz[c_idx]
            + ddPx[b_idx] * Py[c_idx] * Pz[a_idx]
            + ddPx[c_idx] * Py[a_idx] * Pz[b_idx]
        )
        out = out + (blk @ (6.0 * wg * dx * dx))
        blk = sc * (
            Px[a_idx] * ddPy[b_idx] * Pz[c_idx]
            + Px[b_idx] * ddPy[c_idx] * Pz[a_idx]
            + Px[c_idx] * ddPy[a_idx] * Pz[b_idx]
        )
        out = out + (blk @ (6.0 * wg * dy * dy))
        blk = sc * (
            Px[a_idx] * Py[b_idx] * ddPz[c_idx]
            + Px[b_idx] * Py[c_idx] * ddPz[a_idx]
            + Px[c_idx] * Py[a_idx] * ddPz[b_idx]
        )
        out = out + (blk @ (6.0 * wg * dz * dz))
        blk = sc * (
            dPx[a_idx] * dPy[b_idx] * Pz[c_idx]
            + dPx[b_idx] * dPy[c_idx] * Pz[a_idx]
            + dPx[c_idx] * dPy[a_idx] * Pz[b_idx]
        )
        out = out + (blk @ (12.0 * wg * dx * dy))
        blk = sc * (
            dPx[a_idx] * Py[b_idx] * dPz[c_idx]
            + dPx[b_idx] * Py[c_idx] * dPz[a_idx]
            + dPx[c_idx] * Py[a_idx] * dPz[b_idx]
        )
        out = out + (blk @ (12.0 * wg * dx * dz))
        blk = sc * (
            Px[a_idx] * dPy[b_idx] * dPz[c_idx]
            + Px[b_idx] * dPy[c_idx] * dPz[a_idx]
            + Px[c_idx] * dPy[a_idx] * dPz[b_idx]
        )
        out = out + (blk @ (12.0 * wg * dy * dz))
        off = 4 * kg
        if kd > 0:
            td, ctd, vd, wd, Vd, Vdt, Vdtt = dstate
            dst = vel[off:off + kd]
            dvd = vel[off + kd:off + 2 * kd]
            dt = ctd * dst
            out = out + (Vd @ (4.0 * dvd * dvd))
            out = out + (Vdt @ (8.0 * vd * dvd * dt - 2.0 * wd * td * dst * dst))
            out = out + (Vdtt @ (2.0 * wd * dt * dt))
            off += 2 * kd
        if has_centre:
            dvc = vel[off]
            out = out + v_origin * (2.0 * dvc * dvc)
        return out

    t_start = time.time()
    geo_active = geo and (geo_tau <= 0.0)
    r, J, state = eval_full(q, geo_active)
    r_init = float(np.linalg.norm(r))
    cost = float(r @ r)
    best_cost = cost
    best_q = q.copy()
    lam = lam0
    hist = [best_cost]
    prev_step = None
    cur_bold = bold
    restarted = False
    nvar = len(q)
    used = 0

    for it in range(steps):
        used = it + 1
        if geo and not geo_active and math.sqrt(cost) < geo_tau:
            geo_active = True
            r, J, state = eval_full(q, True)
        g = J.T @ r
        H = J.T @ J
        diag = np.diag(H)
        dmax = float(diag.max()) + 1e-300
        dscale = diag + 1e-12 * dmax
        accepted = False
        for _ in range(25):
            Hd = H.copy()
            Hd.flat[::nvar + 1] += lam * dscale
            try:
                cf = cho_factor(Hd, lower=True, check_finite=False)
                vel = -cho_solve(cf, g, check_finite=False)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            step = vel
            if geo_active and alpha > 0.0:
                rpp = d2r(state, vel)
                accel = -cho_solve(cf, J.T @ rpp, check_finite=False)
                nv = math.sqrt(float(np.dot(dscale, vel * vel)))
                na = math.sqrt(float(np.dot(dscale, accel * accel)))
                if nv > 0.0 and na <= alpha * nv:
                    step = vel + 0.5 * accel
            qn = q + step
            rn = eval_res(qn)
            cn = float(rn @ rn)
            if step is not vel:
                qv = q + vel
                rv = eval_res(qv)
                cv = float(rv @ rv)
                if not np.isfinite(cn) or cv < cn:
                    qn = qv
                    rn = rv
                    cn = cv
                    step = vel
            ok = False
            uphill = False
            if np.isfinite(cn):
                if cn < cost:
                    ok = True
                elif geo_active and cur_bold > 0 and prev_step is not None:
                    sn1 = math.sqrt(float(np.dot(dscale, step * step)))
                    sn0 = math.sqrt(float(np.dot(dscale, prev_step * prev_step)))
                    if sn1 > 0.0 and sn0 > 0.0:
                        cos_b = float(np.dot(dscale, step * prev_step)) / (sn1 * sn0)
                        cos_b = max(-1.0, min(1.0, cos_b))
                        if cos_b > 0.0 and ((1.0 - cos_b) ** cur_bold) * cn <= best_cost:
                            ok = True
                            uphill = True
            if ok:
                q = qn
                r, J, state = eval_full(q, geo_active)
                cost = float(r @ r)
                if cost < best_cost:
                    best_cost = cost
                    best_q = q.copy()
                prev_step = step
                if not uphill:
                    factor = lam_down if geo_active else (1.0 / 0.3)
                    lam = max(lam / factor, 1e-15)
                accepted = True
                break
            lam *= lam_up
        hist.append(best_cost)
        if verbose and (used <= 5 or used % 10 == 0 or math.sqrt(best_cost) < 1e-13):
            dt = time.time() - t_start
            print(
                f"  it={used:4d} res={math.sqrt(best_cost):.3e} cur={math.sqrt(cost):.3e} "
                f"lam={lam:.1e} geo={int(geo_active)} ({dt:.1f}s)",
                flush=True,
            )
        if math.sqrt(best_cost) < 1e-13:
            break
        stagnated = not accepted
        if not stagnated and win > 0 and len(hist) > win:
            if math.sqrt(best_cost) > 0.995 * math.sqrt(hist[-1 - win]):
                stagnated = True
        if stagnated:
            if geo_active and not restarted and cost > 1.01 * best_cost:
                restarted = True
                q = best_q.copy()
                r, J, state = eval_full(q, True)
                cost = best_cost
                lam = min(lam, 1e-4)
                cur_bold = 0
                prev_step = None
                hist = [best_cost]
                continue
            break

    q = best_q
    xg = np.sin(q[:kg])
    yg = np.sin(q[kg:2 * kg])
    zg = np.sin(q[2 * kg:3 * kg])
    wg = q[3 * kg:4 * kg] ** 2
    off = 4 * kg
    if kd > 0:
        td = np.sin(q[off:off + kd])
        wd = q[off + kd:off + 2 * kd] ** 2
        off += 2 * kd
    else:
        td = np.array([], dtype=float)
        wd = np.array([], dtype=float)
    wc = float(q[off] ** 2) if has_centre else 0.0
    return r_init, math.sqrt(best_cost), used, xg, yg, zg, wg, td, wd, wc


def solve_s6_plain(
    B,
    x0,
    y0,
    z0,
    wg0,
    td0,
    wd0,
    has_centre,
    wc0,
    steps,
    win,
    lam0,
    geo=True,
    alpha=0.75,
    bold=2,
    geo_tau=0.05,
    lam_down=1.5,
    lam_up=5.0,
    xmax=1.12,
    pull_b=None,
    pull_wt=1.0,
    verbose=False,
):
    """Fold-free Marquardt-scaled geodesic LM in plain (x, y, z, v^2) coordinates.

    Avoiding x = sin(s) removes the dx/ds = 0 fold singularity at |x| = 1 that traps
    outer-face orbits at high degree. An optional piecewise-linear boundary hinge
    pull_wt * max(0, |x| - pull_b) is folded directly into the gradient and diagonal
    of H = J^T J without extra matrix allocations to pull a PO solution strictly
    inside [-1, 1]^3.
    """
    kg = len(x0)
    kd = len(td0)
    parts = [
        x0.copy(),
        y0.copy(),
        z0.copy(),
        np.sqrt(np.maximum(wg0, 1e-14)),
    ]
    if kd > 0:
        parts.append(td0.copy())
        parts.append(np.sqrt(np.maximum(wd0, 1e-14)))
    if has_centre:
        parts.append(np.array([math.sqrt(max(wc0, 1e-14))]))
        V0 = B.eval_values(np.array([0.0]), np.array([0.0]), np.array([0.0]))
        v_origin = V0[:, 0]
    else:
        v_origin = None
    q = np.concatenate(parts)
    a_idx = B.a_idx
    b_idx = B.b_idx
    c_idx = B.c_idx
    sc = B.sc
    if kd > 0:
        coord_idx = np.concatenate(
            [np.arange(0, 3 * kg), np.arange(4 * kg, 4 * kg + kd)]
        ).astype(int)
    else:
        coord_idx = np.arange(0, 3 * kg, dtype=int)
    nvar = len(q)

    def pen_vec(vec):
        xc = vec[coord_idx]
        exc = np.maximum(0.0, np.abs(xc) - pull_b)
        return pull_wt * exc * np.sign(xc)

    def eval_moment_res(vec):
        if len(coord_idx) > 0 and float(np.abs(vec[coord_idx]).max()) > xmax:
            return None
        x = vec[:kg]
        y = vec[kg:2 * kg]
        z = vec[2 * kg:3 * kg]
        vg = vec[3 * kg:4 * kg]
        V = B.eval_values(x, y, z)
        r = 6.0 * (V @ (vg * vg)) - B.rhs
        off = 4 * kg
        if kd > 0:
            td = vec[off:off + kd]
            vd = vec[off + kd:off + 2 * kd]
            Vd = B.eval_diag_values(td)
            r = r + 2.0 * (Vd @ (vd * vd))
            off += 2 * kd
        if has_centre:
            vc = vec[off]
            r = r + v_origin * (vc * vc)
        return r

    def eval_cost(vec):
        rm = eval_moment_res(vec)
        if rm is None:
            return math.inf
        val = float(rm @ rm)
        if pull_b is not None:
            pv = pen_vec(vec)
            val += float(pv @ pv)
        return val

    def eval_full(vec, need_jet):
        x = vec[:kg]
        y = vec[kg:2 * kg]
        z = vec[2 * kg:3 * kg]
        vg = vec[3 * kg:4 * kg]
        wg = vg * vg
        if need_jet:
            V, Vx, Vy, Vz, Px, dPx, ddPx, Py, dPy, ddPy, Pz, dPz, ddPz = B.eval_jet(
                x, y, z
            )
        else:
            V, Vx, Vy, Vz = B.eval_and_grad(x, y, z)
            Px = None
            dPx = None
            ddPx = None
            Py = None
            dPy = None
            ddPy = None
            Pz = None
            dPz = None
            ddPz = None
        r = 6.0 * (V @ wg) - B.rhs
        cols = [
            Vx * (6.0 * wg)[None, :],
            Vy * (6.0 * wg)[None, :],
            Vz * (6.0 * wg)[None, :],
            V * (12.0 * vg)[None, :],
        ]
        off = 4 * kg
        dstate = None
        if kd > 0:
            td = vec[off:off + kd]
            vd = vec[off + kd:off + 2 * kd]
            wd = vd * vd
            if need_jet:
                Vd, Vdt, Vdtt = B.eval_diag_jet(td)
                dstate = (vd, wd, Vd, Vdt, Vdtt)
            else:
                Vd, Vdt = B.eval_diag_grad(td)
            r = r + 2.0 * (Vd @ wd)
            cols.append(Vdt * (2.0 * wd)[None, :])
            cols.append(Vd * (4.0 * vd)[None, :])
            off += 2 * kd
        if has_centre:
            vc = vec[off]
            r = r + v_origin * (vc * vc)
            cols.append((v_origin * (2.0 * vc))[:, None])
        if need_jet:
            state = (
                vg, wg,
                Px, dPx, ddPx, Py, dPy, ddPy, Pz, dPz, ddPz,
                V, Vx, Vy, Vz, dstate,
            )
        else:
            state = None
        return r, np.hstack(cols), state

    def d2r(state, vel):
        """Exact directional second derivative r''(vel) in plain (x, y, z, v^2)."""
        (
            vg, wg,
            Px, dPx, ddPx, Py, dPy, ddPy, Pz, dPz, ddPz,
            V, Vx, Vy, Vz, dstate,
        ) = state
        dx = vel[:kg]
        dy = vel[kg:2 * kg]
        dz = vel[2 * kg:3 * kg]
        dvg = vel[3 * kg:4 * kg]
        out = (
            (V @ (12.0 * dvg * dvg))
            + (Vx @ (24.0 * vg * dvg * dx))
            + (Vy @ (24.0 * vg * dvg * dy))
            + (Vz @ (24.0 * vg * dvg * dz))
        )
        blk = sc * (
            ddPx[a_idx] * Py[b_idx] * Pz[c_idx]
            + ddPx[b_idx] * Py[c_idx] * Pz[a_idx]
            + ddPx[c_idx] * Py[a_idx] * Pz[b_idx]
        )
        out = out + (blk @ (6.0 * wg * dx * dx))
        blk = sc * (
            Px[a_idx] * ddPy[b_idx] * Pz[c_idx]
            + Px[b_idx] * ddPy[c_idx] * Pz[a_idx]
            + Px[c_idx] * ddPy[a_idx] * Pz[b_idx]
        )
        out = out + (blk @ (6.0 * wg * dy * dy))
        blk = sc * (
            Px[a_idx] * Py[b_idx] * ddPz[c_idx]
            + Px[b_idx] * Py[c_idx] * ddPz[a_idx]
            + Px[c_idx] * Py[a_idx] * ddPz[b_idx]
        )
        out = out + (blk @ (6.0 * wg * dz * dz))
        blk = sc * (
            dPx[a_idx] * dPy[b_idx] * Pz[c_idx]
            + dPx[b_idx] * dPy[c_idx] * Pz[a_idx]
            + dPx[c_idx] * dPy[a_idx] * Pz[b_idx]
        )
        out = out + (blk @ (12.0 * wg * dx * dy))
        blk = sc * (
            dPx[a_idx] * Py[b_idx] * dPz[c_idx]
            + dPx[b_idx] * Py[c_idx] * dPz[a_idx]
            + dPx[c_idx] * Py[a_idx] * dPz[b_idx]
        )
        out = out + (blk @ (12.0 * wg * dx * dz))
        blk = sc * (
            Px[a_idx] * dPy[b_idx] * dPz[c_idx]
            + Px[b_idx] * dPy[c_idx] * dPz[a_idx]
            + Px[c_idx] * dPy[a_idx] * dPz[b_idx]
        )
        out = out + (blk @ (12.0 * wg * dy * dz))
        off = 4 * kg
        if kd > 0:
            vd, wd, Vd, Vdt, Vdtt = dstate
            dt = vel[off:off + kd]
            dvd = vel[off + kd:off + 2 * kd]
            out = out + (Vd @ (4.0 * dvd * dvd))
            out = out + (Vdt @ (8.0 * vd * dvd * dt))
            out = out + (Vdtt @ (2.0 * wd * dt * dt))
            off += 2 * kd
        if has_centre:
            dvc = vel[off]
            out = out + v_origin * (2.0 * dvc * dvc)
        return out

    t_start = time.time()
    geo_active = geo and (geo_tau <= 0.0)
    rm, Jm, state = eval_full(q, geo_active)
    r_init = float(np.linalg.norm(rm))
    cost = float(rm @ rm)
    if pull_b is not None:
        pv = pen_vec(q)
        cost += float(pv @ pv)
    best_cost = cost
    best_q = q.copy()
    lam = lam0
    hist = [best_cost]
    prev_step = None
    cur_bold = bold
    restarted = False
    used = 0

    for it in range(steps):
        used = it + 1
        if geo and not geo_active and math.sqrt(cost) < geo_tau:
            geo_active = True
            rm, Jm, state = eval_full(q, True)
        g = Jm.T @ rm
        H = Jm.T @ Jm
        if pull_b is not None:
            xc = q[coord_idx]
            active = np.abs(xc) > pull_b
            if np.any(active):
                ai = coord_idx[active]
                exc = (np.abs(xc[active]) - pull_b) * np.sign(xc[active])
                pw2 = pull_wt * pull_wt
                g[ai] += pw2 * exc
                H[ai, ai] += pw2
        diag = np.diag(H)
        dmax = float(diag.max()) + 1e-300
        dscale = diag + 1e-12 * dmax
        accepted = False
        for _ in range(25):
            Hd = H.copy()
            Hd.flat[::nvar + 1] += lam * dscale
            try:
                cf = cho_factor(Hd, lower=True, check_finite=False)
                vel = -cho_solve(cf, g, check_finite=False)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            step = vel
            if geo_active and alpha > 0.0:
                rpp = d2r(state, vel)
                accel = -cho_solve(cf, Jm.T @ rpp, check_finite=False)
                nv = math.sqrt(float(np.dot(dscale, vel * vel)))
                na = math.sqrt(float(np.dot(dscale, accel * accel)))
                if nv > 0.0 and na <= alpha * nv:
                    step = vel + 0.5 * accel
            qn = q + step
            cn = eval_cost(qn)
            if step is not vel:
                qv = q + vel
                cv = eval_cost(qv)
                if not np.isfinite(cn) or cv < cn:
                    qn = qv
                    cn = cv
                    step = vel
            ok = False
            uphill = False
            if np.isfinite(cn):
                if cn < cost:
                    ok = True
                elif geo_active and cur_bold > 0 and prev_step is not None:
                    sn1 = math.sqrt(float(np.dot(dscale, step * step)))
                    sn0 = math.sqrt(float(np.dot(dscale, prev_step * prev_step)))
                    if sn1 > 0.0 and sn0 > 0.0:
                        cos_b = float(np.dot(dscale, step * prev_step)) / (sn1 * sn0)
                        cos_b = max(-1.0, min(1.0, cos_b))
                        if cos_b > 0.0 and ((1.0 - cos_b) ** cur_bold) * cn <= best_cost:
                            ok = True
                            uphill = True
            if ok:
                q = qn
                rm, Jm, state = eval_full(q, geo_active)
                cost = float(rm @ rm)
                if pull_b is not None:
                    pv = pen_vec(q)
                    cost += float(pv @ pv)
                if cost < best_cost:
                    best_cost = cost
                    best_q = q.copy()
                prev_step = step
                if not uphill:
                    factor = lam_down if geo_active else (1.0 / 0.3)
                    lam = max(lam / factor, 1e-15)
                accepted = True
                break
            lam *= lam_up
        hist.append(best_cost)
        if verbose and (used <= 5 or used % 10 == 0 or math.sqrt(best_cost) < 1e-13):
            dt = time.time() - t_start
            mx = float(np.abs(q[coord_idx]).max()) if len(coord_idx) > 0 else 0.0
            print(
                f"  it={used:4d} tot={math.sqrt(best_cost):.3e} mom={float(np.linalg.norm(rm)):.3e} "
                f"bd={1.0 - mx:+.2e} lam={lam:.1e} geo={int(geo_active)} ({dt:.1f}s)",
                flush=True,
            )
        if math.sqrt(best_cost) < 1e-13:
            break
        stagnated = not accepted
        if not stagnated and win > 0 and len(hist) > win:
            if math.sqrt(best_cost) > 0.995 * math.sqrt(hist[-1 - win]):
                stagnated = True
        if stagnated:
            if geo_active and not restarted and cost > 1.01 * best_cost:
                restarted = True
                q = best_q.copy()
                rm, Jm, state = eval_full(q, True)
                cost = best_cost
                lam = min(lam, 1e-4)
                cur_bold = 0
                prev_step = None
                hist = [best_cost]
                continue
            break

    q = best_q
    xg = q[:kg].copy()
    yg = q[kg:2 * kg].copy()
    zg = q[2 * kg:3 * kg].copy()
    wg = q[3 * kg:4 * kg] ** 2
    off = 4 * kg
    if kd > 0:
        td = q[off:off + kd].copy()
        wd = q[off + kd:off + 2 * kd] ** 2
        off += 2 * kd
    else:
        td = np.array([], dtype=float)
        wd = np.array([], dtype=float)
    wc = float(q[off] ** 2) if has_centre else 0.0
    rm_best = eval_moment_res(q)
    mom_res = float(np.linalg.norm(rm_best)) if rm_best is not None else math.inf
    return r_init, mom_res, used, xg, yg, zg, wg, td, wd, wc


def expand_s6(xg, yg, zg, wg, td, wd, has_centre, wc):
    """Expand S6 orbit representatives to full node and weight arrays."""
    nodes = []
    weights = []
    for xi, yi, zi, wi in zip(xg, yg, zg, wg):
        x = float(xi)
        y = float(yi)
        z = float(zi)
        w = float(wi)
        nodes.append([x, y, z])
        nodes.append([y, z, x])
        nodes.append([z, x, y])
        nodes.append([-x, -y, -z])
        nodes.append([-y, -z, -x])
        nodes.append([-z, -x, -y])
        weights.extend([w] * 6)
    for ti, wi in zip(td, wd):
        t = float(ti)
        w = float(wi)
        nodes.append([t, t, t])
        nodes.append([-t, -t, -t])
        weights.extend([w, w])
    if has_centre:
        nodes.append([0.0, 0.0, 0.0])
        weights.append(float(wc))
    return np.array(nodes), np.array(weights)


def monomial_residual(n, nodes, weights):
    """Maximum |sum w x^a y^b z^c - int x^a y^b z^c| / 8 over all a + b + c <= n."""
    x = nodes[:, 0]
    y = nodes[:, 1]
    z = nodes[:, 2]
    count = len(x)
    xp = np.ones((n + 1, count))
    yp = np.ones((n + 1, count))
    zp = np.ones((n + 1, count))
    for k in range(1, n + 1):
        xp[k] = xp[k - 1] * x
        yp[k] = yp[k - 1] * y
        zp[k] = zp[k - 1] * z
    max_err = 0.0
    for a in range(n + 1):
        for b in range(n + 1 - a):
            wxy = weights * xp[a] * yp[b]
            c_max = n - a - b
            vals = zp[:c_max + 1] @ wxy
            for c in range(c_max + 1):
                if a % 2 == 0 and b % 2 == 0 and c % 2 == 0:
                    exact = 8.0 / ((a + 1) * (b + 1) * (c + 1))
                else:
                    exact = 0.0
                err = abs(float(vals[c]) - exact) / 8.0
                if err > max_err:
                    max_err = err
    return max_err


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--degree", type=int, required=True, help="odd polynomial degree n")
    ap.add_argument("--nodes", type=int, default=0, help="exact node count N (0 to infer from --extra-unk)")
    ap.add_argument("--extra-unk", type=int, default=0, help="surplus unknowns above M_S6 when --nodes is omitted")
    ap.add_argument("--centre", action="store_true", help="include centre node when inferring N from --extra-unk")
    ap.add_argument("--no-centre", action="store_true", help="omit centre node when inferring N from --extra-unk")
    ap.add_argument("--angle", type=float, default=8.0, help="chiral (1,1,1) twist angle in degrees")
    ap.add_argument("--r0", type=float, default=0.35, help="boundary offset in units of 2*pi/n")
    ap.add_argument("--pnorm", type=float, default=0.0, help="L_p orbit cutoff norm (0 for L_infinity, e.g. 6..12 to chamfer low-weight corners)")
    ap.add_argument("--steps", type=int, default=500, help="maximum LM iterations")
    ap.add_argument("--win", type=int, default=35, help="stagnation window (0 to disable)")
    ap.add_argument("--lam0", type=float, default=1e-2, help="initial LM damping")
    ap.add_argument("--no-geo", action="store_true", help="disable geodesic acceleration and bold uphill steps")
    ap.add_argument("--alpha", type=float, default=0.75, help="geodesic acceleration ratio cap |a|/|v|")
    ap.add_argument("--bold", type=int, default=2, help="Transtrum uphill boldness exponent (0 to disable)")
    ap.add_argument("--geo-tau", type=float, default=0.0, help="residual threshold to activate geodesic phase (0 for immediate)")
    ap.add_argument("--coords", choices=("sin", "plain"), default="sin", help="coordinate map: sin (x=sin(s) PI by construction) or plain (fold-free x + hinge pull)")
    ap.add_argument("--pull-margin", type=float, default=5e-4, help="target interior margin 1 - max|x| for --coords plain hinge pull")
    ap.add_argument("--pull-weight", type=float, default=1.0, help="hinge penalty weight for --coords plain phase 2")
    ap.add_argument("--pull-steps", type=int, default=120, help="maximum LM iterations for --coords plain phase 2 pull")
    ap.add_argument("--pull-lam0", type=float, default=1e-3, help="initial LM damping for --coords plain phase 2 pull")
    ap.add_argument("--xmax", type=float, default=1.12, help="coordinate escape cap for --coords plain")
    ap.add_argument("--verbose", action="store_true", help="print iteration progress")
    ap.add_argument("--out", required=True, help="output atlas JSON path")
    a = ap.parse_args()
    if a.degree % 2 == 0:
        raise ValueError("degree must be odd for S6-symmetric cube rules")
    B = CubeS6Basis(a.degree)
    if a.nodes > 0:
        target_nodes = a.nodes
    else:
        if a.centre and a.no_centre:
            raise ValueError("cannot pass both --centre and --no-centre")
        if a.centre:
            hc = True
        elif a.no_centre:
            hc = False
        else:
            hc = (a.degree % 4 == 3)
        target_nodes = bcc_natural_nodes(
            a.degree, B.M + a.extra_unk, a.angle, hc, pnorm=a.pnorm
        )
    use_geo = not a.no_geo
    t0 = time.time()
    xg0, yg0, zg0, wg0, td0, wd0, has_centre, wc0 = bcc_cheb_start(
        a.degree, target_nodes, a.angle, a.r0, pnorm=a.pnorm
    )
    unk = 4 * len(xg0) + 2 * len(td0) + (1 if has_centre else 0)
    if a.coords == "sin":
        r0_val, res, used, xg, yg, zg, wg, td, wd, wc = solve_s6_lm(
            B,
            xg0,
            yg0,
            zg0,
            wg0,
            td0,
            wd0,
            has_centre,
            wc0,
            a.steps,
            a.win,
            a.lam0,
            geo=use_geo,
            alpha=a.alpha,
            bold=a.bold,
            geo_tau=a.geo_tau,
            verbose=a.verbose,
        )
    else:
        r0_val, res, used, xg, yg, zg, wg, td, wd, wc = solve_s6_plain(
            B,
            xg0,
            yg0,
            zg0,
            wg0,
            td0,
            wd0,
            has_centre,
            wc0,
            a.steps,
            a.win,
            a.lam0,
            geo=use_geo,
            alpha=a.alpha,
            bold=a.bold,
            geo_tau=a.geo_tau,
            xmax=a.xmax,
            pull_b=None,
            verbose=a.verbose,
        )
        nodes_p1, _ = expand_s6(xg, yg, zg, wg, td, wd, has_centre, wc)
        mx_p1 = float(np.abs(nodes_p1).max())
        if res < 1e-12 and mx_p1 >= 1.0 - a.pull_margin and a.pull_steps > 0:
            _, res, it2, xg, yg, zg, wg, td, wd, wc = solve_s6_plain(
                B,
                xg,
                yg,
                zg,
                wg,
                td,
                wd,
                has_centre,
                wc,
                a.pull_steps,
                a.win,
                a.pull_lam0,
                geo=use_geo,
                alpha=a.alpha,
                bold=a.bold,
                geo_tau=0.0,
                xmax=a.xmax,
                pull_b=1.0 - a.pull_margin,
                pull_wt=a.pull_weight,
                verbose=a.verbose,
            )
            used += it2
            nodes_p2, _ = expand_s6(xg, yg, zg, wg, td, wd, has_centre, wc)
            mx_p2 = float(np.abs(nodes_p2).max())
            if res >= 1e-12 and mx_p2 < 1.0 - 1e-7:
                # Phase 2 pulled every node inside [-1, 1]^3; try fold-free pure-moment
                # polish capped inside the cube before falling back to sin(s).
                _, res3, it3, x3, y3, z3, w3, t3, wd3, wc3 = solve_s6_plain(
                    B,
                    xg,
                    yg,
                    zg,
                    wg,
                    td,
                    wd,
                    has_centre,
                    wc,
                    a.steps,
                    a.win,
                    a.pull_lam0,
                    geo=use_geo,
                    alpha=a.alpha,
                    bold=a.bold,
                    geo_tau=0.0,
                    xmax=1.0 - 1e-7,
                    pull_b=None,
                    verbose=a.verbose,
                )
                used += it3
                nodes_p3, _ = expand_s6(x3, y3, z3, w3, t3, wd3, has_centre, wc3)
                if res3 < 1e-12 and float(np.abs(nodes_p3).max()) < 1.0 - 1e-7:
                    res = res3
                    xg = x3
                    yg = y3
                    zg = z3
                    wg = w3
                    td = t3
                    wd = wd3
                    wc = wc3
                else:
                    _, res, it4, xg, yg, zg, wg, td, wd, wc = solve_s6_lm(
                        B,
                        xg,
                        yg,
                        zg,
                        wg,
                        td,
                        wd,
                        has_centre,
                        wc,
                        a.steps,
                        a.win,
                        a.pull_lam0,
                        geo=use_geo,
                        alpha=a.alpha,
                        bold=a.bold,
                        geo_tau=0.0,
                        verbose=a.verbose,
                    )
                    used += it4
    nodes, weights = expand_s6(xg, yg, zg, wg, td, wd, has_centre, wc)
    merr = monomial_residual(a.degree, nodes, weights)
    min_w = float(weights.min())
    bdist = float(1.0 - np.abs(nodes).max())
    elapsed = time.time() - t0
    print(
        f"cube deg={a.degree} N={target_nodes} (gen={len(xg)} diag={len(td)} c={int(has_centre)} "
        f"unk={unk}/{B.M}) ang={a.angle} r0={a.r0} p={a.pnorm} coords={a.coords}: "
        f"init={r0_val:.3e} -> res={res:.2e} monom={merr:.2e} min_w={min_w:.3e} "
        f"bdist={bdist:.3e} it={used} ({elapsed:.1f}s)",
        flush=True,
    )
    if res >= 1e-12 or merr >= 1e-11 or min_w <= 0.0 or bdist <= 0.0:
        print("no solution found", flush=True)
        sys.exit(1)
    rule_id = f"CA-C3-{a.degree:02d}-{target_nodes}-PI-MPC"
    args = [
        "--degree", str(a.degree),
        "--nodes", str(target_nodes),
        "--angle", f"{a.angle:.17g}",
        "--r0", f"{a.r0:.17g}",
        "--steps", str(a.steps),
        "--win", str(a.win),
    ]
    if a.pnorm > 0.0:
        args.extend(["--pnorm", f"{a.pnorm:.17g}"])
    if a.lam0 != 1e-2:
        args.extend(["--lam0", f"{a.lam0:.17g}"])
    if a.no_geo:
        args.append("--no-geo")
    else:
        if a.alpha != 0.75:
            args.extend(["--alpha", f"{a.alpha:.17g}"])
        if a.bold != 2:
            args.extend(["--bold", str(a.bold)])
        if a.geo_tau != 0.0:
            args.extend(["--geo-tau", f"{a.geo_tau:.17g}"])
    if a.coords != "sin":
        args.extend(["--coords", a.coords])
        if a.pull_margin != 5e-4:
            args.extend(["--pull-margin", f"{a.pull_margin:.17g}"])
        if a.pull_weight != 1.0:
            args.extend(["--pull-weight", f"{a.pull_weight:.17g}"])
        if a.pull_steps != 120:
            args.extend(["--pull-steps", str(a.pull_steps)])
        if a.pull_lam0 != 1e-3:
            args.extend(["--pull-lam0", f"{a.pull_lam0:.17g}"])
        if a.xmax != 1.12:
            args.extend(["--xmax", f"{a.xmax:.17g}"])
    args.extend(["--out", a.out])
    cmd = "python code/cube_cheb_s6.py " + shlex.join(args)
    solver_desc = (
        "geodesic-accelerated Marquardt-scaled Levenberg-Marquardt"
        if use_geo
        else "Marquardt-scaled Levenberg-Marquardt"
    )
    if a.coords == "plain":
        solver_desc += " in fold-free (x, y, z, v^2) coordinates with boundary hinge pull"
    norm_desc = f", L_{a.pnorm:.17g} cutoff" if a.pnorm > 0.0 else ""
    record = {
        "id": rule_id,
        "family": "minimal_pointcount_for_degree_n_PI",
        "domain": "cube",
        "dimension": 3,
        "degree": a.degree,
        "points": target_nodes,
        "quality": "PI",
        "origin": "found",
        "method": (
            f"Deterministic S6-symmetric rotated BCC Chebyshev equilibrium lattice start "
            f"(angle {a.angle:.17g} deg around (1,1,1), boundary offset {a.r0:.17g} * 2*pi/n"
            f"{norm_desc}, equilibrium weights) solved directly with {solver_desc} (no randomness)"
        ),
        "author": "Antigravity (AI-assisted)",
        "reproduce": {
            "script": "code/cube_cheb_s6.py",
            "seed": 0,
            "command": cmd,
        },
        "status": "proven",
        "residual_reported": merr,
        "nodes": [[float(p[0]), float(p[1]), float(p[2])] for p in nodes],
        "weights": [float(wt) for wt in weights],
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
        fh.write("\n")
    print(f"solution found: wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()

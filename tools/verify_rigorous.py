"""Rigorous check of the cubature rules of the atlas with interval arithmetic.

Part 1, the stored numbers: every rule file is read with its decimal numbers taken exactly.
  - Node positions and weight signs are checked exactly in rational arithmetic.
  - The moment residual of the stored rule is enclosed rigorously for every basis polynomial up to the degree.
Part 2, existence of an exact rule: a Krawczyk test proves that the moment equations have an exact solution
  in a small box around the stored rule, and that every rule in that box has the stored quality
  (all nodes inside and / or all weights positive).
  - Box domains whose rule is symmetric under the full symmetry group (signs and permutations of the
    coordinates) use the symmetry-reduced equations; the proven rule is then symmetric as well.
  - All other rules use the full system in all node coordinates and weights.
  - If there are more unknowns than equations, the unknowns are restricted to an affine subspace through
    the stored rule (spanned by the rows of the Jacobian), which gives a square system.

Basis: on boxes products of Legendre polynomials P_k(x_c); on the simplex the orthogonal Dubiner (PKD) basis
for the existence proof (the Legendre products in t = 2x - 1 are far too ill-conditioned there) and the
Legendre products for the residual of the stored numbers.
Exact integrals: 2^d resp. the volume for the constant and 0 otherwise; Legendre products on the simplex by exact
rational arithmetic.

Rounding model: IEEE 754 double precision, round to nearest.
  - Elementwise operations are widened by one unit in the last place (nextafter) in the safe direction.
  - A float matrix product of inner dimension k is enclosed by fl(A B) +- gamma(k) |A| |B| (plus an underflow
    term); this bound holds for every order of summation, with or without FMA, so also for blocked and
    multithreaded BLAS.
  - Basis values at the box centres are computed exactly with fractions and then enclosed by floats.
  - Legendre products over a box of radius rho around t0: exact derivative bound max |P_k^(m)| on |t| <= R,
    which is P_k^(m)(R) for R >= 1.
  - PKD basis over a box: second-order Taylor form with exact derivatives at the centre; the remainder is
    bounded by interval enclosures of the third derivatives over the box.
  - Sums over the nodes use math.fsum (correctly rounded), widened by one unit in the last place.

Usage:
  python tools/verify_rigorous.py                     all rules
  python tools/verify_rigorous.py --changed origin/main   only rule files changed against a git revision
  python tools/verify_rigorous.py --report reports/interval_proofs.json
Exit code 1 if a checked rule fails part 1, or (with --strict) if part 2 cannot prove it.
"""

import argparse
import glob
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from fractions import Fraction
from itertools import permutations
from itertools import product

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIM = {"interval": 1, "triangle": 2, "square": 2, "tetra": 3, "cube": 3}
VOL = {"interval": Fraction(2), "triangle": Fraction(1, 2), "square": Fraction(4), "tetra": Fraction(1, 6), "cube": Fraction(8)}
BOX = {"interval", "square", "cube"}
RES_TOL = 1e-11
U = 2.0 ** -53
TINY = 2.0 ** -1000
# Points per block in the basis enclosure (memory: the temporaries grow with block size x number of basis functions).
POINT_BLOCK = int(os.environ.get("VR_POINT_BLOCK", "128"))
INF = np.inf


# ---------------------------------------------------------------- rounding helpers


def dn(a):
    return np.nextafter(a, -INF)


def up(a):
    return np.nextafter(a, INF)


def gamma(k):
    """Upper bound of k u / (1 - k u), slightly enlarged."""
    return 1.01 * k * U / (1.0 - k * U)


def frac_lo(q):
    """Largest float that is <= the fraction q."""
    f = float(q)
    if Fraction(f) > q:
        f = float(dn(f))
    return f


def frac_hi(q):
    """Smallest float that is >= the fraction q."""
    f = float(q)
    if Fraction(f) < q:
        f = float(up(f))
    return f


def ab_bound(A, B, k):
    """Rigorous upper bound of |A| |B| (entrywise abs, float matrices) including its own rounding."""
    S = np.abs(A) @ np.abs(B)
    return up(S * (1.0 + gamma(k + 2)) + k * TINY)


def mm(A, B):
    """Enclosure of the exact product of float matrices: midpoint and radius."""
    k = A.shape[1] if A.ndim > 1 else A.shape[0]
    M = A @ B
    R = up(gamma(k) * ab_bound(A, B, k) + k * TINY)
    return M, R


def mm_iv(Am, Ar, B):
    """Enclosure of (Am +- Ar) B for a float matrix B: midpoint and radius."""
    k = Am.shape[1] if Am.ndim > 1 else Am.shape[0]
    M, R = mm(Am, B)
    R2 = up((np.abs(Ar) @ np.abs(B)) * (1.0 + gamma(k + 2)) + k * TINY)
    return M, up(R + R2)


def iv_mul(alo, ahi, blo, bhi):
    p1 = alo * blo
    p2 = alo * bhi
    p3 = ahi * blo
    p4 = ahi * bhi
    lo = np.minimum(np.minimum(p1, p2), np.minimum(p3, p4))
    hi = np.maximum(np.maximum(p1, p2), np.maximum(p3, p4))
    return dn(lo), up(hi)


def to_midrad(lo, hi):
    m = 0.5 * lo + 0.5 * hi
    r = up(np.maximum(up(hi - m), up(m - lo)))
    return m, r


# ---------------------------------------------------------------- Legendre values


_pt_cache = {}


def legendre_point(q, n):
    """Exact P_k(q) and P_k'(q) for k = 0..n at a rational q, enclosed by floats: arrays (lo, hi) of shape (2, n+1)."""
    key = (q, n)
    if key in _pt_cache:
        return _pt_cache[key]
    P = [Fraction(1), q]
    D = [Fraction(0), Fraction(1)]
    for k in range(1, n):
        P.append(((2 * k + 1) * q * P[k] - k * P[k - 1]) / (k + 1))
        D.append(D[k - 1] + (2 * k + 1) * P[k])
    lo = np.array([[frac_lo(v) for v in P[:n + 1]], [frac_lo(v) for v in D[:n + 1]]])
    hi = np.array([[frac_hi(v) for v in P[:n + 1]], [frac_hi(v) for v in D[:n + 1]]])
    _pt_cache[key] = (lo, hi)
    return lo, hi


_bd_cache = {}


def legendre_bounds(R, n):
    """Upper bounds of max |P_k'| and max |P_k''| on |t| <= R (R >= 1 a float): array shape (2, n+1)."""
    key = (R, n)
    if key in _bd_cache:
        return _bd_cache[key]
    q = Fraction(R)
    P = [Fraction(1), q]
    D = [Fraction(0), Fraction(1)]
    E = [Fraction(0), Fraction(0)]
    for k in range(1, n):
        P.append(((2 * k + 1) * q * P[k] - k * P[k - 1]) / (k + 1))
        D.append(D[k - 1] + (2 * k + 1) * P[k])
        E.append(E[k - 1] + (2 * k + 1) * D[k])
    out = np.array([[frac_hi(v) for v in D[:n + 1]], [frac_hi(v) for v in E[:n + 1]]])
    _bd_cache[key] = out
    return out


def coord_tables(dom, n, c0, cr):
    """Enclosures of P_k(t) and dP_k/dx over the boxes c0 +- cr of one coordinate (arrays over points).

    Returns (Plo, Phi, Dlo, Dhi), each of shape (n+1, number of points)."""
    s = 1.0 if dom in BOX else 2.0
    npt = len(c0)
    Plo = np.empty((n + 1, npt))
    Phi = np.empty((n + 1, npt))
    Dlo = np.empty((n + 1, npt))
    Dhi = np.empty((n + 1, npt))
    for i in range(npt):
        x = Fraction(float(c0[i]))
        t = x if dom in BOX else 2 * x - 1
        lo, hi = legendre_point(t, n)
        rho = float(up(s * float(cr[i])))
        if rho > 0:
            R = max(1.0, float(up(frac_hi(abs(t)) + rho)))
            bd = legendre_bounds(R, n)
            wP = up(rho * bd[0])
            wD = up(rho * bd[1])
        else:
            wP = np.zeros(n + 1)
            wD = np.zeros(n + 1)
        Plo[:, i] = dn(lo[0] - wP)
        Phi[:, i] = up(hi[0] + wP)
        Dlo[:, i] = dn(s * dn(lo[1] - wD))
        Dhi[:, i] = up(s * up(hi[1] + wD))
    return Plo, Phi, Dlo, Dhi


# ---------------------------------------------------------------- orthogonal basis on the simplex (Dubiner / PKD)
#
# On the unit triangle psi_ij = q_i(A, S) * P_j^(2i+1,0)(2y - 1) with A = 2x + y - 1, S = 1 - y, and
# on the unit tetrahedron psi_ijk = q_i(A1, S1) * h_ij(A2, S2) * P_k^(2i+2j+2,0)(2z - 1) with
# A1 = 2x + y + z - 1, S1 = 1 - y - z, A2 = 2y + z - 1, S2 = 1 - z.
# q_i(A, S) = S^i P_i(A / S) and h_ij(A, S) = S^j P_j^(2i+1,0)(A / S) are homogeneous polynomials.
# The products are orthogonal on the simplex, so the integral is the volume for the constant and 0 otherwise.
# Values at the box centres are computed exactly (fractions), with exact first and second derivatives.
# Over a box the values and gradients are enclosed by a second-order Taylor form whose remainder is bounded
# by interval enclosures of the third derivatives (interval three-term recurrences over the box).


def pkd_index_maps(dom, idx):
    """Positions of the factors of every basis function in the factor tables."""
    if dom == "triangle":
        return dict(i=np.array([a[0] for a in idx]), r=np.array(list(range(len(idx)))))
    ij = sorted(set((a[0], a[1]) for a in idx))
    pos_ij = {v: k for k, v in enumerate(ij)}
    mk = sorted(set((a[0] + a[1], a[2]) for a in idx))
    pos_mk = {v: k for k, v in enumerate(mk)}
    return dict(i=np.array([a[0] for a in idx]), ij=ij, mk=mk,
                h=np.array([pos_ij[(a[0], a[1])] for a in idx]), r=np.array([pos_mk[(a[0] + a[1], a[2])] for a in idx]))


_tab_cache = {}


def pkd_factor_tables(dom, n, idx, maps, c0):
    """Exact factor values (and partial derivatives) at the box centres, enclosed by floats (cached).

    Every table is a pair (lo, hi) of arrays with one column per point."""
    key = (dom, n, tuple(idx), c0.tobytes())
    if key not in _tab_cache:
        _tab_cache.clear()
        _tab_cache[key] = _pkd_factor_tables(dom, n, idx, maps, c0)
    return _tab_cache[key]


def dyadic(q):
    """Integer numerator and binary exponent E of a dyadic rational q = num / 2^E."""
    den = q.denominator
    E = den.bit_length() - 1
    if den != 1 << E:
        raise ValueError("not dyadic")
    return q.numerator, E


def homog_jacobi_int(alpha, A, S, N):
    """H_m = S^m P_m^(alpha,0)(A / S) and its partials up to order 2 in A and S, m = 0..N, for dyadic A and S.

    Exact integer recurrences with a common denominator, then one correctly rounded division per value
    (Python int / int is correctly rounded). Returns 6 lists of floats: H, HA, HS, HAA, HAS, HSS."""
    an, ea = dyadic(A)
    sn, es = dyadic(S)
    E = max(ea, es)
    a = an << (E - ea)
    s = sn << (E - es)
    G = [2, (alpha + 2) * a + alpha * s]
    GA = [0, alpha + 2]
    GS = [0, alpha]
    GAA = [0, 0]
    GAS = [0, 0]
    GSS = [0, 0]
    K = [1, 1]
    for m in range(1, N):
        a1 = 2 * (m + 1) * (m + alpha + 1) * (2 * m + alpha)
        a2 = (2 * m + alpha + 1) * alpha * alpha
        a3 = (2 * m + alpha) * (2 * m + alpha + 1) * (2 * m + alpha + 2)
        a4 = 2 * (m + alpha) * m * (2 * m + alpha + 2)
        c = a4 * (2 * (m) * (m + alpha) * (2 * m + alpha - 2) if m >= 2 else 1)
        lin = a2 * s + a3 * a
        ss = s * s
        G.append(lin * G[m] - c * ss * G[m - 1])
        GA.append(a3 * G[m] + lin * GA[m] - c * ss * GA[m - 1])
        GS.append(a2 * G[m] + lin * GS[m] - c * (2 * s * G[m - 1] + ss * GS[m - 1]))
        GAA.append(2 * a3 * GA[m] + lin * GAA[m] - c * ss * GAA[m - 1])
        GAS.append(a3 * GS[m] + a2 * GA[m] + lin * GAS[m] - c * (2 * s * GA[m - 1] + ss * GAS[m - 1]))
        GSS.append(2 * a2 * GS[m] + lin * GSS[m] - c * (2 * G[m - 1] + 4 * s * GS[m - 1] + ss * GSS[m - 1]))
        K.append(K[m] * a1)
    out = []
    for Gx, order in ((G, 0), (GA, 1), (GS, 1), (GAA, 2), (GAS, 2), (GSS, 2)):
        vals = []
        for m in range(N + 1):
            e = (m - order) * E
            num = Gx[m]
            den = 2 * K[m]
            if e >= 0:
                den = den << e
            else:
                num = num << (-e)
            vals.append(num / den)
        out.append(vals)
    return out


def _pkd_factor_tables(dom, n, idx, maps, c0):
    npt = c0.shape[0]
    cols = []
    for p in range(npt):
        x = [Fraction(float(v)) for v in c0[p]]
        cache = {}

        def hj(alpha, A, S, N):
            key = (alpha, N)
            if key not in cache:
                cache[key] = homog_jacobi_int(alpha, A, S, N)
            return cache[key]

        if dom == "triangle":
            A = 2 * x[0] + x[1] - 1
            S = 1 - x[1]
            t = 2 * x[1] - 1
            q, qA, qS, qAA, qAS, qSS = homog_jacobi_int(0, A, S, n)
            rv = []
            rd = []
            r2 = []
            for a in idx:
                H, HA, HS, HAA, HAS, HSS = hj(2 * a[0] + 1, t, Fraction(1), n - a[0])
                rv.append(H[a[1]])
                rd.append(2 * HA[a[1]])
                r2.append(4 * HAA[a[1]])
            cols.append(dict(q=q, qA=qA, qS=qS, qAA=qAA, qAS=qAS, qSS=qSS, rv=rv, rd=rd, r2=r2))
        else:
            A1 = 2 * x[0] + x[1] + x[2] - 1
            S1 = 1 - x[1] - x[2]
            A2 = 2 * x[1] + x[2] - 1
            S2 = 1 - x[2]
            t = 2 * x[2] - 1
            q, qA, qS, qAA, qAS, qSS = homog_jacobi_int(0, A1, S1, n)
            hv = []
            hA = []
            hS = []
            hAA = []
            hAS = []
            hSS = []
            for i, j in maps["ij"]:
                H, HA, HS, HAA, HAS, HSS = hj(2 * i + 1, A2, S2, n - i)
                hv.append(H[j])
                hA.append(HA[j])
                hS.append(HS[j])
                hAA.append(HAA[j])
                hAS.append(HAS[j])
                hSS.append(HSS[j])
            cache.clear()
            rv = []
            rd = []
            r2 = []
            for m, k in maps["mk"]:
                H, HA, HS, HAA, HAS, HSS = hj(2 * m + 2, t, Fraction(1), n - m)
                rv.append(H[k])
                rd.append(2 * HA[k])
                r2.append(4 * HAA[k])
            cols.append(dict(q=q, qA=qA, qS=qS, qAA=qAA, qAS=qAS, qSS=qSS, hv=hv, hA=hA, hS=hS, hAA=hAA, hAS=hAS, hSS=hSS, rv=rv, rd=rd, r2=r2))
    tabs = {}
    for key in cols[0]:
        f = np.array([cols[p][key] for p in range(npt)], dtype=float).T
        # every value is a correctly rounded float (the factors 2 and 4 are exact), so one step outward encloses it
        tabs[key] = (dn(f), up(f))
    return tabs


def iv_add(a, b):
    return dn(a[0] + b[0]), up(a[1] + b[1])


def iv_sub(a, b):
    return dn(a[0] - b[1]), up(a[1] - b[0])


def iv_take(t, rows):
    return t[0][rows], t[1][rows]


def iv_m(a, b):
    return iv_mul(a[0], a[1], b[0], b[1])


def iv_2(a):
    return 2.0 * a[0], 2.0 * a[1]


ORDERS = [(p, q) for t in range(4) for p in range(t + 1) for q in [t - p]]


def iv_lin(c0, cr, coef, const):
    """Interval of the linear form const + sum coef[c] x_c over the boxes c0 +- cr."""
    lo = np.full(c0.shape[0], float(const))
    hi = lo.copy()
    for c, a in enumerate(coef):
        if a == 0:
            continue
        xl = dn(c0[:, c] - cr[:, c])
        xh = up(c0[:, c] + cr[:, c])
        tl = np.minimum(a * xl, a * xh)
        th = np.maximum(a * xl, a * xh)
        lo = dn(lo + tl)
        hi = up(hi + th)
    return lo, hi


def homog_jacobi_box(alpha, A, S, N):
    """Interval enclosures of all partial derivatives of order <= 3 (in A and S) of
    H_m = S^m P_m^(alpha,0)(A / S), m = 0..N, for interval arguments A and S.

    Returns a dict (p, q) -> list over m of (lo, hi). Plain interval arithmetic: the result is
    rigorous but wider than the true range; it is only used for the third-order remainder."""
    one = np.ones_like(A[0])
    zero = np.zeros_like(A[0])
    H = {o: [(zero, zero)] for o in ORDERS}
    H[(0, 0)] = [(one, one)]
    if N >= 1:
        h1 = iv_add((dn((alpha + 2) * A[0] / 2), up((alpha + 2) * A[1] / 2)), (dn(alpha * S[0] / 2), up(alpha * S[1] / 2)))
        for o in ORDERS:
            H[o].append((zero, zero))
        H[(0, 0)][1] = h1
        H[(1, 0)][1] = (one * (alpha + 2) / 2, one * (alpha + 2) / 2)
        H[(0, 1)][1] = (one * alpha / 2, one * alpha / 2)
    S2 = iv_mul(S[0], S[1], S[0], S[1])
    S2x = (dn(2 * S[0]), up(2 * S[1]))
    for m in range(1, N):
        a1 = float(2 * (m + 1) * (m + alpha + 1) * (2 * m + alpha))
        a2 = float((2 * m + alpha + 1) * alpha * alpha)
        a3 = float((2 * m + alpha) * (2 * m + alpha + 1) * (2 * m + alpha + 2))
        a4 = float(2 * (m + alpha) * m * (2 * m + alpha + 2))
        L = iv_add((dn(a2 * S[0]), up(a2 * S[1])), (dn(a3 * A[0]), up(a3 * A[1])))
        for p, q in ORDERS:
            t = iv_mul(L[0], L[1], *H[(p, q)][m])
            if p > 0:
                v = H[(p - 1, q)][m]
                t = iv_add(t, (dn(p * a3 * v[0]), up(p * a3 * v[1])))
            if q > 0:
                v = H[(p, q - 1)][m]
                t = iv_add(t, (dn(q * a2 * v[0]), up(q * a2 * v[1])))
            b = iv_mul(S2[0], S2[1], *H[(p, q)][m - 1])
            if q > 0:
                v = H[(p, q - 1)][m - 1]
                b = iv_add(b, iv_mul(dn(q * S2x[0]), up(q * S2x[1]), v[0], v[1]))
            if q > 1:
                v = H[(p, q - 2)][m - 1]
                b = iv_add(b, (dn(q * (q - 1) * v[0]), up(q * (q - 1) * v[1])))
            t = iv_sub(t, (dn(a4 * b[0]), up(a4 * b[1])))
            H[(p, q)].append((dn(t[0] / a1), up(t[1] / a1)))
    return H


def absb(t):
    return np.maximum(np.abs(t[0]), np.abs(t[1]))


def factor_box_bounds(H, rows, ga, gs, cr):
    """Abs bounds over the boxes of a factor F(A, S) and its directional derivatives.

    Returns P0, Pc (list over c of |dF/dx_c|), P1 = sum_e |dF/dx_e| rho_e, P2 = D^2F[rho, rho] bound,
    Pch (list over c of sum_e |d2F/dx_c dx_e| rho_e) and Pchh (list over c of the third-order bound)."""
    d = cr.shape[1]
    g = lambda o: np.array([absb(H[o][m]) for m in rows])
    F0 = g((0, 0))
    FA = g((1, 0))
    FS = g((0, 1))
    FAA = g((2, 0))
    FAS = g((1, 1))
    FSS = g((0, 2))
    FAAA = g((3, 0))
    FAAS = g((2, 1))
    FASS = g((1, 2))
    FSSS = g((0, 3))
    al = sum(abs(ga[c]) * cr[:, c] for c in range(d))[None, :]
    si = sum(abs(gs[c]) * cr[:, c] for c in range(d))[None, :]
    P1 = FA * al + FS * si
    P2 = FAA * al * al + 2 * FAS * al * si + FSS * si * si
    Pc = [FA * abs(ga[c]) + FS * abs(gs[c]) for c in range(d)]
    Pch = [abs(ga[c]) * (FAA * al + FAS * si) + abs(gs[c]) * (FAS * al + FSS * si) for c in range(d)]
    Pchh = [abs(ga[c]) * (FAAA * al * al + 2 * FAAS * al * si + FASS * si * si) + abs(gs[c]) * (FAAS * al * al + 2 * FASS * al * si + FSSS * si * si) for c in range(d)]
    return F0, Pc, P1, P2, Pch, Pchh


def pkd_remainders(dom, n, idx, maps, c0, cr):
    """Rigorous bounds of the second-order Taylor remainders over the boxes, per basis function and point:
    R2 >= |D^2 psi[h, h]| and R3[c] >= |d/dx_c D^2 psi[h, h]| for all |h_e| <= cr_e and all points of the box."""
    d = DIM[dom]
    unit = (np.ones(c0.shape[0]), np.ones(c0.shape[0]))
    facs = []
    if dom == "triangle":
        A = iv_lin(c0, cr, (2, 1), -1)
        S = iv_lin(c0, cr, (0, -1), 1)
        T = iv_lin(c0, cr, (0, 2), -1)
        Hq = homog_jacobi_box(0, A, S, n)
        facs.append(factor_box_bounds(Hq, [a[0] for a in idx], (2, 1), (0, -1), cr))
        cache = {}
        rows = []
        for a in idx:
            if a[0] not in cache:
                cache[a[0]] = homog_jacobi_box(2 * a[0] + 1, T, unit, n - a[0])
        # gather the r-factor bounds basis by basis
        parts = [factor_box_bounds(cache[a[0]], [a[1]], (0, 2), (0, 0), cr) for a in idx]
        facs.append(tuple(np.concatenate([pt[k] for pt in parts]) if not isinstance(parts[0][k], list) else [np.concatenate([pt[k][c] for pt in parts]) for c in range(d)] for k in range(6)))
    else:
        A1 = iv_lin(c0, cr, (2, 1, 1), -1)
        S1 = iv_lin(c0, cr, (0, -1, -1), 1)
        A2 = iv_lin(c0, cr, (0, 2, 1), -1)
        S2 = iv_lin(c0, cr, (0, 0, -1), 1)
        T = iv_lin(c0, cr, (0, 0, 2), -1)
        Hq = homog_jacobi_box(0, A1, S1, n)
        facs.append(factor_box_bounds(Hq, [a[0] for a in idx], (2, 1, 1), (0, -1, -1), cr))
        ch = {}
        cr_ = {}
        ph = []
        pr = []
        for a in idx:
            i, j, k = a
            if i not in ch:
                ch[i] = homog_jacobi_box(2 * i + 1, A2, S2, n - i)
            m = i + j
            if m not in cr_:
                cr_[m] = homog_jacobi_box(2 * m + 2, T, unit, n - m)
        hb = {}
        rb = {}
        for i in ch:
            hb[i] = factor_box_bounds(ch[i], list(range(n - i + 1)), (0, 2, 1), (0, 0, -1), cr)
        for m in cr_:
            rb[m] = factor_box_bounds(cr_[m], list(range(n - m + 1)), (0, 0, 2), (0, 0, 0), cr)

        def gather(tab, key_row):
            out = []
            for k in range(6):
                if isinstance(tab[next(iter(tab))][k], list):
                    out.append([np.array([tab[kk][k][c][rr] for kk, rr in key_row]) for c in range(d)])
                else:
                    out.append(np.array([tab[kk][k][rr] for kk, rr in key_row]))
            return tuple(out)

        facs.append(gather(hb, [(a[0], a[1]) for a in idx]))
        facs.append(gather(rb, [(a[0] + a[1], a[2]) for a in idx]))
    nf = len(facs)
    P0 = [f[0] for f in facs]
    Pc = [f[1] for f in facs]
    P1 = [f[2] for f in facs]
    P2 = [f[3] for f in facs]
    Pch = [f[4] for f in facs]
    Pchh = [f[5] for f in facs]

    def others(excl):
        v = 1.0
        for e in range(nf):
            if e not in excl:
                v = v * P0[e]
        return v

    R2 = 0.0
    for f in range(nf):
        R2 = R2 + P2[f] * others({f})
        for g in range(nf):
            if g != f:
                R2 = R2 + P1[f] * P1[g] * others({f, g})
    R3 = []
    for c in range(d):
        t = 0.0
        for f in range(nf):
            t = t + Pchh[f][c] * others({f})
            for g in range(nf):
                if g == f:
                    continue
                t = t + 2 * Pch[f][c] * P1[g] * others({f, g})
                t = t + Pc[f][c] * P2[g] * others({f, g})
                for k in range(nf):
                    if k != f and k != g:
                        t = t + Pc[f][c] * P1[g] * P1[k] * others({f, g, k})
        R3.append(up(t * (1 + 64 * U)))
    return up(R2 * (1 + 64 * U)), R3


def pkd_enclosure(par, n, c0, cr, want_grad):
    """Enclosures of the PKD basis (rows) at the points (columns) and of their gradients over the boxes."""
    dom = par.dom
    d = DIM[dom]
    idx = par.idx
    maps = pkd_index_maps(dom, idx)
    tabs = pkd_factor_tables(dom, n, idx, maps, c0)
    qi = maps["i"]
    q = iv_take(tabs["q"], qi)
    qA = iv_take(tabs["qA"], qi)
    qS = iv_take(tabs["qS"], qi)
    if dom == "triangle":
        r = tabs["rv"]
        rd = tabs["rd"]
        val = iv_m(q, r)
        g = [iv_m(iv_2(qA), r), iv_add(iv_m(iv_sub(qA, qS), r), iv_m(q, rd))]
    else:
        h = iv_take(tabs["hv"], maps["h"])
        hA = iv_take(tabs["hA"], maps["h"])
        hS = iv_take(tabs["hS"], maps["h"])
        r = iv_take(tabs["rv"], maps["r"])
        rd = iv_take(tabs["rd"], maps["r"])
        qh = iv_m(q, h)
        hr = iv_m(h, r)
        val = iv_m(qh, r)
        qmS = iv_sub(qA, qS)
        g = [iv_m(iv_2(qA), hr),
             iv_add(iv_m(qmS, hr), iv_m(iv_m(q, iv_2(hA)), r)),
             iv_add(iv_add(iv_m(qmS, hr), iv_m(iv_m(q, iv_sub(hA, hS)), r)), iv_m(qh, rd))]
    # boxes: second-order Taylor form around the centre with the exact gradient and Hessian at the centre;
    # the remainder uses interval enclosures of the third derivatives over the boxes
    rad = cr.sum(axis=1)
    if np.any(rad > 0):
        absv = lambda t: np.maximum(np.abs(t[0]), np.abs(t[1]))
        # factors: (abs value, abs partials in A and S, abs second partials, gradients of A and S)
        if dom == "triangle":
            facs = [(q, qA, qS, iv_take(tabs["qAA"], qi), iv_take(tabs["qAS"], qi), iv_take(tabs["qSS"], qi), (2, 1), (0, -1)),
                    (r, rd, None, tabs["r2"], None, None, (0, 1), None)]
        else:
            facs = [(q, qA, qS, iv_take(tabs["qAA"], qi), iv_take(tabs["qAS"], qi), iv_take(tabs["qSS"], qi), (2, 1, 1), (0, -1, -1)),
                    (h, hA, hS, iv_take(tabs["hAA"], maps["h"]), iv_take(tabs["hAS"], maps["h"]), iv_take(tabs["hSS"], maps["h"]), (0, 2, 1), (0, 0, -1)),
                    (r, rd, None, iv_take(tabs["r2"], maps["r"]), None, None, (0, 0, 1), None)]
        F = []
        G = []
        GR = []
        HR = []
        for fv, fA, fS, fAA, fAS, fSS, ga, gs in facs:
            F.append(absv(fv))
            vA = absv(fA)
            vS = absv(fS) if fS is not None else 0.0
            aA = [abs(v) for v in ga]
            aS = [abs(v) for v in gs] if gs is not None else [0] * d
            arho = sum(aA[c] * cr[:, c] for c in range(d))
            srho = sum(aS[c] * cr[:, c] for c in range(d))
            Gc = [vA * aA[c] + vS * aS[c] for c in range(d)]
            G.append(Gc)
            GR.append(sum(Gc[c] * cr[:, c][None, :] for c in range(d)))
            wAA = absv(fAA)
            wAS = absv(fAS) if fAS is not None else 0.0
            wSS = absv(fSS) if fSS is not None else 0.0
            HR.append([wAA * aA[c] * arho[None, :] + wAS * (aA[c] * srho[None, :] + aS[c] * arho[None, :]) + wSS * aS[c] * srho[None, :] for c in range(d)])
        nf = len(facs)
        w1 = 0.0
        for c in range(d):
            gc = np.maximum(np.abs(g[c][0]), np.abs(g[c][1]))
            w1 = w1 + gc * cr[:, c][None, :]
        R2, R3 = pkd_remainders(dom, n, idx, maps, c0, cr)
        w1 = up((w1 + 0.5 * R2) * (1 + 16 * U))
        val = (dn(val[0] - w1), up(val[1] + w1))
        ng = []
        for c in range(d):
            hb = 0.0
            for f in range(nf):
                other = 1.0
                for e in range(nf):
                    if e != f:
                        other = other * F[e]
                hb = hb + other * HR[f][c]
                for e in range(nf):
                    if e == f:
                        continue
                    rest = 1.0
                    for k in range(nf):
                        if k != f and k != e:
                            rest = rest * F[k]
                    hb = hb + rest * G[f][c] * GR[e]
            w2 = up((hb + 0.5 * R3[c]) * (1 + 32 * U))
            ng.append((dn(g[c][0] - w2), up(g[c][1] + w2)))
        g = ng
    return val, (g if want_grad else [])


def pkd_float(dom, n, idx, X):
    """PKD basis in floating point at the points X (rows): matrix points x basis."""
    X = np.asarray(X, dtype=float)

    def hj(alpha, A, S, N):
        H = [np.ones_like(A)]
        if N >= 1:
            H.append(((alpha + 2) * A + alpha * S) / 2)
        for m in range(1, N):
            a1 = 2 * (m + 1) * (m + alpha + 1) * (2 * m + alpha)
            a2 = (2 * m + alpha + 1) * alpha * alpha
            a3 = (2 * m + alpha) * (2 * m + alpha + 1) * (2 * m + alpha + 2)
            a4 = 2 * (m + alpha) * m * (2 * m + alpha + 2)
            H.append(((a2 * S + a3 * A) * H[m] - a4 * S * S * H[m - 1]) / a1)
        return H

    one = np.ones(len(X))
    out = np.zeros((len(X), len(idx)))
    if dom == "triangle":
        q = hj(0, 2 * X[:, 0] + X[:, 1] - 1, 1 - X[:, 1], n)
        cache = {}
        for k, a in enumerate(idx):
            if a[0] not in cache:
                cache[a[0]] = hj(2 * a[0] + 1, 2 * X[:, 1] - 1, one, n - a[0])
            out[:, k] = q[a[0]] * cache[a[0]][a[1]]
        return out
    q = hj(0, 2 * X[:, 0] + X[:, 1] + X[:, 2] - 1, 1 - X[:, 1] - X[:, 2], n)
    ch = {}
    cr = {}
    for k, a in enumerate(idx):
        if a[0] not in ch:
            ch[a[0]] = hj(2 * a[0] + 1, 2 * X[:, 1] + X[:, 2] - 1, 1 - X[:, 2], n - a[0])
        m = a[0] + a[1]
        if m not in cr:
            cr[m] = hj(2 * m + 2, 2 * X[:, 2] - 1, one, n - m)
        out[:, k] = q[a[0]] * ch[a[0]][a[1]] * cr[m][a[2]]
    return out


# ---------------------------------------------------------------- basis and exact integrals


def full_indices(d, n):
    return [a for a in product(range(n + 1), repeat=d) if sum(a) <= n]


def shifted_legendre_coeffs(k):
    """Integer coefficients of P_k(2x - 1) in powers of x."""
    return [(-1) ** (k + j) * math.comb(k, j) * math.comb(k + j, j) for j in range(k + 1)]


_int_cache = {}


def exact_integrals(dom, idx):
    """Exact integrals of the basis products over the reference domain (fractions)."""
    key = (dom, tuple(idx))
    if key not in _int_cache:
        _int_cache[key] = _exact_integrals(dom, idx)
    return _int_cache[key]


def _exact_integrals(dom, idx):
    d = DIM[dom]
    if dom in BOX:
        return [Fraction(2 ** d) if sum(a) == 0 else Fraction(0) for a in idx]
    n = max(sum(a) for a in idx)
    big = math.factorial(n + d)
    fact = [math.factorial(j) for j in range(n + 1)]
    coef = [shifted_legendre_coeffs(k) for k in range(n + 1)]
    out = []
    for a in idx:
        tot = 0
        for js in product(*[range(k + 1) for k in a]):
            c = 1
            num = 1
            for k, j in zip(a, js):
                c *= coef[k][j]
                num *= fact[j]
            tot += c * num * (big // math.factorial(sum(js) + d))
        out.append(Fraction(tot, big))
    return out


# ---------------------------------------------------------------- parametrisation of a rule


class Param:
    """Rule as a function of an unknown vector u.

    Every point p has a weight u[widx[p]]. Coordinate c of point p is the affine form
    const[p, c] + sum_t coef[p, c, t] * u[kidx[p, c, t]] (kidx = -1 for unused terms, coef = +-1).
    cons: extra linear equations sum_k c_k u[k] = 1 with integer c_k (barycentric sums of simplex orbits).
    bary: indices of u that are barycentric coordinates (all nodes are inside iff these are >= 0)."""

    def __init__(self, dom, mode, u0, widx, cidx, csgn, cconst, idx, cons=None, bary=None):
        self.cons = cons or []
        self.bary = bary
        self.dom = dom
        self.mode = mode
        self.u0 = np.array(u0, dtype=float)
        self.widx = np.array(widx, dtype=int)
        ci = np.array(cidx, dtype=int)
        self.kidx = ci[:, :, None].copy()
        self.coef = np.array(csgn, dtype=float)[:, :, None].copy()
        self.const = np.array(cconst, dtype=float)
        self.idx = idx

    def set_affine(self, p, c, terms, const):
        """Make coordinate c of point p the affine form const + sum coef * u[k] over terms [(k, coef)]."""
        T = max(self.kidx.shape[2], len(terms))
        if T > self.kidx.shape[2]:
            pad = T - self.kidx.shape[2]
            self.kidx = np.concatenate([self.kidx, -np.ones(self.kidx.shape[:2] + (pad,), dtype=int)], axis=2)
            self.coef = np.concatenate([self.coef, np.zeros(self.coef.shape[:2] + (pad,))], axis=2)
        self.kidx[p, c, :] = -1
        self.coef[p, c, :] = 0.0
        for t, (k, cf) in enumerate(terms):
            self.kidx[p, c, t] = k
            self.coef[p, c, t] = cf
        self.const[p, c] = const

    def points(self, u, ru):
        """Centres and radii of weights and coordinates for the box u +- ru (radii include rounding)."""
        w0 = u[self.widx]
        wr = ru[self.widx]
        valid = self.kidx >= 0
        k = np.where(valid, self.kidx, 0)
        terms = np.where(valid, self.coef * u[k], 0.0)
        nterm = valid.sum(axis=2)
        c0 = self.const + terms.sum(axis=2)
        cr = (np.where(valid, np.abs(self.coef) * ru[k], 0.0)).sum(axis=2)
        # a single term without constant is exact; otherwise add a bound for the rounding of the sum
        inexact = (nterm >= 2) | ((nterm == 1) & (self.const != 0))
        err = (nterm + 1) * 2 * U * (np.abs(self.const) + np.abs(terms).sum(axis=2))
        cr = np.where(inexact, up(cr + err), cr)
        cr = np.where(nterm > 0, up(cr * (1 + 4 * U)) * (cr > 0), 0.0)
        return w0, wr, c0, cr

    def bary_forms(self):
        """Barycentric coordinates of every point as affine forms: list over points of d+1 (dict, const)."""
        d = self.const.shape[1]
        out = []
        for p in range(self.const.shape[0]):
            forms = []
            tot = {}
            totc = 1.0
            for c in range(d):
                f = {}
                for t in range(self.kidx.shape[2]):
                    k = int(self.kidx[p, c, t])
                    if k >= 0:
                        f[k] = f.get(k, 0.0) + float(self.coef[p, c, t])
                        tot[k] = tot.get(k, 0.0) - float(self.coef[p, c, t])
                forms.append((f, float(self.const[p, c])))
                totc -= float(self.const[p, c])
            forms.insert(0, ({k: v for k, v in tot.items() if v != 0.0}, totc))
            out.append(forms)
        return out


def full_param(dom, n, X, W, Xq=None, zero_tol=0.0):
    """All weights and coordinates free, except coordinates that lie exactly on a boundary face.

    On the simplex a node with 1 - sum x = 0 (or |1 - sum x| <= zero_tol) keeps that exactly: its largest
    coordinate becomes 1 minus the others."""
    d = DIM[dom]
    u0 = []
    widx = []
    cidx = []
    csgn = []
    cconst = []
    for p in range(len(W)):
        widx.append(len(u0))
        u0.append(W[p])
        ci = []
        cs = []
        cc = []
        for c in range(d):
            v = X[p][c]
            fixed = (dom in BOX and abs(v) == 1.0) or (dom not in BOX and v == 0.0)
            if fixed:
                ci.append(-1)
                cs.append(1.0)
                cc.append(v)
            else:
                ci.append(len(u0))
                cs.append(1.0)
                cc.append(0.0)
                u0.append(v)
        cidx.append(ci)
        csgn.append(cs)
        cconst.append(cc)
    par = Param(dom, "full", u0, widx, cidx, csgn, cconst, full_indices(d, n))
    if dom not in BOX and Xq is not None:
        drop = []
        for p in range(len(W)):
            lam0 = 1 - sum(Xq[p])
            if lam0 == 0 or abs(lam0) <= zero_tol:
                cm = int(np.argmax(X[p]))
                if par.kidx[p, cm, 0] < 0:
                    continue
                drop.append(int(par.kidx[p, cm, 0]))
                terms = [(int(par.kidx[p, c, 0]), -1.0) for c in range(d) if c != cm and par.kidx[p, c, 0] >= 0]
                par.set_affine(p, cm, terms, 1.0)
        if drop:
            # remove the unknowns that became dependent and renumber
            keep = np.array([k for k in range(len(par.u0)) if k not in set(drop)])
            new = -np.ones(len(par.u0), dtype=int)
            new[keep] = np.arange(len(keep))
            par.u0 = par.u0[keep]
            par.widx = new[par.widx]
            par.kidx = np.where(par.kidx >= 0, new[np.maximum(par.kidx, 0)], -1)
    return par


def group_elements(d):
    """All signed permutations g of R^d as (perm, signs): (g x)[c] = signs[c] * x[perm[c]]."""
    return [(pm, sg) for pm in permutations(range(d)) for sg in product([1.0, -1.0], repeat=d)]


def apply_g(g, x):
    pm, sg = g
    return tuple(sg[c] * x[pm[c]] for c in range(len(x)))


def sym_param(dom, n, X, W, tol=1e-9):
    """Symmetry-reduced parametrisation for box domains.

    The group G is the set of all signed permutations that map the stored rule onto itself.
    Returns (param, deviation, group size) or None if G is trivial; deviation bounds the distance of
    the stored nodes and weights from the symmetrised rule at u0."""
    d = DIM[dom]
    keyof = lambda x: tuple(int(round(v / tol)) for v in x)
    where = {}
    for p in range(len(W)):
        where.setdefault(keyof(X[p]), []).append(p)
    G = []
    for g in group_elements(d):
        ok = True
        for p in range(len(W)):
            img = where.get(keyof(apply_g(g, X[p])))
            if not img or not any(abs(W[q] - W[p]) <= tol * max(1.0, abs(W[p])) for q in img):
                ok = False
                break
        if ok:
            G.append(g)
    if len(G) <= 1:
        return None
    # equations: one per orbit of Legendre indices whose stabiliser has no sign change
    idx = []
    seen = set()
    for a in full_indices(d, n):
        if a in seen:
            continue
        orb = {}
        good = True
        for pm, sg in G:
            b = [0] * d
            sign = 1.0
            for c in range(d):
                b[pm[c]] = a[c]
                sign *= sg[c] ** a[c]
            b = tuple(b)
            if b in orb and orb[b] != sign:
                good = False
            orb[b] = sign
        seen.update(orb)
        if good:
            idx.append(a)
    # unknowns: per node orbit a weight and the free coordinates of its representative
    u0 = []
    widx = []
    cidx = []
    csgn = []
    cconst = []
    dev = 0.0
    done = set()
    for p in range(len(W)):
        if p in done:
            continue
        rep = X[p]
        orbit = set()
        for g in G:
            q = where[keyof(apply_g(g, rep))]
            orbit.update(q)
        done.update(orbit)
        stab = [g for g in G if keyof(apply_g(g, rep)) == keyof(rep)]
        # union-find with signs for the relations y[c] = s * y[perm[c]] of the stabiliser
        parent = list(range(d))
        rel = [1.0] * d
        zero = [False] * d

        def find(c):
            if parent[c] == c:
                return c, 1.0
            r, s2 = find(parent[c])
            return r, s2 * rel[c]

        for pm, sg in stab:
            for c in range(d):
                ra, sa = find(c)
                rb, sb = find(pm[c])
                want = sg[c]
                if ra == rb:
                    if sa != want * sb:
                        zero[ra] = True
                else:
                    parent[ra] = rb
                    rel[ra] = want * sb * sa
                    zero[rb] = zero[rb] or zero[ra]
        roots = {}
        for c in range(d):
            r, sc = find(c)
            roots.setdefault(r, []).append((c, sc))
        wi = len(u0)
        u0.append(sum(W[q] for q in orbit) / len(orbit))
        dev = max(dev, max(abs(W[q] - u0[wi]) for q in orbit))
        sym = {}
        for r, mem in roots.items():
            vals = [sc * rep[c] for c, sc in mem]
            v = sum(vals) / len(vals)
            if zero[find(r)[0]]:
                kind = ("const", 0.0)
            elif all(abs(rep[c]) == 1.0 for c, sc in mem) and all(x == vals[0] for x in vals):
                kind = ("const", vals[0])
            else:
                kind = ("par", len(u0))
                u0.append(v)
            for c, sc in mem:
                sym[c] = (kind, sc)
        pts = {}
        for g in G:
            pm, sg = g
            pt = []
            for c in range(d):
                (kind, val), sc = sym[pm[c]]
                if kind == "const":
                    pt.append(("const", sg[c] * sc * val))
                else:
                    pt.append(("par", val, sg[c] * sc))
            pts[tuple(pt)] = True
        if len(pts) != len(orbit):
            return None
        for pt in pts:
            widx.append(wi)
            ci = []
            cs = []
            cc = []
            for e in pt:
                if e[0] == "const":
                    ci.append(-1)
                    cs.append(1.0)
                    cc.append(e[1] if e[1] != 0 else 0.0)
                else:
                    ci.append(e[1])
                    cs.append(e[2])
                    cc.append(0.0)
            cidx.append(ci)
            csgn.append(cs)
            cconst.append(cc)
    par = Param(dom, "sym", u0, widx, cidx, csgn, cconst, idx)
    # distance of the stored nodes from the symmetrised points
    w0, wr, c0, cr = par.points(par.u0, np.zeros(len(u0)))
    P = np.array(X)
    for i in range(len(c0)):
        dev = max(dev, float(np.min(np.max(np.abs(P - c0[i]), axis=1))))
    return par, float(up(dev * (1 + 1e-6) + 1e-300)), len(G)


def count_invariants(G, d, n):
    """Exact dimension of the G-invariant polynomials of degree <= n on the d-simplex (Burnside).

    Polynomials of degree <= n in x correspond to homogeneous polynomials of degree n in the d+1
    barycentric coordinates; a permutation fixes a monomial iff the exponents are constant on its cycles."""
    tot = 0
    for sg in G:
        seen = [False] * (d + 1)
        lens = []
        for i in range(d + 1):
            if not seen[i]:
                j = i
                L = 0
                while not seen[j]:
                    seen[j] = True
                    j = sg[j]
                    L += 1
                lens.append(L)
        ways = [1] + [0] * n
        for L in lens:
            for m in range(L, n + 1):
                ways[m] += ways[m - L]
        tot += ways[n]
    assert tot % len(G) == 0
    return tot // len(G)


def bary(x):
    return (1.0 - sum(x),) + tuple(x)


def apply_s(sg, x):
    """Permutation of the barycentric coordinates: lambda'_i = lambda_sg(i)."""
    lam = bary(x)
    return tuple(lam[sg[c]] for c in range(1, len(lam)))


def legendre_float(t, n):
    P = [np.ones_like(t), t]
    for k in range(1, n):
        P.append(((2 * k + 1) * t * P[k] - k * P[k - 1]) / (k + 1))
    return P[:n + 1]


def choose_equations(G, d, n, dim, seed=1):
    """Legendre indices whose symmetrisations are linearly independent, by greedy Gram-Schmidt.

    Soundness does not depend on this numerical choice: the Krawczyk test succeeds only if the
    chosen equations have a nonsingular Jacobian, which implies that they are independent, and
    their number equals the exact dimension of the invariant space."""
    idx = full_indices(d, n)
    rng = np.random.default_rng(seed)
    npt = 3 * dim + 20
    lam = rng.dirichlet(np.ones(d + 1), size=npt)
    cols = np.zeros((npt, len(idx)))
    for sg in G:
        xs = lam[:, [sg[c] for c in range(1, d + 1)]]
        cols += pkd_float("triangle" if d == 2 else "tetra", n, idx, xs)
    norms = np.linalg.norm(cols, axis=0)
    # symmetrisations that vanish are only rounding noise; drop them before normalising
    dead = norms < 1e-9 * norms.max()
    norms[dead] = 1
    cols = cols / norms
    cols[:, dead] = 0
    chosen = []
    R = cols.copy()
    for _ in range(dim):
        nr = np.linalg.norm(R, axis=0)
        j = int(np.argmax(nr))
        if nr[j] < 1e-8:
            return None
        chosen.append(j)
        qv = R[:, j] / nr[j]
        R = R - np.outer(qv, qv @ R)
        R[:, j] = 0
    return [idx[j] for j in sorted(chosen)]


def simplex_sym_param(dom, n, X, W, Xq, tol=1e-9, zero_tol=0.0):
    """Symmetry-reduced parametrisation for the simplex: G = permutations of the barycentric
    coordinates that map the stored rule onto itself. Unknowns per orbit: a weight and one barycentric
    value per class of the stabiliser; a linear equation per orbit keeps the barycentric sum at 1."""
    d = DIM[dom]
    keyof = lambda x: tuple(int(round(v / tol)) for v in x)
    where = {}
    for p in range(len(W)):
        where.setdefault(keyof(X[p]), []).append(p)
    G = []
    for sg in permutations(range(d + 1)):
        ok = True
        for p in range(len(W)):
            img = where.get(keyof(apply_s(sg, X[p])))
            if not img or not any(abs(W[q] - W[p]) <= tol * max(1.0, abs(W[p])) for q in img):
                ok = False
                break
        if ok:
            G.append(sg)
    if len(G) <= 1:
        return None
    dim = count_invariants(G, d, n)
    idx = choose_equations(G, d, n, dim)
    if idx is None:
        return None
    u0 = []
    widx = []
    cidx = []
    csgn = []
    cconst = []
    cons = []
    bvars = []
    dev = 0.0
    done = set()
    for p in range(len(W)):
        if p in done:
            continue
        rep = X[p]
        lam = bary(rep)
        lamq = [1 - sum(Xq[p])] + list(Xq[p])
        orbit = set()
        for sg in G:
            orbit.update(where[keyof(apply_s(sg, rep))])
        done.update(orbit)
        stab = [sg for sg in G if keyof(apply_s(sg, rep)) == keyof(rep)]
        parent = list(range(d + 1))

        def find(i):
            while parent[i] != i:
                i = parent[i]
            return i

        for sg in stab:
            for i in range(d + 1):
                a = find(i)
                b = find(sg[i])
                if a != b:
                    parent[a] = b
        classes = {}
        for i in range(d + 1):
            classes.setdefault(find(i), []).append(i)
        wi = len(u0)
        u0.append(sum(W[q] for q in orbit) / len(orbit))
        dev = max(dev, max(abs(W[q] - u0[wi]) for q in orbit))
        of = {}
        row = []
        for r, mem in classes.items():
            if all(abs(lamq[i]) <= zero_tol for i in mem):
                of[r] = None
            else:
                k = len(u0)
                u0.append(sum(lam[i] for i in mem) / len(mem))
                of[r] = k
                bvars.append(k)
                row.append((k, len(mem)))
        cons.append(row)
        pts = {}
        for sg in G:
            pt = tuple(of[find(sg[c])] for c in range(1, d + 1))
            pts[pt] = True
        if len(pts) != len(orbit):
            return None
        for pt in pts:
            widx.append(wi)
            cidx.append([-1 if k is None else k for k in pt])
            csgn.append([1.0] * d)
            cconst.append([0.0] * d)
    par = Param(dom, "sym", u0, widx, cidx, csgn, cconst, idx, cons=cons, bary=bvars)
    w0, wr, c0, cr = par.points(par.u0, np.zeros(len(u0)))
    P = np.array(X)
    for i in range(len(c0)):
        dev = max(dev, float(np.min(np.max(np.abs(P - c0[i]), axis=1))))
    return par, float(up(dev * (1 + 1e-6) + 1e-300)), len(G)


def float_residual(par, n, u):
    """Moment residual in floating point (only used to improve the centre, not for any proof)."""
    w0, wr, c0, cr = par.points(u, np.zeros(len(u)))
    d = DIM[par.dom]
    if par.dom in BOX:
        A = np.array(par.idx, dtype=int)
        tabs = [np.array(legendre_float(c0[:, c], n)) for c in range(d)]
        B = np.ones((len(A), len(w0)))
        for c in range(d):
            B = B * tabs[c][A[:, c]]
        ex = np.array([float(2 ** d) if sum(a) == 0 else 0.0 for a in par.idx])
        F = B @ w0 - ex
    else:
        B = pkd_float(par.dom, n, par.idx, c0)
        ex = np.array([float(VOL[par.dom]) if sum(a) == 0 else 0.0 for a in par.idx])
        F = w0 @ B - ex
    if par.cons:
        F = np.concatenate([F, [sum(c * u[k] for k, c in row) - 1.0 for row in par.cons]])
    return F


def refine(par, n, exact, steps=4):
    """Chord-Newton steps (minimum norm) on the centre in floating point; returns the shift."""
    u = par.u0.copy()
    Jm, Jr = jacobian(par, n, u, np.zeros(len(u)))
    fm = float_residual(par, n, u)
    best = float(np.max(np.abs(fm)))
    for _ in range(steps):
        un = u - np.linalg.lstsq(Jm, fm, rcond=None)[0]
        fn = float_residual(par, n, un)
        val = float(np.max(np.abs(fn)))
        if not np.isfinite(val) or val >= best:
            break
        u = un
        fm = fn
        best = val
    shift = float(up(np.max(np.abs(u - par.u0)) * (1 + 1e-6) + 1e-300))
    par.u0 = u
    return shift


# ---------------------------------------------------------------- residual and Jacobian enclosures


def basis_tables(par, n, c0, cr):
    """Per coordinate the enclosures of P_k and dP_k/dx at all points."""
    d = DIM[par.dom]
    return [coord_tables(par.dom, n, c0[:, c], cr[:, c]) for c in range(d)]


def basis_enclosure(par, n, c0, cr, want_grad):
    """Legendre products on boxes (and for the stored-number check), the PKD basis on simplices.

    The enclosure of every point (column) depends only on that point, so the points are processed in
    blocks of POINT_BLOCK and the columns are joined afterwards. The results are identical; only the
    temporary arrays, which are many times larger than the result, exist for one block at a time."""
    npts = c0.shape[0]
    if npts <= POINT_BLOCK:
        return basis_enclosure_block(par, n, c0, cr, want_grad)
    vlo = []
    vhi = []
    glo = None
    ghi = None
    for s in range(0, npts, POINT_BLOCK):
        (blo, bhi), bg = basis_enclosure_block(par, n, c0[s:s + POINT_BLOCK], cr[s:s + POINT_BLOCK], want_grad)
        vlo.append(blo)
        vhi.append(bhi)
        if glo is None:
            glo = [[] for _ in bg]
            ghi = [[] for _ in bg]
        for c, (a, b) in enumerate(bg):
            glo[c].append(a)
            ghi[c].append(b)
        del blo, bhi, bg
    val = (np.concatenate(vlo, axis=1), np.concatenate(vhi, axis=1))
    grads = [(np.concatenate(glo[c], axis=1), np.concatenate(ghi[c], axis=1)) for c in range(len(glo))]
    return val, grads


def basis_enclosure_block(par, n, c0, cr, want_grad):
    """Basis enclosure for one block of points."""
    if par.dom not in BOX and par.mode != "stored":
        return pkd_enclosure(par, n, c0, cr, want_grad)
    return phi_enclosure(par, basis_tables(par, n, c0, cr), want_grad)


def phi_enclosure(par, tabs, want_grad):
    """Enclosures of the basis products (rows) at all points (columns), and of their gradients."""
    d = DIM[par.dom]
    A = np.array(par.idx, dtype=int)
    flo = np.ones((len(A), tabs[0][0].shape[1]))
    fhi = np.ones_like(flo)
    for c in range(d):
        flo, fhi = iv_mul(flo, fhi, tabs[c][0][A[:, c]], tabs[c][1][A[:, c]])
    grads = []
    if want_grad:
        for c in range(d):
            glo = np.ones_like(flo)
            ghi = np.ones_like(flo)
            for e in range(d):
                if e == c:
                    glo, ghi = iv_mul(glo, ghi, tabs[e][2][A[:, e]], tabs[e][3][A[:, e]])
                else:
                    glo, ghi = iv_mul(glo, ghi, tabs[e][0][A[:, e]], tabs[e][1][A[:, e]])
            grads.append((glo, ghi))
    return (flo, fhi), grads


def residual(par, n, u, ru, exact):
    """Enclosure (mid, rad) of F(u') = sum_p w_p phi(x_p) - integral, for all u' in u +- ru."""
    w0, wr, c0, cr = par.points(u, ru)
    (flo, fhi), _ = basis_enclosure(par, n, c0, cr, False)
    wlo = dn(w0 - wr)
    whi = up(w0 + wr)
    plo, phi = iv_mul(flo, fhi, wlo[None, :], whi[None, :])
    # math.fsum is correctly rounded, so one outward step encloses the exact sum of the bounds
    slo = dn(np.array([math.fsum(row) for row in plo]))
    shi = up(np.array([math.fsum(row) for row in phi]))
    s, rad = to_midrad(slo, shi)
    if par.dom not in BOX and par.mode != "stored":
        exact = [VOL[par.dom] if sum(a) == 0 else Fraction(0) for a in par.idx]
    elo = np.array([frac_lo(q) for q in exact])
    ehi = np.array([frac_hi(q) for q in exact])
    em, er = to_midrad(elo, ehi)
    fm = s - em
    fr = up(rad + er + up(np.abs(fm) * 2 * U) + TINY)
    if par.cons:
        cm = []
        cr = []
        for row in par.cons:
            q = sum(c * Fraction(float(u[k])) for k, c in row) - 1
            lo = frac_lo(q)
            hi = frac_hi(q)
            m0, r0 = to_midrad(np.array([lo]), np.array([hi]))
            rr = sum(abs(c) * float(ru[k]) for k, c in row)
            cm.append(float(m0[0]))
            cr.append(float(up(up(r0[0] + rr) * (1 + 4 * U))))
        fm = np.concatenate([fm, cm])
        fr = np.concatenate([fr, cr])
    return fm, fr


def jacobian(par, n, u, ru):
    """Enclosure (mid, rad) of dF/du over the box u +- ru: shape (equations, unknowns)."""
    d = DIM[par.dom]
    w0, wr, c0, cr = par.points(u, ru)
    (flo, fhi), grads = basis_enclosure(par, n, c0, cr, True)
    q = len(u)
    M = len(par.idx)
    Jm = np.zeros((M, q))
    Jabs = np.zeros((M, q))
    Jr = np.zeros((M, q))
    cnt = np.zeros(q)
    m, r = to_midrad(flo, fhi)
    np.add.at(Jm.T, par.widx, m.T)
    np.add.at(Jabs.T, par.widx, np.abs(m).T)
    np.add.at(Jr.T, par.widx, r.T)
    np.add.at(cnt, par.widx, 1)
    wlo = dn(w0 - wr)
    whi = up(w0 + wr)
    for c in range(d):
        glo, ghi = grads[c]
        for t in range(par.kidx.shape[2]):
            free = par.kidx[:, c, t] >= 0
            if not np.any(free):
                continue
            sg = par.coef[:, c, t]
            tlo, thi = iv_mul(glo, ghi, (sg * wlo)[None, :], (sg * whi)[None, :])
            m, r = to_midrad(tlo[:, free], thi[:, free])
            cols = par.kidx[free, c, t]
            np.add.at(Jm.T, cols, m.T)
            np.add.at(Jabs.T, cols, np.abs(m).T)
            np.add.at(Jr.T, cols, r.T)
            np.add.at(cnt, cols, 1)
    k = max(1.0, float(cnt.max()))
    Jr = up(Jr * (1 + gamma(k + 2)) + gamma(k) * Jabs * (1 + gamma(k + 2)) + k * TINY)
    if par.cons:
        Cm = np.zeros((len(par.cons), q))
        for i, row in enumerate(par.cons):
            for kk, c in row:
                Cm[i, kk] += c
        Jm = np.vstack([Jm, Cm])
        Jr = np.vstack([Jr, np.zeros_like(Cm)])
    return Jm, Jr


# ---------------------------------------------------------------- Krawczyk test


def greedy_pivots(B, k):
    """k column indices of B chosen greedily (Gram-Schmidt with column pivoting)."""
    R = B.copy()
    chosen = []
    for _ in range(k):
        nr = np.einsum("ij,ij->j", R, R)
        nr[chosen] = -1
        j = int(np.argmax(nr))
        chosen.append(j)
        qv = R[:, j] / max(math.sqrt(max(nr[j], 0.0)), 1e-300)
        R -= np.outer(qv, qv @ R)
    return chosen


def pivot_columns(J, M):
    """M columns of the M x q matrix J that form a well-conditioned square matrix.

    The q - M columns to leave out are chosen by pivoting on the rows of a null-space basis
    (a heuristic; the proof itself does not depend on the choice)."""
    q = J.shape[1]
    Js = J / np.maximum(np.linalg.norm(J, axis=0), 1e-300)
    Q = np.linalg.qr(Js.T, mode="complete")[0]
    N = Q[:, M:]
    drop = set(greedy_pivots(N.T, q - M))
    return np.array([j for j in range(q) if j not in drop])


def krawczyk(par, n, exact, tries=(1.5, 3.0, 10.0, 1e2, 1e4)):
    """Try to prove a zero of F near u0. Returns a dict with the result."""
    u0 = par.u0
    q = len(u0)
    M = len(par.idx) + len(par.cons)
    if q < M:
        return dict(proved=False, reason="more equations (%d) than unknowns (%d)" % (M, q))
    zero = np.zeros(q)
    Fm, Fr = residual(par, n, u0, zero, exact)
    J0m, J0r = jacobian(par, n, u0, zero)
    # Scale the unknowns so that all Jacobian columns have comparable size: the Krawczyk box is a cube
    # in the scaled variables. A node with a tiny weight then gets a correspondingly wider box for its
    # coordinates, which hardly influence the moments. Powers of two keep the scaling exact.
    cn = np.linalg.norm(J0m, axis=0)
    ref = float(np.median(cn[cn > 0])) if np.any(cn > 0) else 1.0
    sc = np.where(cn > 0, ref / np.maximum(cn, 1e-300), 1.0)
    sc = np.exp2(np.round(np.log2(np.clip(sc, 2.0 ** -20, 2.0 ** 40))))
    if q > M:
        # keep q - M unknowns fixed at their values: choose M well-conditioned columns (greedy pivoting)
        sel = pivot_columns(J0m, M)
    else:
        sel = np.arange(q)
    # z -> u = u0 + V z with V[sel[k], k] = vs[k]; the products below are exact (powers of two)
    vs = sc[sel]
    A0 = J0m[:, sel] * vs
    try:
        C = np.linalg.inv(A0)
    except np.linalg.LinAlgError:
        return dict(proved=False, reason="singular Jacobian")
    if not np.all(np.isfinite(C)):
        return dict(proved=False, reason="singular Jacobian")
    # rounding errors of C A alone are about gamma(M) |C| |A|; if that is not small the test cannot succeed
    cond = float(np.max(np.abs(C) @ (np.abs(A0) @ np.ones(A0.shape[1]))))
    if not np.isfinite(cond) or gamma(M) * cond > 0.25:
        return dict(proved=False, reason="Jacobian too ill-conditioned in this basis (cond ~ %.1e)" % cond, cond=cond)
    absC = np.abs(C)
    CGm, CGr = mm(C, Fm)
    CGr = up(CGr + up((absC @ Fr) * (1 + gamma(M + 2)) + M * TINY))
    # size of the Newton step including its enclosure error; the box must be larger than that
    delta = float(np.max(np.abs(CGm) + CGr))
    last = None
    for f in tries:
        r = max(f * delta, 1e-300)
        rz = np.full(M, r)
        ru = np.zeros(q)
        ru[sel] = vs * rz
        Jm, Jr = jacobian(par, n, u0, ru)
        Am = Jm[:, sel] * vs
        Ar = Jr[:, sel] * vs
        CAm, CAr = mm(C, Am)
        CAr = up(CAr + up((absC @ Ar) * (1 + gamma(M + 2)) + M * TINY))
        Bm = np.eye(M) - CAm
        Br = up(CAr + up(np.abs(Bm) * 2 * U) + TINY)
        row = up(((np.abs(Bm) + Br) @ rz) * (1 + gamma(M + 2)) + M * TINY)
        K = up(up(np.abs(CGm) + CGr) + row)
        last = float(np.max(K) / r)
        if np.all(K < r):
            return dict(proved=True, r=r, ru=ru, kratio=last)
    return dict(proved=False, reason="Krawczyk test not satisfied (max |K|/r = %.3g)" % last, kratio=last)


def quality_over_box(par, u0, ru):
    """Rigorous quality of all rules in the box u0 +- ru: (positive weights, all nodes inside, some node outside)."""
    w0, wr, c0, cr = par.points(u0, ru)
    pos = bool(np.all(dn(w0 - wr) > 0))
    # fixed coordinates (radius 0) are exact constants and must not be widened
    lo = np.where(cr > 0, dn(c0 - cr), c0)
    hi = np.where(cr > 0, up(c0 + cr), c0)
    outside = False
    if par.dom in BOX:
        outside = bool(np.any(lo > 1.0) or np.any(hi < -1.0))
    if par.bary is not None:
        b = np.array(par.bary, dtype=int)
        inside = bool(np.all(dn(u0[b] - ru[b]) >= 0.0)) if len(b) else True
        outside = bool(np.any(up(u0[b] + ru[b]) < 0.0)) if len(b) else False
    elif par.dom in BOX:
        inside = bool(np.all(lo >= -1.0) and np.all(hi <= 1.0))
    else:
        inside = True
        ulo = dn(u0 - ru)
        uhi = up(u0 + ru)
        for forms in par.bary_forms():
            for f, cst in forms:
                if not f:
                    ok = cst >= 0
                else:
                    v = cst
                    for k, cf in f.items():
                        v = float(dn(v + (cf * ulo[k] if cf > 0 else cf * uhi[k])))
                        v = float(dn(v))
                    ok = v >= 0
                if not ok:
                    inside = False
                if f or cst < 0:
                    v = cst
                    for k, cf in f.items():
                        v = float(up(v + (cf * uhi[k] if cf > 0 else cf * ulo[k])))
                    if v < 0:
                        outside = True
    return pos, inside, outside


# ---------------------------------------------------------------- part 1: stored numbers


def stored_check(dom, n, Xq, Wq):
    """Exact node/weight checks and a rigorous residual bound for the stored decimal numbers."""
    d = DIM[dom]
    if dom in BOX:
        marg = min(1 - abs(v) for p in Xq for v in p)
    else:
        marg = min(min(min(p), 1 - sum(p)) for p in Xq)
    minw = min(Wq)
    X = np.array([[float(v) for v in p] for p in Xq])
    W = np.array([float(v) for v in Wq])
    Xr = np.array([[float(up(frac_hi(abs(Fraction(float(v)) - v)))) for v in p] for p in Xq])
    Wr = np.array([float(up(frac_hi(abs(Fraction(float(v)) - v)))) for v in Wq])
    u = []
    ru = []
    widx = []
    cidx = []
    for p in range(len(W)):
        widx.append(len(u))
        u.append(W[p])
        ru.append(Wr[p])
        ci = []
        for c in range(d):
            ci.append(len(u))
            u.append(X[p][c])
            ru.append(Xr[p][c])
        cidx.append(ci)
    par = Param(dom, "stored", u, widx, cidx, np.ones((len(W), d)), np.zeros((len(W), d)), full_indices(d, n))
    exact = exact_integrals(dom, par.idx)
    fm, fr = residual(par, n, par.u0, np.array(ru), exact)
    bound = float(up(np.max(up(np.abs(fm) + fr)) / float(VOL[dom])))
    q = ("P" if minw > 0 else "N") + ("I" if marg >= 0 else "O")
    return dict(res_bound=bound, res_ok=bool(bound <= RES_TOL), margin_exact=float(marg), minw_exact=float(minw), quality_exact=q)


# ---------------------------------------------------------------- driver


def snap(dom, X, eps=1e-12):
    """Move coordinates within eps of the boundary onto it; returns the new nodes and the largest move."""
    out = []
    moved = 0.0
    for p in X:
        q = []
        for v in p:
            w = v
            if dom in BOX and 1.0 - abs(v) < eps:
                w = math.copysign(1.0, v)
            if dom not in BOX and abs(v) < eps:
                w = 0.0
            moved = max(moved, abs(w - v))
            q.append(w)
        if dom not in BOX and abs(1.0 - sum(q)) < eps:
            # placed on the face sum x = 1 by full_param (largest coordinate = 1 - the others)
            moved = max(moved, float(up(abs(1.0 - sum(q)) * 2 + 4 * U)))
        out.append(q)
    return out, moved


def prove(dom, n, X, W, Xq, zero_tol):
    """Existence proof: symmetry-reduced if the rule has symmetries, otherwise in all unknowns."""
    cands = []
    if dom in BOX:
        sp = sym_param(dom, n, X, W)
    else:
        sp = simplex_sym_param(dom, n, X, W, Xq, zero_tol=zero_tol)
    if sp is not None:
        cands.append(sp)
    else:
        fp = full_param(dom, n, X, W, Xq, zero_tol)
        w0, wr, c0, cr = fp.points(fp.u0, np.zeros(len(fp.u0)))
        dev = float(up(np.max(np.abs(c0 - np.array(X))) * (1 + 1e-6) + 1e-300))
        cands.append((fp, dev, 1))
    res = None
    tried = []
    for par, dev, gsize in cands:
        mode = par.mode if gsize == 1 else "sym%d" % gsize
        q = len(par.u0)
        M = len(par.idx) + len(par.cons)
        if q < M:
            res = dict(proved=False, mode=mode, equations=M, unknowns=q, reason="more equations (%d) than unknowns (%d)" % (M, q))
            tried.append(res)
            continue
        if q * M > 6e7:
            res = dict(proved=False, mode=mode, equations=M, unknowns=q, reason="system too large (%d x %d)" % (M, q))
            tried.append(res)
            continue
        exact = exact_integrals(dom, par.idx)
        shift = refine(par, n, exact)
        k = krawczyk(par, n, exact)
        k["mode"] = mode
        k["equations"] = M
        k["unknowns"] = q
        if k["proved"]:
            pos, ins, outs = quality_over_box(par, par.u0, k["ru"])
            k["dist"] = float(up(float(np.max(k["ru"])) + dev + shift))
            k["pos"] = pos
            k["inside"] = ins
            k["outside"] = outs
            k["quality_proved"] = ("P" if pos else "?") + ("I" if ins else ("O" if outs else "?"))
            del k["ru"]
            res = k
            break
        res = k
        tried.append(k)
    return res


def check_rule(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    r = json.loads(raw.decode("utf8"), parse_float=Fraction)
    dom = r["domain"]
    n = int(r["degree"])
    Xq = [[Fraction(v) for v in p] for p in r["nodes"]]
    Wq = [Fraction(v) for v in r["weights"]]
    t0 = time.time()
    out = dict(id=r["id"], file=os.path.relpath(path, ROOT).replace(os.sep, "/"), domain=dom, degree=n, points=len(Wq), quality=r.get("quality"), status=r.get("status"), sha=content_sha(raw.decode("utf8")))
    out["stored"] = stored_check(dom, n, Xq, Wq)
    X = [[float(v) for v in p] for p in Xq]
    W = [float(v) for v in Wq]
    res = prove(dom, n, X, W, Xq, 0.0)
    q = r.get("quality") or ""
    if res.get("proved") and q[1:] == "I" and not res.get("inside"):
        # nodes within 1e-12 of the boundary: retry with them placed exactly on the boundary
        Xs, moved = snap(dom, X)
        if moved > 0:
            Xsq = [[Fraction(v) for v in p] for p in Xs]
            res2 = prove(dom, n, Xs, W, Xsq, 1e-12)
            if res2.get("proved") and res2.get("inside"):
                res2["dist"] = float(up(res2["dist"] + moved))
                res2["snapped"] = moved
                res = res2
    out["exist"] = res
    out["seconds"] = round(time.time() - t0, 2)
    return out


def changed_files(base):
    out = subprocess.run(["git", "diff", "--name-only", "--diff-filter=AM", base, "--", "rules"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [os.path.join(ROOT, f) for f in out.split() if f.endswith(".json") and not f.endswith("family.json")]


def verdict(res):
    """Part 1 passes if the residual bound holds and the exact quality agrees; part 2 if the stored quality is proven."""
    st = res["stored"]
    q = res["quality"] or ""
    ok1 = st["res_ok"] and (st["quality_exact"] == q or (q == "PI" and st["quality_exact"] == "PO" and st["margin_exact"] >= -1e-12))
    ex = res["exist"]
    ok2 = bool(ex.get("proved")) and (q[:1] != "P" or ex.get("pos")) and (q[1:] != "I" or ex.get("inside")) and (q[1:] != "O" or ex.get("outside"))
    return ok1, ok2


def content_sha(text):
    """Hash of what a proof depends on: domain, degree, quality and the exact decimal strings of nodes and
    weights. Other metadata (names, status, numerical checks) can change without invalidating a proof."""
    r = json.loads(text, parse_float=str, parse_int=str)
    core = {k: r.get(k) for k in ("domain", "degree", "quality", "nodes", "weights")}
    blob = json.dumps(core, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf8")).hexdigest()[:16]


def missing_files(path):
    """Rule files whose numbers have no successful proof in the report."""
    proved = set()
    if os.path.exists(path):
        with open(path) as fh:
            for e in json.load(fh).get("rules", []):
                if e.get("ok_stored") and e.get("ok_exist") and e.get("exist", {}).get("quality_proved") == e.get("quality"):
                    proved.add((e["id"], e.get("sha")))
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, "rules", "*", "*", "*.json"))):
        with open(f, encoding="utf8") as fh:
            text = fh.read()
        rid = json.loads(text).get("id")
        if (rid, content_sha(text)) not in proved:
            out.append(f)
    return out


def merge_report(path, results):
    """Replace the entries of the checked rules in an existing report; drop entries whose file is gone."""
    old = []
    if os.path.exists(path):
        with open(path) as fh:
            old = json.load(fh).get("rules", [])
    new_ids = {r["id"] for r in results}
    keep = [e for e in old if e["id"] not in new_ids and os.path.exists(os.path.join(ROOT, e["file"]))]
    merged = sorted(keep + results, key=lambda e: e["file"])
    write_report(path, merged)
    return len(keep), len(results)


def write_report(path, results):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(dict(tool="tools/verify_rigorous.py", rules=results), fh, indent=1)
        fh.write("\n")


def report_key(results):
    """What a report says, without run times and rounding noise: per rule file hash and verdicts."""
    out = []
    for r in results:
        ex = r.get("exist", {})
        out.append((r.get("id"), r.get("sha"), r.get("ok_stored"), r.get("ok_exist"), ex.get("quality_proved"), ex.get("mode")))
    return sorted(out, key=lambda t: str(t))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="*", help="rule files (default: all)")
    ap.add_argument("--changed", help="only rule files added or changed against this git revision")
    ap.add_argument("--report", help="write the results as JSON to this file")
    ap.add_argument("--strict", action="store_true", help="also fail if the existence proof does not succeed")
    ap.add_argument("--merge", help="put the results of the checked rules into this report file, keeping the other entries")
    ap.add_argument("--missing", help="only rule files without a successful proof of exactly their numbers in this report file")
    ap.add_argument("--update", help="rewrite this report file, but only if a result changed (new, changed or removed rule files, or a different verdict); run times and last digits of distances do not count")
    ap.add_argument("--summary", help="append a Markdown summary to this file (e.g. $GITHUB_STEP_SUMMARY)")
    args = ap.parse_args()
    if args.files:
        files = [os.path.abspath(f) for f in args.files]
    elif args.changed:
        files = changed_files(args.changed)
    elif args.missing:
        files = missing_files(args.missing)
    else:
        files = sorted(f for f in glob.glob(os.path.join(ROOT, "rules", "*", "*", "*.json")))
    results = []
    bad1 = 0
    bad2 = 0
    hard2 = 0
    if not files:
        print("no rule files to check")
    for f in files:
        res = check_rule(f)
        ok1, ok2 = verdict(res)
        res["ok_stored"] = ok1
        res["ok_exist"] = ok2
        bad1 += not ok1
        bad2 += not ok2
        hard2 += not ok2
        ex = res["exist"]
        if ex.get("proved"):
            info = "%s exact rule within %.1e, proven %s" % (ex["mode"], ex["dist"], ex["quality_proved"])
        else:
            info = "%s not proven: %s" % (ex.get("mode", "-"), ex.get("reason"))
        print("%-34s %s %s  residual <= %.1e  %s  (%.1fs)" % (res["id"], "ok " if ok1 else "BAD", "ok " if ok2 else "-- ", res["stored"]["res_bound"], info, res["seconds"]), flush=True)
        results.append(res)
    print("checked %d rules: stored numbers %d ok, %d bad; existence proven %d, not proven %d" % (len(results), len(results) - bad1, bad1, len(results) - bad2, bad2))
    if args.report:
        write_report(args.report, results)
    if args.update:
        old = None
        if os.path.exists(args.update):
            with open(args.update) as fh:
                old = json.load(fh).get("rules", [])
        if old is None or report_key(old) != report_key(results):
            write_report(args.update, results)
            print("report %s updated" % args.update)
        else:
            print("report %s unchanged" % args.update)
    if args.merge:
        kept, added = merge_report(args.merge, results)
        print("report %s: %d entries kept, %d checked rules written" % (args.merge, kept, added))
    if args.summary:
        with open(args.summary, "a") as fh:
            fh.write("### Interval proofs (tools/verify_rigorous.py)\n\n")
            fh.write("%d rules: stored numbers %d ok, %d bad; existence of an exact rule proven for %d, not proven for %d.\n\n" % (len(results), len(results) - bad1, bad1, len(results) - bad2, bad2))
            fh.write("| rule | stored numbers | existence | detail |\n|---|---|---|---|\n")
            for res in results:
                ex = res["exist"]
                det = ("%s, within %.1e, %s" % (ex["mode"], ex["dist"], ex["quality_proved"])) if ex.get("proved") else (ex.get("reason") or "")
                fh.write("| %s | %s | %s | %s |\n" % (res["id"], "ok" if res["ok_stored"] else "**bad**", "proven" if res["ok_exist"] else "not proven", det))
    if bad1 or (args.strict and hard2):
        sys.exit(1)


if __name__ == "__main__":
    main()

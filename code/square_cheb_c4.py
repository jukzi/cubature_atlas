#!/usr/bin/env python3
"""Deterministic C4-symmetric cubature search on the square [-1, 1]^2 for odd degrees.

Constructs starting nodes in Chebyshev equilibrium angle coordinates [0, pi]^2 by
taking a regular square lattice (Z^2 with a centre node or (Z + 1/2)^2 without),
rotating it by a deterministic chiral angle near 45 degrees to break D4 -> C4
(avoiding the O(n) Burnside penalty of mirror reflections while keeping central
inversion -I so all odd-degree moments vanish identically), selecting the first
k orbits in L_infinity order, scaling them to boundary offset r0_fac * (2*pi/n),
mapping back via x = sin(u), y = sin(v), and initializing weights proportional to
the equilibrium Jacobian sqrt((1 - x^2)(1 - y^2)).

Solves the exact C4-invariant Legendre moment system with Marquardt-scaled
Levenberg-Marquardt (PI by construction: x = sin(sx), y = sin(sy), w = v^2)
using Cholesky factorization, residual-only trial evaluations, and exact analytic
geodesic acceleration (Transtrum & Sethna 2012, arXiv:1201.5885):
- Second derivatives P_k''(x) come directly from the Legendre ODE
  (1 - x^2) P_k''(x) = 2 x P_k'(x) - k(k + 1) P_k(x) with zero extra recurrences
  and no finite-difference cancellation down to 1e-15.
- Reuses the Cholesky factor of H + lam * D for both the first-order velocity v
  and the geodesic acceleration a = -(H + lam * D)^{-1} J^T r''(v).
- Uses monotone Marquardt LM during initial descent (res >= geo_tau) to preserve
  basin selection, then activates geodesic acceleration, delayed gratification
  (lam /= 1.5, lam *= 5.0), and bold uphill acceptance ((1 - cos beta)^b * c_new
  <= c_best with lam kept unchanged on uphill steps) inside the narrow curved
  canyon, with a monotone best-point polish fallback if an uphill excursion stalls.

Usage:
    python code/square_cheb_c4.py --degree 51 --nodes 452 --angle 49 --out rule.json
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
    """Evaluate orthonormal P_k(x), P_k'(x), and exact P_k''(x) via Legendre ODE."""
    P, dP = legendre_eval(n, x)
    # The Legendre ODE (1 - x^2) P_k'' = 2 x P_k' - k(k + 1) P_k is linear and
    # homogeneous in P_k, so it holds identically for the orthonormal scaling.
    inv = 1.0 / np.maximum(1.0 - x * x, 1e-30)
    ks = np.arange(n + 1, dtype=float)[:, None]
    ddP = (2.0 * x[None, :] * dP - (ks * (ks + 1.0)) * P) * inv[None, :]
    return P, dP, ddP


class SquareC4Basis:
    """Orthonormal C4-invariant polynomial basis on [-1, 1]^2 up to odd degree n."""

    def __init__(self, n):
        self.n = n
        self.pairs = []
        for a in range(0, n + 1, 2):
            for b in range(a, n + 1 - a, 2):
                sc = 0.5 if a == b else 1.0 / math.sqrt(2.0)
                self.pairs.append((a, b, 1.0, sc))
        for a in range(1, n + 1, 2):
            for b in range(a + 2, n + 1 - a, 2):
                self.pairs.append((a, b, -1.0, 1.0 / math.sqrt(2.0)))
        self.M = len(self.pairs)
        self.a_idx = np.array([p[0] for p in self.pairs])
        self.b_idx = np.array([p[1] for p in self.pairs])
        self.sgn = np.array([p[2] for p in self.pairs])[:, None]
        self.sc = np.array([p[3] for p in self.pairs])[:, None]
        self.rhs = np.zeros(self.M)
        self.rhs[0] = 2.0

    def eval_values(self, x, y):
        """Return invariant basis values V without derivatives."""
        Px = legendre_values(self.n, x)
        Py = legendre_values(self.n, y)
        a = self.a_idx
        b = self.b_idx
        return self.sc * (Px[a] * Py[b] + self.sgn * Px[b] * Py[a])

    def eval_and_grad(self, x, y):
        """Return invariant basis values V and partial derivatives Vx, Vy."""
        Px, dPx = legendre_eval(self.n, x)
        Py, dPy = legendre_eval(self.n, y)
        a = self.a_idx
        b = self.b_idx
        V = self.sc * (Px[a] * Py[b] + self.sgn * Px[b] * Py[a])
        Vx = self.sc * (dPx[a] * Py[b] + self.sgn * dPx[b] * Py[a])
        Vy = self.sc * (Px[a] * dPy[b] + self.sgn * Px[b] * dPy[a])
        return V, Vx, Vy

    def eval_jet(self, x, y):
        """Return V, Vx, Vy and 1D Legendre jets for exact directional 2nd derivatives."""
        Px, dPx, ddPx = legendre_eval2(self.n, x)
        Py, dPy, ddPy = legendre_eval2(self.n, y)
        a = self.a_idx
        b = self.b_idx
        V = self.sc * (Px[a] * Py[b] + self.sgn * Px[b] * Py[a])
        Vx = self.sc * (dPx[a] * Py[b] + self.sgn * dPx[b] * Py[a])
        Vy = self.sc * (Px[a] * dPy[b] + self.sgn * Px[b] * dPy[a])
        return V, Vx, Vy, Px, dPx, ddPx, Py, dPy, ddPy


def rotated_cheb_start(n, k, has_centre, angle_deg, r0_fac):
    """Deterministic rotated Chebyshev lattice start with k C4-orbits (+ optional centre)."""
    ang = math.radians(angle_deg)
    ca = math.cos(ang)
    sa = math.sin(ang)
    rad = int(math.ceil(math.sqrt(4 * k + 10))) + 4
    offs = 0.0 if has_centre else 0.5
    orbs = []
    for i in range(-rad, rad + 1):
        for j in range(-rad, rad + 1):
            ii = i + offs
            jj = j + offs
            if abs(ii) < 1e-12 and abs(jj) < 1e-12:
                continue
            if ii > 1e-12 and jj >= -1e-12:
                u = ca * ii - sa * jj
                v = sa * ii + ca * jj
                linf = max(abs(u), abs(v))
                l2 = math.hypot(u, v)
                orbs.append((linf, l2, u, v))
    orbs.sort(key=lambda t: (round(t[0], 9), t[1]))
    chosen = orbs[:k]
    us = np.array([t[2] for t in chosen])
    vs = np.array([t[3] for t in chosen])
    umax = max(float(np.abs(us).max()), float(np.abs(vs).max()))
    scale = 2.0 * math.pi / n
    hw = 0.5 * math.pi - r0_fac * scale
    us = us * (hw / umax)
    vs = vs * (hw / umax)
    x = np.sin(us)
    y = np.sin(vs)
    w_raw = np.sqrt(np.maximum(1e-12, (1.0 - x * x) * (1.0 - y * y)))
    wc_raw = 1.0 if has_centre else 0.0
    tot = 4.0 * float(w_raw.sum()) + wc_raw
    w = 4.0 * w_raw / tot
    wc = 4.0 * wc_raw / tot
    return x, y, w, wc


def solve_c4_lm(
    B,
    x0,
    y0,
    w0,
    has_centre,
    wc0,
    steps,
    win,
    lam0,
    geo=True,
    alpha=0.75,
    bold=2,
    geo_tau=1e-3,
    lam_down=1.5,
    lam_up=5.0,
):
    """Marquardt-scaled LM with PI-by-construction map and exact geodesic acceleration."""
    k = len(x0)
    sx0 = np.arcsin(np.clip(x0, -0.999999, 0.999999))
    sy0 = np.arcsin(np.clip(y0, -0.999999, 0.999999))
    v0 = np.sqrt(np.maximum(w0, 1e-14))
    a_idx = B.a_idx
    b_idx = B.b_idx
    sgn = B.sgn
    sc = B.sc
    if has_centre:
        q = np.concatenate([sx0, sy0, v0, [math.sqrt(max(wc0, 1e-14))]])
        V0 = B.eval_values(np.array([0.0]), np.array([0.0]))
        v_origin = V0[:, 0]
    else:
        q = np.concatenate([sx0, sy0, v0])
        v_origin = None

    def eval_res(vec):
        sx = vec[:k]
        sy = vec[k:2 * k]
        v = vec[2 * k:3 * k]
        x = np.sin(sx)
        y = np.sin(sy)
        w = v * v
        V = B.eval_values(x, y)
        r = 4.0 * (V @ w) - B.rhs
        if has_centre:
            vc = vec[3 * k]
            r = r + v_origin * (vc * vc)
        return r

    def eval_full(vec, need_jet):
        sx = vec[:k]
        sy = vec[k:2 * k]
        v = vec[2 * k:3 * k]
        x = np.sin(sx)
        y = np.sin(sy)
        cx = np.cos(sx)
        cy = np.cos(sy)
        w = v * v
        if need_jet:
            V, Vx, Vy, Px, dPx, ddPx, Py, dPy, ddPy = B.eval_jet(x, y)
            state = (x, y, cx, cy, v, w, Px, dPx, ddPx, Py, dPy, ddPy, V, Vx, Vy)
        else:
            V, Vx, Vy = B.eval_and_grad(x, y)
            state = None
        r = 4.0 * (V @ w) - B.rhs
        Jsx = Vx * (4.0 * w * cx)[None, :]
        Jsy = Vy * (4.0 * w * cy)[None, :]
        Jv = V * (8.0 * v)[None, :]
        if has_centre:
            vc = vec[3 * k]
            r = r + v_origin * (vc * vc)
            Jvc = (v_origin * (2.0 * vc))[:, None]
            J = np.hstack([Jsx, Jsy, Jv, Jvc])
        else:
            J = np.hstack([Jsx, Jsy, Jv])
        return r, J, state

    def d2r(state, vel):
        """Exact directional second derivative r''(vel) in (sin sx, sin sy, v^2) coords."""
        x, y, cx, cy, v, w, Px, dPx, ddPx, Py, dPy, ddPy, V, Vx, Vy = state
        dsx = vel[:k]
        dsy = vel[k:2 * k]
        dv = vel[2 * k:3 * k]
        Vxx = sc * (ddPx[a_idx] * Py[b_idx] + sgn * ddPx[b_idx] * Py[a_idx])
        Vxy = sc * (dPx[a_idx] * dPy[b_idx] + sgn * dPx[b_idx] * dPy[a_idx])
        Vyy = sc * (Px[a_idx] * ddPy[b_idx] + sgn * Px[b_idx] * ddPy[a_idx])
        # Chain rule for r = 4 * sum_i v_i^2 * V(sin sx_i, sin sy_i) + vc^2 * v_origin - rhs:
        # d/dsx(sin sx) = cos sx, d^2/dsx^2(sin sx) = -sin sx.
        cV = 8.0 * (dv * dv)
        cVx = 16.0 * v * dv * cx * dsx - 4.0 * w * x * (dsx * dsx)
        cVy = 16.0 * v * dv * cy * dsy - 4.0 * w * y * (dsy * dsy)
        cVxx = 4.0 * w * (cx * dsx) ** 2
        cVxy = 8.0 * w * (cx * dsx) * (cy * dsy)
        cVyy = 4.0 * w * (cy * dsy) ** 2
        out = (V @ cV) + (Vx @ cVx) + (Vy @ cVy) + (Vxx @ cVxx) + (Vxy @ cVxy) + (Vyy @ cVyy)
        if has_centre:
            dvc = vel[3 * k]
            out = out + v_origin * (2.0 * dvc * dvc)
        return out

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
                    # Keep damping unchanged on accepted uphill steps, and use delayed
                    # gratification once inside the geodesic canyon phase.
                    factor = lam_down if geo_active else (1.0 / 0.3)
                    lam = max(lam / factor, 1e-15)
                accepted = True
                break
            lam *= lam_up
        hist.append(best_cost)
        if math.sqrt(best_cost) < 1e-14:
            break
        stagnated = not accepted
        if not stagnated and win > 0 and len(hist) > win:
            if math.sqrt(best_cost) > 0.995 * math.sqrt(hist[-1 - win]):
                stagnated = True
        if stagnated:
            if geo_active and not restarted and cost > 1.01 * best_cost:
                # If an uphill excursion stalled, snap back to best_q and polish monotonically.
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
    x = np.sin(q[:k])
    y = np.sin(q[k:2 * k])
    w = q[2 * k:3 * k] ** 2
    wc = float(q[3 * k] ** 2) if has_centre else 0.0
    return r_init, math.sqrt(best_cost), used, x, y, w, wc


def expand_c4(x, y, w, has_centre, wc):
    """Expand C4 orbit representatives to full node and weight arrays."""
    nodes = []
    weights = []
    for xi, yi, wi in zip(x, y, w):
        nodes.append([float(xi), float(yi)])
        nodes.append([float(-yi), float(xi)])
        nodes.append([float(-xi), float(-yi)])
        nodes.append([float(yi), float(-xi)])
        weights.extend([float(wi)] * 4)
    if has_centre:
        nodes.append([0.0, 0.0])
        weights.append(float(wc))
    return np.array(nodes), np.array(weights)


def monomial_residual(n, nodes, weights):
    """Maximum |sum w x^a y^b - int x^a y^b| / 4 over all a + b <= n."""
    x = nodes[:, 0]
    y = nodes[:, 1]
    xp = np.ones((n + 1, len(x)))
    yp = np.ones((n + 1, len(y)))
    for k in range(1, n + 1):
        xp[k] = xp[k - 1] * x
        yp[k] = yp[k - 1] * y
    max_err = 0.0
    for a in range(n + 1):
        for b in range(n + 1 - a):
            val = float(np.dot(weights, xp[a] * yp[b]))
            if a % 2 == 0 and b % 2 == 0:
                exact = 4.0 / ((a + 1) * (b + 1))
            else:
                exact = 0.0
            err = abs(val - exact) / 4.0
            if err > max_err:
                max_err = err
    return max_err


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--degree", type=int, required=True, help="odd polynomial degree n")
    ap.add_argument("--nodes", type=int, required=True, help="node count N (0 or 1 mod 4)")
    ap.add_argument("--angle", type=float, default=49.0, help="lattice rotation angle in degrees")
    ap.add_argument("--r0", type=float, default=0.60, help="boundary offset in units of 2*pi/n")
    ap.add_argument("--steps", type=int, default=1500, help="maximum LM iterations")
    ap.add_argument("--win", type=int, default=35, help="stagnation window (0 to disable)")
    ap.add_argument("--no-geo", action="store_true", help="disable geodesic acceleration and bold uphill steps")
    ap.add_argument("--alpha", type=float, default=0.75, help="geodesic acceleration ratio cap |a|/|v|")
    ap.add_argument("--bold", type=int, default=2, help="Transtrum uphill boldness exponent (0 to disable)")
    ap.add_argument("--geo-tau", type=float, default=1e-3, help="residual threshold to activate geodesic phase (0 for immediate)")
    ap.add_argument("--out", required=True, help="output atlas JSON path")
    a = ap.parse_args()
    if a.degree % 2 == 0:
        raise ValueError("degree must be odd for C4-symmetric rules")
    if a.nodes % 4 not in (0, 1):
        raise ValueError("nodes must be 0 or 1 mod 4 for C4 symmetry")
    has_centre = (a.nodes % 4 == 1)
    k = a.nodes // 4
    use_geo = not a.no_geo
    t0 = time.time()
    B = SquareC4Basis(a.degree)
    x0, y0, w0, wc0 = rotated_cheb_start(a.degree, k, has_centre, a.angle, a.r0)
    r0, res, used, x, y, w, wc = solve_c4_lm(
        B,
        x0,
        y0,
        w0,
        has_centre,
        wc0,
        a.steps,
        a.win,
        1e-2,
        geo=use_geo,
        alpha=a.alpha,
        bold=a.bold,
        geo_tau=a.geo_tau,
    )
    nodes, weights = expand_c4(x, y, w, has_centre, wc)
    merr = monomial_residual(a.degree, nodes, weights)
    min_w = float(weights.min())
    bdist = float(min(1.0 - np.abs(nodes[:, 0]).max(), 1.0 - np.abs(nodes[:, 1]).max()))
    elapsed = time.time() - t0
    print(f"square deg={a.degree} N={a.nodes} ang={a.angle} r0={a.r0}: "
          f"init={r0:.3e} -> res={res:.2e} monom={merr:.2e} min_w={min_w:.3e} bdist={bdist:.3e} "
          f"it={used} ({elapsed:.1f}s)", flush=True)
    if res >= 1e-12 or merr >= 1e-11 or min_w <= 0.0 or bdist < -1e-12:
        print("no solution found", flush=True)
        sys.exit(1)
    rule_id = f"CA-C2-{a.degree:02d}-{a.nodes}-PI-MPC"
    args = [
        "--degree", str(a.degree),
        "--nodes", str(a.nodes),
        "--angle", f"{a.angle:.17g}",
        "--r0", f"{a.r0:.17g}",
        "--steps", str(a.steps),
        "--win", str(a.win),
    ]
    if a.no_geo:
        args.append("--no-geo")
    else:
        if a.alpha != 0.75:
            args.extend(["--alpha", f"{a.alpha:.17g}"])
        if a.bold != 2:
            args.extend(["--bold", str(a.bold)])
        if a.geo_tau != 1e-3:
            args.extend(["--geo-tau", f"{a.geo_tau:.17g}"])
    args.extend(["--out", a.out])
    cmd = "python code/square_cheb_c4.py " + shlex.join(args)
    solver_desc = (
        "geodesic-accelerated Marquardt-scaled Levenberg-Marquardt"
        if use_geo
        else "Marquardt-scaled Levenberg-Marquardt"
    )
    record = {
        "id": rule_id,
        "family": "minimal_pointcount_for_degree_n_PI",
        "domain": "square",
        "dimension": 2,
        "degree": a.degree,
        "points": a.nodes,
        "quality": "PI",
        "origin": "found",
        "method": (
            f"Deterministic C4-symmetric rotated Chebyshev equilibrium lattice start "
            f"(angle {a.angle:.17g} deg, boundary offset {a.r0:.17g} * 2*pi/n, equilibrium weights) "
            f"solved directly with {solver_desc} (no randomness)"
        ),
        "author": "Antigravity (AI-assisted)",
        "reproduce": {
            "script": "code/square_cheb_c4.py",
            "seed": 0,
            "command": cmd,
        },
        "status": "proven",
        "residual_reported": merr,
        "nodes": [[float(p[0]), float(p[1])] for p in nodes],
        "weights": [float(wt) for wt in weights],
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
        fh.write("\n")
    print(f"solution found: wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()

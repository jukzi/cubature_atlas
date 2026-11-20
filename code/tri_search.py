#!/usr/bin/env python3
"""Search for triangle cubature rules of degree n with few nodes N (PI: weights > 0,
nodes in the closed triangle (0,0),(1,0),(0,1)).

Usage examples
    python code/tri_search.py --degree 20 --nodes 77 --mode random --seed 1 --tries 50 --out /tmp/t20.jsonl
    python code/tri_search.py --degree 20 --nodes 77 --mode po2pi --start rules/.../CA-T2-20-77-PO-MPC.json --seed 1 --out ...
    python code/tri_search.py --degree 14 --nodes 40 --mode elim --start rules/.../CA-T2-14-41-PI-MPC.json --seed 1 --out ...

Every solution (residual below --accept) is appended to --out as one JSON line with
degree, N, seed, mode, command, nodes, weights.  The result is a floating point
solution; it is NOT a proof (see tools/verify_rigorous.py).

Design decisions (each one explains an idea of this program; "IDEA n" is referenced
in the iteration log at the bottom of this docstring section):

IDEA 1  Orthonormal (Dubiner/PKD) moment equations instead of monomials.
        Monomial moments are exponentially ill-conditioned for degree > 12; with an
        orthogonal basis the residual vector is well scaled, the Jacobian has
        condition number ~ that of the geometry only, and Levenberg-Marquardt/TRF
        converges quadratically near a solution.  Equations: sum_i w_i phi_k(x_i)
        = integral of phi_k = area*delta_k0, with phi_k scaled to unit L2 norm.
IDEA 2  Duffy parametrisation x = s, y = (1-s) t with s,t in [0,1] and w >= 0 as box
        bounds.  The PI quality (nodes in the closed triangle, positive weights)
        becomes a pure box constraint, so a bounded trust-region least squares
        (scipy 'trf') enforces it exactly, no penalties.  Nodes can sit on the
        boundary (t=0, t=1, s=0) which is where optimal rules like to put nodes.
IDEA 3  Analytic Jacobian.  The basis is evaluated with its x- and y-derivatives
        by differentiating the three-term recurrences, so one Jacobian costs about
        as much as one residual; finite differences would cost 3N residuals.
IDEA 4  Degree continuation (homotopy in the degree).  A rule with N nodes for
        degree n-1 is a 3N-dimensional manifold-ish set when 3N > M(n-1); we solve
        degree n-2, n-1, n successively, each time starting from the previous
        solution.  It lets the random start find the "big basin" of the structure
        (boundary layer with dense nodes) with a cheap system first.
IDEA 5  Good start distributions.  Optimal rules put more nodes near the boundary
        (Gauss-Lobatto-like density).  Starts draw s,t from a Chebyshev-type
        arcsine law mixed with a uniform one; weights start at area/N.  Uniform
        random starts converge far more rarely at N close to the counting value.
IDEA 6  Node elimination (mode elim).  From a rule with N+1 nodes remove one node
        (candidates ordered by smallest weight, or random), distribute its weight
        to the neighbours and re-solve.  This is how rules near the counting value
        are usually found, since the N+1 rule lies in the right basin.
IDEA 8  C3 (rotation) symmetric search (mode c3): a third of the unknowns.
IDEA 9  Basin hopping on the residual norm (mode basin): relocate 1-3 nodes, re-solve.
IDEA 10 Own projected Levenberg-Marquardt (lm_bounded) instead of scipy trf: more than
        10x faster time to solution (16/52 by basin hopping: 4 of 4 seeds in 6-16 s,
        before 2 of 4 in 240 s).  The single largest gain of this program.
IDEA 11 Shell start + noise (mode shell): nested rings of Chebyshev-Lobatto points,
        noise proportional to the bubble function 27 b0 b1 b2, short runs.
        Reproduces the atlas counts of the cells 10/24, 11/27, 16/52, 20/78, 22/93
        with a C3 symmetric rule in minutes (random starts never did).
IDEA 12 Shell start for N = 2 mod 3 (mode shellasym): C3 pattern with N-1 or N-2
        nodes plus 1-2 free nodes, unsymmetric solve (reproduces 11/27 unsymmetric).
IDEA 13 Unsymmetric polish of C3 near-solutions (--asym_polish): no gain so far.
IDEA 16 Mirror (x <-> y) symmetric search (mode mirror): N = 2a + b, half the
        unknowns, possible for N = 2 mod 3 where C3 is not.
IDEA 17 Mirror shell start (mode mirrorshell): S3 symmetric rings of a C3 pattern with
        N - 1 or N - 2 nodes, paired by the mirror x <-> y, plus 1-2 extra nodes, then
        the mirror symmetric LM; for N = 2 mod 3 at degrees where random mirror
        starts are hopeless.
IDEA 18 Patterns near known successes (--near, shell_pattern_near): successful ring
        patterns of neighbouring degrees are similar; sampling near them focuses the
        search.  See the measurements in SEARCH_PROMPT.md.
IDEA 14 Variable projection (--vp 1, solve_vp): weights eliminated by linear least
        squares.  MEASURED: no gain (16/52 time to solution equal within noise to the
        full 3N-unknown LM).  Kept as an option, default off.
IDEA 15 Inner-shell rotation in the shell start (shell_start, --rot): shells with a
        small radius are turned about the centroid by a random angle.  The C3
        structure is kept (the rotation commutes with the cyclic permutation) but the
        relative phase of neighbouring rings becomes a search variable; a staggered
        (rotated) ring pattern is typical for optimal rules.
IDEA 7  PO -> PI conversion (mode po2pi).  An atlas PO rule with N nodes has one
        node outside.  We project the node into the triangle and re-solve with
        bounds; if that fails we shrink the outside node's coordinates slowly
        towards the triangle while re-solving (homotopy in the constraint).
"""
import argparse
import json
import math
import sys
import time

import numpy as np
from scipy.optimize import least_squares


# ---------------------------------------------------------------- basis (IDEA 1, 3)
def jacobi_a0(q_max, a, z):
    """P_q^{(a,0)}(z) and derivative for q=0..q_max; returns arrays [q][points]."""
    P = np.zeros((q_max + 1,) + z.shape)
    D = np.zeros_like(P)
    P[0] = 1.0
    if q_max >= 1:
        P[1] = (a + 1) + (a + 2) * (z - 1) / 2
        D[1] = (a + 2) / 2
    for n in range(2, q_max + 1):
        c0 = 2 * n * (n + a) * (2 * n + a - 2)
        c1 = (2 * n + a - 1) * (2 * n + a) * (2 * n + a - 2)
        c2 = (2 * n + a - 1) * a * a
        c3 = 2 * (n + a - 1) * (n - 1) * (2 * n + a)
        P[n] = ((c1 * z + c2) * P[n - 1] - c3 * P[n - 2]) / c0
        D[n] = (c1 * P[n - 1] + (c1 * z + c2) * D[n - 1] - c3 * D[n - 2]) / c0
    return P, D


def index_pairs(n):
    return [(p, q) for p in range(n + 1) for q in range(n + 1 - p)]


class Basis:
    """Dubiner basis phi_pq(x,y) = A_p(x,y) * P_q^{(2p+1,0)}(2y-1), scaled to
    unit L2 norm on the triangle, with A_p(x,y) = (1-y)^p P_p((2x+y-1)/(1-y))
    computed by a polynomial recurrence (no division by 1-y)."""

    def __init__(self, n):
        self.n = n
        self.pq = index_pairs(n)
        self.M = len(self.pq)
        # unit-norm scaling: int phi_pq^2 over the triangle = 1/((2p+1)(2p+2q+2)) * 1/2 *... computed exactly:
        # int_T A_p^2 P_q^2 = (1/(2p+1)) * (1/ (2(p+q)+2)) * 2^{-?}; we use the closed form
        #   ||phi_pq||^2 = 1 / ((2p+1) * (2p+2q+2)) * 2   -> verified numerically in selftest()
        self.scale = np.array([math.sqrt((2 * p + 1) * (2 * p + 2 * q + 2)) for p, q in self.pq])

    def eval(self, x, y):
        """Return B (M x N), Bx, By."""
        n = self.n
        N = x.size
        s = 2 * x + y - 1  # in [-(1-y), (1-y)]
        r2 = (1 - y) ** 2
        A = np.zeros((n + 1, N))
        Ax = np.zeros_like(A)
        Ay = np.zeros_like(A)
        A[0] = 1.0
        if n >= 1:
            A[1] = s
            Ax[1] = 2.0
            Ay[1] = 1.0
        for p in range(2, n + 1):
            A[p] = ((2 * p - 1) * s * A[p - 1] - (p - 1) * r2 * A[p - 2]) / p
            Ax[p] = ((2 * p - 1) * (2 * A[p - 1] + s * Ax[p - 1]) - (p - 1) * r2 * Ax[p - 2]) / p
            Ay[p] = ((2 * p - 1) * (A[p - 1] + s * Ay[p - 1])
                     - (p - 1) * (r2 * Ay[p - 2] - 2 * (1 - y) * A[p - 2])) / p
        z = 2 * y - 1
        B = np.empty((self.M, N))
        Bx = np.empty_like(B)
        By = np.empty_like(B)
        cache = {}
        for k, (p, q) in enumerate(self.pq):
            if p not in cache:
                cache[p] = jacobi_a0(n - p, 2 * p + 1, z)
            P, D = cache[p]
            B[k] = A[p] * P[q]
            Bx[k] = Ax[p] * P[q]
            By[k] = Ay[p] * P[q] + 2 * A[p] * D[q]
        sc = self.scale[:, None]
        return B * sc, Bx * sc, By * sc

    def rhs(self):
        """Integral of phi_k over the triangle: only phi_00 = 1 survives (area 1/2)."""
        r = np.zeros(self.M)
        r[0] = 0.5 * self.scale[0]
        return r


_basis_cache = {}


def basis(n):
    if n not in _basis_cache:
        _basis_cache[n] = Basis(n)
    return _basis_cache[n]


# ------------------------------------------------------------ residual (IDEA 2)
def unpack(v, N):
    s = v[:N]
    t = v[N:2 * N]
    w = v[2 * N:]
    return s, t, w


def to_xy(s, t):
    return s, (1 - s) * t


def residual_and_jac(v, n, N):
    bs = basis(n)
    s, t, w = unpack(v, N)
    x, y = to_xy(s, t)
    B, Bx, By = bs.eval(x, y)
    res = B @ w - bs.rhs()
    # d x/ds = 1, d y/ds = -t, d y/dt = 1-s
    Js = (Bx - By * t) * w
    Jt = By * (1 - s) * w
    J = np.hstack([Js, Jt, B])
    return res, J


def lm_bounded(fun_jac, v0, lb, ub, max_nfev, tol=1e-15):
    """IDEA 10: own projected Levenberg-Marquardt instead of scipy 'trf'.
    Profiling showed that the SVD inside scipy's trf took > 50 % of the time and
    that residual and Jacobian were evaluated separately (two basis evaluations
    per iteration).  Here one evaluation gives both, the damped normal equations
    (J^T J + mu diag(J^T J)) dv = -J^T r are solved by Cholesky on the free
    variables only (variables sitting on a bound with the gradient pushing outward
    are frozen), and the step is projected onto the box.  The orthonormal basis
    (IDEA 1) keeps J^T J well enough conditioned for this."""
    v = v0.copy()
    r, J = fun_jac(v)
    cost = 0.5 * float(r @ r)
    mu = 1e-3
    for _ in range(max_nfev):
        g = J.T @ r
        free = ~(((v <= lb + 1e-14) & (g > 0)) | ((v >= ub - 1e-14) & (g < 0)))
        if not free.any() or np.abs(g[free]).max() < tol:
            break
        Jf = J[:, free]
        # Use the smaller of the primal and dual damped systems.  Preserve the
        # diagonal LM scaling of the primal formulation exactly:
        #
        #   (Jf.T Jf + mu D) step = -Jf.T r,  D = diag(Jf.T Jf).
        #
        # With K = Jf D^{-1/2}, the equivalent dual step is
        #
        #   step = -D^{-1/2} K.T (K K.T + mu I)^{-1} r.
        #
        # The dual system is cheaper when there are fewer moment equations than
        # free variables.  This scaling matters: replacing D by a scalar damping
        # term would change the LM trajectory and confound the benchmark.
        d = np.sum(Jf * Jf, axis=0)
        d[d < 1e-14] = 1e-14
        use_dual = Jf.shape[0] < Jf.shape[1]
        if use_dual:
            inv_sqrt_d = 1.0 / np.sqrt(d)
            K = Jf * inv_sqrt_d[None, :]
            system = K @ K.T
        else:
            system = Jf.T @ Jf
        improved = False
        for _try in range(12):
            try:
                damped = system.copy()
                if use_dual:
                    damped.flat[::damped.shape[0] + 1] += mu
                    L = np.linalg.cholesky(damped)
                    y = np.linalg.solve(L.T, np.linalg.solve(L, r))
                    step = -(Jf.T @ y) / d
                else:
                    damped.flat[::damped.shape[0] + 1] += mu * d
                    L = np.linalg.cholesky(damped)
                    step = -np.linalg.solve(L.T, np.linalg.solve(L, g[free]))
            except np.linalg.LinAlgError:
                mu *= 10
                continue
            vn = v.copy()
            vn[free] = np.clip(v[free] + step, lb[free], ub[free])
            rn, Jn = fun_jac(vn)
            cn = 0.5 * float(rn @ rn)
            if cn < cost:
                dc = cost - cn
                v, r, J, cost = vn, rn, Jn, cn
                mu = max(mu / 5, 1e-12)
                improved = True
                break
            mu *= 4
        if not improved or dc < tol * max(cost, 1e-300) * 1e-3 and cost < 1e-28:
            break
        if cost < 1e-30:
            break
    return v, float(np.linalg.norm(r, np.inf))


def solve(v0, n, N, max_nfev=300, ftol=1e-15, xtol=1e-15, gtol=1e-15):
    lb = np.zeros(3 * N)
    ub = np.concatenate([np.ones(2 * N), np.full(N, np.inf)])
    v0 = np.clip(v0, lb + 1e-12, np.where(np.isinf(ub), 1e9, ub - 1e-12))
    return lm_bounded(lambda v: residual_and_jac(v, n, N), v0, lb, ub, max_nfev)


# ---------------------------------------------- variable projection (IDEA 14)
def vp_residual_jac(q, n, N):
    """IDEA 14: variable projection (Golub-Pereyra / Kaufman).  The moment equations
    are linear in the weights, so for given nodes the best weights solve a linear
    least squares problem.  Eliminating them leaves 2N unknowns (node positions)
    instead of 3N, a smaller and often better conditioned nonlinear problem, and
    every iterate has the optimal weights.  Residual r(q) = (I - B B^+) rhs,
    Jacobian (Kaufman approximation) J = +P_perp (dB/dq) diag(w).  Positivity of the
    weights is not enforced inside; a result with negative weights is polished
    with the full bounded solver afterwards (which decides acceptance)."""
    bs = basis(n)
    s = q[:N]
    t = q[N:]
    x, y = to_xy(s, t)
    B, Bx, By = bs.eval(x, y)
    rhs = bs.rhs()
    Q, R = np.linalg.qr(B)
    w = np.linalg.lstsq(R, Q.T @ rhs, rcond=None)[0]
    r = B @ w - rhs
    Js = (Bx - By * t) * w
    Jt = By * (1 - s) * w
    J = np.hstack([Js, Jt])
    J = J - Q @ (Q.T @ J)
    r = r - Q @ (Q.T @ r)  # projected residual (equals -P_perp rhs)
    return r, J, w


def solve_vp(v0, n, N, max_nfev=300):
    """Solve in (s,t) with eliminated weights, return a full vector (s,t,w)."""
    q0 = np.clip(v0[:2 * N], 1e-12, 1 - 1e-12)
    lb = np.zeros(2 * N)
    ub = np.ones(2 * N)

    def fj(q):
        r, J, _ = vp_residual_jac(q, n, N)
        return r, J

    q, res = lm_bounded(fj, q0, lb, ub, max_nfev)
    _, _, w = vp_residual_jac(q, n, N)
    v = np.concatenate([q, w])
    return v, res


# ---------------------------------------------------------------- starts (IDEA 5)
def random_start(N, rng):
    def law(m):
        u = rng.random(m)
        a = (1 - np.cos(np.pi * u)) / 2  # arcsine law: dense near 0 and 1
        mix = rng.random(m) < 0.5
        return np.where(mix, a, rng.random(m))

    s = law(N)
    t = law(N)
    w = np.full(N, 0.5 / N)
    return np.concatenate([s, t, w])


def xy_to_st(x, y):
    s = np.clip(x, 0, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(s < 1, y / (1 - s), 0.0)
    return s, np.clip(t, 0, 1)


def load_rule(path):
    r = json.load(open(path))
    xy = np.array(r["nodes"], float)
    return xy[:, 0], xy[:, 1], np.array(r["weights"], float)


def continuation(v0, n, N, steps):
    """IDEA 4: solve degrees steps[0..] in turn, ending at n."""
    v = v0
    for d in steps:
        v, res = solve(v, d, N, max_nfev=150)
    return v


def accept(v, n, N, tol):
    res, _ = residual_and_jac(v, n, N)
    s, t, w = unpack(v, N)
    return np.linalg.norm(res, np.inf) < tol and w.min() > 0


def save(path, n, N, v, seed, mode, cmd, res):
    s, t, w = unpack(v, N)
    x, y = to_xy(s, t)
    rec = {"degree": n, "N": N, "seed": seed, "mode": mode, "command": cmd, "residual": res,
           "nodes": [[float(a), float(b)] for a, b in zip(x, y)], "weights": [float(a) for a in w]}
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


# ------------------------------------------------------------------ modes
def mode_random(a, rng, cmd):
    n, N = a.degree, a.nodes
    found = 0
    t0 = time.time()
    for k in range(a.tries):
        if a.minutes and time.time() - t0 > 60 * a.minutes:
            break
        v = random_start(N, rng)
        steps = [d for d in (n - 4, n - 2, n - 1) if d >= 1] + [n]
        v = continuation(v, n, N, steps) if a.continuation else v
        v, res = solve(v, n, N, max_nfev=a.maxfev)
        ok = accept(v, n, N, a.accept)
        print(f"try {k} res {res:.2e} {'FOUND' if ok else ''}", flush=True)
        if ok:
            found += 1
            save(a.out, n, N, v, a.seed, "random", cmd, res)
    print("found", found, "of", k + 1)


def mode_elim(a, rng, cmd):
    """IDEA 6: remove one node from an (N+1)-node rule, re-solve."""
    x, y, w = load_rule(a.start)
    n, N = a.degree, a.nodes
    assert len(x) == N + 1
    s, t = xy_to_st(x, y)
    order = list(np.argsort(w)) if a.order == "weight" else list(rng.permutation(N + 1))
    found = 0
    for k, i in enumerate(order[:a.tries]):
        keep = np.arange(N + 1) != i
        w2 = w[keep].copy()
        # give the removed weight to the nearest neighbours proportional to 1/d^2
        d2 = (x[keep] - x[i]) ** 2 + (y[keep] - y[i]) ** 2 + 1e-6
        wt = 1 / d2
        w2 += w[i] * wt / wt.sum()
        v = np.concatenate([s[keep], t[keep], w2])
        v += np.concatenate([rng.normal(0, a.noise, 2 * N), np.zeros(N)])
        v, res = solve(v, n, N, max_nfev=a.maxfev)
        ok = accept(v, n, N, a.accept)
        print(f"remove {i} (w={w[i]:.2e}) res {res:.2e} {'FOUND' if ok else ''}", flush=True)
        if ok:
            found += 1
            save(a.out, n, N, v, a.seed, "elim", cmd, res)
    print("found", found)


def mode_po2pi(a, rng, cmd):
    """IDEA 7: PO rule with N nodes, push outside nodes in by a homotopy."""
    x, y, w = load_rule(a.start)
    n, N = a.degree, a.nodes
    assert len(x) == N
    out = (x < 0) | (y < 0) | (x + y > 1)
    print("nodes outside:", int(out.sum()), [(float(x[i]), float(y[i])) for i in np.where(out)[0]])
    # homotopy: contract the outside nodes towards the centroid gradually
    found = 0
    for rep in range(a.tries):
        xx, yy, ww = x.copy(), y.copy(), w.copy()
        ww = np.maximum(ww, 1e-6)
        ok = False
        for lam in np.linspace(0, 1, a.hsteps):
            xt = xx.copy()
            yt = yy.copy()
            for i in np.where(out)[0]:
                cx, cy = 1 / 3, 1 / 3
                # projection onto triangle along the ray to the centroid
                lo, hi = 0.0, 1.0
                for _ in range(60):
                    m = (lo + hi) / 2
                    px, py = cx + m * (xx[i] - cx), cy + m * (yy[i] - cy)
                    if px >= 0 and py >= 0 and px + py <= 1:
                        lo = m
                    else:
                        hi = m
                px, py = cx + lo * (xx[i] - cx), cy + lo * (yy[i] - cy)
                xt[i] = (1 - lam) * xx[i] + lam * px
                yt[i] = (1 - lam) * yy[i] + lam * py
            s, t = xy_to_st(np.clip(xt, 0, 1), np.clip(yt, 0, 1))
            v = np.concatenate([s, t, ww]) + np.concatenate([rng.normal(0, a.noise * rep, 2 * N), np.zeros(N)])
            v, res = solve(v, n, N, max_nfev=a.maxfev)
            print(f"rep {rep} lam {lam:.2f} res {res:.2e}", flush=True)
            if accept(v, n, N, a.accept):
                ok = True
                break
        if ok:
            found += 1
            save(a.out, n, N, v, a.seed, "po2pi", cmd, res)
            break
    print("found", found)


# ------------------------------------------------------------ C3 symmetry (IDEA 8)
def c3_expand(v, k, centroid, n):
    """Parameters v = [s(k), t(k), w(k), (wc)] -> residual and Jacobian w.r.t. v.
    Each base node (x,y) with barycentric (1-x-y, x, y) stands for the 3 nodes
    (x,y), (y,1-x-y), (1-x-y,x), all with weight w.  The centroid (optional) has its
    own weight.  Because the 3 rotated copies enter the full moment equations, the
    equations are the plain ones, only the unknowns are reduced to a third."""
    bs = basis(n)
    s = v[:k]
    t = v[k:2 * k]
    w = v[2 * k:3 * k]
    x, y = to_xy(s, t)
    X = np.concatenate([x, y, 1 - x - y])
    Y = np.concatenate([y, 1 - x - y, x])
    B, Bx, By = bs.eval(X, Y)
    W = np.concatenate([w, w, w])
    res = B @ W - bs.rhs()
    # derivative of the expanded coordinates with respect to base (x,y)
    # node copy 1: (x,y); copy 2: (y, 1-x-y); copy 3: (1-x-y, x)
    dXdx = np.concatenate([np.ones(k), np.zeros(k), -np.ones(k)])
    dXdy = np.concatenate([np.zeros(k), np.ones(k), -np.ones(k)])
    dYdx = np.concatenate([np.zeros(k), -np.ones(k), np.ones(k)])
    dYdy = np.concatenate([np.ones(k), -np.ones(k), np.zeros(k)])
    Gx = (Bx * dXdx + By * dYdx) * W  # M x 3k : d residual / d x_base (per copy)
    Gy = (Bx * dXdy + By * dYdy) * W
    Gx = Gx[:, :k] + Gx[:, k:2 * k] + Gx[:, 2 * k:]
    Gy = Gy[:, :k] + Gy[:, k:2 * k] + Gy[:, 2 * k:]
    Js = Gx - Gy * t
    Jt = Gy * (1 - s)
    Jw = B[:, :k] + B[:, k:2 * k] + B[:, 2 * k:]
    cols = [Js, Jt, Jw]
    if centroid:
        Bc, _, _ = bs.eval(np.array([1 / 3]), np.array([1 / 3]))
        res = res + Bc[:, 0] * v[3 * k]
        cols.append(Bc)
    return res, np.hstack(cols)


def solve_c3(v0, k, centroid, n, max_nfev=300):
    m = 3 * k + (1 if centroid else 0)
    lb = np.zeros(m)
    ub = np.concatenate([np.ones(2 * k), np.full(m - 2 * k, np.inf)])
    v0 = np.clip(v0, lb + 1e-12, np.where(np.isinf(ub), 1e9, ub - 1e-12))
    return lm_bounded(lambda v: c3_expand(v, k, centroid, n), v0, lb, ub, max_nfev)


def c3_to_full(v, k, centroid):
    s = v[:k]
    t = v[k:2 * k]
    w = v[2 * k:3 * k]
    x, y = to_xy(s, t)
    X = np.concatenate([x, y, 1 - x - y])
    Y = np.concatenate([y, 1 - x - y, x])
    W = np.concatenate([w, w, w])
    if centroid:
        X = np.append(X, 1 / 3)
        Y = np.append(Y, 1 / 3)
        W = np.append(W, v[3 * k])
    s2, t2 = xy_to_st(X, Y)
    return np.concatenate([s2, t2, W])


def mode_c3(a, rng, cmd):
    """IDEA 8: C3 (rotation) symmetric rules.  N = 3k (+1 with the centroid).
    The search space has a third of the unknowns, the basins are much larger, and
    a symmetric solution found is also better in the sense of the atlas (C3 beats
    unsymmetric at equal N)."""
    n, N = a.degree, a.nodes
    centroid = N % 3 == 1
    k = N // 3
    assert N % 3 in (0, 1)
    t0 = time.time()
    found = 0
    for i in range(a.tries):
        if a.minutes and time.time() - t0 > 60 * a.minutes:
            break
        v = random_start(k, rng)
        if centroid:
            v = np.append(v, 0.5 / N)
        # degree continuation (IDEA 4) in the symmetric space
        for d in [d for d in (n - 4, n - 2, n - 1) if d >= 1]:
            v, _ = solve_c3(v, k, centroid, d, max_nfev=120)
        v, res = solve_c3(v, k, centroid, n, max_nfev=a.maxfev)
        full = c3_to_full(v, k, centroid)
        ok = accept(full, n, N, a.accept)
        print(f"try {i} res {res:.2e} {'FOUND' if ok else ''}", flush=True)
        if ok:
            found += 1
            save(a.out, n, N, full, a.seed, "c3", cmd, res)
    print("found", found)


def solve_dispatch(v, n, N, max_nfev, vp):
    if vp:
        v, _ = solve_vp(v, n, N, max_nfev)
        if v[2 * N:].min() <= 0:
            v[2 * N:] = np.maximum(v[2 * N:], 1e-9)
    return solve(v, n, N, max_nfev=max_nfev if not vp else 60)


def mode_basin(a, rng, cmd):
    """IDEA 9: iterated local search (basin hopping) on the residual norm.
    Random restarts only sample basins; a near-solution (residual ~1e-3) usually
    differs from a true solution by the placement of a few nodes.  So we keep the
    best local minimum, relocate 1-3 nodes (the ones with the smallest weights, or
    random ones) to fresh random places, re-solve, and accept the result if the
    residual norm does not grow (plus a small chance of accepting worse results,
    to leave dead ends).  Restart from scratch after --stall rounds without gain."""
    n, N = a.degree, a.nodes
    t0 = time.time()
    found = 0
    rounds = 0

    def norm2(v):
        r, _ = residual_and_jac(v, n, N)
        return float(np.linalg.norm(r))

    while not (a.minutes and time.time() - t0 > 60 * a.minutes):
        v = random_start(N, rng)
        v = continuation(v, n, N, [d for d in (n - 4, n - 2, n - 1) if d >= 1])
        v, res = solve(v, n, N, max_nfev=a.maxfev)
        cur = norm2(v)
        best = cur
        stall = 0
        while stall < a.stall:
            if a.minutes and time.time() - t0 > 60 * a.minutes:
                break
            rounds += 1
            s, t, w = [x.copy() for x in unpack(v, N)]
            m = int(rng.integers(1, 4))
            if rng.random() < 0.5:
                idx = np.argsort(w)[:m]
            else:
                idx = rng.choice(N, m, replace=False)
            fresh = random_start(m, rng)
            s[idx] = fresh[:m]
            t[idx] = fresh[m:2 * m]
            w[idx] = np.median(w)
            v2 = np.concatenate([s, t, w])
            v2, res2 = solve_dispatch(v2, n, N, a.hop_fev, a.vp)
            c2 = norm2(v2)
            if accept(v2, n, N, a.accept) or c2 < 1e-9:
                v2, res2 = solve(v2, n, N, max_nfev=a.maxfev)
                if accept(v2, n, N, a.accept):
                    found += 1
                    save(a.out, n, N, v2, a.seed, "basin", cmd, res2)
                    print(f"round {rounds} FOUND res {res2:.2e}", flush=True)
                    return
            if c2 < cur or rng.random() < 0.02:
                v, cur = v2, c2
            if c2 < best * 0.999:
                best = c2
                stall = 0
                print(f"round {rounds} best |r| {best:.3e}", flush=True)
            else:
                stall += 1
    print("found", found, "rounds", rounds)


# ---------------------------------------------------- shell start + noise (IDEA 11)
def shell_pattern_random(N, rng):
    """Random shell pattern with exactly N nodes in C3 form: outer shell has m0
    points per edge (3(m0-1) nodes), then steps of -2 or -3 points, the innermost
    shells have 1..3 points (1 = centroid).  Returns the list of m per shell."""
    for _ in range(10000):
        lo = max(3, int(math.sqrt(N / 1.5)) - 3)
        hi = min(N // 3 + 2, int(math.sqrt(N / 1.5)) + 8)
        m0 = int(rng.integers(lo, hi + 1))
        ms = [m0]
        total = 3 * (m0 - 1)
        while ms[-1] > 1 and total < N:
            m = ms[-1] - int(rng.choice([1, 2, 2, 3, 3]))
            if m < 1:
                break
            ms.append(m)
            total += 3 * (m - 1) if m > 1 else 1
        if total == N:
            return ms
    return None


def pattern_total(ms):
    return sum(3 * (m - 1) if m > 1 else 1 for m in ms)


def shell_pattern_near(N, bases, rng):
    """IDEA 18: a shell pattern near a pattern that already worked.  Successful
    patterns of neighbouring degrees are similar (e.g. 22 20 18 17 14 11 9 8 5 for
    degree 44, 24 21 19 17 16 15 12 10 7 5 2 1 for degree 48): outer ring about
    degree/2, steps of 2 and 3 with a few 1s.  Each ring count is changed by
    -1, 0 or +1 (0 twice as likely), the last ring may be dropped or added; the
    result must be strictly decreasing and have exactly N nodes."""
    for _ in range(20000):
        base = bases[int(rng.integers(len(bases)))]
        ms = [max(1, m + int(rng.choice([-1, 0, 0, 1]))) for m in base]
        r = rng.random()
        if r < 0.15 and len(ms) > 3:
            ms.pop()
        elif r < 0.3 and ms[-1] > 2:
            ms.append(max(1, ms[-1] - int(rng.choice([2, 3]))))
        if any(ms[i] <= ms[i + 1] for i in range(len(ms) - 1)):
            continue
        if pattern_total(ms) == N:
            return ms
    return None


def shell_start(ms, rng, noise, slide, rot=0.0):
    """IDEA 11: C3 start built from nested shells.  Optimal triangle rules look
    like nested rings (dense near the boundary) - the structure the atlas rules of
    the cells at the counting value show.  Shell k is the boundary of the unit
    triangle scaled about the centroid by r_k = (m_k-1)/(m_0-1); its m_k points per
    edge are Chebyshev-Lobatto positions, one edge only (the other two edges are
    the C3 images, one corner excluded per edge).  The outer shell is slid along the
    edge by `slide` times its node spacing (breaks the corner degeneracy).  Noise is
    proportional to the bubble function 27 b0 b1 b2, so it vanishes on the boundary
    and boundary nodes stay on the boundary.  Returns (s, t, w, centroid)."""
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    Cc = np.array([0.0, 1.0])
    c = np.array([1 / 3, 1 / 3])
    pts = []
    centroid = False
    for k, m in enumerate(ms):
        if m == 1:
            centroid = True
            continue
        r = (m - 1) / (ms[0] - 1)
        PA = c + r * (A - c)
        PB = c + r * (B - c)
        j = np.arange(m - 1, dtype=float)
        u = (1 - np.cos(np.pi * j / (m - 1))) / 2
        if k == 0:
            u = u + slide / (m - 1)
        ang = 0.0
        if rot > 0 and k > 0 and r < 0.85:
            ang = rng.uniform(-rot, rot) * (2 * math.pi / 3)
        ca, sa = math.cos(ang), math.sin(ang)
        for ui in u:
            P = PA + ui * (PB - PA)
            d = P - c
            pts.append(c + np.array([ca * d[0] - sa * d[1], sa * d[0] + ca * d[1]]))
    pts = np.array(pts)
    x, y = pts[:, 0], pts[:, 1]
    bub = 27 * x * y * np.clip(1 - x - y, 0, None)
    x = x + noise * bub * rng.normal(size=len(x))
    y = y + noise * bub * rng.normal(size=len(y))
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, 1 - x)
    s, t = xy_to_st(x, y)
    k = len(s)
    N = 3 * k + (1 if centroid else 0)
    w = np.full(k, 0.5 / N)
    return s, t, w, centroid


def mode_shell(a, rng, cmd):
    """IDEA 11: shell start + C3 LM with short noise starts, many patterns.
    Short runs with a stagnation stop (--short_fev steps); only starts whose
    residual falls below --promote get the long run."""
    n, N = a.degree, a.nodes
    assert N % 3 in (0, 1)
    t0 = time.time()
    found = 0
    tried = 0
    best = 1e9
    while not (a.minutes and time.time() - t0 > 60 * a.minutes) and tried < a.tries:
        ms = None
        if a.near and rng.random() < a.near_prob:
            bases = [[int(x) for x in b.split(",")] for b in a.near.split(";")]
            ms = shell_pattern_near(N, bases, rng)
        if ms is None:
            ms = shell_pattern_random(N, rng)
        if ms is None:
            print("no pattern")
            return
        for _ in range(a.starts_per_pattern):
            tried += 1
            s, t, w, centroid = shell_start(ms, rng, a.noise, rng.uniform(0, 0.25), a.rot)
            k = len(s)
            v = np.concatenate([s, t, w] + ([[0.5 / N]] if centroid else []))
            v, res = solve_c3(v, k, centroid, n, max_nfev=a.short_fev)
            if res < best:
                best = res
                print(f"pattern {ms} res {res:.2e} (best so far)", flush=True)
            if a.asym_polish and res < a.asym_polish:
                # IDEA 13: leave the symmetric subspace.  A C3 near-solution is a
                # start in the full space; without the symmetry constraint the
                # residual can still go to zero (the symmetric optimum is only a
                # saddle there).  Small seeded noise breaks the symmetry.
                fullv = c3_to_full(v, k, centroid)
                fullv[:2 * N] += rng.normal(0, 1e-3, 2 * N)
                va, resa = solve(fullv, n, N, max_nfev=a.maxfev)
                if accept(va, n, N, a.accept):
                    found += 1
                    save(a.out, n, N, va, a.seed, "shell+asym:" + " ".join(map(str, ms)), cmd, resa)
                    print(f"FOUND (asym polish) pattern {ms} res {resa:.2e}", flush=True)
            if res < a.promote:
                v, res = solve_c3(v, k, centroid, n, max_nfev=a.maxfev)
                full = c3_to_full(v, k, centroid)
                if accept(full, n, N, a.accept):
                    found += 1
                    save(a.out, n, N, full, a.seed, "shell:" + " ".join(map(str, ms)), cmd, res)
                    print(f"FOUND pattern {ms} res {res:.2e}", flush=True)
                    if a.max_found and found >= a.max_found:
                        print("found", found, "starts", tried, "best", best)
                        return
    print("found", found, "starts", tried, "best", best)


def mode_shellasym(a, rng, cmd):
    """IDEA 12: shell start for N = 2 mod 3 (no C3 rule with exactly N nodes).
    Take a C3 shell pattern with N-1 or N-2 nodes (N-1, N-2 = 0/1 mod 3), expand it
    to all nodes, add 1 or 2 extra nodes at random places and solve WITHOUT symmetry.
    The nearly symmetric start has the right ring structure, the extra nodes break
    the symmetry just enough (the atlas PO rules at N = Z for degrees 8, 10, 12 were
    found this way)."""
    n, N = a.degree, a.nodes
    t0 = time.time()
    found = 0
    tried = 0
    best = 1e9
    while not (a.minutes and time.time() - t0 > 60 * a.minutes) and tried < a.tries:
        extra = int(rng.choice([1, 2]))
        N0 = N - extra
        if N0 % 3 == 2:
            extra = 3 - (N - N0) if False else extra
            continue
        ms = shell_pattern_random(N0, rng)
        if ms is None:
            continue
        for _ in range(a.starts_per_pattern):
            tried += 1
            s, t, w, centroid = shell_start(ms, rng, a.noise, rng.uniform(0, 0.25), a.rot)
            k = len(s)
            vc = np.concatenate([s, t, w] + ([[0.5 / N]] if centroid else []))
            full = c3_to_full(vc, k, centroid)
            fs, ft, fw = unpack(full, N0)
            fs = np.concatenate([fs, rng.random(extra)])
            ft = np.concatenate([ft, rng.random(extra)])
            fw = np.concatenate([fw, np.full(extra, 0.5 / N)])
            v = np.concatenate([fs, ft, fw])
            v[:2 * N] += rng.normal(0, a.noise * 0.2, 2 * N)
            v, res = solve(v, n, N, max_nfev=a.short_fev)
            if res < best:
                best = res
                print(f"pattern {ms}+{extra} res {res:.2e} (best so far)", flush=True)
            if res < a.promote:
                v, res = solve(v, n, N, max_nfev=a.maxfev)
                if accept(v, n, N, a.accept):
                    found += 1
                    save(a.out, n, N, v, a.seed, f"shellasym:{ms}+{extra}", cmd, res)
                    print(f"FOUND pattern {ms}+{extra} res {res:.2e}", flush=True)
                    if a.max_found and found >= a.max_found:
                        print("found", found, "starts", tried, "best", best)
                        return
    print("found", found, "starts", tried, "best", best)


# ----------------------------------------------------- mirror symmetry (IDEA 16)
def mirror_expand(v, a, b, n):
    """IDEA 16: mirror (x <-> y) symmetric rules, N = 2a + b.  a pairs (x,y),(y,x)
    with the same weight and b nodes (u,u) on the mirror axis.  Unknowns:
    s,t,w of the pairs (Duffy coordinates of one node) and u,w of the axis nodes.
    Allows N = 2 mod 3, where no C3 rule exists, with half the unknowns."""
    bs = basis(n)
    s = v[:a]
    t = v[a:2 * a]
    w = v[2 * a:3 * a]
    u = v[3 * a:3 * a + b]
    wu = v[3 * a + b:3 * a + 2 * b]
    x, y = to_xy(s, t)
    X = np.concatenate([x, y, u])
    Y = np.concatenate([y, x, u])
    W = np.concatenate([w, w, wu])
    B, Bx, By = bs.eval(X, Y)
    res = B @ W - bs.rhs()
    Gx = Bx[:, :a] * w + By[:, a:2 * a] * w
    Gy = By[:, :a] * w + Bx[:, a:2 * a] * w
    Js = Gx - Gy * t
    Jt = Gy * (1 - s)
    Jw = B[:, :a] + B[:, a:2 * a]
    Ju = (Bx[:, 2 * a:] + By[:, 2 * a:]) * wu
    Jwu = B[:, 2 * a:]
    return res, np.hstack([Js, Jt, Jw, Ju, Jwu])


def solve_mirror(v0, a, b, n, max_nfev=300):
    m = 3 * a + 2 * b
    lb = np.zeros(m)
    ub = np.concatenate([np.ones(2 * a), np.full(a, np.inf), np.full(b, 0.5), np.full(b, np.inf)])
    v0 = np.clip(v0, lb + 1e-12, np.where(np.isinf(ub), 1e9, ub - 1e-12))
    return lm_bounded(lambda q: mirror_expand(q, a, b, n), v0, lb, ub, max_nfev)


def mirror_to_full(v, a, b):
    s = v[:a]
    t = v[a:2 * a]
    w = v[2 * a:3 * a]
    u = v[3 * a:3 * a + b]
    wu = v[3 * a + b:3 * a + 2 * b]
    x, y = to_xy(s, t)
    X = np.concatenate([x, y, u])
    Y = np.concatenate([y, x, u])
    W = np.concatenate([w, w, wu])
    s2, t2 = xy_to_st(X, Y)
    return np.concatenate([s2, t2, W])


def mode_mirror(a_, rng, cmd):
    """IDEA 16: random starts (arcsine law) with mirror symmetry, b axis nodes drawn
    from --axis_max, degree continuation, short runs with a stagnation cut."""
    n, N = a_.degree, a_.nodes
    t0 = time.time()
    found = 0
    tried = 0
    best = 1e9
    while not (a_.minutes and time.time() - t0 > 60 * a_.minutes) and tried < a_.tries:
        b = int(rng.integers(0, a_.axis_max + 1))
        if b > N or (N - b) % 2:
            continue
        a = (N - b) // 2
        tried += 1
        base = random_start(a, rng)
        v = np.concatenate([base, rng.random(b) * 0.5, np.full(b, 0.5 / N)])
        for d in [d for d in (n - 4, n - 2, n - 1) if d >= 1]:
            v, _ = solve_mirror(v, a, b, d, max_nfev=100)
        v, res = solve_mirror(v, a, b, n, max_nfev=a_.maxfev)
        if res < best:
            best = res
            print(f"try {tried} axis {b} res {res:.2e} (best so far)", flush=True)
        full = mirror_to_full(v, a, b)
        if accept(full, n, N, a_.accept):
            found += 1
            save(a_.out, n, N, full, a_.seed, f"mirror:{b}", cmd, res)
            print(f"FOUND axis {b} res {res:.2e}", flush=True)
            if a_.max_found and found >= a_.max_found:
                break
    print("found", found, "tries", tried, "best", best)


def mode_mirrorshell(a_, rng, cmd):
    """IDEA 17: mirror symmetric shell start for N = 2 mod 3 (no C3 rule possible).
    A C3 shell pattern with N0 = N - extra nodes (N0 = 0 or 1 mod 3) whose rings are
    not slid or turned is S3 symmetric, hence mirror symmetric.  Its nodes off the
    axis x = y are paired (base node with x < y), its nodes on the axis are axis
    nodes.  extra = 1 adds an axis node, extra = 2 adds a mirror pair or two axis
    nodes.  Seeded noise (bubble function) breaks the exact structure; the solve is
    the mirror symmetric LM (IDEA 16), which has half the unknowns."""
    n, N = a_.degree, a_.nodes
    t0 = time.time()
    found = 0
    tried = 0
    best = 1e9
    while not (a_.minutes and time.time() - t0 > 60 * a_.minutes) and tried < a_.tries:
        extra = int(rng.choice([1, 2]))
        N0 = N - extra
        if N0 % 3 == 2:
            continue
        ms = shell_pattern_random(N0, rng)
        if ms is None:
            continue
        for _ in range(a_.starts_per_pattern):
            tried += 1
            s0, t0_, w0, cen = shell_start(ms, rng, 0.0, 0.0, 0.0)
            kk = len(s0)
            full = c3_to_full(np.concatenate([s0, t0_, w0] + ([[0.5 / N]] if cen else [])), kk, cen)
            fs, ft, fw = unpack(full, N0)
            x, y = to_xy(fs, ft)
            on_axis = np.abs(x - y) < 1e-9
            base = (x < y) & ~on_axis
            bx, by = x[base], y[base]
            bw = fw[base]
            ux = x[on_axis]
            uw = fw[on_axis]
            if extra == 1:
                ux = np.append(ux, rng.random() * 0.5)
                uw = np.append(uw, 0.5 / N)
            elif rng.random() < 0.5:
                bx = np.append(bx, rng.random() * 0.4)
                by = np.append(by, 0.5 + rng.random() * 0.4 - 0.1)
                bw = np.append(bw, 0.5 / N)
            else:
                ux = np.concatenate([ux, rng.random(2) * 0.5])
                uw = np.concatenate([uw, np.full(2, 0.5 / N)])
            bub = 27 * bx * by * np.clip(1 - bx - by, 0, None)
            bx = np.clip(bx + a_.noise * bub * rng.normal(size=len(bx)), 0, 1)
            by = np.clip(by + a_.noise * bub * rng.normal(size=len(by)), 0, 1 - bx)
            bs_, bt_ = xy_to_st(bx, by)
            a = len(bx)
            b = len(ux)
            if 2 * a + b != N:
                continue
            v = np.concatenate([bs_, bt_, bw, np.clip(ux, 0, 0.5), uw])
            v, res = solve_mirror(v, a, b, n, max_nfev=a_.short_fev)
            if res < best:
                best = res
                print(f"pattern {ms}+{extra} res {res:.2e} (best so far)", flush=True)
            if res < a_.promote:
                v, res = solve_mirror(v, a, b, n, max_nfev=a_.maxfev)
                fullm = mirror_to_full(v, a, b)
                if accept(fullm, n, N, a_.accept):
                    found += 1
                    save(a_.out, n, N, fullm, a_.seed, f"mirrorshell:{ms}+{extra}", cmd, res)
                    print(f"FOUND pattern {ms}+{extra} res {res:.2e}", flush=True)
                    if a_.max_found and found >= a_.max_found:
                        print("found", found, "starts", tried, "best", best)
                        return
    print("found", found, "starts", tried, "best", best)


def selftest():
    """Check the basis: orthonormality via a high order Duffy-Gauss rule."""
    from numpy.polynomial.legendre import leggauss
    n = 8
    g, wg = leggauss(24)
    g = (g + 1) / 2
    wg = wg / 2
    S, T = np.meshgrid(g, g, indexing="ij")
    W = np.outer(wg, wg) * (1 - S)  # Jacobian of the Duffy map
    x, y = to_xy(S.ravel(), T.ravel())
    B, Bx, By = basis(n).eval(x, y)
    G = (B * W.ravel()) @ B.T
    err = np.abs(G - np.eye(len(G))).max()
    # derivative check
    h = 1e-6
    B2, _, _ = basis(n).eval(x + h, y)
    B3, _, _ = basis(n).eval(x - h, y)
    B4, _, _ = basis(n).eval(x, y + h)
    B5, _, _ = basis(n).eval(x, y - h)
    ex = np.abs((B2 - B3) / (2 * h) - Bx).max()
    ey = np.abs((B4 - B5) / (2 * h) - By).max()
    print("orthonormality error", err, "dx err", ex, "dy err", ey)
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--degree", type=int)
    ap.add_argument("--nodes", type=int)
    ap.add_argument("--mode", choices=["random", "elim", "po2pi", "c3", "basin", "shell", "shellasym", "mirror", "mirrorshell", "selftest"], default="random")
    ap.add_argument("--start")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tries", type=int, default=20)
    ap.add_argument("--minutes", type=float, default=0)
    ap.add_argument("--maxfev", type=int, default=300)
    ap.add_argument("--accept", type=float, default=1e-13)
    ap.add_argument("--noise", type=float, default=1e-3)
    ap.add_argument("--stall", type=int, default=60)
    ap.add_argument("--hop_fev", type=int, default=60)
    ap.add_argument("--short_fev", type=int, default=60)
    ap.add_argument("--promote", type=float, default=1e-2)
    ap.add_argument("--starts_per_pattern", type=int, default=100)
    ap.add_argument("--max_found", type=int, default=0,
                    help="stop after this many solutions (makes the command reproducible)")
    ap.add_argument("--near", default="", help="semicolon separated base patterns, e.g. '22,20,18;24,21,19'")
    ap.add_argument("--near_prob", type=float, default=0.7)
    ap.add_argument("--axis_max", type=int, default=6)
    ap.add_argument("--vp", type=int, default=0)
    ap.add_argument("--rot", type=float, default=0.0)
    ap.add_argument("--asym_polish", type=float, default=0.0)
    ap.add_argument("--hsteps", type=int, default=6)
    ap.add_argument("--order", choices=["weight", "random"], default="weight")
    ap.add_argument("--continuation", type=int, default=1)
    ap.add_argument("--out", default="tri_solutions.jsonl")
    a = ap.parse_args()
    cmd = "python code/tri_search.py " + " ".join(sys.argv[1:])
    if a.mode == "selftest":
        selftest()
        return
    rng = np.random.default_rng(a.seed)
    {"random": mode_random, "elim": mode_elim, "po2pi": mode_po2pi, "c3": mode_c3, "basin": mode_basin, "shell": mode_shell, "shellasym": mode_shellasym, "mirror": mode_mirror, "mirrorshell": mode_mirrorshell}[a.mode](a, rng, cmd)


if __name__ == "__main__":
    main()

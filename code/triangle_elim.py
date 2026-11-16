"""Triangle (0,0),(1,0),(0,1): PI cubature search by node elimination.

Usage:
    python3 triangle_elim.py <degree> <start_points> <seed> <seconds> [--out DIR]
    python3 triangle_elim.py refine <degree> <start.npy> <seed> <seconds> [--out DIR]

Nodes are parametrised by collapsed (Duffy) coordinates (u, v) in [0,1]^2,
x = u (1 - v), y = v, so every node stays in the closed triangle; weights are
kept >= 0.  The moment equations use the orthonormal Dubiner (PKD) basis
psi_ij = P_i(2u-1) (1-v)^i P_j^(2i+1,0)(2v-1), i + j <= degree.

The elimination mode starts from <start_points> random nodes drawn from
numpy's default_rng(seed), solves with a bounded Levenberg-Marquardt method and
removes one node at a time.  The refine mode loads a saved rule, removes one
node and solves again from perturbed nodes (noise from default_rng(seed)).
Every exact rule found is saved as DIR/tri_d<degree>_n<N>_s<seed>.npy with
columns x, y, w.  The check here is a floating-point residual test only; it
is neither a certificate nor a proof.
"""
import argparse
import os
import sys
import time
import numpy as np
from scipy.special import eval_jacobi
from scipy.linalg import cho_factor, cho_solve, LinAlgError


def jacobi_and_derivative(n, a, t):
    """Values and t-derivatives of P_n^(a,0)(t) for an array t."""
    val = eval_jacobi(n, a, 0, t)
    if n == 0:
        der = np.zeros_like(t)
    else:
        der = 0.5 * (n + a + 1) * eval_jacobi(n - 1, a + 1, 1, t)
    return val, der


class Problem:
    """Moment equations on the triangle in collapsed coordinates."""

    def __init__(self, deg):
        self.deg = deg
        pairs = []
        for i in range(deg + 1):
            for j in range(deg + 1 - i):
                pairs.append((i, j))
        self.ij = np.array(pairs)
        self.m = len(pairs)
        self.scale = np.sqrt((2 * self.ij[:, 0] + 1) * 2.0 * (self.ij[:, 0] + self.ij[:, 1] + 1))
        self.target = np.zeros(self.m)
        # integral of the orthonormal constant sqrt(2) over the triangle (area 1/2)
        self.target[0] = np.sqrt(2.0) / 2.0

    def basis(self, u, v):
        """Basis values and derivatives with respect to u and v; arrays (n, m)."""
        n = len(u)
        a = 2.0 * u - 1.0
        b = 2.0 * v - 1.0
        om = 1.0 - v
        val = np.empty((n, self.m))
        du = np.empty((n, self.m))
        dv = np.empty((n, self.m))
        leg = {}
        for i in range(self.deg + 1):
            leg[i] = jacobi_and_derivative(i, 0, a)
        for k, (i, j) in enumerate(self.ij):
            pi, dpi = leg[i]
            qj, dqj = jacobi_and_derivative(j, 2 * i + 1, b)
            omi = om ** i
            if i > 0:
                domi = -i * om ** (i - 1)
            else:
                domi = np.zeros(n)
            s = self.scale[k]
            val[:, k] = s * pi * omi * qj
            du[:, k] = s * 2.0 * dpi * omi * qj
            dv[:, k] = s * pi * (domi * qj + omi * 2.0 * dqj)
        return val, du, dv

    def residual(self, x, n):
        uv = x[:2 * n].reshape(n, 2)
        w = x[2 * n:3 * n]
        val, _, _ = self.basis(uv[:, 0], uv[:, 1])
        return w @ val - self.target

    def jacobian(self, x, n):
        uv = x[:2 * n].reshape(n, 2)
        w = x[2 * n:3 * n]
        val, du, dv = self.basis(uv[:, 0], uv[:, 1])
        jac = np.empty((self.m, 3 * n))
        jac[:, 0:2 * n:2] = (w[:, None] * du).T
        jac[:, 1:2 * n:2] = (w[:, None] * dv).T
        jac[:, 2 * n:3 * n] = val.T
        return jac

    def bounds(self, n):
        lo = np.zeros(3 * n)
        hi = np.r_[np.ones(2 * n), 10.0 * np.ones(n)]
        return lo, hi

    def solve(self, x0, n, max_iter, tol=1e-28):
        """Levenberg-Marquardt with bounds by projection (active set on the gradient)."""
        lo, hi = self.bounds(n)
        x = np.clip(x0, lo, hi)
        r = self.residual(x, n)
        cost = 0.5 * (r @ r)
        mu = 1e-3
        for _ in range(max_iter):
            if cost < tol:
                break
            jac = self.jacobian(x, n)
            g = jac.T @ r
            at_lo = (x <= lo + 1e-14) & (g > 0)
            at_hi = (x >= hi - 1e-14) & (g < 0)
            free = ~(at_lo | at_hi)
            jf = jac[:, free]
            wide = jf.shape[1] >= self.m
            if wide:
                a0 = jf @ jf.T
                rhs = -r
            else:
                a0 = jf.T @ jf
                rhs = -(jf.T @ r)
            k = a0.shape[0]
            t0 = np.trace(a0) / k + 1e-30
            improved = False
            for _ in range(16):
                a = a0.copy()
                a[np.diag_indices(k)] += mu * t0
                try:
                    d = cho_solve(cho_factor(a, lower=True, overwrite_a=True), rhs)
                except (LinAlgError, ValueError):
                    mu *= 10.0
                    continue
                if wide:
                    d = jf.T @ d
                xn = x.copy()
                xn[free] += d
                xn = np.clip(xn, lo, hi)
                rn = self.residual(xn, n)
                cn = 0.5 * (rn @ rn)
                if cn < cost:
                    x = xn
                    r = rn
                    cost = cn
                    mu = max(mu / 4.0, 1e-15)
                    improved = True
                    break
                mu *= 4.0
            if not improved:
                break
        return x, cost


def to_xy(x, n):
    uv = x[:2 * n].reshape(n, 2)
    w = x[2 * n:3 * n].copy()
    pts = np.c_[uv[:, 0] * (1.0 - uv[:, 1]), uv[:, 1]]
    return pts, w


def to_uv(pts):
    v = pts[:, 1]
    om = 1.0 - v
    u = np.where(om > 1e-14, pts[:, 0] / np.maximum(om, 1e-300), 0.5)
    return np.c_[np.clip(u, 0.0, 1.0), np.clip(v, 0.0, 1.0)]


def monomial_error(deg, pts, wts):
    """Largest error over all monomials x^i y^j, i + j <= deg, relative to the area 1/2."""
    from math import factorial
    err = 0.0
    for i in range(deg + 1):
        for j in range(deg + 1 - i):
            exact = factorial(i) * factorial(j) / factorial(i + j + 2)
            q = wts @ (pts[:, 0] ** i * pts[:, 1] ** j)
            err = max(err, abs(q - exact) / 0.5)
    return err


def is_valid(deg, pts, wts):
    if wts.min() <= 0:
        return False
    if pts.min() < 0 or (pts[:, 0] + pts[:, 1]).max() > 1.0 + 1e-15:
        return False
    return monomial_error(deg, pts, wts) < 1e-13


def drop_node(x, n, idx):
    """Remove node idx and give its weight to the nearest remaining node."""
    pts, w = to_xy(x, n)
    uv = x[:2 * n].reshape(n, 2)
    keep = np.ones(n, dtype=bool)
    keep[idx] = False
    near = np.argmin(np.linalg.norm(pts[keep] - pts[idx], axis=1))
    w2 = w[keep]
    w2[near] += w[idx]
    return np.r_[uv[keep].ravel(), w2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("degree", type=int)
    ap.add_argument("start_points", type=int)
    ap.add_argument("seed", type=int)
    ap.add_argument("seconds", type=float)
    ap.add_argument("--tries", type=int, default=12, help="candidates tried per elimination step")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--out", default="triangle_out")
    args = ap.parse_args()
    deg = args.degree
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    prob = Problem(deg)
    n = args.start_points
    t_start = time.time()
    cost = 1.0
    x = None
    for attempt in range(50):
        uv = rng.uniform(0.01, 0.99, (n, 2))
        # sqrt in v spreads the nodes evenly in area
        uv[:, 1] = 1.0 - np.sqrt(1.0 - uv[:, 1])
        x = np.r_[uv.ravel(), np.full(n, 0.5 / n)]
        x, cost = prob.solve(x, n, args.iters * 3)
        print("start %d cost %.2e" % (attempt, cost), flush=True)
        if cost < 1e-26:
            break
        if time.time() - t_start > args.seconds:
            break
    if cost >= 1e-26:
        print("RESULT no start rule")
        return
    best_total = None
    while time.time() - t_start < args.seconds:
        pts, wts = to_xy(x, n)
        if is_valid(deg, pts, wts):
            name = os.path.join(args.out, "tri_d%d_n%d_s%d.npy" % (deg, n, args.seed))
            np.save(name, np.c_[pts, wts])
            best_total = n
            print("FOUND deg %d N %d wmin %.2e  %.0fs" % (deg, n, wts.min(), time.time() - t_start), flush=True)
        order = np.argsort(x[2 * n:3 * n])
        success = False
        for idx in order[:args.tries]:
            if time.time() - t_start > args.seconds:
                break
            xt = drop_node(x, n, idx)
            xt, ct = prob.solve(xt, n - 1, args.iters)
            if ct < 1e-26:
                x = xt
                n -= 1
                success = True
                break
        if not success:
            break
    print("RESULT deg %d best N %s seed %d" % (deg, best_total, args.seed), flush=True)


def refine_main():
    """Start from a saved rule and try to remove one more node."""
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("degree", type=int)
    ap.add_argument("start")
    ap.add_argument("seed", type=int)
    ap.add_argument("seconds", type=float)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--out", default="triangle_out")
    args = ap.parse_args()
    deg = args.degree
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    prob = Problem(deg)
    arr = np.load(args.start)
    uv = to_uv(arr[:, :2])
    w = arr[:, 2]
    n = len(w)
    x0 = np.r_[uv.ravel(), w]
    order = np.argsort(w)
    sigmas = [0.0, 0.003, 0.01, 0.03]
    t_start = time.time()
    trial = 0
    while time.time() - t_start < args.seconds:
        idx = order[trial % n]
        sig = sigmas[(trial // n) % len(sigmas)]
        trial += 1
        xt = drop_node(x0, n, idx)
        xt[:2 * (n - 1)] += sig * rng.standard_normal(2 * (n - 1))
        xt, cost = prob.solve(xt, n - 1, args.iters)
        if cost < 1e-26:
            pts, wts = to_xy(xt, n - 1)
            if is_valid(deg, pts, wts):
                name = os.path.join(args.out, "tri_d%d_n%d_s%d.npy" % (deg, n - 1, args.seed))
                np.save(name, np.c_[pts, wts])
                print("FOUND deg %d N %d trial %d sigma %g wmin %.2e" % (deg, n - 1, trial, sig, wts.min()), flush=True)
                return
        if trial % 20 == 0:
            print("trial %d cost %.1e %.0fs" % (trial, cost, time.time() - t_start), flush=True)
    print("RESULT none")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "refine":
        refine_main()
    else:
        main()

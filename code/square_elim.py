"""Square [-1,1]^2: PI cubature search by node elimination.

Usage:
    python3 square_elim.py <degree> <start_points> <seed> <seconds> [--cs] [--po BOUND] [--out DIR]
    python3 square_elim.py refine <degree> <start.npy> <seed> <seconds> [--to-centre] [--out DIR]

Starts from <start_points> random nodes (drawn from numpy's default_rng(seed)),
solves the moment equations with a bounded Levenberg-Marquardt method
(weights >= 0, nodes in the closed square), then removes one node (or one
pair of nodes with --cs) at a time and solves again.  Every exact rule that
is found is saved as DIR/sq_d<degree>_n<N>_s<seed>.npy (columns x, y, w).

--cs: centrally symmetric rules (pairs +p, -p and a centre node when the
number of nodes is odd); only the even moments are imposed, the odd ones
vanish by symmetry.  This is the useful choice for odd degrees.

--po BOUND: PO rules: the nodes may leave the square, the bounds of the coordinates are [-BOUND, BOUND] (for example 3); the weights stay positive.  Rules are saved as
DIR/sqpo_d<degree>_n<N>_s<seed>.npy.  With --cs this gives centrally symmetric PO rules (symmetry group of order 2, rotation by 180 degrees).

Basis: products of orthonormal Legendre polynomials.  The check here is a
floating-point residual test only; it is neither a certificate nor a proof.
"""
import argparse
import sys
import os
import time
import numpy as np
from numpy.polynomial import legendre as leg
from scipy.linalg import cho_factor, cho_solve, LinAlgError


class Problem:
    """Moment equations on the square, optionally centrally symmetric."""

    def __init__(self, deg, cs):
        self.deg = deg
        self.cs = cs
        self.bound = 1.0
        pairs = []
        for i in range(deg + 1):
            for j in range(deg + 1 - i):
                if (not cs) or (i + j) % 2 == 0:
                    pairs.append((i, j))
        self.ij = np.array(pairs)
        self.scale = np.sqrt((2 * self.ij[:, 0] + 1) * (2 * self.ij[:, 1] + 1) / 4.0)
        self.m = len(pairs)
        self.target = np.zeros(self.m)
        # integral of the orthonormal constant 1/2 over the square (area 4)
        self.target[0] = 2.0
        dmat = np.zeros((deg + 1, deg + 1))
        for i in range(1, deg + 1):
            e = np.zeros(i + 1)
            e[i] = 1.0
            c = leg.legder(e)
            dmat[i, :len(c)] = c
        self.dmat = dmat
        p0 = leg.legvander(np.zeros(1), deg)[0]
        self.v0 = p0[self.ij[:, 0]] * p0[self.ij[:, 1]] * self.scale
        self.mult = 2.0 if cs else 1.0

    def unpack(self, x, n, centre):
        p = x[:2 * n].reshape(n, 2)
        w = x[2 * n:3 * n]
        wc = x[3 * n] if centre else 0.0
        return p, w, wc

    def residual(self, x, n, centre):
        p, w, wc = self.unpack(x, n, centre)
        ax = leg.legvander(p[:, 0], self.deg)[:, self.ij[:, 0]]
        ay = leg.legvander(p[:, 1], self.deg)[:, self.ij[:, 1]]
        v = ax * ay * self.scale
        r = self.mult * (w @ v) - self.target
        if centre:
            r = r + wc * self.v0
        return r

    def jacobian(self, x, n, centre):
        p, w, wc = self.unpack(x, n, centre)
        lx = leg.legvander(p[:, 0], self.deg)
        ly = leg.legvander(p[:, 1], self.deg)
        ax = lx[:, self.ij[:, 0]]
        ay = ly[:, self.ij[:, 1]]
        dx = (lx @ self.dmat.T)[:, self.ij[:, 0]]
        dy = (ly @ self.dmat.T)[:, self.ij[:, 1]]
        nun = 3 * n + (1 if centre else 0)
        jac = np.empty((self.m, nun))
        jac[:, 0:2 * n:2] = (self.mult * w[:, None] * dx * ay * self.scale).T
        jac[:, 1:2 * n:2] = (self.mult * w[:, None] * ax * dy * self.scale).T
        jac[:, 2 * n:3 * n] = (self.mult * ax * ay * self.scale).T
        if centre:
            jac[:, 3 * n] = self.v0
        return jac

    def bounds(self, n, centre):
        lo = np.r_[-self.bound * np.ones(2 * n), np.zeros(n)]
        hi = np.r_[self.bound * np.ones(2 * n), 10.0 * np.ones(n)]
        if centre:
            lo = np.r_[lo, 0.0]
            hi = np.r_[hi, 10.0]
        return lo, hi

    def solve(self, x0, n, centre, max_iter, tol=1e-28):
        """Levenberg-Marquardt with bounds by projection (active set on the gradient)."""
        lo, hi = self.bounds(n, centre)
        x = np.clip(x0, lo, hi)
        r = self.residual(x, n, centre)
        cost = 0.5 * (r @ r)
        mu = 1e-3
        for _ in range(max_iter):
            if cost < tol:
                break
            jac = self.jacobian(x, n, centre)
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
                rn = self.residual(xn, n, centre)
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

    def full_rule(self, x, n, centre):
        p, w, wc = self.unpack(x, n, centre)
        if not self.cs:
            return p.copy(), w.copy()
        pts = np.vstack([p, -p])
        wts = np.r_[w, w]
        if centre:
            pts = np.vstack([pts, np.zeros((1, 2))])
            wts = np.r_[wts, wc]
        return pts, wts


def monomial_error(deg, pts, wts):
    """Largest error over all monomials x^i y^j, i + j <= deg, relative to the area 4."""
    err = 0.0
    for i in range(deg + 1):
        for j in range(deg + 1 - i):
            exact = 0.0
            if i % 2 == 0 and j % 2 == 0:
                exact = 4.0 / ((i + 1) * (j + 1))
            q = wts @ (pts[:, 0] ** i * pts[:, 1] ** j)
            err = max(err, abs(q - exact) / 4.0)
    return err


def random_start(rng, n, cs, centre):
    """Random nodes in the square, a little away from the boundary, equal weights."""
    p = rng.uniform(-0.98, 0.98, (n, 2))
    total = 2 * n + (1 if centre else 0) if cs else n
    w = np.full(n, 4.0 / total)
    x = np.r_[p.ravel(), w]
    if centre:
        x = np.r_[x, 4.0 / total]
    return x


def drop_node(x, n, centre, idx):
    """Remove node idx (or pair idx) and give its weight to the nearest remaining node."""
    p = x[:2 * n].reshape(n, 2)
    w = x[2 * n:3 * n].copy()
    keep = np.ones(n, dtype=bool)
    keep[idx] = False
    dist = np.linalg.norm(p[keep] - p[idx], axis=1)
    near = np.argmin(dist)
    w2 = w[keep]
    w2[near] += w[idx]
    xn = np.r_[p[keep].ravel(), w2]
    if centre:
        xn = np.r_[xn, x[3 * n]]
    return xn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("degree", type=int)
    ap.add_argument("start_points", type=int)
    ap.add_argument("seed", type=int)
    ap.add_argument("seconds", type=float)
    ap.add_argument("--cs", action="store_true")
    ap.add_argument("--po", type=float, default=0.0, help="PO rules: nodes may leave the square, coordinate bounds [-PO, PO] (0: nodes stay in the square)")
    ap.add_argument("--centre", type=int, default=-1, help="1/0: force a centre node (cs only); default: by parity of start_points")
    ap.add_argument("--tries", type=int, default=12, help="candidates tried per elimination step")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--out", default="square_out")
    args = ap.parse_args()
    deg = args.degree
    cs = args.cs
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    prob = Problem(deg, cs)
    if args.po > 0:
        prob.bound = args.po
    if cs:
        centre = (args.start_points % 2 == 1) if args.centre < 0 else bool(args.centre)
        n = args.start_points // 2
    else:
        centre = False
        n = args.start_points
    t_start = time.time()
    cost = 1.0
    x = None
    for attempt in range(50):
        x = random_start(rng, n, cs, centre)
        x, cost = prob.solve(x, n, centre, args.iters * 3)
        print("start %d cost %.2e" % (attempt, cost), flush=True)
        if cost < 1e-26:
            break
    if cost >= 1e-26:
        print("RESULT no start rule")
        return
    best_total = None
    while time.time() - t_start < args.seconds:
        pts, wts = prob.full_rule(x, n, centre)
        err = monomial_error(deg, pts, wts)
        total = len(wts)
        inside = np.abs(pts).max() <= (args.po if args.po > 0 else 1.0)
        if err < 1e-13 and wts.min() > 0 and inside:
            name = os.path.join(args.out, "%s_d%d_n%d_s%d.npy" % ("sqpo" if args.po > 0 else "sq", deg, total, args.seed))
            np.save(name, np.c_[pts, wts])
            best_total = total
            print("FOUND deg %d N %d err %.1e wmin %.2e  %.0fs" % (deg, total, err, wts.min(), time.time() - t_start), flush=True)
        w = x[2 * n:3 * n]
        order = np.argsort(w)
        success = False
        for idx in order[:args.tries]:
            if time.time() - t_start > args.seconds:
                break
            xt = drop_node(x, n, centre, idx)
            xt, ct = prob.solve(xt, n - 1, centre, args.iters)
            if ct < 1e-26:
                x = xt
                n -= 1
                success = True
                break
        if not success:
            break
    print("RESULT deg %d best N %s seed %d" % (deg, best_total, args.seed), flush=True)




def refine_main():
    """Start from a saved rule and try to remove one more pair (or one node with --free).

    Usage: python3 square_elim.py refine <degree> <start.npy> <seed> <seconds> [--to-centre] [--free] [--out DIR]
    --free: the start rule is treated without symmetry and one node is removed.
    --to-centre: the removed pair becomes a centre node (start rule must have no centre).
    Each candidate pair is removed, then the system is solved from the nodes perturbed
    with Gaussian noise of several sizes drawn from default_rng(seed).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("degree", type=int)
    ap.add_argument("start")
    ap.add_argument("seed", type=int)
    ap.add_argument("seconds", type=float)
    ap.add_argument("--to-centre", action="store_true")
    ap.add_argument("--free", action="store_true")
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--out", default="square_out")
    args = ap.parse_args()
    deg = args.degree
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    prob = Problem(deg, not args.free)
    arr = np.load(args.start)
    pts = arr[:, :2]
    wts = arr[:, 2]
    if args.free:
        is_c = np.zeros(len(wts), dtype=bool)
    else:
        is_c = np.abs(pts).max(axis=1) < 1e-12
    has_c = bool(is_c.any())
    rest = pts[~is_c]
    rw = wts[~is_c]
    if args.free:
        half = np.ones(len(rw), dtype=bool)
    else:
        half = (rest[:, 0] > 1e-12) | ((np.abs(rest[:, 0]) <= 1e-12) & (rest[:, 1] > 0))
    p = rest[half]
    w = rw[half]
    n = len(w)
    wc = wts[is_c][0] if has_c else 0.0
    t_start = time.time()
    order = np.argsort(w)
    sigmas = [0.0, 0.003, 0.01, 0.03, 0.06]
    centre_new = has_c or args.to_centre
    trial = 0
    while time.time() - t_start < args.seconds:
        idx = order[trial % n]
        sig = sigmas[(trial // n) % len(sigmas)]
        trial += 1
        keep = np.ones(n, dtype=bool)
        keep[idx] = False
        q = p[keep] + sig * rng.standard_normal((n - 1, 2))
        q = np.clip(q, -1.0, 1.0)
        w2 = w[keep].copy()
        if args.to_centre and not has_c:
            c_w = 2.0 * w[idx]
        else:
            c_w = wc
            near = np.argmin(np.linalg.norm(p[keep] - p[idx], axis=1))
            w2[near] += w[idx]
        x = np.r_[q.ravel(), w2]
        if centre_new:
            x = np.r_[x, c_w]
        x, cost = prob.solve(x, n - 1, centre_new, args.iters)
        if cost < 1e-26:
            fp, fw = prob.full_rule(x, n - 1, centre_new)
            err = monomial_error(deg, fp, fw)
            if err < 1e-13 and fw.min() > 0 and np.abs(fp).max() <= 1.0:
                name = os.path.join(args.out, "sq_d%d_n%d_s%d.npy" % (deg, len(fw), args.seed))
                np.save(name, np.c_[fp, fw])
                print("FOUND deg %d N %d trial %d sigma %g err %.1e wmin %.2e" % (deg, len(fw), trial, sig, err, fw.min()), flush=True)
                return
        if trial % 20 == 0:
            print("trial %d cost %.1e %.0fs" % (trial, cost, time.time() - t_start), flush=True)
    print("RESULT none")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "refine":
        refine_main()
    else:
        main()

"""PO node elimination for the cube [-1,1]^3 and the tetrahedron (0,0,0)(1,0,0)(0,1,0)(0,0,1).

Starts from the best PI rule of the atlas (or from a given .npy file with columns x, y, z, w)
and repeatedly removes one node (delete the node or merge two close nodes, small random
perturbation, then damped Gauss-Newton / Levenberg-Marquardt with bounds).
Weights must stay positive; nodes may leave the domain (PO).
A reduced rule is accepted only after the monomial test AND the Newton-Kantorovich certificate.

Usage:
    python3 po_elim.py <cube|tetra> <degree> <seconds> <seed> [start.npy]
Output files: po_<domain>_d<degree>_n<N>.npy (columns x, y, z, w), one per accepted node count.
"""

# Imports.
import os
import sys
import glob
import json
import math
import time
import numpy as np
from scipy.linalg import solve_triangular

# Line-search solver with Cholesky steps, shared with the other programs.
os.environ["CUB_LM"] = "1"
os.environ["FASTJAC"] = "0"
# Program directory and repository root.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import cs
import dom
import cubature_atlas as CA


class PhysDom(dom.Dom):
    """Basis of the tetrahedron, triangle or square evaluated directly in physical coordinates (no Duffy map), so nodes may leave the domain."""

    def to_phys(self, Q):
        # Identity map: parameters are the physical coordinates.
        N = len(Q)
        Jm = np.zeros((N, self.d, self.d))
        for k in range(self.d):
            Jm[:, k, k] = 1.0
        return Q, Jm


class GenProb(dom.ProbG):
    """Moment system on the tetrahedron, triangle or square with nodes in a box around the domain."""

    def bounds(self, N, mode):
        # Nodes in a box BOXOUT around the domain, weights >= 0.
        d = self.d
        if self.dom.name == "square":
            a = -1.0 - BOXOUT
            b = 1.0 + BOXOUT
        else:
            a = -BOXOUT
            b = 1.0 + BOXOUT
        lo = np.r_[np.full(d * N, a), np.full(N, 0.0)]
        hi = np.r_[np.full(d * N, b), np.full(N, 50.0)]
        return lo, hi


class CubeProb(cs.Prob):
    """Moment system on the cube with nodes in [-1 - BOXOUT, 1 + BOXOUT]^3.
    With cs=True the unknowns are one node of each pair (p, -p) and its weight, plus the centre weight if center=True."""

    def bounds(self, N, mode):
        # Nodes in a box around the cube, weights >= 0.
        PB = 1.0 + BOXOUT
        lo = np.r_[np.full(3 * N, -PB), np.full(N, 0.0)]
        hi = np.r_[np.full(3 * N, PB), np.full(N, 50.0)]
        if self.center:
            lo = np.r_[lo, 0.0]
            hi = np.r_[hi, 50.0]
        return lo, hi

    def full_points(self, x, N):
        # Expand the unknowns to the full list of nodes and weights.
        P = x[:3 * N].reshape(N, 3)
        w = x[3 * N:4 * N]
        if not self.cs:
            return P, w
        Pf = np.vstack([P, -P])
        wf = np.r_[w, w]
        if self.center:
            Pf = np.vstack([Pf, np.zeros((1, 3))])
            wf = np.r_[wf, x[4 * N]]
        return Pf, wf


def duffy_inverse(name, P):
    """Physical points of the triangle or tetrahedron -> Duffy parameters q in [0, 1]^d (inverse of dom.Dom.to_phys)."""
    if name == "triangle":
        v = P[:, 1]
        u = P[:, 0] / (1.0 - v)
        return np.c_[u, v]
    w = P[:, 2]
    v = P[:, 1] / (1.0 - w)
    u = P[:, 0] / ((1.0 - v) * (1.0 - w))
    return np.c_[u, v, w]


def split_central(P, w):
    """Split a centrally symmetric rule into one node per pair (p, -p), the pair weight and the centre weight (or None)."""
    r = np.linalg.norm(P, axis=1)
    cen = r < 1e-12
    c = None
    if cen.any():
        c = float(w[cen][0])
    Q = P[~cen]
    v = w[~cen]
    # Keep the node whose first nonzero coordinate is positive.
    keep = []
    for i in range(len(Q)):
        nz = np.nonzero(np.abs(Q[i]) > 1e-12)[0][0]
        if Q[i, nz] > 0:
            keep.append(i)
    keep = np.array(keep)
    assert 2 * len(keep) == len(Q), "rule is not centrally symmetric"
    return Q[keep], v[keep], c


# How far nodes may leave the domain.
BOXOUT = float(os.environ.get("BOXOUT", "0.5"))
# Scale of the random perturbation (0 = none) and "small" = only delete the smallest weights.
NOISE = float(os.environ.get("NOISE", "1"))
CANDS = os.environ.get("CANDS", "all")
# INSIDE=1: PI search (nodes stay in the closed domain, mode "d" of dom.ProbG).
INSIDE = os.environ.get("INSIDE", "0") == "1"
MODE = "d" if INSIDE else "b"
# Solver: lm (default), lmstop (LM with early stop), chord (rank-1 downdates + chord steps, then lmstop), gn.
SOLVER = os.environ.get("SOLVER", "lm")
# Number of smallest-weight (and of random) nodes tried per chord round.
CHORDK = int(os.environ.get("CHORDK", "30"))
# Nodes removed per deletion step.
DELK = int(os.environ.get("DELK", "1"))
# Iterations of the local solver per attempt.
MI = int(os.environ.get("MI", "300"))


def solve_gn_minnorm(pr, x0, N, iters):
    """Minimum-norm Gauss-Newton for underdetermined systems: x <- x - J^T (J J^T + lam I)^-1 r, projected to the bounds."""
    lo, hi = pr.bounds(N, "b")
    x = np.clip(x0, lo, hi)
    r = pr.res(x, N)
    for it in range(iters):
        c = 0.5 * float(r @ r)
        if c < 1e-26:
            break
        J = pr.jac(x, N)
        G = J @ J.T
        lam = 1e-14 * np.trace(G) / len(G)
        G[np.diag_indices_from(G)] += lam
        L = np.linalg.cholesky(G)
        y = np.linalg.solve(L.T, np.linalg.solve(L, r))
        xn = np.clip(x - J.T @ y, lo, hi)
        rn = pr.res(xn, N)
        if float(rn @ rn) > 4.0 * float(r @ r):
            break
        x = xn
        r = rn
    class R:
        pass
    out = R()
    out.x = x
    out.cost = 0.5 * float(r @ r)
    return out


def chol_downdate(U, v):
    """Rank-1 downdate of an upper Cholesky factor in place: U^T U - v v^T. Returns False if it is not positive definite."""
    v = v.copy()
    n = len(v)
    for k in range(n):
        d = U[k, k] * U[k, k] - v[k] * v[k]
        if d <= 0.0:
            return False
        r = math.sqrt(d)
        c = r / U[k, k]
        s = v[k] / U[k, k]
        U[k, k] = r
        if k + 1 < n:
            U[k, k + 1:] = (U[k, k + 1:] - s * v[k + 1:]) / c
            v[k + 1:] = c * v[k + 1:] - s * U[k, k + 1:]
    return True


def chord_factor(pr, x, N):
    """Jacobian J at x and upper Cholesky factor U of J J^T + lam I (computed once per accepted rule)."""
    J = pr.jac(x, N)
    G = J @ J.T
    lam = 1e-13 * np.trace(G) / len(G)
    G[np.diag_indices_from(G)] += lam
    U = np.linalg.cholesky(G).T.copy()
    return J, U


def solve_chord(pr, J, U, x, N, cols, iters):
    """Delete the unknowns cols (one node): rank-1 downdates of U, then chord steps x <- x - J'^T (J' J'^T)^-1 r with fixed J'.
    Early stop when the residual does not halve within 3 steps. Returns (x', cost) or (None, inf)."""
    Ud = U.copy()
    for c in cols:
        if not chol_downdate(Ud, J[:, c]):
            return None, np.inf
    keep = np.delete(np.arange(len(x)), cols)
    Jd = J[:, keep]
    xn = x[keep].copy()
    lo, hi = pr.bounds(N - 1, MODE)
    r = pr.res(xn, N - 1)
    hist = [float(r @ r)]
    for it in range(iters):
        y = solve_triangular(Ud, solve_triangular(Ud, r, trans="T"))
        xn = xn - Jd.T @ y
        if np.any(xn < lo) or np.any(xn > hi):
            return None, np.inf
        r = pr.res(xn, N - 1)
        hist.append(float(r @ r))
        if hist[-1] < 1e-22:
            break
        # Early stop on stagnation: the squared residual must drop by 4 (residual by 2) within 3 steps.
        if len(hist) > 3 and hist[-1] > 0.25 * hist[-4]:
            return None, np.inf
    return xn, 0.5 * hist[-1]


def solve_lm_stop(pr, x0, N, maxit):
    """Levenberg-Marquardt in blocks of 15 iterations; stop as soon as a block reduces the cost by less than a factor 10."""
    x = x0
    best = np.inf
    done = 0
    r = None
    while done < maxit:
        r = pr.solve_lm(x, N, MODE, max_iter=15)
        done += 15
        if r.cost < 1e-24 or r.cost > 0.1 * best:
            break
        best = r.cost
        x = r.x
    return r


def atlas_best_pi(name, deg):
    """Smallest PI rule of the atlas for this degree (or higher degree)."""
    best = None
    pattern = os.path.join(ROOT, "rules", "minimal_pointcount_for_degree_n_PI", name, "*.json")
    for f in glob.glob(pattern):
        r = json.load(open(f))
        if r["degree"] < deg:
            continue
        if best is None or r["points"] < best["points"]:
            best = r
    X = np.array(best["nodes"], dtype=float)
    W = np.array(best["weights"], dtype=float)
    return X, W


def monomial_error(name, deg, X, W):
    """Largest absolute monomial moment error divided by the volume."""
    exps = [t for t in CA.indices(name, deg)]
    err = 0.0
    for e in exps:
        v = np.sum(W * np.prod(X ** np.array(e), axis=1))
        if name in ("cube", "square"):
            ex = 1.0
            for k in e:
                ex *= (2.0 / (k + 1)) if k % 2 == 0 else 0.0
        else:
            ex = math.prod(math.factorial(k) for k in e) / math.factorial(sum(e) + len(e))
        err = max(err, abs(v - ex))
    return err / CA.VOL[name]


def ok_rule(name, deg, X, W):
    """Monomial test first, then the certificate; returns the certificate dict or None."""
    if W.min() <= 0:
        return None
    e = monomial_error(name, deg, X, W)
    if e > 1e-12:
        return None
    c = CA.certificate(name, deg, X, W)
    print("    monomial error %.1e, certificate h=%.2e certified=%s" % (e, c["h"], c["certified"]), flush=True)
    if not c["certified"]:
        return None
    return c


def run_elim(name, deg, tl, rng, P, w):
    """Node elimination from the rule (P, w); saves every accepted rule; returns (N, nodes, weights) of the last accepted rule."""
    D = 3 if name in ("cube", "tetra") else 2
    best = (len(w), None, None)
    # Centrally symmetric search on the cube (environment CS=1).
    sym = name == "cube" and os.environ.get("CS", "0") == "1"
    tail = np.zeros(0)
    if sym:
        P, w, c = split_central(P, w)
        pr = CubeProb(deg, True, c is not None)
        if c is not None:
            tail = np.array([c])
    elif name == "cube":
        pr = CubeProb(deg, False, False)
    elif INSIDE and name in ("triangle", "tetra"):
        # PI search: Duffy parameters in [0, 1] keep every node inside the domain.
        pr = dom.ProbG(dom.Dom(name, deg))
        P = duffy_inverse(name, P)
    else:
        pr = GenProb(PhysDom(name, deg))
    N = len(w)
    print("start", name, "degree", deg, "N", N, "unknowns", (D + 1) * N + len(tail), "moments", pr.M, "symmetric", sym, flush=True)
    x = np.r_[P.ravel(), w, tail]
    t0 = time.time()
    fails = 0
    Jc = None
    Uc = None
    # Stop when one node (pair) less would give fewer unknowns than moments (generic count reached).
    while time.time() - t0 < tl and (D + 1) * (N - 1) + len(tail) >= pr.M:
        Qc = x[:D * N].reshape(N, D)
        wc = x[D * N:(D + 1) * N]
        order = np.argsort(wc)
        # Candidates: smallest weights, closest pairs, random nodes.
        cands = [("del", int(k)) for k in order[:6]]
        dm = np.linalg.norm(Qc[:, None] - Qc[None], axis=2)
        np.fill_diagonal(dm, 9.0)
        flat = np.argsort(dm, axis=None)[:12:2]
        pairs = np.dstack(np.unravel_index(flat, dm.shape))[0]
        cands += [("mer", (int(a), int(b))) for a, b in pairs if a != b]
        cands += [("del", int(k)) for k in rng.choice(N, min(4, N), replace=False)]
        rng.shuffle(cands)
        if CANDS == "small":
            cands = [("del", int(k)) for k in order[:8]]
        done = False
        if SOLVER == "chord":
            # One factorization per accepted rule; every deletion candidate costs only rank-1 downdates and chord steps.
            if fails == 0 or Jc is None:
                Jc, Uc = chord_factor(pr, x, N)
            ks = [int(k) for k in order[:CHORDK]]
            ks += [int(k) for k in rng.choice(N, min(CHORDK, N), replace=False) if int(k) not in ks]
            for k in ks:
                cols = [D * k + j for j in range(D)] + [D * N + k]
                xs, cost = solve_chord(pr, Jc, Uc, x, N, cols, MI)
                if xs is None or cost > 1e-16:
                    continue
                if cost > 1e-24:
                    rr = pr.solve_lm(xs, N - 1, MODE, max_iter=10)
                    xs = rr.x
                    cost = rr.cost
                if cost < 1e-24:
                    Xp, Wp = pr.full_points(xs, N - 1)
                    cert = ok_rule(name, deg, Xp, Wp)
                    if cert is not None:
                        np.save("po_%s_d%d_n%d.npy" % (name, deg, len(Wp)), np.c_[Xp, Wp])
                        best = (len(Wp), Xp, Wp)
                        x = xs
                        N = N - 1
                        done = True
                        fails = 0
                        Jc = None
                        print("  -> %d nodes (chord) t=%.0fs" % (len(Wp), time.time() - t0), flush=True)
                        break
                if time.time() - t0 > tl:
                    break
            if not done:
                fails += 1
                print("  round without success at %d (#%d) t=%.0fs" % (N, fails, time.time() - t0), flush=True)
                # The chord candidates are deterministic: after a failed round fall back to the perturbed LM round.
            else:
                continue
        for kind, c in cands:
            if kind == "del":
                # Remove node c and the DELK - 1 nodes with the smallest weights besides it.
                drop = [c] + [int(k) for k in order if k != c][:DELK - 1]
                keep = np.delete(np.arange(N), drop)
                Qn = Qc[keep].copy()
                wn = wc[keep] * (wc.sum() / wc[keep].sum())
            else:
                a, b = c
                keep = np.delete(np.arange(N), [a, b])
                wm = wc[a] + wc[b]
                Qn = np.vstack([Qc[keep], (Qc[a] * wc[a] + Qc[b] * wc[b]) / wm])
                wn = np.r_[wc[keep], wm]
            scale = 2.0 if name == "cube" else 1.0
            Qn = Qn + rng.normal(0, scale * NOISE * rng.choice([0.002, 0.008, 0.03]), Qn.shape)
            Nn = len(wn)
            if (D + 1) * Nn + len(tail) < pr.M:
                continue
            lo, hi = pr.bounds(Nn, MODE)
            xn = np.clip(np.r_[Qn.ravel(), wn, x[(D + 1) * N:]], lo, hi)
            if SOLVER == "gn":
                r = solve_gn_minnorm(pr, xn, Nn, MI)
            elif SOLVER in ("lmstop", "chord"):
                r = solve_lm_stop(pr, xn, Nn, MI)
            else:
                r = pr.solve_lm(xn, Nn, MODE, max_iter=MI)
            if r.cost < 1e-24:
                Xp, Wp = pr.full_points(r.x, Nn)
                cert = ok_rule(name, deg, Xp, Wp)
                if cert is not None:
                    np.save("po_%s_d%d_n%d.npy" % (name, deg, len(Wp)), np.c_[Xp, Wp])
                    best = (len(Wp), Xp, Wp)
                    x = r.x
                    N = Nn
                    done = True
                    fails = 0
                    print("  -> %d nodes (%s) t=%.0fs" % (len(Wp), kind, time.time() - t0), flush=True)
                    break
            if time.time() - t0 > tl:
                break
        if not done:
            fails += 1
            print("  round without success at %d (#%d) t=%.0fs" % (N, fails, time.time() - t0), flush=True)
    print("END", name, deg, N, flush=True)
    return best


def main():
    # Arguments.
    name = sys.argv[1]
    deg = int(sys.argv[2])
    tl = float(sys.argv[3])
    seed = int(sys.argv[4])
    rng = np.random.default_rng(seed)
    # Dimension of the domain.
    D = 3 if name in ("cube", "tetra") else 2
    # Start rule.
    if len(sys.argv) > 5:
        A = np.load(sys.argv[5])
        P = A[:, :D]
        w = A[:, D]
    else:
        P, w = atlas_best_pi(name, deg)
    best = run_elim(name, deg, tl, rng, P, w)



if __name__ == "__main__":
    main()

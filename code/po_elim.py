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


class PhysTetra(dom.Dom):
    """Tetrahedron basis evaluated directly in physical coordinates (no Duffy map), so nodes may leave the domain."""

    def to_phys(self, Q):
        # Identity map: parameters are the physical coordinates.
        N = len(Q)
        Jm = np.zeros((N, 3, 3))
        Jm[:, 0, 0] = 1.0
        Jm[:, 1, 1] = 1.0
        Jm[:, 2, 2] = 1.0
        return Q, Jm


class TetraProb(dom.ProbG):
    """Moment system on the tetrahedron with nodes in a box around the domain."""

    def bounds(self, N, mode):
        # Nodes in [-BOXOUT, 1 + BOXOUT]^3, weights >= 0.
        lo = np.r_[np.full(3 * N, -BOXOUT), np.full(N, 0.0)]
        hi = np.r_[np.full(3 * N, 1.0 + BOXOUT), np.full(N, 50.0)]
        return lo, hi


class CubeProb(cs.Prob):
    """Moment system on the cube with nodes in [-1 - BOXOUT, 1 + BOXOUT]^3."""

    def bounds(self, N, mode):
        # Nodes in a box around the cube, weights >= 0.
        PB = 1.0 + BOXOUT
        lo = np.r_[np.full(3 * N, -PB), np.full(N, 0.0)]
        hi = np.r_[np.full(3 * N, PB), np.full(N, 50.0)]
        return lo, hi

    def full_points(self, x, N):
        # Plain (non-symmetric) layout: nodes then weights.
        P = x[:3 * N].reshape(N, 3)
        w = x[3 * N:4 * N]
        return P, w


# How far nodes may leave the domain.
BOXOUT = float(os.environ.get("BOXOUT", "0.5"))
# Iterations of the local solver per attempt.
MI = int(os.environ.get("MI", "300"))


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
        if name == "cube":
            ex = 1.0
            for k in e:
                ex *= (2.0 / (k + 1)) if k % 2 == 0 else 0.0
        else:
            ex = math.prod(math.factorial(k) for k in e) / math.factorial(sum(e) + 3)
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


def main():
    # Arguments.
    name = sys.argv[1]
    deg = int(sys.argv[2])
    tl = float(sys.argv[3])
    seed = int(sys.argv[4])
    rng = np.random.default_rng(seed)
    # Problem.
    if name == "cube":
        pr = CubeProb(deg, False, False)
    else:
        pr = TetraProb(PhysTetra("tetra", deg))
    # Start rule.
    if len(sys.argv) > 5:
        A = np.load(sys.argv[5])
        P = A[:, :3]
        w = A[:, 3]
    else:
        P, w = atlas_best_pi(name, deg)
    N = len(w)
    print("start", name, "degree", deg, "N", N, "unknowns", 4 * N, "moments", pr.M, flush=True)
    x = np.r_[P.ravel(), w]
    t0 = time.time()
    fails = 0
    while time.time() - t0 < tl and N > 1:
        Qc = x[:3 * N].reshape(N, 3)
        wc = x[3 * N:4 * N]
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
        done = False
        for kind, c in cands:
            if kind == "del":
                keep = np.delete(np.arange(N), c)
                Qn = Qc[keep].copy()
                wn = wc[keep] * (wc.sum() / wc[keep].sum())
            else:
                a, b = c
                keep = np.delete(np.arange(N), [a, b])
                wm = wc[a] + wc[b]
                Qn = np.vstack([Qc[keep], (Qc[a] * wc[a] + Qc[b] * wc[b]) / wm])
                wn = np.r_[wc[keep], wm]
            scale = 2.0 if name == "cube" else 1.0
            Qn = Qn + rng.normal(0, scale * rng.choice([0.002, 0.008, 0.03]), Qn.shape)
            lo, hi = pr.bounds(N - 1, "b")
            xn = np.clip(np.r_[Qn.ravel(), wn], lo, hi)
            r = pr.solve_lm(xn, N - 1, "b", max_iter=MI)
            if r.cost < 1e-24:
                Xp, Wp = pr.full_points(r.x, N - 1)
                cert = ok_rule(name, deg, Xp, Wp)
                if cert is not None:
                    np.save("po_%s_d%d_n%d.npy" % (name, deg, N - 1), np.c_[Xp, Wp])
                    x = r.x
                    N = N - 1
                    done = True
                    fails = 0
                    print("  -> %d nodes (%s) t=%.0fs" % (N, kind, time.time() - t0), flush=True)
                    break
            if time.time() - t0 > tl:
                break
        if not done:
            fails += 1
            print("  round without success at %d (#%d) t=%.0fs" % (N, fails, time.time() - t0), flush=True)
    print("END", name, deg, N, flush=True)


if __name__ == "__main__":
    main()

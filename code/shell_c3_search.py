"""shell_c3_search: triangle quadrature rules with exactly Z nodes from Gauss-shell patterns, C3 symmetric search, no noise.
Z = ceil(M / 3) with M = (n + 1)(n + 2) / 2 the number of moment equations.  The nodes are free in the plane (no search inside the triangle, no clipping); the weights are positive (w = v^2).
For every shell pattern (Gauss points per edge, outside to inside, steps of -2 or -3 points between neighbouring shells, exactly Z nodes):
  1. pattern   fully symmetric (S3) node set: the ring of every shell is made of the Gauss points of its edges, the next shell starts at the middle of the four nodes next to each corner.
  2. S3        the pattern is improved by Levenberg-Marquardt steps on the C3 reduced system (a start with S3 symmetry keeps S3 symmetry in exact arithmetic).
  3. rotation  the non-corner nodes of the outer shell slide by 1/8 of their spacing along their edge; the innermost shells (--turn, default 2; "all" = every inner shell) turn about the centroid by 1/8 of
               the angle that moves their nodes by one node spacing.  The result is C3 symmetric but no longer S3 symmetric.
  4. C3        Levenberg-Marquardt on the C3 reduced system (one representative per orbit of the rotation group C3, the centre node if there is one), float64 on the CPU.
The run of a pattern stops when the residual is below 1e-12 (a solution), or when over the last --win steps neither the residual fell by at least --gain (relative) nor any unknown changed by more than --dq.
The search ends at the first solution.  Every pattern writes <out>/shell_c3_<deg>_<pattern>_<turn>.json (stages pattern, S3, rotation, C3 with all nodes, residuals, shell of every node, command) so that the way
pattern -> rotation -> best C3 approximation can be drawn; a solution is also written with all digits to <out>/solution_<deg>_<N>.json.  A summary line per pattern goes to <out>/summary.txt.
Usage: python code/shell_c3_search.py --deg 36 [--out shell_c3_out] [--turn 2,all] [--part 0 --parts 1] [--win 15 --gain 0.05 --dq 1e-3] [--steps 600] [--max-patterns 50]
Needs numpy and scipy (code/cubature_atlas.py)."""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cubature_atlas as C

CORNERS = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
SQ3 = np.sqrt(3.0)
CENTROID = np.array([1.0 / 3.0, 1.0 / 3.0])


def nodes_of_shell(m):
    """number of nodes of a shell with m Gauss points per edge"""
    if m <= 1:
        return 1
    return 3 * (m - 1)


def gauss01(m):
    return (np.polynomial.legendre.leggauss(m)[0] + 1) / 2


def ring(A, m):
    """nodes of the ring with corner points A (3 x 2) and m Gauss points per edge; the Gauss points next to a corner are merged to one node"""
    if m <= 1:
        return np.array([np.mean(A, axis=0)])
    g = gauss01(m)
    edges = [[A[e] + t * (A[(e + 1) % 3] - A[e]) for t in g] for e in range(3)]
    out = []
    for e in range(3):
        previous = edges[(e - 1) % 3]
        out.append((previous[m - 1] + edges[e][0]) / 2)
        out.extend(edges[e][1:m - 1])
    return np.array(out)


def next_corners(R, m):
    """corner points of the next shell: middle of the four ring nodes around every corner"""
    if m <= 1:
        return [R[0]] * 3
    k = m - 1
    n = len(R)
    return [np.mean([R[(e * k + o) % n] for o in (1, 2, -1, -2)], axis=0) for e in range(3)]


def to_plane(x):
    """(x, y) of the triangle (0,0), (1,0), (0,1) -> equilateral frame"""
    return np.stack([x[..., 0] + 0.5 * x[..., 1], SQ3 / 2 * x[..., 1]], axis=-1)


def from_plane(p):
    y = p[..., 1] * 2 / SQ3
    return np.stack([p[..., 0] - 0.5 * y, y], axis=-1)


def turn(x, angle):
    """rotation about the centroid (angle in the equilateral frame)"""
    c = to_plane(CENTROID)
    p = to_plane(x) - c
    ca = np.cos(angle)
    sa = np.sin(angle)
    q = np.stack([ca * p[..., 0] - sa * p[..., 1], sa * p[..., 0] + ca * p[..., 1]], axis=-1) + c
    return from_plane(q)


def expand(reps):
    """all nodes of the C3 orbits of the representatives: (x, y), (l0, l1), (l2, l0) with l0 = 1 - x - y, l1 = x, l2 = y"""
    out = []
    for x in reps:
        l0 = 1 - x[0] - x[1]
        l1 = x[0]
        l2 = x[1]
        out.extend([[l1, l2], [l0, l1], [l2, l0]])
    return np.array(out)


class Pattern:
    """Gauss shell pattern: representatives of the orbits, shell index of every representative, centre node flag, node spacing angle per shell"""

    def __init__(self, seq):
        self.seq = list(seq)
        A = [v.copy() for v in CORNERS]
        reps = []
        idx = []
        centre = False
        unit = []
        c = to_plane(CENTROID)
        for j, m in enumerate(self.seq):
            R = ring(A, m)
            if m <= 1:
                centre = True
                unit.append(0.0)
            else:
                reps.extend(R[:m - 1])
                idx.extend([j] * (m - 1))
                P = to_plane(R)
                d = np.linalg.norm(P[:, None] - P[None], axis=2) + 9 * np.eye(len(P))
                unit.append(float(d.min(axis=1).mean() / np.linalg.norm(P - c, axis=1).mean()))
            A = next_corners(R, m)
        self.reps = np.array(reps)
        self.idx = np.array(idx)
        self.centre = centre
        self.unit = np.array(unit)
        self.k = len(reps)
        self.N = 3 * self.k + (1 if centre else 0)
        self.shell = np.repeat(self.idx, 3).tolist() + ([len(self.seq) - 1] if centre else [])

    def nodes(self):
        P = expand(self.reps)
        if self.centre:
            P = np.vstack([P, CENTROID])
        return P

    def c3_orbits(self, reps=None):
        """orbits of the rotation group C3 (generic orbits in the order of the representatives, then the centre node)"""
        reps = self.reps if reps is None else reps
        orbits = [("g3", [x[0], x[1]]) for x in reps]
        if self.centre:
            orbits.append(("ce", []))
        return orbits

    def spacing_outer(self):
        """typical spacing of the representatives of the outer shell along its edge (barycentric units)"""
        sel = np.where(self.idx == 0)[0][1:]
        return float(np.median(np.diff(np.sort(self.reps[sel, 0])))) if len(sel) > 1 else 0.0


IDENT = (np.eye(2), np.zeros(2))
ROT_A = (np.array([[-1.0, -1.0], [1.0, 0.0]]), np.array([1.0, 0.0]))        # (x, y) -> (1 - x - y, x)
ROT_B = (np.array([[0.0, 1.0], [-1.0, -1.0]]), np.array([0.0, 1.0]))        # (x, y) -> (y, 1 - x - y)
SWAP = (np.array([[0.0, 1.0], [1.0, 0.0]]), np.zeros(2))                    # (x, y) -> (y, x)


def compose(f, g):
    """f after g, both affine maps (M, t)"""
    return (f[0] @ g[0], f[0] @ g[1] + f[1])


IMAGES = {
    "g3": [IDENT, ROT_A, ROT_B],                                                           # generic orbit of C3: 3 nodes
    "g6": [IDENT, ROT_A, ROT_B, compose(IDENT, SWAP), compose(ROT_A, SWAP), compose(ROT_B, SWAP)],   # generic orbit of S3: 6 nodes
    "ax": [(np.array([[1.0], [1.0]]), np.zeros(2)), (np.array([[-2.0], [1.0]]), np.array([1.0, 0.0])), (np.array([[1.0], [-2.0]]), np.array([0.0, 1.0]))],   # node on a mirror line: (s, s), (1 - 2s, s), (s, 1 - 2s)
    "ce": [(np.zeros((2, 0)), np.array([1.0 / 3.0, 1.0 / 3.0]))],
}
NPARAM = {"g3": 2, "g6": 2, "ax": 1, "ce": 0}


class Reduced:
    """Reduced unknowns q = [v (one per orbit), parameters of the orbits (x, y of the representative; s for a node on a mirror line; none for the centre)]; weights w = v^2.
    orbits: list of (kind, parameter values) with kind g3 (generic orbit of C3), g6 (generic orbit of S3), ax (three nodes on the mirror lines of S3), ce (centre node)."""

    def __init__(self, orbits):
        self.orbits = orbits
        self.n_orb = len(orbits)
        self.N = sum(len(IMAGES[kind]) for kind, par in orbits)
        self.pq = self.n_orb + sum(NPARAM[kind] for kind, par in orbits)
        self.AX = np.zeros((2 * self.N, self.pq))
        self.bX = np.zeros(2 * self.N)
        self.orb = np.zeros(self.N, dtype=int)
        col = self.n_orb
        i = 0
        for j, (kind, par) in enumerate(orbits):
            npar = NPARAM[kind]
            for M2, t in IMAGES[kind]:
                self.orb[i] = j
                for c in range(2):
                    if npar:
                        self.AX[2 * i + c, col:col + npar] = M2[c]
                    self.bX[2 * i + c] = t[c]
                i += 1
            col += npar

    def q_from(self, weights):
        """weights: weight of every orbit"""
        pars = []
        for kind, par in self.orbits:
            pars.extend(np.atleast_1d(par).tolist() if NPARAM[kind] else [])
        return np.concatenate([np.sqrt(np.asarray(weights, float)), np.array(pars, float)])

    def nodes_weights(self, q):
        X = (self.AX @ q + self.bX).reshape(self.N, 2)
        W = q[:self.n_orb][self.orb] ** 2
        return X, W

    def jacobian(self, B, q, X, W):
        J = C.jacobian(B, X, W)
        N = self.N
        Jq = J[:, :2 * N] @ self.AX
        v = q[:self.n_orb]
        DW = np.zeros((N, self.pq))
        DW[np.arange(N), self.orb] = 2 * v[self.orb]
        return Jq + J[:, 2 * N:] @ DW


def barycentric(P):
    return np.stack([1 - P[:, 0] - P[:, 1], P[:, 0], P[:, 1]], axis=1)


def s3_orbits(P, tol=1e-9):
    """orbits of the full symmetry group S3 of a node set P that is S3 symmetric: generic (6 nodes), on a mirror line (3 nodes), centre"""
    L = barycentric(P)
    used = np.zeros(len(P), dtype=bool)
    out = []
    for i in range(len(P)):
        if used[i]:
            continue
        key = np.sort(L[i])
        same = np.where(np.abs(np.sort(L, axis=1) - key).max(axis=1) < tol)[0]
        used[same] = True
        if len(same) == 1:
            out.append(("ce", []))
        elif len(same) == 3:
            s = float(np.median(L[i]))        # the repeated barycentric coordinate; the odd one is 1 - 2s
            out.append(("ax", [s]))
        elif len(same) == 6:
            out.append(("g6", [P[i][0], P[i][1]]))
        else:
            raise ValueError("node set is not S3 symmetric (orbit of %d nodes)" % len(same))
    return out


def lm(B, red, q, steps, win, gain, dq_tol, target=1e-13, lam0=1e-1):
    """Levenberg-Marquardt; stops at a solution, after `steps` steps, or when over the last `win` steps neither the residual fell by `gain` (relative) nor an unknown changed by more than dq_tol"""
    X, W = red.nodes_weights(q)
    r = C.residual(B, X, W)
    cost = float(r @ r)
    lam = lam0
    hist = [(cost, q.copy())]
    reason = "steps"
    used = 0
    for it in range(steps):
        used = it + 1
        J = red.jacobian(B, q, X, W)
        g = J.T @ r
        H = J.T @ J
        accepted = False
        for _ in range(25):
            try:
                d = -np.linalg.solve(H + lam * np.eye(len(q)), g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            qn = q + d
            Xn, Wn = red.nodes_weights(qn)
            rn = C.residual(B, Xn, Wn)
            cn = float(rn @ rn)
            if np.isfinite(cn) and cn < cost:
                q, X, W, r, cost = qn, Xn, Wn, rn, cn
                lam = max(lam * 0.3, 1e-15)
                accepted = True
                break
            lam *= 5
        hist.append((cost, q.copy()))
        if np.sqrt(cost) < target:
            reason = "solution"
            break
        if not accepted:
            reason = "no step"
            break
        if len(hist) > win:
            cost_ref, q_ref = hist[-1 - win]
            slow = np.sqrt(cost) > (1 - gain) * np.sqrt(cost_ref)
            still = float(np.abs(q - q_ref).max()) < dq_tol
            if slow and still:
                reason = "stagnation"
                break
    return q, float(np.sqrt(cost)), used, reason


def enumerate_patterns(n, target, offsets):
    """shell sequences m0 > m1 > ... with steps of 2 or 3 points per edge (1 to the centre from 2), the last shell with at most 3 points and exactly `target` nodes"""
    m0 = -(-(n + 1) // 2)
    found = []

    def rec(seq, total):
        m = seq[-1]
        if total == target and m <= 3:
            found.append(list(seq))
        if total >= target:
            return
        steps = (1, 2, 3) if m == 2 else (2, 3)
        for s in steps:
            if m - s >= 1:
                seq.append(m - s)
                rec(seq, total + nodes_of_shell(m - s))
                seq.pop()

    for off in offsets:
        rec([m0 + off], nodes_of_shell(m0 + off))
    found.sort(key=lambda q: (q[0] != m0, sum(1 for a, b in zip(q, q[1:]) if a - b != 2), len(q), q))
    return found


def rotate(pat, P, mode):
    """slide the outer shell and turn the inner shells: the S3 approximation P (all nodes in the order of the pattern) becomes C3 symmetric"""
    P = P.copy()
    shell = np.array(pat.shell)
    L = np.stack([1 - P[:, 0] - P[:, 1], P[:, 0], P[:, 1]], axis=1)
    amount = pat.spacing_outer() / 8.0
    for n in range(len(P)):
        if shell[n] == 0 and n >= 3:       # nodes 0, 1, 2 are the merged corner node of the outer shell and its two rotations
            i = int(np.argmin(L[n]))
            L[n][(i + 1) % 3] -= amount
            L[n][(i + 2) % 3] += amount
    P = L[:, 1:]
    n_shells = len(pat.seq)
    inner = [j for j in range(1, n_shells) if pat.seq[j] > 1]
    if mode != "all":
        inner = inner[-int(mode):]
    for j in inner:
        sel = shell == j
        P[sel] = turn(P[sel], -pat.unit[j] / 8.0)
    return P


def match(P_ref, P):
    """P[c[i]] is the node of P that belongs to the node P_ref[i] (optimal one to one assignment)"""
    D = np.linalg.norm(P_ref[:, None] - P[None], axis=2)
    rows, cols = linear_sum_assignment(D)
    return cols


def run_pattern(B, pat, mode, a):
    P0 = pat.nodes()
    red_s = Reduced(s3_orbits(P0))
    q = red_s.q_from(np.full(red_s.n_orb, 0.5 / pat.N))
    q, res, used, why = lm(B, red_s, q, a.s3_steps, 10 ** 9, 0.0, 0.0)
    Xs, Ws = red_s.nodes_weights(q)
    order = match(P0, Xs)
    X = Xs[order]
    W = Ws[order]
    stages = [{"stage": "pattern", "residual": None, "nodes": np.round(P0, 6).tolist()}]
    stages.append({"stage": "S3", "residual": res, "nodes": np.round(X, 6).tolist()})
    X2 = rotate(pat, X, mode)
    stages.append({"stage": "rotation", "residual": None, "nodes": np.round(X2, 6).tolist()})
    k = pat.k
    red_c = Reduced(pat.c3_orbits(X2[0:3 * k:3]))
    w_orb = np.concatenate([W[0:3 * k:3], W[3 * k:]])
    q2 = red_c.q_from(w_orb)
    q2, res2, used2, why2 = lm(B, red_c, q2, a.steps, a.win, a.gain, a.dq)
    X3, W3 = red_c.nodes_weights(q2)
    stages.append({"stage": "C3", "residual": res2, "nodes": np.round(X3, 6).tolist()})
    solved = res2 < 1e-12 and bool((W3 > 0).all())
    return stages, X3, W3, res2, used2, why2, solved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deg", type=int, required=True)
    ap.add_argument("--out", default="shell_c3_out")
    ap.add_argument("--turn", default="2,all", help="comma separated: number of innermost shells that turn, or all")
    ap.add_argument("--offsets", default="0", help="comma separated offsets of the first shell relative to ceil((deg + 1) / 2), for example 0,-1,1")
    ap.add_argument("--target", type=int, default=0, help="number of nodes (default Z = ceil(M / 3))")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--s3-steps", type=int, default=60)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--win", type=int, default=15)
    ap.add_argument("--gain", type=float, default=0.05)
    ap.add_argument("--dq", type=float, default=1e-3)
    ap.add_argument("--max-patterns", type=int, default=50)
    a = ap.parse_args()
    n = a.deg
    M = (n + 1) * (n + 2) // 2
    Z = -(-M // 3)
    target = a.target or Z
    os.makedirs(a.out, exist_ok=True)
    offsets = [int(s) for s in a.offsets.split(",")]
    patterns = enumerate_patterns(n, target, offsets)[:a.max_patterns]
    patterns = patterns[a.part::a.parts]
    B = C.Basis("triangle", n)
    summary = os.path.join(a.out, "summary.txt")
    print("degree %d: M = %d, Z = %d, target %d nodes, %d shell patterns in this part" % (n, M, Z, target, len(patterns)), flush=True)
    for seq in patterns:
        pat = Pattern(seq)
        diffs = "".join(" -%d" % (x - y) for x, y in zip(seq, seq[1:]))
        for mode in a.turn.split(","):
            t0 = time.time()
            stages, X, W, res, used, why, solved = run_pattern(B, pat, mode, a)
            tag = "_".join(str(m) for m in seq)
            cmd = "python code/shell_c3_search.py --deg %d --turn %s --offsets %s --steps %d --win %d --gain %g --dq %g" % (n, mode, a.offsets, a.steps, a.win, a.gain, a.dq)
            data = {"degree": n, "N": pat.N, "seq": seq, "shell": pat.shell, "stages": stages, "solved_by": "C3" if solved else None, "turn": mode,
                    "repro": {"command": cmd + " (shells %s)" % seq}}
            with open(os.path.join(a.out, "shell_c3_%d_%s_turn%s.json" % (n, tag, mode)), "w") as fh:
                json.dump(data, fh)
            line = "degree %d, N %d, shells %d%s, turn %s: S3 residual %.2e, C3 residual %.2e after %d steps (%s), %.0fs%s" % (
                n, pat.N, seq[0], diffs, mode, stages[1]["residual"], res, used, why, time.time() - t0, "  *** SOLUTION" if solved else "")
            print(line, flush=True)
            with open(summary, "a") as fh:
                fh.write(line + "\n")
            if solved:
                L = np.c_[1 - X.sum(1), X]
                sol = {"degree": n, "N": pat.N, "seq": seq, "turn": mode, "nodes": X.tolist(), "weights": W.tolist(), "residual": res,
                       "min_barycentric": float(L.min()), "nodes_outside": int((L.min(axis=1) < 0).sum()), "command": cmd}
                with open(os.path.join(a.out, "solution_%d_%d.json" % (n, pat.N)), "w") as fh:
                    json.dump(sol, fh)
                print("solution: min weight %.2e, nodes outside the triangle %d" % (W.min(), sol["nodes_outside"]), flush=True)
                return
    print("no solution for degree %d in this part" % n, flush=True)


if __name__ == "__main__":
    main()

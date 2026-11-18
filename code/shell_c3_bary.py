"""shell_c3_bary: like code/shell_c3_search.py, but the nodes stay inside the triangle by construction: the barycentric coordinates of every node are b_i = s_i^2 / (s_0^2 + s_1^2 + s_2^2)
with free numbers s_i, so every solution is a PI rule (all nodes inside, all weights positive, w = v^2).  Pipeline per Gauss shell pattern with exactly Z nodes (steps of -2 or -3 points per edge):
  pattern -> S3 approximation (orbits of the full symmetry group; the pattern is first pulled into the triangle) -> rotation (outer shell slides by 1/8 spacing along its edge, the --turn innermost shells
  turn, then pulled into the triangle again) -> C3 symmetric Levenberg-Marquardt (one representative per orbit of the rotation group), float64, CPU, no noise.
A run of a pattern stops at a solution (residual below 1e-12), or when over the last --win steps neither the residual fell by --gain (relative) nor an unknown changed by more than --dq.
Writes <out>/shell_c3_bary_<deg>_<pattern>_turn<mode>.json (stages pattern, S3, rotation, C3 with nodes and residuals), <out>/summary.txt and, at the first solution, <out>/solution_<deg>_<N>.json.
--auto: runs all variants one after the other (default settings; first shell one below/above and other turn counts; more steps with a longer stagnation window) with several processes (--procs, default 4, one
thread each) until the first solution; if there is no pattern with exactly Z nodes it uses Z + 1.
Usage: python code/shell_c3_bary.py --deg 36 --auto [--out shell_c3_bary_out] [--procs 4]   or   python code/shell_c3_bary.py --deg 36 --target 235 --turn 2,all --offsets 0,-1,1 [--part 0 --parts 4]
Needs numpy and scipy (code/cubature_atlas.py) and code/shell_c3_search.py."""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cubature_atlas as C
import shell_c3_search as SC

NODES_OF = {"g3": 3, "g6": 6, "ax": 3, "ce": 1}
NPARAM = {"g3": 3, "g6": 3, "ax": 2, "ce": 0}
PAIRS = {"g3": [(1, 2), (0, 1), (2, 0)], "g6": [(1, 2), (0, 1), (2, 0), (2, 1), (0, 2), (1, 0)]}     # node (x, y) = (b_i, b_j) of the orbit images
EPS = 1e-3


def pull_inside(P, eps=EPS):
    """barycentric coordinates at least eps, renormalised"""
    L = np.stack([1 - P[..., 0] - P[..., 1], P[..., 0], P[..., 1]], axis=-1).clip(eps, None)
    L = L / L.sum(axis=-1, keepdims=True)
    return L[..., 1:]


def bary_of(P):
    """barycentric coordinates (l0, l1, l2) of the nodes"""
    return np.stack([1 - P[:, 0] - P[:, 1], P[:, 0], P[:, 1]], axis=1)


class Bary:
    """unknowns q = [v (one per orbit), s of the orbits]; weights w = v^2; generic orbits: b = s^2 / sum s^2 (3 numbers s), node (x, y) = (b_i, b_j) for the pairs of the orbit images;
    orbits on a mirror line (kind ax): barycentric coordinates (a, a, c) with a = s_1^2 / T, c = s_2^2 / T, T = 2 s_1^2 + s_2^2 (2 numbers s); centre node: fixed"""

    def __init__(self, orbits):
        self.orbits = orbits
        self.n_orb = len(orbits)
        self.N = sum(NODES_OF[k] for k, s in orbits)
        self.pq = self.n_orb + sum(NPARAM[k] for k, s in orbits)
        self.orb = np.concatenate([np.full(NODES_OF[k], j) for j, (k, s) in enumerate(orbits)])

    def q_from(self, weights):
        pars = []
        for k, s in self.orbits:
            pars.extend(list(np.atleast_1d(s)) if NPARAM[k] else [])
        return np.concatenate([np.sqrt(np.asarray(weights, float)), np.array(pars, float)])

    def _gen(self, s, kind):
        """nodes (n, 2) and derivative d(x, y)/ds (n, 2, 3) of a generic orbit"""
        S = float(s @ s)
        b = s * s / S
        db = 2 * np.diag(s) / S - 2 * np.outer(s * s, s) / S ** 2         # db[i, j] = d b_i / d s_j
        pairs = PAIRS[kind]
        X = np.array([[b[i], b[j]] for i, j in pairs])
        D = np.array([[db[i], db[j]] for i, j in pairs])
        return X, D

    def _ax(self, s):
        s1, s2 = s
        T = 2 * s1 * s1 + s2 * s2
        a = s1 * s1 / T
        c = s2 * s2 / T
        da = np.array([2 * s1 / T - s1 * s1 * 4 * s1 / T ** 2, -s1 * s1 * 2 * s2 / T ** 2])
        dc = np.array([-s2 * s2 * 4 * s1 / T ** 2, 2 * s2 / T - s2 * s2 * 2 * s2 / T ** 2])
        X = np.array([[a, a], [c, a], [a, c]])
        D = np.array([[da, da], [dc, da], [da, dc]])
        return X, D

    def _orbit(self, kind, s):
        if kind == "ce":
            return np.array([[1 / 3, 1 / 3]]), np.zeros((1, 2, 0))
        if kind == "ax":
            return self._ax(s)
        return self._gen(s, kind)

    def nodes_weights(self, q):
        X = []
        col = self.n_orb
        for k, s0 in self.orbits:
            n = NPARAM[k]
            Xo, D = self._orbit(k, q[col:col + n])
            X.append(Xo)
            col += n
        return np.vstack(X), q[:self.n_orb][self.orb] ** 2

    def jacobian(self, B, q, X, W):
        J = C.jacobian(B, X, W)
        N = self.N
        DX = np.zeros((2 * N, self.pq))
        i = 0
        col = self.n_orb
        for k, s0 in self.orbits:
            n = NPARAM[k]
            Xo, D = self._orbit(k, q[col:col + n])
            for m in range(len(Xo)):
                if n:
                    DX[2 * i:2 * i + 2, col:col + n] = D[m]
                i += 1
            col += n
        DW = np.zeros((N, self.pq))
        DW[np.arange(N), self.orb] = 2 * q[:self.n_orb][self.orb]
        return J[:, :2 * N] @ DX + J[:, 2 * N:] @ DW


def s_from_bary(kind, b):
    """starting numbers s for a node with barycentric coordinates b (pulled into the triangle before)"""
    if kind == "ax":
        a = float(np.median(b))                       # the repeated coordinate
        c = 1 - 2 * a
        return [np.sqrt(a), np.sqrt(c)]
    return list(np.sqrt(b / b.sum()))


def s3_orbits_bary(P):
    """S3 orbits of the (pulled in, S3 symmetric) node set P with starting numbers s"""
    out = []
    for kind, par in SC.s3_orbits(P):
        if kind == "ce":
            out.append(("ce", []))
        elif kind == "ax":
            out.append(("ax", s_from_bary("ax", np.array([par[0], par[0], 1 - 2 * par[0]]))))
        else:
            x, y = par
            out.append(("g6", s_from_bary("g6", np.array([1 - x - y, x, y]))))
    return out


def run_pattern(B, pat, mode, a):
    P0 = pull_inside(pat.nodes())
    red_s = Bary(s3_orbits_bary(P0))
    q = red_s.q_from(np.full(red_s.n_orb, 0.5 / pat.N))
    q, res, used, why = SC.lm(B, red_s, q, a.s3_steps, 10 ** 9, 0.0, 0.0)
    Xs, Ws = red_s.nodes_weights(q)
    order = SC.match(P0, Xs)
    X = Xs[order]
    W = Ws[order]
    stages = [{"stage": "pattern", "residual": None, "nodes": np.round(P0, 6).tolist()}, {"stage": "S3", "residual": res, "nodes": np.round(X, 6).tolist()}]
    X2 = pull_inside(SC.rotate(pat, X, mode))
    stages.append({"stage": "rotation", "residual": None, "nodes": np.round(X2, 6).tolist()})
    k = pat.k
    orbits = [("g3", s_from_bary("g3", bary_of(X2[3 * j:3 * j + 1])[0])) for j in range(k)] + ([("ce", [])] if pat.centre else [])
    red_c = Bary(orbits)
    q2 = red_c.q_from(np.concatenate([W[0:3 * k:3], W[3 * k:]]))
    q2, res2, used2, why2 = SC.lm(B, red_c, q2, a.steps, a.win, a.gain, a.dq)
    X3, W3 = red_c.nodes_weights(q2)
    stages.append({"stage": "C3", "residual": res2, "nodes": np.round(X3, 6).tolist()})
    solved = res2 < 1e-12 and bool((W3 > 0).all())
    return stages, X3, W3, res2, used2, why2, solved


_STATE = {}


def init_worker(deg):
    _STATE["B"] = C.Basis("triangle", deg)


def job(args):
    deg, seq, mode, params = args
    a = argparse.Namespace(**params)
    pat = SC.Pattern(seq)
    t0 = time.time()
    stages, X, W, res, used, why, solved = run_pattern(_STATE["B"], pat, mode, a)
    return {"seq": seq, "mode": mode, "N": pat.N, "shell": pat.shell, "stages": stages, "res": res, "used": used, "why": why, "solved": solved,
            "X": X.tolist(), "W": W.tolist(), "seconds": time.time() - t0, "s3": stages[1]["residual"]}


def save_result(a, deg, r, cmd):
    seq = r["seq"]
    tag = "_".join(str(m) for m in seq)
    data = {"degree": deg, "N": r["N"], "seq": seq, "shell": r["shell"], "stages": r["stages"], "solved_by": "C3" if r["solved"] else None, "turn": r["mode"],
            "repro": {"command": cmd + " (shells %s)" % seq}}
    with open(os.path.join(a.out, "shell_c3_bary_%d_%s_turn%s.json" % (deg, tag, r["mode"])), "w") as fh:
        json.dump(data, fh)
    diffs = "".join(" -%d" % (x - y) for x, y in zip(seq, seq[1:]))
    line = "degree %d, N %d, shells %d%s, turn %s: S3 residual %.2e, C3 residual %.2e after %d steps (%s), %.0fs%s" % (
        deg, r["N"], seq[0], diffs, r["mode"], r["s3"], r["res"], r["used"], r["why"], r["seconds"], "  *** SOLUTION" if r["solved"] else "")
    print(line, flush=True)
    with open(os.path.join(a.out, "summary.txt"), "a") as fh:
        fh.write(line + "\n")
    if r["solved"]:
        X = np.array(r["X"])
        W = np.array(r["W"])
        L = bary_of(X)
        sol = {"degree": deg, "N": r["N"], "seq": seq, "turn": r["mode"], "nodes": X.tolist(), "weights": W.tolist(), "residual": r["res"], "min_barycentric": float(L.min()),
               "nodes_outside": int((L.min(axis=1) < 0).sum()), "command": cmd}
        with open(os.path.join(a.out, "solution_%d_%d.json" % (deg, r["N"])), "w") as fh:
            json.dump(sol, fh)
        print("solution: min weight %.2e, smallest barycentric coordinate %.2e (all nodes inside the triangle)" % (W.min(), L.min()), flush=True)


def run_variant(a, deg, target, offsets, modes, steps, win, gain, dq, procs, label, pats=None):
    from multiprocessing import Pool
    if pats is None:
        pats = SC.enumerate_patterns(deg, target, offsets)[:a.max_patterns][a.part::a.parts]
    print("%s: degree %d, target %d nodes, %d shell patterns, turn %s, steps %d" % (label, deg, target, len(pats), modes, steps), flush=True)
    if not pats:
        return None
    params = {"s3_steps": a.s3_steps, "steps": steps, "win": win, "gain": gain, "dq": dq}
    jobs = [(deg, seq, m, params) for seq in pats for m in modes]
    cmd = "python code/shell_c3_bary.py --deg %d --target %d --turn %s --offsets %s --steps %d --win %d --gain %g --dq %g" % (deg, target, ",".join(modes), ",".join(str(o) for o in offsets), steps, win, gain, dq)
    found = None
    if procs <= 1:
        init_worker(deg)
        for j in jobs:
            r = job(j)
            save_result(a, deg, r, cmd.replace("--turn %s" % ",".join(modes), "--turn %s" % r["mode"]))
            if r["solved"]:
                return r
        return None
    with Pool(procs, initializer=init_worker, initargs=(deg,)) as pool:
        for r in pool.imap_unordered(job, jobs):
            save_result(a, deg, r, cmd.replace("--turn %s" % ",".join(modes), "--turn %s" % r["mode"]))
            if r["solved"]:
                found = r
                pool.terminate()
                break
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deg", type=int, required=True)
    ap.add_argument("--out", default="shell_c3_bary_out")
    ap.add_argument("--auto", action="store_true", help="all variants one after the other with several processes until the first solution")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--target", type=int, default=0, help="number of nodes (default Z, in --auto Z and then Z + 1)")
    ap.add_argument("--seq", default="", help="run only this shell pattern (points per edge of the shells, outside to inside, comma separated) with the --turn modes, for reproducing a solution")
    ap.add_argument("--turn", default="2,all")
    ap.add_argument("--offsets", default="0,-1,1")
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
    Z = -(-((n + 1) * (n + 2) // 2) // 3)
    os.makedirs(a.out, exist_ok=True)
    if a.seq:
        pats = [[int(s) for s in a.seq.split(",")]]
        r = run_variant(a, n, a.target or sum(SC.nodes_of_shell(m) for m in pats[0]), [0], a.turn.split(","), a.steps, a.win, a.gain, a.dq, 1, "pattern", pats)
        print("solution found" if r else "no solution", flush=True)
        return
    if not a.auto:
        target = a.target or Z
        r = run_variant(a, n, target, [int(o) for o in a.offsets.split(",")], a.turn.split(","), a.steps, a.win, a.gain, a.dq, 1, "run")
        print("solution found" if r else "no solution", flush=True)
        return
    nprocs = max(1, min(a.procs, os.cpu_count() or 1))
    variants = [("default", [0, -1, 1], ["2", "all"], 600, 15, 0.05, 1e-3),
                ("more first shells and turns", [-2, -1, 0, 1, 2], ["1", "2", "3", "all"], 600, 15, 0.05, 1e-3),
                ("longer runs", [0, -1, 1], ["1", "2", "all"], 1500, 40, 0.01, 1e-4)]
    targets = [a.target] if a.target else [Z, Z + 1]
    for target in targets:
        if not SC.enumerate_patterns(n, target, [-2, -1, 0, 1, 2]):
            print("no shell pattern with %d nodes" % target, flush=True)
            continue
        for label, offs, modes, steps, win, gain, dq in variants:
            r = run_variant(a, n, target, offs, modes, steps, win, gain, dq, nprocs, "%s (target %d)" % (label, target))
            if r:
                print("solution found: %s" % os.path.join(a.out, "solution_%d_%d.json" % (n, r["N"])), flush=True)
                return
    print("no solution found", flush=True)


if __name__ == "__main__":
    main()

"""shell_asym_search: triangle quadrature rules with exactly N nodes without any symmetry, started from Gauss shell patterns.
The counterpart of shell_c3_search.py (nodes free in the plane, C3 symmetric) and shell_c3_bary.py (nodes inside the triangle, C3 symmetric): here every node and every weight is a free unknown.
  --mode po   nodes free in the plane, weights w = wmin + v^2 (unknowns v, x, y of every node)
  --mode pi   nodes inside the triangle by construction: barycentric coordinates b = s^2 / (s0^2 + s1^2 + s2^2), weights w = wmin + v^2 (a solution is a PI rule)
The start is, for every shell pattern and every --starts entry:
  pattern   the S3 approximation of the pattern (nodes of the pattern improved on the fully symmetric reduced system)
  rotated   the C3 start of the other programs: the S3 approximation after the rotation (outer shell slides by 1/8 spacing, the --turn innermost shells turn)
  random    uniformly random nodes in the triangle (a baseline without a shell pattern)
Start number 0 is exact; the further starts of --nstarts add normal noise of --noise / sqrt(N) to the node coordinates (seed --seed + number, so a run can be repeated).  A symmetric start stays symmetric under the
Levenberg-Marquardt steps in exact arithmetic, so the noise is what lets the search leave the symmetric subspace.
--wmin (default 0) is a lower bound of every weight.  Without it a free node can escape to infinity with a weight that goes to zero and the residual then falls below 1e-12 without a rule that is usable
(three nodes at distance 3.6 from the triangle with weights of 1e-10 were seen at degree 8); a solution must also have a smallest weight above --min-weight.
--polish-c3: a solution whose nodes are C3 symmetric (every node has its two rotated images among the nodes, within 1e-6) is rebuilt from the exact orbits and polished with a few Levenberg-Marquardt
steps on the C3 reduced system (the same system as in shell_c3_search.py), so that the stored rule is exactly C3 symmetric.
The run of a start stops when the residual is below 1e-12 (a solution; the smallest weight must be above --min-weight), or when over the last --win steps neither the residual fell by at least --gain (relative)
nor any unknown changed by more than --dq.  The search ends at the first solution.  Per pattern, start kind and mode a file <out>/shell_asym_<deg>_<pattern>_<mode>_<start>.json holds the stages of the start
with the smallest final residual (stages pattern, S3, rotation (the start), asym (the final nodes)); a solution is also written with all digits to <out>/solution_<deg>_<N>_<mode>.json.
Usage: python code/shell_asym_search.py --deg 8 --seq 5,2 --mode pi --starts rotated,pattern,random --nstarts 50 [--noise 0.3] [--seed 0] [--wmin 1e-5] [--min-weight 0] [--polish-c3] [--out shell_asym_out]
Needs numpy and scipy (code/cubature_atlas.py)."""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cubature_atlas as C
import shell_c3_search as SC


class AsymPO:
    """unknowns q = [v (N), x and y of every node (2 N)]; weights w = wmin + v^2"""

    def __init__(self, n, wmin=0.0):
        self.N = n
        self.wmin = wmin

    def q_from(self, X, W):
        return np.concatenate([np.sqrt(np.clip(W - self.wmin, 1e-12, None)), X.reshape(-1)])

    def nodes_weights(self, q):
        n = self.N
        return q[n:].reshape(n, 2), self.wmin + q[:n] ** 2

    def jacobian(self, B, q, X, W):
        n = self.N
        J = C.jacobian(B, X, W)
        Jw = J[:, 2 * n:] * (2 * q[:n])[None, :]
        return np.hstack([Jw, J[:, :2 * n]])


class AsymPI:
    """unknowns q = [v (N), s of every node (3 N)]; barycentric coordinates b = s^2 / sum(s^2), so every node is inside the triangle; weights w = wmin + v^2"""

    def __init__(self, n, wmin=0.0):
        self.N = n
        self.wmin = wmin

    def q_from(self, X, W):
        L = np.c_[1 - X.sum(1), X]
        L = np.clip(L, 1e-3, None)
        L = L / L.sum(axis=1, keepdims=True)
        return np.concatenate([np.sqrt(np.clip(W - self.wmin, 1e-12, None)), np.sqrt(L).reshape(-1)])

    def nodes_weights(self, q):
        n = self.N
        s2 = q[n:].reshape(n, 3) ** 2
        b = s2 / s2.sum(axis=1, keepdims=True)
        return b[:, 1:3], self.wmin + q[:n] ** 2

    def jacobian(self, B, q, X, W):
        n = self.N
        s = q[n:].reshape(n, 3)
        s2 = s ** 2
        tot = s2.sum(axis=1)
        J = C.jacobian(B, X, W)
        Jw = J[:, 2 * n:] * (2 * q[:n])[None, :]
        Js = np.zeros((J.shape[0], 3 * n))
        for i in range(n):
            # db_k / ds_j = (2 s_k delta_kj tot - s_k^2 2 s_j) / tot^2
            D = (2 * np.diag(s[i]) * tot[i] - 2 * np.outer(s2[i], s[i])) / tot[i] ** 2
            Js[:, 3 * i:3 * i + 3] = J[:, 2 * i:2 * i + 2] @ D[1:3, :]
        return np.hstack([Jw, Js])


def polish_c3(B, X, W, tol=1e-6):
    """returns (X, W, residual) of the polished C3 symmetric rule, or None if the nodes are not C3 symmetric"""
    n = len(X)
    used = np.zeros(n, dtype=bool)
    reps = []
    weights = []
    centre_weight = None
    for i in range(n):
        if used[i]:
            continue
        j = int(np.linalg.norm(X - (SC.ROT_A[0] @ X[i] + SC.ROT_A[1]), axis=1).argmin())
        k = int(np.linalg.norm(X - (SC.ROT_B[0] @ X[i] + SC.ROT_B[1]), axis=1).argmin())
        near = max(np.linalg.norm(X[j] - (SC.ROT_A[0] @ X[i] + SC.ROT_A[1])), np.linalg.norm(X[k] - (SC.ROT_B[0] @ X[i] + SC.ROT_B[1])))
        if near > tol:
            return None
        if j == i and k == i:
            used[i] = True
            centre_weight = W[i]
            continue
        if len({i, j, k}) < 3 or used[j] or used[k]:
            return None
        used[[i, j, k]] = True
        reps.append(X[i])
        weights.append(float(np.mean(W[[i, j, k]])))
    orbits = [("g3", [r[0], r[1]]) for r in reps]
    if centre_weight is not None:
        orbits.append(("ce", []))
        weights.append(float(centre_weight))
    red = SC.Reduced(orbits)
    q = red.q_from(np.array(weights))
    q, res, used_steps, why = SC.lm(B, red, q, 400, 10 ** 9, 0.0, 0.0, target=1e-15)
    Xp, Wp = red.nodes_weights(q)
    return Xp, Wp, res


def start_of(kind, pat, B, a, mode_turn):
    """returns the stages of the start and the node and weight arrays for the start kind"""
    P0 = pat.nodes()
    n = pat.N
    stages = [{"stage": "pattern", "residual": None, "nodes": np.round(P0, 6).tolist()}]
    if kind == "random":
        rng = np.random.default_rng(12345)
        r = np.sqrt(rng.random((n, 1)))
        u = rng.random((n, 1))
        X = np.hstack([r * (1 - u), r * u])
        return stages, X, np.full(n, 0.5 / n)
    red_s = SC.Reduced(SC.s3_orbits(P0))
    q = red_s.q_from(np.full(red_s.n_orb, 0.5 / n))
    q, res, used, why = SC.lm(B, red_s, q, a.s3_steps, 10 ** 9, 0.0, 0.0)
    Xs, Ws = red_s.nodes_weights(q)
    order = SC.match(P0, Xs)
    X = Xs[order]
    W = Ws[order]
    stages.append({"stage": "S3", "residual": res, "nodes": np.round(X, 6).tolist()})
    if kind == "rotated":
        X = SC.rotate(pat, X, mode_turn)
        stages.append({"stage": "rotation", "residual": None, "nodes": np.round(X, 6).tolist()})
    return stages, X, W


def run_start(B, red, X, W, noise, seed, a):
    """one Levenberg-Marquardt run from the start nodes with the noise of this start number"""
    n = len(X)
    Xn = X.copy()
    if seed is not None:
        rng = np.random.default_rng(seed)
        Xn = X + rng.normal(0.0, noise / np.sqrt(n), X.shape)
    q = red.q_from(Xn, W)
    q, res, used, why = SC.lm(B, red, q, a.steps, a.win, a.gain, a.dq)
    Xf, Wf = red.nodes_weights(q)
    return Xf, Wf, res, used, why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deg", type=int, required=True)
    ap.add_argument("--mode", choices=["po", "pi"], default="pi")
    ap.add_argument("--seq", default="", help="one shell pattern (points per edge, outside to inside, comma separated); default: all patterns with steps of 2 or 3 points")
    ap.add_argument("--target", type=int, default=0, help="number of nodes (default Z = ceil(M / 3))")
    ap.add_argument("--starts", default="rotated,pattern,random")
    ap.add_argument("--nstarts", type=int, default=20)
    ap.add_argument("--noise", type=float, default=0.3, help="standard deviation of the noise in units of 1 / sqrt(N)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--turn", default="2")
    ap.add_argument("--s3-steps", type=int, default=60)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--win", type=int, default=30)
    ap.add_argument("--gain", type=float, default=0.02)
    ap.add_argument("--dq", type=float, default=1e-4)
    ap.add_argument("--wmin", type=float, default=0.0, help="lower bound of every weight (w = wmin + v^2)")
    ap.add_argument("--min-weight", type=float, default=1e-9, help="a solution needs a smallest weight above this value (0: only positive)")
    ap.add_argument("--polish-c3", action="store_true", help="polish a C3 symmetric solution on the C3 reduced system")
    ap.add_argument("--offsets", default="0")
    ap.add_argument("--max-patterns", type=int, default=50)
    ap.add_argument("--out", default="shell_asym_out")
    a = ap.parse_args()
    n = a.deg
    Z = -(-((n + 1) * (n + 2) // 2) // 3)
    target = a.target or Z
    os.makedirs(a.out, exist_ok=True)
    if a.seq:
        patterns = [[int(s) for s in a.seq.split(",")]]
    else:
        patterns = SC.enumerate_patterns(n, target, [int(s) for s in a.offsets.split(",")])[:a.max_patterns]
    B = C.Basis("triangle", n)
    summary = os.path.join(a.out, "summary.txt")
    for seq in patterns:
        pat = SC.Pattern(seq)
        red = AsymPI(pat.N, a.wmin) if a.mode == "pi" else AsymPO(pat.N, a.wmin)
        tag = "_".join(str(m) for m in seq)
        diffs = "".join(" -%d" % (x - y) for x, y in zip(seq, seq[1:]))
        for kind in a.starts.split(","):
            t0 = time.time()
            stages, X0, W0 = start_of(kind, pat, B, a, a.turn)
            best = None
            for k in range(a.nstarts):
                seed = None if k == 0 else a.seed + k
                Xf, Wf, res, used, why = run_start(B, red, X0, W0, a.noise, seed, a)
                if best is None or res < best[2]:
                    best = (Xf, Wf, res, used, why, k)
                if res < 1e-12 and float(Wf.min()) > a.min_weight:
                    break
            Xf, Wf, res, used, why, k = best
            solved = res < 1e-12 and float(Wf.min()) > a.min_weight
            stages.append({"stage": "asym", "residual": res, "nodes": np.round(Xf, 6).tolist()})
            cmd = "python code/shell_asym_search.py --deg %d --seq %s --target %d --mode %s --starts %s --nstarts %d --noise %g --seed %d --turn %s --wmin %g%s" % (
                n, ",".join(str(m) for m in seq), pat.N, a.mode, kind, a.nstarts, a.noise, a.seed, a.turn, a.wmin, (" --min-weight %g" % a.min_weight if a.min_weight != 1e-9 else "") + (" --polish-c3" if a.polish_c3 else ""))
            data = {"degree": n, "N": pat.N, "seq": seq, "shell": pat.shell, "stages": stages, "solved_by": "asym" if solved else None, "mode": a.mode, "start": kind,
                    "best_start": k, "command": cmd}
            with open(os.path.join(a.out, "shell_asym_%d_%s_%s_%s.json" % (n, tag, a.mode, kind)), "w") as fh:
                json.dump(data, fh)
            line = "degree %d, N %d, shells %d%s, %s, start %s: best residual %.2e (start %d of %d, %d steps, %s), %.0fs%s" % (
                n, pat.N, seq[0], diffs, a.mode.upper(), kind, res, k, a.nstarts, used, why, time.time() - t0, "  *** SOLUTION" if solved else "")
            print(line, flush=True)
            with open(summary, "a") as fh:
                fh.write(line + "\n")
            if solved and a.polish_c3:
                pol = polish_c3(B, Xf, Wf)
                if pol is not None and pol[2] < 1e-12 and float(pol[1].min()) > a.min_weight:
                    Xf, Wf, res = pol
                    print("polished on the C3 reduced system: residual %.2e" % res, flush=True)
            if solved:
                L = np.c_[1 - Xf.sum(1), Xf]
                sol = {"degree": n, "N": pat.N, "seq": seq, "mode": a.mode, "start": kind, "nodes": Xf.tolist(), "weights": Wf.tolist(), "residual": res,
                       "min_barycentric": float(L.min()), "nodes_outside": int((L.min(axis=1) < 0).sum()), "command": cmd}
                with open(os.path.join(a.out, "solution_%d_%d_%s.json" % (n, pat.N, a.mode)), "w") as fh:
                    json.dump(sol, fh)
                print("solution: min weight %.2e, smallest barycentric coordinate %.2e, nodes outside the triangle %d" % (Wf.min(), L.min(), sol["nodes_outside"]), flush=True)
                return
    print("no solution for degree %d" % n, flush=True)


if __name__ == "__main__":
    main()

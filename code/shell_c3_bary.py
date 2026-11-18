"""shell_c3_bary: like code/shell_c3_search.py, but the nodes stay inside the triangle by construction: the barycentric coordinates of every node are b_i = s_i^2 / (s_0^2 + s_1^2 + s_2^2)
with free numbers s_i, so every solution is a PI rule (all nodes inside, all weights positive, w = v^2).  Pipeline per Gauss shell pattern with exactly Z nodes (steps of -2 or -3 points per edge):
  pattern -> S3 approximation (orbits of the full symmetry group; the pattern is first pulled into the triangle) -> rotation (outer shell slides by 1/8 spacing along its edge, the --turn innermost shells
  turn, then pulled into the triangle again) -> C3 symmetric Levenberg-Marquardt (one representative per orbit of the rotation group), float64, CPU, no noise.
A run of a pattern stops at a solution (residual below 1e-12), or when over the last --win steps neither the residual fell by --gain (relative) nor an unknown changed by more than --dq.
Writes <out>/shell_c3_bary_<deg>_<pattern>_turn<mode>.json (stages pattern, S3, rotation, C3 with nodes and residuals), <out>/summary.txt and, at the first solution, <out>/solution_<deg>_<N>.json.
--softplus: the weights are w = c ln(1 + e^x) with c = 1 / (2 N) (positive, smooth) instead of w = v^2.  --exact: --auto uses only Z nodes (no Z + 1).  --relax K: the shell patterns may have up to K steps of 1 or 4 points per edge
in addition to the steps of 2 and 3.
--damping scaled: the damping of the Levenberg-Marquardt steps is scaled by the diagonal of J^T J (Marquardt) with a start value of 1e-3 (default identity: lambda_0 = 0.1); both: both, one after the other.
--pull-inside: if a solution has nodes on the boundary of the triangle (a barycentric coordinate below 1e-9, the parametrisation ran to s = 0), it is solved again with the barycentric
coordinates b = delta + (1 - 3 delta) s^2 / sum s^2 for delta = 1e-6, 1e-5, 1e-4, 1e-3 (each solution is the start of the next delta); the solution with the largest delta is kept, so that every node has
the distance delta to the boundary (needed for an interval proof of a PI rule).
--staged: only Z nodes; the shell patterns are searched in stages: first only steps of 2 and 3 points per edge (relax 0), then with one step from --extra-steps (default 1 and 4), up to --relax steps; every stage
runs the patterns that were not in an earlier stage with the turn modes --turn and the weight/damping pairs --configs (for example "v2:identity,softplus:identity") until the first solution or --time-limit seconds.
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
    """unknowns q = [x (one per orbit), s of the orbits]; weights w = x^2, or w = c ln(1 + e^x) with softplus=True (c = 1 / (2 N), the mean weight, so that x is of order 1); generic orbits: b = s^2 / sum s^2 (3 numbers s), node (x, y) = (b_i, b_j) for the pairs of the orbit images;
    orbits on a mirror line (kind ax): barycentric coordinates (a, a, c) with a = s_1^2 / T, c = s_2^2 / T, T = 2 s_1^2 + s_2^2 (2 numbers s); centre node: fixed"""

    def __init__(self, orbits, softplus=False):
        self.orbits = orbits
        self.softplus = softplus
        self.c = 0.0
        self.n_orb = len(orbits)
        self.N = sum(NODES_OF[k] for k, s in orbits)
        self.pq = self.n_orb + sum(NPARAM[k] for k, s in orbits)
        self.orb = np.concatenate([np.full(NODES_OF[k], j) for j, (k, s) in enumerate(orbits)])
        self.c = 0.5 / self.N

    def _x_from_w(self, w):
        w = np.maximum(w, 1e-12)
        if self.softplus:
            return np.log(np.expm1(w / self.c))          # inverse of w = c ln(1 + e^x)
        return np.sqrt(w)

    def _w_of(self, x):
        if self.softplus:
            return self.c * np.logaddexp(0.0, x)
        return x ** 2

    def _dw_of(self, x):
        if self.softplus:
            return self.c * 0.5 * (1.0 + np.tanh(0.5 * x))        # c times the logistic function, the derivative of c ln(1 + e^x)
        return 2 * x

    def q_from(self, weights):
        pars = []
        for k, s in self.orbits:
            pars.extend(list(np.atleast_1d(s)) if NPARAM[k] else [])
        return np.concatenate([self._x_from_w(np.asarray(weights, float)), np.array(pars, float)])

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
        return np.vstack(X), self._w_of(q[:self.n_orb][self.orb])

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
        DW[np.arange(N), self.orb] = self._dw_of(q[:self.n_orb][self.orb])
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


def lm_scaled(B, red, q, steps, win, gain, dq_tol, target=1e-13, lam0=1e-3):
    """Levenberg-Marquardt like SC.lm, but the damping is scaled by the diagonal of J^T J (Marquardt), so that unknowns of very different size (weights, barycentric numbers) are damped alike"""
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
        D = np.diag(np.diag(H) + 1e-12 * (np.diag(H).max() + 1e-300))
        accepted = False
        for _ in range(25):
            try:
                d = -np.linalg.solve(H + lam * D, g)
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
            if np.sqrt(cost) > (1 - gain) * np.sqrt(cost_ref) and float(np.abs(q - q_ref).max()) < dq_tol:
                reason = "stagnation"
                break
    return q, float(np.sqrt(cost)), used, reason


class BaryM(Bary):
    """Bary with a distance delta to the boundary: b = delta + (1 - 3 delta) s^2 / sum s^2 for the generic orbits"""

    def __init__(self, orbits, delta, softplus=False):
        super().__init__(orbits, softplus)
        self.delta = delta

    def _gen(self, s, kind):
        X, D = Bary._gen(self, s, kind)
        f = 1 - 3 * self.delta
        return self.delta + f * X, f * D


def solve_with_margin(B, pat, X, W, a, lm):
    """solve the C3 system again with a growing distance delta to the boundary; returns (nodes, weights, residual, delta) of the largest delta with a solution, or None"""
    k = pat.k
    best = None
    cur_X = X
    cur_w = np.concatenate([W[0:3 * k:3], W[3 * k:]])
    for delta in (1e-6, 1e-5, 1e-4, 1e-3):
        f = 1 - 3 * delta
        orbits = []
        for j in range(k):
            b = bary_of(cur_X[3 * j:3 * j + 1])[0]
            b0 = np.clip((b - delta) / f, 1e-8, None)
            orbits.append(("g3", list(np.sqrt(b0 / b0.sum()))))
        if pat.centre:
            orbits.append(("ce", []))
        red = BaryM(orbits, delta, a.softplus)
        q = red.q_from(cur_w)
        q, res, used, why = SC.lm(B, red, q, 600, 25, 0.02, 1e-4, lam0=1e-4)
        Xn, Wn = red.nodes_weights(q)
        if not (res < 1e-12 and bool((Wn > 0).all())):
            break
        best = (Xn, Wn, res, delta)
        cur_X = Xn
        cur_w = np.concatenate([Wn[0:3 * k:3], Wn[3 * k:]])
    return best


def run_pattern(B, pat, mode, a):
    lm = lm_scaled if a.damping == "scaled" else SC.lm
    P0 = pull_inside(pat.nodes())
    red_s = Bary(s3_orbits_bary(P0), a.softplus)
    q = red_s.q_from(np.full(red_s.n_orb, 0.5 / pat.N))
    q, res, used, why = lm(B, red_s, q, a.s3_steps, 10 ** 9, 0.0, 0.0)
    Xs, Ws = red_s.nodes_weights(q)
    order = SC.match(P0, Xs)
    X = Xs[order]
    W = Ws[order]
    stages = [{"stage": "pattern", "residual": None, "nodes": np.round(P0, 6).tolist()}, {"stage": "S3", "residual": res, "nodes": np.round(X, 6).tolist()}]
    X2 = pull_inside(SC.rotate(pat, X, mode))
    stages.append({"stage": "rotation", "residual": None, "nodes": np.round(X2, 6).tolist()})
    k = pat.k
    orbits = [("g3", s_from_bary("g3", bary_of(X2[3 * j:3 * j + 1])[0])) for j in range(k)] + ([("ce", [])] if pat.centre else [])
    red_c = Bary(orbits, a.softplus)
    q2 = red_c.q_from(np.concatenate([W[0:3 * k:3], W[3 * k:]]))
    q2, res2, used2, why2 = lm(B, red_c, q2, a.steps, a.win, a.gain, a.dq)
    X3, W3 = red_c.nodes_weights(q2)
    stages.append({"stage": "C3", "residual": res2, "nodes": np.round(X3, 6).tolist()})
    solved = res2 < 1e-12 and bool((W3 > 0).all())
    a.margin_result = 0.0
    if solved and getattr(a, "pull_inside", False) and bary_of(X3).min() < 1e-9:
        r = solve_with_margin(B, pat, X3, W3, a, lm)
        if r is not None:
            X3, W3, res2, a.margin_result = r
            stages.append({"stage": "inside", "residual": res2, "nodes": np.round(X3, 6).tolist()})
    return stages, X3, W3, res2, used2, why2, solved


def enumerate_relaxed(n, target, offsets, relax, last_max=3, extra_steps=(1, 4)):
    """like SC.enumerate_patterns (steps of 2 or 3 points per edge, last shell with at most 3 points, exactly `target` nodes), but with up to `relax` steps of 1 or 4 points per edge in addition"""
    if relax <= 0 and last_max == 3:
        return SC.enumerate_patterns(n, target, offsets)
    m0 = -(-(n + 1) // 2)
    found = []

    def rec(seq, total, spent):
        m = seq[-1]
        if total == target and m <= last_max:
            found.append(list(seq))
        if total >= target:
            return
        for s in ((1,) if m == 2 else tuple(sorted({2, 3} | set(extra_steps)))):
            extra = spent + (1 if (s in extra_steps and m != 2) else 0)
            if extra > relax or m - s < 1:
                continue
            seq.append(m - s)
            rec(seq, total + SC.nodes_of_shell(m - s), extra)
            seq.pop()

    for off in offsets:
        rec([m0 + off], SC.nodes_of_shell(m0 + off), 0)

    def key(q):
        steps = [x - y for x, y in zip(q, q[1:])]
        odd = sum(1 for i, s in enumerate(steps) if s in extra_steps and q[i] != 2)
        return (q[0] != m0, odd, sum(1 for s in steps if s != 2), len(q), q)

    found.sort(key=key)
    return found


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
            "X": X.tolist(), "W": W.tolist(), "seconds": time.time() - t0, "s3": stages[1]["residual"], "damping": a.damping, "margin": a.margin_result, "softplus": a.softplus, "params": params}


def save_result(a, deg, r, cmd):
    seq = r["seq"]
    tag = "_".join(str(m) for m in seq)
    data = {"degree": deg, "N": r["N"], "seq": seq, "shell": r["shell"], "stages": r["stages"], "solved_by": "C3" if r["solved"] else None, "turn": r["mode"],
            "repro": {"command": cmd + " (shells %s)" % seq}}
    with open(os.path.join(a.out, "shell_c3_bary_%d_%s_turn%s%s%s.json" % (deg, tag, r["mode"], "_sp" if r.get("softplus") else "", "" if r["damping"] == "identity" else "_" + r["damping"])), "w") as fh:
        json.dump(data, fh)
    diffs = "".join(" -%d" % (x - y) for x, y in zip(seq, seq[1:]))
    line = "degree %d, N %d, shells %d%s, turn %s%s: S3 residual %.2e, C3 residual %.2e after %d steps (%s), %.0fs%s" % (
        deg, r["N"], seq[0], diffs, r["mode"], ("" if r["damping"] == "identity" else ", " + r["damping"] + " damping") + (", softplus" if r.get("softplus") else ""), r["s3"], r["res"], r["used"], r["why"], r["seconds"], ("  *** SOLUTION" + (" (pulled inside, distance %.0e to the boundary)" % r["margin"] if r["margin"] else "")) if r["solved"] else "")
    print(line, flush=True)
    with open(os.path.join(a.out, "summary.txt"), "a") as fh:
        fh.write(line + "\n")
    if r["solved"]:
        X = np.array(r["X"])
        W = np.array(r["W"])
        L = bary_of(X)
        pr = r.get("params")
        seqcmd = cmd if not pr else ("python code/shell_c3_bary.py --deg %d --seq %s --target %d --turn %s --steps %d --win %d --gain %g --dq %g" % (
            deg, ",".join(str(m) for m in seq), r["N"], r["mode"], pr["steps"], pr["win"], pr["gain"], pr["dq"]) + (" --softplus" if r.get("softplus") else "")
            + ("" if r["damping"] == "identity" else " --damping " + r["damping"]) + (" --pull-inside" if pr.get("pull_inside") else ""))
        sol = {"degree": deg, "N": r["N"], "seq": seq, "turn": r["mode"], "nodes": X.tolist(), "weights": W.tolist(), "residual": r["res"], "min_barycentric": float(L.min()),
               "nodes_outside": int((L.min(axis=1) < 0).sum()), "margin_delta": r["margin"], "command": seqcmd}
        with open(os.path.join(a.out, "solution_%d_%d.json" % (deg, r["N"])), "w") as fh:
            json.dump(sol, fh)
        print("solution: min weight %.2e, smallest barycentric coordinate %.2e (all nodes inside the triangle)" % (W.min(), L.min()), flush=True)


def run_variant(a, deg, target, offsets, modes, steps, win, gain, dq, procs, label, pats=None):
    from multiprocessing import Pool
    if pats is None:
        pats = enumerate_relaxed(deg, target, offsets, a.relax, a.last_max)[:a.max_patterns][a.part::a.parts]
    print("%s: degree %d, target %d nodes, %d shell patterns, turn %s, steps %d" % (label, deg, target, len(pats), modes, steps), flush=True)
    if not pats:
        return None
    dampings = ["identity", "scaled"] if a.damping == "both" else [a.damping]
    jobs = [(deg, seq, m, {"s3_steps": a.s3_steps, "steps": steps, "win": win, "gain": gain, "dq": dq, "softplus": a.softplus, "damping": dmp, "pull_inside": a.pull_inside}) for seq in pats for m in modes for dmp in dampings]
    cmd = "python code/shell_c3_bary.py --deg %d --target %d --turn %s --offsets %s --steps %d --win %d --gain %g --dq %g%s%s" % (deg, target, ",".join(modes), ",".join(str(o) for o in offsets), steps, win, gain, dq, " --softplus" if a.softplus else "", (" --relax %d" % a.relax) if a.relax else "") + ((" --last-max %d" % a.last_max) if a.last_max != 3 else "") + (" --pull-inside" if a.pull_inside else "")
    found = None
    if procs <= 1:
        init_worker(deg)
        for j in jobs:
            r = job(j)
            save_result(a, deg, r, cmd.replace("--turn %s" % ",".join(modes), "--turn %s" % r["mode"]) + ("" if r["damping"] == "identity" else " --damping " + r["damping"]))
            if r["solved"]:
                return r
        return None
    with Pool(procs, initializer=init_worker, initargs=(deg,)) as pool:
        for r in pool.imap_unordered(job, jobs):
            save_result(a, deg, r, cmd.replace("--turn %s" % ",".join(modes), "--turn %s" % r["mode"]) + ("" if r["damping"] == "identity" else " --damping " + r["damping"]))
            if r["solved"]:
                found = r
                pool.terminate()
                break
    return found


def run_staged(a, n, Z, nprocs):
    """see the module docstring (--staged); returns True if a solution was found"""
    from multiprocessing import Pool
    deadline = time.time() + a.time_limit if a.time_limit else None
    extra = tuple(int(s) for s in a.extra_steps.split(",") if s)
    configs = []
    for c in a.configs.split(","):
        w, d = c.split(":")
        configs.append((w == "softplus", d))
    base = {"s3_steps": a.s3_steps, "steps": a.steps, "win": a.win, "gain": a.gain, "dq": a.dq, "pull_inside": a.pull_inside}
    seen = set()
    for relax in range(0, a.relax + 1):
        pats = [q for q in enumerate_relaxed(n, Z, [-2, -1, 0, 1, 2], relax, a.last_max, extra) if tuple(q) not in seen][:a.max_patterns]
        seen.update(tuple(q) for q in pats)
        print("stage relax %d: degree %d, %d nodes, %d new shell patterns" % (relax, n, Z, len(pats)), flush=True)
        jobs = [(n, seq, m, dict(base, softplus=sp, damping=dmp)) for seq in pats for m in a.turn.split(",") for sp, dmp in configs]
        if not jobs:
            continue
        with Pool(nprocs, initializer=init_worker, initargs=(n,)) as pool:
            for r in pool.imap_unordered(job, jobs):
                save_result(a, n, r, "")
                if r["solved"]:
                    pool.terminate()
                    print("solution found: %s" % os.path.join(a.out, "solution_%d_%d.json" % (n, r["N"])), flush=True)
                    return True
                if deadline and time.time() > deadline:
                    pool.terminate()
                    print("no solution found (time limit of %d s reached)" % a.time_limit, flush=True)
                    return False
    print("no solution found", flush=True)
    return False


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
    ap.add_argument("--damping", default="identity", choices=["identity", "scaled", "both"], help="damping of the Levenberg-Marquardt steps")
    ap.add_argument("--pull-inside", action="store_true", help="solve a solution with nodes on the boundary again with a distance delta to the boundary (1e-6 up to 1e-3)")
    ap.add_argument("--softplus", action="store_true", help="weights w = ln(1 + e^x) instead of w = v^2")
    ap.add_argument("--exact", action="store_true", help="--auto: only Z nodes, no Z + 1")
    ap.add_argument("--last-max", type=int, default=3, help="largest number of points per edge of the last shell")
    ap.add_argument("--relax", type=int, default=0, help="up to this many steps of 1 or 4 points per edge in the shell patterns, in addition to the steps of 2 and 3")
    ap.add_argument("--staged", action="store_true", help="only Z nodes, shell patterns in stages (relax 0 first), see the module docstring")
    ap.add_argument("--extra-steps", default="1,4", help="steps (points per edge) allowed in addition to 2 and 3, comma separated (default 1,4)")
    ap.add_argument("--time-limit", type=int, default=0, help="--staged: stop after this many seconds (0 = no limit)")
    ap.add_argument("--configs", default="v2:identity,softplus:identity", help="--staged: weights:damping pairs, weights v2 or softplus, damping identity or scaled")
    a = ap.parse_args()
    n = a.deg
    Z = -(-((n + 1) * (n + 2) // 2) // 3)
    os.makedirs(a.out, exist_ok=True)
    if a.staged:
        if Z % 3 == 2:
            print("Z = %d is 2 modulo 3: there is no C3 symmetric rule with exactly Z nodes" % Z, flush=True)
            return
        found = run_staged(a, n, Z, max(1, min(a.procs, os.cpu_count() or 1)))
        return
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
    targets = [a.target] if a.target else ([Z] if a.exact else [Z, Z + 1])
    for target in targets:
        if not enumerate_relaxed(n, target, [-2, -1, 0, 1, 2], a.relax, a.last_max):
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

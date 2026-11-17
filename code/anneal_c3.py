"""anneal_c3: annealing search for triangle rules with exactly Z nodes in the C3 symmetric subspace (the rotation group of the triangle), nodes free in the plane (PO search, no clipping into the triangle).
Start state: the C3 approximation of code/shell_c3_search.py for one Gauss shell pattern (pattern -> S3 -> rotation -> C3).  Then several chains; in every iteration each chain takes a base state (with probability --pbest the best
state found so far, otherwise its own state), adds Gaussian noise to the representatives of the orbits (sigma log-uniform in [--smin, --smax], larger near the centroid: sigma * (1 + (--centre-gain - 1) * (1 - rho)),
rho = distance to the centroid / distance of a vertex, at most 1) and solves with Levenberg-Marquardt on the C3 reduced system (stop when the residual did not fall by --gain over the last --patience steps and is above
--floor).  A chain accepts the result if its residual is smaller, otherwise with probability exp(-(log10 r_new - log10 r_cur) / T), T = T0 * decay^(iteration - 1) in decades of the residual.  After a solution
(residual < 1e-12, all weights positive) the chain restarts from the start state, so that solutions can be counted.  All draws are seeded: numpy default_rng([seed, degree, iteration, chain]).
Every solution is written to --out with seed, command, iteration, chain and the number of nodes outside the triangle (PI if 0); the best state of the run goes to <out minus .json>_best.json.
float64 on the CPU; needs numpy and scipy (code/cubature_atlas.py) and code/shell_c3_search.py.
Usage: python code/anneal_c3.py --deg 34 --seq 18,16,14,11,9,6,3 --seed 5 --out anneal_c3_34.json [--iters 30] [--chains 8] [--turn all]"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cubature_atlas as C
import shell_c3_search as SC

VERTEX_R = 0.4714       # distance of a vertex of the triangle (0,0), (1,0), (0,1) to its centroid, as used for the noise profile


def lm_stop(B, red, q, steps, patience, gain, floor, lam0=1e-6):
    """Levenberg-Marquardt with a stagnation stop; returns the unknowns and the residual (2-norm of the moment equations)"""
    X, W = red.nodes_weights(q)
    r = C.residual(B, X, W)
    cost = float(r @ r)
    lam = lam0
    hist = [np.sqrt(cost)]
    for t in range(steps):
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
        res = float(np.sqrt(cost))
        hist.append(res)
        if res < 1e-13 or not accepted:
            break
        if t + 1 >= patience and res > (1 - gain) * hist[-1 - patience] and res > floor:
            break
    return q, float(np.sqrt(cost))


def noisy(red, q, rng, sigma, centre_gain):
    """add noise to the representatives of the orbits; larger near the centroid"""
    q = q.copy()
    k = red.n_orb - (1 if red.orbits[-1][0] == "ce" else 0)
    reps = q[red.n_orb:red.n_orb + 2 * k].reshape(k, 2)
    rho = np.minimum(np.linalg.norm(reps - SC.CENTROID, axis=1) / VERTEX_R, 1.0)
    scale = sigma * (1 + (centre_gain - 1) * (1 - rho))
    q[red.n_orb:red.n_orb + 2 * k] = (reps + rng.normal(0, 1, reps.shape) * scale[:, None]).reshape(-1)
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deg", type=int, required=True)
    ap.add_argument("--seq", required=True, help="shell pattern: points per edge of the shells, outside to inside, comma separated")
    ap.add_argument("--turn", default="all", help="number of innermost shells that turn in the rotation step, or all")
    ap.add_argument("--chains", type=int, default=8)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--pbest", type=float, default=0.25)
    ap.add_argument("--T0", type=float, default=0.5)
    ap.add_argument("--decay", type=float, default=0.9)
    ap.add_argument("--smin", type=float, default=0.003)
    ap.add_argument("--smax", type=float, default=0.02)
    ap.add_argument("--centre-gain", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--gain", type=float, default=0.1)
    ap.add_argument("--floor", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="anneal_c3.json")
    a = ap.parse_args()
    seq = [int(s) for s in a.seq.split(",")]
    B = C.Basis("triangle", a.deg)
    pat = SC.Pattern(seq)
    cmd = "python code/anneal_c3.py --deg %d --seq %s --turn %s --chains %d --iters %d --pbest %g --T0 %g --decay %g --smin %g --smax %g --centre-gain %g --seed %d" % (
        a.deg, a.seq, a.turn, a.chains, a.iters, a.pbest, a.T0, a.decay, a.smin, a.smax, a.centre_gain, a.seed)
    start_args = SimpleNamespace(s3_steps=60, steps=250, win=15, gain=0.05, dq=1e-3)
    stages, X3, W3, res3, used, why, solved = SC.run_pattern(B, pat, a.turn, start_args)
    k = pat.k
    red = SC.Reduced(pat.c3_orbits(X3[0:3 * k:3]))
    q0 = red.q_from(np.concatenate([W3[0:3 * k:3], W3[3 * k:]]))
    q0, r0 = lm_stop(B, red, q0, 250, 10 ** 9, 0.0, 0.0)
    print("pattern %s, N %d, %d unknowns, start residual %.2e" % (seq, pat.N, red.pq, r0), flush=True)
    if r0 < 1e-12:
        X, W = red.nodes_weights(q0)
        L = np.c_[1 - X.sum(1), X]
        hit = {"iteration": 0, "chain": 0, "sigma": 0.0, "from_best": False, "nodes": X.tolist(), "weights": W.tolist(), "residual": r0, "nodes_outside": int((L.min(axis=1) < 0).sum()),
               "min_barycentric": float(L.min()), "repro": {"script": "code/anneal_c3.py", "seed": a.seed, "degree": a.deg, "seq": seq, "command": cmd + " (the start state is a solution)"}}
        with open(a.out, "w") as fh:
            json.dump([hit], fh)
        print("the C3 start state is already a solution (nodes outside the triangle: %d)" % hit["nodes_outside"], flush=True)
        return
    start = (r0, q0)
    cur = [start for _ in range(a.chains)]
    best = start
    found = []
    runs = 0
    t0 = time.time()
    for it in range(1, a.iters + 1):
        T = a.T0 * a.decay ** (it - 1)
        acc = 0
        for c in range(a.chains):
            rng = np.random.default_rng([a.seed, a.deg, it, c])
            use_best = rng.random() < a.pbest
            base = best if use_best else cur[c]
            sigma = float(np.exp(rng.uniform(np.log(a.smin), np.log(a.smax))))
            qn, rn = lm_stop(B, red, noisy(red, base[1], rng, sigma, a.centre_gain), a.steps, a.patience, a.gain, a.floor)
            runs += 1
            if rn < 1e-12:
                X, W = red.nodes_weights(qn)
                if (W > 0).all():
                    L = np.c_[1 - X.sum(1), X]
                    found.append({"iteration": it, "chain": c, "sigma": sigma, "from_best": bool(use_best), "nodes": X.tolist(), "weights": W.tolist(), "residual": rn,
                                  "nodes_outside": int((L.min(axis=1) < 0).sum()), "min_barycentric": float(L.min()),
                                  "repro": {"script": "code/anneal_c3.py", "seed": a.seed, "degree": a.deg, "seq": seq, "command": cmd + " (iteration %d, chain %d)" % (it, c)}})
                    cur[c] = start
                    continue
            rng2 = np.random.default_rng([a.seed, a.deg, it, 999, c])
            r_cur = cur[c][0]
            if rn < r_cur or rng2.random() < np.exp(-(np.log10(max(rn, 1e-16)) - np.log10(max(r_cur, 1e-16))) / T):
                cur[c] = (rn, qn)
                acc += 1
            if rn < best[0]:
                best = (rn, qn)
        print("   iteration %2d (T %.3f): accepted %d/%d, runs %d, best residual %.2e, hits so far %d, %.0fs" % (it, T, acc, a.chains, runs, best[0], len(found), time.time() - t0), flush=True)
        with open(a.out, "w") as fh:
            json.dump(found, fh)
        Xb, Wb = red.nodes_weights(best[1])
        with open(a.out.replace(".json", "") + "_best.json", "w") as fh:
            json.dump({"degree": a.deg, "seq": seq, "seed": a.seed, "residual": best[0], "nodes": Xb.tolist(), "weights": Wb.tolist(), "command": cmd}, fh)
    print("result: %d hits in %d runs, %.0fs" % (len(found), runs, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

"""asym_remove_node: triangle quadrature rule with N - 1 nodes and no symmetry, started from a stored PI rule with N nodes from which one node is removed.
A C3 symmetric rule has N = 0 or 1 modulo 3 nodes.  An unsymmetric rule can have 2 modulo 3 nodes, one node less than the next C3 rule, so the stored C3 rule with N nodes is the natural start for N - 1.
Idea: the removed node leaves a hole; the weight of the removed node is given to its nearest neighbours, the nodes get seeded normal noise of --noise / sqrt(N) (so that the search leaves the symmetric subspace),
then Levenberg-Marquardt (code/shell_c3_search.py, `lm`) on all nodes and weights.  The nodes stay inside the triangle by construction (barycentric b = s^2 / sum s^2, weights wmin + v^2, class AsymPI of code/shell_asym_search.py),
so a solution is a PI rule.  A run stops at a residual below 1e-12, or when over the last --win steps neither the residual fell by --gain (relative) nor an unknown changed by more than --dq.
The candidates to remove are the nodes in the order of --order (weight: smallest weight first; random: seeded permutation); start number k removes candidate k // --noises and uses noise --noise * (k % --noises + 1) / --noises.
Every result (also without a solution) is appended to <out>/summary.txt with seed and command; a solution is written to <out>/solution_<deg>_<N-1>_asym.json with all digits, seed and command.
Usage: python code/asym_remove_node.py --rule rules/minimal_pointcount_for_degree_n_PI/triangle/CA-T2-44-348-PI-MPC.json --starts 40 --noise 0.05 --noises 2 --seed 1 [--part 0 --parts 4] --out asym_out
Needs numpy and scipy (code/cubature_atlas.py, code/shell_c3_search.py, code/shell_asym_search.py)."""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cubature_atlas as C
import shell_c3_search as SC
import shell_asym_search as SA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", required=True)
    ap.add_argument("--starts", type=int, default=20)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--noises", type=int, default=2)
    ap.add_argument("--order", default="weight", choices=["weight", "random"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--win", type=int, default=15)
    ap.add_argument("--gain", type=float, default=0.05)
    ap.add_argument("--dq", type=float, default=0.001)
    ap.add_argument("--wmin", type=float, default=1e-7)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--out", default="asym_remove_out")
    a = ap.parse_args()
    rule = json.load(open(a.rule))
    n = rule["degree"]
    X0 = np.array(rule["nodes"], float)
    W0 = np.array(rule["weights"], float)
    N = len(X0)
    B = C.Basis("triangle", n)
    os.makedirs(a.out, exist_ok=True)
    if a.order == "weight":
        order = list(np.argsort(W0))
    else:
        order = list(np.random.default_rng(a.seed).permutation(N))
    red = SA.AsymPI(N - 1, a.wmin)
    for k in range(a.starts):
        if k % a.parts != a.part:
            continue
        cand = int(order[(k // a.noises) % N])
        noise = a.noise * (k % a.noises + 1) / a.noises
        t0 = time.time()
        keep = np.arange(N) != cand
        X = X0[keep].copy()
        W = W0[keep].copy()
        near = np.argsort(np.linalg.norm(X - X0[cand], axis=1))[:3]
        W[near] += W0[cand] / 3.0
        rng = np.random.default_rng(a.seed * 1000003 + k)
        Xn = X + rng.normal(0.0, noise / np.sqrt(N), X.shape)
        q = red.q_from(Xn, W)
        q, res, used, why = SC.lm(B, red, q, a.steps, a.win, a.gain, a.dq)
        Xf, Wf = red.nodes_weights(q)
        cmd = "python code/asym_remove_node.py --rule %s --starts %d --noise %g --noises %d --order %s --seed %d --steps %d --win %d --gain %g --dq %g --wmin %g" % (
            a.rule, a.starts, a.noise, a.noises, a.order, a.seed, a.steps, a.win, a.gain, a.dq, a.wmin)
        solved = res < 1e-12 and float(Wf.min()) > 0.0
        line = "degree %d, N %d, removed node %d (weight %.2e), noise %.3f, start %d: residual %.2e (%d steps, %s), %.0fs%s" % (
            n, N - 1, cand, W0[cand], noise, k, res, used, why, time.time() - t0, "  *** SOLUTION" if solved else "")
        print(line, flush=True)
        with open(os.path.join(a.out, "summary.txt"), "a") as fh:
            fh.write(line + "\n")
        if solved:
            L = np.c_[1 - Xf.sum(1), Xf]
            sol = {"degree": n, "N": N - 1, "nodes": Xf.tolist(), "weights": Wf.tolist(), "residual": res, "min_barycentric": float(L.min()),
                   "nodes_outside": int((L.min(axis=1) < 0).sum()), "start": k, "removed": cand, "noise": noise, "command": cmd}
            with open(os.path.join(a.out, "solution_%d_%d_asym.json" % (n, N - 1)), "w") as fh:
                json.dump(sol, fh)
            print("solution found: min weight %.2e, smallest barycentric coordinate %.2e" % (Wf.min(), L.min()), flush=True)
            return
    print("no solution", flush=True)


if __name__ == "__main__":
    main()

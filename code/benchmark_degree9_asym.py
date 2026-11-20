#!/usr/bin/env python3
"""Paired elimination experiment on the proven degree-9, 19-node PI triangle rule.

Baseline: existing Duffy-box solver (code/tri_search.py).
Alternative: AsymPI barycentric-square nodes + positive squared weights
(code/shell_asym_search.py), the specialized solver used by asym_remove_node.py.

Both methods receive exactly the same deleted node, redistributed weights and
physical Gaussian perturbation for every paired trial. The goal is an 18-node
PI rule (and optionally a 17-node rule if an optimized weight becomes dummy).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
sys.path.insert(0, str(CODE))

import tri_search as TS
import cubature_atlas as C
import shell_c3_search as SC
import shell_asym_search as SA


START = ROOT / "rules/minimal_pointcount_for_degree_n_PI/triangle/CA-T2-09-19-PI-MPC.json"
DEGREE = 9
NOISE_LEVELS = (0.05, 0.10, 0.20)
SEEDS = (9104, 9105, 9106)
STEPS = 600
WIN = 15
GAIN = 0.05
DQ = 0.001
WMIN = 1e-7
DUMMY_CUTOFF = 1e-6
ACCEPT = 1e-12


def redistributed_start(rule, removed, noise, seed):
    X0 = np.asarray(rule["nodes"], float)
    W0 = np.asarray(rule["weights"], float)
    keep = np.arange(len(X0)) != removed
    X = X0[keep].copy()
    W = W0[keep].copy()
    near = np.argsort(np.linalg.norm(X - X0[removed], axis=1))[:3]
    W[near] += W0[removed] / 3.0
    rng = np.random.default_rng(seed)
    Xn = X + rng.normal(0.0, noise / np.sqrt(len(X)), X.shape)
    return Xn, W


def solve_box(X, W, n, max_iter):
    N = len(W)
    s, t = TS.xy_to_st(np.clip(X[:, 0], 0, 1),
                       np.clip(X[:, 1], 0, 1 - np.clip(X[:, 0], 0, 1)))
    v0 = np.concatenate([s, t, W])
    lb = np.zeros(3 * N)
    ub = np.concatenate([np.ones(2 * N), np.full(N, np.inf)])
    v, res = TS.lm_bounded(
        lambda v: TS.residual_and_jac(v, n, N), v0, lb, ub, max_iter
    )
    s, t, w = TS.unpack(v, N)
    x, y = TS.to_xy(s, t)
    return np.column_stack([x, y]), w, float(res)


def solve_asym(X, W, n, max_iter):
    red = SA.AsymPI(len(W), WMIN)
    q = red.q_from(X, W)
    B = C.Basis("triangle", n)
    q, _res2, used, why = SC.lm(B, red, q, max_iter, WIN, GAIN, DQ,
                                target=1e-13)
    Xf, Wf = red.nodes_weights(q)
    residual = C.residual(B, Xf, Wf)
    return np.asarray(Xf), np.asarray(Wf), float(np.linalg.norm(residual, np.inf)), used, why


def validate(X, W, n):
    L = np.column_stack([1 - X.sum(axis=1), X])
    B = C.Basis("triangle", n)
    residual = C.residual(B, X, W)
    rinf = float(np.linalg.norm(residual, np.inf))
    return {
        "residual": rinf,
        "min_weight": float(np.min(W)),
        "min_barycentric": float(np.min(L)),
        "inside": bool(np.min(L) >= -1e-13),
        "positive": bool(np.min(W) > 0.0),
        "accepted": bool(rinf < ACCEPT and np.min(W) > 1e-8 and np.min(L) >= -1e-13),
    }


def maybe_drop_dummy(method, X, W, n):
    """Try removing tiny-weight nodes and re-solving; return the best real rule."""
    history = []
    while len(W) > 2 and float(np.min(W)) < DUMMY_CUTOFF:
        k = int(np.argmin(W))
        X2 = np.delete(X, k, axis=0)
        W2 = np.delete(W, k)
        if method == "box_duffy":
            Xn, Wn, res = solve_box(X2, W2, n, STEPS)
            used, why = STEPS, "fixed-iteration Duffy"
        else:
            Xn, Wn, res, used, why = solve_asym(X2, W2, n, STEPS)
        v = validate(Xn, Wn, n)
        history.append({"removed_dummy": k, "N": len(Wn), "result": v,
                        "steps": used, "stop": why})
        if not v["accepted"]:
            break
        X, W = Xn, Wn
    return X, W, history


def trial(method, rule, removed, noise, seed):
    X, W = redistributed_start(rule, removed, noise, seed)
    start = time.perf_counter()
    if method == "box_duffy":
        Xf, Wf, res = solve_box(X, W, DEGREE, STEPS)
        used, why = STEPS, "fixed-iteration Duffy"
    else:
        Xf, Wf, res, used, why = solve_asym(X, W, DEGREE, STEPS)
    first = validate(Xf, Wf, DEGREE)
    drop_history = []
    if first["residual"] < ACCEPT:
        Xf, Wf, drop_history = maybe_drop_dummy(method, Xf, Wf, DEGREE)
    final = validate(Xf, Wf, DEGREE)
    return {
        "method": method,
        "removed_index": removed,
        "removed_weight": float(rule["weights"][removed]),
        "noise": noise,
        "seed": seed,
        "seconds": time.perf_counter() - start,
        "steps": used,
        "stop_reason": why,
        "first_result": first,
        "final_result": final,
        "dummy_drop_attempts": drop_history,
        "rule": ({"nodes": Xf.tolist(), "weights": Wf.tolist(),
                  "degree": DEGREE, "N": len(Wf)}
                 if final["accepted"] else None),
    }


def main():
    rule = json.loads(START.read_text())
    if rule["degree"] != DEGREE or rule["points"] != 19 or rule["quality"] != "PI":
        raise RuntimeError("Unexpected start-rule metadata")
    order = list(np.argsort(np.asarray(rule["weights"], float)))
    results = []
    t0 = time.perf_counter()
    for removed in order:
        for i, noise in enumerate(NOISE_LEVELS):
            seed = SEEDS[i] * 100 + removed
            # Alternate order so one solver does not always run under the same
            # machine-load conditions.
            methods = ("box_duffy", "asym_barycentric")
            if (removed + i) % 2:
                methods = methods[::-1]
            paired = {}
            for method in methods:
                print(f"START {method} remove={removed} noise={noise} seed={seed}",
                      flush=True)
                r = trial(method, rule, removed, noise, seed)
                results.append(r)
                paired[method] = r
                print(
                    f"END {method} remove={removed} noise={noise} "
                    f"N={r['final_result']['accepted'] and r['rule']['N'] or 0} "
                    f"res={r['final_result']['residual']:.3e} "
                    f"wmin={r['final_result']['min_weight']:.3e} "
                    f"success={r['final_result']['accepted']} "
                    f"seconds={r['seconds']:.2f}",
                    flush=True,
                )
            if paired["box_duffy"]["final_result"]["accepted"] != paired["asym_barycentric"]["final_result"]["accepted"]:
                print("PAIRED OUTCOME DIFFERENCE", removed, noise, flush=True)

    summary = {
        "target": "triangle degree 9: 18 PI nodes (and 17 if a genuine dummy node can be dropped)",
        "start_rule": "CA-T2-09-19-PI-MPC",
        "removal_order": order,
        "noise_levels": NOISE_LEVELS,
        "seeds": SEEDS,
        "iterations_per_trial": STEPS,
        "accept_tolerance": ACCEPT,
        "dummy_weight_cutoff": DUMMY_CUTOFF,
        "methods": {},
        "paired": [],
        "candidates": [],
        "elapsed_seconds": time.perf_counter() - t0,
    }
    for method in ("box_duffy", "asym_barycentric"):
        part = [r for r in results if r["method"] == method]
        by_n = {}
        for N in (17, 18):
            vals = [r for r in part if r["final_result"]["accepted"] and r["rule"]["N"] == N]
            by_n[str(N)] = len(vals)
        summary["methods"][method] = {
            "trials": len(part),
            "solutions_by_N": by_n,
            "successes_total": sum(r["final_result"]["accepted"] for r in part),
            "total_seconds": sum(r["seconds"] for r in part),
            "median_seconds": float(np.median([r["seconds"] for r in part])),
            "best_residual": min(r["final_result"]["residual"] for r in part),
        }
    for removed in order:
        for noise in NOISE_LEVELS:
            pair = [
                next(r for r in results if r["removed_index"] == removed
                     and r["noise"] == noise and r["method"] == m)
                for m in ("box_duffy", "asym_barycentric")
            ]
            summary["paired"].append({
                "removed_index": removed,
                "noise": noise,
                "box_duffy_success": pair[0]["final_result"]["accepted"],
                "asym_barycentric_success": pair[1]["final_result"]["accepted"],
                "box_duffy_residual": pair[0]["final_result"]["residual"],
                "asym_barycentric_residual": pair[1]["final_result"]["residual"],
                "box_duffy_seconds": pair[0]["seconds"],
                "asym_barycentric_seconds": pair[1]["seconds"],
            })
    summary["candidates"] = [
        {"method": r["method"], "removed_index": r["removed_index"],
         "noise": r["noise"], "seed": r["seed"], **r["final_result"],
         "nodes": r["rule"]["nodes"], "weights": r["rule"]["weights"]}
        for r in results if r["final_result"]["accepted"] and r["rule"] is not None
    ]
    Path("degree9_asym_elimination_results.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print("FINAL SUMMARY")
    print(json.dumps(summary["methods"], indent=2))
    print("Candidates:", len(summary["candidates"]))
    print("Elapsed seconds:", summary["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    main()

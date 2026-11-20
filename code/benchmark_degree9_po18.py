#!/usr/bin/env python3
"""Paired degree-9 N=18 search: PI/Duffy bounds versus physical PO bounds.

Starts from both the original and the improved proven 19-node PI rules.
For each of the 19 possible first deletions and three perturbation scales, the
same redistributed/perturbed 18-node configuration is given to:
  * the existing bounded Duffy PI solver;
  * a physical-coordinate solver with a wide box and positive weights, so
    nodes may leave the triangle (PO).

A numerical solution is accepted only with positive non-dummy weights and a
degree-9 orthonormal-basis residual below 1e-13. Any candidate still needs the
repository's strict interval proof.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import tri_search as TS


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ("original", ROOT / "rules/minimal_pointcount_for_degree_n_PI/triangle/CA-T2-09-19-PI-MPC.json"),
    ("improved_s3", ROOT / "scratch_candidates/degree9_s3_19_po_start.json"),
)
DEGREE = 9
TARGET_N = 18
NOISES = (0.002, 0.02, 0.10)
PO_BOUND = 1.0
MAXFEV = 500
ACCEPT = 1e-13
MIN_WEIGHT = 1e-8


def fj_physical(v):
    N = TARGET_N
    X = np.column_stack((v[:N], v[N:2*N]))
    W = v[2*N:]
    bs = TS.basis(DEGREE)
    B, Bx, By = bs.eval(X[:, 0], X[:, 1])
    r = B @ W - bs.rhs()
    J = np.hstack((Bx * W, By * W, B))
    return r, J


def redistribute(X0, W0, removed, noise, seed):
    keep = np.arange(len(W0)) != removed
    X = X0[keep].copy()
    W = W0[keep].copy()
    d2 = ((X - X0[removed]) ** 2).sum(axis=1) + 1e-6
    inv = 1.0 / d2
    W += W0[removed] * inv / inv.sum()
    rng = np.random.default_rng(seed)
    X += rng.normal(0.0, noise, X.shape)
    return X, W


def solve_po(X, W):
    lo = np.r_[np.full(TARGET_N, -PO_BOUND),
               np.full(TARGET_N, -PO_BOUND),
               np.zeros(TARGET_N)]
    hi = np.r_[np.full(TARGET_N, 1.0 + PO_BOUND),
               np.full(TARGET_N, 1.0 + PO_BOUND),
               np.full(TARGET_N, np.inf)]
    v0 = np.r_[X[:, 0], X[:, 1], W]
    v0 = np.clip(v0, lo + 1e-12, np.where(np.isinf(hi), 1e9, hi - 1e-12))
    v, _ = TS.lm_bounded(fj_physical, v0, lo, hi, MAXFEV)
    Xf = np.column_stack((v[:TARGET_N], v[TARGET_N:2*TARGET_N]))
    Wf = v[2*TARGET_N:]
    r, _ = fj_physical(v)
    return Xf, Wf, float(np.linalg.norm(r, np.inf))


def solve_pi(X, W):
    # Project the common noisy start onto the triangle for a valid Duffy start.
    s, t = TS.xy_to_st(np.clip(X[:, 0], 0, 1), np.clip(X[:, 1], 0, 1))
    v0 = np.r_[s, t, W]
    lb = np.zeros(3 * TARGET_N)
    ub = np.r_[np.ones(2 * TARGET_N), np.full(TARGET_N, np.inf)]
    v0 = np.clip(v0, lb + 1e-12, np.r_[np.ones(2 * TARGET_N) - 1e-12, np.full(TARGET_N, 1e9)])
    v, _ = TS.solve(v0, DEGREE, TARGET_N, max_nfev=MAXFEV)
    s, t, Wf = TS.unpack(v, TARGET_N)
    x, y = TS.to_xy(s, t)
    Xf = np.column_stack((x, y))
    r, _ = TS.residual_and_jac(v, DEGREE, TARGET_N)
    return Xf, Wf, float(np.linalg.norm(r, np.inf))


def assess(X, W, residual):
    L = np.column_stack((X[:, 0], X[:, 1], 1.0 - X.sum(axis=1)))
    margin = float(L.min())
    minw = float(W.min())
    outside = bool(margin < -1e-9)
    accepted = bool(residual < ACCEPT and minw > MIN_WEIGHT)
    return {
        "residual": float(residual),
        "min_weight": minw,
        "min_barycentric": margin,
        "outside_nodes": int((L.min(axis=1) < -1e-9).sum()),
        "quality": "PO" if outside else "PI",
        "accepted": accepted,
    }


def main():
    results = []
    t_all = time.perf_counter()
    for source_id, path in SOURCES:
        rule = json.loads(path.read_text())
        X0 = np.asarray(rule["nodes"], float)
        W0 = np.asarray(rule["weights"], float)
        if len(W0) != 19:
            raise RuntimeError(f"Unexpected source rule {source_id}")
        for removed in range(19):
            for ni, noise in enumerate(NOISES):
                seed = 990000 + (0 if source_id == "original" else 100000) + removed * 10 + ni
                X, W = redistribute(X0, W0, removed, noise, seed)
                pair = {}
                for method, solver in (("pi_duffy", solve_pi), ("po_physical", solve_po)):
                    t0 = time.perf_counter()
                    Xf, Wf, res = solver(X.copy(), W.copy())
                    q = assess(Xf, Wf, res)
                    row = {
                        "source": source_id,
                        "method": method,
                        "removed_index": removed,
                        "noise": noise,
                        "seed": seed,
                        "seconds": time.perf_counter() - t0,
                        "quality": q,
                        "rule": ({"degree": DEGREE, "N": TARGET_N,
                                  "nodes": Xf.tolist(), "weights": Wf.tolist(),
                                  "quality": q["quality"]}
                                 if q["accepted"] else None),
                    }
                    results.append(row)
                    pair[method] = row
                    print(
                        f"END source={source_id} {method} remove={removed} noise={noise} "
                        f"res={res:.3e} wmin={q['min_weight']:.3e} "
                        f"margin={q['min_barycentric']:.3e} outside={q['outside_nodes']} "
                        f"success={q['accepted']} seconds={row['seconds']:.2f}",
                        flush=True,
                    )
                if pair["pi_duffy"]["quality"]["accepted"] != pair["po_physical"]["quality"]["accepted"]:
                    print(f"PAIRED DIFFERENCE source={source_id} remove={removed} noise={noise}", flush=True)

    summary = {
        "target": "triangle degree 9, N=18, positive weights, PI or PO",
        "sources": [x[0] for x in SOURCES],
        "noise_levels": NOISES,
        "maxfev": MAXFEV,
        "po_coordinate_bound": PO_BOUND,
        "accept_tolerance": ACCEPT,
        "methods": {},
        "paired_results": results,
        "candidates": [
            {"source": r["source"], "method": r["method"],
             "removed_index": r["removed_index"], "noise": r["noise"],
             "seed": r["seed"], **r["quality"], **r["rule"]}
            for r in results if r["quality"]["accepted"] and r["rule"] is not None
        ],
        "elapsed_seconds": time.perf_counter() - t_all,
    }
    for source_id, _ in SOURCES:
        summary["methods"][source_id] = {}
        for method in ("pi_duffy", "po_physical"):
            part = [r for r in results if r["source"] == source_id and r["method"] == method]
            accepted = [r for r in part if r["quality"]["accepted"]]
            summary["methods"][source_id][method] = {
                "trials": len(part),
                "accepted": len(accepted),
                "PI": sum(r["quality"]["quality"] == "PI" for r in accepted),
                "PO": sum(r["quality"]["quality"] == "PO" for r in accepted),
                "best_residual": min(r["quality"]["residual"] for r in part),
                "total_seconds": sum(r["seconds"] for r in part),
            }
    Path("degree9_po18_physical_results.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print("FINAL SUMMARY")
    print(json.dumps(summary["methods"], indent=2))
    print("Candidates:", len(summary["candidates"]), flush=True)


if __name__ == "__main__":
    main()

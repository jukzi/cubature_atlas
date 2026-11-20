#!/usr/bin/env python3
"""Search for an 18-node PI triangle rule of degree 9 by deleting the centroid.

The start is the newly proved S3-symmetric 19-node rule. Its 18 non-centroid
nodes consist of six C3 orbits of size three. We remove the centroid, distribute
its weight equally over the 18 remaining physical nodes, and solve:
  1. C3-reduced LM with degree continuation;
  2. full asymmetric LM from perturbed expansions of the same C3 starts.

This directly tests whether the 19 -> 18 reduction is possible from the
improved symmetric basin. No atlas rule is modified.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import tri_search as TS


ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "scratch_candidates/degree9_s3_19_start.json"
DEGREE = 9
SEEDS = tuple(range(9700, 9732))
NOISE_LEVELS = (0.0, 0.002, 0.01, 0.03, 0.08)
CONTINUATION = (5, 7, 8)
CONT_MAXFEV = 120
FINAL_MAXFEV = 700
ACCEPT = 1e-13
MIN_WEIGHT = 1e-8


def rotate_xy(point):
    x, y = point
    return np.array([[x, y], [y, 1.0 - x - y], [1.0 - x - y, x]])


def extract_c3_orbits(X, W):
    center = np.array([1.0 / 3.0, 1.0 / 3.0])
    center_idx = int(np.argmin(np.linalg.norm(X - center, axis=1)))
    if np.linalg.norm(X[center_idx] - center) > 1e-10:
        raise RuntimeError("The S3 start does not contain an exact centroid")
    remaining = set(range(len(W))) - {center_idx}
    reps = []
    while remaining:
        i = min(remaining)
        targets = rotate_xy(X[i])
        orbit = []
        for target in targets:
            distances = np.linalg.norm(X - target, axis=1)
            j = int(np.argmin(distances))
            if distances[j] > 1e-9:
                raise RuntimeError("Could not reconstruct a C3 orbit")
            orbit.append(j)
        if len(set(orbit)) != 3 or not set(orbit).issubset(remaining):
            raise RuntimeError("C3 orbit partition is inconsistent")
        if np.max(np.abs(W[orbit] - W[i])) > 1e-10:
            raise RuntimeError("Weights are not constant on a C3 orbit")
        reps.append(i)
        remaining.difference_update(orbit)
    if len(reps) != 6:
        raise RuntimeError(f"Expected 6 C3 orbits, found {len(reps)}")
    return center_idx, reps


def make_c3_start(X, W, center_idx, reps):
    wc = float(W[center_idx])
    Xbase = X[reps]
    Wbase = W[reps] + wc / 18.0
    s, t = TS.xy_to_st(Xbase[:, 0], Xbase[:, 1])
    return np.concatenate((s, t, Wbase))


def c3_candidate(q, k):
    full = TS.c3_to_full(q, k, centroid=False)
    s, t, w = TS.unpack(full, 3 * k)
    x, y = TS.to_xy(s, t)
    X = np.column_stack((x, y))
    residual, _ = TS.residual_and_jac(full, DEGREE, 3 * k)
    return X, w, float(np.linalg.norm(residual, np.inf))


def full_candidate(v, N):
    s, t, w = TS.unpack(v, N)
    x, y = TS.to_xy(s, t)
    X = np.column_stack((x, y))
    residual, _ = TS.residual_and_jac(v, DEGREE, N)
    return X, w, float(np.linalg.norm(residual, np.inf))


def quality(X, W, residual):
    L = np.column_stack((X[:, 0], X[:, 1], 1.0 - X.sum(axis=1)))
    margin = float(L.min())
    minw = float(W.min())
    accepted = bool(residual < ACCEPT and minw > MIN_WEIGHT and margin >= -1e-13)
    return {
        "residual": float(residual),
        "min_weight": minw,
        "min_barycentric": margin,
        "accepted": accepted,
    }


def solve_c3_start(q0, noise, seed):
    rng = np.random.default_rng(seed)
    k = 6
    q = q0.copy()
    q[:2 * k] += rng.normal(0.0, noise, 2 * k)
    q[:2 * k] = np.clip(q[:2 * k], 1e-8, 1.0 - 1e-8)
    q[2 * k:] *= np.exp(rng.normal(0.0, noise, k))
    for degree in CONTINUATION:
        q, _ = TS.solve_c3(q, k, False, degree, max_nfev=CONT_MAXFEV)
    q, _ = TS.solve_c3(q, k, False, DEGREE, max_nfev=FINAL_MAXFEV)
    return c3_candidate(q, k)


def solve_full_start(q0, noise, seed):
    k = 6
    N = 3 * k
    rng = np.random.default_rng(seed)
    q = q0.copy()
    q[:2 * k] += rng.normal(0.0, noise, 2 * k)
    q[:2 * k] = np.clip(q[:2 * k], 1e-8, 1.0 - 1e-8)
    q[2 * k:] *= np.exp(rng.normal(0.0, noise, k))
    v = TS.c3_to_full(q, k, centroid=False)
    # Add independent perturbations to the 18 physical nodes to allow the
    # full solver to leave the C3-symmetric subspace.
    s, t, w = TS.unpack(v, N)
    rng2 = np.random.default_rng(seed + 1000000)
    s = np.clip(s + rng2.normal(0.0, noise, N), 1e-8, 1.0 - 1e-8)
    t = np.clip(t + rng2.normal(0.0, noise, N), 1e-8, 1.0 - 1e-8)
    v = np.concatenate((s, t, w))
    for degree in CONTINUATION:
        v, _ = TS.solve(v, degree, N, max_nfev=CONT_MAXFEV)
    v, _ = TS.solve(v, DEGREE, N, max_nfev=FINAL_MAXFEV)
    return full_candidate(v, N)


def main():
    rule = json.loads(START.read_text())
    X = np.asarray(rule["nodes"], float)
    W = np.asarray(rule["weights"], float)
    if rule["degree"] != 9 or len(W) != 19:
        raise RuntimeError("Unexpected S3 starting rule")
    center_idx, reps = extract_c3_orbits(X, W)
    q0 = make_c3_start(X, W, center_idx, reps)
    print(f"centroid index={center_idx}; C3 representatives={reps}; "
          f"removed weight={W[center_idx]:.8g}", flush=True)

    results = []
    t_all = time.perf_counter()
    for noise_idx, noise in enumerate(NOISE_LEVELS):
        for seed in SEEDS:
            for method, solver in (("c3", solve_c3_start), ("full", solve_full_start)):
                run_seed = seed + noise_idx * 10000
                t0 = time.perf_counter()
                Xf, Wf, residual = solver(q0, noise, run_seed)
                q = quality(Xf, Wf, residual)
                row = {
                    "method": method,
                    "noise": noise,
                    "seed": run_seed,
                    "seconds": time.perf_counter() - t0,
                    "quality": q,
                    "rule": ({"degree": 9, "N": 18,
                              "nodes": Xf.tolist(), "weights": Wf.tolist()}
                             if q["accepted"] else None),
                }
                results.append(row)
                print(
                    f"END {method} noise={noise} seed={run_seed} "
                    f"res={residual:.3e} wmin={q['min_weight']:.3e} "
                    f"margin={q['min_barycentric']:.3e} "
                    f"success={q['accepted']} seconds={row['seconds']:.2f}",
                    flush=True,
                )

    summary = {
        "target": "triangle degree 9 PI, N=18 (current proven count 19)",
        "start_rule": "CA-T2-09-19-PI-MPC-a",
        "start_centroid_index": center_idx,
        "c3_representatives": reps,
        "centroid_weight_distributed_per_physical_node": float(W[center_idx] / 18.0),
        "continuation_degrees": CONTINUATION,
        "continuation_maxfev": CONT_MAXFEV,
        "final_maxfev": FINAL_MAXFEV,
        "seeds": list(SEEDS),
        "noise_levels": NOISE_LEVELS,
        "methods": {},
        "results": results,
        "candidates": [
            {"method": r["method"], "noise": r["noise"], "seed": r["seed"],
             **r["quality"], "nodes": r["rule"]["nodes"], "weights": r["rule"]["weights"]}
            for r in results if r["rule"] is not None
        ],
        "elapsed_seconds": time.perf_counter() - t_all,
    }
    for method in ("c3", "full"):
        part = [r for r in results if r["method"] == method]
        summary["methods"][method] = {
            "trials": len(part),
            "successes": sum(r["quality"]["accepted"] for r in part),
            "best_residual": min(r["quality"]["residual"] for r in part),
            "median_residual": float(np.median([r["quality"]["residual"] for r in part])),
            "total_seconds": sum(r["seconds"] for r in part),
        }
    Path("degree9_c3_18_search_results.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print("FINAL SUMMARY")
    print(json.dumps(summary["methods"], indent=2))
    print("Candidates:", len(summary["candidates"]), flush=True)


if __name__ == "__main__":
    main()

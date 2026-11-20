#!/usr/bin/env python3
"""C3-orbit elimination from proven higher-degree triangle rules to degree-9 N=18.

Sources:
- degree 10, 24 PI nodes = 8 C3 orbits; delete 2 orbits;
- degree 11, 27 PI nodes = 9 C3 orbits; delete 3 orbits.

For every possible orbit deletion set, test equal and inverse-distance
weight redistribution, four perturbation amplitudes, and both:
  * C3-reduced continuation LM;
  * full asymmetric continuation LM from the same physical start.

This preserves the correct C3 structure in one path, while permitting symmetry
breaking in the other. Every accepted candidate must still be interval-proved.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np

import tri_search as TS


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ("degree10_24", ROOT / "rules/minimal_pointcount_for_degree_n_PI/triangle/CA-T2-10-24-PI-MPC.json", 2),
    ("degree11_27", ROOT / "rules/minimal_pointcount_for_degree_n_PI/triangle/CA-T2-11-27-PI-MPC.json", 3),
)
DEGREE = 9
TARGET_N = 18
NOISES = (0.0, 0.005, 0.02, 0.08)
REDISTRIBUTIONS = ("equal", "inverse_distance")
CONTINUATION = (5, 7, 8, 9)
CONT_MAXFEV = 100
FINAL_MAXFEV = 600
ACCEPT = 1e-13
MIN_WEIGHT = 1e-8
EPS = 1e-10


def rotate_orbit(point):
    x, y = point
    return np.array([[x, y], [y, 1.0 - x - y], [1.0 - x - y, x]])


def group_c3_orbits(rule):
    X = np.asarray(rule["nodes"], float)
    W = np.asarray(rule["weights"], float)
    unused = set(range(len(W)))
    orbits = []
    while unused:
        i = min(unused)
        idx = []
        for target in rotate_orbit(X[i]):
            distances = np.linalg.norm(X - target, axis=1)
            j = int(np.argmin(distances))
            if distances[j] > 2e-9:
                raise RuntimeError(f"{rule['id']} is not C3-symmetric")
            idx.append(j)
        if len(set(idx)) != 3 or not set(idx).issubset(unused):
            raise RuntimeError(f"Invalid C3 orbit partition for {rule['id']}")
        if np.max(np.abs(W[idx] - W[i])) > 2e-9:
            raise RuntimeError(f"Weights are not C3-invariant in {rule['id']}")
        orbits.append({"rep": i, "idx": idx, "weight": float(W[i]),
                       "nodes": X[idx].copy()})
        unused.difference_update(idx)
    return X, W, orbits


def make_orbit_start(X, W, orbits, deleted, redistribution):
    kept = [i for i in range(len(orbits)) if i not in deleted]
    if len(kept) != 6:
        raise RuntimeError("The 18-node C3 start must have six orbits")
    w = np.array([orbits[i]["weight"] for i in kept], float)
    delta = np.zeros(6)
    if redistribution == "equal":
        total = sum(orbits[i]["weight"] for i in deleted)
        delta[:] = total / 6.0
    else:
        Xkeep = [orbits[i]["nodes"] for i in kept]
        for i in deleted:
            source = orbits[i]
            distances = np.array([
                min(np.linalg.norm(a - b) for a in source["nodes"] for b in nodes)
                for nodes in Xkeep
            ])
            inv = 1.0 / np.maximum(distances, 1e-8) ** 2
            delta += source["weight"] * inv / inv.sum()
    reps = np.array([orbits[i]["rep"] for i in kept], dtype=int)
    Xbase = X[reps].copy()
    Wbase = w + delta
    s, t = TS.xy_to_st(Xbase[:, 0], Xbase[:, 1])
    return np.concatenate((s, t, Wbase)), kept, reps, delta


def perturb_c3(q0, noise, seed):
    k = TARGET_N // 3
    q = q0.copy()
    rng = np.random.default_rng(seed)
    q[:2*k] += rng.normal(0.0, noise, 2*k)
    q[:2*k] = np.clip(q[:2*k], EPS, 1.0 - EPS)
    if noise:
        q[2*k:] *= np.exp(rng.normal(0.0, noise, k))
    return q


def solve_c3(q0, noise, seed):
    k = TARGET_N // 3
    q = perturb_c3(q0, noise, seed)
    for d in CONTINUATION:
        q, _ = TS.solve_c3(q, k, False, d, max_nfev=CONT_MAXFEV)
    q, _ = TS.solve_c3(q, k, False, DEGREE, max_nfev=FINAL_MAXFEV)
    full = TS.c3_to_full(q, k, False)
    s, t, w = TS.unpack(full, TARGET_N)
    x, y = TS.to_xy(s, t)
    return np.column_stack((x, y)), w


def solve_full(q0, noise, seed):
    k = TARGET_N // 3
    q = perturb_c3(q0, noise, seed)
    full = TS.c3_to_full(q, k, False)
    s, t, w = TS.unpack(full, TARGET_N)
    rng = np.random.default_rng(seed + 1234567)
    if noise:
        s = np.clip(s + rng.normal(0.0, noise, TARGET_N), EPS, 1.0 - EPS)
        t = np.clip(t + rng.normal(0.0, noise, TARGET_N), EPS, 1.0 - EPS)
    v = np.concatenate((s, t, w))
    for d in CONTINUATION:
        v, _ = TS.solve(v, d, TARGET_N, max_nfev=CONT_MAXFEV)
    v, _ = TS.solve(v, DEGREE, TARGET_N, max_nfev=FINAL_MAXFEV)
    s, t, w = TS.unpack(v, TARGET_N)
    x, y = TS.to_xy(s, t)
    return np.column_stack((x, y)), w


def assess(X, W):
    bary = np.column_stack((X[:, 0], X[:, 1], 1.0 - X.sum(axis=1)))
    margin = float(bary.min())
    minw = float(W.min())
    s, t = TS.xy_to_st(X[:, 0], X[:, 1])
    v = np.r_[s, t, W]
    residual = TS.residual_and_jac(v, DEGREE, TARGET_N)[0]
    rinf = float(np.linalg.norm(residual, np.inf))
    accepted = bool(rinf < ACCEPT and minw > MIN_WEIGHT and margin >= -1e-13)
    return {
        "residual": rinf,
        "min_weight": minw,
        "min_barycentric": margin,
        "accepted": accepted,
    }


def main():
    results = []
    all_candidates = []
    t_all = time.perf_counter()
    for source_id, path, delete_count in SOURCES:
        rule = json.loads(path.read_text())
        X, W, orbits = group_c3_orbits(rule)
        if len(orbits) - delete_count != TARGET_N // 3:
            raise RuntimeError(f"Unexpected orbit count for {source_id}")
        combos = list(itertools.combinations(range(len(orbits)), delete_count))
        print(f"SOURCE {source_id}: {len(orbits)} C3 orbits; "
              f"{len(combos)} possible deletion sets", flush=True)
        for combo_i, deleted in enumerate(combos):
            for red_i, redistribution in enumerate(REDISTRIBUTIONS):
                q0, kept, reps, delta = make_orbit_start(
                    X, W, orbits, deleted, redistribution
                )
                for noise_i, noise in enumerate(NOISES):
                    seed = (9900000 + (0 if source_id == "degree10_24" else 1000000)
                            + combo_i * 100 + red_i * 10 + noise_i)
                    pair = {}
                    for method, solver in (("c3_reduced", solve_c3),
                                           ("full_asymmetric", solve_full)):
                        t0 = time.perf_counter()
                        Xf, Wf = solver(q0, noise, seed)
                        quality = assess(Xf, Wf)
                        row = {
                            "source": source_id,
                            "method": method,
                            "deleted_orbits": list(deleted),
                            "kept_orbits": kept,
                            "redistribution": redistribution,
                            "redistributed_base_weights": delta.tolist(),
                            "noise": noise,
                            "seed": seed,
                            "seconds": time.perf_counter() - t0,
                            "quality": quality,
                            "rule": ({"degree": DEGREE, "N": TARGET_N,
                                      "nodes": Xf.tolist(), "weights": Wf.tolist()}
                                     if quality["accepted"] else None),
                        }
                        results.append(row)
                        pair[method] = row
                        print(
                            f"END source={source_id} method={method} "
                            f"deleted={deleted} redist={redistribution} noise={noise} "
                            f"res={quality['residual']:.3e} "
                            f"wmin={quality['min_weight']:.3e} "
                            f"margin={quality['min_barycentric']:.3e} "
                            f"success={quality['accepted']} "
                            f"seconds={row['seconds']:.2f}",
                            flush=True,
                        )
                        if quality["accepted"] and row["rule"] is not None:
                            all_candidates.append({
                                "source": source_id,
                                "method": method,
                                "deleted_orbits": list(deleted),
                                "redistribution": redistribution,
                                "noise": noise,
                                "seed": seed,
                                **quality,
                                **row["rule"],
                            })
                    if pair["c3_reduced"]["quality"]["accepted"] != pair["full_asymmetric"]["quality"]["accepted"]:
                        print("PAIRED DIFFERENCE", source_id, deleted, redistribution, noise, flush=True)

    summary = {
        "target": "triangle degree 9 PI, 18 nodes (current proven count 19)",
        "sources": [{"name": x[0], "delete_orbits": x[2]} for x in SOURCES],
        "redistributions": REDISTRIBUTIONS,
        "noise_levels": NOISES,
        "continuation_degrees": CONTINUATION,
        "continuation_maxfev": CONT_MAXFEV,
        "final_maxfev": FINAL_MAXFEV,
        "methods": {},
        "candidates": all_candidates,
        "results": results,
        "elapsed_seconds": time.perf_counter() - t_all,
    }
    for source_id, _, _ in SOURCES:
        summary["methods"][source_id] = {}
        for method in ("c3_reduced", "full_asymmetric"):
            part = [r for r in results if r["source"] == source_id and r["method"] == method]
            summary["methods"][source_id][method] = {
                "trials": len(part),
                "successes": sum(r["quality"]["accepted"] for r in part),
                "best_residual": min(r["quality"]["residual"] for r in part),
                "median_residual": float(np.median([r["quality"]["residual"] for r in part])),
                "total_seconds": sum(r["seconds"] for r in part),
            }
    Path("degree9_c3_orbit_elimination_results.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print("FINAL SUMMARY")
    print(json.dumps(summary["methods"], indent=2))
    print("Candidates:", len(all_candidates), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Test chained node elimination from higher-degree proven triangle rules.

A degree-10 or degree-11 PI rule is also an exact starting rule for degree 9.
The extra nodes provide a higher-dimensional solution manifold. This script
reduces those rules one node at a time down to N=17, comparing:
  direct: one bounded degree-9 LM solve per deletion;
  continuation: solve the same start at degrees 5 -> 7 -> 8 -> 9, then polish.

The candidate deletions, redistributed weights and physical perturbations
are paired between the two policies at the first chain step. Every successful
step becomes the start of the next step. No atlas rule files are modified.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import tri_search as TS


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ("degree10_24", ROOT / "rules/minimal_pointcount_for_degree_n_PI/triangle/CA-T2-10-24-PI-MPC.json"),
    ("degree11_27", ROOT / "rules/minimal_pointcount_for_degree_n_PI/triangle/CA-T2-11-27-PI-MPC.json"),
)
TARGET_DEGREE = 9
TARGET_N = 17
MAX_CANDIDATES = 12
NOISE_LEVELS = (1e-3, 1e-2, 5e-2)
CONTINUATION = (5, 7, 8, 9)
CONT_MAXFEV = 100
DIRECT_MAXFEV = 800
FINAL_MAXFEV = 400
ACCEPT = 1e-13
MIN_WEIGHT = 1e-8


def remove_start(X, W, removed, noise, seed):
    keep = np.arange(len(W)) != removed
    Xk = X[keep].copy()
    Wk = W[keep].copy()
    # Same redistribution used by tri_search.mode_elim.
    d2 = ((Xk - X[removed]) ** 2).sum(axis=1) + 1e-6
    inv = 1.0 / d2
    Wk += W[removed] * inv / inv.sum()
    s, t = TS.xy_to_st(Xk[:, 0], Xk[:, 1])
    v = np.concatenate([s, t, Wk])
    rng = np.random.default_rng(seed)
    v[:2 * len(Wk)] += rng.normal(0.0, noise, 2 * len(Wk))
    return v


def attempt(v0, policy, N):
    if policy == "direct":
        v, res = TS.solve(v0, TARGET_DEGREE, N, max_nfev=DIRECT_MAXFEV)
    else:
        v = v0.copy()
        for d in CONTINUATION:
            v, _ = TS.solve(v, d, N, max_nfev=CONT_MAXFEV)
        v, res = TS.solve(v, TARGET_DEGREE, N, max_nfev=FINAL_MAXFEV)
    s, t, w = TS.unpack(v, N)
    x, y = TS.to_xy(s, t)
    X = np.column_stack([x, y])
    L = np.column_stack([1.0 - x - y, x, y])
    r, _ = TS.residual_and_jac(v, TARGET_DEGREE, N)
    rinf = float(np.linalg.norm(r, np.inf))
    quality = {
        "residual": rinf,
        "min_weight": float(w.min()),
        "min_barycentric": float(L.min()),
        "accepted": bool(rinf < ACCEPT and w.min() > MIN_WEIGHT and L.min() >= -1e-13),
    }
    return X, w, quality


def candidate_order(W, rng):
    # A seeded permutation pairs the exact deletion indices across policies.
    # The noise perturbation is also seeded by (source, N, node, amplitude).
    return [int(i) for i in rng.permutation(len(W))]


def run_chain(source_id, source_path, policy):
    rule = json.loads(source_path.read_text())
    if rule.get("quality") != "PI" or rule.get("points") != len(rule["nodes"]):
        raise RuntimeError(f"bad source metadata: {source_path}")
    X = np.asarray(rule["nodes"], float)
    W = np.asarray(rule["weights"], float)
    rng = np.random.default_rng(9300 + len(X) * 100)
    history = []
    chain_started = time.perf_counter()
    print(f"CHAIN_START source={source_id} policy={policy} N={len(W)}", flush=True)

    while len(W) > TARGET_N:
        Nnew = len(W) - 1
        order = candidate_order(W, rng)[:MAX_CANDIDATES]
        stage = {
            "N": int(Nnew),
            "tested": 0,
            "success": False,
            "best_residual": float("inf"),
            "best_removed": None,
            "best_noise": None,
            "elapsed_seconds": 0.0,
        }
        best_success = None
        step_start = time.perf_counter()
        for candidate in order:
            for j, noise in enumerate(NOISE_LEVELS):
                # Identical deletion and perturbation seed for direct and
                # continuation policies at the same source and stage.
                seed = 930000 + len(rule["nodes"]) * 10000 + Nnew * 100 + int(candidate) * 3 + j
                v0 = remove_start(X, W, int(candidate), noise, seed)
                t0 = time.perf_counter()
                Xf, Wf, quality = attempt(v0, policy, Nnew)
                used = time.perf_counter() - t0
                stage["tested"] += 1
                if quality["residual"] < stage["best_residual"]:
                    stage["best_residual"] = quality["residual"]
                    stage["best_removed"] = int(candidate)
                    stage["best_noise"] = float(noise)
                if quality["accepted"]:
                    # Among successful candidates at this step prefer a larger
                    # minimum weight and then a larger boundary margin.
                    score = (quality["min_weight"], quality["min_barycentric"])
                    if best_success is None or score > best_success[0]:
                        best_success = (score, Xf, Wf, quality, int(candidate), float(noise))
            if best_success is not None:
                # Still test all configured candidates in the stage so the
                # selected start is the best-quality exact result.
                continue

        stage["elapsed_seconds"] = time.perf_counter() - step_start
        if best_success is None:
            history.append(stage)
            print(
                f"CHAIN_STOP source={source_id} policy={policy} target={Nnew} "
                f"best={stage['best_residual']:.3e} tried={stage['tested']} "
                f"seconds={stage['elapsed_seconds']:.1f}",
                flush=True,
            )
            break

        _, X, W, quality, removed, noise = best_success
        stage["success"] = True
        stage["selected_removed"] = removed
        stage["selected_noise"] = noise
        stage["selected_quality"] = quality
        history.append(stage)
        print(
            f"CHAIN_STEP source={source_id} policy={policy} N={Nnew} "
            f"res={quality['residual']:.3e} wmin={quality['min_weight']:.3e} "
            f"margin={quality['min_barycentric']:.3e} tried={stage['tested']}",
            flush=True,
        )
        if Nnew == TARGET_N:
            break

    final = None
    if len(W) <= rule["points"] - 1:
        final = {
            "N": int(len(W)),
            "nodes": X.tolist(),
            "weights": W.tolist(),
            "quality": TS.accept(
                np.concatenate([
                    X[:, 0],
                    np.divide(X[:, 1], 1.0 - X[:, 0],
                              out=np.zeros(len(X)),
                              where=(1.0 - X[:, 0]) > 1e-14),
                    W,
                ]),
                TARGET_DEGREE, len(W), ACCEPT
            ) if len(W) > 0 else False,
        }
    return {
        "source": source_id,
        "policy": policy,
        "start_N": int(rule["points"]),
        "final_N": int(len(W)),
        "reached_target": bool(len(W) <= TARGET_N),
        "elapsed_seconds": time.perf_counter() - chain_started,
        "history": history,
        "final_rule": final,
    }


def main():
    results = []
    for source_id, path in SOURCES:
        if not path.exists():
            raise FileNotFoundError(path)
        for policy in ("direct", "continuation"):
            results.append(run_chain(source_id, path, policy))

    summary = {
        "target": "triangle degree 9 PI with 17 nodes; current proven count 19",
        "source_rules": [name for name, _ in SOURCES],
        "policies": ["direct", "continuation"],
        "max_candidates_per_stage": MAX_CANDIDATES,
        "noise_levels": NOISE_LEVELS,
        "continuation_degrees": CONTINUATION,
        "results": results,
    }
    Path("degree9_high_degree_chain_results.json").write_text(
        json.dumps(summary, indent=2, default=lambda obj: obj.item() if hasattr(obj, "item") else str(obj)) + "\n"
    )
    print("FINAL SUMMARY")
    print(json.dumps([
        {k: r[k] for k in ("source", "policy", "start_N", "final_N",
                           "reached_target", "elapsed_seconds")}
        for r in results
    ], indent=2), flush=True)


if __name__ == "__main__":
    main()

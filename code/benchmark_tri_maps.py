#!/usr/bin/env python3
"""Paired benchmark: box/Duffy search vs barycentric-node + softplus-weight search.

Target: triangle degree 8. N=16 reproduces the current PI node count; N=15
tests the counting-value improvement. Both methods use the same original
primal LM implementation, same random starts, continuation, seeds and budgets.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import tri_search as ts


SEEDS = list(range(2001, 2041))
DEGREE = 8
NODE_COUNTS = (16, 15)
CONTINUATION_DEGREES = (4, 6, 7, 8)
CONTINUATION_MAXFEV = 150
FINAL_MAXFEV = 300
ACCEPT = 1e-13


def softplus(q):
    return np.logaddexp(0.0, q)


def sigmoid(q):
    # Stable sigmoid, without requiring a new dependency.
    out = np.empty_like(q)
    pos = q >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-q[pos]))
    ez = np.exp(q[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def decode_barycentric(q, n_nodes):
    qx = q[:n_nodes]
    qy = q[n_nodes:2 * n_nodes]
    logits = np.vstack((np.zeros(n_nodes), 2.0 * qx, 2.0 * qy))
    logits -= np.max(logits, axis=0, keepdims=True)
    ex = np.exp(logits)
    bary = ex / np.sum(ex, axis=0, keepdims=True)
    x, y = bary[1], bary[2]
    w = softplus(q[2 * n_nodes:])
    return x, y, w


def encode_barycentric(v, n_nodes):
    s, t, w = ts.unpack(v, n_nodes)
    x, y = ts.to_xy(s, t)
    b1 = np.maximum(x, 1e-14)
    b2 = np.maximum(y, 1e-14)
    b0 = np.maximum(1.0 - x - y, 1e-14)
    total = b0 + b1 + b2
    b0, b1, b2 = b0 / total, b1 / total, b2 / total
    # Inverse softplus, stable for the small positive initial weights here.
    qw = np.log(np.expm1(np.maximum(w, 1e-14)))
    return np.concatenate((0.5 * np.log(b1 / b0),
                           0.5 * np.log(b2 / b0), qw))


def mapped_residual_and_jac(q, n, n_nodes):
    x, y, w = decode_barycentric(q, n_nodes)
    bs = ts.basis(n)
    B, Bx, By = bs.eval(x, y)
    residual = B @ w - bs.rhs()

    # Softmax barycentric map b=(b0,b1,b2) with logits (0,2qx,2qy).
    dx_dqx = 2.0 * x * (1.0 - x)
    dy_dqx = -2.0 * x * y
    dx_dqy = -2.0 * x * y
    dy_dqy = 2.0 * y * (1.0 - y)

    Jqx = (Bx * w) * dx_dqx + (By * w) * dy_dqx
    Jqy = (Bx * w) * dx_dqy + (By * w) * dy_dqy
    Jqw = B * sigmoid(q[2 * n_nodes:])[None, :]
    return residual, np.hstack((Jqx, Jqy, Jqw))


def mapped_solve(v0, n, n_nodes, max_nfev=300):
    q0 = encode_barycentric(v0, n_nodes)
    lb = np.full(3 * n_nodes, -np.inf)
    ub = np.full(3 * n_nodes, np.inf)
    q, _ = ts.lm_bounded(
        lambda z: mapped_residual_and_jac(z, n, n_nodes),
        q0, lb, ub, max_nfev,
    )
    x, y, w = decode_barycentric(q, n_nodes)
    s = x
    t = np.divide(y, 1.0 - x, out=np.zeros_like(y), where=(1.0 - x) > 1e-14)
    v = np.concatenate((s, t, w))
    residual, _ = ts.residual_and_jac(v, n, n_nodes)
    return v, float(np.linalg.norm(residual, np.inf))


def run_one(method, n_nodes, seed):
    rng = np.random.default_rng(seed)
    v = ts.random_start(n_nodes, rng)
    solve = ts.solve if method == "box_duffy" else mapped_solve
    t0 = time.perf_counter()
    trace = []
    for d in CONTINUATION_DEGREES:
        v, res = solve(v, d, n_nodes, max_nfev=CONTINUATION_MAXFEV)
        trace.append(float(res))
    v, res = solve(v, DEGREE, n_nodes, max_nfev=FINAL_MAXFEV)
    elapsed = time.perf_counter() - t0
    accepted = bool(ts.accept(v, DEGREE, n_nodes, ACCEPT))
    s, t, w = ts.unpack(v, n_nodes)
    x, y = ts.to_xy(s, t)
    boundary_distance = np.minimum(np.minimum(x, y), 1.0 - x - y)
    rule = None
    if accepted:
        rule = {
            "degree": DEGREE,
            "N": n_nodes,
            "seed": seed,
            "nodes": [[float(a), float(b)] for a, b in zip(x, y)],
            "weights": [float(a) for a in w],
            "residual": float(res),
            "min_weight": float(w.min()),
            "min_boundary_distance": float(boundary_distance.min()),
        }
    return {
        "method": method,
        "N": n_nodes,
        "seed": seed,
        "success": accepted,
        "residual": float(res),
        "continuation_residuals": trace,
        "seconds": elapsed,
        "rule": rule,
    }


def main():
    results = []
    for n_nodes in NODE_COUNTS:
        for seed in SEEDS:
            # Alternate order to reduce systematic timing bias.
            methods = ("box_duffy", "barycentric_softplus")
            if seed % 2:
                methods = methods[::-1]
            for method in methods:
                print(f"START method={method} N={n_nodes} seed={seed}", flush=True)
                row = run_one(method, n_nodes, seed)
                results.append(row)
                print(
                    f"END method={method} N={n_nodes} seed={seed} "
                    f"res={row['residual']:.3e} success={row['success']} "
                    f"seconds={row['seconds']:.2f}",
                    flush=True,
                )

    summary = {
        "target": "triangle degree 8 PI: N=16 current count; N=15 counting value",
        "seeds": SEEDS,
        "continuation_degrees": CONTINUATION_DEGREES,
        "continuation_maxfev": CONTINUATION_MAXFEV,
        "final_maxfev": FINAL_MAXFEV,
        "accept_tolerance": ACCEPT,
        "methods": {},
        "paired_by_seed_and_N": [],
        "candidates": [],
    }
    for method in ("box_duffy", "barycentric_softplus"):
        for n_nodes in NODE_COUNTS:
            part = [r for r in results if r["method"] == method and r["N"] == n_nodes]
            summary["methods"].setdefault(method, {})[str(n_nodes)] = {
                "starts": len(part),
                "successes": sum(r["success"] for r in part),
                "success_rate": sum(r["success"] for r in part) / len(part),
                "total_seconds": sum(r["seconds"] for r in part),
                "median_seconds_per_start": float(np.median([r["seconds"] for r in part])),
                "median_final_residual": float(np.median([r["residual"] for r in part])),
                "best_final_residual": min(r["residual"] for r in part),
            }
    for n_nodes in NODE_COUNTS:
        for seed in SEEDS:
            old = next(r for r in results if r["N"] == n_nodes and r["seed"] == seed and r["method"] == "box_duffy")
            new = next(r for r in results if r["N"] == n_nodes and r["seed"] == seed and r["method"] == "barycentric_softplus")
            summary["paired_by_seed_and_N"].append({
                "N": n_nodes,
                "seed": seed,
                "box_duffy_success": old["success"],
                "barycentric_softplus_success": new["success"],
                "box_duffy_residual": old["residual"],
                "barycentric_softplus_residual": new["residual"],
                "box_duffy_seconds": old["seconds"],
                "barycentric_softplus_seconds": new["seconds"],
            })
    summary["candidates"] = [
        r["rule"] for r in results if r["success"] and r["rule"] is not None
    ]
    Path("triangle_barycentric_benchmark.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print("\nSUMMARY")
    print(json.dumps(summary["methods"], indent=2))
    print("Candidates:", len(summary["candidates"]))
    print("Full results: triangle_barycentric_benchmark.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Paired benchmark of NNLS variable projection for degree-9 triangle rules.

At each node iterate, solve the linear weights subproblem exactly with w >= 0
(Scipy NNLS), then update only the 2N Duffy node coordinates using the
Kaufman projected Jacobian. Finish with the existing full bounded LM polish.

The baseline and NNLS methods share exactly the same random node starts and
degree-continuation schedule. Targets N=18 and N=17 are compared with the
existing proven degree-9 PI count of 19 nodes.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

import tri_search as TS


DEGREE = 9
NODE_COUNTS = (18, 17)
SEEDS = tuple(range(9201, 9241))
CONTINUATION = (5, 7, 8, 9)
CONT_MAXFEV = 150
FINAL_MAXFEV = 300
POLISH_MAXFEV = 150
ACCEPT = 1e-13
ACTIVE_EPS = 1e-12


def nnls_residual_jac(q, n, N):
    """NNLS-profiled residual and projected Kaufman Jacobian.

    For fixed nodes, w is the nonnegative least-squares minimizer of
    ||B(q) w - rhs||. On a fixed positive support, the residual is orthogonal
    to the active columns of B. Projecting the raw node Jacobian into the
    orthogonal complement eliminates the first-order weight response.
    """
    s = q[:N]
    t = q[N:]
    x, y = TS.to_xy(s, t)
    bs = TS.basis(n)
    B, Bx, By = bs.eval(x, y)
    rhs = bs.rhs()
    w, _ = nnls(B, rhs)
    r = B @ w - rhs

    Js = (Bx - By * t) * w
    Jt = By * (1.0 - s) * w
    Jraw = np.hstack([Js, Jt])

    active = w > ACTIVE_EPS
    if np.any(active):
        Q, _ = np.linalg.qr(B[:, active], mode="reduced")
        J = Jraw - Q @ (Q.T @ Jraw)
    else:
        J = Jraw
    return r, J, w


def solve_nnls_profile(q0, n, N, max_nfev):
    lb = np.zeros(2 * N)
    ub = np.ones(2 * N)

    def fun_jac(q):
        r, J, _ = nnls_residual_jac(q, n, N)
        return r, J

    q, _ = TS.lm_bounded(fun_jac, q0.copy(), lb, ub, max_nfev)
    r, _, w = nnls_residual_jac(q, n, N)
    return q, w, float(np.linalg.norm(r, np.inf))


def run_baseline(v0, N):
    v = v0.copy()
    for d in CONTINUATION:
        v, _ = TS.solve(v, d, N, max_nfev=CONT_MAXFEV)
    v, res = TS.solve(v, DEGREE, N, max_nfev=FINAL_MAXFEV)
    s, t, w = TS.unpack(v, N)
    x, y = TS.to_xy(s, t)
    return np.column_stack([x, y]), w, res


def run_nnls(v0, N):
    q = v0[:2 * N].copy()
    for d in CONTINUATION:
        q, w, _ = solve_nnls_profile(q, d, N, CONT_MAXFEV)
    q, w, _ = solve_nnls_profile(q, DEGREE, N, FINAL_MAXFEV)
    v = np.concatenate([q[:N], q[N:], w])
    # Full bounded LM polish converts the profiled near-solution into the
    # ordinary atlas variables, with exactly the same box and weight bounds.
    v, res = TS.solve(v, DEGREE, N, max_nfev=POLISH_MAXFEV)
    s, t, w = TS.unpack(v, N)
    x, y = TS.to_xy(s, t)
    return np.column_stack([x, y]), w, res


def check(X, W):
    L = np.column_stack([1.0 - X.sum(axis=1), X])
    residual, _ = TS.residual_and_jac(
        np.concatenate([
            X[:, 0],
            np.divide(X[:, 1], 1.0 - X[:, 0],
                      out=np.zeros(len(X)),
                      where=(1.0 - X[:, 0]) > 1e-14),
            W,
        ]),
        DEGREE, len(W)
    )
    rinf = float(np.linalg.norm(residual, np.inf))
    return {
        "residual": rinf,
        "min_weight": float(np.min(W)),
        "min_barycentric": float(np.min(L)),
        "accepted": bool(rinf < ACCEPT and np.min(W) > 1e-8 and np.min(L) >= -1e-13),
    }


def one_trial(N, seed, method, v0):
    t0 = time.perf_counter()
    if method == "full_lm":
        X, W, solver_res = run_baseline(v0, N)
    else:
        X, W, solver_res = run_nnls(v0, N)
    quality = check(X, W)
    return {
        "N": int(N),
        "seed": int(seed),
        "method": method,
        "seconds": time.perf_counter() - t0,
        "solver_residual": float(solver_res),
        "quality": quality,
        "rule": ({"degree": DEGREE, "N": int(N), "nodes": X.tolist(),
                  "weights": W.tolist()}
                 if quality["accepted"] else None),
    }


def main():
    results = []
    for N in NODE_COUNTS:
        for seed in SEEDS:
            # Generate the start once; both methods receive the identical copy.
            v0 = TS.random_start(N, np.random.default_rng(seed))
            methods = ("full_lm", "nnls_profile")
            if seed % 2:
                methods = methods[::-1]
            pair = {}
            for method in methods:
                print(f"START {method} N={N} seed={seed}", flush=True)
                row = one_trial(N, seed, method, v0)
                results.append(row)
                pair[method] = row
                print(
                    f"END {method} N={N} seed={seed} "
                    f"res={row['quality']['residual']:.3e} "
                    f"success={row['quality']['accepted']} "
                    f"wmin={row['quality']['min_weight']:.3e} "
                    f"seconds={row['seconds']:.2f}",
                    flush=True,
                )
            if pair["full_lm"]["quality"]["accepted"] != pair["nnls_profile"]["quality"]["accepted"]:
                print(f"PAIRED OUTCOME DIFFERENCE N={N} seed={seed}", flush=True)

    summary = {
        "target": "triangle degree 9 PI, compare N=18 and N=17 (atlas current PI count 19)",
        "seeds": list(SEEDS),
        "continuation_degrees": list(CONTINUATION),
        "continuation_maxfev": CONT_MAXFEV,
        "final_maxfev": FINAL_MAXFEV,
        "polish_maxfev": POLISH_MAXFEV,
        "active_weight_threshold": ACTIVE_EPS,
        "accept_tolerance": ACCEPT,
        "methods": {},
        "paired": [],
        "candidates": [],
    }
    for method in ("full_lm", "nnls_profile"):
        summary["methods"][method] = {}
        for N in NODE_COUNTS:
            part = [r for r in results if r["method"] == method and r["N"] == N]
            summary["methods"][method][str(N)] = {
                "starts": len(part),
                "successes": sum(r["quality"]["accepted"] for r in part),
                "success_rate": sum(r["quality"]["accepted"] for r in part) / len(part),
                "total_seconds": sum(r["seconds"] for r in part),
                "median_seconds_per_start": float(np.median([r["seconds"] for r in part])),
                "median_residual": float(np.median([r["quality"]["residual"] for r in part])),
                "best_residual": min(r["quality"]["residual"] for r in part),
            }
    for N in NODE_COUNTS:
        for seed in SEEDS:
            old = next(r for r in results if r["method"] == "full_lm" and r["N"] == N and r["seed"] == seed)
            new = next(r for r in results if r["method"] == "nnls_profile" and r["N"] == N and r["seed"] == seed)
            summary["paired"].append({
                "N": N,
                "seed": seed,
                "full_lm_success": old["quality"]["accepted"],
                "nnls_profile_success": new["quality"]["accepted"],
                "full_lm_residual": old["quality"]["residual"],
                "nnls_profile_residual": new["quality"]["residual"],
                "full_lm_seconds": old["seconds"],
                "nnls_profile_seconds": new["seconds"],
            })
    summary["candidates"] = [
        {"N": r["N"], "seed": r["seed"], "method": r["method"],
         **r["quality"], "nodes": r["rule"]["nodes"], "weights": r["rule"]["weights"]}
        for r in results if r["rule"] is not None
    ]
    Path("degree9_nnls_projection_results.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print("FINAL SUMMARY")
    print(json.dumps(summary["methods"], indent=2))
    print("Candidates:", len(summary["candidates"]), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Paired benchmark: main-branch primal LM vs branch primal/dual LM.

Target: reduce the proven degree-10 PI triangle rule from 24 to 23 nodes.
Every seed gives both solvers the same random node-removal order and noise.
The run uses a fixed number of candidates and the same maximum LM iterations.
This tests search success, not just the algebraic equivalence of one LM step.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import tri_search as ts


ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "rules/minimal_pointcount_for_degree_n_PI/triangle/CA-T2-10-24-PI-MPC.json"
SEEDS = list(range(801, 809))
TRIES_PER_SEED = 24
MAXFEV = 300
NOISE = 1e-3
ACCEPT = 1e-13


def primal_lm_bounded(fun_jac, v0, lb, ub, max_nfev, tol=1e-15):
    """Original primal solver from main before the adaptive primal/dual change."""
    v = v0.copy()
    r, J = fun_jac(v)
    cost = 0.5 * float(r @ r)
    mu = 1e-3
    for _ in range(max_nfev):
        g = J.T @ r
        free = ~(((v <= lb + 1e-14) & (g > 0)) |
                 ((v >= ub - 1e-14) & (g < 0)))
        if not free.any() or np.abs(g[free]).max() < tol:
            break
        Jf = J[:, free]
        A = Jf.T @ Jf
        d = np.diag(A).copy()
        d[d < 1e-14] = 1e-14
        improved = False
        for _try in range(12):
            try:
                L = np.linalg.cholesky(A + mu * np.diag(d))
                step = -np.linalg.solve(L.T, np.linalg.solve(L, g[free]))
            except np.linalg.LinAlgError:
                mu *= 10
                continue
            vn = v.copy()
            vn[free] = np.clip(v[free] + step, lb[free], ub[free])
            rn, Jn = fun_jac(vn)
            cn = 0.5 * float(rn @ rn)
            if cn < cost:
                dc = cost - cn
                v, r, J, cost = vn, rn, Jn, cn
                mu = max(mu / 5, 1e-12)
                improved = True
                break
            mu *= 4
        if not improved or (dc < tol * max(cost, 1e-300) * 1e-3
                            and cost < 1e-28):
            break
        if cost < 1e-30:
            break
    return v, float(np.linalg.norm(r, np.inf))


def run_one(solver_name: str, solver, seed: int, out_path: Path):
    ts.lm_bounded = solver
    args = SimpleNamespace(
        degree=10,
        nodes=23,
        start=str(START),
        order="random",
        tries=TRIES_PER_SEED,
        noise=NOISE,
        maxfev=MAXFEV,
        accept=ACCEPT,
        out=str(out_path),
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    log = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(log):
        ts.mode_elim(args, rng, f"paired-benchmark:{solver_name}:seed={seed}")
    elapsed = time.perf_counter() - t0
    rows = []
    if out_path.exists():
        rows = [json.loads(line) for line in out_path.read_text().splitlines()
                if line.strip()]
    return {
        "solver": solver_name,
        "seed": seed,
        "attempts": TRIES_PER_SEED,
        "solutions": len(rows),
        "seconds": elapsed,
        "best_residual": min((r["residual"] for r in rows), default=None),
        "rules": rows,
        "log": log.getvalue(),
    }


def main():
    if not START.exists():
        raise FileNotFoundError(START)
    base = json.loads(START.read_text())
    if base.get("degree") != 10 or base.get("points") != 24:
        raise RuntimeError("Unexpected benchmark start rule metadata")

    results = []
    with tempfile.TemporaryDirectory(prefix="tri-lm-bench-") as tmp:
        tmp = Path(tmp)
        for seed in SEEDS:
            # Alternate order by seed to reduce systematic machine-load bias.
            variants = [
                ("primal", primal_lm_bounded),
                ("adaptive", ts.lm_bounded),
            ]
            if seed % 2:
                variants.reverse()
            for name, solver in variants:
                out = tmp / f"{name}-{seed}.jsonl"
                results.append(run_one(name, solver, seed, out))

    summary = {
        "target": "triangle degree 10 PI, 23 nodes (atlas baseline: 24)",
        "seeds": SEEDS,
        "tries_per_seed": TRIES_PER_SEED,
        "attempts_per_solver": len(SEEDS) * TRIES_PER_SEED,
        "maxfev": MAXFEV,
        "noise": NOISE,
        "accept": ACCEPT,
        "solvers": {},
        "paired": [],
        "rule_candidates": [],
    }
    for name in ("primal", "adaptive"):
        part = [r for r in results if r["solver"] == name]
        summary["solvers"][name] = {
            "attempts": sum(r["attempts"] for r in part),
            "solutions": sum(r["solutions"] for r in part),
            "seconds": sum(r["seconds"] for r in part),
            "median_seconds_per_seed": float(np.median([r["seconds"] for r in part])),
        }
    for seed in SEEDS:
        p = next(r for r in results if r["seed"] == seed and r["solver"] == "primal")
        a = next(r for r in results if r["seed"] == seed and r["solver"] == "adaptive")
        summary["paired"].append({
            "seed": seed,
            "primal_solutions": p["solutions"],
            "adaptive_solutions": a["solutions"],
            "primal_seconds": p["seconds"],
            "adaptive_seconds": a["seconds"],
            "primal_best_residual": p["best_residual"],
            "adaptive_best_residual": a["best_residual"],
        })
        for result in (p, a):
            for rule in result["rules"]:
                summary["rule_candidates"].append({
                    "solver": result["solver"],
                    "seed": seed,
                    "residual": rule["residual"],
                    "nodes": rule["nodes"],
                    "weights": rule["weights"],
                })

    Path("tri_lm_benchmark_results.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "rule_candidates"}, indent=2))
    print("\nCandidate rules:", len(summary["rule_candidates"]))
    print("Full results: tri_lm_benchmark_results.json")


if __name__ == "__main__":
    main()

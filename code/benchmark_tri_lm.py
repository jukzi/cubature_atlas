#!/usr/bin/env python3
"""Paired benchmark: old primal LM vs adaptive primal/dual LM.

Target: turn the existing degree-12 PO triangle rule (31 nodes) into a PI
rule with the same 31 nodes, improving the atlas PI count from 32 to 31.
Each seed, continuation step, and perturbation is identical for both solvers.
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import tri_search as ts


ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "rules/minimal_pointcount_for_degree_n_PO/triangle/CA-T2-12-31-PO-MPC.json"
SEEDS = list(range(801, 805))
TRIES = 6
HSTEPS = 6
MAXFEV = 250
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
        degree=12,
        nodes=31,
        start=str(START),
        tries=TRIES,
        noise=NOISE,
        hsteps=HSTEPS,
        maxfev=MAXFEV,
        accept=ACCEPT,
        out=str(out_path),
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    log = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(log):
        ts.mode_po2pi(args, rng, f"paired-benchmark:{solver_name}:seed={seed}")
    elapsed = time.perf_counter() - t0
    rows = []
    if out_path.exists():
        rows = [json.loads(line) for line in out_path.read_text().splitlines()
                if line.strip()]
    return {
        "solver": solver_name,
        "seed": seed,
        "max_repetitions": TRIES,
        "homotopy_steps": HSTEPS,
        "lm_calls": log.getvalue().count(" res "),
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
    if base.get("degree") != 12 or base.get("points") != 31 or base.get("quality") != "PO":
        raise RuntimeError("Unexpected benchmark start rule metadata")

    results = []
    with tempfile.TemporaryDirectory(prefix="tri-lm-bench-") as tmp:
        tmp = Path(tmp)
        for seed in SEEDS:
            variants = [("primal", primal_lm_bounded), ("adaptive", ts.lm_bounded)]
            if seed % 2:
                variants.reverse()
            for name, solver in variants:
                results.append(run_one(name, solver, seed, tmp / f"{name}-{seed}.jsonl"))

    summary = {
        "target": "triangle degree 12 PI, 31 nodes (atlas baseline: PI 32; proven PO 31)",
        "start_rule": "CA-T2-12-31-PO-MPC",
        "seeds": SEEDS,
        "max_repetitions_per_seed": TRIES,
        "homotopy_steps": HSTEPS,
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
            "max_lm_calls": len(SEEDS) * TRIES * HSTEPS,
            "actual_lm_calls": sum(r["lm_calls"] for r in part),
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
            "primal_lm_calls": p["lm_calls"],
            "adaptive_lm_calls": a["lm_calls"],
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

    Path("tri_lm_benchmark_results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rule_candidates"}, indent=2))
    print("\nCandidate rules:", len(summary["rule_candidates"]))
    print("Full results: tri_lm_benchmark_results.json")


if __name__ == "__main__":
    main()

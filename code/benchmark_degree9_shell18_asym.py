#!/usr/bin/env python3
"""Search degree-9, 18-node PI triangle rules from random nested C3 shells.

This search is independent of node deletion: each start is built directly from
nested Chebyshev-Lobatto shells, with bubble-scaled perturbation and inner-shell
rotation, then optimized in the C3-symmetric 18-variable space.
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tri_search as TS


DEGREE = 9
NODES = 18
SEEDS = tuple(range(9811, 9817))
TRIES_PER_SEED = 250
STARTS_PER_PATTERN = 10
SHORT_FEV = 80
MAXFEV = 500
NOISE = 0.05
ROT = 0.5
PROMOTE = 0.0
ACCEPT = 1e-13


def run_seed(seed, out):
    args = SimpleNamespace(
        degree=DEGREE,
        nodes=NODES,
        seed=seed,
        mode="shell",
        tries=TRIES_PER_SEED,
        minutes=0.0,
        maxfev=MAXFEV,
        accept=ACCEPT,
        noise=NOISE,
        rot=ROT,
        asym_polish=0.05,
        short_fev=SHORT_FEV,
        promote=PROMOTE,
        starts_per_pattern=STARTS_PER_PATTERN,
        near="",
        near_prob=0.0,
        out=str(out),
        max_found=1,
    )
    log = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(log):
        TS.mode_shell(args, np.random.default_rng(seed),
                      f"python code/tri_search.py --mode shell --degree 9 --nodes 18 --seed {seed} --tries 250 --starts_per_pattern 10 --short_fev 80 --maxfev 500 --noise 0.05 --rot 0.5 --asym_polish 0.05 --promote 0 --max_found 1 --out {out}")
    elapsed = time.perf_counter() - t0
    text = log.getvalue()
    best = None
    for m in re.finditer(r"pattern (\[[^\]]+\]) res ([0-9.eE+-]+) \(best so far\)", text):
        best = float(m.group(2))
    found = []
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                found.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return {
        "seed": seed,
        "elapsed_seconds": elapsed,
        "best_short_residual": best,
        "solutions": len(found),
        "best_candidate": found[0] if found else None,
        "log": text,
    }


def main():
    results = []
    with tempfile.TemporaryDirectory(prefix="tri9-shell18-") as tmp:
        tmp = Path(tmp)
        for seed in SEEDS:
            out = tmp / f"shell-{seed}.jsonl"
            print(f"SEED_START {seed}", flush=True)
            r = run_seed(seed, out)
            results.append(r)
            print(
                f"SEED_END {seed} best={r['best_short_residual']} "
                f"solutions={r['solutions']} seconds={r['elapsed_seconds']:.1f}",
                flush=True,
            )
            if r["solutions"]:
                print(f"SOLUTION_FOUND seed={seed}", flush=True)

    summary = {
        "target": "triangle degree 9 PI, 18 nodes (current proven count 19)",
        "method": "random nested C3 shells, then full asymmetric LM polish of near-solutions",
        "seeds": SEEDS,
        "tries_per_seed": TRIES_PER_SEED,
        "starts_per_pattern": STARTS_PER_PATTERN,
        "short_fev": SHORT_FEV,
        "asym_polish_threshold": 0.05,
        "maxfev": MAXFEV,
        "noise": NOISE,
        "rotation": ROT,
        "promote_residual": PROMOTE,
        "accept": ACCEPT,
        "total_starts": sum(min(TRIES_PER_SEED, STARTS_PER_PATTERN * 1000) for _ in SEEDS),
        "total_solutions": sum(r["solutions"] for r in results),
        "best_residual": min((r["best_short_residual"] for r in results if r["best_short_residual"] is not None), default=None),
        "elapsed_seconds": sum(r["elapsed_seconds"] for r in results),
        "results": results,
    }
    Path("degree9_shell18_asym_results.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print("FINAL SUMMARY")
    print(json.dumps({k:v for k,v in summary.items() if k != "results"}, indent=2), flush=True)


if __name__ == "__main__":
    main()

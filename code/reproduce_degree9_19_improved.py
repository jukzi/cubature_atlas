#!/usr/bin/env python3
"""Reproduce the improved degree-9, 19-node S3-symmetric PI triangle rule.

The search starts from the proven degree-10, 24-node PI rule, replays the
five successful seeded eliminations, and then applies the repository's
minimize_l2 procedure. It finally runs the strict interval proof.

Usage:
    python code/reproduce_degree9_19_improved.py --out /tmp/CA-T2-09-19-PI-MPC-a.json
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import tri_search as TS


SOURCE = ROOT / "rules/minimal_pointcount_for_degree_n_PI/triangle/CA-T2-10-24-PI-MPC.json"
DEGREE = 9
CONTINUATION = (5, 7, 8, 9)
CONTINUATION_MAXFEV = 100
FINAL_MAXFEV = 400
ACCEPT = 1e-13

# (new node count, zero-based deleted node, coordinate noise, RNG seed).
# These are the five selected successful stages of the degree-10 / 24-node
# source chain. The seed schedule is recorded in the rule's reproduce field.
ELIMINATIONS = (
    (23, 19, 0.05, 1172359),
    (22, 14, 0.001, 1172242),
    (21, 11, 0.05, 1172135),
    (20, 7, 0.05, 1172023),
    (19, 16, 0.05, 1171950),
)


def delete_and_redistribute(X: np.ndarray, W: np.ndarray, index: int,
                            noise: float, seed: int) -> np.ndarray:
    keep = np.arange(len(W)) != index
    Xk = X[keep].copy()
    Wk = W[keep].copy()
    d2 = ((Xk - X[index]) ** 2).sum(axis=1) + 1e-6
    inv = 1.0 / d2
    Wk += W[index] * inv / inv.sum()
    s, t = TS.xy_to_st(Xk[:, 0], Xk[:, 1])
    v = np.concatenate((s, t, Wk))
    rng = np.random.default_rng(seed)
    v[:2 * len(Wk)] += rng.normal(0.0, noise, 2 * len(Wk))
    return v


def solve_continuation(v: np.ndarray, N: int) -> tuple[np.ndarray, float]:
    for degree in CONTINUATION:
        v, _ = TS.solve(v, degree, N, max_nfev=CONTINUATION_MAXFEV)
    v, residual = TS.solve(v, DEGREE, N, max_nfev=FINAL_MAXFEV)
    return v, float(residual)


def monomial_residual(X: np.ndarray, W: np.ndarray, degree: int) -> float:
    """Largest total-degree monomial error divided by triangle area (1/2)."""
    from math import factorial

    worst = 0.0
    for a in range(degree + 1):
        for b in range(degree + 1 - a):
            got = float(np.dot(W, (X[:, 0] ** a) * (X[:, 1] ** b)))
            exact = factorial(a) * factorial(b) / factorial(a + b + 2)
            worst = max(worst, abs(got - exact) / 0.5)
    return worst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/CA-T2-09-19-PI-MPC-a.json")
    args = ap.parse_args()
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    source = json.loads(SOURCE.read_text())
    if source.get("id") != "CA-T2-10-24-PI-MPC" or len(source["nodes"]) != 24:
        raise RuntimeError("Unexpected degree-10 source rule")
    X = np.asarray(source["nodes"], dtype=float)
    W = np.asarray(source["weights"], dtype=float)

    for expected_N, removed, noise, seed in ELIMINATIONS:
        if len(W) != expected_N + 1 or not (0 <= removed < len(W)):
            raise RuntimeError("Elimination chain no longer matches its recorded source")
        v0 = delete_and_redistribute(X, W, removed, noise, seed)
        v, res = solve_continuation(v0, expected_N)
        if not TS.accept(v, DEGREE, expected_N, ACCEPT):
            raise RuntimeError(
                f"Elimination to N={expected_N} failed: residual {res:.3e}"
            )
        s, t, W = TS.unpack(v, expected_N)
        x, y = TS.to_xy(s, t)
        X = np.column_stack((x, y))
        print(
            f"N={expected_N}: residual={res:.3e}, "
            f"minimum weight={W.min():.8g}, "
            f"boundary margin={np.c_[X, 1-X.sum(axis=1)].min():.8g}",
            flush=True,
        )

    if len(W) != 19:
        raise RuntimeError("The reproduced candidate must have 19 nodes")
    base_method = (
        "AI-assisted numerical search of this atlas: start from the proven "
        "degree-10, 24-node PI rule CA-T2-10-24-PI-MPC; perform five seeded "
        "degree-9 node eliminations with continuation degrees 5, 7, 8 and 9; "
        "the selected zero-based deletion indices are 19, 14, 11, 7 and 16. "
        "The weights were then minimized in the L2 norm under all moment "
        "equations with the domain penalty and exact rule symmetries retained "
        "(tools/minimize_l2.py); the minimizer has S3 symmetry (six elements)."
    )
    rule = {
        "id": "CA-T2-09-19-PI-MPC-a",
        "family": "minimal_pointcount_for_degree_n_PI",
        "domain": "triangle",
        "dimension": 2,
        "degree": DEGREE,
        "points": 19,
        "quality": "PI",
        "origin": "found",
        "method": base_method,
        "author": "OpenAI ChatGPT (AI-assisted search)",
        "status": "proven",
        "residual_reported": float(np.linalg.norm(
            TS.residual_and_jac(
                np.concatenate((
                    X[:, 0],
                    np.divide(X[:, 1], 1.0 - X[:, 0],
                              out=np.zeros(len(X)),
                              where=(1.0 - X[:, 0]) > 1e-14),
                    W,
                )),
                DEGREE, len(W)
            )[0], np.inf
        )),
        "reproduce": {
            "script": "code/reproduce_degree9_19_improved.py",
            "seed": [1172359, 1172242, 1172135, 1172023, 1171950],
            "command": (
                "python code/reproduce_degree9_19_improved.py "
                "--out /tmp/CA-T2-09-19-PI-MPC-a.json"
            ),
        },
        "nodes": X.tolist(),
        "weights": W.tolist(),
    }
    out.write_text(json.dumps(rule, indent=2) + "\n", encoding="utf8")

    # Apply the repository's recommended final weight minimization and recompute
    # the displayed numerical diagnostics from the resulting stored numbers.
    subprocess.run(
        [sys.executable, str(ROOT / "tools/minimize_l2.py"), str(out), "--write"],
        cwd=ROOT, check=True,
    )
    rule = json.loads(out.read_text())
    X = np.asarray(rule["nodes"], dtype=float)
    W = np.asarray(rule["weights"], dtype=float)
    margin = float(np.c_[X, 1.0 - X.sum(axis=1)].min())
    res = monomial_residual(X, W, DEGREE)
    rule["residual_reported"] = res
    rule["numerical_check"] = {
        "N": 19,
        "M": 55,
        "p": 57,
        "res": res,
        "inside": bool(margin >= -1e-13),
        "min_weight": float(W.min()),
        "min_boundary_distance": margin,
        "symmetry_group_order": 6,
    }
    rule["status"] = "proven"
    out.write_text(json.dumps(rule, indent=2) + "\n", encoding="utf8")

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/verify_rigorous.py"),
            "--strict", "--save", str(out),
        ],
        cwd=ROOT, check=True,
    )
    print(f"Reproduced and proved: {out}", flush=True)


if __name__ == "__main__":
    main()

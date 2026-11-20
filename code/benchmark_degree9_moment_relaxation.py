#!/usr/bin/env python3
"""Numerical moment-matrix relaxation for the triangle degree-9 lower bound.

For degree 9 (k=4), test whether the truncated moment Gram matrix of P_5
can be positive semidefinite with ranks r=1,2,3,4. A positive cubature rule
with N nodes necessarily has rank <= N - dim(P_4), and dim(P_4)=15.
Thus r=2 and r=3 are the relaxation targets for N=17 and N=18.

This is a numerical necessary-condition test, not a proof of nonexistence or
existence of cubature rules. The best moment deviations and factor matrices
are saved for any feasible relaxation, for possible follow-up extraction.
"""
from __future__ import annotations

import json
from math import factorial
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


K = 4
DOMAIN = "triangle"
RANKS = (1, 2, 3, 4)
TRIALS = 200
MAX_NFEV = 3000
TOL = 1e-12


def exact_degree(dim, degree):
    if dim == 1:
        return [(degree,)]
    out = []
    for a in range(degree, -1, -1):
        for rest in exact_degree(dim - 1, degree - a):
            out.append((a,) + rest)
    return out


def moment(exponents):
    numerator = 1
    for a in exponents:
        numerator *= factorial(a)
    return numerator / factorial(sum(exponents) + len(exponents))


def build_system():
    basis = []
    for degree in range(K + 2):
        basis.extend(exact_degree(2, degree))
    gram = np.array([
        [moment(tuple(x + y for x, y in zip(a, b))) for b in basis]
        for a in basis
    ])
    coef = np.linalg.inv(np.linalg.cholesky(gram))
    top = [i for i, e in enumerate(basis) if sum(e) == K + 1]
    C = coef[np.ix_(top, top)]
    m = len(top)
    high = exact_degree(2, 2 * K + 2)
    position = {e: i for i, e in enumerate(high)}
    index = np.array([
        [position[tuple(x + y for x, y in zip(basis[top[i]], basis[top[j]]))]
         for j in range(m)]
        for i in range(m)
    ])
    upper = np.triu_indices(m)
    dim_pk = len(basis) - m
    return C, m, high, index, upper, dim_pk


def run_rank(r, C, m, high, index, upper, dim_pk, rng):
    num_high = len(high)

    def residual(z):
        H = z[:num_high][index]
        F = z[num_high:].reshape(m, r)
        return (np.eye(m) + C @ H @ C.T - F @ F.T)[upper]

    best_norm = float("inf")
    best_z = None
    best_trial = None
    for trial in range(TRIALS):
        z0 = np.r_[0.01 * rng.standard_normal(num_high),
                   rng.standard_normal(m * r)]
        sol = least_squares(
            residual, z0, method="trf",
            xtol=1e-15, ftol=1e-15, gtol=1e-15,
            max_nfev=MAX_NFEV,
        )
        norm = float(np.linalg.norm(sol.fun))
        if norm < best_norm:
            best_norm = norm
            best_z = sol.x.copy()
            best_trial = trial
            print(f"rank={r} trial={trial+1} best={best_norm:.3e}", flush=True)
        if best_norm < TOL:
            break

    F = best_z[num_high:].reshape(m, r)
    s = best_z[:num_high]
    return {
        "rank": r,
        "nodes_lower_bound": dim_pk + r,
        "equations": len(upper),
        "unknowns_before_gauge": num_high + m * r,
        "unknowns_after_gauge": num_high + m * r - r * (r - 1) // 2,
        "trials": best_trial + 1,
        "best_residual": best_norm,
        "numerically_feasible": bool(best_norm < TOL),
        "moment_error_norm": float(np.linalg.norm(s)),
        "factor_norm": float(np.linalg.norm(F)),
    }, s, F


def main():
    C, m, high, index, upper, dim_pk = build_system()
    rng = np.random.default_rng(0)
    summaries = []
    factors = {}
    for r in RANKS:
        report, s, F = run_rank(r, C, m, high, index, upper, dim_pk, rng)
        summaries.append(report)
        factors[str(r)] = {"moment_deviation": s.tolist(), "factor": F.tolist()}

    result = {
        "domain": DOMAIN,
        "degree": 2 * K + 1,
        "dim_P_k": dim_pk,
        "ranks_tested": RANKS,
        "trials_per_rank_max": TRIALS,
        "max_nfev": MAX_NFEV,
        "tolerance": TOL,
        "interpretation": (
            "Numerical PSD-rank relaxation only. A feasible factorization is "
            "necessary but not sufficient for a cubature rule; failure to find "
            "one in finitely many starts is not a proof of impossibility."
        ),
        "results": summaries,
        "factors": factors,
    }
    Path("degree9_moment_relaxation.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print("FINAL SUMMARY")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()

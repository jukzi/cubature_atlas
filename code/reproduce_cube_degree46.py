"""Prepare a degree-46 cube rule in tetrahedral rotation orbits."""
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from cubature_search import Problem, Rule, actual_points, apply_group, eliminate, search_degree, solve_rule, verify


def load_subgroup(path, degree=46, group="T"):
    problem = Problem(degree, group)
    data = {tuple(row[:3]): row[3] for row in np.loadtxt(path)}
    patterns, coords, weights = [], [], []
    while data:
        node = np.array(next(iter(data)))
        images = apply_group(problem.perms, problem.signs, node[None, :])[:, 0, :]
        orbit = {tuple(image) for image in images}
        weight = sum(data.pop(image) for image in sorted(orbit)) / 8
        for index, pattern in enumerate(problem.patterns):
            if pattern.size != len(orbit):
                continue
            fits = (pattern.fit(image, tol=1e-12) for image in images)
            fit = next((fit for fit in fits if fit is not None), None)
            if fit is not None:
                patterns.append(index)
                coords.append(fit)
                weights.append(weight)
                break
        else:
            raise ValueError("Unmatched orbit")
    return Rule(problem, patterns, coords, weights)


def pull_inside(rule, margin=1e-6):
    """Move the boundary orbit inward and refit the remaining variables."""
    variables = rule.pack()
    coordinate_count = len(variables) - len(rule.weights)
    fixed = np.zeros_like(variables, dtype=bool)
    fixed[:coordinate_count] = np.abs(variables[:coordinate_count]) > 1 - margin
    variables[fixed] = np.sign(variables[fixed]) * (1 - margin)
    rule.unpack(variables)
    ok, _ = solve_rule(rule, tol=1e-14, maxit=100, fixed=fixed)
    if not ok:
        raise ValueError("Could not refit the rule with its boundary orbit inside the cube")


def minimise_weights(rule):
    """Minimise sum of squared node weights with the orbit moments constrained."""
    start = rule.pack()
    coordinate_count = len(start) - len(rule.weights)
    sizes = np.array([rule.prob.patterns[i].size for i in rule.pat_ids])
    scale = np.sum(rule.weights ** 2 / sizes)

    def objective(variables):
        return np.sum(variables[coordinate_count:] ** 2 / sizes) / scale

    def gradient(variables):
        return np.r_[np.zeros(coordinate_count), 2 * variables[coordinate_count:] / sizes / scale]

    def residual(variables):
        rule.unpack(variables)
        return rule.residual_jacobian(deriv=False)[0]

    def jacobian(variables):
        rule.unpack(variables)
        return rule.residual_jacobian()[1]

    bounds = [(-1 + 1e-6, 1 - 1e-6)] * coordinate_count + [(1e-9, 1)] * len(sizes)
    result = minimize(
        objective, start, jac=gradient, method="SLSQP", bounds=bounds,
        constraints={"type": "eq", "fun": residual, "jac": jacobian},
        options={"maxiter": 30, "ftol": 1e-13},
    )
    rule.unpack(result.x)
    ok, _ = solve_rule(rule, tol=1e-14)
    if not ok or objective(rule.pack()) >= 1:
        raise ValueError("Could not recover a feasible rule with smaller squared weights")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=112)
    parser.add_argument("--start", type=Path, help="Reuse a saved degree-47 Oh search result")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.start is None:
        source = search_degree(
            47, group="Oh", restarts=1, time_limit=0, seed=84,
            outdir=str(args.out), escape_tries=0,
        )
        args.start = source["file"]
    start = load_subgroup(args.start)
    ok, _ = solve_rule(start)
    if not ok:
        raise ValueError("Starting rule does not satisfy the degree-46 moments")
    rule = eliminate(start, rng=np.random.default_rng(args.seed), noise=0.1, power=4.0, log=print)
    nodes, weights = actual_points(rule)
    raw = args.out / f"cube3d_n46_m{len(nodes)}_T_raw.txt"
    np.savetxt(raw, np.c_[nodes, 8 * weights], fmt="%.17e", header=f"seed={args.seed} source={args.start}")
    rule = load_subgroup(raw)
    print("orbit variables", len(rule.pack()), "equations", rule.prob.neq, flush=True)
    pull_inside(rule)
    minimise_weights(rule)
    nodes, weights = actual_points(rule)
    residual = verify(nodes, weights, 46)
    if residual > 1e-13 or np.any(weights <= 0) or np.any(np.abs(nodes) >= 1):
        raise ValueError("The rule failed the numerical moment or quality check")
    out = args.out / f"cube3d_n46_m{len(nodes)}_T_minimized.txt"
    np.savetxt(out, np.c_[nodes, 8 * weights], fmt="%.17e", header=f"seed={args.seed} residual={residual:.3e}")
    print(out)


if __name__ == "__main__":
    main()

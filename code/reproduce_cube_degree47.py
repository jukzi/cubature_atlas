"""Reproduce the degree-47 cube rule using orbit elimination and weight minimisation."""
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from cubature_search import Problem, Rule, actual_points, orbit_points, search_degree, solve_rule, verify


def load_octahedral_rule(path):
    """Group signed permutations by their sorted absolute coordinates."""
    problem = Problem(47, "Oh")
    groups = {}
    for row in np.loadtxt(path):
        key = tuple(sorted(np.abs(row[:3])))
        groups.setdefault(key, []).append(row[3])
    patterns, coordinates, weights = [], [], []
    for point, orbit_weights in groups.items():
        images = orbit_points(problem.perms, problem.signs, np.array(point))
        for index, pattern in enumerate(problem.patterns):
            if pattern.size != len(orbit_weights):
                continue
            fits = (pattern.fit(image, tol=1e-12) for image in images)
            fit = next((fit for fit in fits if fit is not None), None)
            if fit is not None:
                patterns.append(index)
                coordinates.append(fit)
                weights.append(sum(orbit_weights) / 8)
                break
        else:
            raise ValueError("No pattern matches an input orbit")
    return Rule(problem, patterns, coordinates, weights)


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

    bounds = [(-1 + 1e-7, 1 - 1e-7)] * coordinate_count + [(1e-9, 1)] * len(sizes)
    result = minimize(
        objective, start, jac=gradient, method="SLSQP", bounds=bounds,
        constraints={"type": "eq", "fun": residual, "jac": jacobian},
        options={"maxiter": 120, "ftol": 1e-13},
    )
    rule.unpack(result.x)
    ok, _ = solve_rule(rule, tol=1e-14)
    if not ok or objective(rule.pack()) >= 1:
        raise ValueError("Could not recover a feasible rule with smaller squared weights")
    print(f"Squared weight ratio: {objective(rule.pack()):.10f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=84)
    parser.add_argument("--start", type=Path, help="Reuse a saved degree-47 Oh search result")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.start is None:
        # A single elimination run makes the result independent of run time.
        source = search_degree(
            47, group="Oh", restarts=1, time_limit=0, seed=args.seed,
            outdir=str(args.out), escape_tries=0,
        )
        args.start = source["file"]
    rule = load_octahedral_rule(args.start)
    minimise_weights(rule)
    nodes, weights = actual_points(rule)
    residual = verify(nodes, weights, 47)
    if residual > 1e-13 or np.any(weights <= 0) or np.any(np.abs(nodes) >= 1):
        raise ValueError("The rule failed its moment or quality check")
    filename = args.out / f"cube3d_n47_m{len(nodes)}_Oh_minimized.txt"
    np.savetxt(filename, np.c_[nodes, 8 * weights], fmt="%.17e",
               header=f"degree=47 seed={args.seed} max_moment_error={residual:.3e}")
    print(filename)


if __name__ == "__main__":
    main()

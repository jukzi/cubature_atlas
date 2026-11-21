"""Find square rules by continuing one weight to zero in Legendre moments."""
import argparse
import json
from pathlib import Path
import shlex
import sys

import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError

from square_elim import Problem


def correct(problem, values, fixed, goal, bound):
    """Correct all moments while keeping one weight prescribed."""
    count = len(values) // 3
    values = values.copy()
    values[fixed] = goal
    free = np.arange(len(values)) != fixed
    damping = 1e-10
    for _ in range(18):
        residual = problem.residual(values, count, False)
        error = np.linalg.norm(residual)
        if error < 2e-13:
            return values
        jac = problem.jacobian(values, count, False)[:, free]
        gram = jac @ jac.T
        for _ in range(6):
            system = gram.copy()
            system.flat[::len(system) + 1] += damping
            try:
                step = jac.T @ cho_solve(cho_factor(system, lower=True), -residual)
            except LinAlgError:
                damping *= 10
                continue
            trial = values.copy()
            trial[free] += step
            if (np.max(np.abs(trial[:2 * count])) <= bound
                    and np.linalg.norm(problem.residual(trial, count, False)) < error):
                values = trial
                damping = max(1e-14, damping / 5)
                break
            damping *= 10
        else:
            return None
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("degree", type=int)
    parser.add_argument("start", type=Path)
    parser.add_argument("seed", type=int)
    parser.add_argument("--finish", action="store_true")
    parser.add_argument("--tangent-order", action="store_true")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--bound", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.start.suffix == ".json":
        rule = json.loads(args.start.read_text())
        start = np.c_[rule["nodes"], rule["weights"]]
    else:
        start = np.load(args.start)
    count = len(start)
    problem = Problem(args.degree, False)
    if 3 * (count - 1) < problem.m:
        parser.error("the remaining system would have fewer variables than moments")
    rng = np.random.default_rng(args.seed)
    values = np.r_[start[:, :2].ravel(), start[:, 2]]
    scores = start[:, 2] * np.exp(rng.normal(0, 0.12, count))
    if args.tangent_order:
        # The nullspace estimates how much the weights can move locally.
        _, singular_values, right = np.linalg.svd(problem.jacobian(values, count, False),
                                                  full_matrices=True)
        tangent = np.linalg.norm(right[len(singular_values):, 2 * count:], axis=0)
        scores /= np.maximum(tangent, 1e-15)
    order = np.argsort(scores)
    args.out.mkdir(parents=True, exist_ok=True)
    for node in order[args.offset:args.offset + args.trials]:
        values = np.r_[start[:, :2].ravel(), start[:, 2]]
        fixed = 2 * count + node
        weight = values[fixed]
        fraction = 1.0
        stride = 0.2
        while fraction > 0 and stride >= 0.0005:
            next_fraction = max(0.0, fraction - stride)
            candidate = correct(problem, values, fixed, next_fraction * weight, args.bound)
            if candidate is None:
                stride /= 2
            else:
                values = candidate
                fraction = next_fraction
                stride = min(0.3, stride * 1.4)
        print(f"node {node} remaining weight fraction {fraction:.6g}", flush=True)
        keep = np.arange(count) != node
        points, weights, _ = problem.unpack(values, count, False)
        points = points[keep]
        weights = weights[keep]
        if fraction > 0:
            if not args.finish or fraction > 0.35:
                continue
            problem.bound = args.bound
            reduced = np.r_[points.ravel(), weights]
            reduced, cost = problem.solve(reduced, count - 1, False, 150)
            print(f"node {node} finish cost {cost:.6g}", flush=True)
            if cost >= 1e-26:
                continue
            points, weights, _ = problem.unpack(reduced, count - 1, False)
        if weights.min() <= 0:
            continue
        result = np.c_[points, weights]
        file = args.out / f"square-d{args.degree}-n{count-1}-s{args.seed}-drop{node}.npy"
        np.save(file, result)
        file.with_suffix(".json").write_text(json.dumps({
            "seed": args.seed, "start": str(args.start),
            "command": shlex.join(["python", *sys.argv]), "removed_index": int(node),
        }, indent=2) + "\n")
        print(f"FOUND {file} max coordinate {abs(result[:, :2]).max():.10g}", flush=True)
        break


if __name__ == "__main__":
    main()

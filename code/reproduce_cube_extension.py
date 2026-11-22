"""Reproduce cube rules by raising the degree of an octahedral rule."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.linalg import qr, solve_triangular

from cubature_search import actual_points, augment, canonicalize, eliminate, load_rule_for, solve_rule

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "rules/diallo_worku_2026/cube/CA-C3-45-4414-PI-DW.json"


def warm_start(seed):
    """Raise the degree from 45 to 47 with seeded orbit augmentation."""
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    with TemporaryDirectory() as directory:
        points = Path(directory) / "source.txt"
        np.savetxt(points, np.column_stack((source["nodes"], source["weights"])))
        base = load_rule_for({"n": 47, "file": points}, "Oh")
    rng = np.random.default_rng(seed)
    generic = max(range(len(base.prob.patterns)), key=lambda i: base.prob.patterns[i].size)
    for count in (0, 4, 8, 12, 20, 30):
        candidate = augment(base, {generic: count}, rng) if count else base.copy()
        converged, residual = solve_rule(candidate, maxit=200, patience=35)
        print(f"Augmented orbits {count}, residual {residual:.3e}", flush=True)
        if converged:
            canonicalize(candidate)
            return candidate
    raise RuntimeError("Could not raise the degree to 47")


def minimise_weights(rule):
    """Descend the squared weight norm with the orbit moments constrained."""
    sizes = np.array([rule.prob.patterns[p].size for p in rule.pat_ids])
    lower, upper = rule.bounds()
    weights = slice(-len(sizes), None)
    initial = rule.pack()
    margin = 1 - np.max(np.abs(rule.generators()))
    lower[:-len(sizes)] = np.maximum(lower[:-len(sizes)], -(1 - margin))
    upper[:-len(sizes)] = np.minimum(upper[:-len(sizes)], 1 - margin)

    def objective(variables):
        return np.sum(variables[weights] ** 2 / sizes)

    current = initial
    for _ in range(20):
        rule.unpack(current)
        _, jacobian = rule.residual_jacobian()
        basis, factorization = qr(jacobian.T, mode="economic")
        gradient = np.zeros_like(current)
        gradient[weights] = 2 * current[weights] / sizes
        direction = basis @ (basis.T @ gradient) - gradient
        if np.max(np.abs(direction)) < 1e-15:
            break
        for factor in np.geomspace(1000, 0.1, num=16):
            trial = current + factor * direction
            for _ in range(8):
                if not np.all((lower <= trial) & (trial <= upper)):
                    break
                rule.unpack(trial)
                residual, _ = rule.residual_jacobian(deriv=False)
                if np.max(np.abs(residual)) < 1e-14:
                    if objective(trial) < objective(current):
                        current = trial
                    break
                trial -= basis @ solve_triangular(factorization.T, residual, lower=True)
            if current is trial:
                break
        else:
            break
    rule.unpack(current)
    print(f"Squared weight norm {64 * objective(initial):.12g} -> {64 * objective(current):.12g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, choices=(46, 47), default=47)
    parser.add_argument("--seed", type=int, default=740)
    parser.add_argument("--moves", type=int, default=29)
    parser.add_argument("--start", type=Path, help="Saved degree-47 raw rule for the degree-46 search")
    parser.add_argument("--raw", action="store_true", help="Save the result before weight minimisation")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    group = "Oh" if args.degree == 47 else "Td"
    if args.start is None:
        if args.degree != 47:
            parser.error("--start is required for degree 46")
        with TemporaryDirectory() as directory:
            source = Path(directory) / "warm.txt"
            nodes, weights = actual_points(warm_start(args.seed))
            np.savetxt(source, np.column_stack((nodes, 8 * weights)), fmt="%.17e")
            rule = load_rule_for({"n": args.degree, "file": source}, group)
    else:
        rule = load_rule_for({"n": args.degree, "file": args.start}, group)
    ok, residual = solve_rule(rule)
    if not ok:
        raise RuntimeError(f"Starting rule residual {residual:.3e}")
    rule = eliminate(rule, max_steps=args.moves, log=lambda line: print(line, flush=True))

    # Reload the expanded orbits before minimisation to use the same ordering as a saved search result.
    args.out.mkdir(parents=True, exist_ok=True)
    nodes, weights = actual_points(rule)
    path = args.out / f"cube3d_n{args.degree}_m{len(nodes)}_{group}.txt"
    np.savetxt(path, np.column_stack((nodes, 8 * weights)), fmt="%.17e")
    if not args.raw:
        rule = load_rule_for({"n": args.degree, "file": path}, group)
        minimise_weights(rule)
        nodes, weights = actual_points(rule)
        if np.any(weights <= 0) or np.any(np.abs(nodes) > 1):
            raise RuntimeError("The resulting rule failed its quality check")
        np.savetxt(path, np.column_stack((nodes, 8 * weights)), fmt="%.17e")
    print(path)


if __name__ == "__main__":
    main()

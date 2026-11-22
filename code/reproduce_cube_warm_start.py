"""Reduce a stored cube rule to a smaller rule of lower degree."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from cubature_search import actual_points, eliminate, load_rule_for, solve_rule
from reproduce_cube_degree49 import pull_inside
from reproduce_cube_extension import minimise_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Atlas JSON rule used as a warm start")
    parser.add_argument("--degree", type=int, required=True)
    parser.add_argument("--group", default="Td")
    parser.add_argument("--moves", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0)
    parser.add_argument("--pull-inside", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.source.read_text(encoding="utf-8"))
    with TemporaryDirectory() as directory:
        path = Path(directory) / "start.txt"
        np.savetxt(path, np.column_stack((source["nodes"], source["weights"])), fmt="%.17e")
        rule = load_rule_for({"n": args.degree, "file": path}, args.group)

    converged, residual = solve_rule(rule)
    if not converged:
        raise RuntimeError(f"Starting rule residual {residual:.3e}")
    rule = eliminate(
        rule, max_steps=args.moves, rng=np.random.default_rng(args.seed), noise=args.noise,
        log=lambda line: print(line, flush=True),
    )
    nodes, weights = actual_points(rule)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"cube3d_n{args.degree}_m{len(nodes)}_{args.group}.txt"
    np.savetxt(path, np.column_stack((nodes, 8 * weights)), fmt="%.17e")

    rule = load_rule_for({"n": args.degree, "file": path}, args.group)
    if args.pull_inside:
        pull_inside(rule)
    minimise_weights(rule)
    nodes, weights = actual_points(rule)
    if np.any(weights <= 0) or np.any(np.abs(nodes) > 1):
        raise RuntimeError("The resulting rule failed its quality check")
    np.savetxt(path, np.column_stack((nodes, 8 * weights)), fmt="%.17e")
    print(path, flush=True)


if __name__ == "__main__":
    main()

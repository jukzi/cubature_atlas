"""Reduce the atlas degree-49 cube rule at degree 48 in tetrahedral orbits."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from cubature_search import actual_points, eliminate, solve_rule, verify
from reproduce_cube_degree46 import load_subgroup, pull_inside
from reproduce_cube_extension import minimise_weights

SOURCE = Path(__file__).resolve().parents[1] / (
    "rules/minimal_pointcount_for_degree_n_PI/cube/CA-C3-49-5739-PI-MPC.json"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--group", choices=("T", "Td"), required=True)
    parser.add_argument("--start", type=Path, help="Resume at weight minimisation with a saved degree-48 rule")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.start is None:
        source = json.loads(SOURCE.read_text(encoding="utf-8"))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "source.txt"
            np.savetxt(path, np.column_stack((source["nodes"], source["weights"])), fmt="%.17e")
            initial = load_subgroup(path, degree=48, group=args.group)
        ok, residual = solve_rule(initial)
        if not ok:
            raise ValueError(f"The starting rule did not converge, residual {residual:.3e}")
        rule = eliminate(initial, rng=np.random.default_rng(args.seed), noise=0.1, power=4,
                         log=lambda line: print(line, flush=True))
        nodes, weights = actual_points(rule)
        args.start = args.out / f"cube3d_n48_m{len(nodes)}_{args.group}_raw.txt"
        np.savetxt(args.start, np.column_stack((nodes, 8 * weights)), fmt="%.17e", header=f"seed={args.seed}")
    rule = load_subgroup(args.start, degree=48, group=args.group)
    pull_inside(rule)
    minimise_weights(rule)
    nodes, weights = actual_points(rule)
    residual = verify(nodes, weights, 48)
    if residual > 1e-13 or np.any(weights <= 0) or np.any(np.abs(nodes) >= 1):
        raise ValueError("The result failed the numerical moment or quality checks")
    path = args.out / f"cube3d_n48_m{len(nodes)}_{args.group}_minimized.txt"
    np.savetxt(path, np.column_stack((nodes, 8 * weights)), fmt="%.17e",
               header=f"seed={args.seed} residual={residual:.3e}")
    print(path)


if __name__ == "__main__":
    main()

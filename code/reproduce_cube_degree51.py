"""Extend the atlas degree-49 cube rule to degree 51 with octahedral symmetry."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from cubature_search import Problem, actual_points, augment, canonicalize, eliminate, solve_rule, verify
from reproduce_cube_degree46 import pull_inside
from reproduce_cube_degree49 import load_octahedral_rule
from reproduce_cube_extension import minimise_weights

SOURCE = Path(__file__).resolve().parents[1] / (
    "rules/minimal_pointcount_for_degree_n_PI/cube/CA-C3-49-5739-PI-MPC.json"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--orbits", type=int, required=True)
    parser.add_argument("--moves", type=int, default=35)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", type=Path, help="Resume from a saved raw degree-51 result")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.start is None:
        raw = json.loads(SOURCE.read_text(encoding="utf-8"))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "source.txt"
            np.savetxt(path, np.column_stack((raw["nodes"], raw["weights"])), fmt="%.17e")
            base = load_octahedral_rule(path)
        base.prob = Problem(51, "Oh")
        rng = np.random.default_rng(args.seed)
        generic = max(range(len(base.prob.patterns)), key=lambda i: base.prob.patterns[i].size)
        rule = augment(base, {generic: args.orbits}, rng)
        ok, residual = solve_rule(rule, maxit=260, patience=45)
        print(f"Augmented start: {rule.npoints()} nodes, residual {residual:.3e}", flush=True)
        if not ok:
            raise RuntimeError("Could not extend the degree-49 start")
        canonicalize(rule)
        rule = eliminate(rule, rng=rng, noise=0.1, power=4, max_steps=args.moves,
                         log=lambda message: print(message, flush=True))
        nodes, weights = actual_points(rule)
        args.start = args.out / f"cube3d_n51_m{len(nodes)}_Oh_raw.txt"
        np.savetxt(args.start, np.column_stack((nodes, 8 * weights)), fmt="%.17e")
    rule = load_octahedral_rule(args.start)
    rule.prob = Problem(51, "Oh")
    pull_inside(rule)
    minimise_weights(rule)
    nodes, weights = actual_points(rule)
    residual = verify(nodes, weights, 51)
    if residual > 1e-13 or np.any(weights <= 0) or np.any(np.abs(nodes) > 1):
        raise ValueError("The result failed the numerical moment or quality checks")
    path = args.out / f"cube3d_n51_m{len(nodes)}_Oh_minimized.txt"
    np.savetxt(path, np.column_stack((nodes, 8 * weights)), fmt="%.17e",
               header=f"seed={args.seed} orbits={args.orbits} moves={args.moves} residual={residual:.3e}")
    print(path)


if __name__ == "__main__":
    main()

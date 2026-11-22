"""Reproduce a cube rule by seeded large-neighbourhood search."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from cubature_search import actual_points, lns, load_rule_for, solve_rule, structure_target
from reproduce_cube_extension import minimise_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--group", choices=("Td", "Th", "Oh"), required=True)
    parser.add_argument("--pre-group", choices=("Td", "Th", "Oh"), help="Re-export the source through this orbit representation")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    degree = int(source["degree"])
    with TemporaryDirectory() as directory:
        path = Path(directory) / "source.txt"
        np.savetxt(path, np.column_stack((source["nodes"], source["weights"])), fmt="% .17e")
        if args.pre_group:
            prepared = load_rule_for({"n": degree, "file": path}, args.pre_group)
            converged, residual = solve_rule(prepared)
            if not converged:
                raise RuntimeError(f"Preprocessing residual {residual:.3e}")
            nodes, weights = actual_points(prepared)
            np.savetxt(path, np.column_stack((nodes, 8 * weights)), fmt="% .17e")
        rule = load_rule_for({"n": degree, "file": path}, args.group)
    converged, residual = solve_rule(rule)
    if not converged:
        raise RuntimeError(f"Starting rule residual {residual:.3e}")

    _, target = structure_target(rule.prob)
    rng = np.random.default_rng(args.seed + degree)
    result = lns(
        rule, rng, iters=args.iterations, temp0=4.0, target=target,
        log=lambda line: print(line, flush=True),
    )
    nodes, weights = actual_points(result)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"cube3d_n{degree:02d}_m{len(nodes)}_{args.group}.txt"
    np.savetxt(path, np.column_stack((nodes, 8 * weights)), fmt="% .17e")
    result = load_rule_for({"n": degree, "file": path}, args.group)
    minimise_weights(result)
    nodes, weights = actual_points(result)
    if np.any(weights <= 0) or np.any(np.abs(nodes) > 1):
        raise RuntimeError("The resulting rule failed its quality check")
    np.savetxt(path, np.column_stack((nodes, 8 * weights)), fmt="%.17e")
    print(path, flush=True)


if __name__ == "__main__":
    main()

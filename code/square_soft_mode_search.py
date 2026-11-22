"""Delete a square-rule node and polish seeded soft-mode perturbations."""

import argparse
import json
from pathlib import Path
import shlex
import sys

import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError, svd

from square_elim import Problem


def solve(residual, jacobian, x, max_iterations, bound=1.01):
    r = residual(x)
    cost = r @ r
    damping = 1e-4
    for iteration in range(max_iterations):
        if cost < 1e-26:
            break
        j = jacobian(x)
        scale = 1 / np.linalg.norm(j, axis=0).clip(1e-10)
        j *= scale
        gram = j.T @ j
        gradient = j.T @ r
        diagonal = np.diag_indices(len(x))
        for _ in range(12):
            regularized = gram.copy()
            regularized[diagonal] += damping
            try:
                step = cho_solve(cho_factor(regularized, lower=True, check_finite=False),
                                 -gradient, check_finite=False)
            except LinAlgError:
                damping *= 4
                continue
            trial = x + scale*step
            if abs(trial[:2*len(x)//3]).max() <= bound:
                tr = residual(trial)
                trial_cost = tr @ tr
                if trial_cost < cost:
                    decrease = cost-trial_cost
                    x, r, cost = trial, tr, trial_cost
                    damping = max(damping*.4, 1e-15)
                    break
            damping *= 4
        else:
            break
        if decrease < 1e-13*cost and np.linalg.norm(step*scale) < 1e-10:
            break
    return x, cost


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("start", type=Path)
    parser.add_argument("seed", type=int)
    parser.add_argument("node", type=int, help="zero-based index of the deleted node")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rule = json.loads(args.start.read_text())
    seed, node = args.seed, args.node
    if rule["domain"] != "square" or not 0 <= node < len(rule["nodes"]):
        parser.error("provide a square rule and a valid node index")
    rng = np.random.default_rng(seed)
    problem = Problem(rule["degree"], False)
    source = np.c_[rule["nodes"], rule["weights"]]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    n = len(source) - 1
    keep = np.arange(n + 1) != node
    xy, w = source[keep, :2], source[keep, 2]
    dist = np.linalg.norm(xy - source[node, :2], axis=1)
    near = np.argsort(dist)[:4]
    share = np.exp(-(dist[near] / dist[near[-1]]) ** 2)
    w[near] += source[node, 2] * share / share.sum()
    initial = np.r_[xy.ravel(), w]
    base = initial.copy()
    best = np.inf
    modes = None
    for attempt in range(args.trials):
        x = base.copy()
        print(f"{seed=} {node=} {attempt=} starting", flush=True)
        if modes is not None:
            kick = modes @ rng.normal(size=modes.shape[1])
            kick *= 10 ** rng.uniform(-3.5, -1.8) / np.linalg.norm(kick)
            x += kick
        moment_scale = np.ones(problem.m)
        if attempt % 3 == 1:
            moment_scale *= np.exp(rng.normal(0, .3, problem.m))
        elif attempt % 3 == 2:
            moment_scale[problem.ij.sum(axis=1) == problem.deg] *= rng.uniform(.15, .7)

        def residual(z):
            return moment_scale * problem.residual(z, n, False)

        def jacobian(z):
            return moment_scale[:, None] * problem.jacobian(z, n, False)

        result, _ = solve(residual, jacobian, x, 400)
        if attempt % 3:
            moment_scale[:] = 1
            result, _ = solve(residual, jacobian, result, 180)
        xy, w, _ = problem.unpack(result, n, False)
        err = np.linalg.norm(problem.residual(result, n, False))
        print(f"{seed=} {node=} {attempt=} residual={err:.10g} "
              f"weight={min(w):.8g} radius={abs(xy).max():.8g}", flush=True)
        if err < best:
            best = err
            base = result
            np.save(out / f"best-{seed}.npy", np.c_[xy, w])
            jac = problem.jacobian(base, n, False)
            scale = 1 / np.linalg.norm(jac, axis=0).clip(1e-8)
            _, _, vh = svd(jac * scale, full_matrices=False, check_finite=False)
            modes = vh[-12:].T * scale[:, None]
        if err < 1e-12 and min(w) > 0 and abs(xy).max() < 1:
            result_path = out / f"square-d{problem.deg}-n{n}-s{seed}.npy"
            np.save(result_path, np.c_[xy, w])
            result_path.with_suffix(".json").write_text(json.dumps({
                "seed": seed,
                "start": str(args.start),
                "deleted_node": node,
                "command": shlex.join(["python", *sys.argv]),
                "residual_l2": float(err),
            }, indent=2) + "\n")
            print(f"solution found: {result_path}", flush=True)
            return
    raise SystemExit("no solution found")


if __name__ == "__main__":
    main()

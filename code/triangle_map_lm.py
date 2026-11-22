#!/usr/bin/env python3
"""Search for positive interior triangle rules using mapped LM."""
import argparse
import json
from pathlib import Path
import shlex
import sys
from time import perf_counter

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit, logit

from tri_search import basis, lm_bounded, residual_and_jac, to_xy, unpack


def evaluate(v, moments, n, node_map, weight_map):
    a, b, c = unpack(v, n)
    if node_map == "duffy":
        # Two logits give independent directions and keep both coordinates interior.
        x = expit(a)
        t = expit(b)
        y = (1 - x) * t
        dx = x * (1 - x)
        xa = dx
        xb = 0
        ya, yb = -dx * t, (1 - x) * t * (1 - t)
    else:
        # Fix the third square to one to remove the redundant radial direction.
        den = 1 + a * a + b * b
        x = a * a / den
        y = b * b / den
        xa, xb = 2 * a * (1 - x) / den, -2 * b * x / den
        ya, yb = -2 * a * y / den, 2 * b * (1 - y) / den
    if weight_map == "softplus":
        w = np.logaddexp(0, c)
        dw = expit(c)
    elif weight_map == "exp":
        w = np.exp(c)
        dw = w
    else:
        w = c * c
        dw = 2 * c
    values, dx, dy = moments.eval(x, y)
    jx = (dx * xa + dy * ya) * w
    jy = (dx * xb + dy * yb) * w
    return values @ w - moments.rhs(), np.hstack([jx, jy, values * dw]), (x, y, w)


def solve(v, evaluate_at, iterations, damping):
    r, jac, points = evaluate_at(v)
    cost = r @ r
    history = []
    for _ in range(iterations):
        if np.linalg.norm(r, np.inf) < 5e-14:
            break
        # Scale columns to accommodate different coordinate and weight derivatives.
        scale = np.maximum(np.linalg.norm(jac, axis=0), 1e-8)
        j = jac / scale
        dual = j.shape[0] < j.shape[1]
        if dual:
            normal = j @ j.T
            rhs = -r
        else:
            normal = j.T @ j
            rhs = -(j.T @ r)
        for _ in range(12):
            matrix = normal + damping * np.eye(normal.shape[0])
            try:
                factor = cho_factor(matrix, check_finite=False)
            except np.linalg.LinAlgError:
                damping *= 4
                continue
            step = cho_solve(factor, rhs, check_finite=False)
            if dual:
                step = j.T @ step
            # Cap the largest move to keep the maps away from floating-point overflow.
            step = step / scale
            step *= min(1, 4 / max(np.linalg.norm(step, np.inf), 4))
            candidate = v + step
            with np.errstate(over='ignore', invalid='ignore'):
                rn, jn, pn = evaluate_at(candidate)
                cn = rn @ rn
            if cn < cost and np.isfinite(jn).all():
                history.append((cost, np.linalg.norm(step, np.inf)))
                v, r, jac, points, cost = candidate, rn, jn, pn, cn
                damping = max(1e-12, damping / 3)
                break
            damping *= 4
        else:
            break
        # Require both a flat objective and negligible motion before giving up.
        if len(history) >= 15:
            old_cost = history[-15][0]
            largest_move = max(move for _, move in history[-15:])
            if cost > 0.999 * old_cost and largest_move < 1e-5:
                break
    return points, float(np.linalg.norm(r, np.inf))


def start_points(rng, n, start, noise):
    if start is None:
        nodes = rng.dirichlet(np.ones(3), n)[:, :2]
        return nodes, np.full(n, 0.5 / n)
    nodes = np.array(start["nodes"])
    w = np.array(start["weights"])
    if len(w) == n + 1:
        # Redistribute the removed weight locally before solving the moments.
        drop = rng.integers(len(w))
        near = np.argsort(np.sum((nodes - nodes[drop]) ** 2, axis=1))[1:4]
        w[near] += w[drop] / len(near)
        nodes = np.delete(nodes, drop, axis=0)
        w = np.delete(w, drop)
    bary = np.column_stack([nodes, 1 - nodes.sum(axis=1)])
    # A PO warm start must first move into the open triangle.
    bary = np.maximum(bary + rng.normal(0, noise, bary.shape), 1e-5)
    bary /= bary.sum(axis=1, keepdims=True)
    return bary[:, :2], w


def search(args, nodes, w):
    x, y = nodes.T
    if args.node_map == "projected":
        v = np.r_[x, y / (1 - x), w]
        n = args.nodes
        lo = np.zeros(3 * n)
        hi = np.r_[np.ones(2 * n), np.full(n, np.inf)]
        fun = lambda q: residual_and_jac(q, args.degree, n)
        v, residual = lm_bounded(fun, v, lo, hi, args.iterations)
        s, t, w = unpack(v, n)
        return (*to_xy(s, t), w), residual
    if args.node_map == "duffy":
        a = logit(x)
        b = logit(y / (1 - x))
    else:
        remainder = 1 - x - y
        a = np.sqrt(x / remainder)
        b = np.sqrt(y / remainder)
    if args.weight_map == "softplus":
        c = np.log(np.expm1(w))
    elif args.weight_map == "exp":
        c = np.log(w)
    else:
        c = np.sqrt(w)
    moments = basis(args.degree)

    def at(v):
        return evaluate(v, moments, args.nodes, args.node_map, args.weight_map)

    return solve(np.r_[a, b, c], at, args.iterations, args.damping)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tries", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--node-map", choices=["duffy", "square", "projected"], default="duffy")
    parser.add_argument("--weight-map", choices=["softplus", "exp", "square"], default="softplus")
    parser.add_argument("--damping", type=float, default=0.1)
    parser.add_argument("--start", type=Path)
    parser.add_argument("--noise", type=float, default=0.001)
    args = parser.parse_args()
    if min(args.degree, args.nodes, args.tries, args.iterations) < 1 or args.nodes < 2:
        parser.error("degree, tries and iterations must be positive, with at least two nodes")
    if not np.isfinite([args.damping, args.noise]).all() or args.damping <= 0 or args.noise < 0:
        parser.error("damping must be positive and noise must be nonnegative")
    if args.node_map == "projected" and args.weight_map != "softplus":
        parser.error("projected baseline uses direct weights")
    start = json.loads(args.start.read_text()) if args.start else None
    if start and (len(start["weights"]) not in {args.nodes, args.nodes + 1}
                  or min(start["weights"]) <= 0):
        parser.error("start rule needs positive weights and N or N+1 nodes")
    args.out.mkdir(parents=True, exist_ok=True)
    command = shlex.join(["python", *sys.argv])
    best = np.inf
    successes = 0
    began = perf_counter()
    for i in range(args.tries):
        seed = args.seed + i
        rng = np.random.default_rng(seed)
        nodes, w = start_points(rng, args.nodes, start, args.noise)
        (x, y, w), residual = search(args, nodes, w)
        if residual < best:
            best = residual
            print(f"best so far: seed={seed} residual={best:.6g}", flush=True)
        if residual < 1e-12 and np.min(w) > 1e-8 and np.min(1 - (x + y)) >= 0:
            path = args.out / f"seed-{seed}"
            with path.with_suffix(".npy").open("xb") as data:
                np.save(data, np.column_stack([x, y, w]))
            replay = vars(args).copy()
            replay["seed"] = seed
            replay["tries"] = 1
            replay["out"] = args.out.parent / f"{args.out.name}-replay-{seed}"
            argv = ["python", sys.argv[0]]
            for key, value in replay.items():
                if value is not None:
                    argv.extend(["--" + key.replace("_", "-"), str(value)])
            metadata = dict(seed=seed, start=str(args.start) if start else None,
                            command=command, reproduce=shlex.join(argv), residual=residual)
            path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
            successes += 1
            print(f"solution found: seed={seed} residual={residual:.6g} "
                  f"elapsed={perf_counter() - began:.3f}", flush=True)
    print(f"summary: solutions={successes}/{args.tries} best={best:.6g} "
          f"seconds={perf_counter() - began:.3f}", flush=True)


if __name__ == "__main__":
    main()

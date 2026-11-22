#!/usr/bin/env python3
"""Remove tetrahedral orbits using invariant Dubiner bases by total degree."""
import argparse
import json
from pathlib import Path
import shlex
import sys
from time import perf_counter

import numpy as np
from scipy.linalg import qr

import sym_elim as se
import po_elim


class TetraBasis(se.BlockDom):
    def _calibrate(self):
        # Jacobi norms and the Duffy Jacobian give the exact diagonal scaling.
        p, q, r = np.asarray(self.idx).T
        self.norm = np.sqrt((2 * p + 1) * (2 * p + 2 * q + 2) * (2 * (p + q + r) + 3))
        self.T = np.zeros(len(p))
        self.T[0] = self.norm[0] / 6


def invariant_dimension(group, degree):
    # On homogeneous barycentric monomials, Burnside counts permutation orbits.
    count = 0
    for perm in group:
        unused = set(range(4))
        ways = np.zeros(degree + 1, dtype=int)
        ways[0] = 1
        while unused:
            first = min(unused)
            i = first
            length = 0
            while i in unused:
                unused.remove(i)
                i = perm[i]
                length += 1
            for j in range(length, degree + 1):
                ways[j] += ways[j - length]
        count += ways[-1]
    return count // len(group)


def invariant_basis(group, basis, seed):
    # Group actions preserve each orthogonal total-degree layer. A QR on each
    # small layer extracts its invariant coefficient space from orbit averages.
    deg = basis.deg
    rng = np.random.default_rng(seed)
    cumulative = [invariant_dimension(group, n) for n in range(deg + 1)]
    layer_sizes = np.diff(cumulative, prepend=0)
    bary = rng.dirichlet(np.full(4, 0.65), int(3 * max(layer_sizes) + 20))
    samples = np.zeros((len(bary), basis.M))
    for perm in group:
        samples += basis.values(bary[:, perm[1:]]) / len(group)
    degrees = np.sum(basis.idx, axis=1)
    U = np.zeros((basis.M, cumulative[-1]))
    offset = 0
    for n, size in enumerate(layer_sizes):
        if size == 0:
            continue
        mask = degrees == n
        q, r, _ = qr(samples[:, mask].T, mode="economic", pivoting=True)
        if abs(r[size - 1, size - 1]) < 1e-10 * abs(r[0, 0]):
            raise RuntimeError(f"rank deficiency in degree {n}")
        block = q[:, :size]
        error = np.linalg.norm(samples[:, mask] - (samples[:, mask] @ block) @ block.T)
        if error > 1e-9 * np.linalg.norm(samples[:, mask]):
            raise RuntimeError(f"incomplete invariant basis in degree {n}")
        U[mask, offset:offset + size] = block
        offset += size
    return U


class OrbitProblem(se.SymProb):
    def __init__(self, basis, invariant):
        self.D = basis
        self.U = invariant
        self.M = invariant.shape[1]
        self.T = invariant.T @ basis.T


def linear_order(problem, x):
    # In scaled coordinates, projecting a weight axis onto the Jacobian
    # nullspace predicts how much the other unknowns must move to zero it.
    q = problem.off[-1]
    scale = np.r_[np.maximum(np.minimum(x[:q], 1 - x[:q]), 0.02), x[q:]]
    jacobian = problem.jac(x) * scale
    vectors, triangular, _ = qr(jacobian.T, mode="full", pivoting=True)
    diagonal = np.abs(np.diag(triangular))
    rank = np.count_nonzero(diagonal > diagonal[0] * 1e-10)
    null = vectors[:, rank:]
    scores = np.full(problem.n, np.inf)
    for orbit, axis in enumerate(null[q:]):
        freedom = axis @ axis
        if freedom <= 1e-14:
            continue
        move = -(null @ axis) / freedom
        trial = x + move * scale
        violation = np.maximum(-trial, 0) / scale
        violation[:q] += np.maximum(trial[:q] - 1, 0) / scale[:q]
        # The removed weight is fixed to zero. Penalize other coordinates
        # predicted to cross an active constraint, then prefer smaller steps.
        scores[orbit] = np.linalg.norm(move) + 10 * np.linalg.norm(violation)
    return np.argsort(scores, kind="stable"), scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("degree", type=int)
    parser.add_argument("start", type=Path)
    parser.add_argument("--group", choices=["s4", "a4", "s3", "c3"], default="s4")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tries", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--noise", type=float, default=0)
    parser.add_argument("--order", choices=["light", "heavy", "random", "linear"], default="light")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    then = perf_counter()
    rule = json.loads(args.start.read_text())
    group = se.make_group(se.GENS[args.group])
    basis = TetraBasis("tetra", args.degree)
    prob = OrbitProblem(basis, invariant_basis(group, basis, args.seed))
    orbits = se.decompose(group, np.asarray(rule["nodes"]), np.asarray(rule["weights"]))
    x = prob.setup(orbits)
    start_residual = np.linalg.norm(prob.res(x), np.inf)
    print(f"start: {len(rule['nodes'])} points, {prob.M} equations, {len(x)} unknowns, "
          f"residual {start_residual:.3g}, setup {perf_counter() - then:.3f}s", flush=True)
    rng = np.random.default_rng(args.seed)
    order = np.argsort([orbit.size * w for orbit, _, w in orbits])
    if args.order == "heavy":
        order = order[::-1]
    if args.order == "random":
        rng.shuffle(order)
    if args.order == "linear":
        order, scores = linear_order(prob, x)
        print("linear ordering:", [(int(i), int(orbits[i][0].size), float(scores[i]))
                                  for i in order[:max(5, args.tries)]], flush=True)
    for attempt, drop in enumerate(order[:args.tries], 1):
        orbit, q, w = orbits[drop]
        # Preserve total mass and move the dropped mass into its closest orbit.
        points, _ = prob.points(prob.setup(orbits))
        image = np.array([se.apply(g, points[drop:drop + 1])[0] for g in group])
        distances = np.min(np.linalg.norm(points[:, None] - image, axis=2), axis=1)
        distances[drop] = np.inf
        near = np.argmin(distances)
        trial = []
        for i, (ob, p, wt) in enumerate(orbits):
            if i == drop:
                continue
            if i == near:
                wt += orbit.size * w / ob.size
            p = np.clip(p + args.noise * rng.normal(size=p.shape), 0, 1)
            trial.append((ob, p, wt))
        x = prob.setup(trial)
        x, cost = prob.solve_lm(x, args.iterations)
        print(f"trial {attempt} orbit {drop} size {orbit.size} cost {cost:.7g} "
              f"elapsed {perf_counter() - then:.3f}s", flush=True)
        if cost >= 1e-24:
            continue
        points, weights = se.expand(group, prob, x)
        if (np.min(se.bary(points)) < -1e-15 or np.min(weights) <= 0
                or po_elim.monomial_error("tetra", args.degree, points, weights) >= 1e-11):
            continue
        stem = args.out / f"tetra-d{args.degree}-n{len(weights)}-s{args.seed}-t{attempt}"
        np.save(stem.with_suffix(".npy"), np.column_stack((points, weights)))
        info = dict(seed=args.seed, start=str(args.start), command=shlex.join(["python", *sys.argv]),
                    attempt=attempt, orbit=int(drop), cost=float(cost))
        stem.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
        print(f"solution found: {len(weights)} points at trial {attempt}", flush=True)
        break


if __name__ == "__main__":
    main()

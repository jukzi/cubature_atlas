"""Search odd-degree square cubature rules with twofold or fourfold rotations.

Start from seeded random nodes or from a JSON rule with the requested symmetry.
With a warm start, remove one orbit at a time and solve the remaining moment
equations. With multiple continuation steps, prescribe successively smaller
weights for the orbit before removing it. Save every accepted intermediate rule.
"""
import argparse
import json
import shlex
import sys
import time
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import legvander
from scipy.linalg import qr

from square_elim import Problem, monomial_error


class RotationProblem(Problem):
    def __init__(self, degree, rotation, bound):
        super().__init__(degree, True)
        self.rotation = rotation
        self.bound = bound
        self.fixed = None
        if rotation == 4:
            i, j = self.ij.T
            keep = (i > j) | ((i == j) & (j % 2 == 0))
            self.ij = self.ij[keep]
            i, j = self.ij.T
            self.sign = (-1.0) ** j
            self.scale = self.scale[keep] / np.sqrt(2 - (i == j))
            self.m = len(i)
            self.target = self.target[keep]
            p0 = legvander([0.0], degree)[0]
            self.v0 = (1 + self.sign) / 2 * p0[i] * p0[j] * self.scale

    def bounds(self, n, centre):
        lo, hi = super().bounds(n, centre)
        if self.fixed is not None:
            index, weight = self.fixed
            lo[index] = hi[index] = weight
        return lo, hi

    def evaluate(self, x, n, centre, jacobian):
        p, w, wc = self.unpack(x, n, centre)
        i, j = self.ij.T
        ax = legvander(p[:, 0], self.deg)
        ay = legvander(p[:, 1], self.deg)
        v = ax[:, i] * ay[:, j]
        if jacobian:
            dx = ax @ self.dmat.T
            dy = ay @ self.dmat.T
            vx = dx[:, i] * ay[:, j]
            vy = ax[:, i] * dy[:, j]
        if self.rotation == 4:
            # Average the two axis orders. The antipodal image gives the same value.
            v += self.sign * ax[:, j] * ay[:, i]
            if jacobian:
                vx += self.sign * dx[:, j] * ay[:, i]
                vy += self.sign * ax[:, j] * dy[:, i]
        scale = 2 * self.scale
        if not jacobian:
            return (w @ v) * scale + wc * self.v0 - self.target
        jac = np.empty((self.m, len(x)))
        jac[:, :2 * n:2] = (w[:, None] * vx * scale).T
        jac[:, 1:2 * n:2] = (w[:, None] * vy * scale).T
        jac[:, 2 * n:3 * n] = (v * scale).T
        if centre:
            jac[:, -1] = self.v0
        return jac

    def residual(self, x, n, centre):
        return self.evaluate(x, n, centre, False)

    def jacobian(self, x, n, centre):
        return self.evaluate(x, n, centre, True)

    def full_rule(self, x, n, centre):
        p, w, wc = self.unpack(x, n, centre)
        nodes = [p, -p]
        if self.rotation == 4:
            q = p[:, ::-1] * [-1, 1]
            nodes.extend([q, -q])
        weights = np.tile(w, self.rotation)
        if centre:
            nodes.append(np.zeros((1, 2)))
            weights = np.r_[weights, wc]
        return np.vstack(nodes), weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('degree', type=int)
    parser.add_argument('rotation', type=int, choices=[2, 4])
    parser.add_argument('target', type=int)
    parser.add_argument('seed', type=int)
    parser.add_argument('--start', type=Path)
    parser.add_argument('--bound', type=float, default=1)
    parser.add_argument('--seconds', type=float, default=120)
    parser.add_argument('--iters', type=int, default=160)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--add-centre', action='store_true')
    parser.add_argument('--remove-centre', action='store_true')
    parser.add_argument('--direct', action='store_true')
    parser.add_argument('--rank', action='store_true')
    parser.add_argument('--noise', type=float, default=0)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.degree < 1 or args.degree % 2 == 0:
        parser.error('the degree must be positive and odd')
    if args.target < 1 or args.target % args.rotation not in (0, 1):
        parser.error('target must consist of full orbits and at most one centre node')
    if args.steps < 1:
        parser.error('steps must be positive')
    if args.direct and args.steps != 1:
        parser.error('direct elimination requires one step')
    rng = np.random.default_rng(args.seed)
    prob = RotationProblem(args.degree, args.rotation, args.bound)
    stop = time.monotonic() + args.seconds
    args.out.mkdir(parents=True, exist_ok=True)
    if args.start:
        data = json.loads(args.start.read_text())
        p = np.array(data['nodes'])
        w = np.array(data['weights'], dtype=float)
        half = (p[:, 0] > 1e-12) | ((abs(p[:, 0]) <= 1e-12) & (p[:, 1] > 1e-12))
        if args.rotation == 4:
            half = (p[:, 0] > 1e-12) & (p[:, 1] >= -1e-12)
        centre = bool(np.any(np.max(abs(p), axis=1) < 1e-12))
        wc = w[np.argmin(np.max(abs(p), axis=1))] if centre else 0
        p, w = p[half], w[half]
        print('start', len(w) * args.rotation + centre, 'moments', prob.m, flush=True)
    else:
        centre = bool(args.target % args.rotation)
    if args.add_centre:
        centre = True
    if args.remove_centre:
        centre = False
    count = 0
    best = np.inf
    while time.monotonic() < stop:
        if not args.start:
            n = args.target // args.rotation
            q = rng.uniform(-0.85, 0.85, (n, 2))
            v = np.full(n, 4 / args.target)
            c = 4 / args.target
            fractions = [0]
        else:
            k = len(w)
            if count == 0:
                score = w
                if args.rank:
                    initial = np.r_[p.ravel(), w, wc] if centre else np.r_[p.ravel(), w]
                    jac = prob.jacobian(initial, k, centre)
                    if jac.shape[1] > jac.shape[0]:
                        basis, _ = qr(jac.T)
                        freedom = np.linalg.norm(basis[2 * k:3 * k, prob.m:], axis=1)
                        score = w / np.maximum(freedom, 1e-15)
                order = np.argsort(score)
            if count and count % k == 0:
                order = rng.permutation(k)
            index = order[count % k]
            keep = (np.arange(k) != index) | args.remove_centre
            if args.direct:
                drops = k - args.target // args.rotation
                choices = np.r_[index, order[order != index]]
                if count:
                    choices = np.r_[index, rng.permutation(choices[1:])]
                keep[choices[:drops]] = False
            n = int(keep.sum())
            q = p[keep] + args.noise * rng.normal(size=(n, 2))
            v = w[keep].copy()
            c = wc
            fractions = np.linspace(1, 0, args.steps + 1)[1:] if not args.remove_centre else [0]
        count += 1
        x = np.r_[q.ravel(), v, c] if centre else np.r_[q.ravel(), v]
        for fraction in fractions:
            prob.fixed = None
            if args.start and fraction > 0:
                # Keep the disappearing orbit free in position but fix its weight.
                if fraction == fractions[0]:
                    x = np.r_[q.ravel(), p[index], v, w[index], *([c] if centre else [])]
                prob.fixed = (3 * n + 2, w[index] * fraction)
                x, cost = prob.solve(x, n + 1, centre, args.iters)
            else:
                if args.start and args.steps > 1 and not args.remove_centre:
                    x = np.r_[x[:2 * n], x[2 * n + 2:3 * n + 2], *([x[-1]] if centre else [])]
                x, cost = prob.solve(x, n, centre, args.iters)
        residual = np.sqrt(2 * cost)
        if residual < best:
            best = residual
            print('trial', count, 'best', f'{best:.8g}', flush=True)
        if residual < 1e-13:
            nodes, weights = prob.full_rule(x, n, centre)
            error = monomial_error(args.degree, nodes, weights)
            print("numerical candidate", len(weights), error, weights.min(), abs(nodes).max(), flush=True)
            if error < 1e-13 and weights.min() > 1e-12 and abs(nodes).max() <= args.bound:
                name = args.out / f'd{args.degree}_n{len(weights)}_s{args.seed}_t{count}.json'
                name.write_text(json.dumps(dict(nodes=nodes.tolist(), weights=weights.tolist(),
                                                seed=args.seed, command=shlex.join(sys.argv))))
                print('FOUND', name, 'monomial error', error, 'min weight', weights.min(), flush=True)
                if len(weights) <= args.target or not args.start:
                    break
                p, v, wc = prob.unpack(x, n, centre)
                p, w = p.copy(), v.copy()
                count = 0
                best = np.inf
                args.remove_centre = False
    print('RESULT trials', count, 'best', best, flush=True)


if __name__ == '__main__':
    main()

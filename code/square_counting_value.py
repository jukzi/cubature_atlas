"""Compute positive interior square rules at the counting value, Z = ceil(dim(P_d)/3).

Examples from the project root (NumPy, SciPy; use one BLAS thread for replay):
    python code/square_counting_value.py 35 --out rule35.json
    python code/square_counting_value.py 38 --out rule38.json
    python code/square_counting_value.py 42 --out rule42.json

Even degrees: continue one weight down, merge it with a neighbour, and solve
the smaller system using scaled LM with a directional curvature correction.
For a square system stalled away from zero, continue the residual along a
one-dimensional ray by pseudo-arclength. Odd degree 35: split an antipodal
pair and refit in the centrally symmetric subspace.
"""
import argparse
import json
import shlex
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve, svd
from scipy.spatial import cKDTree

from square_elim import Problem


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    35: ('festa_sommariva_2012', 'CA-C2-35-220-PI-FS', None),
    38: ('minimal_pointcount_for_degree_n_PI', 'CA-C2-38-261-PI-MPC', 166),
    42: ('minimal_pointcount_for_degree_n_PI', 'CA-C2-42-317-PI-MPC', 111),
}


def fit(problem, q, count, steps=400):
    """Scaled LM; accept a curvature correction only when it improves the trial."""
    q = q.copy()
    residual = problem.residual(q, count, False)
    damping = 1e-4
    for iteration in range(steps):
        if np.linalg.norm(residual) < 3e-14:
            break
        jac = problem.jacobian(q, count, False)
        scale = 1 / np.maximum(np.linalg.norm(jac, axis=0), 1e-9)
        jac *= scale
        gram = jac.T @ jac
        gradient = jac.T @ residual
        diagonal = np.diag_indices_from(gram)
        for _ in range(15):
            system = gram.copy()
            system[diagonal] += damping
            try:
                factor = cho_factor(system, lower=True, check_finite=False)
            except np.linalg.LinAlgError:
                damping *= 4
                continue
            step = cho_solve(factor, -gradient, check_finite=False)
            if np.max(np.abs((step * scale)[:2 * count])) > .08:
                damping *= 4
                continue
            trial = q + step * scale
            if not admissible(trial, count, 1.00001, 0.):
                damping *= 4
                continue
            trial_residual = problem.residual(trial, count, False)
            curvature = trial_residual - residual - jac @ step
            correction = cho_solve(factor, -jac.T @ curvature, check_finite=False)
            if np.linalg.norm(correction) < .8 * np.linalg.norm(step):
                curved = q + (step + correction) * scale
                if admissible(curved, count, 1.00001, 0.):
                    curved_residual = problem.residual(curved, count, False)
                    if curved_residual @ curved_residual < trial_residual @ trial_residual:
                        trial, trial_residual = curved, curved_residual
            if trial_residual @ trial_residual < residual @ residual:
                q, residual = trial, trial_residual
                damping = max(1e-15, damping * .42)
                break
            damping *= 4
        else:
            break
    return q, np.linalg.norm(residual)


def admissible(q, count, bound, min_weight):
    return np.max(np.abs(q[:2 * count])) < bound and np.min(q[2 * count:]) >= min_weight


def prescribed_weight(problem, q, count, constraint, goal):
    residual = np.r_[problem.residual(q, count, False), constraint @ q - goal]
    for _ in range(9):
        if np.linalg.norm(residual) < 1e-11:
            return q
        jac = np.vstack((problem.jacobian(q, count, False), constraint))
        scale = 1 / np.maximum(np.linalg.norm(jac, axis=0), 1e-8)
        jac *= scale
        gram = jac @ jac.T
        for damping in (1e-13, 1e-10, 1e-7):
            system = gram.copy()
            system.flat[::len(system) + 1] += damping
            try:
                factor = cho_factor(system, lower=True, check_finite=False)
                step = scale * (jac.T @ cho_solve(factor, -residual, check_finite=False))
            except np.linalg.LinAlgError:
                continue
            for alpha in (1., .5, .25):
                trial = q + step * alpha
                if not admissible(trial, count, 1.001, -1e-7):
                    continue
                new = np.r_[problem.residual(trial, count, False), constraint @ trial - goal]
                if new @ new < residual @ residual:
                    break
            else:
                continue
            q, residual = trial, new
            break
        else:
            return None
    return q if np.linalg.norm(residual) < 2e-10 else None


def remove_node(problem, points, weights, node):
    count = len(weights)
    q = np.r_[points.ravel(), weights]
    constraint = np.zeros((1, len(q)))
    constraint[0, 2 * count + node] = 1 / weights[node]
    start = constraint @ q
    fraction, stride = 1., .12
    while fraction > .008 and stride > .0015:
        next_fraction = max(.005, fraction - stride)
        trial = prescribed_weight(problem, q, count, constraint, start * next_fraction)
        if trial is None:
            stride *= .45
        else:
            q, fraction = trial, next_fraction
            stride = min(.22, stride * 1.4)
    points, weights, _ = problem.unpack(q.copy(), count, False)
    distances = np.linalg.norm(points - points[node], axis=1)
    distances[node] = np.inf
    neighbour = np.argmin(distances)
    total = weights[node] + weights[neighbour]
    points[neighbour] = (points[node] * weights[node] + points[neighbour] * weights[neighbour]) / total
    weights[neighbour] = total
    return np.r_[np.delete(points, node, axis=0).ravel(), np.delete(weights, node)]


def trace_fold(problem, q0, count, sign):
    """Trace F(q) = t*u and correct the rule when the curve crosses t = 0."""
    r = problem.residual(q0, count, False)
    t = np.linalg.norm(r)
    direction = r / t
    jac = problem.jacobian(q0, count, False)
    scale = 1 / np.linalg.norm(jac, axis=0)
    _, _, right = svd(jac * scale, full_matrices=False, check_finite=False)
    tangent = np.r_[sign * right[-1], 0.]
    q = q0.copy()
    stride = .02
    for _ in range(1200):
        anchor = np.r_[q / scale, t] + stride * tangent
        trial = anchor.copy()
        for correction in range(7):
            points = trial[:-1] * scale
            if not admissible(points, count, 1.015, -.0004):
                break
            rhs = np.r_[problem.residual(points, count, False) - direction * trial[-1],
                        (trial - anchor) @ tangent]
            augmented = np.empty((len(trial), len(trial)))
            augmented[:-1, :-1] = problem.jacobian(points, count, False) * scale
            augmented[:-1, -1] = -direction
            augmented[-1] = tangent
            if np.linalg.norm(rhs) < 1e-12:
                break
            delta = solve(augmented, -rhs, assume_a='gen', check_finite=False)
            if np.linalg.norm(delta) > stride:
                break
            trial += delta
        else:
            correction = 7
        if (correction >= 7 or not admissible(points, count, 1.015, -.0004)
                or np.linalg.norm(problem.residual(points, count, False) - direction * trial[-1]) > 2e-11):
            stride *= .5
            if stride < 2e-4:
                break
            continue
        new_t = trial[-1]
        if t * new_t <= 0 or abs(new_t) < 1e-10:
            guess = (abs(new_t) * q + abs(t) * points) / (abs(new_t) + abs(t))
            result, error = fit(problem, guess, count, 120)
            if error < 2e-13 and admissible(result, count, 1., 0.):
                return result
            break
        q, t = points, new_t
        rhs = np.zeros(len(trial))
        rhs[-1] = 1.
        tangent = solve(augmented, rhs, assume_a='gen', check_finite=False)
        tangent /= np.linalg.norm(tangent)
        stride = min(.22, stride * (1.3 if correction < 4 else .8))
    return None


def split_pair(points, weights, node):
    mask = (points[:, 0] > 1e-12) | ((abs(points[:, 0]) <= 1e-12) & (points[:, 1] > 0))
    points, weights = points[mask].copy(), weights[mask].copy()
    index = int(np.argmax(weights)) if node is None else node
    distances = np.linalg.norm(points - points[index], axis=1)
    distances[index] = np.inf
    offset = .24 * (points[np.argmin(distances)] - points[index])
    points = np.vstack((points, points[index] + offset))
    points[index] -= offset
    weights[index] *= .5
    return np.r_[points.ravel(), weights, weights[index]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('degree', type=int, choices=sorted(SOURCES))
    parser.add_argument('--node', type=int, help='source node index (antipodal representative for degree 35)')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    degree = args.degree
    family, source_id, default_node = SOURCES[degree]
    source = json.loads((ROOT / 'rules' / family / 'square' / (source_id + '.json')).read_text())
    points, weights = np.array(source['nodes']), np.array(source['weights'])
    node = args.node if args.node is not None else default_node
    target = ((degree + 1) * (degree + 2) // 2 + 2) // 3
    symmetric = degree == 35
    problem = Problem(degree, symmetric)
    count = target // 2 if symmetric else target
    if node is not None and not 0 <= node < (len(weights)//2 if symmetric else len(weights)):
        parser.error('source node index outside the rule')
    q = split_pair(points, weights, node) if symmetric else remove_node(problem, points, weights, node)
    q, error = fit(problem, q, count)
    method = ('Split an antipodal pair' if symmetric else f'Continue the weight at source index {node}, then merge with its nearest neighbour')
    method += f' starting from {source_id}; column-scaled Levenberg-Marquardt with a curvature correction'
    print(f'After refit: residual {error:.6g}', flush=True)
    if error >= 2e-13 and len(q) == problem.m:
        for sign in (1, -1):
            result = trace_fold(problem, q, count, sign)
            if result is not None:
                q = result
                method += '; pseudo-arclength continuation through a fold and Newton polishing'
                break
    error = float(np.linalg.norm(problem.residual(q, count, False)))
    points, weights = problem.full_rule(q, count, False)
    separation = cKDTree(points).query(points, k=2)[0][:, 1].min()
    if not (len(points) == target and error < 2e-13 and min(weights) > 0
            and abs(points).max() < 1 and separation > 1e-7):
        raise SystemExit(f'No positive interior rule: residual {error:.6g}')
    script = 'code/square_counting_value.py'
    record = dict(id=f'CA-C2-{degree}-{target}-PI-MPC', domain='square', dimension=2,
                  family='minimal_pointcount_for_degree_n_PI', degree=degree, points=target,
                  quality='PI', origin='found', status='candidate', author='OpenAI Codex',
                  method=method, residual_reported=error,
                  reproduce=dict(script=script, seed=0, command=shlex.join(['python', script, *sys.argv[1:]])),
                  nodes=points.tolist(), weights=weights.tolist())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + '\n')
    print(f'Wrote {target} nodes, residual {error:.3g}, separation {separation:.4g}', flush=True)


if __name__ == '__main__':
    main()

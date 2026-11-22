import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import null_space

from square_rotation_search import RotationProblem

parser = argparse.ArgumentParser()
parser.add_argument("start", type=Path)
parser.add_argument("seed", type=int)
parser.add_argument("mode", choices=["shell", "walk", "merge"])
parser.add_argument("rotation", type=int, choices=[2, 4])
parser.add_argument("out", type=Path)
args = parser.parse_args()
rng = np.random.default_rng(args.seed)
data = json.loads(args.start.read_text())
p = np.array(data["nodes"])
w = np.array(data["weights"], dtype=float)
centre = np.max(abs(p), axis=1) < 1e-12
wc = w[centre][0]
keep = (p[:, 0] > 1e-12) | ((abs(p[:, 0]) <= 1e-12) & (p[:, 1] > 1e-12))
if args.rotation == 4:
    keep = (p[:, 0] > 1e-12) & (p[:, 1] >= -1e-12)
p, w = p[keep], w[keep]
n = len(w)
prob = RotationProblem(49, args.rotation, 1 - 1e-9)
print("start", n * args.rotation + 1, "seed", args.seed, "mode", args.mode, flush=True)
best = np.inf
for trial in range(5):
    x = np.r_[p.ravel(), w, wc]
    if args.mode == "walk":
        for _ in range(8 + 4 * trial):
            j = prob.jacobian(x, n, True)
            scale = np.r_[np.ones(2*n), np.full(n+1, .025)]
            basis = null_space(j * scale)
            direction = scale * (basis @ rng.normal(size=basis.shape[1]))
            if direction[-1] > 0:
                direction = -direction
            direction *= .002 / np.linalg.norm(direction, np.inf)
            guess = x + direction
            fitted, cost = prob.solve(guess, n, True, 25)
            if cost < 5e-25:
                x = fitted
        print("walk centre", x[-1], flush=True)
        x = x[:-1]
    elif args.mode == "merge":
        q = p * np.sign(p @ rng.normal(size=2))[:, None]
        idx = np.argsort(np.linalg.norm(q, axis=1))[:4]
        i = rng.choice(idx)
        q[i] *= (w[i] / (w[i] + wc)) ** .5
        v = w.copy()
        v[i] += wc
        x = np.r_[q.ravel(), v]
    else:
        x = x[:-1] + rng.normal(size=3*n) * [0, .0001, .0003, .0008, .0015][trial]
    if args.mode == "shell":
        scale, v0 = prob.scale.copy(), prob.v0.copy()
        shell = prob.ij.sum(axis=1) >= 46
        for factor in (.025, .08, .25, .55, 1):
            multiplier = np.where(shell, factor, 1.)
            prob.scale = scale * multiplier
            prob.v0 = v0 * multiplier
            x, cost = prob.solve(x, n, False, 60)
    else:
        x, cost = prob.solve(x, n, False, 180)
    residual = float(np.linalg.norm(prob.residual(x, n, False)))
    nodes, weights = prob.full_rule(x, n, False)
    print(trial, residual, weights.min(), abs(nodes).max(), flush=True)
    if residual < best:
        best = residual
        args.out.write_text(json.dumps(dict(seed=args.seed, trial=trial, mode=args.mode,
            rotation=args.rotation, residual=residual, nodes=nodes.tolist(),
            weights=weights.tolist())))
    if residual < 1e-13 and weights.min() > 1e-12 and abs(nodes).max() < 1:
        break

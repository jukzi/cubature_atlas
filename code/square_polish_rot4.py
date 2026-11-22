import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import root
from square_rotation_search import RotationProblem

parser = argparse.ArgumentParser()
parser.add_argument("start", type=Path)
parser.add_argument("seed", type=int)
parser.add_argument("out", type=Path)
args = parser.parse_args()
data = json.loads(args.start.read_text())
n = (len(data["weights"]) - 1) // 4
p, w = np.array(data["nodes"]), np.array(data["weights"])
x = np.r_[p[:n].ravel(), w[:n], w[-1]]
prob = RotationProblem(49, 4, 1)
rng = np.random.default_rng(args.seed)
best = np.inf
for trial in range(18):
    s = 0 if trial == 0 else 10 ** rng.uniform(-4.5, -2.4)
    start = x + rng.normal(size=x.size) * s * np.r_[np.ones(2*n), np.full(n+1, .05)]
    result = root(prob.residual, start, jac=prob.jacobian, args=(n, True),
                  method="lm", options={"maxiter": 240, "ftol": 1e-12, "xtol": 1e-12})
    if not np.isfinite(result.x).all():
        continue
    nodes, weights = prob.full_rule(result.x, n, True)
    residual = np.linalg.norm(result.fun)
    print(trial, residual, weights.min(), abs(nodes).max(), flush=True)
    if residual < best:
        best = residual
        args.out.write_text(json.dumps(dict(nodes=nodes.tolist(), weights=weights.tolist(),
            residual=float(residual), seed=args.seed, trial=trial)))
    if residual < 1e-13 and weights.min() > 1e-10 and abs(nodes).max() < 1:
        print("CANDIDATE", len(weights), flush=True)
        break

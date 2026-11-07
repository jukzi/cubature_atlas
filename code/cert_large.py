"""Usage: python code/cert_large.py rules/.../<id>.json [...]  (writes certificates.json)

Newton-Kantorovich certificate as in code/cubature_atlas.py certificate(), for large simplex rules.
Same quantities (sigma_min, eta, L from 40 random directions times factor, h, radius);
the spectral norms of the Jacobian differences are computed by power iteration instead of a full SVD,
and eta = |J^+ F| via the normal equations of the underdetermined system."""
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cubature_atlas import Basis, System, DIM, margin
import cubature_atlas as ca

_orig_raw = ca.raw


def fast_raw(name, n, X):
    """Same basis as cubature_atlas.raw; for the tetrahedron the Jacobi recursions are cached per i and i+j."""
    if name != "tetra":
        return _orig_raw(name, n, X)
    idx = ca.indices(name, n)
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    Q = ca.homLegendre(n, x, 1 - y - z)
    one = np.ones_like(z)
    Rc = {}
    Pc = {}
    out = []
    for i, j, k in idx:
        if i not in Rc:
            Rc[i] = ca.homJacobi(n, 2 * i + 1, y, 1 - z)
        if i + j not in Pc:
            Pc[i + j] = ca.homJacobi(n, 2 * i + 2 * j + 2, z, one)
        out.append(Q[i] * Rc[i][j] * Pc[i + j][k])
    return np.stack(out, 1)


ca.raw = fast_raw


def norm2(A, its=60, seed=0):
    """Largest singular value by power iteration on A^T A, with a 1% safety margin."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=A.shape[1])
    v /= np.linalg.norm(v)
    s = 0.0
    for _ in range(its):
        w = A.T @ (A @ v)
        s_new = np.sqrt(np.linalg.norm(w))
        v = w / np.linalg.norm(w)
        if abs(s_new - s) < 1e-6 * s_new:
            s = s_new
            break
        s = s_new
    return 1.01 * s


def cert(path, seeds=(0, 1, 2), factor=2.0):
    r = json.load(open(path))
    dom = r["domain"]
    n = r["degree"]
    d = DIM[dom]
    X = np.array(r["nodes"], dtype=float)
    W = np.array(r["weights"], dtype=float)
    t0 = time.time()
    B = Basis(dom, n)
    S = System(B, d, False, False)
    x0 = np.r_[X.ravel(), W]
    F0 = S.F(x0)
    J0 = S.J(x0)
    print("basis+J %.0fs" % (time.time() - t0), J0.shape, flush=True)
    sv = np.linalg.svd(J0, compute_uv=False)
    smin = sv[-1]
    print("svd %.0fs sigma_min %.3e cond %.2e" % (time.time() - t0, smin, sv[0] / smin), flush=True)
    G = J0 @ J0.T
    y = np.linalg.solve(G, F0)
    eta = float(np.linalg.norm(J0.T @ y))
    out = None
    for seed in seeds:
        rng = np.random.default_rng(seed)
        rho = 1e-5
        L = 0.0
        for _ in range(40):
            u = rng.normal(size=len(x0))
            u /= np.linalg.norm(u)
            L = max(L, norm2(S.J(x0 + rho * u) - J0) / rho)
        L *= factor
        h = L * eta / smin ** 2
        print("seed %d L %.3e h %.3e  %.0fs" % (seed, L, h, time.time() - t0), flush=True)
        if out is None or h > out["h"]:
            out = {"h": h, "L": L}
    rad = 2 * eta
    wmin = float(W.min())
    mg = float(margin(dom, X))
    c = {"N": len(W), "M": J0.shape[0], "p": J0.shape[1], "res": float(np.abs(F0).max()), "sigma": float(smin),
         "h": float(out["h"]), "rad": float(rad), "cert": bool(out["h"] <= 0.5),
         "inside": bool(out["h"] <= 0.5 and wmin > rad and mg > rad)}
    return c, wmin, mg


if __name__ == "__main__":
    res = {}
    for p in sys.argv[1:]:
        c, wmin, mg = cert(p)
        print(p.split("/")[-1], json.dumps(c), "wmin %.2e margin %.2e" % (wmin, mg), flush=True)
        res[p] = c
        json.dump(res, open("certificates.json", "w"))

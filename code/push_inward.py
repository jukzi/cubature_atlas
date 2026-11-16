"""push_inward: move the nodes of a square rule away from the boundary without changing degree or node count.

The interval proof (tools/verify_rigorous.py) can fail for a rule with nodes very close to the boundary,
because the second derivatives of the Legendre polynomials are large there. A rule with more unknowns than
moment equations has a whole family of solutions. This program walks along that family: each step moves the
nodes nearest to the boundary inward (projected onto the null space of the Jacobian), then least-norm Newton
steps restore the moment equations. Weights stay positive and nodes stay inside. The run is deterministic.
If that is not enough, --cond adds steps that improve the conditioning of the Jacobian (columns scaled to
equal length): they follow the finite-difference gradient of the sum of the logarithms of its three smallest
singular values inside the null space.

Usage: python code/push_inward.py --start <rule.json> --out <rule.json> [--steps 150] [--tau 3e-4] [--step 1e-3] [--cond 0]
Needs numpy.
"""
import argparse
import json
import sys

import numpy as np
from numpy.polynomial import legendre


def system(n, X, W):
    """Residual and Jacobian of the moment equations on [-1,1]^2 in the orthonormal Legendre product basis."""
    k = np.arange(n + 1)
    s = np.sqrt((2 * k + 1) / 2.0)
    eye = np.eye(n + 1)
    Px = legendre.legvander(X[:, 0], n) * s
    Py = legendre.legvander(X[:, 1], n) * s
    Dx = np.zeros((len(X), n + 1))
    Dy = np.zeros((len(X), n + 1))
    for i in range(1, n + 1):
        der = legendre.legder(eye[i])
        Dx[:, i] = legendre.legval(X[:, 0], der) * s[i]
        Dy[:, i] = legendre.legval(X[:, 1], der) * s[i]
    I = np.array([i for i in range(n + 1) for j in range(n + 1 - i)])
    J = np.array([j for i in range(n + 1) for j in range(n + 1 - i)])
    B = Px[:, I] * Py[:, J]
    rhs = np.zeros(len(I))
    rhs[0] = 2.0
    F = B.T @ W - rhs
    Jw = B.T
    Jx = (Dx[:, I] * Py[:, J] * W[:, None]).T
    Jy = (Px[:, I] * Dy[:, J] * W[:, None]).T
    return F, np.hstack([Jw, Jx, Jy])


def newton(n, X, W, iters):
    """Least-norm Newton steps back onto the solution set."""
    N = len(W)
    for _ in range(iters):
        F, J = system(n, X, W)
        du = np.linalg.lstsq(J, -F, rcond=None)[0]
        W = W + du[:N]
        X = X + np.stack([du[N:2 * N], du[2 * N:]], axis=1)
        if np.max(np.abs(F)) < 3e-15:
            break
    return X, W


def boundary_distance(X):
    return float(np.min(1.0 - np.abs(X)))


def push(n, X, W, steps, tau, step):
    N = len(W)
    best = (boundary_distance(X), X.copy(), W.copy())
    print("start: boundary distance %.6e, smallest weight %.3e" % (best[0], W.min()))
    for it in range(steps):
        F, J = system(n, X, W)
        d = 1.0 - np.abs(X)
        # wanted move: inward, only for the coordinates nearest to the boundary
        g = np.exp(-(d - d.min()) / tau) * (-np.sign(X))
        want = np.concatenate([np.zeros(N), g[:, 0], g[:, 1]])
        t = want - np.linalg.lstsq(J, J @ want, rcond=None)[0]
        size = np.max(np.abs(t[N:]))
        if size < 1e-9:
            print("no direction left at step", it)
            break
        t = t * (step / size)
        Xn, Wn = newton(n, X + np.stack([t[N:2 * N], t[2 * N:]], axis=1), W + t[:N], 6)
        if Wn.min() <= 0 or boundary_distance(Xn) <= 0:
            step = step / 2
            continue
        X = Xn
        W = Wn
        m = boundary_distance(X)
        if m > best[0]:
            best = (m, X.copy(), W.copy())
        else:
            step = step * 0.7
        if it % 10 == 0:
            print("step %d: boundary distance %.6e, smallest weight %.3e" % (it, m, W.min()))
        if step < 1e-8:
            break
    X, W = newton(n, best[1], best[2], 10)
    F, J = system(n, X, W)
    print("final: boundary distance %.6e, smallest weight %.3e, residual %.1e" % (boundary_distance(X), W.min(), np.max(np.abs(F))))
    return X, W


def conditioning(n, X, W):
    """Sum of the logarithms of the three smallest singular values of the column-scaled Jacobian, and the Jacobian."""
    F, J = system(n, X, W)
    cn = np.linalg.norm(J, axis=0)
    sv = np.linalg.svd(J * (np.median(cn) / cn), compute_uv=False)
    return float(np.sum(np.log(sv[-3:]))), J


def split(u, N):
    return np.stack([u[N:2 * N], u[2 * N:]], axis=1), u[:N]


def improve_conditioning(n, X, W, steps, step):
    N = len(W)
    val, J = conditioning(n, X, W)
    print("conditioning start: value %.6f" % val)
    for it in range(steps):
        u = np.concatenate([W, X[:, 0], X[:, 1]])
        Z = np.linalg.svd(J, full_matrices=True)[2][J.shape[0]:].T
        h = 1e-6
        g = np.zeros(Z.shape[1])
        for k in range(Z.shape[1]):
            Xk, Wk = split(u + h * Z[:, k], N)
            g[k] = (conditioning(n, Xk, Wk)[0] - val) / h
        t = Z @ g
        t = t * (step / np.max(np.abs(t)))
        found = False
        for shrink in (1.0, 0.5, 0.25, 0.1):
            Xt, Wt = split(u + shrink * t, N)
            Xn, Wn = newton(n, Xt, Wt, 6)
            if Wn.min() <= 0 or boundary_distance(Xn) <= 0:
                continue
            vn, Jn = conditioning(n, Xn, Wn)
            if vn > val:
                found = True
                break
        if not found:
            print("conditioning: no improvement at step", it)
            break
        X = Xn
        W = Wn
        val = vn
        J = Jn
        print("conditioning step %d: value %.6f, boundary distance %.6e, smallest weight %.3e" % (it, val, boundary_distance(X), W.min()), flush=True)
    return X, W


def main():
    ap = argparse.ArgumentParser(description="Move the nodes of a square rule away from the boundary along its solution family.")
    ap.add_argument("--start", required=True, help="rule file (square, more unknowns than moment equations)")
    ap.add_argument("--out", required=True, help="file for the resulting rule (same fields, new nodes and weights)")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--tau", type=float, default=3e-4, help="width of the band near the boundary that is pushed")
    ap.add_argument("--step", type=float, default=1e-3, help="largest coordinate change of one step")
    ap.add_argument("--cond", type=int, default=0, help="number of additional steps that improve the conditioning of the Jacobian")
    ap.add_argument("--cond-step", type=float, default=5e-4, help="largest change of an unknown in one conditioning step")
    a = ap.parse_args()
    with open(a.start, encoding="utf8") as fh:
        rule = json.load(fh)
    if rule["domain"] != "square":
        sys.exit("only the square is implemented")
    X, W = push(rule["degree"], np.array(rule["nodes"], dtype=float), np.array(rule["weights"], dtype=float), a.steps, a.tau, a.step)
    if a.cond > 0:
        X, W = improve_conditioning(rule["degree"], X, W, a.cond, a.cond_step)
    rule["nodes"] = [[float(x), float(y)] for x, y in X]
    rule["weights"] = [float(w) for w in W]
    rule["push_inward"] = {"command": " ".join(["python", "code/push_inward.py"] + sys.argv[1:])}
    with open(a.out, "w", encoding="utf8", newline="\n") as fh:
        json.dump(rule, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


if __name__ == "__main__":
    main()

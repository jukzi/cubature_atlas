"""minimize_l2: minimise the L2 norm of the weights (sum of w^2) of a rule under all moment equations, keeping the symmetry of the rule, plus a penalty for nodes outside the domain.
A rule with more unknowns than independent equations lies on a family of rules; among them the one with the smallest sum of w^2 has the weights spread most evenly and, in the tests so far,
the nodes farthest from the boundary.  Before a rule is checked in, run this program on it (CLAUDE.md).
The objective is  f = sum(w^2) / sum(w^2 at the start) + lambda * sum over all nodes i and all faces k of (exp(a_ik / tau) - 1 - a_ik / tau),  where a_ik = max(0, -c_ik) is the distance of node i outside face k and c_ik its distance from the face measured in barycentric coordinates
(simplices: the barycentric coordinates; boxes: half the distance to the two faces of every axis).  Only nodes outside are punished: a node inside or on the boundary costs nothing, a node 0.1 outside a face costs lambda * (e^(0.1 / tau) - 1 - 0.1 / tau).  The penalty is smooth where the boundary is crossed (value and slope are 0 there) and grows exponentially outward.
The penalty is soft: it trades the weights against the distance outside instead of a hard bound, and for a PO rule it pulls the outer nodes back towards the domain.
The method: projected gradient on the solution manifold.  The unknowns are the node coordinates and v with w = v^2 (so the weights stay positive).  In every step the gradient of f is projected onto the null
space of the Jacobian of the moment equations (singular value decomposition, so the projection does not depend on a basis and respects every symmetry of the rule), a step along the projected direction is corrected
back onto the solution manifold by Gauss-Newton steps, and the result is made exactly symmetric by averaging over the symmetry group of the rule (the subgroup of the symmetries of the domain that maps the
nodes and weights of the rule onto themselves).  A step is accepted only if the residual stays below 1e-13, f falls and, for a PI rule, every node stays inside.  When the nodes come within 1e-4 of a larger
symmetry group during the descent, they are made exactly symmetric and the descent goes on (the minimum often has more symmetry than the start).
Usage: python tools/minimize_l2.py <rule.json> [--write] [--lambda 0.05] [--tau 0.1] [--allow-closer] [--max-vars 2500] [--max-iter 1000]
Without --write the result is only reported.  With --write the nodes, the weights and residual_reported of the rule file are replaced and a sentence is added to its method text; the interval proof has to be
made again afterwards (python tools/verify_rigorous.py --strict --save <rule.json>).
Needs numpy and scipy (code/cubature_atlas.py)."""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
import cubature_atlas as C

SENTENCE = "; the weights were then minimised in the L2 norm under all moment equations with a penalty for nodes outside the domain and the symmetry kept (tools/minimize_l2.py)"


def domain_symmetries(domain):
    """list of functions that map the nodes X (N x d) of the domain onto the nodes of a symmetric image"""
    out = []
    if domain in ("triangle", "tetra"):
        d = 2 if domain == "triangle" else 3
        for perm in itertools.permutations(range(d + 1)):
            def f(X, perm=perm):
                L = np.c_[1 - X.sum(axis=1), X]
                return L[:, list(perm)][:, 1:]
            out.append(f)
    else:
        d = 2 if domain == "square" else 3
        for perm in itertools.permutations(range(d)):
            for signs in itertools.product([1.0, -1.0], repeat=d):
                def f(X, perm=perm, signs=signs):
                    return X[:, list(perm)] * np.array(signs)
                out.append(f)
    return out


def rule_group(domain, X, W, tol=1e-8):
    """the symmetries of the domain that map the rule onto itself: list of (function, permutation of the node indices)"""
    tree = cKDTree(X)
    group = []
    for f in domain_symmetries(domain):
        Y = f(X)
        dist, idx = tree.query(Y)
        if dist.max() > tol or len(set(idx.tolist())) != len(X):
            continue
        if np.abs(W[idx] - W).max() > tol * max(1.0, W.max()):
            continue
        group.append((f, idx))
    return group


def symmetrize(X, W, group):
    """average over the group: node j gets the mean of the images of the nodes that the group maps onto j"""
    Xs = np.zeros_like(X)
    Ws = np.zeros_like(W)
    for f, idx in group:
        Y = f(X)
        Xs[idx] += Y
        Ws[idx] += W
    return Xs / len(group), Ws / len(group)


def inside_margin(domain, X):
    """distance to the boundary: smallest barycentric coordinate (simplices) or 1 - max |x| (boxes)"""
    if domain in ("triangle", "tetra"):
        return float(np.c_[1 - X.sum(axis=1), X].min())
    return float(1.0 - np.abs(X).max())


def face_distances(domain, X):
    """distances c_ik of every node from every face (barycentric coordinates, or half the distance to the faces of a box)"""
    if domain in ("triangle", "tetra"):
        return np.c_[1 - X.sum(axis=1), X]
    return np.c_[(1 - X) / 2, (1 + X) / 2]


def penalty(domain, X, tau):
    """penalty for nodes outside the domain: sum over nodes and faces of exp(a / tau) - 1 - a / tau with a = max(0, -c) the distance outside the face, and its gradient with respect to the node coordinates;
    nodes inside (also on the boundary) cost nothing"""
    c = face_distances(domain, X)
    a = np.maximum(0.0, -c)
    e = np.exp(np.minimum(a / tau, 600.0))
    value = float((e - 1.0 - a / tau).sum())
    g = np.where(c < 0, -(e - 1.0) / tau, 0.0)      # derivative of the penalty with respect to c
    if domain in ("triangle", "tetra"):
        grad = -g[:, :1] + g[:, 1:]
    else:
        d = X.shape[1]
        grad = (-g[:, :d] + g[:, d:]) / 2
    return value, grad


def build_jac(B, X, v):
    """Jacobian of the moment equations with respect to the node coordinates and v (w = v^2), built in chunks of nodes so that the complex-step arrays stay small"""
    n, d = X.shape
    m = len(B.idx)
    W = v ** 2
    J = np.empty((m, n * d + n))
    chunk = max(1, 4000000 // m)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        Xa = X[a:b]
        J[:, n * d + a:n * d + b] = B(Xa).T * (2 * v[a:b])[None, :]
        for c in range(d):
            Xc = Xa.astype(complex)
            Xc[:, c] += 1e-30j
            J[:, d * a + c:d * b:d] = (B(Xc).imag / 1e-30 * W[a:b, None]).T
    return J


class Linear:
    """projection onto the null space of J and minimum norm solutions of J dq = -r.  Small systems: singular value decomposition.  Large systems: LSMR (matrix-free least squares) on the dense Jacobian,
    which needs no second copy of J, no square matrix of the size of the basis and no full decomposition; the memory is the Jacobian plus a few vectors."""

    def __init__(self, J, dense_limit, lsmr_iter):
        self.J = J
        self.dense = min(J.shape) <= dense_limit
        self.lsmr_iter = lsmr_iter
        if self.dense:
            U, S, Vt = np.linalg.svd(J, full_matrices=False)
            rank = int((S > 1e-10 * S[0]).sum())
            self.U, self.S, self.Vt, self.rank = U[:, :rank], S[:rank], Vt[:rank], rank
        else:
            self.op = LinearOperator(J.shape, matvec=lambda x: J @ x, rmatvec=lambda y: J.T @ y, dtype=float)
            self.opT = LinearOperator((J.shape[1], J.shape[0]), matvec=lambda y: J.T @ y, rmatvec=lambda x: J @ x, dtype=float)

    def project(self, g):
        if self.dense:
            return g - self.Vt.T @ (self.Vt @ g)
        y = lsmr(self.opT, g, atol=1e-14, btol=1e-14, conlim=1e10, maxiter=self.lsmr_iter)[0]
        return g - self.J.T @ y

    def solve(self, r):
        if self.dense:
            return -(self.Vt.T @ ((self.U.T @ r) / self.S))
        return -lsmr(self.op, r, atol=1e-14, btol=1e-14, conlim=1e10, maxiter=self.lsmr_iter)[0]


def minimise(domain, degree, X, W, inside, lam=0.05, tau=0.1, max_iter=1000, time_limit=1800.0, dense_limit=1500, lsmr_iter=3000):
    B = C.Basis(domain, degree, check_ortho=False)
    n, d = X.shape
    group = rule_group(domain, X, W)
    v = np.sqrt(W)
    s0 = float(np.sum(W ** 2))

    def res(Xq, vq):
        return C.residual(B, Xq, vq ** 2)

    def jac(Xq, vq):
        return build_jac(B, Xq, vq)

    def objective(Xq, vq):
        return float(np.sum(vq ** 4)) / s0 + lam * penalty(domain, Xq, tau)[0]

    t0 = time.time()
    f_start = objective(X, v)
    total_iters = 0

    def descend(X, v, group):
        fval = objective(X, v)
        step = 1.0
        stall = 0
        iters = 0
        for it in range(max_iter):
            if time.time() - t0 > time_limit:
                break
            iters = it + 1
            lin = Linear(jac(X, v), dense_limit, lsmr_iter)
            gx = lam * penalty(domain, X, tau)[1]
            g = np.concatenate([gx.reshape(-1), 4 * v ** 3 / s0])
            direction = -lin.project(g)
            gnorm = float(np.linalg.norm(direction))
            if gnorm < 1e-12:
                break
            direction = direction / gnorm
            accepted = False
            for _ in range(40):
                q = np.concatenate([X.reshape(-1), v]) + step * direction * min(1.0, 1e3 * gnorm + 0.05)
                Xq = q[:n * d].reshape(n, d)
                vq = q[n * d:]
                for _ in range(4):
                    dq = lin.solve(res(Xq, vq))
                    Xq = Xq + dq[:n * d].reshape(n, d)
                    vq = vq + dq[n * d:]
                Xq, Wq = symmetrize(Xq, vq ** 2, group) if group else (Xq, vq ** 2)
                vq = np.sqrt(np.clip(Wq, 0.0, None))
                r_new = float(np.linalg.norm(res(Xq, vq)))
                f_new = objective(Xq, vq)
                ok = r_new < 1e-13 and f_new < fval and np.isfinite(f_new) and vq.min() > 0
                if ok and inside:
                    ok = inside_margin(domain, Xq) > 1e-12
                if ok:
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                break
            gain = fval - f_new
            X, v, fval = Xq, vq, f_new
            step = min(step * 1.5, 1.0)
            stall = stall + 1 if gain < 1e-10 * abs(fval) else 0
            if stall >= 4:
                break
        return X, v, fval, iters

    X, v, fval, iters = descend(X, v, group)
    total_iters += iters
    # the minimum often has more symmetry than the start: when the nodes are within 1e-4 of a larger group, make them exactly symmetric, correct onto the solution manifold and go on
    for _ in range(3):
        bigger = rule_group(domain, X, v ** 2, tol=1e-4)
        if len(bigger) <= len(group):
            break
        Xs, Ws = symmetrize(X, v ** 2, bigger)
        vs = np.sqrt(np.clip(Ws, 0.0, None))
        for _ in range(8):
            dq = Linear(jac(Xs, vs), dense_limit, lsmr_iter).solve(res(Xs, vs))
            Xs = Xs + dq[:n * d].reshape(n, d)
            vs = vs + dq[n * d:]
            Xs, Ws = symmetrize(Xs, vs ** 2, bigger)
            vs = np.sqrt(np.clip(Ws, 0.0, None))
        f_s = objective(Xs, vs)
        ok = float(np.linalg.norm(res(Xs, vs))) < 1e-13 and f_s <= fval + 1e-6 * abs(fval) and vs.min() > 0 and bool(rule_group(domain, Xs, vs ** 2, tol=1e-9))
        if ok and inside:
            ok = inside_margin(domain, Xs) > 1e-12
        if not ok:
            break
        group = rule_group(domain, Xs, vs ** 2, tol=1e-9)
        X, v, fval = Xs, vs, f_s
        X, v, fval, iters = descend(X, v, group)
        total_iters += iters
    W = v ** 2
    return X, W, f_start, fval, len(group), total_iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rule")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.05, help="weight of the penalty for nodes outside, in units of the starting sum of w^2")
    ap.add_argument("--tau", type=float, default=0.1, help="length scale of the penalty exp(a / tau) - 1 - a / tau, a the barycentric distance outside a face")
    ap.add_argument("--max-vars", type=int, default=40000, help="skip rules with more unknowns than this (the dense Jacobian needs (equations x unknowns) x 8 bytes)")
    ap.add_argument("--dense-limit", type=int, default=1500, help="systems with at most this many equations or unknowns use the singular value decomposition, larger ones LSMR")
    ap.add_argument("--lsmr-iter", type=int, default=3000, help="iteration limit of every LSMR solve")
    ap.add_argument("--allow-closer", action="store_true", help="with --write: also write a PI rule whose smallest distance to the boundary became smaller (the penalty only punishes outer nodes, so the descent may move inner nodes onto the boundary)")
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--time-limit", type=float, default=1800.0)
    a = ap.parse_args()
    with open(a.rule, encoding="utf-8-sig") as fh:
        rule = json.load(fh)
    domain = rule["domain"]
    X = np.array(rule["nodes"], float)
    W = np.array(rule["weights"], float)
    n, d = X.shape
    if (d + 1) * n > a.max_vars:
        print("%s: %d unknowns, skipped" % (rule["id"], (d + 1) * n))
        return
    inside = rule.get("quality") == "PI"
    m0 = inside_margin(domain, X)
    X1, W1, f0, f1, ng, iters = minimise(domain, rule["degree"], X, W, inside, a.lam, a.tau, a.max_iter, a.time_limit, a.dense_limit, a.lsmr_iter)
    B = C.Basis(domain, rule["degree"], check_ortho=False)
    r1 = float(np.max(np.abs(C.residual(B, X1, W1))))
    m1 = inside_margin(domain, X1)
    out0 = int((face_distances(domain, X).min(axis=1) < 0).sum())
    out1 = int((face_distances(domain, X1).min(axis=1) < 0).sum())
    print("%s: f %.5f -> %.5f; sum w^2 %.6f -> %.6f (%.1f%%), distance to the boundary %.3e -> %.3e, nodes outside %d -> %d, min weight %.3e -> %.3e, symmetry group of order %d, residual %.1e, %d steps" % (
        rule["id"], f0, f1, np.sum(W ** 2), np.sum(W1 ** 2), 100 * (np.sum(W1 ** 2) / np.sum(W ** 2) - 1), m0, m1, out0, out1, W.min(), W1.min(), ng, r1, iters))
    if a.write and inside and m1 < 0.99 * m0 and not a.allow_closer:
        print("not written: the distance to the boundary of the PI rule became smaller (%.3e -> %.3e); use --allow-closer to write it anyway" % (m0, m1))
        return
    if a.write and f1 < f0 * (1 - 1e-6) and r1 < 1e-12:
        rule["nodes"] = X1.tolist()
        rule["weights"] = W1.tolist()
        rule["residual_reported"] = r1
        if SENTENCE not in rule.get("method", ""):
            rule["method"] = rule.get("method", "") + SENTENCE
        old_check = rule.get("numerical_check")
        if old_check is not None:
            cert = C.certificate(domain, rule["degree"], X1, W1)
            old_check.update({"res": float(cert["resid"]), "sigma": float(cert["sigma"]), "h": float(cert["h"]), "rad": float(cert["radius"]), "cert": bool(cert["certified"]),
                              "inside": bool(cert["inside"])})
        # the layout of the rule files: one header key per line, then nodes and weights one row per line
        out = ["{"]
        for key, value in rule.items():
            if key not in ("nodes", "weights"):
                out.append("  %s: %s," % (json.dumps(key), json.dumps(value)))
        out.append('  "nodes": [')
        out.append(",\n".join("    " + json.dumps([float(x) for x in row]) for row in X1))
        out.append("  ],")
        out.append('  "weights": [')
        out.append(",\n".join("    " + json.dumps(float(w)) for w in W1))
        out.append("  ]")
        out.append("}")
        with open(a.rule, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(out) + "\n")
        print("written; make the interval proof again: python tools/verify_rigorous.py --strict %s --save" % a.rule)


if __name__ == "__main__":
    main()

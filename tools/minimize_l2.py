"""minimize_l2: minimise the L2 norm of the weights (sum of w^2) of a rule under all moment equations, keeping the symmetry of the rule.
A rule with more unknowns than independent equations lies on a family of rules; among them the one with the smallest sum of w^2 has the weights spread most evenly and, in the tests so far,
the nodes farthest from the boundary.  Before a rule is checked in, run this program on it (CLAUDE.md).
The method: projected gradient on the solution manifold.  The unknowns are the node coordinates and v with w = v^2 (so the weights stay positive).  In every step the gradient of sum(v^4) is projected onto the null
space of the Jacobian of the moment equations (singular value decomposition, so the projection does not depend on a basis and respects every symmetry of the rule), a step along the projected direction is corrected
back onto the solution manifold by Gauss-Newton steps, and the result is made exactly symmetric by averaging over the symmetry group of the rule (the subgroup of the symmetries of the domain that maps the
nodes and weights of the rule onto themselves).  A step is accepted only if the residual stays below 1e-13, the sum of w^2 falls and, for a PI rule, every node stays inside.
Usage: python tools/minimize_l2.py <rule.json> [--write] [--max-vars 2500] [--max-iter 300]
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
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
import cubature_atlas as C

SENTENCE = "; the weights were then minimised in the L2 norm under all moment equations with the symmetry kept (tools/minimize_l2.py)"


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
    n = len(X)
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


def minimise(domain, degree, X, W, inside, max_iter=300, time_limit=1800.0):
    B = C.Basis(domain, degree)
    n, d = X.shape
    group = rule_group(domain, X, W)
    v = np.sqrt(W)
    margin0 = inside_margin(domain, X)

    def res(Xq, vq):
        return C.residual(B, Xq, vq ** 2)

    def jac(Xq, vq):
        J = C.jacobian(B, Xq, vq ** 2)
        return np.hstack([J[:, :n * d], J[:, n * d:] * (2 * vq)[None, :]])

    t0 = time.time()
    obj = float(np.sum(v ** 4))
    obj_start = obj
    step = 1.0
    stall = 0
    for it in range(max_iter):
        if time.time() - t0 > time_limit:
            break
        J = jac(X, v)
        U, S, Vt = np.linalg.svd(J, full_matrices=False)
        rank = int((S > 1e-10 * S[0]).sum())
        Vn = np.linalg.svd(J, full_matrices=True)[2][rank:] if rank < J.shape[1] else np.zeros((0, J.shape[1]))
        g = np.concatenate([np.zeros(n * d), 4 * v ** 3])
        direction = -Vn.T @ (Vn @ g)
        gnorm = float(np.linalg.norm(direction))
        if gnorm < 1e-12:
            break
        direction = direction / gnorm
        accepted = False
        for _ in range(30):
            q = np.concatenate([X.reshape(-1), v]) + step * direction * min(1.0, 1e3 * gnorm + 0.05)
            Xq = q[:n * d].reshape(n, d)
            vq = q[n * d:]
            for _ in range(4):
                r = res(Xq, vq)
                dq = -(Vt[:rank].T @ ((U[:, :rank].T @ r) / S[:rank]))
                Xq = Xq + dq[:n * d].reshape(n, d)
                vq = vq + dq[n * d:]
            Xq, Wq = symmetrize(Xq, vq ** 2, group) if group else (Xq, vq ** 2)
            vq = np.sqrt(np.clip(Wq, 0.0, None))
            r_new = float(np.linalg.norm(res(Xq, vq)))
            obj_new = float(np.sum(vq ** 4))
            ok = r_new < 1e-13 and obj_new < obj and np.isfinite(obj_new) and vq.min() > 0
            if ok and inside:
                ok = inside_margin(domain, Xq) >= 0.99 * margin0 and inside_margin(domain, Xq) > 1e-12
            if ok:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
        gain = obj - obj_new
        X, v, obj = Xq, vq, obj_new
        step = min(step * 1.5, 1.0)
        stall = stall + 1 if gain < 1e-10 * obj else 0
        if stall >= 4:
            break
    W = v ** 2
    return X, W, obj_start, obj, len(group), it + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rule")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--max-vars", type=int, default=2500, help="skip rules with more unknowns than this (the singular value decomposition of the Jacobian is dense)")
    ap.add_argument("--max-iter", type=int, default=300)
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
    X1, W1, o0, o1, ng, iters = minimise(domain, rule["degree"], X, W, inside, a.max_iter, a.time_limit)
    B = C.Basis(domain, rule["degree"])
    r1 = float(np.max(np.abs(C.residual(B, X1, W1))))
    m1 = inside_margin(domain, X1)
    print("%s: sum w^2 %.6f -> %.6f (%.1f%%), distance to the boundary %.3e -> %.3e, min weight %.3e -> %.3e, symmetry group of order %d, residual %.1e, %d steps" % (
        rule["id"], o0, o1, 100 * (o1 / o0 - 1), m0, m1, W.min(), W1.min(), ng, r1, iters))
    if a.write and o1 < o0 * (1 - 1e-6) and r1 < 1e-12:
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
        print("written; make the interval proof again: python tools/verify_rigorous.py --strict --save %s" % a.rule)


if __name__ == "__main__":
    main()

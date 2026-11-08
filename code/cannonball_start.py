#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cannonball_start.py
===================

Tetrahedron (d = 3) cubature search that starts from a cannonball packing.

Idea
----
The counting value for degree n is
    Z(n) = dim P_n / (d + 1) = (n+1)(n+2)(n+3) / 24,
the number of points for which the moment equations (dim P_n of them) and the
unknowns (d + 1 per point: d coordinates and one weight) balance.
We place N = ceil(Z(n)) equal balls into the regular tetrahedron like a stack of
cannonballs (the tetrahedral pile = principal lattice of the simplex), remove the
surplus balls of the smallest complete pile and relax the remaining ones by
soft-sphere repulsion until no two balls and no ball and wall overlap more than
necessary.  The ball centres (in barycentric coordinates, so the start is affine
invariant) with equal weights 1/N are the start for the moment equations.

The equations are then solved by continuation in the degree: the rule is made
exact for degree m0 (heavily under-determined, easy), then m0+1, ..., n.
Three solvers are available (--solver):
  trf   scipy least_squares, trust region reflective, with bounds.  Points are
        written in collapsed (Duffy) coordinates u in [0,1]^3,
        x = u1, y = (1-u1) u2, z = (1-u1)(1-u2) u3, weights w >= 0.
        Points can reach and leave the boundary.  Slow for large n
        (one SVD of the Jacobian per iteration).
  lm    Levenberg-Marquardt of simplex_cubature.py (group C1, lambda = s^2/sum,
        w = v^2).  Fast, but points that reach the boundary get stuck there
        (zero gradient), so it rarely converges from this start.
  free  Levenberg-Marquardt without any constraint (Cartesian coordinates and
        weights).  Fast; the result may have nodes outside or negative weights.
If the final stage fails, the program reports the residual; if N is larger than
needed, it can continue with node elimination (--eliminate, PI rules only).

Results so far (seed 1 to 4 unless stated)
------------------------------------------
  n = 10, trf: N = 73 never exact (seed 1 to 4; seed 1 also with --maxfev 3000),
               N = 74 PI-exact for seed 4 with --maxfev 3000 (rule CA-T3-10-74-PI-MPC,
               previously 75 points in the atlas), N = 75 exact for seed 2 and 4,
               N = 76, 77 exact for every seed.
  n = 16, trf: N = 243 = ceil(Z) not exact (seed 1, residual 4e-2).
  n = 24, free: N = 732 = ceil(Z) exact up to degree 23, degree 24 not exact
               (seed 1, residual 1e-2 after 500 iterations).

Usage
-----
    python3 cannonball_start.py --n 10 --extra 2 --seed 4 --maxfev 3000
    python3 cannonball_start.py --n 24 --solver free
    python3 cannonball_start.py --n 24 --extra 60 --solver trf --eliminate 3600
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simplex_cubature as sc


# Vertices of a regular tetrahedron with edge length 1 (rows)
REG = np.array([[0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, math.sqrt(3.0) / 2.0, 0.0],
                [0.5, math.sqrt(3.0) / 6.0, math.sqrt(2.0 / 3.0)]])

# Height of the regular unit tetrahedron
HEIGHT = math.sqrt(2.0 / 3.0)


def counting_value(n):
    """Z(n) = (n+1)(n+2)(n+3)/24 for the tetrahedron."""
    # Exact rational value as float
    return (n + 1) * (n + 2) * (n + 3) / 24.0


def pile(k):
    """Barycentric coordinates of a tetrahedral cannonball pile with k layers (Tet(k) balls)."""
    # Collect lattice points
    pts = []
    # Layer index from the bottom face (lambda_3 = layer / (k-1))
    for a in range(k):
        # Second index
        for b in range(k - a):
            # Third index
            for c in range(k - a - b):
                # Last index is determined
                e = k - 1 - a - b - c
                # Store integer lattice point
                pts.append((a, b, c, e))
    # Normalise to barycentric coordinates
    return np.array(pts, dtype=float) / max(k - 1, 1)


def to_cart(L):
    """Barycentric -> Cartesian coordinates in the regular tetrahedron."""
    # Linear map
    return L @ REG


def to_bary(X):
    """Cartesian (regular tetrahedron) -> barycentric coordinates."""
    # Matrix of edge vectors
    T = (REG[1:] - REG[0]).T
    # Solve for lambda_1..3
    l = np.linalg.solve(T, (X - REG[0]).T).T
    # lambda_0
    l0 = 1.0 - l.sum(axis=1)
    # Stack
    return np.column_stack([l0, l])


def cannonball_start(N, rng, removal="spread", iters=3000, log=print):
    """N ball centres (barycentric) of a relaxed cannonball packing in the regular tetrahedron."""
    # Smallest pile with at least N balls
    k = 1
    # Grow until the pile is large enough
    while k * (k + 1) * (k + 2) // 6 < N:
        k += 1
    # Lattice points of the pile
    P = pile(k)
    # Ball radius of the pile when the balls touch each other and the faces:
    # outer edge E = (k-1) a + sqrt(6) a with a = 2 r = 1/(k-1+sqrt(6))
    a = 1.0 / (k - 1 + math.sqrt(6.0))
    # Barycentric margin of the centres (distance r to every face / height)
    m = 0.5 * a / HEIGHT
    # Shrink the pile so that the balls lie inside the tetrahedron
    L = m + (1.0 - 4.0 * m) * P
    # Surplus balls to remove
    surplus = len(L) - N
    # Report
    log("cannonball pile: %d layers, %d balls, remove %d (%s)" % (k, len(L), surplus, removal))
    # Remove balls
    if surplus > 0:
        # Remove the top layers (apex = vertex 3) first
        if removal == "top":
            # Order by height (lambda_3), lowest first
            order = np.argsort(L[:, 3], kind="stable")
            # Keep the lowest N
            L = L[order[:N]]
        else:
            # Remove balls spread uniformly at random
            keep = rng.choice(len(L), size=N, replace=False)
            # Keep them
            L = L[np.sort(keep)]
    # Cartesian coordinates
    X = to_cart(L)
    # Inward unit normals and offsets of the faces: n.x + c >= 0 inside
    normals = []
    # Offsets
    offsets = []
    # Face opposite vertex i
    for i in range(4):
        # Other vertices
        o = [j for j in range(4) if j != i]
        # Normal from the cross product
        nv = np.cross(REG[o[1]] - REG[o[0]], REG[o[2]] - REG[o[0]])
        # Unit length
        nv = nv / np.linalg.norm(nv)
        # Orient towards vertex i
        if np.dot(nv, REG[i] - REG[o[0]]) < 0:
            nv = -nv
        # Store
        normals.append(nv)
        # Offset
        offsets.append(-np.dot(nv, REG[o[0]]))
    # Arrays
    Nf = np.array(normals)
    # Arrays
    cf = np.array(offsets)
    # Target radius: equal-volume estimate, the solver grows it as long as possible
    r = 0.5 * a
    # Relaxation (soft spheres, overlap energy with growing radius)
    for it in range(iters):
        # Pairwise differences
        D = X[:, None, :] - X[None, :, :]
        # Distances
        dist = np.sqrt((D * D).sum(axis=2))
        # Ignore self pairs
        np.fill_diagonal(dist, np.inf)
        # Overlap of each pair
        ov = np.maximum(2.0 * r - dist, 0.0)
        # Push apart along the connecting line
        F = (ov / dist)[:, :, None] * D
        # Total force
        G = 0.5 * F.sum(axis=1)
        # Wall distances
        wd = X @ Nf.T + cf
        # Wall overlap
        wo = np.maximum(r - wd, 0.0)
        # Push away from the walls
        G += wo @ Nf
        # Move
        X = X + 0.5 * G
        # Largest overlap
        mo = max(float(ov.max()), float(wo.max()))
        # Grow when the packing is (almost) free of overlaps, shrink when jammed
        if mo < 1e-4 * r:
            r *= 1.002
        elif mo > 0.05 * r:
            r *= 0.999
    # Final minimum distances
    D = X[:, None, :] - X[None, :, :]
    # Pair distances
    dist = np.sqrt((D * D).sum(axis=2))
    # Ignore self pairs
    np.fill_diagonal(dist, np.inf)
    # Report the packing
    log("relaxed packing: radius %.5f (pile %.5f), min distance %.5f, min wall distance %.5f"
        % (r, 0.5 * a, float(dist.min()), float((X @ Nf.T + cf).min())))
    # Back to barycentric coordinates, clipped into the simplex
    L = np.maximum(to_bary(X), 1e-6)
    # Renormalise
    return L / L.sum(axis=1, keepdims=True)


def rule_from_points(L, w):
    """C1 rule of simplex_cubature from barycentric points and weights (sum 1)."""
    # One orbit per point
    mults = [(1, 1, 1, 1)] * len(w)
    # s = sqrt(lambda)
    svals = [np.sqrt(np.maximum(p, 0.0)) for p in L]
    # v = sqrt(w)
    rule = sc.Rule(3, "C1", mults, svals, np.sqrt(w))
    # Fix gauge
    rule.normalise()
    # Return
    return rule


def duffy_to_bary(U):
    """Collapsed coordinates u in [0,1]^3 -> barycentric points and dX/dU (P, 3, 3)."""
    # Components
    a = U[:, 0]
    # Second component
    b = U[:, 1]
    # Third component
    c = U[:, 2]
    # Cartesian x
    x = a
    # Cartesian y
    y = (1.0 - a) * b
    # Cartesian z
    z = (1.0 - a) * (1.0 - b) * c
    # Barycentric coordinates
    L = np.column_stack([1.0 - x - y - z, x, y, z])
    # Jacobian of the map
    D = np.zeros((len(U), 3, 3))
    # dx/da
    D[:, 0, 0] = 1.0
    # dy/da
    D[:, 1, 0] = -b
    # dy/db
    D[:, 1, 1] = 1.0 - a
    # dz/da
    D[:, 2, 0] = -(1.0 - b) * c
    # dz/db
    D[:, 2, 1] = -(1.0 - a) * c
    # dz/dc
    D[:, 2, 2] = (1.0 - a) * (1.0 - b)
    # Return both
    return L, D


def bary_to_duffy(L):
    """Barycentric points -> collapsed coordinates in [0,1]^3."""
    # Cartesian coordinates
    x = L[:, 1]
    # Cartesian y
    y = L[:, 2]
    # Cartesian z
    z = L[:, 3]
    # First collapsed coordinate
    a = x
    # Second
    b = y / np.maximum(1.0 - a, 1e-300)
    # Third
    c = z / np.maximum((1.0 - a) * (1.0 - b), 1e-300)
    # Clip into the unit cube
    return np.clip(np.column_stack([a, b, c]), 0.0, 1.0)


def moments_cart(pkd, L, w, deriv):
    """Residual (E,) and Jacobian w.r.t. Cartesian coordinates (E, 3P) and weights (E, P)."""
    # Basis values and barycentric gradients
    V, dV = pkd.evaluate(L, deriv)
    # Moment residual
    r = V @ w
    # Exact mean of the constant
    r[0] -= 1.0
    # Values only
    if not deriv:
        return r, None, None
    # Cartesian gradient: d/dx_i = d/dlambda_i - d/dlambda_0
    G = dV[:, 1:, :] - dV[:, :1, :]
    # Return residual, gradient (E, 3, P) times weight, and values
    return r, G * w[None, None, :], V


def solve_trf(m, L, w, maxfev=400):
    """Bounded least squares (scipy trf) for degree m; returns (ok, L, w, residual norm)."""
    # Basis of degree m
    pkd = sc.PKDBasis(3, m)
    # Number of points
    P = len(w)
    # Tolerance as in simplex_cubature.solve_lm
    tol = 2e-14 * math.sqrt(pkd.size)

    # Residual function
    def fun(p):
        # Barycentric points
        Lp, _ = duffy_to_bary(p[:3 * P].reshape(P, 3))
        # Residual
        r, _, _ = moments_cart(pkd, Lp, p[3 * P:], False)
        # Return
        return r

    # Jacobian function
    def jac(p):
        # Barycentric points and map Jacobian
        Lp, D = duffy_to_bary(p[:3 * P].reshape(P, 3))
        # Moment Jacobian in Cartesian coordinates
        _, Gw, V = moments_cart(pkd, Lp, p[3 * P:], True)
        # Chain rule to collapsed coordinates
        GU = np.einsum("eip,pij->ejp", Gw, D)
        # Columns ordered point by point (u1, u2, u3), then the weights
        JU = GU.transpose(0, 2, 1).reshape(V.shape[0], -1)
        # Full Jacobian
        return np.concatenate([JU, V], axis=1)

    # Lower bounds
    lb = np.zeros(4 * P)
    # Upper bounds
    ub = np.concatenate([np.ones(3 * P), np.full(P, np.inf)])
    # Start vector
    p0 = np.clip(np.concatenate([bary_to_duffy(L).ravel(), w]), lb, ub)
    # Solve
    res = least_squares(fun, p0, jac=jac, bounds=(lb, ub), method="trf", x_scale="jac",
                        xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=maxfev)
    # Result points
    Lr, _ = duffy_to_bary(res.x[:3 * P].reshape(P, 3))
    # Residual norm
    nr = float(np.linalg.norm(res.fun))
    # Return
    return nr < tol, Lr, res.x[3 * P:].copy(), nr


def solve_free(m, L, w, maxit=500):
    """Unconstrained Levenberg-Marquardt for degree m; returns (ok, L, w, residual norm)."""
    # Basis of degree m
    pkd = sc.PKDBasis(3, m)
    # Number of points
    P = len(w)
    # Tolerance
    tol = 2e-14 * math.sqrt(pkd.size)
    # Cartesian coordinates
    X = L[:, 1:].copy()
    # Weights
    w = w.copy()

    # Residual and Jacobian at (X, w)
    def evaluate(X, w, deriv):
        # Barycentric points
        Lx = np.column_stack([1.0 - X.sum(axis=1), X])
        # Residual and parts of the Jacobian
        r, Gw, V = moments_cart(pkd, Lx, w, deriv)
        # Values only
        if not deriv:
            return r, None
        # Full Jacobian
        J = np.concatenate([Gw.transpose(0, 2, 1).reshape(V.shape[0], -1), V], axis=1)
        # Return
        return r, J

    # Initial residual and Jacobian
    r, J = evaluate(X, w, True)
    # Residual norm
    nr = float(np.linalg.norm(r))
    # Damping
    mu = 1e-4
    # Main loop
    for it in range(maxit):
        # Converged
        if nr < tol:
            break
        # Column scaling
        cn = np.linalg.norm(J, axis=0)
        # Avoid zero columns
        cn[cn < 1e-300] = 1.0
        # Scaled Jacobian
        Js = J / cn
        # Normal matrix of the under-determined system
        M = Js @ Js.T
        # Damping scale
        scale = float(np.trace(M)) / M.shape[0]
        # Acceptance flag
        accepted = False
        # Damping loop
        for inner in range(25):
            # Minimum-norm step
            y = np.linalg.solve(M + mu * scale * np.eye(M.shape[0]), r)
            # Unscaled step
            dx = -(Js.T @ y) / cn
            # Trial coordinates
            Xt = X + dx[:3 * P].reshape(P, 3)
            # Trial weights
            wt = w + dx[3 * P:]
            # Trial residual
            rt, _ = evaluate(Xt, wt, False)
            # Trial norm
            nt = float(np.linalg.norm(rt))
            # Accept on decrease
            if nt < nr:
                # Update coordinates
                X = Xt
                # Update weights
                w = wt
                # Update norm
                nr = nt
                # Less damping
                mu = max(mu / 3.0, 1e-15)
                # Mark
                accepted = True
                # Leave damping loop
                break
            # More damping
            mu *= 4.0
        # No progress possible
        if not accepted:
            break
        # New Jacobian
        r, J = evaluate(X, w, True)
    # Barycentric points
    Lx = np.column_stack([1.0 - X.sum(axis=1), X])
    # Return
    return nr < tol, Lx, w, nr


def continuation_points(L, w, n, m0, solver, log, maxfev=400, deadline=float("inf")):
    """Degree continuation with the trf or free solver; returns (ok, L, w, residual norm)."""
    # Residual norm
    nr = float("inf")
    # Success flag
    ok = False
    # Loop over degrees
    for m in range(m0, n + 1):
        # Out of time: stop before the next degree
        if time.time() > deadline:
            log("  time budget exhausted before degree %d" % m)
            ok = False
            break
        # Start time
        t0 = time.time()
        # Solve this degree
        if solver == "trf":
            ok, L, w, nr = solve_trf(m, L, w, maxfev)
        else:
            ok, L, w, nr = solve_free(m, L, w)
        # Report
        log("  degree %2d: %d equations, %s, residual %.3e, %.0fs"
            % (m, math.comb(m + 3, 3), "ok" if ok else "FAILED", nr, time.time() - t0))
        # Give up
        if not ok:
            break
    # Return
    return ok, L, w, nr


def continuation(rule, n, m0, log, maxit=200):
    """Solve degree m0, m0+1, ..., n in turn with solve_lm; returns (ok, rule, residual norm)."""
    # Current rule
    cur = rule
    # Loop over degrees
    for m in range(m0, n + 1):
        # Basis of degree m
        basis = sc.InvariantBasis(3, m, "C1")
        # Start time
        t0 = time.time()
        # Solve, restarting LM while it keeps making progress
        ok = False
        # Residual norm
        nr = float("inf")
        # Restarts
        for rep in range(6):
            # One LM run
            ok, cur, nr_new = sc.solve_lm(basis, cur, maxit=maxit)
            # Stop when converged or stuck
            if ok or nr_new > 0.7 * nr:
                nr = nr_new
                break
            # Remember
            nr = nr_new
            # Move away from the boundary for the next run
            cur = sc.unstick(cur, floor=0.01)
        # Report
        log("  degree %2d: %d equations, %s, residual %.3e, %.0fs"
            % (m, basis.size, "ok" if ok else "FAILED", nr, time.time() - t0))
        # Give up
        if not ok:
            return False, cur, nr
    # All degrees solved
    return True, cur, nr


def write_atlas_json(path, n, L, w, err, start_info):
    """Write a rule (reference tetrahedron, weights sum 1/6) together with its quality."""
    # Cartesian coordinates of the reference tetrahedron: x = lambda_1, y = lambda_2, z = lambda_3
    nodes = [[float(p[1]), float(p[2]), float(p[3])] for p in L]
    # Weights scaled by the volume 1/6
    weights = [float(x) / 6.0 for x in w]
    # Positive weights
    positive = bool(np.min(w) > 0.0)
    # Nodes in the closed simplex
    inside = bool(np.min(L) >= -1e-14)
    # Quality code
    quality = ("P" if positive else "N") + ("I" if inside else "O")
    # Data
    data = {"degree": n, "points": len(weights), "quality": quality, "max_moment_error_pkd": err,
            "start": start_info, "nodes": nodes, "weights": weights}
    # Write
    with open(path, "w") as f:
        json.dump(data, f)
    # Return the quality code
    return quality


def main():
    """Command line driver."""
    # Parser
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Degree
    ap.add_argument("--n", type=int, default=24, help="polynomial degree")
    # Extra points above ceil(Z(n))
    ap.add_argument("--extra", type=int, default=0, help="points in addition to ceil(Z(n))")
    # Removal mode
    ap.add_argument("--removal", choices=["spread", "top"], default="spread", help="which surplus balls of the pile to remove")
    # Solver
    ap.add_argument("--solver", choices=["trf", "lm", "free"], default="trf", help="nonlinear solver (see module docstring)")
    # Total wall-clock budget
    ap.add_argument("--budget", type=float, default=0.0, help="total wall-clock budget in seconds (0 = none); elimination ends by then")
    # Function evaluations per degree for trf
    ap.add_argument("--maxfev", type=int, default=400, help="function evaluations per degree (trf)")
    # First degree of the continuation
    ap.add_argument("--m0", type=int, default=0, help="first degree of the continuation (default n/2)")
    # Random seed
    ap.add_argument("--seed", type=int, default=1)
    # Elimination time
    ap.add_argument("--eliminate", type=float, default=0.0, help="seconds of node elimination after success (PI only)")
    # Output directory
    ap.add_argument("--out", type=str, default="cannonball_out")
    # Parse
    args = ap.parse_args()
    # Output directory
    os.makedirs(args.out, exist_ok=True)

    # Logger
    def log(msg):
        # Print with flush
        print(msg, flush=True)

    # Start time
    t_start = time.time()
    # Deadline of the whole run
    deadline = t_start + args.budget if args.budget > 0 else float("inf")
    # Counting value
    Z = counting_value(args.n)
    # Number of points
    N = int(math.ceil(Z)) + args.extra
    # Report
    log("degree %d: Z(n) = %.4f, start with N = %d points (%d unknowns, %d equations), solver %s"
        % (args.n, Z, N, 4 * N, math.comb(args.n + 3, 3), args.solver))
    # Random generator
    rng = np.random.default_rng(args.seed)
    # Cannonball start
    L = cannonball_start(N, rng, args.removal, log=log)
    # Equal weights
    w = np.full(N, 1.0 / N)
    # First degree
    m0 = args.m0 if args.m0 > 0 else args.n // 2
    # Continuation
    if args.solver == "lm":
        # Orbit form of simplex_cubature
        ok, rule, nr = continuation(rule_from_points(L, w), args.n, m0, log)
        # Expanded rule
        L, w = rule.expand()
    else:
        # Point form
        ok, L, w, nr = continuation_points(L, w, args.n, m0, args.solver, log, args.maxfev, deadline)
    # Drop points with zero weight (the bounded solver may switch points off)
    keep = np.abs(w) > 0.0
    # Points
    L = L[keep]
    # Weights
    w = w[keep]
    # Independent check
    ver = sc.Verifier(3, args.n)
    # Moment error
    err = ver.error(L, w)
    # Start description
    info = "cannonball packing N=%d removal=%s seed=%d, solver %s, continuation from degree %d" % (N, args.removal, args.seed, args.solver, m0)
    # Output path
    path = os.path.join(args.out, "tet_n%02d_m%d_seed%d_%s_%s.json" % (args.n, len(w), args.seed, args.solver, "ok" if ok else "fail"))
    # Write
    quality = write_atlas_json(path, args.n, L, w, err, info)
    # Report
    log("result: %s, N = %d, quality %s, max moment error %.3e, min weight %.3e, min lambda %.3e"
        % ("exact" if ok else "not exact", len(w), quality, err, float(w.min()), float(L.min())))
    # Report
    log("written " + path)
    # Optional elimination (PI rules only)
    if ok and quality == "PI" and args.eliminate > 0:
        # Basis of the target degree
        basis = sc.InvariantBasis(3, args.n, "C1")
        # End of the elimination
        stop = min(time.time() + args.eliminate, deadline)
        # Current rule
        best = rule_from_points(L, w)
        # Point count before elimination
        before = len(w)
        # Point count of the last snapshot
        last = before
        # Rounds of at most 20 minutes, each followed by a saved snapshot
        while time.time() < stop - 60.0:
            # End of this round
            round_end = min(time.time() + 1200.0, stop)
            # Eliminate
            best = sc.eliminate(basis, best, round_end, 10, log, rng)
            # Expanded
            Le, we = best.expand()
            # Error
            err = ver.error(Le, we)
            # Report
            log("after elimination round: N = %d, max moment error %.3e" % (len(we), err))
            # Output path
            path = os.path.join(args.out, "tet_n%02d_m%d_seed%d_elim.json" % (args.n, len(we), args.seed))
            # Write
            write_atlas_json(path, args.n, Le, we, err, info + ", then node elimination from %d points" % before)
            # Report
            log("written " + path)
            # Stop when a whole round brought no progress
            if len(we) >= last:
                break
            # Remember the count
            last = len(we)


if __name__ == "__main__":
    main()

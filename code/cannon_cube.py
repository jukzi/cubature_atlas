#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cannon_cube.py
==============

Cube [-1,1]^3, degree n: start the search from a cannonball packing
(face-centred cubic lattice, FCC) instead of a tensor Gauss rule.

Modes
-----
free : exactly N = ceil(Z(n)) nodes, Z(n) = (n+1)(n+2)(n+3)/24, no symmetry.
       Projected Levenberg-Marquardt (minimum-norm steps) on all
       dim P_n moment equations in the orthonormal Legendre basis.
sym  : FCC packing invariant under a cube group (default T_d), decomposed
       into orbits, solved with cubature_search.solve_rule, then reduced by
       cubature_search.eliminate and cubature_search.lns.

Usage
-----
   python3 cannon_cube.py free 24 --time 1800
   python3 cannon_cube.py sym 24 --m 6 --group Td --time 1800

Style rule of this file: one command per line.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import itertools
import math
import time
import numpy as np
from scipy.optimize import nnls
from scipy.linalg import cho_factor
from scipy.linalg import cho_solve
import cubature_search as cs


def fcc_points(m, parity=0):
    """FCC lattice points (i,j,k), |i|,|j|,|k| <= m, i+j+k = parity mod 2."""
    # All integer points of the box.
    r = np.arange(-m, m + 1)
    # Cartesian product.
    P = np.array(list(itertools.product(r, r, r)), dtype=float)
    # Keep one parity class (the FCC lattice or its shifted copy).
    keep = (np.sum(P, axis=1).astype(int) % 2) == parity
    # Return the lattice points.
    return P[keep]


def fcc_in_cube(N, rng):
    """Cannonball packing with exactly N centres in [-1,1]^3.

    The FCC lattice is scaled until about N centres fit into the box
    [-1+a/2, 1-a/2]^3 (a = nearest-neighbour distance); surplus centres
    are removed at random.
    """
    # Large lattice to cut from.
    base = fcc_points(14, 0)
    # Nearest-neighbour distance of the unit lattice.
    nn = math.sqrt(2.0)
    # Best scale found so far.
    best = None
    # Sweep the lattice spacing.
    for s in np.linspace(0.05, 0.4, 4000):
        # Half width of the admissible box for the centres.
        L = 1.0 - 0.5 * s * nn
        # Centres inside.
        X = base * s
        # Selection mask.
        inside = np.max(np.abs(X), axis=1) <= L + 1e-12
        # Count.
        c = int(np.sum(inside))
        # Keep the smallest surplus.
        if c >= N and (best is None or c < best[0]):
            best = (c, s, X[inside])
    # Unpack.
    c, s, X = best
    # Remove surplus centres at random.
    if c > N:
        # Random subset of size N.
        idx = rng.choice(c, N, replace=False)
        # Apply.
        X = X[np.sort(idx)]
    # Report.
    print("FCC spacing %.4f, %d centres fit, %d kept" % (s, c, X.shape[0]), flush=True)
    # Return the centres.
    return X


def multi_indices(n):
    """All (a,b,c) with a+b+c <= n."""
    # List comprehension over the box.
    return np.array([a for a in itertools.product(range(n + 1), repeat=3) if sum(a) <= n])


def basis_and_grad(X, idx, n):
    """V[k,i] = prod_j L_{idx[k,j]}(X[i,j]) and its partial derivatives."""
    # Legendre tables (n+1, N, 3).
    L, dL = cs.legendre_table(X, n)
    # Factors per coordinate.
    F = [L[idx[:, j], :, j] for j in range(3)]
    # Derivative factors.
    G = [dL[idx[:, j], :, j] for j in range(3)]
    # Basis values.
    V = F[0] * F[1] * F[2]
    # Gradients.
    D0 = G[0] * F[1] * F[2]
    # Second coordinate.
    D1 = F[0] * G[1] * F[2]
    # Third coordinate.
    D2 = F[0] * F[1] * G[2]
    # Return all.
    return V, D0, D1, D2


def free_solve(n, N, time_limit, seed, out):
    """Projected minimum-norm Levenberg-Marquardt on all moment equations."""
    # Random generator.
    rng = np.random.default_rng(seed)
    # Multi-indices.
    idx = multi_indices(n)
    # Number of equations.
    K = idx.shape[0]
    # Right-hand side (probability measure, orthonormal basis).
    e0 = np.zeros(K)
    # Constant function.
    e0[0] = 1.0
    # Cannonball start.
    X = fcc_in_cube(N, rng)
    # Basis at the start.
    V, _, _, _ = basis_and_grad(X, idx, n)
    # Non-negative weights by NNLS.
    W, rn = nnls(V, e0, maxiter=50 * N)
    # Report.
    print("K=%d equations, %d unknowns; NNLS residual %.3e, zero weights %d" % (K, 4 * N, rn, int(np.sum(W == 0))), flush=True)
    # Lift zero weights a little so that their nodes can move.
    W = np.maximum(W, 0.05 / N)
    # Damping.
    lam = 1e-2
    # Start time.
    t0 = time.time()
    # Iteration counter.
    it = 0
    # Current residual.
    V, D0, D1, D2 = basis_and_grad(X, idx, n)
    # Residual vector.
    r = V @ W - e0
    # Norm.
    f = float(np.linalg.norm(r))
    # Main loop.
    while time.time() - t0 < time_limit and f > 1e-14:
        # Jacobian blocks: columns x_i, y_i, z_i, w_i.
        J = np.hstack([D0 * W, D1 * W, D2 * W, V])
        # Active set: coordinates at the box and weights at zero that push outward are frozen.
        g = J.T @ r
        # Packed unknowns.
        z = np.concatenate([X[:, 0], X[:, 1], X[:, 2], W])
        # Lower bounds.
        lo = np.concatenate([-np.ones(3 * N), np.zeros(N)])
        # Upper bounds.
        hi = np.concatenate([np.ones(3 * N), np.full(N, np.inf)])
        # Frozen variables.
        frozen = ((z <= lo + 1e-15) & (g > 0)) | ((z >= hi - 1e-15) & (g < 0))
        # Free Jacobian.
        Jf = J[:, ~frozen]
        # Gram matrix.
        A = Jf @ Jf.T
        # Try damping values.
        accepted = False
        # Loop over damping factors.
        for lam_try in (lam, lam * 10.0, lam * 100.0, lam * 1e3, lam * 1e4):
            # Regularised Gram matrix.
            Ad = A + lam_try * np.eye(K)
            # Cholesky.
            c = cho_factor(Ad)
            # Minimum-norm step.
            step = np.zeros(4 * N)
            # Free part.
            step[~frozen] = -Jf.T @ cho_solve(c, r)
            # New point (projected).
            zn = np.clip(z + step, lo, hi)
            # Unpack coordinates.
            Xn = np.stack([zn[:N], zn[N:2 * N], zn[2 * N:3 * N]], axis=1)
            # Unpack weights.
            Wn = zn[3 * N:]
            # Basis at the trial point.
            Vn, D0n, D1n, D2n = basis_and_grad(Xn, idx, n)
            # Trial residual.
            rn_ = Vn @ Wn - e0
            # Its norm.
            fn = float(np.linalg.norm(rn_))
            # Accept on decrease.
            if fn < f:
                # Take the step.
                X, W, V, D0, D1, D2, r, f = Xn, Wn, Vn, D0n, D1n, D2n, rn_, fn
                # Reduce damping.
                lam = max(lam_try / 30.0, 1e-14)
                # Mark.
                accepted = True
                # Stop trying.
                break
        # No decrease: increase damping and perturb slightly.
        if not accepted:
            # Larger damping.
            lam = lam * 1e5
        # Next iteration.
        it += 1
        # Progress.
        if it % 5 == 0 or f < 1e-12:
            # Print.
            print("it %4d  |r| %.3e  lam %.1e  minw %.2e  frozen %d  %.0fs" % (it, f, lam, W.min() * N, int(np.sum(frozen)), time.time() - t0), flush=True)
        # Damping exploded: give up.
        if lam > 1e12:
            break
    # Save the final state.
    np.save(out, np.c_[X, W * 8.0])
    # Report.
    print("FINAL |r| %.3e  minw*N %.3e  zero weights %d  max|x| %.6f  saved %s" % (f, W.min() * N, int(np.sum(W < 1e-13)), np.max(np.abs(X)), out), flush=True)
    # Return.
    return X, W, f


def rule_from_points(prob, X, W):
    """Decompose a group-invariant point set into orbits of the problem."""
    # Assignment flags.
    used = np.zeros(X.shape[0], dtype=bool)
    # Lookup table.
    lookup = {tuple(np.round(x * 1e9).astype(np.int64)): i for i, x in enumerate(X)}
    # Orbit lists.
    pids = []
    # Coordinates.
    crds = []
    # Weights.
    wts = []
    # Loop over points.
    for i in range(X.shape[0]):
        # Skip assigned.
        if used[i]:
            continue
        # Orbit.
        orb = cs.orbit_points(prob.perms, prob.signs, X[i])
        # Mark the orbit.
        for y in orb:
            # Flag.
            used[lookup[tuple(np.round(y * 1e9).astype(np.int64))]] = True
        # Find a pattern.
        q_found = None
        # Coordinates found.
        v_found = None
        # Loop over patterns.
        for q, pat in enumerate(prob.patterns):
            # Size must agree.
            if pat.size != orb.shape[0]:
                continue
            # Try the orbit points.
            for y in orb:
                # Fit.
                v = pat.fit(y)
                # Success.
                if v is not None:
                    q_found = q
                    v_found = v
                    break
            # Done.
            if q_found is not None:
                break
        # Store.
        pids.append(q_found)
        # Coordinates.
        crds.append(v_found)
        # Total orbit weight.
        wts.append(W[i] * orb.shape[0])
    # Build the rule.
    return cs.Rule(prob, pids, crds, wts)


def write_rule(fname, n, group, X, W):
    """Write x y z w with weights summing to 8."""
    # Open.
    with open(fname, "w") as fh:
        # Header.
        fh.write("# cubature rule on [-1,1]^3, degree %d, %d points, group %s (cannonball start)\n" % (n, X.shape[0], group))
        # Header 2.
        fh.write("# columns: x y z w ; sum(w) = 8 ; max moment error %.2e\n" % cs.verify(X, W, n))
        # Points.
        for x, w in zip(X, W * 8.0):
            # One line.
            fh.write("% .17e % .17e % .17e % .17e\n" % (x[0], x[1], x[2], w))


def sym_search(n, m, parity, group, time_limit, seed, outdir, scale):
    """Group-invariant cannonball start, then elimination and LNS."""
    # Random generator.
    rng = np.random.default_rng(seed)
    # Problem.
    prob = cs.Problem(n, group)
    # Lattice centres.
    P = fcc_points(m, parity)
    # Scale into the cube: centres at distance scale*spacing/2 from the faces.
    X = P / (m + scale)
    # Equal start weights.
    W = np.full(X.shape[0], 1.0 / X.shape[0])
    # Orbit decomposition.
    rule = rule_from_points(prob, X, W)
    # Orbit statistics.
    struct = {}
    # Count families.
    for p in rule.pat_ids:
        # Name.
        key = prob.patterns[p].name()
        # Increment.
        struct[key] = struct.get(key, 0) + 1
    # Degrees of freedom.
    dof = rule.pack().size
    # Report.
    print("[%s] FCC m=%d parity=%d: %d points, %d orbits, %d unknowns, %d equations, %s" % (group, m, parity, X.shape[0], len(rule.pat_ids), dof, prob.neq, struct), flush=True)
    # Solve the full nonlinear system from the equal-weight start.
    ok, res = cs.solve_rule(rule, maxit=400, patience=60)
    # Report.
    print("[%s] start solved: ok=%s residual %.2e, points %d" % (group, ok, res, cs.actual_points(rule)[0].shape[0]), flush=True)
    # Not exact: stop.
    if not ok:
        return None
    # Output folder.
    os.makedirs(outdir, exist_ok=True)
    # Saver.
    def save(r):
        # Expand.
        Xc, Wc = cs.actual_points(r)
        # Write.
        write_rule(os.path.join(outdir, "cube3d_n%02d_m%d_%s.txt" % (n, Xc.shape[0], group)), n, group, Xc, Wc)
    # Save the start.
    save(rule)
    # Greedy elimination.
    t0 = time.time()
    # Eliminate.
    red = cs.eliminate(rule, rng=rng, time_limit=0.4 * time_limit)
    # Report.
    print("[%s] elimination: %d points (%.0fs)" % (group, red.npoints(), time.time() - t0), flush=True)
    # Save.
    save(red)
    # Target structure.
    _, target = cs.structure_target(prob)
    # LNS.
    best = cs.lns(red, rng, iters=10 ** 6, time_limit=max(10.0, time_limit - (time.time() - t0)), temp0=4.0, target=target, on_improve=save, log=lambda s: print("[%s] %s (%.0fs)" % (group, s.strip(), time.time() - t0), flush=True))
    # Save.
    save(best)
    # Report.
    print("[%s] FINAL %d points" % (group, cs.actual_points(best)[0].shape[0]), flush=True)
    # Return.
    return best


def continue_search(n, fname, group, time_limit, seed, outdir):
    """Continue LNS from a saved rule (x y z w, sum w = 8)."""
    # Random generator.
    rng = np.random.default_rng(seed)
    # Problem.
    prob = cs.Problem(n, group)
    # Load.
    D = np.loadtxt(fname)
    # Orbit decomposition (weights normalised to 1).
    rule = rule_from_points(prob, D[:, :3], D[:, 3] / 8.0)
    # Polish.
    ok, res = cs.solve_rule(rule, maxit=200, patience=40)
    # Report.
    print("[%s] loaded %d points, ok=%s residual %.2e" % (group, D.shape[0], ok, res), flush=True)
    # Output folder.
    os.makedirs(outdir, exist_ok=True)
    # Saver.
    def save(r):
        # Expand.
        Xc, Wc = cs.actual_points(r)
        # Write.
        write_rule(os.path.join(outdir, "cube3d_n%02d_m%d_%s.txt" % (n, Xc.shape[0], group)), n, group, Xc, Wc)
    # Target structure.
    _, target = cs.structure_target(prob)
    # Start time.
    t0 = time.time()
    # LNS.
    best = cs.lns(rule, rng, iters=10 ** 6, time_limit=time_limit, temp0=4.0, target=target, on_improve=save, log=lambda s: print("[%s] %s (%.0fs)" % (group, s.strip(), time.time() - t0), flush=True))
    # Report.
    print("[%s] FINAL %d points" % (group, cs.actual_points(best)[0].shape[0]), flush=True)
    # Return.
    return best


def main():
    """Command line interface."""
    # Parser.
    ap = argparse.ArgumentParser()
    # Mode.
    ap.add_argument("mode", choices=["free", "sym", "cont"])
    # Start file (cont mode).
    ap.add_argument("--start", default="")
    # Degree.
    ap.add_argument("n", type=int)
    # Time limit.
    ap.add_argument("--time", type=float, default=1800.0)
    # Seed.
    ap.add_argument("--seed", type=int, default=1)
    # Number of nodes (free mode, default ceil(Z)).
    ap.add_argument("--N", type=int, default=0)
    # Lattice half width (sym mode).
    ap.add_argument("--m", type=int, default=6)
    # Lattice parity (sym mode).
    ap.add_argument("--parity", type=int, default=0)
    # Group (sym mode).
    ap.add_argument("--group", default="Td")
    # Boundary offset (sym mode): X = P/(m+scale).
    ap.add_argument("--scale", type=float, default=0.5)
    # Output.
    ap.add_argument("--out", default="cannon_out")
    # Parse.
    a = ap.parse_args()
    # Free mode.
    if a.mode == "free":
        # Node count.
        N = a.N if a.N > 0 else int(math.ceil((a.n + 1) * (a.n + 2) * (a.n + 3) / 24.0))
        # Output folder.
        os.makedirs(a.out, exist_ok=True)
        # Run.
        free_solve(a.n, N, a.time, a.seed, os.path.join(a.out, "free_n%02d_N%d_s%d.npy" % (a.n, N, a.seed)))
    # Continue mode.
    elif a.mode == "cont":
        # Run.
        continue_search(a.n, a.start, a.group, a.time, a.seed, a.out)
    # Symmetric mode.
    else:
        # Run.
        sym_search(a.n, a.m, a.parity, a.group, a.time, a.seed, a.out, a.scale)


if __name__ == "__main__":
    main()

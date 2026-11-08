#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cannon_tetra.py
===============

Tetrahedron, degree n: start the symmetric search of simplex_cubature.py from a
cannonball pile (tetrahedral stack of K layers of equal balls) instead of a
random NNLS start.

The ball centres of a pile with K layers are, in barycentric coordinates,
    lambda = (i + a, j + a, k + a, l + a) / (K - 1 + 4 a),   i + j + k + l = K - 1,
with the wall offset a (a free parameter: distance of the outer centres from
the faces in units of the layer spacing).
The pile is invariant under the full symmetric group S4, so it decomposes into
S4 orbits.  With equal weights it is the start for the Levenberg-Marquardt
solve of the S4-invariant moment equations; then orbit elimination (group S),
transfer to group A (rotations A4) and further elimination.

Usage
-----
   python3 cannon_tetra.py 24 --K 20 --a 0.25 --time 1500
   python3 cannon_tetra.py 28 --K 25 --a 0.25 --cont 8 --time 2400

Style rule of this file: one command per line.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import time
import numpy as np
from scipy.optimize import nnls
import simplex_cubature as sc


def pile(K, a):
    """Barycentric centres of a tetrahedral pile with K layers."""
    # Output list.
    out = []
    # Loop over the first three indices.
    for i in range(K):
        # Second index.
        for j in range(K - i):
            # Third index.
            for k in range(K - i - j):
                # Fourth index.
                l = K - 1 - i - j - k
                # Shifted and normalised barycentric coordinates.
                out.append(np.array([i + a, j + a, k + a, l + a]) / (K - 1 + 4.0 * a))
    # Return as array.
    return np.array(out)


def pile_rule(K, a):
    """S4-orbit rule of the pile with equal weights."""
    # All centres.
    P = pile(K, a)
    # Distinct orbits: sorted coordinate tuples.
    keys = {}
    # Loop over the centres.
    for p in P:
        # Canonical key (descending values).
        key = tuple(np.round(np.sort(p)[::-1], 12))
        # Count.
        keys[key] = keys.get(key, 0) + 1
    # Multiplicities.
    mults = []
    # s values.
    svals = []
    # Loop over orbits.
    for key in keys:
        # Distinct values and their multiplicities.
        vals, cnt = np.unique(np.array(key), return_counts=True)
        # Descending order.
        vals = vals[::-1]
        # Matching counts.
        cnt = cnt[::-1]
        # Store multiplicities.
        mults.append(tuple(int(c) for c in cnt))
        # s = sqrt(value).
        svals.append(np.sqrt(vals))
    # Equal weight per point (total 1).
    v = np.full(len(mults), np.sqrt(1.0 / P.shape[0]))
    # Build the rule.
    rule = sc.Rule(3, "S", mults, svals, v)
    # Fix the gauge.
    rule.normalise()
    # Return rule and number of centres.
    return rule, P.shape[0]


def main():
    """Command line interface."""
    # Parser.
    ap = argparse.ArgumentParser()
    # Degree.
    ap.add_argument("n", type=int)
    # Layers.
    ap.add_argument("--K", type=int, default=0)
    # Start file (continue mode).
    ap.add_argument("--start", default="")
    # Group for the continue mode.
    ap.add_argument("--group", default="A")
    # Wall offset.
    ap.add_argument("--a", type=float, default=0.5)
    # Time limit in seconds.
    ap.add_argument("--time", type=float, default=1200.0)
    # Share of the time for group S.
    ap.add_argument("--sshare", type=float, default=0.5)
    # Seed.
    ap.add_argument("--seed", type=int, default=1)
    # Moves tried per elimination step.
    ap.add_argument("--ntry", type=int, default=12)
    # NNLS start weights instead of equal weights.
    ap.add_argument("--nnls", type=int, default=1)
    # Start degree of the degree continuation (0 = none).
    ap.add_argument("--cont", type=int, default=0)
    # Output folder.
    ap.add_argument("--out", default="cannon_tetra_out")
    # Parse.
    a = ap.parse_args()
    # Random generator.
    rng = np.random.default_rng(a.seed)
    # Start time.
    t0 = time.time()
    # Log function.
    def log(msg):
        # Print with time stamp.
        print("[n=%d %.0fs] %s" % (a.n, time.time() - t0, msg), flush=True)
    # Complete basis.
    pkd = sc.PKDBasis(3, a.n)
    # Verifier.
    ver = sc.Verifier(3, a.n, pkd)
    # Invariant bases.
    basisS = sc.InvariantBasis(3, a.n, "S", pkd)
    # Rotation invariants.
    basisA = sc.InvariantBasis(3, a.n, "A", pkd)
    # Continue from a saved rule (JSON orbit form of simplex_cubature.save_rule).
    if a.start:
        # Load.
        start, _ = sc.load_rule(a.start)
        # Target group.
        g = a.group
        # Basis of the target group.
        Bg = basisA if g == "A" else sc.InvariantBasis(3, a.n, g, pkd)
        # Transfer the rule.
        rule0 = sc.to_group(start, g)
        # Report.
        log("loaded %s: %d points, continue with group %s" % (a.start, rule0.npoints(), g))
        # Output folder.
        os.makedirs(a.out, exist_ok=True)
        # Search with warm start and kicks.
        best = sc.search(Bg, 3, g, [rule0], t0 + a.time, rng, a.ntry, log, nnls_starts=False)
        # Verify.
        pts, wts = best.expand()
        # Error.
        err = ver.error(pts, wts)
        # Save if valid.
        if err < 1e-11 and pts.min() > -1e-14 and wts.min() > 0.0:
            # Write.
            sc.save_rule(os.path.join(a.out, "simplex_d3_n%02d_%s_m%d.txt" % (a.n, g, len(wts))), 3, a.n, g, best, err)
        # Report.
        log("FINAL %s: %d points, error %.1e" % (g, len(wts), err))
        # Done.
        return
    # Pile.
    rule, N = pile_rule(a.K, a.a)
    # Report.
    log("pile K=%d a=%.3f: %d balls, %d S-orbits, %d parameters, equations S %d A %d" % (a.K, a.a, N, len(rule.mults), rule.nparams(), basisS.size, basisA.size))
    # Weights of the pile orbits by NNLS (orbits with zero weight are dropped).
    if a.nnls and a.cont == 0:
        # Basis values at the representatives.
        Q, _ = basisS.evaluate(rule.reps(), False)
        # Orbit contributions.
        Amat = Q * rule.sizes()[None, :]
        # Non-negative least squares.
        w, rn = nnls(Amat, basisS.b, maxiter=5000)
        # Support.
        keep = np.where(w > 1e-15)[0]
        # Rule on the support.
        rule = sc.Rule(3, "S", [rule.mults[i] for i in keep], [rule.s[i] for i in keep], np.sqrt(w[keep]))
        # Report.
        log("NNLS weights: residual %.2e, %d orbits kept, %d points" % (rn, len(keep), rule.npoints()))
    # Degree continuation: solve for degree m0, m0+1, ..., n (equal start weights).
    if a.cont > 0:
        # Loop over the degrees below n.
        for m in range(a.cont, a.n):
            # Invariant basis of degree m.
            Bm = sc.InvariantBasis(3, m, "S", sc.PKDBasis(3, m))
            # Solve.
            ok, rule, res = sc.solve_lm(Bm, rule, maxit=800)
            # Stop on failure.
            if not ok:
                # Report.
                log("continuation failed at degree %d, residual %.2e" % (m, res))
                # Give up.
                return
        # Report.
        log("continuation reached degree %d" % (a.n - 1))
    # Solve the S system.
    ok, rule, res = sc.solve_lm(basisS, rule, maxit=800)
    # Report.
    log("start solved: ok=%s residual %.2e, points %d" % (ok, res, rule.npoints()))
    # Give up.
    if not ok:
        return
    # Output folder.
    os.makedirs(a.out, exist_ok=True)
    # Saver.
    def save(r, g):
        # Expand.
        pts, wts = r.expand()
        # Error.
        err = ver.error(pts, wts)
        # Save if valid.
        if err < 1e-11 and pts.min() > -1e-14 and wts.min() > 0.0:
            # Write.
            sc.save_rule(os.path.join(a.out, "simplex_d3_n%02d_%s_m%d.txt" % (a.n, g, len(wts))), 3, a.n, g, r, err)
        # Return the error.
        return err
    # Canonical form.
    rule = sc.canonicalise(rule)
    # Elimination with group S.
    bestS = sc.eliminate(basisS, rule, t0 + a.sshare * a.time, a.ntry, lambda s: None, rng)
    # Report.
    log("S elimination: %d points, error %.1e" % (bestS.npoints(), save(bestS, "S")))
    # Transfer to group A.
    ruleA = sc.to_group(bestS, "A")
    # Further search with group A (warm start, kicks).
    bestA = sc.search(basisA, 3, "A", [ruleA], t0 + a.time, rng, a.ntry, log, nnls_starts=False)
    # Report.
    if bestA is not None:
        # Final line.
        log("FINAL A: %d points, error %.1e" % (bestA.npoints(), save(bestA, "A")))


if __name__ == "__main__":
    main()

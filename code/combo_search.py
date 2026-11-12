"""Combined search for few-node cubature rules on the triangle and the tetrahedron.

Combines the ideas of the atlas programs:
  1. Diverse starts (simplex_cubature.py): fresh NNLS starts and orbit elimination with the
     symmetry groups S_{d+1} and A_{d+1}; the reduced systems are small, so a symmetric
     rule is found in seconds to minutes. The atlas rule is used as an extra start.
  2. Breaking the symmetry (po_elim.py): the expanded symmetric rule is the start of the
     asymmetric node elimination with random perturbations, merges and Levenberg-Marquardt
     re-solves (fresh Jacobian per iteration, no early stop: that was measured to be best).
     INSIDE=1 keeps the nodes in the closed domain (PI, Duffy parameters in [0,1]);
     INSIDE=0 allows nodes in a box around it (PO).
  3. Acceptance only after the monomial test and the floating-point Newton-Kantorovich test
     (po_elim.ok_rule); the interval proof is run afterwards with tools/verify_rigorous.py.

Each round picks a group, finds a symmetric rule within TSYM seconds, and runs the
elimination for TELIM seconds if the symmetric rule is at most SLACK nodes above the best
rule so far. The best rule of the run is written as po_<domain>_d<degree>_n<N>.npy.

Usage:
    INSIDE=1 python3 combo_search.py <triangle|tetra> <degree> <seconds> <seed>
Environment: TSYM (default 120), TELIM (default 300), SLACK (default 6), MI (LM iterations).
"""

# Imports.
import os
import sys
import math
import time
import numpy as np

# Program directory.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import simplex_cubature as SC
import po_elim as PE

# Time per symmetric search and per elimination, allowed excess of the symmetric start.
TSYM = float(os.environ.get("TSYM", "120"))
TELIM = float(os.environ.get("TELIM", "300"))
SLACK = int(os.environ.get("SLACK", "6"))


def to_cartesian(rule, d):
    """Expanded symmetric rule -> Cartesian nodes of the atlas simplex and weights scaled to its volume."""
    pts, wts = rule.expand()
    X = np.array(pts)[:, 1:]
    W = np.array(wts) / math.factorial(d)
    return X, W


def main():
    # Arguments.
    name = sys.argv[1]
    deg = int(sys.argv[2])
    total = float(sys.argv[3])
    seed = int(sys.argv[4])
    d = 2 if name == "triangle" else 3
    rng = np.random.default_rng(seed)
    t0 = time.time()
    # Orthonormal basis and invariant bases of the orbit search.
    pkd = SC.PKDBasis(d, deg)
    bases = {g: SC.InvariantBasis(d, deg, g, pkd) for g in ("S", "A")}
    # Best rule so far: the atlas PI rule.
    X0, W0 = PE.atlas_best_pi(name, deg)
    bestN = len(W0)
    print("start", name, deg, "atlas PI", bestN, flush=True)
    # The first round starts from the atlas rule itself.
    starts = [("atlas", X0, W0)]
    rnd = 0
    while time.time() - t0 < total:
        rnd += 1
        if not starts:
            # Symmetric start: fresh NNLS start and orbit elimination with a random group.
            g = "S" if rng.random() < 0.4 else "A"
            dl = min(time.time() + TSYM, t0 + total)
            rule = SC.search(bases[g], d, g, [], dl, rng, 10, lambda s: None)
            if rule is None:
                continue
            X, W = to_cartesian(rule, d)
            print("round %d: group %s -> %d nodes (%.0fs)" % (rnd, g, len(W), time.time() - t0), flush=True)
            if len(W) > bestN + SLACK:
                continue
            starts.append((g, X, W))
        label, X, W = starts.pop(0)
        # Asymmetric elimination with perturbations and merges.
        left = t0 + total - time.time()
        res = PE.run_elim(name, deg, min(TELIM, left), rng, X, W)
        n = res[0]
        print("round %d: elimination from %s (%d) -> %d nodes (%.0fs)" % (rnd, label, len(W), n, time.time() - t0), flush=True)
        if res[1] is not None and n < bestN:
            bestN = n
            print("  NEW BEST %d" % n, flush=True)
    print("END", name, deg, bestN, flush=True)


if __name__ == "__main__":
    main()

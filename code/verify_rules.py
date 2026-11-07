#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_rules.py
===============

Independent check of cubature rule files written by cubature_search.py.

For every file  rules/cube3d_nXX_mYYY_<group>.txt  (columns x y z w on the
cube [-1,1]^3, weights summing to 8) it checks

  * exactness for ALL monomials x^a y^b z^c with a+b+c <= n, using the exact
    integral  prod_i (1 - (-1)^(a_i+1)) / (a_i + 1)  (0 for odd exponents),
    with compensated summation (math.fsum);
  * positivity of all weights;
  * that all points lie in [-1,1]^3 (boundary allowed).

It uses only the Python standard library and numpy, and none of the code of
the search program, so it is an independent test.

Style rule of this file: one command per line.
"""

# Imports.
import glob
import math
import os
import re
import sys
import numpy as np


def exact_monomial(a, b, c):
    """Exact integral of x^a y^b z^c over [-1,1]^3."""
    # Product of the one-dimensional integrals.
    val = 1.0
    # Loop over the exponents.
    for e in (a, b, c):
        # Odd exponent integrates to zero.
        if e % 2 == 1:
            return 0.0
        # Even exponent: 2 / (e + 1).
        val *= 2.0 / (e + 1)
    # Return the product.
    return val


def check_file(path):
    """Return (n, m, max relative error, min weight, max |coordinate|)."""
    # Degree from the file name.
    n = int(re.search(r"_n(\d+)_", os.path.basename(path)).group(1))
    # Points and weights.
    data = np.loadtxt(path, ndmin=2)
    # Coordinates.
    x = data[:, 0]
    # Second coordinate.
    y = data[:, 1]
    # Third coordinate.
    z = data[:, 2]
    # Weights.
    w = data[:, 3]
    # Powers of the coordinates up to degree n.
    px = np.vstack([x ** k for k in range(n + 1)])
    # Powers of y.
    py = np.vstack([y ** k for k in range(n + 1)])
    # Powers of z.
    pz = np.vstack([z ** k for k in range(n + 1)])
    # Largest error, relative to the integral of |x^a y^b z^c| (well-scaled).
    worst = 0.0
    # Loop over all exponents with total degree <= n.
    for a in range(n + 1):
        # Second exponent.
        for b in range(n + 1 - a):
            # Third exponent.
            for c in range(n + 1 - a - b):
                # Terms of the quadrature sum.
                terms = w * px[a] * py[b] * pz[c]
                # Compensated sum.
                q = math.fsum(terms)
                # Exact value.
                ex = exact_monomial(a, b, c)
                # Scale: integral of the absolute value of the monomial.
                scale = 8.0 / ((a + 1) * (b + 1) * (c + 1))
                # Relative error.
                err = abs(q - ex) / scale
                # Track the maximum.
                worst = max(worst, err)
    # Return the statistics.
    return n, data.shape[0], worst, float(w.min()), float(np.abs(data[:, :3]).max())


def main():
    """Check all rule files in the given directory (default: rules)."""
    # Directory.
    folder = sys.argv[1] if len(sys.argv) > 1 else "rules"
    # Files sorted by degree.
    files = sorted(glob.glob(os.path.join(folder, "cube3d_n*_m*_*.txt")))
    # Header.
    print("file                                   n      m    max_rel_err   min_weight   max|x|   ok")
    # All good flag.
    all_ok = True
    # Loop over files.
    for f in files:
        # Check.
        n, m, err, wmin, xmax = check_file(f)
        # Verdict.
        ok = err < 1e-12 and wmin > 0 and xmax <= 1.0 + 1e-15
        # Update the global flag.
        all_ok = all_ok and ok
        # Report.
        print("%-36s %3d %6d %12.2e %12.3e %8.5f   %s" % (os.path.basename(f), n, m, err, wmin, xmax, "yes" if ok else "NO"))
    # Summary.
    print("ALL OK" if all_ok else "SOME RULES FAILED")


# Entry point.
if __name__ == "__main__":
    main()

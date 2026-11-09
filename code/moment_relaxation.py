"""Moment relaxation for odd degrees 2k+1 (numerical test, not a proof).

For a rule with N nodes and positive weights let K be the discrete Gram matrix
of the orthonormal polynomials P_{k+1}.  Then N >= dim P_k + rank K, K is
positive semidefinite, and K depends only on the (unknown) discrete moments of
degree 2k+2:  K = I + C H(s) C^T, where C holds the leading coefficients of
P_{k+1}, s the differences "discrete minus true moment" of degree 2k+2 and
H(s)[a][b] = s[lead(a) + lead(b)].

The program asks numerically whether s and an m x r matrix F with
I + C H(s) C^T = F F^T exist.  If not (for many random starts), rules with
dim P_k + r nodes and positive weights are excluded at this level, but only
numerically.  If yes, this level cannot exclude them, and a proof needs moments
of higher degree (flat extension).

Usage:  python moment_relaxation.py DOMAIN k r [r ...]      DOMAIN = triangle or tetra
        python moment_relaxation.py triangle 4 1 2 3       (degree 9)
        python moment_relaxation.py tetra 2 2 3            (degree 5)
"""
import sys
import numpy as np
from math import factorial
from scipy.optimize import least_squares


def exact_degree(dim, e):
    """Exponent tuples of total degree exactly e."""
    if dim == 1:
        return [(e,)]
    out = []
    for a in range(e, -1, -1):
        for rest in exact_degree(dim - 1, e - a):
            out.append((a,) + rest)
    return out


def moment(e):
    """Integral of a monomial over the reference simplex: prod(e_i!) / (|e| + d)!."""
    num = 1
    for a in e:
        num *= factorial(a)
    return num / factorial(sum(e) + len(e))


def main():
    domain = sys.argv[1]
    k = int(sys.argv[2])
    ranks = [int(x) for x in sys.argv[3:]]
    dim = 2 if domain == "triangle" else 3
    basis = []
    for e in range(k + 2):
        basis.extend(exact_degree(dim, e))
    gram = np.array([[moment(tuple(x + y for x, y in zip(a, b))) for b in basis] for a in basis])
    coef = np.linalg.inv(np.linalg.cholesky(gram))
    top = [i for i, m in enumerate(basis) if sum(m) == k + 1]
    C = coef[np.ix_(top, top)]
    m = len(top)
    high = exact_degree(dim, 2 * k + 2)
    position = {e: i for i, e in enumerate(high)}
    index = np.array([[position[tuple(x + y for x, y in zip(basis[top[i]], basis[top[j]]))] for j in range(m)] for i in range(m)])
    upper = np.triu_indices(m)
    dim_pk = len(basis) - m
    rng = np.random.default_rng(0)
    for r in ranks:

        def residual(z):
            H = z[:len(high)][index]
            F = z[len(high):].reshape(m, r)
            return (np.eye(m) + C @ H @ C.T - F @ F.T)[upper]

        best = np.inf
        for trial in range(200):
            z0 = np.r_[0.01 * rng.standard_normal(len(high)), rng.standard_normal(m * r)]
            sol = least_squares(residual, z0, method="trf", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=3000)
            best = min(best, np.linalg.norm(sol.fun))
            if best < 1e-12:
                break
        unknowns = len(high) + m * r - r * (r - 1) // 2
        print("%s degree %d, N = %d (rank %d): %d equations, %d unknowns, best residual %.2e after %d starts" % (domain, 2 * k + 1, dim_pk + r, r, len(upper[0]), unknowns, best, trial + 1), flush=True)


if __name__ == "__main__":
    main()

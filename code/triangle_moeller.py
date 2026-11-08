"""Exact lower bounds for cubature on the triangle (odd degrees).

Triangle (0,0), (1,0), (0,1) with the Lebesgue measure.

Moeller's lower bound (general weight, no symmetry needed), in the form of
Y. Xu (structural matrices of the three-term relation):

    degree 2k+1:  N >= dim P_k + rank(M_k) / 2
    M_k = A_{k,1} A_{k,2}^T - A_{k,2} A_{k,1}^T

A_{k,i} is the (k+1) x (k+2) matrix in  x_i P_k = A_{k,i} P_{k+1} + B_{k,i} P_k + A_{k-1,i}^T P_{k-1},
P_k the orthonormal polynomials of degree k.  M_k is skew-symmetric, so its
rank is even and at most k+1; the bound adds at most floor((k+1)/2).

M_k = 0 is the Gauss case: then and only then a rule with dim P_k nodes can
exist (Mysovskikh).  rank(M_k) > 0 therefore also proves that no rule with
dim P_k nodes exists.

Computation without square roots: for any basis Q_0..Q_k of the orthogonal
polynomials of degree k (orthogonal to all of P_{k-1}, not normalised) put

    G[j][l] = < proj_V(x Q_j), proj_V(y Q_l) >,   V = orthogonal polynomials of degree k+1.

Then G - G^T is congruent to M_k, so it has the same rank.  Because x Q_j and
y Q_l have degree k+1, proj_V(f) = f - proj_{P_k}(f), and

    G[j][l] = <x Q_j, y Q_l> - <proj_{P_k}(x Q_j), proj_{P_k}(y Q_l)>.

All arithmetic is exact (fractions), so the printed ranks are exact.

Usage:  python triangle_moeller.py [max_degree]      (default 29)
"""
import sys
import time
from fractions import Fraction
from math import factorial


def monomials(deg):
    """Exponent pairs (a, b) with a + b <= deg, ordered by total degree."""
    out = []
    for e in range(deg + 1):
        for b in range(e + 1):
            out.append((e - b, b))
    return out


def moment(a, b):
    """Integral of x^a y^b over the reference triangle: a! b! / (a+b+2)!."""
    return Fraction(factorial(a) * factorial(b), factorial(a + b + 2))


def solve(A, B):
    """Solve A X = B exactly (A square, B list of columns); Gauss-Jordan with fractions."""
    n = len(A)
    m = len(B)
    # Augmented matrix [A | B].
    T = []
    for i in range(n):
        row = list(A[i])
        for c in range(m):
            row.append(B[c][i])
        T.append(row)
    for col in range(n):
        piv = col
        while T[piv][col] == 0:
            piv += 1
        T[col], T[piv] = T[piv], T[col]
        inv = 1 / T[col][col]
        T[col] = [v * inv for v in T[col]]
        for i in range(n):
            if i != col and T[i][col] != 0:
                f = T[i][col]
                T[i] = [a - f * b for a, b in zip(T[i], T[col])]
    # Columns of the solution.
    X = []
    for c in range(m):
        X.append([T[i][n + c] for i in range(n)])
    return X


def rank(M):
    """Exact rank of a matrix of fractions."""
    R = [list(r) for r in M]
    rows = len(R)
    cols = len(R[0]) if rows else 0
    r = 0
    for col in range(cols):
        piv = None
        for i in range(r, rows):
            if R[i][col] != 0:
                piv = i
                break
        if piv is None:
            continue
        R[r], R[piv] = R[piv], R[r]
        for i in range(rows):
            if i != r and R[i][col] != 0:
                f = R[i][col] / R[r][col]
                R[i] = [a - f * b for a, b in zip(R[i], R[r])]
        r += 1
    return r


def moeller_rank(k):
    """Rank of M_k for the triangle, exact."""
    # Monomial basis up to degree k+1 and its Gram matrix.
    basis = monomials(k + 1)
    index = {m: i for i, m in enumerate(basis)}
    size = len(basis)
    H = []
    for (a1, b1) in basis:
        H.append([moment(a1 + a2, b1 + b2) for (a2, b2) in basis])
    n_km1 = len(monomials(k - 1))
    n_k = len(monomials(k))
    # Orthogonal polynomials Q_j of degree k: monomial minus projection onto P_{k-1}.
    Hkm1 = [row[:n_km1] for row in H[:n_km1]]
    rhs = []
    for j in range(k + 1):
        col = index[(k - j, j)]
        rhs.append([H[i][col] for i in range(n_km1)])
    coef = solve(Hkm1, rhs)
    Q = []
    for j in range(k + 1):
        q = [Fraction(0)] * size
        q[index[(k - j, j)]] = Fraction(1)
        for i in range(n_km1):
            q[i] -= coef[j][i]
        Q.append(q)

    def shift(p, da, db):
        """Coefficient vector of x^da y^db * p (p of degree <= k)."""
        out = [Fraction(0)] * size
        for i, c in enumerate(p):
            if c != 0:
                a, b = basis[i]
                out[index[(a + da, b + db)]] += c
        return out

    X = [shift(q, 1, 0) for q in Q]
    Y = [shift(q, 0, 1) for q in Q]

    def gram_rows(f, rows):
        """Inner products <basis_i, f> for i < rows."""
        return [sum(H[i][t] * f[t] for t in range(size) if f[t] != 0) for i in range(rows)]

    UX = [gram_rows(f, n_k) for f in X]
    UY = [gram_rows(f, n_k) for f in Y]
    Hk = [row[:n_k] for row in H[:n_k]]
    ZY = solve(Hk, UY)
    G = []
    for j in range(k + 1):
        full = gram_rows(X[j], size)
        row = []
        for l in range(k + 1):
            direct = sum(full[t] * Y[l][t] for t in range(size) if Y[l][t] != 0)
            proj = sum(UX[j][t] * ZY[l][t] for t in range(n_k))
            row.append(direct - proj)
        G.append(row)
    M = [[G[j][l] - G[l][j] for l in range(k + 1)] for j in range(k + 1)]
    return rank(M)


def main():
    max_degree = int(sys.argv[1]) if len(sys.argv) > 1 else 29
    print("degree  dim P_k  rank M_k  bound  (dim P_k + rank/2)")
    for n in range(3, max_degree + 1, 2):
        k = (n - 1) // 2
        t0 = time.time()
        r = moeller_rank(k)
        dim = (k + 1) * (k + 2) // 2
        print("%6d  %7d  %8d  %5d   %.1fs" % (n, dim, r, dim + r // 2, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

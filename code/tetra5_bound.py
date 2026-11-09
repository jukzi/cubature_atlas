"""Tetrahedron, degree 5: no rule with 12 nodes, for any real weights (exact proof).

Reference tetrahedron (0,0,0), (1,0,0), (0,1,0), (0,0,1).  Nodes and weights
are arbitrary real numbers; the result is N >= 13 (Moeller's bound gives 12).

Notation (as in moeller_bound.py and tetra3_positive.py, here k = 2):
  Q_j   orthogonal polynomials of degree 2 (monomial minus projection onto P_1),
  R_a   orthogonal polynomials of degree 3 (monomial minus projection onto P_2),
  A_i   6 x 10 structural matrices; x_i Q_j = R_{lead(Q_j)+e_i} + (degree <= 2),
        so A_i is a 0/1 shift matrix,
  H     true Gram matrix of R,  K  discrete Gram matrix sum_m w_m R(x_m) R(x_m)^T,
  M_ij = A_i H A_j^T - A_j H A_i^T  (skew-symmetric 6 x 6), and for every rule
         A_i K A_j^T - A_j K A_i^T = M_ij.

Proof for N = 12:
  1. The discrete Gram matrix of P_3 has rank dim P_2 + rank K = 10 + rank K <= 12,
     and rank M_ij = 4 forces rank K >= 2.  So K = G S G^T with G (10 x 2) of rank 2
     and S = diag(+-1, +-1) (any signature, since the weights may be negative).
  2. B_i = A_i G gives M_ij = B_i S B_j^T - B_j S B_i^T.  As rank M_ij = 4, the column
     spaces S_i = col(B_i) are 2-dimensional, S_i + S_j = U_ij := col(M_ij) and
     S_i, S_j are complementary.  S_i also lies in W_i = intersection of the two U's
     that contain it.
  3. Exact computation: dim W_i = 3 and the three U_ij share a 2-dimensional
     subspace T (W_i = T + one vector w_i).  S_i = T is impossible (then S_i and S_j
     would meet), so S_i meets T in a line t_i and S_i = span(t_i, w_i + l_i v).
  4. Let Om_ij be the 2-form on U_ij inverse to M_ij.  M_ij lies in S_i (x) S_j, so S_i
     and S_j are Lagrangian for Om_ij.  The two Lagrangian conditions for S_i give
     one linear equation for t_i inside T and then l_i; both have unique solutions
     here, so S_1, S_2, S_3 are determined exactly.
  5. With B_i = S_i X_i (X_i invertible 2 x 2) the equations become linear in
     P_ij = X_i S X_j^T and have unique solutions.  Then
        Q = (P_12 P_23^{-T})^{-1} P_13 = X_3 S X_3^T
     must be symmetric.  The exact Q is not symmetric: contradiction.

All arithmetic is exact (fractions).  The option --selftest builds artificial data
M_ij from random G and signature S of the same structure (column spaces summing to
5 dimensions) and checks that steps 3 to 5 recover a symmetric Q of signature S.

Usage:  python tetra5_bound.py [--selftest]
"""
import random
import sys
from fractions import Fraction
from math import factorial


def monomials(deg):
    """Exponent triples of total degree exactly deg."""
    out = []
    for a in range(deg, -1, -1):
        for b in range(deg - a, -1, -1):
            out.append((a, b, deg - a - b))
    return out


def add(a, b):
    """Sum of two exponent triples."""
    return tuple(x + y for x, y in zip(a, b))


def moment(e):
    """Integral of x^a y^b z^c over the reference tetrahedron."""
    return Fraction(factorial(e[0]) * factorial(e[1]) * factorial(e[2]), factorial(sum(e) + 3))


def transpose(X):
    """Matrix transpose."""
    return [list(r) for r in zip(*X)]


def matmul(X, Y):
    """Matrix product."""
    return [[sum(X[a][t] * Y[t][b] for t in range(len(Y))) for b in range(len(Y[0]))] for a in range(len(X))]


def rref(M):
    """Reduced row echelon form (nonzero rows) and pivot columns."""
    R = [list(r) for r in M]
    rows = len(R)
    cols = len(R[0])
    piv = []
    r = 0
    for c in range(cols):
        p = None
        for i in range(r, rows):
            if R[i][c] != 0:
                p = i
                break
        if p is None:
            continue
        R[r], R[p] = R[p], R[r]
        pv = R[r][c]
        R[r] = [v / pv for v in R[r]]
        for i in range(rows):
            if i != r and R[i][c] != 0:
                f = R[i][c]
                R[i] = [a - f * b for a, b in zip(R[i], R[r])]
        piv.append(c)
        r += 1
        if r == rows:
            break
    return R[:r], piv


def nullspace(M):
    """Basis of the null space (list of vectors)."""
    R, piv = rref(M)
    cols = len(M[0])
    out = []
    for fc in range(cols):
        if fc in piv:
            continue
        v = [Fraction(0)] * cols
        v[fc] = Fraction(1)
        for i, pc in enumerate(piv):
            v[pc] = -R[i][fc]
        out.append(v)
    return out


def column(X, j):
    """Column j as a list."""
    return [X[r][j] for r in range(len(X))]


def colspace(M):
    """Basis of the column space, as the columns of a matrix."""
    R, piv = rref(transpose(M))
    return transpose(R)


def intersect(Ua, Ub):
    """Basis (columns) of the intersection of two column spaces."""
    na = len(Ua[0])
    C = [list(ra) + [-x for x in rb] for ra, rb in zip(Ua, Ub)]
    vecs = []
    for v in nullspace(C):
        vecs.append([sum(Ua[r][c] * v[c] for c in range(na)) for r in range(len(Ua))])
    return transpose(vecs)


def coords(Ub, v):
    """Coordinates of v in the column basis Ub (v must lie in the span)."""
    n = len(Ub[0])
    R, piv = rref([list(Ub[r]) + [v[r]] for r in range(len(Ub))])
    if n in piv:
        raise ValueError("vector not in span")
    c = [Fraction(0)] * n
    for i, pc in enumerate(piv):
        c[pc] = R[i][n]
    return c


def inverse(C):
    """Inverse of a square matrix."""
    n = len(C)
    R, piv = rref([list(C[i]) + [Fraction(1) if i == j else Fraction(0) for j in range(n)] for i in range(n)])
    return [row[n:] for row in R]


def inv2(X):
    """Inverse of a 2 x 2 matrix."""
    d = X[0][0] * X[1][1] - X[0][1] * X[1][0]
    return [[X[1][1] / d, -X[0][1] / d], [-X[1][0] / d, X[0][0] / d]]


def structure():
    """Structural matrices A_i and the matrices M_ij of the tetrahedron, k = 2."""
    low = monomials(0) + monomials(1) + monomials(2)
    top = monomials(3)
    deg2 = monomials(2)
    H_low = [[moment(add(a, b)) for b in low] for a in low]
    cross = [[moment(add(a, b)) for b in low] for a in top]
    H_inv = inverse(H_low)
    Z = matmul(cross, H_inv)
    H = []
    for i in range(len(top)):
        H.append([moment(add(top[i], top[j])) - sum(Z[i][t] * cross[j][t] for t in range(len(low))) for j in range(len(top))])
    unit = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    A = []
    for i in range(3):
        A.append([[Fraction(1) if add(deg2[j], unit[i]) == top[c] else Fraction(0) for c in range(len(top))] for j in range(len(deg2))])
    M = {}
    for i, j in ((0, 1), (0, 2), (1, 2)):
        P = matmul(matmul(A[i], H), transpose(A[j]))
        M[(i, j)] = [[P[a][b] - P[b][a] for b in range(len(P))] for a in range(len(P))]
    return A, M


def two_form(U, Mp):
    """Om(x, y) for x, y in U, the 2-form inverse to M = U C U^T."""
    n = len(U[0])
    Y = [coords(U, column(Mp, j)) for j in range(len(Mp))]
    C = [coords(U, [Y[j][a] for j in range(len(Mp))]) for a in range(n)]
    Ci = inverse(C)

    def om(x, y):
        a = coords(U, x)
        b = coords(U, y)
        return sum(a[r] * Ci[r][s] * b[s] for r in range(n) for s in range(n))

    return om


def decide(M, verbose):
    """Steps 3 to 5; returns the 2 x 2 matrix Q."""
    U = {p: colspace(M[p]) for p in M}
    for p in U:
        if len(U[p][0]) != 4:
            raise ValueError("rank M_ij is not 4")
    W = {0: intersect(U[(0, 1)], U[(0, 2)]), 1: intersect(U[(0, 1)], U[(1, 2)]), 2: intersect(U[(0, 2)], U[(1, 2)])}
    Tb = intersect(W[0], U[(1, 2)])
    if verbose:
        print("dim W_i:", [len(W[i][0]) for i in range(3)], " dim T:", len(Tb[0]))
    if any(len(W[i][0]) != 3 for i in range(3)) or len(Tb[0]) != 2:
        raise ValueError("subspace dimensions differ from the case treated here")
    om = {p: two_form(U[p], M[p]) for p in M}
    t1 = column(Tb, 0)
    t2 = column(Tb, 1)
    pairs = {0: ((0, 1), (0, 2)), 1: ((0, 1), (1, 2)), 2: ((0, 2), (1, 2))}
    S = {}
    for i, (p, q) in pairs.items():
        w = None
        for j in range(3):
            v = column(W[i], j)
            if len(rref([[t1[r], t2[r], v[r]] for r in range(6)])[1]) == 3:
                w = v
                break
        cp = om[p](t1, t2)
        cq = om[q](t1, t2)
        fa = cq * om[p](t1, w) - cp * om[q](t1, w)
        fb = cq * om[p](t2, w) - cp * om[q](t2, w)
        if fa == 0 and fb == 0:
            raise ValueError("line t_i not unique")
        t = [-fb * x + fa * y for x, y in zip(t1, t2)]
        v = t2 if fa == 0 else t1
        if om[p](t, v) == 0:
            raise ValueError("l_i not unique")
        lam = -om[p](t, w) / om[p](t, v)
        s2 = [x + lam * y for x, y in zip(w, v)]
        if om[p](t, s2) != 0 or om[q](t, s2) != 0:
            raise ValueError("Lagrangian conditions not satisfied")
        S[i] = [[t[r], s2[r]] for r in range(6)]
    P = {}
    for (i, j), Mij in M.items():
        rows = []
        for a in range(6):
            for b in range(6):
                coef = [S[i][a][x] * S[j][b][y] - S[j][a][y] * S[i][b][x] for x in range(2) for y in range(2)]
                rows.append(coef + [Mij[a][b]])
        R, piv = rref(rows)
        if 4 in piv:
            raise ValueError("no P_ij: no rule (stronger contradiction)")
        P[(i, j)] = [[R[0][4], R[1][4]], [R[2][4], R[3][4]]]
    return matmul(inv2(matmul(P[(0, 1)], inv2(transpose(P[(1, 2)])))), P[(0, 2)])


def selftest(A):
    """Artificial data with known G and signature: Q must be symmetric with that signature."""
    for seed, sig in ((1, (1, 1)), (2, (1, -1)), (3, (-1, -1))):
        random.seed(seed)
        y = [Fraction(random.randint(-3, 3)) for _ in range(6)]
        cons = [[sum(y[r] * A[i][r][c] for r in range(6)) for c in range(10)] for i in range(3)]
        basis = nullspace(cons)
        coef = [[Fraction(random.randint(-3, 3)) for _ in basis] for _ in range(2)]
        G = [[sum(coef[c][q] * basis[q][r] for q in range(len(basis))) for c in range(2)] for r in range(10)]
        M = {}
        for i, j in ((0, 1), (0, 2), (1, 2)):
            Bi = matmul(A[i], G)
            Bj = matmul(A[j], G)
            X = matmul([[Bi[r][c] * sig[c] for c in range(2)] for r in range(6)], transpose(Bj))
            M[(i, j)] = [[X[a][b] - X[b][a] for b in range(6)] for a in range(6)]
        Q = decide(M, False)
        det = Q[0][0] * Q[1][1] - Q[0][1] * Q[1][0]
        tr = Q[0][0] + Q[1][1]
        if det > 0:
            found = (1, 1) if tr > 0 else (-1, -1)
        else:
            found = (1, -1)
        ok = Q[0][1] == Q[1][0] and sorted(found) == sorted(sig)
        print("selftest signature", sig, ": Q symmetric", Q[0][1] == Q[1][0], ", signature recovered", ok)


def main():
    A, M = structure()
    if "--selftest" in sys.argv:
        selftest(A)
        return
    print("rank M_ij:", [len(colspace(M[p])[0]) for p in sorted(M)])
    Q = decide(M, True)
    print("Q =", [[str(x) for x in r] for r in Q])
    if Q[0][1] != Q[1][0]:
        print("Q is not symmetric => no degree-5 rule with 12 nodes (any real weights); N >= 13")
    else:
        print("Q is symmetric: this test does not exclude 12 nodes")


if __name__ == "__main__":
    main()

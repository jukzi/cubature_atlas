"""Tetrahedron, degree 3: no rule with 5 nodes and positive weights (exact proof).

Reference tetrahedron (0,0,0), (1,0,0), (0,1,0), (0,0,1).  Nodes may lie
anywhere; only the signs of the weights matter.

Argument (notation of moeller_bound.py, here with k = 1):
  * Q_j: orthogonal polynomials of degree 1, R_alpha = x^alpha - proj_{P_1} x^alpha
    for the six monomials of degree 2 (orthogonal polynomials of degree 2).
  * x_i Q_j = R_{lead(Q_j) + e_i} + (degree <= 1), so the structural matrices
    A_i (3 x 6) are 0/1 shift matrices in these bases.
  * H = true Gram matrix of R, K = discrete Gram matrix sum_m w_m R(x_m) R(x_m)^T.
    For every pair i < j:  A_i K A_j^T - A_j K A_i^T = A_i H A_j^T - A_j H A_i^T = M_ij.
  * The discrete Gram matrix of P_2 has rank dim P_1 + rank K = 4 + rank K <= N.
    Moeller's bound gives rank K >= 1, so N = 5 forces K = s g g^T, s = +1 or -1.
    Positive weights make K positive semidefinite, so s = +1.
  * With a_i = A_i g:  s (a_i x a_j) = m_ij, where m_ij is the vector of the
    skew-symmetric 3 x 3 matrix M_ij.  Hence a_1 = l_1 u_1 with u_1 = m_12 x m_13,
    a_2 = l_2 u_2 with u_2 = m_12 x m_23, a_3 = l_3 u_3 with u_3 = m_13 x m_23, and
    s l_i l_j = c_ij with m_ij = c_ij (u_i x u_j).  Multiplying the three equations:
    (l_1 l_2 l_3)^2 = s c_12 c_13 c_23.  A real solution needs s c_12 c_13 c_23 > 0.
  * The exact rational values below give c_12 c_13 c_23 < 0, so s = -1: impossible
    for positive weights.  (s = -1 is the known 5-node rule with a negative weight.)

Consequence: a degree-3 rule with positive weights on the tetrahedron needs at
least 6 nodes, so the 6-node rule of the atlas is minimal among positive rules.

All arithmetic is exact (fractions).  Usage:  python tetra3_positive.py
"""
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
    """Integral of x^a y^b z^c over the reference tetrahedron: a! b! c! / (a+b+c+3)!."""
    return Fraction(factorial(e[0]) * factorial(e[1]) * factorial(e[2]), factorial(sum(e) + 3))


def solve(A, B):
    """Solve A X = B exactly; B is a list of right-hand sides, result a list of solutions."""
    n = len(A)
    T = []
    for i in range(n):
        T.append(list(A[i]) + [rhs[i] for rhs in B])
    for c in range(n):
        p = c
        while T[p][c] == 0:
            p += 1
        T[c], T[p] = T[p], T[c]
        piv = T[c][c]
        T[c] = [v / piv for v in T[c]]
        for i in range(n):
            if i != c and T[i][c] != 0:
                f = T[i][c]
                T[i] = [a - f * b for a, b in zip(T[i], T[c])]
    return [[T[i][n + c] for i in range(n)] for c in range(len(B))]


def cross(u, v):
    """Cross product."""
    return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]


def skew_vector(S):
    """Vector m of a skew-symmetric 3x3 matrix with S = a b^T - b a^T  <->  m = a x b."""
    return [S[1][2], S[2][0], S[0][1]]


def ratio(v, w):
    """The number c with v = c w (raises if v and w are not parallel)."""
    c = None
    for x, y in zip(v, w):
        if y == 0:
            if x != 0:
                raise ValueError("not parallel")
            continue
        q = x / y
        if c is not None and c != q:
            raise ValueError("not parallel")
        c = q
    return c


def main():
    low = monomials(0) + monomials(1)
    top = monomials(2)
    deg1 = monomials(1)
    unit = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    # Gram matrix of P_1 and inner products of degree-2 monomials with P_1.
    H_low = [[moment(add(a, b)) for b in low] for a in low]
    cross_ip = [[moment(add(a, b)) for b in low] for a in top]
    Z = solve(H_low, cross_ip)
    # Gram matrix H of R (Schur complement).
    H = []
    for i in range(6):
        row = []
        for j in range(6):
            proj = sum(cross_ip[i][t] * Z[j][t] for t in range(len(low)))
            row.append(moment(add(top[i], top[j])) - proj)
        H.append(row)
    # Structural matrices A_i (3 x 6): x_i Q_j -> R_{lead(Q_j) + e_i}.
    A = []
    for i in range(3):
        Ai = []
        for j in range(3):
            target = add(deg1[j], unit[i])
            Ai.append([Fraction(1) if top[c] == target else Fraction(0) for c in range(6)])
        A.append(Ai)
    # m_ij from M_ij = A_i H A_j^T - A_j H A_i^T.
    m = {}
    for i, j in ((0, 1), (0, 2), (1, 2)):
        P = [[sum(A[i][a][s] * H[s][t] * A[j][b][t] for s in range(6) for t in range(6)) for b in range(3)] for a in range(3)]
        S = [[P[a][b] - P[b][a] for b in range(3)] for a in range(3)]
        m[(i, j)] = skew_vector(S)
        print("m_%d%d =" % (i + 1, j + 1), [str(x) for x in m[(i, j)]])
    u1 = cross(m[(0, 1)], m[(0, 2)])
    u2 = cross(m[(0, 1)], m[(1, 2)])
    u3 = cross(m[(0, 2)], m[(1, 2)])
    c12 = ratio(m[(0, 1)], cross(u1, u2))
    c13 = ratio(m[(0, 2)], cross(u1, u3))
    c23 = ratio(m[(1, 2)], cross(u2, u3))
    product = c12 * c13 * c23
    print("c_12 =", c12, " c_13 =", c13, " c_23 =", c23)
    print("c_12 c_13 c_23 =", product)
    if product < 0:
        print("=> s = -1 only: no 5-node degree-3 rule with positive weights; positive rules need >= 6 nodes")
    else:
        print("=> s = +1 possible: this test does not exclude 5 positive nodes")


if __name__ == "__main__":
    main()

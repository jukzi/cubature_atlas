"""Moeller-type lower bounds for odd degrees in 2D and 3D, exact modulo a prime.

Domains: triangle, square, tetra (tetrahedron), cube (reference domains of the atlas).

Statement.  Let a cubature rule with N nodes (any real weights) be exact for
all polynomials of degree <= 2k+1.  Let P_k be the orthonormal polynomials of
degree k (a vector of length r_k) and A_{k,i} the structural matrices of the
three-term relation  x_i P_k = A_{k,i} P_{k+1} + B_{k,i} P_k + A_{k-1,i}^T P_{k-1}.
Put  M_ij = A_{k,i} A_{k,j}^T - A_{k,j} A_{k,i}^T  (skew-symmetric, r_k x r_k).  Then

    N >= dim P_k + rho,
    rho >= ceil(rank(M_ij) / 2)               for every pair i < j,
    rho >= ceil(rank([M_12 | M_13 | M_23]) / 3)   (3D).

For d = 2 this is Moeller's bound in the form of Y. Xu.

Proof (short).  Let <f,g>_N = sum w_m f(x_m) g(x_m).  It equals the integral
<f,g> when deg f + deg g <= 2k+1.  The Gram matrix of the orthonormal basis of
P_{k+1} in <.,.>_N is block diagonal: identity on P_0..P_k, the block K of
P_{k+1}, and zero coupling (degree <= 2k+1).  Its rank is at most N, so
N >= dim P_k + rank K; put rho = rank K.  The matrix <x_i P_k, x_j P_k^T>_N equals
sum w x_i x_j P_k P_k^T, so it is unchanged by swapping i and j.  Inserting the
three-term relation, all terms except A_i K A_j^T are exact, and the same
identity holds for the integral (K replaced by the identity).  Subtracting:
A_i K A_j^T - A_j K A_i^T = M_ij.  With K = F F^T, rank F = rho, the column space of
M_ij lies in col(A_i F) + col(A_j F) (dimension <= 2 rho), and the column space of
all M_ij together lies in col(A_1 F) + col(A_2 F) + col(A_3 F) (dimension <= 3 rho).

Computation.  Any basis Q of the orthogonal polynomials of degree k gives
matrices congruent to M_ij (Q = T P_k, T invertible), with the same ranks;
the concatenation [M_12|M_13|M_23] changes by T on the left and diag(T^T, T^T, T^T)
on the right, also rank preserving.  With proj_V(f) = f - proj_{P_k}(f) for
deg f = k+1:  (A_i A_j^T)[a][b] -> < proj_V(x_i Q_a), proj_V(x_j Q_b) >.

All quantities are rational.  They are computed in the field Z/pZ with a prime
p larger than every denominator factor; the reduction mod p of a rational
matrix has rank <= its rational rank.  So every printed rank is a proven lower
bound for the true rank, and the bound is rigorous.  If a Gram matrix is
singular mod p the program stops (choose another prime).  When the computed
rank already equals the maximum possible (r_k, rounded down to even), it is
the exact rank.

Usage:  python moeller_bound.py DOMAIN [max_degree]
        python moeller_bound.py tetra 29
"""
import sys
import time
import numpy as np

# Prime below 2^31: products of two residues fit into int64.
PRIME = 2147483629


def monomials(dim, deg):
    """Exponent tuples of total degree <= deg, ordered by total degree."""
    out = []
    for e in range(deg + 1):
        out.extend(exact_degree(dim, e))
    return out


def exact_degree(dim, e):
    """Exponent tuples of total degree exactly e."""
    if dim == 1:
        return [(e,)]
    out = []
    for a in range(e, -1, -1):
        for rest in exact_degree(dim - 1, e - a):
            out.append((a,) + rest)
    return out


def inv(a):
    """Inverse modulo PRIME."""
    return pow(int(a) % PRIME, PRIME - 2, PRIME)


def fact(n):
    """n! modulo PRIME."""
    r = 1
    for i in range(2, n + 1):
        r = r * i % PRIME
    return r


MOMENT_CACHE = {}


def moment(domain, e):
    """Cached integral of a monomial (see moment_raw)."""
    key = (domain, e)
    if key not in MOMENT_CACHE:
        MOMENT_CACHE[key] = moment_raw(domain, e)
    return MOMENT_CACHE[key]


def moment_raw(domain, e):
    """Integral of the monomial with exponents e over the reference domain, mod PRIME."""
    if domain in ("triangle", "tetra"):
        # Simplex: prod(e_i!) / (|e| + d)!.
        num = 1
        for a in e:
            num = num * fact(a) % PRIME
        return num * inv(fact(sum(e) + len(e))) % PRIME
    # Box [-1,1]^d: prod 2/(e_i+1) for even e_i, 0 otherwise.
    r = 1
    for a in e:
        if a % 2 == 1:
            return 0
        r = r * 2 * inv(a + 1) % PRIME
    return r


def matmul(A, B):
    """A @ B modulo PRIME without int64 overflow (B split into 16-bit halves)."""
    lo = B & 0xFFFF
    hi = B >> 16
    r_lo = (A @ lo) % PRIME
    r_hi = (A @ hi) % PRIME
    return (r_lo + (r_hi << 16)) % PRIME


def solve(A, B):
    """Solve A X = B modulo PRIME (Gauss-Jordan); stops if A is singular mod PRIME."""
    n = A.shape[0]
    T = np.concatenate([A % PRIME, B % PRIME], axis=1).astype(np.int64)
    for col in range(n):
        nz = np.nonzero(T[col:, col])[0]
        if len(nz) == 0:
            sys.exit("Gram matrix singular modulo the prime; choose another PRIME")
        piv = col + nz[0]
        if piv != col:
            T[[col, piv]] = T[[piv, col]]
        T[col] = T[col] * inv(T[col, col]) % PRIME
        f = T[:, col].copy()
        f[col] = 0
        T = (T - np.outer(f, T[col]) % PRIME) % PRIME
    return T[:, n:]


def rank(M):
    """Rank modulo PRIME (a lower bound for the rank over the rationals)."""
    T = M.copy() % PRIME
    rows, cols = T.shape
    r = 0
    for col in range(cols):
        if r == rows:
            break
        nz = np.nonzero(T[r:, col])[0]
        if len(nz) == 0:
            continue
        piv = r + nz[0]
        if piv != r:
            T[[r, piv]] = T[[piv, r]]
        T[r] = T[r] * inv(T[r, col]) % PRIME
        f = T[:, col].copy()
        f[r] = 0
        T = (T - np.outer(f, T[r]) % PRIME) % PRIME
        r += 1
    return r


def structural_ranks(domain, k):
    """Ranks of M_ij for all pairs and of their concatenation."""
    dim = 2 if domain in ("triangle", "square") else 3
    basis = monomials(dim, k + 1)
    index = {m: i for i, m in enumerate(basis)}
    size = len(basis)
    n_km1 = len(monomials(dim, k - 1))
    n_k = len(monomials(dim, k))
    # Gram matrix of the monomials up to degree k+1.
    H = np.zeros((size, size), dtype=np.int64)
    for i, a in enumerate(basis):
        for j in range(i, size):
            b = basis[j]
            v = moment(domain, tuple(x + y for x, y in zip(a, b)))
            H[i, j] = v
            H[j, i] = v
    # Orthogonal polynomials of degree k: monomial minus projection onto P_{k-1}.
    deg_k = list(range(n_km1, n_k))
    C = solve(H[:n_km1, :n_km1], H[:n_km1, deg_k])
    Q = np.zeros((size, len(deg_k)), dtype=np.int64)
    for c, i in enumerate(deg_k):
        Q[i, c] = 1
    Q[:n_km1, :] = (-C) % PRIME
    # Projection onto V_{k+1}: R_i = x_i Q - proj_{P_k}(x_i Q).
    R = []
    for axis in range(dim):
        X = np.zeros((size, len(deg_k)), dtype=np.int64)
        for i in range(n_k):
            m = list(basis[i])
            m[axis] += 1
            X[index[tuple(m)], :] = Q[i, :]
        coef = solve(H[:n_k, :n_k], matmul(H[:n_k, :], X))
        Ri = X.copy()
        Ri[:n_k, :] = (Ri[:n_k, :] - coef) % PRIME
        R.append(Ri)
    # Skew-symmetric matrices M_ij = G_ij - G_ij^T with G_ij = R_i^T H R_j.
    ranks = {}
    blocks = []
    for i in range(dim):
        HRi = matmul(H, R[i])
        for j in range(i + 1, dim):
            G = matmul(np.ascontiguousarray(R[j].T), HRi).T % PRIME
            M = (G - G.T) % PRIME
            ranks[(i, j)] = rank(M)
            blocks.append(M)
    total = rank(np.concatenate(blocks, axis=1)) if dim == 3 else None
    return ranks, total, len(deg_k)


def main():
    domain = sys.argv[1] if len(sys.argv) > 1 else "tetra"
    max_degree = int(sys.argv[2]) if len(sys.argv) > 2 else 29
    dim = 2 if domain in ("triangle", "square") else 3
    print("%s: degree 2k+1, r_k = dim of degree-k orthogonal polynomials" % domain)
    print("degree  dim P_k   r_k  pair ranks        concat  rho  bound   time")
    for n in range(3, max_degree + 1, 2):
        k = (n - 1) // 2
        t0 = time.time()
        ranks, total, r_k = structural_ranks(domain, k)
        rho = max((r + 1) // 2 for r in ranks.values())
        if total is not None:
            rho = max(rho, (total + 2) // 3)
        dim_pk = len(monomials(dim, k))
        pair_text = " ".join(str(ranks[p]) for p in sorted(ranks))
        total_text = "-" if total is None else str(total)
        print("%6d  %7d  %4d  %-16s  %6s  %3d  %5d  %5.1fs" % (n, dim_pk, r_k, pair_text, total_text, rho, dim_pk + rho, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

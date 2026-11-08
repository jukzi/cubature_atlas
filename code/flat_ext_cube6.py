"""Open question: is there a degree-6 cubature rule for the cube [-1,1]^3 with only 20 = dim P3 nodes
(positive weights, nodes anywhere in R^3)?

Such a rule exists exactly if the moment matrix M(3) of the cube (positive definite, 20 x 20) has a flat
extension M(4) (Curto-Fialkow).  The unknowns are the moments of the rule of degree 7 (36 numbers y_m) and
degree 8 (45 numbers).  Flatness means, for all degree-4 monomials a, b:

    L(x^(a+b)) = sum over g, d in P3 of L(x^(g+a)) * Minv[g, d] * L(x^(d+b)),

where L agrees with the cube integral up to degree 6.  Eliminating the degree-8 moments (all pairs a, b with
the same sum a + b must give the same value) leaves 75 quadratic equations in the 36 degree-7 moments.

This program builds these equations exactly (rational arithmetic, pure Python), checks their number and
their rank, and writes them in the input format of msolve (first line: variables, second line: 0 for the
rational numbers, then the polynomials).  A Groebner basis {1} would prove that no 20-node rule exists,
so that N_PO(6) = 21 is minimal.

Findings so far (see the open question in Table 1 of the atlas):
  - numerically, the system has no real and no complex solution (all searches stop at residual 0.187);
  - no Nullstellensatz certificate with multipliers of degree <= 2 exists (checked modulo primes, with the
    cube symmetry), and none of the form sum g_k f_k = q(y) + a with q positive semidefinite either.

Usage: python3 flat_ext_cube6.py [output.ms]
"""
import sys
import itertools
from fractions import Fraction


def monomials(deg):
    # all exponent triples of total degree deg
    out = []
    for a in range(deg + 1):
        for b in range(deg + 1 - a):
            out.append((a, b, deg - a - b))
    return out


def moment_1d(k):
    # moment of x^k for the uniform probability measure on [-1, 1]
    if k % 2 == 1:
        return Fraction(0)
    return Fraction(1, k + 1)


def moment(m):
    # moment of x^m for the uniform probability measure on the cube
    return moment_1d(m[0]) * moment_1d(m[1]) * moment_1d(m[2])


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def inverse(matrix):
    # exact inverse by Gauss-Jordan elimination
    n = len(matrix)
    aug = []
    for i in range(n):
        row = list(matrix[i])
        for j in range(n):
            row.append(Fraction(1) if i == j else Fraction(0))
        aug.append(row)
    for c in range(n):
        p = c
        while aug[p][c] == 0:
            p += 1
        aug[c], aug[p] = aug[p], aug[c]
        inv = 1 / aug[c][c]
        aug[c] = [v * inv for v in aug[c]]
        for r in range(n):
            if r != c and aug[r][c] != 0:
                f = aug[r][c]
                aug[r] = [x - f * y for x, y in zip(aug[r], aug[c])]
    return [row[n:] for row in aug]


def rank(rows):
    # exact rank of a list of rows (lists of Fractions)
    rows = [list(r) for r in rows]
    r = 0
    ncols = len(rows[0]) if rows else 0
    for c in range(ncols):
        p = None
        for k in range(r, len(rows)):
            if rows[k][c] != 0:
                p = k
                break
        if p is None:
            continue
        rows[r], rows[p] = rows[p], rows[r]
        for k in range(len(rows)):
            if k != r and rows[k][c] != 0:
                f = rows[k][c] / rows[r][c]
                rows[k] = [x - f * y for x, y in zip(rows[k], rows[r])]
        r += 1
    return r


def build_equations():
    basis3 = []
    for k in range(4):
        basis3.extend(monomials(k))
    deg4 = monomials(4)
    deg7 = monomials(7)
    var = {}
    for m in deg7:
        var[m] = len(var)
    gram = [[moment(add(g, d)) for d in basis3] for g in basis3]
    ginv = inverse(gram)

    def entry(g, a):
        # L(x^(g+a)): a constant up to degree 6, the unknown y_m for degree 7
        m = add(g, a)
        if sum(m) <= 6:
            v = moment(m)
            if v == 0:
                return {}
            return {(): v}
        return {(var[m],): Fraction(1)}

    def flat_rhs(a, b):
        # right-hand side of the flatness condition as a polynomial in y
        poly = {}
        for i, g in enumerate(basis3):
            ea = entry(g, a)
            if not ea:
                continue
            for j, d in enumerate(basis3):
                w = ginv[i][j]
                if w == 0:
                    continue
                eb = entry(d, b)
                for m1, v1 in ea.items():
                    for m2, v2 in eb.items():
                        key = tuple(sorted(m1 + m2))
                        poly[key] = poly.get(key, Fraction(0)) + w * v1 * v2
        return poly

    by_sum = {}
    for a, b in itertools.combinations_with_replacement(deg4, 2):
        by_sum.setdefault(add(a, b), []).append((a, b))
    equations = []
    for s in sorted(by_sum):
        pairs = by_sum[s]
        first = flat_rhs(*pairs[0])
        for pair in pairs[1:]:
            other = flat_rhs(*pair)
            diff = dict(first)
            for m, v in other.items():
                diff[m] = diff.get(m, Fraction(0)) - v
            diff = {m: v for m, v in diff.items() if v != 0}
            if diff:
                equations.append(diff)
    names = ["y%d%d%d" % m for m in deg7]
    return equations, names


def to_integer_string(poly, names):
    # scale to integer coefficients and print in msolve syntax
    lcm = 1
    for v in poly.values():
        d = v.denominator
        a = lcm
        b = d
        while b:
            a, b = b, a % b
        lcm = lcm * d // a
    terms = []
    for m in sorted(poly):
        c = poly[m] * lcm
        mono = "*".join(names[x] for x in m)
        if mono:
            terms.append("%+d*%s" % (c.numerator, mono))
        else:
            terms.append("%+d" % c.numerator)
    return "".join(terms)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "cube_deg6_20nodes.ms"
    equations, names = build_equations()
    monos = sorted({m for e in equations for m in e})
    index = {m: i for i, m in enumerate(monos)}
    rows = []
    for e in equations:
        row = [Fraction(0)] * len(monos)
        for m, v in e.items():
            row[index[m]] = v
        rows.append(row)
    r = rank(rows)
    print("equations:", len(equations), "rank:", r, "unknowns:", len(names))
    lines = [",".join(names), "0", ",\n".join(to_integer_string(e, names) for e in equations)]
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("written:", out)


if __name__ == "__main__":
    main()

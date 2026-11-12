"""Square [-1,1]^2, degree 8: are there rules with 15 = dim P4 nodes, and are all of them outside the square?

A rule of degree 8 with 15 nodes and positive weights exists exactly if the moment matrix M(4) of the square
(positive definite, 15 x 15) has a flat extension M(5) (Curto-Fialkow); its nodes are then the common
eigenvalues of the multiplication matrices and its weights are positive. The unknowns are the moments of the
rule of degree 9 (10 numbers y0..y9, y_a = L(x^a y^(9-a))) and degree 10 (11 numbers). Flatness means, for all
degree-5 monomials a, b:

    L(x^(a+b)) = sum over g, d in P4 of L(x^(g+a)) * Minv[g, d] * L(x^(d+b)),

where L agrees with the integral over the square up to degree 8. Eliminating the degree-10 moments (all pairs
a, b with the same sum a + b must give the same value) leaves 10 quadratic equations in the 10 unknowns.

Numerically (many random starts) the system has exactly 8 real solutions; they are the images of the PO rule
CA-C2-08-15-PO under the 8 symmetries of the square, and that rule has a node at |x| = 1.21 outside the square.
If msolve confirms that there are no further real solutions, then
  - the degree-8 rule with 15 nodes is unique up to symmetry, and
  - no PI rule with 15 nodes exists, so N_PI(8) = 16 is minimal.

This program builds the equations exactly (rational arithmetic, pure Python), checks them at the numerical
solution, and writes them in the input format of msolve (first line: variables, second line: 0 for the
rational numbers, then the polynomials). Real solutions with msolve:  msolve -f square8.ms -o square8.out

Usage: python3 flat_ext_square8.py [output.ms] [rule.json]
"""
import sys
import json
import itertools
from fractions import Fraction


def exact(a, b):
    # integral of x^a y^b over [-1, 1]^2
    if a % 2 or b % 2:
        return Fraction(0)
    return Fraction(4, (a + 1) * (b + 1))


def monomials(deg):
    # exponent pairs of total degree deg, x-power descending
    return [(a, deg - a) for a in range(deg, -1, -1)]


def inverse(M):
    # exact inverse by Gauss-Jordan elimination
    n = len(M)
    A = [list(row) + [Fraction(int(i == j)) for j in range(n)] for i, row in enumerate(M)]
    for c in range(n):
        p = next(r for r in range(c, n) if A[r][c] != 0)
        A[c], A[p] = A[p], A[c]
        piv = A[c][c]
        A[c] = [v / piv for v in A[c]]
        for r in range(n):
            if r != c and A[r][c] != 0:
                f = A[r][c]
                A[r] = [vr - f * vc for vr, vc in zip(A[r], A[c])]
    return [row[n:] for row in A]


# Index sets: P4 rows, degree-5 columns.
mon4 = [m for k in range(5) for m in monomials(k)]
mon5 = monomials(5)
# Known part of the moment matrix and its inverse.
A = [[exact(p[0] + q[0], p[1] + q[1]) for q in mon4] for p in mon4]
Ainv = inverse(A)


def entry(p, q):
    """Entry of B: a polynomial (dict monomial-tuple -> coefficient) of degree <= 1 in y."""
    a = p[0] + q[0]
    b = p[1] + q[1]
    if a + b <= 8:
        return {(): exact(a, b)}
    # degree 9: the unknown y_a
    return {(a,): Fraction(1)}


B = [[entry(p, q) for q in mon5] for p in mon4]


def mul(f, g):
    # product of two polynomials in the dict representation
    out = {}
    for mf, cf in f.items():
        for mg, cg in g.items():
            m = tuple(sorted(mf + mg))
            out[m] = out.get(m, Fraction(0)) + cf * cg
    return out


def add_to(acc, f, c):
    # acc += c * f
    for m, v in f.items():
        acc[m] = acc.get(m, Fraction(0)) + c * v


def C_entry(i, j):
    """(B^T A^-1 B)[i, j] as a polynomial of degree <= 2 in y."""
    acc = {}
    for g in range(15):
        if not any(Ainv[g][d] for d in range(15)):
            continue
        for d in range(15):
            c = Ainv[g][d]
            if c == 0:
                continue
            add_to(acc, mul(B[g][i], B[d][j]), c)
    return {m: v for m, v in acc.items() if v != 0}


# Group the pairs (i <= j) by the degree-10 monomial they represent; differences within a group are the equations.
groups = {}
for i, j in itertools.combinations_with_replacement(range(6), 2):
    s = (mon5[i][0] + mon5[j][0], mon5[i][1] + mon5[j][1])
    groups.setdefault(s, []).append((i, j))
equations = []
for s in sorted(groups):
    prs = groups[s]
    c0 = C_entry(*prs[0])
    for pr in prs[1:]:
        f = dict(C_entry(*pr))
        add_to(f, c0, Fraction(-1))
        f = {m: v for m, v in f.items() if v != 0}
        if f:
            equations.append(f)
print("equations:", len(equations), "unknowns: 10")


def evaluate(f, y):
    # value of a polynomial at the float point y
    tot = 0.0
    for m, c in f.items():
        t = float(c)
        for k in m:
            t *= y[k]
        tot += t
    return tot


# Check at the moments of a rule file (degree-9 moments of the rule), if given.
if len(sys.argv) > 2:
    r = json.load(open(sys.argv[2]))
    y = [sum(w * x ** a * yy ** (9 - a) for (x, yy), w in zip(r["nodes"], r["weights"])) for a in range(10)]
    print("max |equation| at the moments of", r["id"], ":", max(abs(evaluate(f, y)) for f in equations))


def to_msolve(f):
    # polynomial as text for msolve, integer coefficients after clearing denominators
    den = 1
    for v in f.values():
        den = den * v.denominator // __import__("math").gcd(den, v.denominator)
    terms = []
    for m, c in sorted(f.items()):
        k = int(c * den)
        mon = "*".join("y%d" % i for i in m)
        terms.append(("%d*%s" % (k, mon)) if mon else ("%d" % k))
    return "+".join(terms).replace("+-", "-")


out = sys.argv[1] if len(sys.argv) > 1 else "square8.ms"
with open(out, "w") as fh:
    fh.write(",".join("y%d" % i for i in range(10)) + "\n")
    fh.write("0\n")
    fh.write(",\n".join(to_msolve(f) for f in equations) + "\n")
print("written", out)

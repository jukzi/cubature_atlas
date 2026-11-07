"""Fully symmetric positive-interior cubature search on triangle / tetrahedron.

Symmetry group: all permutations of the barycentric coordinates (S3 / S4).
Unknowns: per orbit its free barycentric parameters plus one weight.
Equations: exactness on the invariant polynomials (orthonormal Dubiner basis,
projected onto the invariant subspace with an SVD).
"""
import itertools
import sys
import time
import numpy as np
from scipy.optimize import least_squares
import kubatur as K


def unique_perms(labels):
    seen = {}
    for perm in itertools.permutations(range(len(labels))):
        image = tuple(labels[i] for i in perm)
        if image not in seen:
            seen[image] = perm
    return [np.array(p) for p in seen.values()]


class Sym:
    def __init__(self, name, degree):
        self.name = name
        self.degree = degree
        self.dom = K.Domain(name, degree)
        self.d = self.dom.d
        d = self.d
        if d == 2:
            # (number of params, labels of the barycentric pattern)
            self.types = [(0, [0, 0, 0]), (1, [0, 0, 1]), (2, [0, 1, 2])]
        else:
            self.types = [(0, [0, 0, 0, 0]), (1, [0, 0, 0, 1]), (1, [0, 0, 1, 1]), (2, [0, 0, 1, 2]), (3, [0, 1, 2, 3])]
        self.perms = [unique_perms(t[1]) for t in self.types]
        self.group = len(list(itertools.permutations(range(d + 1))))
        self.build_invariants()

    def bary(self, t, q):
        """Barycentric vector (length d+1) and its Jacobian (d+1, nparams) for orbit type t."""
        d = self.d
        npar = self.types[t][0]
        if d == 2:
            if t == 0:
                return np.full(3, 1 / 3), np.zeros((3, 0))
            if t == 1:
                a = q[0] / 2
                return np.array([a, a, 1 - 2 * a]), np.array([[.5], [.5], [-1.0]])
            P, J = self.dom.to_physical(q[None, :])
            l1, l2 = P[0]
            dl = J[0]
            b = np.array([1 - l1 - l2, l1, l2])
            db = np.vstack([-(dl[0] + dl[1]), dl[0], dl[1]])
            return b, db
        if t == 0:
            return np.full(4, .25), np.zeros((4, 0))
        if t == 1:
            a = q[0] / 3
            return np.array([a, a, a, 1 - 3 * a]), np.array([[1 / 3], [1 / 3], [1 / 3], [-1.0]])
        if t == 2:
            a = q[0] / 2
            return np.array([a, a, .5 - a, .5 - a]), np.array([[.5], [.5], [-.5], [-.5]])
        if t == 3:
            s1, s2 = q
            a = s1 / 2
            b = s2 * (1 - s1)
            c = (1 - s1) * (1 - s2)
            db = np.array([[.5, 0], [.5, 0], [-s2, 1 - s1], [-(1 - s2), -(1 - s1)]])
            return np.array([a, a, b, c]), db
        P, J = self.dom.to_physical(q[None, :])
        l = P[0]
        dl = J[0]
        b = np.concatenate([[1 - l.sum()], l])
        db = np.vstack([-dl.sum(axis=0), dl])
        return b, db

    def build_invariants(self):
        rng = np.random.default_rng(5)
        M = self.dom.M
        count = int(1.6 * M / self.group) + 40
        rows = np.zeros((count, M))
        for i in range(count):
            b = rng.dirichlet(np.ones(self.d + 1))
            images = []
            for perm in itertools.permutations(range(self.d + 1)):
                images.append(b[list(perm)][1:])
            rows[i] = self.dom.values(np.array(images)).sum(axis=0)
        u, s, vt = np.linalg.svd(rows.T, full_matrices=False)
        rank = int((s > 1e-9 * s[0]).sum())
        self.U = u[:, :rank]
        self.rank = rank
        self.target = self.U.T @ self.dom.target

    def orbit_points(self, t, q):
        b, db = self.bary(t, q)
        return np.array([b[p][1:] for p in self.perms[t]]), b, db

    def evaluate(self, counts, x, with_jac):
        """Residual (and Jacobian) for the structure counts (one entry per type)."""
        orbits = []
        for t, c in enumerate(counts):
            orbits += [t] * c
        norb = len(orbits)
        pos = 0
        qs = []
        for t in orbits:
            npar = self.types[t][0]
            qs.append(x[pos:pos + npar])
            pos += npar
        w = x[pos:pos + norb]
        M = self.dom.M
        S = np.zeros((M, norb))
        dS = []
        for o, t in enumerate(orbits):
            P, b, db = self.orbit_points(t, qs[o])
            if with_jac and self.types[t][0] > 0:
                V, D = self.dom.values_and_derivatives(P)
                S[:, o] = V.sum(axis=0)
                cols = []
                for k in range(self.types[t][0]):
                    col = np.zeros(M)
                    for axis in range(self.d):
                        coef = np.array([db[p[axis + 1], k] for p in self.perms[t]])
                        col += (D[axis] * coef[:, None]).sum(axis=0)
                    cols.append(col)
                dS.append(np.array(cols).T)
            else:
                S[:, o] = self.dom.values(P).sum(axis=0)
                dS.append(np.zeros((M, self.types[t][0])))
        r = self.U.T @ (S @ w - self.dom.target)
        if not with_jac:
            return r
        US = self.U.T @ S
        blocks = []
        for o in range(norb):
            if dS[o].shape[1]:
                blocks.append(w[o] * (self.U.T @ dS[o]))
        J = np.hstack(blocks + [US]) if blocks else US
        return r, J

    def unknowns(self, counts):
        return sum(c * (self.types[t][0] + 1) for t, c in enumerate(counts))

    def npoints(self, counts):
        return sum(c * len(self.perms[t]) for t, c in enumerate(counts))

    def try_structure(self, counts, rng, nfev=120):
        orbits = []
        for t, c in enumerate(counts):
            orbits += [t] * c
        npar = sum(self.types[t][0] for t in orbits)
        norb = len(orbits)
        q0 = rng.random(npar)
        w0 = np.full(norb, self.dom.vol / max(self.npoints(counts), 1))
        w0 = w0 * np.array([len(self.perms[t]) for t in orbits])
        lo = np.r_[np.zeros(npar), np.zeros(norb)]
        hi = np.r_[np.ones(npar), np.full(norb, np.inf)]
        x0 = np.r_[q0, w0]

        def fun(x):
            return self.evaluate(counts, x, False)

        def jac(x):
            return self.evaluate(counts, x, True)[1]

        sol = least_squares(fun, x0, jac=jac, bounds=(lo, hi), method="trf", x_scale="jac", max_nfev=nfev, xtol=1e-15, ftol=1e-15, gtol=1e-15)
        return sol

    def expand(self, counts, x):
        orbits = []
        for t, c in enumerate(counts):
            orbits += [t] * c
        pos = 0
        qs = []
        for t in orbits:
            npar = self.types[t][0]
            qs.append(x[pos:pos + npar])
            pos += npar
        w = x[pos:]
        pts = []
        wts = []
        for o, t in enumerate(orbits):
            P, b, db = self.orbit_points(t, qs[o])
            for row in P:
                pts.append(row)
                wts.append(w[o])
        return np.array(pts), np.array(wts)

    def compositions(self, Nmax, slack):
        """Structures with enough unknowns, ascending in the number of points."""
        d = self.d
        sizes = [len(p) for p in self.perms]
        out = []
        rank = self.rank

        def rec(t, left, counts):
            if t == len(sizes):
                c = tuple(counts)
                u = self.unknowns(c)
                n = self.npoints(c)
                if rank <= u <= rank + slack and n <= Nmax and n > 0:
                    out.append((n, c))
                return
            maxc = left // sizes[t]
            if t == 0:
                maxc = min(maxc, 1)
            for k in range(maxc + 1):
                u = self.unknowns(counts + [k] + [0] * (len(sizes) - t - 1))
                if u > rank + slack:
                    break
                rec(t + 1, left - k * sizes[t], counts + [k])

        rec(0, Nmax, [])
        out.sort()
        return [c for n, c in out]


def pi_ok(name, degree, X, W):
    """Monomial check first, then the Newton-Kantorovich certificate; only then is it a solution."""
    err, wmin, outside = K.verify(name, degree, X, W)
    if not (err < 1e-13 and wmin > 0 and outside <= 1e-9):
        return False, err, wmin, outside
    cert = K.certificate(name, degree, X, W)
    print("  certificate: h=%.2e certified=%s inside=%s" % (cert["h"], cert["certified"], cert["inside"]), flush=True)
    return cert["certified"], err, wmin, outside


def search(name, degree, Nmax, slack, starts, deadline, rng):
    """Go downwards: find any rule near Nmax first, then try to remove points."""
    sym = Sym(name, degree)
    comps = sym.compositions(Nmax, slack)
    print("%s deg %d invariants %d candidate structures %d" % (name, degree, sym.rank, len(comps)), flush=True)
    best = None
    limit = Nmax
    while time.time() < deadline:
        level = [c for c in comps if sym.npoints(c) <= limit]
        if not level:
            break
        top = max(sym.npoints(c) for c in level)
        group = [c for c in level if sym.npoints(c) == top]
        found = False
        for rounds in range(starts):
            for counts in group:
                if time.time() > deadline:
                    return best
                sol = sym.try_structure(counts, rng)
                if np.linalg.norm(sol.fun) < 1e-11:
                    X, W = sym.expand(counts, sol.x)
                    keep = W > 1e-14
                    ok, err, wmin, outside = pi_ok(name, degree, X[keep], W[keep])
                    if ok:
                        best = (len(W[keep]), counts, X[keep], W[keep], err, wmin)
                        print("  FOUND N=%d counts=%s err=%.1e wmin=%.1e" % (best[0], counts, err, wmin), flush=True)
                        np.save("sym_%s_d%d_n%d.npy" % (name, degree, best[0]), np.c_[best[2], best[3]])
                        found = True
                        break
            if found:
                break
        limit = (best[0] - 1) if found else top - 1
    return best


if __name__ == "__main__":
    name = sys.argv[1]
    degree = int(sys.argv[2])
    Nmax = int(sys.argv[3])
    slack = int(sys.argv[4])
    starts = int(sys.argv[5])
    seconds = float(sys.argv[6])
    rng = np.random.default_rng(int(sys.argv[7]) if len(sys.argv) > 7 else 1)
    res = search(name, degree, Nmax, slack, starts, time.time() + seconds, rng)
    print("best", res[0] if res else None)

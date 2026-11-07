"""Search for small positive interior cubature rules on the triangle and the tetrahedron.

Domains:
    triangle: (0,0), (1,0), (0,1)
    tetra:    (0,0,0), (1,0,0), (0,1,0), (0,0,1)

Method:
    1. Heuristic start: Lloyd relaxation with a boundary-heavy density.
    2. Levenberg-Marquardt on the moment equations in an orthonormal basis.
    3. Node elimination: delete or merge nodes and re-solve.

Requirements: Python >= 3.9, numpy, scipy.

Usage:
    python kubatur.py find   triangle 12 34 --starts 8
    python kubatur.py reduce triangle 12 --time 600
    python kubatur.py auto   triangle 12 --time 600
    python kubatur.py check  rule_triangle_d12_n34.npy

A rule is stored as rule_<domain>_d<degree>_n<nodes>.npy.
Columns: coordinates, then the weight.
A rule is valid if the moment error is below 1e-12, all weights are positive
and all nodes lie inside the domain.
"""

import argparse
import glob
import itertools
import math
import re
import sys
import time

import numpy as np
from scipy.linalg import LinAlgError
from scipy.linalg import cho_factor
from scipy.linalg import cho_solve
from scipy.spatial import cKDTree

STEP = 1e-30
SCALE_COLUMNS = True
VOLUME = {"triangle": 0.5, "tetra": 1.0 / 6.0}
DIMENSION = {"triangle": 2, "tetra": 3}
EXTRA_START_NODES = {"triangle": 2, "tetra": 12}


def counting_value(name, degree):
    """Number of unknowns per node is d+1, number of moments is C(n+d, d)."""
    d = DIMENSION[name]
    moments = math.comb(degree + d, d)
    return -(-moments // (d + 1))


def jacobi_hom(alpha, nmax, ew, w):
    """Homogenised Jacobi polynomials H_n = w^n P_n^(alpha,0)(ew / w), n = 0..nmax.

    The recurrence also works for complex arguments (used for complex-step derivatives).
    """
    out = [np.ones_like(ew)]
    previous = None
    current = out[0]
    for n in range(nmax):
        if alpha == 0:
            a = (2 * n + 1) / (n + 1.0)
            b = 0.0
            c = n / (n + 1.0)
        else:
            a = (2 * n + alpha + 1) * (2 * n + alpha + 2) / (2.0 * (n + 1) * (n + alpha + 1))
            b = alpha ** 2 * (2 * n + alpha + 1) / (2.0 * (n + 1) * (n + alpha + 1) * (2 * n + alpha))
            c = n * (n + alpha) * (2 * n + alpha + 2) / ((n + 1.0) * (n + alpha + 1) * (2 * n + alpha))
        following = (a * ew + b * w) * current
        if previous is not None:
            following = following - c * w * w * previous
        out.append(following)
        previous = current
        current = following
    return out


class Domain:
    """Orthogonal polynomial basis (Dubiner type) on a simplex."""

    def __init__(self, name, degree):
        self.name = name
        self.degree = degree
        self.d = DIMENSION[name]
        self.vol = VOLUME[name]
        if name == "triangle":
            self.index = [(p, q) for p in range(degree + 1) for q in range(degree + 1 - p)]
        else:
            self.index = [
                (p, q, r)
                for p in range(degree + 1)
                for q in range(degree + 1 - p)
                for r in range(degree + 1 - p - q)
            ]
        self.M = len(self.index)
        self.norm = np.ones(self.M)
        self.target = np.zeros(self.M)
        self.calibrate()

    def raw_basis(self, P):
        """Unnormalised basis values, shape (number of points, M); P may be complex."""
        n = self.degree
        count = P.shape[0]
        V = np.empty((count, self.M), dtype=P.dtype)
        x = P[:, 0]
        y = P[:, 1]
        if self.name == "triangle":
            v = 1 - y
            ew = 2 * x - v
            legendre = [np.ones_like(x)]
            if n >= 1:
                legendre.append(ew * 1.0)
            for p in range(1, n):
                legendre.append(((2 * p + 1) * ew * legendre[p] - p * v * v * legendre[p - 1]) / (p + 1))
            eta = 2 * y - 1
            column = 0
            for p in range(n + 1):
                tail = jacobi_hom(2 * p + 1, n - p, eta, np.ones_like(x))
                for q in range(n - p + 1):
                    V[:, column] = legendre[p] * tail[q]
                    column += 1
            return V
        z = P[:, 2]
        v = 1 - y - z
        ew = 2 * x - v
        legendre = [np.ones_like(x)]
        if n >= 1:
            legendre.append(ew * 1.0)
        for p in range(1, n):
            legendre.append(((2 * p + 1) * ew * legendre[p] - p * v * v * legendre[p - 1]) / (p + 1))
        w2 = 1 - z
        ew2 = 2 * y - w2
        zeta = 2 * z - 1
        column = 0
        for p in range(n + 1):
            middle = jacobi_hom(2 * p + 1, n - p, ew2, w2)
            for q in range(n - p + 1):
                tail = jacobi_hom(2 * p + 2 * q + 2, n - p - q, zeta, np.ones_like(x))
                base = legendre[p] * middle[q]
                for r in range(n - p - q + 1):
                    V[:, column] = base * tail[r]
                    column += 1
        return V

    def calibrate(self):
        """Normalise the basis with a collapsed Gauss-Legendre grid (exact for polynomials).

        Only the diagonal of the Gram matrix and the integrals of the basis are needed.
        The grid is processed in chunks to keep memory small.
        """
        m = self.degree + 3
        u, w = np.polynomial.legendre.leggauss(m)
        u = (u + 1) / 2
        w = w / 2
        if self.d == 2:
            grid_u, grid_v = np.meshgrid(u, u, indexing="ij")
            weights = np.outer(w, w) * (1 - grid_v)
            points = np.c_[(grid_u * (1 - grid_v)).ravel(), grid_v.ravel()]
        else:
            grid_u, grid_v, grid_w = np.meshgrid(u, u, u, indexing="ij")
            weights = w[:, None, None] * w[None, :, None] * w[None, None, :]
            weights = weights * (1 - grid_v) * (1 - grid_w) ** 2
            px = grid_u * (1 - grid_v) * (1 - grid_w)
            py = grid_v * (1 - grid_w)
            points = np.c_[px.ravel(), py.ravel(), grid_w.ravel()]
        weights = weights.ravel()
        diagonal = np.zeros(self.M)
        integral = np.zeros(self.M)
        chunk = max(1, 4000000 // self.M)
        for start in range(0, len(weights), chunk):
            block = self.raw_basis(points[start:start + chunk])
            wb = weights[start:start + chunk]
            diagonal += (block ** 2 * wb[:, None]).sum(axis=0)
            integral += (block * wb[:, None]).sum(axis=0)
        self.norm = 1.0 / np.sqrt(diagonal)
        self.target = integral * self.norm

    def values(self, P):
        return self.raw_basis(P) * self.norm

    def values_and_derivatives(self, P):
        """Basis values and their derivatives with respect to the coordinates (complex step)."""
        V = self.values(P)
        derivatives = []
        for axis in range(self.d):
            shifted = P.astype(complex)
            shifted[:, axis] += 1j * STEP
            derivatives.append(self.raw_basis(shifted).imag / STEP * self.norm)
        return V, derivatives

    def to_physical(self, Q):
        """Collapsed (Duffy) parameters in [0,1]^d -> points and the matrix dp/dq."""
        count = len(Q)
        Jm = np.zeros((count, self.d, self.d))
        if self.d == 2:
            u = Q[:, 0]
            v = Q[:, 1]
            P = np.c_[u * (1 - v), v]
            Jm[:, 0, 0] = 1 - v
            Jm[:, 0, 1] = -u
            Jm[:, 1, 1] = 1
            return P, Jm
        u = Q[:, 0]
        v = Q[:, 1]
        w = Q[:, 2]
        P = np.c_[u * (1 - v) * (1 - w), v * (1 - w), w]
        Jm[:, 0, 0] = (1 - v) * (1 - w)
        Jm[:, 0, 1] = -u * (1 - w)
        Jm[:, 0, 2] = -u * (1 - v)
        Jm[:, 1, 1] = 1 - w
        Jm[:, 1, 2] = -v
        Jm[:, 2, 2] = 1
        return P, Jm

    def to_parameters(self, P):
        """Inverse of to_physical (clipped to the unit cube)."""
        if self.d == 2:
            v = P[:, 1]
            u = P[:, 0] / np.clip(1 - v, 1e-9, None)
            return np.clip(np.c_[u, v], 0, 1)
        x = P[:, 0]
        y = P[:, 1]
        z = P[:, 2]
        v = y / np.clip(1 - z, 1e-9, None)
        u = x / np.clip(1 - y - z, 1e-9, None)
        return np.clip(np.c_[u, v, z], 0, 1)


class Problem:
    """Moment equations sum_i w_i phi_k(p_i) = integral(phi_k) with 0 <= q <= 1 and w >= 0."""

    def __init__(self, domain):
        self.domain = domain
        self.d = domain.d
        self.M = domain.M

    def residual(self, x, N):
        d = self.d
        Q = x[:d * N].reshape(N, d)
        w = x[d * N:(d + 1) * N]
        P, _ = self.domain.to_physical(Q)
        V = self.domain.values(P)
        return w @ V - self.domain.target

    def jacobian(self, x, N):
        d = self.d
        Q = x[:d * N].reshape(N, d)
        w = x[d * N:(d + 1) * N]
        P, Jm = self.domain.to_physical(Q)
        V, dV = self.domain.values_and_derivatives(P)
        J = np.empty((self.M, (d + 1) * N))
        for k in range(d):
            dq = sum(dV[axis] * Jm[:, axis, k][:, None] for axis in range(d))
            J[:, k:d * N:d] = (w[:, None] * dq).T
        J[:, d * N:(d + 1) * N] = V.T
        return J

    def bounds(self, N):
        d = self.d
        lower = np.r_[np.zeros(d * N), np.zeros(N)]
        upper = np.r_[np.ones(d * N), np.full(N, 50.0)]
        return lower, upper

    def solve(self, x0, N, max_iter=1500, tol=1e-26):
        """Damped Gauss-Newton (Levenberg-Marquardt) with projection onto the bounds."""
        lower, upper = self.bounds(N)
        x = np.clip(x0, lower, upper)
        r = self.residual(x, N)
        cost = 0.5 * r @ r
        mu = 1e-3
        M = self.M
        for _ in range(max_iter):
            if cost < tol:
                break
            J = self.jacobian(x, N)
            gradient = J.T @ r
            at_lower = (x <= lower + 1e-13) & (gradient > 0)
            at_upper = (x >= upper - 1e-13) & (gradient < 0)
            free = ~(at_lower | at_upper)
            Jf = J[:, free]
            if SCALE_COLUMNS:
                scale = 1.0 / np.sqrt((Jf ** 2).sum(axis=0) + 1e-300)
                Jf = Jf * scale
            improved = False
            for _ in range(14):
                try:
                    if Jf.shape[1] >= M:
                        A = Jf @ Jf.T
                        A[np.diag_indices(M)] += mu * (np.trace(A) / M + 1e-30)
                        step = Jf.T @ cho_solve(cho_factor(A, lower=True), -r)
                    else:
                        A = Jf.T @ Jf
                        k = A.shape[0]
                        A[np.diag_indices(k)] += mu * (np.trace(A) / k + 1e-30)
                        step = cho_solve(cho_factor(A, lower=True), -(Jf.T @ r))
                except (LinAlgError, ValueError):
                    mu *= 10
                    continue
                if SCALE_COLUMNS:
                    step = step * scale
                trial = x.copy()
                trial[free] += step
                trial = np.clip(trial, lower, upper)
                r_trial = self.residual(trial, N)
                cost_trial = 0.5 * r_trial @ r_trial
                if cost_trial < cost:
                    x = trial
                    r = r_trial
                    cost = cost_trial
                    mu = max(mu / 4, 1e-15)
                    improved = True
                    break
                mu *= 4
            if not improved:
                break
        return x, cost

    def solve_with_restarts(self, x0, N, max_iter=1500):
        x, cost = self.solve(x0, N, max_iter)
        for _ in range(4):
            if cost < 1e-24 or cost > 1e-4:
                break
            x, cost = self.solve(x, N, max_iter)
        return x, cost

    def points_and_weights(self, x, N):
        d = self.d
        Q = x[:d * N].reshape(N, d)
        P, _ = self.domain.to_physical(Q)
        return P, x[d * N:(d + 1) * N]


def sample_simplex(d, count, rng):
    """Uniform random points in the reference simplex."""
    if d == 2:
        u = rng.random((count, 2))
        flip = u.sum(axis=1) > 1
        u[flip] = 1 - u[flip]
        return u
    u = np.sort(rng.random((count, 3)), axis=1)
    return np.c_[u[:, 0], u[:, 1] - u[:, 0], u[:, 2] - u[:, 1]]


def min_barycentric(S):
    return np.c_[S, 1 - S.sum(axis=1)].min(axis=1)


def heuristic_start(name, N, rng, power=0.5, samples=20000, iterations=15):
    """Nodes roughly equidistributed, denser near the boundary; weights larger near the centre."""
    d = DIMENSION[name]
    vol = VOLUME[name]
    S = sample_simplex(d, samples, rng)
    density = (0.08 + min_barycentric(S)) ** (-power)
    chosen = rng.choice(samples, N, replace=False, p=density / density.sum())
    C = S[chosen].copy()
    for _ in range(iterations):
        owner = cKDTree(C).query(S)[1]
        mass = np.bincount(owner, weights=density, minlength=N)
        for axis in range(d):
            moment = np.bincount(owner, weights=density * S[:, axis], minlength=N)
            C[:, axis] = np.where(mass > 0, moment / np.maximum(mass, 1e-300), C[:, axis])
    owner = cKDTree(C).query(S)[1]
    cell = np.bincount(owner, minlength=N) / samples * vol
    centre_value = 1.0 / (d + 1)
    W = cell * (0.6 + 2.4 * min_barycentric(C) / centre_value)
    W = W * vol / W.sum()
    return C, W


def verify(name, degree, X, W):
    """Independent check with monomials: (max moment error, min weight, outside distance)."""
    d = X.shape[1]
    worst = 0.0
    for e in itertools.product(range(degree + 1), repeat=d):
        if sum(e) > degree:
            continue
        exact = math.prod(math.factorial(k) for k in e) / math.factorial(sum(e) + d)
        value = np.sum(W * np.prod(X ** np.array(e), axis=1))
        worst = max(worst, abs(value - exact))
    outside = max(-X.min(), X.sum(axis=1).max() - 1.0, 0.0)
    return worst, W.min(), outside


def system_jacobian(domain, X, W):
    """Residual and Jacobian of the moment equations in the physical coordinates (nodes and weights)."""
    N, d = X.shape
    V, dV = domain.values_and_derivatives(X)
    residual = W @ V - domain.target
    J = np.empty((domain.M, (d + 1) * N))
    for axis in range(d):
        J[:, axis:d * N:d] = (W[:, None] * dV[axis]).T
    J[:, d * N:] = V.T
    return residual, J


def certificate(name, degree, X, W, directions=24, factor=2.0, seed=0):
    """Newton-Kantorovich certificate for the underdetermined system F(x) = 0 (Ben-Israel).

    sigma = smallest singular value of J(x0), eta = |J^+ F(x0)|, L = Lipschitz constant of J.
    If h = L * eta / sigma^2 <= 1/2, an exact rule exists within distance 2 * eta of x0.
    L is estimated from differences of J in random directions and multiplied by `factor`
    (a floating-point certificate, not an interval proof).
    'inside' tells whether the whole ball keeps all weights positive and all nodes inside the domain.
    """
    domain = Domain(name, degree)
    N, d = X.shape
    x0 = np.r_[X.ravel(), W]
    F0, J0 = system_jacobian(domain, X, W)
    sv = np.linalg.svd(J0, compute_uv=False)
    result = {"residual": float(np.abs(F0).max()), "sigma": float(sv[-1]), "certified": False, "inside": False, "h": float("inf")}
    if J0.shape[0] > J0.shape[1] or sv[-1] <= 1e-9 * sv[0]:
        return result
    eta = float(np.linalg.norm(np.linalg.lstsq(J0, F0, rcond=None)[0]))
    rng = np.random.default_rng(seed)
    rho = 1e-5
    L = 0.0
    for _ in range(directions):
        u = rng.normal(size=len(x0))
        u /= np.linalg.norm(u)
        y = x0 + rho * u
        _, Jy = system_jacobian(domain, y[:d * N].reshape(N, d), y[d * N:])
        L = max(L, np.linalg.norm(Jy - J0, 2) / rho)
    L *= factor
    h = L * eta / sv[-1] ** 2
    radius = 2 * eta
    margin = min(X.min(), (1 - X.sum(axis=1)).min())
    result.update(eta=eta, L=float(L), h=float(h), radius=float(radius))
    result["certified"] = bool(h <= 0.5)
    result["inside"] = bool(h <= 0.5 and W.min() > radius and margin > radius)
    return result


def is_valid(name, degree, X, W, need_certificate=True):
    """A rule counts as a solution only if the monomial check AND the certificate succeed."""
    error, wmin, outside = verify(name, degree, X, W)
    if not (error < 1e-12 and wmin > 0 and outside <= 1e-9):
        return False
    if not need_certificate:
        return True
    cert = certificate(name, degree, X, W)
    print("  certificate: h=%.2e certified=%s inside=%s" % (cert["h"], cert["certified"], cert["inside"]), flush=True)
    return cert["certified"]


def rule_file(name, degree, nodes):
    return "rule_%s_d%d_n%d.npy" % (name, degree, nodes)


def save_rule(name, degree, X, W):
    np.save(rule_file(name, degree, len(W)), np.c_[X, W])


def load_smallest(name, degree):
    """Smallest valid stored rule: (X, W) or None."""
    found = []
    for path in glob.glob("rule_%s_d%d_n*.npy" % (name, degree)):
        nodes = int(re.search(r"_n(\d+)\.npy$", path).group(1))
        found.append((nodes, path))
    for nodes, path in sorted(found):
        A = np.load(path)
        if A.shape[0] != nodes:
            continue
        if is_valid(name, degree, A[:, :-1], A[:, -1]):
            return A[:, :-1], A[:, -1]
    return None


def find(name, degree, N, starts, max_iter=1500):
    """Try several heuristic starts with N nodes; save and return the first valid rule."""
    domain = Domain(name, degree)
    problem = Problem(domain)
    for seed in range(starts):
        rng = np.random.default_rng(1000 + seed)
        t0 = time.time()
        C, W = heuristic_start(name, N, rng)
        x0 = np.r_[domain.to_parameters(C).ravel(), W]
        x, cost = problem.solve_with_restarts(x0, N, max_iter)
        X, W = problem.points_and_weights(x, N)
        keep = W > 1e-13
        X = X[keep]
        W = W[keep]
        print("start %d: cost %.1e, %d nodes, %.0f s" % (seed, cost, len(W), time.time() - t0), flush=True)
        if cost < 1e-24 and is_valid(name, degree, X, W):
            save_rule(name, degree, X, W)
            print("saved", rule_file(name, degree, len(W)), flush=True)
            return X, W
    return None


def reduce_rule(name, degree, X, W, time_limit, max_iter=300, seed=1):
    """Repeatedly delete or merge nodes and re-solve; every smaller valid rule is saved."""
    domain = Domain(name, degree)
    problem = Problem(domain)
    d = domain.d
    rng = np.random.default_rng(seed)
    t0 = time.time()
    Q = domain.to_parameters(X)
    w = W.copy()
    N = len(w)
    failures = 0
    while time.time() - t0 < time_limit and N > 1:
        P, _ = domain.to_physical(Q)
        candidates = []
        for k in np.argsort(w)[:6]:
            candidates.append(("delete", int(k)))
        distance = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
        np.fill_diagonal(distance, 9.0)
        order = np.argsort(distance, axis=None)[:12:2]
        for a, b in zip(*np.unravel_index(order, distance.shape)):
            candidates.append(("merge", (int(a), int(b))))
        for k in rng.choice(N, min(4, N), replace=False):
            candidates.append(("delete", int(k)))
        rng.shuffle(candidates)
        success = False
        for kind, c in candidates:
            if kind == "delete":
                keep = np.delete(np.arange(N), c)
                Qn = Q[keep].copy()
                wn = w[keep].copy()
            else:
                a, b = c
                keep = np.delete(np.arange(N), [a, b])
                merged_weight = w[a] + w[b]
                merged_point = (Q[a] * w[a] + Q[b] * w[b]) / merged_weight
                Qn = np.vstack([Q[keep], merged_point])
                wn = np.r_[w[keep], merged_weight]
            Qn = Qn + rng.normal(0, rng.choice([0.001, 0.004, 0.015]), Qn.shape)
            x0 = np.r_[np.clip(Qn, 0, 1).ravel(), wn]
            x, cost = problem.solve(x0, N - 1, max_iter)
            if cost < 1e-24:
                Xn, Wn = problem.points_and_weights(x, N - 1)
                if is_valid(name, degree, Xn, Wn):
                    save_rule(name, degree, Xn, Wn)
                    Q = x[:d * (N - 1)].reshape(N - 1, d)
                    w = x[d * (N - 1):]
                    N = N - 1
                    success = True
                    failures = 0
                    print("-> %d nodes (%s), %.0f s" % (N, kind, time.time() - t0), flush=True)
                    break
            if time.time() - t0 > time_limit:
                break
        if not success:
            failures += 1
            print("round without success at %d nodes (#%d), %.0f s" % (N, failures, time.time() - t0), flush=True)
    print("finished with", N, "nodes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["find", "reduce", "auto", "check"])
    parser.add_argument("domain", help="triangle, tetra, or a rule file for 'check'")
    parser.add_argument("degree", nargs="?", type=int)
    parser.add_argument("nodes", nargs="?", type=int)
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--time", type=float, default=600.0)
    parser.add_argument("--iterations", type=int, default=1500)
    args = parser.parse_args()
    if args.command == "check":
        match = re.search(r"rule_(\w+)_d(\d+)_n(\d+)\.npy$", args.domain)
        A = np.load(args.domain)
        error, wmin, outside = verify(match.group(1), int(match.group(2)), A[:, :-1], A[:, -1])
        print("nodes %d, moment error %.1e, min weight %.1e, outside %.1e" % (len(A), error, wmin, outside))
        return
    name = args.domain
    degree = args.degree
    print("counting value Z =", counting_value(name, degree), flush=True)
    if args.command == "find":
        if args.nodes is None:
            sys.exit("find needs the number of nodes")
        if find(name, degree, args.nodes, args.starts, args.iterations) is None:
            sys.exit("no valid rule found")
        return
    if args.command == "auto":
        start_nodes = counting_value(name, degree) + EXTRA_START_NODES[name]
        if load_smallest(name, degree) is None:
            if find(name, degree, start_nodes, args.starts, args.iterations) is None:
                sys.exit("no valid start rule found")
    stored = load_smallest(name, degree)
    if stored is None:
        sys.exit("no valid rule file found; run 'find' or 'auto' first")
    reduce_rule(name, degree, stored[0], stored[1], args.time)


if __name__ == "__main__":
    main()

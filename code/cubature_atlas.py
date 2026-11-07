"""Cubature atlas: search and certificate for cubature rules with positive weights.
Domains: interval [-1,1], square [-1,1]^2, cube [-1,1]^3,
         triangle (0,0)(1,0)(0,1), tetra (0,0,0)(1,0,0)(0,1,0)(0,0,1).
Needs only numpy and scipy."""
import numpy as np
from itertools import product
from scipy.optimize import least_squares

DIM = {"interval": 1, "square": 2, "cube": 3, "triangle": 2, "tetra": 3}
VOL = {"interval": 2.0, "square": 4.0, "cube": 8.0, "triangle": 0.5, "tetra": 1 / 6}
BOX = ("interval", "square", "cube")


def indices(name, n):
    """Multi-indices (i,j,k) with i+j+k <= n."""
    d = DIM[name]
    return [t for t in product(range(n + 1), repeat=d) if sum(t) <= n]


def legendre(n, x):
    P = [np.ones_like(x), x]
    for i in range(1, n):
        P.append(((2 * i + 1) * x * P[i] - i * P[i - 1]) / (i + 1))
    return P[: n + 1]


def homLegendre(n, x, s):
    """s^i * P_i((2x-s)/s), without dividing by s."""
    Q = [np.ones_like(x), 2 * x - s]
    for i in range(1, n):
        Q.append(((2 * i + 1) * (2 * x - s) * Q[i] - i * s * s * Q[i - 1]) / (i + 1))
    return Q[: n + 1]


def homJacobi(n, a, y, t):
    """t^j * P_j^(a,0)((2y-t)/t), without dividing by t."""
    R = [np.ones_like(y), (a + 2) * y - t]
    for m in range(1, n):
        an = 2 * (m + 1) * (m + a + 1) * (2 * m + a)
        R.append(((2 * m + a + 1) * (((2 * m + a + 2) * (2 * m + a)) * (2 * y - t) + a * a * t) * R[m]
                  - 2 * (m + a) * m * (2 * m + a + 2) * t * t * R[m - 1]) / an)
    return R[: n + 1]


def raw(name, n, X):
    """Orthogonal polynomials (Legendre products, Dubiner basis), matrix (points x basis). X may be complex."""
    idx = indices(name, n)
    if name in BOX:
        L = [legendre(n, X[:, c]) for c in range(X.shape[1])]
        return np.stack([np.prod([L[c][t[c]] for c in range(len(t))], axis=0) for t in idx], 1)
    if name == "triangle":
        x, y = X[:, 0], X[:, 1]
        Q = homLegendre(n, x, 1 - y)
        Rs = {i: homJacobi(n, 2 * i + 1, y, np.ones_like(y)) for i in range(n + 1)}
        return np.stack([Q[i] * Rs[i][j] for i, j in idx], 1)
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    Q = homLegendre(n, x, 1 - y - z)
    out = []
    for i, j, k in idx:
        R = homJacobi(n, 2 * i + 1, y, 1 - z)[j]
        P = homJacobi(n, 2 * i + 2 * j + 2, z, np.ones_like(z))[k]
        out.append(Q[i] * R * P)
    return np.stack(out, 1)


def reference_rule(name, m):
    """Tensor Gauss rule (Duffy map for simplices), exact up to degree 2m-3."""
    g, w = np.polynomial.legendre.leggauss(m)
    if name in BOX:
        d = DIM[name]
        pts = np.array(list(product(g, repeat=d)))
        wt = np.prod(np.array(list(product(w, repeat=d))), axis=1)
        return pts, wt
    u, wu = (g + 1) / 2, w / 2
    if name == "triangle":
        P = np.array(list(product(u, u))); W = np.array(list(product(wu, wu)))
        return np.c_[P[:, 0] * (1 - P[:, 1]), P[:, 1]], W[:, 0] * W[:, 1] * (1 - P[:, 1])
    P = np.array(list(product(u, u, u))); W = np.array(list(product(wu, wu, wu)))
    a, b, c = P.T
    return np.c_[a * (1 - b) * (1 - c), b * (1 - c), c], W.prod(1) * (1 - b) * (1 - c) ** 2


class Basis:
    """Orthonormal basis phi_k and targets T_k = integral of phi_k."""
    def __init__(self, name, n):
        self.name, self.n, self.idx = name, n, indices(name, n)
        pts, wt = reference_rule(name, n + 4)
        # blockwise (memory: the full points x basis matrix can reach several GB for the cube at degree 30)
        chunk = max(1, 2_000_000 // max(1, len(self.idx)))
        sq = np.zeros(len(self.idx)); T = np.zeros(len(self.idx)); G = np.zeros((len(self.idx), len(self.idx)))
        for a in range(0, len(pts), chunk):
            B = raw(name, n, pts[a:a + chunk]); w = wt[a:a + chunk]
            sq += (B * B * w[:, None]).sum(0); T += (B * w[:, None]).sum(0)
            G += B.T @ (B * w[:, None])
        self.scale = 1 / np.sqrt(sq)
        self.T = T * self.scale
        self.ortho = np.abs(G * self.scale[:, None] * self.scale[None, :] - np.eye(len(self.idx))).max()
        self.parity = np.array([sum(t) % 2 for t in self.idx])

    def __call__(self, X):
        return raw(self.name, self.n, X) * self.scale


def residual(B, X, W, sym=False):
    """Moment residual sum_i w_i phi_k(x_i) - T_k."""
    Phi = B(X)
    r = Phi.T @ W - B.T
    return r[B.parity == 0] if sym else r


def jacobian(B, X, W, sym=False):
    """Jacobian via complex step: columns = coordinates of all points, then weights."""
    N, d = X.shape
    Phi = B(X)
    J = np.empty((Phi.shape[1], N * d + N))
    for c in range(d):
        Xc = X.astype(complex); Xc[:, c] += 1e-30j
        J[:, c:N * d:d] = (B(Xc).imag / 1e-30 * W[:, None]).T
    J[:, N * d:] = Phi.T
    return J


def duffy(name, Q):
    if name in BOX:
        return Q
    if name == "triangle":
        return np.c_[Q[:, 0] * (1 - Q[:, 1]), Q[:, 1]]
    a, b, c = Q.T
    return np.c_[a * (1 - b) * (1 - c), b * (1 - c), c]


def search(name, n, N, tries=50, seed=0, tol=1e-26):
    """Random start + bounded least squares (trf).
    Parameters: points in [-1,1]^d or Duffy parameters in [0,1]^d, weights >= 0."""
    B, d, rng = Basis(name, n), DIM[name], np.random.default_rng(seed)
    lo, hi = (-1.0, 1.0) if name in BOX else (0.0, 1.0)

    def res(x):
        return residual(B, duffy(name, x[:N * d].reshape(N, d)), x[N * d:])

    for _ in range(tries):
        x0 = np.r_[rng.uniform(lo, hi, N * d), np.full(N, VOL[name] / N)]
        r = least_squares(res, x0, bounds=(np.r_[np.full(N * d, lo), np.zeros(N)], np.r_[np.full(N * d, hi), np.full(N, np.inf)]),
                          xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=300)
        if 2 * r.cost < tol:
            return duffy(name, r.x[:N * d].reshape(N, d)), r.x[N * d:]
    return None


def margin(name, X):
    """Distance of the points to the domain boundary (negative = outside)."""
    if name in BOX:
        return (1 - np.abs(X)).min()
    return np.c_[X, 1 - X.sum(1)].min()


def split_symmetric(X, W, tol=1e-8):
    """Detect a centrally symmetric rule: pairs (p,-p) with equal weight plus optional centre.
    Returns (P, w, wc) with wc=None without centre, or None if the rule is not symmetric."""
    used = np.zeros(len(X), bool); P, w, wc = [], [], None
    for i in range(len(X)):
        if used[i]:
            continue
        if np.abs(X[i]).max() < tol:
            used[i] = True; wc = W[i]; continue
        dist = np.abs(X + X[i]).max(1); dist[used] = 9; dist[i] = 9
        j = int(np.argmin(dist))
        if dist[j] > tol or abs(W[j] - W[i]) > tol:
            return None
        used[i] = used[j] = True; P.append(X[i]); w.append(W[i])
    return np.array(P).reshape(-1, X.shape[1]), np.array(w), wc


class System:
    """Moment system F(x)=0 with parameter vector x (points, weights[, centre weight]).
    sym=True: only pairs +-p (even moments); otherwise all points (all moments)."""
    def __init__(self, B, d, sym, center):
        self.B, self.d, self.sym, self.center = B, d, sym, center
        self.rows = (B.parity == 0) if sym else np.ones(len(B.T), bool)
        self.f = 2.0 if sym else 1.0

    def unpack(self, x):
        d = self.d; nc = 1 if self.center else 0
        N = (len(x) - nc) // (d + 1)
        return x[:N * d].reshape(N, d), x[N * d:N * d + N], (x[-1] if nc else 0.0)

    def F(self, x):
        P, w, wc = self.unpack(x)
        r = self.f * (self.B(P).T @ w) - self.B.T
        if self.center:
            r = r + wc * self.B(np.zeros((1, self.d))).ravel()
        return r[self.rows]

    def J(self, x):
        P, w, wc = self.unpack(x); N, d = P.shape
        Phi = self.B(P)
        J = np.empty((len(self.B.T), len(x)))
        for c in range(d):
            Pc = P.astype(complex); Pc[:, c] += 1e-30j
            J[:, c:N * d:d] = (self.B(Pc).imag / 1e-30 * (self.f * w)[:, None]).T
        J[:, N * d:N * d + N] = self.f * Phi.T
        if self.center:
            J[:, -1] = self.B(np.zeros((1, d))).ravel()
        return J[self.rows]


def certificate(name, n, X, W, factor=2.0, seed=0):
    """Newton-Kantorovich certificate (Ben-Israel) for the underdetermined system F(x)=0.
    s = smallest singular value of J(x0) (full row rank), eta = |J^+ F(x0)|, L = Lipschitz constant of J.
    If h = L*eta/s^2 <= 1/2, an exact solution x* exists with |x* - x0| <= 2*eta.
    L is estimated from differences of J in 40 random directions and multiplied by `factor`
    (floating-point certificate, not an interval proof). Also reports whether the ball around x0
    keeps all weights positive and all points inside the domain."""
    B, d = Basis(name, n), DIM[name]
    sp = split_symmetric(X, W) if name in BOX else None
    if sp is not None:
        P, w, wc = sp
        S = System(B, d, True, wc is not None)
        x0 = np.r_[P.ravel(), w] if wc is None else np.r_[P.ravel(), w, wc]
        wmin = min(list(w) + ([wc] if wc is not None else []))
        mg = margin(name, np.vstack([P, -P])) if len(P) else 1.0
    else:
        S = System(B, d, False, False)
        x0 = np.r_[X.ravel(), W]; wmin = W.min(); mg = margin(name, X)
    F0 = S.F(x0); J0 = S.J(x0)
    sv = np.linalg.svd(J0, compute_uv=False)
    rank_ok = J0.shape[0] <= J0.shape[1] and sv[-1] > 1e-9 * sv[0]
    out = dict(M=J0.shape[0], p=J0.shape[1], sym=sp is not None, resid=float(np.abs(F0).max()),
               sigma=float(sv[-1]), cond=float(sv[0] / sv[-1]), rank_ok=bool(rank_ok), wmin=float(wmin), margin=float(mg))
    if not rank_ok:
        out.update(h=float("inf"), radius=float("inf"), certified=False, inside=False); return out
    eta = float(np.linalg.norm(np.linalg.lstsq(J0, F0, rcond=None)[0]))
    rng = np.random.default_rng(seed); rho = 1e-5; L = 0.0
    for _ in range(40):
        u = rng.normal(size=len(x0)); u /= np.linalg.norm(u)
        L = max(L, np.linalg.norm(S.J(x0 + rho * u) - J0, 2) / rho)
    L *= factor
    h = L * eta / sv[-1] ** 2
    r = 2 * eta
    out.update(eta=eta, L=float(L), h=float(h), radius=float(r), certified=bool(h <= 0.5),
               inside=bool(h <= 0.5 and wmin > r and mg > r))
    return out

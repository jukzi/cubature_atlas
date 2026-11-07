"""Quadratur auf Quadrat [-1,1]^2, Dreieck (0,0),(1,0),(0,1) und Tetraeder (0,0,0),(1,0,0),(0,1,0),(0,0,1).
Orthonormale Basis (Legendre-Produkte bzw. Koornwinder/Dubiner), Ableitungen per Complex-Step, Duffy-Abbildung fuer Simplizes."""
import numpy as np, os, time, itertools, math
from numpy.polynomial import legendre as L
from math import factorial
from cs import Prob

H = 1e-30


def jacobi_hom(alpha, nmax, ew, w):
    """H_n = w^n P_n^{(alpha,0)}(eta) mit ew = eta*w, n = 0..nmax (komplexfaehig)."""
    out = [np.ones_like(ew)]
    if nmax == 0:
        return out
    prev = None
    cur = out[0]
    for n in range(nmax):
        if alpha == 0:
            a = (2 * n + 1) / (n + 1.0); b = 0.0; c = n / (n + 1.0)
        else:
            a = (2 * n + alpha + 1) * (2 * n + alpha + 2) / (2.0 * (n + 1) * (n + alpha + 1))
            b = alpha ** 2 * (2 * n + alpha + 1) / (2.0 * (n + 1) * (n + alpha + 1) * (2 * n + alpha))
            c = n * (n + alpha) * (2 * n + alpha + 2) / ((n + 1.0) * (n + alpha + 1) * (2 * n + alpha))
        nxt = (a * ew + b * w) * cur - (c * w * w * prev if prev is not None else 0.0)
        out.append(nxt); prev, cur = cur, nxt
    return out


class Dom:
    def __init__(self, name, deg):
        self.name, self.deg = name, deg
        self.d = 2 if name in ("square", "triangle") else 3
        self.vol = {"square": 4.0, "triangle": 0.5, "tetra": 1.0 / 6}[name]
        if name == "square":
            self.idx = [(i, j) for i in range(deg + 1) for j in range(deg + 1 - i)]
            I, J = np.array(self.idx).T
            self.s = np.sqrt((2 * I + 1) * (2 * J + 1) / 4.0)
            self.parity = (-1) ** (I + J)
            C = np.zeros((deg + 1, deg + 1))
            for i in range(1, deg + 1):
                e = np.zeros(i + 1); e[i] = 1; c = L.legder(e); C[i, :len(c)] = c
            self.C = C
            self.T = np.zeros(len(self.idx)); self.T[0] = 4 * self.s[0]
            self.norm = np.ones(len(self.idx))
        else:
            if name == "triangle":
                self.idx = [(p, q) for p in range(deg + 1) for q in range(deg + 1 - p)]
            else:
                self.idx = [(p, q, r) for p in range(deg + 1) for q in range(deg + 1 - p) for r in range(deg + 1 - p - q)]
            self.parity = None
            self.norm = np.ones(len(self.idx))
            self._calibrate()
        self.M = len(self.idx)

    # ---------- Basis -------------
    def _raw(self, P):
        n = self.deg
        if self.name == "triangle":
            x, y = P[:, 0], P[:, 1]
            v = 1 - y; ew = 2 * x - v
            Q = [np.ones_like(x)]
            if n >= 1: Q.append(ew * 1.0)
            for p in range(1, n):
                Q.append(((2 * p + 1) * ew * Q[p] - p * v * v * Q[p - 1]) / (p + 1))
            eta = 2 * y - 1
            cols = {}
            for p in range(n + 1):
                R = jacobi_hom(2 * p + 1, n - p, eta, np.ones_like(x))
                for q in range(n - p + 1):
                    cols[(p, q)] = Q[p] * R[q]
            return np.stack([cols[k] for k in self.idx], axis=1)
        if self.name == "tetra":
            x, y, z = P[:, 0], P[:, 1], P[:, 2]
            v = 1 - y - z; ew = 2 * x - v
            Q = [np.ones_like(x)]
            if n >= 1: Q.append(ew * 1.0)
            for p in range(1, n):
                Q.append(((2 * p + 1) * ew * Q[p] - p * v * v * Q[p - 1]) / (p + 1))
            w2 = 1 - z; ew2 = 2 * y - w2; zeta = 2 * z - 1
            cols = {}
            for p in range(n + 1):
                Hq = jacobi_hom(2 * p + 1, n - p, ew2, w2)
                for q in range(n - p + 1):
                    S = jacobi_hom(2 * p + 2 * q + 2, n - p - q, zeta, np.ones_like(x))
                    base = Q[p] * Hq[q]
                    for r in range(n - p - q + 1):
                        cols[(p, q, r)] = base * S[r]
            return np.stack([cols[k] for k in self.idx], axis=1)
        raise ValueError

    def basis(self, P):
        """Orthonormale Basis V (N,M) und Ableitungen dV[l] (N,M) nach den physikalischen Koordinaten."""
        if self.name == "square":
            Lx = [L.legvander(P[:, k], self.deg) for k in range(2)]
            D = [Lx[k] @ self.C.T for k in range(2)]
            A = [Lx[k][:, [t[k] for t in self.idx]] for k in range(2)]
            Dd = [D[k][:, [t[k] for t in self.idx]] for k in range(2)]
            V = A[0] * A[1] * self.s
            return V, [Dd[0] * A[1] * self.s, A[0] * Dd[1] * self.s]
        V = self._raw(P) * self.norm
        dV = []
        for l in range(self.d):
            Pc = P.astype(complex).copy(); Pc[:, l] += 1j * H
            dV.append(self._raw(Pc).imag / H * self.norm)
        return V, dV

    def values(self, P):
        if self.name == "square":
            return self.basis(P)[0]
        return self._raw(P) * self.norm

    def _calibrate(self):
        m = self.deg + 3
        u, w = np.polynomial.legendre.leggauss(m); u = (u + 1) / 2; w = w / 2
        if self.d == 2:
            U, Vv = np.meshgrid(u, u, indexing="ij"); WW = np.outer(w, w)
            X = U * (1 - Vv); Y = Vv; jac = (1 - Vv)
            P = np.c_[X.ravel(), Y.ravel()]; wq = (WW * jac).ravel()
        else:
            U, Vv, Ww = np.meshgrid(u, u, u, indexing="ij"); WW = w[:, None, None] * w[None, :, None] * w[None, None, :]
            X = U * (1 - Vv) * (1 - Ww); Y = Vv * (1 - Ww); Z = Ww; jac = (1 - Vv) * (1 - Ww) ** 2
            P = np.c_[X.ravel(), Y.ravel(), Z.ravel()]; wq = (WW * jac).ravel()
        B = self._raw(P)
        G = (B * wq[:, None]).T @ B
        dg = np.sqrt(np.diag(G)); self.norm = 1.0 / dg
        Gn = G * np.outer(self.norm, self.norm)
        off = np.abs(Gn - np.eye(len(dg))).max()
        self.ortho_err = off
        self.T = (B * wq[:, None]).sum(0) * self.norm

    # ---------- Parametrisierung ---------
    def to_phys(self, Q):
        """Parameter q (N,d) -> physikalische Punkte (N,d) und Matrix dp/dq (N,d,d) [l,k]."""
        N = len(Q)
        if self.name == "square":
            Jm = np.zeros((N, 2, 2)); Jm[:, 0, 0] = 1; Jm[:, 1, 1] = 1
            return Q, Jm
        if self.name == "triangle":
            u, v = Q[:, 0], Q[:, 1]
            P = np.c_[u * (1 - v), v]
            Jm = np.zeros((N, 2, 2)); Jm[:, 0, 0] = 1 - v; Jm[:, 0, 1] = -u; Jm[:, 1, 1] = 1
            return P, Jm
        u, v, w = Q[:, 0], Q[:, 1], Q[:, 2]
        P = np.c_[u * (1 - v) * (1 - w), v * (1 - w), w]
        Jm = np.zeros((N, 3, 3))
        Jm[:, 0, 0] = (1 - v) * (1 - w); Jm[:, 0, 1] = -u * (1 - w); Jm[:, 0, 2] = -u * (1 - v)
        Jm[:, 1, 1] = 1 - w; Jm[:, 1, 2] = -v; Jm[:, 2, 2] = 1
        return P, Jm


class ProbG(Prob):
    """Nichtsymmetrisch (cs=False) oder zentralsymmetrisch (nur Quadrat, cs=True; optional Mittelpunkt)."""
    def __init__(self, dom, cs=False, center=False):
        self.dom = dom; self.cs = cs; self.center = center; self.deg = dom.deg; self.d = dom.d
        if cs:
            assert dom.name == "square"
            self.sel = np.where(dom.parity == 1)[0]
        else:
            self.sel = np.arange(dom.M)
        self.M = len(self.sel); self.T = dom.T[self.sel]; self.f = 2.0 if cs else 1.0
        p0 = np.zeros((1, dom.d)); self.V0 = dom.values(p0)[0][self.sel] if center else None

    def nunk(self, N): return (self.d + 1) * N + (1 if self.center else 0)

    def res(self, x, N):
        d = self.d
        Q = x[:d * N].reshape(N, d); w = x[d * N:(d + 1) * N]
        P, _ = self.dom.to_phys(Q)
        V = self.dom.values(P)[:, self.sel]
        r = self.f * (w @ V) - self.T
        if self.center: r = r + x[(d + 1) * N] * self.V0
        return r

    def jac(self, x, N):
        d = self.d
        Q = x[:d * N].reshape(N, d); w = x[d * N:(d + 1) * N]
        P, Jm = self.dom.to_phys(Q)
        V, dV = self.dom.basis(P)
        V = V[:, self.sel]
        J = np.empty((self.M, self.nunk(N)))
        for k in range(d):
            dq = sum(dV[l][:, self.sel] * Jm[:, l, k][:, None] for l in range(d))
            J[:, k:d * N:d] = (self.f * w[:, None] * dq).T
        J[:, d * N:(d + 1) * N] = (self.f * V).T
        if self.center: J[:, (d + 1) * N] = self.V0
        return J

    def bounds(self, N, mode):
        d = self.d
        inside = mode in ("c", "d")
        if self.dom.name == "square":
            PB = (1.0 if inside else 3.0)
            if getattr(self, "PBover", None) is not None: PB = self.PBover
            lo_q, hi_q = -PB, PB
        else:
            PB = 0.0 if inside else -0.7
            if getattr(self, "PBover", None) is not None: PB = self.PBover
            lo_q, hi_q = PB, 1.0 - PB
        WL = 0.0 if mode in ("b", "d") else -50.0
        lo = np.r_[np.full(d * N, lo_q), np.full(N, WL)]; hi = np.r_[np.full(d * N, hi_q), np.full(N, 50.0)]
        if self.center: lo = np.r_[lo, WL]; hi = np.r_[hi, 50.0]
        return lo, hi

    def random_start(self, N, rng):
        d = self.d; npts = 2 * N + int(self.center) if self.cs else N
        if self.dom.name == "square": Q = rng.uniform(-.95, .95, d * N)
        else: Q = rng.uniform(0.02, .98, d * N)
        x = np.r_[Q, np.full(N, self.dom.vol / npts) * rng.uniform(.5, 1.5, N)]
        if self.center: x = np.r_[x, self.dom.vol / npts]
        return x

    def full_points(self, x, N):
        d = self.d
        Q = x[:d * N].reshape(N, d); w = x[d * N:(d + 1) * N]
        P, _ = self.dom.to_phys(Q)
        if not self.cs: return P, w
        X = np.vstack([P, -P]); W = np.r_[w, w]
        if self.center: X = np.vstack([X, np.zeros((1, d))]); W = np.r_[W, x[(d + 1) * N]]
        return X, W


def verify_dom(name, deg, X, W):
    """Unabhaengige Pruefung mit Monomen: (max Fehler, min w, ausserhalb?, Summe w)."""
    d = X.shape[1]
    if name == "square":
        ex = lambda e: np.prod([2.0 / (k + 1) if k % 2 == 0 else 0.0 for k in e])
        inside = np.abs(X).max()
    else:
        ex = lambda e: math.prod([factorial(k) for k in e]) / factorial(sum(e) + d)
        inside = max(-X.min(), X.sum(1).max() - 1, 0.0) if True else 0
    m = 0.0
    for e in itertools.product(range(deg + 1), repeat=d):
        if sum(e) > deg: continue
        v = np.sum(W * np.prod(X ** np.array(e), axis=1))
        m = max(m, abs(v - ex(e)))
    return m, W.min(), inside, W.sum()


def abz(name, n, cs=False):
    from math import comb, ceil
    if name == "tetra": return ceil(comb(n + 3, 3) / 4)
    C = comb(n + 2, 2)
    if name == "triangle" or not cs: return ceil(C / 3)
    m = n if n % 2 == 0 else n - 1
    E = sum(e + 1 for e in range(0, m + 1, 2) for _ in [0])  # Monome gerader Gesamtgrad <= m, 2D: Grad e hat e+1 Monome
    E = sum(e + 1 for e in range(0, m + 1, 2))
    return 2 * ceil(E / 3) if True else 0

if os.environ.get("FASTJAC","0")=="1":
    try:
        import fj; fj.install(ProbG)
    except Exception as _e:
        print("fastjac nicht verfuegbar:",_e)

"""Cubature-Solver Würfel [-1,1]^3: Legendre-Basis, analytische Jacobi-Matrix.
cs=True: zentralsymmetrisch (Paare ±p, optional Mittelpunkt), nur gerade Momente.
Modi: a = beliebig, b = Gewichte >= 0, c = Punkte im Würfel, d = beides."""
import numpy as np, time, sys, os
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from numpy.polynomial import legendre as L
from scipy.optimize import least_squares


class Prob:
    def __init__(self, deg, cs, center=False):
        self.deg, self.cs, self.center = deg, cs, center
        trip = np.array([(i, j, k) for i in range(deg + 1) for j in range(deg + 1 - i)
                         for k in range(deg + 1 - i - j) if (not cs) or ((i + j + k) % 2 == 0)])
        self.trip = trip
        I, J, K = trip.T
        self.s = np.sqrt((2 * I + 1) * (2 * J + 1) * (2 * K + 1) / 8.0)
        self.M = len(trip)
        self.T = np.zeros(self.M)
        self.T[0] = 8 * self.s[0]
        C = np.zeros((deg + 1, deg + 1))
        for i in range(1, deg + 1):
            e = np.zeros(i + 1); e[i] = 1
            c = L.legder(e); C[i, :len(c)] = c
        self.C = C
        p0 = L.legvander(np.zeros(1), deg)[0]
        self.V0 = p0[I] * p0[J] * p0[K] * self.s
        self.f = 2.0 if cs else 1.0

    def nunk(self, N):
        return 4 * N + (1 if self.center else 0)

    def _A(self, P):
        return [L.legvander(P[:, d], self.deg)[:, self.trip[:, d]] for d in range(3)]

    def res(self, x, N):
        P = x[:3 * N].reshape(N, 3); w = x[3 * N:4 * N]
        A = self._A(P)
        V = A[0] * A[1] * A[2] * self.s
        r = self.f * (w @ V) - self.T
        if self.center:
            r = r + x[4 * N] * self.V0
        return r

    def jac(self, x, N):
        P = x[:3 * N].reshape(N, 3); w = x[3 * N:4 * N]
        Ls = [L.legvander(P[:, d], self.deg) for d in range(3)]
        A = [Ls[d][:, self.trip[:, d]] for d in range(3)]
        D = [(Ls[d] @ self.C.T)[:, self.trip[:, d]] for d in range(3)]
        V = A[0] * A[1] * A[2] * self.s
        J = np.empty((self.M, self.nunk(N)))
        for d in range(3):
            o = [a for a in range(3) if a != d]
            dV = D[d] * A[o[0]] * A[o[1]] * self.s
            J[:, d:3 * N:3] = (self.f * w[:, None] * dV).T
        J[:, 3 * N:4 * N] = (self.f * V).T
        if self.center:
            J[:, 4 * N] = self.V0
        return J

    def bounds(self, N, mode):
        PB = 1.0 - 1e-6 if mode in ("c", "d") else 3.0
        if getattr(self, 'PBover', None) is not None:
            PB = self.PBover
        WL = 0.0 if mode in ("b", "d") else -50.0
        lo = np.r_[-PB * np.ones(3 * N), WL * np.ones(N)]
        hi = np.r_[PB * np.ones(3 * N), 50 * np.ones(N)]
        if self.center:
            lo = np.r_[lo, WL]; hi = np.r_[hi, 50.0]
        return lo, hi

    def solve(self, x0, N, mode, max_nfev=300):
        if os.environ.get('CUB_LM') == '1':
            return self.solve_lm(x0, N, mode, max_iter=max(60, max_nfev // 2))
        lo, hi = self.bounds(N, mode)
        x0 = np.clip(x0, lo + 1e-12, hi - 1e-12)
        r = least_squares(lambda x: self.res(x, N), x0, jac=lambda x: self.jac(x, N),
                          bounds=(lo, hi), xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=max_nfev)
        return r


    def solve_lm(self, x0, N, mode, max_iter=120, tol=1e-26):
        """Gedämpftes Gauß-Newton (Levenberg-Marquardt) mit Schranken durch Projektion; Cholesky statt SVD."""
        lo, hi = self.bounds(N, mode)
        x = np.clip(x0, lo, hi)
        r = self.res(x, N); c = 0.5 * r @ r; mu = 1e-3; M = self.M
        for it in range(max_iter):
            if c < tol:
                break
            J = self.jac(x, N)
            g = J.T @ r
            free = ~(((x <= lo + 1e-13) & (g > 0)) | ((x >= hi - 1e-13) & (g < 0)))
            Jf = J[:, free]
            ok = False
            for _ in range(14):
                try:
                    if Jf.shape[1] >= M:
                        A = Jf @ Jf.T
                        A[np.diag_indices(M)] += mu * (np.trace(A) / M + 1e-30)
                        d = Jf.T @ cho_solve(cho_factor(A, lower=True), -r)
                    else:
                        A = Jf.T @ Jf
                        k = A.shape[0]
                        A[np.diag_indices(k)] += mu * (np.trace(A) / k + 1e-30)
                        d = cho_solve(cho_factor(A, lower=True), -(Jf.T @ r))
                except (LinAlgError, ValueError):
                    mu *= 10; continue
                xn = x.copy(); xn[free] += d; xn = np.clip(xn, lo, hi)
                rn = self.res(xn, N); cn = 0.5 * rn @ rn
                if cn < c:
                    x, r, c = xn, rn, cn; mu = max(mu / 4, 1e-15); ok = True; break
                mu *= 4
            if not ok:
                break
        class R: pass
        out = R(); out.x = x; out.cost = c
        return out

    def random_start(self, N, rng):
        npts = 2 * N + (1 if self.center else 0) if self.cs else N
        x0 = np.r_[rng.uniform(-.95, .95, 3 * N), np.full(N, 8.0 / npts) * rng.uniform(.5, 1.5, N)]
        if self.center:
            x0 = np.r_[x0, 8.0 / npts]
        return x0

    def full_points(self, x, N):
        P = x[:3 * N].reshape(N, 3); w = x[3 * N:4 * N]
        if not self.cs:
            return P, w
        X = np.vstack([P, -P]); W = np.r_[w, w]
        if self.center:
            X = np.vstack([X, np.zeros((1, 3))]); W = np.r_[W, x[4 * N]]
        return X, W


def verify(X, W, deg):
    """Unabhängige Prüfung mit Monomen (alle Monome Grad <= deg), gibt (max Fehler, min w, max|x|, sum w)."""
    I = lambda k: 2.0 / (k + 1) if k % 2 == 0 else 0.0
    m = 0.0
    for i in range(deg + 1):
        for j in range(deg + 1 - i):
            for k in range(deg + 1 - i - j):
                v = np.sum(W * X[:, 0] ** i * X[:, 1] ** j * X[:, 2] ** k)
                m = max(m, abs(v - I(i) * I(j) * I(k)))
    return m, W.min(), np.abs(X).max(), W.sum()


def search(deg, N, mode, cs, center, seed, tl, tag=""):
    pr = Prob(deg, cs, center)
    rng = np.random.default_rng(seed)
    t0 = time.time(); best = None; tries = 0
    while time.time() - t0 < tl:
        tries += 1
        r = pr.solve(pr.random_start(N, rng), N, mode)
        if best is None or r.cost < best.cost:
            best = r
        if r.cost < 1e-24:
            break
    ok = best.cost < 1e-24
    return pr, best, ok, tries


if __name__ == "__main__":
    deg, N, mode, cs, center, seed, tl = (int(sys.argv[1]), int(sys.argv[2]), sys.argv[3],
                                          sys.argv[4] == "cs", sys.argv[5] == "1", int(sys.argv[6]), float(sys.argv[7]))
    pr, best, ok, tries = search(deg, N, mode, cs, center, seed, tl)
    npts = (2 * N + int(center)) if cs else N
    msg = f"deg {deg} pts {npts} N {N} mode {mode} {'cs' if cs else 'ns'} c{int(center)} seed {seed}: tries {tries} cost {best.cost:.2e} ok={ok}"
    if ok:
        X, W = pr.full_points(best.x, N)
        err, wmin, xmax, ws = verify(X, W, deg + 1 if cs else deg)
        msg += f" | err {err:.1e} wmin {wmin:.3g} max|x| {xmax:.6f} sumw {ws:.12f}"
        np.save(f"rule_d{deg}_{mode}_{'cs' if cs else 'ns'}_{npts}.npy", np.c_[X, W])
    print(msg, flush=True)

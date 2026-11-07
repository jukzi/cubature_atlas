"""Newton-Kantorovich certificate for a symmetric cubature rule on the cube [-1,1]^3 in the symmetry-reduced system.

A rule invariant under G (Oh = all signed permutations, Td = permutations with an even number of sign changes)
is described by orbit parameters: every orbit is {g B y : g in G} with B an orthonormal basis of the fixed space
of the stabiliser of its representative.  Unknowns z = (y of all orbits, one weight per orbit).
Equations: the moments of the G-invariant polynomials (orthonormal basis U of that space), F_red = U^T F.
Then h = L * eta / sigma^2 <= 1/2 certifies an exact symmetric rule within |dz| <= 2*eta (Ben-Israel, as in cubature_atlas.certificate).
L is estimated from differences of J in random directions and multiplied by `factor` (floating-point certificate)."""
import itertools
import numpy as np
import cubature_atlas as CA


def group(kind):
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            if kind == "Td" and np.prod(signs) != 1:
                continue
            M = np.zeros((3, 3))
            for i, p in enumerate(perm):
                M[i, p] = signs[i]
            mats.append(M)
    return mats


def orbits(X, W, G, tol=1e-7):
    N = len(X)
    used = np.zeros(N, bool)
    out = []
    for i in range(N):
        if used[i]:
            continue
        idx = []
        for g in G:
            q = g @ X[i]
            d = np.abs(X - q).max(axis=1)
            j = int(np.argmin(d))
            if d[j] > tol or abs(W[j] - W[i]) > tol:
                return None
            if j not in idx:
                idx.append(j)
        for j in idx:
            used[j] = True
        out.append((i, idx))
    return out


def build_parametrisation(X, W, G, orb):
    """Returns Pmap (4N x nz), z0, bookkeeping.  x = (X.ravel(), W) = Pmap z."""
    N = len(X)
    cols = []
    z0 = []
    blocks = []
    nz = 0
    for rep, idx in orb:
        S = [g for g in G if np.abs(g @ X[rep] - X[rep]).max() < 1e-7]
        A = np.vstack([g - np.eye(3) for g in S]) if S else np.zeros((1, 3))
        u, s, vt = np.linalg.svd(A)
        r = int((s > 1e-9).sum())
        B = vt[r:].T
        f = B.shape[1]
        y0 = B.T @ X[rep]
        seen = []
        coset = []
        for g in G:
            q = g @ X[rep]
            for k, (qq, _) in enumerate(seen):
                if np.abs(q - qq).max() < 1e-7:
                    break
            else:
                seen.append((q, g))
        blocks.append((idx, seen, B, f, nz))
        z0.extend(y0)
        nz += f
    nw = len(orb)
    Pm = np.zeros((4 * N, nz + nw))
    wv = []
    pos = 0
    for o, (idx, seen, B, f, off) in enumerate(blocks):
        for (q, g) in seen:
            # find the node index equal to q
            j = int(np.argmin(np.abs(X - q).max(axis=1)))
            for c in range(3):
                Pm[3 * j + c, off:off + f] = (g @ B)[c]
            Pm[3 * N + j, nz + o] = 1.0
        wv.append(W[idx[0]])
    z0 = np.r_[np.array(z0), np.array(wv)]
    return Pm, z0


def invariant_basis(Bs, G, rng, extra=40):
    """Orthonormal basis U of the span of all residual-type vectors sum_g phi(g p) (the invariant polynomial space)."""
    cols = []
    K = Bs.idx.__len__() // 8 + extra
    pts = rng.uniform(-1, 1, (K * 3, 3))
    R = np.zeros((len(Bs.idx), len(pts)))
    for k, p in enumerate(pts):
        R[:, k] = Bs(np.array([g @ p for g in G])).sum(axis=0)
    u, s, vt = np.linalg.svd(R, full_matrices=False)
    r = int((s > 1e-10 * s[0]).sum())
    return u[:, :r], r, len(pts)


def certificate_cube(n, X, W, factor=2.0, directions=24, seed=0):
    rng = np.random.default_rng(seed)
    Bs = CA.Basis("cube", n)
    for kind in ("Oh", "Td"):
        G = group(kind)
        orb = orbits(X, W, G)
        if orb is not None:
            break
    else:
        return dict(certified=False, inside=False, h=float("inf"), group=None)
    Pm, z0 = build_parametrisation(X, W, G, orb)
    N = len(X)
    U, r, npts = invariant_basis(Bs, G, rng)
    if r >= npts:
        raise RuntimeError("not enough sample points for the invariant space")

    def unpack(z):
        x = Pm @ z
        return x[:3 * N].reshape(N, 3), x[3 * N:]

    def F(z):
        Xz, Wz = unpack(z)
        return U.T @ CA.residual(Bs, Xz, Wz)

    def J(z):
        # J_red = U^T J_full Pm, assembled per coordinate so that the full M x 4N matrix is never formed
        Xz, Wz = unpack(z)
        Phi = Bs(Xz)
        out = (U.T @ Phi.T) @ Pm[3 * N:, :]
        for c in range(3):
            Xc = Xz.astype(complex)
            Xc[:, c] += 1e-30j
            D = (Bs(Xc).imag / 1e-30) * Wz[:, None]
            out += (U.T @ D.T) @ Pm[c:3 * N:3, :]
            del D, Xc
        return out

    F0 = F(z0)
    J0 = J(z0)
    sv = np.linalg.svd(J0, compute_uv=False)
    out = dict(group=kind, orbits=len(orb), M=int(J0.shape[0]), p=int(J0.shape[1]), sym=True,
               resid=float(np.abs(CA.residual(Bs, X, W)).max()), sigma=float(sv[-1]), cond=float(sv[0] / sv[-1]))
    # consistency: the full residual lies in the invariant space
    full = CA.residual(Bs, X, W)
    out["leak"] = float(np.linalg.norm(full - U @ (U.T @ full)))
    rank_ok = J0.shape[0] <= J0.shape[1] and sv[-1] > 1e-9 * sv[0]
    out["rank_ok"] = bool(rank_ok)
    if not rank_ok:
        out.update(h=float("inf"), radius=float("inf"), certified=False, inside=False)
        return out
    eta = float(np.linalg.norm(np.linalg.lstsq(J0, F0, rcond=None)[0]))
    rho = 1e-5
    L = 0.0
    for _ in range(directions):
        u = rng.normal(size=len(z0))
        u /= np.linalg.norm(u)
        L = max(L, np.linalg.norm(J(z0 + rho * u) - J0, 2) / rho)
    L *= factor
    h = L * eta / sv[-1] ** 2
    radius = 2 * eta
    scale = float(np.linalg.norm(Pm, axis=1).max())  # every node coordinate / weight is a row of Pm applied to z: |dx| <= scale * |dz|
    wmin = float(W.min())
    margin = float((1 - np.abs(X)).min())
    out.update(eta=eta, L=float(L), h=float(h), radius=float(radius), wmin=wmin, margin=margin,
               certified=bool(h <= 0.5), inside=bool(h <= 0.5 and wmin > radius * scale and margin > radius * scale))
    return out

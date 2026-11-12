"""Total-degree homotopy for the 10 quadratic flat-extension equations of the square, degree 8.

H(y, t) = (1 - t) * gamma * G(y) + t * F(y),  G_i(y) = y_i^2 - 1,  2^10 = 1024 start points y_i = +-1.
Predictor (Euler) - corrector (Newton) with adaptive step; paths that blow up go to infinity.
Prints the number of finite solutions, of real ones, and the real solutions' node data."""
import os
import sys
import itertools
import numpy as np

sys.argv = ["flat_ext_square8.py", os.devnull]
ns = {}
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "flat_ext_square8.py")).read(), ns)
eqs = ns["equations"]
n = 10
# F_k(y) = c_k + b_k . y + y^T Q_k y (Q symmetric)
c = np.zeros(n)
b = np.zeros((n, n))
Q = np.zeros((n, n, n))
for k, f in enumerate(eqs):
    for m, v in f.items():
        v = float(v)
        if len(m) == 0:
            c[k] += v
        elif len(m) == 1:
            b[k, m[0]] += v
        else:
            i, j = m
            if i == j:
                Q[k, i, i] += v
            else:
                Q[k, i, j] += v / 2
                Q[k, j, i] += v / 2
# Scale each equation to unit size.
s = np.maximum(np.abs(c), np.max(np.abs(Q).reshape(n, -1), axis=1))
c = c / s
b = b / s[:, None]
Q = Q / s[:, None, None]


def F(y):
    return c + b @ y + np.einsum("kij,i,j->k", Q, y, y)


def JF(y):
    return b + 2 * np.einsum("kij,j->ki", Q, y)


rng = np.random.default_rng(7)
gamma = np.exp(2j * np.pi * rng.random())


def H(y, t):
    return (1 - t) * gamma * (y * y - 1) + t * F(y)


def Hy(y, t):
    return (1 - t) * gamma * np.diag(2 * y) + t * JF(y)


def Ht(y, t):
    return -gamma * (y * y - 1) + F(y)


def track(y):
    t = 0.0
    h = 0.01
    while t < 1.0:
        h = min(h, 1.0 - t)
        # Euler predictor
        dy = -np.linalg.solve(Hy(y, t), Ht(y, t))
        yp = y + h * dy
        tp = t + h
        ok = False
        for it in range(6):
            corr = np.linalg.solve(Hy(yp, tp), H(yp, tp))
            yp = yp - corr
            if np.linalg.norm(corr) < 1e-10 * (1 + np.linalg.norm(yp)):
                ok = True
                break
        if ok:
            y = yp
            t = tp
            h = min(h * 1.5, 0.05)
        else:
            h = h / 2
            if h < 1e-12:
                return None, t
        if np.linalg.norm(y) > 1e8:
            return None, t
    # final Newton on F
    for it in range(20):
        y = y - np.linalg.solve(JF(y), F(y))
    return y, 1.0


sols = []
fail = 0
for signs in itertools.product([1.0, -1.0], repeat=n):
    y0 = np.array(signs, dtype=complex)
    try:
        y, t = track(y0)
    except np.linalg.LinAlgError:
        y = None
    if y is None or not np.all(np.isfinite(y)) or np.linalg.norm(F(y)) > 1e-9:
        fail += 1
        continue
    if not any(np.linalg.norm(y - z) < 1e-6 * (1 + np.linalg.norm(z)) for z in sols):
        sols.append(y)
real = [z for z in sols if np.max(np.abs(z.imag)) < 1e-7 * (1 + np.linalg.norm(z))]
print("paths 1024, diverged or failed", fail, "distinct finite solutions", len(sols), "real", len(real), flush=True)
for z in real:
    print("real solution, |y| = %.6f" % np.linalg.norm(z.real))

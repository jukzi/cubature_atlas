"""Dreieck PI: Start aus Regel des Grads deg-k plus Knoten in den groessten Luecken; Speichern nur nach Monomtest UND Zertifikat.
python3 seed_tri2.py <deg> <k> <N> <starts> [maxsec]  ->  tri_d_d<deg>_n<N>.npy"""
import os, sys, time, glob, json
os.environ["CUB_LM"] = "1"
import numpy as np
from scipy.spatial import cKDTree
from dom import Dom, ProbG
import kubatur as K
deg, k, N, maxs = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
maxsec = float(sys.argv[5]) if len(sys.argv) > 5 else 1e9
MI = int(os.environ.get("MI", "1500")); D = Dom("triangle", deg); pr = ProbG(D)
def base_rule(dg):
    b = None
    for f in glob.glob("tri_d_d%d_n*.npy" % dg):
        A = np.load(f)
        if b is None or len(A) < len(b): b = A
    if b is None: b = np.array(json.load(open("allrules30.json"))["rules"]["triangle"][str(dg)]["pts"], float)
    return b
base = base_rule(deg - k); X0, W0 = base[:, :-1], base[:, -1]; N0 = len(W0)
if N <= N0: sys.exit("N <= N0")
def to_q(P):
    y = P[:, 1]; u = P[:, 0] / np.clip(1 - y, 1e-9, None); return np.c_[np.clip(u, 0, 1), np.clip(y, 0, 1)]
def start(rng, M=30000):
    u = rng.random((M, 2)); m = u.sum(1) > 1; u[m] = 1 - u[m]; C = list(X0)
    for _ in range(N - N0):
        dist = cKDTree(np.array(C)).query(u)[0]; top = np.argsort(-dist)[:max(3, M // 200)]; C.append(u[rng.choice(top)])
    C = np.array(C); W = np.r_[W0 * N0 / N * 0.8, np.full(N - N0, 0.5 / N)]; W *= 0.5 / W.sum(); return C, W
t0 = time.time(); ok = 0
for s in range(maxs):
    if time.time() - t0 > maxsec: break
    rng = np.random.default_rng(100 + s); C, W = start(rng); x = np.r_[to_q(C).ravel(), W]
    r = pr.solve_lm(x, N, "d", max_iter=MI)
    for _k in range(4):
        if r.cost < 1e-24 or r.cost > 1e-4: break
        r = pr.solve_lm(r.x, N, "d", max_iter=MI)
    print("seed", s, "cost %.1e" % r.cost, "%.0fs" % (time.time() - t0), flush=True)
    if r.cost < 1e-24:
        X, Wt = pr.full_points(r.x, N)
        e, wm, out = K.verify("triangle", deg, X, Wt)
        if e < 1e-12 and wm > 0 and out <= 1e-9:
            c = K.certificate("triangle", deg, X, Wt)
            print("   certificate h=%.2e certified=%s inside=%s" % (c["h"], c["certified"], c["inside"]), flush=True)
            if c["certified"] and c["inside"]:
                np.save("tri_d_d%d_n%d.npy" % (deg, N), np.c_[X, Wt]); print("FOUND", deg, N, flush=True); ok = 1; break
print("RESULT deg", deg, "N", N, "found" if ok else "none", flush=True)

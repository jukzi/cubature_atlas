"""Quadrat: Start aus der Regel eines niedrigeren Grads plus Knoten in den groessten Luecken.
python3 seed_sq.py <deg> <k> <N> <mode d|b> <starts> [maxsec]   (Startregel: Grad deg-k aus sq_*.npy oder allrules30.json)
Nur Regeln mit Monomtest UND Zertifikat werden gespeichert: sq_<mode>_d<deg>_n<N>.npy"""
import os, sys, time, glob, json
os.environ["CUB_LM"] = "1"; os.environ["FASTJAC"] = "0"
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, "lib")
from dom import Dom, ProbG, verify_dom
import cubature_atlas as CA
deg, k, N, mode, maxs = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
maxsec = float(sys.argv[6]) if len(sys.argv) > 6 else 1e9
MI = int(os.environ.get("MI", "1500")); PB = 1.0 if mode == "d" else 3.0
D = Dom("square", deg); pr = ProbG(D)
def best_file(dg, md):
    b = None
    for f in glob.glob("sq_%s_d%d_n*.npy" % (md, dg)):
        A = np.load(f)
        if b is None or len(A) < len(b): b = A
    return b
base = best_file(deg - k, mode)
if base is None and mode == "b": base = best_file(deg - k, "d")
if base is None: base = np.array(json.load(open("allrules30.json"))["rules"]["square"][str(deg - k)]["pts"], float)
X0, W0 = base[:, :2], base[:, 2]; N0 = len(W0); VOL = 4.0
if N <= N0: sys.exit("N <= N0=%d" % N0)
def start(rng, M=40000):
    S = rng.uniform(-1, 1, (M, 2)); C = list(X0)
    for _ in range(N - N0):
        dist = cKDTree(np.array(C)).query(S)[0]; top = np.argsort(-dist)[:max(3, M // 200)]; C.append(S[rng.choice(top)])
    C = np.array(C); W = np.r_[W0 * N0 / N * 0.8, np.full(N - N0, VOL / N)]; W *= VOL / W.sum(); return C, W
t0 = time.time(); ok = 0
for s in range(maxs):
    if time.time() - t0 > maxsec: break
    rng = np.random.default_rng(100 + s); C, W = start(rng); x = np.r_[np.clip(C, -PB, PB).ravel(), W]
    r = pr.solve_lm(x, N, mode, max_iter=MI)
    for _k in range(4):
        if r.cost < 1e-24 or r.cost > 1e-4: break
        r = pr.solve_lm(r.x, N, mode, max_iter=MI)
    print("seed", s, "cost %.1e" % r.cost, "%.0fs" % (time.time() - t0), flush=True)
    if r.cost < 1e-24:
        X, Wt = pr.full_points(r.x, N)
        e, wm, ins, _ = verify_dom("square", deg, X, Wt); print("   check e=%.1e wmin=%.1e outside=%.1e" % (e, wm, ins - 1), flush=True)
        if e < 1e-12 and wm > 0 and (mode != "d" or ins <= 1 + 1e-9):
            c = CA.certificate("square", deg, X, Wt)
            print("   certificate h=%.2e certified=%s inside=%s" % (c["h"], c["certified"], c["inside"]), flush=True)
            if c["certified"] and (mode != "d" or c["inside"]):
                np.save("sq_%s_d%d_n%d.npy" % (mode, deg, N), np.c_[X, Wt]); print("FOUND deg", deg, "N", N, flush=True); ok = 1; break
print("RESULT deg", deg, "N", N, "mode", mode, "found" if ok else "none", flush=True)

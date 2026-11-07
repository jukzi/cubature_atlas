"""Quadrat, zentralsymmetrisch (Paare +-p, optional Mittelpunkt): Start aus symmetrischer Regel niedrigeren Grads plus neue Paare.
python3 seed_cs.py <deg> <N gesamt> <mode d|b> <starts> [maxsec]
Speichert nur nach Monomtest UND Zertifikat: sq_<mode>_d<deg>_n<N>.npy (volle Punktliste) """
import os, sys, time, glob, re
os.environ["CUB_LM"] = "1"; os.environ["FASTJAC"] = "0"
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, "lib")
from dom import Dom, ProbG, verify_dom
import cubature_atlas as CA
deg, N, mode, maxs = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
maxsec = float(sys.argv[5]) if len(sys.argv) > 5 else 1e9
MI = int(os.environ.get("MI", "1500")); PB = 1.0 if mode == "d" else 3.0
centre = N % 2 == 1; half = N // 2
D = Dom("square", deg); pr = ProbG(D, cs=True, center=centre)
def is_cs(A):
    X = A[:, :2]; t = cKDTree(X); d, i = t.query(-X); return d.max() < 1e-7 and np.abs(A[i, 2] - A[:, 2]).max() < 1e-7
def candidates():
    out = []
    for dg in range(deg - 1, 0, -1):
        for f in glob.glob("r2_square_d%d_cs_*.npy" % dg) + glob.glob("sq_%s_d%d_n*.npy" % (mode, dg)) + glob.glob("sq_d_d%d_n*.npy" % dg):
            A = np.load(f)
            if len(A) < N - 1 and is_cs(A): out.append((dg, len(A), A))
        if out: break
    return out
cands = candidates()
if not cands: sys.exit("keine symmetrische Startregel")
dg0, N0, base = min(cands, key=lambda c: c[1])
X0, W0 = base[:, :2], base[:, 2]
sel = (X0[:, 0] > 1e-9) | ((np.abs(X0[:, 0]) <= 1e-9) & (X0[:, 1] > 1e-9))
H0, HW = X0[sel], W0[sel]; c0 = np.abs(X0).max(1) < 1e-9; cw = W0[c0][0] if c0.any() else 0.0
print("Start Grad", dg0, "N0", N0, "Paare", len(H0), "Mittelpunkt", bool(c0.any()), "-> N", N, "Paare", half, "Mittelpunkt", centre, flush=True)
if half < len(H0): sys.exit("zu wenige Paare")
def start(rng, M=40000):
    S = rng.uniform(-1, 1, (M, 2)); C = list(H0); allp = lambda: np.vstack([np.array(C), -np.array(C)])
    for _ in range(half - len(H0)):
        dist = cKDTree(allp()).query(S)[0]; top = np.argsort(-dist)[:max(3, M // 200)]; C.append(S[rng.choice(top)])
    C = np.array(C); w = np.r_[HW * 0.85, np.full(half - len(H0), 4.0 / N)]
    x = np.r_[np.clip(C, -PB, PB).ravel(), w]
    if centre: x = np.r_[x, 4.0 / N]
    return x
t0 = time.time(); ok = 0
for s in range(maxs):
    if time.time() - t0 > maxsec: break
    rng = np.random.default_rng(200 + s); x = start(rng)
    r = pr.solve_lm(x, half, mode, max_iter=MI)
    for _k in range(4):
        if r.cost < 1e-24 or r.cost > 1e-4: break
        r = pr.solve_lm(r.x, half, mode, max_iter=MI)
    print("seed", s, "cost %.1e" % r.cost, "%.0fs" % (time.time() - t0), flush=True)
    if r.cost < 1e-24:
        X, W = pr.full_points(r.x, half)
        e, wm, ins, _ = verify_dom("square", deg, X, W)
        if e < 1e-12 and wm > 0 and (mode != "d" or ins <= 1 + 1e-9):
            c = CA.certificate("square", deg, X, W)
            print("   certificate h=%.2e certified=%s inside=%s" % (c["h"], c["certified"], c["inside"]), flush=True)
            if c["certified"] and (mode != "d" or c["inside"]):
                np.save("sq_%s_d%d_n%d.npy" % (mode, deg, len(W)), np.c_[X, W]); print("FOUND deg", deg, "N", len(W), flush=True); ok = 1; break
print("RESULT deg", deg, "N", N, "mode", mode, "found" if ok else "none", flush=True)

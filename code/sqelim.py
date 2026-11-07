"""Knotenreduktion auf dem Quadrat [-1,1]^2: PI (mode d) oder PO (mode b).
Aufruf: python3 sqelim.py <deg> <mode d|b> <sekunden> <seed> [startdatei.npy]
Jede Regel wird erst nach Monomtest UND Zertifikat gespeichert (sq_<mode>_d<deg>_n<N>.npy)."""
import os, sys, time, json, glob, re
os.environ["CUB_LM"] = "1"; os.environ["FASTJAC"] = "0"
import numpy as np
sys.path.insert(0, "lib")
from dom import Dom, ProbG, verify_dom
import cubature_atlas as CA
deg, mode, tl, seed = int(sys.argv[1]), sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
D = Dom("square", deg); pr = ProbG(D); rng = np.random.default_rng(seed); d = 2; t0 = time.time()
MI = int(os.environ.get("MI", "300"))
PB = 1.0 if mode == "d" else 3.0

def ok_rule(X, W):
    e, wm, ins, _ = verify_dom("square", deg, X, W)
    if not (e < 1e-12 and wm > 0 and (mode != "d" or ins <= 1 + 1e-9)):
        return False
    c = CA.certificate("square", deg, X, W)
    print("    certificate h=%.2e certified=%s inside=%s" % (c["h"], c["certified"], c["inside"]), flush=True)
    return bool(c["certified"]) and (mode != "d" or bool(c["inside"]))

def load_start():
    if len(sys.argv) > 5:
        A = np.load(sys.argv[5]); return A[:, :2], A[:, 2]
    best = None
    for f in glob.glob("sq_%s_d%d_n*.npy" % (mode, deg)):
        A = np.load(f)
        if best is None or len(A) < len(best): best = A
    if best is None and mode == "b":
        for f in glob.glob("sq_d_d%d_n*.npy" % deg):
            A = np.load(f)
            if best is None or len(A) < len(best): best = A
    if best is None:
        R = json.load(open("allrules30.json"))["rules"]["square"][str(deg)]
        best = np.array(R["pts"])
    return best[:, :2], best[:, 2]

P, w = load_start(); N = len(w)
print("Start N =", N, "deg", deg, "mode", mode, flush=True)
x = np.r_[P.ravel(), w]
fails = 0
while time.time() - t0 < tl and N > 1:
    Qc = x[:2 * N].reshape(N, 2); wc = x[2 * N:]
    order = np.argsort(wc)
    cands = [("del", int(k)) for k in order[:6]]
    dm = np.linalg.norm(Qc[:, None] - Qc[None], axis=2); np.fill_diagonal(dm, 9)
    pairs = np.dstack(np.unravel_index(np.argsort(dm, axis=None)[:12:2], dm.shape))[0]
    cands += [("mer", (int(a), int(b))) for a, b in pairs if a != b]
    cands += [("del", int(k)) for k in rng.choice(N, min(4, N), replace=False)]
    rng.shuffle(cands); done = False
    for kind, c in cands:
        if kind == "del":
            keep = np.delete(np.arange(N), c); Qn = Qc[keep].copy(); wn = wc[keep].copy()
        else:
            a, b = c; keep = np.delete(np.arange(N), [a, b]); wm_ = wc[a] + wc[b]
            Qn = np.vstack([Qc[keep], (Qc[a] * wc[a] + Qc[b] * wc[b]) / wm_]); wn = np.r_[wc[keep], wm_]
        Qn = Qn + rng.normal(0, rng.choice([0.002, 0.008, 0.03]), Qn.shape)
        xn = np.r_[np.clip(Qn, -PB, PB).ravel(), wn]
        r = pr.solve_lm(xn, N - 1, mode, max_iter=MI)
        if r.cost < 1e-24:
            Xp, Wp = pr.full_points(r.x, N - 1)
            if ok_rule(Xp, Wp):
                np.save("sq_%s_d%d_n%d.npy" % (mode, deg, N - 1), np.c_[Xp, Wp])
                x, N = r.x, N - 1; done = True; fails = 0
                print("  -> %d Punkte (%s) t=%.0fs" % (N, kind, time.time() - t0), flush=True); break
        if time.time() - t0 > tl: break
    if not done:
        fails += 1; print("  Runde ohne Erfolg bei %d (#%d) t=%.0fs" % (N, fails, time.time() - t0), flush=True)
print("ENDE square", deg, mode, N, flush=True)

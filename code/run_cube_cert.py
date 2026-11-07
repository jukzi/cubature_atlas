import json, sys, time, numpy as np
sys.path.insert(0, "lib")
import cube_cert as CC
A = json.load(open("allrules30.json"))["rules"]["cube"]
degs = [int(a) for a in sys.argv[1].split(",")]
out = {}
try: out = json.load(open("cube_cert_out_%s.json" % sys.argv[2]))
except Exception: pass
for n in degs:
    P = np.array(A[str(n)]["pts"]); t = time.time()
    c = CC.certificate_cube(n, P[:, :3], P[:, 3]); c["N"] = len(P); c["wsum"] = float(P[:, 3].sum()); c["time"] = time.time() - t
    out[str(n)] = c
    print(n, len(P), c.get("group"), "M", c.get("M"), "p", c.get("p"), "h=%.2e" % c["h"], "cert", c["certified"], "inside", c["inside"], "leak %.1e" % c.get("leak", -1), "%.0fs" % c["time"], flush=True)
    json.dump(out, open("cube_cert_out_%s.json" % sys.argv[2], "w"))

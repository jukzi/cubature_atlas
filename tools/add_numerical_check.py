"""add_numerical_check: store the floating-point check (Newton-Kantorovich test of code/cubature_atlas.py) as "numerical_check" in rule files that have none.
Only the line `"numerical_check": {...},` is inserted in front of `"nodes"`; nothing else in the file changes (a locked rule may change this key).
The check is numerical, not a proof.  Needs numpy and scipy (code/cubature_atlas.py imports it).
Usage: python tools/add_numerical_check.py [--max-p 4000] [--dry-run] <rule files>   (without files: every rule without numerical_check)"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
import cubature_atlas as C


def check_of(r):
    X = np.array(r["nodes"], float)
    W = np.array(r["weights"], float)
    cert = C.certificate(r["domain"], r["degree"], X, W)
    return {"N": int(len(X)), "M": int(cert["M"]), "p": int(cert["p"]), "res": float(cert["resid"]), "sigma": float(cert["sigma"]),
            "h": float(cert["h"]), "rad": float(cert["radius"]), "cert": bool(cert["certified"]), "inside": bool(cert["inside"])}


def insert(path, nc):
    text = open(path, encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.split(nl)
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith('"nodes"'):
            indent = ln[:len(ln) - len(ln.lstrip())]
            lines.insert(i, indent + '"numerical_check": ' + json.dumps(nc) + ",")
            break
    else:
        raise SystemExit("%s: no nodes line" % path)
    out = nl.join(lines)
    json.loads(out)
    open(path, "w", encoding="utf-8", newline="").write(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--max-p", type=int, default=4000, help="skip rules with more unknowns than this (the test needs 40 Jacobians and singular values)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    files = a.files or sorted(glob.glob(os.path.join(ROOT, "rules", "*", "*", "CA-*.json")))
    for f in files:
        r = json.load(open(f, encoding="utf-8"))
        if r.get("numerical_check"):
            continue
        d = {"interval": 1, "triangle": 2, "square": 2, "tetra": 3, "cube": 3}[r["domain"]]
        p = (d + 1) * len(r["nodes"])
        if p > a.max_p:
            print("skipped (p = %d > %d): %s" % (p, a.max_p, os.path.basename(f)), flush=True)
            continue
        t0 = time.time()
        nc = check_of(r)
        print("%s: M %d p %d sigma %.2e cert %s inside %s (%.0fs)" % (os.path.basename(f), nc["M"], nc["p"], nc["sigma"], nc["cert"], nc["inside"], time.time() - t0), flush=True)
        if not a.dry_run:
            insert(f, nc)


if __name__ == "__main__":
    main()

"""import_burkardt_rules: literature rules whose nodes and weights are printed in public rule files, brought into the atlas format (one rule file each).
Source files (data tables of published rules, one Fortran subroutine per degree, 16 digits):
  triangle_xg: Xiao, Gimbutas (2010), fully symmetric triangle rules of the degrees 0 to 50 (file triangle_symq_rule.f90, barycentric nodes a, b, c and weights w).
The numbers are read from the printed arrays; nothing is recomputed or tuned. The nodes (b, c) are the Cartesian coordinates on the atlas triangle (0,0), (1,0), (0,1); the weights are scaled to the sum 1/2.
The program stops with an error if the node count differs from the count in reference_data/literature_counts.json, if the moment error is larger than 1e-12, if a weight is not positive or if a node lies outside the domain.
Usage: python code/import_burkardt_rules.py --source triangle_xg --file <triangle_symq_rule.f90> --degrees 32 33 35 [--root .]
Afterwards: python tools/verify_rigorous.py --strict --save <rule files>; python tools/build_atlas.py --base origin/main --changed
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cubature_atlas as C
import po_elim

NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[DdEe][-+]?\d+)?")


def read_subroutines(path):
    """Return {subroutine name: {array name: [floats]}} for all arrays given as name = (/ ... /) or data name / ... /."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    kept = []
    for line in lines:
        s = line.strip()
        if s.startswith("!"):
            continue
        kept.append(s.replace("&", " "))
    text = "\n".join(kept)
    chunks = re.split(r"(?im)^\s*subroutine\s+", text)
    result = {}
    for chunk in chunks[1:]:
        name = re.match(r"\s*(\w+)", chunk).group(1).lower()
        arrays = {}
        for m in re.finditer(r"(?is)(\w+)\s*=\s*\(/(.*?)/\)", chunk):
            arrays[m.group(1).lower()] = parse_numbers(m.group(2))
        for m in re.finditer(r"(?is)\bdata\s+(\w+)\s*/(.*?)/", chunk):
            arrays[m.group(1).lower()] = parse_numbers(m.group(2))
        result[name] = arrays
    return result


def parse_numbers(text):
    return [float(t.replace("D", "e").replace("d", "e")) for t in NUMBER.findall(text)]


def literature_count(root, domain, key, degree):
    path = os.path.join(root, "reference_data", "literature_counts.json")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)["domains"][domain]
    for entry in data:
        if entry["k"] == key:
            return entry["arr"][degree - 1]
    return None


def triangle_xg(subs, degree):
    """Fully symmetric triangle rule of the given degree: barycentric a, b, c and weights w."""
    arrays = subs["rule%02d" % degree]
    a = np.array(arrays["a"])
    b = np.array(arrays["b"])
    c = np.array(arrays["c"])
    w = np.array(arrays["w"])
    X = np.stack([b, c], axis=1)
    inside = float(min(a.min(), b.min(), c.min()))
    scale = float(w.sum()) / 0.5
    return X, w / scale, inside, scale


def read_data_post(path, degree):
    """Arrays xs<degree>, ys<degree>, ws<degree> of the block '%%% nmax = <degree>' in a file data_post.m (35 digits, all nodes listed)."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    blocks = re.split(r"(?m)^%%% nmax = ", text)
    for block in blocks[1:]:
        if int(re.match(r"\d+", block).group(0)) != degree:
            continue
        out = {}
        for name in ("xs", "ys", "ws"):
            m = re.search(r"%s%d = \[(.*?)\];" % (name, degree), block, re.S)
            out[name] = np.array([float(t) for t in m.group(1).replace(",", " ").split()])
        return out
    raise SystemExit("degree %d not found in %s" % (degree, path))


def triangle_xg_rot(path, degree):
    """Rotationally symmetric Xiao-Gimbutas triangle rule (file tables/triarotq/data_post.m of the authors' repository) on the triangle
    (-1,-1/sqrt3), (1,-1/sqrt3), (0,2/sqrt3); mapped affinely to (0,0), (1,0), (0,1), weights scaled to the sum 1/2."""
    d = read_data_post(os.path.join(path, "tables", "triarotq", "data_post.m"), degree)
    P = np.stack([d["xs"], d["ys"]], axis=1)
    w = d["ws"]
    s3 = math.sqrt(3.0)
    v0 = np.array([-1.0, -1.0 / s3])
    v1 = np.array([1.0, -1.0 / s3])
    v2 = np.array([0.0, 2.0 / s3])
    m = np.array([v1 - v0, v2 - v0]).T
    X = np.linalg.solve(m, (P - v0).T).T
    inside = float(min(X.min(), (1.0 - X.sum(axis=1)).min()))
    scale = float(w.sum()) / 0.5
    return X, w / scale, inside, scale


def tetra_xg(subs, degree):
    """Xiao-Gimbutas tetrahedron rule on the tetrahedron with the vertices
    (-1,-1/sqrt3,-1/sqrt6), (0,2/sqrt3,-1/sqrt6), (1,-1/sqrt3,-1/sqrt6), (0,0,3/sqrt6); mapped affinely to (0,0,0), (1,0,0), (0,1,0), (0,0,1)."""
    arrays = subs["rule%02d" % degree]
    P = np.stack([arrays["xs"], arrays["ys"], arrays["zs"]], axis=1)
    w = np.array(arrays["ws"])
    s3 = math.sqrt(3.0)
    s6 = math.sqrt(6.0)
    v0 = np.array([-1.0, -1.0 / s3, -1.0 / s6])
    v1 = np.array([0.0, 2.0 / s3, -1.0 / s6])
    v2 = np.array([1.0, -1.0 / s3, -1.0 / s6])
    v3 = np.array([0.0, 0.0, 3.0 / s6])
    m = np.array([v1 - v0, v2 - v0, v3 - v0]).T
    X = np.linalg.solve(m, (P - v0).T).T
    inside = float(min(X.min(), (1.0 - X.sum(axis=1)).min()))
    scale = float(w.sum()) / (1.0 / 6.0)
    return X, w / scale, inside, scale


def box_rule(names, volume):
    """Reader of a rule on [-1,1]^d given as arrays named by `names` and a weight array `ws`."""
    def reader(subs, degree):
        key = "rule%02d" % degree
        arrays = subs[key]
        X = np.stack([arrays[n] for n in names], axis=1)
        w = np.array(arrays["ws"])
        inside = float(1.0 - np.abs(X).max())
        scale = float(w.sum()) / volume
        return X, w / scale, inside, scale
    return reader


def smr_square(subs, degree):
    """Festa-Sommariva rule smrNN: a flat array of triples (x, y, w) on [-1,1]^2."""
    arrays = subs["smr%02d" % degree]
    flat = np.array(arrays["xyw"])
    t = flat.reshape(-1, 3)
    X = t[:, :2]
    w = t[:, 2]
    inside = float(1.0 - np.abs(X).max())
    scale = float(w.sum()) / 4.0
    return X, w / scale, inside, scale


def tetra_wk(path, degree):
    """Worku-Hicken-Zingg tetrahedron rule: file tet_q<degree>_n<N>_ext.dat in the directory `path`; three header lines, then one block per node
    (x, y, z, w on separate lines, blank line between blocks) on the tetrahedron with the vertex coordinates -1 and 1; mapped by x -> (x+1)/2."""
    names = [n for n in os.listdir(path) if n.startswith("tet_q%d_n" % degree) and n.endswith("_ext.dat")]
    if len(names) != 1:
        raise SystemExit("degree %d: expected one file, found %s" % (degree, names))
    with open(os.path.join(path, names[0]), encoding="utf-8") as fh:
        lines = [t.strip() for t in fh.read().splitlines()]
    numbers = [float(t) for t in lines[3:] if t]
    t = np.array(numbers).reshape(-1, 4)
    X = (t[:, :3] + 1.0) / 2.0
    w = t[:, 3] / 8.0
    inside = float(min(X.min(), (1.0 - X.sum(axis=1)).min()))
    scale = float(w.sum()) / (1.0 / 6.0)
    return X, w / scale, inside, scale


def cube_pk(path, degree):
    """Pinkse rule: file legendre_d3_p<degree>_q<q>_n<N>.csv in the directory `path` (header lines starting with #, then a line x1,x2,x3,w and one line per node)
    on the unit cube with the weights summing to 1; mapped by x -> 2x - 1, weights times 8."""
    names = [n for n in os.listdir(path) if n.startswith("legendre_d3_p%d_q" % degree) and n.endswith(".csv")]
    if len(names) != 1:
        raise SystemExit("degree %d: expected one file, found %s" % (degree, names))
    rows = []
    with open(os.path.join(path, names[0]), encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("x1") or not line.strip():
                continue
            rows.append([float(t) for t in line.split(",")])
    t = np.array(rows)
    X = 2.0 * t[:, :3] - 1.0
    w = 8.0 * t[:, 3]
    inside = float(1.0 - np.abs(X).max())
    scale = float(w.sum()) / 8.0
    return X, w / scale, inside, scale


def central_symmetry(X, W):
    """True if the node set and the weights are invariant under x -> -x (up to 1e-13)."""
    for p, w in zip(X, W):
        d = np.abs(X + p).max(axis=1)
        j = int(np.argmin(d))
        if d[j] > 1e-13 or abs(W[j] - w) > 1e-13 * max(1.0, abs(w)):
            return False
    return True


SOURCES = {
    "tetra_xg": {
        "family": "xiao_gimbutas_2010",
        "tag": "XG",
        "domain": "tetra",
        "letter": "T3",
        "key": "xg",
        "reader": tetra_xg,
        "symmetry": "none",
        "author": "Xiao-Gimbutas",
        "source": "H. Xiao, Z. Gimbutas: A numerical algorithm for the construction of efficient quadrature rules in two and higher dimensions. Comput. Math. Appl. 59 (2010) 663-676; numbers as printed (16 digits) in J. Burkardt's tetrahedron_arbq_rule, mapped affinely from the tetrahedron with the vertices (-1,-1/sqrt3,-1/sqrt6), (0,2/sqrt3,-1/sqrt6), (1,-1/sqrt3,-1/sqrt6), (0,0,3/sqrt6) to the atlas tetrahedron, weights scaled to the sum 1/6.",
    },
    "cube_xg": {
        "family": "xiao_gimbutas_2010",
        "tag": "XG",
        "domain": "cube",
        "letter": "C3",
        "key": "xg",
        "reader": box_rule(["xs", "ys", "zs"], 8.0),
        "symmetry": "none",
        "author": "Xiao-Gimbutas",
        "source": "H. Xiao, Z. Gimbutas: A numerical algorithm for the construction of efficient quadrature rules in two and higher dimensions. Comput. Math. Appl. 59 (2010) 663-676; numbers as printed (16 digits) in J. Burkardt's cube_arbq_rule on [-1,1]^3, weights scaled to the sum 8.",
    },
    "square_xg": {
        "family": "xiao_gimbutas_2010",
        "tag": "XG",
        "domain": "square",
        "letter": "C2",
        "key": "xg",
        "reader": box_rule(["xs", "ys"], 4.0),
        "symmetry": "none",
        "author": "Xiao-Gimbutas",
        "source": "H. Xiao, Z. Gimbutas: A numerical algorithm for the construction of efficient quadrature rules in two and higher dimensions. Comput. Math. Appl. 59 (2010) 663-676; numbers as printed (16 digits) in J. Burkardt's square_arbq_rule on [-1,1]^2, weights scaled to the sum 4.",
    },
    "cube_pk": {
        "family": "pinkse_2026",
        "tag": "PK",
        "domain": "cube",
        "letter": "C3",
        "key": "pk",
        "reader": cube_pk,
        "symmetry": "auto",
        "author": "Pinkse",
        "source": "J. Pinkse: Positive weight Hermite and Legendre quadrature rules. arXiv:2609.26840 (2026); data: Zenodo, doi:10.5281/zenodo.22881864 (CC BY 4.0), transformed from the unit cube to [-1,1]^3 (weights times 8), double precision as published.",
    },
    "tetra_wk": {
        "family": "worku_hicken_zingg_2024",
        "tag": "WHZ",
        "domain": "tetra",
        "letter": "T3",
        "key": "wk",
        "reader": tetra_wk,
        "symmetry": "full",
        "author": "Worku–Hicken–Zingg",
        "source": "Z. A. Worku, J. E. Hicken, D. W. Zingg: Very high-order symmetric positive-interior quadrature rules on triangles and tetrahedra. J. Comput. Appl. Math. 472 (2026) 116782, arXiv:2409.02027; data: github.com/OptimalDesignLab/SummationByParts.jl, pi_quadrature_data/high_precision (MIT license), mapped from the simplex with vertex coordinates -1 and 1 to the unit simplex (x -> (x+1)/2, weights divided by 2^d) and rounded to double precision.",
    },
    "square_fs": {
        "family": "festa_sommariva_2012",
        "tag": "FS",
        "domain": "square",
        "letter": "C2",
        "key": "fs",
        "reader": smr_square,
        "symmetry": "auto",
        "author": "Festa-Sommariva",
        "source": "M. Festa, A. Sommariva: Computing almost minimal formulas on the square. J. Comput. Appl. Math. 236 (2012) 4296-4302; numbers as printed (16 digits) in J. Burkardt's square_minimal_rule (GNU LGPL), which reproduces the authors' sets.",
    },
    "triangle_xg_rot": {
        "family": "xiao_gimbutas_2010",
        "tag": "XG",
        "domain": "triangle",
        "letter": "T2",
        "key": "xg",
        "reader": triangle_xg_rot,
        "symmetry": "rotation (group of order 3)",
        "author": "Xiao-Gimbutas",
        "source": "H. Xiao, Z. Gimbutas: A numerical algorithm for the construction of efficient quadrature rules in two and higher dimensions. Comput. Math. Appl. 59 (2010) 663-676; rotationally symmetric rule triarotq as tabulated (35 digits) by the authors, mapped affinely from the triangle (-1,-1/sqrt3), (1,-1/sqrt3), (0,2/sqrt3) to the atlas triangle, weights scaled to the sum 1/2.",
    },
    "triangle_xg": {
        "family": "xiao_gimbutas_2010",
        "tag": "XG",
        "domain": "triangle",
        "letter": "T2",
        "key": "xg",
        "reader": triangle_xg,
        "symmetry": "full",
        "author": "Xiao-Gimbutas",
        "source": "H. Xiao, Z. Gimbutas: A numerical algorithm for the construction of efficient quadrature rules in two and higher dimensions. Comput. Math. Appl. 59 (2010) 663-676; fully symmetric rule as printed (16 digits) in J. Burkardt's triangle_symq_rule, barycentric nodes (b, c) used as Cartesian coordinates of the atlas triangle, weights scaled to the sum 1/2.",
    },
}


def build_rule(cfg, degree, root, subs, max_p):
    X, W, inside, scale = cfg["reader"](subs, degree)
    n = len(W)
    expected = literature_count(root, cfg["domain"], cfg["key"], degree)
    if expected is not None and expected != n:
        raise SystemExit("degree %d: %d nodes, literature count %s" % (degree, n, expected))
    if W.min() <= 0:
        raise SystemExit("degree %d: weight not positive" % degree)
    if inside < -1e-15:
        raise SystemExit("degree %d: node outside the domain (%g)" % (degree, inside))
    err = po_elim.monomial_error(cfg["domain"], degree, X, W)
    print("degree %d: %d nodes, weight sum factor %.3e, moment error %.2e, min barycentric %.2e" % (degree, n, scale, err, inside), flush=True)
    if err > 1e-12:
        raise SystemExit("degree %d: moment error %.2e too large" % (degree, err))
    symmetry = cfg["symmetry"]
    if symmetry == "auto":
        symmetry = "central" if central_symmetry(X, W) else "none"
    unknowns = n * (X.shape[1] + 1)
    nc = None
    if unknowns <= max_p:
        cert = C.certificate(cfg["domain"], degree, X, W)
        nc = {"N": int(n), "M": int(cert["M"]), "p": int(cert["p"]), "res": float(cert["resid"]), "sigma": float(cert["sigma"]),
              "h": float(cert["h"]), "rad": float(cert["radius"]), "cert": bool(cert["certified"]), "inside": bool(cert["inside"])}
    rid = "CA-%s-%02d-%d-PI-%s" % (cfg["letter"], degree, n, cfg["tag"])
    rule = {"id": rid, "family": cfg["family"], "domain": cfg["domain"], "dimension": X.shape[1], "degree": degree, "points": int(n),
            "quality": "PI", "origin": "literature",
            "method": "taken from the literature (printed numbers read from the data table, weights scaled to the domain volume), moment equations re-checked by the build",
            "author": cfg["author"], "source": cfg["source"], "status": "proven", "symmetry": symmetry}
    if nc is not None:
        rule["numerical_check"] = nc
    rule["nodes"] = [[float(repr(float(v))) for v in row] for row in X]
    rule["weights"] = [float(v) for v in W]
    return rule


def write_rule(root, r):
    d = os.path.join(root, "rules", r["family"], r["domain"])
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, r["id"] + ".json")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(r, ensure_ascii=False, indent=1) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=sorted(SOURCES))
    ap.add_argument("--file", required=True)
    ap.add_argument("--degrees", type=int, nargs="+", required=True)
    ap.add_argument("--root", default=os.path.dirname(HERE))
    ap.add_argument("--max-p", type=int, default=4000, help="no numerical_check for rules with more unknowns than this (the check is slow)")
    args = ap.parse_args()
    cfg = SOURCES[args.source]
    subs = args.file if os.path.isdir(args.file) else read_subroutines(args.file)
    for degree in args.degrees:
        path = write_rule(args.root, build_rule(cfg, degree, args.root, subs, args.max_p))
        print("  wrote", path, flush=True)


if __name__ == "__main__":
    main()

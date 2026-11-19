#!/usr/bin/env python3
"""Print the state of the atlas: per domain and degree the counting value Z,
the smallest stored PI and PO node count and the gap N - Z.

Usage: python tools/state_table.py [--domain tetra] [--min-degree 20] [--csv]

Cells without any stored rule are listed with "-".
Counts of literature rules without stored points are not included
(see reference_data/literature_counts.json).
"""
import argparse
import glob
import json
import os

DIM = {"interval": 1, "triangle": 2, "square": 2, "tetra": 3, "cube": 3}


def moment_count(dim, degree):
    # Number of monomials of total degree <= degree in dim variables.
    # Square and cube are exact for total degree, as in the atlas.
    count = 1
    for k in range(1, dim + 1):
        count = count * (degree + k) // k
    return count


def counting_value(dim, degree):
    # Z = ceil(M / (dim + 1)).
    m = moment_count(dim, degree)
    return -(-m // (dim + 1))


def load_rules(root):
    best = {}
    pattern = os.path.join(root, "rules", "*", "*", "*.json")
    for path in sorted(glob.glob(pattern)):
        if path.endswith("family.json"):
            continue
        with open(path, encoding="utf-8") as handle:
            rule = json.load(handle)
        domain = rule.get("domain")
        degree = rule.get("degree")
        quality = rule.get("quality")
        points = rule.get("points")
        if domain is None or degree is None or points is None:
            continue
        key = (domain, degree, quality)
        if key not in best or points < best[key][0]:
            best[key] = (points, rule.get("id", os.path.basename(path)))
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--min-degree", type=int, default=1)
    parser.add_argument("--csv", action="store_true")
    args = parser.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    best = load_rules(root)
    cells = sorted({(d, n) for (d, n, _q) in best})
    header = ["domain", "degree", "Z", "N_PI", "gap_PI", "N_PO", "gap_PO", "PO<PI"]
    rows = []
    for domain, degree in cells:
        if args.domain and domain != args.domain:
            continue
        if degree < args.min_degree:
            continue
        z = counting_value(DIM[domain], degree)
        pi = best.get((domain, degree, "PI"), (None, ""))[0]
        po = best.get((domain, degree, "PO"), (None, ""))[0]
        gap_pi = "-" if pi is None else pi - z
        gap_po = "-" if po is None else po - z
        better = "yes" if (pi is not None and po is not None and po < pi) else ""
        rows.append([domain, degree, z, "-" if pi is None else pi, gap_pi,
                     "-" if po is None else po, gap_po, better])
    if args.csv:
        print(",".join(header))
        for row in rows:
            print(",".join(str(x) for x in row))
        return
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    for row in [header] + rows:
        print("  ".join(str(x).rjust(widths[i]) for i, x in enumerate(row)))


if __name__ == "__main__":
    main()

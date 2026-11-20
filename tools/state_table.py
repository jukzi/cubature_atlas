#!/usr/bin/env python3
"""Print the state of the atlas: per domain and degree the counting value Z,
the smallest stored PI and PO node count and the gap N - Z.

Usage: python tools/state_table.py [--domain tetra] [--min-degree 20] [--csv]

Lists every degree from 1 to 30 for every domain, plus higher degrees up to the highest stored rule in that domain. Missing cells are listed with "-".
Counts of literature rules without stored points are not included
(see reference_data/literature_counts.json).
"""
import argparse
import glob
import json
import os

DIM = {"interval": 1, "triangle": 2, "square": 2, "tetra": 3, "cube": 3}
DEFAULT_MAX_DEGREE = 30


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


def state_rows(best, domain_filter=None, min_degree=1):
    """Return a complete domain/degree matrix, including cells without stored rules.

    The atlas historically covers degrees 1..30 in every domain. Higher-degree
    work is uneven across domains, so extend each domain's range to its highest
    stored degree without inventing a larger search range for the other domains.
    """
    max_degree = {domain: DEFAULT_MAX_DEGREE for domain in DIM}
    for domain, degree, _quality in best:
        if domain in max_degree:
            max_degree[domain] = max(max_degree[domain], degree)

    rows = []
    for domain in sorted(DIM):
        if domain_filter and domain != domain_filter:
            continue
        for degree in range(max(1, min_degree), max_degree[domain] + 1):
            z = counting_value(DIM[domain], degree)
            pi = best.get((domain, degree, "PI"), (None, ""))[0]
            po = best.get((domain, degree, "PO"), (None, ""))[0]
            gap_pi = "-" if pi is None else pi - z
            gap_po = "-" if po is None else po - z
            better = "yes" if (pi is not None and po is not None and po < pi) else ""
            rows.append([domain, degree, z, "-" if pi is None else pi, gap_pi,
                         "-" if po is None else po, gap_po, better])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--min-degree", type=int, default=1)
    parser.add_argument("--csv", action="store_true")
    args = parser.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    best = load_rules(root)
    header = ["domain", "degree", "Z", "N_PI", "gap_PI", "N_PO", "gap_PO", "PO<PI"]
    rows = state_rows(best, args.domain, args.min_degree)
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

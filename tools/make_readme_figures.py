"""Static figures for README.md (PNG, light and dark), computed from the rule files.

  assets/readme_gallery_<theme>.png   four rules, marker area proportional to the weight
  assets/readme_map_<theme>.png       the 150 cells (domain x degree): optimal / better than the literature / ...
  assets/readme_efficiency_<theme>.png nodes relative to the counting value Z, atlas vs literature vs lower bound
  assets/readme_outside_<theme>.png   the degree-8 square rule with 15 nodes, one node outside

The counting value Z and the lower bound L are the formulas of the website (site_template/index.html).
Needs numpy and matplotlib. Usage: python tools/make_readme_figures.py
"""
import os
import glob
import json
from math import comb
from math import ceil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets")
DOMAINS = ["interval", "triangle", "square", "tetra", "cube"]
LABEL = {"interval": "Interval", "triangle": "Triangle", "square": "Square", "tetra": "Tetrahedron", "cube": "Cube"}
DIM = {"interval": 1, "triangle": 2, "square": 2, "tetra": 3, "cube": 3}
MAXN = 30
# Colors: reference categorical slots (blue, orange, aqua) and neutral inks per theme.
THEMES = {
    "light": {"bg": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df", "dom": "#bdbcb6",
              "atlas": "#2a78d6", "lit": "#eb6834", "po": "#1baf7a", "bound": "#8a8984",
              "opt": "#1b4f91", "better": "#5b9ae6", "equal": "#c9dcf3", "nolit": "#e9e8e4"},
    "dark": {"bg": "#0d1117", "ink": "#f0efec", "ink2": "#c3c2b7", "grid": "#2a2e35", "dom": "#5b5f66",
             "atlas": "#3987e5", "lit": "#d95926", "po": "#199e70", "bound": "#9a9990",
             "opt": "#8fbcf5", "better": "#3987e5", "equal": "#23466f", "nolit": "#20242b"},
}


def counting_value(dom, n):
    """Counting value Z(n) of the website (with the symmetric variant on square and cube)."""
    if dom == "interval":
        return ceil((n + 1) / 2)
    d = DIM[dom]
    z = ceil(comb(n + d, d) / (d + 1))
    if dom in ("square", "cube"):
        e = sum(comb(j + d - 1, d - 1) for j in range(0, n + 1, 2))
        z = min(z, ceil(2 * e / (d + 1)))
    return z


def lower_bound(dom, n):
    """Proven lower bound L(n) of the website."""
    k = n // 2
    if dom == "interval":
        return ceil((n + 1) / 2)
    if dom in ("triangle", "square"):
        extra = (k + 1) // 2 if n % 2 else 0
        return comb(k + 2, 2) + extra
    if n % 2 == 0:
        return comb(k + 3, 3)
    if dom == "tetra":
        if n == 5:
            return 13
        r = (k + 1) * (k + 2) // 2
        return comb(k + 3, 3) + (r - (k + 2) // 2) // 2
    ev = sum(comb(j + 2, 2) for j in range(0, k + 1, 2))
    od = sum(comb(j + 2, 2) for j in range(1, k + 1, 2))
    if k % 2 == 0:
        return 2 * ev - 1
    return 2 * od


def load():
    """Best node counts per (domain, degree): atlas PI, atlas PO, literature; and the rule files themselves."""
    roles = {}
    for f in glob.glob(os.path.join(ROOT, "rules", "*", "family.json")):
        F = json.load(open(f))
        roles[F["id"]] = F["role"]
    best = {}
    files = {}
    for f in glob.glob(os.path.join(ROOT, "rules", "*", "*", "*.json")):
        if f.endswith("family.json"):
            continue
        r = json.load(open(f))
        key = "lit" if roles[r["family"]] == "literature" else r["quality"]
        files.setdefault((r["domain"], r["degree"], key), []).append(f)
        for d in range(1, r["degree"] + 1):
            k = (r["domain"], d, key)
            best[k] = min(best.get(k, 10 ** 9), r["points"])
    lc = json.load(open(os.path.join(ROOT, "reference_data", "literature_counts.json")))["domains"]
    for dom, cols in lc.items():
        for c in cols:
            for i, v in enumerate(c["arr"]):
                if v is not None:
                    k = (dom, i + 1, "lit")
                    best[k] = min(best.get(k, 10 ** 9), v)
    return best, files


def rule(files, dom, deg, q):
    """The rule file with the fewest nodes for this domain, degree and quality."""
    rs = [json.load(open(f)) for f in files.get((dom, deg, q), [])]
    return min(rs, key=lambda r: r["points"])


def style(ax, t):
    """Recessive axes and grid."""
    ax.set_facecolor(t["bg"])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=t["ink2"], labelsize=9, length=0)
    ax.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def proj3(P):
    """Simple oblique projection of 3D points."""
    a = np.radians(28)
    x = P[:, 0] + 0.45 * P[:, 1] * np.cos(a)
    y = P[:, 2] + 0.45 * P[:, 1] * np.sin(a)
    return np.c_[x, y]


def draw_rule(ax, r, t, color, outside_color=None):
    """Nodes of a rule, marker area proportional to the weight; the domain outline in gray."""
    dom = r["domain"]
    P = np.array(r["nodes"], dtype=float)
    W = np.array(r["weights"], dtype=float)
    vol = {"triangle": 0.5, "square": 4.0, "tetra": 1 / 6, "cube": 8.0}[dom]
    # Average marker area about 2000 / N^0.8 points^2, at least 6.
    s = np.maximum(2000.0 * (W / vol) * len(W) / len(W) ** 0.8, 6.0)
    if dom == "triangle":
        ax.add_patch(Polygon([[0, 0], [1, 0], [0, 1]], closed=True, fill=False, edgecolor=t["dom"], linewidth=1.2))
        Q = P
    elif dom == "square":
        ax.add_patch(Rectangle((-1, -1), 2, 2, fill=False, edgecolor=t["dom"], linewidth=1.2))
        Q = P
    else:
        if dom == "tetra":
            V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
            E = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        else:
            V = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
            E = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.sum(np.abs(V[i] - V[j])) == 2]
        PV = proj3(V)
        for i, j in E:
            ax.plot(PV[[i, j], 0], PV[[i, j], 1], color=t["dom"], linewidth=1.0, zorder=1)
        order = np.argsort(-P[:, 1])
        P = P[order]
        W = W[order]
        s = s[order]
        Q = proj3(P)
    cols = [color] * len(W)
    if outside_color is not None:
        lo = -1 - 1e-12
        hi = 1 + 1e-12
        cols = [outside_color if (np.any(p < lo) or np.any(p > hi)) else color for p in P]
    ax.scatter(Q[:, 0], Q[:, 1], s=s, c=cols, alpha=0.85, edgecolors=t["bg"], linewidths=0.6, zorder=2)
    if outside_color is not None:
        for (x, y), c in zip(Q, cols):
            if c == outside_color:
                ax.scatter([x], [y], s=260, facecolors="none", edgecolors=outside_color, linewidths=1.6, zorder=3)
                ax.annotate("node outside", (x, y), xytext=(x - 0.95, y + 0.12), color=t["ink"], fontsize=9,
                            arrowprops=dict(arrowstyle="-", color=t["ink2"], linewidth=0.8, shrinkB=9))
    ax.set_aspect("equal")
    ax.axis("off")


def gallery(best, files, t, name):
    picks = [("triangle", 20), ("square", 20), ("tetra", 12), ("cube", 12)]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6), facecolor=t["bg"])
    for ax, (dom, deg) in zip(axs, picks):
        r = rule(files, dom, deg, "PI")
        draw_rule(ax, r, t, t["atlas"])
        ax.set_title("%s, degree %d\n%d nodes" % (LABEL[dom], deg, r["points"]), color=t["ink"], fontsize=11, pad=6)
    fig.text(0.5, 0.02, "Marker area proportional to the weight. Every rule integrates all polynomials up to its degree exactly.", ha="center", color=t["ink2"], fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.1, wspace=0.08)
    fig.savefig(os.path.join(OUT, "readme_gallery_%s.png" % name), dpi=150, facecolor=t["bg"])
    plt.close(fig)


# Proven minimal among rules with positive weights although above L (website: LPOS).
LPOS = {("tetra", 3): 6}


def cell_class(best, dom, n):
    pi = best.get((dom, n, "PI"))
    if dom == "interval" and pi is None:
        pi = ceil((n + 1) / 2)
    lit = best.get((dom, n, "lit"))
    if pi == lower_bound(dom, n) or LPOS.get((dom, n)) == pi:
        return "opt"
    if lit is None:
        return "nolit"
    if pi < lit:
        return "better"
    return "equal"


def cellmap(best, t, name):
    fig, ax = plt.subplots(figsize=(13, 2.9), facecolor=t["bg"])
    ax.set_facecolor(t["bg"])
    for i, dom in enumerate(DOMAINS):
        for n in range(1, MAXN + 1):
            c = cell_class(best, dom, n)
            ax.add_patch(Rectangle((n - 0.5 + 0.06, -i - 0.5 + 0.06), 0.88, 0.88, facecolor=t[c], edgecolor="none"))
    ax.set_xlim(0.4, MAXN + 0.6)
    ax.set_ylim(-len(DOMAINS) + 0.4, 0.6)
    ax.set_yticks([-i for i in range(len(DOMAINS))])
    ax.set_yticklabels([LABEL[d] for d in DOMAINS], color=t["ink"], fontsize=10)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.tick_params(colors=t["ink2"], length=0, labelsize=9)
    ax.set_xlabel("degree n", color=t["ink2"], fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    labels = [("opt", "provably optimal"), ("better", "fewer nodes than the published rules"),
              ("equal", "same number as the published rules"), ("nolit", "no published comparison")]
    handles = [Rectangle((0, 0), 1, 1, facecolor=t[k]) for k, _ in labels]
    leg = ax.legend(handles, [s for _, s in labels], loc="upper center", bbox_to_anchor=(0.5, 1.32), ncol=4, frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(t["ink"])
    fig.subplots_adjust(left=0.09, right=0.995, top=0.8, bottom=0.17)
    fig.savefig(os.path.join(OUT, "readme_map_%s.png" % name), dpi=150, facecolor=t["bg"])
    plt.close(fig)


def efficiency(best, t, name):
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.4), facecolor=t["bg"], sharey=True)
    for ax, dom in zip(axs, ["triangle", "square", "tetra", "cube"]):
        style(ax, t)
        ns = np.arange(1, MAXN + 1)
        Z = np.array([counting_value(dom, n) for n in ns], dtype=float)
        pi = np.array([best[(dom, n, "PI")] for n in ns], dtype=float)
        lit = np.array([best.get((dom, n, "lit"), np.nan) for n in ns], dtype=float)
        L = np.array([lower_bound(dom, n) for n in ns], dtype=float)
        ax.plot(ns, L / Z, color=t["bound"], linewidth=1.5, linestyle=(0, (3, 2)), label="proven lower bound")
        ax.plot(ns, lit / Z, color=t["lit"], linewidth=2, label="published rules")
        ax.plot(ns, pi / Z, color=t["atlas"], linewidth=2, label="Cubature Atlas")
        ax.axhline(1.0, color=t["ink2"], linewidth=0.8)
        ax.set_title(LABEL[dom], color=t["ink"], fontsize=11)
        ax.set_xlabel("degree n", color=t["ink2"], fontsize=9)
        ax.set_xlim(1, MAXN)
        ax.set_ylim(0.45, 1.75)
    axs[0].set_ylabel("nodes / counting value Z", color=t["ink2"], fontsize=9)
    leg = axs[0].legend(loc="upper left", frameon=False, fontsize=8.5)
    for txt in leg.get_texts():
        txt.set_color(t["ink"])
    fig.subplots_adjust(left=0.06, right=0.995, top=0.88, bottom=0.17, wspace=0.08)
    fig.savefig(os.path.join(OUT, "readme_efficiency_%s.png" % name), dpi=150, facecolor=t["bg"])
    plt.close(fig)


def outside(files, t, name):
    r = rule(files, "square", 8, "PO")
    fig, ax = plt.subplots(figsize=(3.6, 3.6), facecolor=t["bg"])
    draw_rule(ax, r, t, t["atlas"], outside_color=t["po"])
    ax.set_xlim(-1.3, 1.45)
    ax.set_ylim(-1.3, 1.45)
    ax.set_title("Square, degree 8: 15 nodes,\none of them outside", color=t["ink"], fontsize=10)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.02)
    fig.savefig(os.path.join(OUT, "readme_outside_%s.png" % name), dpi=150, facecolor=t["bg"])
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    best, files = load()
    for name, t in THEMES.items():
        gallery(best, files, t, name)
        cellmap(best, t, name)
        efficiency(best, t, name)
        outside(files, t, name)
    print("written to", OUT)


if __name__ == "__main__":
    main()

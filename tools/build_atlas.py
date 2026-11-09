"""Cubature Atlas: verify all rule files and build the static website.

Usage:
    python tools/build_atlas.py --check              # verify only (used on pull requests)
    python tools/build_atlas.py                      # verify and build site/
    python tools/build_atlas.py --base origin/main   # also refuse changes to locked rules
    python tools/build_atlas.py --strict             # stale translations are errors, not warnings
    python tools/build_atlas.py --sync-hashes        # after updating German texts: mark them as current

Needs only Python 3.9+ and numpy.
"""
import argparse
import hashlib
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
import zipfile
from itertools import product

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RULES = os.path.join(ROOT, "rules")
SITE = os.path.join(ROOT, "site")
BASE_URL = "https://jukzi.github.io/cubature_atlas/"
REPO_URL = "https://github.com/jukzi/cubature_atlas"
CONCEPT_DOI = "10.5281/zenodo.24037245"
MAXN = 30

DOMAINS = ["interval", "triangle", "square", "tetra", "cube"]
SYM = {"interval": "C1", "triangle": "T2", "square": "C2", "tetra": "T3", "cube": "C3"}
DIM = {"interval": 1, "triangle": 2, "square": 2, "tetra": 3, "cube": 3}
VOL = {"interval": 2.0, "triangle": 0.5, "square": 4.0, "tetra": 1.0 / 6.0, "cube": 8.0}
BOX = {"interval", "square", "cube"}
ROLES = {"pi", "po", "literature", "own"}
QUALITIES = {"PI", "PO", "NI", "NO"}

# Tolerances of the build check
RES_TOL = 1e-11          # max |sum w x^a - int x^a| / volume over all monomials up to the degree
INSIDE_TOL = 1e-12       # a node counts as inside if its distance to the domain is below this
ID_RE = re.compile(r"^CA-(C1|T2|C2|T3|C3)-(\d{2})-(\d+)-(PI|PO|NI|NO)-([A-Za-z0-9]+)(-[a-z])?$")
FAMILY_RE = re.compile(r"^[A-Za-z0-9_]+$")

errors = []
warnings = []


def err(msg):
    errors.append(msg)


def warn(msg):
    warnings.append(msg)


def sha(text):
    return hashlib.sha256(text.encode("utf8")).hexdigest()[:16]


def rel(path):
    return os.path.relpath(path, ROOT).replace(os.sep, "/")


# ---------------------------------------------------------------- exact monomial integrals

_mono_cache = {}


def monomials(dom, n):
    """Exponent tuples of total degree <= n and their exact integrals over the reference domain."""
    key = (dom, n)
    if key in _mono_cache:
        return _mono_cache[key]
    d = DIM[dom]
    exps = [t for t in product(range(n + 1), repeat=d) if sum(t) <= n]
    vals = []
    for t in exps:
        if dom in BOX:
            v = 1.0
            for a in t:
                v *= 0.0 if a % 2 else 2.0 / (a + 1)
        else:
            num = 1
            for a in t:
                num *= math.factorial(a)
            v = num / math.factorial(d + sum(t))
        vals.append(v)
    out = (np.array(exps, dtype=int), np.array(vals))
    _mono_cache[key] = out
    return out


def moment_residual(dom, n, X, W):
    """Largest absolute moment error over all monomials up to degree n, divided by the volume."""
    exps, exact = monomials(dom, n)
    d = DIM[dom]
    powers = [np.vstack([X[:, c] ** k for k in range(n + 1)]) for c in range(d)]
    vals = np.ones((len(exps), len(W)))
    for c in range(d):
        vals *= powers[c][exps[:, c]]
    q = vals @ W
    return float(np.max(np.abs(q - exact)) / VOL[dom])


def margin(dom, X):
    """Distance of the nodes to the domain boundary (negative = outside)."""
    if dom in BOX:
        return float((1 - np.abs(X)).min())
    return float(np.c_[X, 1 - X.sum(1)].min())


def quality(minw, marg):
    q = "P" if minw > 0 else "N"
    q += "I" if marg >= -INSIDE_TOL else "O"
    return q


# ---------------------------------------------------------------- Gauss-Jacobi tables for the page

def gauss_jacobi(m, a):
    """Gauss-Jacobi nodes and weights for weight (1-x)^a on [-1,1] (Golub-Welsch)."""
    b = 0.0
    k = np.arange(m, dtype=float)
    diag = np.zeros(m)
    for i in range(m):
        den = (2 * i + a + b) * (2 * i + a + b + 2)
        diag[i] = (b * b - a * a) / den if den != 0 else (b - a) / (a + b + 2)
    off = np.zeros(max(m - 1, 0))
    for i in range(1, m):
        num = 4 * i * (i + a) * (i + b) * (i + a + b)
        den = (2 * i + a + b) ** 2 * (2 * i + a + b + 1) * (2 * i + a + b - 1)
        off[i - 1] = math.sqrt(num / den)
    J = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    x, V = np.linalg.eigh(J)
    mu0 = 2 ** (a + b + 1) * math.gamma(a + 1) * math.gamma(b + 1) / math.gamma(a + b + 2)
    w = mu0 * V[0, :] ** 2
    del k
    return [float(v) for v in x], [float(v) for v in w]


def gj_tables():
    out = {"L": {}, "J1": {}, "J2": {}}
    for m in range(1, MAXN // 2 + 2):
        x, w = np.polynomial.legendre.leggauss(m)
        out["L"][str(m)] = [[float(v) for v in x], [float(v) for v in w]]
        out["J1"][str(m)] = list(gauss_jacobi(m, 1.0))
        out["J2"][str(m)] = list(gauss_jacobi(m, 2.0))
    return out


# ---------------------------------------------------------------- loading and checking

def load_json(path):
    try:
        with open(path, encoding="utf8") as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        err("%s: cannot read JSON (%s)" % (rel(path), e))
        return None


def load_families():
    fams = []
    tags = {}
    roles = {}
    for fid in sorted(os.listdir(RULES)):
        fdir = os.path.join(RULES, fid)
        if not os.path.isdir(fdir):
            continue
        fp = os.path.join(fdir, "family.json")
        if not os.path.exists(fp):
            err("rules/%s: family.json missing" % fid)
            continue
        f = load_json(fp)
        if f is None:
            continue
        if not FAMILY_RE.match(fid):
            err("rules/%s: directory name may only contain letters, digits and _" % fid)
        if f.get("id") != fid:
            err("rules/%s/family.json: id must equal the directory name" % fid)
        for key in ("name", "short", "description"):
            v = f.get(key)
            if not isinstance(v, dict) or not v.get("en") or not v.get("de"):
                err("rules/%s/family.json: '%s' needs an English and a German text (en, de)" % (fid, key))
        role = f.get("role", "own")
        f["role"] = role
        if role not in ROLES:
            err("rules/%s/family.json: role must be one of %s" % (fid, sorted(ROLES)))
        if role in ("pi", "po"):
            if role in roles:
                err("rules/%s/family.json: role '%s' already used by %s" % (fid, role, roles[role]))
            roles[role] = fid
        tag = f.get("tag", "")
        if not re.match(r"^[A-Za-z0-9]+$", tag):
            err("rules/%s/family.json: tag must be alphanumeric" % fid)
        if tag in tags and not {f["role"], tags[tag][1]} <= {"pi", "po"}:
            err("rules/%s/family.json: tag %s already used by %s" % (fid, tag, tags[tag][0]))
        tags[tag] = (fid, role)
        q = f.get("quality")
        if q is not None and q not in QUALITIES:
            err("rules/%s/family.json: quality must be one of %s" % (fid, sorted(QUALITIES)))
        f["references"] = f.get("references", [])
        f["rules"] = []
        fams.append(f)
    return fams


def load_rules(fam):
    fdir = os.path.join(RULES, fam["id"])
    for dom in sorted(os.listdir(fdir)):
        ddir = os.path.join(fdir, dom)
        if not os.path.isdir(ddir):
            continue
        if dom not in DOMAINS:
            err("%s: unknown domain directory (allowed: %s)" % (rel(ddir), ", ".join(DOMAINS)))
            continue
        for fn in sorted(os.listdir(ddir)):
            if not fn.endswith(".json"):
                continue
            path = os.path.join(ddir, fn)
            r = load_json(path)
            if r is None:
                continue
            r["_path"] = path
            with open(path, "rb") as fh:
                r["_sha"] = hashlib.sha256(fh.read()).hexdigest()[:16]
            check_rule(fam, dom, r, fn)
            fam["rules"].append(r)


def check_rule(fam, dom, r, fn):
    where = rel(r["_path"])
    rid = r.get("id", "")
    if fn != rid + ".json":
        err("%s: file name must be <id>.json (id is '%s')" % (where, rid))
    m = ID_RE.match(rid)
    if not m:
        err("%s: id '%s' does not match CA-<domain>-<degree 2 digits>-<points>-<quality>-<tag>[-a]" % (where, rid))
        return
    if r.get("family") != fam["id"]:
        err("%s: family must be '%s'" % (where, fam["id"]))
    if r.get("domain") != dom:
        err("%s: domain must be '%s' (directory)" % (where, dom))
    d = DIM[dom]
    nodes = r.get("nodes")
    weights = r.get("weights")
    if not isinstance(nodes, list) or not isinstance(weights, list) or len(nodes) != len(weights) or not nodes:
        err("%s: nodes and weights must be non-empty lists of equal length" % where)
        return
    if any(not isinstance(p, list) or len(p) != d for p in nodes):
        err("%s: every node needs %d coordinates" % (where, d))
        return
    X = np.array(nodes, dtype=float)
    W = np.array(weights, dtype=float)
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(W)):
        err("%s: non-finite numbers" % where)
        return
    n = r.get("degree")
    if not isinstance(n, int) or n < 1:
        err("%s: degree must be a positive integer" % where)
        return
    if r.get("points") != len(W):
        err("%s: points=%s but %d nodes stored" % (where, r.get("points"), len(W)))
    if m.group(1) != SYM[dom] or int(m.group(2)) != n or int(m.group(3)) != len(W) or m.group(5) != fam.get("tag"):
        err("%s: id does not agree with domain/degree/points/family tag" % where)
    res = moment_residual(dom, n, X, W)
    marg = margin(dom, X)
    minw = float(W.min())
    q = quality(minw, marg)
    r["_res"] = res
    r["_margin"] = marg
    r["_minw"] = minw
    r["_q"] = q
    if res > RES_TOL:
        err("%s: moment residual %.2e exceeds %.0e (rule is not exact to degree %d)" % (where, res, RES_TOL, n))
    if r.get("quality") != q:
        err("%s: stored quality %s but the data give %s" % (where, r.get("quality"), q))
    if m.group(4) != q:
        err("%s: quality in the id is %s but the data give %s" % (where, m.group(4), q))
    fq = fam.get("quality")
    if fq and q != fq:
        err("%s: family %s holds %s rules, this rule is %s" % (where, fam["id"], fq, q))


def attach_interval_proofs(fams):
    """Results of tools/verify_rigorous.py (reports/interval_proofs.json) for rule files that are unchanged since."""
    path = os.path.join(ROOT, "reports", "interval_proofs.json")
    rep = {}
    if os.path.exists(path):
        for e in load_json(path).get("rules", []):
            rep[e["id"]] = e
    for f in fams:
        for r in f["rules"]:
            e = rep.get(r["id"])
            if e and e.get("sha") == r["_sha"]:
                ex = e["exist"]
                r["_iv"] = {"ok": bool(e.get("ok_stored") and e.get("ok_exist")), "d": ex.get("dist"),
                            "q": ex.get("quality_proved"), "m": ex.get("mode")}
            else:
                r["_iv"] = None


def iv_text(iv, t):
    """Text of the interval proof for a rule page."""
    if not iv:
        return t["rpIvNone"]
    if not iv["ok"]:
        return t["rpIvNo"]
    m = iv["m"] or ""
    mode = t["ivSym"].replace("{g}", m[3:]) if m.startswith("sym") else t["ivFull"]
    return t["rpIvOk"].replace("{q}", iv["q"]).replace("{d}", "%.1e" % iv["d"]).replace("{m}", mode)


def check_unique(fams):
    seen = {}
    for f in fams:
        for r in f["rules"]:
            if r["id"] in seen:
                err("%s: id %s also used in %s" % (rel(r["_path"]), r["id"], seen[r["id"]]))
            seen[r["id"]] = rel(r["_path"])


def check_locked(base):
    """Refuse modification or deletion of rule files that were locked in the base revision."""
    try:
        out = subprocess.run(["git", "diff", "--name-status", base, "--", "rules"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout
    except Exception as e:  # noqa: BLE001
        err("cannot compare with %s (%s)" % (base, e))
        return
    for line in out.splitlines():
        parts = line.split("\t")
        status = parts[0]
        path = parts[1]
        if status.startswith("A") or not path.endswith(".json"):
            continue
        try:
            old = subprocess.run(["git", "show", "%s:%s" % (base, path)], cwd=ROOT,
                                 capture_output=True, text=True, check=True).stdout
            old_rule = json.loads(old)
        except Exception:  # noqa: BLE001
            continue
        if old_rule.get("locked"):
            err("%s: rule is locked and must not be changed or deleted (status %s)" % (path, status))


def check_i18n(strict):
    en = load_json(os.path.join(ROOT, "i18n", "en.json")) or {}
    de = load_json(os.path.join(ROOT, "i18n", "de.json")) or {}
    src = de.get("_source", {})
    for k in en:
        if k not in de:
            err("i18n/de.json: key '%s' missing (English: %s)" % (k, str(en[k])[:60]))
        elif src.get(k) != sha(json.dumps(en[k], ensure_ascii=False, sort_keys=True)):
            msg = "i18n/de.json: key '%s' may be stale, the English text changed (update the text and _source hash)" % k
            if strict:
                err(msg)
            else:
                warn(msg)
    for k in de:
        if k != "_source" and k not in en:
            warn("i18n/de.json: key '%s' has no English source" % k)
    doc_en = open(os.path.join(ROOT, "content", "en", "documentation.html"), encoding="utf8").read()
    doc_de = open(os.path.join(ROOT, "content", "de", "documentation.html"), encoding="utf8").read()
    mm = re.match(r"<!-- source: en/documentation.html source_hash: ([0-9a-f]+) -->\n", doc_de)
    if not mm:
        err("content/de/documentation.html: first line must be the source_hash comment")
    elif mm.group(1) != sha(doc_en):
        msg = "content/de/documentation.html: English documentation changed (source_hash %s, now %s)" % (mm.group(1), sha(doc_en))
        if strict:
            err(msg)
        else:
            warn(msg)
    stale_doc = bool(mm) and mm.group(1) != sha(doc_en)
    doc_de_body = doc_de[mm.end():] if mm else doc_de
    return en, {k: v for k, v in de.items() if k != "_source"}, doc_en, doc_de_body, stale_doc


def sync_hashes():
    """Mark every German text as translated from the current English text. Run only after checking the translation."""
    en = load_json(os.path.join(ROOT, "i18n", "en.json"))
    dp = os.path.join(ROOT, "i18n", "de.json")
    de = load_json(dp)
    de["_source"] = {k: sha(json.dumps(en[k], ensure_ascii=False, sort_keys=True)) for k in en if k in de}
    out = {"_source": de.pop("_source")}
    out.update(de)
    with open(dp, "w", encoding="utf8") as fh:
        fh.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    doc_en = open(os.path.join(ROOT, "content", "en", "documentation.html"), encoding="utf8").read()
    pp = os.path.join(ROOT, "content", "de", "documentation.html")
    doc_de = open(pp, encoding="utf8").read()
    doc_de = re.sub(r"^<!-- source: en/documentation.html source_hash: [0-9a-f]+ -->\n", "", doc_de)
    with open(pp, "w", encoding="utf8") as fh:
        fh.write("<!-- source: en/documentation.html source_hash: %s -->\n" % sha(doc_en) + doc_de)
    print("German source hashes updated")


# ---------------------------------------------------------------- best known

def best_known(fams):
    """best[(dom, n, kind)] = smallest node count over rules with degree >= n (kind PI or PO)."""
    best = {}
    for f in fams:
        for r in f["rules"]:
            q = r["_q"]
            kinds = []
            if q == "PI":
                kinds = ["PI", "PO"]
            elif q == "PO":
                kinds = ["PO"]
            for kind in kinds:
                for n in range(1, r["degree"] + 1):
                    key = (r["domain"], n, kind)
                    if key not in best or r["points"] < best[key][0]:
                        best[key] = (r["points"], r["id"])
    return best


# ---------------------------------------------------------------- site

def rule_url(r, lang="en", ext=".html"):
    pre = "" if lang == "en" else "de/"
    return "%srules/%s/%s/%s%s" % (pre, r["family"], r["domain"], r["id"], ext)


def payload(fams, en, de, doc_en, doc_de, stale_doc, lang):
    refs = load_json(os.path.join(ROOT, "content", "references.json"))
    lc = load_json(os.path.join(ROOT, "reference_data", "literature_counts.json"))["domains"]
    out_f = []
    for f in fams:
        rules = {}
        for r in sorted(f["rules"], key=lambda x: (x["domain"], x["degree"], x["points"])):
            d = DIM[r["domain"]]
            pts = [list(p) + [w] for p, w in zip(r["nodes"], r["weights"])]
            rules.setdefault(r["domain"], []).append({
                "id": r["id"],
                "deg": r["degree"],
                "n": r["points"],
                "q": r["_q"],
                "origin": r.get("origin", ""),
                "err": r["_res"],
                "cert": r.get("certificate"),
                "iv": r.get("_iv"),
                "url": rule_url(r, lang),
                "pts": pts,
            })
            del d
        out_f.append({k: f[k] for k in ("id", "tag", "role", "name", "short", "description", "references") if k in f} |
                     {"quality": f.get("quality"), "searched_degrees": f.get("searched_degrees", {}), "rules": rules})
    return {
        "lang": lang,
        "maxn": MAXN,
        "i18n": {"en": en, "de": de},
        "doc": {"en": doc_en, "de": doc_de},
        "docStale": stale_doc,
        "refs": refs,
        "gj": gj_tables(),
        "litcounts": lc,
        "links": {"repo": REPO_URL, "data": ("" if lang == "en" else "../") + "data/atlas.json",
                  "zip": ("" if lang == "en" else "../") + "downloads/cubature_atlas_rules.zip"},
        "families": out_f,
    }


def page_html(template, data, lang, en, de):
    t = en if lang == "en" else de
    other = "de/" if lang == "en" else "../"
    js = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    head = ('<title>%s</title>\n<meta name="description" content="%s">\n'
            '<link rel="canonical" href="%s">\n'
            '<link rel="alternate" hreflang="en" href="%s">\n<link rel="alternate" hreflang="de" href="%sde/">\n'
            '<link rel="alternate" hreflang="x-default" href="%s">\n') % (
        html.escape(t["pageTitle"]), html.escape(t["metaDesc"]),
        BASE_URL + ("" if lang == "en" else "de/"), BASE_URL, BASE_URL, BASE_URL)
    out = template.replace("<!--__HEAD__-->", head)
    out = out.replace("/*__ATLAS__*/null", js)
    out = out.replace('<html lang="en">', '<html lang="%s">' % lang)
    out = out.replace("__OTHER_LANG_URL__", other)
    return out


RULE_CSS = """body{font:16px/1.5 system-ui,sans-serif;max-width:980px;margin:0 auto;padding:24px;color:#101B26;background:#EDF0F3}
a{color:#0B6E8E}table{border-collapse:collapse;background:#fff}td,th{padding:4px 10px;border-bottom:1px solid #E4E9ED;text-align:left;vertical-align:top}
pre{background:#F6F8FA;border:1px solid #D2D9DF;border-radius:8px;padding:12px;overflow:auto;font:12.5px/1.45 ui-monospace,Menlo,monospace}
h1{font-weight:500}.ai{font:13px ui-monospace,Menlo,monospace;background:#E4E9ED;border-left:3px solid #0B6E8E;padding:8px 12px}"""


def rule_page(r, fam, best, lang, t):
    up = "../../../" if lang == "en" else "../../../../"
    d = DIM[r["domain"]]
    names = ["x", "y", "z"][:d] + ["w"]
    rows = ["  #  " + "  ".join("%22s" % s for s in names)]
    for i, (p, w) in enumerate(zip(r["nodes"], r["weights"])):
        rows.append("%4d " % (i + 1) + "  ".join("%22.15e" % v for v in list(p) + [w]))
    kind = "PI" if r["_q"] == "PI" else "PO"
    b = best.get((r["domain"], r["degree"], kind))
    if b and b[1] == r["id"]:
        bestline = "<p><b>%s</b></p>" % t["rpBest"]
    elif b:
        br = BYID.get(b[1])
        bestline = '<p>%s <a href="%s%s">%s</a> (%d)</p>' % (t["rpSuper"], up, rule_url(br, lang), b[1], b[0])
    else:
        bestline = ""
    cert = r.get("certificate")
    if cert:
        ctext = "h = %.2e, radius = %.2e, %s" % (cert.get("h", float("nan")), cert.get("rad", float("nan")),
                                                 t["rpYes"] if cert.get("cert") else t["rpNo"])
    else:
        ctext = "–"
    dname = t.get("dom_" + r["domain"], r["domain"])
    if fam["role"] == "literature":
        lic = html.escape(t["rpLicLit"])
    else:
        lic = '<a href="https://creativecommons.org/licenses/by/4.0/" rel="license">%s</a>' % html.escape(t["linkLicense"])
    cite = "Jörg Kubitz: Cubature Atlas. Zenodo. https://doi.org/%s, rule %s (%s, %s, degree %d, %d nodes), %s%s" % (
        CONCEPT_DOI, r["id"], fam["name"]["en"], r["domain"], r["degree"], r["points"], BASE_URL, rule_url(r, "en"))
    title = "%s – %s %s, %s %d, %d %s" % (r["id"], t["title"], dname, t["rpDegree"], r["degree"], r["points"], t["nodes"])
    alt_en = BASE_URL + rule_url(r, "en")
    alt_de = BASE_URL + rule_url(r, "de")
    body = """<!doctype html><html lang="%s"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>%s</title><meta name="description" content="%s">
<link rel="alternate" hreflang="en" href="%s"><link rel="alternate" hreflang="de" href="%s">
<style>%s</style></head><body>
<p><a href="%s">%s</a> · <a href="%srules/index.html">%s</a></p>
<h1>%s</h1>
<p class="ai">%s</p>
%s
<p>%s</p>
<table>
<tr><th>%s</th><td>%s<br><small>rules/%s/</small></td></tr>
<tr><th>%s</th><td>%s (%s)</td></tr>
<tr><th>%s</th><td>%d</td></tr>
<tr><th>%s</th><td>%d</td></tr>
<tr><th>%s</th><td>%s</td></tr>
<tr><th>%s</th><td>%.2e</td></tr>
<tr><th>%s</th><td>%.6e</td></tr>
<tr><th>%s</th><td>%.3e</td></tr>
<tr><th>%s</th><td>%s</td></tr>
<tr><th>%s</th><td>%s</td></tr>
<tr><th>%s</th><td>%s</td></tr>
</table>
<p><a href="%s%s">%s</a></p>
<h2>%s</h2>
<pre>%s</pre>
<h2>%s</h2>
<pre>%s</pre>
</body></html>
""" % (lang, html.escape(title), html.escape(fam["description"][lang][:150]), alt_en, alt_de, RULE_CSS,
       up + ("" if lang == "en" else "de/"), t["rpBack"], up + ("" if lang == "en" else "de/"), t["rpIndex"],
       html.escape(r["id"]), html.escape(t["ai"]), bestline,
       lic,
       t["rpFamily"], html.escape(fam["name"][lang]), fam["id"],
       t["rpDomain"], html.escape(dname), SYM[r["domain"]],
       t["rpDegree"], r["degree"], t["rpPoints"], r["points"], t["rpQuality"], r["_q"],
       t["rpResidual"], r["_res"], t["rpMinW"], r["_minw"], t["rpMargin"], r["_margin"],
       t["rpCert"], ctext, t["rpIv"], html.escape(iv_text(r.get("_iv"), t)), t["rpOrigin"], html.escape(r.get("method", r.get("source", ""))),
       up, rule_url(r, "en", ".json"), t["rpDownload"],
       t["rpCite"], html.escape(cite), t["rpNodes"], "\n".join(rows))
    return body


def index_page(fams, lang, t):
    up = "../" if lang == "en" else "../../"
    parts = []
    for f in fams:
        parts.append("<h2>%s</h2><p>%s<br><small>rules/%s/</small></p><ul>" % (
            html.escape(f["name"][lang]), html.escape(f["description"][lang]), f["id"]))
        for r in sorted(f["rules"], key=lambda x: (DOMAINS.index(x["domain"]), x["degree"], x["points"])):
            parts.append('<li><a href="%s/%s/%s.html">%s</a> – %s, %s %d, %d %s</li>' % (
                f["id"], r["domain"], r["id"], r["id"], html.escape(t.get("dom_" + r["domain"], r["domain"])),
                t["rpDegree"], r["degree"], r["points"], t["nodes"]))
        parts.append("</ul>")
    return """<!doctype html><html lang="%s"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>%s – %s</title><style>%s</style></head><body><p><a href="%s">%s</a></p><h1>%s</h1>%s</body></html>
""" % (lang, t["rpIndex"], t["title"], RULE_CSS, up + ("" if lang == "en" else "de/"), t["rpBack"], t["rpIndex"], "".join(parts))


BYID = {}


def build_site(fams, en, de, doc_en, doc_de, stale_doc):
    if os.path.exists(SITE):
        shutil.rmtree(SITE)
    os.makedirs(SITE)
    for lang in ("en", "de"):
        t = en if lang == "en" else de
        for dom in DOMAINS:
            t.setdefault("dom_" + dom, dom)
    template = open(os.path.join(ROOT, "site_template", "index.html"), encoding="utf8").read()
    # Files in site_template/static/ are copied unchanged to the site root
    # (for example the Google Search Console verification file).
    static_dir = os.path.join(ROOT, "site_template", "static")
    if os.path.isdir(static_dir):
        for name in sorted(os.listdir(static_dir)):
            shutil.copy(os.path.join(static_dir, name), os.path.join(SITE, name))
    best = best_known(fams)
    for f in fams:
        for r in f["rules"]:
            BYID[r["id"]] = r
    for lang in ("en", "de"):
        data = payload(fams, en, de, doc_en, doc_de, stale_doc, lang)
        target = SITE if lang == "en" else os.path.join(SITE, "de")
        os.makedirs(target, exist_ok=True)
        with open(os.path.join(target, "index.html"), "w", encoding="utf8") as fh:
            fh.write(page_html(template, data, lang, en, de))
        if lang == "en":
            os.makedirs(os.path.join(SITE, "data"))
            with open(os.path.join(SITE, "data", "atlas.json"), "w", encoding="utf8") as fh:
                json.dump({k: data[k] for k in ("families", "refs", "litcounts")}, fh, ensure_ascii=False)
    urls = [BASE_URL, BASE_URL + "de/"]
    for lang in ("en", "de"):
        t = en if lang == "en" else de
        base = SITE if lang == "en" else os.path.join(SITE, "de")
        os.makedirs(os.path.join(base, "rules"), exist_ok=True)
        with open(os.path.join(base, "rules", "index.html"), "w", encoding="utf8") as fh:
            fh.write(index_page(fams, lang, t))
        urls.append(BASE_URL + ("" if lang == "en" else "de/") + "rules/index.html")
        for f in fams:
            for r in f["rules"]:
                p = os.path.join(SITE, rule_url(r, lang))
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "w", encoding="utf8") as fh:
                    fh.write(rule_page(r, f, best, lang, t))
                urls.append(BASE_URL + rule_url(r, lang))
                if lang == "en":
                    shutil.copy(r["_path"], os.path.join(SITE, rule_url(r, "en", ".json")))
    for f in fams:
        dst = os.path.join(SITE, "rules", f["id"], "family.json")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(os.path.join(RULES, f["id"], "family.json"), dst)
    os.makedirs(os.path.join(SITE, "downloads"))
    with zipfile.ZipFile(os.path.join(SITE, "downloads", "cubature_atlas_rules.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for f in fams:
            z.write(os.path.join(RULES, f["id"], "family.json"), "rules/%s/family.json" % f["id"])
            for r in f["rules"]:
                z.write(r["_path"], rel(r["_path"]))
    with open(os.path.join(SITE, "sitemap.xml"), "w", encoding="utf8") as fh:
        fh.write('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n')
        for u in urls:
            fh.write("  <url><loc>%s</loc></url>\n" % html.escape(u))
        fh.write("</urlset>\n")
    with open(os.path.join(SITE, "robots.txt"), "w") as fh:
        fh.write("User-agent: *\nAllow: /\nSitemap: %ssitemap.xml\n" % BASE_URL)
    with open(os.path.join(SITE, ".nojekyll"), "w") as fh:
        fh.write("")
    return len(urls)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify only, do not build the site")
    ap.add_argument("--base", help="git revision to compare with; locked rules must be unchanged")
    ap.add_argument("--strict", action="store_true", help="treat stale translations as errors")
    ap.add_argument("--sync-hashes", action="store_true", help="record the current English texts as source of the German ones")
    args = ap.parse_args()
    if args.sync_hashes:
        sync_hashes()
        return
    fams = load_families()
    for f in fams:
        load_rules(f)
    check_unique(fams)
    attach_interval_proofs(fams)
    if args.base:
        check_locked(args.base)
    en, de, doc_en, doc_de, stale_doc = check_i18n(args.strict)
    total = sum(len(f["rules"]) for f in fams)
    for f in fams:
        doms = sorted({r["domain"] for r in f["rules"]}, key=DOMAINS.index)
        worst = max([r.get("_res", 0.0) for r in f["rules"]] or [0.0])
        print("%-40s %-10s %4d rules  %s  max residual %.1e" % (f["id"], f["role"], len(f["rules"]), ",".join(doms), worst))
    for w in warnings:
        print("WARNING:", w)
    if errors:
        for e in errors:
            print("ERROR:", e)
        print("%d error(s) in %d rules" % (len(errors), total))
        sys.exit(1)
    print("OK: %d families, %d rules verified" % (len(fams), total))
    if not args.check:
        n = build_site(fams, en, de, doc_en, doc_de, stale_doc)
        print("site/ built: %d pages in the sitemap" % n)


if __name__ == "__main__":
    main()

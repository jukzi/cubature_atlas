"""Cubature Atlas: verify all rule files and build the static website.

Usage:
    python tools/build_atlas.py --check              # verify only (used on pull requests)
    python tools/build_atlas.py                      # verify and build site/
    python tools/build_atlas.py --base origin/main --changed   # before a commit: check only the rules this change adds or modifies
    python tools/build_atlas.py --strict             # stale translations are errors, not warnings
    python tools/build_atlas.py --sync-hashes        # after updating German texts: mark them as current

Needs only Python 3.9+ and numpy.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import glob
import hashlib
import html
import json
import math
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import zipfile
from itertools import product

# Keep BLAS/OpenMP strictly single-threaded per process so parallel verification
# does not oversubscribe CPU threads (must be set before importing numpy).
BLAS_THREAD_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for _v in BLAS_THREAD_VARS:
    os.environ[_v] = "1"

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RULES = os.path.join(ROOT, "rules")
SITE = os.path.join(ROOT, "site")
BASE_URL = "https://jukzi.github.io/cubature_atlas/"
REPO_URL = "https://github.com/jukzi/cubature_atlas"
CONCEPT_DOI = "10.5281/zenodo.24037245"
MIN_MAXN = 50

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
ID_RE = re.compile(r"^CA-(C1|T2|C2|T3|C3)-(\d{2,})-(\d+)-(PI|PO|NI|NO)-([A-Za-z0-9]+)(-[a-z])?$")
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
    worst = 0.0
    m_tot = len(exps)
    for start in range(0, m_tot, 2048):
        sl = slice(start, min(start + 2048, m_tot))
        e_sl = exps[sl]
        vals = powers[0][e_sl[:, 0]]
        for c in range(1, d):
            vals = vals * powers[c][e_sl[:, c]]
        q = vals @ W
        w = float(np.max(np.abs(q - exact[sl])))
        if w > worst:
            worst = w
    return worst / VOL[dom]


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


def max_degree(fams, lc=None):
    """Highest degree present in any rule file or literature count table (at least MIN_MAXN)."""
    m = MIN_MAXN
    for f in fams:
        for r in f["rules"]:
            if r["degree"] > m:
                m = r["degree"]
    if lc:
        for cols in lc.values():
            for c in cols:
                for idx, val in enumerate(c.get("arr") or [], 1):
                    if val is not None and idx > m:
                        m = idx
    return m


def gj_tables(maxn=MIN_MAXN):
    out = {"L": {}, "J1": {}, "J2": {}}
    for m in range(1, maxn // 2 + 2):
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


def _load_and_check_rule(task):
    """Worker-safe rule loader and verifier: returns (rule_dict_or_None, error_messages)."""
    fam_id, fam_tag, fam_quality, dom, fn, path, do_check = task
    where = rel(path)
    try:
        with open(path, encoding="utf8") as fh:
            raw = fh.read()
        r = json.loads(raw)
    except Exception as e:  # noqa: BLE001
        return None, ["%s: cannot read JSON (%s)" % (where, e)]
    r["_path"] = path
    r["_sha"] = content_sha(raw)
    errs = []
    if do_check:
        errs = _check_rule_data(fam_id, fam_tag, fam_quality, dom, r, fn)
    return r, errs


def load_rules(fam, only=None):
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
            do_check = only is None or os.path.abspath(path) in only
            r, r_errs = _load_and_check_rule((fam["id"], fam.get("tag"), fam.get("quality"), dom, fn, path, do_check))
            for m in r_errs:
                err(m)
            if r is not None:
                fam["rules"].append(r)


def load_all_rules(fams, only=None, jobs=None):
    """Load and verify all rules across families, optionally in parallel using single-threaded spawned workers."""
    tasks = []
    for fi, fam in enumerate(fams):
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
                do_check = only is None or os.path.abspath(path) in only
                tasks.append((fi, (fam["id"], fam.get("tag"), fam.get("quality"), dom, fn, path, do_check)))
    checked_count = sum(1 for _, t in tasks if t[6])
    if jobs is None or jobs <= 0:
        jobs = min(os.cpu_count() or 1, 8)
    jobs = max(1, min(jobs, len(tasks) or 1))
    results = [None] * len(tasks)
    if jobs > 1 and checked_count >= 4:
        # Longest-processing-time-first (largest files first) avoids tail stragglers on 3D high-degree rules.
        order = sorted(range(len(tasks)), key=lambda i: (tasks[i][1][6], os.path.getsize(tasks[i][1][5])), reverse=True)
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
            for idx, res in zip(order, ex.map(_load_and_check_rule, [tasks[i][1] for i in order])):
                results[idx] = res
    else:
        for idx, (_, task) in enumerate(tasks):
            results[idx] = _load_and_check_rule(task)
    for (fi, _), (r, r_errs) in zip(tasks, results):
        for m in r_errs:
            err(m)
        if r is not None:
            fams[fi]["rules"].append(r)


def content_sha(text):
    """Same hash as tools/verify_rigorous.py: domain, degree, quality and the exact decimal strings of
    nodes and weights. Metadata can change without invalidating a proof."""
    r = json.loads(text, parse_float=str, parse_int=str)
    core = {k: r.get(k) for k in ("domain", "degree", "quality", "nodes", "weights")}
    blob = json.dumps(core, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf8")).hexdigest()[:16]


def _check_rule_data(fam_id, fam_tag, fam_quality, dom, r, fn):
    errs = []
    where = rel(r["_path"])
    rid = r.get("id", "")
    if fn != rid + ".json":
        errs.append("%s: file name must be <id>.json (id is '%s')" % (where, rid))
    m = ID_RE.match(rid)
    if not m:
        errs.append("%s: id '%s' does not match CA-<domain>-<degree 2 digits>-<points>-<quality>-<tag>[-a]" % (where, rid))
        return errs
    if r.get("family") != fam_id:
        errs.append("%s: family must be '%s'" % (where, fam_id))
    if r.get("domain") != dom:
        errs.append("%s: domain must be '%s' (directory)" % (where, dom))
    d = DIM[dom]
    nodes = r.get("nodes")
    weights = r.get("weights")
    if not isinstance(nodes, list) or not isinstance(weights, list) or len(nodes) != len(weights) or not nodes:
        errs.append("%s: nodes and weights must be non-empty lists of equal length" % where)
        return errs
    if any(not isinstance(p, list) or len(p) != d for p in nodes):
        errs.append("%s: every node needs %d coordinates" % (where, d))
        return errs
    X = np.array(nodes, dtype=float)
    W = np.array(weights, dtype=float)
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(W)):
        errs.append("%s: non-finite numbers" % where)
        return errs
    n = r.get("degree")
    if not isinstance(n, int) or n < 1:
        errs.append("%s: degree must be a positive integer" % where)
        return errs
    if r.get("points") != len(W):
        errs.append("%s: points=%s but %d nodes stored" % (where, r.get("points"), len(W)))
    if m.group(1) != SYM[dom] or int(m.group(2)) != n or int(m.group(3)) != len(W) or m.group(5) != fam_tag:
        errs.append("%s: id does not agree with domain/degree/points/family tag" % where)
    res = moment_residual(dom, n, X, W)
    marg = margin(dom, X)
    minw = float(W.min())
    q = quality(minw, marg)
    r["_res"] = res
    r["_margin"] = marg
    r["_minw"] = minw
    r["_q"] = q
    if res > RES_TOL:
        errs.append("%s: moment residual %.2e exceeds %.0e (rule is not exact to degree %d)" % (where, res, RES_TOL, n))
    if r.get("quality") != q:
        errs.append("%s: stored quality %s but the data give %s" % (where, r.get("quality"), q))
    if m.group(4) != q:
        errs.append("%s: quality in the id is %s but the data give %s" % (where, m.group(4), q))
    if fam_quality and q != fam_quality:
        errs.append("%s: family %s holds %s rules, this rule is %s" % (where, fam_id, fam_quality, q))
    if "certificate" in r:
        errs.append("%s: the floating-point check is not a proof; store it as \"numerical_check\", not \"certificate\"" % where)
    if r.get("status") != "proven":
        errs.append("%s: status must be \"proven\" (every rule needs an interval proof, see CONTRIBUTING.md)" % where)
    return errs


def check_rule(fam, dom, r, fn):
    for msg in _check_rule_data(fam["id"], fam.get("tag"), fam.get("quality"), dom, r, fn):
        err(msg)


def attach_interval_proofs(fams, only=None):
    """Result of tools/verify_rigorous.py for rule files that are unchanged since; the proof of a rule is the
    file next to it with the extension .proof instead of .json."""
    for f in fams:
        for r in f["rules"]:
            if only is not None and os.path.abspath(r["_path"]) not in only:
                continue   # --changed: rules this change does not touch are not checked here
            pp = os.path.splitext(r["_path"])[0] + ".proof"
            e = load_json(pp) if os.path.exists(pp) else None
            if e and e.get("sha") == r["_sha"]:
                ex = e["exist"]
                r["_iv"] = {"ok": bool(e.get("ok_stored") and e.get("ok_exist")), "d": ex.get("dist"),
                            "q": ex.get("quality_proved"), "m": ex.get("mode")}
            else:
                r["_iv"] = None
            iv = r["_iv"]
            if iv is None:
                err("%s: no interval proof for these numbers (%s); run "
                    "python tools/verify_rigorous.py --strict --save %s" % (rel(r["_path"]), rel(pp), rel(r["_path"])))
            elif not iv["ok"] or iv["q"] != r.get("quality"):
                err("%s: the interval proof does not prove the stored quality %s (proved: %s)" % (rel(r["_path"]), r.get("quality"), iv["q"]))
    for pp in glob.glob(os.path.join(RULES, "*", "*", "*.proof")):
        if not os.path.exists(os.path.splitext(pp)[0] + ".json"):
            err("%s: proof without a rule file" % rel(pp))


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


def changed_rule_paths(base):
    """Absolute paths of the rule files this change adds or modifies against `base`
    (committed, staged, unstaged and new untracked files)."""
    paths = set()
    try:
        out = subprocess.run(["git", "diff", "--name-status", base, "--", "rules"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout
        for line in out.splitlines():
            parts = line.split("\t")
            if parts[0][:1] in ("A", "M", "R", "C") and parts[-1].endswith(".json"):
                paths.add(os.path.abspath(os.path.join(ROOT, parts[-1])))
        new = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "--", "rules"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout
        for line in new.splitlines():
            if line.endswith(".json"):
                paths.add(os.path.abspath(os.path.join(ROOT, line)))
    except Exception as e:  # noqa: BLE001
        err("cannot determine the changed rules against %s (%s)" % (base, e))
    return {p for p in paths if os.path.basename(p) != "family.json"}


def check_references():
    """content/references.json must be sorted by year (undated entries last) and numbered 1, 2, 3, ..."""
    refs = load_json(os.path.join(ROOT, "content", "references.json"))
    numbers = [r["n"] for r in refs]
    if numbers != list(range(1, len(refs) + 1)):
        err("content/references.json: the numbers n must be 1, 2, 3, ... in the order of the list")
    keys = []
    for r in refs:
        year = str(r.get("y", "")).strip()
        keys.append((0, int(year)) if year.isdigit() else (1, 0))
    if keys != sorted(keys):
        err("content/references.json is not sorted by year (undated entries last); run python tools/sort_references.py")


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


def domain_pts(fams):
    """Return per-domain JSON strings ({rule_id: [[x, ..., w], ...]}) and 8-char SHA-256 hashes for lazy loading."""
    by_dom = {d: {} for d in DOMAINS}
    for f in fams:
        for r in f["rules"]:
            by_dom[r["domain"]][r["id"]] = [list(p) + [w] for p, w in zip(r["nodes"], r["weights"])]
    chunks, hashes = {}, {}
    for d in DOMAINS:
        raw = json.dumps(by_dom[d], ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        chunks[d] = raw
        hashes[d] = hashlib.sha256(raw.encode("utf8")).hexdigest()[:8]
    return chunks, hashes


def payload(fams, en, de, doc_en, doc_de, stale_doc, lang, include_pts=True, pts_hashes=None):
    refs = load_json(os.path.join(ROOT, "content", "references.json"))
    lc = load_json(os.path.join(ROOT, "reference_data", "literature_counts.json"))["domains"]
    maxn = max_degree(fams, lc)
    out_f = []
    for f in fams:
        rules = {}
        for r in sorted(f["rules"], key=lambda x: (x["domain"], x["degree"], x["points"])):
            entry = {
                "id": r["id"],
                "deg": r["degree"],
                "n": r["points"],
                "q": r["_q"],
                "origin": r.get("origin", ""),
                "err": r["_res"],
                "cert": r.get("numerical_check"),
                "iv": r.get("_iv"),
                "url": rule_url(r, lang),
                "method": r.get("method", "") if f.get("role") == "pi" else "",
            }
            if include_pts:
                entry["pts"] = [list(p) + [w] for p, w in zip(r["nodes"], r["weights"])]
            rules.setdefault(r["domain"], []).append(entry)
        out_f.append({k: f[k] for k in ("id", "tag", "role", "name", "short", "description", "references") if k in f} |
                     {"quality": f.get("quality"), "searched_degrees": f.get("searched_degrees", {}), "rules": rules})
    pre = "" if lang == "en" else "../"
    links = {"repo": REPO_URL, "data": pre + "data/atlas.json",
             "zip": pre + "downloads/cubature_atlas_rules.zip"}
    res = {
        "lang": lang,
        "maxn": maxn,
        "i18n": {"en": en, "de": de},
        "doc": {"en": doc_en, "de": doc_de},
        "docStale": stale_doc,
        "refs": refs,
        "gj": gj_tables(maxn),
        "litcounts": lc,
        "links": links,
        "families": out_f,
    }
    if not include_pts:
        links["ptsBase"] = pre + "data/"
        res["ptsHash"] = pts_hashes or {}
    return res


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
    cert = r.get("numerical_check")
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


def render_404(fams):
    """Build a single 404.html router that redirects deep application paths (<dom>/<src>/<n>/ and de/<dom>/<src>/<n>/)
    to the index page with query parameters, replacing thousands of static redirect stub files."""
    codes = {"npi": "found", "npo": "out", "lit": "lit", "gauss": "gauss"}
    srcs = {}
    for dom in DOMAINS:
        m = {"gauss": "gauss", "lob": "lob", "cc": "cc", "nc": "nc"} if dom == "interval" else dict(codes)
        for f in fams:
            if f["role"] not in ("pi", "po") and any(r["domain"] == dom for r in f["rules"]):
                m[f["id"]] = "f:" + f["id"]
        srcs[dom] = m
    src_js = json.dumps(srcs, separators=(",", ":"), sort_keys=True)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="robots" content="noindex">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Cubature Atlas</title><style>%s</style><script>\n'
        '(function(){\n'
        'var SRC=%s;\n'
        'var m=location.pathname.match(/^(\\/(?:[^\\/]+\\/)?(?:pr-preview\\/pr-\\d+\\/)?)(de\\/)?(interval|triangle|square|tetra|cube)(?:\\/([^\\/]+)(?:\\/(\\d+))?)?\\/?$/);\n'
        'if(!m)return;\n'
        'var base=m[1]+(m[2]||""),dom=m[3],sc=m[4],n=m[5];\n'
        'var q=["dom="+dom];\n'
        'if(sc){var sm=SRC[dom];if(!sm||!Object.prototype.hasOwnProperty.call(sm,sc))return;q.push("src="+encodeURIComponent(sm[sc]));}\n'
        'if(n){if(parseInt(n,10)<1)return;q.push("n="+n);}\n'
        'var extra=location.search?location.search.slice(1):"";\n'
        'location.replace(base+"?"+q.join("&")+(extra?"&"+extra:"")+location.hash);\n'
        '})();\n'
        '</script></head><body><h1>404 – Page not found</h1>'
        '<p><a href="%s">Cubature Atlas (EN)</a> · <a href="%sde/">Cubature Atlas (DE)</a></p>'
        '</body></html>\n'
    ) % (RULE_CSS, src_js, BASE_URL, BASE_URL)


def write_404(fams):
    with open(os.path.join(SITE, "404.html"), "w", encoding="utf8") as fh:
        fh.write(render_404(fams))


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
    maxn = MIN_MAXN
    pts_chunks, pts_hashes = domain_pts(fams)
    os.makedirs(os.path.join(SITE, "data"))
    for dom, raw in pts_chunks.items():
        with open(os.path.join(SITE, "data", "pts-%s.json" % dom), "w", encoding="utf8") as fh:
            fh.write(raw)
    for lang in ("en", "de"):
        data = payload(fams, en, de, doc_en, doc_de, stale_doc, lang, include_pts=False, pts_hashes=pts_hashes)
        maxn = data["maxn"]
        target = SITE if lang == "en" else os.path.join(SITE, "de")
        os.makedirs(target, exist_ok=True)
        with open(os.path.join(target, "index.html"), "w", encoding="utf8") as fh:
            fh.write(page_html(template, data, lang, en, de))
        if lang == "en":
            full = payload(fams, en, de, doc_en, doc_de, stale_doc, lang, include_pts=True)
            with open(os.path.join(SITE, "data", "atlas.json"), "w", encoding="utf8") as fh:
                json.dump({k: full[k] for k in ("families", "refs", "litcounts")}, fh, ensure_ascii=False)
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
    write_404(fams)
    return len(urls)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify only, do not build the site")
    ap.add_argument("--base", help="git revision to compare with (with --changed: check only the rules this change adds or modifies)")
    ap.add_argument("--strict", action="store_true", help="treat stale translations as errors")
    ap.add_argument("--changed", action="store_true",
                    help="with --base: check only the rule files this change adds or modifies (before a commit); implies --check")
    ap.add_argument("--sync-hashes", action="store_true", help="record the current English texts as source of the German ones")
    ap.add_argument("-j", "--jobs", type=int, default=0,
                    help="number of single-threaded worker processes for rule verification (0 = auto, 1 = sequential)")
    args = ap.parse_args()
    if args.sync_hashes:
        sync_hashes()
        return
    only = None
    if args.changed:
        if not args.base:
            ap.error("--changed needs --base")
        only = changed_rule_paths(args.base)
        args.check = True
    fams = load_families()
    load_all_rules(fams, only, jobs=args.jobs)
    check_unique(fams)
    attach_interval_proofs(fams, only)
    check_references()
    en, de, doc_en, doc_de, stale_doc = check_i18n(args.strict)
    total =sum(len(f["rules"]) for f in fams)
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
    if only is not None:
        print("OK: %d changed rule file(s) checked; the other %d rules were not checked here (the check in CI verifies all rules)" % (len(only), total - len(only)))
    else:
        print("OK: %d families, %d rules verified" % (len(fams), total))
    if not args.check:
        n = build_site(fams, en, de, doc_en, doc_de, stale_doc)
        print("site/ built: %d pages in the sitemap" % n)


if __name__ == "__main__":
    main()

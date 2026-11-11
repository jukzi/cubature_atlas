#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_cannonball.py
==================

One unattended search attempt for a better tetrahedron rule, used by the
GitHub workflow `.github/workflows/auto_cannonball.yml`.

What it does
------------
1. Reads all PI tetrahedron rules of the atlas and the branches of open
   automatic pull requests (`ai/auto-tet-nXX-mYYY`) to know the best node
   count for the degree.
2. Runs code/cannonball_start.py: cannonball packing of ceil(Z(n)) + extra
   balls as start, degree continuation with bounded least squares (scipy trf),
   then node elimination until the time budget is used up.  The program writes
   a snapshot after every elimination round, so a result survives a timeout.
3. Takes the smallest exact PI result.  If it has fewer nodes than every known
   rule of at least that degree: snaps barycentric coordinates below 1e-13 onto
   the boundary, computes the Newton-Kantorovich certificate
   (code/cubature_atlas.py), writes the rule file into
   rules/minimal_pointcount_for_degree_n_PI/tetra/, runs
   `python tools/verify_rigorous.py --strict --merge reports/interval_proofs.json <rule>` (interval proof)
   and `python tools/build_atlas.py --base origin/main`, and requires both to succeed.
4. Writes the pull-request title and body and, for GitHub Actions, the step
   outputs found / branch / title / rule_id / rule_path / body.

Usage
-----
   python tools/auto_cannonball.py --degree 24 --extra 58 --seed 3 --minutes 320
   python tools/auto_cannonball.py --degree 8 --extra 4 --seed 1 --minutes 3

Style rule of this file: one command per line.
"""

# Imports.
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import numpy as np

# Repository root (this file lives in tools/).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Folder of the PI family for the tetrahedron.
PI_DIR = os.path.join(ROOT, "rules", "minimal_pointcount_for_degree_n_PI", "tetra")
# Search program.
SEARCH = os.path.join(ROOT, "code", "cannonball_start.py")
# Largest moment error accepted for a candidate (the build check is stricter in its own way).
MAX_ERR = 1e-13


def log(msg):
    """Print a progress line immediately."""
    # Flush so that the GitHub log shows it at once.
    print("[auto_cannonball] " + msg, flush=True)


def atlas_tet_rules():
    """All PI tetrahedron rules of the atlas as (degree, points, path)."""
    # Result list.
    out = []
    # All rule files of all families for the tetrahedron.
    for path in glob.glob(os.path.join(ROOT, "rules", "*", "tetra", "*.json")):
        # Load the rule.
        with open(path) as fh:
            rule = json.load(fh)
        # Only rules with positive weights and nodes in the closed tetrahedron.
        if rule.get("quality") != "PI":
            continue
        # Store degree, point count and file.
        out.append((int(rule["degree"]), int(rule["points"]), path))
    # Return the list.
    return out


def open_auto_prs():
    """Degrees and point counts of open automatic pull requests (from branch names)."""
    # Result list.
    out = []
    # Ask the remote for the branches of automatic searches.
    try:
        res = subprocess.run(["git", "ls-remote", "--heads", "origin", "ai/auto-tet-*"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    except Exception:
        return out
    # Parse the branch names.
    for line in res.stdout.splitlines():
        # Branch name pattern ai/auto-tet-nXX-mYYY.
        m = re.search(r"ai/auto-tet-n(\d+)-m(\d+)", line)
        # Keep matches.
        if m:
            out.append((int(m.group(1)), int(m.group(2)), line.split()[-1]))
    # Return the list.
    return out


def best_known(entries, n):
    """Smallest point count over all entries of degree >= n (None if there is none)."""
    # Candidate counts.
    counts = [p for d, p, _ in entries if d >= n]
    # Minimum or None.
    return min(counts) if counts else None


def run_search(n, extra, seed, minutes, maxfev, out):
    """Run code/cannonball_start.py within the time budget."""
    # Create the output folder.
    os.makedirs(out, exist_ok=True)
    # Budget in seconds.
    budget = 60.0 * minutes
    # Command line: elimination may use the whole remaining budget.
    cmd = [sys.executable, SEARCH, "--n", str(n), "--extra", str(extra), "--seed", str(seed), "--solver", "trf", "--maxfev", str(maxfev), "--budget", "%.0f" % budget, "--eliminate", "%.0f" % budget, "--out", out]
    # Log the command.
    log("running: " + " ".join(cmd))
    # Run with a safety timeout (one elimination step may overrun the budget).
    try:
        subprocess.run(cmd, cwd=os.path.join(ROOT, "code"), timeout=budget + 900)
    except subprocess.TimeoutExpired:
        log("search exceeded its time limit; using the files written so far")


def best_result(out, n):
    """Smallest exact PI result of degree n written by the search: (points, X, W, data) or None."""
    # Best candidate.
    best = None
    # Loop over the output files.
    for path in glob.glob(os.path.join(out, "tet_n*_m*_*.json")):
        # Load the result.
        with open(path) as fh:
            data = json.load(fh)
        # Only the target degree.
        if int(data["degree"]) != n:
            continue
        # Only PI.
        if data.get("quality") != "PI":
            continue
        # Only exact rules.
        if not float(data.get("max_moment_error_pkd", 1.0)) < MAX_ERR:
            continue
        # Nodes.
        X = np.array(data["nodes"], dtype=float)
        # Weights (volume 1/6).
        W = np.array(data["weights"], dtype=float)
        # Keep the smallest one.
        if best is None or len(W) < best[0]:
            best = (len(W), X, W, data)
    # Return it.
    return best


def snap(X):
    """Put barycentric coordinates below 1e-13 exactly onto the boundary."""
    # Barycentric coordinates.
    L = np.c_[1.0 - X.sum(axis=1), X]
    # Snap tiny values.
    L[np.abs(L) < 1e-13] = 0.0
    # Renormalise.
    L = L / L.sum(axis=1, keepdims=True)
    # Cartesian coordinates.
    return L[:, 1:].copy()


def certify(degree, X, W):
    """Newton-Kantorovich certificate with the atlas program code/cubature_atlas.py."""
    # Make the atlas code importable.
    sys.path.insert(0, os.path.join(ROOT, "code"))
    # Import the atlas module.
    import cubature_atlas
    # Compute.
    try:
        return cubature_atlas.certificate("tetra", degree, X, W)
    except Exception as exc:
        log("certificate failed: %r" % (exc,))
        return None


def write_rule(degree, X, W, cert, start):
    """Write the rule file in the atlas format; return (id, path)."""
    # Identifier.
    rid = "CA-T3-%02d-%d-PI-MPC" % (degree, len(X))
    # Path.
    path = os.path.join(PI_DIR, rid + ".json")
    # Certified?
    ok = bool(cert is not None and cert.get("certified"))
    # Header fields.
    head = {
        "id": rid,
        "family": "minimal_pointcount_for_degree_n_PI",
        "domain": "tetra",
        "dimension": 3,
        "degree": int(degree),
        "points": int(len(X)),
        "quality": "PI",
        "origin": "search",
        "method": "automatic search in GitHub Actions (tools/auto_cannonball.py): code/cannonball_start.py, %s; no symmetry; checked numerically and proven with interval arithmetic (tools/verify_rigorous.py)" % start,
        "author": "Claude (Anthropic), commissioned and directed by Joerg Kubitz; found by the automatic search in GitHub Actions",
        "status": "proven",
        "locked": False,
        "residual_reported": float(cert["resid"]) if cert is not None and "resid" in cert else 1e-15,
    }
    # Certificate block (only if the computation ran).
    if cert is not None and "sigma" in cert and "h" in cert:
        head["numerical_check"] = {"N": int(len(X)), "M": int(cert["M"]), "p": int(cert["p"]), "res": float(cert["resid"]), "sigma": float(cert["sigma"]), "h": float(cert["h"]), "rad": float(cert["radius"]), "cert": bool(cert["certified"]), "inside": bool(cert["inside"])}
    # Lines of the file in the layout of the existing rule files.
    lines = ["{"]
    # Header fields, one per line.
    for key, val in head.items():
        lines.append("  %s: %s," % (json.dumps(key), json.dumps(val)))
    # Nodes, one per line.
    lines.append('  "nodes": [')
    # Join the node rows.
    lines.append(",\n".join("    " + json.dumps([float(v) for v in row]) for row in X))
    # Close the node list.
    lines.append("  ],")
    # Weights, one per line.
    lines.append('  "weights": [')
    # Join the weights.
    lines.append(",\n".join("    " + json.dumps(float(w)) for w in W))
    # Close the weight list.
    lines.append("  ]")
    # Close the object.
    lines.append("}")
    # Write.
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    # Return id and path.
    return rid, path


def build_check(rule_path):
    """Prove the rule with interval arithmetic (recorded in reports/interval_proofs.json), then run the
    atlas check against origin/main; return (ok, last lines of the output)."""
    # Interval proof of the new rule; the build requires it.
    proof = subprocess.run([sys.executable, "tools/verify_rigorous.py", "--strict", "--merge", "reports/interval_proofs.json", rule_path], cwd=ROOT, capture_output=True, text=True)
    # Run the build.
    res = subprocess.run([sys.executable, "tools/build_atlas.py", "--base", "origin/main"], cwd=ROOT, capture_output=True, text=True)
    # Combined output.
    text = proof.stdout + proof.stderr + res.stdout + res.stderr
    # A failed proof fails the check.
    if proof.returncode != 0:
        return False, "\n".join(text.strip().splitlines()[-6:])
    # Success marker.
    ok = res.returncode == 0 and re.search(r"^OK", text, re.M) is not None
    # Return the verdict and the tail of the output.
    return ok, "\n".join(text.strip().splitlines()[-6:])


def set_output(path, values):
    """Append key=value lines to the GitHub step output file (if given)."""
    # Nothing to do outside GitHub Actions.
    if not path:
        return
    # Append.
    with open(path, "a") as fh:
        # One line per value.
        for key, val in values.items():
            fh.write("%s=%s\n" % (key, val))


def main():
    """Parse the arguments and perform one search attempt."""
    # Argument parser.
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Degree.
    ap.add_argument("--degree", type=int, default=24)
    # Points above ceil(Z(n)).
    ap.add_argument("--extra", type=int, default=58)
    # Seed of the packing.
    ap.add_argument("--seed", type=int, default=1)
    # Time budget in minutes.
    ap.add_argument("--minutes", type=float, default=320.0)
    # Function evaluations per degree of the bounded solver.
    ap.add_argument("--maxfev", type=int, default=400)
    # Working folder.
    ap.add_argument("--work", default=os.path.join(ROOT, "auto_cannonball_work"))
    # File for the pull-request body.
    ap.add_argument("--pr-body", default=os.path.join(ROOT, "auto_cannonball_work", "pr_body.md"))
    # GitHub step output file.
    ap.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT", ""))
    # Parse.
    args = ap.parse_args()
    # Degree.
    n = args.degree
    # Known rules of the atlas.
    atlas = atlas_tet_rules()
    # Best known count including open automatic pull requests.
    target = best_known(atlas + open_auto_prs(), n)
    # Report the plan.
    log("degree %d, extra %d, seed %d, %.0f min, maxfev %d; best known %s" % (n, args.extra, args.seed, args.minutes, args.maxfev, target))
    # Default outputs.
    outputs = {"found": "false", "degree": n}
    # Fresh working folder.
    shutil.rmtree(args.work, ignore_errors=True)
    # Output folder of the search.
    out = os.path.join(args.work, "out")
    # Run the search.
    run_search(n, args.extra, args.seed, args.minutes, args.maxfev, out)
    # Best result.
    res = best_result(out, n)
    # Nothing usable.
    if res is None:
        log("no exact PI rule written by the search")
        set_output(args.github_output, outputs)
        return
    # Unpack.
    points, X, W, data = res
    # Report.
    log("search result: %d points (best known %s)" % (points, target))
    # Re-read the open pull requests (another job may have finished meanwhile).
    target = best_known(atlas + open_auto_prs(), n)
    # Not an improvement.
    if target is not None and points >= target:
        log("no improvement")
        set_output(args.github_output, outputs)
        return
    # Snap boundary nodes.
    X = snap(X)
    # Sanity: positive weights, nodes in the closed tetrahedron.
    if W.min() <= 0 or np.c_[X, 1.0 - X.sum(axis=1)].min() < 0.0:
        log("result is not PI; discarded")
        set_output(args.github_output, outputs)
        return
    # Certificate.
    log("computing the Newton-Kantorovich certificate")
    # Compute it.
    cert = certify(n, X, W)
    # Report it.
    log("certificate: %s" % ({k: cert.get(k) for k in ("certified", "inside", "h")} if cert else None))
    # Only certified rules are proposed.
    if cert is None or not cert.get("certified"):
        log("not certified; discarded")
        set_output(args.github_output, outputs)
        return
    # Write the rule file.
    rid, rpath = write_rule(n, X, W, cert, data.get("start", ""))
    # Atlas check.
    ok, tail = build_check(rpath)
    # Report.
    log("build check: " + ("OK" if ok else "FAILED") + "\n" + tail)
    # Reject the rule if the check fails.
    if not ok:
        os.remove(rpath)
        set_output(args.github_output, outputs)
        return
    # Previous best of the atlas alone.
    old = best_known(atlas, n)
    # Title of the pull request.
    title = "Tetrahedron degree %d: %s -> %d nodes (automatic cannonball search)" % (n, old, points)
    # Body of the pull request.
    body = "\n".join([
        "## What was computed",
        "",
        "The automatic search (`.github/workflows/auto_cannonball.yml`, `tools/auto_cannonball.py`) found a smaller PI rule on the tetrahedron:",
        "",
        "| | |",
        "|---|---|",
        "| Rule | `%s` |" % rid,
        "| Degree | %d |" % n,
        "| Nodes | %d (best in the atlas so far: %s) |" % (points, old),
        "| Start | %s |" % data.get("start", ""),
        "| Budget | %.0f min, maxfev %d |" % (args.minutes, args.maxfev),
        "",
        "## How it was verified",
        "",
        "- Moment error with the full PKD basis (`simplex_cubature.Verifier`): %.2e." % float(data["max_moment_error_pkd"]),
        "- Newton–Kantorovich certificate (`code/cubature_atlas.py`, full system, no symmetry): `cert = true`, h = %.2e, radius %.2e, `inside = %s`." % (cert["h"], cert["radius"], str(bool(cert.get("inside"))).lower()),
        "- `python tools/build_atlas.py --base origin/main`:",
        "",
        "```",
        tail,
        "```",
        "",
        "## What is still open",
        "",
        "- Review by the maintainer. The rule file is new; no existing rule was changed.",
        "- `inside = false` means at least one node lies on the boundary (snapped from below 1e-13), as for several existing tetrahedron rules.",
        "- If the pull request was opened with the default `GITHUB_TOKEN`, GitHub does not start the pull-request checks and the preview for it. Close and reopen the pull request to start them, or set the secret `AUTO_SEARCH_TOKEN`.",
        "",
        "Rule found by an automatic search program written by an AI (Claude, Anthropic), commissioned and directed by Jörg Kubitz.",
    ])
    # Create the folder of the body file.
    os.makedirs(os.path.dirname(args.pr_body), exist_ok=True)
    # Write the body.
    with open(args.pr_body, "w") as fh:
        fh.write(body + "\n")
    # Outputs for the workflow.
    outputs.update({"found": "true", "branch": "ai/auto-tet-n%02d-m%d" % (n, points), "title": title, "rule_id": rid, "rule_path": os.path.relpath(rpath, ROOT), "body": args.pr_body})
    # Write them.
    set_output(args.github_output, outputs)
    # Done.
    log("improvement found: " + title)


# Entry point.
if __name__ == "__main__":
    main()

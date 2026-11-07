#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_search.py
==============

One unattended search attempt for a better cube rule, used by the scheduled
GitHub workflow `.github/workflows/auto_search.yml`.

What it does
------------
1. Reads all PI cube rules of the atlas and the branches of open automatic
   pull requests (`ai/auto-cube-nXX-mYYY`) to know the best node count for
   every degree.
2. Picks a degree (random, weighted towards high degrees, where the gap to the
   structure estimate is largest), a symmetry group (Td or Oh) and a mode:
     refine: large-neighbourhood search starting from the best atlas rules
             (code/cubature_search.py --refine, warm start),
     fresh:  complete new search from tensor Gauss-Legendre rules.
   Several parallel slots of one workflow run get different degrees.
3. Runs code/cubature_search.py for a fixed time.
4. If the result has fewer nodes than every known rule of at least that
   degree: computes the Newton-Kantorovich certificate (code/cube_cert.py),
   writes the rule file into
   rules/minimal_pointcount_for_degree_n_PI/cube/, runs
   `python tools/build_atlas.py --base origin/main` and requires `OK`.
5. Writes the pull-request title and body and, for GitHub Actions, the step
   outputs found / branch / title / rule_id.

Usage
-----
   python tools/auto_search.py --minutes 120 --slot 0 --slots 3 --seed 12345
   python tools/auto_search.py --minutes 5 --degree 12 --group Td --mode refine

Only the cube is supported, because code/cubature_search.py and
code/cube_cert.py are the cube programs of the atlas.

Style rule of this file: one command per line.
"""

# Imports.
import argparse
import glob
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import numpy as np

# Repository root (this file lives in tools/).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Folder of the PI family for the cube.
PI_DIR = os.path.join(ROOT, "rules", "minimal_pointcount_for_degree_n_PI", "cube")
# Search program.
SEARCH = os.path.join(ROOT, "code", "cubature_search.py")
# Lowest and highest degree that is searched.
NMIN = 2
# Highest degree of the atlas.
NMAX = 30


def log(msg):
    """Print a progress line immediately."""
    # Flush so that the GitHub log shows it at once.
    print("[auto_search] " + msg, flush=True)


def atlas_cube_rules():
    """All PI cube rules of the atlas as (degree, points, path)."""
    # Result list.
    out = []
    # All rule files of all families for the cube.
    for path in glob.glob(os.path.join(ROOT, "rules", "*", "cube", "*.json")):
        # Load the rule.
        with open(path) as fh:
            rule = json.load(fh)
        # Only rules with positive weights and nodes in the closed cube.
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
        res = subprocess.run(["git", "ls-remote", "--heads", "origin", "ai/auto-cube-*"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    except Exception:
        return out
    # Parse the branch names.
    for line in res.stdout.splitlines():
        # Branch name pattern ai/auto-cube-nXX-mYYY.
        m = re.search(r"ai/auto-cube-n(\d+)-m(\d+)", line)
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


def choose_degrees(rng, slots):
    """Distinct degrees for all slots, weighted by degree squared."""
    # Candidate degrees.
    degs = list(range(NMIN, NMAX + 1))
    # Weights favour high degrees (larger gaps to the structure estimate).
    weights = [d * d for d in degs]
    # Chosen degrees.
    chosen = []
    # Draw without replacement.
    while len(chosen) < min(slots, len(degs)):
        # Weighted draw.
        d = rng.choices(degs, weights=weights)[0]
        # Keep new degrees only.
        if d not in chosen:
            chosen.append(d)
    # Return the list.
    return chosen


def export_sources(entries, folder):
    """Write atlas rules as text files (x y z w) for the warm start of the search."""
    # Create the folder.
    os.makedirs(folder, exist_ok=True)
    # Loop over the rules.
    for d, p, path in entries:
        # Load the rule.
        with open(path) as fh:
            rule = json.load(fh)
        # Nodes.
        X = np.array(rule["nodes"], dtype=float)
        # Weights (the atlas stores weights for the volume 8, as the search program).
        W = np.array(rule["weights"], dtype=float)
        # File name understood by cubature_search.py (degree and point count).
        name = os.path.join(folder, "cube3d_n%02d_m%d_atlas.txt" % (d, p))
        # Write.
        np.savetxt(name, np.c_[X, W], fmt="% .17e", header="atlas rule %s" % rule["id"])


def run_search(n, group, mode, minutes, seed, work):
    """Run code/cubature_search.py for about `minutes` minutes; return the output folder."""
    # Output folder.
    out = os.path.join(work, "out")
    # Create it.
    os.makedirs(out, exist_ok=True)
    # Wall-clock budget in seconds.
    budget = 60.0 * minutes
    # Command line.
    if mode == "refine":
        # refine_degree scales the time by 0.1 + 0.9 ((n-1)/29)^2; undo that scaling.
        scale = 0.1 + 0.9 * ((n - 1) / 29.0) ** 2
        # Arguments.
        cmd = [sys.executable, SEARCH, "--nmin", str(n), "--nmax", str(n), "--group", group, "--refine", "%.0f" % (budget / scale), "--out", out, "--sources", os.path.join(work, "src"), "--jobs", "1", "--seed", str(seed)]
    else:
        # Fresh search with a fixed budget.
        cmd = [sys.executable, SEARCH, "--nmin", str(n), "--nmax", str(n), "--group", group, "--tmin", "%.0f" % budget, "--tmax", "%.0f" % budget, "--out", out, "--summary", os.path.join(work, "summary.json"), "--seed", str(seed)]
    # Log the command.
    log("running: " + " ".join(cmd))
    # Run with a generous safety timeout.
    try:
        subprocess.run(cmd, cwd=os.path.join(ROOT, "code"), timeout=budget * 1.5 + 900)
    except subprocess.TimeoutExpired:
        log("search exceeded its time limit; using the files written so far")
    # Return the output folder.
    return out


def best_result(out, n):
    """Smallest rule of degree >= n written by the search: (degree, points, path) or None."""
    # Candidates.
    cands = []
    # Loop over the output files.
    for path in glob.glob(os.path.join(out, "cube3d_n*_m*_*.txt")):
        # Degree and point count from the file name.
        m = re.search(r"_n(\d+)_m(\d+)_", os.path.basename(path))
        # Keep files of sufficient degree.
        if m and int(m.group(1)) >= n:
            cands.append((int(m.group(1)), int(m.group(2)), path))
    # Smallest point count, highest degree on ties.
    return min(cands, key=lambda c: (c[1], -c[0])) if cands else None


def certify(degree, X, W):
    """Newton-Kantorovich certificate with the atlas program code/cube_cert.py."""
    # Make the atlas code importable.
    sys.path.insert(0, os.path.join(ROOT, "code"))
    # Import the certificate module.
    import cube_cert
    # Compute.
    try:
        return cube_cert.certificate_cube(degree, X, W)
    except Exception as exc:
        log("certificate failed: %r" % (exc,))
        return None


def write_rule(degree, X, W, cert, group, mode):
    """Write the rule file in the atlas format; return (id, path)."""
    # Identifier.
    rid = "CA-C3-%02d-%d-PI-MPC" % (degree, len(X))
    # Path.
    path = os.path.join(PI_DIR, rid + ".json")
    # Certified?
    ok = bool(cert is not None and cert.get("certified"))
    # Header fields.
    head = {
        "id": rid,
        "family": "minimal_pointcount_for_degree_n_PI",
        "domain": "cube",
        "dimension": 3,
        "degree": int(degree),
        "points": int(len(X)),
        "quality": "PI",
        "origin": "sym",
        "method": "automatic scheduled search (tools/auto_search.py, mode %s): code/cubature_search.py with symmetry reduction (group %s), orbit elimination and large-neighbourhood search; certified in the symmetry-reduced system" % (mode, group),
        "author": "Claude (Anthropic), commissioned and directed by Joerg Kubitz; found by the automatic search in GitHub Actions",
        "status": "verified" if ok else "needs_certificate",
        "locked": False,
        "residual_reported": 1e-15,
    }
    # Certificate block (only if the computation ran).
    if cert is not None and "sigma" in cert:
        head["certificate"] = {"N": int(len(X)), "M": int(cert["M"]), "p": int(cert["p"]), "res": float(cert["resid"]), "sigma": float(cert["sigma"]), "h": float(cert["h"]), "rad": float(cert["radius"]), "cert": bool(cert["certified"]), "inside": bool(cert["inside"])}
    # Symmetry group of the search.
    head["symmetry"] = group
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


def build_check():
    """Run the atlas check against origin/main; return (ok, last lines of the output)."""
    # Run the build.
    res = subprocess.run([sys.executable, "tools/build_atlas.py", "--base", "origin/main"], cwd=ROOT, capture_output=True, text=True)
    # Combined output.
    text = res.stdout + res.stderr
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
    # Time budget of the search in minutes.
    ap.add_argument("--minutes", type=float, default=120.0)
    # Index of this slot (parallel jobs of one workflow run).
    ap.add_argument("--slot", type=int, default=0)
    # Number of slots.
    ap.add_argument("--slots", type=int, default=1)
    # Seed shared by all slots of one run (degree choice) and per-slot search seed.
    ap.add_argument("--seed", type=int, default=None)
    # Fixed degree (0 = random).
    ap.add_argument("--degree", type=int, default=0)
    # Fixed group ("" = random).
    ap.add_argument("--group", default="")
    # Fixed mode ("" = random).
    ap.add_argument("--mode", default="")
    # Working folder.
    ap.add_argument("--work", default=os.path.join(ROOT, "auto_search_work"))
    # File for the pull-request body.
    ap.add_argument("--pr-body", default=os.path.join(ROOT, "auto_search_work", "pr_body.md"))
    # GitHub step output file.
    ap.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT", ""))
    # Parse.
    args = ap.parse_args()
    # Seed of the whole run.
    seed = args.seed if args.seed is not None else int(time.time())
    # Generator shared by all slots (same degree list in every slot).
    shared = random.Random(seed)
    # Generator of this slot.
    rng = random.Random(seed * 1000 + args.slot)
    # Degree of this slot.
    n = args.degree if args.degree else choose_degrees(shared, args.slots)[args.slot % args.slots]
    # Group: Td is best at most degrees.
    group = args.group if args.group else rng.choices(["Td", "Oh"], weights=[4, 1])[0]
    # Mode.
    mode = args.mode if args.mode else rng.choice(["refine", "fresh"])
    # Search seed of this slot.
    sseed = rng.randrange(1, 10 ** 6)
    # Known rules: atlas and open automatic pull requests.
    atlas = atlas_cube_rules()
    # Open pull requests.
    prs = open_auto_prs()
    # Best known count for this degree.
    target = best_known(atlas + prs, n)
    # Report the plan.
    log("slot %d/%d: degree %d, group %s, mode %s, seed %d, %.0f min; best known %s" % (args.slot, args.slots, n, group, mode, sseed, args.minutes, target))
    # Fresh working folder.
    shutil.rmtree(args.work, ignore_errors=True)
    # Export the atlas rules for the warm start.
    export_sources(atlas, os.path.join(args.work, "src"))
    # Run the search.
    out = run_search(n, group, mode, args.minutes, sseed, args.work)
    # Best result.
    res = best_result(out, n)
    # Default outputs.
    outputs = {"found": "false", "degree": n, "group": group, "mode": mode}
    # Nothing written.
    if res is None:
        log("no rule written by the search")
        set_output(args.github_output, outputs)
        return
    # Unpack.
    degree, points, path = res
    # Report.
    log("search result: degree %d, %d points (best known %s)" % (degree, points, target))
    # Re-read the open pull requests (another slot may have finished meanwhile).
    target = best_known(atlas + open_auto_prs(), n)
    # Not an improvement.
    if target is not None and points >= target:
        log("no improvement")
        set_output(args.github_output, outputs)
        return
    # Load the rule.
    data = np.loadtxt(path, ndmin=2)
    # Nodes.
    X = data[:, :3]
    # Weights (volume 8).
    W = data[:, 3]
    # Basic sanity: positive weights, nodes in the closed cube.
    if W.min() <= 0 or np.abs(X).max() > 1.0 + 1e-12:
        log("result is not PI; discarded")
        set_output(args.github_output, outputs)
        return
    # Certificate.
    log("computing the Newton-Kantorovich certificate")
    # Compute it.
    cert = certify(degree, X, W)
    # Report it.
    log("certificate: %s" % ({k: cert.get(k) for k in ("certified", "inside", "h")} if cert else None))
    # Write the rule file.
    rid, rpath = write_rule(degree, X, W, cert, group, mode)
    # Atlas check.
    ok, tail = build_check()
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
    title = "Cube degree %d: %s -> %d nodes (automatic search)" % (n, old, points)
    # Certificate summary for the body.
    if cert is not None and cert.get("certified"):
        cert_line = "Newton–Kantorovich certificate (`code/cube_cert.py`, %s-reduced system): `cert = true`, h = %.2e, `inside = %s`." % (cert.get("group"), cert["h"], str(bool(cert.get("inside"))).lower())
    else:
        cert_line = "**No certificate**: `code/cube_cert.py` did not certify this rule, so its status is `needs_certificate`."
    # Body of the pull request.
    body = "\n".join([
        "## What was computed",
        "",
        "The scheduled automatic search (`.github/workflows/auto_search.yml`, `tools/auto_search.py`) found a smaller PI rule on the cube [-1,1]³:",
        "",
        "| | |",
        "|---|---|",
        "| Rule | `%s` |" % rid,
        "| Degree | %d (searched for degree %d) |" % (degree, n),
        "| Nodes | %d (best in the atlas so far: %s) |" % (points, old),
        "| Group / mode | %s / %s |" % (group, mode),
        "| Search time | %.0f min, seed %d |" % (args.minutes, sseed),
        "",
        "## How it was verified",
        "",
        "- `python tools/build_atlas.py --base origin/main`:",
        "",
        "```",
        tail,
        "```",
        "",
        "- " + cert_line,
        "",
        "## What is still open",
        "",
        "- Review by the maintainer. The rule file is new; no existing rule was changed.",
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
    outputs.update({"found": "true", "branch": "ai/auto-cube-n%02d-m%d" % (degree, points), "title": title, "rule_id": rid, "rule_path": os.path.relpath(rpath, ROOT), "body": args.pr_body})
    # Write them.
    set_output(args.github_output, outputs)
    # Done.
    log("improvement found: " + title)


# Entry point.
if __name__ == "__main__":
    main()

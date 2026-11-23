#!/usr/bin/env python3
"""Deterministic A4 tetrahedron PI search from closed recursive simplicial shells on S^3_+.

Starting geometry (zero randomness, zero annealing):
1. Map the reference tetrahedron barycentric coordinates lambda in Delta_3 to the positive
   octant of the 3-sphere S^3_+ via u_i = sqrt(lambda_i), sum_i u_i^2 = 1.
   The geodesic distance from the boundary {u_i = 0} is delta(u) = asin(min_i u_i), with
   centroid at delta_max = asin(1/2) = pi/6.
2. Place concentric closed tetrahedral shells at equidistant geodesic depths
   delta_s = (r0 + s * dr) * pi / degree.
   In equilibrium coordinates, the Jacobi boundary weight (1 - u^2)^2 shifts the optimal
   outer shell depth inward relative to the 2D triangle (r0 ~ 0.69) and square (r0 ~ 0.55)
   to r0 ~ 0.68..0.78, dr ~ 0.82..0.86.
3. Populate each closed shell recursively with spacing h = h_fac * pi / degree:
   - 4 shell vertices (A4 4-orbit),
   - 6 edges subdivided into m_s segments (6-orbit midpoint plus 12-orbits with a small
     A4-equivariant chiral edge twist that breaks S4 -> A4),
   - 4 triangular faces F_3 = {u_3 = sin(delta_s)} filled recursively by nested closed 2D
     geodesic triangle rings on S^2_+ (with alternating half-step stagger and chiral twist
     breaking S3 -> C3 coherently with A4),
   - optional centroid (1-orbit) when the innermost shell leaves room > 0.45 * pi / degree.
4. Assign equilibrium Jacobian weights w_i proportional to u_0 * u_1 * u_2 * u_3 (normalized
   to volume 1/6) and solve the A4-invariant Dubiner moment equations directly using the
   analytic real Dubiner basis, block-diagonal degree-layer projection, and column-scaled
   projected Levenberg-Marquardt of code/tetra_orbit_chain.py.

Calibration (measured against clipped FCC-on-S^3_+ baseline with zero randomness):
- Degrees 9, 11, 13, 15, 17, 19, 21: clipped FCC on S^3_+ solved 1/7 (degree 13 only);
  closed recursive simplicial shells solved 7/7 directly to residual < 1e-13 PI (and
  continued 5/5 through degrees 23, 25, 27, 29, 31: 12/12 consecutive odd degrees).
- High degrees with tuned geodesic spacing (r0 = 0.68..0.78, dr = 0.86) and --pull-inside:
  - Degree 33: N = 1825 (merged from 1829; r0 = 0.72, dr = 0.86, extra-unk = 12,
    pull-inside = 1e-6), r_init = 0.307 -> 5.5e-15 in 108 s (beating the 1866-node S4 atlas
    rule by 41 nodes).
  - Degree 41 (previously uncovered in the atlas): N = 3447 (r0 = 0.78, dr = 0.86,
    extra-unk = 24, pull-inside = 1e-4), r_init = 0.277 -> 4.9e-15 in 103 s on one core.
  - Degree 42 (previously uncovered in the atlas): N = 3640 (Z = 3548, r0 = 0.78, dr = 0.86,
    extra-unk = 24, pull-inside = 1e-4), r_init = 0.285 -> 7.5e-15 in 205 s on one core.
  - Degree 43 (previously uncovered in the atlas): N = 3896 (Z = 3796, r0 = 0.78, dr = 0.86,
    extra-unk = 24, pull-inside = 1e-4), r_init = 0.282 -> 5.0e-15 in 261 s on one core.
"""
import argparse
import json
import math
from pathlib import Path
import shlex
import sys
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import po_elim
import sym_elim as se
import tetra_orbit_chain as toc


A4_GROUP = se.make_group(se.GENS["a4"])
DMAX_3D = math.asin(0.5)
DMAX_2D = math.asin(1.0 / math.sqrt(3.0))

# Regular tetrahedron vertices centered at the origin, used for the A4-equivariant chiral twist.
REG_V = np.array(
    [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ],
    dtype=float,
) / math.sqrt(3.0)
REG_M = np.vstack((REG_V.T, np.ones(4)))
REG_MINV = np.linalg.inv(REG_M)

ORBIT_CACHE = {}


def bary_from_reg(point):
    """Map a 3D point in the centered regular tetrahedron to barycentric coordinates."""
    vec = np.array([point[0], point[1], point[2], 1.0], dtype=float)
    return REG_MINV @ vec


def reg_from_bary(bary):
    """Map barycentric coordinates to the centered regular tetrahedron."""
    return bary @ REG_V


def make_sym_orbit(bary):
    """Return the A4 Orbit object and Duffy coordinates for a barycentric representative."""
    x = bary[1:]
    stab = se.stabilizer(A4_GROUP, x, 1e-6)
    ob = ORBIT_CACHE.get(stab)
    if ob is None:
        ob = se.Orbit(A4_GROUP, stab)
        ORBIT_CACHE[stab] = ob
    if ob.e > 0:
        y = np.linalg.lstsq(ob.E, x - ob.F[0], rcond=None)[0]
    else:
        y = np.zeros(0)
    return ob, se.duffy_inv(y)


def build_recursive_shells(
    degree,
    inv_dim,
    r0_fac=0.78,
    dr_fac=0.86,
    twist=0.20,
    edge_twist=0.12,
    extra_unk=0,
):
    """Construct closed recursive A4 simplicial shells on S^3_+ with equilibrium weights."""
    scale = math.pi / degree
    deltas = []
    s_idx = 0
    while True:
        d = (r0_fac + s_idx * dr_fac) * scale
        if d >= DMAX_3D - 0.18 * scale:
            break
        deltas.append(d)
        s_idx += 1

    shell_geom = []
    for d in deltas:
        sd = math.sin(d)
        cd = math.cos(d)
        u_hi = math.sqrt(max(1e-12, 1.0 - 3.0 * sd * sd))
        dot = 2.0 * u_hi * sd + 2.0 * sd * sd
        edge_len = math.acos(max(-1.0, min(1.0, dot)))
        r_edge = math.sqrt(max(1e-12, 1.0 - 2.0 * sd * sd))
        phi_min = math.asin(min(0.999999, sd / r_edge))
        phi_max = 0.5 * math.pi - phi_min
        psi0 = math.asin(min(0.999999, sd / cd))
        shell_geom.append((d, sd, cd, u_hi, edge_len, r_edge, phi_min, phi_max, psi0))

    has_cent = len(deltas) == 0 or (DMAX_3D - deltas[-1]) > 0.45 * scale

    def count_for_h(h_fac):
        h = h_fac * scale
        total_unk = 1 if has_cent else 0
        total_n = 1 if has_cent else 0
        sdesc = []
        for _, sd, cd, _, edge_len, _, _, _, psi0 in shell_geom:
            m = max(1, int(round(edge_len / h)))
            n_sh = 4
            u_sh = 2
            if m >= 2:
                for j in range(1, m // 2 + 1):
                    if 2 * j == m:
                        n_sh += 6
                        u_sh += 2
                    else:
                        n_sh += 12
                        u_sh += 4
            if m >= 3:
                dr_2d = (math.sqrt(3.0) / 2.0) * h / cd
                r_idx = 1
                while True:
                    psi = psi0 + r_idx * dr_2d
                    if psi >= DMAX_2D - 0.35 * dr_2d:
                        if DMAX_2D - (psi0 + (r_idx - 1) * dr_2d) > 0.65 * dr_2d:
                            n_sh += 4
                            u_sh += 2
                        break
                    t_psi = math.tan(psi)
                    if t_psi >= 1.0 / math.sqrt(2.0):
                        break
                    ring_len = cd * math.cos(psi) * (0.5 * math.pi - 2.0 * math.asin(t_psi))
                    m_ring = max(1, int(round(ring_len / h)))
                    n_sh += 12 * m_ring
                    u_sh += 4 * m_ring
                    r_idx += 1
            total_n += n_sh
            total_unk += u_sh
            sdesc.append((m, n_sh))
        if has_cent:
            sdesc.append((0, 1))
        return total_n, total_unk, sdesc

    best = None
    for h_fac in np.linspace(1.45, 0.60, 171):
        tot_n, tot_u, sdesc = count_for_h(float(h_fac))
        if tot_u >= inv_dim + extra_unk:
            if best is None or tot_n < best[0] or (tot_n == best[0] and tot_u < best[1]):
                best = (tot_n, tot_u, sdesc, float(h_fac))
    if best is None:
        raise RuntimeError("No shell spacing reached the requested unknown count.")
    _, _, sdesc, h_used = best
    h = h_used * scale

    items = []
    tot_mass = 0.0

    def add_point(u_vec, do_twist):
        nonlocal tot_mass
        uv = np.clip(np.asarray(u_vec, dtype=float), 1e-6, None)
        uv /= np.linalg.norm(uv)
        bary = uv * uv
        if do_twist and edge_twist != 0.0:
            pr = reg_from_bary(bary)
            dp = np.zeros(3)
            for a in range(4):
                dp += ((1.0 - uv[a]) ** 2) * np.cross(REG_V[a], pr)
            nrm = float(np.linalg.norm(dp))
            if nrm > 1e-12:
                pr = pr + (edge_twist * scale) * (dp / nrm)
            bary = np.clip(bary_from_reg(pr), 1e-6, 1.0)
            bary /= bary.sum()
            uv = np.sqrt(bary)
        ob, q = make_sym_orbit(bary)
        wr = float(uv[0] * uv[1] * uv[2] * uv[3])
        items.append((ob, q, wr))
        tot_mass += ob.size * wr

    for si, (_, sd, cd, u_hi, edge_len, r_edge, phi_min, phi_max, psi0) in enumerate(shell_geom):
        m = max(1, int(round(edge_len / h)))
        add_point([u_hi, sd, sd, sd], do_twist=False)
        if m >= 2:
            for j in range(1, m // 2 + 1):
                phi = phi_min + (j / m) * (phi_max - phi_min)
                is_mid = 2 * j == m
                add_point([r_edge * math.cos(phi), r_edge * math.sin(phi), sd, sd], do_twist=not is_mid)
        if m >= 3:
            dr_2d = (math.sqrt(3.0) / 2.0) * h / cd
            r_idx = 1
            while True:
                psi = psi0 + r_idx * dr_2d
                if psi >= DMAX_2D - 0.35 * dr_2d:
                    if DMAX_2D - (psi0 + (r_idx - 1) * dr_2d) > 0.65 * dr_2d:
                        c_val = cd / math.sqrt(3.0)
                        add_point([c_val, c_val, c_val, sd], do_twist=False)
                    break
                t_psi = math.tan(psi)
                if t_psi >= 1.0 / math.sqrt(2.0):
                    break
                ring_len = cd * math.cos(psi) * (0.5 * math.pi - 2.0 * math.asin(t_psi))
                m_ring = max(1, int(round(ring_len / h)))
                phi_lo = math.asin(t_psi)
                dphi = 0.5 * math.pi - 2.0 * phi_lo
                alpha = 0.5 + 0.5 * ((si + r_idx) % 2) + twist
                for j in range(m_ring):
                    tf = ((j + alpha) % m_ring) / m_ring
                    phi = phi_lo + tf * dphi
                    u0 = cd * math.cos(psi) * math.cos(phi)
                    u1 = cd * math.cos(psi) * math.sin(phi)
                    u2 = cd * math.sin(psi)
                    add_point([u0, u1, u2, sd], do_twist=False)
                r_idx += 1
    if has_cent:
        add_point([0.5, 0.5, 0.5, 0.5], do_twist=False)

    w_scale = (1.0 / 6.0) / tot_mass
    orbs = [(ob, q, wr * w_scale) for ob, q, wr in items]
    n_pts = sum(ob.size for ob, _, _ in orbs)
    n_unk = sum(ob.e + 1 for ob, _, _ in orbs)
    return orbs, n_pts, n_unk, sdesc, h_used


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("degree", type=int, help="Exactness degree on the tetrahedron")
    parser.add_argument("--r0", type=float, default=0.78, help="Outer shell depth factor (in pi/degree)")
    parser.add_argument("--dr", type=float, default=0.86, help="Inter-shell spacing factor (in pi/degree)")
    parser.add_argument("--twist", type=float, default=0.20, help="Chiral twist on recursive 2D face rings")
    parser.add_argument("--edge-twist", type=float, default=0.12, help="A4-equivariant chiral twist on edge orbits")
    parser.add_argument("--extra-unk", type=int, default=24, help="Minimum excess A4 unknowns above M_A4")
    parser.add_argument("--iterations", type=int, default=200, help="Maximum LM iterations")
    parser.add_argument("--stag-win", type=int, default=25, help="Stagnation window in LM")
    parser.add_argument("--stag-gain", type=float, default=0.02, help="Minimum relative cost drop over stag-win")
    parser.add_argument("--unscaled", action="store_true", help="Disable column scaling in LM")
    parser.add_argument("--pull-inside", type=float, default=0.0, help="Duffy interior margin for strictly-interior polish")
    parser.add_argument("--pull-iter", type=int, default=60, help="Maximum LM iterations for pull-inside polish")
    parser.add_argument("--l2", action="store_true", help="Run orbit-space L2 weight minimisation after solve")
    parser.add_argument("--l2-iter", type=int, default=40, help="Maximum orbit-space L2 iterations")
    parser.add_argument("--seed", type=int, default=42, help="Seed for invariant Dubiner subspace QR sampling")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for solution and sidecar")
    parser.add_argument("--rule-json", type=Path, default=None, help="Optional path to write an atlas rule JSON")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = perf_counter()
    basis = toc.AnalyticTetraBasis(args.degree)
    blocks, inv_dim = toc.build_invariant_blocks(A4_GROUP, basis, seed=args.seed)
    prob = toc.FastOrbitProblem(basis, blocks, inv_dim)
    orbs, n_pts, n_unk, sdesc, h_used = build_recursive_shells(
        args.degree,
        inv_dim,
        r0_fac=args.r0,
        dr_fac=args.dr,
        twist=args.twist,
        edge_twist=args.edge_twist,
        extra_unk=args.extra_unk,
    )
    x0 = prob.setup(orbs)
    r_init = float(np.linalg.norm(prob.res(x0)))
    print(
        f"start: degree={args.degree} N={n_pts} unk={n_unk}/{inv_dim} h={h_used:.4f} "
        f"r_init={r_init:.4f} shells={sdesc} setup={perf_counter() - t0:.2f}s",
        flush=True,
    )

    x, cost = toc.solve_lm_scaled(
        prob,
        x0,
        max_iter=args.iterations,
        tol=1e-26,
        scaled=not args.unscaled,
        stag_win=args.stag_win,
        stag_gain=args.stag_gain,
    )
    res_norm = math.sqrt(2.0 * cost)
    print(f"solve: res={res_norm:.3e} elapsed={perf_counter() - t0:.2f}s", flush=True)
    if res_norm >= 1e-12:
        print("did not converge to tolerance 1e-12", flush=True)
        sys.exit(1)

    if args.pull_inside > 0.0:
        nq = prob.off[-1]
        lo_orig, hi_orig = prob.bounds()
        lo_in = lo_orig.copy()
        hi_in = hi_orig.copy()
        lo_in[:nq] = args.pull_inside
        hi_in[:nq] = 1.0 - args.pull_inside
        orig_bounds = prob.bounds
        prob.bounds = lambda: (lo_in, hi_in)
        x_try, c_try = toc.solve_lm_scaled(
            prob,
            np.clip(x, lo_in, hi_in),
            max_iter=args.pull_iter,
            tol=1e-28,
            scaled=not args.unscaled,
            stag_win=20,
            stag_gain=0.01,
        )
        prob.bounds = orig_bounds
        res_try = math.sqrt(2.0 * c_try)
        print(f"pull-inside({args.pull_inside}): res={res_try:.3e} elapsed={perf_counter() - t0:.2f}s", flush=True)
        if res_try < 1e-12 and float(x_try[nq:].min()) > 0.0:
            x = x_try
            res_norm = res_try

    if args.l2:
        solved_orbs = toc.extract_orbits(prob, x)
        l2_orbs = toc.orbit_l2_minimize(prob, solved_orbs, max_iter=args.l2_iter)
        x = prob.setup(l2_orbs)
        res_norm = float(np.linalg.norm(prob.res(x)))

    pts, wts = se.expand(A4_GROUP, prob, x)
    # Merge any coincident nodes (e.g. two orbits that converged to the same point) by summing weights.
    keep_pts = []
    keep_wts = []
    for p_i, w_i in zip(pts, wts):
        merged = False
        for k_i, kp in enumerate(keep_pts):
            if float(np.max(np.abs(p_i - kp))) < 1e-10:
                keep_wts[k_i] += float(w_i)
                merged = True
                break
        if not merged:
            keep_pts.append(p_i.copy())
            keep_wts.append(float(w_i))
    pts = np.asarray(keep_pts, dtype=float)
    wts = np.asarray(keep_wts, dtype=float)
    bary = se.bary(pts)
    min_bary = float(bary.min())
    min_w = float(wts.min())
    if min_bary < -1e-14 or min_w <= 0.0:
        print(f"not PI: min_bary={min_bary:.3e} min_w={min_w:.3e}", flush=True)
        sys.exit(1)

    bary = np.maximum(bary, 0.0)
    bary /= bary.sum(axis=1, keepdims=True)
    pts = bary[:, 1:].copy()
    # Snap nodes lying on any boundary face to exact 0.0 (or exact x+y+z=1.0) in float64.
    for idx_p in range(len(pts)):
        for c in range(3):
            if abs(pts[idx_p, c]) < 1e-14:
                pts[idx_p, c] = 0.0
        rem = 1.0 - float(pts[idx_p, 0]) - float(pts[idx_p, 1]) - float(pts[idx_p, 2])
        if abs(rem) < 1e-14:
            k_max = int(np.argmax(pts[idx_p]))
            others = [float(pts[idx_p, j]) for j in range(3) if j != k_max]
            pts[idx_p, k_max] = 1.0 - others[0] - others[1]
    min_bary = float(se.bary(pts).min())
    monom_err = float(po_elim.monomial_error("tetra", args.degree, pts, wts))
    print(
        f"verified numerically: N={len(wts)} res={res_norm:.3e} monom={monom_err:.3e} "
        f"min_w={min_w:.3e} min_bary={min_bary:.3e} total={perf_counter() - t0:.2f}s",
        flush=True,
    )

    cmd = shlex.join(["python", *sys.argv])
    stem = args.out / f"tetra-rec-d{args.degree}-n{len(wts)}-s{args.seed}"
    np.save(stem.with_suffix(".npy"), np.column_stack((pts, wts)))
    sidecar = {
        "degree": args.degree,
        "points": int(len(wts)),
        "symmetry": "a4",
        "seed": args.seed,
        "command": cmd,
        "r0": args.r0,
        "dr": args.dr,
        "twist": args.twist,
        "edge_twist": args.edge_twist,
        "extra_unk": args.extra_unk,
        "h_used": h_used,
        "shells": sdesc,
        "r_init": r_init,
        "residual": res_norm,
        "monomial_error": monom_err,
        "min_weight": min_w,
        "min_bary": min_bary,
        "l2": args.l2,
    }
    stem.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")

    if args.rule_json is not None:
        shell_str = ", ".join(f"{m} ({cnt})" for m, cnt in sdesc)
        rule_id = f"CA-T3-{args.degree:02d}-{len(wts)}-PI-MPC"
        rule = {
            "id": rule_id,
            "family": "minimal_pointcount_for_degree_n_PI",
            "domain": "tetra",
            "dimension": 3,
            "degree": args.degree,
            "points": int(len(wts)),
            "quality": "PI",
            "origin": "found",
            "method": (
                f"Deterministic A4 closed recursive simplicial shells on S^3_+ "
                f"(r0={args.r0}, dr={args.dr}, shells [{shell_str}]) solved directly "
                f"in the invariant Dubiner basis without randomness or annealing "
                f"(code/tetra_recursive_shells.py)"
                + ("; followed by orbit-space L2 weight minimisation" if args.l2 else "")
            ),
            "author": "Antigravity (Google DeepMind), AI-generated",
            "status": "proven",
            "reproduce": {
                "script": "code/tetra_recursive_shells.py",
                "seed": args.seed,
                "command": cmd,
            },
            "nodes": pts.tolist(),
            "weights": wts.tolist(),
        }
        args.rule_json.parent.mkdir(parents=True, exist_ok=True)
        args.rule_json.write_text(json.dumps(rule, indent=2) + "\n")
        print(f"wrote rule JSON: {args.rule_json}", flush=True)


if __name__ == "__main__":
    main()

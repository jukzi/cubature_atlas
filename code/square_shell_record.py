"""Search the square from concentric square shells in Chebyshev angle coordinates.

Use the Atlas C4 Legendre equations and scaled Levenberg-Marquardt solver.
The shell index is fixed only during initialization. The optimizer releases
each four-point orbit's position and weight.
"""
import argparse
import json
import math
import shlex
from pathlib import Path
import sys
import numpy as np


def shell_start(sq, degree, count, angle, layers, offset, mode, phase):
    k, centre = divmod(count, 4)
    x, y, _, _ = sq.rotated_cheb_start(degree, k, centre, angle, offset)
    edge = math.pi / 2 - offset * 2 * math.pi / degree
    if mode == 'lattice':
        u, v = np.arcsin(x), np.arcsin(y)
        radius = np.maximum(abs(u), abs(v))
        labels = np.minimum(layers - 1, (layers * radius / edge).astype(int))
        for ring in np.unique(labels):
            chosen = labels == ring
            scale = np.mean(radius[chosen]) / radius[chosen]
            u[chosen] *= scale
            v[chosen] *= scale
    else:
        # Allocate orbits in proportion to each shell's area in angle space.
        bounds = np.rint(k * (np.arange(layers + 1) / layers) ** 2).astype(int)
        u, v = [], []
        for ring, (left, right) in enumerate(zip(bounds[:-1], bounds[1:])):
            p = right - left
            if p == 0:
                continue
            radius = edge * np.sqrt((left + right) / (2 * k)) / np.sqrt((bounds[-2] + k) / (2 * k))
            shift = (phase + .5 * (ring % 2)) % 1
            u.extend(radius * (2 * (np.arange(p) + shift) / p - 1))
            v.extend(np.full(p, -radius))
        u, v = np.array(u), np.array(v)
    w = np.cos(u) * np.cos(v)
    total = 4 * np.sum(w) + centre
    return np.sin(u), np.sin(v), 4 * w / total, 4 * centre / total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--atlas', type=Path, required=True)
    ap.add_argument('--degree', type=int, required=True)
    ap.add_argument('--nodes', type=int, required=True)
    ap.add_argument('--angle', type=float, default=49)
    ap.add_argument('--offset', type=float, default=.60)
    ap.add_argument('--layers', type=int, required=True)
    ap.add_argument('--mode', choices=['lattice', 'regular'], default='lattice')
    ap.add_argument('--phase', type=float, default=.25)
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.degree % 2 != 1 or a.nodes % 4 > 1 or a.layers < 1:
        ap.error('Use odd degree, N = 0 or 1 mod 4, and positive layer count.')
    sys.path.insert(0, str((a.atlas / 'code').resolve()))
    import square_cheb_c4 as sq
    centre = a.nodes % 4 == 1
    x, y, w, wc = shell_start(sq, a.degree, a.nodes, a.angle, a.layers, a.offset, a.mode, a.phase)
    B = sq.SquareC4Basis(a.degree)
    start, res, it, x, y, w, wc = sq.solve_c4_lm(B, x, y, w, centre, wc, a.steps, 35, .01)
    nodes, weights = sq.expand_c4(x, y, w, centre, wc)
    error = sq.monomial_residual(a.degree, nodes, weights)
    margin = 1 - np.abs(nodes).max()
    print(f'n={a.degree} N={a.nodes} init={start:.3g} residual={res:.3g} monomial={error:.3g} steps={it} margin={margin:.3g} minw={weights.min():.3g}', flush=True)
    if not (error <= 1e-12 and res <= 1e-12 and margin > 0 and weights.min() > 0):
        return
    method = (f'{a.layers} concentric square shell bins in Chebyshev angle coordinates, '
              f'{a.mode} tangential placement, then free C4 orbit coordinates and weights')
    rule = dict(id=f'CA-C2-{a.degree:02}-{a.nodes}-PI-MPC', domain='square',
                family='minimal_pointcount_for_degree_n_PI', dimension=2, degree=a.degree,
                points=a.nodes, quality='PI', origin='found', method=method,
                status='candidate', author='OpenAI Codex', residual_reported=error,
                nodes=nodes.tolist(), weights=weights.tolist())
    script = 'code/square_shell_record.py'
    rule['reproduce'] = dict(script=script, seed=0, command=shlex.join(['python', script, *sys.argv[1:]]))
    a.out.write_text(json.dumps(rule, indent=2) + '\n')


if __name__ == '__main__':
    main()

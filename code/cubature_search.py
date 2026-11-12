#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cubature_search.py
==================

Search for cubature rules with few points on the hypercube [-1, 1]^d
(implemented and tuned for d = 3) that integrate every polynomial of
total degree <= n exactly, with
  * strictly positive weights,
  * all points inside the cube or on its boundary.

Method (summary)
----------------
1. Symmetry reduction (Sobolev's invariance theorem):
   If the point set and the weights are invariant under a finite group G
   of symmetries of the cube, the rule is exact for all polynomials of
   degree <= n as soon as it is exact for the G-invariant polynomials of
   degree <= n.  This shrinks the number of moment equations roughly by |G|
   and the number of unknowns by the same factor.
     S. L. Sobolev, "Formulas of mechanical cubature on the surface of a
     sphere", Sibirsk. Mat. Zh. 3 (1962) 769-796.
   The default group is the full octahedral group O_h (48 signed
   permutations); subgroups (O, T_d, T_h, D_4h, D_2h, C_i, C_1) are
   available as well.

2. Well-conditioned moment equations:
   The invariant polynomials are obtained by applying the Reynolds operator
   to tensor products of orthonormal Legendre polynomials.  Distinct
   orbits of multi-indices give mutually orthogonal invariants, so the
   moment equations are formulated in an orthonormal basis.  The exact
   moments are then simply (1, 0, 0, ..., 0).
     F. D. Witherden, P. E. Vincent, "On the identification of symmetric
     quadrature rules for finite element methods", Comput. Math. Appl. 69
     (2015) 1232-1241.  https://arxiv.org/abs/1409.1865

3. Orbit parametrisation:
   Every point orbit is described by a "pattern" (e.g. (a,0,0), (a,a,b),
   (a,b,c)), a vector of free coordinates and one total orbit weight.
   Patterns are enumerated automatically for the chosen group.

4. Solver:
   A bounded Levenberg-Marquardt / Gauss-Newton method with an SVD of the
   (usually under-determined) Jacobian.  Several damping values reuse one
   SVD.  Box constraints (coordinates in [-1,1], weights >= 0) are handled
   by projection plus an active set.  The Jacobian is analytic.

5. Node (orbit) elimination:
   Start from an exact symmetric rule (tensor Gauss-Legendre rule),
   then repeatedly remove an orbit or "demote" it to a smaller orbit type
   (e.g. (a,b,c) -> (a,b,0)), re-solving the moment equations each time.
   Candidate moves are ranked by the norm of the minimum-norm linearised
   correction, i.e. the criterion of Slobodkins & Tausch.
     H. Xiao, Z. Gimbutas, "A numerical algorithm for the construction of
     efficient quadrature rules in two and higher dimensions",
     Comput. Math. Appl. 59 (2010) 663-676.
     https://doi.org/10.1016/j.camwa.2009.10.027
     A. Slobodkins, J. Tausch, "A node elimination algorithm for cubature
     of high-dimensional polytopes", Comput. Math. Appl. 144 (2023) 229-236.
     https://arxiv.org/abs/2207.10737
     M. Diallo, Z. A. Worku, "High-order symmetric positive interior
     quadrature rules on two and three dimensional domains" (2026).
     https://arxiv.org/abs/2601.14488
   Randomised restarts of the elimination path are used to escape poor
   greedy decisions.  Fast scoring: the minimum-norm correction satisfies
   ||delta||^2 = r'^T (J' J'^T)^{-1} r', where J'J'^T is a low-rank update of
   J J^T, so one (neq x neq) solve per candidate move suffices.

5b. Iterated large-neighbourhood search (phase 2):
   Add 1-4 random orbits to the best rule (exactness is kept, the new
   weights are raised by continuation), re-eliminate with randomised
   ordering, accept with a simulated-annealing rule.  This gave the largest
   improvements at high degree.

5c. Targeted search (version 3):
   For every subset S of orbit types, the Jacobian columns of S have a
   maximal rank rank_S (estimated with Chebyshev-distributed samples).  The
   cheapest orbit structure satisfying all necessary rank conditions solves
   the integer linear program (structure_target, scipy.optimize.milp)
       minimise   sum_f c_f size_f
       subject to sum_{f not in S} c_f dof_f >= neq - rank_S  for all S.
   This is an estimate, not a proven lower bound (positivity is ignored).
   Half of the LNS perturbations add orbits preferentially in families with
   a deficit w.r.t. this target structure.

5d. Warm starts (--refine with --sources):
   Every stored rule of degree >= n that is invariant under the chosen
   group is an exact starting rule for degree n; O_h rules are T_d- and
   T_h-invariant.  Improvements are written to disk immediately.

5e. Speed (version 2): Cholesky factorisation of the smaller Gram matrix
   instead of an SVD in the LM step, using
       (J^T J + lam I)^{-1} J^T = J^T (J J^T + lam I)^{-1},
   Woodbury updates for the move scores, gathered Legendre tables and
   closed-form projections.

Usage
-----
   python3 cubature_search.py --nmin 1 --nmax 30 --group Oh --jobs 2      # search
   python3 cubature_search.py --nmin 2 --nmax 30 --group Td --out rules_Td # other group
   python3 cubature_search.py --nmin 20 --nmax 30 --group Td --refine 600 --out rules_ref --sources rules_best --jobs 2 --seed 7   # warm-start LNS
   python3 cubature_search.py --nmin 1 --nmax 31 --finalize               # clean + summary
   python3 verify_rules.py rules_best                                     # independent check

6. Lower bounds printed for comparison:
     m >= dim P_{floor(n/2)}  (classical bound for positive rules), and for
     odd n the stronger bound of Moeller for centrally symmetric regions:
     H. M. Moeller, "Kubaturformeln mit minimaler Knotenzahl",
     Numer. Math. 25 (1976) 185-200.  https://doi.org/10.1007/BF01462272

Style rule of this file: one command per line.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import glob
import re
import itertools
import json
import math
import sys
import time
import numpy as np
from scipy.linalg import cho_factor
from scipy.linalg import cho_solve


# ---------------------------------------------------------------------------
# One-dimensional orthonormal Legendre polynomials
# ---------------------------------------------------------------------------
def legendre_table(x, kmax):
    """Return (L, dL) with L[k] = sqrt(2k+1) P_k(x) and dL[k] its derivative.

    The scaling makes {L_k} orthonormal w.r.t. the probability measure dx/2
    on [-1, 1].  x may have any shape; the result has shape (kmax+1,)+x.shape.
    """
    # Allocate the tables for the classical (unnormalised) Legendre values.
    p = np.empty((kmax + 1,) + x.shape)
    # Allocate the table for the derivatives.
    dp = np.empty((kmax + 1,) + x.shape)
    # P_0 = 1.
    p[0] = 1.0
    # P_0' = 0.
    dp[0] = 0.0
    # Only continue if degree 1 is requested.
    if kmax >= 1:
        # P_1 = x.
        p[1] = x
        # P_1' = 1.
        dp[1] = 1.0
    # Three-term recurrence (Bonnet) for k >= 1.
    for k in range(1, kmax):
        # (k+1) P_{k+1} = (2k+1) x P_k - k P_{k-1}.
        p[k + 1] = ((2 * k + 1) * x * p[k] - k * p[k - 1]) / (k + 1)
        # P'_{k+1} = P'_{k-1} + (2k+1) P_k  (stable, no division by 1-x^2).
        dp[k + 1] = dp[k - 1] + (2 * k + 1) * p[k]
    # Normalisation factors sqrt(2k+1).
    scale = np.sqrt(2.0 * np.arange(kmax + 1) + 1.0)
    # Reshape so that the factors broadcast over the trailing axes.
    scale = scale.reshape((kmax + 1,) + (1,) * x.ndim)
    # Return the normalised values and derivatives.
    return p * scale, dp * scale


# ---------------------------------------------------------------------------
# Symmetry groups of the cube (signed permutation matrices)
# ---------------------------------------------------------------------------
def cube_group(name, d=3):
    """Return (perms, signs) of a subgroup of the hyperoctahedral group.

    An element acts by (g x)_i = signs[i] * x[perms[i]].
    Supported names (d = 3): Oh, O, Td, Th, T, D3d, D3, C3v, S6, C3, D4h, D2h, Ci, C1.
    For general d: 'Bd' (full group), 'Ci', 'C1', 'D2h' (sign flips only).
    """
    # Container for the permutation parts.
    perms = []
    # Container for the sign parts.
    signs = []
    # Loop over all permutations of the coordinates.
    for perm in itertools.permutations(range(d)):
        # Loop over all sign vectors.
        for sgn in itertools.product((1, -1), repeat=d):
            # Matrix representation, used for determinants and tests.
            mat = np.zeros((d, d))
            # Fill the signed permutation matrix row by row.
            for i in range(d):
                # Row i picks coordinate perm[i] with sign sgn[i].
                mat[i, perm[i]] = sgn[i]
            # Determinant (+1 rotation, -1 improper).
            det = round(np.linalg.det(mat))
            # Parity of the permutation (+1 even, -1 odd).
            parity = round(np.linalg.det(np.eye(d)[list(perm)]))
            # Product of the signs.
            sprod = int(np.prod(sgn))
            # True if the permutation is the identity.
            is_id = list(perm) == list(range(d))
            # Decide membership according to the requested group.
            if name in ("Oh", "Bd"):
                keep = True
            elif name == "O":
                keep = det == 1
            elif name == "Td":
                keep = sprod == 1
            elif name == "Th":
                keep = parity == 1
            elif name == "T":
                keep = parity == 1 and sprod == 1
            elif name == "D3d":
                keep = len(set(sgn)) == 1
            elif name == "D3":
                keep = len(set(sgn)) == 1 and det == 1
            elif name == "C3v":
                keep = all(x == 1 for x in sgn)
            elif name == "S6":
                keep = len(set(sgn)) == 1 and parity == 1
            elif name == "C3":
                keep = all(x == 1 for x in sgn) and parity == 1
            elif name == "D4h":
                keep = perm[d - 1] == d - 1
            elif name == "D2h":
                keep = is_id
            elif name == "Ci":
                keep = is_id and (len(set(sgn)) == 1)
            elif name == "C1":
                keep = is_id and all(s == 1 for s in sgn)
            else:
                raise ValueError("unknown group " + name)
            # Store the element if it belongs to the group.
            if keep:
                perms.append(perm)
                signs.append(sgn)
    # Convert to integer arrays.
    return np.array(perms, dtype=int), np.array(signs, dtype=float)


def apply_group(perms, signs, pts):
    """Apply every group element to every point.

    pts: (..., d) -> result (G, ..., d) with result[g, ..., i] =
    signs[g, i] * pts[..., perms[g, i]].
    """
    # Gather the permuted coordinates for every group element.
    gathered = pts[..., perms]
    # gathered has shape (..., G, d); move the group axis to the front.
    gathered = np.moveaxis(gathered, -2, 0)
    # Reshape signs so that they broadcast over the point axes.
    shape = (signs.shape[0],) + (1,) * (pts.ndim - 1) + (signs.shape[1],)
    # Return the signed coordinates.
    return gathered * signs.reshape(shape)


def orbit_points(perms, signs, p, tol=1e-10):
    """Return the distinct points of the orbit G p."""
    # All images of p.
    imgs = apply_group(perms, signs, p[None, :])[:, 0, :]
    # Round to merge numerically identical images.
    keys = np.round(imgs / tol).astype(np.int64)
    # Unique rows.
    _, idx = np.unique(keys, axis=0, return_index=True)
    # Return the distinct images.
    return imgs[np.sort(idx)]


# ---------------------------------------------------------------------------
# Orbit patterns
# ---------------------------------------------------------------------------
class Pattern:
    """A family of orbits: generator x = A v with free coordinates v."""

    def __init__(self, spec, A, size):
        # Integer specification, e.g. (1, 1, 0) means (a, a, 0).
        self.spec = tuple(spec)
        # Matrix mapping the free coordinates to the generator point.
        self.A = A
        # Number of free coordinates.
        self.dim = A.shape[1]
        # Number of points of a generic orbit of this family.
        self.size = size
        # Pseudo-inverse of A (closed-form projection onto the family).
        self.Apinv = np.linalg.pinv(A) if A.shape[1] > 0 else np.zeros((0, A.shape[0]))

    def fit(self, q, tol=1e-9):
        """Return v with A v = q if q belongs to this family, else None."""
        # Least-squares solution (A has orthogonal columns).
        v = self.Apinv @ q
        # Check that the reconstruction is exact.
        if np.max(np.abs(self.A @ v - q)) > tol:
            return None
        # Return the free coordinates.
        return v

    def name(self):
        """Human readable form such as (a,a,b)."""
        # Letters for the free coordinates.
        letters = "abcdefgh"
        # Build each coordinate symbol.
        parts = []
        # Loop over coordinates.
        for s in self.spec:
            # Zero coordinate.
            if s == 0:
                parts.append("0")
            # Positive variable.
            elif s > 0:
                parts.append(letters[s - 1])
            # Negative variable.
            else:
                parts.append("-" + letters[-s - 1])
        # Join to a tuple string.
        return "(" + ",".join(parts) + ")"


def enumerate_patterns(perms, signs, d, rng):
    """Enumerate inequivalent orbit patterns for the group.

    Coordinates are 0 or +-v_j.  Patterns that produce the same orbit family
    are merged; patterns that are special cases of a larger family with the
    same orbit size (no point saving) are dropped.
    """
    # Candidate list.
    cands = []
    # Every coordinate is in {-d..d}; 0 means zero, +-j means +-v_j.
    for spec in itertools.product(range(-d, d + 1), repeat=d):
        # Variables that appear, in order of first appearance.
        seen = []
        # Canonical form flag.
        ok = True
        # Check canonical ordering and positive first appearance.
        for s in spec:
            # Skip zeros.
            if s == 0:
                continue
            # Index of the variable.
            j = abs(s)
            # First appearance of variable j.
            if j not in seen:
                # It must be the next unused variable and appear with + sign.
                if j != len(seen) + 1 or s < 0:
                    ok = False
                # Record it.
                seen.append(j)
        # Skip non-canonical specifications.
        if not ok:
            continue
        # Number of free coordinates.
        nv = len(seen)
        # Build the matrix A.
        A = np.zeros((d, nv))
        # Fill A.
        for i, s in enumerate(spec):
            # Non-zero coordinate.
            if s != 0:
                A[i, abs(s) - 1] = np.sign(s)
        # Generic values for the free coordinates.
        v = rng.uniform(0.15, 0.85, nv)
        # Generic point of the family.
        p = A @ v
        # Size of the generic orbit.
        size = orbit_points(perms, signs, p).shape[0]
        # Store the candidate together with a generic point.
        cands.append((Pattern(spec, A, size), p))
    # Sort: few variables first, few minus signs, then lexicographic.
    cands.sort(key=lambda c: (c[0].dim, sum(1 for s in c[0].spec if s < 0), [-abs(s) for s in c[0].spec]))
    # Keep only one representative per equivalence class.
    reps = []
    # Loop over candidates.
    for pat, p in cands:
        # All points of the generic orbit.
        orb = orbit_points(perms, signs, p)
        # Assume it is new.
        new = True
        # Compare with representatives kept so far.
        for rep, _ in reps:
            # Same dimension and same size are necessary for equivalence.
            if rep.dim == pat.dim and rep.size == pat.size:
                # Equivalent if some orbit point fits the representative.
                if any(rep.fit(q) is not None for q in orb):
                    new = False
                    break
        # Keep new classes.
        if new:
            reps.append((pat, p))
    # Remove families covered by a larger family of the same orbit size.
    final = []
    # Loop over representatives.
    for pat, p in reps:
        # Orbit of the generic point.
        orb = orbit_points(perms, signs, p)
        # Assume not covered.
        covered = False
        # Look for a bigger family with equal orbit size containing it.
        for big, _ in reps:
            # Bigger dimension and same number of points.
            if big.dim > pat.dim and big.size == pat.size:
                # Check containment.
                if any(big.fit(q) is not None for q in orb):
                    covered = True
                    break
        # Keep uncovered families.
        if not covered:
            final.append(pat)
    # Return the list of patterns.
    return final


# ---------------------------------------------------------------------------
# Invariant orthonormal basis
# ---------------------------------------------------------------------------
def invariant_basis(perms, signs, n, d):
    """Multi-indices and norms of the G-invariant orthonormal basis.

    For a multi-index alpha, (P_alpha o g)(x) = sign * P_beta(x) with
    beta[perms[g,i]] = alpha[i].  The Reynolds projection of P_alpha is
    (1/|G|) sum_g P_alpha o g.  Different orbits are orthogonal.
    Returns (alphas, norms) where the invariant is
    f(x) = mean_g P_alpha(g x) / norm.
    """
    # All multi-indices of total degree <= n, sorted by degree.
    idx = [a for a in itertools.product(range(n + 1), repeat=d) if sum(a) <= n]
    # Sort by total degree so that the constant comes first.
    idx.sort(key=lambda a: (sum(a), a))
    # Indices already assigned to an orbit.
    done = set()
    # Output containers.
    alphas = []
    # Norms of the Reynolds projections.
    norms = []
    # Loop over multi-indices.
    for a in idx:
        # Skip already processed ones.
        if a in done:
            continue
        # Coefficients of the projection in the orthonormal product basis.
        coef = {}
        # Loop over group elements.
        for g in range(perms.shape[0]):
            # Image multi-index.
            beta = [0] * d
            # Sign factor.
            sg = 1.0
            # Build beta and the sign.
            for i in range(d):
                # Coordinate i of g x is signs[g,i] * x[perms[g,i]].
                beta[perms[g, i]] = a[i]
                # Multiply the sign (-1)^a_i if the sign is negative.
                sg *= signs[g, i] ** a[i]
            # Key of the image.
            key = tuple(beta)
            # Accumulate.
            coef[key] = coef.get(key, 0.0) + sg / perms.shape[0]
        # Mark the orbit as processed.
        done.update(coef.keys())
        # Norm of the projection.
        nrm = math.sqrt(sum(c * c for c in coef.values()))
        # Keep only non-vanishing invariants.
        if nrm > 1e-12:
            alphas.append(a)
            norms.append(nrm)
    # Return as arrays.
    return np.array(alphas, dtype=int), np.array(norms)


# ---------------------------------------------------------------------------
# The cubature problem for a fixed group and degree
# ---------------------------------------------------------------------------
class Problem:
    """Moment equations for G-invariant rules of degree n on [-1,1]^d."""

    def __init__(self, n, group="Oh", d=3, seed=0):
        # Degree of exactness.
        self.n = n
        # Dimension.
        self.d = d
        # Name of the symmetry group.
        self.group = group
        # Random generator (pattern enumeration only).
        rng = np.random.default_rng(seed)
        # Group elements.
        self.perms, self.signs = cube_group(group, d)
        # Group order.
        self.G = self.perms.shape[0]
        # Orbit patterns.
        self.patterns = enumerate_patterns(self.perms, self.signs, d, rng)
        # Invariant basis.
        self.alphas, self.norms = invariant_basis(self.perms, self.signs, n, d)
        # Number of moment equations.
        self.neq = self.alphas.shape[0]
        # Exact moments in the orthonormal basis.
        self.rhs = np.zeros(self.neq)
        # Only the constant has a non-zero integral (probability measure).
        self.rhs[0] = 1.0
        # Distinct coordinate permutations occurring in the group.
        self.uperms = np.unique(self.perms, axis=0)
        # Sign sums: coef[k, u] = sum over elements with permutation u of prod_i s_i^alpha_ki,
        # divided by |G| and by the norm of invariant k.  Then
        # f_k(p) = sum_u coef[k, u] prod_i L_{alpha_ki}(p[perm_u(i)]).
        self.coef = np.zeros((self.neq, self.uperms.shape[0]))
        # Loop over group elements.
        for g in range(self.G):
            # Index of its permutation.
            u = int(np.where(np.all(self.uperms == self.perms[g], axis=1))[0][0])
            # Sign factor for every invariant.
            sg = np.prod(self.signs[g][None, :] ** self.alphas, axis=1)
            # Accumulate.
            self.coef[:, u] += sg
        # Normalise.
        self.coef /= self.G * self.norms[:, None]

    def eval_basis(self, pts, deriv=True):
        """Invariants F[k, o] = f_k(p_o) and gradients dF[k, o, j].

        Only the distinct coordinate permutations of the group are summed;
        the sign flips are folded into self.coef (exact, since
        L_a(-t) = (-1)^a L_a(t)).
        """
        # Legendre tables at the generator coordinates: (n+1, O, d).
        L, dL = legendre_table(pts, self.n)
        # Number of orbits.
        O = pts.shape[0]
        # Values accumulator.
        F = np.zeros((self.neq, O))
        # Gradient accumulator.
        dF = np.zeros((self.neq, O, self.d)) if deriv else None
        # Gather tables once: T[i][c][k, o] = L_{alpha_ki}(p_o[c]) (d*d gathers instead of |perms|*d).
        T = [[L[self.alphas[:, i], :, c] for c in range(self.d)] for i in range(self.d)]
        # Same for the derivatives.
        dT = [[dL[self.alphas[:, i], :, c] for c in range(self.d)] for i in range(self.d)] if deriv else None
        # Loop over the distinct permutations.
        for u, perm in enumerate(self.uperms):
            # Factors: fac[i][k, o] = L_{alpha_ki}(p_o[perm_i]).
            fac = [T[i][perm[i]] for i in range(self.d)]
            # Product over the coordinates.
            prod = fac[0].copy()
            # Multiply the remaining factors.
            for i in range(1, self.d):
                prod *= fac[i]
            # Add the weighted contribution.
            F += self.coef[:, u:u + 1] * prod
            # Skip derivatives if not requested.
            if not deriv:
                continue
            # Loop over the differentiated factor.
            for i in range(self.d):
                # Derivative factor.
                term = dT[i][perm[i]]
                # Multiply the other factors.
                for l in range(self.d):
                    # Skip the differentiated one.
                    if l != i:
                        term = term * fac[l]
                # The factor depends on coordinate perm[i] of the generator.
                dF[:, :, perm[i]] += self.coef[:, u:u + 1] * term
        # Return values and gradients.
        return F, dF


# ---------------------------------------------------------------------------
# A symmetric rule = list of orbits
# ---------------------------------------------------------------------------
class Rule:
    """Collection of orbits (pattern index, free coordinates, total weight)."""

    def __init__(self, prob, pat_ids, coords, weights):
        # The underlying problem.
        self.prob = prob
        # Pattern index of every orbit.
        self.pat_ids = list(pat_ids)
        # Free coordinates of every orbit (list of 1-D arrays).
        self.coords = [np.array(c, dtype=float) for c in coords]
        # Total weight of every orbit (sum of its point weights).
        self.weights = np.array(weights, dtype=float)

    def copy(self):
        """Deep copy."""
        # Copy all lists and arrays.
        return Rule(self.prob, list(self.pat_ids), [c.copy() for c in self.coords], self.weights.copy())

    def generators(self):
        """Generator points of all orbits, shape (O, d)."""
        # Map the free coordinates through the pattern matrices.
        rows = [self.prob.patterns[p].A @ c for p, c in zip(self.pat_ids, self.coords)]
        # Stack (handle the empty rule).
        return np.array(rows).reshape(len(rows), self.prob.d)

    def npoints(self):
        """Nominal number of points (sum of generic orbit sizes)."""
        # Sum of the pattern sizes.
        return int(sum(self.prob.patterns[p].size for p in self.pat_ids))

    def pack(self):
        """Pack all unknowns into one vector."""
        # Coordinates first, then the weights.
        parts = [c for c in self.coords] + [self.weights]
        # Concatenate.
        return np.concatenate(parts) if parts else np.zeros(0)

    def unpack(self, z):
        """Inverse of pack."""
        # Running offset.
        off = 0
        # Loop over orbits.
        for o in range(len(self.coords)):
            # Number of free coordinates.
            k = self.coords[o].size
            # Copy the slice.
            self.coords[o] = z[off:off + k].copy()
            # Advance.
            off += k
        # Remaining entries are the weights.
        self.weights = z[off:].copy()

    def bounds(self):
        """Lower and upper bounds for the packed vector."""
        # Number of coordinate unknowns.
        nc = sum(c.size for c in self.coords)
        # Lower bounds.
        lo = np.concatenate([-np.ones(nc), np.zeros(len(self.weights))])
        # Upper bounds.
        hi = np.concatenate([np.ones(nc), np.full(len(self.weights), np.inf)])
        # Return both.
        return lo, hi

    def residual_jacobian(self, deriv=True):
        """Residual r = F T - e0 and Jacobian w.r.t. the packed unknowns."""
        # Generator points.
        pts = self.generators()
        # Basis values and gradients.
        F, dF = self.prob.eval_basis(pts, deriv)
        # Residual.
        r = F @ self.weights - self.prob.rhs
        # Return without Jacobian if not needed.
        if not deriv:
            return r, None
        # Coordinate blocks of the Jacobian.
        blocks = []
        # Loop over orbits.
        for o, p in enumerate(self.pat_ids):
            # Pattern matrix.
            A = self.prob.patterns[p].A
            # Skip orbits without free coordinates.
            if A.shape[1] == 0:
                continue
            # d r / d v_o = T_o * dF[:, o, :] @ A.
            blocks.append(self.weights[o] * (dF[:, o, :] @ A))
        # Weight block is simply F.
        blocks.append(F)
        # Assemble.
        J = np.hstack(blocks)
        # Return.
        return r, J


# ---------------------------------------------------------------------------
# Bounded Levenberg-Marquardt solver
# ---------------------------------------------------------------------------
# Default iteration limits (can be changed from the command line).
SOLVER = {"maxit": 80, "patience": 15}

def solve_rule(rule, tol=1e-14, maxit=None, verbose=False, patience=None, fixed=None):
    """Solve the moment equations for the given rule (in place).

    Projected Levenberg-Marquardt with an active set for the bounds.
    One SVD per iteration is reused for several damping parameters.
    Returns (success, residual norm).
    """
    # Use the module-wide defaults if not given.
    if maxit is None:
        maxit = SOLVER["maxit"]
    # Stagnation window.
    if patience is None:
        patience = SOLVER["patience"]
    # Current unknowns.
    z = rule.pack()
    # Bounds.
    lo, hi = rule.bounds()
    # Frozen unknowns are modelled by collapsing their bounds.
    if fixed is not None:
        # Lower bound equals the current value.
        lo = np.where(fixed, z, lo)
        # Upper bound equals the current value.
        hi = np.where(fixed, z, hi)
    # Clip the start into the box.
    z = np.clip(z, lo, hi)
    # Write it back.
    rule.unpack(z)
    # Initial residual and Jacobian.
    r, J = rule.residual_jacobian()
    # Residual norm.
    nr = np.linalg.norm(r)
    # Relative damping parameter.
    mu = 1e-6
    # History for stagnation detection.
    hist = [nr]
    # Main loop.
    for it in range(maxit):
        # Converged.
        if nr < tol:
            return True, nr
        # Gradient of 0.5 ||r||^2.
        grad = J.T @ r
        # Active set: variables at a bound with the descent direction pointing outwards.
        at_lo = (z <= lo + 1e-15) & (grad > 0)
        # Variables at the upper bound.
        at_hi = (z >= hi - 1e-15) & (grad < 0)
        # Free variables.
        free = ~(at_lo | at_hi)
        # Reduced Jacobian.
        Jf = J[:, free]
        # Use the smaller Gram matrix: (J^T J + lam I)^{-1} J^T = J^T (J J^T + lam I)^{-1}.
        dual = Jf.shape[0] <= Jf.shape[1]
        # Gram matrix of the smaller dimension.
        Gm = Jf @ Jf.T if dual else Jf.T @ Jf
        # Right-hand side for the primal form.
        g = None if dual else Jf.T @ r
        # Scale of the problem (largest diagonal entry ~ largest squared singular value).
        smax2 = float(np.max(np.diag(Gm))) if Gm.size else 1.0
        # Flag for an accepted step.
        accepted = False
        # Try increasing damping until the residual decreases.
        for _ in range(14):
            # Damping value.
            lam = mu * smax2
            # Damped Gram matrix.
            Ad = Gm + lam * np.eye(Gm.shape[0])
            # Cholesky factorisation (may fail if lam is too small).
            try:
                # Factor.
                cf = cho_factor(Ad, check_finite=False)
            except np.linalg.LinAlgError:
                # Increase the damping and retry.
                mu *= 10.0
                continue
            # Step in the free variables.
            step = -(Jf.T @ cho_solve(cf, r, check_finite=False)) if dual else -cho_solve(cf, g, check_finite=False)
            # Full step vector.
            dz = np.zeros_like(z)
            # Insert the free part.
            dz[free] = step
            # Projected trial point.
            zt = np.clip(z + dz, lo, hi)
            # Evaluate the residual only.
            rule.unpack(zt)
            # Residual at the trial point.
            rt, _ = rule.residual_jacobian(deriv=False)
            # Its norm.
            nrt = np.linalg.norm(rt)
            # Accept on decrease.
            if nrt < nr:
                accepted = True
                break
            # Otherwise increase the damping.
            mu *= 10.0
        # Give up if no decrease was possible.
        if not accepted:
            rule.unpack(z)
            return nr < tol, nr
        # Accept the trial point.
        z = zt
        # Decrease the damping after a successful step.
        mu = max(mu / 30.0, 1e-15)
        # Recompute residual and Jacobian.
        r, J = rule.residual_jacobian()
        # New norm.
        nr = np.linalg.norm(r)
        # Record history.
        hist.append(nr)
        # Optional trace.
        if verbose:
            print("   it", it, "res", nr, "mu", mu)
        # Stagnation: little progress over `patience` iterations while far from a solution.
        if len(hist) > patience and nr > 1e-8 and nr > 0.5 * hist[-patience]:
            return False, nr
    # Final verdict.
    return nr < tol, nr


# ---------------------------------------------------------------------------
# Rule clean-up: drop zero weights, demote degenerate orbits
# ---------------------------------------------------------------------------
def canonicalize(rule, wtol=1e-13):
    """Remove empty orbits and re-label degenerate ones (in place)."""
    # Problem shortcut.
    prob = rule.prob
    # New orbit lists.
    pids = []
    # New coordinates.
    crds = []
    # New weights.
    wts = []
    # Loop over orbits.
    for p, c, w in zip(rule.pat_ids, rule.coords, rule.weights):
        # Drop orbits with vanishing weight.
        if w < wtol:
            continue
        # Generator point.
        x = prob.patterns[p].A @ c
        # Actual number of distinct points.
        size = orbit_points(prob.perms, prob.signs, x).shape[0]
        # Degenerate orbit: find the smaller family that describes it.
        if size < prob.patterns[p].size:
            # Orbit points.
            orb = orbit_points(prob.perms, prob.signs, x)
            # Search families of the right size.
            for q, pat in enumerate(prob.patterns):
                # Only families with matching size.
                if pat.size != size:
                    continue
                # Try every orbit point.
                v = None
                # Loop over the orbit.
                for y in orb:
                    # Fit.
                    v = pat.fit(y, tol=1e-12)
                    # Stop on success.
                    if v is not None:
                        break
                # Re-label on success.
                if v is not None:
                    p = q
                    c = v
                    break
        # Keep the orbit.
        pids.append(p)
        # Keep its coordinates.
        crds.append(c)
        # Keep its weight.
        wts.append(w)
    # Write back.
    rule.pat_ids = pids
    # Coordinates.
    rule.coords = crds
    # Weights.
    rule.weights = np.array(wts)


def actual_points(rule):
    """Expand the rule into all points and weights (merging duplicates)."""
    # Problem shortcut.
    prob = rule.prob
    # Point list.
    X = []
    # Weight list.
    W = []
    # Loop over orbits.
    for p, c, w in zip(rule.pat_ids, rule.coords, rule.weights):
        # Distinct orbit points.
        orb = orbit_points(prob.perms, prob.signs, prob.patterns[p].A @ c)
        # Append the points.
        X.append(orb)
        # Equal share of the orbit weight.
        W.append(np.full(orb.shape[0], w / orb.shape[0]))
    # Stack.
    X = np.vstack(X)
    # Concatenate.
    W = np.concatenate(W)
    # Merge points that coincide across orbits.
    keys = np.round(X * 1e10).astype(np.int64)
    # Unique rows and inverse map.
    uk, inv = np.unique(keys, axis=0, return_inverse=True)
    # Merged weights.
    Wm = np.zeros(uk.shape[0])
    # Accumulate.
    np.add.at(Wm, inv.ravel(), W)
    # Representative coordinates.
    Xm = np.zeros((uk.shape[0], X.shape[1]))
    # Take the first occurrence.
    Xm[inv.ravel()] = X
    # Return points and weights (probability measure).
    return Xm, Wm


# ---------------------------------------------------------------------------
# Initial rule: tensor Gauss-Legendre decomposed into orbits
# ---------------------------------------------------------------------------
def initial_rule(prob, k):
    """Tensor Gauss-Legendre rule with k points per direction, as orbits."""
    # One-dimensional nodes and weights.
    t, w = np.polynomial.legendre.leggauss(k)
    # Normalise to the probability measure on [-1,1].
    w = w / 2.0
    # All tensor points.
    grid = np.array(list(itertools.product(range(k), repeat=prob.d)))
    # Coordinates.
    X = t[grid]
    # Weights.
    W = np.prod(w[grid], axis=1)
    # Assignment flags.
    used = np.zeros(X.shape[0], dtype=bool)
    # Fast lookup of grid points.
    lookup = {tuple(np.round(x * 1e10).astype(np.int64)): i for i, x in enumerate(X)}
    # Output lists.
    pids = []
    # Coordinates.
    crds = []
    # Weights.
    wts = []
    # Loop over all points.
    for i in range(X.shape[0]):
        # Skip assigned points.
        if used[i]:
            continue
        # Orbit of the point.
        orb = orbit_points(prob.perms, prob.signs, X[i])
        # Mark the orbit points as used.
        for y in orb:
            # Index of the orbit point.
            j = lookup[tuple(np.round(y * 1e10).astype(np.int64))]
            # Flag.
            used[j] = True
        # Find a family that represents this orbit.
        found = False
        # Loop over patterns.
        for q, pat in enumerate(prob.patterns):
            # Size must agree.
            if pat.size != orb.shape[0]:
                continue
            # Try all orbit points.
            for y in orb:
                # Fit.
                v = pat.fit(y)
                # Success.
                if v is not None:
                    found = True
                    break
            # Stop searching.
            if found:
                break
        # Should never fail.
        if not found:
            raise RuntimeError("no pattern for orbit")
        # Store the orbit.
        pids.append(q)
        # Coordinates.
        crds.append(v)
        # Total orbit weight.
        wts.append(W[i] * orb.shape[0])
    # Build the rule.
    return Rule(prob, pids, crds, wts)


def augment(rule, extra, rng, wsmall=1e-6):
    """Add extra orbits (dict family index -> count) with tiny weights.

    An exact rule plus orbits of (almost) zero weight stays (almost) exact,
    so the subsequent re-solve converges immediately.  The additional cheap
    orbits give the elimination the freedom to remove expensive orbits.
    """
    # Work on a copy.
    new = rule.copy()
    # Loop over the requested families.
    for q, cnt in extra.items():
        # Add cnt orbits.
        for _ in range(cnt):
            # Family index.
            new.pat_ids.append(q)
            # Random coordinates.
            new.coords.append(rng.uniform(0.05, 1.0, rule.prob.patterns[q].dim))
            # Tiny weight.
            new.weights = np.append(new.weights, wsmall)
    # Renormalise.
    new.weights /= new.weights.sum()
    # Re-solve.
    solve_rule(new)
    # Return.
    return new


# ---------------------------------------------------------------------------
# Elimination moves
# ---------------------------------------------------------------------------
def generate_moves(rule, max_dist=0.35):
    """All single-orbit deletions and demotions with their initial guesses.

    Returns a list of (saved_points, new_rule) pairs.
    """
    # Problem shortcut.
    prob = rule.prob
    # Output.
    moves = []
    # Loop over orbits.
    for o, (p, c) in enumerate(zip(rule.pat_ids, rule.coords)):
        # Size of this orbit.
        size = prob.patterns[p].size
        # Deletion: drop the orbit and spread its weight proportionally.
        new = rule.copy()
        # Remove pattern id.
        del new.pat_ids[o]
        # Remove coordinates.
        del new.coords[o]
        # Remove the weight.
        w_removed = new.weights[o]
        # Delete the entry.
        new.weights = np.delete(new.weights, o)
        # Re-scale the remaining weights to keep the total mass.
        if new.weights.size:
            new.weights *= 1.0 / (1.0 - w_removed) if w_removed < 1 else 1.0
        # Record the move (saved points, rule, orbit, target family or None).
        moves.append((size, new, o, None))
        # Generator of the orbit.
        x = prob.patterns[p].A @ c
        # All images of the generator.
        imgs = apply_group(prob.perms, prob.signs, x[None, :])[:, 0, :]
        # Demotions to smaller families.
        for q, pat in enumerate(prob.patterns):
            # Only strictly smaller orbits.
            if pat.size >= size:
                continue
            # Best projection over all images.
            best = None
            # Loop over images.
            for y in imgs:
                # Least-squares coordinates in the smaller family.
                v = pat.Apinv @ y
                # Clip into the box.
                v = np.clip(v, -1.0, 1.0)
                # Distance to the projection.
                dist = np.linalg.norm(pat.A @ v - y)
                # Keep the closest.
                if best is None or dist < best[0]:
                    best = (dist, v)
            # Skip far-away projections.
            if best[0] > max_dist:
                continue
            # Build the demoted rule.
            new = rule.copy()
            # New family.
            new.pat_ids[o] = q
            # New coordinates.
            new.coords[o] = best[1]
            # Record the move.
            moves.append((size - pat.size, new, o, q))
    # Return all moves.
    return moves


def fast_scores(rule, moves, eps=1e-13):
    """Linearised-correction scores for many moves using one Jacobian.

    For a modified rule with Jacobian J' and residual r', the minimum-norm
    correction delta solves J' delta = -r' and satisfies
        ||delta||^2 = r'^T (J' J'^T)^{-1} r'.
    J' differs from the current Jacobian J only in the columns of the moved
    orbit, so J' J'^T is a low-rank update of J J^T, and only an
    (neq x neq) solve per move is needed.  Variables are scaled so that the
    weights are O(1).  The current rule must be (nearly) exact.
    """
    # Problem shortcut.
    prob = rule.prob
    # Current residual and Jacobian.
    r, J = rule.residual_jacobian()
    # Values and gradients at the generators.
    F, dF = prob.eval_basis(rule.generators())
    # Number of coordinate unknowns.
    nc = sum(c.size for c in rule.coords)
    # Weight scaling.
    wscale = max(float(np.mean(rule.weights)), 1e-300)
    # Scaled Jacobian (weights measured in units of wscale).
    Js = J.copy()
    # Scale the weight columns.
    Js[:, nc:] *= wscale
    # Gram matrix.
    M = Js @ Js.T
    # Regularisation.
    reg = eps * np.trace(M) / M.shape[0]
    # Inverse of the regularised Gram matrix (used for Woodbury updates).
    Minv = np.linalg.inv(M + reg * np.eye(M.shape[0]))
    # Scores.
    out = np.empty(len(moves))
    # Collect the new generator points of all demotions for one batched evaluation.
    newpts = []
    # Map from move index to row in newpts.
    rowmap = {}
    # Loop over moves.
    for idx, (saved, new, o, q) in enumerate(moves):
        # Demotions have a new orbit.
        if q is not None:
            # Remember the row.
            rowmap[idx] = len(newpts)
            # Generator of the demoted orbit.
            newpts.append(prob.patterns[q].A @ new.coords[o])
    # Batched evaluation.
    if newpts:
        Fn, dFn = prob.eval_basis(np.array(newpts))
    # Loop over moves.
    for idx, (saved, new, o, q) in enumerate(moves):
        # Columns of orbit o in the current Jacobian.
        cidx, widx = unknown_slices(rule, o)
        # Old columns (scaled).
        Uold = Js[:, list(cidx) + [widx]]
        # Weight of the orbit.
        To = rule.weights[o]
        # Residual change: remove the old orbit contribution.
        rp = r - F[:, o] * To
        # Low-rank factors of the Gram update: M' = M + U C U^T.
        U = Uold
        # Signs: removed columns.
        csig = -np.ones(Uold.shape[1])
        # Demotion: add the new orbit.
        if q is not None:
            # Row of the new point.
            k = rowmap[idx]
            # Pattern matrix of the new family.
            A = prob.patterns[q].A
            # New coordinate columns.
            Unew = To * (dFn[:, k, :] @ A) if A.shape[1] > 0 else np.zeros((prob.neq, 0))
            # New weight column (scaled).
            Unew = np.hstack([Unew, wscale * Fn[:, k:k + 1]])
            # Append the new columns.
            U = np.hstack([U, Unew])
            # Signs: added columns.
            csig = np.concatenate([csig, np.ones(Unew.shape[1])])
            # Add the new contribution to the residual.
            rp = rp + Fn[:, k] * To
        # Woodbury: (M + U C U^T)^{-1} = Minv - Minv U (C^{-1} + U^T Minv U)^{-1} U^T Minv.
        MU = Minv @ U
        # Small capacitance matrix.
        cap = np.diag(1.0 / csig) + U.T @ MU
        # Minv applied to the residual.
        Mr = Minv @ rp
        # Solve the small system.
        try:
            corr = np.linalg.solve(cap, U.T @ Mr)
        except np.linalg.LinAlgError:
            out[idx] = np.inf
            continue
        # Updated solution lambda = M'^{-1} r'.
        lam = Mr - MU @ corr
        # Squared norm of the minimum-norm correction.
        out[idx] = math.sqrt(max(float(rp @ lam), 0.0))
    # Return the scores.
    return out


def score_moves(rule, moves, apply_correction=True):
    """Slobodkins-Tausch type score: norm of the min-norm linearised correction."""
    # Scores.
    scores = []
    # Loop over moves.
    for saved, new, *_ in moves:
        # Residual and Jacobian of the modified rule at its initial guess.
        r, J = new.residual_jacobian()
        # Minimum-norm correction of the linearised system.
        dz = np.linalg.lstsq(J, -r, rcond=1e-12)[0]
        # Remaining inconsistency of the linear model.
        incons = np.linalg.norm(J @ dz + r)
        # Number of coordinate unknowns.
        nc = sum(c.size for c in new.coords)
        # Typical weight scale.
        wscale = max(np.mean(new.weights), 1e-300) if new.weights.size else 1.0
        # Scaled correction norm.
        step = math.sqrt(np.sum(dz[:nc] ** 2) + np.sum((dz[nc:] / wscale) ** 2))
        # Penalise inconsistency strongly.
        scores.append(step + 1e3 * incons)
        # Use the linearised correction as a better starting point (projected into the box).
        if apply_correction:
            # Bounds of the modified rule.
            lo, hi = new.bounds()
            # Corrected and projected unknowns.
            new.unpack(np.clip(new.pack() + dz, lo, hi))
    # Return as array.
    return np.array(scores)


def unknown_slices(rule, o):
    """Index ranges of orbit o's coordinates and weight in the packed vector."""
    # Offset of the coordinates of orbit o.
    off = sum(c.size for c in rule.coords[:o])
    # Coordinate indices.
    cidx = np.arange(off, off + rule.coords[o].size)
    # Total number of coordinate unknowns.
    nc = sum(c.size for c in rule.coords)
    # Weight index.
    widx = nc + o
    # Return both.
    return cidx, widx


def homotopy_delete(rule, o, steps=(0.3, 0.6, 0.85, 1.0), min_dt=0.004, max_steps=200):
    """Remove orbit o by driving its weight to zero along a continuation path.

    The weight of orbit o is frozen at (1-t) w0 while all other unknowns
    follow; the step in t is halved on failure.  If the path reaches the
    boundary of the feasible set because the weight of ANOTHER orbit drops
    to zero, that orbit is removed as well and the continuation goes on.
    Returns an exact rule with fewer points or None.
    """
    # Work on a copy.
    cur = rule.copy()
    # Initial weight.
    w0 = cur.weights[o]
    # Current homotopy parameter.
    t = 0.0
    # Current step.
    dt = steps[0]
    # Step counter.
    nstep = 0
    # Continuation loop.
    while t < 1.0 and nstep < max_steps:
        # Count the step.
        nstep += 1
        # Trial parameter.
        tt = min(1.0, t + dt)
        # Trial rule.
        trial = cur.copy()
        # Prescribed weight.
        trial.weights[o] = (1.0 - tt) * w0
        # Frozen mask.
        fixed = np.zeros(trial.pack().size, dtype=bool)
        # Freeze the weight of orbit o.
        fixed[unknown_slices(trial, o)[1]] = True
        # Solve the remaining unknowns.
        ok, _ = solve_rule(trial, fixed=fixed)
        # Accept or shrink the step.
        if ok:
            cur = trial
            t = tt
            dt = min(2.0 * dt, 0.5)
            continue
        # Shrink the step.
        dt *= 0.5
        # Below the minimal step: try to drop other orbits whose weight vanished.
        if dt < min_dt:
            # Orbits (other than o) with (almost) zero weight.
            dead = [k for k in range(len(cur.weights)) if k != o and cur.weights[k] < 1e-6 * float(np.mean(cur.weights))]
            # Nothing to drop: give up.
            if not dead:
                return None
            # Remove them (from the back so that indices stay valid).
            for k in sorted(dead, reverse=True):
                # Remove the pattern id.
                del cur.pat_ids[k]
                # Remove the coordinates.
                del cur.coords[k]
                # Remove the weight.
                cur.weights = np.delete(cur.weights, k)
                # Shift the index of orbit o.
                if k < o:
                    o -= 1
            # Re-solve with the weight of o frozen at its current value.
            fixed = np.zeros(cur.pack().size, dtype=bool)
            # Mark the weight.
            fixed[unknown_slices(cur, o)[1]] = True
            # Solve.
            ok, _ = solve_rule(cur, fixed=fixed)
            # Failure.
            if not ok:
                return None
            # Restart with a moderate step.
            dt = 0.05
    # Not finished.
    if t < 1.0:
        return None
    # Remove the now empty orbit.
    del cur.pat_ids[o]
    # Remove its coordinates.
    del cur.coords[o]
    # Remove its (zero) weight.
    cur.weights = np.delete(cur.weights, o)
    # Final polish.
    ok, _ = solve_rule(cur)
    # Clean up.
    if ok:
        canonicalize(cur)
        ok, _ = solve_rule(cur)
    # Return the result.
    return cur if ok else None


def homotopy_demote(rule, o, q, v_target, min_dt=0.05):
    """Move orbit o continuously onto family q (coordinates v_target).

    The generator of orbit o is frozen on the straight line from its current
    position to the projection onto family q; the rest of the rule follows.
    """
    # Problem shortcut.
    prob = rule.prob
    # Work on a copy.
    cur = rule.copy()
    # Old pattern.
    pold = prob.patterns[cur.pat_ids[o]]
    # Start point.
    x0 = pold.A @ cur.coords[o]
    # All images of the target point.
    target = prob.patterns[q].A @ v_target
    # Choose the image of the target closest to x0 (in old coordinates).
    imgs = apply_group(prob.perms, prob.signs, target[None, :])[:, 0, :]
    # Images that are representable in the old pattern.
    best = None
    # Loop over the images.
    for y in imgs:
        # Coordinates in the old pattern.
        v = np.linalg.lstsq(pold.A, y, rcond=None)[0]
        # Must be exactly representable.
        if np.max(np.abs(pold.A @ v - y)) > 1e-12:
            continue
        # Distance in parameter space.
        dist = np.linalg.norm(v - cur.coords[o])
        # Keep the closest.
        if best is None or dist < best[0]:
            best = (dist, v)
    # Not reachable within the old pattern.
    if best is None:
        return None
    # Parameter start and end.
    v0 = cur.coords[o].copy()
    # End point.
    v1 = best[1]
    # Homotopy state.
    t = 0.0
    # Initial step.
    dt = 0.5
    # Continuation loop.
    while t < 1.0:
        # Trial parameter.
        tt = min(1.0, t + dt)
        # Trial rule.
        trial = cur.copy()
        # Prescribed coordinates.
        trial.coords[o] = (1.0 - tt) * v0 + tt * v1
        # Frozen mask.
        fixed = np.zeros(trial.pack().size, dtype=bool)
        # Freeze the coordinates of orbit o.
        fixed[unknown_slices(trial, o)[0]] = True
        # Solve.
        ok, _ = solve_rule(trial, fixed=fixed)
        # Accept or shrink the step.
        if ok:
            cur = trial
            t = tt
            dt = min(2.0 * dt, 0.5)
        else:
            dt *= 0.5
            # Give up.
            if dt < min_dt:
                return None
    # Re-label the orbit with the smaller family.
    canonicalize(cur)
    # Final polish.
    ok, _ = solve_rule(cur)
    # Return.
    return cur if ok and cur.npoints() < rule.npoints() else None


def homotopy_candidates(rule):
    """List (saved, kind, o, q, v) of deletion / demotion homotopies."""
    # Problem shortcut.
    prob = rule.prob
    # Output list.
    out = []
    # Loop over orbits.
    for o, (p, c) in enumerate(zip(rule.pat_ids, rule.coords)):
        # Orbit size.
        size = prob.patterns[p].size
        # Deletion.
        out.append((size, "del", o, None, None))
        # Generator.
        x = prob.patterns[p].A @ c
        # Images.
        imgs = apply_group(prob.perms, prob.signs, x[None, :])[:, 0, :]
        # Demotions.
        for q, pat in enumerate(prob.patterns):
            # Only smaller families with free coordinates or the centre.
            if pat.size >= size:
                continue
            # Closest projection.
            best = None
            # Loop over images.
            for y in imgs:
                # Projection coordinates.
                v = pat.Apinv @ y
                # Clip.
                v = np.clip(v, -1.0, 1.0)
                # Distance.
                dist = np.linalg.norm(pat.A @ v - y)
                # Keep the closest.
                if best is None or dist < best[0]:
                    best = (dist, v)
            # Record.
            out.append((size - pat.size, "dem", o, q, best[1]))
    # Return.
    return out


def homotopy_step(rule, max_try=10, power=2.0, rng=None, noise=0.0):
    """Try continuation moves ordered by saved points and orbit weight."""
    # Candidates.
    cands = homotopy_candidates(rule)
    # Priority: many saved points, small weight (deletion) or short distance (demotion).
    keys = []
    # Loop over candidates.
    for saved, kind, o, q, v in cands:
        # Weight-based cost for deletions.
        if kind == "del":
            cost = rule.weights[o] * len(rule.weights)
        else:
            # Parameter distance for demotions.
            cost = np.linalg.norm(rule.prob.patterns[q].A @ v - rule.prob.patterns[rule.pat_ids[o]].A @ rule.coords[o]) + 1e-3
        # Combine with the saving.
        keys.append(cost / saved ** power)
    # Random perturbation.
    keys = np.array(keys)
    # Apply noise.
    if rng is not None and noise > 0:
        keys = keys * np.exp(noise * rng.standard_normal(keys.size))
    # Try in order.
    for idx in np.argsort(keys)[:max_try]:
        # Unpack.
        saved, kind, o, q, v = cands[idx]
        # Run the continuation.
        res = homotopy_delete(rule, o) if kind == "del" else homotopy_demote(rule, o, q, v)
        # Accept the first success.
        if res is not None and res.npoints() < rule.npoints():
            return res
    # Nothing worked.
    return None


def escape_step(rule, rng, tries=30, max_try=6):
    """Node-addition escape from a stalled elimination.

    One or two cheap orbits are added at random positions with a small weight
    (an exact rule plus a zero-weight orbit is still exact, so the re-solve is
    easy).  Then a deletion/demotion of an ORIGINAL orbit that saves more
    points than were added is forced.  Returns an improved exact rule or None.
    """
    # Problem shortcut.
    prob = rule.prob
    # Current number of points.
    m0 = rule.npoints()
    # Largest orbit size present.
    smax = max(prob.patterns[p].size for p in rule.pat_ids)
    # Families that are cheaper than the largest orbit and have free coordinates.
    cheap = [q for q, pat in enumerate(prob.patterns) if pat.size < smax and pat.dim > 0]
    # Nothing to add.
    if not cheap:
        return None
    # Number of original orbits.
    norig = len(rule.pat_ids)
    # Random trials.
    for _ in range(tries):
        # Copy of the rule.
        aug = rule.copy()
        # Number of orbits to add.
        nadd = int(rng.integers(1, 3))
        # Added points.
        added = 0
        # Add the orbits.
        for _ in range(nadd):
            # Random cheap family.
            q = int(rng.choice(cheap))
            # Append the family.
            aug.pat_ids.append(q)
            # Random coordinates.
            aug.coords.append(rng.uniform(0.0, 1.0, prob.patterns[q].dim))
            # Small weight.
            aug.weights = np.append(aug.weights, 1e-3)
            # Count points.
            added += prob.patterns[q].size
        # Renormalise the total mass.
        aug.weights /= aug.weights.sum()
        # Make the augmented rule exact again.
        ok, _ = solve_rule(aug)
        # Skip failures.
        if not ok:
            continue
        # Moves of the augmented rule.
        moves = generate_moves(aug)
        # Keep moves that act on original orbits and save more than was added.
        keep = []
        # Loop over moves.
        for saved, new, *_ in moves:
            # Net saving required.
            if saved <= added:
                continue
            # The move must leave all added orbits in place (they are at the end).
            if new.pat_ids[-nadd:] != aug.pat_ids[-nadd:] or len(new.pat_ids) < norig - 1 + nadd:
                continue
            # Keep the move.
            keep.append((saved, new, None, None))
        # Nothing promising.
        if not keep:
            continue
        # Score the moves.
        sc = score_moves(aug, keep)
        # Try the best ones.
        for idx in np.argsort(sc)[:max_try]:
            # Candidate.
            cand = keep[idx][1].copy()
            # Solve.
            ok, _ = solve_rule(cand)
            # Clean up and polish on success.
            if ok:
                # Remove degenerate or empty orbits.
                canonicalize(cand)
                # Polish.
                ok, _ = solve_rule(cand)
            # Accept only a real improvement.
            if ok and cand.npoints() < m0:
                return cand
        # Continuation deletion of the largest original orbits (smallest weights first).
        order = sorted(range(norig), key=lambda o: (-prob.patterns[aug.pat_ids[o]].size, aug.weights[o]))
        # Only orbits whose removal saves more than was added.
        order = [o for o in order if prob.patterns[aug.pat_ids[o]].size > added]
        # Try a few.
        for o in order[:3]:
            # Continuation deletion.
            res = homotopy_delete(aug, o)
            # Accept improvements.
            if res is not None and res.npoints() < m0:
                return res
    # No improvement found.
    return None


def eliminate(rule, max_try=12, rng=None, noise=0.0, log=None, time_limit=None, escape_tries=0, power=2.0, homotopy_tries=0, lexi=False):
    """Greedy (optionally randomised) orbit elimination starting from an exact rule."""
    # Start time.
    t0 = time.time()
    # Work on a copy.
    cur = rule.copy()
    # Random generator for the escape moves.
    erng = rng if rng is not None else np.random.default_rng(0)
    # Main loop.
    while True:
        # Time limit check.
        if time_limit is not None and time.time() - t0 > time_limit:
            break
        # All candidate moves.
        moves = generate_moves(cur)
        # Nothing to do.
        if not moves:
            break
        # Scores (fast low-rank formula).
        sc = fast_scores(cur, moves)
        # Prefer moves that save more points: divide by saved**power.
        sc = sc / np.array([m[0] for m in moves], dtype=float) ** power
        # Random perturbation for diversified restarts.
        if rng is not None and noise > 0:
            sc = sc * np.exp(noise * rng.standard_normal(sc.size))
        # Order of trial.
        order = np.argsort(sc)
        # Lexicographic mode: biggest saving first, then score (Diallo-Worku-like priority).
        if lexi:
            # Savings of the moves.
            saved = np.array([m[0] for m in moves])
            # Collect the best max_try moves of every saving class, largest class first.
            order = np.concatenate([[i for i in order if saved[i] == sv][:max_try] for sv in sorted(set(saved), reverse=True)]).astype(int)
        else:
            # Plain mode: only the best max_try moves overall.
            order = order[:max_try]
        # Success flag.
        ok = False
        # Try the candidates.
        for idx in order:
            # Candidate rule.
            cand = moves[idx][1].copy()
            # Apply the minimum-norm linearised correction as starting point.
            score_moves(cand, [(0, cand)])
            # Re-solve.
            ok, res = solve_rule(cand)
            # Accept on convergence.
            if ok:
                # Clean up.
                canonicalize(cand)
                # Polish again after clean-up.
                ok, res = solve_rule(cand)
            # Accept the move.
            if ok:
                cur = cand
                break
        # Continuation moves if the direct jumps failed.
        if not ok and homotopy_tries > 0:
            # Attempt the continuation.
            hom = homotopy_step(cur, max_try=homotopy_tries, power=power, rng=rng, noise=noise)
            # Accept.
            if hom is not None:
                cur = hom
                ok = True
                # Log.
                if log is not None:
                    log("    homotopy -> m=%d" % cur.npoints())
        # Escape by node addition if plain elimination is stuck.
        if not ok and escape_tries > 0:
            # Attempt the escape.
            esc = escape_step(cur, erng, tries=escape_tries)
            # Continue eliminating from the improved rule.
            if esc is not None:
                cur = esc
                ok = True
                # Log the escape.
                if log is not None:
                    log("    escape -> m=%d" % cur.npoints())
        # Stop if no move succeeded.
        if not ok:
            break
        # Progress log.
        if log is not None:
            log("    m=%d orbits=%d" % (cur.npoints(), len(cur.pat_ids)))
    # Return the reduced rule.
    return cur


def perturb(rule, rng, nadd, frac=0.5, target=None):
    """Add nadd random orbits and move real weight into them.

    The orbits are added with a tiny weight (keeps exactness), then their
    weights are raised to frac * (mean orbit weight) by continuation with the
    new weights frozen, so that the rule moves inside the solution manifold.
    Returns the perturbed exact rule (or None).
    """
    # Problem shortcut.
    prob = rule.prob
    # Families with free coordinates.
    fams = [q for q, p in enumerate(prob.patterns) if p.dim > 0]
    # Choose the families of the new orbits.
    extra = {}
    # Selection probabilities: uniform, or proportional to the deficit w.r.t. the target structure.
    prob_f = np.ones(len(fams))
    # Targeted mode.
    if target is not None:
        # Current counts per family.
        cur = np.array([rule.pat_ids.count(q) for q in fams])
        # Deficit (plus a small floor so that every family stays possible).
        prob_f = np.maximum(np.array([target[q] for q in fams]) - cur, 0) + 0.3
    # Normalise.
    prob_f = prob_f / prob_f.sum()
    # Draw nadd families.
    for _ in range(nadd):
        # Random family.
        q = int(rng.choice(fams, p=prob_f))
        # Count it.
        extra[q] = extra.get(q, 0) + 1
    # Add with tiny weights.
    cur = augment(rule, extra, rng)
    # Indices of the added orbits.
    added = list(range(len(rule.pat_ids), len(cur.pat_ids)))
    # Target weight.
    target = frac * float(np.mean(rule.weights))
    # Continuation in the weight.
    for scale in (1.0, 0.3, 0.1, 0.03):
        # Trial copy.
        trial = cur.copy()
        # Set the new weights.
        for o in added:
            # Prescribed weight.
            trial.weights[o] = scale * target
        # Freeze them.
        fixed = np.zeros(trial.pack().size, dtype=bool)
        # Mark the weights.
        for o in added:
            # Weight index.
            fixed[unknown_slices(trial, o)[1]] = True
        # Solve.
        ok, _ = solve_rule(trial, fixed=fixed)
        # Release the weights and polish.
        if ok:
            # Unfrozen polish.
            ok, _ = solve_rule(trial)
        # Accept the first success.
        if ok:
            return trial
    # Fall back to the tiny-weight version.
    return cur


def drive_to_target(rule, target, rng, log=None, max_fail=6):
    """Steer an exact rule towards the target orbit structure.

    1. Add the missing orbits of every family with a deficit (exactness is
       preserved; their weights are raised by continuation).
    2. Remove surplus orbits of the families with an excess, largest family
       first, each by a weight homotopy (homotopy_delete) so that the rule
       stays exact along the way; the smallest-weight orbits are tried first.
    3. Finish with an ordinary elimination.
    Returns the resulting exact rule.
    """
    # Problem shortcut.
    prob = rule.prob
    # Current counts.
    cur = np.array([rule.pat_ids.count(q) for q in range(len(prob.patterns))])
    # Deficits of families with free coordinates.
    extra = {q: int(target[q] - cur[q]) for q in range(len(prob.patterns)) if prob.patterns[q].dim > 0 and target[q] > cur[q]}
    # Add the missing orbits.
    work = rule.copy()
    # One family at a time (continuation in the weights).
    for q, cnt in extra.items():
        # Add cnt orbits of family q.
        for _ in range(cnt):
            # Add one orbit with tiny weight.
            work = augment(work, {q: 1}, rng)
            # Index of the new orbit.
            o = len(work.pat_ids) - 1
            # Raise its weight to half of the mean by continuation.
            for scale in (0.5, 0.15, 0.05):
                # Trial copy.
                trial = work.copy()
                # Prescribed weight.
                trial.weights[o] = scale * float(np.mean(work.weights))
                # Freeze it.
                fixed = np.zeros(trial.pack().size, dtype=bool)
                # Mark the weight.
                fixed[unknown_slices(trial, o)[1]] = True
                # Solve.
                ok, _ = solve_rule(trial, fixed=fixed)
                # Accept.
                if ok:
                    work = trial
                    break
    # Families sorted by orbit size (largest first).
    order_f = sorted(range(len(prob.patterns)), key=lambda q: -prob.patterns[q].size)
    # Remove the surplus.
    for q in order_f:
        # Failures for this family.
        fails = 0
        # While the family has an excess.
        while work.pat_ids.count(q) > target[q] and fails < max_fail:
            # Orbits of this family sorted by weight.
            cands = sorted([o for o, p in enumerate(work.pat_ids) if p == q], key=lambda o: work.weights[o])
            # Success flag.
            done = False
            # Try the lightest orbits first.
            for o in cands[:3]:
                # Weight homotopy.
                res = homotopy_delete(work, o)
                # Accept.
                if res is not None:
                    work = res
                    done = True
                    break
            # Count failures.
            if not done:
                fails += 1
                # Give up on this family.
                break
        # Log.
        if log is not None:
            log("    drive: family %s now %d (target %d), m=%d" % (prob.patterns[q].name(), work.pat_ids.count(q), target[q], work.npoints()))
    # Final ordinary elimination.
    return eliminate(work, rng=rng, noise=0.2, power=3.0)


def lns(rule, rng, iters=50, time_limit=None, log=None, max_add=4, temp0=6.0, target=None, on_improve=None):
    """Iterated large-neighbourhood search: perturb (add orbits) + re-eliminate.

    Accepts the new rule if it does not have more points (plateau moves are
    allowed); returns the best rule found.
    """
    # Start time.
    t0 = time.time()
    # Current rule.
    cur = rule.copy()
    # Best rule.
    best = rule.copy()
    # Iterations.
    for it in range(iters):
        # Time limit.
        if time_limit is not None and time.time() - t0 > time_limit:
            break
        # Number of added orbits.
        nadd = int(rng.integers(1, max_add + 1))
        # Targeted perturbation in half of the iterations (if a target is given).
        tgt = target if (target is not None and rng.random() < 0.5) else None
        # Perturb.
        pert = perturb(cur, rng, nadd, target=tgt)
        # Size preference (stronger when targeting: remove the big orbits).
        power = float(rng.choice([2.0, 3.0, 4.0])) if tgt is not None else float(rng.choice([1.0, 2.0, 3.0]))
        # Re-eliminate with randomised ordering.
        cand = eliminate(pert, rng=rng, noise=float(rng.choice([0.2, 0.5])), power=power)
        # Simulated-annealing temperature (points), decreasing linearly in time or iterations.
        progress = (time.time() - t0) / time_limit if time_limit else it / max(iters, 1)
        # Temperature.
        temp = temp0 * max(1.0 - progress, 0.0)
        # Increase in points.
        delta = cand.npoints() - cur.npoints()
        # Metropolis acceptance (plateau and improving moves always accepted).
        if delta <= 0 or (temp > 0 and rng.random() < math.exp(-delta / temp)):
            cur = cand
        # Track the best.
        if cand.npoints() < best.npoints():
            best = cand.copy()
            # Log improvements.
            if log is not None:
                log("    lns it %d -> m=%d" % (it, best.npoints()))
            # Checkpoint callback (e.g. write the rule to disk).
            if on_improve is not None:
                on_improve(best)
    # Return the best rule.
    return best


# ---------------------------------------------------------------------------
# Exhaustive structure search for small degrees (polyquad-like)
# ---------------------------------------------------------------------------
def rank_caps(prob, seed=0):
    """Rank of the Jacobian columns generated by every subset of families.

    For a subset S of families, rank_S is the dimension of the span of the
    Jacobian columns of arbitrarily many generic orbits from S.  A structure
    with dof_S unknowns in S and dof_total unknowns overall can only have a
    full-rank Jacobian if  dof_S - rank_S <= dof_total - neq  for every S
    (cf. the consistency criteria of Witherden & Vincent, 2015).
    """
    # Random generator.
    rng = np.random.default_rng(seed)
    # Number of families.
    P = len(prob.patterns)
    # Jacobian blocks of large random pools per family.
    blocks = []
    # Loop over families.
    for f, pat in enumerate(prob.patterns):
        # Pool size.
        cnt = 1 if pat.dim == 0 else prob.neq
        # Random rule consisting of this family only.
        rule = Rule(prob, [f] * cnt, [np.cos(np.pi * rng.uniform(0.0, 1.0, pat.dim)) for _ in range(cnt)], rng.uniform(0.5, 1.5, cnt))
        # Its Jacobian (Chebyshev-distributed samples keep it well conditioned at high degree).
        blocks.append(rule.residual_jacobian()[1])
    # Ranks of all subsets (bit masks).
    ranks = {}
    # Loop over non-empty subsets.
    for mask in range(1, 2 ** P):
        # Columns of the subset.
        J = np.hstack([blocks[f] for f in range(P) if mask >> f & 1])
        # Singular values.
        sv = np.linalg.svd(J, compute_uv=False)
        # Numerical rank.
        ranks[mask] = int(np.sum(sv > 1e-11 * sv[0]))
    # Return the table.
    return ranks


def structure_target(prob):
    """Cheapest orbit structure that satisfies all rank conditions (integer LP).

    minimise   sum_f c_f size_f
    subject to sum_{f not in S} c_f dof_f >= neq - rank_S   for every subset S,
               c_f >= 0 integer, at most one centre.
    Returns (points, counts) or (None, None).  Used as a lower estimate and
    as a target for the large-neighbourhood search.
    """
    # Local import (scipy is only needed here).
    from scipy.optimize import milp, LinearConstraint, Bounds
    # Rank table.
    ranks = rank_caps(prob)
    # Number of families.
    P = len(prob.patterns)
    # Orbit sizes.
    size = np.array([p.size for p in prob.patterns], dtype=float)
    # Unknowns per orbit.
    dof = np.array([p.dim + 1 for p in prob.patterns], dtype=float)
    # Constraint rows.
    rows = [dof.copy()]
    # Lower bounds.
    lbs = [prob.neq]
    # One row per subset.
    for mask, rk in ranks.items():
        # Unknowns outside the subset.
        rows.append(np.array([0.0 if mask >> f & 1 else dof[f] for f in range(P)]))
        # Required amount.
        lbs.append(prob.neq - rk)
    # Upper bounds (centre at most once).
    ub = np.array([1.0 if p.dim == 0 else np.inf for p in prob.patterns])
    # Solve the integer program.
    res = milp(size, constraints=LinearConstraint(np.array(rows), lbs, np.inf), integrality=np.ones(P), bounds=Bounds(0, ub))
    # Infeasible or failed.
    if res.x is None:
        return None, None
    # Return points and counts.
    return int(round(res.fun)), np.round(res.x).astype(int)


def rank_feasible(prob, counts, ranks):
    """Check the necessary rank conditions for a structure (counts per family)."""
    # Unknowns per family.
    dof = [c * (p.dim + 1) for c, p in zip(counts, prob.patterns)]
    # Surplus of unknowns over equations.
    surplus = sum(dof) - prob.neq
    # Not enough unknowns at all.
    if surplus < 0:
        return False
    # Check every subset.
    for mask, rk in ranks.items():
        # Unknowns inside the subset.
        ds = sum(d for f, d in enumerate(dof) if mask >> f & 1)
        # Violated condition.
        if ds - rk > surplus:
            return False
    # All conditions hold.
    return True


def structures_with_points(prob, m, max_count=None):
    """Yield count vectors c (orbits per pattern) with exactly m points and enough unknowns."""
    # Sizes of the patterns.
    sizes = [p.size for p in prob.patterns]
    # Unknowns per orbit.
    dofs = [p.dim + 1 for p in prob.patterns]
    # Number of patterns.
    P = len(sizes)

    # Recursive enumeration.
    def rec(i, remaining, acc):
        # All patterns assigned.
        if i == P:
            # Exactly m points required.
            if remaining == 0:
                # Enough degrees of freedom.
                if sum(a * b for a, b in zip(acc, dofs)) >= prob.neq:
                    yield list(acc)
            return
        # Maximal count for this pattern (dimension-0 patterns at most once).
        cmax = remaining // sizes[i]
        # A pattern without free coordinates can appear only once.
        if prob.patterns[i].dim == 0:
            cmax = min(cmax, 1)
        # Loop over counts.
        for cnt in range(cmax, -1, -1):
            # Recurse.
            yield from rec(i + 1, remaining - cnt * sizes[i], acc + [cnt])

    # Start the recursion.
    yield from rec(0, m, [])


def random_rule(prob, counts, rng):
    """Random initial guess for a given structure."""
    # Pattern ids.
    pids = []
    # Coordinates.
    crds = []
    # Loop over patterns.
    for q, cnt in enumerate(counts):
        # Repeat cnt times.
        for _ in range(cnt):
            # Store the pattern.
            pids.append(q)
            # Random coordinates in (0, 1).
            crds.append(rng.uniform(0.05, 0.98, prob.patterns[q].dim))
    # Random positive weights summing to one.
    wts = rng.dirichlet(np.ones(len(pids)))
    # Build the rule.
    return Rule(prob, pids, crds, wts)


def exhaustive_search(prob, m_lo, m_hi, starts=30, rng=None, log=None, time_limit=60.0):
    """Try all structures with m_lo <= m < m_hi points; return the first rule found."""
    # Start time.
    t0 = time.time()
    # Rank table for pruning.
    ranks = rank_caps(prob)
    # Loop over point counts.
    for m in range(m_lo, m_hi):
        # Loop over structures.
        for counts in structures_with_points(prob, m):
            # Skip structures that violate the rank conditions.
            if not rank_feasible(prob, counts, ranks):
                continue
            # Random starts.
            for s in range(starts):
                # Time limit.
                if time.time() - t0 > time_limit:
                    return None
                # Random initial rule.
                rule = random_rule(prob, counts, rng)
                # Solve.
                ok, res = solve_rule(rule, maxit=60)
                # Success.
                if ok:
                    # Clean up degenerate orbits.
                    canonicalize(rule)
                    # Re-solve.
                    ok, res = solve_rule(rule)
                # Return the first success.
                if ok:
                    # Log it.
                    if log is not None:
                        log("    exhaustive: found m=%d structure %s" % (rule.npoints(), counts))
                    return rule
    # Nothing found.
    return None


# ---------------------------------------------------------------------------
# Verification on the full (non-symmetric) basis
# ---------------------------------------------------------------------------
def verify(X, W, n, d=3):
    """Max error over all orthonormal Legendre products of degree <= n.

    X: points in [-1,1]^d, W: weights summing to one (probability measure).
    """
    # Legendre tables at all coordinates: (n+1, N, d).
    L, _ = legendre_table(X, n)
    # All multi-indices.
    idx = np.array([a for a in itertools.product(range(n + 1), repeat=d) if sum(a) <= n])
    # Product basis values: (K, N).
    V = np.ones((idx.shape[0], X.shape[0]))
    # Multiply coordinate factors.
    for i in range(d):
        V *= L[idx[:, i], :, i]
    # Quadrature approximations of the moments.
    q = V @ W
    # Exact moments.
    exact = np.zeros(idx.shape[0])
    # Constant has moment one.
    exact[np.all(idx == 0, axis=1)] = 1.0
    # Maximal error.
    return float(np.max(np.abs(q - exact)))


# ---------------------------------------------------------------------------
# Lower bounds
# ---------------------------------------------------------------------------
def dim_poly(s, d=3):
    """Dimension of the space of polynomials of total degree <= s in d variables."""
    # Binomial coefficient C(s+d, d).
    return math.comb(s + d, d) if s >= 0 else 0


def lower_bound(n, d=3):
    """max of the classical bound and Moeller's bound (odd n, central symmetry)."""
    # Classical bound for positive rules.
    lb = dim_poly(n // 2, d)
    # Moeller's bound applies to odd degrees n = 2k-1.
    if n % 2 == 1:
        # s = k-1.
        s = (n - 1) // 2
        # Number of monomials of even total degree <= s.
        even = sum(math.comb(j + d - 1, d - 1) for j in range(0, s + 1, 2))
        # Number of monomials of odd total degree <= s.
        odd = sum(math.comb(j + d - 1, d - 1) for j in range(1, s + 1, 2))
        # Moeller: 2*even-1 if s even, 2*odd if s odd.
        mb = 2 * even - 1 if s % 2 == 0 else 2 * odd
        # Take the larger bound.
        lb = max(lb, mb)
    # Return.
    return lb


# ---------------------------------------------------------------------------
# Driver for one degree
# ---------------------------------------------------------------------------
def search_degree(n, group="Oh", restarts=6, time_limit=600.0, exhaustive_time=0.0, seed=1, outdir="rules", quiet=False, escape_tries=60):
    """Find a rule of degree n with as few points as possible."""
    # Start time.
    t0 = time.time()
    # Logger.
    def log(msg):
        # Print unless quiet.
        if not quiet:
            print("[n=%d %s] %s" % (n, group, msg), flush=True)
    # Random generator.
    rng = np.random.default_rng(seed + 1000 * n)
    # Problem setup.
    prob = Problem(n, group)
    # Report the size.
    log("equations=%d patterns=%s" % (prob.neq, [p.name() + ":" + str(p.size) for p in prob.patterns]))
    # Starting rules (exact by construction): Gauss-Legendre with k, k+1, k+2 points.
    k0 = n // 2 + 1
    # Build and polish them.
    starts = []
    # Loop over the three sizes.
    for k in (k0, k0 + 1, k0 + 2):
        # Tensor rule decomposed into orbits.
        st = initial_rule(prob, k)
        # Polish to full accuracy.
        solve_rule(st)
        # Store.
        starts.append(st)
    # Pool of finished rules (point count, run, rule).
    pool = []
    # Phase 1: portfolio of quick randomised eliminations (about 25% of the budget).
    run = 0
    # Loop until the phase-1 budget is used (at least `restarts` runs).
    while run < restarts or time.time() - t0 < 0.25 * time_limit:
        # Hard stop at the full budget.
        if time.time() - t0 > time_limit and pool:
            break
        # Starting rule (cycle through the three Gauss-Legendre sizes).
        st = starts[run % len(starts)]
        # Size preference exponent (cycle).
        power = (2.0, 3.0, 1.0)[(run // len(starts)) % 3]
        # Noise level (deterministic first round).
        noise = 0.0 if run < len(starts) else float(rng.choice([0.3, 0.6]))
        # Eliminate.
        cand = eliminate(st, rng=rng, noise=noise, power=power)
        # Store.
        pool.append((cand.npoints(), run, cand))
        # Next run.
        run += 1
    # Sort the pool by point count.
    pool.sort(key=lambda e: (e[0], e[1]))
    # Log phase 1.
    log("phase1: %d runs, best counts %s  (%.0fs)" % (run, [e[0] for e in pool[:6]], time.time() - t0))
    # Best rule so far.
    best = pool[0][2]
    # Target structure for the targeted large-neighbourhood search.
    _, target = structure_target(prob)
    # Phase 2: large-neighbourhood search with annealing on the three best rules.
    chunk = 0
    # Loop until the time is used up.
    while time.time() - t0 < time_limit:
        # Pick the seed rule (round robin over the best three, then the global best).
        seed_rule = pool[chunk % min(3, len(pool))][2] if chunk < 3 else best
        # Time for this chunk.
        tl = min(0.25 * time_limit, time_limit - (time.time() - t0))
        # Run the LNS.
        res = lns(seed_rule, rng, iters=10 ** 6, time_limit=tl, temp0=6.0 if chunk < 3 else 3.0, target=target, max_add=6)
        # Log.
        log("phase2 chunk %d: %d -> %d  (%.0fs)" % (chunk, seed_rule.npoints(), res.npoints(), time.time() - t0))
        # Keep the best.
        if res.npoints() < best.npoints():
            best = res
        # Next chunk.
        chunk += 1
    # Optional exhaustive search below the best count (small n).
    if exhaustive_time > 0:
        # Lower bound.
        lb = lower_bound(n)
        # Current best count.
        mb = actual_points(best)[0].shape[0]
        # Search structures with fewer points.
        ex = exhaustive_search(prob, lb, mb, rng=rng, log=log, time_limit=exhaustive_time)
        # Take it if found.
        if ex is not None and actual_points(ex)[0].shape[0] < mb:
            best = ex
    # Final polish with a tighter tolerance.
    solve_rule(best, tol=1e-15, maxit=30)
    # Expand into all points.
    X, W = actual_points(best)
    # Independent verification on the full basis.
    err = verify(X, W, n)
    # Structure summary.
    struct = {}
    # Count orbits per family.
    for p in best.pat_ids:
        # Pattern name.
        key = prob.patterns[p].name()
        # Increment.
        struct[key] = struct.get(key, 0) + 1
    # Number of points on the boundary.
    nb = int(np.sum(np.max(np.abs(X), axis=1) > 1 - 1e-12))
    # Result record.
    res = {
        "n": n,
        "group": group,
        "m": int(X.shape[0]),
        "equations": prob.neq,
        "lower_bound": lower_bound(n),
        "max_moment_error": err,
        "min_weight": float(np.min(W) * 8.0),
        "max_abs_coord": float(np.max(np.abs(X))),
        "boundary_points": nb,
        "orbits": struct,
        "seconds": round(time.time() - t0, 1),
    }
    # Make sure the output directory exists.
    os.makedirs(outdir, exist_ok=True)
    # File name.
    fname = os.path.join(outdir, "cube3d_n%02d_m%d_%s.txt" % (n, X.shape[0], group))
    # Write the rule: x y z w with weights for the volume 8 of [-1,1]^3.
    with open(fname, "w") as fh:
        # Header line 1.
        fh.write("# cubature rule on [-1,1]^3, degree %d, %d points, group %s\n" % (n, X.shape[0], group))
        # Header line 2.
        fh.write("# columns: x y z w ; sum(w) = 8 ; max moment error %.2e\n" % err)
        # One line per point.
        for x, w in zip(X, W * 8.0):
            # Write with full double precision.
            fh.write("% .17e % .17e % .17e % .17e\n" % (x[0], x[1], x[2], w))
    # Remember the file.
    res["file"] = fname
    # Log.
    log("RESULT m=%d err=%.1e minw=%.2e boundary=%d %s" % (res["m"], err, res["min_weight"], nb, struct))
    # Return the record.
    return res


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------
def finish_from_rule(n, rule, group, outdir, t_used):
    """Write and verify a rule found for degree n+1 as a degree-n rule."""
    # Expand.
    X, W = actual_points(rule)
    # Verification at degree n.
    err = verify(X, W, n)
    # Orbit statistics.
    struct = {}
    # Count families.
    for p in rule.pat_ids:
        # Pattern name.
        key = rule.prob.patterns[p].name()
        # Increment.
        struct[key] = struct.get(key, 0) + 1
    # Boundary points.
    nb = int(np.sum(np.max(np.abs(X), axis=1) > 1 - 1e-12))
    # File name.
    fname = os.path.join(outdir, "cube3d_n%02d_m%d_%s.txt" % (n, X.shape[0], group))
    # Write the file.
    with open(fname, "w") as fh:
        # Header line 1.
        fh.write("# cubature rule on [-1,1]^3, degree %d (exact to degree %d), %d points, group %s\n" % (n, n + 1, X.shape[0], group))
        # Header line 2.
        fh.write("# columns: x y z w ; sum(w) = 8 ; max moment error %.2e\n" % err)
        # Points.
        for x, w in zip(X, W * 8.0):
            # One line per point.
            fh.write("% .17e % .17e % .17e % .17e\n" % (x[0], x[1], x[2], w))
    # Result record.
    return {"n": n, "group": group, "m": int(X.shape[0]), "equations": Problem(n, group).neq, "lower_bound": lower_bound(n), "max_moment_error": err, "min_weight": float(np.min(W) * 8.0), "max_abs_coord": float(np.max(np.abs(X))), "boundary_points": nb, "orbits": struct, "seconds": t_used, "file": fname, "note": "same rule as degree %d" % (n + 1)}


def load_rule_for(res, group):
    """Rebuild the orbit representation of a stored result (for even-degree reuse)."""
    # Problem of the stored degree.
    prob = Problem(res["n"], group)
    # Points and weights from the file.
    data = np.loadtxt(res["file"], ndmin=2)
    # Decompose into orbits like the initial rule.
    used = np.zeros(data.shape[0], dtype=bool)
    # Lookup table.
    lookup = {tuple(np.round(x * 1e8).astype(np.int64)): i for i, x in enumerate(data[:, :3])}
    # Orbit lists.
    pids = []
    # Coordinates.
    crds = []
    # Weights.
    wts = []
    # Loop over points.
    for i in range(data.shape[0]):
        # Skip assigned points.
        if used[i]:
            continue
        # Orbit.
        orb = orbit_points(prob.perms, prob.signs, data[i, :3])
        # Mark.
        for y in orb:
            # Flag the orbit point.
            used[lookup[tuple(np.round(y * 1e8).astype(np.int64))]] = True
        # Find the family.
        for q, pat in enumerate(prob.patterns):
            # Size must match.
            if pat.size != orb.shape[0]:
                continue
            # Fit any orbit point.
            v = next((pat.fit(y, 1e-9) for y in orb if pat.fit(y, 1e-9) is not None), None)
            # Found.
            if v is not None:
                break
        # Store.
        pids.append(q)
        # Coordinates.
        crds.append(v)
        # Orbit weight (probability measure).
        wts.append(data[i, 3] / 8.0 * orb.shape[0])
    # Build the rule.
    return Rule(prob, pids, crds, wts)


def refine_degree(n, group, outdir, time_limit, seed=7, sources=None):
    """Continue the LNS for degree n from the best known rule; save improvements.

    Warm start: every stored rule of degree >= n (from any of the source
    folders) that is invariant under `group` is a valid exact starting rule
    for degree n.  In particular an O_h-invariant rule is also T_d-invariant.
    """
    # Start time.
    t0 = time.time()
    # Folders to search for starting rules.
    dirs = [outdir] + list(sources or [])
    # Point count from a file name.
    count = lambda f: int(re.search(r"_m(\d+)_", f).group(1))
    # Degree from a file name.
    degree = lambda f: int(re.search(r"_n(\d+)_", f).group(1))
    # All candidate files of degree >= n.
    files = [f for d in dirs for f in glob.glob(os.path.join(d, "cube3d_n*_m*_*.txt")) if degree(f) >= n]
    # Sort by point count.
    files.sort(key=count)
    # Best rule for this degree that is invariant under the group.
    rule = None
    # Try the candidates in order.
    for f in files:
        # Rebuild the orbit representation under the requested group (fails if not invariant).
        try:
            rule = load_rule_for({"n": n, "file": f}, group)
        except Exception:
            rule = None
        # Check the point count (a non-invariant rule decomposes incorrectly).
        if rule is not None and actual_points(rule)[0].shape[0] == count(f):
            # Polish and check exactness.
            ok, _ = solve_rule(rule)
            # Accept the first valid one.
            if ok:
                fbest = f
                break
        # Not usable.
        rule = None
    # Nothing usable.
    if rule is None:
        return n, -1, -1
    # Points of the start.
    m0 = count(fbest)
    # Random generator.
    rng = np.random.default_rng(seed + n)
    # Target structure from the rank conditions.
    tpts, target = structure_target(rule.prob)
    # Report it.
    print("[n=%d refine] start m=%d, rank-ILP estimate m=%s target %s" % (n, m0, tpts, None if target is None else list(target)), flush=True)
    # Checkpoint writer: save every improvement immediately.
    def save(r):
        # Expand.
        Xc, Wc = actual_points(r)
        # File name.
        fn = os.path.join(outdir, "cube3d_n%02d_m%d_%s.txt" % (n, Xc.shape[0], group))
        # Write.
        with open(fn, "w") as fh:
            # Header line 1.
            fh.write("# cubature rule on [-1,1]^3, degree %d, %d points, group %s\n" % (n, Xc.shape[0], group))
            # Header line 2.
            fh.write("# columns: x y z w ; sum(w) = 8 ; max moment error %.2e\n" % verify(Xc, Wc, n))
            # Points.
            for x, w in zip(Xc, Wc * 8.0):
                # One line per point.
                fh.write("% .17e % .17e % .17e % .17e\n" % (x[0], x[1], x[2], w))
    # Large-neighbourhood search.
    res = lns(rule, rng, iters=10 ** 6, time_limit=time_limit, temp0=4.0, target=target, on_improve=save, log=lambda s: print("[n=%d refine] %s (%.0fs)" % (n, s.strip(), time.time() - t0), flush=True))
    # Expand.
    X, W = actual_points(res)
    # Save if better than everything stored in the output folder for this degree.
    own = [f for f in glob.glob(os.path.join(outdir, "cube3d_n%02d_m*_%s.txt" % (n, group)))]
    # Best own count.
    mown = min([count(f) for f in own], default=10 ** 9)
    # Write when improved.
    if X.shape[0] < mown:
        # Verification.
        err = verify(X, W, n)
        # File name.
        fname = os.path.join(outdir, "cube3d_n%02d_m%d_%s.txt" % (n, X.shape[0], group))
        # Write.
        with open(fname, "w") as fh:
            # Header line 1.
            fh.write("# cubature rule on [-1,1]^3, degree %d, %d points, group %s\n" % (n, X.shape[0], group))
            # Header line 2.
            fh.write("# columns: x y z w ; sum(w) = 8 ; max moment error %.2e\n" % err)
            # Points.
            for x, w in zip(X, W * 8.0):
                # One line per point.
                fh.write("% .17e % .17e % .17e % .17e\n" % (x[0], x[1], x[2], w))
    # Report.
    return n, m0, int(X.shape[0])


def finalize(args):
    """Keep only the best rule per degree, derive even degrees, verify, summarise."""
    # Centrally symmetric group?
    central = args.group in ("Oh", "Th", "D4h", "D2h", "Ci", "Bd")
    # Point count from a file name.
    count = lambda f: int(re.search(r"_m(\d+)_", f).group(1))
    # Best file per degree.
    best = {}
    # Loop over all stored files.
    for f in glob.glob(os.path.join(args.out, "cube3d_n*_m*_%s.txt" % args.group)):
        # Degree.
        n = int(re.search(r"_n(\d+)_", f).group(1))
        # Keep the smallest.
        if n not in best or count(f) < count(best[n]):
            best[n] = f
    # For central groups a rule of degree n+1 also serves degree n.
    if central:
        # Loop over degrees.
        for n in range(args.nmin, args.nmax + 1):
            # Better rule one degree higher?
            if n + 1 in best and (n not in best or count(best[n + 1]) < count(best[n])):
                # Rebuild it.
                rule = load_rule_for({"n": n + 1, "file": best[n + 1]}, args.group)
                # Write it as a degree-n rule.
                rec = finish_from_rule(n, rule, args.group, args.out, 0.0)
                # Use it.
                best[n] = rec["file"]
    # Remove dominated files.
    for f in glob.glob(os.path.join(args.out, "cube3d_n*_m*_%s.txt" % args.group)):
        # Degree.
        n = int(re.search(r"_n(\d+)_", f).group(1))
        # Delete if not the best.
        if best.get(n) != f:
            os.remove(f)
    # Summary records.
    results = []
    # Loop over degrees.
    for n in range(args.nmin, args.nmax + 1):
        # Skip missing degrees.
        if n not in best:
            continue
        # Load.
        data = np.loadtxt(best[n], ndmin=2)
        # Points.
        X = data[:, :3]
        # Weights (probability measure).
        W = data[:, 3] / 8.0
        # Rebuild the orbit structure for the statistics.
        rule = load_rule_for({"n": n, "file": best[n]}, args.group)
        # Orbit statistics.
        struct = {}
        # Count families.
        for p in rule.pat_ids:
            # Name.
            key = rule.prob.patterns[p].name()
            # Increment.
            struct[key] = struct.get(key, 0) + 1
        # Record.
        results.append({"n": n, "m": int(X.shape[0]), "lower_bound": lower_bound(n), "equations": rule.prob.neq, "max_moment_error": verify(X, W, n), "min_weight": float(data[:, 3].min()), "max_abs_coord": float(np.abs(X).max()), "boundary_points": int(np.sum(np.max(np.abs(X), axis=1) > 1 - 1e-12)), "orbits": struct, "file": best[n]})
    # Print the table.
    print("  n      m   LB  eqs  boundary  max_err  orbits")
    # Rows.
    for r in results:
        # Format.
        print("%3d %6d %4d %4d %9d %8.1e  %s" % (r["n"], r["m"], r["lower_bound"], r["equations"], r["boundary_points"], r["max_moment_error"], r["orbits"]))
    # Save.
    with open(args.summary, "w") as fh:
        # JSON.
        json.dump(results, fh, indent=1)


def main():
    """Parse arguments and run the search for a range of degrees."""
    # Argument parser.
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Smallest degree.
    ap.add_argument("--nmin", type=int, default=1)
    # Largest degree.
    ap.add_argument("--nmax", type=int, default=30)
    # Symmetry group.
    ap.add_argument("--group", default="Oh")
    # Minimal number of phase-1 elimination runs per degree.
    ap.add_argument("--restarts", type=int, default=6)
    # Time budget for degree 1 (seconds).
    ap.add_argument("--tmin", type=float, default=60.0)
    # Time budget for degree 31 (seconds); quadratic interpolation in between.
    ap.add_argument("--tmax", type=float, default=900.0)
    # Time for the exhaustive structure search (0 = off), used for n <= 11 only.
    ap.add_argument("--exhaustive", type=float, default=0.0)
    # Number of random node additions per escape attempt.
    ap.add_argument("--escape", type=int, default=60)
    # Output directory.
    ap.add_argument("--out", default="rules")
    # Number of parallel worker processes.
    ap.add_argument("--jobs", type=int, default=1)
    # Summary file.
    ap.add_argument("--summary", default="summary.json")
    # Finalisation mode: collect the best stored rule per degree and write the summary.
    ap.add_argument("--finalize", action="store_true")
    # Refinement mode: continue from stored rules for this many seconds (degree 30; scaled quadratically).
    ap.add_argument("--refine", type=float, default=0.0)
    # Random seed (default: 7 for --refine, 1 for the fresh search, as in earlier versions).
    ap.add_argument("--seed", type=int, default=None)
    # Only every step-th degree starting at nmin (e.g. 2 = odd degrees only).
    ap.add_argument("--step", type=int, default=1)
    # Additional folders with starting rules for the refinement (comma separated).
    ap.add_argument("--sources", default="")
    # Parse.
    args = ap.parse_args()
    # Finalisation mode.
    if args.finalize:
        # Run it.
        finalize(args)
        # Done.
        return
    # Refinement mode.
    if args.refine > 0:
        # Degrees to refine.
        degs = list(range(args.nmin, args.nmax + 1))
        # Parallel execution.
        from concurrent.futures import ProcessPoolExecutor
        # Worker pool.
        with ProcessPoolExecutor(max(args.jobs, 1)) as ex:
            # Submit.
            # Source folders.
            src = [d for d in args.sources.split(",") if d]
            # Time per degree: 10% of the budget plus a quadratic part.
            tl = lambda n: args.refine * (0.1 + 0.9 * ((n - 1) / 29.0) ** 2)
            # Submit (largest first for load balance).
            futs = [ex.submit(refine_degree, n, args.group, args.out, tl(n), 7 if args.seed is None else args.seed, src) for n in sorted(degs, reverse=True) if n % args.step == args.nmin % args.step]
            # Collect and print.
            for f in futs:
                # Result tuple.
                n, before, after = f.result()
                # Report.
                print("refine n=%d: %d -> %d" % (n, before, after))
        # Done.
        return
    # Centrally symmetric groups integrate odd polynomials automatically.
    central = args.group in ("Oh", "Th", "D4h", "D2h", "Ci", "Bd")
    # Requested degrees.
    wanted = list(range(args.nmin, args.nmax + 1))
    # Degrees that are actually searched (even n reuse the rule of n+1 if central).
    searched = sorted(set((n + 1 if (central and n % 2 == 0) else n) for n in wanted))
    # Time budget per degree.
    def budget(n):
        # Quadratic interpolation between tmin and tmax.
        return args.tmin + (args.tmax - args.tmin) * ((n - 1) / 30.0) ** 2
    # Keyword arguments shared by all degrees.
    kw = dict(group=args.group, restarts=args.restarts, outdir=args.out, escape_tries=args.escape, seed=1 if args.seed is None else args.seed)
    # Run the searches.
    found = {}
    # Parallel or sequential.
    if args.jobs > 1:
        # Local import.
        from concurrent.futures import ProcessPoolExecutor
        # Worker pool.
        with ProcessPoolExecutor(args.jobs) as ex:
            # Submit (largest first for load balance).
            futs = {n: ex.submit(search_degree, n, time_limit=budget(n), exhaustive_time=args.exhaustive if n <= 11 else 0.0, **kw) for n in sorted(searched, reverse=True)}
            # Collect.
            found = {n: futs[n].result() for n in searched}
    else:
        # Sequential loop.
        found = {n: search_degree(n, time_limit=budget(n), exhaustive_time=args.exhaustive if n <= 11 else 0.0, **kw) for n in searched}
    # Assemble the results for all wanted degrees.
    results = []
    # Loop over wanted degrees.
    for n in wanted:
        # Directly searched degree.
        if n in found:
            results.append(found[n])
        else:
            # Reuse the rule of degree n+1.
            results.append(finish_from_rule(n, load_rule_for(found[n + 1], args.group), args.group, args.out, 0.0))
    # Header of the table.
    print("\n  n      m   LB  eqs  boundary  max_err   time  orbits")
    # One line per degree.
    for r in results:
        # Format.
        print("%3d %6d %4d %4d %9d %8.1e %6.0f  %s" % (r["n"], r["m"], r["lower_bound"], r["equations"], r["boundary_points"], r["max_moment_error"], r["seconds"], r["orbits"]))
    # Save the summary.
    with open(args.summary, "w") as fh:
        # JSON dump.
        json.dump(results, fh, indent=1)


# Entry point.
if __name__ == "__main__":
    main()

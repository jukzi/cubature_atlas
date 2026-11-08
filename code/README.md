# Code

The programs that produced and certified the rules, as published on the original atlas page. They need numpy and scipy.

| File | Purpose |
|---|---|
| `counts.py` | Counting value, lower bounds, Gauss product |
| `cubature_atlas.py` | Basis, search, Newton–Kantorovich certificate |
| `demo.py` | Find a rule, certify it, integrate |
| `kubatur.py` | Triangle/tetrahedron search with certificate |
| `dom.py`, `cs.py`, `fj.py` | Basis and solver (square, triangle, tetrahedron) |
| `seed_sq.py`, `seed_cs.py` | Square, start from a lower degree |
| `sqelim.py` | Square, node reduction (PI and PO) |
| `seed_tri2.py`, `free_search.py` | Triangle/tetrahedron, start from the previous degree, PO search |
| `symsearch.py` | Orbit search (triangle, tetrahedron) |
| `simplex_cubature.py` | d-simplex search with symmetry groups S_{d+1}, A_{d+1} or none: invariant basis from the orthonormal Dubiner basis, NNLS start, orbit elimination with Levenberg–Marquardt and manifold walks; produced the tetrahedron rules CA-T3-24-766 … CA-T3-30-1704 |
| `cannonball_start.py` | Tetrahedron search started from a cannonball packing of about Z(n) = (n+1)(n+2)(n+3)/24 equal balls (the counting value), degree continuation with bounded least squares (scipy trf), optional node elimination; produced CA-T3-10-74 |
| `cannon_cube.py` | Cube: start from a T_d- or O_h-invariant cannonball packing (FCC lattice) with equal weights, solve the symmetry-reduced system, then orbit elimination and LNS of `cubature_search.py` (`sym`, `cont` modes); free mode with exactly Z(n) nodes without symmetry; produced CA-C3-24-730 and CA-C3-22-551 |
| `cert_large.py` | Certificate for large simplex rules (tetrahedron degree 28 and up): same test as `certificate()`, with cached basis recursions and power iteration for the norms |
| `po_elim.py` | Cube/tetrahedron PO search: node elimination from the PI rule with nodes allowed outside the domain (CS=1: centrally symmetric pairs on the cube); produced CA-C3-06-21-PO … CA-C3-29-1181-PO and CA-T3-12-114-PO … CA-T3-24-757-PO |
| `cubature_search.py`, `verify_rules.py`, `cube_cert.py`, `run_cube_cert.py` | Cube: symmetric search and certificate |

`cubature_search.py` (version 3) produced the cube rules CA-C3-21-486 … CA-C3-30-1324. It searches in a chosen subgroup of the cube group (Oh, Td, Th, O, T, …): orbit elimination, large-neighbourhood search with simulated annealing, a rank-ILP target structure, and warm starts from stored rules (`--refine T --sources dir`). Run `python3 cubature_search.py --help` for the options; the module docstring describes the method and lists the literature.

The rule files in `rules/` are the authoritative data. The build (`tools/build_atlas.py`) checks them independently of this code.

# Development

Everything needed to add rules, run the checks and the proofs, and work on the website. The overview for readers is in [README.md](README.md); the workflow for pull requests is in [CONTRIBUTING.md](CONTRIBUTING.md); the search programs are described in [code/README.md](code/README.md).

## What is in here

| Path | Content |
|---|---|
| `rules/<family>/family.json` | Name and description of a rule family, in English and German |
| `rules/<family>/<domain>/<id>.json` | One rule per file: metadata, nodes, weights |
| `literature.md` | Literature on the topic: used in the atlas, tried but not worthwhile, not used yet (with links and abstracts) |
| `reference_data/literature_counts.json` | Node counts from the literature without stored points (Xiao–Gimbutas, symmetric rules, Pinkse) |
| `i18n/en.json`, `i18n/de.json` | Texts of the website (English is the source) |
| `content/en/`, `content/de/` | Documentation section of the website |
| `content/references.json` | Bibliography, sorted by year (undated last); the position is the citation number `[n]`; after adding an entry run `tools/sort_references.py` |
| `tools/sort_references.py` | Sorts the bibliography by year and renumbers every use of the numbers |
| `tools/state_table.py` | Prints per domain and degree the counting value Z, the smallest stored PI and PO node counts and the gaps N - Z (options `--domain`, `--min-degree`, `--csv`) |
| `site_template/index.html` | The interactive page; the build inserts the data |
| `code/` | Search programs that produced the rules, with numerical checks (not proofs) |
| `tools/build_atlas.py` | Verifier and site generator (Python 3.9+, numpy) |
| `tools/verify_rigorous.py`, `.github/workflows/rigorous.yml` | Rigorous check with interval arithmetic: stored numbers and existence of an exact rule (Krawczyk test) |
| `rules/<family>/<domain>/<id>.proof` | Result of the rigorous check of one rule, next to the rule file (same name, extension `.proof`) |
| `tools/make_readme_gif.py`, `tools/make_readme_figures.py`, `assets/` | Figures of the README: figure A of the built page as an animated GIF (Playwright, Pillow), and the static charts computed from the rule files (matplotlib) |
| `tools/auto_search.py`, `.github/workflows/auto_search.yml` | Scheduled automatic search for smaller cube rules; opens a pull request when it finds one |
| `tools/auto_cannonball.py`, `.github/workflows/auto_cannonball.yml` | Automatic search for smaller tetrahedron rules from a cannonball start, started by hand; opens a pull request when it finds one |

## Rule families

Each family is one directory under `rules/`. The family name, including its translation, is stored in the directory's `family.json`.

| Directory | Role | Content |
|---|---|---|
| `minimal_pointcount_for_degree_n_PI` | `pi` | Smallest rules of the atlas's own search: positive weights, nodes in the closed domain |
| `minimal_pointcount_for_degree_n_PO` | `po` | Positive weights, at least one node genuinely outside the domain |
| `witherden_vincent_2015` | `literature` | Fully symmetric triangle rules (reproduced for comparison) |
| `jaskowiec_sukumar_2020` | `literature` | Fully symmetric tetrahedron rules (reproduced for comparison) |
| `diallo_worku_2026` | `literature` | Fully symmetric square and cube rules (reproduced for comparison) |
| `pinkse_2026` | `literature` | Cube rules of the degrees 11, 13, 15, 17, 21, 23, 25, 27 and 29 to 45 (odd) and square rules of the odd degrees 31 to 51 (reproduced for comparison) |
| `festa_sommariva_2012` | `literature` | Square rules of the degrees 16, 20, 22, 27, 29 and 31 to 50 (reproduced for comparison; degrees 16 to 29 made by `code/import_burkardt_rules.py`) |
| `worku_hicken_zingg_2024` | `literature` | Fully symmetric triangle rules of the degrees 31 to 50 and tetrahedron rules of the degrees 21 to 40 (reproduced for comparison) |
| `wandzura_xiao_2003` | `literature` | The six fully symmetric triangle rules printed in the paper (degrees 5, 10, 15, 20, 25, 30), made by `code/import_xiao_wandzura.py` |
| `xiao_gimbutas_2010` | `literature` | The four rules with printed nodes (triangle degrees 10, 30, 50 and the cube rule of degree 5, made by `code/import_xiao_wandzura.py`) and the rules of the authors' data tables (fully symmetric triangle degrees 32 to 49, rotationally symmetric triangle degrees 23, 28, 29, square degrees 16, 20, 22, tetrahedron degrees 11 to 15, cube degrees 9, 11, 12, 13, 15), made by `code/import_burkardt_rules.py` |

Quality codes, as used by Cools' encyclopaedia: **PI** = positive weights, all nodes inside (the boundary counts as inside); **PO** = positive weights, some node outside; **NI** / **NO** = some negative weight, nodes inside / outside. Families for rules with negative weights would be named `minimal_pointcount_for_degree_n_NI` and `minimal_pointcount_for_degree_n_NO`.

The build computes the quality from the numbers and rejects a file whose stored quality is wrong. Example: the PO search had returned a degree-30 square rule with 166 nodes whose nodes all lie inside. That makes it a PI rule, so it now lives in the PI family, one node below the 167-node rule. Both are proven with interval arithmetic.

### Adding a family

1. Create `rules/<family_id>/family.json` (letters, digits and `_` only):

   ```json
   {
     "id": "my_new_family",
     "tag": "MNF",
     "role": "own",
     "quality": "PI",
     "name":        {"en": "My new family", "de": "Meine neue Familie"},
     "short":       {"en": "My family",     "de": "Meine Familie"},
     "description": {"en": "What the rules are and where they come from.", "de": "Was die Regeln sind und woher sie kommen."},
     "references": []
   }
   ```

2. Put the rules into `rules/<family_id>/<domain>/` (domains: `interval`, `triangle`, `square`, `tetra`, `cube`).
3. Run `python tools/build_atlas.py`. A family with role `own` automatically shows up on the website as a new selectable type: a button in the viewer, a column in the table, a series in both charts, and an entry in the documentation. A family with role `literature` joins the "Literature" column instead.

Roles: `pi` and `po` exist once each (the atlas's own best rules), `literature` for published rules, `own` for everything else. `quality` is optional; if it is set, every rule in the family must have it.

## Rule files

File name = id + `.json`. The id has the form

```
CA-<domain>-<degree, 2 digits>-<points>-<quality>-<family tag>[-a]
CA-T2-13-35-PI-MPC
```

with the domain symbols `C1` interval [−1,1], `T2` triangle (0,0),(1,0),(0,1), `C2` square [−1,1]², `T3` tetrahedron (0,0,0),(1,0,0),(0,1,0),(0,0,1), `C3` cube [−1,1]³. Published ids are never reused or renamed; a better rule gets a new id.

```json
{
  "id": "CA-T2-03-4-PI-MPC",
  "family": "minimal_pointcount_for_degree_n_PI",
  "domain": "triangle",
  "dimension": 2,
  "degree": 3,
  "points": 4,
  "quality": "PI",
  "origin": "found",
  "method": "...",
  "author": "...",
  "reproduce": {"script": "code/<program>.py", "seed": 12345, "command": "python code/<program>.py ..."},
  "status": "proven",
  "numerical_check": {"N": 4, "M": 10, "p": 12, "res": 1e-16, "sigma": 0.3, "h": 1e-14, "rad": 1e-16, "cert": true, "inside": true},
  "nodes": [[x, y], ...],
  "weights": [w, ...]
}
```

`reproduce` is optional and not checked by the build; give it to every new rule with `origin` `"found"`: `script` is a file of the repository, `seed` an integer or a list of integers, `command` the call that gives the same rule again.

"Best known" is not stored. The build and the page compute it: for each domain and degree, the rule with the fewest nodes among all rules of at least that degree.

## Verification and build

```bash
pip install numpy
python tools/build_atlas.py --check               # verify all rules
python tools/build_atlas.py                       # verify and build site/
python tools/build_atlas.py --base origin/main --changed   # before a commit: check only the rules this change adds or modifies
```

Every rule must have an interval proof in the file `<id>.proof` next to it, for exactly its numbers (see below); the build fails otherwise. Every rule is also checked for: exact integration of all monomials up to its degree (maximum error per volume ≤ 1e-11), the quality code computed from the data (weights and position of the nodes), and agreement of the id with domain, degree, point count and family tag. Ids must be unique.

### Rigorous check with interval arithmetic

`tools/build_atlas.py` works in floating point, and so does the numerical Newton–Kantorovich check stored in the rule files as `numerical_check`. Neither is a proof. The proof is `tools/verify_rigorous.py` (Python, numpy), in which every rounding error is accounted for:

1. **Stored numbers.** The decimal numbers of the file are taken exactly. Node positions and weight signs are checked in exact rational arithmetic, and the moment residual of the stored rule is enclosed rigorously for every basis polynomial up to the degree (it must be ≤ 1e-11 per volume).
2. **Existence of an exact rule.** A Krawczyk test (interval Newton method) proves that the moment equations have an exact solution within a small distance of the stored numbers, and that every rule in that neighbourhood has the stored quality (positive weights, nodes inside). Rules with symmetries are proven in the symmetry-reduced equations (signed permutations on square and cube, permutations of the barycentric coordinates on triangle and tetrahedron); nodes on the boundary are kept exactly on it. On interval, square and cube the residual at the centre of the Krawczyk box is summed exactly in integer arithmetic (the stored floats are dyadic rationals), so rounding errors of the basis values do not widen the box, and the same exact residual is used to move the centre to the nearest floats of the exact rule before the test (needed for ill-conditioned rules); for very large rules (more than `VR_EXACT_PAIRS` pairs of basis function and node, default 4 million) and on the simplices it is enclosed with interval arithmetic.

Basis: Legendre products on boxes, the orthogonal Dubiner (PKD) basis on triangle and tetrahedron. Basis values at the centres of the interval boxes are computed exactly with fractions; matrix products are enclosed with rigorous error bounds that hold for any BLAS.

```bash
python tools/verify_rigorous.py                                   # all rules
python tools/verify_rigorous.py --changed origin/main             # only added or changed rule files
python tools/verify_rigorous.py --strict --save rules/.../<id>.json   # prove new rules and write rules/.../<id>.proof
python tools/verify_rigorous.py rules/minimal_pointcount_for_degree_n_PI/triangle/CA-T2-20-78-PI-MPC.json
```

The result of each rule is in its `.proof` file next to the rule file (so pull requests that prove different rules never touch the same file; per rule: residual bound, exact quality of the stored numbers, proof mode, distance within which the exact rule lies, proven quality). The build requires a proof file for every rule, and refuses a `.proof` file without a rule file (delete the proof together with the rule) and shows it on the website (table "Proofs", column "interval proof") and on every rule page. The proof file stores a hash of what the proof depends on (domain, degree, quality and the exact decimal strings of nodes and weights), so a proof stays valid when only metadata change. `.github/workflows/rigorous.yml` runs the check for the changed rule files of every pull request (for all rules if the proof program itself changed) and, when started by hand, for all rules (if the result differs from `reports/interval_proofs.json`, that run opens a pull request with the new report from the branch `auto/interval-report`); a pull request fails if a changed rule does not pass. There are no exceptions: every rule needs a proof.

For every rule the stored numbers pass, and an exact rule with the stored quality is proven to exist within about 10<sup>-10</sup> of the stored numbers (the per-rule distance is in the report). Some tetrahedron PI rules store a node about 10<sup>-15</sup> outside the face x + y + z = 1 (accepted by the build's tolerance of 10<sup>-12</sup>); for them the proof places that node exactly on the face, and the proven exact rule lies inside. The basis enclosure is computed in blocks of points (`VR_POINT_BLOCK`, default 128) and the matrix products of the Krawczyk test in blocks of rows (`VR_ROW_BLOCK`, default 512); the two QR factorisations of the heuristic parts keep their Householder reflectors instead of forming Q. The proof of the largest rule (cube, degree 45, 4308 nodes, a system of about 8400 equations) was measured at 2.5 GB of memory. Systems with more than `VR_MAX_SYSTEM` entries (equations times unknowns, default 1e8) are not attempted. The linear algebra runs on several cores through the BLAS library of numpy (`OPENBLAS_NUM_THREADS` limits it); the proof does not depend on the number of threads.

The build writes `site/`: the English page, the German page under `de/`, one static page per rule (both languages, for search engines), `data/atlas.json`, a ZIP of all rules, and `sitemap.xml`. GitHub Actions run the check on every pull request and publish a live preview of the pull request under `pr-preview/pr-<number>/`. On every push to `main` they deploy `site/` to the `gh-pages` branch, which GitHub Pages serves (`.github/workflows/atlas.yml`).

## Automatic search

Once a day, `.github/workflows/auto_search.yml` runs three parallel slots of about 150 minutes each (`workflow_dispatch` allows other values).

Each slot (`tools/auto_search.py`):
- picks a cube degree at random, weighted towards high degrees;
- picks the symmetry group Td or Oh;
- picks the mode: a warm-start large-neighbourhood search from the best atlas rules, or a fresh search;
- runs `code/cubature_search.py`.

If a rule has fewer nodes than every rule of the atlas and of the open automatic pull requests of at least that degree, the slot:
- checks it numerically with `code/cube_cert.py`;
- writes the rule file, proves it with `tools/verify_rigorous.py --strict --save` and runs `tools/build_atlas.py --base origin/main`;
- opens a pull request from `ai/auto-cube-nXX-mYYY` if the check prints `OK`.

Settings:
- **Settings → Actions → General → "Allow GitHub Actions to create and approve pull requests"** must be enabled.
- Pull requests opened with the default `GITHUB_TOKEN` do not start the pull-request checks and the preview. With a repository secret `AUTO_SEARCH_TOKEN` (a fine-grained token with *Contents* and *Pull requests* write access to this repository) they do.

The tetrahedron has its own workflow, `.github/workflows/auto_cannonball.yml`, started by hand (`workflow_dispatch`: degree, extra points, number of seeds, minutes, maxfev). Each job (`tools/auto_cannonball.py`):
- runs `code/cannonball_start.py` for one seed: a cannonball packing of ceil(Z(n)) + extra balls, Z(n) = (n+1)(n+2)(n+3)/24, as start, degree continuation with bounded least squares, then node elimination until the budget is used up (a snapshot after every elimination round);
- keeps all results as a workflow artifact;
- takes the smallest exact PI result and, if it has fewer nodes than every tetrahedron rule of the atlas and of the open pull requests `ai/auto-tet-*` of at least that degree, checks it numerically with `code/cubature_atlas.py`, writes the rule file, proves it with `tools/verify_rigorous.py`, runs `tools/build_atlas.py --base origin/main` and opens a pull request from `ai/auto-tet-nXX-mYYY` if the check prints `OK`.

The same settings as above apply.

## Translations

English is the source. `i18n/de.json` stores, for each key, a hash of the English text it was translated from (`_source`), and `content/de/documentation.html` does the same in its first line. If an English text changes, the build warns that the German one is stale (`--strict` turns this into an error), and the German page shows a notice. After updating the German text, run `python tools/build_atlas.py --sync-hashes`.

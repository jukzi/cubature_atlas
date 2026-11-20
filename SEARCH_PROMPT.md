# Search prompt for AI agents

A prompt for an AI agent (or a person) that searches for better rules in the Cubature Atlas. It collects what worked and what did not. Numbers in it (rule counts, node counts, rule ids, pull request numbers, residuals) are examples from earlier sessions: re-check every one against the current atlas before relying on it, and name its source when you quote it.

## 0. Quick start

1. Read README.md, CONTRIBUTING.md, DEVELOPMENT.md, CLAUDE.md; check `nproc`, `free -g`, whether the shell has network access, and `uptime`.
2. Fetch `origin/main`, run `python tools/state_table.py` and pick a cell (see "Where to search").
3. Calibrate: reproduce the atlas count of an easy cell from random starts with your own program.
4. Search in short runs with many different parameters (see "Persistence and reporting"); save every solution with seed and command.
5. Minimise the weights (`tools/minimize_l2.py`), prove the rule (`tools/verify_rigorous.py --strict --save`), run `tools/build_atlas.py --base origin/main --changed`, open one pull request per rule.
6. Write what worked and what did not into this file, with numbers and their source.

## 1. Core

**Task:** Cubature Atlas – find rules with fewer nodes (or better rules), prove them, submit pull requests.

### Project

- Repository: https://github.com/jukzi/cubature_atlas. Read README.md, CONTRIBUTING.md, DEVELOPMENT.md and CLAUDE.md first and follow them. In short: branch from the latest `main`, one pull request per improvement, rebase merges, `python tools/build_atlas.py --base origin/main --changed` must print OK before a pull request, never weaken a check, never push to `main`.
- Force-push: never on other people's branches. On your own `ai/` branch only with `--force-with-lease`, and only if the environment allows it. If it is refused, do not ask and do not wait: push the rebased state to a new branch, open a new pull request and close the old one with a pointer.
- Goal: for a domain and degree n, find rules with fewer nodes N than the atlas (columns N_PI / N_PO). Also welcome: the same N with better quality (see "Quality").
- Counting value Z = ceil(M / (dim + 1)): M is the number of moment equations (total degree at most n), every node brings dim + 1 unknowns (coordinates and weight). Triangle and square: M = (n+1)(n+2)/2, Z = ceil(M/3). Tetrahedron and cube: M = (n+1)(n+2)(n+3)/6, Z = ceil(M/4). Z is the natural target.
- PI = all weights positive and all nodes in the closed domain. PO = weights positive, nodes may be outside; a PO rule only shows that a rule with that many nodes exists in principle.
- The atlas changes all the time (other people merge pull requests). Before every campaign, and again before every long run, fetch and read it from `origin/main`. Generate the state table with `python tools/state_table.py` (options `--domain`, `--min-degree`, `--csv`; it reads `rules/*/*/*.json`) instead of copying numbers from this text. From it determine:
  - where the gap N − Z is largest,
  - where a PO rule has fewer nodes than the best PI rule,
  - which cells are unsolved,
  - which cells have only a literature count without points (a count is not a rule; see `reference_data/literature_counts.json`).

### Quality of a rule (what "better" means, in this order)

1. fewer nodes
2. symmetric (C3/S3, mirror, central) instead of unsymmetric
3. larger weights and larger distance of the nodes to the boundary

Overwrite or delete an existing rule file only if the new one is better in this sense; otherwise add a new file with a new id. Never modify a rule with `"locked": true`. Rules taken from the literature are never changed.

Before checking in a rule of this atlas, run `python tools/minimize_l2.py <rule.json> --write` and redo the interval proof. The tool minimises the sum of w^2 under all moment equations with the symmetry group of the rule kept exactly, with a smooth penalty (zero slope at the boundary, exponential outward) for nodes outside the domain only. It needs freedom (unsymmetric rule, or N larger than the number of independent equations); for an isolated solution (N = Z, symmetric) it changes nothing, which is expected. It does not write a PI rule whose smallest distance to the boundary would shrink unless you pass `--allow-closer`.

### Where to search (decide from the atlas, not from habit)

- Largest gap N − Z, a cell where a PO rule beats the PI rule, or an open cell.
- Shell starts (appendix of your domain) found rules for triangle degrees 7 to 49. For triangle degrees 51 to 70 twenty cloud workers without the S3 stage found nothing in four hours (best residuals about 2e-5 to 4e-4). There, literature rules are the better start.
- Stop spending budget where the residual plateaus (1e-2 to 1e-3 over many starts and patterns): this suggests that no such rule exists. Say so in the report instead of running on.

### Method

- Write your own program (one file in `code/`, pure Python with numpy/scipy, English comments, one statement per line). Justify every idea in a comment and measure it on a KNOWN case (time to solution before and after).
- Calibrate first: reproduce the atlas node counts of easy and medium cells from random starts before you trust a negative result on a target cell.
- Every search program writes each solution to a file (`--out`) together with seed and exact command (a `.json` sidecar with seed, start file and command per result), not only counts.
- Every rule found by a search gets a `reproduce` field (`script`, `seed`, `command`). Draw the starting points from a generator seeded with the stored seed. For staged searches the field lists the whole chain of stages (group:seed in order, each stage starting from the smallest result of the one before); trace the chain through the sidecars. Do not replay old runs just to fill the field; say honestly what was and was not replayed. Long LM runs round differently on different machines: about half of the replays of cloud solutions did not reproduce.
- The `method` field names a shell pattern twice: absolute (points per edge) and relative ("25 -2 -3 -2 ...").
- Every rule needs an interval proof: `python tools/verify_rigorous.py --strict --save <rule.json>`; commit the `.proof` file next to the rule file. Never call a floating-point check a certificate or a proof.

### What worked in every domain (reuse, do not rediscover)

- Orthonormal basis (Dubiner/PKD on simplices, Legendre products on boxes) with analytic or complex-step Jacobian; monomials are too ill-conditioned. Evaluate the recurrences once per index, not once per basis function (30x faster for the tetrahedron, bit-identical).
- Own Levenberg–Marquardt instead of `scipy.optimize.least_squares(trf)`: Cholesky on the free variables, one basis evaluation per iteration, float64 on the CPU. This was the largest single speed-up (10x time to solution on the triangle). A start damping of 1e-4 gave wrong minima in one implementation, 0.1 worked; test your own.
- Stagnation stop: stop a run when, over the last `win` steps, neither the residual fell by `gain` (relative) nor an unknown changed by `dq`. Hopeless starts then end after a few steps, and the saved time goes into more starts.
- PI by construction, no penalties. Two variants worked: barycentric squares b = s^2 / sum s^2 (weights v^2 or softplus), or box bounds on a collapsed (Duffy) parametrisation with w >= 0. Nodes on the boundary can be pulled inside by re-solving with b = delta + (1 − 3 delta) s^2 / sum s^2.
- Which map for weights and nodes (measured in one session, see the table below; re-measure with your own solver before relying on it). Weights: softplus w = ln(1 + e^s), w = e^s and w = s^2 all work; w = |s| and unconstrained w are much worse (the solver finds solutions, but almost all have negative weights, so positivity is the hard part). Likely reasons, not tested: |s| has a kink at 0, s^2 has zero slope at w = 0, so a weight near 0 gets stuck. Nodes: barycentric squares and a sigmoid on the Duffy coordinates were at least as good as box bounds; the sigmoid map was the fastest in wall time (no bounds, 2N instead of 3N node unknowns). Neither of them reaches the boundary exactly, so polish rules with boundary nodes afterwards.

  Setup of the measurement: triangle, unsymmetric (all nodes and weights free), N = 12, 16, 19 for degrees 6, 8, 9, 40 random starts per cell with the same seeds for every variant, `scipy.optimize.least_squares(trf)` with a finite-difference Jacobian, Dubiner basis; success means residual below 1e-12 and all weights above 1e-8. Cells show successes out of 40.

  | map | degree 6 | degree 8 | degree 9 |
  |---|---|---|---|
  | weights s^2 (box nodes) | 22 | 5 | 26 |
  | weights e^s (box nodes) | 37 | 4 | 22 |
  | weights softplus (box nodes) | 33 | 4 | 29 |
  | weights \|s\| (box nodes) | 3 | 0 | 2 |
  | weights free, sign not forced (box nodes) | 3 | 0 | 2 |
  | nodes box bounds (softplus weights) | 33 | 4 | 29 |
  | nodes barycentric squares (softplus weights) | 40 | 12 | 24 |
  | nodes sigmoid on Duffy (softplus weights) | 40 | 5 | 31 |

  With 40 starts only large gaps count (for example 3 against 33 or 40, or 12 against 4 at degree 8); differences of a few runs are noise. The scripts of this measurement are not in the repository.
- Shell start + symmetric search (details per domain below), noise starts in short runs, staged pattern search (first coarse patterns, then relax).
- PO search without any lower bound on the weights: nodes may go far outside with tiny weights (triangle degree 8: three nodes about 3.6 outside with weights 1.6e-10) and the interval proof decides. A bound such as w >= 1e-5 hid these solutions. Still demand w > 0 and guard against dummy nodes (weight about 1e-12) when N = Z + 1.
- Asymmetric search (all nodes and weights free) started from the symmetric pattern with seeded noise; polish a symmetric result on the reduced system. The minimum of sum w^2 often turns out symmetric by itself.
- Cube, rule with a symmetry group: warm-start `code/cubature_search.py --refine` in a SUBGROUP of the group of the start rule. A Td rule is also a T rule (rotations only, 12 elements); the orbits are half as large, so single half-orbits can be removed and the system is still about 12 times smaller than without symmetry. Degree 28: 1133 -> 1132 nodes in 90 s (seed 71, `--group T --refine 3300`); the same run with Td found nothing in 60 minutes.
- Cube, unsymmetric rule: `BOXOUT=0 python code/po_elim.py cube <n> <seconds> <seed>` starts from the best atlas PI rule and keeps all nodes in [-1,1]^3 (PI). Degree 16: 244 -> 243 nodes after 2012 s (seed 41); seeds 42 and 43 found nothing in 33 minutes. It stops by itself when 4 (N - 1) is below the number of moment equations.
- `tools/minimize_l2.py` skips a rule with more than `--max-vars` unknowns (default is small); for about 4500 unknowns (cube degree 28) pass `--max-vars 6000` (28 minutes, two cores). The symmetric interval proof of the same rule took 6 seconds.
- Literature tables: if a paper prints nodes, import them with a script (see `code/import_xiao_wandzura.py`): read the printed generator nodes, expand the orbits with the symmetry group, map the domain affinely, scale the weights, check the moment equations. The check also finds misprints (a wrong printed weight; weights printed per generator instead of per node). A published count without points is not a rule: the viewer then shows only the number and no nodes.
- Triangle degrees 41 to 50: a target of Z + 2 or Z + 3 nodes (C3 symmetric, N = 0 or 1 modulo 3) with `shell_c3_bary.py --target N --turn 2,all --offsets 0,-1,1 --pull-inside`, split over 4 processes with `--part`/`--parts`, found rules within about 30 to 90 minutes each; kill the other parts after the first solution. `--pull-inside` is needed for the interval proof (nodes on the boundary otherwise).

### Tried without gain

Repeat only with a new twist; add every new negative result with domain, degree, N, method, best residual.

- Random-start LM without a shell pattern where shells worked; node elimination from N + 1 rules; long basin hopping; GPU float32 searches; S3 stage for triangle degrees below 20; PO to PI conversion by shrinking outside nodes towards the centroid.
- Triangle, N = Z, PI: degrees 8, 10, 12 (best residual symmetric 1.4e-1, 8.0e-2, 1.5e-1; unsymmetric 9.5e-2, 6.9e-2, not finished for 12), even degrees 10 to 18 with C3 symmetry up to `--relax 4`, degree 18 with noise starts (150 per pattern, residual about 1e-3).
- Triangle degrees 51 to 70, C3 symmetric PI, no S3 stage, 20 cloud sessions of 4 hours: no rule.
- Cube, `--refine` silently skips start rules that are not invariant under the chosen group and then starts from a larger symmetric rule. The atlas rule of degree 16 (244 nodes) had no symmetry, so `--group Td` and `--group Oh` started from the 265 and 282 node rules and ended at 252 and 282 nodes; `fresh` (tensor Gauss start, Td, 60 minutes) ended at 248. Always check the `start m=` line of the log against the atlas.
- Cube degree 6, PI with 21 nodes, `BOXOUT=0 po_elim.py` from the 22-node rule: more than 2600 rounds without success in 43 minutes.
- Cube degree 28, `po_elim.py` without symmetry from the 1133-node rule (4495 equations, 4532 unknowns, one core per run): no elimination round finished in 8 to 55 minutes. Use a subgroup instead (see "What worked").
- Cube degree 28, `--group Td --refine` from the 1133-node rule, 60 minutes: no improvement. Degree 16, `--group Td` and `--group Oh` refine: see above.
- Square degree 28, centrally symmetric PO with 146 nodes: `square_elim.py --cs --po 3` stops at 150 nodes within seconds (the elimination gives up after 12 failed candidates).
- Triangle, PI, C3 symmetric shell search (`code/shell_c3_bary.py --pull-inside`, all patterns, `--turn 2,all`, 4 cores, about 30 to 100 s per pattern): no solution for degree 41 with N = 301 and 302 (about 170 runs, best residual 3.2e-5), degree 42 with N = 316 and 317 (about 260 runs, best 1.2e-4), degree 44 with N = 345 (best 9.3e-5, also `--staged --relax 3`, 55 min) and N = 346 (best residual not recorded), degree 46 with N = 376 (best 1.8e-4, also `--staged --relax 3`, 55 min), degree 48 with N = 411 and 412 (about 100 runs, best 4.5e-5), degree 50 with N = 442 (best residual not recorded). Pattern: the residual stalls at 1e-4 to 1e-5; the rules found so far all have N = Z + 2 or Z + 3 (degree 41: 303, degree 42: 318, degree 44: 348, degree 46: 378, degree 50: 444).
- Weight maps w = |s| and unconstrained w (positivity not forced) for PI search on the triangle, unsymmetric, degrees 6, 8, 9 with N = 12, 16, 19, 40 random starts each with `scipy` trf: only 3, 0 and 2 runs ended with residual below 1e-12 and positive weights (unconstrained w: 36 of 40 runs converged at degree 6, but only 3 with positive weights). Best residuals of the failed runs were not recorded. Use softplus, e^s or s^2 instead.

### Open ideas

- Own Levenberg–Marquardt with the best maps from the table above (softplus weights, barycentric nodes) on the triangle cells with the largest gap N − Z: measure whether the success rates carry over from the unsymmetric `scipy` runs; a two-phase run (forced-positive maps first, free weights or the plain moment system to polish) is untested.
- Noise-start search for the unsolved cells (check the atlas first); asymmetric search started from the best symmetric near-solution; variable projection (solve the weights linearly each step, fewer unknowns); Z − 1 attempts with symmetric shells; adaptive noise amplitude per shell; the same pipeline for the square (C2/C4) and the tetrahedron (shell-like starts); a c1 phase after the C3 phase has stalled; an iterative `--tries`/restart loop for the square elimination.

### Persistence and reporting

- The stopping criterion is a budget, not success (a target may not exist): at least X hours of total compute (X is given for the session; if it is not given, X = 4) and at least 15 different ideas tried; each search run at most 1 hour (interval proofs are exempt, they can take hours). Prefer many short runs: a configuration that has found no solution after about 30 minutes is replaced by other parameters (pattern, start, noise, map, N) instead of being run longer, unless the stagnation criterion shows it still improving. "No more ideas" does not count: derive new ones from near-solutions (which nodes move, where the residual sits, which pattern was closest).
- Do not end your turn with a question or a pure status report, and do not end it while searches or proofs are running (see the environment notes: they may die with the turn). Report in short notes about every hour and keep working after each note. Each note names: what is searched and with which algorithm, the best state per degree, open proofs, open pull requests, chance of success. All numbers come from files (`git log`, the atlas, the logs), not from memory.
- Wait inside the turn (poll loops). If the session cannot continue by itself, first test whether a scheduled wake-up (`/loop`, a scheduled task, a "send later" message) is actually delivered in this session; scheduled tasks may start a fresh session without your context. Schedule a follow-up before you stop only if it works.
- Commit and push intermediate states to your branch (name it `ai/<topic>`). Pull requests: first the program (own pull request, with a line in `code/README.md`), then the rule (its `reproduce` command needs the program; do not open the rule pull request before the program one exists), one pull request per proven rule. The description says what was computed, how it was proven and what is still open. If a better proven rule exists for the same cell, close the older pull request with a pointer to the newer one.
- Keep this file current: when you find a new idea, a useful insight or a tool that helps, add it to `SEARCH_PROMPT.md` in the same pull request or in a pull request of its own. Document what does not work just as carefully (domain, degree, N, method, best residual, time spent) under "Tried without gain", so nobody repeats it. Move an entry between "Open ideas", "What worked" and "Tried without gain" when its status changes, and correct numbers that are out of date.
- Never invent a citation, never state a count you have not checked. If you use a publication, extend `literature.md` as CLAUDE.md describes.
- Do not start GitHub Actions search runs (`workflow_dispatch` of `auto_search.yml`, `auto_cannonball.yml`) on your own; only the maintainer does.

### Environment notes

Observed in earlier sessions; verify first, they differ per session.

- Check cores and memory first (`nproc`, `free -g`). Earlier sessions had 2 or 4 cores, 8 or 16 GB, a shallow clone, and network access only to GitHub and the package registries; the shell network may be off (then no pip installs; work with what is installed and say which package you did not use).
- **Background jobs may die.** In one session, processes survived only while a foreground command was running, and the machine was restarted between rounds (`uptime` showed a few minutes). Therefore: start campaigns with `nohup nice -n 15`, one process per core, `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`; keep a foreground poll running (`sleep 595` worked there; test whether a long `sleep` is blocked in your session, then poll in loops of at most the allowed time); after every restart and before every report check `uptime` and `pgrep -f <program>` and restart the loops. Proofs that die with a restart start from zero.
- A foreground command was limited to about 10 minutes (tool timeout 600000 ms).
- Pull requests: in earlier sessions `gh pr create` (GraphQL) was blocked; the REST API worked: `gh api repos/<owner>/<repo>/pulls -f title=... -f head=<branch> -f base=main -F body=@file`; patch a description with `gh api -X PATCH repos/.../pulls/<n>`. Try the normal way first.
- A shallow clone may track only one branch; if a hook complains about "no remote branch", add the fetch refspec for your branch and fetch it.
- Headless Chrome for website checks needs much CPU and memory (pages for high degrees do not finish, and a browser started next to a proof got the proof killed by the out-of-memory killer); check pages at low degrees, and never while a proof is running.
- Cloud sessions do not report when they end: push results to a branch (`ai/<name>-results-<degree>`) so progress is visible from the commit times. Keep cloud sessions lean: one start block, one waiting monitor, a report of at most 10 lines, no edits to `rules/`, `tools/` or `code/` unless that is the task.
- When waiting for a log line, grep for `^solution found:`; plain "solution found" also matches "no solution found".

## 2. Appendix: triangle

- Rule files live in `rules/<family_id>/triangle/`, named `CA-T2-<degree>-<nodes>-<type>`; PI means positive weights with all nodes inside, PO means positive weights with nodes allowed outside. Reference cells are in the atlas table and in `reference_data/literature_counts.json`.
- Literature rules imported from Xiao and Gimbutas (2010) and Wandzura and Xiao (2003) by `code/import_xiao_wandzura.py` (orbit expansion with the D3 group, affine map to the triangle (0,0), (1,0), (0,1), weights scaled to sum 1/2, moment check with `po_elim.monomial_error`).
- Where a cell has only a literature node count but no stored points, the viewer shows the count and no points. Such cells are the first candidates: reproduce them from the published formulas or find an equal or smaller rule.
- Method: orbit elimination under D3 symmetry, then `python tools/minimize_l2.py <rule.json> --write`, then `python tools/verify_rigorous.py --strict --save <rule files>`. Overwrite an existing rule only if the new one is better (see CLAUDE.md).

## 3. Appendix: tetrahedron

- Most promising region: degrees 31 and higher, where stored counts are furthest from the lower bound. Rule files: `rules/minimal_pointcount_for_degree_n_PI/tetra/CA-T3-<degree>-<nodes>-PI-MPC.json`.
- Search program: `code/sym_elim.py` with group c3 (C3-symmetric orbit elimination). Run it in stages of 2400 s; every stage writes a sidecar `.json` with seed, start and command, so the `reproduce` field can chain the stages. Settings used before: `EXTTOL=1e-5`, `EXTIT=150`, two chains with different seed bases (8700 and 9200).
- Interval proofs for N between 2100 and 2600 took 1.5 to 3 hours and need much memory; do not run a browser next to a proof.
- Never replay old runs only to fill `reproduce`; draw start points from a generator seeded with the stored seed and save every solution with seed and command (`--out`).

## 4. Appendix: square and cube

- Rule files: `rules/<family_id>/square/` and `rules/<family_id>/cube/`, for example `CA-C3-05-13-PI-XG` (Xiao and Gimbutas 2010, degree 5, 13 nodes, weights scaled to sum 8).
- Symmetry groups used for orbit expansion: S2, S5 and S6 (printed orbit data is expanded by the group, then the moments are checked).
- The moment check finds typos in printed data: one cube weight pair was wrong in print and was recomputed by least squares; a 6-orbit weight in one triangle table was doubled. Always run the moment check and say in the pull request what was corrected.
- Same acceptance rules as in the other domains: minimise the L2 norm of weights for atlas-found rules, make the interval proof (`--strict --save`), commit the `.proof` file, one pull request per improvement.

## 5. Literature cells that still have only a count (state 2026-11-20)

Cell = (domain, source k of `reference_data/literature_counts.json`, node count at degree n). Re-check against `python tools/state_table.py` and the atlas before relying on it. Public data tables found for the cells of the Xiao-Gimbutas rules: J. Burkardt's rule collections (triangle_symq_rule, tetrahedron_arbq_rule, cube_arbq_rule, square_arbq_rule, square_minimal_rule), the authors' repository `zgimbutas/triasymq` (BSD licence; tables with 35 digits under `tables/*/data_post.m`; its README lists the node counts of all rule types). They are imported by `code/import_burkardt_rules.py`.

Xiao-Gimbutas (2010), still without node data:
- triangle k=xg: 110@24 and 118@25 (the table of the paper lists reflection-symmetric rules; the authors' repository has only the rotationally symmetric rules with 111 nodes at degree 24 and 117 at degree 25, no reflection-symmetric code `triasyml.f`).
- square k=xg: 18@9, 27@11, 35@13, 45@15, 71@19. The published code gives fewer nodes (17, 24, 33, 44, 68 in Burkardt's square_arbq_rule; 17, 24, 33, 43, 67 in the authors' current tables); no data with the counts of the table of the paper was found.
- cube k=xg: 74@10 (the published code has 73 nodes at degree 10).
- Other counts of the current tables of the authors (square, tetrahedron and cube rules with fewer nodes than the paper) are not imported: they do not match the counts in the file.

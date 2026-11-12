# Contributing

The maintainer owns `main`. All other changes, including those made by AI agents, arrive as pull requests and are merged only when the automatic check passes and the maintainer has reviewed them.

## Workflow

1. Branch from the latest `main` (`ai/<topic>` for AI agents, any name otherwise).
2. Make one improvement per pull request, e.g. "tetrahedron degree 21: 509 → 505 nodes".
3. Run `python tools/build_atlas.py --base origin/main --changed` locally. It checks only the rules you added or modified (not the earlier ones) and must end with `OK`. The check in CI verifies all rules.
4. Open the pull request. GitHub Actions run the same check, attach the built site as an artifact and publish a live preview at `https://jukzi.github.io/cubature_atlas/pr-preview/pr-<number>/`. The link appears as a comment in the pull request; the preview is removed when the pull request is closed or merged. Pull requests from forks get only the artifact.
5. After the merge, the website updates automatically.

## Rules for rule files

- One rule per file, in `rules/<family>/<domain>/<id>.json`. See the README for the format and the id scheme.
- **Never change or delete a file with `"locked": true`.** The check rejects it. A better rule is a new file with a new id. The old one stays; the page shows the best one automatically.
- Do not store "best known" flags. They are computed.
- Store at least 15 significant digits. The check requires all monomial moments to be correct to 1e-11 per volume.
- `quality` must match the data (PI, PO, NI, NO), and the family's `quality` if the family sets one.
- **Every rule needs a proof.** Prove new and changed rules with interval arithmetic and record the proof in the report: `python tools/verify_rigorous.py --strict --merge reports/interval_proofs.json <rule files>`, then commit the rule files together with `reports/interval_proofs.json`. The build fails for any rule without a proof of exactly its numbers, and every pull request re-runs the proof for the changed rules. There are no exceptions; a rule that cannot be proven is not added. Set `"status": "proven"`.
- Floating-point checks (moment residuals, the Newton–Kantorovich test in `code/`) are useful during the search but are not proofs. Store their result as `"numerical_check"`, never as "certificate".
- Fill in `origin`, `method` and `author`. AI-generated rules say so in `author`.

## Locking

After review, the maintainer sets `"locked": true` on rules that are published or cited. Literature rules are locked from the start.

## Literature

`literature.md` lists the literature on the topic in three sections: used in the atlas, tried but not worthwhile, not used yet. Every pull request that uses, tests or finds a publication extends it (right section, sorted by date, with link and abstract, in English). Check each reference against its DOI or arXiv page; never invent a citation.

## Translations

English texts are the source (`i18n/en.json`, `content/en/`). If you change an English text, update the German one in the same pull request (`i18n/de.json`, `content/de/`) and then run `python tools/build_atlas.py --sync-hashes`. Do not run `--sync-hashes` without actually updating the translation.

## Commit messages

Commits made with AI assistance carry a trailer, e.g. `Co-Authored-By: Claude <noreply@anthropic.com>`. `git log --grep` then gives the record of AI involvement that is needed for publications.

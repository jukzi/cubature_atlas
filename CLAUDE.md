# Instructions for AI agents working on this repository

Read README.md and CONTRIBUTING.md first. In short:

- Work on a new branch from the latest `main`; never push to `main`. Open one pull request per improvement.
- Never modify or delete a rule file with `"locked": true`. Add a new file with a new id instead.
- Before committing, run `python tools/build_atlas.py --base origin/main`. Do not open a pull request unless it prints `OK`.
- Do not weaken the checks in `tools/build_atlas.py` (tolerances, quality rules, locked check) to make a rule pass.
- New families: create `rules/<family_id>/family.json` with `name`, `short` and `description` in both `en` and `de`.
- If you change English texts (`i18n/en.json`, `content/en/`), update the German texts in the same pull request, then run `--sync-hashes`.
- Every reference you add must be checked to exist; never invent citations.
- Code conventions: pure Python, a single file per program, English comments, one statement per line.
- Every rule needs an interval proof: run `python tools/verify_rigorous.py --strict --merge reports/interval_proofs.json <rule files>` and commit the updated report with the rules. Never call a floating-point check a certificate or a proof; it is a numerical check.
- State in the pull request description what was computed, how it was proven, and what is still open.
- Tooltips on the website: never hover only (`title` alone). Every tooltip needs a clickable, keyboard-reachable ⓘ (or marker) in addition to hover, so it also works on touch screens; use the shared popup (`td .oq` / `th .hw` with `data-tip`).
- Exponents on the website: never Unicode superscript characters (⁰ ¹ ² ³ ⁴ … ⁻); depending on the font some of them slip down. In HTML use `<sup>`; in plain text (e.g. popups filled with `textContent`) write `10^-14` or `1e-14`.

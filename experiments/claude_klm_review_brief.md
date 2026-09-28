# Review brief — TODO 036 (harness fixes k, l, m, j, gate alignment), branch `fix/harness-klm`

You are the **independent reviewer**. A pi agent (Sol 6.0) built this branch, and Claude (the
manager) will act on your findings.
- **Do not fix code yourself.** Report findings.
- Do not spawn subagents.
- Be concrete and skeptical: treat the builder's report as claims to verify, not facts.

## Scope and inputs

- **Worktree:** `/home/james/bet_project/.worktrees/BayesianFootball-harness-klm`.
  - Diff under review: `git diff feat/w2-tier-components...fix/harness-klm`.
  - Pull the branch head first.
- **Contract:** `experiments/pi_harness_klm_prompt.md` (the agreed design; do not re-litigate the
  decisions in it) and the acceptance criteria in
  `todos/036_harness_persistence_and_reporting_fixes_k_l_m_j_gate_alignment.md`.
- **Builder's report:** `docs/architecture/harness_klm_report.md`.
- **Repo rules:** `AGENTS.md`, `docs/guides/julia_coding_context_for_agents.md`,
  `docs/guides/experiment_database_and_config_truth_guide.md` §0.

## Rules

- No credentials in files or output.
- **No Julia on the laptop.** Run Julia on mcmc-beast (`ssh root@mcmc-beast`), in your own tmux
  session there named `claude_klm_review`:
  - from a clean `bash /root/BF_runs/8effca72/scripts/beast_checkout.sh <head-sha>` checkout;
  - copy `/root/BF_runs/a76a65df/.cache/datastore_Scottish*.jls` into its `.cache/`;
  - `/root/.juliaup/bin/julia --project`, after `set -a; source .env; set +a`;
  - logs in `/root/BF_runs/logs/klm_review/`.
- Check `pgrep -af julia` first, and run one heavy job at a time. The Kaimon process in tmux
  `julia` is expected.
- **No sampling.** Production `mcmc_experiments` is **read-only** (SELECT and `load_fit` only).
  Integration tests use `mcmc_experiments_test` via `BF_EXPERIMENTS_TEST_DB_URL`.
- You may commit **only** `docs/architecture/harness_klm_review.md` on `fix/harness-klm` and push
  it. Touch nothing else.

## What to check

1. **(l) per-fold storage.**
   - Is `save_fit` one transaction?
   - Is each fold blob bound in binary and well under 1 GiB?
   - Is peak memory about one fold?
   - Is the layout marker unambiguous?
   - Is `ensure_schema!` purely additive and idempotent against an existing production schema
     (no destructive DDL, safe to re-run)?
   - Does `load_fit` read both layouts, bit-exact?
   - Do legacy runs (`ce7ea22f`, `c4a0fa94`, `132df5c2`) still load?
   - `extend_fit`: are fold blobs appended, never rewritten? Is the legacy-run behaviour as
     documented?
   - Does `ON DELETE CASCADE` hold?
   - Is the stride fallback gone, with grids at stride 1?
   - Are FileStorage and DualStorage unaffected?
2. **(j):**
   - Is the smoke nonce inside `config_hash`?
   - Is the grid `recipe_hash` unchanged? (Compare against `harness_checks.recipe_hash` of the W2
     runs for a few recipes.)
   - Is the `on_duplicate` default unchanged, and does the grid path use `:error` with a clear
     message?
   - Can re-smoking twice pass `fit_parity`?
3. **(k) + gate:**
   - Do failing diagnostics persist their value? Do unexpected exceptions still record `{}` plus
     the error?
   - Are the thresholds exactly R̂ > 1.05 and divergence rate > 0.1%, with the right comparison
     direction and divergence-rate denominator?
   - Is ESS a note only?
   - Are `Training` defaults and `fold_results.converged` untouched?
4. **(m):**
   - Does the leaderboard show every (run, control) pairing, with no panel mixing?
   - Does the v1.2 board, generated from the test data or a read-only dry run, keep every run on
     the committed v1.1 board (`git show 090db11e:experiments/LEADERBOARD.md`)?
   - Does the MAP-screen section survive?
   - Runbook CSVs: are the 14 W0 runs and the MAP-screen runs complete, with the same controls as
     in v1.1, and the right panel flags and loaders?
   - Would the production steps write only additive v1.2 rows?
5. **Additions made under Amendment 1** (`experiments/pi_harness_klm_amendment1.md`):
   - **The `--test-db` flag on `run_candidates.jl`.** Is it impossible for it to hit production by
     accident (e.g. the env var unset, or naming the wrong DB)?
   - **The completed-register protection for re-screens.** Manager's note: the W1 register row's
     `run_ids` are **semicolon**-separated (seeded), while `write_experiment!` merges by splitting
     on commas. Would the production re-screen produce a mixed-separator or corrupted run list?
     Check this against the real production value (read-only).
   - **The NaN → JSON null mapping.** Is it confined to the persistence boundary?
   - **Screen check-write failures.** Are they now surfaced rather than swallowed?
   - **The production runbook** (`docs/architecture/harness_klm_prod_runbook.md`). Is it correct,
     ordered, and additive only?
6. **Tests:**
   - Independently re-run on the beast: `test/test_db_storage.jl` and `test/test_extension.jl`
     (with the test DB), the harness scoring and runner suites, and the synthetic > 1 GiB round-trip
     script. Compare the counts with the builder's report.
   - Look for tests that pass vacuously: skipped, gated off, or asserting on the wrong object.
7. **Code quality:** matches the surrounding idiom, surgical edits, docstrings current, and no
   dead code left from the stride ladder.

## Output

Write `docs/architecture/harness_klm_review.md`:
- a verdict line, `VERDICT: ACCEPT` or `VERDICT: CHANGES_REQUIRED`;
- a findings table: ID, severity (blocker / major / minor / nit), `file:line`, the defect, a
  concrete failure scenario, and the fix you suggest;
- the commands you ran with their pass counts, and anything you could not verify.

Only blockers and majors justify `CHANGES_REQUIRED`. Commit and push that one file, then print
exactly `KLM_REVIEW_DONE` on its own line and stop.

For a re-review after a fix round, append a "Re-review N" section to the same file with the updated
verdict, and print `KLM_REREVIEW<N>_DONE`.

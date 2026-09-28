# Work package — TODO 036: harness persistence and reporting fixes (k, l, m, j, gate alignment)

You are a **solo implementer**; Claude is the manager.
- **No subagents, and no subagent review.** When you are done, a separate Claude CLI (Opus 5.5)
  reviews your work.
- The design below was agreed with the human in a grill-me session on 2026-09-28. **Do not
  re-open these decisions.** If one turns out to be impossible, stop and report (§6).

## 0. Rules

- **Worktree** `/home/james/bet_project/.worktrees/BayesianFootball-harness-klm`, branch
  **`fix/harness-klm`** (cut from `feat/w2-tier-components` at the commit that adds this brief).
  - Work only there, commit in small logical steps, and push **this branch only**.
  - Never commit to or push another branch, and never merge. The manager merges.
- **Credentials:** none in files, prompts or logs. `BF_DB_URL` comes from `.env`; `PostgresStorage`
  resolves the experiments DB. Never print `conn_str`.
- **The laptop runs no Julia.** All Julia runs on **mcmc-beast** (`ssh root@mcmc-beast` works
  non-interactively), in your own tmux session there, named `pi_klm`.
  - **Iterating:** rsync the worktree to `/root/BF_runs/klm_dev/` with
    `--exclude '/.cache/' --exclude '/data/' --exclude '/.git/'`. Symlink `.env` and
    `Manifest.toml` from `/root/BayesianFootball/`, and copy
    `/root/BF_runs/a76a65df/.cache/datastore_Scottish*.jls` into its `.cache/`.
  - **Final evidence:** from a clean `bash /root/BF_runs/8effca72/scripts/beast_checkout.sh <sha>`
    checkout of a pushed SHA. Copy the same datastore caches into it.
  - Julia is `/root/.juliaup/bin/julia --project`, after `set -a; source .env; set +a`. Use `-t 16`
    for test suites.
  - Logs go in `/root/BF_runs/logs/klm/`, never inside a checkout.
  - Before anything heavy, check `pgrep -af julia`. The Kaimon process in tmux session `julia` is
    expected; leave it alone. One heavy job at a time.
- **No sampling.** No `run_candidates.jl` smoke or grid, and no model fits beyond what the existing
  test suites already do.
- **Databases.**
  - **Production `mcmc_experiments` is read-only for you until §5.** You may `load_fit` and
    `SELECT` from it. Never `INSERT`, `UPDATE`, `DELETE`, `ALTER` or `CREATE` there.
  - Create and use a **separate test database** `mcmc_experiments_test` on the same server
    (`CREATE DATABASE`), and point `BF_EXPERIMENTS_TEST_DB_URL` at it (built from the same
    credentials source; never printed). All integration tests and the synthetic round trip use it.
    Drop test objects you create when finished, but keep the database.
- **Code style:** read `docs/guides/julia_coding_context_for_agents.md` first. Match the
  surrounding code, keep edits surgical, and keep docstrings current. No `@model` changes are
  expected.

## 1. Read first

1. `todos/036_harness_persistence_and_reporting_fixes_k_l_m_j_gate_alignment.md` and the TODO 031
   Work Log entries for 2026-09-27 and 2026-09-28 (follow-ups j, k, l, m).
2. `src/training/inference/db_storage.jl` in full: `save_fit`, `load_fit`, `config_hash`,
   `_db_recipe_tags`, `_db_artifact_blob`/`_db_artifact_value`, `_db_exec_binary`,
   `_db_query_blob`, `ensure_schema!`, the latent insert/load.
3. `src/training/inference/extension.jl` (`extend_fit` and its PostgreSQL path),
   `src/training/inference/types.jl` (`Fit`, `FoldFit`) and `src/training/inference/io.jl`
   (FileStorage/DualStorage, which must keep working).
4. `src/harness/stages.jl` (smoke and grid stages, the stride ladder),
   `src/harness/checks.jl` (`_run_diagnostic!`, `_convergence_diagnostic`, `thin_for_persistence`),
   `src/training/inference/convergence.jl` (`summarise_convergence`, thresholds).
5. `scripts/leaderboard.jl`, `src/harness/scoring.jl` (`leaderboard`, `_headline`),
   `src/harness/store.jl` (`read_scores`, `write_scores!`), `scripts/score_runs.jl`.
6. `docs/guides/experiment_harness_guide.md` and
   `docs/guides/experiment_database_and_config_truth_guide.md` §0 and §2.
7. Tests: `test/test_db_storage.jl`, `test/test_extension.jl`, `test/harness_*_tests.jl`.

## 2. What to build

### Item 1 — (k) diagnostics keep their metrics on failure
- A failing diagnostic must persist the metrics it computed in `harness_checks.value`, with the
  reason in `detail` and status `fail`.
- Suggested approach: diagnostics signal failure by returning or throwing a value-carrying result,
  e.g. a `DiagnosticFailure(value, detail)` exception that `_run_diagnostic!` unpacks.
- A diagnostic that errors for an unexpected reason (a real exception) still records `{}` plus the
  error text.
- This applies to every `_run_diagnostic!` caller, not just convergence.

### Item 2 — convergence gate aligned with the human rule
- The harness convergence **review** fails only when **max R̂ > 1.05** or **divergences > 0.1% of
  post-warm-up draws** (the human decision of 2026-09-27; W2 README §7). Name both thresholds as
  constants that cite that decision.
- ESS below 400 (bulk or tail), BFMI and tree-depth findings are recorded in `value`
  (e.g. `notes = [...]`) and stay `pass`. The `[SUMMARY] review=` line follows the new status.
- **Scope: the harness review only.**
  - Do not change `Training` convergence defaults or `fold_results.converged`, which MatchDay and
    other code may use.
  - List every other site that uses 1.01 or 400 thresholds in your report, without changing it.

### Item 3 — (l) per-fold fit artefacts (the main item)
- **Schema** (additive, created by `ensure_schema!` with `IF NOT EXISTS`): a new table
  `fit_fold_artifacts`.
  - Columns: `run_id uuid REFERENCES runs(run_id) ON DELETE CASCADE`, `fold_idx int`,
    `fold_blob bytea`, `PRIMARY KEY (run_id, fold_idx)`.
  - `fit_artifacts` keeps one row per run, which becomes a **shell**: the `Fit` without chain data
    (config, diagnostics, metadata, fold metadata). The layout must be detectable without
    guessing, e.g. an additive `layout text` column (`'single'` default for legacy rows,
    `'per_fold'` for new ones) or a versioned shell type.
- **`save_fit(fit, ::PostgresStorage)`** writes the shell plus one zstd-compressed blob per fold,
  each bound with `_db_exec_binary`, in **one transaction**. Serialize and compress one fold at a
  time, so peak memory is about one fold, not the whole fit.
- **`load_fit`** reads both layouts.
  - Per-fold: reassemble the `Fit` in fold order, **bit-exact** equal to what was saved.
  - Legacy: exactly today's behaviour. **All 324 existing runs must still load; do not migrate or
    rewrite them.**
- **`extend_fit`** on a per-fold run appends new fold rows and updates the shell and
  `fold_results`. It never rewrites existing fold blobs. On a legacy single-blob run, either keep
  today's rewrite behaviour or convert it to per-fold on first extension. Choose one, test it and
  document it.
- **Stride:** remove the 1 → 2 → 4 thinning fallback in `src/harness/stages.jl`. Grids always
  persist at stride 1, and a save failure is a real error.
  - Keep the `persistence_stride` info check (it records 1) for continuity.
  - `thin_for_persistence` may stay as a utility.
- FileStorage and DualStorage must keep working unchanged.

### Item 4 — (j) re-smoke and duplicate-hash handling
- `save_fit(...; on_duplicate::Symbol = :return)`:
  - `:return` is today's silent return of the existing run ID (the default, so other callers are
    unchanged);
  - `:error` throws a clear error naming the existing run and the fix.
- **Smoke:** each smoke save carries a per-invocation nonce in the recipe tags, so its
  `config_hash` differs and re-smoking an unchanged recipe works. The nonce tag must be **included**
  in the hash (`_db_recipe_tags` drops `time:` tags, so do not use that prefix). Smoke runs stay
  in `<experiment>_smoke`. The grid's recipe hash (`harness_checks.recipe_hash`) must **not**
  change: smoke validity stays keyed by recipe.
- **Grid:** the harness grid path calls `save_fit(...; on_duplicate = :error)`.

### Item 5 — (m) leaderboard, W0 and MAP-screen re-scoring (code and runbook only)
- `scripts/leaderboard.jl` / `Harness.leaderboard` shows each row's Δ LogLoss against **its own
  recorded control** (`harness_scores.control_run_id`), with the control named in a column. Remove
  the hard-coded `W0_CONTROL_UUID` dependence.
  - Where a run has several controls (e.g. `m12_td_ref` against both `td_lower_base` and
    `td_base`), show each pairing without dropping any.
  - Rows from different panels are still never mixed.
- **Keep the MAP-screen section** on the v1.2 board.
- **Runbook for §5:** `docs/architecture/harness_klm_prod_runbook.md` plus CSVs that re-score, at
  v1.2 (scoring only):
  - the **14 W0 reference runs** present on the committed v1.1 `experiments/LEADERBOARD.md`
    (`git show 090db11e:experiments/LEADERBOARD.md`) but absent from the v1.2 board, each with the
    control it had in v1.1;
  - the MAP-screen runs.

  Work out their experiments, panels (`--target-seasons` / `--expected-fixtures`) and loaders from
  the DB and the code. Dry-run every CSV's `load_run_refs`/control grouping locally in the test DB
  context where possible. **Do not write v1.2 rows to production yet.**

## 3. Tests and validation (all on the beast; report counts and wall times)

1. **Unit and integration** (test DB):
   - (k): a failing diagnostic persists its value; an unexpected exception records `{}`.
   - Gate: R̂ 1.04 with ESS 300 passes with a note; R̂ 1.06 fails; divergence rate 0.11% fails.
   - (l): save/load parity, bit-exact, for a multi-fold fit; a legacy-layout row written the old way
     still loads; `extend_fit` appends fold rows without rewriting existing ones (compare blob
     hashes before and after); `ON DELETE CASCADE` removes fold rows.
   - (j): re-smoking the same recipe twice saves two distinct smoke runs and passes `fit_parity`;
     `on_duplicate = :error` throws; the default `:return` is unchanged.
   - (m): a synthetic score table with two controls and two panels renders every pairing, with none
     lost and no panel mixing.
2. **Synthetic > 1 GiB round trip** (test DB, no sampling):
   - Load the real W2 `grw_base` fit (`ce7ea22f-61d8-40cc-a505-94d101b98604`) read-only from
     production. Replicate its folds (renumbered) until the serialized size exceeds **1.2 GiB**.
   - Save it through the per-fold path, then `load_fit` it back.
   - Require `==` equality of chains, config, diagnostics and metadata. Report the total bytes,
     largest fold blob, save and load seconds, and peak RSS.
3. **Legacy loads:** `load_fit` via the new code loads, read-only from production and unchanged:
   - `ce7ea22f` (W2, stride 2);
   - `c4a0fa94` (W2, stride 4);
   - `132df5c2` (W0 `m12_td`);
   - one pre-PR #42 hex-era run.

   Report chain dimensions.
4. **Existing suites stay green**, with counts before and after: harness scoring, harness runner,
   portfolio, pyramid tiers, splitter, `test/test_db_storage.jl` and `test/test_extension.jl` (both
   with `BF_EXPERIMENTS_TEST_DB_URL` set), and `test/run_parallel_tests.jl` (no worse than the
   known T007).
5. `./scripts/todo.sh check` passes.

## 4. Report, then stop (build phase)

- Write `docs/architecture/harness_klm_report.md` with:
  - evidence per item: commands, SHAs, numbers;
  - the validation tables;
  - the threshold-site list (Item 2);
  - design choices, including the legacy `extend_fit` choice;
  - anything you could not do.
- Add a dated `@pi` line to the TODO 036 Work Log. Commit, push `fix/harness-klm`, and print the
  head SHA.
- Then print exactly `KLM_BUILD_DONE` on its own line and **stop**.
- **Review loop:** the manager sends you review findings (`docs/architecture/harness_klm_review.md`).
  Fix them, push, append a "Fix round N" section to your report, and print `KLM_FIX<N>_DONE`
  (N = 1 or 2). Then stop again.

## 5. Production phase — ONLY when the manager explicitly says "go prod" (after merge)

- Work from a clean beast checkout of the **merged** `feat/w2-tier-components` SHA the manager
  names. Follow your runbook:
  1. `ensure_schema!` against `mcmc_experiments`, which creates `fit_fold_artifacts` and the layout
     marker. Verify with `\d`.
  2. Re-score the W0 and MAP-screen runs at v1.2 with `scripts/score_runs.jl`. Do not delete or
     overwrite v1.1 rows.
  3. Run `scripts/leaderboard.jl`. Diff the new board against the v1.1 board: **every run on the
     v1.1 board must appear**, and the W2 section must show per-base Δs that match the W2 README.
  4. Copy `experiments/LEADERBOARD.{csv,md}` and `REGISTER.md` back, commit them on
     `feat/w2-tier-components` (the manager will tell you the worktree), and push.
  5. Append a "Production" section to your report.
- Print `KLM_PROD_DONE` and stop.

## 6. If blocked

Write the problem under a "Blocked" heading in your report, push, print `KLM_BLOCKED` with a
one-line reason, and stop. Blocked means: a design decision above is impossible, a test fails the
same way twice, a production write would be needed before §5, or sampling would be needed.

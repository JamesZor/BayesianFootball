# Harness KLM — build phase blocked (2026-09-28)

## Blocked

The agreed §2 Item 5 / §5 production procedure requires **re-scoring the 12 MAP-screen runs at v1.2 with `scripts/score_runs.jl`, without sampling**. This is impossible under the current data contract:

- `src/harness/stages.jl:screen` fits MAP in memory, persists only `harness_scores` and `harness_checks`, and never calls `save_fit`. The v1.1 leaderboard itself warns: screen recipe UUIDs are **not** rows in `mcmc_experiments.runs`.
- Read-only production query (on mcmc-beast, `docker exec mcmc_experiments_postgres psql -U postgres -d mcmc_experiments`): `SELECT count(DISTINCT h.run_id), count(DISTINCT r.run_id) FROM harness_scores h LEFT JOIN runs r ON r.run_id=h.run_id WHERE h.scorecard_version='v1.1' AND h.stage='screen';` → **12 | 0**.
- `scripts/score_runs.jl:main` calls `Harness.score_runs`; that function calls `Training.load_fit(PostgresStorage(ref.experiment), ref.run_id)` for each run. No screen fit exists to load, so a CSV cannot re-score these rows. Re-running `screen` would fit MAP again (sampling/fits disallowed by §0); copying old v1.1 scores into v1.2 is not re-scoring and was expressly rejected as a version fallback.

Per work package §6, stopping rather than substituting an unapproved method. **Manager decision needed:** permit a deterministic MAP screen re-run (new fits), or accept preserving the v1.1 MAP section explicitly labelled v1.1 instead of claiming v1.2 re-scoring. No production writes were made.

## Partial build (not release-ready)

- Implemented candidate changes for diagnostic metric-carrying failures, harness-only 1.05 / 0.1% gate, per-fold PostgreSQL shell/blob layout and legacy load, append-on-extension (legacy stays single-blob), no thinning fallback, smoke nonce and duplicate policy, and per-panel/per-control leaderboard rows. These are **partial and not approved for production**.
- Added unit/integration checks for failures, gate boundaries, multi-control leaderboard, duplicate policy, legacy-row load, cascade, and fold-blob hash stability.
- Beast development sync: `rsync -a --exclude '/.cache/' --exclude '/data/' --exclude '/.git/' ./ root@mcmc-beast:/root/BF_runs/klm_dev/`; test DB `mcmc_experiments_test` created via Docker Postgres; `.env` and `Manifest.toml` linked, Scottish caches copied. Credentials were derived at runtime and not logged.
- `test/harness_scoring_tests.jl` and `test/harness_runner_tests.jl`: runner **159/159 in 1m15.8s**, scoring tests to this point passed (see `/root/BF_runs/logs/klm/focused.log`). Initial synthetic scoring test errored due to a typed `Vector` of missing-only control IDs; corrected and second run passed the scoring section. The second combined invocation could not connect to the test DB because the production-only pgpass entry did not match the new database; resolved using the same Docker credential in process environment (never printed).
- `test/test_extension.jl` with test DB: **43/43, 27.0 s** (`/root/BF_runs/logs/klm/extension.log`). `test/test_db_storage.jl`: **133/133, 33.5 s** (`/root/BF_runs/logs/klm/db_storage.log`). Test-created run rows removed from `mcmc_experiments_test`; database retained.
- The 14 W0 UUIDs missing from v1.2 were identified read-only by comparing `git show 090db11e:experiments/LEADERBOARD.md` against the production v1.2 grid UUIDs; all are recoverable from production, but the required MAP CSV cannot be made truthful.

**Not done:** §3 synthetic >1.2 GiB round trip, four production legacy loads, complete suite matrix and clean pushed-SHA validation, production runbook/CSVs, acceptance sign-off. `experiments/LEADERBOARD.md`, `REGISTER.md` and production `mcmc_experiments` were not changed. No sampling was run.

## Other threshold sites (read-only inventory so far)

`src/training/inference/types.jl:224-247` documents 1.01 and 400 (actual `ConvergenceThresholds.max_rhat` is already 1.05; `min_ess` is 400). `src/training/inference/convergence.jl:summarise_convergence` applies these Training defaults, and `src/training/inference/db_storage.jl:save_fit` plus `src/training/inference/extension.jl:_extension_insert_fold!` use them for `fold_results.converged`. No Training gate was changed. Full site audit remains outstanding because of the block.

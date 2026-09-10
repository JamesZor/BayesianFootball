# 005 — Rerun Scottish Lower Experiment 01 Poisson grid with ReverseDiff and verify posterior parity

| Field | Value |
|---|---|
| ID | 005 |
| Title | Rerun Scottish Lower Experiment 01 Poisson grid with ReverseDiff and verify posterior parity |
| Status | IN_PROGRESS |
| Priority | P1 |
| Assignee | pi |
| Created | 2026-09-10 |
| Updated | 2026-09-10 |
| Related Files / Commits / PRs | [r20_train_5models_2426_unified.jl](../experiments/scottish_lower/01_poisson_2426_grid/r20_train_5models_2426_unified.jl); [r21_sync_to_postgres.jl](../experiments/scottish_lower/01_poisson_2426_grid/r21_sync_to_postgres.jl); [README.md](../experiments/scottish_lower/01_poisson_2426_grid/README.md) |

## Context & Problem Statement

With the NUTS sampler `adtype` bug fixed (committing compiled ReverseDiff directly into `Turing.NUTS`), sampling performance on 16 cores has accelerated by ~66x. Experiment 01 (`experiments/scottish_lower/01_poisson_2426_grid/`) originally evaluated 5 composable Poisson models across a 40-fold walk-forward grid (Seasons 24/25 and 25/26, 710 out-of-sample matches).

The user requested launching a `pi` agent in tmux to re-run the Experiment 01 Poisson grid using the new ReverseDiff sampler, persist the resulting fits to PostgreSQL `mcmc_experiments`, ensure the runs are clearly distinguished by version / backend (e.g. `ad_backend = 'reversediff'`), and verify that the posterior parameter estimates and proper scores match the historical baseline published in `experiments/scottish_lower/01_poisson_2426_grid/README.md`.

## Acceptance Criteria

- [ ] Add and populate `ad_backend` column in PostgreSQL `mcmc_experiments.runs` (or record `tags = ["reversediff"]`) to distinguish ReverseDiff runs from earlier runs.
- [ ] Run the 5-model walk-forward grid (`m00_baseline`, `m02_wealth`, `m03_distance`, `m04_joint`, `m05_production_wealth`) with compiled ReverseDiff across all 40 folds under `QueuedNUTSConfig` on `mcmc-beast`.
- [ ] Persist the resulting fits and latents to PostgreSQL `mcmc_experiments`.
- [ ] Compare the posterior parameter means (home advantage γ, feature weights w) and proper scores (LogLoss, Brier, RPS) against the reference values in `01_poisson_2426_grid/README.md` to verify statistical equivalence.
- [ ] Document findings and update work log in this TODO.

## Ideas & Candidate Solutions

- **Runner Script**: Use or adapt `r20_train_5models_2426_unified.jl` to persist directly to `PostgresStorage("scottish_lower_poisson_2426")` with `ad_backend = "reversediff"`.
- **Deduplication**: `save_fit` deduplicates on `config_hash`. By adding `"reversediff"` or `"sampler_v2"` to `FitConfig.tags`, each run gets a fresh distinct `config_hash` without colliding with historical synthetic entries.

## Work Log & Progress

- [2026-09-10 @antigravity] Created task and assigned to `@pi`. Verified `runs.ad_backend` column added on `mcmc_experiments`. Prepared tmux window `exp01_poisson` in `pi_solo` for agent delegation.

## Verification & Findings

Pending execution by pi agent.

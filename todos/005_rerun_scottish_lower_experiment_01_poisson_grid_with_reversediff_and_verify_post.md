# 005 — Rerun Scottish Lower Experiment 01 Poisson grid with ReverseDiff and verify posterior parity

| Field | Value |
|---|---|
| ID | 005 |
| Title | Rerun Scottish Lower Experiment 01 Poisson grid with ReverseDiff and verify posterior parity |
| Status | COMPLETED |
| Priority | P1 |
| Assignee | pi |
| Created | 2026-09-10 |
| Updated | 2026-09-10 |
| Related Files / Commits / PRs | [r20_train_5models_2426_unified.jl](../experiments/scottish_lower/01_poisson_2426_grid/r20_train_5models_2426_unified.jl); [r21_sync_to_postgres.jl](../experiments/scottish_lower/01_poisson_2426_grid/r21_sync_to_postgres.jl); [README.md](../experiments/scottish_lower/01_poisson_2426_grid/README.md); [T009](../docs/tickets/T009-experiment-01-runners-select-oldest-fit.md) |

## Context & Problem Statement

With the NUTS sampler `adtype` bug fixed (committing compiled ReverseDiff directly into `Turing.NUTS`), sampling performance on 16 cores has accelerated by ~66x. Experiment 01 (`experiments/scottish_lower/01_poisson_2426_grid/`) originally evaluated 5 composable Poisson models across a 40-fold walk-forward grid (Seasons 24/25 and 25/26, 710 out-of-sample matches).

The user requested launching a `pi` agent in tmux to re-run the Experiment 01 Poisson grid using the new ReverseDiff sampler, persist the resulting fits to PostgreSQL `mcmc_experiments`, ensure the runs are clearly distinguished by version / backend (e.g. `ad_backend = 'reversediff'`), and verify that the posterior parameter estimates and proper scores match the historical baseline published in `experiments/scottish_lower/01_poisson_2426_grid/README.md`.

## Acceptance Criteria

- [x] Add and populate `ad_backend` column in PostgreSQL `mcmc_experiments.runs` (or record `tags = ["reversediff"]`) to distinguish ReverseDiff runs from earlier runs.
- [x] Run the 5-model walk-forward grid (`m00_baseline`, `m02_wealth`, `m03_distance`, `m04_joint`, `m05_production_wealth`) with compiled ReverseDiff across all 40 folds under `QueuedNUTSConfig` on `mcmc-beast`.
- [x] Persist the resulting fits and latents to PostgreSQL `mcmc_experiments`.
- [x] Compare posterior parameter means and proper scores against `01_poisson_2426_grid/README.md`. The parity hypothesis is **accepted for LogLoss/Brier but rejected for the published fold-1 parameters and RPS**; details below.
- [x] Document findings and update work log in this TODO.

## Ideas & Candidate Solutions

- **Runner Script**: Use or adapt `r20_train_5models_2426_unified.jl` to persist directly to `PostgresStorage("scottish_lower_poisson_2426")` with `ad_backend = "reversediff"`.
- **Deduplication**: `save_fit` deduplicates on `config_hash`. By adding `"reversediff"` or `"sampler_v2"` to `FitConfig.tags`, each run gets a fresh distinct `config_hash` without colliding with historical synthetic entries.
- **Convergence retries**: Preserve failed/partial attempts rather than deleting database history. Add an attempt tag and rerun only the affected model under the unchanged 4×800, 800-warmup, 0.65-target recipe; select only a 40/40-converged UUID for the final comparison.
- **Parity interpretation**: Treat a negative parity result as a finding. Do not widen tolerances or relabel the current standard 1X2 RPS merely to reproduce the README table.

## Work Log & Progress

- [2026-09-10 @antigravity] Created task and assigned to `@pi`. Verified `runs.ad_backend` column added on `mcmc_experiments`. Prepared tmux window `exp01_poisson` in `pi_solo` for agent delegation.
- [2026-09-10 @pi] Read the execution, Julia, remote-compute and experiment-database contracts. Confirmed the server was idle from production sampling, the target checkout was branch `feat/scottish-lower-goal-decomposition` at `2adc1d7f`, the 53 MB Scottish Lower cache existed, and only the five historical `synthetic-no-mcmc` rows occupied the experiment namespace.
- [2026-09-10 @pi] Updated `r20_train_5models_2426_unified.jl` to register ReverseDiff-tagged recipes, preflight completed tagged runs, persist every Fit immediately through `PostgresStorage`, persist proper scores, retain local artifacts, and support model/attempt filters for convergence retries. Launched it in server tmux sessions with Julia `-t 16`, `pinthreads(:cores)` and BLAS=1.
- [2026-09-10 @pi] Completed the five-arm grid. The first full run took about 14 minutes wall time including package precompilation, feature construction, PostgreSQL/local persistence, evaluation and portfolio reporting; individual Fit metadata recorded 118–160 seconds for the final selected arms.
- [2026-09-10 @pi] The first `m02` and `m04` attempts each had one fold with four divergences (39/40 relational fold gates). An unchanged-recipe `m02` retry passed 40/40. The second `m04` attempt failed tail ESS and two fold gates; a third unchanged-recipe attempt passed 40/40. Failed attempts remain in PostgreSQL with explicit attempt/convergence tags; no history was deleted.
- [2026-09-10 @pi] Loaded all five selected Fits back from PostgreSQL and compared every chain array and both latent matrices with the corresponding local artifact: exact equality for 40/40 folds and 710/710 `CountLatents` rows in every arm.
- [2026-09-10 @pi] Found that Experiment 01's compare, portfolio and sync runners choose the oldest artifact from `list_fits`, whose API is newest-first. Kept the fix out of this rerun and raised [T009](../docs/tickets/T009-experiment-01-runners-select-oldest-fit.md).

## Verification & Findings

### Selected PostgreSQL runs

All rows are in experiment `scottish_lower_poisson_2426`, have `status = 'completed'`,
`ad_backend = 'reversediff'`, include the `reversediff` tag, contain 40 fold rows and 710
relational latent rows, and have all 40 fold gates passing. The checkout was based on
`2adc1d7f`; PostgreSQL records `2adc1d7f-dirty` because the persistence-only runner changes
were synced without creating an unauthorised Git commit.

| Model | Run UUID | Fit time | Folds | Max R-hat | Min bulk/tail ESS | Divergences |
|---|---|---:|---:|---:|---:|---:|
| `m00_baseline` | `2722f7e2-0ee6-4040-95cc-55420800b1c3` | 133.4 s | 40/40 | 1.0095 | 814 / 1078 | 7 |
| `m02_wealth` | `6b23f2ca-700a-4c56-b520-b97510770f82` | 144.0 s | 40/40 | 1.0092 | 920 / 489 | 10 |
| `m03_distance` | `e20b7b86-cad8-407c-a8ca-002af00dc357` | 120.5 s | 40/40 | 1.0116 | 887 / 565 | 8 |
| `m04_joint` | `91a798c3-d369-4fbe-9670-70acb50f956e` | 160.3 s | 40/40 | 1.0100 | 1079 / 553 | 11 |
| `m05_production_wealth` | `9239e392-897f-490e-aad4-9e3cc7c6cf5b` | 117.8 s | 40/40 | 1.0091 | 959 / 502 | 5 |

Divergences are totals across 160 chains per arm (128,000 retained draws). No selected fold
exceeded the configured per-fold convergence threshold. Every max R-hat is below 1.05.

### Proper-score comparison

| Model | LogLoss new / published | Delta | Brier new / published | Delta | RPS new / published | Delta |
|---|---:|---:|---:|---:|---:|---:|
| `m00_baseline` | 0.660353 / 0.6603 | +0.000053 | 0.234162 / 0.2341 | +0.000062 | 0.227259 / 0.1770 | +0.050259 |
| `m02_wealth` | 0.660101 / 0.6601 | +0.000001 | 0.234049 / 0.2341 | -0.000051 | 0.226718 / 0.1769 | +0.049818 |
| `m03_distance` | 0.660345 / 0.6603 | +0.000045 | 0.234157 / 0.2341 | +0.000057 | 0.227343 / 0.1770 | +0.050343 |
| `m04_joint` | 0.660042 / 0.6600 | +0.000042 | 0.234016 / 0.2340 | +0.000016 | 0.226787 / 0.1768 | +0.049987 |
| `m05_production_wealth` | 0.659731 / 0.6597 | +0.000031 | 0.233879 / 0.2338 | +0.000079 | 0.226222 / 0.1766 | +0.049622 |

LogLoss and Brier reproduce the published ranking and are within `8e-5` of the rounded
reference values, which is also within the observed between-retry score variation (`8.3e-5`
for `m02` LogLoss). **RPS does not reproduce**: the current, standard ordered-1X2 evaluator
is consistently about `0.050` higher for all five models. The RPS implementation already
existed before the README was published and no historical real Fit artifact survives in the
experiment database (the five older rows are explicitly synthetic), so this run cannot
reconstruct how the README's RPS column was produced. It would be incorrect to call this
RPS parity.

### Fold-1 posterior comparison

The original `r20` runner extracts these values from fold 1, so the rerun uses that same
scope. MCSE is the current fold-1 chain's `MCMCChains.summarystats` value.

| Model / parameter | New | Published | Delta | Current MCSE | abs(Delta)/MCSE |
|---|---:|---:|---:|---:|---:|
| `m00` home advantage | 0.147117 | 0.158 | -0.010883 | 0.001267 | 8.59 |
| `m02` home advantage | 0.141879 | 0.156 | -0.014121 | 0.001140 | 12.39 |
| `m02` raw wealth | 0.112237 | 0.125 | -0.012763 | 0.000726 | 17.58 |
| `m03` home advantage | 0.133881 | 0.140 | -0.006119 | 0.001337 | 4.58 |
| `m03` distance | 0.062202 | 0.045 | +0.017202 | 0.000548 | 31.38 |
| `m04` home advantage | 0.133054 | 0.137 | -0.003946 | 0.001086 | 3.63 |
| `m04` raw wealth | 0.110524 | 0.124 | -0.013476 | 0.000754 | 17.88 |
| `m04` distance | 0.062614 | 0.044 | +0.018614 | 0.000507 | 36.70 |
| `m05` home advantage | 0.142097 | 0.155 | -0.012903 | 0.001194 | 10.81 |
| `m05` production wealth | 0.127331 | 0.132 | -0.004669 | 0.000839 | 5.57 |

**Posterior-mean parity is rejected at the requested Monte Carlo-error standard.** The
published-minus-current gaps are 3.6–36.7 current MCSEs. Independent current-code retries
also reproduce one another much more closely than they reproduce the README: for example,
the first and selected third `m04` attempts differ by only 0.000019 in home advantage and
0.000105 in distance weight, versus README gaps of 0.003946 and 0.018614. Because the
historical real chains and their exact data snapshot are unavailable, this establishes a
non-parity finding but does **not** by itself attribute the difference to ReverseDiff rather
than historical artifact/data/code provenance.

### Evidence locations on `mcmc-beast`

- Full first-grid log: `experiments/scottish_lower/01_poisson_2426_grid/results/reversediff_rerun_2026-09-10.log`
- Convergence retry logs: `.../reversediff_retry2_2026-09-10.log`, `.../reversediff_retry3_2026-09-10.log`
- Exact PostgreSQL/local round-trip and parity report: `.../reversediff_parity_verification_2026-09-10.log`

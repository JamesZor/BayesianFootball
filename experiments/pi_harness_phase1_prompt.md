# Work package — Experiment harness, Phase 1: scorer, cohorts, register, leaderboard (TODO 030)

You are the implementer. Claude is the manager: it will review your work against the
acceptance criteria below before Phase 2 is released. **Do Phase 1 only, then stop.**

## 0. Read first (in this order)

1. `AGENTS.md` (the whole thing), then `docs/guides/julia_coding_context_for_agents.md`.
2. **The plan:** `docs/architecture/rfc_experiment_harness.md`, especially §1–3, §5, §6 Phase 1,
   §7 and §8 (decisions are final).
3. Why this exists: `experiments/scottish_lower/REVIEW.md`.
4. The code you are graduating (read it fully):
   - `current_development/grw_player_hybrid/l01_loader.jl` (`gph_*` runner helpers; Phase 1 only needs `gph_run_by_name` / load helpers)
   - `current_development/grw_player_hybrid/l02_evaluation.jl` (Betfair close, panel restriction, `gph_scores`, `gph_observation_frame`, `gph_paired_bootstrap`)
   - `current_development/grw_pyramid_cups/r04_evaluate.jl` (slopes, the old broad transition rule)
   - `experiments/scottish_lower/compare_scottish_experiments.jl` (re-scoring saved runs from the DB)
   - `experiments/scottish_lower/06_joint_player_lineup_fusion/l66_hierarchical_kappa_eval_loader.jl` §0 (the old-artifact shim, needed to load pre-2026-09-03 joint-observation fits such as m12_td)
   - `src/Calibration/rate_pool.jl` (`invert_market_rates`, `inversion_frame`)
   - `src/evaluation/metrics.jl` (`evaluate_predictions`, `build_evaluation_context`, `evaluation_rows`, `calc_logloss`)
   - `src/training/inference/db_storage.jl` (`PostgresStorage`, `ensure_schema!`, `load_fit`, connection helpers)

## 1. Environment and rules

- **Worktree:** `/home/james/bet_project/.worktrees/BayesianFootball-experiment-harness`,
  branch `feat/experiment-harness`. Work only here.
- **Git:** commit on `feat/experiment-harness` in small logical commits (conventional
  prefixes, as in `git log`). You may `git push origin feat/experiment-harness`, which is
  needed so the beast can check out an exact SHA. **Never** push another branch, force-push,
  merge, rebase onto main, or touch the git stash.
- **Julia locally (archpc):** `julia --project -t 8` for unit tests and light work. No MCMC
  anywhere in Phase 1. Do **not** run `Pkg.update` or add packages: `Distributions` is
  pinned at 0.25.126 (0.25.127 breaks DistributionsAD). Everything you need (CSV,
  DataFrames, LibPQ, SHA, Optim, SpecialFunctions, JSON3) is already a dependency.
- **mcmc-beast** (for loading and scoring the large saved fits): `ssh root@mcmc-beast` works
  non-interactively.
  - Create a checkout at your pushed SHA: `git -C /root/BayesianFootball fetch origin && git -C /root/BayesianFootball worktree add /root/BF_runs/<sha> <sha>`.
  - Link `.env` from `/root/BayesianFootball/.env` and copy `.cache/datastore_ScottishLower.jls` and `.cache/datastore_ScottishAll.jls` from `/root/BayesianFootball-scotland-cross-tier/.cache/` if present (otherwise let it load from SQL).
  - Run inside your **own** tmux session on the beast (`tmux new -d -s harness_p1 ...`) with `JULIA_PKG_PRECOMPILE_AUTO=0 julia --project -t 16`. Do **not** attach to, type into, or kill any existing session (in particular `julia`, which hosts a Kaimon REPL).
- **Databases:**
  - Read runs from `mcmc_experiments` by UUID.
  - Create only the **new additive tables** `harness_scores`, `harness_experiments` (and, if you need it now, `harness_checks`), with `CREATE TABLE IF NOT EXISTS`, and write only to them.
  - Never `ALTER`/`DROP`/`UPDATE`/`DELETE` existing tables. `betdb` is read-only.
- **Credentials:** never print, log or commit a password or a credential-bearing URL.
  `PostgresStorage`'s masked `show` is fine; `storage.conn_str` is not.
- **Honesty:** if an acceptance number doesn't reproduce, say so with the numbers, find out
  why, and record it. Don't loosen a tolerance or pick a subset to make it pass.

## 2. What to build

New module `BayesianFootball.Harness` in `src/harness/`, included in `src/BayesianFootball.jl`
**after** the `MatchDay` include (it needs Evaluation, Portfolio, Calibration). Export
only what's listed here.

| File | Contents |
|---|---|
| `src/harness/harness-module.jl` | module, includes, exports `score_runs`, `score_fits`, `leaderboard`, `club_season_tiers`, `transition_cohort`, `RunRef` |
| `src/harness/cohorts.jl` | `club_season_tiers(ds_leagues)` → Dict((club, season) => tournament_id) from league rows of 54–57. `transition_cohort(ds, panel_ids, tiers; first_n ∈ (10, 20), direction)` → Set of match_ids where a club is within its first `first_n` matches in a tier different from its previous season's. Directions: `:relegated_into_L1` (55→56), `:promoted_into_L1` (57→56), `:l1_l2` (56↔57), `:any`. Also returns which club transitioned (for bias). `favourites(odds; threshold = 0.60)` |
| `src/harness/scoring.jl` | Port `gph_betfair_closing_odds`, `gph_season_panel`, `gph_restrict`, `gph_family`, `gph_family_selections`, `gph_scores`, `gph_observation_frame`, `gph_paired_bootstrap` (logic unchanged). `compression_slopes(latents, odds, ids)`: market log-rate supremacy from `Calibration.invert_market_rates` on the 1X2 close → `compression_slope` (market-on-model, cov(mkt, model)/var(model)) and `model_on_market_slope`. `transition_bias_pp` = mean(model − close) P(win) of the transitioning club, in percentage points. `score_fits(fits::Vector{Pair{RunRef,Fit}}; ds, tiers, control)` → long DataFrame with columns exactly as RFC §5. `score_runs(refs; ...)` loads each run by UUID, scores and frees it (one at a time) |
| `src/harness/store.jl` | `ensure_harness_schema!(db)`, `write_scores!(db, df)`, `read_scores(db; ...)`, `write_experiment!(db, row)`, `read_experiments(db)`. Keyed on `run_id` + `scorecard_version` (`"v1"`); re-scoring replaces rows for the same key |
| `scripts/score_runs.jl` | `julia --project -t 16 scripts/score_runs.jl <runs.csv>`; CSV columns `label,experiment,run_id,role`; applies the l66 shim when needed; writes to `harness_scores` |
| `scripts/leaderboard.jl` | reads `harness_scores` + `harness_experiments` → `experiments/LEADERBOARD.csv`, `experiments/LEADERBOARD.md` (headline: target LogLoss all/1X2, ECE, compression_slope, transition bias and cohort LogLoss, Δ vs control [CI], run UUID) and `experiments/REGISTER.md` |
| `experiments/scotland/W0_runs.csv` | the W0 backfill list (below) |
| `experiments/register_seed.csv` | one row per past suite/task: id, date, todo, question, dimension (likelihood / dynamics / covariate / lineup / data-scope / portfolio / EDA), status, decision, run_ids, readme. Source: `experiments/scottish_lower/REVIEW.md` §1 table plus the suite READMEs; loaded by a small `scripts/seed_register.jl` |
| `test/harness_scoring_tests.jl` | **no DB, no network:** bootstrap on a synthetic frame with known Δ (CI covers it; fixture clustering widens the CI vs row resampling); cohort labelling on a hand-built 3-season, 4-tier toy league (each direction, first_n boundary); slope convention on synthetic supremacy (compressed model ⇒ `compression_slope` > 1); per-subset row counts. Add it to `test/run_parallel_tests.jl` |

**Subsets** (RFC §5): `target` (56/57 fixtures in 24/25 + 25/26), `t56`, `t57`,
`transition_{relegated_into_L1,promoted_into_L1,l1_l2,any}_{first10,first20}`,
`favourites`. Markets: `all`, `1X2`, `OU2.5`, `BTTS`. The reference is the de-vigged Betfair
TWA(−20, 0] close, recorded in a `reference` column. Monitor subsets (54/55) come in
Phase 2; leave the column design able to hold them.

**W0 runs** (`experiments/scotland/W0_runs.csv`):

| label | experiment | run_id | role |
|---|---|---|---|
| m12_td | scottish_lower_joint_player_2426 | 132df5c2-c742-4e95-8693-3aeb2b2cbaef | control |
| m05_joint_td | scottish_lower_joint_player_2426 | ed541a7c-01e2-447e-a771-783517728d47 | candidate |
| m13_joint_td | scottish_lower_joint_player_2426 | 5474e824-8c9d-4613-8e39-841426c3f80f | candidate |
| m12_grw | scottish_lower_grw_player_hybrid | 3a9a4c7e-378b-45d0-a2d2-c8b69b46786b | candidate |
| m05_joint_grw | scottish_lower_multiscale_grw_2426 | f870dbb7-9df0-4dae-a84a-cf570cf8113e | candidate |
| m00_baseline_grw | scottish_lower_multiscale_grw_2426 | f64a00a2-34a0-4f31-8c58-c093c92d54b7 | candidate |
| g1_grw_all_spfl | scottish_pyramid_grw_cups | f00ec78a-28ca-464e-91d4-dd1af384415c | candidate |
| g2_grw_all_spfl_cups | scottish_pyramid_grw_cups | a6f62436-ec8a-461d-8bd5-dc1861a2daaa | candidate |
| g3_grw_joint_all_spfl_cups | scottish_pyramid_grw_cups | 9babf9e9-0a04-43af-855c-619a4b7dac8b | candidate |
| s10_m01_td_poisson | scottish_lower_momentum_grw | 33d85b4a-e929-4738-8125-706e0dc26de1 | candidate |
| s10_m02_grw_poisson | scottish_lower_momentum_grw | f8da493d-db2c-42d5-85e2-0f19d0107b1d | candidate |
| s10_m03_momentum_grw | scottish_lower_momentum_grw | 3e06683b-de96-431f-843d-f98619d9fc13 | candidate |
| s11_m01_td_poisson | scottish_lower_decompression | 90a8c7bc-b55c-4ef2-8e65-07cdaefa2b29 | candidate |
| s11_m02_td_joint | scottish_lower_decompression | 6d9970c8-df28-4c2a-a67d-a5741153e708 | candidate |
| s11_m03_negbin_pxg | scottish_lower_decompression | f866b2bc-d87b-42d0-8c36-8c8d623d1178 | candidate |
| s12_m01_td_poisson | scottish_lower_decoupled_xg | de7fa956-87e8-418f-afb4-61ce01cb9f7d | candidate |
| s12_m02_td_joint | scottish_lower_decoupled_xg | 97c7a3d9-a05a-4029-90cb-e34279b8c791 | candidate |
| s12_m03_funnel_shared | scottish_lower_decoupled_xg | 27d5a9f5-303a-4661-ad3d-464ba776d380 | candidate |
| s12_m04_funnel_hier | scottish_lower_decoupled_xg | b9d1627c-d7a6-4bc5-ab49-84acad0aafdd | candidate |

A run that won't load, or that covers a different panel than the 710 fixtures, is
**reported** in the Phase 1 report with the reason, not silently dropped. Suite 12's funnel
arms may need their prototype loader included (`experiments/scottish_lower/12_decoupled_generative_xg/l12_loader.jl`
/ `l15_cut.jl`). If so, do it in the script, not in `src/`.

## 3. Acceptance (Claude will check each)

1. **Reproduction.** For g1, g2, g3 and m12_td, `score_runs` reproduces
   `current_development/grw_pyramid_cups/results/evaluation/r04_proper_scores.csv`
   LogLoss / Brier / ECE for `all` and `1X2` to **1e-9**, and their `market_on_model` slope
   (`r04_compression_scorecard.csv`, subset `all`) to **±0.02**. If the `invert_market_rates`
   slope misses ±0.02, compute r04's Nelder–Mead inversion alongside, report both, and keep
   `invert_market_rates` as the v1 definition (decision RFC §8.1).
2. **W0 complete.** `harness_scores` holds rows for every W0 run that loads;
   `experiments/LEADERBOARD.csv/.md` are generated and committed, with transition-cohort
   columns populated (per direction, first 10 and 20).
3. **Register.** `harness_experiments` is seeded; `experiments/REGISTER.md` lists every past
   suite and task (01–13, 021–029, the pyramid/cups run) with question, dimension, status
   and decision.
4. **Tests.** `test/harness_scoring_tests.jl` passes standalone, and
   `test/run_parallel_tests.jl` is no worse than before (T007 is the known flaky suite;
   report exact counts).
5. **The two discrepancies from REVIEW.md §2**, investigated and reported, not fixed by
   fiat:
   - (a) Is the 01/02 vs unified disagreement a scorer difference? Answer by re-scoring one Gen 1 run, if a UUID for `scottish_lower_poisson_2426` / `m00_baseline` can be found in `runs`.
   - (b) Why does the TimeDecay Poisson control score 0.6545 in the unified report and 0.6468 in suites 10–12? Compare the runs' configs (`configs` / `config_registry`) and say what differs.
6. **`./scripts/todo.sh check`** passes. TODO 030 Work Log has dated `@pi` entries.

## 4. Reporting and stopping

- Keep TODO 030's Work Log current (dated `[2026-09-.. @pi]` lines); don't change its Status.
- At the end, write `docs/architecture/harness_phase1_report.md`:
  - each acceptance item with evidence (commands, numbers, run UUIDs, commit SHAs);
  - anything that didn't reproduce and why;
  - W0 headline findings (a short table: compression slope and transition bias by model);
  - open questions for Phase 2.
- Commit, push `feat/experiment-harness`, print exactly `PHASE1_DONE` on its own line, and
  **stop**. Don't start Phase 2.
- If you're blocked on a decision that is genuinely the human's, write it into the report
  under "Blocked", print `PHASE1_BLOCKED`, and stop.

# Experiment harness — design and implementation plan

| | |
|---|---|
| Status | **APPROVED 2026-09-26 — Phase 1 in progress** |
| Tasks | [TODO 030](../../todos/030_shared_experiment_harness_recipes_gates_runner_scorecard_leaderboard.md) (this), [TODO 031](../../todos/031_scottish_pyramid_dynamics_x_scope_x_observation_matrix.md) (first use: compression / league-change waves) |
| Date | 2026-09-26 |
| Inputs | [`experiments/scottish_lower/REVIEW.md`](../../experiments/scottish_lower/REVIEW.md), [`experiment_workflow_review.md`](experiment_workflow_review.md), [`experiment_harness_suite_catalogue.md`](experiment_harness_suite_catalogue.md) |

## 1. Goal

A work package should only have to write a **list of candidates** (model × data scope ×
sampler) and a short README. Everything else is shared library code: fold setup,
checks, sampling, persistence, scoring, the leaderboard. The shared code is assembled
from pieces that already work:

| Source | What it already does |
|---|---|
| `current_development/grw_player_hybrid/l01_loader.jl` (`gph_*`) | fold inputs, gradient audit, checkpointed sampling, recipe-hash lookup, thinning, save-and-verify, latent/coverage audits |
| `current_development/grw_player_hybrid/l02_evaluation.jl` | Betfair TWA close, restriction to panel, proper scores per market family, per-observation frame, fixture-clustered paired bootstrap, Option B portfolio |
| `current_development/grw_pyramid_cups/l01_loader.jl` (`pcx_*`) | pooled segment, cup filtering, `PyramidGRWCV` (canonical folds, widened training), `pcx_align_time!` |
| `current_development/grw_pyramid_cups/r04_evaluate.jl` | market-on-model / model-on-market slopes, transition subset |
| `experiments/scottish_lower/compare_scottish_experiments.jl` | re-scoring saved runs from the DB under one recipe |
| `src/` | `CountModelBuilder`, `FitConfig`/`fit_model`, `MAPConfig`, `QueuedExecution`, `audit_convergence`, `PostgresStorage`/`save_fit`/`load_fit`, `evaluate_predictions`, `Calibration.invert_market_rates`, `run_portfolio_simulation`, `MatchDay.option_b_system` |

Non-goals: rewriting old suites (they're frozen; only their saved runs are re-scored), an
agent loop (later — it becomes easy once candidates are data), JSON config frameworks,
dashboards before the CSV exists.

## 2. Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│ experiments/<segment>/NN_topic/candidates.jl   EXPERIMENT = "…"            │
│                                                CANDIDATES = [Candidate(…)] │
│ experiments/<segment>/NN_topic/README.md       question · decision · links │
├────────────────────────────────────────────────────────────────────────────┤
│ scripts/run_candidates.jl  <candidates.jl> --stage screen|smoke|grid       │
│ scripts/score_runs.jl      <candidates.jl | run-list.csv> [--portfolio]    │
│ scripts/leaderboard.jl     → experiments/LEADERBOARD.{csv,md}              │
│ scripts/beast_checkout.sh  <sha> → /root/BF_runs/<sha>  (git worktree)     │
╞════════════════════════════════════════════════════════════════════════════╡
│ NEW  src/harness/  → module BayesianFootball.Harness                       │
│      (included AFTER Evaluation/Portfolio/Calibration/MatchDay)            │
│   candidate.jl   Candidate, recipe_hash                                    │
│   checks.jl      hard checks (throw) · diagnostics (recorded)              │
│   stages.jl      screen · smoke · grid · resume · smoke-before-grid rule   │
│   cohorts.jl     club_season_tiers, transition_cohort, favourites          │
│   scoring.jl     score_fits/score_runs, compression slopes, bootstrap      │
│   portfolio.jl   finalist Option B / T−25 portfolio                        │
│   store.jl       harness_checks + harness_scores tables (additive schema)  │
│ NEW  src/Data/splitting/scoped.jl   DataScope, ScopedWalkForwardCV         │
│ NEW  src/Data/fetchers/segments.jl  ScottishPyramid (54–57 + 73/982/1520)  │
│ NEW  src/features/…                 create_features method for scoped CV   │
╞════════════════════════════════════════════════════════════════════════════╡
│ EXISTING src/ — unchanged                                                  │
└────────────────────────────────────────────────────────────────────────────┘
```

`src/experiments/` (the old `Experiments` module) is included *before* Evaluation,
Portfolio and Calibration, so it can't host scoring. The harness is a new module included
after `MatchDay` in `BayesianFootball.jl`.

## 3. Types

```julia
# src/Data/splitting/scoped.jl
Base.@kwdef struct DataScope
    name::String
    train_tournaments::Vector{Int}   = [56, 57]
    cups::Symbol                     = :none            # :none | :senior_spfl_ties
    target_tournaments::Vector{Int}  = [56, 57]         # held out, scored, selected on
    monitor_tournaments::Vector{Int} = Int[]            # held out, scored, never selected on
    clock_tournaments::Vector{Int}   = [56, 57]         # fold calendar
    target_seasons::Vector{String}   = ["24/25", "25/26"]
    history_seasons::Int             = 2
    dynamics_col::Symbol             = :match_biweek
end

struct ScopedWalkForwardCV <: AbstractSplitter
    scope::DataScope
end
# create_id_boundaries: canonical GroupedCVConfig folds on clock_tournaments, then widen
#   training ids with train_tournaments + scoped cup rows strictly before each fold cutoff
#   (generalises PyramidGRWCV). get_next_matches: target + monitor rows of the next bin.
# Features.create_features(splits, ds, model, ::ScopedWalkForwardCV): the pcx_align_time!
#   clock (history by season, target by clock-tournament biweek, cross-checked).

# src/harness/candidate.jl
Base.@kwdef struct Candidate
    name::String
    model                                   # CountModelBuilder output
    scope::DataScope
    sampler = QueuedNUTSConfig(n_samples = 1000, n_warmup = 500, n_chains = 4, accept_rate = 0.65)
    role::Symbol = :candidate               # :control | :candidate
    hypothesis::String = ""
end
recipe_hash(c) = sha256(string(c.model), string(c.scope))   # sampler/folds excluded:
                                                            # smoke and grid share it
```

## 4. Stages and checks

| Stage | Folds / sampler | Hard checks (throw) | Recorded diagnostics |
|---|---|---|---|
| `screen` | all folds, `MAPConfig` | filtration, latents finite | scores via `score_fits` (not persisted as runs) |
| `smoke` | 2 folds, 2×200 | gradient (compiled = fresh RD = FD, perturbed), filtration + scope asserts, latents, grid coherence, `save_fit`/`load_fit` parity | convergence summary → `review` flag; tape bytes/latency; grid tail mass at K = 12 |
| `grid` | all folds, candidate sampler | refuses without a passing smoke for `recipe_hash`; filtration, coverage (40 / 710 + monitor), latents, round-trip | convergence → `review` flag; git SHA + dirty flag |
| `portfolio` (finalists) | reuses grid run | ledger reload parity | Option B / T−25 metrics |

- Hard checks come from `gph_gradient_audit`, `gph_filtration_report`, `gph_latent_audit`,
  `gph_assert_coverage` and `gph_save_and_verify`, unchanged in logic.
- Convergence uses `audit_convergence` + `ConvergenceThresholds` (src defaults). A failure
  is recorded as `review` and the run is still saved and scored, marked on the leaderboard.
- Resume: `grid` reuses a completed run with the same `config_hash` (`gph_completed_run`)
  and checkpoints per fold.
- Persistence: full draws by default (binary bytea, PR #42); `persist_stride` kept as an
  option.
- Records go to two additive tables in `mcmc_experiments`:
  `harness_checks(run_id, recipe_hash, stage, check, severity, status, value, note, at)` and
  `harness_scores` (§5). They're created by `ensure_schema!`, with no change to existing tables.

## 5. Scoring

`score_runs(run_ids; ds, controls)` loads each run by UUID, then
`score_fits(fits; ds, controls)` does the rest (the same code path `screen` uses in memory).
One long table:

```
run_id · model · stage · scorecard_version · subset · market · metric · value · lo · hi · n_obs · n_fixtures
```

| Dimension | Values (v1) |
|---|---|
| subset | `target` (56/57, 710), `t56`, `t57`, `monitor_t54`, `monitor_t55`, `transition_relegated_into_L1`, `transition_promoted_into_L1`, `transition_L1_L2`, each at `first_10` / `first_20` matches, `favourites` (close ≥ 0.60) |
| market | `all`, `1X2`, `OU2.5`, `BTTS` |
| metric | `logloss`, `brier`, `ece`, `mce`, `rps`, the same for the market; `compression_slope` (market-on-model), `model_on_market_slope`; `transition_bias_pp` (model − close, P(win) of the transitioning club); `delta_logloss_vs_control` with fixture-clustered 95% CI |
| reference | de-vigged Betfair TWA(−20, 0] close (`gph_betfair_closing_odds`), stored per row |

- **Cohorts** (`cohorts.jl`): `club_season_tiers(ds)` from league rows (point-in-time tier
  per club-season), shared with the cup policy. A transition fixture is one inside a club's
  first N matches in a new tier, labelled by direction. This replaces r04's broad
  "tier differs" rule.
- **Slopes:** the market's log-rate supremacy comes from `Calibration.invert_market_rates`
  on the 1X2 close. Checked against r04's Nelder–Mead inversion in phase 1 (see risks).
- `scripts/leaderboard.jl` writes `experiments/LEADERBOARD.csv` (one row per run ×
  headline metrics) and `LEADERBOARD.md`. READMEs cite leaderboard rows, not frozen numbers.

## 6. Implementation plan

Phases are ordered so each delivers something usable on its own. **Phase 1 alone gives W0**
(re-scoring every existing run on compression and transition metrics).

### Phase 1 — scorer, cohorts, leaderboard (W0)

| Step | Work |
|---|---|
| 1.1 | `src/harness/harness-module.jl` skeleton; include after `MatchDay`; export `score_runs`, `score_fits`, `leaderboard` |
| 1.2 | `cohorts.jl`: `club_season_tiers`, `transition_cohort(ds, panel; first_n, direction)`, `favourites`. Needs a league-rows store: use `Data.ScottishAll()` (already on this branch) |
| 1.3 | `scoring.jl`: port `gph_betfair_closing_odds`, `gph_restrict`/`gph_season_panel`, `gph_family*`, `gph_scores`, `gph_observation_frame`, `gph_paired_bootstrap`; add per-tournament and cohort subsets; slopes via `invert_market_rates` |
| 1.4 | `store.jl`: `harness_scores` table + write/read |
| 1.5 | `scripts/score_runs.jl` (accepts a CSV of `label,experiment,run_id,role`), `scripts/leaderboard.jl` |
| 1.6 | Backfill list `experiments/scotland/W0_runs.csv`: m12_td `132df5c2`, m12_grw `3a9a4c7e`, m05_joint_grw `f870dbb7`, m00_baseline_grw `f64a00a2`, g1 `f00ec78a`, g2 `a6f62436`, g3 `9babf9e9`, the suite 10/11/12 controls, Exp 06 m05/m12/m13 |
| 1.7 | **Experiment register:** `harness_experiments` table (id, date, todo, question, scope/dimension, status, decision, run_ids, readme) + `experiments/REGISTER.md` generated by `leaderboard.jl`. Seeded with one row per past suite from `experiments/scottish_lower/REVIEW.md` §1, so "what did we try and what happened" lives in one place |
| 1.8 | Tests `test/harness_scoring_tests.jl`: bootstrap on a synthetic frame (known Δ, CI covers it); cohort labelling on a hand-built 3-season toy league; slope sign and convention on synthetic supremacy; per-subset row counts |

**Acceptance:**
1. For g1/g2/g3/m12_td, `score_runs` reproduces `grw_pyramid_cups/results/evaluation/r04_proper_scores.csv` LogLoss/Brier/ECE to 1e-9 (same code path) and the market-on-model slopes to ±0.02.
2. `LEADERBOARD.csv` exists for all W0 runs, with the transition-cohort columns populated; `REGISTER.md` lists every past suite with its decision.
3. The new test file passes in `run_parallel_tests.jl`.

**Where it runs:** scoring, not sampling, so no pinning. It can run on the beast next to
nothing else; pooled fits are large to load.

### Phase 2 — data scope

| Step | Work |
|---|---|
| 2.1 | `Data.ScottishPyramid()` segment = 54, 55, 56, 57, 73, 982, 1520; cache via `load_datastore_cached` |
| 2.2 | `apply_scope(ds, scope)`: cup season relabel (`pcx_fs_label`), `:senior_spfl_ties` via `club_season_tiers` (both clubs in an SPFL league that season; B/U21/guest excluded) |
| 2.3 | `ScopedWalkForwardCV` + `create_id_boundaries` / `get_next_matches` (target + monitor) |
| 2.4 | `Features.create_features` method for the scoped splitter (port `pcx_align_time!` with its clock cross-check) |
| 2.5 | Tests `test/scoped_splitter_tests.jl`: lower scope identical to `gph_splitter` boundaries; strict cutoff and no overlap on every fold; monitor rows only in held-out sets; clock cross-check |

**Acceptance:**
1. The `lower` scope reproduces the canonical 40 folds / 710 fixtures exactly.
2. The `spfl_cups` scope reproduces `r02_widening_g2_grw_all_spfl_cups.csv` per fold (n_train, n_upper, n_cup), or lists every differing tie with its reason. This is how the rule-based cup policy replaces the 363-row allowlist.

### Phase 3 — runner

| Step | Work |
|---|---|
| 3.1 | `candidate.jl`: `Candidate`, `recipe_hash`, `fit_config(c; stage)` |
| 3.2 | `checks.jl`: port the `gph_*` audits; hard ones throw `HarnessCheckError`; diagnostics return rows for `harness_checks` |
| 3.3 | `stages.jl`: `screen` (MAPConfig), `smoke`, `grid` (resume, checkpoints, smoke-before-grid), recording to `harness_checks` |
| 3.4 | `scripts/run_candidates.jl`: loads a candidates file, runs a stage for all or `--only name`, prints one summary line per candidate, refuses `grid` without `-t ≥ 8` pinned |
| 3.5 | `scripts/beast_checkout.sh <sha>`: `git worktree add /root/BF_runs/<sha> <sha>` + `.env` link, so every run has a real SHA |
| 3.6 | Tests `test/harness_runner_tests.jl`: stage logic on a tiny synthetic league (MAP + 1-fold 2×20 NUTS). Smoke-before-grid refusal; severity routing (a forced convergence failure is saved and flagged, not thrown); recipe_hash stability |

**Acceptance:**
1. On the beast, `smoke` then `grid` of the g2 recipe as a `Candidate` reproduces g2's recorded LogLoss (paired bootstrap CI vs `a6f62436` covers 0). About 80 min of beast time.
2. `screen` for the 12 W1 candidates completes in minutes and writes scores.

### Phase 4 — finalist portfolio and docs

| Step | Work |
|---|---|
| 4.1 | `portfolio.jl`: port `gph_option_b`, `gph_buildable_panel`, `gph_simulate`, `gph_portfolio_row`; optional T−25 calibrated container (`calibrate_fit`) |
| 4.2 | `docs/guides/experiment_harness_guide.md` (how to write `candidates.jl`, run stages, read the leaderboard); link from AGENTS.md §1 and §5; experiment README template |
| 4.3 | Fix AGENTS.md §6 / `model_generations_guide.md` to cite re-scored numbers |

**Acceptance:** Option B portfolio for m12_td via the harness matches the Task 013 r05
result for that run.

### Then TODO 031

W1 is one `candidates.jl` (12 candidates) → `screen` → `smoke` → `grid` over two nights.
W2–W4 follow on the winner.

## 7. Risks and how the plan handles them

| Risk | Handling |
|---|---|
| **Old artifacts won't deserialize.** Pre-2026-09-03 joint-observation fits (incl. live m12_td `132df5c2`) hold a 3-param `JointGammaPoissonObservation` | Phase 1 includes the shim from Exp 06 `l66` §0 in the backfill script; acceptance requires m12_td to load |
| Slope inversion change (`invert_market_rates` vs r04's Nelder–Mead) shifts values | Compute both in phase 1; if they differ by > 0.02, keep r04's inversion as the v1 definition and log the difference |
| `config_hash` depends on `string(model)`; a harness `FitConfig` may hash differently from the original g2 run | Acceptance compares **scores**, not hashes; the harness run is a new run |
| No neutral-venue column in the fetchers (the allowlist removed Hampden semis/finals by hand) | Phase 2.2 checks `sofascore.events` raw JSON for a neutral flag; if absent, keep a small `neutral_venues.csv` (derived from the allowlist diff) as the only hand-maintained input |
| Pooled fits are large in memory when scoring many runs | `score_runs` loads, scores and frees one run at a time |
| Features dispatch for the scoped splitter lives in `Features`, which loads before `Data` extensions | Place the method in `src/features/` next to the existing splitter methods; covered by 2.5 tests |
| Monitor 54/55 fixtures need Betfair rows | `ScottishPyramid` loads `betfair_odds` for all its tournaments; phase 2 records coverage per tier (preflight: 54 at 99.7%, 55 at 99.9% processed odds) |

## 8. Decisions (human, 2026-09-26)

1. **Slope:** `Calibration.invert_market_rates` on the 1X2 close. r04's Nelder–Mead value is
   computed once in phase 1 as a cross-check, not as an alternative definition.
2. **Transition cohort:** first 10 and first 20 matches in the new tier, split by direction.
3. **Scores storage:** `harness_scores` / `harness_checks` / `harness_experiments` tables in
   `mcmc_experiments` on mcmc-beast; CSV/MD files are generated views.
4. **Branching:** `feat/scotland-cross-tier-models` goes to main via PR; the harness is built
   on `feat/experiment-harness` in its own worktree, branched from that tip and retargeted
   to main once the PR merges.
5. **Execution:** a pi agent (`openai-codex/gpt-5.6-sol`, thinking high) implements phase
   by phase in tmux; Claude manages, reviews each phase against its acceptance criteria and
   reports.

# Experiment workflow review — Scottish suites, Aug–Sep 2026

| | |
|---|---|
| Date | 2026-09-26 |
| Scope | `experiments/scottish_lower/01–13`, `experiments/scotland/01–02`, `current_development/{grw_player_hybrid, grw_pyramid_cups, market_inverse_dynamics, fast_slow_grw, hierarchical_home_advantage}`, TODOs 005–029, branch history |
| Companion | [`experiment_harness_suite_catalogue.md`](experiment_harness_suite_catalogue.md) (per-row code inventory), [`rfc_experiment_harness.md`](rfc_experiment_harness.md) (the proposed fix) |

**In one sentence:** the modelling stack in `src/` is good and mostly finished. What is
missing is a thin, shared **experiment layer** between it and each work package. Without
one, every package rebuilds that layer, mixes EDA, new model types and benchmarking in one
folder, and ends on an unmerged branch.

---

## 1. What we did — the suites at a glance

| Suite / task | Question | Outcome | Harness it used |
|---|---|---|---|
| 01–03 | Poisson / NegBin / joint Gamma-Poisson baselines | Joint two-arm is the standard (LogLoss 0.6571) | early unified runners |
| 05–06 | Player-lineup RAPM fusion | Buys calibration (ECE), not LogLoss; m12 goes live | own loaders (`l50`, `l60`) |
| 07 | Calibrated portfolio / trust vector at T−25 | OU1.5 survives OOS; per-line pruning doesn't | own (`l07`, 975 lines) |
| 08 | Goal decomposition | Smoke passed; **grid never run** (TODO 002 still open) | own (`l08_workflow`) |
| 008 | Contextual home advantage | No gain; prototype, not merged | own |
| 013 | GRW + lineup hybrid | GRW and pxG are substitutes; don't promote | **`grw_player_hybrid` (`gph_*`)** — born here |
| 021 | Fast/slow GRW mixtures | Don't decompress | own |
| 022 → suite 10 | Momentum GRW | Marginal; not promoted | `gph_*` |
| 023 | Market-inverse state space | Phase 2 attribution; research only | own |
| 024 → suite 11 | NegBin + pxG covariate | Decompresses 1.72 → 1.30, loses LogLoss and money | `gph_*` |
| 025 → suite 12 | Decoupled xG funnel | **Rejected** | `gph_*` + own `l15_cut` (1,123 lines) |
| 027 → suite 13 | Pedigree / full-time EDA | BLOCKED; work-package premises corrected | EDA scripts (Python + Julia) |
| 028 → scotland/01 | Time-decay cross-tier + cold-start priors | BLOCKED on performance gates; r07 grid crashed after sampling m00 | **own again** (`l02`+`l04`+`l05`, ~700 lines) |
| 029 → scotland/02 | Cup + pyramid EDA | Completed; tier steps ~0.47, relegated prior +0.12 | EDA scripts |
| `grw_pyramid_cups` | GRW on SPFL + cups (g1–g3) | Cups help 1X2; g3 least compressed (slope 1.10) at tied LogLoss | `gph_*` |

Six decompression attempts (021–025, 028), none promoted. The two levers that did show
up, **GRW dynamics** (Task 007, suite 10) and **cup bridges** (`grw_pyramid_cups`), came
from work that reused a harness rather than rebuilding one.

## 2. What works — keep it, build on it

1. **The `src/` modelling stack.** `CountModelBuilder` composition, `FitConfig`/`fit_model`,
   `QueuedExecution`, typed `CountLatents`, score-grid kernels, `evaluate_predictions`,
   `calibrate_fit`, `run_portfolio_simulation`. The work packages rarely needed to touch
   these; when they did, graduation went cleanly (MultiScaleGRW, joint observation,
   generative calibration).
2. **`PostgresStorage` + `config_hash` + `audit_convergence`.** Resume-by-recipe and the
   six-part convergence summary (with worst-parameter names and abstention) are exactly the
   right primitives.
3. **The canonical comparability contract.** 40 folds / 710 fixtures / 2,899 scored
   selections, the Betfair close as reference, paired fixture-clustered bootstrap. Every
   suite that honoured it produced numbers that can be compared.
4. **`current_development/grw_player_hybrid/l01_loader.jl` + `l02_evaluation.jl` (`gph_*`,
   1,154 lines) is already a working proto-harness.** Four later packages reused it (suites 10–12, `grw_pyramid_cups`):
   gradient audit, fold inputs, checkpointed sampling, completed-run lookup by hash,
   thinning, save-and-verify, coverage and latent audits, scores, paired bootstrap,
   Betfair close, T−25 Option B portfolio, edge summary. **This is the seed to graduate**,
   not a blank-page design.
5. **Honest READMEs with run UUIDs and decisions.** Suites 10–13 state a decision, the
   denominators and the CI. That is the right record format.

## 3. What doesn't work

### 3.1 The experiment layer is rebuilt every time
- 028 wrote about 700 lines of smoke/production/pilot helpers that duplicate `gph_*`. 08,
  07, 021 and 023 each have their own too. The agent budget goes on plumbing, not on the
  question.
- **Untested runners run overnight.** 028's `r07` was written at 23:49 and launched at
  00:48. It sampled m00 on all 40 folds, then died on `fit.folds[i].diagnostics` (not a
  field of `FoldFit`), and the other four candidates never ran. A shared, tested runner
  removes this class of failure.

### 3.2 Gates that block on performance, not correctness
- 028 required **zero bytes** per gradient and a per-draw 12×12 grid mass ≥ 0.999. Both
  measure performance or truncation, not correctness, and both blocked the package for
  two days. Meanwhile `grw_pyramid_cups` ran the same model class without them and
  produced the result that mattered.
- Thresholds also differ between `src/` and the suites. `ConvergenceThresholds` defaults
  to `min_ess = 400` and `max_divergence_rate = 0.001`, while 028 used ESS > 100 and a
  divergence rate < eps(). The same fit can pass in one place and fail in another.

### 3.3 Results can't be compared across suites
- **Two slope conventions** are in use (market-on-model in 11/12/023; model-on-market in
  suite 10), and both get called "supremacy slope".
- **Different reference instants**: close, TWA(−20,0], T−25.
- **Different panels** for the same headline: 710 fixtures vs 627 scored vs 622 tradeable
  vs 18 favourites.
- **LogLoss is selection-level binary cross-entropy**, including on 1X2. Correct, but not
  what a reader assumes, and not stated everywhere.
- **No leaderboard.** `unified_paradigm_comparison.csv` stops at suite 06.

### 3.4 EDA, new model types and benchmarking are tangled in one folder
Each package does three different jobs at once:
- **(a) EDA**: SQL, descriptive statistics, GLMs, no MCMC.
- **(b) New model code**: a component, its AD safety and unit tests.
- **(c) Benchmarking**: smoke → grid → score → portfolio.

They need different evidence and different review. Mixed together:
- work-package prompts carry EDA hypotheses in as facts (027 found the "54% ceiling" false;
  029 found the +0.90 relegated prior should be +0.12, but 028 had already built B1 on +0.90);
- new types live as prototypes that later suites `include` across folders. `grw_pyramid_cups`
  includes `../grw_player_hybrid/l01_loader.jl`, and suites 10–12 include it by relative
  path, so a prototype has become a dependency;
- a benchmark can't start until its bespoke EDA and bespoke components are finished.

### 3.5 Work ends on unmerged, stacked branches
- `main` holds suites 01–08 only. Suites 10–13, 027, 028, 029 and `grw_pyramid_cups` exist
  only on a **chain of stacked branches** (momentum → negbin → decoupled → pedigree →
  cross-tier). This branch is 50 commits ahead of main.
- Finished but unmerged elsewhere: 008 (`hierarchical-home-advantage`), 08
  (`goal-decomposition`), and three EDA branches.
- Runs execute from **rsync copies without `.git`** (`/root/BF_*`, 23 directories on the
  beast). Provenance is a hand-written source digest, and the live m12 `model_run_id` is
  NULL.

### 3.6 Data scope is hard-coded per suite
"Train on more tournaments, score on League One/Two" was implemented twice, differently:
`Data.ScottishAll` + `GroupedCVConfig` (028), and `ScottishPyramidCups` + `PyramidGRWCV` +
an allowlist CSV (`grw_pyramid_cups`). Cups have no Data-layer representation, and nothing
scores the upper tiers against the market.

## 4. What to abstract — ranked by payoff

| # | Abstraction | Replaces | Built from |
|---|---|---|---|
| 1 | **Graduate `gph_*` into `src/harness/`** as the standard smoke/grid runner: fold inputs, gradient audit, checkpointed sampling, resume by hash, save-and-verify, coverage/latent audits | ~300–1,100 lines per suite | `grw_player_hybrid/l01` |
| 2 | **Standard scorer + scorecard spec** (JSON: stage × metric × market × subset), one slope convention, the reference instant as a field, fixed denominators reported | `l12/l13/l14_evaluation`, `r04_evaluate`, `gph_scores` | `grw_player_hybrid/l02` + `src/evaluation` |
| 3 | **Data scope + `ScopedWalkForwardCV`**: train tournaments, cups policy, target tournaments, monitor tournaments; canonical folds asserted | `ScottishAll`/`GroupedCVConfig`, `PyramidGRWCV`, allowlist CSV | `grw_pyramid_cups/l01` |
| 4 | **Gate severities** (hard / review / warn), one threshold set, reclassify tape bytes and grid tail as warnings | 028 `l02`, 08 `l08_model_checks` | `src/.../convergence.jl` |
| 5 | **Leaderboard** (`scripts/leaderboard.jl` → CSV/MD + dashboard), backfilled | `unified_paradigm_comparison.csv` | `mcmc_experiments` |
| 6 | **Recipes as data** + component registry, so a candidate is a JSON file, not a script | per-suite `*_models()` / `*_arms()` functions | `gph_models`, `pcx_model` |

Items 1–4 are mostly **moving and naming code that already works**. Item 6 is the step
towards the FunSearch-style loop.

## 5. What's missing

- A **three-lane convention**:
  - **EDA** (`eda/<topic>/`, no MCMC, outputs numbers that become priors or hypotheses);
  - **components** (a PR into `src/models` via the extension recipe, with AD and unit
    tests, no grids);
  - **experiments** (recipes on the harness, rows on the leaderboard).
  A work package names its lane; a package that needs all three becomes three tasks.
- A **merge cadence**: merge each completed package to main before the next one branches,
  so branches stop stacking. Consolidate the current chain now (this branch contains it).
- **Runs from git.** The beast checks out a commit (a worktree per run), not an rsync copy,
  so every run UUID maps to a SHA.
- A **per-tier market monitor** (54/55 against the Betfair close) for pooled fits.
- A **"fit on everything to date" path for live** (partly done by the l17/r17 card
  injection; not yet a harness stage).
- A **smoke-before-overnight rule** enforced by the runner itself: the grid stage refuses
  to start unless a smoke record exists for the same recipe hash.

## 6. Suggested order

1. **Consolidate branches.** Merge this branch (which already contains 10–13, 027–029 and
   `grw_pyramid_cups`) to main via one PR, then decide on 008 / 08 / the EDA branches.
2. **Graduate `gph_*` + scope + scorer + gate severities** (RFC items 1–4) with the g1/g2/g3
   reproduction as the acceptance test.
3. **Run the 12-arm matrix** as the first recipe list (TODO 031), with the leaderboard on.
4. **Recipes + agent loop** once the matrix has exercised the harness.

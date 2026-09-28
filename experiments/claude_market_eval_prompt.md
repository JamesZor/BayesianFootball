# Work package — TODO 038: market evaluation layer (typed axes, pluggable evaluators, unified CLV)

You are the **builder** (Claude CLI, Opus 5.5). Claude, the manager in another session, reviews
your work, and a separate Claude CLI session does the formal review.
- **No subagents.** Do not merge.
- The design below was agreed with the human in a grill-me session on 2026-09-28. **Do not re-open
  it.** If something in it is impossible, stop and report (§8).

## 0. Why this exists

The project's question is whether the models have an **exploitable edge against the market** at a
time we could actually bet. Being more accurate than the market is not required. The harness
cannot answer that today:
- `src/harness/portfolio.jl` hard-codes two monolithic containers (`close_option_b`,
  `t25_calibrated`);
- it reports **no CLV**;
- the two existing CLV functions disagree on the definition.

This package builds a small, typed evaluation layer. Every axis is swappable through Julia
multiple dispatch, and every measurement is a pluggable evaluator.

## 1. Rules

- **Worktree** `/home/james/bet_project/.worktrees/BayesianFootball-market-eval`, branch
  **`feat/market-eval`**, cut from the `feat/w2-tier-components` commit that merges TODO 036 (the
  manager names it).
  - Commit in small logical steps, and push this branch only.
- **The laptop runs no Julia.** All Julia runs on mcmc-beast (`ssh root@mcmc-beast`) in your own
  tmux session there, named `claude_market_eval`.
  - **Iterating:** rsync to `/root/BF_runs/market_dev/` with
    `--exclude '/.cache/' --exclude '/data/' --exclude '/.git/'`. Symlink `.env` and
    `Manifest.toml` from `/root/BayesianFootball/`, and copy
    `/root/BF_runs/a76a65df/.cache/datastore_Scottish*.jls`.
  - **Final evidence:** from a clean `bash /root/BF_runs/8effca72/scripts/beast_checkout.sh <sha>`
    checkout of a pushed SHA, with the caches copied in.
  - Julia is `/root/.juliaup/bin/julia --project -t 16`, after `set -a; source .env; set +a`.
  - Logs go in `/root/BF_runs/logs/market_eval/`.
  - Check `pgrep -af julia` first, and run one heavy job at a time.
- **No sampling.** Everything works from persisted fits (`load_fit`) and the datastore.
- **Databases.**
  - `mcmc_experiments` (production) is **read-only** until §7.
  - All tests and dev runs use `mcmc_experiments_test` (`BF_EXPERIMENTS_TEST_DB_URL`, guarded with
    `scripts/klm_test_db_guard.jl`).
  - `betdb` is read-only until §7.
  - Never print credentials.
- **Style:** read `docs/guides/julia_coding_context_for_agents.md` first. Match the idiom of
  `src/Portfolio/`, which is already dispatch-based (`AbstractPricePolicy`, `AbstractShrinkage`,
  `AbstractTrustModel`, `BookSpec{…}`, `PolicySpec{…}`, `PortfolioSystem{B,P}`). **No Symbol or
  `if`-chain switches on axis values. Behaviour is chosen by dispatch.** Type-stable hot loops;
  prefer concrete parametric fields.

## 2. Read first

1. `src/harness/portfolio.jl` (all of it): the two containers, the panel logic, the bootstrap and
   the metric rows. Also `src/harness/store.jl` and `src/harness/scoring.jl` (`RunRef`,
   `score_runs`, the long-row format).
2. `src/Portfolio/`: `types.jl`, `interfaces.jl`, `simulate.jl`, `slates.jl`, `stake.jl`,
   `attribution.jl` (`edge_summary`, `capture_ratio`), `metrics.jl`, `book.jl`, `pricing.jl`.
3. `src/Calibration/`: `types.jl` (`AbstractCalibrator`, `GenerativeRateCalibrator`, the weight
   laws), `book.jl` (`point_in_time_book`, `PointInTimeBookConfig`, `closing_book`, `bet_clv`,
   `clv_summary`), `pipeline.jl` (`calibrate_fit`, `calibrated_fit`).
4. `src/MatchDay/ledger/settle.jl` (`clv_for_order`, `fill_vwap`, `lay_to_back`),
   `src/MatchDay/ledger/schema.jl` (`clv_audit`), and `MatchDay.option_b_system`.
5. `scripts/run_candidates.jl` (the `--stage portfolio` path) and `scripts/score_runs.jl`
   (`--portfolio`).
6. TODO 038 (acceptance), TODO 030 (the harness), and TODO 031 plus the W2 README (the finalists).

## 3. The design (agreed; names may be refined, but the structure may not)

Domain types live in **`src/Portfolio/evaluation/`**, a new submodule file set included by
`portfolio-module.jl`. The Harness only orchestrates, in **`src/harness/market.jl`**.

```julia
# --- axis 1: pricing -------------------------------------------------------------
abstract type AbstractPricing end
struct RawPosterior <: AbstractPricing end
struct Calibrated{C<:Calibration.AbstractCalibrator} <: AbstractPricing
    calibrator::C
end
# price source for a fit restricted to a panel; Calibrated needs the entry book it calibrates to
pricing_source(::RawPosterior, fit, book)            # -> fit (as today)
pricing_source(p::Calibrated, fit, book)             # -> calibrated_fit(calibrate_fit(p.calibrator, fit, book))

# --- axis 2: book instant (entry price) ------------------------------------------
abstract type AbstractBookInstant end
struct MinutesBefore <: AbstractBookInstant; minutes::Float64; end   # T−25 = MinutesBefore(25.0)
struct Close <: AbstractBookInstant end                              # Betfair closing odds
entry_book(i::MinutesBefore, ds)   # point_in_time_book(ds; as_of_minutes = -i.minutes)
entry_book(::Close, ds)            # the closing odds used by close_option_b today

# --- axis 3: staking ---------------------------------------------------------------
abstract type AbstractStaking end
struct FlatStake <: AbstractStaking; unit::Float64; min_edge::Float64; end
struct SystemStaking{S<:PortfolioSystem} <: AbstractStaking; system::S; end

# --- axis 4: markets -----------------------------------------------------------------
#   Data.MarketConfig (1X2 / OU2.5 / BTTS), reused as is

struct MarketRecipe{P<:AbstractPricing, I<:AbstractBookInstant,
                    S<:AbstractStaking, M<:Data.MarketConfig}
    name::String; pricing::P; instant::I; staking::S; markets::M
end
market_recipe_hash(r::MarketRecipe)   # SHA-256 of a canonical string of the four axes (like recipe_hash)

# --- the ledger ----------------------------------------------------------------------
struct BetLedger   # one row per bet: run_id, match_id, slate, market, selection, side,
                   # odds_taken (effective, back-equivalent), p_model, p_entry_fair, edge_pp,
                   # stake, outcome/return, p_close_fair, odds_close_raw, league/tournament, lead_min
    …
end
ledger(r::MarketRecipe, fit, ds; panel)::BetLedger      # dispatches through the four axes

# --- evaluators ----------------------------------------------------------------------
abstract type AbstractMarketEvaluator end
struct CLVEvaluator  <: AbstractMarketEvaluator; buckets::Vector{Float64}; B::Int; seed::Int; end
struct EdgeQuality   <: AbstractMarketEvaluator end           # predicted vs realised edge, capture ratio
struct SlateGrowth   <: AbstractMarketEvaluator; B::Int; seed::Int; end   # log growth/slate + CI, drawdown
struct ROIRisk       <: AbstractMarketEvaluator end           # ROI, Sharpe, Calmar, P(ROI>0), win rate
evaluate(ev::AbstractMarketEvaluator, L::BetLedger, ctx)::DataFrame   # long score rows
```

Evaluator requirements:
- **Evaluators never know how the ledger was built.** `SlateGrowth` and `ROIRisk` wrap the
  existing `src/Portfolio` metrics rather than re-implementing them.
- **Every evaluator emits the harness long-row schema:** `subset`, `market`, `metric`, `value`,
  `lo`, `hi`, `n_obs`, `n_fixtures`, `reference`.
- **Every evaluator reports breakdowns by market, by league (tournament 56 / 57) and, for CLV and
  edge quality, by edge bucket.** The kill rule is not pre-registered (human decision), so the
  report must show **all** slices with a multiplicity caveat, not a selected one.

## 4. Unified CLV (one definition, used everywhere)

Put a canonical implementation in `src/Portfolio/evaluation/clv.jl`, with the close reference as
a **dispatch type**:

```julia
abstract type AbstractCloseReference end
struct DeviggedClose <: AbstractCloseReference; window::Tuple{Float64,Float64}; end  # TWA(−20,0], as the scorecard
struct RawClose      <: AbstractCloseReference end                                   # raw closing odds (legacy Calibration)
clv(ref::AbstractCloseReference, odds_taken, close…)  # -> (; clv_pp, clv_pct, beat_close)
```

The canonical headline, used by all new code:
- `p_entry = 1/odds_taken`, where `odds_taken` is the **effective**, back-equivalent price
  actually taken (VWAP of fills where fills exist);
- `p_close` = the de-vigged close, `DeviggedClose((-20.0, 0.0))`;
- `clv_pp = 100·(p_close − p_entry)`;
- `clv_pct = 100·(odds_taken·p_close − 1)`, i.e. `odds_entry/odds_close_fair − 1`;
- `beat_close = clv_pp > 0`;
- aggregates: the flat mean, the stake-weighted mean, the share beating the close, and a
  **slate-clustered** bootstrap CI.

The two existing functions become thin wrappers over the canonical code, and their outputs must
stay **bit-identical**, with parity tests on real inputs:
- `Calibration.bet_clv`/`clv_summary`: its `clv_pct` uses the raw close → `RawClose()`.
  `calibration_runs.clv_weighted_pct` depends on it.
- `MatchDay.clv_for_order`: its `clv = p_close_devigged − 1/effective_odds` and
  `clv_pct = clv/entry`. Expose that as its own named quantity, not by redefining the canonical
  `clv_pct`.

Document both legacy definitions and the canonical one in the docstring and in your report.

## 5. Two bet sets, and parity with today

- **Flat audit ledger:** `FlatStake(unit = 1.0, min_edge = e)` bets every selection with model edge
  ≥ `e` against the entry book, policy-independent. `e` is a recipe parameter, default 0.0, and
  CLV is reported by edge bucket, e.g. `[0, 2, 4, 6, Inf]` pp.
- **System ledger:** `SystemStaking(MatchDay.option_b_system(; lambda = 8.0))`, with stake-weighted
  CLV.
- **Migration parity (hard requirement):** re-express `close_option_b` as
  `MarketRecipe(RawPosterior(), Close(), SystemStaking(option_b λ=8), markets)`, and
  `t25_calibrated` as `MarketRecipe(Calibrated(<today's scot_lower_t25_inv calibrator>),
  MinutesBefore(25.0), SystemStaking(option_b λ=8), markets)`.
  - On the existing finalists (`m12` `132df5c2`, `m05` `ed541a7c`), every metric the old stage
    wrote (`total_return_pct`, `cagr_pct`, `growth_per_slate` [lo, hi], `roi_pct`,
    `p_roi_positive`, `sharpe_ann`, `calmar`, `max_drawdown_pct`, `win_rate_pct`, `mean_edge_pp`,
    `mean_exposure`, `capture_ratio`, `n_bets`, `n_slates`, `n_panel`) must reproduce
    **bit-identically** (`==`), on the same common panels (632 / 611). Compute in the test DB
    context and compare with a read-only production SELECT of the existing `stage = 'finalist'`
    rows.
  - Keep `portfolio_runs` and `--portfolio` working as wrappers over the new layer, so old
    invocations still run.

## 6. Harness integration and storage

- `src/harness/market.jl`: `market_runs(refs::AbstractVector{RunRef}, recipes::AbstractVector{<:MarketRecipe}, evaluators; ds, panel = :common, db)`.
  - It loads each fit once and evaluates every recipe on the run's panel. Panels are never mixed.
  - It writes `harness_scores` rows with `stage = "market"`, `scorecard_version = "v1.2"`, the
    panel label, `subset = "<recipe name>/<slice>"`, and an additive `market_recipe_hash` column.
- The additive schema (`ensure_harness_schema!`, `IF NOT EXISTS` only) adds a new table
  `harness_market_recipes (market_recipe_hash PK, name, spec jsonb, created_at)` holding each
  recipe's full typed spec (the axes and their parameters), so any score row can be reproduced.
- CLI: `scripts/run_market_eval.jl <runs.csv> --recipes <recipes.jl> [--test-db] [--panel common|<n>]`.
  - `recipes.jl` defines `RECIPES::Vector{MarketRecipe}` and `EVALUATORS`, the way `candidates.jl`
    does.
  - Refuse `--test-db` unless the storage is the guarded test DB.
- Leaderboard: add a warning-labelled **"Market evaluation"** section (per panel, per recipe), with
  CLV first. Never mix it with the grid Δs.

## 7. Tests, first grid, production (in this order)

1. **Unit tests** (new `test/market_eval_tests.jl`; add it to the parallel runner):
   - Dispatch on each axis, including a test that a **new** axis type plugs in by adding one
     method.
   - The `market_recipe_hash` is stable, and changes when any axis parameter changes.
   - The CLV formulas on hand-worked examples: back, lay (effective odds), VWAP fills, and a
     de-vigged close.
   - Bucket and league slicing, and slate-clustered CI determinism under a fixed seed.
   - Legacy parity for `bet_clv`/`clv_summary` and `clv_for_order`.
2. **Migration parity** (§5) on the test DB, bit-identical.
3. **Existing suites green:** portfolio, unified portfolio, harness portfolio, harness scoring,
   harness runner, the calibration tests, the MatchDay ledger tests, and
   `test/run_parallel_tests.jl` (no worse than T007).
4. **Rehearse the first grid on the test DB, then stop and print `MKT_BUILD_DONE`.**
   - **Recipes:** {`RawPosterior`, `Calibrated`(today's T−25 calibrator)} × {`MinutesBefore(25.0)`,
     `Close()`} × {`FlatStake(1.0, 0.0)`, `SystemStaking(option_b λ=8)`} × all markets.
   - **Runs:**
     - W0 `m12` `132df5c2` and `m05` `ed541a7c` (632 / 611 panels);
     - the W0 floor control `s12_m01_td_poisson` `de7fa956`, a pure team-strength TD Poisson.
       Also include `m00_poisson_control` `d63f8877`, which the paper ledger ran, if it scores on
       the same panel.
     - W2 on the 1,070 panel: `td_lower_base` `0dd74a52`, `td_lower_a2full_carry_jump` `6c0aa3ea`,
       `grw_base` `ce7ea22f`, `grw_step_a2_carry_jump` `c4a0fa94`, `m12_td_ref` `973baafa`.
   - Note: with `Close()`, CLV is 0 by construction. Report that as a sanity check, not a result.
5. **Production, only after the manager says "go prod":**
   - run `ensure_harness_schema!` on production;
   - run the same grid into production `harness_scores`;
   - backfill `betdb.paper_runbook.clv_audit` for the settled paper orders through the canonical
     CLV (`DeviggedClose`, entry = fill VWAP). It must be idempotent and insert-only. Print the
     row count before and after.

   Then regenerate `LEADERBOARD.md`, commit, and print `MKT_PROD_DONE`.

## 8. Report and stop

- Write `docs/architecture/market_eval_report.md` with:
  - the design as built, including an ASCII diagram of the axes → ledger → evaluators flow and a
    type/method table;
  - the parity evidence (tables, commands, SHAs);
  - the test counts;
  - **the rehearsal results for every slice**: CLV (pp and %) with CI by recipe × market × league
    × edge bucket, next to ROI and growth, with a clear statement of what is and is not
    significant, and a multiplicity caveat;
  - the legacy CLV definitions;
  - anything not done.
- Add a dated `@claude` line to the TODO 038 Work Log. Commit, push, print the head SHA, then print
  exactly `MKT_BUILD_DONE` and stop.
- **Review loop:** the manager sends findings (`docs/architecture/market_eval_review.md`). Fix them,
  append "Fix round N", push, and print `MKT_FIX<N>_DONE`.
- If blocked: write it under "Blocked", push, print `MKT_BLOCKED` with a one-line reason, and stop.

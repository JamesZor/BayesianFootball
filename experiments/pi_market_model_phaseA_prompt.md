# Work package — TODO 039 Phase A: KL inversion of the close, and inversion quality

You are the implementer. Claude is the manager and reviews your work against §4; the human signs
off before Phase B. **Do Phase A only, then stop.** Phase A is all on the laptop: no MCMC, no
beast, no database writes.

## 0. Rules

- **Worktree:** `/home/james/bet_project/.worktrees/BayesianFootball-market-model-pi`, branch
  `pi/market-model-phase-a` (already created from the design commit). Work only there. Commit in
  small logical steps with conventional prefixes (`feat(market-model): …`, `test(market-model): …`,
  `results(market-model): …`). **Do not push.** Never merge, rebase, force-push, touch another
  branch or the git stash. The manager merges.
- **Julia:** `julia --project -t 8` from the worktree root. Do **not** run `Pkg.update` or add
  packages. `Distributions` is pinned at 0.25.126 (0.25.127 breaks DistributionsAD). Optim,
  ForwardDiff, CSV, DataFrames, StatsBase, Plots/StatsPlots and SpecialFunctions are already
  dependencies.
- **Data:** load the store with
  `Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 10^6)`. The worktree's
  `.cache/datastore_ScottishLower.jls` is a pinned copy of 2026-09-25; the large `max_age_hours`
  stops a silent refresh from SQL. Record the cache file's mtime and size in your report. `.env`
  is a symlink; never print it, `BF_DB_URL` or any credential.
- **No `src/` changes.** Everything lives in `current_development/market_model/`. TODO 023's
  loader (`current_development/market_inverse_dynamics/l01_market_inverse_loader.jl`) is
  `include`d, not edited.
- **Style:** runners follow `docs/prototype_runner_style_guide.md` (numbered sections; machinery in
  the loader).
- **Honesty:** if a number doesn't reproduce, say so with the numbers and find out why. Don't loosen
  a tolerance or pick a subset to make a check pass.

## 1. Read first

1. `current_development/market_model/DESIGN.md`: **the contract**, especially §2 (decisions), §4.1
   and §4.5 (Phase A rows). If this brief seems to disagree with it, DESIGN wins; say so.
2. `current_development/market_inverse_dynamics/README.md` §1 and
   `PHASE2_FEATURE_ATTRIBUTION.md` §5c and §6: what TODO 023 found about the inversion.
3. `current_development/market_inverse_dynamics/l01_market_inverse_loader.jl`: `closing_book`,
   `MarketPanel`, `build_market_panel`.
4. `src/Calibration/rate_pool.jl` (`market_targets`, `invert_market_rates`) and
   `src/Calibration/types.jl` (`MarketInversionConfig`, `L2_INVERSION_LINES`).
5. `src/Calibration/book.jl` (`devig_book!`, `expected_selection_count`,
   `PointInTimeBookConfig`): its completeness-before-normalisation rule.
6. `src/features/market_inverse_utils.jl` (`build_probability_matrix`, `_calculate_error`) and
   `src/features/types.jl` (`DoublePoissonMarketFeature`).
7. Tickets `docs/tickets/T014-betfair-1x2-home-away-swap.md` and
   `docs/tickets/T015-inversion-accepts-books-without-1x2.md`.

## 2. What to build

### 2.1 Loader `current_development/market_model/l01_market_model.jl` (module `MarketModel`)

**(a) The gated close book.** Start from TODO 023's `closing_book(ds)` (Betfair TWA(−20, 0], same
window), but gate before de-vigging:

- **Completeness before normalisation:** a (match, market, line) group must carry its full
  selection set (1X2: 3; OverUnder: 2; BTTS: 2; use `expected_selection_count`). `closing_book`
  today normalises whatever runners are present, so a one-sided group becomes a probability of 1.
  Count how many groups this changes in the 710-fixture panel.
- **Overround limits** (0.90, 1.10), as `PointInTimeBookConfig`.
- **Market-level refusal frame:** match, market, line, n_selections, overround, reason.

Keep only the lines of `L2_INVERSION_LINES`: 1X2, BTTS, O/U 0.5, 1.5, 2.5, 3.5.

**(b) KL inversion, double Poisson, per fixture in isolation.** Minimise over θ = (log λ_h, log λ_a):

```
L(θ) = Σ_k KL(p_k ‖ q_k(θ)) = Σ_k Σ_s p_{k,s} · log(p_{k,s} / q_{k,s}(θ))
```

- k runs over the fixture's quoted markets; p are the de-vigged probabilities; q comes from the
  double-Poisson grid with `max_goals = 10`.
- The grid must be **numerically identical** to
  `Features.build_probability_matrix(DoublePoissonMarketFeature(), θ, 10)`. That function is typed
  `Vector{Float64}`, so either call it under a derivative-free optimiser, or write a generic
  (AD-friendly) local grid and test it equal to the Features one to ≤ 1e-14.
- Optimiser: your choice, with tight tolerances. Prove on the panel that you reach the global
  minimum: re-solve every accepted fixture from at least 3 dispersed starts and report the max
  |Δθ| across starts.

**Gates, applied in this order, each refusal named:**

| Order | Gate |
|---:|---|
| 1 | no Betfair book |
| 2 | **no complete 1X2 market** (T015) |
| 3 | fewer than 3 quoted selections |
| 4 | optimiser not converged |
| 5 | summed KL > 0.01 |
| 6 | λ outside [0.05, 6] |

Drop match **14035501** (T014's swapped book) with reason "T014 swapped book".

**(c) Panel bridge.** Add `build_market_panel(ds, frame; seasons, tournaments, step_days = 7)`, a
method that builds TODO 023's `MarketPanel` from an **external** inversion frame. Phase B needs it
to run the Kalman ladder on KL rates. Test: feeding it the frame TODO 023's own
`build_market_panel(ds)` builds internally reproduces that panel **exactly** (`obs_y`, `obs_att`,
`obs_def`, `obs_week`, `week_ptr` equal).

**(d) Helpers** for the Phase A tables below. Keep them in the loader, not the runner.

### 2.2 Runner `current_development/market_model/r01_kl_inversion.jl`

Panel: tournaments 56/57, seasons 24/25 + 25/26 (710 fixtures). Outputs go under
`current_development/market_model/results/A/`; figures under `results/figures/A_*.png`.

1. **Coverage table** (`coverage.csv`): the 710 fixtures through each gate in order, with
   counts. Separately, the market-level refusals by (market, reason) (`market_refusals.csv`).
2. **Rates** (`rates.csv`, one row per panel fixture): match_id, date, season, tournament, home,
   away, lambda_h, lambda_a, kl, n_markets, n_selections, lines_quoted, has_1x2, accepted, reason.
3. **Global-optimum check:** max |Δθ| across starts.
4. **KL vs SSE** (`kl_vs_sse.csv` + summary):
   - run `invert_market_rates` with the default `MarketInversionConfig` on the **same gated book**
     as the control;
   - report Δ log λ_h, Δ log λ_a, Δ supremacy log(λ_h/λ_a) and Δ log(λ_h + λ_a): quantiles and the
     20 largest;
   - separately, a reproduction check of the **ungated** SSE inversion against
     `market_inverse_dynamics/results/production/market_targets.csv`: n in common and max |Δ| (the
     store snapshot differs; report, don't force);
   - list which of the 623 TODO 023 fixtures are now refused, and why.
5. **In-sample line residuals** (`line_residuals.csv`): r = q_model − p_market per quoted selection,
   by line × selection. Give mean, sd and a fixture-bootstrap 95% CI (2,000 reps, fixed seed),
   overall and split by tournament (56 / 57) and by n_markets.
6. **Held-out-line test** (`heldout_line_test.csv`):
   - take the accepted fixtures with 1X2 + O/U 2.5 plus at least one of {O/U 0.5, 1.5, 3.5, BTTS};
   - invert on 1X2 + O/U 2.5 only, and predict the held-out lines;
   - per held-out selection: n, mean signed residual with fixture-bootstrap 95% CI, mean |r|, and the
     in-sample residual on the same fixtures for comparison.

   This tests whether the double-Poisson grid matches the book's shape; it is not a forecast.
7. **Figures:**
   - KL distribution (log x) by n_selections;
   - line residuals with CIs, in-sample and held-out;
   - KL vs SSE log λ scatter, plus a Δ histogram;
   - λ_h vs λ_a coloured by tournament.

### 2.3 Tests `current_development/market_model/t01_market_model_tests.jl`

Runnable with `julia --project -t 8 current_development/market_model/t01_market_model_tests.jl`,
using `Test`:

1. **Grid parity:** local grid (if any) vs `Features.build_probability_matrix` ≤ 1e-14 on 50 random θ.
2. **Synthetic recovery:**
   - build exact books from known (λ_h, λ_a) on a grid covering 0.3–3.5 per side;
   - full lines recover log λ to ≤ 1e-6;
   - the 1X2 + O/U 2.5 subset also recovers to ≤ 1e-6;
   - a totals-only book is **refused** by gate 2.
3. **Completeness:** a 1X2 group with 2 of 3 runners is refused before normalisation; a complete
   group de-vigs to probabilities summing to 1.
4. **KL identity:** L(θ) = 0 at the generating θ of an exact book; L > 0 elsewhere on a small grid.
5. **Gate order:** a fixture failing several gates reports the first in §2.1(b) order.
6. **Panel bridge parity** (§2.1(c)) on the real store.

## 3. Out of scope

Any team-strength fit or Kalman run (Phase B); Turing; the pooled model; changes to `src/`;
fixing T015 or T014 in `src/`.

## 4. Acceptance (the manager reruns each item)

1. `t01_market_model_tests.jl` passes. Give the command and the pass count.
2. The coverage table sums to 710. All 27 fixtures with `has_1x2 = false` in
   `market_inverse_dynamics/results/phase2/fixture_features.csv` are refused by gate 2 (list them),
   and 14035501 is dropped.
3. The global-optimum check: max |Δθ| across starts ≤ 1e-6.
4. KL vs SSE: the quantile table, the 20 largest, and the reproduction check against
   `market_targets.csv`.
5. The line-residual and held-out tables and the four figures exist, and every number in your
   report is copied from a CSV under `results/A/`.
6. `r01_kl_inversion.jl` runs top to bottom in a fresh session. Give the wall time.
7. TODO 039's Work Log has a dated `@pi` line.

## 5. Report and stop

- Write `current_development/market_model/results/A/PHASE_A_REPORT.md`:
  - evidence per acceptance item (commands, commit SHAs, numbers);
  - the design choices you made where §2 left room (optimiser, grid implementation, starts);
  - what the residual tables say about the double-Poisson grid, in plain words;
  - anything in DESIGN.md you believe is wrong or ambiguous.
- Commit, then print exactly `PHASEA_DONE` on its own line and stop. If you are blocked on a human
  decision, write it under "Blocked" in the report, print `PHASEA_BLOCKED`, and stop.

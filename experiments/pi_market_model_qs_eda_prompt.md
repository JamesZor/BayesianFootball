# Work package: TODO 039 quality/style EDA across all betdb leagues

**Roles:** you are the implementer. Claude is the manager and reviews your work against §6. The
human decides what follows. **Do this work package only, then stop.**

**The question.** Is the quality/style formulation of the double-Poisson team model suitable
across leagues, or is it a Scottish League One/Two result? The formulation is:

    α = q + s,   β = s − q,   independent GRW steps on q and s,   r = σ_s/σ_q

It's Spec A in `current_development/market_model/research/ad_correlation/REPORT.md`.

**Settled for you by the human (2026-10-07):**
- all 11 league tournaments with Betfair data, each fitted separately, with no cups;
- market fits plus a goal check;
- C1 is parked;
- model gpt-6-sol.

## 0. Rules

### Worktree and git

Work only in `/home/james/bet_project/.worktrees/BayesianFootball-market-model-pi-qs`, on branch
`pi/market-model-qs-eda`.
- The manager created the branch: the reviewed stream (phases A–B3 and the literature report) plus
  the Phase C code (merge `7c093bf4`).
- Commit in small logical steps with conventional prefixes.
- You may `git push origin pi/market-model-qs-eda`, and no other branch.
- Never force-push, merge, rebase, or touch the stash.

### Julia

Julia runs **only through a persistent REPL in tmux**.
- Never use one-shot `julia file.jl` or `julia -e`.
- Every runner and test file prints an anchored end marker (`R06_DONE`, `T06_DONE`, …).

### tmux

**Target panes by pane ID only.**
- Create sessions with `tmux new -d -P -F '#{pane_id}' -s <name> …`, record the `%N`, and use
  only that ID.
- The human opens their own windows inside agent sessions, so a session name can point at the
  wrong pane.
- Your laptop REPL session is `pi_julia_mm_qs`.
- Never touch `pi_julia_mm_b`, `agent_claude_research_ad`, `agent_pi_market_model_c8`, or any
  session you did not create.

### mcmc-beast: every sampling run

- **Checkout:** `git -C /root/BayesianFootball worktree add --detach /root/BF_runs/market_model_qs origin/pi/market-model-qs-eda`.
  Link `/root/BayesianFootball/.env` and `/root/BayesianFootball/Manifest.toml` into it.
- **Session:** your own session `pi_mm_qs`, by pane ID, started with
  `JULIA_PKG_PRECOMPILE_AUTO=0 /root/.juliaup/bin/julia --project -t 16`, then
  `BLAS.set_num_threads(1)`.
- **Logs** go in `/root/BF_runs/logs/market_model_qs/`.
- **Other sessions:** never touch them. `julia` hosts Kaimon. The `pi_mm_c*` sessions hold
  finished Phase C runs: leave them alone.
- **Bringing results back:** copy them to the laptop worktree and commit them there. Exclude
  `/.env` and `/Manifest.toml` from any rsync.

### Data

**Data comes from fresh pinned snapshots.**
- `BF_DB_URL` is in the laptop environment. Never print it.
- On the laptop, build each segment's `DataStore` once with
  `load_datastore_cached(seg; force=true)`.
- Record each cache file's SHA256 in `results/QS/DATA_SNAPSHOTS.md`, copy the files to the beast,
  and verify the hash there. After that, load with `max_age_hours = 10^6`.
- Don't use the old `.cache/datastore_EnglishLower.jls` (2026-09-12). Its segment code is on no
  branch.

**Segments, with no `src/` changes.** Use the existing segments:
- `ScottishUpper` (54, 55);
- `ScottishLower` (56, 57);
- `IrelandAll` (79, 718);
- `Veikkausliiga` (31).

For England, define a stream-local singleton in `current_development/market_model/`:

    struct MarketModelEnglish <: BayesianFootball.Data.DataTournemantSegment end
    BayesianFootball.Data.tournament_ids(::MarketModelEnglish) = [1, 2, 3, 84]

This is a new type, so it is not type piracy. Its cache file name won't clash with anything.
`MarketModelConfig.tournaments` then selects one tournament per fit.

### Packages and the database

- Don't add or update packages. `Distributions` is pinned at 0.25.126.
- No database writes.
- Everything lives in `current_development/market_model/`.

### Honesty

- If a gate fails or a number doesn't reproduce, report it with the numbers.
- Don't loosen thresholds, pick favourable subsets, or tune priors per league.
- **The priors, seeds, budgets and price window are the Phase B2/C ones, unchanged, for every
  league.**

## 1. Read first

1. `current_development/market_model/DESIGN.md`, in particular §5 (B2 and the Phase C revisions)
   and §8 (Other leagues). Also `README.md`, "Running on another league".
2. `results/B2/PHASE_B2_REPORT.md`, especially the quality/style addendum and the levels
   definition.
3. `results/C/` on this branch: `PHASE_C_PROGRESS_REPORT.md` and `HANDOVER.md`, plus
   `r05_fast_rungs.jl` and `l05_fast_gaussian.jl` (the C0 quality/style rung).
4. `research/ad_correlation/REPORT.md`: RQ3, and Spec A–C.
5. `l01_market_model.jl` (`MarketModelConfig`, `closing_book`, the gates), `l02_two_stage.jl`,
   `l03_covariance.jl`, and the TODO 023 engine they call.

**C1 is parked.** Its synthetic recovery missed two parameters: σ_s (truth 0.010, interval
[0.0101, 0.0136]) and σ_u (truth 0.060, interval [0.048, 0.059]). The source is the beast
`results/C/v7_recovery/synthetic_recovery_c.csv`. Don't run C1 or the full-book engine.

## 2. Leagues, seasons and protocols

| Config name | Tournament | Seasons with Betfair |
|---|---|---|
| eng_premier | 1 | 22/23–25/26 |
| eng_championship | 2 | 22/23–25/26 |
| eng_league_one | 3 | 22/23–25/26 |
| eng_league_two | 84 | 22/23–25/26 |
| sco_premiership | 54 | 21/22–25/26 (only rounds 1–33 are in the database) |
| sco_championship | 55 | 21/22–25/26 (data ends 2025-09) |
| sco_league_one | 56 | 21/22–25/26 |
| sco_league_two | 57 | 21/22–25/26 |
| irl_premier | 79 | 2021–2026 |
| irl_first_division | 718 | 2021–2026 |
| fin_veikkausliiga | 31 | 2021–2026 |

**Seasons:**
- Use every season that has accepted books.
- Exclude the in-progress 2026 seasons (Ireland, Finland) and any season the data ends partway
  through (e.g. sco_championship 25/26) **from both protocols**, and list each exclusion with its
  reason.
- Keep the week grid and season handling exactly as in Phase B2.

**Protocols:**
- **10a (retrospective):** full panel.
- **10b (honest):** θ is fitted on all earlier seasons. Score the last complete season one step
  ahead, predicting before each week's updates, exactly as in B2/C.

**Expected counts:** don't assert any for the new configs, because these are measured here.

### Regression gate

The existing `scottish_lower_2425_2526()` pooled config must still reproduce. Rerun it, and check
that the published B2 R6 and C8 C0 summary CSVs are byte-identical, with the same seeds. If they
aren't, stop with `PHASEQS_BLOCKED`.

## 3. What to compute, per league

### 3.1 Phase A coverage and inversion

Same gates as Phase A, per league-season:
- the Betfair close coverage;
- the market line mix;
- the shares of 1X2-only and totals-only books;
- the overround distribution;
- the accepted and rejected counts, with reasons;
- the draw mispricing (DP misfit) on in-sample books.

Write `results/QS/coverage_by_league.csv`.

### 3.2 Rungs

Fit each rung on the Phase B2 panel construction (full books as in B2), in both protocols:

| Rung | Model |
|---|---|
| **R2** | independent α/β GRW |
| **R6** | correlated α/β steps: σ_α, σ_β, ρ_αβ |
| **C0** | quality/style GRW: σ_q, σ_s, Gaussian σ_obs, home advantage γ |

Write `results/QS/parameters_by_league.csv`. Report each value as a posterior median with a 90%
interval:
- σ_q, σ_s, **r = σ_s/σ_q** (per draw), σ_obs, γ (C0);
- σ_α, σ_β, ρ_αβ, and the per-draw rotated **corr(Δq, Δs)** and r (R6);
- R̂ and ESS for every parameter.

### 3.3 Level geometry

Use the B2 levels definition (season-average RTS α/β across active teams, at median θ):
- corr(ᾱ, β̄) per league-season;
- the implied level r = √((1 + ρ)/(1 − ρ));
- **corr(q̄, s̄) across teams.** This is the unchecked assumption in the literature report's RQ3.

Write `results/QS/levels_by_league.csv`.

### 3.4 Honest predictive comparison

Score 10b paired by fixture, with the score B2 used (the sum of the supremacy and level marginal
log densities):
- C0 vs R2;
- R6 vs R2;
- C0 vs R6.

Give mean differences with fixture SEs, and 90% coverage on both axes.

Write `results/QS/paired_scores.csv`.

### 3.5 Goal check

Does style carry information about real goals? Use 10b only. For each test fixture, take C0's
one-step predicted rates, which use only earlier weeks' closes. Score the realised goals
(y_h, y_a) under a double Poisson with three variants:
- (i) the full predicted rates;
- (ii) style set to zero (s_h = s_a = 0, leaving μ, γ and quality);
- (iii) quality set to zero.

Report the paired log score differences (i − ii) and (i − iii), with fixture SEs. Also report the
calibration slope of realised total goals on predicted expected total goals.

Write `results/QS/goal_check.csv`.

This is an **exploratory** check. One season per league gives wide intervals, so don't
over-read single leagues; also report the result pooled over all leagues.

### 3.6 Identifiability of style against fixture noise

C1's recovery suggests σ_s and the fixture noise trade off against each other.
- For every league, report the posterior correlation of σ_s with σ_obs from the C0 draws.
- For **two** leagues, eng_premier (sharp) and sco_league_two (thin), run a C0 synthetic recovery
  with 3 generation seeds each:
  - truth = that league's 10a posterior medians;
  - same panel shape and budget;
  - report whether each truth lies in its 90% interval.

Write `results/QS/c0_recovery.csv`.

### 3.7 Figures

- a forest plot of r (C0) and ρ_αβ (R6) by league, with the Scottish B2 values marked;
- level scatter plots of q̄ against s̄ per league;
- a paired-score forest plot;
- the goal-check forest plot.

## 4. Verdict criteria

These are fixed now, before any result. They are the manager's defaults. Apply them mechanically.
Report every number even where a criterion passes.

The formulation is **suitable** for a league if all of these hold:
1. **Alignment:** R6's corr(Δq, Δs) 90% interval includes 0, or |median| < 0.3.
2. **Quality dominance:** the upper end of C0's 90% interval for r is below 0.6.
3. **Prediction:** C0 is not worse than R6 by more than 2 fixture SEs on the 10b paired score.
   Also report C0 vs R2.

It is **unsuitable** if criterion 1 or 2 fails clearly. Otherwise it is **inconclusive**, with a
reason.

The goal check (§3.5) and identifiability (§3.6) are supporting evidence. They aren't pass/fail,
but call out any league where style has no goal information, or where σ_s doesn't recover.

## 5. Order of work and stopping

**If time or quota runs short, work in this order:**
1. data snapshots, segments and the regression gate;
2. coverage for all leagues;
3. C0 and R2 for all leagues, both protocols;
4. R6;
5. levels;
6. paired scores and verdicts;
7. the goal check;
8. C0 recovery;
9. figures;
10. reproduction.

**Before full runs,** time one league end to end and post the ETA. If the whole package exceeds
about 12 hours on the beast, do items 1–6 first and hand over.

**Reproduction:** rerun `parameters_by_league.csv` and `paired_scores.csv` on the beast from a
fresh REPL. They must be byte-identical, and the SHA256s go in `results/QS/REPRODUCIBILITY.md`.

**Context:** when your context passes about 70%, commit, write `results/QS/HANDOVER.md` (what's
done, what's next, the open sessions with their pane IDs) and stop with the handover sentinel.

**If quota runs out:** stop cleanly. The manager resumes you.

## 6. Deliverables and acceptance

1. **`results/QS/PHASE_QS_REPORT.md`:**
   - a verdict table per league, with the §4 criteria and their numbers;
   - a cross-league summary: is r ≈ 0.3 general, and does it differ between sharp and thin
     markets?
   - the goal check, identifiability, exclusions, and anything that didn't go to plan.
2. **All the CSVs above,** the figures, `DATA_SNAPSHOTS.md` and `REPRODUCIBILITY.md`.
3. **Tests in `t06_qs_eda_tests.jl`:**
   - the English segment returns the expected tournament IDs;
   - the configs build;
   - the per-draw transform for r and corr(Δq, Δs) matches a closed-form toy;
   - the goal-check variants reduce to the full model when nothing is zeroed.

   All t01–t05 tests (excluding the labelled `C2-pending` set), plus t06, pass in a fresh laptop
   REPL.
4. **README:** a "Cross-league quality/style EDA" section covering how to run it, the stream-local
   segment, and where the outputs are.
5. **Wrap-up:** commit, push, leave the beast tidy (close your sessions), and print exactly one
   of these alone on its line:

       PHASEQS_DONE
       PHASEQS_BLOCKED
       PHASEQS_HANDOVER

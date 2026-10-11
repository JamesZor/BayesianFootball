QS transition experiment, Stage 2: build and smoke the S3 arms (opt-in components, gated)
Role: pi implementer (gpt-6.1-sol) in .worktrees/BayesianFootball-qs-transition, branch exp/qs-transition. Stage 2 only.
Goal: make 5 S3 arms runnable and gated (control, F3 QS+A2, F1F2 entrant prior, F4 summer step, F6 weekly), with smokes only.
Done: §5 DONE WHEN; write REPORT_S2.md in experiments/scotland/11_qs_transition/.
Stop: print PHASES2_DONE alone on its line at the end. No full 60-fold grids (that is S3). No DB writes outside the harness's normal smoke runs.

## 0. Records

`~/bet_project/hub/experiments/` does not exist. Records go in `experiments/scotland/11_qs_transition/build/`,
and the report goes one level up. Local commits only; the manager pushes. Don't ask about either.

## 1. Context card

Read in full: `experiments/scotland/11_qs_transition/DESIGN.md`, `MANAGER_DECISIONS.md` (D4–D8 are
your scope), and `REPORT_S1.md` §4 (the prior) and §6 (the build list). Beyond those, only the
ranges listed here.

- **Control recipe** (qs_marketobs, unchanged):
  - `experiments/scotland/06_qs_joint_and_market_observation/candidates.jl:1-60`;
  - market table: `l01_market_table.jl` (wave 2);
  - wave-2 run `81fdf817-f82d-4f66-9bfd-5f456c4fe29f`, scope `lower`, `history_seasons = 2`,
    `dynamics_col = :match_biweek`, sampler 4 × (500 + 1000).

  S3 needs it on **W2's panel**: target seasons 23/24, 24/25, 25/26; 60 folds; 1,070 fixtures. The
  frozen market input must cover 21/22–25/26 League One/Two.
- **A2:** the `PyramidTiers` builder predictor term, `src/models/pregame/builder/pyramid_tiers.jl:1-80`.
  The W2 lower arm is in `experiments/scotland/04_pyramid_tier_transition_arms/candidates.jl:160-172`
  (`estimated_tiers(anchor = :full_pyramid)`, carry plus jump). S1 found the total step is 0.235 in
  q units (`eda/results/q3_a2_units.csv`).
- **QS:** `src/models/pregame/components/dynamics/team_level/quality_style.jl` (zero-centred,
  non-centred `z_init ~ N(0,1)·τ₀`) and `src/models/pregame/builder/quality_style_dynamics.jl`.
  Tests: `test/test_quality_style_grw.jl`.
- **Summer step precedent:** `MultiScaleGRW(target_season_step = true)` in
  `src/models/pregame/components/dynamics/team_level/multiscale.jl`, `src/models/pregame/builder/grw_dynamics.jl`
  and `builder.jl` (`rg -n target_season_step`). That includes the season-opening-fold rule
  (integrate σₛ·z per draw, centred across teams).
- **Prior inputs:** S1's per-fold construction and manifests: `eda/results/q12_source_manifest.csv`,
  `q12_fold_prior_hashes.csv`, `q2_prior_predictions.csv`, and its code in `eda/` (see `eda/README.md`).
- **Gates for every src change** (DESIGN §3):
  - exact AD ≤ 1e-10 plus an FD convergence audit;
  - zero-allocation tape (`test/tape_allocation_tests.jl` pattern);
  - 3-seed recovery (no parameter misses all 3; pooled ≥ 80%);
  - unit tests;
  - **legacy reproduction:** with every new option off, a wave-2 qs_marketobs fold reproduces
    bit-identically.

**Reading rules:** `read` with offset/limit or `rg -n`; at most 40 lines of output per command (longer
output goes to `build/logs/`); keep `build/PROGRESS.md` and `build/DECISIONS_LOG.md` current.
**After a compaction:** re-read §5 of this brief, then PROGRESS and DECISIONS_LOG; never guess IDs or
paths.

## 2. Rules

- Julia runs only on mcmc-beast, via `jl` (`jl start s2 --beast --threads 16`; long jobs `--bg` plus
  `watch_job`). BLAS 1. Checkout `/root/BF_runs/qs_transition` (update it to your commits).
- Never touch other tmux sessions. On the beast, `julia` and `mbtop` are James's.
- src changes: **opt-in, off by default**. Don't change the default behaviour of any existing component.
- No 26/27 outcomes. No tuning on the W2 panel's target seasons: the prior parameters come from seasons
  before each target season (D5/D6).

## 3. Tasks

1. **Control on W2's panel.** A candidates file for this experiment with the unchanged qs_marketobs
   recipe on target seasons 23/24–25/26.
   - Freeze the market input table (21/22–25/26).
   - Smoke 2 folds, including a 23/24 season-opening fold.
   - **Gate:** with the original 24/25–25/26 scope, one wave-2 fold reproduces bit-identically.
2. **F3, QS+A2.** Compose `PyramidTiers(anchor = :full_pyramid, carry, jump)` as in
   `td_lower_a2full_carry_jump` with `QualityStyleGRW`.
   - Verify the tier term enters once and on the right axis (q mostly; 0.48/0.52 split). S1 noted the
     total-step vs q-unit conversion.
   - Smoke 2 folds. If composition needs a src change, keep it opt-in and gated.
3. **F1F2, the entrant prior (D5/D6).** An opt-in option on `QualityStyleGRW` (or a builder feature)
   giving per-club prior means and SDs for the initial q/s state, **only for clubs with no in-scope
   history in the fold's training window**.
   - Means: S1's A2-centred P2 transport from the club's out-of-scope (54/55) closes in seasons before
     the target season, with parameters estimated from earlier movers.
   - Fallback: the A2 tier position for clubs with no out-of-scope market history.
   - Write a per-fold provenance manifest: club, source match IDs and seasons, maximum source date,
     parameters, fallback flag.
   - **Hard check:** the source match IDs overlap the fold's market-likelihood rows **0** times, and no
     source date is on or after the target season's start.
   - Smoke 2 folds, with all src gates.
4. **F4, QS target-season step.** An opt-in `target_season_step` for QS, mirroring MultiScaleGRW's,
   including the season-opening-fold rule. src gates, then smoke 2 folds.
5. **F6, weekly refits.** Check whether the fold cadence can be weekly while the dynamics clock stays
   biweekly.
   - If it can, make the arm (weekly folds, everything else unchanged), smoke 2 folds, and report the
     fold count.
   - If cadence and clock are tied, don't change src. Report it, and propose the cheapest separable
     alternative. The manager decides.
6. **Time estimate:** per-fold wall time on the beast for each arm, and the projected 60-fold grid time
   (5 arms).

## 4. Out of scope

- 60-fold grids (S3) and pooled scopes, cups and pxG (deferred by D7).
- Any change to wave-1–5 results or the frozen wave-5 config.
- An overlapping empirical-Bayes prior arm.

## 5. DONE WHEN

1. All src changes pass their gates (AD, zero allocation, 3-seed recovery, unit tests, legacy bit
   identity), with commands and results in `build/` CSVs and logs.
2. Each arm has a committed recipe and a 2-fold smoke (R̂, ESS, divergences). The F1F2 provenance
   manifest shows overlap 0.
3. `REPORT_S2.md` contains:
   - a summary of at most 12 lines;
   - the per-arm status;
   - the gate table;
   - the F6 decision note;
   - the timing projection;
   - **the exact S3 run commands**.
4. Committed (local). Beast REPL stopped. Print `PHASES2_DONE`.

## 6. Budget and blocks

- gpt-6.1-sol, thinking high.
- If you hit the usage limit, write `build/HANDOVER.md` with the exact next step and stop.
- If a gate fails twice, stop and `ask_user_question` (the manager answers in scope).

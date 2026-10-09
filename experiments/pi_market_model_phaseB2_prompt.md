# Work package — TODO 039 Phase B2: covariance structure, a config-driven pipeline, and the process README

You are the implementer. Claude is the manager and reviews your work against §5; the human signs
off before Phase C. **Do Phase B2 only, then stop.**

## 0. Rules (Phase B's rules, plus the two changes in bold)

- **Worktree:** `/home/james/bet_project/.worktrees/BayesianFootball-market-model-pi-b2`, branch
  `pi/market-model-phase-b2` (already created from the reviewed Phase B merge). Work only there.
  - Commit in small logical steps with conventional prefixes.
  - You may `git push origin pi/market-model-phase-b2`, and no other branch.
  - Never force-push, merge, rebase or touch the stash.
- **Julia runs only through a persistent REPL in tmux.** Never use one-shot `julia file.jl` or
  `julia -e`. Every runner and test file prints an anchored end marker (`R03_DONE`, `T03_DONE`, …).
  A newly started REPL counts as a fresh session.
- **Target tmux panes by pane ID only.**
  - Create each session with `tmux new -d -P -F '#{pane_id}' -s <name> …`.
  - Record the returned `%N` and use only that ID for `send-keys` and `capture-pane`.
  - Never address a pane by session name or by `session:window`.
  - The human opens their own windows inside agent sessions. In Phase B, a command meant for
    your REPL landed in the human's Claude session `%13`.
- **Laptop:** your own new session `pi_julia_mm_b2`. Never touch `pi_julia_mm_b`, which hosts
  the human's Claude session, or `claude_review_b`, which is the manager's.
- **mcmc-beast: every sampling run.**
  - Your own checkout:
    `git -C /root/BayesianFootball worktree add --detach /root/BF_runs/market_model_b2 origin/pi/market-model-phase-b2`.
  - In it, link `/root/BayesianFootball/.env` and `/root/BayesianFootball/Manifest.toml`.
  - Copy the pinned cache from `/root/BF_runs/market_model_b/.cache/datastore_ScottishLower.jls`
    and check its SHA256 is `c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4`.
  - Your own session `pi_mm_b2`, pane ID only: `JULIA_PKG_PRECOMPILE_AUTO=0 /root/.juliaup/bin/julia --project -t 16`,
    then `BLAS.set_num_threads(1)`.
  - Logs go in `/root/BF_runs/logs/market_model_b2/`.
  - **Load check:** the manager's session `claude_mm_b_review` may still be running a Phase B
    rerun. That is expected: don't touch it and don't block on it. Use `-t 8` while it runs. Print
    `PHASEB2_BLOCKED` only if the load average exceeds 20.
  - Never touch any other session (`julia` hosts Kaimon).
  - Bring `results/B2/` and the `B2_*.png` figures back to the laptop worktree and commit them
    there. Exclude `/.env` and `/Manifest.toml` from any rsync.
- **Data and packages:** pinned cache, `max_age_hours = 10^6`. Don't add or update packages.
  `Distributions` is pinned at 0.25.126.
- **No `src/` changes, no database writes.** Everything lives in `current_development/market_model/`.
- **Honesty:** if a number doesn't reproduce or a gate fails, say so with the numbers. Don't loosen
  thresholds or pick subsets.

## 1. Read first

1. `current_development/market_model/DESIGN.md`: §5 **Phase B2**, the Phase C revisions, §7 and
   **§8 Other leagues**. DESIGN wins over this brief if they disagree; say so.
2. `results/B/PHASE_B_REPORT.md`, `ladder_summary.csv`, `onestep_metrics.csv`,
   `onestep_fixture.csv`.
3. `l01_market_model.jl`, `l02_two_stage.jl`, `r02_two_stage_ladder.jl`, `t02_two_stage_tests.jl`,
   and the TODO 023 engine in `current_development/market_inverse_dynamics/l01_market_inverse_loader.jl`
   (state layout §3, `kalman_loglik`, `ffbs`, process-noise construction, `mid_gates`).
4. `docs/prototype_runner_style_guide.md`.

## 2. What to build

### 2.1 A config-driven pipeline, with no change to any published number

League facts are hard-coded today. Examples:
- `A_SEASONS` / `A_TOURNAMENTS` in `r01`;
- the `invert_panel` and bridge defaults, and match 14035501 in `l01`;
- the 78 / 517 / 1,034 assertions and the `"24/25"` / `"25/26"` strings in `l02`;
- the 10b split, and the season-boundary week in the engine.

Replace them with one config:

```julia
Base.@kwdef struct MarketModelConfig
    name::String                          # e.g. "scottish_lower_2425_2526"
    segment                               # e.g. Data.ScottishLower()
    tournaments::Vector{Int}
    seasons::Vector{String}               # chronological
    honest_train::Vector{String}          # 10b θ-fit seasons
    honest_test::Vector{String}           # 10b scored seasons
    price_window::Tuple{Float64,Float64} = (-20.0, 0.0)
    min_selections_ladder::Int = 5        # Phase B's 1X2-only exclusion
    excluded_matches::Dict{Int,String} = Dict()   # e.g. 14035501 => "T014 swapped book"
    expected::NamedTuple = (;)            # optional count assertions, per config
end
```

- `scottish_lower_2425_2526()` returns today's setup:
  - expected = (panel = 710, accepted = 595, ladder = 517, one_x2_only = 78, obs = 1034);
  - excluded = {14035501 ⇒ "T014 swapped book"}.
- Runners take the config from one constant at the top.
- Derive the season-boundary week from `seasons` and the dates, not a literal.
- **Regression gate:**
  - `r01` run through the config reproduces every Phase A CSV **byte-identically** (laptop);
  - the config-built Phase B panel equals the Phase B panel array by array;
  - the R2 collapsed log-likelihood at 3 fixed θ equals Phase B's code to ≤ 1e-12.

  You don't need to rerun the full Phase B runner.

### 2.2 Rungs R5–R7 (new loader `l03_covariance.jl`, runner `r03_covariance_ladder.jl`)

All are built on R2 (weekly GRW, league γ, Gaussian noise). R3 and R4 are **not** carried forward.

**R5 — fixture shocks, i.e. noise correlated between sides.** Observe each fixture through the
rotated pair, as two scalar observations with independent noise:

```
d_m = y_{m,h} − y_{m,a}        h_d = h_h − h_a           noise σ_sup²
ℓ_m = (y_{m,h} + y_{m,a})/2    h_ℓ = (h_h + h_a)/2       noise σ_lev²
```

- **Equivalence.** This is (ε_h, ε_a) bivariate normal with equal side variance s² and correlation
  ρ_ε, where σ_sup² = 2s²(1 − ρ_ε) and σ_lev² = s²(1 + ρ_ε)/2.
- **The transform has determinant 1**, so the likelihood of (d, ℓ) is the likelihood of (y_h, y_a).
- **Report** s² = (σ_sup² + 4σ_lev²)/4 and ρ_ε = (4σ_lev² − σ_sup²)/(4σ_lev² + σ_sup²).
- **Priors:** σ_sup ~ HN(0.20), σ_lev ~ HN(0.20).

**R6 — team quality, i.e. correlated innovations.** Each team's weekly step
(Δα̃_i, Δβ̃_i) ~ N(0, Q), with

```
Q = [σ_att²            ρ_αβ σ_att σ_def]
    [ρ_αβ σ_att σ_def  σ_def²          ]
```

- Independent across teams. Same shape at the season boundary as R2.
- Initial states as R2 (independent N(0, 0.5²)).
- Prior ρ_αβ ~ Uniform(−1, 1), sampled on atanh with its Jacobian.
- Expect ρ_αβ < 0, because β is "concedes more".

**R7:** R5 + R6.

**Inference:** exact collapsed Kalman likelihood plus coordinate slice sampling, as R2. Use
4 × (2,000 + 3,000), thin 1, both protocols:
- **10a:** full-panel θ;
- **10b:** θ fitted on `honest_train`, scored on `honest_test`.

**Gates** (`engine_gates_b2.csv`):
- R5 at σ_sup = √2·s, σ_lev = s/√2 equals R2 at σ = s, to ≤ 1e-9;
- R6 at ρ_αβ = 0 equals R2 to ≤ 1e-9;
- R5, R6 and R7 each match the batch joint Gaussian on the toy panel: likelihood ≤ 1e-9, RTS
  means ≤ 1e-8;
- FFBS moments for R6, as `mid_gates` does;
- `mid_gates()` and Phase B's engine gates still pass.

### 2.3 Measures

For **R2 (refitted as the control), R5, R6 and R7**, with the same definitions and file schemas as
Phase B:
- smoothed R² and RMSE on log λ_h, log λ_a, supremacy d, level ℓ, and log(λ_h + λ_a) by MC;
- one-step 10a/10b: RMSE, MAE, mean log predictive density, 90% coverage;
- "warm" and season-open subsets;
- the collapsed log-likelihood.

Plus three diagnostics aimed at the question:
1. **Residual correlation:** corr(e_h, e_a) of the one-step residuals per rung and protocol. R2's
   10b value was −0.585.
2. **Persistence test:** the lag-1 autocorrelation of **team-signed** one-step supremacy residuals.
   For each team, take its matches in date order, sign the residual from the team's side, and
   pair consecutive matches within a season. Give the pooled autocorrelation with a
   team-cluster bootstrap CI, per rung and protocol.
   - If team quality moves persistently, R2's residuals carry positive lag-1 autocorrelation
     and R6 removes it.
   - Fixture shocks leave none.
3. **Paired comparison** with R2 on 10b mean log predictive density per fixture: the fixture-level
   sum of both axes; Δ and its SE for R5, R6 and R7.

Outputs go to `results/B2/` (`ladder_summary_b2.csv`, `smoothed_fit_b2.csv`,
`onestep_metrics_b2.csv`, `onestep_fixture_b2.csv`, `residual_correlation.csv`,
`persistence.csv`, `paired_vs_r2.csv`, `convergence_b2.csv`), plus figures `B2_coverage.png` and
`B2_persistence.png`.

### 2.4 The stream README (`current_development/market_model/README.md`)

This documents the experiment and the process so that the human, or another agent, can rerun it on
another league. It must cover:

1. **What and why:** the human's market-model note, and the question each phase answers.
2. **The pipeline as it runs**, as a short diagram plus steps:
   - odds book → completeness and overround gates → KL double-Poisson inversion → inversion gates;
   - → panel → state-space ladder → covariance rungs.

   For each step: inputs, outputs (file paths), gates and seeds.
3. **Decisions and their reasons**, dated, including the human's: the ladder of isolated then
   pooled, double Poisson, KL, the 1X2-only exclusion, beast vs laptop, and the RBPF approval.
4. **Results so far** for A, B and B2, with every number copied from a CSV and its path given;
   the limitations Phase B listed.
5. **How to run:**
   - the tmux REPL procedure with pane IDs;
   - laptop vs beast;
   - the pinned cache and its SHA;
   - the commands per phase and the expected wall times.
6. **Running on another league:**
   - prerequisites: a `DataTournemantSegment` with betdb tournament IDs, a `src/` change that no
     branch has yet for the English leagues;
   - writing a config;
   - why Phase A's coverage table comes first;
   - which Scottish findings may not transfer: the 1X2-only totals bias, the noise size, the
     covariance winner, and the 10b split.

Keep it to what the code and CSVs show. No marketing tone.

## 3. Tests (`t03_covariance_tests.jl`, laptop)

1. Config regression (§2.1): Phase A CSVs byte-identical, panel arrays equal, R2 log-likelihood
   equal.
2. Rotation identity (§2.2 R5), with determinant 1.
3. The ρ_αβ = 0 reduction for R6; Q stays positive definite across the sampled ρ range.
4. Batch joint-Gaussian gates for R5, R6 and R7 on the toy panel.
5. Persistence statistic: on a simulated panel with planted persistent team shocks it is positive;
   with i.i.d. fixture shocks it is ≈ 0, within its CI.
6. Phase B's `t02` still passes (count).

## 4. Out of scope

Phase C (the pooled model); English or other-league segments and runs; any change to Phase A or
Phase B numbers; R3/R4/RBPF.

## 5. Acceptance (the manager reruns each item)

1. `t03` and `t02` pass in a fresh laptop REPL (commands, counts, end markers).
2. The config regression gate holds (Phase A byte-identity, panel equality, log-likelihood ≤ 1e-12).
3. `engine_gates_b2.csv` passes in full.
4. R2, R5, R6 and R7 meet R̂ ≤ 1.05 and bulk/tail ESS ≥ 200 for every θ, in both protocols
   (`convergence_b2.csv`, beast SHA, wall times).
5. All the §2.3 tables and figures exist; every number in the report is copied from them.
6. `r03_covariance_ladder.jl` runs top to bottom in a fresh beast REPL to `R03_DONE`, and a second
   fresh run gives byte-identical CSVs.
7. The README exists and covers §2.4 items 1–6.
8. TODO 039's Work Log has a dated `@pi` line; `./scripts/todo.sh check` passes.

## 6. Report and stop

- Write `results/B2/PHASE_B2_REPORT.md`:
  - evidence per acceptance item;
  - design choices made where §2 left room;
  - **in plain words:** which rung wins, fixture shocks or team quality or both; what ρ_ε and
    ρ_αβ are; whether coverage on supremacy and level gets close to 90%; and what that implies for
    Phase C's GRW and u_m;
  - anything in DESIGN or this brief you believe is wrong.
- Commit and push. Leave the beast tidy: close your sessions unless a manager rerun needs one, and
  say which.
- Print exactly `PHASEB2_DONE` on its own line and stop. If you are blocked on a human decision,
  write it under "Blocked", print `PHASEB2_BLOCKED`, and stop.

## 7. Addendum (human, 2026-10-05): is there a hierarchy behind the covariance?

Added mid-run. Post-processing only; no new model fits.

**(A) Quality/style rotation of the team states.** Use the posterior draws of R6 and R7, with R2
as the ρ_αβ = 0 control, and both protocols. Report in `rotation_quality_style.csv` (median and
90% interval for each):

```
quality q = (α − β)/2,   style s = (α + β)/2      ⇒   supremacy d = 2(q_h − q_a),  level ℓ = s_h + s_a
σ_q² = (σ_att² + σ_def² − 2ρ σ_att σ_def)/4
σ_s² = (σ_att² + σ_def² + 2ρ σ_att σ_def)/4
corr(Δq, Δs) = (σ_att² − σ_def²) / (4 σ_q σ_s)
```

- The headline number is σ_s/σ_q. A one-dimensional "team quality" hierarchy means that ratio is
  near 0.
- Also report the cross-sectional correlation of the smoothed α and β **levels** across teams,
  per season (levels, not innovations).

**(B) League-wide shock test.** Per rung and protocol, report the within-week correlation of the
one-step **level** residuals across different fixtures in the same week. Use the intraclass
correlation from a one-way random-effects decomposition by week, with a week-cluster bootstrap CI.
Write it to `league_shock_test.csv`. A positive value means a league-wide weekly component that
the static μ misses.

**Tests:** on simulated data, a planted common weekly shock gives ICC > 0, and none gives ICC ≈ 0.

**Literature these tests address** (cite in the README and the report):
- `koopman2013-dynamic-bivariate-poisson-model-analysing` p. 7, eq. 4: attack and defence
  disturbances independent;
- `baio2010-bayesian-hierarchical-model-prediction-football` p. 4 (exchangeable team effects),
  p. 6–7 (over-shrinkage; a three-group mixture).

**Acceptance adds:** both CSVs, the two simulation tests, and the numbers copied into the
plain-words section. Group/tier hierarchies stay for Phase C.

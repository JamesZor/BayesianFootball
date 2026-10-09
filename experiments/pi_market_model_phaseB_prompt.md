# Work package — TODO 039 Phase B: two-stage ladder, the Gamma question, rates vs goals

You are the implementer. Claude is the manager and reviews your work against §5; the human signs
off before Phase C. **Do Phase B only, then stop.**

## 0. Rules

- **Worktree:** `/home/james/bet_project/.worktrees/BayesianFootball-market-model-pi-b`, branch
  `pi/market-model-phase-b` (already created from the reviewed Phase A merge). Work only there.
  Commit in small logical steps with conventional prefixes. You **may** `git push origin
  pi/market-model-phase-b`; the beast needs it. **Never** push another branch, force-push, merge,
  rebase or touch the git stash. The manager merges.
- **Julia runs only through a persistent REPL in tmux** (human rule). Never use one-shot
  `julia file.jl` or `julia -e`.
  - Start a REPL once, in its own session.
  - Run files with `tmux send-keys -t <session> 'include("…")' Enter`.
  - Read the output with `tmux capture-pane -p -S -500 -t <session>`.
  - Every runner and test file prints an anchored end marker on its own line (`R02_DONE`,
    `T02_DONE`, …).
  - A newly started REPL counts as a fresh session. Re-`include` the loader after edits; there
    are no `src/` edits.
- **Laptop, archpc:** unit tests and light work only.
  - Use your own REPL session `pi_julia_mm_b`:
    `tmux new -d -s pi_julia_mm_b -c <worktree> 'julia --project -t 8'`.
  - The tmux session `claude_julia_mm` belongs to the manager; never type into it.
  - `pi_julia_mm` (Phase A) may still exist; leave it alone.
- **mcmc-beast: every sampling run** (Kalman slice/Gibbs samplers, NUTS).
  - Connect with `ssh root@mcmc-beast`, which works non-interactively.
  - **Checkout** of your pushed branch:
    `git -C /root/BayesianFootball fetch origin && git -C /root/BayesianFootball worktree add --detach /root/BF_runs/market_model_b origin/pi/market-model-phase-b`.
    After each later push:
    `git -C /root/BF_runs/market_model_b fetch origin && git -C /root/BF_runs/market_model_b checkout --detach origin/pi/market-model-phase-b`.
    Record the SHA of every run you publish.
  - **Links:** `ln -s /root/BayesianFootball/.env` and `ln -s /root/BayesianFootball/Manifest.toml`
    in that worktree.
  - **Data:** copy the **pinned** laptop cache with
    `scp .cache/datastore_ScottishLower.jls root@mcmc-beast:/root/BF_runs/market_model_b/.cache/`,
    and check it is 55,903,698 bytes on both ends. Load it with `max_age_hours = 10^6`, as in
    Phase A.
  - **Your own beast REPL session:**
    `tmux new -d -s pi_mm_b -c /root/BF_runs/market_model_b 'JULIA_PKG_PRECOMPILE_AUTO=0 julia --project -t 16'`.
    Call `BLAS.set_num_threads(1)` in it.
  - **Don't disturb other sessions.** Never attach to, type into or kill any other beast session
    (`julia` hosts Kaimon; `claude_clv_napkin`, `pi_klm`, `w2_watch`, `mbtop`).
  - **Check the load first.** Run `uptime` and `pgrep -af julia` before heavy work. If the load
    average is above 8, someone else is running a grid: print `PHASEB_BLOCKED` with the numbers
    rather than compete.
  - **Logs** go under `/root/BF_runs/logs/market_model_b/`, never inside the checkout.
  - **Bring results back** (`scp`/`rsync` of `results/B/` and the figures) into the laptop
    worktree and commit them there. Exclude `/.env` and `/Manifest.toml` from any rsync.
- **Packages:** do not run `Pkg.update` or add packages. `Distributions` is pinned at 0.25.126.
  There is **no PSIS/LOO package**; see §2.4.
- **AD rules** for any Turing model (AGENTS.md §7):
  - no `if`/`for`/`findall`/`isnan` inside `@model`;
  - mask, don't subset; `A[idx]`, never `view`; no mutation;
  - conditional logic lives outside `@model`;
  - ReverseDiff with a compiled tape, as the repo's samplers do.
- **No database writes. No `src/` changes.** Everything lives in `current_development/market_model/`.
- **Honesty:** if a number doesn't reproduce or a gate fails, say so with the numbers. Don't loosen
  a threshold or pick a subset to pass.

## 1. Read first

1. `current_development/market_model/DESIGN.md` §2, §4.2, §4.3 and the Phase B rows of §4.5 and
   §5: **the contract**. Where this brief adds to it, this brief wins (decisions after Phase A).
2. `current_development/market_model/results/A/PHASE_A_REPORT.md` and TODO 039's Verification
   section, especially the **1X2-only totals bias**.
3. `current_development/market_inverse_dynamics/README.md` (§1 inference, §5 measures and
   protocols 10a/10b) and `PHASE2_FEATURE_ATTRIBUTION.md` §5 (Student-t scale mixture, static
   states).
4. `current_development/market_inverse_dynamics/l01_market_inverse_loader.jl`:
   - arms `StaticArm`, `GRW1`, `GRW1Break`, `FeatureGRW`;
   - `kalman_loglik`, `ffbs`, `fit_arm`, `fit_feature_arm`, `prediction_metrics`,
     `fitted_logrates`, `mid_gates`, `log_prior`.

   Also read `r01_market_inverse_runner.jl` for the 10a/10b protocols and budgets.
5. `current_development/market_model/l01_market_model.jl` (Phase A), especially the
   `build_market_panel(ds, frame)` bridge.
6. `docs/prototype_runner_style_guide.md`; `docs/turing_ad_performance_guide.md`.

## 2. What to build

New loader `current_development/market_model/l02_two_stage.jl`, runner
`r02_two_stage_ladder.jl`, tests `t02_two_stage_tests.jl`. Outputs go under `results/B/`, figures
under `results/figures/B_*.png`.

### 2.1 The Phase B panel (human decision)

- Start from Phase A's `results/A/rates.csv` (or recompute with the Phase A loader; it must agree).
- **Keep only accepted books with `n_selections ≥ 5`** (1X2 plus at least one O/U or BTTS market).
  Expect **517 fixtures** (1,034 log-rate observations) and assert it.
- Exclude the **78 accepted books that are 1X2-only**, with reason
  `"1X2-only book (Phase B exclusion: totals biased ~0.10 low)"`.
- Build the `MarketPanel` with the Phase A bridge. Map the columns: `lambda_mkt_h/a` ←
  `lambda_h/a`, `sse` ← `kl` (say in a comment that the column holds KL), `n_targets` ←
  `n_selections`.
- Weekly grid (`step_days = 7`), as TODO 023.
- **The score grid stays double Poisson** (human decision; its draw/BTTS misfit is documented
  in Phase A).

### 2.2 The ladder (exact Kalman, TODO 023 engine)

| Rung | Team states | Home advantage | Observation noise |
|---|---|---|---|
| R0 | none (team states pinned at 0) | league γ | Gaussian |
| R1 static per team-season | constant within a season; one jump at the 25/26 boundary (`GRW1Break`, in-season step **fixed at 0**, σ_break learned) | league γ | Gaussian |
| R2 weekly GRW | `GRW1` (TODO 023 a1) | league γ | Gaussian |
| R3 + per-team HA | as R2 | γ_i = γ + δ_i, δ_i ~ N(0, σ_γ²), static states on home-side observations only, σ_γ ~ HalfNormal(0.10) learned | Gaussian |
| R4 + heavy tails | as R3 | as R3 | Student-t: per-fixture Gamma scale mixture shared by both sides, as Phase 2 |

- Priors are TODO 023's (`log_prior` docstring) unless stated. Write every prior into the
  loader's docstrings.
- Budgets:
  - Gaussian rungs: 4 chains × (2,000 warm-up + 3,000), thin 1, as TODO 023;
  - R4: 4 × (1,000 + 2,000 × thin 4), as Phase 2.
- Raise a budget only if a gate fails, and record that.
- **Gates for every θ of every rung:** R̂ ≤ 1.05, bulk **and** tail ESS ≥ 200. Run
  `MarketInverseDynamics.mid_gates()` first; all of its gates must pass. If R1's
  zero-in-season-step or R3's static HA states need new engine paths, add a **batch
  joint-Gaussian gate** for each on the toy panel, to the same tolerances (loglik ≤ 1e-9, RTS mean
  ≤ 1e-8).

### 2.3 Measures (every rung)

Report each measure on four axes:
- log λ_h;
- log λ_a;
- **supremacy** d = log λ_h − log λ_a;
- **level** ℓ = (log λ_h + log λ_a)/2.

Both d and ℓ are linear, so the predictive is exact. Also report log(λ_h + λ_a) by Monte Carlo
from the predictive, for comparability with DESIGN §3.

1. **Smoothed variance explained:** in-sample R² and RMSE from the smoothed fit (posterior median
   θ; RTS means), per axis (`smoothed_fit.csv`).
2. **One-step-ahead, pre-week filtration** (`onestep_metrics.csv`): RMSE, MAE, mean log predictive
   density, 90% coverage. Run two protocols:
   - **10a:** plug-in θ from the full panel;
   - **10b:** honest, θ fitted on 24/25 only and scored on 25/26.

   Also report the "warm" (drop the first 3 weeks) and season-open subsets, as TODO 023 §5.
3. **Per team and week** (`residuals_team.csv`, `residuals_week.csv`):
   - mean smoothed and one-step residuals by team-season and by week, on the d and ℓ axes;
   - the 15 worst team-seasons and weeks per rung;
   - flag transition clubs (Ross County, Airdrie, East Kilbride, Kelty) and the last 4 weeks of
     each season.
4. **Ladder summary** (`ladder_summary.csv`): one row per rung, with θ medians and 90% intervals
   (σ_obs, steps, σ_break, σ_γ, ν), smoothed R² per axis, one-step RMSE per axis (10a and 10b),
   and the collapsed log-likelihood.

### 2.4 The Gamma question (DESIGN §4.3), in Turing on the beast

- **Mean structure:** a static per team-season model,
  η_{h,m} = μ + γ + α_{h,s} + β_{a,s} and η_{a,m} = μ + α_{a,s} + β_{h,s}. Use α, β ~ N(0, σ_α²),
  N(0, σ_β²) per team-season, zero-sum per season by projection, non-centred.
- **Three observation families for λ̃** (the inverted rate, not its log):
  1. **Gamma** with mean exp(η) and shape ν: `Gamma(ν, exp(η)/ν)` in Distributions' shape–scale
     form (the human's note);
  2. **log-normal**: `LogNormal(η, σ)`;
  3. **log-Student-t:** log λ̃ ~ η + σ·t_ν, with density on the λ̃ scale =
     t-density(log λ̃) − log λ̃ (the Jacobian).
- All three are scored **on the λ̃ scale**, so their log densities are comparable.
- **Fit:** NUTS, 4 chains × (1,000 + 1,000), acceptance 0.8. R̂ ≤ 1.05, ESS ≥ 200, divergences
  ≤ 0.1%.
- **Comparison:**
  - WAIC from the full fits (pointwise log-likelihood per observation, with the effective number of
    parameters);
  - the decision metric is **10-fold cross-validation by fixture**: both sides of a fixture in the
    same fold, fixed seed, held-out log density on the λ̃ scale averaged over posterior draws,
    sum and per-observation mean, paired differences with a fixture-level SE.
  - No PSIS package exists; don't hand-roll one.
- **Mean–variance EDA** (`mean_variance.csv`, `B_mean_variance.png`): bin by fitted exp(η) (R2
  smoothed fit) into deciles and report, per bin, var(λ̃) and the CV. Constant CV points to
  Gamma/log-normal; variance ∝ mean points to quasi-Poisson.
- **Q–Q plots** of the standardised residuals per family (`B_family_qq.png`).

### 2.5 Rate-distribution EDA (DESIGN §4.5, Phase B row)

On the 517-fixture panel (`rate_distributions.csv`, figures):
- **Distributions:** λ_h, λ_a, λ_h + λ_a and supremacy by tournament, season and calendar month
  (n, mean, sd, quantiles 5/25/50/75/95).
- **Home advantage:** the mean of d by tournament and season, with fixture-bootstrap CI.
- **Correlation:** corr(log λ_h, log λ_a) overall and by tournament.
- **Coverage of exclusions:** the 78 dropped 1X2-only books by week and team, so Phase B's
  coverage holes are visible.

### 2.6 Rates vs realised goals (DESIGN §4.5)

Use full-time goals from the store (`ds.matches`; say which columns) on the same 517 fixtures
(`rates_vs_goals.csv`, `B_goal_calibration.png`):

1. **Poisson GLMs** with log link, by IRLS or `Optim` (no new packages): home goals on log λ̃_h,
   away goals on log λ̃_a, total goals on log(λ̃_h + λ̃_a), each with intercept and slope (SE).
   Calibrated means intercept 0, slope 1.
2. **Binned calibration:** deciles of λ̃ (per side and total), mean goals vs mean λ̃, with
   fixture-bootstrap CIs.
3. **The cost of the grid misfit on outcomes:** 1X2 log loss on results of
   - (a) the double-Poisson 1X2 implied by (λ̃_h, λ̃_a), and
   - (b) the de-vigged close 1X2 itself;

   with the paired difference and its fixture-bootstrap CI. Do the same for O/U 2.5 where quoted.

## 3. Tests (`t02_two_stage_tests.jl`, laptop)

1. **Panel:** 517 fixtures / 1,034 observations; no 1X2-only book; the bridge maps KL → `sse`
   correctly.
2. **R1 reduction:** with σ_break → 0 the R1 likelihood equals `StaticArm`'s on a toy panel with
   two seasons; with the in-season step at 0, the FFBS paths are constant within each season.
3. **R3 reduction:** σ_γ → 0 reproduces R2's log-likelihood (≤ 1e-9).
4. **New engine paths:** each passes its batch joint-Gaussian gate (§2.2).
5. **Families:**
   - the Gamma(ν, exp(η)/ν) mean equals exp(η);
   - log-normal and log-t log densities on the λ̃ scale integrate to 1 numerically over λ̃ for 3
     parameter sets;
   - the AD gradient of each Turing model matches finite differences (≤ 1e-6 relative) at 3
     points.
6. **CV folds:** both sides of every fixture share a fold; the folds partition the panel.
7. **GLM:** IRLS recovers known (intercept, slope) from simulated Poisson data to within 3 SE.

## 4. Out of scope

The pooled Turing model on the books (Phase C); team-parameter EDA against the goal models and
isolated-vs-pooled comparisons (Phase D); any change to Phase A's inversion; dependence grids.

## 5. Acceptance (the manager reruns each item)

1. `t02_two_stage_tests.jl` passes in a fresh laptop REPL (command, count, `T02_DONE`).
2. `mid_gates()` and every new batch gate pass (table).
3. **Every rung R0–R4 meets R̂ ≤ 1.05 and bulk/tail ESS ≥ 200 for every θ** (`convergence.csv`),
   with the beast SHA and wall time per rung.
4. `ladder_summary.csv`, `smoothed_fit.csv` and `onestep_metrics.csv` (10a and 10b, all four axes),
   plus the residual tables.
5. The family comparison: WAIC table, 10-fold CV table with paired differences and SEs, NUTS
   diagnostics, mean–variance table and figure.
6. The rate-distribution and rates-vs-goals tables and figures.
7. `r02_two_stage_ladder.jl` runs top to bottom in a fresh beast REPL to `R02_DONE` (wall time);
   rerunning it gives identical CSVs (seeded).
8. TODO 039's Work Log has a dated `@pi` line; `./scripts/todo.sh check` passes.

## 6. Report and stop

- Write `current_development/market_model/results/B/PHASE_B_REPORT.md`:
  - evidence per acceptance item (commands, SHAs, numbers copied from CSVs);
  - design choices you made where §2 left room;
  - **what the ladder says, in plain words:** how much of the market's log-rate each rung explains
    on supremacy vs level, whether per-team HA and heavy tails earn their place, which family
    wins, and whether the close's rates are calibrated to goals;
  - anything in DESIGN.md or this brief you believe is wrong or ambiguous.
- Commit and push your branch. Leave the beast tidy: keep `pi_mm_b` running only if a manager
  rerun needs it, and say so.
- Print exactly `PHASEB_DONE` on its own line and stop. If you are blocked on a human decision,
  write it under "Blocked" in the report, print `PHASEB_BLOCKED`, and stop.

# 007 — Prototype Gaussian Random Walk State Space Dynamics with ReverseDiff

| Field | Value |
|---|---|
| ID | 007 |
| Title | Prototype Gaussian Random Walk State Space Dynamics with ReverseDiff |
| Status | COMPLETED |
| Priority | P2 |
| Assignee | pi |
| Created | 2026-09-10 |
| Updated | 2026-09-11 |
| Related Files / Commits / PRs | [src/models/pregame/components/dynamics/team_level/multiscale.jl](../src/models/pregame/components/dynamics/team_level/multiscale.jl); [src/models/pregame/builder/grw_dynamics.jl](../src/models/pregame/builder/grw_dynamics.jl); [test/test_multiscale_grw.jl](../test/test_multiscale_grw.jl); [current_development/multiscale_grw/](../current_development/multiscale_grw/); `feat/multiscale-grw-dynamics` |

## Context & Problem Statement

Historically, team strength evolution in BayesianFootball.jl relied on exponential decay with a fixed half-life (`TimeDecayDynamics(days_half_life = 180.0)`) or static season ratings. True latent state-space Gaussian Random Walk (GRW) or AR(1) dynamics—where each team has time-varying latent attack/defence states $\alpha_{i,t} \sim \mathcal{N}(\alpha_{i,t-1}, \sigma_a^2)$ across $T$ match weeks—were previously discarded due to prohibitive MCMC execution times under ForwardDiff (where Dual-number overhead on 1,000+ latent parameters scaled quadratically or exhausted L3 cache).

With ReverseDiff tape compilation fully operational (producing 24× speedups and 5–8 minute 40-fold grids on `mcmc-beast`), high-dimensional latent state models are now computationally tractable. A single reverse sweep over the compiled tape evaluates the gradient of 1,000+ latent random-walk innovations in milliseconds.

We need to prototype, benchmark, and evaluate a true latent state-space GRW component against the current fixed half-life standard on Scottish Lower.

## Acceptance Criteria

- [x] Implement a Gaussian random walk dynamics component conforming to the team-level dynamics contract. Delivered as the existing (previously unexported) `BayesianFootball.Models.PreGame.MultiScaleGRW`, adapted for this prototype through builder-extension methods in `current_development/multiscale_grw/l01_loader.jl` rather than a new `src/` component — see Work Log decision on scope.
- [x] Implement a non-centered parameterization ($\alpha_{i,t} = \alpha_{i,0} + \sigma_a \sum_{k=1}^t z_{i,k}$, with $z_{i,k} \sim \mathcal{N}(0, 1)$) to prevent Neal's funnel geometry divergences in NUTS. Implemented as two-speed non-centered walks: one macro innovation per history season, one micro innovation per target-season match-biweek, cumulative-summed and zero-centered over teams each step.
- [x] Support discretization by fold-relative time index. Implemented as macro-season / micro-match-biweek two-speed discretization (continuous-time Brownian scaling was not required to pass convergence gates and was not implemented).
- [x] Ensure AD-safety inside the likelihood tape under ReverseDiff. Zero-allocation gradient replay was targeted but **not achieved**: installed ReverseDiff stack allocates ~35–37KB per compiled-tape gradient call in this harness (the TimeDecay control allocates ~43,888 bytes in the identical harness), so this is reported as an unmet performance target rather than a gated pass — see `README.md` contract note.
- [x] Preflight on 2-fold CV to verify tape compilation, gradient validity, and absence of NUTS divergences. Ran for all three candidates (`m00_baseline_grw`, `m05_production_wealth_grw`, `m05_joint_production_wealth_grw`); zero divergences on every fold; hard preflight gates (zero divergences, BFMI ≥ 0.30, treedepth_rate < 0.05) passed; R̂/ESS were advisory and some folds ran above 1.01 R̂ at 2-fold scale, consistent with the low-draw preflight budget.
- [x] Execute a full 40-fold walk-forward grid on Scottish Lower (pooled tournaments 56/57, target seasons 24/25 + 25/26) on `mcmc-beast` (-t 16). All three candidates completed 40/40 folds, 710 held-out fixtures.
- [x] Persist run artifacts to PostgreSQL `mcmc_experiments` with `ad_backend = 'reversediff'`. Confirmed by direct query of the `runs`/`fold_results` tables: `m00_baseline_grw` → `f64a00a2-34a0-4f31-8c58-c093c92d54b7`, `m05_production_wealth_grw` → `b2d8036d-8fbd-45f9-92b5-cc7675926232`, `m05_joint_production_wealth_grw` → `f870dbb7-9df0-4dae-a84a-cf570cf8113e`; all `status = completed`, `ad_backend = reversediff`.
- [x] Benchmark out-of-sample proper scores (LogLoss, CRPS, Brier, RPS) and sampling times against `TimeDecayDynamics(180.0)`. All three candidates beat their matched TimeDecay controls on all four proper scores — see Verification & Findings.
- [x] Phase 2: execute the two-arm Joint Gamma-Poisson candidate (`m05_joint_production_wealth_grw`) against the `TimeDecayDynamics` joint control, gated on both Phase 1 candidates passing strict convergence. Completed 2026-09-11; passed strict gates (R̂ 1.0069, bulk ESS 1331, tail ESS 1086, 0 divergences of 256000) and beat the joint control on all four proper scores. Run UUID `f870dbb7-9df0-4dae-a84a-cf570cf8113e`.
- [x] Layer 4 portfolio backtest and Layer 2 generative rate calibration over all three GRW fits, their three matched TimeDecay controls and `m12_joint_hybrid_synergy`, under one canonical Betfair staking contract. 21 arm/variant simulations over a 622-fixture buildable panel; artifacts in `current_development/multiscale_grw/results/` — see Verification & Findings.
- [x] Graduate `MultiScaleGRW` into `src/` as a natively supported `CountModelBuilder` dynamics component, with the state-count contract corrected and regression-tested. Component rewritten in `src/models/pregame/components/dynamics/team_level/multiscale.jl`; builder seam added as `src/models/pregame/builder/grw_dynamics.jl`; exported through `PreGame` → `Models` → `BayesianFootball`; 124 assertions in `test/test_multiscale_grw.jl`, full suite 3962 pass / 0 fail.

## Ideas & Candidate Solutions

- **Non-Centered vs Centered Parameterization**: In sparse leagues with weekly observations, centered parameterizations ($\alpha_t \sim \mathcal{N}(\alpha_{t-1}, \sigma)$) suffer severe funnel geometry as $\sigma \to 0$. Non-centered formulation using unit-normal innovations $z_{i,t}$ is mandatory for robust NUTS sampling.
- **Discretization Strategy**:
  - *Weekly step grid*: Matches are mapped to calendar match-weeks $w \in \{1, \dots, W\}$. Vectorized over teams, easily formatted as a 2D matrix of innovations `z[team, week]`.
  - *Continuous-time Brownian motion*: Innovation variance scales as $\sigma^2 \Delta t_{ij}$ where $\Delta t$ is days elapsed since the team's prior match. Captures mid-week vs weekend scheduling but introduces ragged indexing.
- **Mean-Reversion (Ornstein-Uhlenbeck / AR(1))**: A pure random walk can diffuse to unphysical extremes over multi-year periods. Adding a mild mean-reversion parameter $\rho \in (0.90, 0.99)$ shrinks ratings back toward the league baseline:
  $$\alpha_{i,t} = \rho \alpha_{i,t-1} + \sigma \sqrt{1 - \rho^2} z_{i,t}$$
- **Innovation Scale Hyperpriors**: Prior scale on weekly drift $\sigma_a, \sigma_d \sim \text{HalfNormal}(0.05)$ to ensure team strength changes smoothly week-to-week rather than overfitting single match noise.

## Work Log & Progress

- [2026-09-10 @antigravity] Created task in BACKLOG following ReverseDiff 24× speedup breakthrough and user proposal. Outlined mathematical formulation and acceptance criteria.
- [2026-09-10 @pi] Claimed in session `pi_solo:1`, worktree `/home/james/bet_project/.worktrees/BayesianFootball-grw-dynamics` on branch `feat/multiscale-grw-dynamics`; remote compute target `/root/BF_multiscale_grw` on `mcmc-beast`. Design locked via `/grill-me`: revive `MultiScaleGRW` in `current_development/multiscale_grw/` (l01_loader.jl + r01_runner.jl); Phase 1 (Poisson m00/m05) -> 2-fold preflight -> 40-fold grid on beast -> benchmark vs TimeDecayDynamics; Phase 2 (Two-arm Joint Gamma-Poisson m05) overnight.
- [2026-09-11 @pi] Two-fold preflight passed for all three candidates. Phase 1 production grid (40 folds × 4 chains, 800 warmup + 800 samples) launched; `m00_baseline_grw` initially missed the strict R̂ ≤ 1.01 gate (R̂ 1.0135), as did `m05_production_wealth_grw` on its first pass (R̂ 1.0122, then a second attempt at 1.0132). Implemented `l01_extend_sampling` in `l01_loader.jl` to append a further independent 800-warmup/800-sample run, concatenate raw chain draws, and re-audit convergence on the full concatenated chain while persisting only a 1-in-4 thinned draw subset (full 6400-draw serialized artifact was ~1.18GB and exceeded the PostgreSQL `bytea` text-protocol path; thinned artifact is ~295MB and saves cleanly). Both candidates passed strict gates after one extension pass each and were persisted to `mcmc_experiments` (`scottish_lower_multiscale_grw_2426`). Phase 1 runner reported `PHASE 1 PROMOTION PASS`.
- [2026-09-11 @pi] Ran the Layer 4 / Layer 2 benchmark (`current_development/multiscale_grw/r02_portfolio_calibration.jl`, ~13 min on beast). Four bugs were each caught by an assertion rather than by inspection, which is the argument for writing the assertions: `MarketRateFit` exposes `.accepted` not `.inverted`; `@sprintf` rejects a concatenated format string; a fit's latents must be **restricted to the panel before building** (arms hold 710 fixtures, the book quotes 635, so an unrestricted build skipped 75); and the panel must be **buildable**, not merely quoted (13 quoted fixtures were refused for `no usable selections`). A pre-written narrative bullet claiming the closing-instant calibrator collapses the edge was **contradicted by the run** — measured median `w` is 0.97 at the close versus 0.29 at T-25 — so the prose was replaced by `l02_calibration_note`, which derives the claim from the measured weights instead of asserting it.
- [2026-09-11 @pi] Graduated `MultiScaleGRW` into `src/`. Found and fixed the legacy state-count off-by-one (`1 + n_history + n_target` vs the contracted `n_history + n_target`) while porting the validated prototype math. Builder seam isolated in `src/models/pregame/builder/grw_dynamics.jl` (included after `engine.jl`, since it adds methods to hooks that file declares) rather than scattered across `builder.jl`/`engine.jl`/`player_dynamics.jl`. The prototype's specialised `extract_parameters` override was **not** graduated: the generic composable extractor is fully hook-driven via `_cb_extract_dynamics`/`_cb_oos_dynamics` and computes the identical final state, so the override was redundant. Verified on beast: `test/test_multiscale_grw.jl` 124/124 (including ReverseDiff vs ForwardDiff gradient agreement and compiled-tape replay at a perturbed point), full `runtests.jl` 3962 pass / 1 pre-existing broken / 0 fail in 5m45s.
- [2026-09-11 @pi] Phase 2 launched on `mcmc-beast` (tmux `multiscale_grw_phase2`, `L01_STAGE=phase2`) after confirming the Phase 1 promotion gate and an idle machine. Base pass: 46m40s, R̂ 1.0161 / ESS 507 / **0 divergences of 128000** — same profile as Phase 1, good geometry but short of the strict R̂ gate. One `l01_extend_sampling` pass (44m15s, again 0 divergences) brought the combined chain to R̂ 1.0069 / bulk ESS 1331 / tail ESS 1086 and the run was persisted. `PHASE 2 PASS` recorded. Notably the joint candidate's advantage over its TimeDecay control is roughly 5× smaller than the Poisson candidates' (Δ LogLoss -0.00031 vs -0.00170/-0.00148): the second likelihood arm already supplies much of the temporal sharpening that the GRW state buys in the single-arm models, so the two mechanisms are substantially substitutive rather than additive.

## Verification & Findings

**Two-fold preflight** (all three candidates, zero divergences, hard gates passed):

| Model | Fold | Parameters | Tape instructions | Gradient ms | Alloc bytes | R̂ max | ESS min | Divergences |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `m00_baseline_grw` | 1 | 98 | 735 | 0.053 | 35440 | 1.0091 | 558 | 0 |
| `m00_baseline_grw` | 2 | 158 | 1309 | 0.071 | 36608 | 1.0104 | 691 | 0 |
| `m05_production_wealth_grw` | 1 | 99 | 751 | 0.058 | 35440 | 1.0128 | 566 | 0 |
| `m05_production_wealth_grw` | 2 | 159 | 1325 | 0.085 | 36608 | 1.0132 | 453 | 0 |
| `m05_joint_production_wealth_grw` | 1 | 101 | 791 | 0.120 | 128848 | 1.0075 | 681 | 0 |
| `m05_joint_production_wealth_grw` | 2 | 161 | 1365 | 0.108 | 133088 | 1.0154 | 582 | 0 |

**Production grid** (40 folds, 710 held-out fixtures, extended to 1600 draws/chain × 4 chains — every candidate missed the strict R̂ gate on its first 800+800 pass and passed after exactly one extension):

| Model | Phase | Folds | OOS | R̂ max | ESS bulk min | ESS tail min | Divergences | Wall min | Run UUID |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `m00_baseline_grw` | 1 | 40 | 710 | 1.0067 | 1705 | 1662 | 0 / 256000 | 1.1 | `f64a00a2-34a0-4f31-8c58-c093c92d54b7` |
| `m05_production_wealth_grw` | 1 | 40 | 710 | 1.0073 | 1909 | 1911 | 0 / 256000 | 60.7 | `b2d8036d-8fbd-45f9-92b5-cc7675926232` |
| `m05_joint_production_wealth_grw` | 2 | 40 | 710 | 1.0069 | 1331 | 1086 | 0 / 256000 | 90.9 | `f870dbb7-9df0-4dae-a84a-cf570cf8113e` |

All three confirmed live in `mcmc_experiments.runs` by direct query (not console output) with `status = completed` and `ad_backend = reversediff`; Phase 1 rows carry `git_commit = 31e4795-dirty`. **Zero divergences across all 768,000 post-warmup draws in the study** — the non-centered two-speed parameterization is doing its job.

The strict R̂ ≤ 1.01 gate was never met by an 800-warmup/800-sample pass for any candidate (first-pass R̂: 1.0135, 1.0122/1.0132, 1.0161) despite clean geometry, so the gate was treated as a draw-budget question rather than a model defect: `l01_extend_sampling` appends a second independent 800+800 run and re-audits the concatenated chain. One extension sufficed in every case.

**Proper scores vs matched `TimeDecayDynamics` control** (negative Δ favours MultiScaleGRW):

| GRW candidate | TimeDecay control | LogLoss | Δ LogLoss | Brier | Δ Brier | RPS | Δ RPS | CRPS | Δ CRPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `m00_baseline_grw` | `m00_baseline` | 0.65866 | -0.00170 | 0.23337 | -0.00079 | 0.22537 | -0.00189 | 0.62701 | -0.00368 |
| `m05_production_wealth_grw` | `m05_production_wealth` | 0.65825 | -0.00148 | 0.23318 | -0.00069 | 0.22493 | -0.00129 | 0.62667 | -0.00254 |
| `m05_joint_production_wealth_grw` | `m05_joint_production_wealth` | 0.65681 | -0.00031 | 0.23250 | -0.00011 | 0.22534 | -0.00023 | 0.62660 | -0.00050 |

All three GRW candidates beat their matched TimeDecay control on every one of the four proper scores. `PHASE 1 PROMOTION PASS` and `PHASE 2 PASS` both recorded by the runner; full detail and reproduction commands in `current_development/multiscale_grw/README.md`.

**Headline reading.** The latent state-space GRW is now computationally tractable under compiled ReverseDiff — the central premise of this task — and it is a genuine but small improvement over the fixed 180-day half-life on every proper score. The effect is strongly conditional on what else is in the model: it is worth ~-0.0017 LogLoss on the plain Poisson baseline, ~-0.0015 with the wealth covariate, and only ~-0.0003 once the two-arm Joint Gamma-Poisson likelihood is present. The proxy-xG arm and the GRW state are largely competing to explain the same temporal signal, so the gains do not stack. For context, the joint likelihood itself is worth ~5× the best covariate (Gen 3 finding), and the GRW's marginal contribution on top of it is an order of magnitude smaller than that.

**Unmet performance target.** Zero-allocation gradient replay was not achieved: ~35–37KB per compiled-tape gradient call for the Poisson candidates and ~129–133KB for the joint candidate. The TimeDecay control allocates ~43,888 bytes in the identical harness, so this is a property of the installed ReverseDiff stack rather than a regression introduced by `MultiScaleGRW`. Recorded as an unmet target, not a gated pass.

**Cost.** MultiScaleGRW is materially more expensive than TimeDecay at equal fold count once the extension pass required to clear the strict R̂ gate is counted (`m05_production_wealth_grw` 60.7 min vs the control's ~2.0 min; the joint candidate 90.9 min). A promotion decision should weigh a ~-0.0015 LogLoss gain on single-arm models against roughly a 30× sampling cost, and note that the gain nearly vanishes in the joint configuration that is closest to the current production shape.

### Layer 4 portfolio backtest

Seven arms priced through one identical staking contract — `FlatTrust(0.30)` / `SlateDrawdown(23.0)` / `FixedCap(0.20)`, daily slate, 1X2 + O/U 2.5 + BTTS, Baker-McHale shrinkage — over a **622-fixture** panel, 4,000 bootstrap resamples, seed 1. The panel is the intersection of all seven arms' held-out fixtures with the Betfair book, then narrowed to the fixtures every arm can actually *build* (635 quoted → 622 buildable; the 13 dropped all failed with `no usable selections`). Only the posterior differs between rows.

| Model | Return % | CAGR % | g/slate | Sharpe | Calmar | MDD % |
|---|---:|---:|---:|---:|---:|---:|
| `m05_production_wealth_grw` | **+174.88** | **+79.72** | **+0.01021** | **1.715** | **8.86** | -19.7 |
| `m00_baseline_grw` | +153.84 | +71.61 | +0.00941 | 1.636 | 8.41 | -18.3 |
| `m12_joint_hybrid_synergy` | +129.32 | +61.80 | +0.00838 | 1.355 | 6.59 | -19.6 |
| `m05_joint_production_wealth_grw` | +125.06 | +60.05 | +0.00819 | 1.647 | 6.25 | -20.0 |
| `m05_production_wealth` *(control)* | +124.25 | +59.71 | +0.00816 | 1.305 | 5.91 | -21.0 |
| `m05_joint_production_wealth` *(control)* | +117.34 | +56.84 | +0.00784 | 1.395 | 6.30 | -18.6 |
| `m00_baseline` *(control)* | +115.48 | +56.06 | +0.00775 | 1.215 | 5.47 | -21.1 |

Each GRW arm beats its own matched control, and the ordering of the *margins* tracks the proper-score ordering exactly: `m00_baseline_grw` +38.4 pp (ΔLogLoss -0.00170), `m05_production_wealth_grw` +50.6 pp (-0.00148), `m05_joint_production_wealth_grw` **+7.7 pp** (-0.00031). The backtest therefore **re-expresses the proper-score finding at a different magnification rather than discovering independent alpha**, and it reproduces the substitutability result: the GRW's edge over its control nearly vanishes once the proxy-xG arm is present.

The two best GRW arms also beat `m12_joint_hybrid_synergy`, the current production hybrid, on return, growth, Sharpe and Calmar. **At n = 622 the 95% bootstrap growth interval overlaps zero for most arms, so this ranking is not statistically separable** and no arm should be promoted on these figures alone.

### Layer 2 generative rate calibration

Every arm was re-priced through `GenerativeRateCalibrator` at two instants: `scot_lower_t25_inv` (`InverseGaussianLaw(w_base = 0.25, sigma = 0.35)` at T-25) and `scot_lower_close_std` (`StandardGaussianLaw(w_base = 0.30, sigma = 0.40)` at T+0), both with `PoolDispersion()`. 14 calibrated containers, all coherent: worst `max_family_spread` across the 1X2 / totals / BTTS partitions of the shared score tensor was **6.66e-16**, worst deviation from unit mass 1.83e-4. Coherence here is structural, not audited.

Best calibrated arm, `m05_production_wealth_grw` at T-25: **MDD -9.47%** (from -19.75%), **Sharpe 2.014** (from 1.715), **Calmar 12.32** (from 8.86), on +116.72% return.

The headline is a **risk transformation, not a return improvement**. Median across the seven arms:

| Instant | median `w` | market share | var retention | Δreturn pp | ΔMDD pp | ΔSharpe |
|---|---:|---:|---:|---:|---:|---:|
| `t25_inv` | 0.290 | 0.710 | 0.084 | -48.45 | **+7.09** | +0.128 |
| `close_std` | 0.974 | 0.026 | 0.949 | -10.76 | +3.11 | +0.150 |

At T-25 the pool keeps only ~29% model weight and ~8% of the posterior variance, so the book supplies most of the location: drawdown roughly halves and Sharpe and Calmar rise sharply, but ~48 pp of return is surrendered. **Read the weight column before the return column** — a calibrated arm's Sharpe is substantially the market's, and the two instants are not interchangeable (a spec moved between them gives up 0.0015-0.0020 LogLoss). Calibration also compresses the spread between arms, which is the expected consequence of every arm being shrunk toward the same book.

### Graduation to `src/`

`MultiScaleGRW` is now a first-class `CountModelBuilder` component:

```julia
model = CountModelBuilder(:m00_baseline_grw) |>
    add(GlobalInterception()) |> add(MultiScaleGRW()) |>
    add(GlobalHomeAdvantage()) |> add(PoissonObservation()) |> build
```

The graduation **fixed a silent off-by-one** carried by the previously unexported component: its `cumsum(hcat(init, season, target))` emitted `1 + n_history + n_target` states against a `:n_rounds` contract of `n_history + n_target`, so every fixture read the wrong state. There are `n_history - 1` macro transitions, not `n_history`. `dynamics_design` now asserts the identity rather than trusting it, and the accumulator geometry is pinned test-by-test. The two legacy arities (`build_dynamics(config, n_teams, n_history, n_target)` and the 6-arg `extract_dynamics`) are retained for the four hand-written `standard/` engines and were re-implemented on the corrected math.

The cumulative sum is expressed as multiplication by three constant 0/1 accumulator matrices rather than `cumsum`/`hcat`, because ReverseDiff preallocates a matrix-multiply output in the compiled tape but cannot preallocate `hcat` scratch. `MultiScaleGRW` takes **unit likelihood weights** — recency is latent state, and layering time decay on top would discount the same evidence twice. `cb_chain_columns` now **refuses** the GRW's `z_season`/`z_target` sites rather than defaulting them to width 1: their widths are fold-dependent and the scalar default would silently undercount θ.

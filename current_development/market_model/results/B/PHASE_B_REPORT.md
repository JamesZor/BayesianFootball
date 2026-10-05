# TODO 039 — Phase B report (2026-10-05)

**Status: Phase B implemented and verified; ready for manager/human review.** Phase B only. No Phase C/D, `src/` edits, database writes, package additions or updates.

## Reproduction and acceptance evidence

Scientific source SHA: `8663544edf6a494bb2edb7d2ad791b64408b63bc`.

- Pinned cache: **55,903,698 bytes**, SHA256 `c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4`, identical on laptop/beast; `max_age_hours=10^6`.
- Fresh laptop Julia 1.12.1, own `pi_julia_mm_b:0.0` / `%21`, `julia --project -t 8`: `using LinearAlgebra; BLAS.set_num_threads(1); include("current_development/market_model/t02_two_stage_tests.jl")` → **131/131**, **88.1 s**, anchored `T02_DONE`.
- Fresh beast Julia 1.12.4, own `pi_mm_b:0.0` / `%246`, `JULIA_PKG_PRECOMPILE_AUTO=0 /root/.juliaup/bin/julia --project -t 16`: `include("current_development/market_model/r02_two_stage_ladder.jl")` → anchored **`R02_DONE`**, **3291.801 s**. Runner pins cores and sets BLAS=1. Initial load 0.21 / 0.11 / 0.19; no competing grid was started.
- Independent second fresh beast REPL `%247`, same source/command/seeds, started at load 0.02 / 1.22 / 2.72 → **`R02_DONE`**, **3308.673 s**. All **37 runner-generated CSVs are byte-identical**; two supplemental preflight CSVs remained unchanged. All four original PNGs also matched. Wall times/timestamps deliberately live in `RUN_PROVENANCE.md`, not scientific CSVs. See `REPRODUCIBILITY.md` and `CSV_SHA256SUMS.txt`.
- Presentation-only fixes `17fb7109` / `9469f327` reserve bottom/left margins for previously clipped axis labels. At `9469f327e399ba47f633e56df86438306d90f0d1`, the figure generator was reloaded and run twice on the completed numerical inputs → `R02_FIGURES_DONE`; all four corrected PNGs are byte-identical and visually inspected. No sampling/scoring/data code changed, and all CSV hashes still match. The current plotting-code test re-include also passes **131/131**, **23.7 s**. For exact CSV provenance comparison, rerun the recorded statistical source SHA above; later presentation/artifact commits change HEAD-based provenance columns, not numerical results.
- Logs outside checkout: `/root/BF_runs/logs/market_model_b/8663544e_run1.log` and `8663544e_run2.log`.
- `engine_gates.csv`: **24/24**, including all 13 TODO 023 gates, R0/R1/R3 batch likelihood/RTS checks, deterministic R1 paths/reductions, and the R4 limit. Fixed ω=1 independently matches R3: likelihood error **0**, pre-week mean **5.55e-16**, variance **1.35e-15**; each tolerance **1e-8**. New batch likelihood tolerances remain **1e-9**, RTS **1e-8**.
- `family_ad_gates.csv`: **27/27**; compiled/fresh ReverseDiff, ForwardDiff and finite differences at three linked-coordinate points (displacements 0, +0.8, −0.8). Maximum finite-difference relative error **3.27e-10**, required ≤1e-6.
- `convergence.csv`: **30/30 θ checks**, both full-panel 10a and 24/25-only 10b; max R̂ **1.01622**, min bulk ESS **326.49**, min tail ESS **463.07**. No budget was increased.
- `family_diagnostics.csv` / `nuts_diagnostics.csv`: all **33 full/fold fits**, **2684 parameter checks**, max R̂ **1.01903**, min bulk ESS **375.62**, min tail ESS **581.18**; **zero retained divergences** (132,000 retained draws per full run). Numerical-error internals are explicitly extracted from raw chain storage, not silently discarded by `Array(Chains)`; a synthetic five-divergence/4000-draw regression must fail the 0.1% gate.
- `family_guard_gates.csv`: all 33 fits pass the nonbinding log-rate guard check, including held-out locations; maximum |η| **1.10861**, bounds ±10.

### Ladder wall times (first fresh full run, seconds)

| Rung | 10a | 10b |
|---|---:|---:|
| R0 | 23.208 | 10.911 |
| R1 | 42.320 | 20.713 |
| R2 | 59.011 | 28.622 |
| R3 | 129.048 | 62.103 |
| R4 | 559.829 | 389.741 |

Gaussian budgets: 4 × (2000 warmup + 3000 retained), thin 1. R4: 4 × (1000 warmup + 2000 retained), thin 4. Family fits: 4 × (1000 + 1000), acceptance 0.8. Per-fit SHAs/times are also recorded in `RUN_PROVENANCE.md`.

## Panel and modelling choices

- Exactly **517 fixtures / 1034 observations**; all retained books have ≥5 selections. The 78 accepted 1X2-only books are excluded with the prescribed reason. No inversion or double-Poisson score grid was changed. The legacy `sse` bridge field contains **KL**, not SSE.
- Weekly grid, 22 ladder clubs, Gaussian static league μ/γ integrated by the conditional Kalman engine. R0 pins team states at zero; R1 has exactly zero in-season process noise and one learned season jump; R3's static δ_i indicator is home-only, not the legacy antisymmetric feature design.
- Human-approved static family population: **38 observed team-seasons**, not all 22 clubs crossed with both seasons. Non-centred raw effects are projected to zero sum separately within each season. The same population/design is retained in every CV fold; masks exclude held-out responses.
- Priors are documented in `l02_two_stage.jl` and `l02_two_stage_families.jl`. New family priors were approved: team SDs HalfNormal(.5), log-family σ HalfNormal(.20), Gamma shape Gamma(2, scale 100), log-t df Gamma(2, scale 10).
- **10a is retrospective plug-in**, not honest forecasting. **10b fits θ using only the 244 season-24/25 fixtures**, then sequentially filters and scores the 273 season-25/26 fixtures. All within-week predictions precede every observation update in that week. “Warm” drops the first three panel weeks; season-open means each season's first three observed weeks. Thus 10b all/warm coincide; in-season and season-open are separately supplied.
- R0–R3 linear-axis predictions are exact Gaussian at fixed θ. Human-approved R4 is an RBPF, not an exact unconditional Gaussian: only fresh fixture-shared Gamma weights are sampled, each particle carries exact conditional ratings/HA mean and covariance. Full-panel posterior fixture weights are **never** used for one-step prediction.
- Smoothed R4 is explicitly **RTS at median θ conditional on posterior mean ω**, not an ω-integrated smoothed posterior mean. Its conditional likelihood is separately labelled; the primary `collapsed_loglik` integrates states/scales via RBPF. Nonlinear log-total prediction uses 4000 seeded draws and a labelled Gaussian KDE, not an exact Gaussian density.
- Family decision scores sum the two marginal posterior-predictive λ-scale log densities, with paired differences/SEs at **fixture** level. The joint posterior-predictive fixture density is additionally retained in `family_cv_fixture.csv`. No PSIS approximation. Fixture folds use seed 3903; both sides share their fold. There are 517 fold assignments and 1551 family-fixture scores.
- Calendar-month groups are YYYY-MM. Team supremacy residuals are oriented toward the named club (away signs reversed); level residuals are unchanged. Requested transition-name and last-four-observed-week flags are included.

## What the ladder says

From `ladder_summary.csv` (10b RMSE is genuinely held out; R² is descriptive/smoothed):

| Rung | Smoothed R² supremacy | Smoothed R² level | 10b RMSE supremacy | 10b RMSE level | Full-panel collapsed loglik |
|---|---:|---:|---:|---:|---:|
| R0 | ~0.00% | ~0.00% | .50066 | .09058 | 14.939 |
| R1 | 74.95% | 64.27% | .26528 | .05857 | 518.302 |
| R2 | 98.06% | 92.52% | .19556 | .05049 | 820.137 |
| R3 | 98.10% | 92.62% | .19537 | .05046 | 820.746 |
| R4 | 97.79% | 92.50% | .20280 | .05019 | 834.053 (RBPF estimate) |

All four linear axes, nonlinear log-total and both protocols are in `smoothed_fit.csv`, `onestep_metrics.csv` and `onestep_fixture.csv`; per-team/week residuals and worst-15 tables are supplied. R0's log-total smoothed R² is −.0732: transforming smoothed log-rate means is not the same as taking the predictive expectation of log-total.

**Weekly evolution earns its place.** R1 supplies most static structure, but R2 markedly improves supremacy and level; honest supremacy RMSE falls about 26% relative to R1. Full-panel R2 weekly attack/defence SDs are **.03270 [.02892,.03681] / .03350 [.02952,.03786]** (90% intervals), with σ_obs **.05573 [.05109,.06056]** rather than R1's **.12601 [.12141,.13094]**. R1's break SD is **.10888 [.08646,.13995]**.

**Per-team HA adds almost nothing here.** R3 σ_γ is **.01294 [.00168,.02634]**; the honest supremacy RMSE improvement is only .00019 and level .00003. A positive lower credible quantile under a continuous positive-scale prior is not evidence against the exact-zero nested model.

**Heavy tails buy some density, not better supremacy point forecasts.** R4 σ_obs is **.04303 [.03665,.04939]**, ν **4.115 [2.907,6.630]**; honest-fit ν is **7.534 [3.833,23.594]**. Its 10b mean log density is .15622 for supremacy versus R3 .14444, and 1.47158 for level versus 1.45093; supremacy RMSE worsens. R4's **887.924** conditional smoothed likelihood is **not** the comparable marginal likelihood.

**Intervals are not jointly well calibrated:** 10b nominal-90% coverage is **82.42% supremacy / 97.80% level** for R2 and **80.59% / 97.44%** for R4. High smoothed R² does not make this a production-calibrated forecast model.

### R4 numerical diagnostics and limitations

20,000 particles; independent **prior Gamma(ν/2, scale 2/ν)** proposals, fresh per fixture/week and shared by its two sides. Resampling after the complete week at ESS < Np/2. Seeds **3906, 3907, 3908**, both protocols. `rbpf_week_diagnostics.csv` includes ESS and cumulative/event resampling counts for all 91 grid weeks; `rbpf_worst10_weeks.csv` lists ten worst weeks **per protocol/seed**, with IDs and named fixtures. There are 546 weekly diagnostics.

| Protocol | Max all-fixture linear-axis RMSE spread | Max mean-log-density spread | Full loglik spread |
|---|---:|---:|---:|
| 10a | .000392 | .001390 | 1.26173 |
| 10b | .000280 | .002993 | .88168 |

**Occasional particle collapse is severe:** minimum ESS **11.42 (10a), 2.36 (10b)** despite 20,000 particles; up to 43/38 resampling events. Aggregate metrics are fairly stable, but this does not certify individual worst-week accuracy. Season-open 10b spread reaches .002252 RMSE, .013278 mean log density and 4.35 percentage points coverage. These limitations are not hidden by averaging or dropping weeks; all seeds/subsets, including nonlinear MC sensitivity, are in `rbpf_seed_metrics.csv` / `rbpf_seed_spread.csv`. A better proposal or stronger particle-count study would be needed before operational use.

## Gamma / log-normal / log-t comparison

All densities are on λ, with the log-t Jacobian. The fixed static mean structure is not the weekly ladder.

| Family | CV sum log density | CV per observation | WAIC elpd | p_WAIC |
|---|---:|---:|---:|---:|
| Gamma | 314.180 | .303850 | 313.740 | 69.551 |
| Log-normal | 312.282 | .302014 | 311.758 | 69.114 |
| Log-t | 335.212 | .324189 | 335.319 | 73.007 |

**Log-t has the highest decision score**, but evidence is modest: advantage over Gamma **21.031 ± 10.353 nats (fixture SE)**, over log-normal **22.930 ± 12.540**. Gamma versus log-normal is **1.898 ± 3.113**, effectively tied. Do not treat the point ranking as a decisive universal family winner, or transfer the static-family result untested to weekly dynamics. Full-fit Gamma shape **61.70 [57.22,66.31]**; log-normal σ **.12817 [.12342,.13299]**; log-t σ **.10786 [.10118,.11476]**, ν **7.051 [5.135,10.661]**.

`mean_variance.csv` / `B_mean_variance.png`: interior-bin CVs .0357–.0520, edge bins about .131–.133. Wide edge bins and conditioning on an in-sample smoothed fit confound a simple variance law; this is not clean proof of either constant CV or quasi-Poisson. `family_qq.csv` / `B_family_qq.png` use each family's own standardised reference quantiles.

### Initial failure and correction

The initial `a4cb6364` full attempt passed the ladder/prediction stages but Gamma NUTS failed during step-size search with `Gamma: α > 0` at floating-point shape zero. Uniform linked-coordinate initialisation was unsuitable for the shape scale. Training-only ridge log-rate initialisation plus seeded chain jitter fixes the start without changing priors, likelihoods, budgets or thresholds; held-out-response perturbation tests prove initialisation does not leak. Full-budget Gamma preflight at `2fcf355a` reached `R02_FAMILY_PREFLIGHT_DONE`; the subsequent fresh full run completes all 33 fits. Supplemental preflight CSVs are retained, but are not substitutes for full-run evidence.

## Rate EDA and realised goals

`rate_distributions.csv` includes n/mean/SD/5/25/50/75/95% by tournament, season, calendar month and tournament-season. Overall corr(log λ_h, log λ_a) is **−.7597**. Mean log supremacy/home advantage is **.16089 [.10791,.21466]** in tournament 56, **.16347 [.10631,.21913]** in 57 (2000-replicate fixture-bootstrap 95% intervals). Coverage holes are explicit in `excluded_books.csv`, `excluded_by_week.csv`, `excluded_by_team.csv`; no silent 1X2-only reinstatement.

Full-time goals are **`ds.matches.home_score` / `away_score`**, joined on all 517 fixtures without missing goals. Mean inverted/realised goals: **home 1.5011 / 1.4662; away 1.2799 / 1.2456; total 2.7809 / 2.7118**.

| Poisson GLM | Intercept (Fisher SE) | Slope (Fisher SE) |
|---|---:|---:|
| Home | .03943 (.07612) | .85317 (.15742) |
| Away | .06078 (.05774) | .66976 (.16446) |
| Total | .36992 (.30897) | .61463 (.30070) |

Overall levels are close (about 2.5% total overprediction), but **do not claim perfect goal calibration**: the away slope is about 2 SE below 1, weak evidence of over-sharp away-rate spread; home/total slopes are imprecise. These are Poisson Fisher SEs, not overdispersion-robust SEs. Decile calibration has 2000-replicate fixture-bootstrap CIs (`goal_calibration.csv`, `B_goal_calibration.png`).

The grid's documented market misfit has **no measurable outcome-loss penalty on this sample**:

| Market | n | Grid log loss | De-vigged close log loss | Paired difference, 95% bootstrap CI |
|---|---:|---:|---:|---|
| 1X2 | 517 | 1.052601 | 1.052902 | −.000301 [−.003762,.003245] |
| O/U 2.5 | 353 | .689812 | .689816 | −.000004 [−.001436,.001378] |

This is not evidence that the independent-Poisson grid is structurally correct. Outcomes/paired losses are retained in `rates_vs_goals_fixture.csv`; summary in `rates_vs_goals.csv`. Four PNGs are supplied under `results/figures/B_*.png`.

## Corrections / ambiguities in the design and brief

1. Linear d/ℓ predictions are Gaussian-exact only for R0–R3. R4 is Gaussian conditional on scales; integrating past-scale uncertainty requires a mixture. The human authorised RBPF and the added limit/ESS/seed diagnostics.
2. “Mean structure” is not literally common across the specified families: Gamma has E[λ|η]=exp(η); log-normal has exp(η+σ²/2); **an untruncated log-Student-t has no finite positive λ moments at any finite ν**. Its exp(η) is a median/log-location, not a rate mean. The requested densities are implemented exactly, not silently mean-adjusted or truncated. This matters before using that law generatively in Phase C.
3. Student-t log-rate predictive variance is undefined when ν≤2 (the prior allows it). All fitted plug-in medians here exceed 2; finite Monte Carlo variance must not be mistaken for existence of the true moment. θ uncertainty is not integrated in the requested plug-in protocols.
4. R4 smoothing at a chosen ω and its conditional likelihood must not be labelled as scale-integrated quantities; both labels/likelihoods are retained explicitly.
5. Fixture CV is retrospective decomposition, not a walk-forward family test. Fixture-level SEs satisfy the brief but do not account for additional serial dependence.

## Handoff / blocked

No unresolved blocker. Both fresh full runs, seeded byte comparison, corrected figure re-render and tests pass. `./scripts/todo.sh check` and `git diff --check` pass. TODO 039 remains IN_PROGRESS because Phases C/D are not authorised or completed; only Phase B is handed off.

The dedicated beast `pi_mm_b:0.0` session is left running solely for the manager's acceptance rerun in **clean pane `%248`**, Julia 1.12.4, 16 threads, BLAS=1, without model/data includes. Its checkout is at the figure-fix SHA `9469f327`; choose the recorded statistical SHA for exact provenance-hash comparison. The owned laptop Julia pane was closed; the unrelated Claude pane was left alone. Other **beast** sessions were untouched. A laptop ownership incident was disclosed during implementation: one test-include command was accidentally sent to the unexpected Claude pane `%13` before detection. Subsequent commands targeted verified owned Julia panes only; `%13` was not killed or otherwise operated. This incident is not being concealed by a blanket “all other sessions untouched” claim.

# Market-model experiment and rerun procedure

This stream implements the human's handwritten market-model note
(`~/Downloads/market_model_2026-10-05.pdf`), agreed on 2026-10-05. Instead of fitting
team strength to goals, first invert the market close to match-specific rates,
then ask how those rates decompose into attack, concessions and home advantage.
It is not a forecasting or betting-performance study.

[`DESIGN.md`](DESIGN.md) is the scientific contract. TODO 039 stays open until the
remaining phases are authorised and reviewed.

## Questions by phase

- **A:** can a double-Poisson grid reproduce the different quoted markets, with
  isolated KL inversion? What is lost when lines are held out?
- **B:** how much of the inverted rates is static team structure, weekly evolution,
  home-only team advantage or heavy-tailed fixture noise? Which static rate density
  fits, and how do the inverted rates relate to realised goals?
- **B2:** does the supremacy/level interval mismatch come from transient fixture
  shocks, persistent correlated team innovations, or both? Make the existing pipeline
  config-driven without changing the published A/B results. The human's mid-run
  addendum also asks whether innovations are quality-dominated, whether team
  levels have the same correlation, and whether level errors share a weekly shock.
- **B3:** refiltered FFBS tail/dependence PPCs for R6, and global-dependence
  Dixon–Coles, bivariate Poisson and Frank/Poisson score-grid comparisons.
  These are diagnostic evidence only; Phase C remains double Poisson.
- **C (revision8 fast rungs accepted; C1 IN PROGRESS, full books only):** C0/H1/H2
  have converged both protocols; two fresh fast runs match29/29 CSVs/PNGs.
  Frozen recovery%265 and unpromoted C1 production%267/%268 remain running;
  do not restart, kill, check out or reload their source. No C1 posterior/recovery
  conclusion yet. C2 remains deferred. Current evidence:
  `results/C/PHASE_C_PROGRESS_REPORT.md` and `results/C/HANDOVER.md`.
- **D (not implemented here):** compare isolated/pooled fits and team paths/ratings.

## Pipeline as executed

```text
pinned DataStore.betfair_odds
  -> configured TWA close -> complete markets + overround gates -> de-vig
  -> per-fixture KL double-Poisson inversion -> inversion/refusal gates
  -> accepted log-rate panel -> exclude thin books for the structural ladder
  -> weekly Kalman ladder R0–R4 (B)
  -> refitted R2 / covariance rungs R5–R7 (B2)
  -> convergence gates -> smoothing + pre-week 10a/10b -> diagnostics/figures
  -> B3 grid profiles/book checks + R6 rate refits + refiltered FFBS tail PPCs
```

All paths below are relative to this directory. Loaders define mathematics;
runners expose the scientific workflow. No step writes to either database.

| Step | Inputs and outputs | Gates and seeds |
|---|---|---|
| Snapshot / close | `MarketModelConfig.segment`, pinned `.cache/datastore_<Segment>.jls`; configured price window; `l01_market_model.jl:closing_book` | `max_age_hours=10^6`; Scottish cache hash below; no cache refresh |
| Book gates | TWA odds; `results/A/market_refusal_details.csv`, `market_refusals.csv`, `book_effect.csv` | Exact expected selection set/count for each market; raw implied-probability sum in [0.90,1.10] **before** renormalisation |
| KL inversion | Gated, de-vigged 1X2/BTTS/O-U books; `results/A/rates.csv`, `coverage.csv` | Complete 1X2, at least 3 selections; configured explicit exclusions; dispersed deterministic BFGS starts; all starts converge; summed KL ≤0.01; rates in [0.05,6]; score-grid cutoff 10 |
| Inversion diagnostics | Same-book SSE control, TODO 023 Scottish SSE reference; `results/A/kl_vs_sse*.csv`, `ungated_reproduction.csv`, `optimum_check.csv`, `selection_residuals.csv`, `line_residuals.csv`, `heldout_line_test.csv`; `results/figures/A_*.png` | Multi-start spread ≤1e-6; held-out inversion on 1X2+O/U2.5; same-fixture full-book comparison; fixture bootstrap 2000 reps, seed 3901 |
| Structural panel | Phase A rates; `l02_two_stage.jl:phase_b_panel`; B writes `results/B/market_targets.csv`, `refusals.csv`, `excluded_*.csv` | Configured minimum selections and optional count assertions; ordered home/away pairs; Monday weekly grid; legacy `sse` column contains **KL**, not SSE |
| B ladder | R0–R4 exact conditional Kalman/FFBS engine; `results/B/convergence.csv`, `engine_gates.csv`, `ladder_summary.csv`, `smoothed_fit.csv`, `onestep_metrics.csv`, `onestep_fixture.csv`, residual and RBPF tables | Rhat ≤1.05, bulk/tail ESS ≥200; batch Gaussian gates; seed 3902 with rung/protocol offsets; R4 prediction seeds 3906/3907/3908 |
| B static families / outcome EDA | Same panel; `results/B/family_*.csv`, `nuts_diagnostics.csv`, rate distributions, goal calibration and rates-vs-goals tables; `results/figures/B_*.png` | Compiled/fresh AD gates; all full/CV fits converged; divergence fraction ≤0.001; fixture CV seed 3903; λ-scale Jacobians; seeded fixture bootstraps |
| B2 covariance ladder | Same panel; `l03_covariance.jl`; `results/B2/engine_gates_b2.csv`, `convergence_b2.csv`, `ladder_summary_b2.csv`, `smoothed_fit_b2.csv`, `onestep_metrics_b2.csv`, `onestep_fixture_b2.csv` | R2/R5 isotropic and R2/R6 zero-correlation reductions ≤1e-9; independent original-side batch likelihood ≤1e-9 and RTS ≤1e-8; R6 FFBS moment gates; seed 3922 +100×rung +1 for 10b; 4×(2000+3000), thin 1 |
| B2 decision diagnostics | Fixture predictions; `results/B2/residual_correlation.csv`, `persistence.csv`, `paired_vs_r2.csv`, `noise_equivalence.csv`; `results/figures/B2_coverage.png`, `B2_persistence.png` | Same-fixture comparisons; team-signed within-season lag pairs sorted by date/ID; whole-team bootstrap 2000 reps, seed 3924; prediction MC seed 3906, smoothing MC seed 3923, 4000 draws |
| B2 human addendum | Already-converged R2/R6/R7 draws and full-panel RTS means; all rungs' scored level errors; `results/B2/rotation_quality_style.csv`, `league_shock_test.csv` | No new fits; all theta draws for rotation intervals; season-average RTS point levels across active accepted-panel teams; signed unbalanced one-way ANOVA ICC, whole-week bootstrap 2000 reps, seed 3928; synthetic shock tests seed 3929 |

### Filtration and interpretation

10a estimates hyperparameters on the full panel: retrospective plug-in evaluation.
10b estimates them only on `honest_train`, then filters the full chronological
panel and scores `honest_test`. All predictions for a week precede **every** update
in that week. Test-season observations can inform subsequent weeks, not their own
week. Hyperparameter uncertainty is not integrated in either protocol.

“Warm” drops the first three panel weeks; “season-open” is each season's first
three observed weeks. “In-season” is its complement. The season boundaries come
from the configured seasons and observed dates, not a fixed week number. R2 has
ordinary weekly innovations through the summer gap, **not** a special break jump;
R1 alone adds a learned boundary jump.

R5 rotates observations to supremacy `d=h−a` and level `ell=(h+a)/2`, with separate
noise. The rotation determinant is 1. Equivalent side noise has
`s²=(sigma_sup²+4sigma_lev²)/4` and
`rho_epsilon=(4sigma_lev²−sigma_sup²)/(4sigma_lev²+sigma_sup²)`.
R6 correlates each team's attack/concessions innovations, not different teams;
initial states remain independent. R7 combines these mechanisms. The correlation
prior is Uniform(−1,1), sampled on atanh with its Jacobian. Scale priors and the
shared numerical linked-coordinate bound are documented in the loaders.

The B2 paired score is exactly the requested **sum of supremacy and level marginal
log predictive densities**, paired by fixture, with a fixture SE. It is not the
joint Gaussian predictive density: latent-state uncertainty can correlate the
rotated axes. Serial dependence is not removed by that SE. Persistence uses a
whole-team bootstrap, not a fixture bootstrap; opponents still share residuals,
so it is a diagnostic, not a causal decomposition.

Linear smoothing uses RTS means at median theta. B2 log-total smoothing averages
its nonlinear transform over the smoothed latent Gaussian by MC; B's historical
log-total smoothing transformed the RTS mean instead. Both published B files and
all other B measures remain untouched. Pre-week nonlinear log-total uses the B
MC/KDE convention. High smoothed R² is not evidence of honest interval calibration.

## Decisions and reasons (2026-10-05)

- **Human: isolated then pooled ladder.** Keep the measurement step observable before
  introducing team-structure borrowing. A pooled book likelihood is Phase C, not B2.
- **Human: double Poisson.** “Dixon–Coles” in the original A/B model refers to
  the attack/concessions/home parameterisation, not a low-score rho correction.
  B3 authorises a local diagnostic grid comparison only (including the low-score
  correction). **Double Poisson stays for Phase C; C2 is a later human decision.**
- **Human: KL(market || model).** Fit all quoted selections once per complete market;
  SSE is a same-book control, not the principal inversion.
- **Bad books:** require complete 1X2 locally; refuse the explicitly identified T014
  swapped book. Do not repair T014/T015 in `src/` in this stream.
- **Human/manager: exclude accepted 1X2-only books from B's ladder.** The Phase A
  comparison found a material totals bias; preserve those books in A's coverage/EDA,
  rather than hiding their absence or changing their inversions.
- **Human: beast for every sampling run, laptop for deterministic tests/inversion.**
  Own persistent tmux REPLs, target pane IDs only. Use 8 beast threads during the
  manager's Phase B rerun; do not alter any manager/human session.
- **Human: R4 RBPF approved.** Conditional Gaussian updates do not integrate past
  Student-t scale uncertainty. Fresh fixture-shared prior-Gamma particle proposals,
  plus limit/ESS/seed diagnostics, were authorised for Phase B; not carried into B2.
- **Human: B2 follows R2 only.** Per-team HA and heavy tails were small gains;
  covariance directly addresses the supremacy/level coverage mismatch. Phase C's
  GRW/fixture covariance remains subject to human sign-off.
- **Human mid-run addendum:** inspect quality/style innovation scales and a common
  weekly level-error component without new fits. The manager answered pi's question on
  the definition for the human (the ambiguity was in the manager's addendum wording):
  the per-season levels correlation uses season-average RTS means across active teams,
  as a point statistic; only the innovation rotations require posterior intervals.
  Group/tier mixture hierarchies remain for Phase C.
- **Human: Phase A runtime exception.** Reproduce every non-runtime CSV byte-for-byte;
  wall-clock `runtime.csv` is excluded from that regression, not overwritten with a
  fake elapsed time. Tests use temporary output directories, preserving published A/B.

## Results so far

Every numerical result below comes from a saved CSV; the paths/columns are explicit.
The reports carry fuller tables, caveats and source SHAs.

**A:** `results/A/coverage.csv` records 710 panel fixtures, 78 with no book,
36 without a complete 1X2, 1 configured swapped book, and 595 accepted.
`results/A/line_residuals.csv` (insample, tournament=0, n_markets=0, 1X2/draw)
records mean grid-minus-market probability −0.013354049014150333.
`results/A/heldout_line_test.csv` (heldout, tournament=0, n_markets=0, BTTS/yes)
records mean −0.011972283886809642 on 111 fixtures. The grid is not exactly coherent
with the quoted close. See `results/A/PHASE_A_REPORT.md`.

**B:** `results/B/ladder_summary.csv`, R2 row: smoothed supremacy R²
0.9806140049456532, level R² 0.9251974628245537; honest supremacy RMSE
0.19555570200778602, level RMSE 0.05049170372573143. In
`results/B/onestep_metrics.csv` (R2, 10b, all), nominal-90% supremacy coverage is
0.8241758241758241 and level coverage 0.978021978021978. These motivated B2,
not a claim that R2 was already calibrated.

**B2:** R6 (correlated team innovations) has the highest honest point score.
`results/B2/paired_vs_r2.csv` records its 10b mean fixture-score gain
0.18491913672992602 with SE 0.03849175105895215; R7's gain is
0.18188621186334808 with SE 0.037623351090194566. In
`results/B2/onestep_metrics_b2.csv` (R6, 10b, all), supremacy/level coverage is
0.9010989010989011 / 0.9120879120879121. R6's full-panel innovation correlation
median is −0.8219351600992229 (`results/B2/convergence_b2.csv`, rho_ab, 10a).
Adding fixture correlation gives no clear additional benefit: R7's full-panel
rho_epsilon median is 0.005304793708635777, with 90% interval
[−0.16126436889725151, 0.19277272301680717]
(`results/B2/noise_equivalence.csv`). Team quality is the supported covariance
mechanism here, not a proof that all fixture deviations disappear.

Full results and acceptance evidence are recorded in
[`results/B2/PHASE_B2_REPORT.md`](results/B2/PHASE_B2_REPORT.md). These are new
artifacts, not replacements for A or B. R6's honest level RMSE is slightly worse
than R2's despite its better density/coverage. Read the recommendation for Phase C
with the persistence intervals and paired-score limitations, not smoothed fit alone.

### Quality/style and common-weekly-shock addendum

`results/B2/rotation_quality_style.csv` transforms every retained posterior draw:
`q=(alpha−beta)/2`, `s=(alpha+beta)/2`,
`var(Delta q)=(sigma_att²+sigma_def²−2rho*sigma_att*sigma_def)/4`,
`var(Delta s)=(sigma_att²+sigma_def²+2rho*sigma_att*sigma_def)/4`.
The CSV includes both variances, both SDs, their ratio and the innovation
correlation `(sigma_att²−sigma_def²)/(4sigma_q*sigma_s)`.
For the **full** axes, retain the static terms:
`d=gamma+2(q_h−q_a)`, `ell=mu+gamma/2+s_h+s_a`; the brief's identities are
team-state contributions, not the entire linear predictor.

The headline R6 style/quality SD ratio is **0.3153399879334282**
[0.26234345756667427, 0.37452012202244495] in 10a and
**0.2824111332469862** [0.18609709846977368, 0.396349685252396] in 10b
(`rotation_quality_style.csv`, style_quality_ratio). R2's ratio is exactly 1;
R7 is similar to R6. This is quality-dominated evolution, not an exactly
one-dimensional process or evidence for a discrete tier/group mixture.

Level correlation is a different statistic: average each active team's centred
RTS alpha and beta over all grid weeks between the season's first and last
accepted fixture, then correlate these team averages. “Active” means observed
in accepted-panel fixtures that season; unobserved prior-only team states are
not presented as estimated levels. These are point summaries at median theta,
not posterior-path intervals; 10b smoothing is descriptive and uses the full
panel, unlike its honest theta fit. R6 10a correlations are
−0.8196911956533606 (24/25) and −0.8563675634245145 (25/26); R2 already has
−0.834790094507127 and −0.8630984129327514 despite independent innovations
(`rotation_quality_style.csv`, scope=levels). Correlated fitted levels do not
by themselves imply correlated innovations.

`league_shock_test.csv` uses all scored level residuals, pooling the configured
tournaments, including singleton and opening weeks. For unequal week sizes,
`n0=(N−sum(n_week²)/N)/(K−1)`,
`var_week=(MS_between−MS_within)/n0`, and
`ICC=var_week/(var_week+MS_within)`. Signed method-of-moments estimates are not
truncated to zero. Resample whole week clusters; duplicate sampled weeks become
separate bootstrap groups. R6 10a ICC is **0.0976021529070118**, with 95% CI
[0.010906144777427575, 0.17912775940436865]; honest 10b is
**−0.005788309214321687** [−0.07669348278413401, 0.05821120054758495].
The retrospective panel clusters by week, but the honest sample does not support
an added common weekly component. These raw residual ICCs can also reflect common
filtering/intercept uncertainty or other dependence; a positive ICC alone does
not causally identify a dynamic league intercept.

**Literature checked in the research library:**
- `koopman2013-dynamic-bivariate-poisson-model-analysing`, source-PDF p. 7,
  eq. 4 assumes independent normally distributed attack/defence disturbances.
  Our innovation test probes that independence on market rates, not on their goal
  model. Their beta denotes defensive strength and enters with a minus sign;
  ours denotes concessions, so the sign convention is reversed.
- `baio2010-bayesian-hierarchical-model-prediction-football`, source-PDF p. 4
  gives exchangeable team effects; pp. 6–7 discuss overshrinkage and a three-group
  mixture. The quality/style and common-week diagnostics motivate questions about
  hierarchy; they do not fit or establish that mixture. Group/tier hierarchies
  remain out of this implementation.

### Limitations carried from Phase B

- Full-panel theta and fixture CV are retrospective, not walk-forward forecasts.
- R4's occasional prior-proposal particle collapse was severe even when aggregate
  metrics were fairly stable; its smoothed mean conditional on mean omega was not
  a scale-integrated posterior. B2 does not reuse its engine/results as a candidate.
- An untruncated log-Student-t has no finite positive rate moments; `exp(eta)` is
  its median, not a generative rate mean. Log-normal and Gamma locations likewise
  are not interchangeable rate means without accounting for the variance term.
- Goal-calibration slopes are imprecise; the Poisson GLM SEs are not
  overdispersion-robust. Grid misfit without an observed outcome-loss penalty
  does not establish structural correctness.
- Thin-book exclusion restricts the population; no claim transfers automatically
  to a different league, price window or line mix.

## Phase B3: team-step tails and score-grid evidence

B3 uses new artifacts only (`results/B3/`), preserving A/B/B2. Its local loader,
runner and tests are `l04_copula_grid.jl`, `r04_copula_grid.jl` and
`t04_copula_grid_tests.jl`; supporting definitions remain in local l04 helpers.

- **Gaussian dependence:** `results/B3/tail_ppc.csv` compares observed FFBS
  statistics with newly simulated, **refiltered** FFBS replicates at every theta
  draw. Quality-kurtosis p-values are **0.295 / 0.365** (10a/10b), directional
  asymmetry **0.535 / 0.540**, squared-quality lag-1 correlation **0.280 / 0.325**.
  No saved p-value is extreme; no alternative was fitted. This supports adequacy
  for these observable checks, not proof of no tails in a prior-dominated panel.
  `tail_ppc_draws.csv` preserves both distributions; `largest_quality_steps.csv`
  gives the largest posterior-median quality moves for the human's outside check.
- **Book reproduction:** `results/B3/grid_fit.csv` records total KL
  **0.600179 / 0.228247 / 0.327800 / 0.339442** for G0/G1/G2/G3, with final
  inversion gates passing on every accepted book. `grid_profile.csv` preserves
  coarse/refined profiles, including failed distant candidates. G1's global
  Dixon–Coles rho is **−0.064462526**.
- **Heldout lines:** `results/B3/grid_heldout.csv`, ALL/ALL rows, records pooled
  mean absolute error **0.008846 / 0.008648 / 0.010631 / 0.010946**. G1 is best
  by the predeclared point criterion, but its advantage over G0 is modest.
  `grid_line_residuals.csv` shows the draw deficit shrinking from **−0.013354**
  to **−0.001370**. The shared global parameter uses full books, so this is a
  heldout-line shape check, not independent fixture CV.
- **Thin-book bias:** `results/B3/grid_1x2only_bias.csv` reproduces G0's mean
  Delta log total **−0.102716**; G1 reduces it to **−0.008941**, with CI
  [−0.013462, −0.004813]. The mean is small, not exactly zero; G1's mean absolute
  bias remains **0.040676**. `grid_rate_shift.csv` and raw fixture tables record
  how the grids move rates; `grid_marginals.csv` documents finite-support error.
- **Outcomes:** `results/B3/grid_outcome_logloss.csv` gives paired fixture CIs,
  all crossing zero versus G0 and the quoted close. Grid/G0 losses use the whole
  structural set; OU2.5 close pairs use only its available quoted closes, not
  invented prices. Better book shape is not established outcome superiority.
- **R6 refits:** `results/B3/r6_on_best_grid.csv` retains all axes/protocols on
  G0, G1 and G1 including thin books. On the same population, 10b supremacy RMSE
  is **0.190775 / 0.191056** (G0/G1), level RMSE **0.053156 / 0.051795**. The
  covariance story remains; the extra thin-book row changes the evaluation set.
  G1 10b restricts team-theta fitting but its global grid parameter uses both
  seasons, so it is informational, not fully honest end-to-end evaluation.

Figures: `results/figures/B3_tail_ppc.png`, `B3_grid_profile.png`,
`B3_grid_residuals.png`. Definitions, literature/page corrections and limitations:
[`results/B3/METHODS.md`](results/B3/METHODS.md). Full acceptance evidence:
[`results/B3/PHASE_B3_REPORT.md`](results/B3/PHASE_B3_REPORT.md).

**Human decision: double Poisson for Phase C; C2 later.** B3 changes neither
Phase C's grid nor its implementation. Run the new runner in a fresh owned
beast persistent REPL and wait for `R04_DONE`; run t04/t03/t02 in a fresh owned
laptop REPL. Exact source, pinned cache, pane-ID commands and CSV hash procedure
are in [`results/B3/REPRODUCIBILITY.md`](results/B3/REPRODUCIBILITY.md).

## Phase C: blocked likelihood/projection preflight (2026-10-06)

**Human decision: double Poisson for C; C2 later.** The authorised pooled rung
named C2 is distinct from a later score-grid Phase C2; neither a grid change nor
Phase D was attempted. DESIGN §5 remains unchanged; the new brief selects a
quality/style GRW and collapsed slice sampler instead of its old Turing route.

Before implementing that sampler, `l05_laplace_preflight.jl` and
`r05_laplace_preflight.jl` tested necessary mathematical conditions on the pinned
close. These are **not** the requested pooled loader/runner or acceptance suite.
`results/C/laplace_preflight_summary.csv` records, at the **prior** median
n=1000 and side prediction SD=0.20, full-book median absolute marginal error
**0.013040970693098508** (limit 0.01); supremacy-only 1X2 median
**0.7551135925625077**, p95 **0.9336569033260274** (limits 0.01/0.05).
The unprojected 1X2 median error is only **0.0004892284282393966**.
`totals_projection_preflight.csv` records **7/36** books satisfying the specified
rank-one/10-degree condition; BTTS plus O/U can identify two directions.

Two issues precede sampling: Dirichlet(p; n q)'s large-n loss is KL(q||p), not
Phase A's KL(p||q); discarding identified book information cannot reproduce its
exact likelihood integral against an arbitrary Gaussian prediction. Totals
identify a curved constant-rate-sum contour, not a fixed equal-weight direction
in log-rate coordinates. No threshold was loosened and no corrective model was
silently substituted. See [`results/C/PHASE_C_REPORT.md`](results/C/PHASE_C_REPORT.md)
for evidence, caveats, missing acceptance items and the required manager decision.

Rerun only the diagnostic preflight through an owned laptop tmux REPL:
`include("current_development/market_model/t05_laplace_preflight_tests.jl")`
then `include("current_development/market_model/r05_laplace_preflight.jl")`.
Expect `T05_PREFLIGHT_DONE` and `C_PREFLIGHT05_BLOCKED`, **not** `T05_DONE` or
`R05_DONE`. No Phase C production or posterior claims are available.

## Phase C revision 2: corrected likelihood, failed posterior-mean gate

**Human decision: double Poisson for C; C2 later.** The pooled rung C2 is not
that later score-grid phase. No Phase D, grid change or pooled sampling was run.

Revision 1 incorrectly used KL(p||q) as the Dirichlet large-n loss, tested a
supremacy projection against an unprojected likelihood, and treated a curved
constant-total contour as a fixed direction. Revision 2 uses the exact density,
including normalisers, and a Newton joint mode of likelihood × Gaussian prediction.
Phase A's KL(p||q) inversions remain **scoring targets**, not expansion points.
The manager's logged decision is to retain totals/BTTS-only likelihoods exactly:
O/U constrains the total along a curved contour, while BTTS+O/U may carry weak
supremacy information. 1X2-only books use a flat-measure level-integrated likelihood.

At source `7b6c7659`, `l05_pooled.jl`, `r05_pooled.jl` and `t05_pooled_tests.jl`
contained the revised update, fail-fast accuracy runner and necessary unit checks,
not a completed production or recovery pipeline. `results/C/v2_gate/laplace_gate.csv`
and `v2_gate/laplace_gate_fixture.csv` preserve the first 30 full books, all prescribed n,
spreads and offsets. Marginal accuracy passes; the moment gate fails. The report
records exact numbers and missing acceptance items; no forecast, shrinkage,
concentration, team-path or HA posterior conclusions can yet be made.

The revision-1 report and reproduction notes are retained as
`results/C/PHASE_C_V1_REPORT.md` / `REPRODUCIBILITY_V1.md`; its preflight source and
CSVs remain unchanged. Revision-2 evidence and commands:
[`results/C/PHASE_C_V2_REPORT.md`](results/C/PHASE_C_V2_REPORT.md),
[`results/C/REPRODUCIBILITY_V2.md`](results/C/REPRODUCIBILITY_V2.md).
At that source, a fresh runner stopped at `C05_LIKELIHOOD_BLOCKED`, not
`R05_DONE`; `T05_UPDATE_DONE` denoted only necessary deterministic checks.

## Phase C revision 3: mean fixed, thin-book gates blocked

The manager's addendum is copied unchanged to
`../../experiments/pi_market_model_phaseC_v3_addendum.md`. The loader derives
and implements the cubic skewness mean correction using third derivatives of
the likelihood alone. The revision-2 log-marginal and covariance are unchanged:
all 360 full-book marginals compare identically with their saved revision-2
values. Full-book mean error now has maximum **0.0009284497092163062** of SD,
within the unchanged 0.05 threshold (`results/C/laplace_gate.csv`).

All book types were attempted, but **the complete all-types gate did not finish**.
At n=250, prediction SD=.20 and no offset, O/U-only SD error reaches
**0.058978186176263714** (limit .05); BTTS+O/U median absolute marginal error
is **0.02060874306987226 nats** (limit .01). Independent orders 32/64 resolve
these failures. High-n thin-book quadrature is less stable and is labelled,
not used to dismiss the well-resolved low-n failures. BTTS-only has one available
book and passes its completed settings. The 1X2-only integrated solver stalled;
its revised adaptive-integral third-derivative check then failed twice.

Current necessary tests: **129 pass / 1 fail / 130 assertions**, not pooled
acceptance or synthetic recovery. The current adaptive level-integrated code
is unvalidated. No sampling, convergence, forecast/shrinkage/ratings/HA posterior
or production reproducibility claims exist. Human decision remains **double
Poisson for C; C2 later**; no Phase D or later score-grid phase was attempted.

Evidence, numerical failures, missing acceptance and stop rationale:
[`results/C/PHASE_C_REPORT.md`](results/C/PHASE_C_REPORT.md),
[`results/C/REPRODUCIBILITY.md`](results/C/REPRODUCIBILITY.md).
The current source must not be promoted just because the full-book mean is fixed.

## Phase C revision 4: authorised full-book continuation

Manager scope: `../../experiments/pi_market_model_phaseC_v4_scope.md`.
**Double Poisson for C; later score-grid C2 and Phase D remain unauthorised.**
C0/C1/H1/H2 use only the same 517 full books and isolated scoring targets.
C0/H1/H2 are exact Gaussian filters; C1 uses the exact Dirichlet/Beta density
with an approximate sequential Laplace collapse, revision-3 mean correction,
unchanged revision-2 marginal/clipping and approximate RTS. No thin book enters
these filters. New priors are on sigma_q/sigma_s, not R6's attack/defence/rho prior.
H1 prior clarified by the manager: independent gamma_att N(.15,.25²) and
new gamma_def N(0,.25²). Pinning gamma_def=0 recovers C0.

`l05_fullbook_engine.jl` defines the state engines, conditional identities,
Gaussian/frozen-factor batch checks and smoothing. The included
`l05_fullbook_sampling.jl` adds dedicated slice chains (no R6 prior/±12 box),
ID-joined honest restriction, full-book synthetic generation and exact log-space
Dirichlet storage. Seeds are frozen in `fullbook_seeds.toml`.
`r05_fullbook_recovery.jl` is the beast-only prerequisite, not production.

**Recovery is BLOCKED at statistical source `9de98548`.** Same frozen generation
seed 3962 and full population: the FD attempt failed a joint-mode line search;
synthetic-only AD failed the same check, chain seed4964 at warmup8. Root gradient
[-1.5699131339808048e-5, 3.249019587192592e-5]; peers cancelled. Stop without a
third variation or threshold/seed/prior/data change. No fit/intervals retained.
The preceding probability-underflow issue was resolved by exact log-Gamma/logp
storage, not clipping or selection; the latter two generated panels are
byte-identical. Passing real-book Gate1 does not establish robust recovery.

Final fresh accepted deterministic t05 **270/270**, t04 **92/92**, t03 **131/131**,
t02 **131/131**. C2-pending separately reports **29 pass/10 fail**, excluded by
scope, never labelled passed. `r05_pooled.jl` remains a deterministic preflight:
Gate1 **12/12**, engines **53/53**, all four fixed filters; its six v4_preflight
CSVs regenerate byte-identically to the engine checkpoint. Marker
`C05_C4_ENGINE_PREFLIGHT_DONE`, **not R05_DONE/T05_DONE**. These identities are
not two production runs. Recovery, training, convergence, measures/figures and
production reproduction remain unfinished. Evidence and stop boundary:
`results/C/PHASE_C_REPORT.md`, `REPRODUCIBILITY.md`, `HANDOVER.md`.
Engine-only checkpoint documents are preserved with `_ENGINE_CHECKPOINT` names.

## Phase C revision 5: termination implemented, exact-stall regression fails

The manager's `pi_market_model_phaseC_v5_solver.md` authorises termination only:
Newton decrement <=1e-12 or step infinity norm <=1e-10; on Armijo stall accept
only decrement <=1e-9. The solver now records all three termination counts and
maximum accepted decrement, with a fail-loud <=1e-9 gate. Density, derivatives,
clipping, marginal/correction formulas, scientific settings and seeds are unchanged.

**BLOCKED before recovery sampling.** A test-only replay of the original solver
on the frozen generated panel and seed4964 warmup8 coordinate finds fixture
**12476625**, reproducing the exact handover gradient. The authorised rule accepts
`stalled_converged`, decrement **3.382117013018548e-12** nats, but tighter damped
AD Newton differs by **4.0159674463691175e-7** in mode (limit1e-8) and
**6.658597637709818e-7** nats in Laplace marginal (limit1e-9).
Reference residual **7.670308832530282e-12**; log-determinant shift
**1.3317370690657526e-6** dominates the tiny objective change. A decrement
bounds local objective gain, not first-order Hessian/logdet displacement.
Sources: `results/C/v5_newton/newton_regression.csv`, `newton_regression_book.csv`.

Fresh accepted-boundary t05 is **282 pass / 2 fail / 284**, not T05_DONE.
The same two required checks failed in focused and aggregate runs; no third
numerical variation, threshold change or beast sampling was attempted.
`results/C/newton_termination.csv` labels its two solves `exact_regression`,
not recovery/production. Review of termination/accuracy compatibility is the
next dependency. Full-book scope needs no renewed approval. Historical revision4
reports are preserved as `PHASE_C_V4_REPORT.md`, `REPRODUCIBILITY_V4.md`,
`HANDOVER_V4.md`; v4 and prior-phase scientific artifacts are unchanged.
See current `results/C/PHASE_C_REPORT.md` and `HANDOVER.md` for checks and next steps.

## Phase C revision6: exact-case polish passes, recovery iteration-limit blocked

Manager brief copied unchanged to `experiments/pi_market_model_phaseC_v6_polish.md`
at the repository root. Tolerances and scientific settings remain unchanged.
On qualifying Armijo stalls, at most3 undamped Newton steps use derivatives only;
stop on step infinity-norm<=1e-12 or nondecreasing gradient norm, and fail if final
norm exceeds its pre-polish value or finalδ exceeds1e-9. Counts are `polished`.

Fixture12476625 at the frozen seed4964-coordinate now needs **1** polish step:
mode gap **2.0039525594484076e-13** (limit1e-8), marginal gap
**1.1574741165532032e-11** nats (limit1e-9), acceptedδ
**1.50342686280365e-24**. Source: `results/C/v6_newton/newton_regression.csv`.
Fresh owned %56 t05 **289/289**, T05_C6_DETERMINISTIC_DONE (not T05_DONE);
C2-pending unchanged29pass/10fail, excluded. t04 **92/92**, t03/t02 **131/131**.
Full real Gate1 **12/12**, state gates **53/53** and all four517-fixture fixed
filters/smoothers regenerate in `v6_preflight/`. These are prerequisite checks,
not posterior fits or production reproduction.

**Recovery BLOCKED at source d9e6c003:** seed4961 initial target exhausts100 Newton
iterations; other chains cancel at warmup1, no retained fit/intervals. An observer-
only trace (04e67fa8) identifies fixture12476686: calls7–100 have zero movement,
δ **1.035864027765691e-12** (>1e-12) and step **4.189252610998754e-8** (>1e-10).
Armijo accepts unchanged candidates after roundoff, so its exhaustion/polish branch
never runs. Gradient norm **4.9497130117024364e-5**; raw min precision eigenvalue
**965.3501213272148**. Sources: `v6_diagnostic/newton_mode_trace_summary.csv`,
`newton_mode_trace.csv`, `newton_trace_activity.csv`.
Stop without a solver variation, iteration/tolerance/seed/prior change or recovery
retry. Review zero-motion stall recognition first. Both owned %56/%264 closed.
Root `newton_termination.csv` now labels aborted recovery:1696decrement,1polished,
max acceptedδ9.743716301157053e-13; these counts do not imply accepted chains.
Generated binary remains byte-identical to revision4. No further priorities or
production claims were promoted. Revision5 evidence is archived with `_V5` names.

## Phase C revision7: exact zero-motion case passes; recovery ACTIVE

Manager brief `experiments/pi_market_model_phaseC_v7_stall.md` authorises
accepted-step stall recognition (<1e-14 movement OR equal Float64 density),
reusing the unchanged <=3-step polish/error rules. Statistical source6335316f.
Exact fixture12476686: mode/marginal gaps **0.0/0.0**, both from its recorded
mode and original prediction;1 `zero_motion_polished` step,
delta2.287082719929954e-26. Old fixture12476625 still passes unchanged limits.
Sources: `results/C/v7_newton/`; fresh t05 **305/305**, C2-pending29pass/10fail
separately excluded; t04 **92/92**,t03/t02 **131/131**. Full real Gate1 **12/12**,
state gates **53/53** and four fixed full-panel filters/smoothers pass.

**Recovery is ACTIVE, not passed/blocked:** owned beast **%265**, dedicated
`/root/BF_runs/market_model_c`, detached6335316f,16 threads/core pinning/BLAS1.
Frozen seeds/budgets/truth/population unchanged; all four chains reported warmup200
at checkpoint. Generated binary remains byte-identical to v4/v6. DO NOT restart,
kill/check out/edit running code or launch production before recovery acceptance.
Original log `logs/market_model_c/6335316f_recovery1.log` on the beast.
See current HANDOVER/REPRODUCIBILITY for exact monitor commands and preservation.

Prepared `l05_fullbook_workflow.jl` / `l05_fullbook_reports.jl` add conditional
structure+u smoothing and B2-compatible pre-week scoring/paired C0 measures;
fresh workflow tests **85/85** (independent augmented-Gaussian theta/u gates,
whole-week filtration, schema/input guards). Mock IID hyperdraws are NOT fits.
New `r05_fullbook_core.jl` C0/C1 STAGE is parsed, not executed; it rechecks
recovery draws/intervals/hash before any sampling and prioritises10b. Its marker
is R05_FULLBOOK_CORE_DONE only. H1/H2 posterior uncertainty/figures, remaining
measures, full production workflow and two-run reproduction remain unfinished.
No learned parameter or forecast-benefit conclusion exists. r05_pooled remains
preflight. Local panes close; active beast recovery is transferred at handover.
Revision6 report/repro/handover/source evidence preserved with `_V6` names.

## Phase C revision8: fast rungs now, C1 separately unpromoted

Revision8 supersedes revision7's ordering restriction, **not** its scientific
settings or C1 promotion gates. Source70336b07: full-book C0/H1/H2, both protocols,
4×(2000+3000), thin1;107/107 inherited/state/scalar-parity gates and20/20 theta
convergence rows pass. Both fresh beast%270/%271 reach R05_FAST_RUNGS_DONE; **26 CSVs +3 PNGs are
byte-identical** (`results/C/BYTE_COMPARISON.txt` and both scientific manifests).

**C0 tracks published R6 closely:** 10b paired C0−R6 score
**+0.001285 ±0.004312 fixture SE**. Its supremacy/level coverage is
**90.476%/91.209%** (`results/C/v8_fast_1/onestep_metrics_c.csv`);
`paired_c0_vs_r6.csv` measures the fitted gap, separate from likelihood parity.
**Away suppression is not established:** H1 gamma_def is
**0.026292 [−0.248135,0.300720]**, with gamma_att/def confounded with mu.
Only their supremacy sum is sharply identified. **Quality-linked HA is not
supported:** H2 kappa10b **0.033860 [−0.048858,0.125508]**; honest paired H2−C0
**−0.002457 ±0.003596** (`home_advantage_rungs.csv`, `paired_vs_c0.csv`).
Static HA intervals mix conditional state uncertainty over ALL12,000 retained
hyperdraws; 10b HA uses training only. Forecasts/path bands remain conditional
at median theta as in B2; full-panel smoothing is descriptive.

The manager approved a verified Gaussian-only scalar Kalman callback after
measured dense throughput implied hours; native sampling/adaptation, all priors,
seeds and thresholds stay unchanged. C1's default path and live checkouts were
not edited. Dense-attempt and scalar plot-assertion evidence are preserved.
Ross County/Airdrie are absent from the fixed22-team56/57 population. Manager
approved labelled missing panels, not new data/fake paths; Kelty's actual key
is `kelty-hearts-fc`. Both available clubs and all22 teams' CSV paths remain.
Figures, formulas, source-checked Ridall2024 p.1/3 and limitations:
`results/C/C8_METHODS.md`, `results/C/v8_fast_1/C_fast_*.png`.

**Live, unpromoted:** recovery%265 at frozen6335316f in `/root/BF_runs/market_model_c`;
C1 10b%267 /10a%268 at41e28736 in `/root/BF_runs/market_model_c_fast`, started
04:50:29Z/04:50:32Z. All16 threads/core pinning/BLAS1, prescribed seeds/budgets.
At07:46:59Z, C1 10b/10a measured slowest-chain ETAs are15.6h/36.1h;
recovery's censored conservative ETA is54.6h (chain estimates16.2–54.6h).
These estimates are uncertain during adaptation/contention. Checkpoints, logs, proof and
remaining C1 work are in `results/C/PHASE_C_PROGRESS_REPORT.md` and HANDOVER.
No C1 interpretation until synthetic recovery AND production convergence pass.
No complete Phase C reproduction or pooled forecast/shrinkage/n conclusion yet.

### C2 deferred to the human

**C2 (thin-book pooling) is blocked: Laplace accuracy fails on thin books at realistic-to-wide prediction spreads.**
The integrated 1X2 third derivative also remains unvalidated. Historical
measurements remain under `results/C/v3_gate/` and `PHASE_C_V3_REPORT.md`;
no C2 solver change or tolerance relaxation is made. Options below are for the
human; none is selected:

| Option | What | Cost | Caveat |
|---|---|---|---|
| (i) exact moment matching (assumed-density filtering) | per-book quadrature of the exact likelihood × prediction: mean, covariance and normaliser, for the ~114 thin books only | slower sampling; feasible on the beast with parallel chains | exact up to quadrature order; needs its own order-stability gate |
| (ii) NUTS on the exact Dirichlet likelihood | the original DESIGN §4.4 route: all books, no approximation | hours of sampling; harder geometry (σ, n, u funnels) | exact; no Kalman collapse |
| (iii) gate at the realistic spread only | re-specify Gate 1 at the C1 posterior's actual prediction spreads | cheap | a threshold or setting change: the human's call |
| (iv) restrict thin-book types | e.g. O/U-only books only (they nearly pass), dropping the ~11 BTTS + O/U books | cheap | changes which data are used: the human's call |

## How to run

Do not update packages. `Distributions` is pinned at **0.25.126**; use the existing
`Manifest.toml`. Link `.env` for environment setup but never print credential-bearing
URLs. No SQL writes or `src/` edits are part of this experiment.

Scottish snapshot: `.cache/datastore_ScottishLower.jls`, SHA256
`c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4`.
Check with `sha256sum`; on the beast copy it with `cp -p` from
`/root/BF_runs/market_model_b/.cache/`. Preserving mtime also preserves A's
`snapshot.csv`. Always load with `max_age_hours=10^6`.

### Laptop: own tmux REPL, IDs only

From the worktree root:

```bash
PANE=$(tmux new -d -P -F '#{pane_id}' -s pi_julia_mm_b2 -c "$PWD" \
  'env JULIA_PKG_PRECOMPILE_AUTO=0 GKSwstype=100 julia --project -t 8')
printf 'owned laptop pane: %s\n' "$PANE"
tmux send-keys -t "$PANE" -l -- \
  'using LinearAlgebra; BLAS.set_num_threads(1); include("current_development/market_model/t03_covariance_tests.jl")'
sleep 0.15
tmux send-keys -t "$PANE" Enter
tmux capture-pane -t "$PANE" -p -J -S -80
```

Wait for `T03_DONE`, not a returned shell prompt. This test includes the full
config-driven A runner in a temporary directory (`R01_DONE`), frozen-source B
regression, covariance gates and synthetic persistence checks. Run `t02_two_stage_tests.jl`
in a fresh REPL for Phase B's 131-assertion regression (`T02_DONE`).
To run A alone, include `r01_kl_inversion.jl` and expect `R01_DONE`; unlike the test,
this replaces the A artifacts. The published `results/A/runtime.csv` records
38.179800033569336 seconds for the warm Phase A run; fresh compilation takes longer.

### Beast: dedicated checkout, own pane

Use `ssh root@mcmc-beast`, not the laptop's default username. Fetch the permitted
branch and create a detached worktree at `/root/BF_runs/market_model_b2`, not inside
another agent's checkout. Link `/root/BayesianFootball/.env` and `Manifest.toml`;
copy/check the pinned cache. Inspect load before sampling; the Phase B2 brief
sets a block threshold above 20 and permits 8 threads while manager review runs.

```bash
# On the beast, after setup and load check:
cd /root/BF_runs/market_model_b2
PANE=$(tmux new -d -P -F '#{pane_id}' -s pi_mm_b2 -c "$PWD" \
  'env JULIA_PKG_PRECOMPILE_AUTO=0 /root/.juliaup/bin/julia --project -t 16')
printf 'owned beast pane: %s\n' "$PANE"
tmux pipe-pane -t "$PANE" 'cat >> /root/BF_runs/logs/market_model_b2/run.log'
tmux send-keys -t "$PANE" -l -- \
  'using LinearAlgebra; BLAS.set_num_threads(1); include("current_development/market_model/r03_covariance_ladder.jl")'
sleep 0.15
tmux send-keys -t "$PANE" Enter
tmux capture-pane -t "$PANE" -p -J -S -80
```

For B, include `r02_two_stage_ladder.jl`, expect `R02_DONE`; recorded first full
run was about 55 minutes (`results/B/RUN_PROVENANCE.md`, statistical SHA `8663544e`).
For B2, expect `R03_DONE`; the two fresh 8-thread full runs took
566.8761510848999 / 565.3235380649567 seconds (about 9.4 minutes).
Measured times/source SHA are in `results/B2/RUN_PROVENANCE.md` and the B2 report.
For the addendum alone, include `r03_hierarchy_postprocess.jl` in a fresh beast
REPL and expect `R03_HIERARCHY_DONE` (typically under a minute including loading,
not a sampling budget). It reads the existing converged binary fits and replaces
only the two addendum CSVs. This is how the mid-run addition was verified twice
without launching additional fits. The full R03 runner also calls that same helper
for future clean runs. Never run a script via one-shot
Julia. A second fresh REPL, same source SHA and seeds, must reproduce every B2
CSV byte-for-byte. Convergence tables contain HEAD provenance, so later artifact
commits require checking out the reported statistical SHA for exact hash comparison.

Return `results/B2/` and `results/figures/B2_*.png` to the laptop and commit there.
Exclude `/.env` and `/Manifest.toml` from rsync, and do not sync caches/data blindly.
Binary chains are ignored; include the prototype loaders before deserializing them.
Close only owned panes with `tmux kill-pane -t "$PANE"`. Do not target sessions by name
for input/capture, or touch `pi_julia_mm_b`, `claude_review_b`, `claude_mm_b_review`,
or the beast's `julia` session. A human may open additional windows in an agent session.

## Running on another league

1. **Prerequisite:** an implemented `DataTournemantSegment` with the actual betdb
   tournament IDs and a verified data snapshot. As of the agreed design, no branch
   provides the intended English segments; adding Premier League/Championship/
   League One/League Two segments is a separate `src/` task, outside B2. An old
   English cache on disk is not proof the corresponding segment implementation exists.
2. Construct `MarketModelConfig(name=..., segment=..., tournaments=..., seasons=...,
   honest_train=..., honest_test=..., price_window=..., min_selections_ladder=...,
   excluded_matches=...)`. Chronological seasons and nonoverlapping honest splits
   are research choices. Leave `expected=(;)` on first exploration; add snapshot
   assertions only after inspection. Do not reuse Scottish expected counts or T014 IDs.
3. Change the single runner config constant and use a separate output root/check-out.
   Run **Phase A coverage first**: inspect close availability, quoted-line mix,
   1X2-only and totals-only shares, completeness refusals and price-window semantics.
   Only then choose the structural population. The historical TODO 023 SSE reference
   is Scottish; it is not an independent benchmark for a new league.
4. Feed the new A `rates.csv` to `phase_b_panel(...; config, rates_path=...)`.
   Inspect weekly ordering and season boundaries before training. Run deterministic
   toy engine gates on the laptop, then sampling only on the beast.
5. Do not assume Scottish findings transfer: the 1X2-only totals bias depends on
   draw/grid misfit and line mix; fixture-noise size depends on market quality; the
   covariance winner may change; the 10b train/test split must be redesigned for
   available seasons and the intended temporal question. A sharper market can change
   all four, even if the implementation and gates are identical.

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
- **C (not implemented here):** fit the structure directly to books, sharing
  information and learning book concentration and fixture deviations.
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
- **Human: double Poisson.** “Dixon–Coles” here refers to the attack/concessions/home
  parameterisation, not a low-score rho correction. Other score grids are out of scope.
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

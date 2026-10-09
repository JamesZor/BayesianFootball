# Revision 8 fast-stage methods (2026-10-06)

Scope: full-book C0/H1/H2 now; separately launched C1 draws are **unpromoted**
until the frozen synthetic recovery and production convergence pass. All priors,
seeds, budgets, thresholds, cache, targets, panel and score-grid settings stay
unchanged. `r05_fullbook_core.jl` retains its original recovery prerequisite;
revision8 uses separate runners, not a silent removal of that guard.

## Fast-rung reporting

`r05_fast_rungs.jl` runs 53 inherited/state gates (2–4) plus 54 Gaussian
scalar/dense parity gates on toy/full-real panels, the six prescribed
4×(2000 warmup +3000 retained), thin1 fits, and every-parameter Gate5. Failed
convergence stops before inference. The accepted workflow scores identical
Phase-A isolated targets, with 10a retrospective median hyperparameters, and
10b training-only hyperparameters followed by chronological pre-week filtering
on the full panel, scoring only the test season. Forecast uncertainty is
conditional on median hyperparameters, as in B2; it includes Kalman state and
observation uncertainty, not own-week updates or hyperparameter mixing.

Pairing uses the sum of supremacy and level marginal log densities with fixture
SE (not a joint density or serial-dependence-robust SE). `c0_vs_r6_metrics.csv`
and `paired_c0_vs_r6.csv` compare to the **published B2 R6** fixture table,
with exact fixture/axis/protocol joins and zero-tolerance target identity.
C0 and R6 have matching likelihoods at transformed parameters, but different
priors/dimensions: C0 imposes equal attack/concessions innovation variance and
independent q/s innovations. Actual fitted gaps are measured, not assumed zero.

The five-axis smoothed table includes a nonlinear log-total expectation from
4000 seeded conditional Gaussian draws (seed3964), rather than transforming a
posterior mean. q/s path bands remain conditional at median theta and descriptive
on the full panel for both protocols; they do not integrate hyperparameter
uncertainty. Parameter/ratio summaries transform every retained physical draw.

## Static home-advantage uncertainty and identifiability

`home_advantage_rungs.csv` integrates static-state conditional Normal marginals
against **all 12,000 retained hyperdraws**, using deterministic mixture-CDF
quantiles; no additional MCMC, thinning, selected draws or new random budget.
The final filtered static marginal already conditions on the entire fitting
panel. 10b HA summaries use **training only**, not test-season smoothing.
H2 kappa uses all retained sampled physical draws. Gamma and H1 gamma_att,
gamma_def, their sum (supremacy lift), and half difference (level shift) include
both state and hyperparameter uncertainty. No plug-in static mean is passed off
as a posterior interval.

**H1 has a likelihood gauge:** (mu+c, gamma_att−c, gamma_def+c) leaves both
rates unchanged. The proper frozen priors identify a posterior, but the separate
scoring/suppression interpretation is prior-dependent. Their sum is invariant.
Wide individual intervals or a shifted posterior median cannot establish that
home advantage acts specifically through away suppression. H2's coefficient
multiplies centred home quality; its sign interval is the evidence for scaling,
not a claim inferred from a point estimate alone.

## Source-checked convention

Read the converted source-PDF pages of
`ridall2024-bayesian-state-space-models-modelling`, **p.1** (abstract/common
home goal advantage) and **p.3** (explicitly permits attack, defence and common
home advantage to be dynamic). These rungs test split and quality-linked
alternatives on **market-derived log rates**, with **static** league HA; they
are not that paper's dynamic Gamma-state goal model. The required p.1/3 citation
is therefore a convention/context comparison, not an equation implemented here.

## Manager-authorised Gaussian-only speed correction

The first attempt at41e28736 used the original dense pair-update likelihood.
Observed progress on the244-fixture C0 fit was roughly170 iterations/minute per
chain, implying hours for two six-fit pipelines rather than the brief's minutes.
The manager explicitly approved a verified scalar Gaussian fast path through
`ask_user_question`. The original attempt/evidence is preserved, not cited as a
completed or numerically failed run. No seed/budget was changed and no posterior
result was used to choose the implementation.

`l05_fast_gaussian.jl` reuses B2's scalar Kalman/BLAS rank-one update arithmetic.
The same native slice sampler accepts an optional filter callback; the default
is still the unchanged dense/Laplace filter. C1 forbids the scalar callback and
all live C1/recovery checkouts remain untouched. Toy and full-real parity gates
compare likelihood <=1e-9 and every filtered mean/covariance <=1e-8, at three
fixed parameter points per Gaussian rung. Small deterministic native-chain
checks compare callback/default behaviour without changing adaptation or RNG.
Reporting uses the same exact scalar engine for static HA mixtures; pre-week
predictions and RTS paths still use the original accepted engine.

## Requested club paths: manager-approved absence annotation

The fixed panel has22 teams. Ross County and Airdrie are not among them;
East Kilbride is present and Kelty's canonical key is `kelty-hearts-fc`.
The first scalar-source pipeline passed107 engine gates and20 convergence rows,
then stopped in a report-only assertion for the absent Ross County path.
Its complete sampled fits/accepted tables/log remain preserved; it is NOT counted
as a clean full production reproduction. The manager explicitly approved
annotating missing clubs instead of inventing paths or changing data.
`path_club_inventory.csv` records availability, and the path figure marks the
absent panels plainly. All22 teams remain in `team_paths_c.csv`. Two NEW full
runs at the corrected reporting source are required for clean byte reproduction.

## Artifacts and reproduction

Each run writes a fresh separate directory. Scientific CSVs and figures must
compare byte-for-byte between two fresh beast REPL runs at the same source.
Fit binaries contain real wall time and are retained for rechecking but are not
claimed byte-identical scientific CSVs. Real timing/log/start/ETA metadata are
separate from scientific tables. No copied convergence, invented interim
Newton counts, relabelled mock posterior or C1 promotion is permitted.

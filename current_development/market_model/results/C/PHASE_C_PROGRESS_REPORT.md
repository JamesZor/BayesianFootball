# Phase C revision 8 progress — fast rungs accepted; C1 unpromoted

2026-10-06. **Phase C is NOT complete.** C2 stays deferred. The frozen C1
recovery and both prescribed C1 production protocols are running; no C1
posterior, recovery, convergence or forecast conclusion is promoted here.

Definition of done for revision8: leave recovery alone; complete C0/H1/H2 in
both protocols with all gates and two byte-identical fresh runs; launch C1
10a/10b without promotion; report, commit/push and hand over.
**Revision8 done:** both fresh clean fast runs pass and all29 scientific
CSVs/PNGs are byte-identical; C1 runs remain live/unpromoted as authorised.

## Plain words

- **C0 closely reproduces B2 R6 in practice**, not just at matched parameters.
  Honest paired score difference C0−R6 is **+0.001285 ±0.004312 fixture SE**.
  Supremacy RMSE is slightly lower, level RMSE slightly higher; there is no
  supported material improvement. Different priors/constraints make a fitted
  gap possible despite the matched-parameter likelihood identity.
- **Home advantage is positive, but away suppression is not identified by H1.**
  Its static split is confounded with the league intercept. Under the frozen
  proper priors, gamma_def's honest 90% interval **[−0.248135,0.300720]** spans
  zero widely. Home scoring and away suppression cannot be separated reliably;
  only their summed supremacy lift is sharply estimated. H1 supplies no
  forecast benefit over C0 and has a tiny negative paired score difference.
- **Quality-linked home advantage is not supported.** H2 kappa is
  **0.033860 [−0.048858,0.125508]** in 10b and
  **0.007876 [−0.035922,0.050117]** in 10a. Both span zero. The honest paired
  score difference is **−0.002457 ±0.003596 fixture SE**. A positive point
  estimate is not evidence that stronger teams receive a bigger home edge.
- **Quality-dominated dynamics remain.** C0's style/quality SD ratio is
  **0.309696 [0.256376,0.368873]** in 10a and
  **0.259010 [0.161043,0.373022]** in 10b; B2 R6 gives0.315340/0.282411.
  These are posterior draw transformations, not evidence for exactly zero style
  variance or a discrete hierarchy.

Sources for every result above: `v8_fast_1/paired_c0_vs_r6.csv`,
`paired_vs_c0.csv`, `home_advantage_rungs.csv`, `parameter_posteriors_c.csv`;
R6 ratio reference: `../B2/rotation_quality_style.csv`.

## 1. Accepted full-book measures

Unchanged population: **517 fixtures /91 grid weeks /22 teams**; honest fit
uses244 fixtures and scores273. 10a is retrospective. Both protocols forecast
before **every** update in their week; 10b hyperparameters use training only.
Predictive uncertainty is conditional at median hyperparameters, as in B2.

Source: `v8_fast_1/onestep_metrics_c.csv`, 10b/all. All four requested linear
axes, nonlinear log-total, and all warm/open/in-season subsets in both protocols
are retained in the CSV; no favourable subset is used.

| Rung | Axis | RMSE | MAE | Mean logpd | 90% coverage |
|---|---|---:|---:|---:|---:|
| C0 | log home |0.100054|0.080165|0.887800|90.476%|
| C0 | log away |0.117402|0.087124|0.730507|87.546%|
| C0 | supremacy |0.190344|0.141325|0.254709|90.476%|
| C0 | level |0.053285|0.040811|1.525224|91.209%|
| H1 | log home |0.100095|0.080196|0.887426|90.476%|
| H1 | log away |0.117470|0.087159|0.729738|87.546%|
| H1 | supremacy |0.190378|0.141348|0.254257|90.476%|
| H1 | level |0.053368|0.040859|1.524461|91.209%|
| H2 | log home |0.099911|0.080127|0.887740|91.575%|
| H2 | log away |0.117260|0.087061|0.728944|87.546%|
| H2 | supremacy |0.189812|0.141468|0.255549|90.476%|
| H2 | level |0.053468|0.040852|1.521928|91.209%|

Source: paired tables, same517/273 fixture keys. The score is the requested
sum of supremacy and level marginal log densities, **not joint log density**;
fixture SE is **not serial-dependence robust**.

| Candidate − control | Protocol | n | Mean difference | Fixture SE |
|---|---|---:|---:|---:|
| C0 − R6 |10a|517|−0.000167|0.000381|
| C0 − R6 |10b|273|+0.001285|0.004312|
| H1 − C0 |10a|517|−0.001155|0.000343|
| H1 − C0 |10b|273|−0.001216|0.000486|
| H2 − C0 |10a|517|+0.000145|0.000545|
| H2 − C0 |10b|273|−0.002457|0.003596|

R6's honest supremacy/level RMSE is0.190775/0.053156 and coverage
90.110%/91.209% (`c0_vs_r6_metrics.csv`). C0's shifts are small, not an exact
identity claim about separately fitted posteriors.

## 2. Home advantage and parameter uncertainty

`v8_fast_1/home_advantage_rungs.csv` has14 rows, both protocols; root
`home_advantage_rungs.csv` is a byte-identical convenience copy.
Static coefficients integrate their conditional Gaussian marginals over **all
12,000 retained hyperdraws**; H2 kappa uses all sampled physical draws. 10b HA
summaries use **only the training panel**, not test-season smoothing.

| Rung / coefficient | 10b median | 90% posterior interval |
|---|---:|---|
| C0 gamma |0.182609|[0.170461,0.194751]|
| H1 gamma_att |0.156337|[−0.118091,0.430764]|
| H1 gamma_def |0.026292|[−0.248135,0.300720]|
| H1 supremacy lift (sum) |0.182630|[0.170473,0.194781]|
| H1 level shift (half difference) |0.065022|[−0.209338,0.339382]|
| H2 kappa |0.033860|[−0.048858,0.125508]|
| H2 gamma |0.182542|[0.169950,0.195117]|

H1 gauge: `(mu+c,gamma_att−c,gamma_def+c)` leaves both rates unchanged.
Individual coefficients/level shift therefore depend on the frozen priors;
the identified supremacy sum does not. Do not attribute the HA median mainly
to home attack merely because that coefficient's prior was centred at0.15.

C0 10b sigma_obs/q/s medians are0.063960/0.026582/0.006908, with90% intervals
[0.058406,0.069836]/[0.022627,0.030887]/[0.004276,0.009785]
(`parameter_posteriors_c.csv`). These Gaussian rungs do not learn n or sigma_u;
those book-measurement summaries must await accepted C1.

`team_paths_c.csv` has24,024 conditional q/s rows for all22 teams, both
protocols and all rungs. Bands exclude hyperparameter mixing; 10b full-panel
smoothing is descriptive, not a forecast. `smoothed_fit_c.csv` and
`ladder_summary_c.csv` include all five axes; nonlinear smoothed log-total is a
seeded Gaussian expectation, not a transform of its mean.

## 3. Gates, tests and implementation corrections

Clean fast source **70336b07a81aac7bc3d7188e7ae2138fb3b59072**.
- `engine_gates_c.csv`: **107/107**, including all53 inherited/state gates,
  H1 pinned-defence/H2 zero-kappa reductions, matched R6, independent batch
  Gaussian/RTS/eigen gates, and54 scalar/dense parity gates on toy/full-real
  panels. Unchanged tolerances: likelihood1e-9, means/covariance1e-8.
- `convergence_c.csv`: **20/20**, six prescribed fits, each4×(2000+3000), thin1;
  max Rhat**1.000985775031962**, min bulk ESS**6074.903633053705**, min tail
  ESS**6129.2431286740775**. No chain/parameter was excluded.
- Final-source fresh deterministic tests: scalar/native sampler**17/17**,
  reporting including present/absent plots**22/22**, t05**305/305**,
  workflow**85/85**, t04**92/92**, t03/t02 **each131/131**;
  C2-pending**29pass/10fail**, explicitly excluded.
  `LAPTOP_C8_FINAL_SOURCE_TESTS.txt` captures every anchored marker and count.

The manager explicitly approved two in-scope corrections:
1. **Gaussian-only speed path.** Original dense C0 throughput implied hours;
   new exact scalar Kalman callback uses the same native slice sampler,
   adaptation, priors and seeds. C1's default numerical path is unchanged;
   all live C1/recovery checkouts stay frozen. Original41e28736 attempt is
   preserved in beast `logs/market_model_c/c8/dense_attempt_preserved/` and
   its checkout, stopped only for speed before any saved fit.
2. **Club availability.** The first scalar attempt at5ab9980b passed all engine
   and convergence gates but its path-figure assertion failed on Ross County.
   Ross County/Airdrie are absent from the fixed56/57 population; Kelty's actual
   key is `kelty-hearts-fc`. Manager approved explicit absence annotations,
   without adding data or inventing paths. That attempt remains under
   `v8_scalar_plot_attempt/`; it is not counted as a clean reproduction.

First reporting test failed4pass/1error because the helper guessed the fit's
unconstrained field name; verified `udraws` and corrected it before any accepted
run. No scientific gate/threshold was changed. Logs are retained. An early CSV
query used `rung` instead of the saved `arm` column; schema inspection corrected
the query, not the CSV. Early awk log prefixes proved constant; they are NOT
used for rates. Correct UTC logger files and observed pane checkpoints are used.
Staged diff-check caught CRLF in two new checkpoint/comparison metadata files;
converted those to LF and verified exact bytes apart from line endings. The29
scientific run files were NOT changed.

Figures in `v8_fast_1/`: `C_fast_coverage.png`, `C_fast_home_advantage.png`,
`C_fast_paths.png`, visually inspected and populated. Missing-club annotations
are intentional and recorded in `path_club_inventory.csv`, not invented ratings.
`C8_METHODS.md` records formulas/limitations and the source-checked Ridall2024
**p.1/3** convention (common, dynamic HA). Our alternatives use static HA on
market-derived rates, not that paper's dynamic Gamma-state goal model.

## 4. Two-run clean reproduction PASS

First clean fresh beast `%270`: `R05_FAST_RUNGS_DONE` at
**2026-10-06T05:46:27Z**, start05:26:50Z. Its26 scientific CSVs and3 PNGs
are hashed in `v8_fast_1/SCIENTIFIC_SHA256SUMS.txt` (29/29 laptop checks).
Second fresh `%271`, same source/seeds/settings, start07:29:02Z, reaches
`R05_FAST_RUNGS_DONE` at **07:45:36Z**, separate `v8_fast_2/`. Exact CSV/PNG
file sets match and **29/29 files are byte-identical**, including every gate,
convergence, onestep, path, parameter and HA table/figure.
`BYTE_COMPARISON.txt`, both scientific manifests and BEAST_C8_FAST1/2.txt are
the evidence. Both laptop copies independently pass29/29 checksums.
Fit binaries include genuine wall time and stay on the beast; they are not
claimed byte-identical scientific tables. Individual fit wall times, run1→run2:
C0 10b65.6→40.6s, H1 10b83.5→70.5s, H2 10b136.2→112.1s,
C0 10a153.6→143.9s, H1 10a170.0→144.1s, H2 10a292.4→240.3s.
These are observed timings under concurrent C1 work, not speed guarantees.

## 5. Live C1 runs and what remains

C1 production source **41e28736cdd990d8d701dcb0b48df98196c54674**;
recovery source **6335316f4e93ec91495341fa2f6960a64ce772a9**. All have16 Julia
threads/core pinning/BLAS1, existing Manifest and pinned cache c786e2fc…b423b4.
No source checkout/reload, package/cache/DB/grid/seed/budget change was made
under these runs. Each production protocol regenerated **12/12 full Gate1**
rows and **53/53 engine/state** rows before sampling; immutable gate exports
are in `v8_c1_10a_preflight/` and `v8_c1_10b_preflight/`, not posterior outputs.
No learned concentration, local noise, pooling benefit or convergence is claimed.

| Run | Pane | Start UTC | Checkout | Logs under /root/BF_runs/logs/market_model_c/ |
|---|---|---|---|---|
| Frozen C1 recovery |%265|03:59 approx (process start)|market_model_c|6335316f_recovery1.log|
| C1 10b production |%267|04:50:29|market_model_c_fast|c8/41e28736_c1_10b.log then *_utc.log|
| C1 10a production |%268|04:50:32|market_model_c_fast|c8/41e28736_c1_10a.log then *_utc.log|

At **07:46:59Z**, observed chain1/2/3/4 iterations are:
recovery**1000/750/500/750**; C1 10b**1500/750/1000/1000**;
C1 10a**250/750/500/250**. These are progress, not retained-chain acceptance.
No failure marker/root exception is present in the checkpoint. Load11.14/12.88/13.00.
`BEAST_C8_LIVE_RUNS.txt` and `LIVE_RUN_ETA_CHECKPOINT.csv` record observations,
correct UTC log endpoints, rate formulas and per-chain extrapolations.

| Run | Measured iterations/min | Slowest-chain remaining estimate | Approx finish UTC |
|---|---:|---:|---|
| C1 10b |4.53–10.36|15.6h|Oct6 23:25|
| C1 10a |2.19–4.66|36.1h|Oct7 19:54|
| Recovery |coarse1.37–4.12|conservative54.6h|Oct8 14:25|

**ETAs are uncertain, not acceptance deadlines.** C1 uses each chain's first
and last correctly timestamped marker (after05:11:26Z), `(5000−reported)/rate`.
Recovery has no per-line timestamps: compare the observed250/chain checkpoint
at04:44:53Z with this snapshot. Chains imply16.2–54.6h; the slowest last marker
is stale/censored at500, so54.6h is especially conservative. Re-estimate at
its next marker rather than relying on the earlier ~5/min estimate. Adaptive
warmup,250-step marker censoring, and changing concurrent load may alter rates.
No fabricated interim newton accounting or convergence is substituted.

Remaining: recovery truth intervals + all-parameter Rhat/ESS; C1 production
convergence and recovery-dependent promotion; accepted C1 vs C0 forecasts,
full-book theta=structure+u shrinkage, n/sigma_u/local noise and mixed posterior
paths/figures, then complete C1 workflow/reproduction. No C2/thin pooling,
score-grid change or Phase D is authorised. This is a handover, not Phase C DONE.

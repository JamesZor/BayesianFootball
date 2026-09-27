# 03 — Scottish Pyramid Dynamics × Scope × Observation Matrix (TODO 031 Wave 1)

## Question

How do team dynamics (`TimeDecayDynamics(180)` vs `MultiScaleGRW()`), data scope (`lower` vs `spfl` vs `spfl_cups`), and likelihood (`Poisson` vs `JointGammaPoissonObservation`) interact on market-on-model compression and out-of-sample log-loss across the Scottish league pyramid?

## Candidates (Wave 1: 12 Factorial Arms)

| Candidate | Dynamics | Scope | Likelihood | Role |
|---|---|---|---|---|
| `td_lower_poisson` | TimeDecay(180) | lower (56/57) | Poisson | candidate |
| `td_lower_joint` | TimeDecay(180) | lower (56/57) | Joint pxG | **control** |
| `td_spfl_poisson` | TimeDecay(180) | SPFL (54–57) | Poisson | candidate |
| `td_spfl_joint` | TimeDecay(180) | SPFL (54–57) | Joint pxG | candidate |
| `td_spfl_cups_poisson` | TimeDecay(180) | SPFL + senior cup ties | Poisson | candidate |
| `td_spfl_cups_joint` | TimeDecay(180) | SPFL + senior cup ties | Joint pxG | candidate |
| `grw_lower_poisson` | MultiScaleGRW | lower (56/57) | Poisson | candidate |
| `grw_lower_joint` | MultiScaleGRW | lower (56/57) | Joint pxG | candidate |
| `grw_spfl_poisson` | MultiScaleGRW | SPFL (54–57) | Poisson | candidate |
| `grw_spfl_joint` | MultiScaleGRW | SPFL (54–57) | Joint pxG | candidate |
| `grw_spfl_cups_poisson` | MultiScaleGRW | SPFL + senior cup ties | Poisson | candidate |
| `grw_spfl_cups_joint` | MultiScaleGRW | SPFL + senior cup ties | Joint pxG | candidate |

## Decision Rule (from TODO 031)

- Screen stage (`--stage screen`): fast MAP ranking across all 40 walk-forward folds to verify feature builder integrity, clock stability, and initial calibration ordering.
- Grid selection on the canonical 56/57 walk-forward book (710 held-out matches across 24/25 and 25/26).
- Primary selection criteria:
  1. Decompression: market-on-model compression slope closest to 1.0 (controlling under-confidence vs over-confidence).
  2. Target 1X2 and all-market LogLoss vs control (`td_lower_joint`).
  3. Monitor reporting on 54/55 Premiership and Championship tiers.

## Result — Wave 1 grid (2026-09-27, scorecard `v1.1`)

**Decision: no cell is promoted under the TODO 031 rule.** No cell's overall LogLoss is
non-inferior to the control at the rule's margin (CI upper < +0.001): the best,
`td_spfl_cups_joint`, is −0.00084 [−0.00441, +0.00277]. The transition-bias half of the rule
cannot be tested yet: `transition_bias_pp` carries no bootstrap CI. **The joint observation
wins outright.** Dynamics and scope trade LogLoss against compression and transition bias.
The choice of the W2 base is a human decision; see "Wave 2 base" below.

Reference: de-vigged Betfair TWA (−20, 0] close. Target = 710 held-out 56/57 fixtures across
40 folds. Compression = market-on-model 1X2 slope (ideal 1; above 1 means the model is
compressed). Bias = mean model − market win probability for the transitioning club, in pp
(positive means the club is overrated), over its first 20 league matches. Δ values are paired
fixture-clustered bootstraps (B = 10,000). "Δ vs ctl" is against `td_lower_joint`
(`97c7a3d9`); "Δ vs m12_td" is against the live champion (`132df5c2`).

| Cell | Target LL | ECE | Compression | Δ vs ctl [95% CI] | Δ vs m12_td [95% CI] | Promoted | Relegated | Entered SPFL | Run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `td_lower_poisson` | 0.64679 | 0.0125 | 2.781 | +0.00304 [−0.00017, +0.00625] | +0.00342 [+0.00002, +0.00675] | +3.00 | −5.74 | −16.40 | `de7fa956` ♻ |
| `td_lower_joint` (ctl) | 0.64375 | 0.0143 | 1.928 | 0 | +0.00038 [−0.00205, +0.00284] | +4.80 | −3.90 | −15.42 | `97c7a3d9` ♻ |
| `td_spfl_poisson` | 0.64681 | 0.0155 | 1.913 | +0.00306 [−0.00077, +0.00697] | +0.00344 [−0.00053, +0.00745] | +4.15 | −9.17 | −16.00 | `4712735f` |
| `td_spfl_joint` | 0.64397 | 0.0089 | 1.401 | +0.00023 [−0.00403, +0.00449] | +0.00060 [−0.00403, +0.00524] | +6.36 | −8.73 | −12.03 | `7d8d69e7` |
| `td_spfl_cups_poisson` | 0.64735 | 0.0151 | 2.025 | +0.00361 [−0.00065, +0.00790] | +0.00398 [−0.00056, +0.00853] | +4.11 | −8.68 | −15.23 | `1440ab9c` |
| **`td_spfl_cups_joint`** | **0.64291** | **0.0080** | 1.498 | −0.00084 [−0.00441, +0.00277] | −0.00046 [−0.00471, +0.00379] | +5.84 | −7.54 | −8.95 | `4346604b` |
| `grw_lower_poisson` | 0.64460 | 0.0162 | 1.190 | +0.00085 [−0.00398, +0.00579] | +0.00123 [−0.00420, +0.00681] | +1.87 | −1.97 | −7.90 | `f64a00a2` ♻ |
| `grw_lower_joint` | 0.64351 | 0.0120 | 1.240 | −0.00023 [−0.00394, +0.00349] | +0.00014 [−0.00470, +0.00498] | +4.13 | **−0.31** | −10.73 | `decb01be` |
| `grw_spfl_poisson` | 0.64644 | 0.0122 | 1.099 | +0.00270 [−0.00284, +0.00828] | +0.00307 [−0.00307, +0.00928] | +5.22 | −7.80 | −6.04 | `f00ec78a` ♻ |
| `grw_spfl_joint` | 0.64437 | 0.0161 | **1.066** | +0.00063 [−0.00509, +0.00623] | +0.00100 [−0.00541, +0.00721] | +5.87 | −5.08 | −8.89 | `82e01a3b` † |
| `grw_spfl_cups_poisson` | 0.64617 | 0.0181 | 1.252 | +0.00242 [−0.00364, +0.00861] | +0.00280 [−0.00397, +0.00973] | +3.19 | −5.45 | **−4.60** | `a6f62436` ♻ |
| `grw_spfl_cups_joint` | 0.64385 | 0.0095 | 1.105 | +0.00010 [−0.00593, +0.00609] | +0.00048 [−0.00631, +0.00714] | +4.58 | −2.60 | −5.14 | `9babf9e9` ♻ |

Cohort sizes (first 20): promoted into L1 n = 74, relegated into L1 n = 56, entered SPFL
n = 15 (the entered-SPFL row is noisy). ♻ = reused run (see caveats). † = persisted at
stride 2. Per-tier compression (t56 / t57) is in `harness_scores`. League Two is compressed
more than League One in every cell.

### Main effects (balanced means over the other two factors)

| Factor | Target LL | ECE | Compression | Promoted | Relegated | Entered SPFL |
|---|---|---|---|---|---|---|
| Dynamics td / grw | 0.6453 / 0.6448 | 0.0124 / 0.0140 | 1.92 / **1.16** | +4.7 / +4.1 | −7.3 / **−3.9** | −14.0 / **−7.2** |
| Scope lower / spfl / spfl_cups | 0.6447 / 0.6454 / 0.6451 | 0.0137 / 0.0132 / 0.0127 | 1.78 / 1.37 / 1.47 | **+3.5** / +5.4 / +4.4 | **−3.0** / −7.7 / −6.1 | −12.6 / −10.7 / **−8.5** |
| Observation poisson / joint | 0.6464 / **0.6437** | 0.0149 / **0.0115** | 1.71 / **1.37** | +3.6 / +5.3 | −6.5 / −4.7 | −11.0 / −10.2 |

1. **Observation.** The joint pxG arm beats Poisson in all six paired cells on target LogLoss
   (−0.0011 to −0.0045), with lower ECE and less compression. Keep it in every later wave.
2. **Dynamics.** At the same LogLoss, GRW removes most of the compression (1.16 vs 1.92). It
   roughly halves the under-rating of relegated and newly entered clubs. TimeDecay has the
   better ECE.
3. **Scope.** Pooling the SPFL cuts compression but makes the transition bias **worse**:
   promoted clubs are overrated more, relegated clubs underrated more. The pooled model has one
   rating scale across tiers and no tier term, so a club's rating carries straight across a
   league change. Cup ties partly repair this and give the best entered-SPFL bias. This is the
   gap that W2's league offsets and tier steps target.
4. Promoted clubs are overrated (+1.9 to +6.4 pp) in every cell. No W1 factor fixes it.

### Wave 2 base (human decision)

W2's tier offsets need a pooled scope, so the natural bases are the two `spfl_cups_joint`
cells:

- `td_spfl_cups_joint`: best LogLoss and ECE on the whole leaderboard. Compressed (1.50);
  relegated −7.5 pp.
- `grw_spfl_cups_joint`: nearly uncompressed (1.105); relegated −2.6 pp, entered −5.1 pp.
  LogLoss is +0.0009 worse on the raw difference (not paired). It is the reused g3 run.

The recommendation is to run W2 on both. A TimeDecay cell costs about 10 minutes on the beast
and a pooled GRW cell about 90. Re-run `grw_spfl_cups_joint` under the harness recipe first,
because g3 predates it.

### Caveats

- **Reused runs** (♻) were sampled at acceptance 0.80/0.90 (s12: 4 × (800 + 800)). Their git
  provenance is `80fe1b7d-dirty`, `31e4795-dirty` or `unknown`. g2 and g3 were persisted
  thinned. The six new runs used 4 × (500 + 1000) at acceptance 0.65 from clean commit
  `5307d039`. All new runs pass every hard gate and the convergence review: R̂ ≤ 1.0124;
  divergences 0–16 of 160,000.
- **`grw_spfl_joint` was saved at stride 2** (2,000 draws). Postgres dropped the stride-1
  write of the pooled GRW-joint artefact (the 1 GB field limit), and the harness fell back.
- **No 54/55 monitor rows.** `score_runs.jl` scores on the `ScottishLower` datastore, so the
  Premiership/Championship monitor subsets are empty. Monitor coverage was checked at fit time
  (`monitor_coverage` diagnostic) but not scored.
- **The control-relative bias has no CI**, and the decision rule needs one.

### Reproduction

Smokes ran from `ade6ee9a` (all 5 hard checks passed for 6/6) and grids from `5307d039`
(`run/w1-grid`), both on mcmc-beast via `w1_queue.sh`; logs are in
`/root/BF_runs/logs/{ade6ee9a/w1_smoke,5307d039/w1_grid}/`. Scoring:
`scripts/score_runs.jl experiments/scotland/03_dynamics_scope_matrix/W1_runs.csv` (control
`td_lower_joint`) and `W1_vs_m12td_runs.csv` (control `m12_td`), then `scripts/leaderboard.jl`.

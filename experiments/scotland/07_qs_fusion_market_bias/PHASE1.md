# Phase 1 — the fusion observation

**Result: PASS** at `01f42219`, attempt `a1`, 2026-10-10 00:03–00:36 UTC on mcmc-beast. Eight gates
ran in sequence, each in a fresh REPL (`phase1_gate.sh a1`), with pins checked first (snapshot
`c786e2fc…423b4`, mtime 2026-09-25 12:57:15.480765468 UTC; table `680bc228…2549`). Logs:
`/root/BF_runs/logs/qs_experiment_w3/phase1/a1/`; evidence copied to `results/phase1/`; gate table
`results/phase1_gates.csv`.

## What was built (src only: component, likelihood seam, export, tests)

`JointMarketFusionObservation(; joint, market, kappa_D_prior = LogNormal(0, 0.2), delta_D_prior =
Normal(0, 0.1), delta_M_prior = Normal(0, 0.1))`: a SharedKappa `JointGammaPoissonObservation` plus a
`MarketRateObservation` on the same linear predictors. The market observes the **true goal
log-rates ζ = η + log κ** with D̃ = δ_D + κ_D·D + e_D and M̃ = δ_M + M + e_M (i.i.d. Normal σ_obs on
each log-rate). Each bias prior may be `nothing`, which fixes the term off; all three off dispatches to
wave 2's `_market_rate_ll` on ζ. Predictions are the joint's `λ = κ·μ` through the double-Poisson
grid. Existing components are unchanged (new methods only, plus one new validation row).
Decisions and reasons: `DECISIONS_LOG.md`, "Phase 1 design".

## Tests (`test/test_market_fusion_observation.jl`, 4256/4256)

| # | Test | Evidence |
|---|---|---|
| 1 | Wiring and masking | Sites/θ order for all on/off combinations; validation (NoGuard, κ_D support, σ prior, non-shared κ rejected); recipe display differs for every prior, switch and table change; harness serialise round trip; masked market rows change nothing; no full book ⇒ fusion ll **==** joint ll; fold-40 designs equal the components' own designs field by field (1060 matches, 774 full books) |
| 2 | Likelihood = hand-written joint + biased Normal at the production seam | `Wave2LikelihoodAudit.accumulated` (the `ll ~ to_submodel(…, false); @addlogprob! ll` seam): abs error 0.0 (−9.51137565251); logjoint = ll + priors; linked space likewise; returned-only = priors only. Fold-40 scale: engine vs per-match `logpdf` relative error ≤ 1e-12 (0.0 observed; an independent toy case differs by 2 ulp, so the two are distinct computations) |
| 3 | Bias off = joint + MarketRateObservation, exactly | `==` at the seam; biased path at κ_D = 1, δ = 0 agrees to 1e-12 |
| 4 | Exact AD + FD convergence, fold 40 | below |
| 5 | Zero-allocation compiled gradients | 0 B for MultiScaleGRW and QualityStyleGRW × {bias, nobias, κ_D only, shifts only}; NoGuard rejected |
| 6 | Synthetic recovery, 3 pre-declared seeds per arm | below |

### Exact AD and FD audit (fold 40)

| Arm | θ | ForwardDiff max | fresh RD max | BigFloat AD max | screen union + random | Richardson max | BigFloat trend failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| grw_bias | 980 | 2.30e-15 | 0 | 2.23e-15 | 0 + 50 | 2.6e-22 | 0 |
| qs_bias | 980 | 7.47e-16 | 0 | 3.28e-16 | 68 + 50 | 1.0e-24 | 0 |
| qs_nobias | 977 | 6.29e-16 | 0 | 7.14e-16 | 53 + 50 | 4.1e-24 | 0 |

### Recovery (fold-40 geometry and real masks, 2×(200+200), 90% intervals)

Gate (wave 2): no parameter misses all three seeds — **met**; pooled coverage **84/99 = 84.8% ≥ 80%**.
By arm: grw_bias 32/36, qs_bias 31/36, **qs_nobias 21/27 (77.8%)**. The gate is pooled, as ruled, but the
nobias arm alone is under 80%; its misses are `dyn.r₀` (1/3; median 0.58 vs 0.4 under the weak
LogNormal(0, 1) prior), τ₀, τₖ, ν, σ_obs (2/3 each). Fit wall time 34–60 s.

Bias terms (truth κ_D 0.85, δ_D 0.03, δ_M −0.05):

| Arm | κ_D | δ_D | δ_M | log κ (0.05) |
|---|---|---|---|---|
| grw_bias | [0.836, 1.025] [0.770, 0.893] [0.864, 1.027]✗ | 3/3 | 3/3 | 3/3 |
| qs_bias | [0.851, 0.964]✗ [0.786, 0.889] [0.822, 0.940] | [0.039, 0.094]✗ 2/3 | [−0.046, 0.015]✗ 2/3 | [−0.028, 0.043]✗ 2/3 |

qs_bias seed 2 misses log κ (low) and δ_M (high) together: the market level is δ_M + log κ + level(η),
so the two trade off along a ridge that only the xG arm's grip on η breaks. Read the Phase 4 δ_M and
log κ posteriors jointly for that reason.

## Regressions

| Suite | Pass |
|---|---|
| tape allocation | 184/184 (150 matrix + 2 QS + 8 market + **24 fusion**) |
| MarketRateObservation | 3815/3815 (as Phase 0; AD 4.05e-16 / 4.11e-16; recovery 39/42) |
| JointGammaPoisson | 160/160 |
| builder | 106/106 |
| harness | 179/179 |
| QualityStyleGRW | 373/373 |
| MultiScaleGRW | 124/124 |

The Phase 0 count question (3815 vs 3831) is answered in `PHASE0.md` (addendum).

Development before the gate: one dev REPL (`claude_qsx3_dev`) ran testsets 1–3 and the fold-40 checks
(81/81) from a truncated copy; that copy is not evidence. All `claude_qsx3_*` sessions are closed.

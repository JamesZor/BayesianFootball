# 07 — Fusion of goals+xG and market rates with learned market bias (wave 3, TODO 042)

Brief: `experiments/claude_qs_fusion_bias_brief.md`. Results: **[REPORT.md](REPORT.md)**. Every manager ruling: [DECISIONS_LOG.md](DECISIONS_LOG.md).

## Question and fixed panel

Does one latent that sees goals, proxy xG and full-book market log rates beat the market close, once the market is allowed a static bias? The bias terms are supremacy scale κ_D, home tilt δ_D and totals offset δ_M.

- **Panel:** the same as wave 2. The pinned ScottishLower snapshot, tournaments 56/57, 40 walk-forward folds and 710 held-out fixtures in 24/25 and 25/26, with `lower` scope, two history seasons and the match-biweek clock.
- **Executable source of truth:** [candidates.jl](candidates.jl).
- **Exclusions:** no ROI or staking, no reference refits, and no prior tuning after scoring.

| Candidate | Role | Dynamics | Observation |
|---|---|---|---|
| fusion_qs_bias | control | weak-r QualityStyleGRW | `JointMarketFusionObservation`, bias priors on |
| fusion_qs_nobias | candidate | same QS | same, κ_D = 1, δ_D = δ_M = 0 (no sites) |
| fusion_grw_bias | candidate | MultiScaleGRW | same as control |

**Shared settings:**
- **Goals and xG:** wave 2's joint observation, unchanged: Poisson goals on κμ, Gamma proxy xG on μ, shared κ.
- **Market:** observes the true goal log-rates ζ = η + log κ as D̃ = δ_D + κ_D·D and M̃ = δ_M + M, with Normal(σ_obs) noise on the frozen table (SHA `680bc228…`).
- **Priors:** κ_D ~ LogNormal(0, 0.2), δ_D and δ_M ~ Normal(0, 0.1), σ_obs ~ HalfNormal(0.20).
- **Predictions:** double Poisson on ζ.
- **Sampler:** W2, 4×(500+1000), accept 0.65, depth 10.

## Immutable final runs

Namespace `scottish_lower_qs_wave3_2426`, sampled at `dde8a441` (`results/phase3/RUNS.csv`):

| arm | run | max R̂ | min tail ESS | divergences |
|---|---|---:|---:|---:|
| fusion_qs_bias | `26de19bf-fc29-4818-bfdf-7bf76535901f` | 1.0194 | 330.7 (fold 40, flagged) | 0 |
| fusion_qs_nobias | `5afdfeb3-abad-4d23-82fb-1bfe6babe797` | 1.0123 | 764.6 | 0 |
| fusion_grw_bias | `b2e5a527-e51e-4913-b7bb-e1b94a4d1ff2` | 1.0147 | 882.3 | 0 |

References are used, never refit:
- wave 2's qs_marketobs `81fdf817-f82d-4f66-9bfd-5f456c4fe29f` and qs_joint `2a5eee90-3402-40ce-a5e3-f4c6b57fc8ea`;
- the de-vigged Betfair TWA(−20,0] close.

## Phases and evidence

| phase | result | record |
|---|---|---|
| 0 branch health | PASS `122cbc37`, 16/16 gates | [PHASE0.md](PHASE0.md) |
| 1 fusion component | PASS `01f42219`: tests 4256/4256, exact AD ≤ 2.3e-15, 0 B tapes, recovery 84/99 | [PHASE1.md](PHASE1.md) |
| 2 smoke | PASS `79f003e0`, 6/6 hard checks per arm; grid approved as frozen | [PHASE2.md](PHASE2.md) |
| 3 grid | PASS `dde8a441`, 2.7 h, no reruns | [PHASE3.md](PHASE3.md) |
| 4 scoring, leakage, record | PASS `4f7ee7bc` / `9cf7d0de` | [PHASE4.md](PHASE4.md), [REPORT.md](REPORT.md) |
| 5 reproduction | PASS `8998b355`: fold 1 of each arm byte-identical (chains and latents) | [REPRODUCIBILITY.md](REPRODUCIBILITY.md) |

**Phase 4 scripts:**
- [r06_score.jl](r06_score.jl): scores, intervals, common-panel 3-way scores, bias and r posteriors, and the SVG.
- [r10_leakage_audit.jl](r10_leakage_audit.jl): the leakage audit.
- [r08_record.jl](r08_record.jl): the register row and boards.
- [phase4_gate.sh](phase4_gate.sh): the gate.

**Phase 5 scripts:** [r07_reproduce.jl](r07_reproduce.jl) (one-fold frozen-seed refit and byte comparison) and [phase5_gate.sh](phase5_gate.sh).

## Decision

**No promotion.** The primary comparison, fusion_qs_bias − market close, is −0.000325 1X2 LogLoss with noncircular 90% interval [−0.00707, 0.00113]: no detectable difference. The bias terms give no detectable gain over fusion_qs_nobias.

The market's supremacy is wider than the fused latent's, not compressed: κ_D is 1.17 (QS) / 1.33 (GRW) and above 1 in all 40 folds. Learning κ_D therefore leaves the predictions more compressed. Neither δ_D nor δ_M is distinguishable from 0.

Register status: `completed`. The frozen-seed reproductions are byte-identical (Phase 5).

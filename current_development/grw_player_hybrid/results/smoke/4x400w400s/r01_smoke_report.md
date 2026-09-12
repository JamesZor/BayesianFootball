# r01 smoke gate — Task 013

Generated 2026-09-11 19:58 at `ce9da66a` on mcmc-beast with 16 threads.

Sampler: QueuedNUTS, 4 chains × 400 warmup + 400 retained, δ = 0.8. Folds 1–2.

## Gradient audit (G1)

| model | fold | n_target | n_parameters | tape_instructions | gradient_ms | allocated_bytes | compiled_forward_error | worst_perturbed_error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m00_baseline_grw | 1 | 0 | 98 | 735 | 0.053 | 35440 | 1.0e-15 | 0.0e+00 |
| m00_baseline_grw | 2 | 1 | 158 | 1309 | 0.073 | 36608 | 7.1e-16 | 0.0e+00 |
| m05_wealth_grw | 1 | 0 | 101 | 791 | 0.084 | 128752 | 4.0e-16 | 0.0e+00 |
| m05_wealth_grw | 2 | 1 | 161 | 1365 | 0.104 | 132992 | 5.7e-16 | 0.0e+00 |
| m10_lineup_grw | 1 | 0 | 100 | 760 | 0.064 | 82096 | 5.4e-16 | 0.0e+00 |
| m10_lineup_grw | 2 | 1 | 160 | 1334 | 0.084 | 84800 | 7.1e-16 | 0.0e+00 |
| m12_joint_hybrid_synergy_grw | 1 | 0 | 103 | 816 | 0.098 | 175408 | 5.2e-16 | 0.0e+00 |
| m12_joint_hybrid_synergy_grw | 2 | 1 | 163 | 1390 | 0.113 | 181184 | 5.5e-16 | 0.0e+00 |

## Sampling, latents and persistence (G2–G5)

| model | folds | oos | draws | max_rhat | min_ess_bulk | min_ess_tail | n_divergent | latent_min_sd | gate_pass | run_id |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| m00_baseline_grw | 2 | 39 | 1600 | 1.0151 | 348.4852 | 394.0214 | 0 | 0.1400 | false | 47940765-9e84-40ab-bc5b-b9387340cdee |
| m05_wealth_grw | 2 | 39 | 1600 | 1.0169 | 647.0892 | 576.8076 | 0 | 0.0955 | true | 7d2555b6-e5c2-43ed-b3c7-44c55321164e |
| m10_lineup_grw | 2 | 39 | 1600 | 1.0165 | 428.1108 | 408.9848 | 0 | 0.1235 | true | 89256ad1-611f-4ac8-ab2a-65c0f66e4e16 |
| m12_joint_hybrid_synergy_grw | 2 | 39 | 1600 | 1.0123 | 354.8731 | 342.9939 | 0 | 0.0926 | false | 21f657e6-a990-4682-b292-497365452dbe |

### Failures

* `m00_baseline_grw` — G3 min ESS bulk 348 / tail 394
* `m12_joint_hybrid_synergy_grw` — G3 min ESS bulk 355 / tail 343

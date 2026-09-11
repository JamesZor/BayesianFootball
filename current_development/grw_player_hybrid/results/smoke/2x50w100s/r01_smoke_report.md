# r01 smoke gate — Task 013

Generated 2026-09-11 19:54 at `ce9da66a` on mcmc-beast with 16 threads.

Sampler: QueuedNUTS, 2 chains × 50 warmup + 100 retained, δ = 0.8. Folds 1–2.

## Gradient audit (G1)

| model | fold | n_target | n_parameters | tape_instructions | gradient_ms | allocated_bytes | compiled_forward_error | worst_perturbed_error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m00_baseline_grw | 1 | 0 | 98 | 735 | 0.054 | 35440 | 1.0e-15 | 0.0e+00 |
| m00_baseline_grw | 2 | 1 | 158 | 1309 | 0.076 | 36608 | 7.1e-16 | 0.0e+00 |
| m05_wealth_grw | 1 | 0 | 101 | 791 | 0.085 | 128752 | 4.0e-16 | 0.0e+00 |
| m05_wealth_grw | 2 | 1 | 161 | 1365 | 0.107 | 132992 | 5.7e-16 | 0.0e+00 |
| m10_lineup_grw | 1 | 0 | 100 | 760 | 0.062 | 82096 | 5.4e-16 | 0.0e+00 |
| m10_lineup_grw | 2 | 1 | 160 | 1334 | 0.081 | 84800 | 7.1e-16 | 0.0e+00 |
| m12_joint_hybrid_synergy_grw | 1 | 0 | 103 | 816 | 0.092 | 175408 | 5.2e-16 | 0.0e+00 |
| m12_joint_hybrid_synergy_grw | 2 | 1 | 163 | 1390 | 0.112 | 181184 | 5.5e-16 | 0.0e+00 |

## Sampling, latents and persistence (G2–G5)

| model | folds | oos | draws | max_rhat | min_ess_bulk | min_ess_tail | n_divergent | latent_min_sd | gate_pass | run_id |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| m00_baseline_grw | 2 | 39 | 200 | 1.0927 | 16.7172 | 43.3343 | 0 | 0.1357 | false | 58b5d6fd-8467-425c-9884-0b79eeb81afd |
| m05_wealth_grw | 2 | 39 | 200 | 1.0522 | 51.4952 | 52.7919 | 0 | 0.0871 | false | 8fd4080d-5aa3-4e30-b5cc-9745b6a2d02a |
| m10_lineup_grw | 2 | 39 | 200 | 1.1155 | 37.3554 | 51.7371 | 0 | 0.1363 | false | b3c2e914-a6b3-4a83-a095-2e533a94fdad |
| m12_joint_hybrid_synergy_grw | 2 | 39 | 200 | 1.1315 | 29.9225 | 25.3907 | 0 | 0.0930 | false | 01a11567-e2e7-43cd-a9c4-97a6929e21f7 |

### Failures

* `m00_baseline_grw` — G3 max R̂ 1.0927 (fold 1)
* `m05_wealth_grw` — G3 max R̂ 1.0522 (fold 2)
* `m10_lineup_grw` — G3 max R̂ 1.1155 (fold 2)
* `m12_joint_hybrid_synergy_grw` — G3 max R̂ 1.1315 (fold 2)

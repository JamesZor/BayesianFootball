# r01 smoke gate — Task 013

Generated 2026-09-11 20:04 at `9f08f31e` on mcmc-beast with 16 threads.

Sampler: QueuedNUTS, 4 chains × 500 warmup + 1000 retained, δ = 0.8. Folds 1–2.

## Gradient audit (G1)

| model | fold | n_target | n_parameters | tape_instructions | gradient_ms | allocated_bytes | compiled_forward_error | worst_perturbed_error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m00_baseline_grw | 1 | 0 | 98 | 735 | 0.054 | 35440 | 1.0e-15 | 0.0e+00 |
| m00_baseline_grw | 2 | 1 | 158 | 1309 | 0.074 | 36608 | 7.1e-16 | 0.0e+00 |
| m05_wealth_grw | 1 | 0 | 101 | 791 | 0.087 | 128752 | 4.0e-16 | 0.0e+00 |
| m05_wealth_grw | 2 | 1 | 161 | 1365 | 0.106 | 132992 | 5.7e-16 | 0.0e+00 |
| m10_lineup_grw | 1 | 0 | 100 | 760 | 0.062 | 82096 | 5.4e-16 | 0.0e+00 |
| m10_lineup_grw | 2 | 1 | 160 | 1334 | 0.081 | 84800 | 7.1e-16 | 0.0e+00 |
| m12_joint_hybrid_synergy_grw | 1 | 0 | 103 | 816 | 0.091 | 175408 | 5.2e-16 | 0.0e+00 |
| m12_joint_hybrid_synergy_grw | 2 | 1 | 163 | 1390 | 0.110 | 181184 | 5.5e-16 | 0.0e+00 |

## Sampling, latents and persistence (G2–G5)

| model | folds | oos | draws | max_rhat | min_ess_bulk | min_ess_tail | n_divergent | latent_min_sd | gate_pass | run_id |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| m00_baseline_grw | 2 | 39 | 4000 | 1.0085 | 1129.9986 | 1407.1108 | 0 | 0.1393 | true | 05e8ea2e-4ba4-48f6-a5b8-222192b33a6f |
| m05_wealth_grw | 2 | 39 | 4000 | 1.0051 | 1393.6546 | 1548.2977 | 0 | 0.0936 | true | 3386e82a-dce8-4ba1-b188-50cf09c8ad0a |
| m10_lineup_grw | 2 | 39 | 4000 | 1.0060 | 1195.1953 | 1642.8678 | 0 | 0.1248 | true | 08245882-1765-4da5-88b5-03f8c360fc64 |
| m12_joint_hybrid_synergy_grw | 2 | 39 | 4000 | 1.0124 | 954.1533 | 885.6599 | 0 | 0.0950 | true | 2c25fbfa-a9c3-4f18-91c9-f6d217f317aa |

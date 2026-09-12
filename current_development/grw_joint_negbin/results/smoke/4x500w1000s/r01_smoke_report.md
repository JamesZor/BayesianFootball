# r01 smoke gate — Task 014 (JointGammaNegBinObservation)

Generated 2026-09-12 00:59 at `7e0fc979` on mcmc-beast with 16 threads.

Sampler: QueuedNUTS, 4 chains × 500 warmup + 1000 retained, δ = 0.8. Folds 1–2.

## Likelihood parity vs `equations.jl` (G0)

| arm | points | worst_abs | worst_rel | pass |
|---|---:|---:|---:|---:|
| joint_gamma_poisson_td | 4 | 2.27e-13 | 1.42e-16 | true |
| joint_gamma_negbin_td | 4 | 1.36e-12 | 8.73e-16 | true |
| negbin_td | 4 | 0.00e+00 | 0.00e+00 | true |

## Gradient audit (G1)

| model | fold | n_target | n_parameters | tape_instructions | gradient_ms | allocated_bytes | compiled_forward_error | worst_perturbed_error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m00_baseline_grw_negbin | 1 | 0 | 99 | 756 | 0.744 | 176816 | 4.0e-14 | 0.0e+00 |
| m00_baseline_grw_negbin | 2 | 1 | 159 | 1330 | 0.789 | 182592 | 1.2e-14 | 0.0e+00 |
| m05_wealth_grw_negbin | 1 | 0 | 102 | 812 | 0.789 | 270128 | 1.2e-14 | 0.0e+00 |
| m05_wealth_grw_negbin | 2 | 1 | 162 | 1386 | 0.810 | 278976 | 3.8e-14 | 0.0e+00 |
| m10_lineup_grw_negbin | 1 | 0 | 101 | 781 | 0.764 | 223472 | 2.5e-14 | 0.0e+00 |
| m10_lineup_grw_negbin | 2 | 1 | 161 | 1355 | 0.790 | 230784 | 1.9e-14 | 0.0e+00 |
| m12_joint_hybrid_synergy_negbin | 1 | 0 | 104 | 837 | 0.793 | 316784 | 5.9e-14 | 0.0e+00 |
| m12_joint_hybrid_synergy_negbin | 2 | 1 | 164 | 1411 | 0.818 | 327168 | 1.2e-14 | 0.0e+00 |

## Sampling, latents and persistence (G2–G5)

| model | folds | oos | draws | max_rhat | min_ess_bulk | min_ess_tail | n_divergent | median_r | latent_min_sd | gate_pass | run_id |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| m00_baseline_grw_negbin | 2 | 39 | 4000 | 1.0043 | 1149.4663 | 1262.4152 | 0 | 28.7437 | 0.1366 | true | 90ca88ba-a5d4-4d54-b5e2-8bc102f7240a |
| m05_wealth_grw_negbin | 2 | 39 | 4000 | 1.0056 | 1681.3784 | 1461.0404 | 0 | 29.2865 | 0.0925 | true | fc7dffce-a91b-4fbe-96a5-3da38c896006 |
| m10_lineup_grw_negbin | 2 | 39 | 4000 | 1.0067 | 1107.7123 | 1538.7374 | 0 | 28.6919 | 0.1259 | true | a8637789-53e1-43e5-9f35-0df7f8296095 |
| m12_joint_hybrid_synergy_negbin | 2 | 39 | 4000 | 1.0048 | 935.3514 | 1362.5983 | 0 | 28.6662 | 0.0919 | true | 497c6a0b-aa9f-4df3-9a77-a24c80a27cab |

## Score grid: NegBin vs double-Poisson at the same λ (G6)

`d_*` is this model's grid minus the double-Poisson grid built from the SAME posterior λ draws. A negative binomial moves mass to 0 and to 4+, so the tail markets must move more than 1X2.

| model | fixture | lambda_h | lambda_a | r | mass | d_home | d_over25 | d_over35 | d_btts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m00_baseline_grw_negbin | 12477128 | 1.4007 | 1.1324 | 30.3318 | 1.0000 | -0.0008 | -0.0032 | 0.0021 | -0.0113 |
| m00_baseline_grw_negbin | 12477134 | 1.7190 | 1.1811 | 30.3318 | 1.0000 | -0.0024 | -0.0056 | 0.0003 | -0.0126 |
| m00_baseline_grw_negbin | 12477131 | 1.9449 | 1.5615 | 30.3318 | 1.0000 | -0.0010 | -0.0086 | -0.0035 | -0.0151 |
| m00_baseline_grw_negbin | 12477135 | 1.6716 | 1.2415 | 30.3318 | 1.0000 | -0.0017 | -0.0056 | 0.0002 | -0.0129 |
| m00_baseline_grw_negbin | 12477132 | 1.4499 | 1.2606 | 30.3318 | 1.0000 | -0.0001 | -0.0042 | 0.0014 | -0.0123 |
| m00_baseline_grw_negbin | 12476800 | 1.6545 | 1.4888 | 30.3318 | 1.0000 | 0.0004 | -0.0068 | -0.0010 | -0.0143 |
| m05_wealth_grw_negbin | 12477128 | 1.4614 | 1.1631 | 31.3186 | 1.0000 | -0.0008 | -0.0036 | 0.0018 | -0.0115 |
| m05_wealth_grw_negbin | 12477134 | 1.5428 | 1.2943 | 31.3186 | 1.0000 | -0.0004 | -0.0049 | 0.0007 | -0.0126 |
| m05_wealth_grw_negbin | 12477131 | 1.5936 | 1.5998 | 31.3186 | 1.0000 | 0.0015 | -0.0068 | -0.0013 | -0.0141 |
| m05_wealth_grw_negbin | 12477135 | 1.5427 | 1.2138 | 31.3186 | 1.0000 | -0.0010 | -0.0044 | 0.0012 | -0.0121 |
| m05_wealth_grw_negbin | 12477132 | 1.5863 | 1.3864 | 31.3186 | 1.0000 | 0.0001 | -0.0057 | 0.0001 | -0.0133 |
| m05_wealth_grw_negbin | 12476800 | 1.5460 | 1.4753 | 31.3186 | 1.0000 | 0.0009 | -0.0059 | -0.0002 | -0.0135 |
| m10_lineup_grw_negbin | 12477128 | 1.4544 | 1.1413 | 30.2046 | 1.0000 | -0.0010 | -0.0036 | 0.0019 | -0.0116 |
| m10_lineup_grw_negbin | 12477134 | 1.6525 | 1.2175 | 30.2046 | 1.0000 | -0.0017 | -0.0053 | 0.0005 | -0.0127 |
| m10_lineup_grw_negbin | 12477131 | 1.9643 | 1.4768 | 30.2046 | 1.0000 | -0.0018 | -0.0084 | -0.0031 | -0.0148 |
| m10_lineup_grw_negbin | 12477135 | 1.5964 | 1.3044 | 30.2046 | 1.0000 | -0.0007 | -0.0054 | 0.0003 | -0.0131 |
| m10_lineup_grw_negbin | 12477132 | 1.4643 | 1.2669 | 30.2046 | 1.0000 | -0.0001 | -0.0044 | 0.0013 | -0.0124 |
| m10_lineup_grw_negbin | 12476800 | 1.5767 | 1.5544 | 30.2046 | 1.0000 | 0.0013 | -0.0067 | -0.0010 | -0.0144 |
| m12_joint_hybrid_synergy_negbin | 12477128 | 1.4873 | 1.2027 | 30.9059 | 1.0000 | -0.0007 | -0.0041 | 0.0015 | -0.0120 |
| m12_joint_hybrid_synergy_negbin | 12477134 | 1.4541 | 1.3014 | 30.9059 | 1.0000 | 0.0002 | -0.0044 | 0.0012 | -0.0124 |
| m12_joint_hybrid_synergy_negbin | 12477131 | 1.7342 | 1.4591 | 30.9059 | 1.0000 | -0.0003 | -0.0070 | -0.0013 | -0.0142 |
| m12_joint_hybrid_synergy_negbin | 12477135 | 1.4833 | 1.2466 | 30.9059 | 1.0000 | -0.0004 | -0.0043 | 0.0013 | -0.0122 |
| m12_joint_hybrid_synergy_negbin | 12477132 | 1.6254 | 1.3741 | 30.9059 | 1.0000 | -0.0003 | -0.0059 | -0.0001 | -0.0135 |
| m12_joint_hybrid_synergy_negbin | 12476800 | 1.4532 | 1.6035 | 30.9059 | 1.0000 | 0.0024 | -0.0062 | -0.0004 | -0.0138 |

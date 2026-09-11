# r01 smoke gate — Task 014 (JointGammaNegBinObservation)

Generated 2026-09-12 00:52 at `7e0fc979` on mcmc-beast with 16 threads.

Sampler: QueuedNUTS, 2 chains × 50 warmup + 100 retained, δ = 0.8. Folds 1–2.

## Likelihood parity vs `equations.jl` (G0)

| arm | points | worst_abs | worst_rel | pass |
|---|---:|---:|---:|---:|
| joint_gamma_poisson_td | 4 | 2.27e-13 | 1.42e-16 | true |
| joint_gamma_negbin_td | 4 | 1.36e-12 | 8.73e-16 | true |
| negbin_td | 4 | 0.00e+00 | 0.00e+00 | true |

## Gradient audit (G1)

| model | fold | n_target | n_parameters | tape_instructions | gradient_ms | allocated_bytes | compiled_forward_error | worst_perturbed_error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m00_baseline_grw_negbin | 1 | 0 | 99 | 756 | 0.743 | 176816 | 4.0e-14 | 0.0e+00 |
| m00_baseline_grw_negbin | 2 | 1 | 159 | 1330 | 0.782 | 182592 | 1.2e-14 | 0.0e+00 |
| m05_wealth_grw_negbin | 1 | 0 | 102 | 812 | 0.786 | 270128 | 1.2e-14 | 0.0e+00 |
| m05_wealth_grw_negbin | 2 | 1 | 162 | 1386 | 0.833 | 278976 | 3.8e-14 | 0.0e+00 |
| m10_lineup_grw_negbin | 1 | 0 | 101 | 781 | 0.759 | 223472 | 2.5e-14 | 0.0e+00 |
| m10_lineup_grw_negbin | 2 | 1 | 161 | 1355 | 0.797 | 230784 | 1.9e-14 | 0.0e+00 |
| m12_joint_hybrid_synergy_negbin | 1 | 0 | 104 | 837 | 0.785 | 316784 | 5.9e-14 | 0.0e+00 |
| m12_joint_hybrid_synergy_negbin | 2 | 1 | 164 | 1411 | 0.836 | 327168 | 1.2e-14 | 0.0e+00 |

## Sampling, latents and persistence (G2–G5)

| model | folds | oos | draws | max_rhat | min_ess_bulk | min_ess_tail | n_divergent | median_r | latent_min_sd | gate_pass | run_id |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| m00_baseline_grw_negbin | 2 | 39 | 200 | 1.1597 | 54.6606 | 34.2395 | 0 | 28.4322 | 0.1434 | false | bc024fce-3308-404e-93b6-b01d69383a76 |
| m05_wealth_grw_negbin | 2 | 39 | 200 | 1.0989 | 18.7596 | 66.2079 | 0 | 29.0857 | 0.0832 | false | 49803538-73df-4748-917c-59bb1484f5de |
| m10_lineup_grw_negbin | 2 | 39 | 200 | 1.1082 | 36.0300 | 62.1778 | 0 | 28.7500 | 0.1148 | false | f1664cd5-da34-4f57-999a-d0906cd71ab2 |
| m12_joint_hybrid_synergy_negbin | 2 | 39 | 200 | 1.0775 | 17.5983 | 38.5803 | 0 | 29.6359 | 0.0915 | false | 2a3875aa-fb2b-4192-816f-6247826b5f83 |

## Score grid: NegBin vs double-Poisson at the same λ (G6)

`d_*` is this model's grid minus the double-Poisson grid built from the SAME posterior λ draws. A negative binomial moves mass to 0 and to 4+, so the tail markets must move more than 1X2.

| model | fixture | lambda_h | lambda_a | r | mass | d_home | d_over25 | d_over35 | d_btts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m00_baseline_grw_negbin | 12477128 | 1.3815 | 1.1244 | 30.5606 | 1.0000 | -0.0007 | -0.0029 | 0.0021 | -0.0108 |
| m00_baseline_grw_negbin | 12477134 | 1.7133 | 1.1626 | 30.5606 | 1.0000 | -0.0025 | -0.0053 | 0.0004 | -0.0121 |
| m00_baseline_grw_negbin | 12477131 | 1.9814 | 1.5770 | 30.5606 | 1.0000 | -0.0011 | -0.0085 | -0.0038 | -0.0148 |
| m00_baseline_grw_negbin | 12477135 | 1.6808 | 1.2252 | 30.5606 | 1.0000 | -0.0017 | -0.0054 | 0.0003 | -0.0124 |
| m00_baseline_grw_negbin | 12477132 | 1.4662 | 1.2497 | 30.5606 | 1.0000 | -0.0003 | -0.0041 | 0.0013 | -0.0120 |
| m00_baseline_grw_negbin | 12476800 | 1.6698 | 1.4999 | 30.5606 | 1.0000 | 0.0004 | -0.0067 | -0.0011 | -0.0140 |
| m05_wealth_grw_negbin | 12477128 | 1.4589 | 1.1658 | 30.6983 | 1.0000 | -0.0008 | -0.0036 | 0.0018 | -0.0115 |
| m05_wealth_grw_negbin | 12477134 | 1.5421 | 1.2893 | 30.6983 | 1.0000 | -0.0004 | -0.0048 | 0.0008 | -0.0126 |
| m05_wealth_grw_negbin | 12477131 | 1.5942 | 1.6256 | 30.6983 | 1.0000 | 0.0017 | -0.0070 | -0.0015 | -0.0143 |
| m05_wealth_grw_negbin | 12477135 | 1.5459 | 1.2093 | 30.6983 | 1.0000 | -0.0010 | -0.0044 | 0.0011 | -0.0121 |
| m05_wealth_grw_negbin | 12477132 | 1.5877 | 1.3836 | 30.6983 | 1.0000 | 0.0001 | -0.0057 | 0.0001 | -0.0133 |
| m05_wealth_grw_negbin | 12476800 | 1.5548 | 1.4782 | 30.6983 | 1.0000 | 0.0009 | -0.0061 | -0.0003 | -0.0137 |
| m10_lineup_grw_negbin | 12477128 | 1.4441 | 1.1555 | 30.9951 | 1.0000 | -0.0008 | -0.0036 | 0.0018 | -0.0116 |
| m10_lineup_grw_negbin | 12477134 | 1.6549 | 1.2138 | 30.9951 | 1.0000 | -0.0017 | -0.0053 | 0.0005 | -0.0126 |
| m10_lineup_grw_negbin | 12477131 | 1.9787 | 1.4440 | 30.9951 | 1.0000 | -0.0020 | -0.0083 | -0.0029 | -0.0146 |
| m10_lineup_grw_negbin | 12477135 | 1.5820 | 1.3251 | 30.9951 | 1.0000 | -0.0004 | -0.0054 | 0.0003 | -0.0131 |
| m10_lineup_grw_negbin | 12477132 | 1.4588 | 1.2614 | 30.9951 | 1.0000 | -0.0001 | -0.0043 | 0.0013 | -0.0124 |
| m10_lineup_grw_negbin | 12476800 | 1.5532 | 1.5566 | 30.9951 | 1.0000 | 0.0015 | -0.0065 | -0.0008 | -0.0142 |
| m12_joint_hybrid_synergy_negbin | 12477128 | 1.4834 | 1.2051 | 32.3313 | 1.0000 | -0.0006 | -0.0039 | 0.0014 | -0.0115 |
| m12_joint_hybrid_synergy_negbin | 12477134 | 1.4518 | 1.2878 | 32.3313 | 1.0000 | 0.0001 | -0.0041 | 0.0012 | -0.0118 |
| m12_joint_hybrid_synergy_negbin | 12477131 | 1.7434 | 1.4589 | 32.3313 | 1.0000 | -0.0005 | -0.0067 | -0.0013 | -0.0135 |
| m12_joint_hybrid_synergy_negbin | 12477135 | 1.4873 | 1.2422 | 32.3313 | 1.0000 | -0.0004 | -0.0041 | 0.0012 | -0.0117 |
| m12_joint_hybrid_synergy_negbin | 12477132 | 1.6277 | 1.3572 | 32.3313 | 1.0000 | -0.0004 | -0.0056 | -0.0000 | -0.0128 |
| m12_joint_hybrid_synergy_negbin | 12476800 | 1.4624 | 1.5892 | 32.3313 | 1.0000 | 0.0021 | -0.0059 | -0.0004 | -0.0132 |

### Failures

* `m00_baseline_grw_negbin` — G3 max R̂ 1.1597 (fold 2)
* `m05_wealth_grw_negbin` — G3 max R̂ 1.0989 (fold 2)
* `m10_lineup_grw_negbin` — G3 max R̂ 1.1082 (fold 1)
* `m12_joint_hybrid_synergy_negbin` — G3 max R̂ 1.0775 (fold 1)

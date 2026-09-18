# r01 smoke gate — Task 016 (1-parameter smile spine)

Generated 2026-09-14 00:02 at `20b5d18` on mcmc-beast with 16 threads. Store latest kickoff 2026-09-05.

Sampler: QueuedNUTS, 4 chains × 500 warmup + 1000 retained, δ = 0.8. Folds 1–2.

## G4a anti-diagonal reweighting, synthetic container

| fixture | n_identity | max_draw_cdf_gap | max_mean_cdf_gap | max_mass_gap | min_cell | max_diag_spread | Δp_home | Δp_draw | Δp_away | Δp_under25 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 2.22e-16 | 0.00e+00 | 5.55e-16 | 6.02e-26 | 3.87e-16 | -0.00006 | +0.00008 | +0.00017 | +0.00000 |
| 2 | 0 | 4.44e-16 | 0.00e+00 | 4.44e-16 | 8.80e-25 | 4.01e-16 | +0.00008 | -0.00006 | +0.00018 | +0.00000 |
| 3 | 0 | 3.33e-16 | 5.55e-17 | 4.44e-16 | 7.36e-25 | 3.70e-16 | +0.00034 | -0.00068 | +0.00055 | +0.00000 |
| 4 | 0 | 2.22e-16 | 0.00e+00 | 5.55e-16 | 3.34e-24 | 3.92e-16 | +0.00053 | -0.00068 | +0.00038 | +0.00000 |
| 5 | 0 | 3.33e-16 | 2.78e-17 | 6.66e-16 | 1.96e-25 | 3.94e-16 | +0.00034 | -0.00035 | +0.00023 | +0.00000 |
| 6 | 0 | 2.22e-16 | 5.55e-17 | 5.55e-16 | 3.66e-22 | 4.18e-16 | +0.00071 | -0.00115 | +0.00067 | +0.00000 |

φ ≡ 1 shortcut bit-identical: true. Un-shortcut path max |Δ|: 2.23e-04 against a max grid truncation mass of 1.67e-03 (worst excess over the per-draw bound -8.87e-15; within bound: true). Non-monotone curve refused: true.

## G0 likelihood parity

| check | model | fold | sites | max_abs_base_delta | worst_abs | worst_rel | pass |
|---|---|---:|---|---:|---:|---:|---:|
| G0a | null_anchor | 1 |  | 0.000e+00 | 0.00e+00 | 0.00e+00 | true |
| G0b | m05_joint_grw_smile_spine_w020 | 1 | σ_sup,σ_smile,β_spine | 1.784e+04 | 7.28e-12 | 6.48e-16 | true |
| G0b | m05_joint_grw_smile_spine_w040 | 1 | σ_sup,σ_smile,β_spine | 3.568e+04 | 1.46e-11 | 5.45e-16 | true |
| G0c | m05_joint_grw_smile_spine_w020 | 1 | vs m05_joint_grw_smile_supremacy_w020 | NaN | 0.00e+00 | 0.00e+00 | true |
| G0c | m05_joint_grw_smile_spine_w040 | 1 | vs m05_joint_grw_smile_supremacy_w040 | NaN | 0.00e+00 | 0.00e+00 | true |
| G0a | null_anchor | 2 |  | 0.000e+00 | 0.00e+00 | 0.00e+00 | true |
| G0b | m05_joint_grw_smile_spine_w020 | 2 | σ_sup,σ_smile,β_spine | 8.945e+03 | 1.27e-11 | 9.79e-16 | true |
| G0b | m05_joint_grw_smile_spine_w040 | 2 | σ_sup,σ_smile,β_spine | 1.789e+04 | 2.55e-11 | 1.16e-15 | true |
| G0c | m05_joint_grw_smile_spine_w020 | 2 | vs m05_joint_grw_smile_supremacy_w020 | NaN | 9.09e-13 | 1.98e-16 | true |
| G0c | m05_joint_grw_smile_spine_w040 | 2 | vs m05_joint_grw_smile_supremacy_w040 | NaN | 0.00e+00 | 0.00e+00 | true |

## GB market coverage

| model | fold | n_train | supremacy_observed | supremacy_share | smile_matches | smile_share | smile_per_strike |
|---|---:|---:|---:|---:|---:|---:|---|
| m05_joint_grw_smile_spine_w020 | 1 | 720 | 719 | 0.999 | 705 | 0.979 | 641/641/698/641/641 |
| m05_joint_grw_smile_spine_w020 | 2 | 740 | 739 | 0.999 | 725 | 0.980 | 660/660/718/660/660 |
| m05_joint_grw_smile_spine_w040 | 1 | 720 | 719 | 0.999 | 705 | 0.979 | 641/641/698/641/641 |
| m05_joint_grw_smile_spine_w040 | 2 | 740 | 739 | 0.999 | 725 | 0.980 | 660/660/718/660/660 |

## G1 gradient audit

Tape and gradient-time columns compare each row with the baseline and the five-strike smile @0.40 on the same fold.

| model | fold | n_parameters | tape_instructions | tape_vs_baseline | tape_vs_five_strike | gradient_ms | grad_ratio_vs_five_strike | allocated_bytes | compiled_forward_error | worst_perturbed_error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_grw_baseline | 1 | 101 | 791 | 0 | -52 | 0.085 | 0.50 | 128752 | 4.0e-16 | 0.0e+00 |
| m05_joint_grw_baseline | 2 | 161 | 1365 | 0 | -52 | 0.105 | 0.55 | 132992 | 5.7e-16 | 0.0e+00 |
| m05_joint_grw_smile_supremacy_w040 | 1 | 108 | 843 | 52 | 0 | 0.172 | 1.00 | 395120 | 3.6e-16 | 0.0e+00 |
| m05_joint_grw_smile_supremacy_w040 | 2 | 168 | 1417 | 52 | 0 | 0.191 | 1.00 | 407296 | 1.6e-16 | 0.0e+00 |
| m05_joint_grw_smile_spine_w020 | 1 | 104 | 844 | 53 | 1 | 0.168 | 0.98 | 487824 | 3.0e-16 | 0.0e+00 |
| m05_joint_grw_smile_spine_w020 | 2 | 164 | 1418 | 53 | 1 | 0.191 | 1.00 | 502688 | 5.8e-16 | 0.0e+00 |
| m05_joint_grw_smile_spine_w040 | 1 | 104 | 844 | 53 | 1 | 0.164 | 0.96 | 487824 | 3.2e-16 | 0.0e+00 |
| m05_joint_grw_smile_spine_w040 | 2 | 164 | 1418 | 53 | 1 | 0.187 | 0.98 | 502688 | 6.9e-16 | 0.0e+00 |

## G2–G5 sampling, latents, persistence

| model | folds | oos | draws | max_rhat | min_ess_bulk | min_ess_tail | n_divergent | min_bfmi | wall_min | σ_sup_median | σ_smile_median | κ_median | β_median | φ_median | gate_pass | run_id |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| m05_joint_grw_smile_spine_w020 | 2 | 39 | 4000 | 1.0065 | 1222.8482 | 1802.3474 | 0 | 0.6799 | 2.8 | 0.2303 | 0.0635 | 1.0996 | 0.0528 | 0.900/0.949/1.000/1.054/1.111 | true | 59d3adf2-5bc1-40b1-b89e-8b1934e16335 |
| m05_joint_grw_smile_spine_w040 | 2 | 39 | 4000 | 1.0051 | 1544.4176 | 1984.5007 | 0 | 0.7013 | 3.5 | 0.2153 | 0.0623 | 1.0966 | 0.0529 | 0.900/0.949/1.000/1.054/1.111 | true | ecf43154-8788-4dae-bbdc-5619de431a91 |

## β_spine per fold (H2, early signal)

Task 015's five-strike medians imply β_LS = 0.0525, with line residuals -0.066 / 0.028 / 0.001 / -0.027 / -0.038 at K = 0…4.

| model | fold | β_median | β_q05 | β_q95 | β_sd | φ_at_median | rhat | ess_bulk | ess_tail |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| m05_joint_grw_smile_spine_w020 | 1 | 0.0528 | 0.0499 | 0.0557 | 0.0018 | 0.9/0.949/1.0/1.054/1.111 | 0.9998 | 4236 | 2444 |
| m05_joint_grw_smile_spine_w020 | 2 | 0.0528 | 0.0501 | 0.0557 | 0.0017 | 0.9/0.949/1.0/1.054/1.111 | 1.0010 | 4065 | 2564 |
| m05_joint_grw_smile_spine_w040 | 1 | 0.0528 | 0.0508 | 0.0550 | 0.0012 | 0.9/0.949/1.0/1.054/1.111 | 1.0011 | 3837 | 2200 |
| m05_joint_grw_smile_spine_w040 | 2 | 0.0529 | 0.0509 | 0.0549 | 0.0012 | 0.9/0.949/1.0/1.054/1.112 | 0.9999 | 3983 | 2021 |

## Task 015 smoke at the same budget (wall time and ESS, for comparison)

| model | max_rhat | min_ess_bulk | min_ess_tail | n_divergent | wall_min |
|---|---:|---:|---:|---:|---:|
| m05_joint_grw_baseline | 1.0059 | 1505.2944 | 1216.6773 | 0 | 0.9 |
| m05_joint_grw_supremacy_w040 | 1.0037 | 1345.4525 | 1988.0804 | 0 | 1.0 |
| m05_joint_grw_smile_supremacy_w040 | 1.0064 | 865.5820 | 1383.4499 | 0 | 3.8 |

## G4b reweighting on the fitted containers

| model | fixtures | max_draw_cdf_gap | max_mass_gap | max_diag_spread | mean_Δp_home | mean_Δp_draw | mean_Δp_away | mean_Δp_under25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_grw_smile_spine_w020 | 39 | 4.44e-16 | 6.66e-16 | 3.48e-16 | -0.00325 | +0.00637 | -0.00313 | +0.00000 |
| m05_joint_grw_smile_spine_w040 | 39 | 4.44e-16 | 6.66e-16 | 3.44e-16 | -0.00326 | +0.00640 | -0.00314 | +0.00000 |

| model | shortcut_bit_identical | forced_max_abs_gap | max_truncation_mass | worst_excess_over_truncation | forced_within_truncation |
|---|---:|---:|---:|---:|---:|
| m05_joint_grw_smile_spine_w020 | true | 1.85e-06 | 9.58e-06 | -4.76e-09 | true |
| m05_joint_grw_smile_spine_w040 | true | 1.57e-06 | 7.67e-06 | -6.99e-09 | true |

## G4c portfolio staking off the reweighted grid (T011)

| model | n_books_flat | n_books_grid | n_identical | max_flat_stake_gap | n_books | n_plain_books | n_skipped | max_totals_gap | n_stake_changed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_grw_smile_spine_w020 | 39 | 39 | 39 | 0.00e+00 | 39 | 39 | 0 | 3.89e-15 | 38 |
| m05_joint_grw_smile_spine_w040 | 39 | 39 | 39 | 0.00e+00 | 39 | 39 | 0 | 3.00e-15 | 37 |

## G4b smile O/U 2.5 pricing — three routes and the plain grid

| model | fixture | p_under_ref | p_under_typed | p_under_legacy | p_under_grid | smile_shift |
|---|---:|---:|---:|---:|---:|---:|
| m05_joint_grw_smile_spine_w020 | 12477128 | 0.503397 | 0.503397 | 0.503397 | 0.503397 | -0.00000 |
| m05_joint_grw_smile_spine_w020 | 12477134 | 0.470632 | 0.470632 | 0.470632 | 0.470632 | +0.00000 |
| m05_joint_grw_smile_spine_w020 | 12477131 | 0.420952 | 0.420952 | 0.420952 | 0.420952 | +0.00000 |
| m05_joint_grw_smile_spine_w020 | 12477135 | 0.469311 | 0.469311 | 0.469311 | 0.469311 | +0.00000 |
| m05_joint_grw_smile_spine_w020 | 12477132 | 0.487486 | 0.487486 | 0.487486 | 0.487486 | -0.00000 |
| m05_joint_grw_smile_spine_w020 | 12476800 | 0.450502 | 0.450502 | 0.450502 | 0.450502 | -0.00000 |
| m05_joint_grw_smile_spine_w040 | 12477128 | 0.504152 | 0.504152 | 0.504152 | 0.504152 | +0.00000 |
| m05_joint_grw_smile_spine_w040 | 12477134 | 0.470683 | 0.470683 | 0.470683 | 0.470683 | -0.00000 |
| m05_joint_grw_smile_spine_w040 | 12477131 | 0.419730 | 0.419730 | 0.419730 | 0.419730 | -0.00000 |
| m05_joint_grw_smile_spine_w040 | 12477135 | 0.468600 | 0.468600 | 0.468600 | 0.468600 | -0.00000 |
| m05_joint_grw_smile_spine_w040 | 12477132 | 0.489670 | 0.489670 | 0.489670 | 0.489670 | +0.00000 |
| m05_joint_grw_smile_spine_w040 | 12476800 | 0.451478 | 0.451478 | 0.451478 | 0.451478 | +0.00000 |

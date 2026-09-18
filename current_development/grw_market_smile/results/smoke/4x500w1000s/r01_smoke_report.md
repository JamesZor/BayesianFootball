# r01 smoke gate — Task 015 (market-anchored MultiScaleGRW)

Generated 2026-09-13 01:23 at `unknown` on mcmc-beast with 16 threads. Store latest kickoff 2026-09-05.

Sampler: QueuedNUTS, 4 chains × 500 warmup + 1000 retained, δ = 0.8. Folds 1–2.

## G0 likelihood parity

| case | fold | points | base_sites | pillar_sites | max_abs_base_delta | worst_abs | worst_rel | pass |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| null_anchor | 1 | 4 | 13 |  | 0.000e+00 | 0.00e+00 | 0.00e+00 | true |
| null_anchor | 2 | 4 | 17 |  | 0.000e+00 | 0.00e+00 | 0.00e+00 | true |
| m05_joint_grw_supremacy_w040 | 1 | 4 | 13 | σ_sup | 4.925e+03 | 3.64e-12 | 4.54e-16 | true |
| m05_joint_grw_supremacy_w040 | 2 | 4 | 17 | σ_sup | 8.408e+03 | 7.28e-12 | 6.01e-16 | true |
| m05_joint_grw_smile_supremacy_w040 | 1 | 4 | 13 | σ_sup,σ_smile,log_φ | 4.458e+04 | 2.91e-11 | 1.05e-15 | true |
| m05_joint_grw_smile_supremacy_w040 | 2 | 4 | 17 | σ_sup,σ_smile,log_φ | 1.052e+04 | 1.05e-11 | 1.33e-15 | true |

## GB market coverage (training matches each pillar reads)

| model | fold | n_train | supremacy_observed | supremacy_share | smile_matches | smile_share | smile_per_strike |
|---|---:|---:|---:|---:|---:|---:|---|
| m05_joint_grw_supremacy_w040 | 1 | 720 | 719 | 0.999 | 0 | 0.000 |  |
| m05_joint_grw_supremacy_w040 | 2 | 740 | 739 | 0.999 | 0 | 0.000 |  |
| m05_joint_grw_smile_supremacy_w040 | 1 | 720 | 719 | 0.999 | 705 | 0.979 | 641/641/698/641/641 |
| m05_joint_grw_smile_supremacy_w040 | 2 | 740 | 739 | 0.999 | 725 | 0.980 | 660/660/718/660/660 |

## G1 gradient audit

Δ columns are against the baseline rung on the same fold.

| model | fold | n_target | n_parameters | tape_instructions | delta_tape | gradient_ms | allocated_bytes | delta_alloc | compiled_forward_error | worst_perturbed_error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_grw_baseline | 1 | 0 | 101 | 791 | 0 | 0.086 | 128752 | 0 | 4.0e-16 | 0.0e+00 |
| m05_joint_grw_baseline | 2 | 1 | 161 | 1365 | 0 | 0.108 | 132992 | 0 | 5.7e-16 | 0.0e+00 |
| m05_joint_grw_supremacy_w040 | 1 | 0 | 102 | 811 | 20 | 0.091 | 175408 | 46656 | 9.3e-16 | 0.0e+00 |
| m05_joint_grw_supremacy_w040 | 2 | 1 | 162 | 1385 | 20 | 0.113 | 181184 | 48192 | 3.5e-17 | 0.0e+00 |
| m05_joint_grw_smile_supremacy_w040 | 1 | 0 | 108 | 843 | 52 | 0.171 | 395120 | 266368 | 3.6e-16 | 0.0e+00 |
| m05_joint_grw_smile_supremacy_w040 | 2 | 1 | 168 | 1417 | 52 | 0.195 | 407296 | 274304 | 1.6e-16 | 0.0e+00 |

## G2–G5 sampling, latents, persistence

| model | folds | oos | draws | max_rhat | min_ess_bulk | min_ess_tail | n_divergent | min_bfmi | σ_sup_median | σ_smile_median | κ_median | φ_median | latent_family | gate_pass | run_id |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| m05_joint_grw_baseline | 2 | 39 | 4000 | 1.0059 | 1505.2944 | 1216.6773 | 0 | 0.7153 | n/a | n/a | 1.1237 |  | count | true | e2152be3-f609-4148-b250-96a624e3966c |
| m05_joint_grw_supremacy_w040 | 2 | 39 | 4000 | 1.0037 | 1345.4525 | 1988.0804 | 0 | 0.7399 | 0.2100 | n/a | 1.1169 |  | count | true | cd886262-38d4-4956-a26a-94cb88706e58 |
| m05_joint_grw_smile_supremacy_w040 | 2 | 39 | 4000 | 1.0064 | 865.5820 | 1383.4499 | 0 | 0.7053 | 0.2170 | 0.0529 | 1.1308 | 0.834/0.965/0.991/1.017/1.058 | smile | true | 5ce9ded8-036f-40f5-b878-2452286dbcbc |

## Pillar sites, per fold

| model | fold | site | mean | rhat | ess_bulk | ess_tail |
|---|---:|---|---:|---:|---:|---:|
| m05_joint_grw_supremacy_w040 | 1 | σ_sup | 0.2096 | 1.0008 | 5788 | 2718 |
| m05_joint_grw_supremacy_w040 | 2 | σ_sup | 0.2110 | 1.0000 | 5038 | 3103 |
| m05_joint_grw_smile_supremacy_w040 | 1 | σ_sup | 0.2165 | 1.0017 | 6042 | 2864 |
| m05_joint_grw_smile_supremacy_w040 | 1 | σ_smile | 0.0530 | 1.0001 | 7481 | 2536 |
| m05_joint_grw_smile_supremacy_w040 | 1 | log_φ[1] | -0.1844 | 1.0008 | 1307 | 1939 |
| m05_joint_grw_smile_supremacy_w040 | 1 | log_φ[2] | -0.0379 | 1.0010 | 1301 | 1957 |
| m05_joint_grw_smile_supremacy_w040 | 1 | log_φ[3] | -0.0115 | 1.0011 | 1298 | 1948 |
| m05_joint_grw_smile_supremacy_w040 | 1 | log_φ[4] | 0.0143 | 1.0011 | 1301 | 2014 |
| m05_joint_grw_smile_supremacy_w040 | 1 | log_φ[5] | 0.0539 | 1.0010 | 1306 | 2052 |
| m05_joint_grw_smile_supremacy_w040 | 2 | σ_sup | 0.2181 | 1.0011 | 7421 | 3113 |
| m05_joint_grw_smile_supremacy_w040 | 2 | σ_smile | 0.0528 | 1.0023 | 8213 | 3262 |
| m05_joint_grw_smile_supremacy_w040 | 2 | log_φ[1] | -0.1793 | 1.0019 | 1277 | 1927 |
| m05_joint_grw_smile_supremacy_w040 | 2 | log_φ[2] | -0.0330 | 1.0016 | 1290 | 1977 |
| m05_joint_grw_smile_supremacy_w040 | 2 | log_φ[3] | -0.0066 | 1.0016 | 1253 | 2076 |
| m05_joint_grw_smile_supremacy_w040 | 2 | log_φ[4] | 0.0193 | 1.0017 | 1226 | 1958 |
| m05_joint_grw_smile_supremacy_w040 | 2 | log_φ[5] | 0.0587 | 1.0018 | 1270 | 2005 |

## G4 smile O/U 2.5 pricing — three routes and the plain grid

| model | fixture | p_under_ref | p_under_typed | p_under_legacy | p_under_grid | smile_shift |
|---|---:|---:|---:|---:|---:|---:|
| m05_joint_grw_smile_supremacy_w040 | 12477128 | 0.489531 | 0.489531 | 0.489531 | 0.481842 | +0.00769 |
| m05_joint_grw_smile_supremacy_w040 | 12477134 | 0.456339 | 0.456339 | 0.456339 | 0.448596 | +0.00774 |
| m05_joint_grw_smile_supremacy_w040 | 12477131 | 0.405670 | 0.405670 | 0.405670 | 0.397952 | +0.00772 |
| m05_joint_grw_smile_supremacy_w040 | 12477135 | 0.454379 | 0.454379 | 0.454379 | 0.446634 | +0.00774 |
| m05_joint_grw_smile_supremacy_w040 | 12477132 | 0.475680 | 0.475680 | 0.475680 | 0.467962 | +0.00772 |
| m05_joint_grw_smile_supremacy_w040 | 12476800 | 0.437032 | 0.437032 | 0.437032 | 0.429283 | +0.00775 |

# r01 smoke gate — Task 015 (market-anchored MultiScaleGRW)

Generated 2026-09-13 01:14 at `unknown` on mcmc-beast with 16 threads. Store latest kickoff 2026-09-05.

Sampler: QueuedNUTS, 4 chains × 500 warmup + 500 retained, δ = 0.8. Folds 1–2.

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
| m05_joint_grw_baseline | 2 | 1 | 161 | 1365 | 0 | 0.104 | 132992 | 0 | 5.7e-16 | 0.0e+00 |
| m05_joint_grw_supremacy_w040 | 1 | 0 | 102 | 811 | 20 | 0.093 | 175408 | 46656 | 9.3e-16 | 0.0e+00 |
| m05_joint_grw_supremacy_w040 | 2 | 1 | 162 | 1385 | 20 | 0.115 | 181184 | 48192 | 3.5e-17 | 0.0e+00 |
| m05_joint_grw_smile_supremacy_w040 | 1 | 0 | 108 | 843 | 52 | 0.175 | 395120 | 266368 | 3.6e-16 | 0.0e+00 |
| m05_joint_grw_smile_supremacy_w040 | 2 | 1 | 168 | 1417 | 52 | 0.197 | 407296 | 274304 | 1.6e-16 | 0.0e+00 |

## G2–G5 sampling, latents, persistence

| model | folds | oos | draws | max_rhat | min_ess_bulk | min_ess_tail | n_divergent | min_bfmi | σ_sup_median | σ_smile_median | κ_median | φ_median | latent_family | gate_pass | run_id |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| m05_joint_grw_baseline | 2 | 39 | 2000 | 1.0125 | 615.3649 | 362.8477 | 0 | 0.7538 | n/a | n/a | 1.1229 |  | count | false | 5b884676-ea23-41be-b0e1-3481b2dd4f54 |
| m05_joint_grw_supremacy_w040 | 2 | 39 | 2000 | 1.0083 | 858.5670 | 913.6224 | 0 | 0.7017 | 0.2100 | n/a | 1.1171 |  | count | true | d1f47330-f71e-4daf-8169-f5c359c1d7da |
| m05_joint_grw_smile_supremacy_w040 | 2 | 39 | 2000 | 1.0152 | 409.2698 | 560.9774 | 0 | 0.6857 | 0.2169 | 0.0529 | 1.1314 | 0.833/0.965/0.990/1.016/1.057 | smile | true | b14c785f-bf6b-4c5a-9d53-e8adfd1bc12f |

## Pillar sites, per fold

| model | fold | site | mean | rhat | ess_bulk | ess_tail |
|---|---:|---|---:|---:|---:|---:|
| m05_joint_grw_supremacy_w040 | 1 | σ_sup | 0.2097 | 1.0039 | 3281 | 1426 |
| m05_joint_grw_supremacy_w040 | 2 | σ_sup | 0.2111 | 0.9993 | 2616 | 1468 |
| m05_joint_grw_smile_supremacy_w040 | 1 | σ_sup | 0.2164 | 1.0030 | 3312 | 1365 |
| m05_joint_grw_smile_supremacy_w040 | 1 | σ_smile | 0.0530 | 1.0033 | 3610 | 1315 |
| m05_joint_grw_smile_supremacy_w040 | 1 | log_φ[1] | -0.1844 | 1.0006 | 642 | 825 |
| m05_joint_grw_smile_supremacy_w040 | 1 | log_φ[2] | -0.0379 | 1.0008 | 640 | 869 |
| m05_joint_grw_smile_supremacy_w040 | 1 | log_φ[3] | -0.0117 | 1.0010 | 642 | 871 |
| m05_joint_grw_smile_supremacy_w040 | 1 | log_φ[4] | 0.0142 | 1.0007 | 655 | 841 |
| m05_joint_grw_smile_supremacy_w040 | 1 | log_φ[5] | 0.0538 | 1.0013 | 661 | 969 |
| m05_joint_grw_smile_supremacy_w040 | 2 | σ_sup | 0.2182 | 1.0010 | 3541 | 1613 |
| m05_joint_grw_smile_supremacy_w040 | 2 | σ_smile | 0.0528 | 1.0044 | 3837 | 1717 |
| m05_joint_grw_smile_supremacy_w040 | 2 | log_φ[1] | -0.1801 | 1.0018 | 700 | 1063 |
| m05_joint_grw_smile_supremacy_w040 | 2 | log_φ[2] | -0.0337 | 1.0015 | 710 | 1055 |
| m05_joint_grw_smile_supremacy_w040 | 2 | log_φ[3] | -0.0074 | 1.0016 | 697 | 1058 |
| m05_joint_grw_smile_supremacy_w040 | 2 | log_φ[4] | 0.0184 | 1.0016 | 665 | 1037 |
| m05_joint_grw_smile_supremacy_w040 | 2 | log_φ[5] | 0.0579 | 1.0016 | 691 | 965 |

## G4 smile O/U 2.5 pricing — three routes and the plain grid

| model | fixture | p_under_ref | p_under_typed | p_under_legacy | p_under_grid | smile_shift |
|---|---:|---:|---:|---:|---:|---:|
| m05_joint_grw_smile_supremacy_w040 | 12477128 | 0.489784 | 0.489784 | 0.489784 | 0.481983 | +0.00780 |
| m05_joint_grw_smile_supremacy_w040 | 12477134 | 0.456487 | 0.456487 | 0.456487 | 0.448629 | +0.00786 |
| m05_joint_grw_smile_supremacy_w040 | 12477131 | 0.405662 | 0.405662 | 0.405662 | 0.397827 | +0.00784 |
| m05_joint_grw_smile_supremacy_w040 | 12477135 | 0.454495 | 0.454495 | 0.454495 | 0.446635 | +0.00786 |
| m05_joint_grw_smile_supremacy_w040 | 12477132 | 0.475679 | 0.475679 | 0.475679 | 0.467847 | +0.00783 |
| m05_joint_grw_smile_supremacy_w040 | 12476800 | 0.437115 | 0.437115 | 0.437115 | 0.429250 | +0.00787 |

### Failures

* `m05_joint_grw_baseline` — G3 min ESS bulk 615 / tail 363

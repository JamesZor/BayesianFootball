# r06 smoke gate — Task 008 Phase 2

Generated 2026-09-18 14:06 at `unsynced-rsync` on mcmc-beast with 16 threads.

Sampler: QueuedNUTS, 4 chains × 500 warmup + 1000 retained, δ = 0.8. Folds 1–2.

## G1 — gradient audit vs the flat Exp 06 twin

| model | fold | n_teams | n_parameters | flat_n_parameters | tape_instructions | flat_tape_instructions | gradient_ms | flat_gradient_ms | allocated_bytes | flat_allocated_bytes | compiled_forward_error | worst_perturbed_error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_td_turf_asym | 1 | 23 | 78 | 53 | 240 | 207 | 0.069 | 0.064 | 129072 | 128560 | 2.4e-16 | 0.0e+00 |
| m05_joint_td_turf_asym | 2 | 25 | 84 | 57 | 240 | 207 | 0.069 | 0.065 | 133344 | 132800 | 3.5e-16 | 0.0e+00 |
| m05_joint_td_turf_dual | 1 | 23 | 80 | 53 | 267 | 207 | 0.073 | 0.064 | 129072 | 128560 | 3.0e-16 | 0.0e+00 |
| m05_joint_td_turf_dual | 2 | 25 | 86 | 57 | 267 | 207 | 0.074 | 0.064 | 133344 | 132800 | 7.1e-16 | 0.0e+00 |
| m05_joint_td_contextual | 1 | 23 | 82 | 53 | 293 | 207 | 0.078 | 0.063 | 129072 | 128560 | 2.4e-16 | 0.0e+00 |
| m05_joint_td_contextual | 2 | 25 | 88 | 57 | 293 | 207 | 0.078 | 0.065 | 133344 | 132800 | 1.7e-16 | 0.0e+00 |

## G2/G3 — sampling, latents, persistence

| model | folds | oos | max_rhat | max_site_rhat | min_ess_bulk | min_ess_tail | min_site_ess | n_divergent | min_bfmi | wall_min | gate_pass | run_id |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| m05_joint_td_turf_asym | 2 | 39 | 1.0047 | 1.0047 | 1225.1612 | 1510.4410 | 1696.2508 | 0 | 0.7654 | 0.6280 | true | 6c52b6fc-915f-42b0-a361-ede88be2d6fd |
| m05_joint_td_turf_dual | 2 | 39 | 1.0041 | 1.0037 | 1233.7129 | 1514.0892 | 1524.0686 | 0 | 0.7426 | 0.4629 | true | 31503bc7-ff44-45be-b86c-d6f1c639ebb3 |
| m05_joint_td_contextual | 2 | 39 | 1.0045 | 1.0045 | 1426.9867 | 1059.2012 | 1426.9867 | 0 | 0.7871 | 0.5727 | true | 617dd6ab-0e0f-409b-b6fc-443a288c5210 |

## Contextual design (training fixtures with the term switched on)

| fold | n_train | turf_home | turf_asym | midweek | rest_nonzero | rest_abs_mean | model |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 720 | 468 | 148 | 62 | 154 | 0.9375 | m05_joint_td_turf_asym |
| 2 | 740 | 480 | 153 | 62 | 154 | 0.9122 | m05_joint_td_turf_asym |
| 1 | 720 | 468 | 148 | 62 | 154 | 0.9375 | m05_joint_td_turf_dual |
| 2 | 740 | 480 | 153 | 62 | 154 | 0.9122 | m05_joint_td_turf_dual |
| 1 | 720 | 468 | 148 | 62 | 154 | 0.9375 | m05_joint_td_contextual |
| 2 | 740 | 480 | 153 | 62 | 154 | 0.9122 | m05_joint_td_contextual |

## Coefficients

| model | fold | site | mean | sd | q05 | q95 | p_positive | prior_p_positive | contraction |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_td_turf_asym | 1 | turf_asym.w | 0.0496 | 0.0435 | -0.0213 | 0.1222 | 0.8750 | 0.8413 | 0.1305 |
| m05_joint_td_turf_asym | 1 | ha.γ_base | 0.1254 | 0.0326 | 0.0699 | 0.1770 | 1.0000 | 0.9987 | 0.3471 |
| m05_joint_td_turf_asym | 1 | ha.σ_γ | 0.0389 | 0.0282 | 0.0029 | 0.0919 | 1.0000 | 1.0000 | 0.0653 |
| m05_joint_td_turf_asym | 2 | turf_asym.w | 0.0475 | 0.0445 | -0.0261 | 0.1205 | 0.8555 | 0.8413 | 0.1103 |
| m05_joint_td_turf_asym | 2 | ha.γ_base | 0.1340 | 0.0342 | 0.0786 | 0.1894 | 1.0000 | 0.9987 | 0.3161 |
| m05_joint_td_turf_asym | 2 | ha.σ_γ | 0.0327 | 0.0248 | 0.0025 | 0.0804 | 1.0000 | 1.0000 | 0.1773 |
| m05_joint_td_turf_dual | 1 | turf_asym.w | 0.0567 | 0.0438 | -0.0170 | 0.1285 | 0.9058 | 0.8413 | 0.1242 |
| m05_joint_td_turf_dual | 1 | turf_gen.w | -0.0451 | 0.0407 | -0.1118 | 0.0204 | 0.1422 | 0.5000 | 0.1867 |
| m05_joint_td_turf_dual | 1 | turf_pace.w | -0.0065 | 0.0394 | -0.0720 | 0.0593 | 0.4335 | 0.5000 | 0.2118 |
| m05_joint_td_turf_dual | 1 | ha.γ_base | 0.1445 | 0.0362 | 0.0866 | 0.2052 | 0.9998 | 0.9987 | 0.2756 |
| m05_joint_td_turf_dual | 1 | ha.σ_γ | 0.0371 | 0.0272 | 0.0031 | 0.0880 | 1.0000 | 1.0000 | 0.0976 |
| m05_joint_td_turf_dual | 2 | turf_asym.w | 0.0522 | 0.0439 | -0.0203 | 0.1263 | 0.8858 | 0.8413 | 0.1212 |
| m05_joint_td_turf_dual | 2 | turf_gen.w | -0.0354 | 0.0408 | -0.1017 | 0.0321 | 0.1928 | 0.5000 | 0.1830 |
| m05_joint_td_turf_dual | 2 | turf_pace.w | -0.0059 | 0.0391 | -0.0711 | 0.0581 | 0.4442 | 0.5000 | 0.2177 |
| m05_joint_td_turf_dual | 2 | ha.γ_base | 0.1481 | 0.0375 | 0.0872 | 0.2116 | 1.0000 | 0.9987 | 0.2491 |
| m05_joint_td_turf_dual | 2 | ha.σ_γ | 0.0317 | 0.0240 | 0.0025 | 0.0774 | 1.0000 | 1.0000 | 0.2025 |
| m05_joint_td_contextual | 1 | turf_asym.w | 0.0573 | 0.0425 | -0.0146 | 0.1267 | 0.9048 | 0.8413 | 0.1495 |
| m05_joint_td_contextual | 1 | turf_gen.w | -0.0457 | 0.0390 | -0.1120 | 0.0175 | 0.1172 | 0.5000 | 0.2206 |
| m05_joint_td_contextual | 1 | turf_pace.w | -0.0058 | 0.0385 | -0.0706 | 0.0580 | 0.4445 | 0.5000 | 0.2309 |
| m05_joint_td_contextual | 1 | midweek.w | 0.0505 | 0.0456 | -0.0251 | 0.1224 | 0.8668 | 0.8413 | 0.0888 |
| m05_joint_td_contextual | 1 | rest_diff.w | -0.0010 | 0.0107 | -0.0188 | 0.0166 | 0.4640 | 0.8413 | 0.4636 |
| m05_joint_td_contextual | 1 | ha.γ_base | 0.1414 | 0.0361 | 0.0806 | 0.2012 | 1.0000 | 0.9987 | 0.2772 |
| m05_joint_td_contextual | 1 | ha.σ_γ | 0.0380 | 0.0284 | 0.0024 | 0.0918 | 1.0000 | 1.0000 | 0.0593 |
| m05_joint_td_contextual | 2 | turf_asym.w | 0.0533 | 0.0435 | -0.0167 | 0.1253 | 0.8915 | 0.8413 | 0.1292 |
| m05_joint_td_contextual | 2 | turf_gen.w | -0.0348 | 0.0407 | -0.1027 | 0.0321 | 0.1958 | 0.5000 | 0.1868 |
| m05_joint_td_contextual | 2 | turf_pace.w | -0.0066 | 0.0400 | -0.0723 | 0.0586 | 0.4340 | 0.5000 | 0.1995 |
| m05_joint_td_contextual | 2 | midweek.w | 0.0503 | 0.0457 | -0.0242 | 0.1243 | 0.8640 | 0.8413 | 0.0856 |
| m05_joint_td_contextual | 2 | rest_diff.w | 0.0011 | 0.0122 | -0.0190 | 0.0213 | 0.5360 | 0.8413 | 0.3916 |
| m05_joint_td_contextual | 2 | ha.γ_base | 0.1446 | 0.0370 | 0.0833 | 0.2058 | 1.0000 | 0.9987 | 0.2591 |
| m05_joint_td_contextual | 2 | ha.σ_γ | 0.0311 | 0.0236 | 0.0029 | 0.0760 | 1.0000 | 1.0000 | 0.2158 |

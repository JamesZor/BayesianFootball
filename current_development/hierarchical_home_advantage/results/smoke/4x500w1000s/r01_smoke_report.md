# r01 smoke gate — Task 008 Phase 1

Generated 2026-09-12 23:03 at `unsynced-rsync` on mcmc-beast with 16 threads.

Sampler: QueuedNUTS, 4 chains × 500 warmup + 1000 retained, δ = 0.8. Folds 1–2.

## Gradient audit (G1), hierarchical vs flat twin on the same fold

| model | fold | n_teams | n_parameters | flat_n_parameters | tape_instructions | flat_tape_instructions | gradient_ms | flat_gradient_ms | allocated_bytes | flat_allocated_bytes | delta_allocated_bytes | compiled_forward_error | worst_perturbed_error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_production_wealth_hier_ha | 1 | 23 | 77 | 53 | 227 | 207 | 0.065 | 0.064 | 129072 | 128560 | 512 | 6.7e-16 | 0.0e+00 |
| m05_joint_production_wealth_hier_ha | 2 | 25 | 83 | 57 | 227 | 207 | 0.066 | 0.065 | 133344 | 132800 | 544 | 3.0e-16 | 0.0e+00 |
| m12_joint_hybrid_synergy_hier_ha | 1 | 23 | 79 | 55 | 252 | 232 | 0.074 | 0.071 | 175728 | 175216 | 512 | 4.6e-16 | 0.0e+00 |
| m12_joint_hybrid_synergy_hier_ha | 2 | 25 | 85 | 59 | 252 | 232 | 0.073 | 0.073 | 181536 | 180992 | 544 | 4.9e-16 | 0.0e+00 |
| m12_joint_hybrid_synergy_grw_hier_ha | 1 | 23 | 127 | 103 | 836 | 816 | 0.093 | 0.093 | 175920 | 175408 | 512 | 2.7e-16 | 0.0e+00 |
| m12_joint_hybrid_synergy_grw_hier_ha | 2 | 25 | 189 | 163 | 1410 | 1390 | 0.112 | 0.116 | 181728 | 181184 | 544 | 4.3e-16 | 0.0e+00 |

## Sampling, latents and persistence (G2–G5)

| model | folds | oos | draws | max_rhat | max_ha_rhat | min_ess_bulk | min_ess_tail | n_divergent | min_bfmi | wall_min | gate_pass | run_id |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| m05_joint_production_wealth_hier_ha | 2 | 39 | 4000 | 1.0056 | 1.0044 | 1166.0704 | 1228.1202 | 0 | 0.7154 | 0.6489 | false | 6ced7c3c-0a1e-43bc-b7a4-1a0ef9b468f4 |
| m12_joint_hybrid_synergy_hier_ha | 2 | 39 | 4000 | 1.0050 | 1.0050 | 1298.6881 | 733.6542 | 0 | 0.7953 | 0.4516 | false | 3b748983-a2e4-4380-ab60-ab8c3c20046e |
| m12_joint_hybrid_synergy_grw_hier_ha | 2 | 39 | 4000 | 1.0087 | 1.0042 | 878.2799 | 609.5752 | 0 | 0.5477 | 0.8186 | false | 550a0d00-7fa1-4689-9c92-e0c89637564f |

## HA hyperparameters

| model | fold | gamma_base_mean | gamma_base_sd | sigma_q05 | sigma_q50 | sigma_q95 | p_sigma_below_0p02 |
|---|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_production_wealth_hier_ha | 1 | 0.1228 | 0.0402 | 0.0051 | 0.0517 | 0.1335 | 0.1953 |
| m05_joint_production_wealth_hier_ha | 2 | 0.1335 | 0.0440 | 0.0038 | 0.0398 | 0.1183 | 0.2647 |
| m12_joint_hybrid_synergy_hier_ha | 1 | 0.1147 | 0.0404 | 0.0042 | 0.0450 | 0.1167 | 0.2323 |
| m12_joint_hybrid_synergy_hier_ha | 2 | 0.1261 | 0.0461 | 0.0031 | 0.0328 | 0.0949 | 0.3152 |
| m12_joint_hybrid_synergy_grw_hier_ha | 1 | 0.1204 | 0.0370 | 0.0428 | 0.1150 | 0.1814 | 0.0145 |
| m12_joint_hybrid_synergy_grw_hier_ha | 2 | 0.1245 | 0.0359 | 0.0341 | 0.1029 | 0.1665 | 0.0275 |

## Held-out fixtures with an unmapped home club (T003)

| fold | match_id | home_team | away_team | away_mapped | model |
|---:|---:|---|---|---:|---|
| 1 | 12477132 | inverness-caledonian-thistle | dumbarton | true | m05_joint_production_wealth_hier_ha |
| 1 | 12476799 | arbroath | montrose | true | m05_joint_production_wealth_hier_ha |
| 1 | 12477132 | inverness-caledonian-thistle | dumbarton | true | m12_joint_hybrid_synergy_hier_ha |
| 1 | 12476799 | arbroath | montrose | true | m12_joint_hybrid_synergy_hier_ha |
| 1 | 12477132 | inverness-caledonian-thistle | dumbarton | true | m12_joint_hybrid_synergy_grw_hier_ha |
| 1 | 12476799 | arbroath | montrose | true | m12_joint_hybrid_synergy_grw_hier_ha |


### Failures

* `m05_joint_production_wealth_hier_ha` — G1 hierarchical HA adds allocation: Δ = 512, 544 B
* `m12_joint_hybrid_synergy_hier_ha` — G1 hierarchical HA adds allocation: Δ = 512, 544 B
* `m12_joint_hybrid_synergy_grw_hier_ha` — G1 hierarchical HA adds allocation: Δ = 512, 544 B

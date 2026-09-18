# r01 smoke gate — Task 008 Phase 1

Generated 2026-09-12 22:54 at `unsynced-rsync` on mcmc-beast with 16 threads.

Sampler: QueuedNUTS, 4 chains × 400 warmup + 400 retained, δ = 0.8. Folds 1–2.

## Gradient audit (G1), hierarchical vs flat twin on the same fold

| model | fold | n_teams | n_parameters | flat_n_parameters | tape_instructions | flat_tape_instructions | gradient_ms | flat_gradient_ms | allocated_bytes | flat_allocated_bytes | delta_allocated_bytes | compiled_forward_error | worst_perturbed_error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_production_wealth_hier_ha | 1 | 23 | 77 | 53 | 227 | 207 | 0.066 | 0.064 | 129072 | 128560 | 512 | 6.7e-16 | 0.0e+00 |
| m05_joint_production_wealth_hier_ha | 2 | 25 | 83 | 57 | 227 | 207 | 0.066 | 0.064 | 133344 | 132800 | 544 | 3.0e-16 | 0.0e+00 |
| m12_joint_hybrid_synergy_hier_ha | 1 | 23 | 79 | 55 | 252 | 232 | 0.073 | 0.072 | 175728 | 175216 | 512 | 4.6e-16 | 0.0e+00 |
| m12_joint_hybrid_synergy_hier_ha | 2 | 25 | 85 | 59 | 252 | 232 | 0.073 | 0.072 | 181536 | 180992 | 544 | 4.9e-16 | 0.0e+00 |
| m12_joint_hybrid_synergy_grw_hier_ha | 1 | 23 | 127 | 103 | 836 | 816 | 0.094 | 0.092 | 175920 | 175408 | 512 | 2.7e-16 | 0.0e+00 |
| m12_joint_hybrid_synergy_grw_hier_ha | 2 | 25 | 189 | 163 | 1410 | 1390 | 0.115 | 0.117 | 181728 | 181184 | 544 | 4.3e-16 | 0.0e+00 |

## Sampling, latents and persistence (G2–G5)

| model | folds | oos | draws | max_rhat | max_ha_rhat | min_ess_bulk | min_ess_tail | n_divergent | min_bfmi | wall_min | gate_pass | run_id |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| m05_joint_production_wealth_hier_ha | 2 | 39 | 1600 | 1.0093 | 1.0084 | 564.3209 | 497.2610 | 0 | 0.6948 | 0.5306 | false | 3a3e1509-c242-44d9-a4ef-f6de6a03e1e9 |
| m12_joint_hybrid_synergy_hier_ha | 2 | 39 | 1600 | 1.0104 | 1.0104 | 528.3531 | 250.5325 | 0 | 0.6297 | 0.3637 | false | 50bd541b-4053-4a5c-b3b9-31f909c56614 |
| m12_joint_hybrid_synergy_grw_hier_ha | 2 | 39 | 1600 | 1.0146 | 1.0098 | 360.9854 | 284.6206 | 0 | 0.5639 | 0.7119 | false | 59f31574-a75a-4961-ade6-68955ed6f67d |

## HA hyperparameters

| model | fold | gamma_base_mean | gamma_base_sd | sigma_q05 | sigma_q50 | sigma_q95 | p_sigma_below_0p02 |
|---|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_production_wealth_hier_ha | 1 | 0.1226 | 0.0411 | 0.0051 | 0.0517 | 0.1266 | 0.1963 |
| m05_joint_production_wealth_hier_ha | 2 | 0.1343 | 0.0445 | 0.0032 | 0.0390 | 0.1149 | 0.2756 |
| m12_joint_hybrid_synergy_hier_ha | 1 | 0.1171 | 0.0411 | 0.0043 | 0.0435 | 0.1164 | 0.2331 |
| m12_joint_hybrid_synergy_hier_ha | 2 | 0.1263 | 0.0434 | 0.0039 | 0.0347 | 0.0978 | 0.2938 |
| m12_joint_hybrid_synergy_grw_hier_ha | 1 | 0.1184 | 0.0371 | 0.0508 | 0.1166 | 0.1780 | 0.0119 |
| m12_joint_hybrid_synergy_grw_hier_ha | 2 | 0.1255 | 0.0348 | 0.0327 | 0.1012 | 0.1630 | 0.0275 |

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
* `m12_joint_hybrid_synergy_hier_ha` — G1 hierarchical HA adds allocation: Δ = 512, 544 B; G3 min ESS bulk 528 / tail 251
* `m12_joint_hybrid_synergy_grw_hier_ha` — G1 hierarchical HA adds allocation: Δ = 512, 544 B; G3 min ESS bulk 361 / tail 285

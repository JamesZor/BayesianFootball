# r02 production grid — Task 015 (market-anchored MultiScaleGRW)

Generated 2026-09-13 12:41 at `unknown` on mcmc-beast. Namespace `scottish_lower_grw_market_smile`. Store latest kickoff 2026-09-05; 43 folds.

Sampler: QueuedNUTS 4 × (500 warmup + 1000 retained), δ = 0.8. Audit on all retained draws; artefact keeps every 2nd draw. Baseline rung = Task 013 run `b0961bc4-c40c-4dbe-9c05-57df7ae0839e`.

## Convergence

| model | folds | oos | draws | max_rhat | min_ess_bulk | min_ess_tail | n_divergent | divergence_rate | treedepth_rate | min_bfmi | strict_rhat_pass | gate_pass | wall_min | run_id |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| m05_joint_grw_supremacy_w040 | 43 | 769 | 2000 | 1.0105 | 814 | 516 | 0 | 0.00000 | 0.0000 | 0.6827 | false | true | 60.5 | 0ee58d18-b7e9-4168-8d78-93887b1a8c26 |
| m05_joint_grw_smile_supremacy_w020 | 43 | 769 | 2000 | 1.0139 | 472 | 498 | 0 | 0.00000 | 0.0000 | 0.6233 | false | true | 158.4 | fcd5e974-9a46-4a10-9828-6b987a5484d6 |
| m05_joint_grw_smile_supremacy_w040 | 43 | 769 | 2000 | 1.0200 | 431 | 696 | 0 | 0.00000 | 0.0000 | 0.6650 | false | true | 185.4 | 30620d3e-e4bd-4c05-b1a1-85cefa36b728 |
| m05_joint_grw_smile_supremacy_w070 | 43 | 769 | 4000 | 1.0201 | 312 | 633 | 0 | 0.00000 | 0.0000 | 0.6805 | false | false | 266.2 |  |

## Pillar posteriors (pooled over folds)

| model | σ_sup_median | σ_sup_q05 | σ_sup_q95 | σ_smile_median | σ_smile_q05 | σ_smile_q95 | κ_median | φ_median |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| m05_joint_grw_supremacy_w040 | 0.212 | 0.194 | 0.232 | n/a | n/a | n/a | 1.105 |  |
| m05_joint_grw_smile_supremacy_w020 | 0.237 | 0.213 | 0.264 | 0.052 | 0.049 | 0.055 | 1.116 | 0.844/0.976/1.001/1.026/1.069 |
| m05_joint_grw_smile_supremacy_w040 | 0.220 | 0.202 | 0.240 | 0.050 | 0.048 | 0.053 | 1.115 | 0.843/0.976/1.001/1.026/1.069 |
| m05_joint_grw_smile_supremacy_w070 | 0.210 | 0.194 | 0.227 | 0.049 | 0.047 | 0.052 | 1.114 | 0.844/0.976/1.001/1.026/1.069 |

# r02 production grid — Task 015 (market-anchored MultiScaleGRW)

Generated 2026-09-13 20:03 at `unknown` on mcmc-beast. Namespace `scottish_lower_grw_market_smile`. Store latest kickoff 2026-09-05; 43 folds.

Sampler: QueuedNUTS 4 × (1000 warmup + 2000 retained), δ = 0.8. Audit on all retained draws; artefact keeps every 4nd draw. Baseline rung = Task 013 run `b0961bc4-c40c-4dbe-9c05-57df7ae0839e`.

## Convergence

| model | folds | oos | draws | max_rhat | min_ess_bulk | min_ess_tail | n_divergent | divergence_rate | treedepth_rate | min_bfmi | strict_rhat_pass | gate_pass | wall_min | run_id |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| m05_joint_grw_smile_supremacy_w070 | 43 | 769 | 2000 | 1.0159 | 504 | 1182 | 0 | 0.00000 | 0.0001 | 0.6944 | false | true | 438.1 | 32d588f1-d666-4112-a7e1-5c9545fbbe3d |

## Pillar posteriors (pooled over folds)

| model | σ_sup_median | σ_sup_q05 | σ_sup_q95 | σ_smile_median | σ_smile_q05 | σ_smile_q95 | κ_median | φ_median |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| m05_joint_grw_smile_supremacy_w070 | 0.210 | 0.194 | 0.227 | 0.049 | 0.047 | 0.052 | 1.114 | 0.844/0.976/1.001/1.026/1.069 |

# r02 production grid — Task 014 (JointGammaNegBinObservation)

Generated 2026-09-12 07:01 at `7e0fc979` on mcmc-beast. Namespace `scottish_lower_grw_joint_negbin`.

Sampler: QueuedNUTS 4 × (1000 warmup + 2500 retained), δ = 0.8. Audit on all retained draws; artefact keeps every 5nd draw.

## Convergence

| model | folds | oos | draws | max_rhat | min_ess_bulk | min_ess_tail | n_divergent | divergence_rate | treedepth_rate | min_bfmi | strict_rhat_pass | gate_pass | wall_min | run_id |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| m12_joint_hybrid_synergy_negbin | 40 | 710 | 2000 | 1.0041 | 1911 | 2518 | 1 | 0.00000 | 0.0005 | 0.6230 | true | true | 111.8 | c3d2aede-ddd5-4590-9c75-5937de4b1bbc |

## Posterior dispersion `r` (reported, not gated)

Pooled over folds. Experiment 02 measured `r̂ ≈ 26.0–26.5` on this league with a single-arm NegBin and TimeDecay state.

| model | median_r | mean_r | r_q05 | r_q95 |
|---|---:|---:|---:|---:|
| m12_joint_hybrid_synergy_negbin | 29.54 | 31.47 | 17.80 | 51.63 |

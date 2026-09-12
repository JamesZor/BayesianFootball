# r02 production grid — Task 014 (JointGammaNegBinObservation)

Generated 2026-09-12 05:04 at `7e0fc979` on mcmc-beast. Namespace `scottish_lower_grw_joint_negbin`.

Sampler: QueuedNUTS 4 × (500 warmup + 1000 retained), δ = 0.8. Audit on all retained draws; artefact keeps every 2nd draw.

## Convergence

| model | folds | oos | draws | max_rhat | min_ess_bulk | min_ess_tail | n_divergent | divergence_rate | treedepth_rate | min_bfmi | strict_rhat_pass | gate_pass | wall_min | run_id |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| m00_baseline_grw_negbin | 40 | 710 | 2000 | 1.0111 | 834 | 670 | 0 | 0.00000 | 0.0061 | 0.7322 | false | true | 59.2 | 0c0da991-7d4c-4f01-90e2-2af157f27aaa |
| m05_wealth_grw_negbin | 40 | 710 | 2000 | 1.0102 | 1049 | 807 | 0 | 0.00000 | 0.0001 | 0.6805 | false | true | 56.5 | 019d41d4-9e0e-41eb-bbc1-8984263f0f14 |
| m10_lineup_grw_negbin | 40 | 710 | 2000 | 1.0105 | 1057 | 882 | 0 | 0.00000 | 0.0055 | 0.7443 | false | true | 63.0 | f7fd8385-fa15-4f6a-ae89-e23447907a80 |
| m12_joint_hybrid_synergy_negbin | 40 | 710 | 4000 | 1.0099 | 689 | 236 | 0 | 0.00000 | 0.0002 | 0.5930 | true | false | 62.2 |  |

## Posterior dispersion `r` (reported, not gated)

Pooled over folds. Experiment 02 measured `r̂ ≈ 26.0–26.5` on this league with a single-arm NegBin and TimeDecay state.

| model | median_r | mean_r | r_q05 | r_q95 |
|---|---:|---:|---:|---:|
| m00_baseline_grw_negbin | 28.40 | 30.29 | 17.09 | 49.92 |
| m05_wealth_grw_negbin | 29.53 | 31.50 | 17.77 | 51.93 |
| m10_lineup_grw_negbin | 28.99 | 30.91 | 17.46 | 50.82 |
| m12_joint_hybrid_synergy_negbin | 29.50 | 31.43 | 17.82 | 51.61 |

# r02 production grid — TODO 021 fast & slow GRW

Generated 2026-09-21 15:57 at `6092de68` on mcmc-beast. Namespace `fast_slow_grw_scottish_lower`.

Sampler: QueuedNUTS 4 × (800 warmup + 800 retained), δ = 0.8. Audit on all retained draws; artefact keeps every 2nd draw.

| model | folds | oos | draws | max_rhat | min_ess_bulk | min_ess_tail | n_divergent | divergence_rate | treedepth_rate | min_bfmi | strict_rhat_pass | gate_pass | wall_min | run_id |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| m01_poisson_grw_tight | 40 | 710 | 1600 | 1.0115 | 732 | 692 | 0 | 0.00000 | 0.0086 | 0.6633 | false | true | 38.3 | 2b42d3bf-28d7-47ac-8706-88798c9031ac |
| m02_poisson_grw_loose_var | 40 | 710 | 1600 | 1.0124 | 682 | 461 | 0 | 0.00000 | 0.0062 | 0.6756 | false | true | 35.5 | 27c8f2f2-e130-44ba-a764-2585d39eb2de |
| m03_poisson_grw_loose_tdist | 40 | 710 | 1600 | 1.0140 | 594 | 673 | 0 | 0.00000 | 0.0003 | 0.5556 | false | true | 29.7 | dafbfe00-54c2-42a6-a87f-2c9de470bca2 |
| m04_poisson_grw_loose_fixed_spread | 40 | 710 | 1600 | 1.0141 | 410 | 791 | 0 | 0.00000 | 0.0027 | 0.8502 | false | true | 32.0 | 514c5533-fea4-4786-8a58-30d8262733cf |

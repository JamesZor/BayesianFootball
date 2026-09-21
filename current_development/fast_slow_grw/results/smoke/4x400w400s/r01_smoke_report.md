# r01 smoke gate — TODO 021 fast & slow GRW

Generated 2026-09-21 12:26 at `5cd005d4` on mcmc-beast with 16 threads. QueuedNUTS 4 × (400 + 400), δ = 0.8. Folds 1, 20, 40. 50 OOS fixtures, 43 with an accepted market inversion.

## Gates and supremacy

| model | max_rhat | min_ess_bulk | n_divergent | sigma0_att | sigma0_def | sup_slope | sup_sd | max_win_prob | n_fav70 | fav70_model | fav70_market | gate_pass | run_id |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| m01_poisson_grw_tight | 1.0206 | 364.7651 | 0 | 0.1940 | 0.2022 | 0.3566 | 0.2691 | 0.5817 | 3 | 0.5098 | 0.7497 | true | cea2a810-4ea8-4a53-83b3-2599d20b0e2c |
| m02_poisson_grw_loose_var | 1.0211 | 449.5776 | 0 | 0.2106 | 0.2181 | 0.4064 | 0.2969 | 0.5776 | 3 | 0.5231 | 0.7497 | true | f14b3e33-7337-437f-b395-9af6c03e9c5e |
| m03_poisson_grw_loose_tdist | 1.0226 | 412.7486 | 0 | 0.1518 | 0.1515 | 0.3629 | 0.2643 | 0.5676 | 3 | 0.5147 | 0.7497 | true | 1c578947-196f-4eab-8785-a98e71f3a731 |

## Rate-pooling ladder (tight ⊕ loose)

| loose | w | sup_slope | sup_sd | max_win_prob | fav70_model | fav70_market |
|---|---:|---:|---:|---:|---:|---:|
| m02_poisson_grw_loose_var | 0.0000 | 0.3566 | 0.2691 | 0.5817 | 0.5098 | 0.7497 |
| m02_poisson_grw_loose_var | 0.2000 | 0.3666 | 0.2744 | 0.5813 | 0.5127 | 0.7497 |
| m02_poisson_grw_loose_var | 0.4000 | 0.3765 | 0.2798 | 0.5807 | 0.5155 | 0.7497 |
| m02_poisson_grw_loose_var | 0.6000 | 0.3865 | 0.2854 | 0.5799 | 0.5182 | 0.7497 |
| m02_poisson_grw_loose_var | 0.8000 | 0.3964 | 0.2911 | 0.5789 | 0.5207 | 0.7497 |
| m02_poisson_grw_loose_var | 1.0000 | 0.4064 | 0.2969 | 0.5776 | 0.5231 | 0.7497 |
| m03_poisson_grw_loose_tdist | 0.0000 | 0.3566 | 0.2691 | 0.5817 | 0.5098 | 0.7497 |
| m03_poisson_grw_loose_tdist | 0.2000 | 0.3579 | 0.2681 | 0.5788 | 0.5109 | 0.7497 |
| m03_poisson_grw_loose_tdist | 0.4000 | 0.3591 | 0.2670 | 0.5756 | 0.5120 | 0.7497 |
| m03_poisson_grw_loose_tdist | 0.6000 | 0.3604 | 0.2661 | 0.5723 | 0.5130 | 0.7497 |
| m03_poisson_grw_loose_tdist | 0.8000 | 0.3617 | 0.2652 | 0.5688 | 0.5139 | 0.7497 |
| m03_poisson_grw_loose_tdist | 1.0000 | 0.3629 | 0.2643 | 0.5676 | 0.5147 | 0.7497 |

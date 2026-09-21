# r01 smoke gate — TODO 021 fast & slow GRW

Generated 2026-09-21 13:40 at `422c424d` on mcmc-beast with 16 threads. QueuedNUTS 4 × (400 + 400), δ = 0.8. Folds 1, 20, 40. 50 OOS fixtures, 43 with an accepted market inversion.

## Gates and supremacy

| model | max_rhat | min_ess_bulk | n_divergent | sigma0_att | sigma0_def | sup_slope | sup_sd | max_win_prob | n_fav70 | fav70_model | fav70_market | gate_pass | run_id |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| m01_poisson_grw_tight | 1.0201 | 422.0802 | 0 | 0.1940 | 0.2042 | 0.3553 | 0.2674 | 0.5826 | 3 | 0.5095 | 0.7497 | true | 08c509b4-bc2f-416d-8da4-29f64f40493f |
| m02_poisson_grw_loose_var | 1.0219 | 451.5831 | 0 | 0.2116 | 0.2164 | 0.4061 | 0.2960 | 0.5758 | 3 | 0.5258 | 0.7497 | true | ea908d04-29c0-4cb6-bbc5-ff5c6d24a692 |
| m03_poisson_grw_loose_tdist | 1.0210 | 386.3298 | 0 | 0.1517 | 0.1491 | 0.3600 | 0.2649 | 0.5682 | 3 | 0.5150 | 0.7497 | true | 32513467-ea91-474f-bade-575fa560a282 |
| m04_poisson_grw_loose_fixed_spread | 1.0246 | 319.4869 | 0 | 0.4768 | 0.4771 | 0.3519 | 0.3161 | 0.6672 | 3 | 0.4568 | 0.7497 | false | 1245631d-e791-40f9-9ae9-46f900a30f08 |

## Draw-concatenation ladder (tight ⊕ loose, ρ = loose share)

| loose | rho | draws | sup_slope | sup_sd | max_win_prob | fav70_model | fav70_market |
|---|---:|---:|---:|---:|---:|---:|---:|
| m02_poisson_grw_loose_var | 0.0000 | 1600 | 0.3553 | 0.2674 | 0.5826 | 0.5095 | 0.7497 |
| m02_poisson_grw_loose_var | 0.2500 | 1600 | 0.3677 | 0.2742 | 0.5811 | 0.5133 | 0.7497 |
| m02_poisson_grw_loose_var | 0.5000 | 1600 | 0.3813 | 0.2813 | 0.5810 | 0.5177 | 0.7497 |
| m02_poisson_grw_loose_var | 0.7500 | 1600 | 0.3932 | 0.2891 | 0.5776 | 0.5223 | 0.7497 |
| m02_poisson_grw_loose_var | 1.0000 | 1600 | 0.4061 | 0.2960 | 0.5758 | 0.5258 | 0.7497 |
| m03_poisson_grw_loose_tdist | 0.0000 | 1600 | 0.3553 | 0.2674 | 0.5826 | 0.5095 | 0.7497 |
| m03_poisson_grw_loose_tdist | 0.2500 | 1600 | 0.3554 | 0.2663 | 0.5793 | 0.5098 | 0.7497 |
| m03_poisson_grw_loose_tdist | 0.5000 | 1600 | 0.3585 | 0.2659 | 0.5735 | 0.5129 | 0.7497 |
| m03_poisson_grw_loose_tdist | 0.7500 | 1600 | 0.3583 | 0.2654 | 0.5686 | 0.5132 | 0.7497 |
| m03_poisson_grw_loose_tdist | 1.0000 | 1600 | 0.3600 | 0.2649 | 0.5682 | 0.5150 | 0.7497 |
| m04_poisson_grw_loose_fixed_spread | 0.0000 | 1600 | 0.3553 | 0.2674 | 0.5826 | 0.5095 | 0.7497 |
| m04_poisson_grw_loose_fixed_spread | 0.2500 | 1600 | 0.3533 | 0.2746 | 0.6040 | 0.4971 | 0.7497 |
| m04_poisson_grw_loose_fixed_spread | 0.5000 | 1600 | 0.3515 | 0.2857 | 0.6246 | 0.4822 | 0.7497 |
| m04_poisson_grw_loose_fixed_spread | 0.7500 | 1600 | 0.3537 | 0.3005 | 0.6462 | 0.4714 | 0.7497 |
| m04_poisson_grw_loose_fixed_spread | 1.0000 | 1600 | 0.3519 | 0.3161 | 0.6672 | 0.4568 | 0.7497 |

### Failures

* `m04_poisson_grw_loose_fixed_spread` —  G6 slope not above tight; 

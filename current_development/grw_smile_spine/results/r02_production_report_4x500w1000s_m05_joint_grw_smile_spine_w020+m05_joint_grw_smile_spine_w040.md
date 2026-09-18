# r02 production grid — Task 016 (1-parameter smile spine)

Generated 2026-09-14 05:28 at `d4cceb8` on mcmc-beast with 16 threads, Julia 1.12.4. Namespace `scottish_lower_grw_smile_spine`. Store latest kickoff 2026-09-05; 43 folds; 769 held-out fixtures.

Sampler: QueuedNUTS 4 × (500 warmup + 1000 retained), δ = 0.8. Audit on all retained draws; artefact keeps every 2nd draw.

## Pinned rungs (loaded, not sampled)

Each is asserted to be the recipe `gss_models()` builds, converged, 43 folds, at the production budget. The baseline's per-fold audit was re-run on thinned chains by `extend_fit`; its ESS is not comparable to the others'.

| rung | source | run_id | folds | max_rhat | min_ess_bulk | min_ess_tail | n_divergent | n_transitions | min_bfmi | wall_min | julia_version | n_threads | git_commit | latents |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| m05_joint_grw_baseline | Task 013 m05_wealth_grw | b0961bc4-c40c-4dbe-9c05-57df7ae0839e | 43 | 1.0115 | 609 | 616 | 0 | 86000 | 0.689 | 42.4 | 1.12.4 | 16 | 037d651c-dirty | CountLatents |
| m05_joint_grw_supremacy_w040 | Task 015 | 0ee58d18-b7e9-4168-8d78-93887b1a8c26 | 43 | 1.0105 | 814 | 516 | 0 | 172000 | 0.683 | 60.5 | 1.12.4 | 16 | unknown | CountLatents |
| m05_joint_grw_smile_supremacy_w020 | Task 015 | fcd5e974-9a46-4a10-9828-6b987a5484d6 | 43 | 1.0139 | 472 | 498 | 0 | 172000 | 0.623 | 158.4 | 1.12.4 | 16 | unknown | detached (T010) |
| m05_joint_grw_smile_supremacy_w040 | Task 015 | 30620d3e-e4bd-4c05-b1a1-85cefa36b728 | 43 | 1.0200 | 431 | 696 | 0 | 172000 | 0.665 | 185.4 | 1.12.4 | 16 | unknown | detached (T010) |

## Convergence and persistence — spine rungs

| model | folds | oos | draws | max_rhat | worst_rhat_fold | min_ess_bulk | min_ess_tail | n_divergent | divergence_rate | treedepth_rate | min_bfmi | strict_rhat_pass | passed | gate_pass | wall_min | run_id | reused |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| m05_joint_grw_smile_spine_w020 | 43 | 769 | 2000 | 1.0135 | 33 | 482 | 854 | 0 | 0.00000 | 0.0000 | 0.6532 | false | true | true | 132.2 | eaf53852-a078-4190-b744-089966a306f6 | false |
| m05_joint_grw_smile_spine_w040 | 43 | 769 | 2000 | 1.0139 | 43 | 531 | 922 | 0 | 0.00000 | 0.0000 | 0.6519 | false | true | true | 176.3 | 582035c0-e145-44f7-9f40-89e25388e79a | false |

## H1 — benchmark against Task 015

Work-package targets, reported not gated: wall ≤ 90 min per rung, min bulk ESS ≥ 600. `wall_min` is read from each artefact's metadata; `readme_wall_min` is Task 015's README figure.

| model | source | max_rhat | min_ess_bulk | min_ess_tail | n_divergent | wall_min | readme_wall_min |
|---|---|---:|---:|---:|---:|---:|---|
| m05_joint_grw_baseline | Task 013 m05_wealth_grw | 1.0115 | 609 | 616 | 0 | 42.4 | — |
| m05_joint_grw_supremacy_w040 | Task 015 | 1.0105 | 814 | 516 | 0 | 60.5 | 60 |
| m05_joint_grw_smile_supremacy_w020 | Task 015 | 1.0139 | 472 | 498 | 0 | 158.4 | 158 |
| m05_joint_grw_smile_supremacy_w040 | Task 015 | 1.0200 | 431 | 696 | 0 | 185.4 | 185 |
| m05_joint_grw_smile_spine_w020 | Task 016 (this run) | 1.0135 | 482 | 854 | 0 | 132.2 | — |
| m05_joint_grw_smile_spine_w040 | Task 016 (this run) | 1.0139 | 531 | 922 | 0 | 176.3 | — |

### Fold by fold, against the five-strike rung at the same weight

| spine | reference | folds | median_bulk_ratio | min_bulk_ratio | median_tail_ratio | folds_spine_bulk_below_reference | spine_min_bulk | reference_min_bulk |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_grw_smile_spine_w020 | m05_joint_grw_smile_supremacy_w020 | 43 | 0.83 | 0.38 | 0.94 | 32 | 482 | 472 |
| m05_joint_grw_smile_spine_w040 | m05_joint_grw_smile_supremacy_w040 | 43 | 1.66 | 0.67 | 1.44 | 6 | 531 | 431 |

Per-fold rows: `r02_fold_benchmark_4x500w1000s_m05_joint_grw_smile_spine_w020+m05_joint_grw_smile_spine_w040.csv`.

## H2 — pillar posteriors and β_spine

Task 015's five-strike medians imply β_LS = 0.0525, with line residuals -0.066 / 0.028 / 0.001 / -0.027 / -0.038 at K = 0…4.

| model | σ_sup_median | σ_smile_median | σ_smile_q05 | σ_smile_q95 | κ_median | β_median | β_q05 | β_q95 | β_sd | φ_median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| m05_joint_grw_smile_spine_w020 | 0.236 | 0.062 | 0.059 | 0.065 | 1.094 | 0.0524 | 0.0499 | 0.0550 | 0.0016 | 0.900/0.949/1.000/1.054/1.111 |
| m05_joint_grw_smile_spine_w040 | 0.219 | 0.060 | 0.058 | 0.063 | 1.092 | 0.0524 | 0.0507 | 0.0543 | 0.0011 | 0.900/0.949/1.000/1.054/1.111 |

| model | fold_β_median_min | fold_β_median_max | max_fold_β_sd | max_β_rhat | min_β_ess_bulk |
|---|---:|---:|---:|---:|---:|
| m05_joint_grw_smile_spine_w020 | 0.0519 | 0.0530 | 0.0018 | 1.0049 | 3502 |
| m05_joint_grw_smile_spine_w040 | 0.0519 | 0.0529 | 0.0012 | 1.0045 | 3258 |

Per-fold rows: `r02_beta_by_fold_<model>.csv`.

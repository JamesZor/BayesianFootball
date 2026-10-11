# Stage 1 artifacts

Main result: [`../REPORT_S1.md`](../REPORT_S1.md). Records stay in this worktree; no hub directory exists.
No production fits, `src` edits, writes to either database, pushes or merges.

## Execution

Julia ran only in our `jl s1` session on mcmc-beast, 16 threads, BLAS 1, isolated directory
`/root/BF_runs/qs_transition`. The frozen W2 cache was read without modifying its checkout:
`/root/BF_runs/a76a65df/.cache/datastore_ScottishPyramid.jls`, md5
`db3ed548191bcd5287a6c1a18e34a105`. Project sources were copied from this worktree with root
`.cache/` and `data/` excluded. An existing beast Manifest supplied the installed environment;
auto-precompilation was disabled. Credentials came from the private environment/.env and are
not artifacts. Runtime dotenv loading is explicit where betdb is needed.

Runners execute **in order in one session**. Each wraps its work in the ten-table audit:

```bash
jl start s1 --beast --threads 16 --dir /root/BF_runs/qs_transition
jl run s1 /root/BF_runs/qs_transition/experiments/scotland/11_qs_transition/eda/r00_inventory.jl
jl run s1 /root/BF_runs/qs_transition/experiments/scotland/11_qs_transition/eda/r01_predictions.jl
jl run s1 /root/BF_runs/qs_transition/experiments/scotland/11_qs_transition/eda/r02_auxiliary.jl
jl run s1 /root/BF_runs/qs_transition/experiments/scotland/11_qs_transition/eda/r03_coverage.jl
jl run s1 /root/BF_runs/qs_transition/experiments/scotland/11_qs_transition/eda/r05_cup_metadata.jl
jl run s1 /root/BF_runs/qs_transition/experiments/scotland/11_qs_transition/eda/r06_likelihood_ids.jl
jl run s1 /root/BF_runs/qs_transition/experiments/scotland/11_qs_transition/eda/r07_close_coverage.jl
jl stop s1
```

`r04_cutoffs.jl` was a diagnostic repair pass during development. Final `r01_predictions.jl`
already invokes the corrected `l01_manifests.jl`; replay does not require the repair pass.
The exported `match_latents.fold_idx` is a storage attachment, **not** actual fixture vintage.
The corrected `folds_*.csv` use canonical held-out matching and metadata.

Sync `eda/results/` back, commit inputs, then run small CSV analysis locally (no DB access):

```bash
uv venv /tmp/qs-s1-venv
uv pip install --python /tmp/qs-s1-venv/bin/python -r experiments/scotland/11_qs_transition/eda/requirements.txt
OPENBLAS_NUM_THREADS=1 /tmp/qs-s1-venv/bin/python experiments/scotland/11_qs_transition/eda/analyze_s1.py
OPENBLAS_NUM_THREADS=1 /tmp/qs-s1-venv/bin/python experiments/scotland/11_qs_transition/eda/analyze_cups.py
OPENBLAS_NUM_THREADS=1 /tmp/qs-s1-venv/bin/python experiments/scotland/11_qs_transition/eda/analyze_diagnostics.py
OPENBLAS_NUM_THREADS=1 /tmp/qs-s1-venv/bin/python experiments/scotland/11_qs_transition/eda/analyze_sensitivity.py
OPENBLAS_NUM_THREADS=1 /tmp/qs-s1-venv/bin/python experiments/scotland/11_qs_transition/eda/analyze_endpoint.py
OPENBLAS_NUM_THREADS=1 /tmp/qs-s1-venv/bin/python experiments/scotland/11_qs_transition/eda/verify_s1.py
```

DuckDB handles file inventory/joins; constrained least squares and covariance calculations handle
small market reconstructions. These are the simple team reconstructions requested by DESIGN,
not Turing/forecast fits. Random diagnostics have fixed seeds. Forward prior means are deterministic.
Figures are generated from the same persisted statistics/predictions.

## Question index

| Question | Main CSVs in `results/` |
|---|---|
| Q1 cause curve | `q1_bias_first20`, `q1_bias_bins`, `q1_club_bias`, `q1_common_panel`, `q1_paired_a2`, `q1_history_bias`, `q1_rate_summary`, `q1_joint_fixture_sensitivity` |
| Q2 prior forecast | `q2_prior_headline`, `q2_prior_unique_fixture`, `q2_prior_predictions`, `q2_first5_predictions`, `q2_prior_parameters`, `q2_paired_error`, `q2_lower_gate_prior` |
| Q3 tier value | `q3_reconstruction_graph`, `q3_tier_cup_estimates`, `q3_tier_mover_estimates`, `q3_bridge_club_sensitivity`, `q3_cup_coverage`, `q3_a2_units` |
| Q4 summer | `q4_window_changes`, `q4_dispersion`, `q4_paired_diffusion`, `q4_qs_scales`, `q4_qs_boundary_proxy` |
| Q5 lag | `q5_age_bias`, `q5_own_since_cutoff`, `q5_conditional_age`, `q5_refit_events`, `q5_vintage_availability` |
| Q6 coverage | `q6_coverage`, `q6_transitions`, `q6_quote_totals`, `q6_cohort_gate`, `q6_w2_close_coverage`, `q6_2627_coverage_only`, `q6_bias_thresholds` |
| Q7 quality/style | `q7_mover_shifts`, `q7_shift_summary`, `q7_opening_shift` |
| Q8 selection/carry | `q8_carry_forward`, `q8_selection_slopes`, `q8_nonmover_overlap`, `q8_endpoint_sensitivity`, `q8_scope_status` |
| Q9 source/rotation | `q9_source_bridge`, `q9_source_by_tier`, `q9_status`; rotation not run |
| Q10 outcomes | `q10_market_residual`, `q10_fixture_logloss`, `q10_paired_model_scores`, `q10_matched_market_residual`, `q10_calibration` |
| Q11 squad | `q11_squad_status`; optional omission explicit |
| Q12 filtration | `q12_source_manifest`, `q12_overlap`, `q12_qs_market_likelihood`, `q12_prediction_chronology`, `q12_cutoff_check_*`, `q12_fold_prior_hashes`, `q12_status` |

## Interpretation and verification

- Means in `q1_bias_*.csv` are percentage points; raw club `delta` is probability.
- Prior D/U errors are log-rate rotation units. First quote can occur after game 1.
- `checkpoint=5` is the latest accepted quote at/before game 5. `mean_first5` scores every accepted
  fixture through game 5. Neither is a future-smoothed isolated club rating.
- `q2_prior_summary` is club-observation weighted; headline/unique-fixture tables deduplicate
  fixtures with two movers. Paired MSE uncertainty is exploratory, especially across repeated
  first-five fixtures. Canonical bias intervals cluster club-season; shared-fixture connected
  sensitivity is also supplied. Small cluster counts prohibit powered-null interpretations.
- Cup estimates are descriptive and timestamp/lineup/venue limited. Only explicit regulation
  goals enter scoring contrasts. No precise cup U/style inversion is claimed.
- `q12_status` distinguishes chronology pass from **implementation blocks** for overlap and
  future-informed historical initial nodes. No undocumented factorisation pass is claimed.
- QS posterior scale proxy is not full boundary predictive variance. Reconstruction error is a
  working LS approximation; cross-coordinate/home-effect and opponent covariance are retained,
  but changing within-window states and inversion error are not cleanly separable.
- Final calendar-date embargo removed calendar-year-labelled held-back cup rows from the early
  raw export before any outcome diagnostic. Git development history preserves that correction.
- `verification.csv` records hard artifact checks, including cohort/coverage/mean parity and all
  DB audits. `artifact_sha256.csv` identifies the result CSVs (except itself and verification).
- `logs/` contains setup/run/verification output; old setup failures are not current scientific
  gate failures. Judge each Julia run by its own `@@JLDONE ... ok` and audit files.

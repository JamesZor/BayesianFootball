# 002 — Complete Experiment 08 Goal Decomposition Production Grid

| Field | Value |
|---|---|
| ID | 002 |
| Title | Complete Experiment 08 Goal Decomposition Production Grid |
| Status | IN_PROGRESS |
| Priority | P1 |
| Assignee | claude |
| Created | 2026-09-10 |
| Updated | 2026-09-10 |
| Related Files / Commits / PRs | [r08_production_grid.jl](../experiments/scottish_lower/08_goal_decomposition/r08_production_grid.jl); [l08_workflow.jl](../experiments/scottish_lower/08_goal_decomposition/l08_workflow.jl); [m00 checkpoints on beast](/root/BF_goal_decomposition/experiments/scottish_lower/08_goal_decomposition/results/m00_recombined_control/checkpoints/) |

## Context & Problem Statement

`m00_recombined_control` completed all 40 folds (160,000 transitions) overnight on `mcmc-beast` in 10 hours 6 minutes, achieving max R-hat 1.0101 and min ESS 680 across all 710 out-of-sample fixtures. All 40 fold checkpoints (`split_001.jls` through `split_040.jls`) are saved on disk.

However, a single divergence in Fold 3 (a 0.000625% divergence rate) triggered the zero-tolerance assert in `l08_workflow.jl` (`max_divergence_rate = eps(Float64)`), which halted the script before persisting `m00` to PostgreSQL `mcmc_experiments` and before executing `m01`, `m02`, and `m03`.

Per project decision, the divergence threshold must be a logged diagnostic flag/warning rather than a hard pipeline stopper, matching the core library's standard (`max_divergence_rate = 0.001`).

## Acceptance Criteria

- [x] Relax divergence threshold in `l08_workflow.jl` to `max_divergence_rate = 0.001` and convert hard error in `l08_assert_promotion` into a logged warning. Sync to `mcmc-beast`.
- [x] Resume `r08_production_grid.jl` on `mcmc-beast` so that `m00` loads its 40 existing checkpoints, passes promotion audit, and persists to PostgreSQL `mcmc_experiments`.
- [ ] Sample all 40 folds for `m01_decomposed_baseline`, `m02_decomposed_team_penalties`, and `m03_decomposed_pressure_own_goals`.
- [ ] Run `r08_evaluate.jl` and `r08_portfolio.jl` to evaluate proper scoring rules (LogLoss, CRPS) and Kelly portfolio returns against canonical benchmarks.
- [ ] Generate `README.md` and final research report with immutable run UUIDs.

## Ideas & Candidate Solutions

- **Flag vs Stopper**: Divergences should be logged in diagnostics summaries and printed as warnings without aborting 10+ hour runs when the rate is miniscule (< 0.1%).
- **Checkpoint Reuse**: `fit_model` in BayesianFootball supports checkpoint directories; `m00`'s 40 folds will load directly without re-sampling.
- **Standalone Fold 3 Resample (Optional)**: If scientific publication requires exactly 0 divergences for `m00`, Fold 3 can be resampled standalone with target acceptance 0.96 without discarding the other 39 folds.

## Work Log & Progress

- [2026-09-10 @antigravity] Created task. Updated `l08_workflow.jl` (`max_divergence_rate = 0.001`, replaced hard error with `@warn`) and synced file to `mcmc-beast`. Existing checkpoints verified on disk.
- [2026-09-10 @claude] Claimed at the user's request in session `claude_todo`,
  after TODO 003 found and fixed the sampler AD bug. Committed the
  `l08_workflow.jl` relaxation unchanged as `4231f9bb` (sha256 32acd30f, identical on
  archpc and the beast). It follows the sampler fix `37532512`. Pushed the branch.
  Cleaned the beast worktree: scp'd or modified files were moved to
  `/root/BF_goal_decomposition_presync_backup_20260910`, and all were byte-identical to
  the pulled commits. Then `git pull --ff-only` to `4231f9bb`, and restored the
  git-ignored `chains.jls`. Kept the 09-09 artefacts
  (`manifest_production_authorised_75c3e3db6f3ca4e8.toml`,
  `production_grid_2026-09-09.log`).
- [2026-09-10 @claude] Resumed the grid at 19:10:26 CEST in tmux
  `goal_decomposition_grid`, with the same command as 09-09
  (`L08_RUN_GRID=true L08_STRICT_SMOKE_ALL4=true`, `-t 16`). Log:
  `results/production_grid_2026-09-10.log`; manifest
  `manifest_production_authorised_29dd0a0d51d92004.toml`. **m00:** "all 40 folds
  restored from checkpoints", audit PASS (R̂ ≤ 1.0101, ESS ≥ 680, 1/160,000 divergences,
  logged as a warning), 710 × 4,000 latents, **persisted run
  `4d16a5e7-66a7-44c3-8816-70ae73c52836`**. Checked independently in
  `mcmc_experiments`: `runs.status = completed`, 40 `fold_results` rows (all
  converged), 710 `match_latents` rows for 710 distinct fixtures. **m01** started
  sampling at 19:11:02 with ReverseDiff. Live evidence: sampling threads at about 79%
  CPU and the GC and other threads at 13% of process CPU, against about 48% and 38%
  under ForwardDiff on 09-09. Progress was 23% of 160 tasks after 2 min 21 s
  (2.05 s/it); on 09-09 m00 had reached 1% after 43 min.
- [2026-09-10 @claude] **m01** sampled all 40 folds × 4 chains in **7 min 24 s** with
  ReverseDiff; m00 had taken 9 h 23 min of sampling with ForwardDiff on 09-09. Audit
  PASS: R̂ ≤ 1.0089, ESS ≥ 754, **0/160,000 divergences**. Persisted run
  **`ae3dfbaa-e646-40dc-9bb0-55f93a47686f`**, confirmed in the DB: completed; 40 fold
  rows, all converged; 710 latents; `git_commit = 4231f9bb-dirty`. m02 started sampling
  at 19:18:32; m03 follows in the same process.
- **Provenance: this grid mixes AD backends.** m00's 40 chains were sampled on
  2026-09-09 at `af2c65ea` with **ForwardDiff** (the `run_sampler` bug; TODO 003).
  The run persisted from them records `git_commit = 4231f9bb-dirty`, the promotion
  commit; "dirty" means only untracked grid logs and manifests. m01–m03 sample with
  **ReverseDiff** (`AutoReverseDiff(compile = true)`) from `37532512` onward. The AD
  backend is not part of `FitConfig`, so config hashes are unchanged. TODO 003 showed
  the two backends give equivalent posteriors on Fold 1 (m00, z sd 0.90, sd ratio
  0.955–1.049). The mix is benign for inference but must be stated wherever the arms
  are compared, and m00 can be re-sampled with ReverseDiff (about minutes) if
  like-for-like provenance is wanted.

## Verification & Findings

- `m00_recombined_control` completed 40 folds in 10h 06m.
- Diagnostics: max R-hat 1.0101, min ESS 680, 1/160,000 divergences (Fold 3). All 40 checkpoint files exist on `mcmc-beast`.
- 2026-09-10 resume: m00 promoted from those checkpoints and persisted as run
  `4d16a5e7-66a7-44c3-8816-70ae73c52836`. The DB row was checked directly
  (completed; 40 folds; 710 latents).
- m01 persisted as run `ae3dfbaa-e646-40dc-9bb0-55f93a47686f`: R̂ ≤ 1.0089, ESS ≥ 754,
  0 divergences, 40 folds, 710 latents; 7 min 24 s with ReverseDiff. m02 and m03 are in
  progress.

# Phase 3 — stopped at missing harness control (2026-10-09)

Source: **8f285a6a**. Ordered launcher stops at qs_joint, 20:14:20 UTC.
No grid retry or scientific change; neither market-observation arm started.

## What completed

| Arm | Persisted UUID | Folds | Max Rhat | Min bulk ESS | Min tail ESS | Divergences | Rhat reruns due |
|---|---|---:|---:|---:|---:|---:|---:|
| grw_joint | 98447840-e95c-420f-92b4-2db6545506a3 | 40 | 1.0100755714 | 641.7834262 | 324.4704888 | 0 | 0 |
| qs_joint | 2a5eee90-3402-40ce-a5e3-f4c6b57fc8ea | 40 | 1.0093676210 | 832.3498809 | 761.0222128 | 0 | 0 |

GRW completes the harness lifecycle, 2008.6993 s include wall (33.48 min).
One GRW fold is below the tail-ESS 400 review gate; preserve the flag, no
Rhat-based rerun warranted. QS samples/persists all 40 folds and passes
filtration, latents, target coverage and actual DB fit_parity. Its convergence
review also passes. QS include fails at 1962.5365 s (32.71 min) during scoring,
not sampling or persistence. No completed grid-arm summary/receipt for QS yet.

Exact evidence: results/phase3/diagnostics_grw_joint.csv,
grid_summary_grw_joint.csv, rerun_selection_grw_joint.csv, run_grw_joint.txt;
blocked_runs_qs_joint.csv, blocked_diagnostics_qs_joint.csv,
blocked_checks_qs_joint.csv. Raw launcher and ANSI-cleaned full arm logs in
results/gate_logs/phase3_*.log. Audit script phase3_blocked_audit.jl uses only
SELECT/load_fit and local CSV writes; no DB writes or new sampling.

## Failure and diagnosis

`LoadError: expected exactly one role=:control; found 0`

Stack: src/harness/scoring.jl:479 `_control_ref` → `score_runs` →
src/harness/stages.jl `grid` → l05_grid.jl:87 `grid_stage` → run.

- l05_grid.jl `grid_stage` accepts `control=nothing`, passes it to unchanged
  Harness.grid, but `run` never supplies a control reference.
- Harness.grid infers a self-control only when candidate.role==:control.
  That correctly permits grw_joint, but not qs_joint or either market arm.
- QS is supplied alone to score_runs with role=:candidate; `_control_ref`
  requires exactly one control unless explicitly supplied and correctly stops.
- Persistence/parity/check-row writes occur BEFORE this scoring call. A
  read-only audit in retained REPL %392 confirms one completed QS grid UUID,
  40 folds, four hard check passes and complete diagnostics. Do not refit it.

## Proposed surgical correction — awaiting manager authorisation

Only the experimental l05/r05 wrapper needs changing, not src/harness.
For non-control arms, construct an explicit RunRef from the final grw_joint
UUID in its grid_summary CSV (including its prescribed rerun, if applicable),
and thread it into both primary and prescribed rerun calls to grid_stage.
Check the selected control belongs to this experiment and is complete before
sampling. Preserve control candidate's existing self-control behaviour.
Add an offline regression exercising forwarding for candidate/rerun paths.

After authorisation, resume under a fresh phase3 attempt label and fresh REPLs.
GRW receipt avoids refitting/rescoring; QS completed DB fit/checkpoints avoid
sampling, so its missing scoring/receipt can complete. Then proceed to the
market arms in the approved order. No budgets/priors/seeds/threshold changes,
reference refits, favourable fit selection or third convergence attempts.

## Stop state

All owned panes closed after evidence/read-only audit: launcher %390,
successful GRW %391, stopped QS %392. Other sessions untouched.
All 80 initial fold checkpoints and both DB fits retained. Beast checkout
still at frozen 8f285a6a. TODO 041 BLOCKED on manager review of the wrapper
correction. Phase 3 incomplete; no Phase 4 paired report or Phase 5 reproduction.

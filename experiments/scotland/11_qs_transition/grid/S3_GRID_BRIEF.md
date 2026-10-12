QS transition experiment, Stage 3: four 60-fold grids, one factor at a time vs qs_marketobs
Role: pi runner (gpt-6.1-sol) in .worktrees/BayesianFootball-qs-transition, branch exp/qs-transition. Stage 3 only.
Goal: run qs_control, qs_a2, qs_entrant, qs_summer on W2's panel, then score each fix vs the control on W2's gate.
Done: §5 DONE WHEN; write REPORT_S3.md in experiments/scotland/11_qs_transition/.
Stop: print PHASES3_DONE alone on its line at the end. One grid at a time. No src edits. No tuning.

## 0. Records

`~/bet_project/hub/experiments/` does not exist. Records go in `experiments/scotland/11_qs_transition/grid/`.
Local commits only; the manager pushes. Don't ask about either.

## 1. Context card

Read in full: `REPORT_S2.md` (especially §7, the exact S3 commands) and `MANAGER_DECISIONS.md`
(D13–D14 are your scope). Read DESIGN.md §4 item 5 for the gate. Nothing else in full.

- **Arms:** `experiments/scotland/11_qs_transition/candidates.jl`: `qs_control`, `qs_a2`,
  `qs_entrant`, `qs_summer`.
  - W2 panel: 60 folds, 1,070 fixtures, target seasons 23/24–25/26, tournaments 56/57.
  - Sampler 4 × (500 + 1,000).
  - Projected about 5.6 h per arm (`build/timing_projection.csv`).
- **Beast checkout** `/root/BF_runs/qs_transition`. Update it to the latest pushed `exp/qs-transition`
  commit first, using the S2 handover's sync pattern (`build/HANDOVER.md`). Check the frozen
  datastore SHA (`a1e89da1…fe12`, REPORT_S2 §7) before each grid.
- **Stale harness row (D14, James authorised):** `harness_checks.id=832`, a failed tape_allocation
  check from the D11 REPL incident.
  - First write the full row to `grid/archived_harness_check_832.csv` and commit it.
  - Then remove **only that row**, using the targeted cleanup in the experiment harness guide §3
    (`rg -n '^## 3' docs/` to find the guide).
  - Then re-run the readiness check (`build/s3_readiness.csv` pattern).
  - No other DB row may change; show a before/after count of `harness_checks` for that recipe hash.
- **Scoring:** W2's recipe (`experiments/scotland/04_pyramid_tier_transition_arms/README.md`
  §Reproduction) with `scripts/score_runs.jl`, scorecard v1.2:
  - `--target-seasons 23/24,24/25,25/26 --expected-fixtures 1070`;
  - each arm scored **against `qs_control`**.

  Gate (DESIGN §4.5):
  - **Gate 1:** Δ target LogLoss vs control, paired fixture-clustered bootstrap (B = 10,000), CI
    upper < +0.002.
  - **Gate 2:** transition bias (model − market win probability of the transitioning club, first 20
    league games, club-season-clustered). |bias| must fall vs control on both relegated_into_L1 and
    promoted_into_L1 in point estimate, with at least one Δ|bias| CI excluding 0.

  Also report:
  - first-10 cohorts;
  - entered-SPFL;
  - compression slope;
  - ECE;
  - O/U 2.5 and BTTS LogLoss;
  - the bias-by-game-bin curve (1, 2–5, 6–10, 11–20), as in S1's `q1_bias_bins`.
- **Rerun rule (James):** a fold with R̂ > 1.05 or divergences > 0.1% gets one seeded rerun.

**Reading rules:** ranged reads or `rg -n`; at most 40 lines of output (longer output goes to
`grid/logs/`); keep `grid/PROGRESS.md` (arm, job ID, start time, ETA) and `grid/DECISIONS_LOG.md`
current.

**Waiting:** launch each grid with `jl eval … --bg`, then `watch_job` and end your turn. Don't poll.
Before a long wait with context over 100K, run `/compact` first.

**After a compaction:** re-read §5 of this brief, then PROGRESS and DECISIONS_LOG; take job IDs and
UUIDs from them, never from memory.

## 2. Rules

- Julia runs only on the beast, via `jl`. Use a **fresh owned REPL per arm** (`jl stop s3` /
  `jl start s3 --beast --threads 16 --dir /root/BF_runs/qs_transition`), as S2 learned.
- **A separate F6 build agent shares the beast** (REPL `f6`, 4 threads). Never touch its REPL or
  checkout. On the beast, `julia` and `mbtop` belong to James.
- No src edits. No changes to recipes, priors, thresholds or samplers after the first grid starts.
- No 26/27 outcomes.

## 3. Tasks

1. Sync the checkout; check the datastore SHA; archive row 832 and re-check readiness.
2. Grids in order: `qs_control`, `qs_a2`, `qs_entrant`, `qs_summer`. Use REPORT_S2 §7's commands,
   with a fresh REPL each time.
3. After each grid:
   - record the run UUID, wall time, R̂, ESS and divergences;
   - apply the rerun rule;
   - **score it vs control as soon as the control exists**;
   - commit the CSVs;
   - add one line to PROGRESS.
4. When all four are done, write `REPORT_S3.md`:
   - a summary of at most 12 lines;
   - the gate table per arm;
   - the bias-by-game-bin curves;
   - convergence;
   - timing;
   - what S4 should combine (passing factors only), plus a note on the planned pooled arm.

## 4. Out of scope

- F6 (a separate agent builds it).
- The pooled arm (S4).
- Combinations.
- The 26/27 test.

## 5. DONE WHEN

1. Four grids completed (or a failed arm documented with its reason), run UUIDs committed.
2. Each arm scored vs control on gates 1 and 2, with CSVs committed.
3. Row 832 is archived to CSV, and the before/after counts are committed.
4. `REPORT_S3.md` is written, everything is committed (local), the REPL is stopped, and
   `PHASES3_DONE` is printed.

## 6. Budget and blocks

- gpt-6.1-sol, thinking medium (runs and plumbing). Use high only to diagnose a failure.
- If you hit the usage limit, write `grid/HANDOVER.md` and stop.
- If a grid fails twice, or a gate cannot be computed, stop and `ask_user_question`.

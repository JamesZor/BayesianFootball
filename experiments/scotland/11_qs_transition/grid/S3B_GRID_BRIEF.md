QS transition experiment, Stage 3b: weekly and pooled 60-fold grids, scored vs qs_control exactly as in S3
Role: pi runner (gpt-6.1-sol) in .worktrees/BayesianFootball-qs-transition, branch exp/qs-transition. Stage 3b only.
Goal: run qs_weekly then qs_pooled_a2 and add them to S3's frozen gate table against qs_control 866a947c.
Done: §4 DONE WHEN; write REPORT_S3B.md in experiments/scotland/11_qs_transition/.
Stop: print PHASES3B_DONE alone on its line at the end. One grid at a time. No src edits. No gate changes.

## 0. Records

`~/bet_project/hub/experiments/` does not exist. Records go in `experiments/scotland/11_qs_transition/grid/`.
Local commits only; the manager pushes. Don't ask about either.

## 1. Context card

Read in full: `REPORT_S3.md` (method, gate table, launch shape) and `grid/PROGRESS.md` (S3's working
notes). Read `f6/REPORT_F6.md` and `pooled/REPORT_POOLED.md` for their grid sections only
(`rg -n '^## ' <file>`). Nothing else in full.

- **The branch now merges F6 and the pooled prep** (`60d9a81b`). `candidates.jl` holds all six
  recipes.
- **Recipe hashes smoked and passing** (the harness readiness check must agree):
  - `qs_weekly` `aaed32f47a4c97e032bfb067b6debd2e4ccc14d934df58301955f4bccffae952`;
  - `qs_pooled_a2` `0db99f621d0e430127f62404c530f07bdbdb22a459f8a345856193e0e09df887`.

  If a hash computed from the merged checkout differs, **stop and ask**; don't re-smoke silently.
- **Checkout:** `/root/BF_runs/qs_transition` (S3's). Sync it to the pushed `60d9a81b` using S3's
  pattern.
  - `qs_weekly` uses S3's frozen lower datastore (SHA `a1e89da1…fe12`).
  - `qs_pooled_a2` needs the pooled datastore. REPORT_POOLED says it copied
    `/root/BF_runs/a76a65df/.cache/datastore_ScottishPyramid.jls` into its checkout
    (`/root/BF_runs/qs_transition_pooled`). Use exactly the cache the pooled smoke used, and check
    its SHA from REPORT_POOLED. Log how you made it available.
- **Launch:** REPORT_S3 §1. Use an absolute `scripts/run_candidates.jl` include, preload
  `l01_entrant_prior.jl` (candidates.jl includes it) as a separate `jl run` before `main`, and use a
  fresh owned `s3` REPL per arm.
- **Scoring:** `grid/r02_score_export.jl`, the same as S3: v1.2, B = 10,000, panel
  `56+57|23/24,24/25,25/26|n=1070`, paired against `qs_control` `866a947c-be65-467d-8dfd-c58aed0ad1f9`.
  Also run the bias-by-game-bin bootstrap that S3 used for its curves.
- **Rerun rule (James):** a fold with R̂ > 1.05 or divergences > 0.1% gets one seeded rerun, using
  S3's `grid/r03_seeded_rerun.jl` pattern.
- **Stale allocation rows:** the D19 standing rule applies (backed up, NULL `run_id`, a later passing
  0 B smoke of the same recipe hash). Log every removal in `grid/DECISIONS_LOG.md`.

**Reading rules:** ranged reads or `rg -n`; at most 40 lines of output; keep `grid/PROGRESS.md` current
(add an S3b section).
**Waiting:** `jl eval … --bg`, then `watch_job`, then end your turn. Run `/compact` before a long wait if
context is over 100K.
**After a compaction:** re-read §5, then PROGRESS and DECISIONS_LOG.

## 2. Rules

- The beast is yours alone for this stage; `julia` and `mbtop` are still James's.
- **No changes to the gate, priors, sampler or recipes.** S3 noted that the frozen gate is stringent,
  because the control's signed relegated bias is near zero by cancellation. Report it as it stands;
  the manager and James decide any gate question.
- No 26/27 outcomes.

## 3. Tasks

1. Sync, verify the hashes and datastores, and run the readiness check.
2. The `qs_weekly` grid: rerun rule, scoring, commit.
3. The `qs_pooled_a2` grid: rerun rule, scoring, commit.
4. Write `REPORT_S3B.md`:
   - a summary of at most 10 lines;
   - the gate table rows for both arms, in S3's format, plus a combined six-arm table;
   - first-10 and entered-SPFL diagnostics;
   - bias-by-game-bin curves;
   - convergence and timing.

## 4. DONE WHEN

1. Both grids completed (or a failure documented with its reason). UUIDs and scores committed.
2. `REPORT_S3B.md` is written, everything is committed (local), the REPL is stopped, and
   `PHASES3B_DONE` is printed.

## 5. Budget and blocks

- gpt-6.1-sol, thinking medium.
- If you hit the usage limit, write `grid/HANDOVER.md` and stop.
- If a grid fails twice, or a hash or datastore doesn't match, stop and `ask_user_question`.

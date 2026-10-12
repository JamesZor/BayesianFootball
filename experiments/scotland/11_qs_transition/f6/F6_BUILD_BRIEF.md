QS transition experiment, F6 build: an opt-in weekly fold cadence, separate from the biweekly dynamics clock
Role: pi implementer (gpt-6.1-sol) in .worktrees/BayesianFootball-qs-transition-f6, branch exp/qs-transition-f6. F6 only.
Goal: add an opt-in fold-cadence setting so qs_control can refit weekly while its state clock stays biweekly, gated, plus a qs_weekly recipe and smoke.
Done: §5 DONE WHEN; write REPORT_F6.md in experiments/scotland/11_qs_transition/f6/.
Stop: print PHASEF6_DONE alone on its line at the end. No grids. Beast: 4 threads only, REPL f6 only.

## 0. Records

`~/bet_project/hub/experiments/` does not exist. Records go in `experiments/scotland/11_qs_transition/f6/`.
Local commits only; the manager pushes and merges. Don't ask about either.

## 1. Context card

Read in full: `experiments/scotland/11_qs_transition/build/F6_DECISION.md` (the design note) and
`MANAGER_DECISIONS.md` D14. Read `REPORT_S2.md` §2 for the gate pattern. Nothing else in full.

- **Why:** S1 found that the promoted-club error grows about +1.68 pp per week since the last refit
  (`eda/results/q5_conditional_age.csv`). F6 tests whether weekly refits help, **holding the QS
  innovation clock at biweekly**, so it is one factor.
- **Coupling today:** `DataScope` has one `dynamics_col` (`src/Data/splitting/scoped.jl:8-42`).
  `_scope_grouped` uses it to build fold periods (`:47-59`), `create_id_boundaries` derives the
  held-out rows from those periods (`:209-242`), the feature builder uses `splitter.scope` for the
  target-state clock (`src/features/builder.jl:175-187`), and `Harness.fit_config` always builds
  `ScopedWalkForwardCV(candidate.scope)` (`src/harness/candidate.jl:103-122`).
- **Design (from F6_DECISION.md):**
  - Add an opt-in fold-cadence field (e.g. `fold_col`), **defaulting to `dynamics_col`**, used only
    for boundary and held-out selection. The biweekly state indices in the builder stay as they are.
  - A weekly fold inside a biweek forecasts from the same state; it must not draw a new biweek
    innovation. That forecast rule needs a dedicated test.
  - Count the real non-empty weekly folds on the frozen W2 panel. Don't assume 120.
- **The control recipe** is `qs_control` in `experiments/scotland/11_qs_transition/candidates.jl`.
  Add `qs_weekly` = `qs_control` plus weekly fold cadence, nothing else.
- **Gates:**
  - every existing scope with the field unset gives identical splits (the same boundary IDs and
    held-out sets for the W2 panel and the wave-2 panel);
  - a wave-2 fold reproduces bit-identically (the `build/r00_legacy.jl` pattern);
  - unit tests for the new field and the forecast rule;
  - a filtration check that no weekly fold trains on a fixture on or after its first held-out day;
  - a normal 2-fold harness smoke of `qs_weekly` with all hard checks (including the 0 B tape).

  There are no new model parameters, so no AD or recovery gates are needed beyond the smoke's own.

**Reading rules:** ranged reads or `rg -n`; at most 40 lines of output (longer output goes to
`f6/logs/`); keep `f6/PROGRESS.md` and `f6/DECISIONS_LOG.md` current.
**After a compaction:** re-read §5, then PROGRESS and DECISIONS_LOG.

## 2. Rules

- **The beast is shared with the S3 grid,** which runs 16 threads in REPL `s3`.
  - Use only REPL `f6`, started with `jl start f6 --beast --threads 4 --dir /root/BF_runs/qs_transition_f6`.
  - Use your own checkout `/root/BF_runs/qs_transition_f6`, built the way S2's `build/HANDOVER.md`
    describes. Exclude `.env`, `Manifest.toml`, `.cache` and `data` from rsync, and copy the frozen
    datastore cache from `/root/BF_runs/qs_transition/.cache/` (check SHA `a1e89da1…fe12`).
  - Never touch `s3`, its checkout, `julia` or `mbtop`.
- src changes are opt-in and default off. Don't change any existing default.
- No grid runs. Smokes only.

## 3. Tasks

1. Implement the field and the forecast rule (opt-in), with unit tests.
2. Run the legacy split identity and the wave-2 bit identity.
3. Count the weekly folds on the W2 panel, then run the filtration check.
4. Add the `qs_weekly` recipe and run its 2-fold smoke.
5. Write a timing projection for a 60-plus-fold weekly grid at 16 threads; mark it as an estimate,
   because the smoke ran at 4 threads under load.
6. Write `REPORT_F6.md`: a summary of at most 10 lines, the gates, the fold count, the timing, and
   the exact grid command.

## 4. Out of scope

- Running the weekly grid (the manager schedules it after S3's control).
- Any other arm.
- Merging into `exp/qs-transition`.

## 5. DONE WHEN

1. All gates in §1 pass, with CSVs and logs in `f6/`.
2. `qs_weekly` is committed with a passing 2-fold smoke, and its run UUID is recorded.
3. `REPORT_F6.md` is written, everything is committed (local), REPL `f6` is stopped, and
   `PHASEF6_DONE` is printed.

## 6. Budget and blocks

- gpt-6.1-sol, thinking high (src change).
- If you hit the usage limit, write `f6/HANDOVER.md` and stop.
- If a gate fails twice, stop and `ask_user_question`.

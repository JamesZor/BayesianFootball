QS transition experiment, S4 prep: a pooled-scope arm (QS + A2 trained on League One/Two + Championship goals)
Role: pi implementer (gpt-6.1-sol) in .worktrees/BayesianFootball-qs-transition-pooled, branch exp/qs-transition-pooled. Prep only.
Goal: a gated, smoked recipe qs_pooled_a2 (train 55-57 goals, market obs 56/57 only, A2 tier term, scored on 56/57) ready for S4.
Done: §5 DONE WHEN; write REPORT_POOLED.md in experiments/scotland/11_qs_transition/pooled/.
Stop: print PHASEPOOL_DONE alone on its line at the end. No grids. Beast: 4 threads, REPL pool only.

## 0. Records

`~/bet_project/hub/experiments/` does not exist. Records go in `experiments/scotland/11_qs_transition/pooled/`.
Local commits only; the manager pushes and merges. Don't ask about either.

## 1. Context card

Read in full: `MANAGER_DECISIONS.md` (D14, D16, D17 are your scope) and `REPORT_S2.md` §2–§3.
Nothing else in full.

- **Why (James, D14):** more data. The Championship (55) is where most clubs relegated into League
  One come from. Training QS on it gives the model those clubs' history directly.
- **Design (manager, D17):**
  - `qs_pooled_a2` = `qs_a2` (`experiments/scotland/11_qs_transition/candidates.jl:34-36`) with the
    scope changed to train `[55, 56, 57]`, target `[56, 57]`, monitor `[55]`, clock `[56, 57]`,
    `cups = :none`, history 2, `dynamics_col = :match_biweek`. Everything else is identical: model,
    priors, sampler, market table.
  - **Market observation stays lower-scope only** (the frozen 56/57 table). Championship fixtures
    have no market row and are masked. That keeps the change one-factor: Championship **goals** only.
  - **The A2 tier term is required.** W1 found that pooling across tiers without tier structure
    compresses tiers and makes the transition bias worse. Choose the anchor the way W2's pooled
    scopes did (`experiments/scotland/04_pyramid_tier_transition_arms/README.md` §2.2: pooled
    scopes use `:first_in_window`), and log the choice.
  - **Not combined with the entrant prior.** Pooling observes the Championship directly, so the
    entrant prior would count it twice.
- **W2's pooled scope pattern:** `experiments/scotland/04_pyramid_tier_transition_arms/candidates.jl:28-50`.
- **QS zero-centring is league-wide.** Check that with three tiers in training the centring and A2
  tier offsets remain identified: 3-seed recovery on a synthetic two-tier/three-tier panel with known
  tier step. Also check that the Championship 25/26 gap (very few market rows, but goals present) is
  handled (goals only, so it doesn't matter for the likelihood).
- **Gates:**
  - if no src change is needed: a 2-fold normal harness smoke with all hard checks (0 B tapes,
    filtration, latents, fit parity), plus the 3-seed tier recovery above;
  - if a src change is needed: opt-in only, plus exact AD ≤ 1e-10, FD, 0 B tapes, unit tests and
    wave-2 bit identity.
  - Panel check: the target fixtures must be exactly the same 1,070 as `qs_control` (W2 panel).

**Reading rules:** ranged reads or `rg -n`; at most 40 lines of output (longer output goes to
`pooled/logs/`); keep `pooled/PROGRESS.md` and `pooled/DECISIONS_LOG.md` current.
**After a compaction:** re-read §5, then PROGRESS and DECISIONS_LOG.

## 2. Rules

- **The beast is shared with the S3 grid** (REPL `s3`, 16 threads).
  - Use only REPL `pool`, started with
    `jl start pool --beast --threads 4 --dir /root/BF_runs/qs_transition_pooled`.
  - Use your own checkout, built as in `build/HANDOVER.md`. Exclude `.env`, `Manifest.toml`,
    `.cache` and `data` from rsync.
  - The pooled scope needs a datastore with tournament 55. Build it read-only, or reuse W2's
    pooled cache if its SHA is recorded (`rg -n 'db3ed548|ScottishPyramid' experiments/scotland/04_*`).
    Log which, and why.
  - Never touch `s3`, `/root/BF_runs/qs_transition`, `/root/BF_runs/qs_transition_f6`, `julia` or `mbtop`.
- No grids. No 26/27 outcomes.

## 3. Tasks

1. Add the recipe `qs_pooled_a2` and run the panel check (1,070 target fixtures, identical to control).
2. Run the tier-identification recovery (3 seeds).
3. Run the 2-fold smoke, including the 23/24 opener.
4. Write a timing projection for the 60-fold grid at 16 threads (an estimate).
5. Write `REPORT_POOLED.md`: a summary of at most 10 lines, the gates, the timing, and the exact
   grid command.

## 4. Out of scope

- Cups and pxG (S1 did not clear the cup identification).
- Combining with the entrant prior.
- Any grid run.

## 5. DONE WHEN

1. The recipe is committed; the panel check, recovery and smoke pass (or a documented failure).
2. `REPORT_POOLED.md` is written, everything is committed (local), REPL `pool` is stopped, and
   `PHASEPOOL_DONE` is printed.

## 6. Budget and blocks

- gpt-6.1-sol, thinking high.
- If you hit the usage limit, write `pooled/HANDOVER.md` and stop.
- If a gate fails twice, stop and `ask_user_question`.

# Claude handover — Phase 0 BLOCKED, not a phase-boundary pass

## Stop reason

`assert_readonly` failed at `l00_phase0_inventory.jl:25` on corrected startup attempt 2.
The runner stopped before `load_fit`, fixture-count/parity gates, or snapshot building.
Do not start Phase 1. No betting result or model comparison has been computed.

## What is done

- TODO 043 created, claimed and now BLOCKED with unchecked acceptance criteria.
- Exact canonical construction and two conflicting later/historical stacks inventoried.
- Historical target recorded before execution: terminal unit wealth 2.5152, 1127 bets;
  tolerance ±0.00005 wealth, exact bet count. Not a newly reproduced result.
- Phase 0 loader/runner committed; source untouched outside experiment files and tracker.
- Startup world-age problem corrected once; its original log retained.
- Both logs retained with no credentials. REPORT.md distinguishes startup failure from
  the read-only gate failure and distinguishes historical numbers from new measurements.

## Frozen addresses / compute

- Local worktree: `/home/james/bet_project/.worktrees/BayesianFootball-qs-experiment`.
- Branch: `exp/qs-goal-vs-market`; setup `8c4fccec`, corrected executed source `9249fbcd`.
- Beast isolated detached worktree: `/root/BF_runs/qs_backtest_phase0` @ `9249fbcd`.
- Its source was advanced with `merge --ff-only` (no merge commit) after initial checkout;
  this should have been a detached checkout under the brief's no-merge rule. Neither the
  experiment branch nor any other beast worktree was merged or switched.
- Beast logs: `/root/BF_runs/logs/qs_backtest/phase0_inventory.log` (attempt 2),
  `phase0_inventory_attempt1.log` (attempt 1).
- Committed copies: `logs/phase0_inventory_attempt1.log`, `logs/phase0_inventory_attempt2.log`.
- Runtime: Julia 1.12.4, persistent REPL, 16 threads, BLAS 1, pinthreads(:cores).
- Owned session `pi_qsbt_phase0`, pane `%451`: **closed and verified absent**. No jobs remain.
- Existing unrelated sessions/worktrees were not touched. No sampling was launched.
- Manifest SHA256: `f86f49b72b987dd4c87ac80d16f132e5342e988fededbe27e54c6afe8133b6cc`.
- Cached datastore SHA256 after attempted startup:
  `c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4`.
- m12 UUID (not loaded): `132df5c2-c742-4e95-8693-3aeb2b2cbaef`, namespace
  `scottish_lower_joint_player_2426`.

## Manager decisions needed before resume

1. Authorise and verify a session-level read-only connection setup. The existing PGOPTIONS
   method did not pass `SHOW default_transaction_read_only`; root cause is not diagnosed.
   Preserve the assertion and the database read-only requirement.
2. Resolve the production-stack discrepancy: the brief explicitly names canonical trust;
   r07 uses BakerMcHale/min=0.0001/no calibration, experiment 07 used frac=0.30/min=0.001,
   and Option B uses different trust plus λ=28. No choice can be hidden behind parity tuning.
3. Once authorised, resume **Phase 0**, not later phases. Snapshot coverage is still pending.

## Final checks

- `./scripts/todo.sh check`: PASS, 43 tasks, AGENTS.md 20283 bytes.
- `git diff --check`: PASS.
- Difftastic review confirms the sole source correction moved the loader include out of
  the closure and added safe MethodError diagnostics; no policy/threshold change.
- Runtime gate: FAIL, not a test-suite pass. No broader tests were run; no src changes.
- Definition of done checked: Phase 0 remains incomplete; all later criteria unchecked.
- Final evidence/tracker commit is pushed to `origin/exp/qs-goal-vs-market`; get its immutable
  address from `git log -1` (the executed numerical source remains `9249fbcd`).

Full verification and limits: [REPORT.md](REPORT.md). Decisions: [DECISIONS_LOG.md](DECISIONS_LOG.md).

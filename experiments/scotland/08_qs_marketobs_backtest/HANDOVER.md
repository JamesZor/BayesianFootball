# Claude handover — Phase 0 BLOCKED at canonical parity

## Stop reason and next decision

Exact canonical m12 T−25 produced **wealth 2.773493294562897 / 1132 bets**.
The manager's frozen target is **2.5152 ±0.00005 / 1127 bets**. Both checks failed.
The runner stopped on the first canonical attempt, completed its finally DB audit,
and did not run option_b_live, remaining snapshots or later phases. No retuning/retry.

Review the canonical target/stack mismatch before authorising any further work. The
pre-recorded historical experiment 07 stack differs, but its causal contribution was
not tested. Do not silently substitute it, relax tolerance or proceed on higher wealth.
A successful phase boundary has **not** been reached; Phase 0 remains incomplete.

## Manager rulings applied

- Static read-path review/checker plus table-wide before/after audits replaces the old
  session-default assertion. Actual SHOW value logged once: **off**, informational.
- Before/after snapshots are **byte-identical**, 10/10 tables unchanged, including
  count/maxima review of all fit/config/harness tables. Audit completed on failure.
- canonical and option_b_live are distinct named policies; policy_inventory.csv and
  REPORT.md report both. All model probabilities stay raw; fitted calibration is out
  of scope. plain_kelly 0.25 retained with canonical caps/execution/λ defaults.
- Canonical is the sole reproduction target and runs first. Historical-control run
  removed, no betdb SQL fallback, no PGOPTIONS changes/debugging, no source allocator work.

## Evidence and what's done

- `results/phase0_reproduction.csv`: canonical failure, 99 slates, 611 built books,
  converged=true, zero build errors.
- `results/phase0_canonical_daily.csv`: 99 daily rows, unit initial bankroll.
- `results/phase0_fixture_universe.csv`: 710 held-out IDs after season restriction.
- `results/phase0_t25_book.csv` and refusals: 90-minute point-in-time price book used.
- `results/phase0_db_before.csv`, `phase0_db_after.csv`, `phase0_db_audit.csv`:
  table counts, timestamp maxima/absence markers and per-table unchanged flags.
- `results/phase0_session_default.csv`, `phase0_execution_status.csv`,
  `phase0_arm_status.csv`, `policy_inventory.csv`: record actual session value,
  chronology, unfinished arms and frozen settings.
- `logs/phase0_resume.log`: executed source hashes, static PASS, SHOW off, gate FAIL,
  finally audit PASS. Old attempt 1/2 logs preserved separately.
- `READ_PATH_AUDIT.md`, `DECISIONS_LOG.md`, `REPORT.md`, `PROGRESS.md`: review/rulings/status.
- TODO 043 remains BLOCKED with acceptance criteria unchecked. No conclusions about
  qs_marketobs, other models, CLV or uncertainty are available.

## Frozen addresses / runtime

- Local: `/home/james/bet_project/.worktrees/BayesianFootball-qs-experiment`.
- Branch: `exp/qs-goal-vs-market`. Executed source: `8018dacf59e62d9b2780cc2f6a26f428ab52419d`.
- Beast: `/root/BF_runs/qs_backtest_phase0` @ `8018dacf` (detached checkout, not merged).
- Beast log: `/root/BF_runs/logs/qs_backtest/phase0_resume.log`.
- m12 loaded once by UUID `132df5c2-c742-4e95-8693-3aeb2b2cbaef`, namespace
  `scottish_lower_joint_player_2426`; saved predictions only, no refit.
- Owned session pi_qsbt_phase0 `%452`: closed and verified absent. No jobs remain;
  old %451 also closed. No unrelated sessions/worktrees touched on resume.
- Julia 1.12.4, persistent REPL, 16 threads, BLAS 1, pinthreads(:cores).
- Datastore SHA256 (unchanged):
  `c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4`.
- Manifest SHA256:
  `f86f49b72b987dd4c87ac80d16f132e5342e988fededbe27e54c6afe8133b6cc`.

## Verification / incomplete work

- Static checker PASS; four injected negative cases correctly rejected.
- Runtime audited DB tables unchanged 10/10; local `cmp` confirms byte-identical snapshots.
- Canonical reproduction FAIL; required bet count and wealth both differ.
- Snapshot aggregates T−25/T−6h/T−24h remain unfinished; later snapshots unbuilt.
- Phase 1/2/3/4 work has not started. No broader test suite run; no src changes.
- Final tracker/diff checks and evidence commit are pushed on the allowed branch; use
  `git log -1` for the handback commit (executed source stays `8018dacf`).

# Wave 4 progress

- Phase: 0 — BLOCKED at canonical reproduction; check recorded 2026-10-10 11:10Z.
- Done: manager rulings recorded before execution; canonical / option_b_live / plain_kelly raw-probability inventories reported; transitive read paths reviewed; static gate PASS and four injected negative cases rejected.
- Executed source: `8018dacf`, one resumed runner. Actual SHOW default `off` logged once; no assertion/debugging/PGOPTIONS changes. Before/after counts/timestamp maxima byte-identical across 10 tables, including finally audit after failure.
- Canonical m12 T−25: wealth 2.773493294562897, 1132 bets, 99 slates, 611 built books, converged=true, zero build errors. Target 2.5152 ±0.00005 and 1127 bets: BOTH FAIL.
- Saved fit loaded once by UUID; saved OOS universe restricted to 710 IDs. No refitting/tuning/DB writes/betdb connection/src changes.
- Pending: canonical parity resolution; option_b_live numerical reference; all snapshot coverage/staleness summaries and T−6h/T−24h books; all later phases. T−25 book/refusals already saved from reproduction, not a completed coverage inventory.
- Next action: Claude reviews target/stack mismatch and authorises any further Phase 0 work. No automatic retry or Phase 1. Old session-default blocker is superseded by manager ruling, not current.
- Open panes/jobs: NONE. Resume pane `%452`, pi_qsbt_phase0, closed/session absent; old %451 closed. No Julia backtest process remains.
- Beast isolated worktree: `/root/BF_runs/qs_backtest_phase0` @ `8018dacf`, detached checkout, Julia 1.12.4 with 16 threads/core pinning/BLAS 1. No unrelated sessions/worktrees touched on resume.
- Logs: `/root/BF_runs/logs/qs_backtest/phase0_resume.log`; committed local copy in `logs/`. Old attempt 1/2 logs preserved.
- Evidence: `results/phase0_reproduction.csv`, canonical_daily, fixture_universe, t25_book/refusals, db_before/after/audit, session_default, execution_status, arm_status, policy_inventory. Current REPORT/HANDOVER/DECISIONS_LOG updated.
- Definition of done checked: NOT MET. Canonical reproduction failed; three-snapshot coverage incomplete. TODO 043 BLOCKED, not closed.

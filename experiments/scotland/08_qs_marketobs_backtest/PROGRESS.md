# Wave 4 progress

- Phase: 0 — inventory and reproduction (IN_PROGRESS).
- Done: clean branch verified; TODO 043 created/claimed; canonical and Option B policy sources inspected; historical m12 target identified (wealth 2.5152, 1127 bets); configuration and gate frozen in DECISIONS_LOG.md before numerical work.
- Next: await beast reproduction output; stop if either parity gate fails. Coverage follows only after a passing exact-production gate.
- Open pane: beast `pi_qsbt_phase0`, `%451`, Julia 1.12.4, `--project -t 16 -i`; include submitted 2026-10-10 ~10:40Z. No sampling job IDs (backtest only).
- Isolated beast source: `/root/BF_runs/qs_backtest_phase0` @ `8c4fccec`; log `/root/BF_runs/logs/qs_backtest/phase0_inventory.log`; outputs `experiments/scotland/08_qs_marketobs_backtest/results/` beneath that worktree. Manifest/.env are existing beast symlinks; datastore copied from previously used qs_experiment cache. Other panes/worktrees untouched.
- Databases: read-only; no model fitting permitted.

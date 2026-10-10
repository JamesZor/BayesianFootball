# Wave 4 progress

- Phase: 0 — BLOCKED at experiment DB read-only-session gate (2026-10-10 10:53Z).
- Done: TODO 043 created/claimed; exact canonical / historical / Option B stacks inventoried; target and policy frozen before execution; Phase 0 loader/runner committed; two attempt logs retained; report and handover written.
- Attempt 1 (`8c4fccec`): startup module-binding MethodError, no data or gate reached. Startup-only correction made.
- Attempt 2 (`9249fbcd`): datastore cache loaded, then assert_readonly failed at loader line 25. No load_fit, no numerical reproduction, no snapshots, no later phases.
- Next action: Claude authorises a read-only connection method that passes the retained assertion and resolves the production-stack discrepancy. No automatic retry. Resume Phase 0 only.
- Open panes/jobs: NONE. Beast `pi_qsbt_phase0`, `%451`, closed and verified absent. No sampling jobs were created.
- Beast isolated source: `/root/BF_runs/qs_backtest_phase0` @ `9249fbcd`; runtime Julia 1.12.4, 16 threads, BLAS 1, core-pinned.
- Beast logs: `/root/BF_runs/logs/qs_backtest/phase0_inventory.log` (attempt 2), `phase0_inventory_attempt1.log`. Copies committed in this experiment's `logs/`.
- Results: no fixture/book/reproduction CSVs were produced. `results/phase0_execution_status.csv` records only execution status, not measurements.
- Databases: one experiment SHOW query for the safety assertion; no fit loaded, no SQL writes, betdb not queried. Read-only safety not established, hence STOP.
- Definition of done: NOT MET. Parity and all three snapshot coverage reports remain pending. TODO 043 stays BLOCKED, not closed.

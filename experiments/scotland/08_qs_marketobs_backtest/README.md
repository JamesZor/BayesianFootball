# 08 — qs_marketobs executable-price backtest

**Phase 0 BLOCKED:** the experiment DB connection failed the read-only-session assertion.
No saved fit was loaded and no reproduction, snapshot coverage or betting result was computed.
The policy inventory is complete; the backtest is not. No new fits or database writes occurred.
TODO 043 remains open; the owned beast pane is closed. See [REPORT.md](REPORT.md) and
[HANDOVER.md](HANDOVER.md) for the evidence and decisions required before resuming.

## Scope and source of truth

Brief: [`../../pi_qs_marketobs_backtest_prompt.md`](../../pi_qs_marketobs_backtest_prompt.md).
Manager: Claude. Task: [043](../../../todos/043_wave_4_scenario_kelly_backtest_of_qs_marketobs_at_executable_prices.md).
The frozen stack and distinction between canonical defaults, Option B and experiment
07's historical book are in [DECISIONS_LOG.md](DECISIONS_LOG.md).

## Phase 0 execution

- Loader: [l00_phase0_inventory.jl](l00_phase0_inventory.jl), definitions only.
- Runner: [r00_phase0_inventory.jl](r00_phase0_inventory.jl), Phase 0 only, no server or ledger.
- Run in an owned beast persistent tmux REPL with `julia --project -t 16 -i`.
- The runner pins cores, fixes BLAS to 1, and enforces PostgreSQL read-only sessions.
- Outputs: `results/phase0_*`; beast log: `/root/BF_runs/logs/qs_backtest/phase0_inventory.log`.
- Failed reproduction prevents subsequent snapshots and every later phase from running.

## Interpretation limits

The point-in-time builder uses last-traded prices, not executable back/lay touch or depth.
Even a completed archive-price backtest will not prove that all stakes could be filled.
Historical experiment 07 helped select the production policy using these same two seasons;
freezing that policy now does not remove that in-sample policy-selection risk.

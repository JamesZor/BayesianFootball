# 08 — qs_marketobs executable-price backtest

**Phase 0 BLOCKED at canonical parity:** m12 wealth **2.7734933**, **1132 bets**;
required **2.5152 ±0.00005**, **1127 bets**. No retry or retuning.
Static read-path checks passed and all ten before/after DB table audits were unchanged.
SHOW default was `off`, recorded rather than asserted under the manager ruling.
Both canonical and option_b_live policy inventories are reported; fitted calibration is out of scope.
Option B's numerical run, remaining snapshots and later phases stopped at the failed gate.
TODO 043 remains BLOCKED; owned beast pane closed. See [REPORT.md](REPORT.md) and [HANDOVER.md](HANDOVER.md).

## Scope and source of truth

Brief: [`../../pi_qs_marketobs_backtest_prompt.md`](../../pi_qs_marketobs_backtest_prompt.md).
Manager: Claude. Task: [043](../../../todos/043_wave_4_scenario_kelly_backtest_of_qs_marketobs_at_executable_prices.md).
The frozen stack and distinction between canonical defaults, Option B and experiment
07's historical book are in [DECISIONS_LOG.md](DECISIONS_LOG.md).

## Phase 0 execution

- Loader: [l00_phase0_inventory.jl](l00_phase0_inventory.jl), definitions only.
- Runner: [r00_phase0_inventory.jl](r00_phase0_inventory.jl), Phase 0 only, no server or ledger.
- Run in an owned beast persistent tmux REPL with `julia --project -t 16 -i`.
- The runner pins cores, fixes BLAS to 1, statically checks read paths and audits DB counts/timestamp maxima before/after (including failure exits). See [READ_PATH_AUDIT.md](READ_PATH_AUDIT.md).
- Outputs: `results/phase0_*`; beast log: `/root/BF_runs/logs/qs_backtest/phase0_resume.log`.
- Failed reproduction prevents subsequent snapshots and every later phase from running.

## Interpretation limits

The point-in-time builder uses last-traded prices, not executable back/lay touch or depth.
Even a completed archive-price backtest will not prove that all stakes could be filled.
Historical experiment 07 helped select the production policy using these same two seasons;
freezing that policy now does not remove that in-sample policy-selection risk.

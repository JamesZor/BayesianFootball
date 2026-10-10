# 08 — qs_marketobs executable-price backtest

**Phase 0 PASS under ruling 3:** exact experiment07 historical cell reproduced
**2.515204600600047 wealth / 1127 bets**. [Recipe and line citations](HISTORICAL_REPRODUCTION_RECIPE.md).
Canonical **2.7734933 / 1132** is retained separately, no rerun; raw Option B reference
**2.2077361 / 1172**. All three snapshots plus T−24h sensitivity are complete;
10/10 DB audits unchanged. Owned pane closed, no later phase ran.
**PHASEBT_HANDOVER** — see [REPORT.md](REPORT.md) and [HANDOVER.md](HANDOVER.md).

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
- Current outputs: `results/phase0_historical/`; beast log: `/root/BF_runs/logs/qs_backtest/phase0_historical.log`. Prior canonical outputs/logs preserved separately; `results/phase0_policy_reference.csv` distinguishes gate/policy roles and executed sources.
- Failed historical reproduction prevents snapshots and later phases; a pass still stops at the Phase0 boundary.

## Interpretation limits

The point-in-time builder uses last-traded prices, not executable back/lay touch or depth.
Even a completed archive-price backtest will not prove that all stakes could be filled.
Historical experiment 07 helped select the production policy using these same two seasons;
freezing that policy now does not remove that in-sample policy-selection risk.

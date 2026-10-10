# 08 — qs_marketobs backtest at archive prices (wave 4, TODO 043)

**Done.** All five models, four snapshots and five policies (100 cells) were backtested after the 2% commission. The experiment 07 gate reproduces bit for bit.

- qs_marketobs does not beat m12 at T−25m or T−6h. It leads only in the low-coverage T−24h subsets (78–289 bets), not decisively, and on fixtures where it already leads at T−25m.
- Scenario Kelly helps the market-informed models at T−25m but does not change who is first.
- The trust tiers add nothing consistent.
- CLV is positive but small.
- Everything is in-sample (2 seasons, 99 slates, policies chosen in experiment 07), and the intervals are wide.

The full answers and per-snapshot tables are in **[REPORT.md](REPORT.md)**.

## Scope

- Briefs: [`../../pi_qs_marketobs_backtest_prompt.md`](../../pi_qs_marketobs_backtest_prompt.md) (Phases 0–2, pi) and [`../../claude_qs_backtest_phase34_brief.md`](../../claude_qs_backtest_phase34_brief.md) (Phases 3–4, Claude CLI).
- Task: [043](../../../todos/043_wave_4_scenario_kelly_backtest_of_qs_marketobs_at_executable_prices.md). Frozen choices and rulings: [DECISIONS_LOG.md](DECISIONS_LOG.md).
- No new fits, no DB writes, src frozen. Julia ran only on mcmc-beast, in owned persistent REPLs with 16 threads and BLAS 1.

## Phases

| Phase | Content | Record |
|---|---|---|
| 0 | Policy inventory, experiment 07 reproduction (2.515204600600047 / 1127), snapshot coverage | [HISTORICAL_REPRODUCTION_RECIPE.md](HISTORICAL_REPRODUCTION_RECIPE.md), `results/phase0_historical/` |
| 1 | Allocator audit: per-match Kelly, no shared-draw daily objective | [PHASE1_ALLOCATOR_AUDIT.md](PHASE1_ALLOCATOR_AUDIT.md) |
| 2 | `ScenarioKelly` in src plus the experiment's daily-slate adapter | [PHASE2_SCENARIO_KELLY.md](PHASE2_SCENARIO_KELLY.md), `l01_*` |
| 3 | 5 × 4 × 5 grid, CLV, block bootstrap, paired m12 differences | `l02_phase3_grid.jl`, `r02_phase3_grid.jl`, `results/phase3/`, `logs/phase3_grid.log` |
| 4 | Report, policy contrasts, same-fixture timing check, CLV intervals | [REPORT.md](REPORT.md), `x04_phase4_tables.py`, `results/phase4/` |

## Reproduce

- **Phase 3.** On beast, in an owned `jl` REPL started at the package root with `-t 16`, include `r02_phase3_grid.jl`. It writes to `results/phase3/` and runs the static read check and the before/after DB audit itself. It needs the saved datastore cache; there is no SQL fallback.
- **Phase 4 (local, no Julia or DB).** Run `uv run experiments/scotland/08_qs_marketobs_backtest/x04_phase4_tables.py`. The REPORT tables are duckdb queries over `results/phase3/` and `results/phase4/`.

## Limits

- **Prices are last-traded archive observations**, not executable back/lay touch or depth. T−24h at 360 minutes uses prices up to 6 hours old.
- **Policy selection is in-sample.** Experiment 07 chose these policies on the same two seasons using m12, so the comparison favours m12.
- Open follow-ups are at the end of [REPORT.md](REPORT.md).

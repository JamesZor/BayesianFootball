# Claude handover — wave 4 DONE

Phases 3 and 4 are complete. **QBT_DONE.**

- **Phase 3:**
  - The gate m12 × T−25m × historical_exp07 = 2.515204600600047 / 1127 is bit-identical.
  - 100/100 cells were computed.
  - The DB audit was unchanged 10/10 on both runner attempts.
  - Details are in DECISIONS_LOG.md and `results/phase3/`.
- **Phase 4:** [README.md](README.md) and [REPORT.md](REPORT.md) contain a 10-line summary, tables per snapshot, the five answers, the in-sample risks and the interval widths. The derived tables are in `results/phase4/`, produced locally by `x04_phase4_tables.py` from the committed CSVs.
- **Manager's additions:**
  - REPORT states that T−24h/360 prices are up to 6 hours old and not executable at bet time.
  - Every early-snapshot claim carries its n_bets.
- **Ownership:**
  - No owned beast REPL remains: `jl-claude_qsbt_p3` was stopped after Phase 3, and Phase 4 used no Julia.
  - The beast checkout `/root/BF_runs/qs_backtest_phase0` is detached at `4f1d3d43`.
  - Unrelated panes were not touched.
- **Follow-ups:** listed at the end of REPORT.md and in TODO 043. None is started.

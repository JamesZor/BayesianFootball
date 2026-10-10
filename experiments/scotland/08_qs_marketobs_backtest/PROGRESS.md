# Wave 4 progress — DONE (Claude CLI)

- Phases 0–2 (pi) accepted; Claude CLI ran Phases 3–4 under `experiments/claude_qs_backtest_phase34_brief.md`.
- **Phase 3 PASS.** The gate is bit-identical (2.515204600600047 / 1127). 100/100 cells were computed, the DB audit was unchanged 10/10, and all 1,435 scenario solves converged. Runner `4f1d3d43`, results in `results/phase3/`.
- **Phase 4 done.** README.md and REPORT.md answer the five questions with traceable numbers. Derived tables come from the local `x04_phase4_tables.py` and are in `results/phase4/`.
- Headline: qs_marketobs does not beat m12 at T−25m or T−6h. Its T−24h lead is a fixture-subset effect: low coverage, with T−24h/360 prices up to 6 hours old. Scenario Kelly helps the market-informed models at T−25m. Trust tiers add nothing consistent. CLV is positive but small. All results are in-sample, with wide intervals.
- TODO 043 is COMPLETED with follow-ups, everything is pushed, and no owned beast session remains.

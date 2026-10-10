# Wave 4 progress — Phase 3 PASS (Claude CLI)

- Handover from pi (Codex fault + weekly quota): Phases 0–2 accepted; Claude CLI runs Phases 3–4 under `experiments/claude_qs_backtest_phase34_brief.md`. Rulings 1–3 binding.
- Phase 3 design frozen in DECISIONS_LOG before any grid number.
- **Gate PASS, bit-identical:** m12 × T−25m × historical_exp07 = 2.515204600600047 / 1127 / 99 slates.
- Cross-checks inside the grid: canonical 2.773493294562897 / 1132 and option_b_live 2.207736087761667 / 1172 equal Phase 0 exactly.
- 100/100 cells (5 models × 4 snapshots × 5 policies) COMPUTED; C0 is draw-level (512), so none NOT_AVAILABLE. Bootstrap intervals (8-week noncircular blocks, 999 reps, 90%) and paired m12 differences in committed CSVs.
- All 1,435 scenario-Kelly daily solves converged; KKT > 1e-3 only on budget-binding days (a residual-tolerance artifact, logged as a follow-up).
- DB audit unchanged 10/10 on both runner attempts. Runner `4f1d3d43`; results `results/phase3/`; log `logs/phase3_grid.log`.
- Next: Phase 4 (README.md + REPORT.md answering the five questions), on manager's go.

# Wave 3 progress (TODO 042)

Brief: `experiments/claude_qs_fusion_bias_brief.md`. Branch `exp/qs-goal-vs-market`.
Beast checkout `/root/BF_runs/qs_experiment`; logs `/root/BF_runs/logs/qs_experiment_w3/`;
outputs `/root/BF_runs/qs_experiment_w3_out/`. Namespace `scottish_lower_qs_wave3_2426`.

## Pins

- Snapshot SHA256 `c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4`, mtime `2026-09-25 12:57:15.480765468 UTC`.
- Frozen market table (wave 2) SHA256 `680bc2283e9c0fc7a896bf01b4ba209cb3aae06f50d00667c98b02f379872549`.

## Current

- **Phase:** 4 COMPLETE (a1 PASS, `PHASE4.md`, `REPORT.md`). QF3_HANDOVER printed.
- **Done:** Phases 0–3 (see PHASE0–3.md); Phase 3 runs fusion_qs_bias `26de19bf-fc29-4818-bfdf-7bf76535901f` (control), fusion_qs_nobias `5afdfeb3-abad-4d23-82fb-1bfe6babe797`, fusion_grw_bias `b2e5a527-e51e-4913-b7bb-e1b94a4d1ff2`. Phase 4: score+leakage at 4f7ee7bc, record at 9cf7d0de; references exact; primary −0.000325 1X2 LogLoss n.d.; κ_D > 1 in 40/40 folds; leakage PASS; register status `scored`; boards regenerated (additions only).
- **Flags for manager:** control fold-40 tail ESS 330.7 (bias posteriors typical); qs_nobias recovery 21/27; 6 circular contradictions (none primary), incl. one noncircular "worse" (qs_bias − grw_bias, all markets) and a context-only noncircular "better" (grw_bias − close, all markets).
- **Next action:** on manager approval, Phase 5: refit 1 fold per new arm from the frozen seed in a fresh REPL, byte-for-byte compare (wave 2 `r07_reproduce.jl` pattern), REPRODUCIBILITY.md, record status `completed` (`QSX3_RECORD_STATUS=completed`, r08 expects `phase5/<arm>/reproduction.csv`), then QF3_DONE. Background waits: `ssh -n` until-grep with an explicit long Bash timeout.
- **Open beast panes:** none.

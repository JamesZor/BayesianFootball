# Wave 3 progress (TODO 042)

Brief: `experiments/claude_qs_fusion_bias_brief.md`. Branch `exp/qs-goal-vs-market`.
Beast checkout `/root/BF_runs/qs_experiment`; logs `/root/BF_runs/logs/qs_experiment_w3/`;
outputs `/root/BF_runs/qs_experiment_w3_out/`. Namespace `scottish_lower_qs_wave3_2426`.

## Pins

- Snapshot SHA256 `c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4`, mtime `2026-09-25 12:57:15.480765468 UTC`.
- Frozen market table (wave 2) SHA256 `680bc2283e9c0fc7a896bf01b4ba209cb3aae06f50d00667c98b02f379872549`.

## Current

- **Phase:** 1 COMPLETE (PASS, `PHASE1.md`). Awaiting manager: QF3_HANDOVER printed.
- **Done:** Phase 0; Phase 1 fusion observation at 01f42219, all 8 gates pass (attempt a1); evidence in `results/phase1/`.
- **Flag for manager:** recovery pooled 84.8% passes, but qs_nobias alone is 21/27 (77.8%).
- **Next action:** Phase 2 — `candidates.jl` (fusion_qs_bias control, fusion_qs_nobias, fusion_grw_bias), smoke every arm, then AskUserQuestion before the grid.
- **Open beast panes:** none.

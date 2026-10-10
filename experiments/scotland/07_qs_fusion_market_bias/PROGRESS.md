# Wave 3 progress (TODO 042)

Brief: `experiments/claude_qs_fusion_bias_brief.md`. Branch `exp/qs-goal-vs-market`.
Beast checkout `/root/BF_runs/qs_experiment`; logs `/root/BF_runs/logs/qs_experiment_w3/`;
outputs `/root/BF_runs/qs_experiment_w3_out/`. Namespace `scottish_lower_qs_wave3_2426`.

## Pins

- Snapshot SHA256 `c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4`, mtime `2026-09-25 12:57:15.480765468 UTC`.
- Frozen market table (wave 2) SHA256 `680bc2283e9c0fc7a896bf01b4ba209cb3aae06f50d00667c98b02f379872549`.

## Current

- **Phase:** 2 smoke PASS (hard checks; `PHASE2.md`). Awaiting manager grid approval (AskUserQuestion).
- **Done:** Phase 0; Phase 2 smoke a1 (UUIDs in `PHASE2.md`); Phase 1 fusion observation at 01f42219, all 8 gates pass (attempt a1); evidence in `results/phase1/`.
- **Flag for manager:** recovery pooled 84.8% passes, but qs_nobias alone is 21/27 (77.8%).
- **Next action:** on approval, Phase 3 grid (qs_bias → qs_nobias → grw_bias, control forwarding as wave 2 `l05_grid.jl`). Full ETA 13.343 h.
- **Open beast panes:** none.

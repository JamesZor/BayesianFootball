# Wave 3 progress (TODO 042)

Brief: `experiments/claude_qs_fusion_bias_brief.md`. Branch `exp/qs-goal-vs-market`.
Beast checkout `/root/BF_runs/qs_experiment`; logs `/root/BF_runs/logs/qs_experiment_w3/`;
outputs `/root/BF_runs/qs_experiment_w3_out/`. Namespace `scottish_lower_qs_wave3_2426`.

## Pins

- Snapshot SHA256 `c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4`, mtime `2026-09-25 12:57:15.480765468 UTC`.
- Frozen market table (wave 2) SHA256 `680bc2283e9c0fc7a896bf01b4ba209cb3aae06f50d00667c98b02f379872549`.

## Current

- **Phase:** 5 COMPLETE — package COMPLETE (TODO 042 COMPLETED). QF3_DONE printed.
- **Done:** Phases 0–4 (PHASE0–4.md, REPORT.md). Phase 3 runs fusion_qs_bias `26de19bf-fc29-4818-bfdf-7bf76535901f` (control), fusion_qs_nobias `5afdfeb3-abad-4d23-82fb-1bfe6babe797`, fusion_grw_bias `b2e5a527-e51e-4913-b7bb-e1b94a4d1ff2`. Phase 4: primary −0.000325 1X2 LogLoss n.d.; decision no promotion. Phase 5 a1 PASS at `8998b355` (REPRODUCIBILITY.md): fold 1 of each arm byte-identical (chains and latents, max abs 0), 123 grid checkpoints unchanged, runs count unchanged; register status `completed`.
- **Flags carried into TODO 042 follow-ups:** control fold-40 tail ESS 330.7; qs_nobias recovery 21/27; 6 circular contradictions (none primary).
- **Next action:** none in this package. Follow-ups are listed in TODO 042.
- **Open beast panes:** none; beast checkout clean at `8998b355`.

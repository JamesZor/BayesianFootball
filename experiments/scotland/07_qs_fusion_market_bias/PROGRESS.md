# Wave 3 progress (TODO 042)

Brief: `experiments/claude_qs_fusion_bias_brief.md`. Branch `exp/qs-goal-vs-market`.
Beast checkout `/root/BF_runs/qs_experiment`; logs `/root/BF_runs/logs/qs_experiment_w3/`;
outputs `/root/BF_runs/qs_experiment_w3_out/`. Namespace `scottish_lower_qs_wave3_2426`.

## Pins

- Snapshot SHA256 `c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4`, mtime `2026-09-25 12:57:15.480765468 UTC`.
- Frozen market table (wave 2) SHA256 `680bc2283e9c0fc7a896bf01b4ba209cb3aae06f50d00667c98b02f379872549`.

## Current

- **Phase:** 3 RUNNING — `phase3_gate.sh a1` at dde8a441 launched 01:04 UTC 2026-10-10; queue check PASS (10/10); fusion_qs_bias PASS 01:59 UTC (run `26de19bf-fc29-4818-bfdf-7bf76535901f`, 54 min, max R̂ 1.0194, no reruns, 0 divergences, tail ESS < 400 on fold 40 only: 330.7); fusion_qs_nobias PASS 02:53 UTC (run `5afdfeb3-abad-4d23-82fb-1bfe6babe797`, 53 min, max R̂ 1.0123, no reruns, 0 divergences, min tail ESS 764.6; control score parity exact, 664 rows); fusion_grw_bias started 02:53 UTC.
- **Done:** Phase 0; Phase 2 smoke a1 (UUIDs in `PHASE2.md`); Phase 1 fusion observation at 01f42219, all 8 gates pass (attempt a1); evidence in `results/phase1/`.
- **Flag for manager:** recovery pooled 84.8% passes, but qs_nobias alone is 21/27 (77.8%).
- **Next action:** wait for `PHASE3_GRIDS_PASS` / `PHASE3_BLOCKED` in `/root/BF_runs/logs/qs_experiment_w3/phase3_launcher_a1.log` (background `ssh -n` until-grep with an explicit long Bash timeout: without one, background commands die at the 120 s default). On a block, resume with a fresh attempt label (checkpoints and receipts resume). Full ETA 13.343 h.
- **Open beast panes:** `claude_qsx3_p3_<arm>` (one at a time, owned by the gate).

# Wave 3 progress (TODO 042)

Brief: `experiments/claude_qs_fusion_bias_brief.md`. Branch `exp/qs-goal-vs-market`.
Beast checkout `/root/BF_runs/qs_experiment`; logs `/root/BF_runs/logs/qs_experiment_w3/`;
outputs `/root/BF_runs/qs_experiment_w3_out/`. Namespace `scottish_lower_qs_wave3_2426`.

## Pins

- Snapshot SHA256 `c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4`, mtime `2026-09-25 12:57:15.480765468 UTC`.
- Frozen market table (wave 2) SHA256 `680bc2283e9c0fc7a896bf01b4ba209cb3aae06f50d00667c98b02f379872549`.

## Current

- **Phase:** 3 COMPLETE (grid a1 PASS, `PHASE3.md`). QF3_HANDOVER printed.
- **Done:** Phase 0; Phase 1 at 01f42219 (`PHASE1.md`); Phase 2 smoke a1 (`PHASE2.md`); Phase 3 grid a1 at dde8a441, 01:04–03:48 UTC: fusion_qs_bias `26de19bf-fc29-4818-bfdf-7bf76535901f` (control), fusion_qs_nobias `5afdfeb3-abad-4d23-82fb-1bfe6babe797`, fusion_grw_bias `b2e5a527-e51e-4913-b7bb-e1b94a4d1ff2`; no reruns, 0 divergences, control parity exact; evidence `results/phase3/`.
- **Flags for manager:** recovery pooled 84.8% passes, but qs_nobias alone is 21/27 (77.8%); control fold 40 tail ESS 330.7 < 400 (R̂ 1.0194, not rerun).
- **Next action:** on manager approval, Phase 4 scoring (wave 2 `r06_score.jl` / `r06_interval_audit.jl` / `r10_leakage_audit.jl` pattern), bias posteriors by fold (read δ_M and log κ jointly). Background waits: `ssh -n` until-grep with an explicit long Bash timeout.
- **Open beast panes:** none.

# F6 progress
Phase: COMPLETE — final local commit / handover.
Done: backward-compatible opt-in fold_col wrapper (approved by James); unchanged default DataScope layout; separate held-out clock; fitted-biweek QS forecasts without innovation draws; qs_weekly identical control model/sampler.
Gates: final units505 PASS; split identity1368 signatures PASS; frozen weekly panel115 folds/1070 fixtures PASS; strict day filtration all115 PASS; 55 same-state folds/531 fixtures. Legacy chain396000 +latents160000 values byte-identical, maxdiff0, reference81fdf817-f82d-4f66-9bfd-5f456c4fe29f.
Smoke103140-600 PASS six hard checks; UUID9346d3b3-874c-49eb-9bc3-0db9f07de477. Receipt103819-27315 PASS exactly0 B on both tapes and canonical DB has_passing_smoke=true. Convergence REVIEW Rhat1.06250 and performance REVIEW ratio2.08675 remain disclosed.
Timing: largest-fold four-thread probe47.15384s; sampling-only ideal16-thread estimate5.64864h for115 folds (2.94711h for60). No grid ran. REPORT_F6.md and HANDOVER.md carry command, assumptions and evidence.
Next: manager only — review flags and schedule after S3; do not rerun verified F6 gates.
Open REPLs/jobs: none; owned beast jl f6 STOPPED. Legacy102604-1090 and panel102849-9259 ok; initial legacy deserialization failure and panel-runner getfield error resolved on second attempts and preserved in logs. No gate failed twice. No S3 session/checkout touched.
Owned checkout: /root/BF_runs/qs_transition_f6. Legacy checkpoints archived outside checkout at /root/BF_runs/logs/f6_legacy_checkpoints; manifest copied here. Frozen Pyramid SHA reverified a1e89da1a25aaaa8bc7476307338b9e29a58d5d33beee8e25e25e6126989fe12.
Constraints: local commits only; no pushes/merges/grids; records here because hub directory is absent. End with PHASEF6_DONE.

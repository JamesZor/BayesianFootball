# S3 progress — COMPLETE

## S3b — preflight complete, weekly next
- Confirmed unchanged weekly115 folds, pooled60 folds (James clarification); same1070 fixtures.
- Owned checkout /root/BF_runs/qs_transition fast-forwarded52c87cba; frozen cache and pooled smoke cache both match approved SHA. Both recipe hashes match; readiness true, no stale-row removals.
- Fresh owned s3 beast16 running; separate entrant preload completed. Readiness172219-11336 ok; CSV evidence local.
- Next: launch qs_weekly alone, watch completion; score and seeded rerun offending folds once, commit. Then fresh s3 for qs_pooled_a2, report and stop.
- No active grid yet. Original S3 evidence below remains final.

## Original S3

Phase: four grids, one-factor gates, diagnostics and REPORT_S3.md complete.
Owned REPL s3: STOPPED; no active jobs/watches. No other REPL/checkout touched.

| Arm | Scored UUID | Grid job | Outcome |
|---|---|---|---|
| qs_control | 866a947c-be65-467d-8dfd-c58aed0ad1f9 | 100145-31050 | completed; no rerun |
| qs_a2 | 2e13b786-7d47-47c0-8e04-1529e705c663 | 112704-30834 | completed; one seeded rerun for9 folds, final fold53 review blocker |
| qs_entrant | c6118c34-a2a5-46fd-b4d6-21ef7b795b49 | 142543-15880 | completed after authorised preload retry; no rerun |
| qs_summer | 13b5d0fb-0f6a-43f0-a9bf-9d5b6efb856c | 154435-670 | completed; no rerun |

A2 original21ab856f-7c72-46be-8732-dc6122a58dc4 preserved; rerun130952-11594.
All3 fixes Gate1 PASS, Gate2 FAIL; no qualifying S4 factor.
Stale832 archive committed31a20acb before targeted deletion095941-23352: recipe14→13, global684→683. Readiness095948-15911 all four true.
Frozen datastore SHA checked before every grid; no source/recipe/sampler tuning or26/27 outcomes.
Final W2 score export170809-1508:2656 v1.2 rows; bias bins171027-26163; initial A2 timing171134-12060.
Report: ../REPORT_S3.md. CSVs: runs, scores per arm, fold diagnostics, reruns, gate_table, cohort_bias, target_metrics, bias_game_bins, timing, raw observations; entrant manifests preserved.
Next action: none in S3. See HANDOVER.md for S4 decision. Local commits only; manager pushes.

# S3 progress — COMPLETE

## S3b — COMPLETE
- Confirmed unchanged weekly115 folds, pooled60 folds (James clarification); same1070 fixtures.
- Owned checkout /root/BF_runs/qs_transition fast-forwarded52c87cba; frozen cache and pooled smoke cache both match approved SHA. Both recipe hashes match; readiness true, no stale-row removals.
- Fresh owned s3 beast16 running; separate entrant preload completed. Readiness172219-11336 ok; CSV evidence local.
- Next: launch qs_weekly alone, watch completion; score and seeded rerun offending folds once, commit. Then fresh s3 for qs_pooled_a2, report and stop.
- Weekly grid172312-4833 ok: d62bd802-1434-4c01-a1c5-26512b776d5e,115 folds/1070 fixtures, Rhat1.0218,2/460000 divergences; no possible offending fold (total div2, all Rhat<1.05). Hard/review PASS.
- W2 export195252-9846 and bins195447-1978 ok, weekly score/folds/checks/raw observations committed with five-arm summaries. Gate1PASS/Gate2FAIL. r02/summary only adapt weekly count115; scoring/gates unchanged.
- Weekly commit ecb4321d. Weekly REPL stopped; fresh s3 beast16 with separate preload195611-20748 ok. Pooled195701-7327 ok, initial0fa6219a-97cc-47c9-95bf-aec0235c89ef,60folds/1605 total latents/1070 target. Rhat1.0236,215div; offending41,45,51,53. Initial score222745-25190 ok; initial folds/checks/scores archived. Rerun223346-26600 ok, finaledf3138e-ec55-4e14-aa35-bb624b16f6c4: replacement div0,0,1,2 on41,45,51,53; aggregate38, Rhat1.0236. Final scoring/bins224438-5675 completed; r05 in that job initially rejected extra Championship monitor observations. Explicit frozen lower-panel restriction fixed export224813-23506:1326 paired rows, cohort means/counts parityPASS. No sampling/score change.
- D23 metrics complete:54 CSV rows, all6 posthoc criterionFAIL. Frozen gate all5 candidates Gate1PASS/Gate2FAIL. Synthetic equal-club weighting/abs-before-mean/pairing/single-cluster/self checks pass; all18 raw cohort signed means/counts match scorer.
- Owned s3r STOPPED; James's s3 left untouched as instructed. No active worker watches/jobs. Final cache/source diff receipt clean. Pooled+D23 evidence committed31050d01; REPORT_S3B.md rendered/validated, HANDOVER updated. All requested outputs complete; no next S3b action. Local commits only; manager pushes.
- Manager says leave s3 untouched (James may be attached). Submissions223036-3592/223106-21125 timed out before start; no further s3 commands, no stop. End phase by stopping s3r only.
- Added task D23: DESIGN9 read; compute posthoc MAE_club (equal club-season weighting), paired cluster bootstrap B10000, worst-club signed-mean magnitude and1–10/11–20 splits for all6 arms/3 cohorts. Frozen gate unchanged. Original S3 evidence below remains final.

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

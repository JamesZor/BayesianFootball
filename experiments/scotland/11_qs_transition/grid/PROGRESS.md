# S3 progress

Phase: entrant launch failed once before sampling; retry running; control/A2 completed and scored.
Owned REPL: fresh s3 beast16, /root/BF_runs/qs_transition; checkout c5948a16, SHA checked.
Entrant first job133204-24475 failed before sampling: no extractor method for Main.QSTransitionInputs.EntrantPriorFeature (possible Julia world age after dynamic candidate include).
Active retry133352-18711; fresh s3 beast16, same checkoutca65dcc3, using Base.invokelatest(main). Start2026-10-12 13:33:52 BST; ETA~15:14 BST if successful. If retry fails, STOP and ask per brief. Rerun130952-11594 succeeded~19min; final scoring133007-15380 succeeded.
A2 initial grid112704-30834 done1h35m; UUID21ab856f-7c72-46be-8732-dc6122a58dc4, Rhat1.0365, div293/240000, hardPASS/reviewFAIL. Initial paired scores exported job130716-10483.
One seeded rerun completed for folds6,15,25,35,38,42,44,53,57. Final UUID2e13b786-7d47-47c0-8e04-1529e705c663; maxRhat1.0255, ESSbulk290, div66/240000. Fold53 still25div/4000 (review blocker), no second rerun. Initial CSVs retained *_initial.csv.
Control job100145-31050 completed ok (1h15m reported wall; ~79min job).
Done: pushed checkout d3f86b29 synced; frozen cache SHA verified; row832 archived/committed31a20acb; cleanup095941-23352 recipe rows14→13; readiness095948-15911 all four true.
Control: 866a947c-be65-467d-8dfd-c58aed0ad1f9;60 folds, Rhat1.022804, bulkESS277, tailESS414, div0/240000; no offending folds/no rerun required. Hard/review PASS. W2 score export112304-23673:664 v1.2 score rows (B10000), CSVs saved.
Next: watch retry133352-18711; if second failure ask user and wait; otherwise record entrant UUID/diagnostics/rerun/score vs control, then summer. Gates/bins/report due. Gates/bias-bin curves assembled after paired arm scoring.
Run UUIDs: control866a947c-be65-467d-8dfd-c58aed0ad1f9. Others not started.

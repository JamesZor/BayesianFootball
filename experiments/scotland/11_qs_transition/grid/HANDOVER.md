# S3b handover — complete

## S3b decision and records
- REPORT_S3B.md (one directory up): weekly115 + pooled60 grids, frozen six-arm gates, posthoc DESIGN9 metrics, first10/entered cohorts, bin curves, convergence/timing. Both new arms Gate1PASS/Gate2FAIL; all6 posthoc absolute-error criterionFAIL. No combination or26/27 launch authorised.
- Weekly final d62bd802-1434-4c01-a1c5-26512b776d5e: no offending folds/no reruns; wall8694.71s.
- Pooled final edf3138e-ec55-4e14-aa35-bb624b16f6c4; immutable initial0fa6219a-97cc-47c9-95bf-aec0235c89ef. Exactly one seeded rerun each41,45,51,53; final no offending folds. Initial wall8783.90s; final95.14s is replay-only. See timing_pooled_initial.csv and reruns_qs_pooled_a2.csv.
- runs.csv now6 final UUIDs. Scores, gate_table, convergence, cohort_bias, target_metrics, bins and timing include all6; initial pooled exports preserved.
- abs_error_table.csv:54 rows (6arms×3cohorts×3splits), equal-club-season per-game MAE, paired B10000 bootstrap; worst signed-mean magnitude, point delta/CI, counts, UUIDs. transition_errors.csv1326 paired inputs. All18 signed means/counts reproduce frozen scoring within1e-10. absolute_error.py --test covers weighting/pairing/cancellation/single cluster/self-reference; report_s3b.py reproduces report.
- No S3b stale rows removed. Both exact approved hashes/pass smoke confirmed in s3b_readiness.csv and histories. Frozen cache unchanged, source/recipes/Project/Manifest clean.

## Current operational caveat (overrides historical S3 text below)
Owned s3r STOPPED; no active worker watch/job. Leave James's s3 untouched per manager ruling: he may be attached. Rerun submission223036-3592 and diagnostic223106-21125 in s3 timed out before @@JLSTART; do not assume they ran or clear them without permission. Subsequent reruns/scoring used s3r only. Timing may include s3 contention, not measured isolated-core throughput.
Local commits941584bd (preflight), ecb4321d (weekly),31050d01 (pooled + D23); final report/handover commit follows. No push; manager pushes. No remaining S3b work.

# Historical S3 handover — four grids complete

## Decision
All3 fixes pass W2 target LogLoss non-inferiority but fail the frozen two-direction transition-bias gate. No factor qualifies for S4 combination. A2 additionally retains a per-fold divergence blocker after its one authorised seeded rerun. Retain control; pooled goals-only lower+Championship scope is separate planned S4 work. F6 was not touched. No26/27 outcomes/test.

## Canonical records
- ../REPORT_S3.md: summary, gates, first10/entered cohorts, bin curves, ECE/compression/derivative LL, convergence, timing, S4 decision.
- runs.csv:4 final UUIDs; scores_qs_*.csv: v1.2 paired vs control, B10000,1070 panel.
- gate_table.csv, convergence.csv, cohort_bias.csv, target_metrics.csv, bias_game_bins.csv.
- fold_diagnostics_qs_*.csv and checks_qs_*.csv; A2 *_initial.csv retained.
- rerun_run_qs_a2.csv and reruns_qs_a2.csv: original/final UUID,9 seeds/folds.
- timing.csv (A2 is replay-only), timing_a2_initial.csv (true original grid wall).
- observations_qs_*.csv and entrant_prior_manifests/: reproducibility inputs.
- archived_harness_check_832.csv, stale_check_counts.csv, s3_readiness.csv: authorised removal of only832, recipe14→13/global684→683.
- summarize_s3.py rebuilds local gate/diagnostic tables; r02_score_export.jl and r04_bias_bins.jl run on beast only.

## Run UUIDs
Control866a947c-be65-467d-8dfd-c58aed0ad1f9.
A2 final2e13b786-7d47-47c0-8e04-1529e705c663; initial21ab856f-7c72-46be-8732-dc6122a58dc4.
Entrantc6118c34-a2a5-46fd-b4d6-21ef7b795b49.
Summer13b5d0fb-0f6a-43f0-a9bf-9d5b6efb856c.

## Operational caveats
Owned beast checkout /root/BF_runs/qs_transition only. Owned s3 STOPPED, no pending job.
No push/merge; manager pushes local branch exp/qs-transition.
For an entrant grid, fresh REPL must preload l01_entrant_prior.jl as a separate jl run before main; two initial dynamic dispatch failures occurred before sampling, then James authorised diagnosis/retry. Do not rerun verified grids.
A2 fold53 still25/4000div; no second rerun allowed. Final aggregate rate cannot conceal it.
Entered-SPFL cluster bootstrap has one priced club-season; intervals collapse and are not powered evidence.
Frozen cache SHA a1e89da1a25aaaa8bc7476307338b9e29a58d5d33beee8e25e25e6126989fe12; source/recipes unchanged.
Original four fit walls total5.418h (not S2 projection22.37h), plus~19min A2 reruns and plumbing.

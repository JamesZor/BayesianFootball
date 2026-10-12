# Stage 3b — weekly and pooled grids complete; neither clears both gates

## Summary (9 lines)
- Weekly uses its unchanged 115-fold recipe; James approved the clarification to the brief's 60-fold title.
- Pooled A2 uses 60 folds; both arms score the frozen 1,070 lower-tier fixtures in 23/24–25/26.
- Weekly and pooled both pass LogLoss non-inferiority, but both fail frozen transition-bias Gate 2.
- Weekly needs no reruns; pooled folds 41,45,51,53 each received exactly one seeded rerun.
- Final weekly and pooled fits have no offending folds under James's R̂/divergence rule.
- All six arms also fail the separate **post hoc** DESIGN §9 absolute-error criterion.
- Weekly has the largest primary relegation MAE point reduction (−0.329 pp), but its CI crosses zero.
- Entered-SPFL remains one priced club-season; degenerate bootstrap CIs do not imply certainty.
- No recipes, priors, sampler, frozen gates or src changed; no 26/27 outcomes scored.

## 1. Frozen gate table — combined six arms

Scoring is S3's v1.2, B=10,000, panel `56+57|23/24,24/25,25/26|n=1070`,
paired against `qs_control` `866a947c-be65-467d-8dfd-c58aed0ad1f9`.
Gate 1: fixture-clustered all-market ΔLL upper 95% CI <0.002.
Frozen Gate 2: both first-20 direction absolute signed-mean biases must decrease,
with at least one paired club-season-clustered Δ|bias| CI excluding zero. Bias units pp.
The control's signed aggregate relegation bias is near zero by cancellation; this gate
is intentionally retained, not substituted after seeing results.


| Arm | ΔLL [CI] | Gate1 | Relegated bias; Δ\|bias\| [CI], pp | Promoted bias; Δ\|bias\| [CI], pp | Gate2 | Both |
| --- | --- | --- | --- | --- | --- | --- |
| Control | +0.000000 [+0.000000,+0.000000] | reference | +0.067; +0.000 [+0.000,+0.000] | +1.290; +0.000 [+0.000,+0.000] | reference | — |
| A2, post-rerun | +0.000313 [-0.000429,+0.001104] | PASS | +1.256; +1.189 [-1.855,+1.530] | +0.543; -0.747 [-1.164,-0.373] | FAIL | FAIL |
| Entrant | +0.000212 [-0.000491,+0.001064] | PASS | +0.755; +0.687 [-1.503,+1.862] | +1.170; -0.120 [-0.385,+0.021] | FAIL | FAIL |
| Summer | +0.000267 [-0.000504,+0.001027] | PASS | -0.464; +0.396 [-0.736,+0.841] | +0.684; -0.607 [-1.069,-0.059] | FAIL | FAIL |
| Weekly | -0.000343 [-0.001204,+0.000507] | PASS | +0.255; +0.187 [-0.627,+0.192] | +1.275; -0.015 [-0.356,+0.297] | FAIL | FAIL |
| Pooled A2, post-rerun | +0.000320 [-0.000432,+0.001119] | PASS | +1.282; +1.215 [-1.912,+1.540] | +0.551; -0.739 [-1.158,-0.364] | FAIL | FAIL |

### S3b rows only

| Arm | ΔLL [CI] | Gate1 | Relegated bias; Δ\|bias\| [CI], pp | Promoted bias; Δ\|bias\| [CI], pp | Gate2 | Both |
| --- | --- | --- | --- | --- | --- | --- |
| Weekly | -0.000343 [-0.001204,+0.000507] | PASS | +0.255; +0.187 [-0.627,+0.192] | +1.275; -0.015 [-0.356,+0.297] | FAIL | FAIL |
| Pooled A2, post-rerun | +0.000320 [-0.000432,+0.001119] | PASS | +1.282; +1.215 [-1.912,+1.540] | +0.551; -0.739 [-1.158,-0.364] | FAIL | FAIL |

## 2. Post hoc DESIGN §9 absolute transition error (D23)

James approved D23 after S3. This is **post hoc on 23/24–25/26**, not a replacement
for the frozen table above; it is pre-registered only for the later 26/27 test (not run here).
Primary `MAE_club` = mean across club-seasons of each club-season's mean
`|P_model(win) − P_market(win)|` over its first 20 league games. Absolute value is taken
**before** the per-club mean; club-seasons receive equal weight, regardless of priced game count.
Δ vs control uses paired resampling of club-season clusters, B=10,000, percentile 95% CI,
seed 20260927. Resample whole cluster summaries, never independent arm samples.
Post-hoc criterion: neither relegated/promoted Δ point is positive and at least one CI upper <0;
Gate 1 is still required. Entered-SPFL is descriptive, not a third gate direction.

All six fail this criterion. A2, entrant, weekly and pooled improve both direction points,
but no direction CI upper is negative. Summer worsens both primary points.
Worst-club is the maximum club-season `|mean signed error|`, **not** maximum MAE.
Units throughout are pp. Full precision, UUIDs, cluster/observation counts and all splits:
`grid/abs_error_table.csv`; per-game paired inputs: `grid/transition_errors.csv`.


| Arm | Cohort | MAE_club | ΔMAE [CI] | Worst-club magnitude | Club-seasons |
| --- | --- | --- | --- | --- | --- |
| Control | relegated_into_L1 | 5.802 | +0.000 [+0.000,+0.000] | 5.044 | 5 |
| Control | promoted_into_L1 | 4.952 | +0.000 [+0.000,+0.000] | 2.715 | 6 |
| Control | entered_spfl | 5.088 | +0.000 [+0.000,+0.000] | 0.949 | 1 |
| A2, post-rerun | relegated_into_L1 | 5.750 | -0.052 [-0.822,+0.605] | 4.102 | 5 |
| A2, post-rerun | promoted_into_L1 | 4.837 | -0.115 [-0.317,+0.078] | 1.170 | 6 |
| A2, post-rerun | entered_spfl | 5.493 | +0.406 [+0.406,+0.406] | 0.615 | 1 |
| Entrant | relegated_into_L1 | 5.789 | -0.013 [-0.553,+0.750] | 4.273 | 5 |
| Entrant | promoted_into_L1 | 4.868 | -0.084 [-0.243,+0.017] | 1.913 | 6 |
| Entrant | entered_spfl | 5.639 | +0.551 [+0.551,+0.551] | 1.193 | 1 |
| Summer | relegated_into_L1 | 6.010 | +0.208 [+0.032,+0.426] | 5.678 | 5 |
| Summer | promoted_into_L1 | 5.014 | +0.062 [-0.117,+0.296] | 1.244 | 6 |
| Summer | entered_spfl | 5.372 | +0.284 [+0.284,+0.284] | 1.092 | 1 |
| Weekly | relegated_into_L1 | 5.473 | -0.329 [-0.855,+0.198] | 4.055 | 5 |
| Weekly | promoted_into_L1 | 4.867 | -0.085 [-0.427,+0.236] | 2.719 | 6 |
| Weekly | entered_spfl | 4.468 | -0.619 [-0.619,-0.619] | 0.660 | 1 |
| Pooled A2, post-rerun | relegated_into_L1 | 5.752 | -0.050 [-0.859,+0.624] | 4.072 | 5 |
| Pooled A2, post-rerun | promoted_into_L1 | 4.832 | -0.120 [-0.339,+0.097] | 1.137 | 6 |
| Pooled A2, post-rerun | entered_spfl | 5.470 | +0.382 [+0.382,+0.382] | 0.539 | 1 |

### Post hoc early/late primary metric

| Arm | Cohort | Games1–10 MAE; Δ [CI] | Games11–20 MAE; Δ [CI] |
| --- | --- | --- | --- |
| Control | relegated_into_L1 | 6.639; +0.000 [+0.000,+0.000] | 5.021; +0.000 [+0.000,+0.000] |
| Control | promoted_into_L1 | 4.364; +0.000 [+0.000,+0.000] | 5.517; +0.000 [+0.000,+0.000] |
| Control | entered_spfl | 4.483; +0.000 [+0.000,+0.000] | 5.618; +0.000 [+0.000,+0.000] |
| A2, post-rerun | relegated_into_L1 | 6.584; -0.054 [-1.598,+1.218] | 4.975; -0.046 [-0.097,-0.011] |
| A2, post-rerun | promoted_into_L1 | 4.168; -0.196 [-0.590,+0.210] | 5.477; -0.040 [-0.096,+0.022] |
| A2, post-rerun | entered_spfl | 5.310; +0.827 [+0.827,+0.827] | 5.654; +0.037 [+0.037,+0.037] |
| Entrant | relegated_into_L1 | 6.605; -0.034 [-1.127,+1.465] | 5.002; -0.019 [-0.047,+0.005] |
| Entrant | promoted_into_L1 | 4.218; -0.147 [-0.457,+0.041] | 5.488; -0.029 [-0.057,-0.006] |
| Entrant | entered_spfl | 5.389; +0.906 [+0.906,+0.906] | 5.859; +0.241 [+0.241,+0.241] |
| Summer | relegated_into_L1 | 6.588; -0.051 [-0.373,+0.225] | 5.484; +0.463 [+0.114,+0.851] |
| Summer | promoted_into_L1 | 4.597; +0.233 [-0.078,+0.535] | 5.429; -0.088 [-0.422,+0.271] |
| Summer | entered_spfl | 4.780; +0.297 [+0.297,+0.297] | 5.889; +0.272 [+0.272,+0.272] |
| Weekly | relegated_into_L1 | 6.212; -0.427 [-1.222,+0.294] | 4.796; -0.225 [-0.639,+0.143] |
| Weekly | promoted_into_L1 | 4.251; -0.113 [-0.413,+0.196] | 5.470; -0.047 [-0.451,+0.345] |
| Weekly | entered_spfl | 4.233; -0.249 [-0.249,-0.249] | 4.674; -0.943 [-0.943,-0.943] |
| Pooled A2, post-rerun | relegated_into_L1 | 6.577; -0.062 [-1.691,+1.214] | 4.990; -0.032 [-0.097,+0.022] |
| Pooled A2, post-rerun | promoted_into_L1 | 4.160; -0.204 [-0.608,+0.252] | 5.473; -0.044 [-0.083,-0.002] |
| Pooled A2, post-rerun | entered_spfl | 5.292; +0.809 [+0.809,+0.809] | 5.626; +0.009 [+0.009,+0.009] |

Only 5 relegated and 6 promoted club-seasons are priced. Entered-SPFL has one
club-season (7 first10 /15 first20 observations), so all its paired cluster intervals
collapse to a point. That is a sample limitation, not strong evidence of an effect.


## 3. First-10 and entered-SPFL signed-bias diagnostics


| Arm | Relegated first10, pp | Promoted first10, pp | Entered-SPFL first10 /first20, pp |
| --- | --- | --- | --- |
| Control | +0.626 | +1.151 | -0.650 / +0.949 |
| A2, post-rerun | +3.021 | -0.353 | -1.470 / +0.615 |
| Entrant | +2.039 | +0.928 | -0.570 / +1.193 |
| Summer | +0.283 | +0.476 | -0.817 / +1.092 |
| Weekly | +0.910 | +1.212 | -1.195 / +0.660 |
| Pooled A2, post-rerun | +3.099 | -0.311 | -1.579 / +0.539 |

First10 priced observations: relegated46 (44 fixtures), promoted55 (53 fixtures). Full CIs/counts remain in `grid/cohort_bias.csv` and per-arm score exports.

## 4. Bias-by-game-bin curves

Signed model-minus-market win probability, pp. Game numbers count **all league games before intersecting the priced panel**, as S3; no renumbering around missing prices. `grid/bias_game_bins.csv` includes B=10,000 clustered CIs/counts for all six arms and all three cohorts.

| Direction /arm | Game1 | Games2–5 | Games6–10 | Games11–20 |
| --- | --- | --- | --- | --- |
| relegated_into_L1 / Control | +0.163 | +1.905 | -0.373 | -0.468 |
| relegated_into_L1 / A2, post-rerun | +7.380 | +5.522 | -0.129 | -0.435 |
| relegated_into_L1 / Entrant | +4.775 | +3.804 | -0.108 | -0.476 |
| relegated_into_L1 / Summer | +0.408 | +1.633 | -0.912 | -1.179 |
| relegated_into_L1 / Weekly | +0.204 | +2.676 | -0.455 | -0.374 |
| relegated_into_L1 / Pooled A2, post-rerun | +7.599 | +5.625 | -0.105 | -0.459 |
| promoted_into_L1 / Control | +2.824 | +1.829 | +0.204 | +1.424 |
| promoted_into_L1 / A2, post-rerun | -4.686 | +0.196 | -0.027 | +1.408 |
| promoted_into_L1 / Entrant | -0.769 | +2.043 | +0.226 | +1.403 |
| promoted_into_L1 / Summer | +3.114 | +1.572 | -1.043 | +0.884 |
| promoted_into_L1 / Weekly | +2.799 | +2.026 | +0.156 | +1.337 |
| promoted_into_L1 / Pooled A2, post-rerun | -4.258 | +0.214 | -0.036 | +1.383 |
| entered_spfl / Control | — | -5.979 | +3.347 | +2.347 |
| entered_spfl / A2, post-rerun | — | -7.910 | +3.360 | +2.440 |
| entered_spfl / Entrant | — | -6.951 | +4.217 | +2.735 |
| entered_spfl / Summer | — | -6.094 | +3.140 | +2.762 |
| entered_spfl / Weekly | — | -6.333 | +2.659 | +2.282 |
| entered_spfl / Pooled A2, post-rerun | — | -8.015 | +3.249 | +2.391 |

Promotion game1 has5 priced club-seasons (later bins6); relegation bins each5. Entered-SPFL game1 has no priced observation. These are descriptive, not causal curves.

## 5. UUIDs, convergence and timing


| Arm | Final scored UUID |
| --- | --- |
| Control | `866a947c-be65-467d-8dfd-c58aed0ad1f9` |
| A2, post-rerun | `2e13b786-7d47-47c0-8e04-1529e705c663` |
| Entrant | `c6118c34-a2a5-46fd-b4d6-21ef7b795b49` |
| Summer | `13b5d0fb-0f6a-43f0-a9bf-9d5b6efb856c` |
| Weekly | `d62bd802-1434-4c01-a1c5-26512b776d5e` |
| Pooled A2, post-rerun | `edf3138e-ec55-4e14-aa35-bb624b16f6c4` |

Pooled initial immutable UUID `0fa6219a-97cc-47c9-95bf-aec0235c89ef` is preserved. Final fit retains56 original folds and replaces41,45,51,53 once using seeds2026101200+fold (replacement divergences0,0,1,2). `reruns_qs_pooled_a2.csv` records diagnostics; `*_qs_pooled_a2_initial.csv` retains initial scores/folds/checks. Initial pooled maxR̂1.0236 and215/240000 divergences passed aggregate review but triggered four per-fold reruns. The final fit passed latent/target coverage and save/load parity checks. Pooled has1605 total latent fixtures, including Championship monitor fixtures; all inferential tables here use only the frozen1070 lower-tier targets.

| Final arm | Folds | Max R̂ | Min bulk/tail ESS | Divergences /draws | Fold review |
| --- | --- | --- | --- | --- | --- |
| Control | 60 | 1.0228 | 277 /414 | 0 /240000 | PASS |
| A2, post-rerun | 60 | 1.0255 | 290 /624 | 66 /240000 | FAIL: 53 |
| Entrant | 60 | 1.0358 | 262 /590 | 1 /240000 | PASS |
| Summer | 60 | 1.0301 | 304 /582 | 0 /240000 | PASS |
| Weekly | 115 | 1.0218 | 292 /542 | 2 /460000 | PASS |
| Pooled A2, post-rerun | 60 | 1.0236 | 292 /205 | 38 /240000 | PASS |

Weekly requires no reruns. Final pooled requires no further reruns. A2 from S3 still has its existing fold53 blocker, unchanged. ESS<400 is informational under James's rule; the stricter fit audit can print FAIL despite passing harness review.

| Work | Measured fit wall | Grid /rerun job |
| --- | --- | --- |
| Weekly115 | 8694.71s /144.91min | 172312-4833 |
| Pooled60 initial | 8783.90s /146.40min | 195701-7327 |
| Pooled four seeded reruns | Approximately10min watcher duration (includes replay/persistence) | 223346-26600 |
| Pooled final replay only | 95.14s; NOT grid wall | 223346-26600 |

Both original grids use beast16, pinned cores/BLAS1, four chains × (500 warmup +1,000
retained), one grid at a time, unchanged recipes. Two initial fit walls total
4.855h; rerun/scoring/export overhead is additional.
James may have been attached to `s3` during reruns, so rerun timing is not claimed as
isolated-core throughput. Manager explicitly required leaving `s3` untouched and using fresh
owned `s3r`; the phase ends with `s3r` stopped and `s3` left for James.


## 6. Reproduction and evidence

- Owned checkout `/root/BF_runs/qs_transition` synced by bundle/fast-forward to pushed52c87cba
  (merged60d9a81b plus brief/D22 records). No push; source/recipes/Project/Manifest diff clean.
- Both lower and pooled-smoke cache hashes are
  `a1e89da1a25aaaa8bc7476307338b9e29a58d5d33beee8e25e25e6126989fe12`.
  Pooled smoke copied this from `/root/BF_runs/a76a65df/.cache/datastore_ScottishPyramid.jls`
  to its owned checkout; identical bytes already existed in the S3 checkout, so no new copy,
  SQL refresh or cache regeneration was needed. Scope selects Championship goals from that cache.
- Readiness172219-11336 confirms approved weekly hash
  `aaed32f47a4c97e032bfb067b6debd2e4ccc14d934df58301955f4bccffae952` and pooled hash
  `0db99f621d0e430127f62404c530f07bdbdb22a459f8a345856193e0e09df887`, both passing smoke.
  Full histories: `s3b_smoke_history_*.csv`; no S3b stale rows removed.
- Absolute runner include and a separate `l01_entrant_prior.jl` preload precede each arm's `main`.
  Grid logs: `/root/BF_runs/logs/qs_transition_s3b_{weekly,pooled}_grid`.
- `grid/r02_score_export.jl`: same S3 scoring, only weekly fold assertion extended to115.
  `grid/r04_bias_bins.jl`: unchanged curves. Final score/curve job224438-5675 completed these
  exports before the first posthoc exporter panel assertion failed on monitored Championship rows.
- `grid/r05_transition_errors.jl` now explicitly restricts raw observations to the control priced
  lower-tier panel before pairing; successful224813-23506 exports1326 arm/cohort/game errors.
  Every one of18 arm/cohort signed means and counts matches frozen scoring within1e-10.
- `python3 grid/absolute_error.py --test`: synthetic unequal-game-count weighting, absolute-before-mean,
  paired resampling, self-reference and one-cluster tests PASS. Normal invocation exports54 rows.
  `summarize_s3.py` builds frozen six-arm tables; `report_s3b.py` renders this report from CSVs.
- No frozen gates/priors/sampler/recipes/src edits and no26/27 outcome scoring. Dirty fit telemetry
  reflects owned research outputs, not changes to model source.

## Decision

Neither S3b arm qualifies under the frozen gates; no arm qualifies under D23's posthoc criterion
on this panel either. Retain the control as the comparison baseline. The manager and James decide
any next-stage action; this report does not authorise a combination, gate relaxation or26/27 run.


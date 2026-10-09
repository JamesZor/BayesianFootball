# Explicit control forwarding and persisted QS resume — PASS

Manager-authorised source **081ba5a6**; src/harness unchanged.

## Fresh offline wrapper regression

Fresh beast REPL %393, -t16, pinned cores, BLAS1; no database/datastore/MCMC.
`t05_control_forwarding.jl`: **57/57 PASS**.

| Gate | Passes |
|---|---:|
| Approved final UUID / experiment ownership / completion / all40 folds | 8/8 |
| Score comparison exact/1e-12, row/identity/count/float mutation negatives | 7/7 |
| All three candidates × primary/rerun explicit forwarding, missing/wrong control and score-mismatch receipt rejection | 42/42 |

Evidence: results/gate_logs/control_regression_v1.log.
Both real primary and prescribed rerun calls in run explicitly pass `control`;
stage guard rejects forgotten/foreign control before grid/scoring.

## Authorised v2 resume

Fresh launcher %394, persisted-QS REPL %395, 20:30:48–20:32:29 UTC.
Standalone GRW stage omitted entirely. QS completed recipe preflight resolves
exactly to **2a5eee90-3402-40ce-a5e3-f4c6b57fc8ea**, emits
`[RESUME_NO_SAMPLING]`, and unchanged Harness.grid loads that existing fit.
No new QS fit or chains; scoring/receipt complete, no Rhat rerun needed.

Explicit control verification checks UUID
**98447840-e95c-420f-92b4-2db6545506a3**, name grw_joint, this experiment,
status completed and all40 folds. No new GRW run or standalone scoring.
Manager separately permits unchanged harness internal paired-control scoring.
Before candidate receipt acceptance, snapshot/read-back comparison checks every
existing GRW score field, matching row identities and counts.

**664 rows / 9,960 fields exactly equal; max absolute difference 0.0.**
No threshold use needed. Evidence: control_scores_before_qs_joint.csv and
control_score_parity_qs_joint.csv in results/phase3. Any future mismatch stops
before receipt acceptance; each candidate and prescribed rerun gets its own
baseline and comparison CSV.

SHA256 verification confirms **all80 GRW/QS initial checkpoint files unchanged**;
results/gate_logs/joint_checkpoints_v2_check.txt. Source seeds, recipes, priors,
budgets, thresholds and wave-1 references unchanged. Initial failure retained
in PHASE3_BLOCKED.md and v1 logs.

GRW tail-ESS review flag **324.4704888 (<400)** remains reportable; no Rhat-based
rerun due. QS Rhat1.00937/bulk832.35/tail761.02, zero divergences.

## Still running

Launcher advances to grw_marketobs in fresh %396 at20:32:29 UTC, then
qs_marketobs. Original fixed budget/order/rerun policy intact. Phase3 not yet
complete; Phase4 report and Phase5 reproduction remain. Do not update the beast
checkout during the grid. Continuation panes/status in PROGRESS.md.

# Wave 2 — Phase 3 complete; scoring/reproduction handover

**Resume at Phase 4, NOT sampling.** Read PROGRESS.md/DECISIONS_LOG.md,
PHASE3.md and original experiments/pi_qs_joint_marketobs_prompt.md §4–6.
TODO041 IN_PROGRESS; branch exp/qs-goal-vs-market. Full package NOT done.

## Completed grid facts

Namespace scottish_lower_qs_wave2_2426,40fold Scottish56/57 panel,710fixtures.
Final UUIDs in results/phase3/RUNS.csv:
- grw_joint 98447840-e95c-420f-92b4-2db6545506a3
- qs_joint 2a5eee90-3402-40ce-a5e3-f4c6b57fc8ea
- grw_marketobs e47a71a7-52eb-4aa5-b747-e64099ea9977
- qs_marketobs 81fdf817-f82d-4f66-9bfd-5f456c4fe29f

All160folds,4chains×(500warmup+1000retained),16/16hard grid passes,
zero divergences, maxRhat1.01721: no Rhat>1.05 folds/no reruns required.
Initial=final UUIDs; rerun-selection ledgers40false rows each. Exact diagnostics
and final read-only ownership/check audit in results/phase3/. KEEP report
flags: GRWjoint tailESS324.47, GRWmarket tail339.34 and bulk395.44 (<400).
No extra tail-ESS sampling authorised.

V1 source8f285a6a sampled both joints; QS saved/parity passed before
missing-control scoring failure. Manager authorises ONLY l05/r05 wrapper
fix/source081ba5a6 plus offline regression57/57PASS. V2 skips standaloneGRW,
resolves exact persistedQS2a5eee90 and finishes scoring/receipt without
sampling. All80joint checkpoints remain SHA256-identical after market grids.
No new joint fits/runs, no src/harness edits or scientific-setting changes.
PHASE3_BLOCKED.md remains historical evidence, NOT current blocked state.

Manager explicitly allows unchanged Harness.grid internal paired-control
recomputation, requiring equality with existingGRW score rows. Each candidate
comparison664rows/9960fields EXACT,maxdiff0; before-snapshots/comparisonCSV
committed. Experiment preflight verifies GRW UUID/ownership/status/all40folds,
explicit control forwarded to both primary/rerun calls. All false/mismatch
negative regressions pass; no parity bypass.

## Execution/artifacts

NO owned beast panes open. %393–%398 and all earlier owned panes closed;
other sessions untouched. All Julia on root@mcmc-beast,-t16,pincores,BLAS1.
Beast checkout /root/BF_runs/qs_experiment still081ba5a6 (later docs commits
local/pushed, not checked out during running grid). Before next stage fetch
and detach to the approved committed source; never touch reference outputs.
Logs /root/BF_runs/logs/qs_experiment_w2/; outputs
/root/BF_runs/qs_experiment_w2_out/phase3/. Initial160fold checkpoints in
checkout data/checkpoints/scottish_lower_qs_wave2_2426/<arm>/.
Frozen digests/mtime in PROGRESS.md. Full ANSI-cleaned gate logs committed.

## Phase 4 still to implement/run

Use exact wave1 saved reference rows/UUIDs, DON'T refit them. Wave1 source
experiments/scotland/05_quality_style_goal_vs_market/: r05_score.jl,
r05_block_check.jl, scoring_adapter.jl, r05_goal_cluster.jl, r05_figures.jl.
I read r05_score.jl lines1–140 and r05_goal_cluster.jl entirely; no Wave2
scoring implementation or predictive comparison done yet.

Required:1X2 LogLoss/RPS/Brier/ECE;OU2.5/BTTSLogLoss;total/allocation goal
scores;compression/model-on-market slopes;transitionbias;both circular and
noncircular8week block bootstraps AND harness clustered intervals. Classify
on prescribed noncircular interval, flag circular contradictions. PrimaryA
QSjoint−GRWjoint;primaryB marketarms−savedC0 and−GRWjoint; all secondary pairs
from package. Report posterior r0/rs/rk by fold for bothQSarms,sigma_obs for
marketarms versusC0; r figure, <=12line summary, CSVtraceability, leaderboard.
Commit score/reference parity evidence and README/REPORT/TODO updates.

Phase 5 then fresh REPL onefold per newarm from frozen seeds, byte comparison,
REPRODUCIBILITY.md. Queue chainseed202700000+100000*arm+100*fold+10*attempt+chain
(arms1–4,folds1–40,attempt0/1,chains1–4); postseed202700000+100000*arm+9000+attempt.
Use attempt0 for these saved fits; never resample reference arms.

All prior component/test/smoke gates passed, historical failures preserved.
No ROI/staking, tuning, threshold relaxation or favourable subset. Definition
ofdone includes Phase4/5; never mark TODOdone before report/reproduction evidence.

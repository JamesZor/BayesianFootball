# Phase 3 — four frozen 40-fold grids complete

Final arm completes **2026-10-09 22:05:57 UTC**. Initial source8f285a6a;
manager-authorised wrapper correction/resume source081ba5a6. No src/harness
changes, joint refits, new joint runs or scientific-setting changes.

## Persisted runs and convergence

| Arm | Final UUID | Rhat max | Bulk ESS min | Tail ESS min | Tail<400 folds | Divergences |
|---|---|---:|---:|---:|---:|---:|
| grw_joint | 98447840-e95c-420f-92b4-2db6545506a3 | 1.01008 | 641.78 | **324.47** | **1** | 0 |
| qs_joint | 2a5eee90-3402-40ce-a5e3-f4c6b57fc8ea | 1.00937 | 832.35 | 761.02 | 0 | 0 |
| grw_marketobs | e47a71a7-52eb-4aa5-b747-e64099ea9977 | 1.01721 | 395.44 | **339.34** | **1** | 0 |
| qs_marketobs | 81fdf817-f82d-4f66-9bfd-5f456c4fe29f | 1.01635 | 434.09 | 781.30 | 0 | 0 |

Every arm contains40 folds ×4 chains ×1000 retained draws, with500 warmup
per approved sampler. All160 initial fold Rhat values <=1.05: **no prescribed
reruns required or performed**. Initial UUIDs equal final UUIDs; all selection
ledgers contain40 false rows. Zero divergences passes the <=0.1% gate.

**Do not call these universally convergence-clean:** GRW joint retains its
manager-highlighted tail ESS324 (<400) review flag; GRW market tail339 (<400)
and bulk395 (<400) also require honest reporting. Tail ESS is not a trigger
for extra sampling under the approved Rhat-only rerun policy.

Exact values: results/phase3/final_grid_audit.csv, diagnostics_<arm>.csv,
grid_summary_<arm>.csv and rerun_selection_<arm>.csv. Immutable receipts
run_<arm>.txt identify saved fits. final_db_runs.csv records persisted source
SHAs and ownership/status; final_grid_checks.csv contains32 check rows,
including **16/16 hard passes** (filtration, latents, target coverage, DB parity).
Fresh read-only audit in %398 confirms exactly one completed run per grid name:
no new/duplicate joint run. Script/log in results/gate_logs/phase3_final_audit.*.

## Failure retained, then authorised no-sampling resume

V1 GRW completes; QS persists/parity passes then fails scoring because its
control was not supplied. PHASE3_BLOCKED.md and original v1 logs retained.
Manager authorises experiment-only l05/r05 control RunRef forwarding for every
non-control primary/rerun, with full ownership/completion preflight. Fresh
offline regression57/57 passes; CONTROL_FORWARDING_REVALIDATION.md.

V2 skips standalone GRW. QS preflight requires exact persisted UUID2a5eee90,
and completes scoring/receipt without sampling. All80 initial joint checkpoint
SHA256 hashes remain unchanged even after both market grids. Market arms then
run in the original order in fresh REPLs, at the same fixed budget/priors/seeds.

Include wall seconds (not sampling-only timings): GRW joint2008.6993;
QS original1962.5365 plus scoring resume92.2180;
GRW market2758.9134; QS market2832.6007.

## Mandatory internal control-score consistency

Manager permits unchanged Harness.grid internal paired-control scoring, but
requires every recomputed GRW row equal its existing row exactly/absolute1e-12
before candidate receipt acceptance. For **each of all three candidates**:

- **664 rows /9960 fields checked; exact=true; max absolute difference0.0.**
- Baseline snapshots control_scores_before_<arm>.csv and comparisons
  control_score_parity_<arm>.csv committed under results/phase3/.
- No standalone GRW stage, refit/resampling or new GRW run. No threshold use
  needed for equality; offline mismatch regression proves receipt rejection.

## Definition of done and next phase

Phase3 acceptance met: frozen ordered grids, atomic checkpoints/UUID receipts,
complete diagnostics and prescribed rerun selection, all hard checks, exact
control score parity, no joint resampling, preserved tail review flags.
All owned panes %393–%398 closed; other sessions untouched.

**Phase4 scoring/report/leaderboard and Phase5 fresh frozen-seed byte-for-byte
reproduction remain.** No predictive decision or task completion claimed.
Reuse saved wave1 scores/references, all prescribed metrics/pairs/bootstrap
paths and noncircular classification; no reference refits or ROI/staking.

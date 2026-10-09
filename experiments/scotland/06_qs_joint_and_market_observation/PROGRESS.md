# Wave 2 — progress

Updated: 2026-10-09 after Phase4 gates/report/boards acceptance. Owner: pi. Manager: Claude.
Branch: `exp/qs-goal-vs-market`. TODO: 041.

## Current phase

**Phase 4 COMPLETE; Phase 5 frozen-seed reproduction next.** Canonical board/read-only score audit PASS:4×664rows/9960fields EXACT,maxdiff0; only4new grid rows/cohorts and one register row added. Fresh v2 scorer and interval audit PASS. All56 market/30 goal pairs, three bootstrap paths, exact saved reference parity,236 r rows/120 sigma rows, native SVG and CSV-traceable REPORT. All primary noncircular1X2 intervals cross zero; circular contradicts both marketarms-vs-GRWjoint. No promotion. Wave1 current-scorer provenance follow-up recorded read-only; wave1 source/outputs untouched. All four40-fold arms
persisted, four hard grid checks per arm PASS, zero divergences, all Rhat<=1.05:
no prescribed reruns needed. Source8f285a6a initial joints; wrapper fix/resume
081ba5a6. Offline regression57/57 PASS, QS completes scoring without sampling,
no standalone/new GRW run. All80 joint checkpoint hashes unchanged.
All three candidate internal-control score comparisons664rows/9960fields
EXACT, maxdiff0. Fresh read-only final audit confirms4runs/160folds/640chains,
1000retained per chain,16hard passes. No owned panes remain.

Final UUIDs (namespace scottish_lower_qs_wave2_2426):
- grw_joint: 98447840-e95c-420f-92b4-2db6545506a3
- qs_joint: 2a5eee90-3402-40ce-a5e3-f4c6b57fc8ea
- grw_marketobs: e47a71a7-52eb-4aa5-b747-e64099ea9977
- qs_marketobs: 81fdf817-f82d-4f66-9bfd-5f456c4fe29f

KEEP tail-ESS400 review flags: GRW joint324.47, GRW market339.34 (bulk395.44).
No extra sampling is authorised for tail ESS. PHASE3.md / results/phase3
has complete CSV evidence, checks, UUIDs and rerun-selection ledgers.
Historical missing-control stop retained in PHASE3_BLOCKED.md/v1 logs;
CONTROL_FORWARDING_REVALIDATION.md documents authorised correction/parity.

## Done

- Phase 0 PASS at ec96c079, evidence 2f0c01e9. Fresh QS 373/373,
  MultiScaleGRW 124/124, tapes 150+2 at 0 B, builder 106/106,
  harness 179/179, market t01–t06 with authorised C2-pending exclusion
  (29/39). Proxy-xG coverage every training fold/tournament 56/57:
  180–530 matches, 50–100%; held-out 710/710. PHASE0.md.
- Phase 1 table preparation PASS at 4a71c02e: exact wave-1 value/CSV parity,
  1430 rows / 1107 full books; target counts 710/595/78/517. Manager-authorised
  Main.QSMarketArm namespace, original provenance guard/manifest untouched.
- Amended component PASS at 153239a7: 3787/3787; exact AD <=4.2e-16,
  selected-coordinate FD convergence/Richardson gates; all three recovery
  seeds per arm 39/42 coverage, zero all-seed misses. PHASE1.md.
- Array-backed feature / canonical observation identity at b006743f fully
  revalidated: component 3831/3831, tape entry point 160/160 at 0 B,
  AD <=4.11e-16, all 42 recovery intervals exactly reproduced. Evidence
  d0d933ae / ARRAY_FEATURE_REVALIDATION.md. No src/harness edits; comparator
  AbstractDict gap recorded as out-of-package TODO follow-up.
- Phase 2 all four six-hard-check smokes PASS. Joint source ac30c4d7 retained;
  market-only d0d933ae includes real DB parity. Fresh read-only eligibility
  audit confirms all four recipes. Evidence a7b332a6 / PHASE2.md.
  All short-smoke convergence/performance flags retained: GRW market
  Rhat 1.13787 / bulk ESS 16.10 / tail 57.47, QS market
  1.06250 / 67.14 / 74.33, zero divergences. Prior GRW 1.14/bulk16/4.219 h
  warning disclosed alongside full four-arm 12.3192 h ETA before approval.
- Manager explicitly approves fixed order/budget. l05/r05 durable seeded
  queue at 8f285a6a; mock validation complete 19:06:40 UTC. Grid started
  19:07:51 UTC. Atomic per-fold checkpoints, UUID receipts, one prescribed
  attempt-1 rerun for every initial Rhat>1.05 fold, always select the rerun.

Earlier failures remain in PHASE1_BLOCKED.md, PHASE2_BLOCKED.md and
PHASE2_PARITY_BLOCKED.md; do not remove/relabel failed evidence.

## Next action

Finish r08_record read-only Phase3/Phase4 score equality audit and canonical leaderboard/register materialisation. Commit/push Phase4 boundary, close%401, then Phase5 via phase5_gate.sh: fresh REPL per newarm, fold1 ONLY, unchanged attempt0 chain/post seeds and sampler, separate reproduction checkpoints. r07_reproduce adapts wave1 r06_reproduce comparisons to frozen wave2queue. Compare raw Float64 bytes/SHA256 for chains/latents; stop first mismatch, no retry. All160 original checkpoint digests frozen before reproduction. No reference resampling.
Do not refit references or tune priors/data/budgets/seeds/thresholds. No ROI/staking.
Retain both GRW tail flags in the report. Task remains IN_PROGRESS.

## Beast panes and artifacts

NO owned panes open at Phase4 boundary. %401 closed after score/interval/record PASS. Next phase5_gate launcher creates ONLY fresh reproduction REPLs. Failed %400 closed; failed scoring log/partial outputs retained in phase4_v1. No sampling/DB writes. %399 startup used unavailable shell PATH Julia and exited before execution. Successful market REPLs%396/%397 closed automatically;
launcher%394 closed after PASS. Fresh read-only final audit%398 PASS, then
closed after evidence retrieval. Offline regression%393/resumedQS%395 and
all earlier owned panes closed. Other sessions untouched. All160 initial
fold checkpoints/both original joint fits preserved; joint hashes unchanged.

Checkout: `/root/BF_runs/qs_experiment` at 081ba5a6.
Checkpoints: `data/checkpoints/scottish_lower_qs_wave2_2426/<arm>/` in checkout.
Logs: `/root/BF_runs/logs/qs_experiment_w2/phase3_launcher_v2.log`
and `phase3_grid_v2_<arm>.log`; historical v1 logs retained.
Launcher `/tmp/qsx2_phase3_resume_v2.sh` (committed evidence copy in results/gate_logs)
skips standalone GRW; tracked original phase3_gate.sh unchanged.
Outputs: `/root/BF_runs/qs_experiment_w2_out/`.
Local evidence: experiment `logs/` and `results/`.
Frozen market_rates.csv SHA256:
`680bc2283e9c0fc7a896bf01b4ba209cb3aae06f50d00667c98b02f379872549`.
Original saved inversion manifest SHA256:
`57941606a195010fe86cbfda0564d61f456379d05d0e8988dce3bb2edd600fb1`.
Pinned snapshot SHA256:
`c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4`,
mtime `2026-09-25 12:57:15.480765468 UTC` unchanged.

## Definition of done

All five phase boundaries evidenced and committed/pushed; exact work-package
tests, four smoke hard passes plus manager grid approval, ordered grids,
complete CSV-traceable paired report and exact saved reference parity,
fresh frozen-seed reproduction, todo/diff checks. Stop on failed gates;
no loosened thresholds. Phase3 complete; Phase4/5 are NOT complete.

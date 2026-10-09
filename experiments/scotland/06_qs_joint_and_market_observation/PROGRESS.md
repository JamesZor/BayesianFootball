# Wave 2 — progress

Updated: 2026-10-09 after 20:14 UTC stop (beast). Owner: pi. Manager: Claude.
Branch: `exp/qs-goal-vs-market`. TODO: 041.

## Current phase

**Phase 3 BLOCKED at qs_joint harness scoring, source 8f285a6a.**
GRW control completed as `grw_joint` UUID 98447840-e95c-420f-92b4-2db6545506a3:
40 folds, Rhat max 1.01008, bulk min 641.78, tail min 324.47 (one review flag),
zero divergences, no Rhat reruns. QS 40 folds persisted/parity hard PASS,
UUID 2a5eee90-3402-40ce-a5e3-f4c6b57fc8ea, Rhat 1.00937, bulk/tail
832.35/761.02, zero divergences/reruns due. Scoring stops because the wrapper
never supplies the GRW control to non-control arms: expected exactly one
role=:control; found 0. No market arm started or retry. PHASE3_BLOCKED.md.
All owned panes closed after read-only audit. Mock-chain queue validation PASS;
full smoke-probe ETA 12.3192 h was disclosed before approval.

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

Await manager review/authorisation of experiment-only l05/r05 control-reference
forwarding; do not retry or refit persisted GRW/QS fits. PHASE3_BLOCKED.md has
exact diagnosis, UUIDs, CSV evidence and proposed resume. When authorised,
use a fresh launcher attempt label; immutable evidence/receipts/checkpoints
avoid duplicate sampling. Approved order:
`grw_joint → qs_joint → grw_marketobs → qs_marketobs`, each 4×(500+1000).
Preserve initial/final UUIDs, diagnostics, rerun-selection CSVs and manifests.
Final divergence <=0.1%, finite Rhat; report residual Rhat flags and tail ESS
against 400. Never third attempt or choose a favourable initial fit.
After all four arms: commit Phase 3 evidence, then Phase 4 prescribed scoring
and Phase 5 frozen-seed reproduction. No reference refits, ROI/staking or
prior/data/budget/seed/threshold changes.

## Beast panes and artifacts

NO owned panes open. Launcher %390 stopped on failure; successful GRW %391
closed automatically; QS %392 retained for read-only UUID/diagnostic audit,
then closed along with launcher after evidence retrieval. All earlier owned
panes %342–%389 closed; other sessions untouched. Both fits and all 80 initial
checkpoints preserved. No source update or retry on beast.

Checkout: `/root/BF_runs/qs_experiment` at 8f285a6a.
Checkpoints: `data/checkpoints/scottish_lower_qs_wave2_2426/<arm>/` in checkout.
Logs: `/root/BF_runs/logs/qs_experiment_w2/phase3_launcher_v1.log`
and `phase3_grid_v1_<arm>.log`.
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
no loosened thresholds. Phase 3 and later are not complete.

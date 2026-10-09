# Wave 2 — progress

Updated: 2026-10-09 UTC. Owner: pi. Manager: Claude.
Branch: `exp/qs-goal-vs-market`. TODO: 041.

## Current phase

**BLOCKED Phase 2 at ac30c4d7: grw_marketobs fit_parity fails. Both joint smokes hard PASS; qs_marketobs not started. Phase 1 remains PASS at 153239a7.** Historical recipe-loading failure at 73147ce0 retained in PHASE2_BLOCKED.md. Previous component failure at ec1a4137 (5533 pass / 372 fail) retained. Table preparation remains PASS at 4a71c02e. Manager-authorised Main.QSMarketArm namespace, original guard/manifest unchanged. Exact table-value/count parity passed, including frozen CSV round-trip. Earlier failure at 351b8ddd retained in PHASE1_BLOCKED.md. Amended component 3787/3787; tapes 150/150 + QS 2/2 + MRO 8/8 (0 B); builder 106/106; harness 179/179. Prior failed attempt retained; PHASE1.md is current acceptance evidence. Phase 0 remains PASS at ec96c079, evidence pushed at 2f0c01e9. V2 joint smoke UUIDs and failed market-observation fit UUID are recorded in PHASE2_PARITY_BLOCKED.md. Earlier loader failure had no sampling. No grid or predictive scoring.

## Done

- Local branch clean at 9a86f7b1; task created/claimed.
- Beast checkout `/root/BF_runs/qs_experiment` clean at d16a5800, dependency/environment symlinks intact.
- Pinned snapshot digest and original mtime verified unchanged.
- Read wave-1 summary, panel/fold code and Phase 0 launcher; manager C2 exclusion preserved.
- Setup committed/pushed at ec96c079; isolated beast checkout updated to that exact source; tests launched.
- Fresh QS 373/373 (59.98 s include wall); MultiScaleGRW 124/124 (44.78 s); tape 150/150 plus QS 2/2, 0 B assertions pass (339.35 s); builder 106/106 (71.40 s); harness 179/179 (111.53 s).
- Market t01–t06 pass; pooled full-book 305/305, C2-pending 29/39 explicitly excluded. Exact summaries committed under logs/.
- Exact joint-feature coverage: both 56 and 57 have 180–530 covered training matches per fold (50%–100%); all 710 held-out matches covered. Per-fold CSV recorded.
- Phase 0 complete 15:32:29 UTC; all owned sessions closed.
- MarketRateObservation, injected feature, exports/seams, tests and table adapter prepared at 27c53ede; fresh-REPL launcher at 351b8ddd. No acceptance claimed.
- Phase 1 preparation failed 15:47:15 UTC (50.78 s): saved wave-1 manifest encodes Main.QSMarketArm, new loader nests it under Main.Wave2MarketTable. The exact repr provenance guard stopped before freezing a table. Evidence preserved; owned idle sessions closed.
- Manager authorised original namespace; preparation-v2 PASS 15:58:54 UTC, 36.2634 s. Frozen table 1,430 rows / 1,107 full books, target counts exactly 710/595/78/517, exact values match wave-1 panel after CSV round-trip.
- Frozen market_rates.csv SHA256: 680bc2283e9c0fc7a896bf01b4ba209cb3aae06f50d00667c98b02f379872549. Guard/manifest unchanged; manifest digest identical before/after.
- Component entry point failed 16:07:56 UTC (183.3400 s include): likelihood 22/23, GRW AD 2867/2932, QS AD 2627/2932, recovery 15/16 assertions. All AD failures are central finite differences; compiled/fresh ReverseDiff and ForwardDiff comparisons pass. 13/14 recovery intervals cover; GRW defence micro scale misses. Details/evidence in PHASE1_BLOCKED.md and results/synthetic_recovery.csv. No new gate executed afterward; table and manifest digests unchanged.

- Amended Phase 1 complete 16:47:05 UTC at 153239a7. Exact AD errors <=4.2e-16 at 1e-10 gate, historical FD counts 65/305 reproduce; all selected-coordinate convergence/Richardson checks pass. Recovery 39/42 (92.86%), no parameter misses all 3 seeds. All intervals and derivative evidence frozen in results/; see PHASE1.md.

## Next action

Await manager parity/eligibility review. Read-only diagnosis at 58665779: original and database-restored rate dictionaries have identical 1430 key/value entries, but _structural_equal compares Dict internals and returns false. Serialization-only config clone reproduces; other fields unchanged. No fix/retry/threshold/input change. See PHASE2_PARITY_BLOCKED.md; historical failed row also blocks future smoke eligibility under current policy. Ask before grid. Phase 1 evidence pushed at e86a03fe; no Phase 1 rerun needed. No predictive scores/refits of wave-1 references. Priors/truths/seeds unchanged; manager amendments in DECISIONS_LOG.md.

## Beast panes and artifacts

None open. V2 %375–%378 and read-only audit %379 closed after evidence retrieval. Both joint hard smokes pass (6/6 each); marketobs 5 pass/1 parity failure, other market arm not started. Convergence/performance review flags retained; no grid. Failed grw_joint REPL %374 and launcher %373 closed after evidence retrieval. Initial shell-only launcher %372 also closed; no Julia ran there. All earlier owned panes closed; other sessions untouched. Diagnosis `%365`/`%366` also closed. Failed component REPL `%364` and launcher `%363` closed after evidence retrieval. Preparation-v2 `%361`/`%362` also closed. Original saved manifest SHA256 57941606a195010fe86cbfda0564d61f456379d05d0e8988dce3bb2edd600fb1, unchanged before/after. Never touch other sessions.
Logs: `/root/BF_runs/logs/qs_experiment_w2/`.
Outputs: `/root/BF_runs/qs_experiment_w2_out/`.
Local evidence: `experiments/scotland/06_qs_joint_and_market_observation/logs/` and `results/`.

## Definition of done

All five phase boundaries evidenced and committed/pushed; exact work-package tests, four smoke hard passes plus manager grid approval, ordered grids, complete CSV-traceable paired report and saved reference parity, fresh frozen-seed reproduction, todo/diff checks. Stop on a failed gate; do not loosen thresholds.

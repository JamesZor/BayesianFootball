# Wave 2 — progress

Updated: 2026-10-09 UTC. Owner: pi. Manager: Claude.
Branch: `exp/qs-goal-vs-market`. TODO: 041.

## Current phase

**Phase 1 resumed: manager authorised option 1 (original Main.QSMarketArm namespace).** Original guard/manifest unchanged; exact table-value/count parity added, including frozen CSV round-trip. Earlier failure at 351b8ddd retained in PHASE1_BLOCKED.md. Component source remains unverified; new AD/allocation/recovery tests have not run. Phase 0 remains PASS at ec96c079, evidence pushed at 2f0c01e9. No new fit, smoke, grid or scoring has started.

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

## Next action

Commit/push authorised namespace correction; run fresh table preparation (prepare_v2). Preserve original guard and saved manifest; record exact parity and SHA256. Commit frozen table/hash before fresh unit/AD/recovery and tape/builder/harness gates. Priors/truths/seeds remain frozen in DECISIONS_LOG.md. Phase 2+ not started.

## Beast panes and artifacts

None open. Preparation launcher `%359` and failed idle REPL `%360` closed after evidence retrieval; Phase 0 REPLs `%343`–`%358` and launcher `%342` also closed. IDs retained in committed log summaries. Never touch other sessions.
Logs: `/root/BF_runs/logs/qs_experiment_w2/`.
Outputs: `/root/BF_runs/qs_experiment_w2_out/`.
Local evidence: `experiments/scotland/06_qs_joint_and_market_observation/logs/` and `results/`.

## Definition of done

All five phase boundaries evidenced and committed/pushed; exact work-package tests, four smoke hard passes plus manager grid approval, ordered grids, complete CSV-traceable paired report and saved reference parity, fresh frozen-seed reproduction, todo/diff checks. Stop on a failed gate; do not loosen thresholds.

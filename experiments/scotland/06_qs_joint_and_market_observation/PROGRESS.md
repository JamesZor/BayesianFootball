# Wave 2 — progress

Updated: 2026-10-09 UTC. Owner: pi. Manager: Claude.
Branch: `exp/qs-goal-vs-market`. TODO: 041.

## Current phase

**Phase 0 PASS at ec96c079; Phase 1 next.** All prescribed fresh-REPL gates pass under the manager's C2-pending exclusion. See `PHASE0.md` and `results/phase0_gates.csv`. No component, smoke, grid or scoring has started.

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

## Next action

Commit/push Phase 0 evidence; implement MarketRateObservation and injected feature, tests, allocation matrix and synthetic recovery. Prior derivation: C0 FullBookRung uses HalfNormal(0.20) for sigma_obs (l05_fullbook_engine.jl:30). Record frozen choice before any fit. Stop/report any failed gate.

## Beast panes and artifacts

None open. Fresh gate REPLs `%343`–`%358` and launcher `%342` closed after all evidence was fetched. All pane IDs retained in `logs/phase0_launcher.summary.txt`. Never touch other sessions.
Logs: `/root/BF_runs/logs/qs_experiment_w2/`.
Outputs: `/root/BF_runs/qs_experiment_w2_out/`.
Local evidence: `experiments/scotland/06_qs_joint_and_market_observation/logs/` and `results/`.

## Definition of done

All five phase boundaries evidenced and committed/pushed; exact work-package tests, four smoke hard passes plus manager grid approval, ordered grids, complete CSV-traceable paired report and saved reference parity, fresh frozen-seed reproduction, todo/diff checks. Stop on a failed gate; do not loosen thresholds.

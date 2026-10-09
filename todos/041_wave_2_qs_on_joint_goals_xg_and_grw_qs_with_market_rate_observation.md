# 041 — Wave 2: QS on joint goals+xG and GRW/QS with market-rate observation

| Field | Value |
|---|---|
| ID | 041 |
| Title | Wave 2: QS on joint goals+xG and GRW/QS with market-rate observation |
| Status | IN_PROGRESS |
| Priority | P1 |
| Assignee | pi |
| Created | 2026-10-09 |
| Updated | 2026-10-09 |
| Related Files / Commits / PRs | experiments/pi_qs_joint_marketobs_prompt.md; experiments/scotland/06_qs_joint_and_market_observation/; TODO 040 |

## Context & Problem Statement

Compare QS versus MultiScaleGRW on joint goals + proxy-xG, and both dynamics observing inverted market rates versus saved wave-1 C0. Same pinned snapshot, 40-fold lower panel and scoring; no ROI/staking. Claude manages; human scope fixed 2026-10-09. Work package is authoritative.

## Acceptance Criteria

- [x] Phase 0 fresh beast branch-health tests and per-fold proxy-xG coverage recorded.
- [ ] MarketRateObservation wiring, masked likelihood, real-fold AD, 0 B tapes and synthetic recovery evidenced.
- [ ] Four W2 candidates pass smoke hard gates; manager approval before grid.
- [ ] Ordered 40-fold grids with checkpoints, prescribed convergence reruns and diagnostics.
- [ ] Report/CSV/leaderboard with all prescribed pairs, bootstraps and posterior parameters; saved wave-1 references unchanged.
- [ ] Fresh-REPL frozen-seed fold reproduction for each arm, byte comparison recorded.
- [ ] Phase-boundary commits/pushes, progress/decisions current; todo check and diff check pass.

## Ideas & Candidate Solutions

- Reuse wave-1 panel, splitter and scorer without refitting references. Audit proxy-xG through MatchProxyXGFeature with k=25, fallback=:none and fold-specific history fit IDs.
- Inject a stream-independent rates feature; src has no prototype dependency. Use binary-masked Normal log-rate observations and existing Poisson prediction seam.
- Derive sigma prior from C0 before implementation; freeze priors before scoring. No threshold relaxation or score-driven tuning.

## Work Log & Progress

- [2026-10-09 @pi] Claimed in session 01a12134-5a93-7644-aa40-b5294e798206, worktree /home/james/bet_project/.worktrees/BayesianFootball-qs-experiment, branch exp/qs-goal-vs-market. Phase 0 started; fresh beast tests and coverage next. Read-only betdb; Julia exclusively beast.

- [2026-10-09 @pi] Phase 0 PASS at ec96c079: all fresh beast test entry points pass with the authorised C2-pending exclusion (29/39). Exact joint-feature coverage in every training window of both 56/57; 180–530 covered per tournament/fold, 50%–100%; heldout 710/710. See PHASE0.md, results/phase0_gates.csv and results/proxy_xg_coverage.csv. All owned panes closed; Phase 1 next.

- [2026-10-09 @pi] BLOCKED Phase 1 at 351b8ddd: table preparation fails unchanged wave-1 inversion provenance guard (50.78 s fresh-REPL include). Saved manifest is qualified Main.QSMarketArm; new adapter nests it under Main.Wave2MarketTable. No guard bypass, retry, frozen table or new fits/tests. Component implementation remains unverified. Await Claude authorisation of namespace reuse or separate unchanged panel preparation; see PHASE1_BLOCKED.md. Owned panes closed; Phase 2+ not started.

- [2026-10-09 @pi] Manager authorises option 1: include wave-1 market_arm.jl as Main.QSMarketArm; table adapter calls original namespace. Resume Phase 1 without modifying guard/manifest or priors/seeds; add exact IDs/log rates/full-book parity and 710/595/78/517 counts, record frozen SHA256 before tests.

## Verification & Findings

- Initial branch clean at 9a86f7b1. Beast detached checkout clean at d16a5800; .env and Manifest symlinks intact.
- Beast snapshot SHA256 c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4; original mtime 2026-09-25 12:57:15.480765468 UTC intact.
- Phase 1 preparation: FAIL `inversion provenance differs`; exact source/evidence and proposed next action in PHASE1_BLOCKED.md. No component gate evidence yet.
- Phase 0: QS 373/373, GRW 124/124, tape 150/150 + QS 2/2 (0 B), builder 106/106, harness 179/179, market t01–t06 pass (full-book 305/305). Exact counts/wall times and summaries committed in experiments/scotland/06_qs_joint_and_market_observation/; C2 known failures not hidden.

# 041 — Wave 2: QS on joint goals+xG and GRW/QS with market-rate observation

| Field | Value |
|---|---|
| ID | 041 |
| Title | Wave 2: QS on joint goals+xG and GRW/QS with market-rate observation |
| Status | BLOCKED |
| Priority | P1 |
| Assignee | pi |
| Created | 2026-10-09 |
| Updated | 2026-10-09 |
| Related Files / Commits / PRs | experiments/pi_qs_joint_marketobs_prompt.md; experiments/scotland/06_qs_joint_and_market_observation/; TODO 040 |

## Context & Problem Statement

Compare QS versus MultiScaleGRW on joint goals + proxy-xG, and both dynamics observing inverted market rates versus saved wave-1 C0. Same pinned snapshot, 40-fold lower panel and scoring; no ROI/staking. Claude manages; human scope fixed 2026-10-09. Work package is authoritative.

## Acceptance Criteria

- [x] Phase 0 fresh beast branch-health tests and per-fold proxy-xG coverage recorded.
- [x] MarketRateObservation wiring, masked likelihood, real-fold AD, 0 B tapes and synthetic recovery revalidated for array feature under manager amendments.
- [x] Four W2 candidates pass smoke hard gates; convergence and full ETA reported.
- [ ] Manager approval before grid.
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

- [2026-10-09 @pi] Authorised table preparation PASS at 4a71c02e: exact IDs/log rates/flags match wave-1 prepared panel after CSV round-trip; 1430 rows, 1107 full books, target counts 710/595/78/517. Frozen SHA256 680bc2283e9c0fc7a896bf01b4ba209cb3aae06f50d00667c98b02f379872549 recorded in PROGRESS/DECISIONS_LOG; saved manifest digest unchanged. Fresh component gates next.

- [2026-10-09 @pi] BLOCKED at ec1a4137: fresh component gate 5533 pass / 372 fail (1 standalone likelihood, 370 finite-difference, 1 recovery miss). All 14 frozen intervals preserved; GRW dyn.β.σₖ truth .024 above 90% upper .0227293. No retry/tuning; launcher stopped before allocation/builder/harness, no smoke/grid. Table/manifest digests unchanged; owned panes closed. Await manager review of PHASE1_BLOCKED.md.

- [2026-10-09 @pi] Manager amendments authorise resumption: decompose/diagnose likelihood before fixing; exact AD relative 1e-10 plus selected-coordinate FD convergence/Richardson 1e-8; retain first recovery seed and run two further fixed seeds per arm, no parameter misses all three and pooled coverage >=80%. Record all three and additional seeds before fitting. Diagnosis first; no production/test fix yet.

- [2026-10-09 @pi] Likelihood decomposition diagnosis at bd924940: _observe returns but does not accumulate ll; production parent does. Production ll=2.5623083423195876 exactly equals hand Normal constants-inclusive ll, prior=1.2036465597893726. Linked joint differs by log(.12)=-2.120263536200091. Fix test seam only, no production change; amended tests prepared, new seed runs not started.

- [2026-10-09 @pi] Phase 1 PASS at 153239a7 under manager amendments: component 3787/3787; tapes 150/150 + QS 2/2 + MRO 8/8 (0 B); builder 106/106; harness 179/179. Likelihood test seam corrected after decomposition, no src change; AD <=4.2e-16 at 1e-10 gate; all selected FD/Richardson checks pass; recovery 39/42 (92.86%), no parameter misses all three seeds. Exact evidence in PHASE1.md/results. Table/manifest digests unchanged, owned panes closed. Phase 2 next; approval before grid.

- [2026-10-09 @pi] BLOCKED Phase 2 at 73147ce0: candidates.jl:5 unqualified include is undefined in the bare Module(:Wave2Candidates). First entry point fails before candidate build/checks/sampling/DB writes; other arms not started, no UUID/ETA. Proposed Base.include(@__MODULE__, ...) one-line fix; awaiting manager authorisation, no retry. Phase 1 remains accepted. PHASE2_BLOCKED.md/evidence recorded; owned idle panes closed.

- [2026-10-09 @pi] Manager authorised the one-line candidate loader fix and fresh smoke retry via confirmation. Apply Base.include(@__MODULE__, ...); no priors/seeds/inputs/threshold changes. Failed attempt retained, v2 evidence names; grid still separately gated.

- [2026-10-09 @pi] BLOCKED Phase 2 at ac30c4d7: both joint arms six hard passes; grw_marketobs five pass/fit_parity fails; qs_marketobs not started. Read-only audit 58665779 proves 1430 original/reloaded dictionary key/value tuples identical but structural comparator sees Dict internals; no gate fix or sampling retry. Historical failed row also blocks eligibility under current policy. PHASE2_PARITY_BLOCKED.md reports UUIDs, diagnostics, ETA and proposed manager review; all owned panes closed. Phase 1 remains PASS; no grid.

- [2026-10-09 @pi] Manager authorises in-component parity fix only: MarketRatesFeature sorted plain ID/rate/Bool vectors, binary search extraction; no src/harness edits. Revalidate full component and 0 B tape entry points in fresh beast REPLs, then ONLY market smokes; joint passes stand. Preserve prior GRW market smoke Rhat 1.14 / bulk ESS 16 and 4.2 h ETA for grid approval discussion, with full four-arm ETA. All frozen inputs/priors/budgets/seeds unchanged; new outputs preserve old evidence.
- [2026-10-09 @pi] FOLLOW-UP (out of package): harness _structural_equal has no AbstractDict method, so future Dict-carrying configurations can fail persistence parity despite identical active contents. Record for a separate harness task; do not fix or bypass here. Historical-failure eligibility is also unchanged; component-local canonical display explicitly authorised in a separate confirmation gives the vector recipe a content/prior identity; arrays alone do not change the generic model display. It is not a relabelled failed attempt.

- [2026-10-09 @pi] Array feature at b006743f fully revalidated in fresh beast REPLs %381/%382: component 3831/3831, tape entry point 150+2+8/160 (0 B). Exact AD <=4.11e-16, 735 Richardson checks pass unchanged 1e-8 gate. ALL 3 declared recovery seeds per arm rerun: 39/42 coverage, zero all-seed misses; all 42 interval values exactly reproduce prior evidence. No harness edits. New report ARRAY_FEATURE_REVALIDATION.md / phase1_arrays outputs; next ONLY market smokes, no grid yet.

- [2026-10-09 @pi] Phase 2 all four hard smokes PASS: retained joint UUIDs stand; market-only retry d0d933ae UUIDs 30ad2ca7-ccd4-4cc0-9973-81142b67dcb3 / 3ec74b0f-ee29-4bef-9ed9-dea5ceb8bba5 pass actual DB parity. New market Rhat/bulk/tail: GRW 1.13787/16.10/57.47, QS 1.06250/67.14/74.33; zero divergences, convergence/performance flags NOT waived. Four-arm ETA 1.9374+1.9550+4.1991+4.2278=12.3192 h, before overhead/reruns. Prior GRW 1.14/bulk16/4.219 h clearly retained. Read-only audit confirms all four recipes smoke-eligible without harness changes/history deletion. PHASE2.md and committed phase2_arrays receipts; all owned panes closed. BLOCKED only on manager grid approval; no grid/scoring/reference refit.

## Verification & Findings

- Initial branch clean at 9a86f7b1. Beast detached checkout clean at d16a5800; .env and Manifest symlinks intact.
- Beast snapshot SHA256 c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4; original mtime 2026-09-25 12:57:15.480765468 UTC intact.
- Historical Phase 1 preparation failure `inversion provenance differs` resolved by manager-authorised namespace reuse; earlier component failures resolved under three manager amendments. Historical Phase 1 PASS at 153239a7; new array feature PASS b006743f (ARRAY_FEATURE_REVALIDATION.md). All four Phase 2 hard smokes PASS (PHASE2.md), convergence flags retained. Current blocker is GRID approval only; historical blocked notes remain evidence.
- Phase 0: QS 373/373, GRW 124/124, tape 150/150 + QS 2/2 (0 B), builder 106/106, harness 179/179, market t01–t06 pass (full-book 305/305). Exact counts/wall times and summaries committed in experiments/scotland/06_qs_joint_and_market_observation/; C2 known failures not hidden.

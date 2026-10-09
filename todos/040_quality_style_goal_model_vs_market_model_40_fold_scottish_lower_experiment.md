# 040 — Quality/style goal model vs market model: 40-fold Scottish Lower experiment

| Field | Value |
|---|---|
| ID | 040 |
| Title | Quality/style goal model vs market model: 40-fold Scottish Lower experiment |
| Status | IN_PROGRESS |
| Priority | P1 |
| Assignee | pi |
| Created | 2026-10-09 |
| Updated | 2026-10-09 |
| Related Files / Commits / PRs | — |

## Context & Problem Statement

Execute `experiments/pi_qs_goal_vs_market_prompt.md` phases 0–6: validate the merged base, implement and verify QualityStyleGRW, compare 40-fold Scottish Lower goal and C0 market arms on a pinned snapshot without staking or ROI. No grid before passing smoke and manager approval.

## Acceptance Criteria

- [x] Phase 0 merged-base acceptance tests pass on mcmc-beast; C2-pending explicitly excluded by manager.
- [x] QualityStyleGRW builder, OOS, reconstruction, AD parity and zero-allocation tests pass.
- [ ] Market arm and score adapter validated on matching 40-fold fixtures.
- [ ] All four goal arms and market arm pass smoke; manager approves full grid.
- [ ] 40-fold grid, paired-bootstrap scoring, report and leaderboard completed without ROI.
- [ ] Fresh-REPL frozen-seed reproduction recorded with byte-for-byte comparison.

## Ideas & Candidate Solutions

Follow the phase gates, fixed priors and seeds in the work package. Record decisions in `experiments/scotland/05_quality_style_goal_vs_market/DECISIONS_LOG.md`; operational state in `PROGRESS.md`.

## Work Log & Progress

- [2026-10-09 @pi] Claimed in pi worktree `/home/james/bet_project/.worktrees/BayesianFootball-qs-experiment`; beginning Phase 0 branch-health gate on mcmc-beast. No experiment scores yet.
- [2026-10-09 @pi] Initial Phase 0 stop: pooled t05 C4 full-book deterministic 305/305, C2-pending child testset 29 passed / 10 failed (one integrated derivative, nine archived thin-book Gate 1); manager clarification requested. All other prescribed base tests passed, pinned ScottishLower SHA256 `c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4`.
- [2026-10-09 @pi] Manager explicitly ruled Phase 0 PASS: the expected C2-pending 29/39 is outside acceptance by design; C1 and C2 parked. All acceptance assertions passed. Resumed Phase 1 without changing C2 tests.
- [2026-10-09 @pi] Phase 1 PASS: `QualityStyleGRW` builder, q/s reconstruction and σ_q,K/σ_s,K OOS; full-model linked and unlinked gradient parity on pinned real fold. New test 373/373 (45s), tape matrix 150/150 plus QS 2/2 at 0 B, MultiScaleGRW 124/124, builder and harness 179/179 passed on beast. Starting Phase 2 market arm and scoring adapter. See `experiments/scotland/05_quality_style_goal_vs_market/PROGRESS.md`.

## Verification & Findings

Phase 0 on mcmc-beast isolated checkout `a4729e8e`: `include("test/test_multiscale_grw.jl")` 124/124; `phase0_gate.sh` used fresh Julia REPL per other entry point. Tape 150/150 (5m15s); t01 269/269; t02 131/131; t03 131/131; t04 92/92; t05 pooled C4 305/305 but excluded C2 pending 29/39 (10 failures). Builder, harness, other t05 files, and t06 passed. Phase 1 at `903ed008` (source) / `69d18589` (strengthened test): full new test 373/373, real fold has 40 boundaries and every θ coordinate at three points passes compiled ReverseDiff vs ForwardDiff, fresh ReverseDiff and central finite differences; QS compiled gradients allocate 0 B. Baseline source suites passed. Test logs `/root/BF_runs/logs/qs_experiment/phase1_quality_style.log` and `phase1_ad_full.log`. No smoke, sampling, scores, or reproduction attempted.

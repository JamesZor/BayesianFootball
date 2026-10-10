# 042 — Wave 3: fusion of goals+xG and market rates with learned market bias

| Field | Value |
|---|---|
| ID | 042 |
| Title | Wave 3: fusion of goals+xG and market rates with learned market bias |
| Status | IN_PROGRESS |
| Priority | P1 |
| Assignee | claude |
| Created | 2026-10-10 |
| Updated | 2026-10-10 |
| Related Files / Commits / PRs | experiments/claude_qs_fusion_bias_brief.md; experiments/scotland/07_qs_fusion_market_bias/; TODO 041 |

## Context & Problem Statement

Wave 2 (TODO 041): `qs_marketobs` tied the de-vigged close on 1X2 LogLoss (0.61192 vs
0.61312, not detectable). A market-only model can at best reproduce the market. Wave 3 puts
goals+xG (JointGammaPoissonObservation on the TRUE rates) and the market log-rates on one latent
strength, with learned static market bias: D̃ = δ_D + κ_D·D + e, M̃ = δ_M + M + e, where
D = log λ_h − log λ_a and M = mean log-rate. Priors κ_D ~ LogNormal(0, 0.2), δ_D, δ_M ~ Normal(0, 0.1);
σ_obs as wave 2. Scope set by the human on 2026-10-10; brief `experiments/claude_qs_fusion_bias_brief.md`.

## Acceptance Criteria

- [x] Phase 0: branch health on the beast, frozen table SHA confirmed.
- [x] Phase 1: fusion observation in src with the six test groups passing (likelihood 1e-12, bias-off equality, exact AD 1e-10 + FD convergence, 0 B tapes, 3-seed recovery gate).
- [x] Phase 2: candidates + smoke for fusion_qs_bias / fusion_qs_nobias / fusion_grw_bias; manager approval before grid.
- [x] Phase 3: 40-fold grid, R̂ ≤ 1.05 rerun policy, divergences ≤ 0.1%.
- [x] Phase 4: scoring with wave-2 metric set and intervals, bias posteriors, leakage audit, README/REPORT/leaderboard.
- [ ] Phase 5: one-fold byte-for-byte reproduction per arm.

## Ideas & Candidate Solutions

- See the brief; implementation choices are logged in `07_.../DECISIONS_LOG.md`.

## Work Log & Progress

[2026-10-10 @claude] Created; Phase 0 started.
[2026-10-10 @claude] Phase 0 PASS at 122cbc37: 16/16 gates in fresh beast REPLs, pins match (see 07_.../PHASE0.md).
[2026-10-10 @claude] Phase 1 PASS at 01f42219 (PHASE1.md): JointMarketFusionObservation; fusion tests 4256/4256, exact AD ≤ 2.3e-15, Richardson ≤ 2.6e-22, 0 B tapes, recovery 84/99 pooled (qs_nobias 21/27 flagged). Phase 0 count gap reconciled.
[2026-10-10 @claude] Phase 2 smoke PASS at 79f003e0 (PHASE2.md): 6/6 hard checks on all three arms, 0 divergences; short-smoke R̂ 1.081/1.103/1.059 (review); ETA 13.343 h. Manager approved the grid as frozen.
[2026-10-10 @claude] Phase 3 grid a1 PASS at dde8a441 (PHASE3.md), 2.7 h: fusion_qs_bias 26de19bf (control), fusion_qs_nobias 5afdfeb3, fusion_grw_bias b2e5a527; max R̂ 1.019/1.012/1.015, no reruns, 0 divergences, control score parity exact; tail ESS < 400 only control fold 40 (330.7).
[2026-10-10 @claude] Phase 4 a1 PASS (PHASE4.md, REPORT.md): scorer 4f7ee7bc, references exact to wave 2; primary fusion_qs_bias − close 1X2 LogLoss −0.000325, noncircular 90% [−0.00707, 0.00113] n.d.; bias ablation +0.000235 n.d.; κ_D > 1 in 40/40 folds (QS 1.17, GRW 1.33), δ_D/δ_M include 0 everywhere; leakage audit PASS (120 folds, 1430 rows); register status scored; decision no promotion. Phase 5 next.
Include session/worktree when claiming. Example (replace with actual facts):

<!-- - [2026-10-09 @pi] Claimed in session <id>, worktree <path>; next action: ... -->

## Verification & Findings

Runs (`experiments/scotland/07_qs_fusion_market_bias/results/phase3/RUNS.csv`): fusion_qs_bias
`26de19bf-fc29-4818-bfdf-7bf76535901f` (control), fusion_qs_nobias
`5afdfeb3-abad-4d23-82fb-1bfe6babe797`, fusion_grw_bias `b2e5a527-e51e-4913-b7bb-e1b94a4d1ff2`;
0 divergences, R̂ ≤ 1.0194, control fold-40 tail ESS 330.7 flagged.

Findings (REPORT.md): no arm beats the close on the prescribed noncircular interval (primary
−0.000325 1X2 LogLoss, n.d.; categorical −0.00104, RPS +0.00014 on 595 fixtures, both n.d.); the
bias terms add nothing detectable. The market is not compressed relative to the fused latent:
κ_D > 1 in every fold, and learning it makes predictions more compressed (market-on-model slope
1.277 vs 1.106). δ_M and log κ correlate −0.80; their sum (+0.10) is well identified.
Leakage audit PASS. Reproduction (Phase 5) pending.

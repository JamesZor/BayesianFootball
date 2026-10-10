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
- [ ] Phase 2: candidates + smoke for fusion_qs_bias / fusion_qs_nobias / fusion_grw_bias; manager approval before grid.
- [ ] Phase 3: 40-fold grid, R̂ ≤ 1.05 rerun policy, divergences ≤ 0.1%.
- [ ] Phase 4: scoring with wave-2 metric set and intervals, bias posteriors, leakage audit, README/REPORT/leaderboard.
- [ ] Phase 5: one-fold byte-for-byte reproduction per arm.

## Ideas & Candidate Solutions

- See the brief; implementation choices are logged in `07_.../DECISIONS_LOG.md`.

## Work Log & Progress

[2026-10-10 @claude] Created; Phase 0 started.
[2026-10-10 @claude] Phase 0 PASS at 122cbc37: 16/16 gates in fresh beast REPLs, pins match (see 07_.../PHASE0.md).
Include session/worktree when claiming. Example (replace with actual facts):

<!-- - [2026-10-09 @pi] Claimed in session <id>, worktree <path>; next action: ... -->

## Verification & Findings

Not run yet. Record commands, pass/fail or skips, wall time, benchmark metrics,
R-hat / bulk and tail ESS / divergences / predictive scores as applicable, and
links to artifacts and immutable run IDs. Distinguish hypotheses from findings.
- 2026-10-10: Phase 1 PASS at 01f42219 (PHASE1.md): JointMarketFusionObservation; fusion tests 4256/4256, exact AD ≤ 2.3e-15, Richardson ≤ 2.6e-22, 0 B tapes, recovery 84/99 pooled (qs_nobias 21/27 flagged). Phase 0 count gap reconciled.

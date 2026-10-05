# 039 — Market model: inversion and team-strength EDA

| Field | Value |
|---|---|
| ID | 039 |
| Title | Market model: inversion and team-strength EDA |
| Status | ACTIVE |
| Priority | P1 |
| Assignee | claude (manager), pi (build) |
| Created | 2026-10-05 |
| Updated | 2026-10-05 |
| Related Files / Commits / PRs | `current_development/market_model/DESIGN.md`; TODO 023 (`current_development/market_inverse_dynamics/`); tickets T014, T015; CLV napkin §14 (`proto/clv-napkin`) |

## Context & Problem Statement

The human's design note of 2026-10-05 proposes a "market model": invert each match's odds to
double-Poisson rates, then model those rates with the attack/defence/home structure and a GRW
on α, β, instead of modelling goals. TODO 023 already inverted the close (SSE, per match) and
fitted Kalman GRWs to the log-rates. Three things remain open: how to invert (each match in
isolation, or pooled with other matches), what the latent rates look like, and how well they
decompose into team strength. The full design and the decisions from the 2026-10-05 grill are
in `current_development/market_model/DESIGN.md`.

## Acceptance Criteria

- [ ] Phase A: KL inversion with the 1X2 gate, synthetic recovery, KL vs SSE, inversion-quality
      EDA including the held-out-line test.
- [ ] Phase B: two-stage ladder R0–R4 on the Kalman engine (engine gates pass; R̂ ≤ 1.05,
      ESS ≥ 200), measures on both axes and both θ protocols; Gamma / log-normal / log-t test;
      rate-distribution and rates-vs-goals EDA.
- [ ] Phase C: pooled Turing model (Dirichlet/Beta books, learned n, u_m), synthetic recovery,
      P1 and P2 converged (divergences ≤ 0.1%), one-step MAP refits at 40 fold cutoffs.
- [ ] Phase D: isolated vs pooled and team-parameter EDA; findings README with every number
      traceable to `results/`.

## Ideas & Candidate Solutions

- Isolated vs group is one spectrum: `log λ = structure + u_m`, with u_m free (isolated),
  u_m = 0 (pure structure), or hierarchical (chosen). Pooling fixes T015 by construction.
- A tempered cross-entropy with a free concentration is not a proper likelihood; use
  Dirichlet (1X2) and Beta (two-way) with one learned n.
- Rejected for now: Dixon–Coles ρ and other grids (double Poisson agreed); fixing T015 in `src`
  (local gate instead); builder-native observation (after the EDA).
- Literature: Egidi, Pauli & Torelli (2018) invert per match from 1X2 only; Wunderlich &
  Memmert (2018) find odds-based ratings beat goal- and result-based ratings out of sample.

## Work Log & Progress

- [2026-10-05 @claude] Design agreed with the human over four rounds of questions. Worktree
  `/home/james/bet_project/.worktrees/BayesianFootball-market-model`, branch
  `feat/market-model-inversion-eda` off `feat/market-inverse-grw-dynamics`. ID 039 taken as the
  next free number after the experiment-harness index (038). Next: Phase A brief for pi.

## Verification & Findings

Not run yet.

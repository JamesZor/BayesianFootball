# 039 — Market model: inversion and team-strength EDA

| Field | Value |
|---|---|
| ID | 039 |
| Title | Market model: inversion and team-strength EDA |
| Status | IN_PROGRESS |
| Priority | P1 |
| Assignee | pi |
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
- [2026-10-05 @pi] Phase A loader, runner, and tests implemented locally; 264 tests pass,
  but the runner fails twice at the residual Plots layout assertion. CSV diagnostics were
  written before the failure; figure set/runtime are incomplete. See
  `current_development/market_model/results/A/PHASE_A_REPORT.md`. Phase A blocked,
  not promoted to Phase B; manager/human review pending.
- [2026-10-05 @pi] Phase A follow-up: diagnosed the Plots/GR failure as state left by
  the log-x KL histogram, repaired the empty histogram and figure ordering, added
  same-fixture in-sample held-out comparisons. Fresh `pi_julia_mm` tmux Julia REPL:
  `R01_DONE` and `T01_DONE` (269/269). All four figures and runtime CSV present;
  Phase A ready for manager review, not Phase B authorisation.

## Verification & Findings

Not run yet.

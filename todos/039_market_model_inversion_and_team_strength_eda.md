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
- [2026-10-05 @claude] Phase A reviewed and merged (`42b09664`). Rerun in a fresh tmux REPL
  (`claude_julia_mm`): 269/269 tests, runner 38.6 s, every CSV and PNG byte-identical to
  `efa71a3c`. Accepted deviations: 12473327 refused at gate 1 (no book in the snapshot); KL gate
  fixed at 0.01 (nothing reaches it). Phase B awaits human sign-off.

## Verification & Findings

**Phase A (2026-10-05).** 595 of 710 fixtures accepted (78 no book, 36 no complete 1X2, 1 T014).
The 239 panel market groups removed by the completeness gate are all one-runner O/U 0.5 groups;
TODO 023's target builder already skipped them, and an ungated rerun reproduces its 623 SSE rates
exactly. KL vs SSE: median |Δ log λ| ≈ 0.0002–0.002, max 0.039. Multi-start spread 4e-8.

The double-Poisson grid misprices the close systematically: draw −1.34 pp [−1.41, −1.26],
home/away +0.65/+0.68 pp, BTTS-yes −0.79 pp in sample and −1.20 pp held out. **Consequence
(manager check):** on the 517 accepted books with ≥ 5 selections, inverting from the 1X2 alone
gives a log-total 0.103 lower than the full book (q05/q50/q95 −0.181/−0.105/−0.014) with
supremacy almost unchanged (+0.010). The 78 accepted 1X2-only books therefore carry totals biased
low by about 0.10, the size of the panel's whole log-total sd.

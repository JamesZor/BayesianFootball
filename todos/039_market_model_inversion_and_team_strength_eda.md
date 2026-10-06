# 039 — Market model: inversion and team-strength EDA

| Field | Value |
|---|---|
| ID | 039 |
| Title | Market model: inversion and team-strength EDA |
| Status | IN_PROGRESS |
| Priority | P1 |
| Assignee | pi |
| Created | 2026-10-05 |
| Updated | 2026-10-06 |
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

- [x] Phase A: KL inversion with the 1X2 gate, synthetic recovery, KL vs SSE, inversion-quality
      EDA including the held-out-line test.
- [x] Phase B: two-stage ladder R0–R4 on the Kalman engine (engine gates pass; R̂ ≤ 1.05,
      ESS ≥ 200), measures on both axes and both θ protocols; Gamma / log-normal / log-t test;
      rate-distribution and rates-vs-goals EDA.
- [x] Phase B2: config regression, covariance rungs R2/R5–R7 in both protocols,
      batch/FFBS gates, convergence and byte-reproducible results; stream process README;
      quality/style rotation, season-average RTS levels correlation and weekly level ICC.
- [x] Phase B3: refiltered FFBS Gaussian-step tail/dependence PPCs; four global-parameter
      score grids with book/heldout/bias/outcome comparisons; converged informational R6
      refits, fresh tests and byte-identical scientific artifacts. Phase C stays double Poisson.
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

- [2026-10-06 @pi] Phase B3 claimed on `pi/market-model-phase-b3`, dedicated worktree.
  Local pane `%34`; pinned cache SHA matches locally and on beast, initial load 0.06.
  Building local score-grid profiles and refiltered FFBS posterior predictive tail checks;
  same R6 priors/budgets, no Phase C, `src/`, package/data changes or database writes.

- [2026-10-05 @pi] Phase B2 claimed on `pi/market-model-phase-b2` in the dedicated
  `BayesianFootball-market-model-pi-b2` worktree. Config-driven pipeline and R5–R7 only;
  no C/D or `src/` changes. Human approved excluding only nondeterministic `runtime.csv`
  from Phase A's byte regression; all other CSVs and published A/B files stay unchanged.
  Laptop REPL `%27` is owned; beast uses `root@mcmc-beast` (the default local username
  is not a beast account). Initial load 3.84. Reused engine already derives season breaks
  from observed season dates; no literal boundary exists there to replace.
- [2026-10-05 @pi] Human/manager addendum accepted mid-run, post-processing only:
  posterior quality/style rotation and level-residual weekly ICC, simulations and
  Koopman/Baio citations. Copied manager brief §7 verbatim into this branch's brief.
  Human clarified levels: season-average RTS means across active teams, point correlation
  at median theta; posterior intervals apply to innovation rotation. Implementation
  defines active teams as those observed in accepted-panel fixtures for the season.
  Both completed full B2 runs are preserved; no additional model fits for the addendum.

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
- [2026-10-05 @pi] Phase B claimed in `pi/market-model-phase-b`, worktree
  `BayesianFootball-market-model-pi-b`; only own tmux REPLs. Implementing the 517-fixture
  ladder, conditional Kalman gates, family comparison and outcome EDA. Human approved
  seeded R4 RBPF (prior fixture-shared Gamma proposals; exact conditional Kalman states)
  and family-specific priors for the new static scale/shape parameters. Manager added
  fixed-omega reduction, per-week ESS/resampling and three-seed stability checks.
  No Phase C work. Initial deterministic R1 FFBS test exposed ~1.2e-8 round-off noise
  in zero-process steps; the prototype now copies the mathematically deterministic
  backward step exactly, rather than loosening the test.
- [2026-10-05 @claude] Phase A reviewed and merged (`42b09664`). Rerun in a fresh tmux REPL
  (`claude_julia_mm`): 269/269 tests, runner 38.6 s, every CSV and PNG byte-identical to
  `efa71a3c`. Accepted deviations: 12473327 refused at gate 1 (no book in the snapshot); KL gate
  fixed at 0.01 (nothing reaches it). Phase B awaits human sign-off.

- [2026-10-05 @pi] Phase B verified and handed off for manager/human review. Fresh
  laptop: 131/131, `T02_DONE`. Two fresh beast full runs at `8663544e`:
  `R02_DONE` in 3291.801 / 3308.673 s; all 37 regenerated scientific CSVs byte-identical.
  All 30 ladder θ gates and 33 family-fit gates pass; zero retained NUTS divergences.
  R4 fixed-omega reduction, per-week ESS/resampling, named worst weeks and three-seed
  diagnostics are published; severe occasional particle collapse is explicitly reported.
  Log-t leads fixture CV by 21.03 ± 10.35 nats versus Gamma (fixture SE), not a decisive
  universal winner. Plot-only margins corrected/verified at `9469f327`. Report:
  `current_development/market_model/results/B/PHASE_B_REPORT.md`; seeded proof:
  `results/B/REPRODUCIBILITY.md`. No Phase C/D or `src/` changes, DB writes or package
  changes. TODO remains IN_PROGRESS for unapproved/uncompleted C/D.

- [2026-10-05 @pi] Phase B2 and the human hierarchy addendum verified for manager review.
  Fresh laptop `%31`: t03 131/131, t02 131/131, `R01_DONE`/`T03_DONE`/`T02_DONE`.
  Two fresh beast full runs at `25a408e2`: `R03_DONE` in 566.876 / 565.324 s;
  all 10 core CSVs and both PNGs byte-identical; 34 engine gates and 32 theta gates pass.
  Addendum-only fresh runs at `fd8b23a9`: `R03_HIERARCHY_DONE` twice, both additional
  CSVs byte-identical, no new fits. R6 leads honest point density and brings
  supremacy/level coverage to 90.11% / 91.21%; style/quality innovation SD ratio
  0.315 (10a) / 0.282 (10b), not an exactly one-dimensional hierarchy. Honest weekly
  level ICC is consistent with zero. Full evidence/citations/limitations in
  `current_development/market_model/results/B2/PHASE_B2_REPORT.md`; process in the
  stream README. Published A/B artifacts untouched; owned REPLs closed, other
  sessions not operated. No Phase C/D, group/tier fits, `src/` or DB/package changes.
  TODO remains IN_PROGRESS pending unapproved/unimplemented C/D.
- [2026-10-05 @claude] **Phase B reviewed and merged** (`ea242b20`).
  - Laptop: fresh REPL, t02 131/131.
  - Gates: engine 24/24, convergence 30/30; 0 divergences in 33 family fits.
  - Independent beast rerun at `8663544e` (3,374 s): all 37 runner CSVs byte-identical to
    pi's (the two preflight CSVs are not runner outputs).
- [2026-10-05 @claude] **Phase B2 reviewed and merged.**
  - Laptop: fresh REPL, t03 131/131 and t02 131/131.
  - Gates: engine 34/34, convergence 32/32.
  - Independent beast rerun at `25a408e2` (601 s), then the addendum at `fd8b23a9`: all 12
    CSVs and both PNGs byte-identical to `SCIENTIFIC_SHA256SUMS.txt`.
  - Attribution corrected in the README and report: the manager, not the human, answered
    pi's question on the per-season levels definition.

- [2026-10-06 @pi] Phase B3 verified for manager/human review on
  `pi/market-model-phase-b3`. Fresh laptop `%36`: t04 92/92, t03 131/131,
  t02 131/131, all markers. Two fresh beast runs at `2e02a785`: `R04_DONE`
  in 501.581 / 508.821 s; all 32 CSVs and three PNGs byte-identical. All 24
  R6 theta gates pass. No Gaussian PPC p-value is extreme: no alternative step
  model fitted. G1 Dixon–Coles has lowest heldout point MAE (0.008648 vs G0
  0.008846); mean 1X2-only log-total bias shrinks from -0.102716 to -0.008941.
  Paired outcome-loss CIs include zero. R6 covariance remains strongly negative;
  G1 including the 78 thin books is informational, with a changed evaluation set.
  Only 353 structural fixtures quote OU2.5: grid/G0 outcome losses cover all 517,
  close-paired losses explicitly cover available quotes. Report, methods,
  scientific hashes and reproduction commands in `results/B3/` under the stream.
  All owned REPLs closed; published A/B/B2 untouched. No `src/`, DB, package/data,
  outside-data or Phase C changes. Human decision: double Poisson for C; C2 later.
  TODO remains IN_PROGRESS for C/D; `todo.sh check` and `git diff --check` pass.

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

**Phase B (2026-10-05).** 517 fixtures / 1034 observations, excluding the prescribed
78 1X2-only accepted books. Weekly R2 explains 98.06% supremacy / 92.52% level
(smoothed R²); honest 25/26 RMSE .19556 / .05049. Per-team HA adds negligible gains;
heavy tails modestly improve density but worsen supremacy RMSE and do not fix coverage.
All acceptance evidence and limitations (R4 ESS collapse, static log-t's infinite rate
moments, weak away-goal calibration flattening) are recorded in the Phase B report.
Phases C/D require human sign-off; this task is not closed.

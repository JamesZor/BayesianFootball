# 030 — Shared experiment harness: recipes, gates, runner, scorecard, leaderboard

| Field | Value |
|---|---|
| ID | 030 |
| Title | Shared experiment harness: recipes, gates, runner, scorecard, leaderboard |
| Status | ACTIVE |
| Priority | P1 |
| Assignee | pi |
| Created | 2026-09-26 |
| Updated | 2026-09-26 |
| Related Files / Commits / PRs | `docs/architecture/rfc_experiment_harness.md`, `docs/architecture/experiment_harness_suite_catalogue.md`, TODO 010, TODO 031 |

## Context & Problem Statement

Every Scottish experiment suite (05–13), TODO 028 and `current_development/grw_pyramid_cups`
re-implements the same harness: AD/tape gates, smoke runners, recipe fingerprinting,
promotion manifests, grid runners, evaluation and scoring. That is about 300–1,100 lines
per suite. The copies disagree. TODO 028 blocked on a zero-byte tape gate and a fixed
12×12 grid-mass floor that `grw_pyramid_cups` never applied. Suites use opposite slope
conventions and different reference instants. Some runs come from untracked rsync copies,
and there is no cross-suite leaderboard.

Decision (human, 2026-09-26): build ONE shared harness in `src/` in the spirit of FunSearch.
The evaluator is fixed library code, a candidate is a recipe (data), and a leaderboard is
the memory. Agents compose recipes first (phase 1) and write new components later
(phase 2, gated). Gates are mostly warnings or review flags; only correctness failures are
hard. The scorecard is configured by a JSON spec (stage × metric × market × subset) that we
review together. Design: `docs/architecture/rfc_experiment_harness.md`.

## Acceptance Criteria

Plan: `docs/architecture/rfc_experiment_harness.md` §6. Each phase is usable on its own.

- [ ] Plan reviewed; §8 decisions made (slope inversion, transition cohort, score storage, branching).
- [ ] **Phase 1 — scorer, cohorts, leaderboard (delivers W0):** `BayesianFootball.Harness`
      (`score_runs`/`score_fits`, cohorts, `harness_scores` table), `scripts/score_runs.jl`,
      `scripts/leaderboard.jl`. Reproduces `grw_pyramid_cups` r04 proper scores to 1e-9 and
      slopes to ±0.02 for g1/g2/g3/m12_td; `experiments/LEADERBOARD.csv` for all W0 runs
      incl. transition-cohort columns; `test/harness_scoring_tests.jl` passes.
- [ ] **Phase 2 — data scope:** `Data.ScottishPyramid`, `DataScope`, `apply_scope`
      (`:senior_spfl_ties`), `ScopedWalkForwardCV` + features clock. Lower scope reproduces
      40 / 710 exactly; `spfl_cups` reproduces the g2 per-fold widening CSV or explains
      every difference; `test/scoped_splitter_tests.jl` passes.
- [ ] **Phase 3 — runner:** `Candidate`, checks (hard vs recorded), `screen`/`smoke`/`grid`
      with resume and smoke-before-grid, `harness_checks` table, `scripts/run_candidates.jl`,
      `scripts/beast_checkout.sh`. The g2 recipe re-run through the harness reproduces g2's
      LogLoss (paired CI covers 0); `test/harness_runner_tests.jl` passes.
- [ ] **Phase 4 — finalist portfolio + docs:** Option B portfolio via the harness matches
      Task 013 r05 for m12_td; `docs/guides/experiment_harness_guide.md` linked from AGENTS.md;
      AGENTS.md §6 cites re-scored numbers.

## Ideas & Candidate Solutions

- Wrap existing facilities rather than duplicating them: `FitConfig`/`fit_model`,
  `audit_convergence` + `ConvergenceThresholds`, `PostgresStorage`/`config_hash`,
  `evaluate_predictions`, `calibrate_fit`, `run_portfolio_simulation`.
- Phase 2 (agent loop, overlaps TODO 010): writable surface = `recipes/` +
  `proposals.jsonl`; compute budget file; de-duplicate recipes by hash before sampling.
- Rejected: rewriting historical suites onto the harness. They stay frozen as records;
  only their persisted latents are re-scored.

## Work Log & Progress

- [2026-09-26 @claude] Scoped with the human in worktree
  `.worktrees/BayesianFootball-scotland-cross-tier`, branch `feat/scotland-cross-tier-models`
  (after committing TODO 028's work in progress and merging `feat/scottish-cross-tier-cups-eda`).
  Drafted the RFC; revised the same day into a lean four-phase implementation plan after the suite review (`experiments/scottish_lower/REVIEW.md`). A pi agent (`ag/gemini-3.8-flash-high`, tmux `agent_pi_suite_catalogue`) is
  producing the read-only suite catalogue as the API input. No harness code written yet;
  next action: RFC review.
- [2026-09-26 @claude] Plan approved with decisions (RFC §8). Execution delegated to pi (`openai-codex/gpt-5.6-sol`, thinking high) in worktree `.worktrees/BayesianFootball-experiment-harness`, branch `feat/experiment-harness`; Claude manages and reviews each phase. Phase 1 first.
- [2026-09-26 @pi] Implemented the Phase 1 Harness module skeleton, cohort definitions, versioned scorer, additive score/register store, W0 and register seeds, CLI scripts, and offline synthetic tests. Standalone harness tests pass (19/19); remote W0 scoring and reproduction checks remain before handoff.

## Verification & Findings

Not run yet.

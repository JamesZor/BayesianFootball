# 036 — Harness persistence and reporting fixes (k, l, m, j, gate alignment)

| Field | Value |
|---|---|
| ID | 036 |
| Title | Harness persistence and reporting fixes (k, l, m, j, gate alignment) |
| Status | BLOCKED |
| Priority | P1 |
| Assignee | pi |
| Created | 2026-09-28 |
| Updated | 2026-09-28 |
| Related Files / Commits / PRs | TODO 031 (follow-ups j, k, l, m), TODO 030; branch `fix/harness-klm` off `feat/w2-tier-components` `090db11e`; brief `experiments/pi_harness_klm_prompt.md` |

## Context & Problem Statement

TODO 031 wave 2 surfaced five harness defects. They are logged in the TODO 031 Work Log
(2026-09-27 and 2026-09-28).

- **(k)** A failed diagnostic records `value = {}`. `_run_diagnostic!` (`src/harness/checks.jl`)
  catches the error that `_convergence_diagnostic` throws and discards the metrics it had
  computed, so R̂, ESS and divergences survive only in the grid log.
- **Gate alignment.** The harness convergence review fails on its own thresholds (ESS > 400, and
  R̂ 1.01). The human rule, set on 2026-09-27 (W2 README §7), fails only on R̂ > 1.05 or
  divergences > 0.1% of draws. Three W2 GRW arms show `review=FAIL` purely on ESS.
- **(l)** `save_fit` serializes the whole `Fit` into one `fit_artifacts.fit_blob`
  (`src/training/inference/db_storage.jl`), and PostgreSQL caps a field at 1 GiB, even through the
  binary path.
  - W2 GRW arms had to be thinned to stride 4, the last rung of the harness fallback
    (`src/harness/stages.jl`). There a failed save rethrows and the sampling run is lost.
  - `extend_fit` rewrites the whole blob.
- **(j)** `save_fit` deduplicates on the UNIQUE `configs.config_hash` and silently returns the
  existing run. As a result:
  - An unchanged recipe cannot be re-smoked: `fit_parity` fails, and the failed record blocks the
    grid.
  - An identical grid would silently return an old run.
- **(m)** `scripts/leaderboard.jl` reads v1.2 rows only.
  - Regenerating it drops the 14 W0 reference runs, which have only v1.1 rows (among them the
    live champion `m12_td` `132df5c2`), and the MAP-screen section.
  - It shows Δ only against the hard-coded W0 control, so W2's per-base Δs read "—".
  - `LEADERBOARD.md` has therefore been left at v1.1 since 2026-09-28.

## Acceptance Criteria

The design decisions were agreed with the human in a grill-me session on 2026-09-28. Detail and
test obligations are in the brief `experiments/pi_harness_klm_prompt.md`.

- [ ] (k) A failed diagnostic persists its computed metrics in `harness_checks.value`.
- [ ] Gate alignment: the harness convergence review fails only on R̂ > 1.05 or divergence rate
      > 0.1%. ESS below 400 is recorded as a note and does not fail the review.
- [ ] (l) Per-fold artefacts:
  - [ ] a new table `fit_fold_artifacts`, with one blob per fold and a slim run-level shell;
  - [ ] `load_fit` reads both the new and the legacy layout;
  - [ ] `extend_fit` appends folds;
  - [ ] stride is always 1, and the thinning fallback is removed;
  - [ ] no migration of existing runs.
- [ ] (j) Smoke saves carry a per-invocation nonce. `save_fit(...; on_duplicate = :return | :error)`,
      and the harness grid path uses `:error`.
- [ ] (m) The leaderboard shows each row's Δ against its own recorded control, and a runbook plus
      CSVs re-score the 14 W0 runs and the MAP-screen runs at v1.2. The regenerated board loses no
      run relative to the v1.1 board.
- [ ] Validation:
  - [ ] a synthetic save/load round trip above 1 GiB, bit-exact, on the test DB, starting from the
        real `grw_base` W2 fit;
  - [ ] legacy runs still load;
  - [ ] existing suites green.
- [ ] One Claude CLI (Opus 5.5) review at the end, with at most 2 fix rounds, then escalation to
      the human.
- [ ] Merged into `feat/w2-tier-components`.
- [ ] Production steps, run by pi after the human's go:
  - [ ] schema created in `mcmc_experiments`;
  - [ ] W0 and MAP-screen runs re-scored at v1.2;
  - [ ] `LEADERBOARD.md` regenerated and committed.

## Ideas & Candidate Solutions

- Rejected for (l):
  - byte-chunking one blob: the whole `Fit` stays in memory, and extension still rewrites it;
  - Float32 draws: this loses bit-exact parity;
  - migrating the 324 legacy runs.
- Rejected for (m): a v1.1 fallback in the board instead of re-scoring.

## Work Log & Progress

- [2026-09-28 @claude] Created after the W2 write-up. The design was agreed with the human
  (grill-me, 2026-09-28), and the build is delegated to pi-solo (`openai-codex/gpt-6-sol`,
  thinking high) on `fix/harness-klm`. Review is by Claude CLI (Opus 5.5) at the end.

- [2026-09-28 @pi] Partial build paused under the work package §6: all 12 v1.1 MAP-screen UUIDs lack a persisted `runs` row, so `scripts/score_runs.jl` cannot re-score them at v1.2 without re-fitting MAP (forbidden by §0). See `docs/architecture/harness_klm_report.md` for the read-only SQL evidence, completed test counts, and decision required. Production untouched.

## Verification & Findings

Partial integration: `test/test_extension.jl` 43/43 (27.0s); `test/test_db_storage.jl` 133/133 (33.5s), both on the separate beast test database. Full acceptance not run; blocked on the MAP-screen decision.

# Work package — Experiment harness, Phase 3: runner (TODO 030)

You are the implementer; Claude is the manager and reviews against §3 before Phase 4 is
released. **Do Phase 3 only, then stop.** Same worktree, branch, git, DB, beast and
credential rules as Phase 1 (`experiments/pi_harness_phase1_prompt.md` §1). Re-read them.
This phase **does** run MCMC, so the compute rules in §1 below matter.

## 0. Read first

1. `docs/architecture/rfc_experiment_harness.md` §3 (`Candidate`, `recipe_hash`), §4 (stages
   and checks), §6 Phase 3, §7.
2. `docs/architecture/harness_phase1_report.md`, `docs/architecture/harness_phase2_report.md`
   (especially "Open questions for Phase 3").
3. Code you are graduating: `current_development/grw_player_hybrid/l01_loader.jl` §3–§6
   (`gph_fit_configs`, `gph_register!`, `gph_run_hash`, `gph_completed_run`,
   `gph_fold_inputs`, `gph_filtration_report`, `gph_gradient_audit`, `gph_sample`,
   `gph_thin_for_persistence`, `gph_latent_audit`, `gph_save_and_verify`,
   `gph_assert_coverage`, `gph_convergence_row`); `current_development/grw_pyramid_cups/r01_smoke.jl`,
   `r02_overnight.jl`; `src/training/inference/` (`fit_model`, `FitConfig`,
   `audit_convergence`, `ConvergenceThresholds`, `FitMetadata`/`git_commit_id`);
   `src/samplers/types.jl` (`MAPConfig`, `QueuedNUTSConfig`).

## 1. Compute rules for this phase

- MCMC runs **only on mcmc-beast**: `-t 16`, `pinthreads(:cores)`,
  `BLAS.set_num_threads(1)`, one production grid at a time. Before a grid, check
  `pgrep -af julia` and confirm no other sampling job is running (the Kaimon REPL process
  in tmux session `julia` is expected and must be left alone). If another grid is running,
  wait; don't start a second one.
- Laptop: unit tests only (tiny synthetic NUTS in tests is fine).
- Run from a git checkout at your pushed SHA made by the new `scripts/beast_checkout.sh`
  (below), never from an rsync copy.

## 2. What to build

| Item | Work |
|---|---|
| `src/harness/candidate.jl` | `Candidate` exactly as RFC §3 (`name, model, scope::DataScope, sampler, role, hypothesis`); `recipe_hash(c)` = SHA-256 of canonical `string(model)` + `string(scope)` (sampler/folds excluded, so smoke and grid share it); `fit_config(c; stage, experiment)` → `FitConfig` using `ScopedWalkForwardCV(c.scope)` and `QueuedExecution(max_concurrent_tasks = 16)`, with the smoke sampler (2 chains × 200 + 200 warmup) or the candidate's sampler |
| `src/harness/checks.jl` | port the `gph_*` audits. **Hard** checks throw `HarnessCheckError(check, detail)`: gradient (compiled = fresh RD ≤ 1e-8, RD = FD ≤ 1e-6, three perturbed points), filtration (no overlap, strict cutoff, scope asserts), latents (finite, positive, non-degenerate), score-grid coherence (finite; 1X2/totals/BTTS partitions consistent), **target** coverage (40 folds / 710 target fixtures, unique), `save_fit`/`load_fit` parity. **Recorded** diagnostics (never throw): convergence summary via `audit_convergence` (src default thresholds) → severity `review` if it fails; tape bytes and gradient latency; grid tail mass at K = 12 (max and p99 over draws); **monitor coverage** (held-out monitor fixtures and how many have a Betfair reference, per tournament) |
| `src/harness/store.jl` | add `harness_checks(run_id, recipe_hash, experiment, candidate, stage, check, severity, status, value, detail, git_sha, at)` (additive, `CREATE TABLE IF NOT EXISTS`) + write/read helpers |
| `src/harness/stages.jl` | `screen(candidates; ds, experiment)`: `MAPConfig` on every fold; build latents; `score_fits` in memory; write scores with `stage = "screen"`; no run persisted. If `fit_model` doesn't accept `MAPConfig` with this path, implement a minimal per-fold MAP in the harness and say so. `smoke(c; ...)`: 2 folds (first and last), all hard checks, diagnostics recorded; the smoke Fit is saved under `<experiment>_smoke`. `grid(c; ...)`: **refuses** unless `harness_checks` holds a passing smoke for `recipe_hash(c)` (all hard checks `pass`); resumes a completed run with the same `config_hash`; per-fold checkpoints; full draws saved (binary bytea), retrying at stride 2 then 4 only if the save fails, with the stride recorded; then scores the run via `score_runs` against the candidates file's `:control`. A convergence failure is saved, scored and flagged `review`, not thrown |
| `scripts/run_candidates.jl` | `julia --project -t 16 scripts/run_candidates.jl <candidates.jl> --stage screen|smoke|grid [--only name,...]`. The candidates file defines `EXPERIMENT::String` and `CANDIDATES::Vector{Candidate}`. It pins threads for smoke/grid, refuses smoke/grid with fewer than 8 threads, prints one summary line per candidate (hard checks, review flags, LogLoss, compression slope, run_id), and upserts a `harness_experiments` row for the experiment |
| `scripts/beast_checkout.sh <sha>` | on the beast: `git -C /root/BayesianFootball fetch origin`, `git worktree add /root/BF_runs/<sha> <sha>` (idempotent); symlink `Manifest.toml` and `.env` from `/root/BayesianFootball` (the Manifest is **not** tracked, and a fresh checkout without it fails on `DotEnv`); copy `.cache/datastore_Scottish*.jls` if available; print the path |
| W1 candidates file | `experiments/scotland/03_dynamics_scope_matrix/candidates.jl` (TODO 031 wave 1): the 12 candidates {`TimeDecayDynamics(days_half_life = 180.0)`, `MultiScaleGRW()`} × {lower, spfl (train 54–57, monitor 54/55), spfl_cups (+ `:senior_spfl_ties`, monitor 54/55)} × {`PoissonObservation()`, joint (the exact `gph_joint_observation()` recipe)}, with `GlobalInterception` + `GlobalHomeAdvantage`; control = `td_lower_joint`; default sampler 4 × (500 + 1000), acceptance 0.65. Plus a stub `README.md` (question, candidates, decision rule from TODO 031). **Only the screen stage runs for these in Phase 3** |
| Tests `test/harness_runner_tests.jl` | tiny synthetic league: `recipe_hash` stability (same recipe → same hash; sampler change → same hash; model change → different hash); grid refuses without a passing smoke; severity routing (a forced convergence failure is saved and flagged `review`, a forced hard failure throws and records `fail`); `beast_checkout.sh` has `bash -n` syntax OK. Add to `run_parallel_tests.jl` |

## 3. Acceptance (Claude will check each)

1. **g2 reproduced through the harness.** A candidates file
   `experiments/scotland/00_harness_validation/candidates.jl` (`EXPERIMENT = "harness_validation"`)
   holds `g2_harness_repro`: scope `spfl_cups` (no monitor), model = the g2 recipe
   (`GlobalInterception`, `MultiScaleGRW()`, `GlobalHomeAdvantage`, `PoissonObservation`),
   sampler = **g2's original** `QueuedNUTSConfig(n_samples = 1000, n_warmup = 500, n_chains = 4, accept_rate = 0.80, max_depth = 10)`.
   Run `smoke` then `grid` on the beast from a `beast_checkout.sh` checkout. Then:
   - (a) the convergence audit passes, or is flagged and explained;
   - (b) paired fixture-clustered bootstrap ΔLogLoss vs g2 `a6f62436-ec8a-461d-8bd5-dc1861a2daaa` on the 710 target fixtures has a 95% CI covering 0 and |Δ| < 0.001;
   - (c) `compression_slope` within ±0.05 of g2's 1.2517;
   - (d) `runs.git_commit` for the new run equals the checkout SHA (not "unknown").
   Expected wall time ~80–90 min.
2. **Smoke-before-grid enforced** on the real path: `--stage grid` for a candidate with no
   smoke record refuses with a clear message (show the output).
3. **W1 screen:** `--stage screen` over the 12 W1 candidates completes and writes
   `stage = "screen"` scores. Report the wall time and a table of screen LogLoss and
   compression slope per candidate. Any candidate that fails to build or fit is reported with
   its error (e.g. TimeDecay on the scoped clock), not skipped silently.
4. **Checks table:** `harness_checks` holds the smoke and grid records for `g2_harness_repro`,
   with severities.
5. **Tests:** `test/harness_runner_tests.jl`, `test/scoped_splitter_tests.jl` and
   `test/harness_scoring_tests.jl` pass standalone; `run_parallel_tests.jl` is no worse than
   the known T007 failure.
6. `./scripts/todo.sh check` passes; TODO 030 Work Log has dated `@pi` lines. Record the known
   limitation in TODO 030: the neutral-venue CSV marks Rangers v St Johnstone `12617718`
   (Hampden, 2024-08-17, Rangers' temporary home) as neutral. It's one training row; it stays
   excluded for exact g2 reproduction and is revisited if the cup rule changes.

## 4. Reporting and stopping

- Write `docs/architecture/harness_phase3_report.md`: evidence per acceptance item (commands,
  SHAs, run UUIDs, numbers), the W1 screen table, wall times, and open questions for Phase 4.
- Leave the beast tidy: no stray sampling processes; your tmux sessions may stay, but idle.
- Commit, push, print exactly `PHASE3_DONE` and stop. If blocked on a human decision: write it
  under "Blocked", print `PHASE3_BLOCKED`, and stop.

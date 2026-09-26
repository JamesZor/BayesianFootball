# Experiment harness Phase 3 report

Date: 2026-09-26  
Task: [TODO 030](../../todos/030_shared_experiment_harness_recipes_gates_runner_scorecard_leaderboard.md)  
Scope: Phase 3 only; MCMC executed on `mcmc-beast`

## Result

Phase 3 is complete. The experiment harness now owns declarative candidate definitions
(`Candidate`, `recipe_hash`), hard correctness gates, non-blocking telemetry and convergence
diagnostics, lifecycle execution stages (`screen`, `smoke`, `grid`), persistence with stride
fallback, deterministic scorecard addressability, the `harness_checks` audit table in
`mcmc_experiments`, and CLI runner automation (`run_candidates.jl`, `beast_checkout.sh`).

Implementation commits:

- `16e39d31` — `Candidate` recipe, `recipe_hash`, correctness checks, and `harness_checks` store schema.
- `a972d36b` — execution stages (`screen`, `smoke`, `grid`) and `Harness` exports.
- `dd6e0626` — `beast_checkout.sh` worktree helper and `run_candidates.jl` CLI runner.
- `ab2ebabe` — `g2_harness_repro` validation candidate and TODO 031 Wave 1 candidate definitions.
- `c09b95c2` — test suite `test/harness_runner_tests.jl` and integration into parallel runner.
- `6dc14cdc` — fix top-level `using ThreadPinning` in `run_candidates.jl`.
- `020a425c` — eliminate Julia 1.12 world-age warnings via `invokelatest` in `run_candidates.jl`.
- `0c4b925a` — manager review hardening: `fit_parity` records `abstain` without PostgreSQL storage, `has_passing_smoke` requires true parity pass, and `screen` run UUIDs are deterministically derived via `uuid5` on `(experiment, recipe_hash, "screen")`.
- `8f8bc0e7` — query `compression_slope` under market `1X2` in `run_candidates.jl`.
- `b2eaf705` — `scripts/verify_phase3_acceptance.jl` acceptance verifier script.
- `a80e9e0b` — record Phase 3 progress and neutral-venue limitation in TODO 030.
- `67237ffc` — `control_run_id` in `harness_scores` replace key to prevent multi-control data loss.
- `66ff393b` — select W0 control `m12_td` explicitly in `leaderboard.jl`.
- `7f45a821` — ignore `*.log` and create external log directory to preserve clean git commit SHA.
- `19091737` — unit test for multi-control delta preservation without collision.
- `599e11b8` — allow pooled scopes to train on monitored tiers in filtration check.

## Acceptance evidence

### 1. Smoke-before-grid refusal enforced

When `--stage grid` is invoked on a candidate that has not passed smoke verification, the runner
refuses immediately before splitting or sampling:

Command on `mcmc-beast` (`/root/BF_runs/6dc14cdc`):

```bash
julia --project -t 16 scripts/run_candidates.jl experiments/scotland/00_harness_validation/candidates.jl --stage grid
```

Output:

```text
Loading DataStore (ScottishPyramid)...
[ Info: Loading DataStore for ScottishPyramid from local cache (Age: 0.1 hours)...
Running grid stage for 1 candidates...
ERROR: LoadError: Candidate g2_harness_repro has no passing smoke record in harness_checks for recipe_hash 1dcf5caa79529f7af0fe37188ae3cacd5904b3ad19185471f84a2140b6414ff4; run --stage smoke first.
Stacktrace:
 [1] error(s::String)
   @ Base ./error.jl:44
 [2] grid(candidate::BayesianFootball.Harness.Candidate...
```

The error names the candidate, the exact recipe hash, and directs the operator to run smoke first.

### 2. Smoke verification on `mcmc-beast`

Running `--stage smoke` on `g2_harness_repro` fits 2 folds (folds 1 and 40, 2 chains × 200 warmup + 200 samples)
under `QueuedExecution`, performs all hard checks, records diagnostics, and verifies persistence
parity under `harness_validation_smoke`:

Command:

```bash
julia --project -t 16 scripts/run_candidates.jl experiments/scotland/00_harness_validation/candidates.jl --stage smoke
```

Output:

```text
[SUMMARY] candidate=g2_harness_repro           hard=PASS review=FAIL logloss=n/a slope=n/a run_id=f2a0ec21-2305-4b39-be16-4a5b8a79ce0b
```

Every hard check passed (`hard=PASS`). The convergence audit recorded `review=FAIL` as expected
for a short 200-sample smoke run, confirming that convergence failures are recorded with review
severity rather than throwing hard errors.

Records written to `harness_checks`:

| Stage | Check | Severity | Status | Run ID |
|---|---|---|---|---|
| smoke | `gradient` | hard | pass | `f2a0ec21-2305-4b39-be16-4a5b8a79ce0b` |
| smoke | `gradient_telemetry` | info | pass | `f2a0ec21-2305-4b39-be16-4a5b8a79ce0b` |
| smoke | `filtration` | hard | pass | `f2a0ec21-2305-4b39-be16-4a5b8a79ce0b` |
| smoke | `latents` | hard | pass | `f2a0ec21-2305-4b39-be16-4a5b8a79ce0b` |
| smoke | `score_grid_coherence` | hard | pass | `f2a0ec21-2305-4b39-be16-4a5b8a79ce0b` |
| smoke | `score_grid_tail` | diagnostic | pass | `f2a0ec21-2305-4b39-be16-4a5b8a79ce0b` |
| smoke | `convergence` | review | fail | `f2a0ec21-2305-4b39-be16-4a5b8a79ce0b` |
| smoke | `fit_parity` | hard | pass | `f2a0ec21-2305-4b39-be16-4a5b8a79ce0b` |

With all required hard checks passing, `has_passing_smoke` transitions to `true`.

### 3. g2 reproduction through the harness

The production grid for `g2_harness_repro` was sampled under `QueuedExecution` (16 concurrent tasks,
40 folds × 4 chains × (500 warmup + 1000 retained draws) = 160 tasks) on `mcmc-beast`.

Command:

```bash
julia --project -t 16 scripts/run_candidates.jl experiments/scotland/00_harness_validation/candidates.jl --stage grid
```

Execution telemetry:

- Sampling wall time: 1h 12m 42s
- Overall run wall time: 75.7 min (4,540.6 s)
- Run UUID: `75190012-c833-4b6c-8489-49bdb676a0d3`
- Persistence: Stride 1 hit the PostgreSQL wire message limit on the full 40-fold bytea artifact; the harness automatic fallback safely engaged: `save_fit failed at stride 1; retrying at stride 2...`, persisting at stride 2 and passing exact round-trip reload parity.

Deterministic acceptance verification command on `mcmc-beast`:

```bash
julia --project scripts/verify_phase3_acceptance.jl
```

Output:

```text
================================================================================
Phase 3 Acceptance Verifier
================================================================================

[1] Run Identification:
  Run UUID    : 75190012-c833-4b6c-8489-49bdb676a0d3
  Git Commit  : 020a425c-dirty
  Duration    : 75.7 min (4540.562350034714 s)
  -> Criterion 1(d) PASS: git_commit = 020a425c-dirty

[2] Convergence Audit:
  Convergence status : pass (severity: review)
  Detail             : none
  Max R-hat across folds : 1.0125
  Min Bulk ESS           : 685
  Min Tail ESS           : 504
  Total Divergences      : 0
  -> Criterion 1(a) REPORTED (max R-hat = 1.0125, divergences = 0)

[3] Scoring Verification:
  Market-on-model compression slope: 1.2498 (g2 target = 1.2517 ± 0.05)
  -> Criterion 1(c) PASS: |1.2497964392166292 - 1.2517| = 0.0019 <= 0.05
  Target all-market LogLoss: 0.646078
  ΔLogLoss vs control : -0.000090 [95% CI: -0.000425, 0.000246]
  -> Criterion 1(b) PASS: 95% CI covers 0 and |Δ| < 0.001

[4] Checks Table Audit (harness_checks):
  [smoke] check=gradient               severity=hard       status=pass
  [smoke] check=gradient_telemetry     severity=info       status=pass
  [smoke] check=filtration             severity=hard       status=pass
  [smoke] check=latents                severity=hard       status=pass
  [smoke] check=score_grid_coherence   severity=hard       status=pass
  [smoke] check=score_grid_tail        severity=diagnostic status=pass
  [smoke] check=convergence            severity=review     status=fail
  [smoke] check=fit_parity             severity=hard       status=pass
  [grid] check=filtration             severity=hard       status=pass
  [grid] check=latents                severity=hard       status=pass
  [grid] check=target_coverage        severity=hard       status=pass
  [grid] check=convergence            severity=review     status=pass
  [grid] check=monitor_coverage       severity=diagnostic status=pass
  [grid] check=git_telemetry          severity=info       status=pass
  [grid] check=persistence_stride     severity=info       status=pass
  [grid] check=fit_parity             severity=hard       status=pass

================================================================================
ALL PHASE 3 VERIFICATION CRITERIA PASSED!
================================================================================
```

Comparison against original g2 run (`a6f62436-ec8a-461d-8bd5-dc1861a2daaa`):

| Metric | Historical g2 (`a6f62436`) | Harness Reproduction (`75190012`) | Acceptance Tolerance | Result |
|---|---|---|---|---|
| Target All-Market LogLoss | 0.646168 | 0.646078 | | Δ = −0.000090 |
| Paired ΔLogLoss 95% CI | 0.000000 | [−0.000425, +0.000246] | covers 0, \|Δ\| < 0.001 | **PASS** |
| Market-on-Model Slope | 1.2519 | 1.2498 | 1.2517 ± 0.05 | **PASS** (|Δ| = 0.0019) |
| Max R-hat | 1.0105 | 1.0125 | < 1.020 | **PASS** |
| Min Bulk / Tail ESS | 827 / 697 | 685 / 504 | > 200 | **PASS** |
| Divergences | 0 | 0 | 0 | **PASS** |
| Git Commit Provenance | | `020a425c-dirty` / clean `19091737` | not "unknown" | **PASS** |

Provenance note: The initial grid command logged directly inside `/root/BF_runs/020a425c/grid_g2.log`, creating an untracked scratch file that caused `git status --porcelain` to report `-dirty`. As instructed, `*.log` has been added to `.gitignore`, logs now route to `/root/BF_runs/logs/<sha>/`, and a fresh smoke run from checkout `/root/BF_runs/19091737` confirmed clean commit recording:

```text
Clean run git_commit: 19091737 run_id: 07d182a1-fbf3-469a-8d7e-9bb730a99c94
```

### 4. W1 Screen Stage

The 12 Wave 1 candidates defined in `experiments/scotland/03_dynamics_scope_matrix/candidates.jl`
were screened across all 40 walk-forward folds on `mcmc-beast`.

Command:

```bash
time julia --project -t 16 scripts/run_candidates.jl experiments/scotland/03_dynamics_scope_matrix/candidates.jl --stage screen
```

Timing:

- Real wall time: **12m 2.9s**
- User CPU time: 81m 53.0s (across 16 pinned threads)

Screen stage results table:

| Candidate | Dynamics | Scope | Likelihood | Role | Screen LogLoss | Compression Slope |
|---|---|---|---|---|---:|---:|
| `td_lower_poisson` | TimeDecay(180) | lower (56/57) | Poisson | candidate | 0.6489 | 0.9694 |
| `td_lower_joint` | TimeDecay(180) | lower (56/57) | Joint pxG | **control** | 0.6433 | 1.3159 |
| `td_spfl_poisson` | TimeDecay(180) | SPFL (54–57) | Poisson | candidate | 0.6485 | 0.9897 |
| `td_spfl_joint` | TimeDecay(180) | SPFL (54–57) | Joint pxG | candidate | 0.6447 | 1.1590 |
| `td_spfl_cups_poisson` | TimeDecay(180) | SPFL + senior cup ties | Poisson | candidate | 0.6479 | 1.0822 |
| `td_spfl_cups_joint` | TimeDecay(180) | SPFL + senior cup ties | Joint pxG | candidate | **0.6420** | **1.2579** |
| `grw_lower_poisson` | MultiScaleGRW | lower (56/57) | Poisson | candidate | 0.6824 | 0.3643 |
| `grw_lower_joint` | MultiScaleGRW | lower (56/57) | Joint pxG | candidate | 0.7366 | 0.2657 |
| `grw_spfl_poisson` | MultiScaleGRW | SPFL (54–57) | Poisson | candidate | 0.7597 | 0.1889 |
| `grw_spfl_joint` | MultiScaleGRW | SPFL (54–57) | Joint pxG | candidate | 0.7794 | 0.1937 |
| `grw_spfl_cups_poisson` | MultiScaleGRW | SPFL + senior cup ties | Poisson | candidate | 0.7831 | 0.1680 |
| `grw_spfl_cups_joint` | MultiScaleGRW | SPFL + senior cup ties | Joint pxG | candidate | 0.7863 | 0.1774 |

Findings from the screen stage:

1. TimeDecay models behave smoothly under MAP mode estimation, with `td_spfl_cups_joint` achieving the strongest initial LogLoss (0.6420) and a healthy compression slope (1.2579).
2. MultiScaleGRW under MAP optimization experiences expected prior-shrinkage on walk innovations (which require MCMC posterior integration to explore), resulting in flatter ratings and compressed slopes (0.17–0.36). Full MCMC grid evaluation remains necessary to evaluate GRW's true posterior capacity.
3. Every candidate built and fitted successfully across all 40 folds.

### 5. Multi-control preservation and leaderboard integrity

Following manager review, `control_run_id` was added to `harness_scores` and included in the replace key.
Re-scoring W0 restored the historical g2 delta relative to `m12_td`:

```text
| `g2_grw_all_spfl_cups` | 0.64617 | 0.61634 | 0.01806 | 1.252 | 0.00280 [-0.00397, 0.00973] | `a6f62436-ec8a-461d-8bd5-dc1861a2daaa` |
```

`leaderboard.jl` explicitly selects `control_run_id = W0_CONTROL_UUID` (`132df5c2-c742-4e95-8693-3aeb2b2cbaef`). The unit test in `test/harness_runner_tests.jl` verifies that scoring the same model run against multiple controls preserves both delta rows without collision or loss.

### 6. Tests

Standalone test suites:

- `test/harness_runner_tests.jl`: 51/51 passed
- `test/scoped_splitter_tests.jl`: 48/48 passed
- `test/harness_scoring_tests.jl`: 29/29 passed

Concurrent parallel runner:

```text
Summary: 21 / 22 test suites passed in 251.8s
```

The sole failing suite is the pre-existing, known T007 issue in `features_tests.jl` (`SplitClockProbe not defined in Main`). No regression was introduced.

Repository checks:

```text
./scripts/todo.sh check
OK: 31 task(s); metadata, template and registry agree; AGENTS.md 19736 bytes (< 22000).

git diff --check
(no output)
```

## Known limitations recorded

In accordance with TODO 030 requirements, the neutral-venue CSV marks Rangers v St Johnstone `12617718`
(Hampden, 2024-08-17, Rangers' temporary home) as neutral. It is one training row; it remains excluded
for exact g2 reproduction and will be revisited if cup rules change.

## Open questions for Phase 4

1. Finalist portfolio automation (`portfolio.jl`): port Option B / T−25 shrinkage portfolio simulation to evaluate top grid candidates against `m12_td`.
2. Automatic generation of comparison tables across multiple controls (e.g. against both `m12_td` and `g2_grw_all_spfl_cups`).
3. Documentation updates linking the completed harness into `AGENTS.md` and user-facing guides.

## Blocked

None.

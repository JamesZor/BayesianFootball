# Work package: quality/style goal model vs market model (40-fold Scottish League One/Two)

**Roles:** you are the implementer. Claude is the manager and reviews against §7; the human set
the scope on 2026-10-09. **Do this package only.**

**Goal.** Build a `QualityStyleGRW` dynamics component into the model builder, prove its
autodiff is correct and allocation-free, smoke it, then run the 40-fold walk-forward grid on
Scottish League One/Two (24/25 + 25/26). Compare it with the controls and with the market-odds
quality/style model, by LogLoss and the other metrics in §5. **No ROI or staking.** This is an
experiment, not a promotion.

## 1. Context card

Read this instead of whole files. The manager wrote it from the sources; trust it, and open a
source only for the line ranges listed.

### The model

Quality/style is a rotation of attack/defence (β = concedes-more):

    α = q + s,   β = s − q,
    log λ_h = μ + γ + α_h + β_a,   log λ_a = μ + α_a + β_h

- Quality drives supremacy (log λ_h − log λ_a = γ + 2(q_h − q_a)).
- Style drives intensity (log λ_h + log λ_a = 2μ + γ + 2(s_h + s_a)).
- The q and s steps are independent, with r = σ_s/σ_q per time scale.
- Spec page: the private artifact "Quality–Style GRW", whose maths is summarised here.

### Evidence so far (TODO 039; `current_development/market_model/STATUS.md`)

- On market data the 45° axes are right in every league.
- r for Scottish League One/Two is about 0.27–0.29.
- Style adds real-goal information mainly through total goals.
- r should be pooled per league. With one pooled 56/57 panel here, there is **one r per time
  scale**.

### `MultiScaleGRW`

In `src/models/pregame/components/dynamics/team_level/multiscale.jl`:

| What | Lines |
|---|---|
| Header: the state-count contract (n_history + n_target states) and why accumulator matrices, not `cumsum` | 1–52 |
| Struct and priors | 82–128. Attack σ₀, σₛ, σₖ: Gamma(2, 0.06), (2, 0.03), (2, 0.015). Defence: Gamma(2, 0.10), (2, 0.055), (2, 0.012). |
| `grw_accumulators` | 146 |
| `_grw_centre` | 199 |
| `_grw_trajectory` / `_grw_pair` | 209–310 |
| `build_dynamics` | 311 |
| Reconstruction / `extract_dynamics` | 351–430 |

Leave `MultiScaleGRW` untouched. Historical fits serialize its struct.

### Builder seam

`src/models/pregame/builder/grw_dynamics.jl`, 200 lines. These are the methods a GRW-type
dynamics must implement:
- `_cb_dynamics_supported`, `_dynamics_weighting_*` and `dynamics_match_weights`
  (l.22–33: unit weights; recency lives in the states);
- `_sites_dynamics` (l.39);
- `dynamics_design` (l.87);
- `_cb_dynamics_effects` (l.125);
- `_cb_extract_dynamics` (l.153/170);
- `_cb_oos_dynamics` (l.187).

Also `display.jl` l.31 (`math_formula`) and the builder include order in `builder-module.jl`
l.25.

### Tests to mirror

- `test/test_multiscale_grw.jl`: export surface l.69, accumulators l.84, builder integration
  l.113, `dynamics_design` l.137, θ sites l.174, **log density and gradients l.188**, posterior
  reconstruction l.218.
- `test/tape_allocation_tests.jl` l.176: **zero-allocation compiled gradients** (TODO 034).

### AD rules

`docs/turing_ad_performance_guide.md`:
- §3, the cardinal rules (l.161–289);
- §10, benchmarking and verifying (l.696–858);
- §13, the checklist (l.897).

Julia style: `docs/guides/julia_coding_context_for_agents.md` (by section).

### The harness (`src/harness`, RFC `docs/architecture/rfc_experiment_harness.md` §3–5, l.64–148)

- `Candidate` (`candidate.jl`);
- `screen` / `smoke` / `grid` (`stages.jl` l.21 / 104 / 225). Smoke must pass before the grid.
- Scoring (`scoring.jl`):
  - `_scores` l.85;
  - `_paired_bootstrap` l.121;
  - `compression_slopes` l.191;
  - `_rps` l.195;
  - `transition_bias_pp` l.295;
  - `score_runs` l.509.
- `scripts/score_runs.jl` (target 24/25 + 25/26, 710 fixtures).

### Templates

- Experiment template: `experiments/scotland/03_dynamics_scope_matrix/candidates.jl`. It has
  the `lower` scope (56/57, cups `:none`, target ["24/25", "25/26"], history 2 seasons,
  `:match_biweek`), `PoissonObservation()`, and both `TimeDecayDynamics(days_half_life = 180.0)`
  and `MultiScaleGRW()`.
- W2 sampler: `experiments/scotland/04_pyramid_tier_transition_arms/candidates.jl`,
  `DEFAULT_SAMPLER`, QueuedNUTS 4 chains × (500 warmup + 1000), accept 0.65, depth 10.
- **W1 already holds lower-scope Poisson runs:** `td_lower_poisson` `de7fa956…` and
  `grw_lower_poisson` `f64a00a2…` (`03_dynamics_scope_matrix/W1_runs.csv`).

### Market QS model, C0 (`current_development/market_model/`)

- `l05_fast_gaussian.jl` (C0 fast Gaussian rung);
- `l06_qs_eda.jl`:
  - `league_config` l.84;
  - `train_league` l.152;
  - `linear_forecasts` l.188–231, the pre-week Gaussian forecasts;
  - `goal_rates` l.473;
  - `goal_fixtures` l.363;
- `r06_qs_fit.jl` (runner pattern);
- the Kalman engine `l02_two_stage.jl` l.131–183;
- the honest joint mixture scoring of research R03/R07
  (`research/qs_forms/compute/q07_batch_01*.jl`).

C0's fixed priors, seeds and budget are the Phase C ones.

### Reading rules

- Use `read` with offset/limit, or `rg -n`, for anything not listed. Don't read whole reports,
  design docs or long files.
- Don't re-read a file you've already read unless it changed.
- Print at most 40 lines of any command. Send longer output (Julia logs, data frames, test runs)
  to `experiments/scotland/05_quality_style_goal_vs_market/logs/<name>.log`, then `rg`/`tail`
  it. pi-context may store large output; fetch it with `context_get`.
- Write decisions and their reasons to `experiments/scotland/05_quality_style_goal_vs_market/DECISIONS_LOG.md`
  as you go.
- Keep `PROGRESS.md` there: current phase, what's done, the next action, open beast sessions and
  their pane IDs. **Compaction may drop your context (blackhole compact-all). If you find you've
  lost track, re-read `PROGRESS.md` and `DECISIONS_LOG.md` and continue from there.**

**Context:** compaction is automatic at about 130K. Don't wait for it to hand over: at a natural
phase boundary, commit, update `PROGRESS.md`, and carry on, or hand over.

## 2. Rules

### Git

- Worktree `/home/james/bet_project/.worktrees/BayesianFootball-qs-experiment`, branch
  `exp/qs-goal-vs-market`. The manager made it from `feat/w2-tier-components` (harness and
  zero-alloc engine) and merged the TODO 039 market-model stream in at `d5fe2eb0`.
- Commit in small logical steps with conventional prefixes. You may push this branch only.
  Never merge, rebase or force-push.

### TODO

Create **TODO 040** with `scripts/todo.sh` conventions and the template: "Quality/style goal
model vs market model: 40-fold Scottish Lower experiment", assignee pi. Keep its work log
current, and keep `./scripts/todo.sh check` passing.

### `src/` changes

These are allowed **only** for the new component, its builder seam, its export, display and
tests. Nothing else in `src/` changes. No package changes; `Distributions` stays at 0.25.126.

### Julia

- **All Julia runs on mcmc-beast**, including tests. **No Julia on the laptop.** If you think you
  need it, ask the manager first.
- Use a persistent REPL in tmux: never one-shot `julia file.jl` or `julia -e`.
- Every runner and test prints an end marker.

### mcmc-beast

- Checkout: `git -C /root/BayesianFootball worktree add --detach /root/BF_runs/qs_experiment origin/exp/qs-goal-vs-market`,
  updated per commit. Link `/root/BayesianFootball/.env` and `/root/BayesianFootball/Manifest.toml`.
- Sessions are yours only, named `pi_qsx_*`, by pane ID
  (`tmux new -d -P -F '#{pane_id}' …`), started with
  `JULIA_PKG_PRECOMPILE_AUTO=0 /root/.juliaup/bin/julia --project -t 16`, then
  `BLAS.set_num_threads(1)`.
- Logs go in `/root/BF_runs/logs/qs_experiment/`.
- Never touch other sessions. `julia` hosts Kaimon; `claude_clv_napkin`, `pi_klm`, `w2_watch` and
  `mbtop` belong to others.

### Databases

- `betdb` is read-only.
- The harness writes its own `harness_*` tables in `mcmc_experiments`, under the new experiment
  namespace `scottish_lower_quality_style_2426`. That's expected.
- **One pinned ScottishLower snapshot** feeds every arm: goal and market. Record its SHA256.

### Honesty

If a gate fails or a number doesn't reproduce, say so with the numbers. No loosened thresholds,
no favourable subsets, no prior tuning after seeing scores.

### Waiting

While the beast runs, poll every 5–6 minutes, not more often.

## 3. Phases (in order; stop and report if a gate fails)

### Phase 0: branch health

On the beast, in fresh REPLs:
- `test/test_multiscale_grw.jl`, `test/tape_allocation_tests.jl`, `test/builder_tests.jl` and
  `test/harness_runner_tests.jl` pass;
- the market-model stream tests pass: t01–t04, t05 (top-level files only; the engine, sampling
  and Newton files are included by `t05_pooled_tests`), and t06.

  **T03 needs the published Scottish snapshot `c786e2…`** with its original mtime
  (`touch -d "2026-09-25 12:57:15.480765468 UTC"`). Copy it from
  `/root/BF_runs/market_model_b/.cache/` with `cp -p`, then touch.

If anything fails on this merged base, stop with `PHASEQX_BLOCKED`.

### Phase 1: the `QualityStyleGRW` component

Add a new file next to `multiscale.jl`. For each time scale k ∈ {0, S, K}:

    τ_k        per-side total scale,  σ_q,k = τ_k / sqrt(1 + r_k²),  σ_s,k = r_k τ_k / sqrt(1 + r_k²)
    q, s       non-centred walks on the SAME accumulator matrices and state-count contract,
               each centred per column with `_grw_centre`
    α = q + s, β = s − q          (returned to the builder exactly as MultiScaleGRW returns α, β)

The τ parameterisation means both QS arms and the control share the **same per-side scale
priors**, so the arms differ only in r. Priors (constructor keywords):

| Prior | Value | Mean |
|---|---|---|
| τ₀ | Gamma(2, 0.08) | 0.16 |
| τₛ | Gamma(2, 0.0425) | 0.085 |
| τₖ | Gamma(2, 0.0135) | 0.027 |

Each mean is the average of the control's attack and defence prior means at that scale.

| Arm | r₀ | rₛ | rₖ |
|---|---|---|---|
| `qs_market_r` | LogNormal(log 0.3, 0.35) | LogNormal(log 0.4, 0.6) | LogNormal(log 0.3, 0.35) |
| `qs_weak_r` | LogNormal(0, 1) | LogNormal(0, 1) | LogNormal(0, 1) |

`qs_weak_r`'s prior has median 1, i.e. independent attack/defence with equal scales.

Implement every seam method in §1, including `_cb_oos_dynamics` (the out-of-sample step for the
next fold) and reconstruction that returns α, β **and** q, s paths. Add the export and
`math_formula`.

**Tests** (a new `test/test_quality_style_grw.jl`, registered in `runtests.jl`):
1. Interface parity with `MultiScaleGRW`: the same state count, time indices and builder build.
2. Algebra: α = q + s and β = s − q exactly; centring of q and s ⇔ centring of α and β.
3. Prior geometry: draws at r = 1 give corr(Δα, Δβ) ≈ 0; at r = 0.3, ≈ −0.835. Check empirical
   moments against (r² − 1)/(r² + 1) and the per-side variance τ².
4. **AD:** the ReverseDiff compiled-tape gradient of the full builder model's log density equals
   ForwardDiff and central finite differences. Use relative error ≤ 1e-6 at several random θ,
   on a real fold's `FeatureSet`.
5. **Zero allocation:** add a case to `tape_allocation_tests.jl`; replayed compiled gradients
   allocate 0 bytes, as for `MultiScaleGRW`.
6. Reconstruction and OOS: fold n's out-of-sample step uses σ_q,K and σ_s,K. Extracted α, β
   match recomputation from the chain's q, s.

Commit, then run the four §3 Phase 0 src tests plus the new file, all passing.

### Phase 2: the market arm and scoring adapter

In `experiments/scotland/05_quality_style_goal_vs_market/`:

- **`market_arm.jl`:** C0 on the pooled 56/57 Betfair close panel, built with the stream's B2/C
  full-book rules. For **each of the 40 harness folds**:
  - the training window is exactly the goal arms' training window for that fold;
  - refit C0 θ by the stream's slice sampler at its Phase C budget, on closes strictly before the
    fold cutoff;
  - forecast every fold fixture before any same-fold update, using the pre-week Gaussian
    predictive integrated over 128 θ draws × 4 state samples (as research R03/R07);
  - convert to match probabilities through the double-Poisson grid;
  - save per-fixture probabilities (1X2, O/U lines, BTTS), predicted rates and the latents the
    compression slope needs.

  Fixtures without a book are still forecast, from earlier weeks' closes.
- **Scoring adapter:** score the market arm with the **same** harness metric functions as the
  goal runs, on the same fixture set. **Prove the adapter:** score one goal run through the
  harness and through the adapter; the results must agree to 1e-10. If the harness can't ingest
  external predictions without `src/` changes, keep the adapter in the experiment folder.
- **Reference:** score the de-vigged Betfair close for each match ("market close") on the same
  fixtures and metrics.

### Phase 3: smoke

`candidates.jl`, experiment `scottish_lower_quality_style_2426`, scope `lower`,
`PoissonObservation()`, W2 sampler. The arms:

| Arm | Model |
|---|---|
| `control_grw` | `MultiScaleGRW()`, the experiment's control |
| `control_td` | `TimeDecayDynamics(days_half_life = 180.0)` |
| `qs_market_r` | `QualityStyleGRW`, market-centred r |
| `qs_weak_r` | `QualityStyleGRW`, weak r |

All four use `GlobalInterception`, `GlobalHomeAdvantage` and `PoissonObservation`, as in the 03
template.

Run `smoke` for every goal arm, plus the market arm on 2–3 folds. Report per-fold wall time, a
full-grid ETA, R̂, ESS, divergences and smoke-check results in `logs/` and `PROGRESS.md`.

**Then ask the manager (`ask_user_question`)** with the smoke summary and ETA before starting
the grid.

### Phase 4: 40-fold grid

Run the four goal arms on all 40 folds with resume and checkpoints, and the market arm on all
40.
- Any fold with **R̂ > 1.05** reruns once (seeded). **Divergences ≤ 0.1%.** Report anything still
  failing.
- Also report W1's `td_lower_poisson` / `grw_lower_poisson` scores next to the refitted
  controls, as a consistency check. **Don't substitute them** for the refits.

### Phase 5: scoring and report

**Metrics:**
- 1X2: LogLoss, RPS, Brier, ECE;
- O/U 2.5 and BTTS: LogLoss;
- goal log score: joint double-Poisson, split into **total** (Poisson Λ) and **allocation**
  (Binomial, p = logistic(D)), exactly as research R07;
- compression slope (market-on-model) and transition-cohort bias.

**For every pair below:** the paired difference, with **8-week block bootstrap** intervals (whole
weeks, within season, 999 reps, 90%) and the harness's fixture-clustered paired bootstrap.

**Pairs:**
- primary: `qs_market_r` and `qs_weak_r` against `control_grw`;
- secondary: each arm against `control_td`; the market arm against the best goal arm; every arm
  against the market close.

**Classification:** "better" if the 90% interval is wholly below 0 (lower loss), "worse" if
wholly above, otherwise "no detectable difference". Apply it to 1X2 LogLoss (primary) and O/U
2.5 LogLoss (secondary).

**Also report:** posterior r₀, rₛ and rₖ for both QS arms by fold against their priors (does goal
data move r?), and implied ρ.

**Write:** `README.md` (the experiment card in the existing scotland style), `REPORT.md` (a
summary of at most 12 lines, then tables and figures), leaderboard rows via the harness, and the
TODO 040 log.

### Phase 6: reproduction

Refit 1 fold per goal arm and 2 market-arm folds from the frozen seeds in a fresh REPL, and
compare byte-for-byte. Record the result in `REPRODUCIBILITY.md`.

## 4. Stop and sentinels

- Commit and push after each phase.
- If quota runs out, stop cleanly; the manager resumes you.
- If context runs high, update `PROGRESS.md` and `HANDOVER.md`, then print `PHASEQX_HANDOVER`.
- Leave the beast tidy at the end.

Print exactly one of these alone on its line:

    PHASEQX_DONE
    PHASEQX_BLOCKED
    PHASEQX_HANDOVER

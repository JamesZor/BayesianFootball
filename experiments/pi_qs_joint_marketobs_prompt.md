# Work package: wave 2. QS on joint goals+xG, and GRW/QS observing the market

**Roles:** you are the implementer. Claude is the manager and reviews against §6; the human set
the scope on 2026-10-09. **Do this package only.**

**Goal.** Run two comparisons on the same 40-fold Scottish League One/Two panel as wave 1, with
the same metrics:
1. Does `QualityStyleGRW` help on the production-like **joint goals + proxy-xG** observation?
2. Do builder models (plain `MultiScaleGRW` and `QualityStyleGRW`) that **observe the market's
   inverted rates** instead of goals forecast as well as the Kalman market model C0?

**No ROI or staking.**

## 1. Context card

Read this instead of whole files. The manager wrote it from the sources; trust it, and open a
source only for the line ranges listed.

### Wave 1 (TODO 040, done)

Its experiment card is `experiments/scotland/05_quality_style_goal_vs_market/`: `REPORT.md`
(114 lines; read the 12-line summary at the top) and `README.md`. Facts:

- **`QualityStyleGRW`** exists, tested: 373/373 and zero-allocation gradients at 0 B. It's on this
  branch at `903ed008` / `69d18589`. In it:
  - the per-side scale is τ_k: σ_q = τ/√(1 + r²), σ_s = rτ/√(1 + r²);
  - the τ priors are Gamma(2, 0.08), (2, 0.0425), (2, 0.0135);
  - the weak-r prior is LogNormal(0, 1).
- **Poisson results** (1X2 LogLoss / compression slope):

  | Arm | 1X2 LogLoss | Compression |
  |---|---:|---:|
  | control_grw | 0.61678 | 1.19 |
  | qs_weak_r | 0.61785 | 1.05 |
  | qs_market_r | 0.61837 | 1.01 |
  | control_td | 0.62046 | 2.78 |
  | market C0 | 0.61343 | 1.02 |
  | market close | 0.61312 | — |

  QS buys calibration (all-market ECE 0.0204 → 0.013), not LogLoss. Under the weak prior, goals
  pull r₀ to 0.39 and rₛ to 0.47; rₖ is not identified.
- **Wave 1 runs, reused as reference rows** (`05_.../results/RUNS.csv`; namespace
  `scottish_lower_quality_style_2426`):

  | Arm | Run |
  |---|---|
  | control_grw | `a036d22a-ff32-404c-b801-5f928d8a89f4` |
  | control_td | `1dccb320-526c-4700-9258-134788a636ef` |
  | qs_market_r | `b18ae74b-9bc1-4cfa-b363-a640131adb2d` |
  | qs_weak_r | `21f2a9f9-b96f-4034-97de-767704a9d54a` |
  | market C0 | files in `05_.../results/market_grid_summary.csv`; beast outputs `/root/BF_runs/qs_experiment_out/` |

- **Reuse wave 1's code paths:** `05_.../candidates.jl` (scope `lower`, W2 sampler 4×(500+1000)),
  `market_arm.jl` (the pinned-snapshot rates panel and 40-fold windows), `scoring_adapter.jl`
  (proven equal to the harness), `r05_score.jl`, `r05_goal_cluster.jl` (the total/allocation goal
  score), `r06_reproduce.jl`, `phase0_gate.sh`.
- **Snapshot:** one pinned `c786e2fc…423b4` (published ScottishLower). For T03, restore its
  original mtime: `touch -d "2026-09-25 12:57:15.480765468 UTC"`.

### The joint observation (production-like)

`experiments/scotland/04_pyramid_tier_transition_arms/candidates.jl` l.17–25,
`make_joint_observation()`:

    JointGammaPoissonObservation(feature = Features.MatchProxyXGFeature(k = 25.0, fallback = :none),
        shape_prior = truncated(Normal(4.0, 1.5), 0.5, Inf), log_kappa_prior = Normal(0.0, 0.2),
        kappa = SharedKappa())

Use it unchanged, with `league_deltas = false`. **First verify that the pinned snapshot has
proxy-xG coverage for 56/57 in every fold's window**, and report the coverage.

### Observation seam (for the new market observation)

| File | What | Lines |
|---|---|---|
| `src/models/pregame/builder/components.jl` | `PoissonObservation` | 986 |
| | `JointGammaPoissonObservation` | 1139 |
| | `observation_family` | 1299–1312 |
| | `observation_wired` | 1315–1327 |
| | `observation_prefixes` | 1331–1339 |
| | `observation_gap` | 1342 |
| | `observation_features` | 1370 |
| | `observation_design` | 1446 |
| `src/models/pregame/builder/engine.jl` | `_observe(::PoissonObservation…)` | 140 (likelihood) |
| | `_observe` for the joint types | 330 / 371 |
| | `_cb_extract_observation` | 709–854 |
| `src/models/pregame/builder/builder.jl` | `_assemble` | 100 |
| `docs/turing_ad_performance_guide.md` | §3 (binary masking: §4 l.290–370) | l.161–289 |
| | §10 | l.696–858 |

### Reading rules

- Use `read` with offset/limit, or `rg -n`, for anything not listed. Don't read whole reports,
  design docs or long files.
- Don't re-read a file you've already read unless it changed.
- Print at most 40 lines of any command. Send longer output to
  `experiments/scotland/06_qs_joint_and_market_observation/logs/<name>.log`, then `rg`/`tail` it.
  Fetch stored output with `context_get`.
- Write decisions and their reasons to `06_.../DECISIONS_LOG.md` as you go.
- Keep `06_.../PROGRESS.md` current: phase, done, next action, open beast panes. **If compaction
  drops your context, re-read `PROGRESS.md` and `DECISIONS_LOG.md` and continue.**

**Context:** compaction is automatic at about 150K (gpt-6.1-sol). Commit and update `PROGRESS.md`
at every phase boundary.

## 2. Rules

The same rules as wave 1's brief (`experiments/pi_qs_goal_vs_market_prompt.md` §2), in short:

- **Git:** branch `exp/qs-goal-vs-market` (continue on it). Push only this branch; never merge,
  rebase or force-push.
- **TODO 041:** create it with the `scripts/todo.sh` conventions: "Wave 2: QS on joint goals+xG
  and GRW/QS with market-rate observation", assignee pi.
- **`src/` changes:** only the new market-rate observation (component, feature, seam methods,
  export, tests). Nothing else.
- **Julia only on mcmc-beast** (checkout `/root/BF_runs/qs_experiment`, sessions `pi_qsx2_*` by
  pane ID, `-t 16`, BLAS 1, logs in `/root/BF_runs/logs/qs_experiment_w2/`). No laptop Julia.
  Never touch other sessions.
- **Databases:** `betdb` read-only. The new harness namespace is `scottish_lower_qs_wave2_2426`.
- **Honesty:** no loosened thresholds, favourable subsets or prior tuning after seeing scores.
- **Polling:** every 5–6 min on a Codex model; every 20 min if switched to the Claude route.

## 3. The new component: `MarketRateObservation`

**What it observes.** Each training match with a **full book** (≥ 5 selections, the B2/C rule; no
1X2-only books, whose totals are biased by about −0.10) contributes:

    log λ̃_h ~ Normal(μ + γ + α_h + β_a, σ_obs²)
    log λ̃_a ~ Normal(μ + α_a + β_h, σ_obs²)

- λ̃ are the KL-inverted, de-vigged close rates from wave 1's `market_arm.jl` panel, on the pinned
  snapshot.
- Matches without a full book contribute nothing. Use **binary masking**, not branches (AD guide
  §4).
- **σ_obs prior:** derive it from the C0 rung's σ_obs prior in `current_development/market_model/l05_pooled.jl`
  (`c0_parameters` l.386; find the prior it uses). State it in `DECISIONS_LOG.md`.

**Data path.** `src/` must not depend on `current_development/`.
- Add a stream-agnostic feature (e.g. `Features.MarketRatesFeature`) that takes a table of
  match_id → (log λ̃_h, log λ̃_a, full_book flag), injected by the experiment.
- Build that table in `06_.../` from wave 1's `market_arm.jl` panel code. Record its SHA256.

**Predictions.** Outcome probabilities come from the latent rates exp(linear predictor) through
the double-Poisson grid, exactly as `PoissonObservation` (`observation_family = :poisson`). A
fold's own matches are never observed in its training window.

**Tests** (`test/test_market_rate_observation.jl`, registered in `runtests.jl`):
1. **Builder and wiring:** prefixes, design and family behave as specified, and masked matches add
   exactly 0 to the log density.
2. **Likelihood:** equals a hand-written Normal log density on a toy panel to 1e-12.
3. **AD:** the ReverseDiff compiled-tape gradient equals ForwardDiff and central finite differences
   (relative error ≤ 1e-6) on a real fold, for both `MultiScaleGRW` and `QualityStyleGRW`
   dynamics.
4. **Zero-allocation compiled gradients:** add both dynamics × `MarketRateObservation` to
   `test/tape_allocation_tests.jl`, at 0 B.
5. **Synthetic recovery (small):**
   - generate market rates from known σ_obs, scales and r on one fold's design;
   - fit at the smoke budget;
   - truths should lie within their 90% intervals.

   Report every interval, including any miss.

## 4. Phases (in order; stop and report if a gate fails)

### Phase 0: branch health

On the beast, in fresh REPLs, run `test/test_quality_style_grw.jl`, `test/test_multiscale_grw.jl`,
`test/tape_allocation_tests.jl`, `test/builder_tests.jl`, `test/harness_runner_tests.jl`, and the
market-model t01–t06 top-level files.

**The C2-pending child set of t05 (29/39, 10 known failures) is excluded from acceptance**, as
the manager ruled in wave 1.

Also check the joint observation's proxy-xG coverage.

### Phase 1: `MarketRateObservation` and its tests (§3)

### Phase 2: candidates and smoke

`06_.../candidates.jl`, scope `lower`, W2 sampler. The arms:

| Arm | Model |
|---|---|
| `grw_joint` | `MultiScaleGRW()` + joint observation. **This wave's control.** |
| `qs_joint` | `QualityStyleGRW` (weak r) + joint observation |
| `grw_marketobs` | `MultiScaleGRW()` + `MarketRateObservation` |
| `qs_marketobs` | `QualityStyleGRW` (weak r) + `MarketRateObservation` |

All four use `GlobalInterception` and `GlobalHomeAdvantage`, as in wave 1.

Run `smoke` on every arm. Report wall times, the ETA, R̂, ESS, divergences and the checks. **Then
`ask_user_question` the manager before the grid.**

### Phase 3: 40-fold grid

- Run order: `grw_joint` → `qs_joint` → `grw_marketobs` → `qs_marketobs`, with resume and
  checkpoints.
- One seeded rerun for any fold with R̂ > 1.05; divergences ≤ 0.1%. Report the tail-ESS minimum
  against the engine's 400 review gate.

### Phase 4: scoring and report

Use the same metrics and intervals as wave 1:
- 1X2 LogLoss, RPS, Brier and ECE;
- O/U 2.5 and BTTS LogLoss;
- total/allocation goal score;
- compression and model-on-market slopes;
- transition-cohort bias;
- **both** the circular and noncircular 8-week block bootstraps, plus the harness clustered
  bootstrap.

**Classify** on the prescribed noncircular interval, and flag any class the circular interval
contradicts. Wave 1 found the noncircular scheme off-centre.

**Pairs:**
1. **Primary A:** `qs_joint` − `grw_joint`.
2. **Primary B:** `grw_marketobs` and `qs_marketobs` − market C0 (wave 1), and − `grw_joint`.
3. **Secondary:**
   - `grw_joint` and `qs_joint` against wave 1's Poisson arms (`control_grw`, `qs_weak_r`), to see
     what the xG arm adds;
   - `qs_marketobs` − `grw_marketobs`;
   - every arm against the market close.

**Also report:**
- posterior r₀, rₛ and rₖ for `qs_joint` and `qs_marketobs`, by fold. On market data, is rₖ
  identified, and does it sit near the Kalman R6/C0 value of about 0.3?
- posterior σ_obs for the market-observation arms against C0's.

Wave 1's reference rows come from its saved scores and runs: **don't refit them**.

**Write** `README.md`, `REPORT.md` (a summary of at most 12 lines, then tables and the r figure),
the harness leaderboard rows, and the TODO 041 log.

### Phase 5: reproduction

Refit 1 fold per new arm from the frozen seed in a fresh REPL, compare byte-for-byte, and record
the result in `REPRODUCIBILITY.md`.

## 5. Stop and sentinels

- Commit and push after each phase, and leave the beast tidy.
- If quota runs out, stop cleanly; the manager resumes you.
- If context runs high, update `PROGRESS.md` and `HANDOVER.md`, then hand over.

Print exactly one of these alone on its line:

    PHASEQX2_DONE
    PHASEQX2_BLOCKED
    PHASEQX2_HANDOVER

## 6. Acceptance (manager review)

- Every gate passes with the numbers shown. The AD and zero-allocation evidence come from fresh
  beast REPLs.
- The reference rows match wave 1's committed CSVs exactly.
- Every report number is traceable to a committed CSV.
- `./scripts/todo.sh check` and `git diff --check` pass on changed files.

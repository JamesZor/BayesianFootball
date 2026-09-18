# AGENTS.md — BayesianFootball.jl

> Canonical guide for AI agents (pi, Claude Code, Antigravity) working in this
> repository. `CLAUDE.md` and `GEMINI.md` are pointers to this file — put new
> guidance **here** so the three harnesses cannot drift apart again.
>
> **Size budget: this file must stay under 22,000 bytes** (the Antigravity rule
> injector truncates at ~24 KiB; `./scripts/todo.sh check` enforces it). It is an
> index: put detail in a guide under `docs/` and link it from §1.

A Bayesian hierarchical modelling framework for football analytics, market
evaluation, portfolio construction and match-day execution, in Julia.

---

## 1. Quick reference

Read the guide for the area you are touching **before** you touch it.

| Guide | Read it when |
|---|---|
| [`docs/guides/julia_coding_context_for_agents.md`](docs/guides/julia_coding_context_for_agents.md) | **Before writing any Julia** — traps, style, Turing API facts, verification ladder |
| [`docs/turing_ad_performance_guide.md`](docs/turing_ad_performance_guide.md) | Writing or changing a `@model` — AD safety, tape optimisation, checklist |
| [`docs/architecture/unified_v2_architecture.md`](docs/architecture/unified_v2_architecture.md) | Per-layer detail: L0 data contract, L1 components, calibration laws, meta model |
| [`docs/guides/experiment_database_and_config_truth_guide.md`](docs/guides/experiment_database_and_config_truth_guide.md) | Touching `mcmc_experiments`; §0 agent protocol, §2 the `betdb` split |
| [`docs/guides/model_generations_guide.md`](docs/guides/model_generations_guide.md) | Gen 3/4 formulations, Experiment 06 results |
| [`docs/guides/matchday_console_guide.md`](docs/guides/matchday_console_guide.md) | Live (8085) / replay (8086) consoles: isolation, workspace, re-solver, API |
| [`docs/guides/extension_recipes.md`](docs/guides/extension_recipes.md) | Adding a league, component, extractor, metric, calibration law, MatchDay source |
| [`docs/guides/testing_and_verification_guide.md`](docs/guides/testing_and_verification_guide.md) | Test tiers, known T007 failure, replay suite tiers |
| [`docs/prototype_runner_style_guide.md`](docs/prototype_runner_style_guide.md) | Creating or refactoring an `rXX_*.jl` runner |
| [`docs/architecture/ai_agent_infrastructure_and_execution_context.md`](docs/architecture/ai_agent_infrastructure_and_execution_context.md) | Hosts, thread pinning, rsync/cache safety, Distributions pin, prompting block |
| [`docs/setup/agy_remote_execution_guide.md`](docs/setup/agy_remote_execution_guide.md) | Running on `mcmc-beast`: pre-flight, nested tmux, 5-step SOP, recovery |
| [`docs/setup/agy_tmux_agent_and_repl_control_guide.md`](docs/setup/agy_tmux_agent_and_repl_control_guide.md) | Driving subagent panes and the warm Julia REPL over tmux |
| [`current_development/match_day_inference/QUICKSTART_LIVE.md`](current_development/match_day_inference/QUICKSTART_LIVE.md) | Operating the live MatchDay loop |
| [`eda/README.md`](eda/README.md) | Portfolio trust & market-capacity findings |
| [`docs/README.md`](docs/README.md) | Index of everything else under `docs/` |

### Task tracking — `todos/`

Actionable work is tracked in [`todos/`](todos/README.md), one `NNN_slug.md` file
per task; the files are canonical and `todos/README.md` is their index.

```bash
./scripts/todo.sh list | new "Task title" | view 019 | check
```

Claim a task by setting `Status`/`Assignee`, bumping `Updated`, adding a dated
Work Log line **and** updating the matching README row together. Close only once
the acceptance criteria are met and Verification is recorded; use `BLOCKED` (with
the dependency) rather than closing unfinished work. **Run `./scripts/todo.sh
check` before handing off or committing** — it also enforces this file's size budget.

---

## 2. Architecture

A multi-tier Bayesian predictive, portfolio and execution system. The
**Unified V2 stack** is the production standard:

| Layer | Name | Lives in |
|---|---|---|
| L0 | **Data** — typed PostgreSQL extraction, memory-optimised `DataStore`, vig-removed market math | `src/Data/` |
| L1 | **Bayesian engines** — Turing.jl hierarchical models on compiled ReverseDiff tapes | `src/models/pregame/` |
| L2 | **Unified inference, latents & experiment truth** — convergence audits, `CountLatents`, zero-alloc score tensors, Postgres run tracking | `src/training/inference/`, `src/models/latents/` |
| L3 | **Unified evaluation** — point-in-time pricing, LogLoss, CRPS, Brier, RPS, ECE vs closing odds | `src/evaluation/` |
| L4 | **Zero-alloc portfolio, staking & audit** — `OddsIndex`, `BookWorkspace`, Baker-McHale shrinkage, fractional Kelly, Postgres persistence | `src/Portfolio/` |
| L5 | **MatchDay operational execution** — point-in-time slate pricing, paper ledger, live and replay consoles | `src/MatchDay/`, `current_development/match_day_inference/` |

> **Numbering caveat.** An older four-layer scheme survives in some docs and
> archive material, where "Layer 2" means GLM calibration (`src/Calibration/`)
> and "Layer 3" the meta-model (`current_development/MetaModels/`). When a doc
> says "L2/L3", check which scheme it means. The table above is canonical.

Per-layer detail (the L0 Fetch → Process → QA contract, L1 components and
engines, L5 seams, the meta model) is in
[`docs/architecture/unified_v2_architecture.md`](docs/architecture/unified_v2_architecture.md).
Every model must implement `Features.required_features(model)` returning a
`Vector{Symbol}`.

**Generative rate calibration** (`src/Calibration/`, the L2 calibrator tier).
The tradeable book is inverted to `(lambda_mkt_h, lambda_mkt_a)`, pooled with
every posterior log-rate draw, and priced through the **same** score-grid
kernels, evaluator and portfolio as the raw fit — so 1X2, totals and BTTS are
partitions of one score tensor and `src/Portfolio/` is unchanged:

```julia
book, refusals = point_in_time_book(ds; config = PointInTimeBookConfig(as_of_minutes = -25.0))
cal = GenerativeRateCalibrator(name = "scot_lower_t25_inv",
                               law  = InverseGaussianLaw(w_base = 0.25, sigma = 0.35),
                               book_as_of_minutes = -25.0)
cf  = calibrate_fit(cal, fit, book)                        # -> CalibratedFit
result, books, rep = run_portfolio_simulation(spec, policy, cf, book, ds)
```

Which law wins depends on the **sharpness of the book**, not the league
(standard at the Betfair close, inverse at T−25), so `calibrate_fit` refuses a
book from a different instant than `book_as_of_minutes`. **`BasicLogitShift` is
DEPRECATED** — per-selection GLM offsets are incoherent across derivative
markets. Design: [`rfc_layer2_calibration_v2.md`](docs/architecture/rfc_layer2_calibration_v2.md).

### Module map

| Module | Responsibility |
|---|---|
| `Data` | SQL extraction, ETL, `Markets` (vig removal, fair odds, CLM) |
| `Features` | `DataStore` → `FeatureSet`. AD-safe flattening, team/time indexing, lineup valuations, player RAPM ratings. `SplitBoundary` (match-ID pointers, not copies) for temporal folds. Extractors in `src/features/extractors/` |
| `Models` | `CountModelBuilder`, Turing components, master engines |
| `Samplers` | NUTS, MAP, MLE, ADVI wrappers. `QueuedNUTSConfig` flattens K-splits × N-chains into one global queue |
| `Training` | `fit_model`, execution dispatchers, convergence auditing, `PostgresStorage`/`FileStorage`/`DualStorage` |
| `Experiments` | Task creation, execution, persistence, loading |
| `Predictions` | Typed latents (`CountLatents`), in-place score-grid kernels (`SmileScoreGrid`), PPD generation |
| `Evaluation` | `OddsView`, `evaluate_predictions`, LogLoss, CRPS, RPS, ECE |
| `Portfolio` | `OddsIndex`, `BookWorkspace`, Kelly log-utility allocator, `simulate_portfolio` |
| `MatchDay` | Point-in-time slate pricing, gates, instruments, paper ledger (`betdb`), operator console |
| `Calibration` | Generative rate calibration (`GenerativeRateCalibrator`, `calibrate_fit`); deprecated `BasicLogitShift` |
| `Signals` | Betting signal generation and Kelly staking |
| `BackTesting` | `AbstractWealthMetric` (Sharpe, Calmar, Sortino, …) and `AbstractDistributionalMetric` (Hurdle ROI) |
| `MyDistributions` | `RobustNegativeBinomial`, `FrankCopulaNegBin`, `DixonColes`, `BivariatePoissonDist`, … |

---

## 3. The two databases

There are **two** PostgreSQL services answering two different questions.
Confusing them is the most common orientation error in this repository.

| | **`betdb`** — what happened, and what we did | **`mcmc_experiments`** — what we fitted, and what it scored |
|---|---|---|
| Env var | `BF_DB_URL` (**required**, no default) | `BF_EXPERIMENTS_DB_URL`, else `~/.pgpass` |
| Host | `archpc:5433` (LAN `192.168.1.88:5433`, Tailscale `100.124.38.117:5433`) | `mcmc-beast:5432` — `localhost:5432` on the beast |
| Reached from Julia by | `Data.load_datastore_sql`, `MatchDay.paper_connection` | `Training.PostgresStorage(experiment_name)` |
| Holds | schemas `sofascore`, `bbc` (proxy-xG commentary), `betfair` (closing-line archive), `betfair_live` (1-min ladders, ≤ 3 levels), and the paper ledgers `paper_runbook` (live, 8085) / `paper_replay` (replay, 8086) | `config_registry`, `configs`, `runs`, `fold_results`, `match_latents`, `fit_artifacts`, `portfolio_*`, `calibration_*` |
| Failure mode if down | no data, no live pricing, no ledger | no canonical fits; a live slate cannot be priced |

`paper_slates.model_run_id` carries `runs.run_id` as an **opaque UUID with no
foreign key** — separate servers. The paper ledger lives in `betdb` on purpose
(availability at T−12, locality with the order book, and separation from the
**backtest** ledger `portfolio_bets`). Schema-by-schema reference, unique
constraints and the full rationale:
[experiment DB guide §2](docs/guides/experiment_database_and_config_truth_guide.md#2-the-two-databases).

---

## 4. Unified V2 pipeline — the production path

```julia
using BayesianFootball
using DataFrames, Dates, ThreadPinning

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

# 1. Data layer
ds = Data.load_datastore_cached(Data.ScottishLower())

# 2. Composable model builder — here the current production shape:
#    two-arm joint observation + team time decay + lineup RAPM + squad wealth.
model = CountModelBuilder(:m12_joint_hybrid_synergy) |>
    add(GlobalInterception()) |>
    add(TimeDecayDynamics(days_half_life = 180.0)) |>
    add(GlobalHomeAdvantage()) |>
    add(PlayerLineupPillar(rating = :shots_rapm,
                           aggregation = BenchWeightedPlayerAggregation(w_bench = 0.10),
                           fit_on = :history)) |>
    add(ProductionWealthCovariate(role = SupremacyRole())) |>
    add(JointGammaPoissonObservation()) |>
    build

# 3. Unified inference lifecycle & convergence gating
fit_cfg = FitConfig(
    name      = "m12_joint_hybrid_synergy",
    model     = model,
    splitter  = Data.CVConfig(target_seasons = ["24/25", "25/26"], window_seasons = 3),
    sampler   = NUTSConfig(n_samples = 1_000, n_chains = 4),
    execution = AutoExecution(),   # → QueuedExecution or ThreadedExecution
)
fit = fit_model(fit_cfg, ds)

# 4. Unified evaluation (LogLoss, CRPS, Brier, RPS, ECE vs closing odds)
eval_report = evaluate_predictions(fit, ds)

# 5. Zero-alloc portfolio & staking simulation
spec   = BookSpec(markets = Data.MarketConfig([Data.Market1X2(), Data.MarketOverUnder(2.5)]),
                  shrink  = BakerMcHale())
policy = PolicySpec(trust = FlatTrust(0.25), risk = SlateDrawdown(20.0), cap = FixedCap(0.25))
result, books, rep = run_portfolio_simulation(spec, policy, fit, ds.odds, ds)
```

The exact component names for a candidate are in that experiment's `lXX_loader.jl`;
treat the block above as the shape of the call, not as a copy-paste recipe.

---

## 5. Prototyping and experiment layout

**`current_development/` — prototypes.** New features are prototyped here
**before** moving to `src/`, always as a pair:

- **`lXX_*.jl` (loader)** — structs, functions, mathematical logic; a temporary module.
- **`rXX_*.jl` (runner)** — load data, call the loader, run, visualise. Runners stay
  human-readable research notebooks with numbered package / configuration / data /
  model / training / diagnostic / inference sections; checkpoint and persistence
  machinery belongs in the loader. Read the
  [runner style guide](docs/prototype_runner_style_guide.md) first.
- **`XX`** — two-digit iteration counter; increment for a fresh approach.

Graduate to `src/` only once the prototype is validated in the runner. Active
streams include `match_day_inference/` (the live/replay consoles),
`player_lineup_dynamics/`, `plus_minus_ratings/`, `bbc_xg_proxy/`,
`calibration_generative_eda/`, segment folders (`scottish_lower/`, …), in-play and
exchange-microstructure research, and `MetaModels/`. `archived/` is **not** the
active path.

**`experiments/<segment>/NN_<topic>/` — completed benchmark suites.** A loader,
smoke / production / comparison / portfolio runners, and a `README.md` carrying
the *measured* results and run UUIDs — **source performance claims from here**. A
production grid is not launched until the smoke runner passes every gate for every
candidate: gradient tape, sampling, the six-part convergence audit, latent
extraction, score grid, `save_fit`/`load_fit` round-trip, and portfolio
persistence with an identical reloaded bet ledger. Top-level `experiments/*.md`
files are work-package prompts — inputs, not records; never cite one as a result.

---

## 6. Model generations — Scottish Lower (tournaments 56 / 57)

Four paradigms, each a full 40-fold walk-forward grid over seasons 24/25 + 25/26
(710 held-out matches, 2,899 scored market observations). Numbers are the
recorded outcomes in each suite's README, not targets. Formulations and the full
Experiment 06 table: [`docs/guides/model_generations_guide.md`](docs/guides/model_generations_guide.md).

| Gen | Suite | Paradigm | Headline |
|---|---|---|---|
| **1** | [`01_poisson_2426_grid/`](experiments/scottish_lower/01_poisson_2426_grid/README.md) | Poisson; baseline, squad wealth, travel distance, age-adjusted production wealth | `m05_production_wealth` LogLoss **0.6597**; Betfair backtest +125% to +140% |
| **2** | [`02_negbin_2426_grid/`](experiments/scottish_lower/02_negbin_2426_grid/README.md) | Negative Binomial | `r̂ ≈ 26.0–26.5`; LogLoss **0.6598**, no material gain |
| **3** | [`03_joint_gamma_poisson/`](experiments/scottish_lower/03_joint_gamma_poisson/README.md) | **Two-arm joint**: shared `μ`, Gamma arm on BBC proxy xG, Poisson arm on goals | LogLoss **0.6571** vs close 0.6568 — worth ~5× the best covariate |
| **4** | [`05_.../`](experiments/scottish_lower/05_player_lineup_and_pxg_fusion/README.md) + [`06_.../`](experiments/scottish_lower/06_joint_player_lineup_fusion/README.md) | **Joint + player-lineup hybrid**: `PlayerLineupPillar` (RAPM, `w_bench = 0.10`) beside team time decay | `m12` ECE **0.0100** vs close **0.0139**; +136.6% bankroll, Sharpe 1.416 |

**Read Gen 4 honestly:** the lineup arms do **not** win on LogLoss — the
team-state control `m05` is still sharpest; they buy **calibration**, which is
what converts into Kelly growth. `m12` is the hybrid the MatchDay consoles load;
`m00`/`m05` are the controls that make a lineup move attributable.

---

## 7. Non-negotiable rules

**AD safety inside `@model`** (authority:
[`docs/turing_ad_performance_guide.md`](docs/turing_ad_performance_guide.md)):

- Feature vectors are pure `Float64`/`Int` — **no `missing` inside `@model`**; the
  feature builder does all conditional logic and imputation.
- **No `if`/`else`, no `for` loops, no `isnan`/`findall` inside `@model`** — use
  binary masks and broadcast arithmetic; dispatch on types for optional terms.
- **Mask, don't subset**: compute the likelihood on all rows and multiply by a
  0/1 availability mask built in the builder (e.g. `coalesce.(xg, NaN)` → mask +
  distribution-safe imputation).
- Select parameters with `A[idx]`, **not** `view(A, idx)`; no array mutation.
- Guard numerics branch-free: `clamp` log-rates, `Turing.@addlogprob! ifelse(bad, -Inf, 0.0)`.

**Credentials.** **Never** commit, paste into a prompt, or print a raw password or
a credential-bearing URL. `BF_DB_URL` comes from the environment (`.env`, git-ignored);
`PostgresStorage` resolves `BF_EXPERIMENTS_DB_URL` or `~/.pgpass`. Its masked
`show` is safe; printing `storage.conn_str` is not.

**Database separation.** A live or replay slate **reads** a converged run out of
`mcmc_experiments` (`MD.canonical_fit`) and **writes** only to
`betdb.<paper_schema>`. Never write paper-trading rows into `mcmc_experiments`.
Register configs, check `configs.config_hash` before sampling, and persist run
UUIDs — the seven-rule protocol is
[experiment DB guide §0](docs/guides/experiment_database_and_config_truth_guide.md#0-agent-protocol--the-seven-rules).

**Compute.** Never run heavy Turing models on the laptop. `-t 16` on
`mcmc-beast`, `-t 8` on `archpc`; always `pinthreads(:cores)` and
`BLAS.set_num_threads(1)` before sampling; one production grid at a time on the
beast. `rsync` with `--exclude '.cache/' --exclude 'data/'`; regenerate
`.cache/datastore_<Segment>.jls` when a new SQL column lands.

---

## 8. Infrastructure & remote execution

| Host | Role |
|---|---|
| `archpc` (local laptop) | Development — `/home/james/bet_project/BayesianFootball`; 8 cores / 16 SMT; PostgreSQL `betdb` on **5433**; both MatchDay consoles (8085, 8086) |
| `mcmc-beast` | Compute — 16 cores (32 SMT), 64 GB, `/root/BayesianFootball`; PostgreSQL `mcmc_experiments` on **5432** |

The loop: edit `lXX`/`rXX` locally → commit & push (only with permission) →
`git pull` in the beast shell window → `include("…/rXX_runner.jl")` in the warm
REPL → capture the pane. Nested-tmux prefixes, window map and recovery are in
[`agy_remote_execution_guide.md`](docs/setup/agy_remote_execution_guide.md).

```bash
tmux ls                                                      # indices drift — verify first
tmux send-keys -t scottish_runner:1.1 'include("current_development/<stream>/rXX_runner.jl")' C-m
tmux capture-pane -t scottish_runner:1.1 -p -S -60
```

---

## 9. Operational guides in brief

**MatchDay consoles.** Live (`r07_serve_console.jl`, **8085**, `betdb.paper_runbook`,
wall clock) and replay (`r08_replay_console.jl`, **8086**, `betdb.paper_replay`,
scrubbed clock) run the **same** pipeline; replay swaps only the sources that read
a clock or network for point-in-time twins, and isolation is structural
(`assert_replay_schema`, `serve_replay` refuses 8085). `LadderSweep` replay P&L is
an **upper bound**; post-kick-off "edges" are not a signal. Check it is up with
`curl -s localhost:8086/api/health`. Everything else — the Gödel workspace, the
dynamic slate re-solver, the API — is in the
[MatchDay console guide](docs/guides/matchday_console_guide.md).

**Tests.** Pick the fastest tier that covers your change: one module
(`include("test/<suite>.jl")`, ~20 s) → `test/run_parallel_tests.jl` (~45 s) →
`test/runtests.jl` (full, ~3.5–6 min) → `test/test_matchday_replay.jl` (not in the
parallel runner). `features_tests.jl` failing only under the parallel runner is
the known [T007](docs/tickets/T007-parallel-feature-test-hidden-dependency.md). A
tier that *skipped* is not evidence. Detail:
[testing guide](docs/guides/testing_and_verification_guide.md).

**Extending the system** (new league, component, extractor, metric, calibration
law, MatchDay source): [extension recipes](docs/guides/extension_recipes.md).

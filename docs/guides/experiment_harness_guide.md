# Experiment harness guide

> **Use this guide before creating or running an experiment.** The harness is the
> reproducible path from a declared scientific recipe to persisted runs, checks,
> a versioned scorecard, and the experiment record. It does not replace feature
> or component development.

## 1. Choose the right lane

| Lane | Use it for | Primary record | Do not use it for |
|---|---|---|---|
| **EDA** | Data availability, provider semantics, point-in-time validity, descriptive plots, and hypotheses worth testing | `eda/`, a suite's EDA report, or `current_development/` research notes | A claim that a model won without an out-of-sample fit |
| **Component** | A new feature, likelihood, splitter, dynamics component, or model implementation | paired `current_development/<stream>/lXX_*.jl` and `rXX_*.jl`, then module tests | A production candidate matrix before the component's contract, AD safety, and filtration are established |
| **Experiment** | A declared comparison of already-valid components across models and scopes | `experiments/<segment>/NN_topic/candidates.jl`, its `README.md`, `harness_checks`, and the generated register/leaderboard | Ad-hoc REPL fitting or a new unvalidated component |

Start in EDA when the question is whether a signal exists or is available at
prediction time. Start in a component prototype when the question is whether
an implementation is mathematically, temporally, and AD-safe. Start an
experiment only after those contracts are settled and the question is a
comparison with a named control and decision rule.

For prototypes, read the [runner style guide](../prototype_runner_style_guide.md).
For data and database provenance, read the
[experiment database and config truth guide](experiment_database_and_config_truth_guide.md).

## 2. Define an experiment

Create `experiments/<segment>/NN_topic/` with both:

- `candidates.jl`: the executable, declarative model × data-scope recipe;
- `README.md`: the human record of the question, candidates, decision rule,
  commands, run UUIDs, scorecard version, results, caveats, and decision.

Use [`experiments/TEMPLATE_README.md`](../../experiments/TEMPLATE_README.md)
as the README starting point. The candidates file must define:

```julia
const EXPERIMENT = "stable_experiment_namespace"
const QUESTION = "What decision does this comparison support?"
const DIMENSION = "likelihood"       # or dynamics, data_scope, lineup, ...
const STATUS = "active"
const DECISION = "pending"
const README = joinpath(@__DIR__, "README.md")
const CONTROL = "named_control"
const CANDIDATES = Harness.Candidate[
    Harness.Candidate(
        name = "named_control",
        model = control_model,
        scope = control_scope,
        sampler = sampler,
        role = :control,
        hypothesis = "The reference recipe held fixed for this decision.",
    ),
    # Candidate(...) rows with role = :candidate.
]
```

Each `Candidate` contains a name, built model, `Data.DataScope`, sampler,
`:control` or `:candidate` role, and a concise hypothesis. Its `recipe_hash`
is the SHA-256 identity of the **model and scope**, deliberately excluding the
sampler budget so a smoke validates the grid recipe. Use exactly one control in
a comparison file. A candidate name is part of the record: do not reuse it for
a materially different recipe.

The harness loads the file in its own module. Put ordinary Julia declarations
and constructors in it; do not rely on state from a warm REPL. The W1 example
in [`experiments/scotland/03_dynamics_scope_matrix/candidates.jl`](../../experiments/scotland/03_dynamics_scope_matrix/candidates.jl)
shows a factorial candidate matrix and a named control.

## 3. Run the stages in order

Run sampling on `mcmc-beast`, never on the laptop. From the immutable beast
checkout (see §4), use:

```bash
julia --project -t 16 scripts/run_candidates.jl \
  experiments/<segment>/NN_topic/candidates.jl --stage screen
julia --project -t 16 scripts/run_candidates.jl \
  experiments/<segment>/NN_topic/candidates.jl --stage smoke --only candidate_name
julia --project -t 16 scripts/run_candidates.jl \
  experiments/<segment>/NN_topic/candidates.jl --stage grid --only candidate_name
julia --project -t 16 scripts/run_candidates.jl \
  experiments/<segment>/NN_topic/candidates.jl --stage portfolio \
  --container close_option_b
```

Use `--only name1,name2` to limit a smoke or grid. `smoke` and `grid` require
at least eight Julia threads and pin threads; the normal beast invocation is
`-t 16`. Every smoke invocation gets a new `smoke_nonce:` config tag, so re-smoking an
unchanged recipe saves a distinct fit while its `harness_checks.recipe_hash` remains stable.
Grid fits are saved at full stride 1; persistence failures are errors, not thinning requests. `portfolio` reuses completed grid runs and accepts
`close_option_b` or `t25_calibrated`; it does not sample a new fit.

| Stage | Purpose | Pass condition and interpretation |
|---|---|---|
| `screen` | Cheap MAP fit over all folds to expose construction, filtration, and obvious ranking problems | Filtration and finite-latent checks must pass. Scores are in-memory MAP diagnostics; no fit run is persisted. |
| `smoke` | Two-fold, 2-chain NUTS correctness gate | Gradient, filtration, latent, score-grid coherence, and database save/load parity are **hard** checks. A hard failure stops the candidate. Tape timing/size and grid-tail mass are recorded diagnostics. |
| `grid` | Full walk-forward NUTS run and score | Refuses to start without passing smoke hard checks for the same `recipe_hash`. It checks filtration, latents, target coverage, persistence parity, saves the run, and scores it against the control. Checkpoints and completed recipe hashes support resume. |
| `portfolio` | Finalist comparison under one buildable panel and one pricing/staking container | Reuses grid UUIDs, rejects skipped fixtures on the common panel, and writes `finalist` score rows. It is a portfolio result, not evidence that the likelihood fit converged. |

A **hard** check failure makes the stage invalid. A `review` diagnostic—most
notably convergence—does not discard the persisted grid run, but it is a
promotion blocker until reviewed. `info` and `diagnostic` rows provide
telemetry, not a pass certificate. Inspect `harness_checks` and the runner's
`[SUMMARY]` lines; never report a run as accepted merely because it has a UUID.

### MAP screen caveat

Screen is a triage tool, not posterior inference. Screen fits are in memory only; the
synthetic recipe UUIDs cannot be re-scored with `score_runs.jl`. To refresh a MAP screen
for a new scorecard, re-run `--stage screen`. For a test-database rehearsal, set
`BF_EXPERIMENTS_TEST_DB_URL` to the isolated `mcmc_experiments_test` and pass `--test-db`:
the script rejects missing or mispointed test URLs and preserves the metadata of an existing
completed experiment register row on re-screen.

It is useful for simple within-class ranking, but it cannot replace the NUTS grid. In particular,
models with learned hierarchical or random-walk scales (including
`TimeDecayDynamics`, which learns attack/defence σ, and `MultiScaleGRW`) receive
`screen_validity = "limited"`: MAP shrinkage can
flatten innovations and make their ranking or compression misleading. Retain
the recorded screen row, but do not eliminate or promote such a model from MAP
scores alone.

## 4. Use an immutable beast checkout and retain its log

After the intended commit is available on the remote, create the checkout:

```bash
CHECKOUT="$(scripts/beast_checkout.sh <git-sha>)"
cd "$CHECKOUT"
mkdir -p "/root/BF_runs/logs/<git-sha>"
julia --project -t 16 scripts/run_candidates.jl \
  experiments/<segment>/NN_topic/candidates.jl --stage smoke \
  2>&1 | tee "/root/BF_runs/logs/<git-sha>/smoke.log"
```

`beast_checkout.sh` creates (or reuses) `/root/BF_runs/<git-sha>`, links the
canonical `Manifest.toml` and local `.env`, and copies the Scottish DataStore
cache when available. Logs belong outside the checkout at
`/root/BF_runs/logs/<git-sha>/`; do not write scratch logs into the worktree,
because that changes the recorded Git provenance to `-dirty`. Preserve the
command, checkout SHA, log path, scorecard version, and returned run UUIDs in
the suite README.

## 5. Read scorecards without mixing controls

`harness_scores` stores a `scorecard_version` with every row. Compare rows only
when the scorecard version, reference price instant, subset, market, and metric
match. A new scorecard can change the market frame or definition; it is not a
cosmetic label.

Scorecard v1.2 rows also carry `panel` (`56+57|target-seasons|n=fixtures`).
Rank and compare only within the same panel; `scripts/score_runs.jl` accepts
`--target-seasons 23/24,24/25,25/26 --expected-fixtures N|none` (W1 defaults:
24/25,25/26 and 710). Existing v1.1 panels are backfilled from each run's
own priced target fixture count where a frozen panel is known (W1's 627
Betfair-covered fixtures identify its 710-fixture target panel); unknown counts remain
explicitly labelled unknown, never assumed to be W1.

Paired deltas additionally carry `control_run_id`. A CSV with an optional `control` column (paired base label or UUID) can
score multiple control groups in one invocation; with no column the sole
`role=:control` remains the base. The same candidate run may
legitimately have different deltas against different controls, so never merge,
quote, or overwrite delta rows while ignoring that UUID. State both the
candidate run UUID and control run UUID in a README when quoting a delta or its
confidence interval.

The generated [`experiments/LEADERBOARD.md`](../../experiments/LEADERBOARD.md)
is the current cross-run comparison surface: read its version/reference header,
then the posterior-grid headline and cohort rows. MAP rows appear only in a
separate warning-labelled diagnostic section and are never mixed into grid Δs. The generated
[`experiments/REGISTER.md`](../../experiments/REGISTER.md) is the inventory of
questions and decisions. Historical rows preserve their original evidence and
caveats; a `completed` row is not permission to transfer its claim to a different
scorecard or price source.

## 6. Add or update a register row

`run_candidates.jl` upserts `harness_experiments` from the metadata in
`candidates.jl` after a stage. Its `run_ids` are atomically merged on conflict
(`--only` cannot erase another candidate's run), with UUID-keyed `run_commits`
JSONB preserving each fit's original commit SHA. Before running it, set `EXPERIMENT`, `TODO`,
`QUESTION`, `DIMENSION`, `STATUS`, `DECISION`, and `README` truthfully. After a
screen, use a status such as `screened` with `grid pending`; after a completed
decision, record the conclusion and immutable run UUIDs. Do not call a
screened, smoke-only, blocked, or EDA-only item completed.

Regenerate `LEADERBOARD.md` and `REGISTER.md` with the project leaderboard
workflow after the database record changes, then review the generated row:

1. ID and TODO identify the same work item as its suite README.
2. Question, dimension, status, and decision say what actually happened.
3. Runs list immutable UUIDs, or explicitly say `not applicable` / `not run`.
4. The record link resolves to the suite README or the appropriate EDA/prototype
   record.

Do not hand-edit generated leaderboard/register output as a substitute for the
database record. For a historical item that cannot be run through the harness,
create or update its provenance row through the register workflow and make the
README caveat explicit.

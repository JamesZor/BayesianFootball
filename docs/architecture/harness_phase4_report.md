# Experiment harness Phase 4 report

Date: 2026-09-27  
Task: [TODO 030](../../todos/030_shared_experiment_harness_recipes_gates_runner_scorecard_leaderboard.md)  
Scope: Phase 4 only; no MCMC was run

## Result

Phase 4 is complete. The harness now evaluates completed grid runs under the pinned Task 013
closing Option-B contract or the T−25 inverse-Gaussian calibrated container, uses one common
buildable panel, refuses any subsequent fixture skip, and writes versioned `stage = "finalist"`
rows without replacing grid scores or another finalist container. MAP screen rows are now visibly
separate from posterior-grid results and carry a per-candidate validity diagnostic.

Implementation commits:

- `09cfcba3` — finalist portfolio stage, portfolio CLI, screen validity, safe purge tool and tests.
- `5761971d` — practical harness guide, README template, provenance corrections and register seed.
- `9ae71ebe` — preserve each finalist container independently in `harness_scores`.

Remote evaluation used immutable checkout `/root/BF_runs/9ae71ebe`; logs are outside the checkout
under `/root/BF_runs/logs/9ae71ebe/`. The permitted stale-row purge and validity/register backfill
were first executed from checkout `5761971d`, with logs under its matching external log directory.

## Acceptance evidence

### 1. Closing Option-B reproduces r05

Command on `mcmc-beast`:

```bash
JULIA_PKG_PRECOMPILE_AUTO=0 julia --project -t 16 scripts/score_runs.jl \
  /root/BF_runs/logs/9ae71ebe/finalists.csv --portfolio close_option_b
```

The harness deliberately pins `MatchDay.option_b_system(lambda = 8.0)`, which is the Task 013/r05
contract. The live factory's later default must not rewrite a historical reproduction target.
Every acceptance field below equals
`current_development/grw_player_hybrid/results/portfolio/r05_portfolio_summary.csv` to `1e-9`
(the displayed values are in fact identical).

| Model / UUID | Panel | Slates | Bets | Return % | ROI % | Sharpe | Calmar | Max DD % | Win % | Mean edge pp | Exposure | Capture |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `m12_hybrid_td_raw` / `132df5c2-c742-4e95-8693-3aeb2b2cbaef` | 632 | 100 | 1302 | 606.5369616689156 | 14.154537865990093 | 1.4865272265515141 | 14.31841428421878 | -42.360623853258524 | 33.94777265745007 | 4.5886556727853485 | 0.1758156791115612 | 0.9492059179227964 |
| `m05_joint_td_raw` / `ed541a7c-01e2-447e-a771-783517728d47` | 632 | 100 | 1280 | 495.50427678971437 | 13.571361709481343 | 1.3882019490912514 | 12.305718507532267 | -40.26618002731159 | 33.828125 | 4.427969518174084 | 0.16901289778308748 | 0.9454866495697053 |

Additional exact fields:

| Model | CAGR % | Growth/slate | Growth 95% CI | P(ROI > 0) |
|---|---:|---:|---:|---:|
| `m12_hybrid_td_raw` | 210.667669213786 | 0.01955205331355309 | [-0.00008781806689157227, 0.039332446750636255] | 0.991 |
| `m05_joint_td_raw` | 181.35117732001876 | 0.017842383846281815 | [-0.001457720775256878, 0.03696198450467225] | 0.987 |

**What defined the r05 panel.** r05 loaded all seven arms (four GRW ladder fits and the three
pinned controls), required each to contain the same 710 target fixtures, intersected the first
arm with the closing odds to get 635 quoted fixtures, then unioned build refusals over all seven;
three had no usable selections, leaving 632. The Phase-4 acceptance invocation supplied the two
named finalists. Starting from all 710 fixtures, the harness explicitly recorded the 75 no-quote
and three no-selection refusals and independently reached the same 632 IDs. Exact downstream
metrics establish that this is the same effective panel, not a hand-picked substitute.

The database holds 30 long rows (15 metrics × 2 runs) at
`subset = portfolio_close_option_b`, `market = book`, `stage = finalist`, all addressed by
control UUID `132df5c2-c742-4e95-8693-3aeb2b2cbaef`.

### 2. T−25 calibrated finalist container

Command:

```bash
JULIA_PKG_PRECOMPILE_AUTO=0 julia --project -t 16 scripts/score_runs.jl \
  /root/BF_runs/logs/9ae71ebe/finalists.csv --portfolio t25_calibrated
```

This builds the complete point-in-time book with `as_of_minutes = -25`, calibrates every
posterior draw with `InverseGaussianLaw(w_base = 0.25, sigma = 0.35)` and enforces the book
instant. Both finalists used one 611-fixture common buildable panel.

| Model / UUID | Panel | Slates | Bets | Return % | ROI % | Sharpe | Calmar | Max DD % | Win % | Mean edge pp | Exposure | Capture |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `m12_hybrid_td_raw` / `132df5c2-c742-4e95-8693-3aeb2b2cbaef` | 611 | 99 | 1001 | 265.7839196016244 | 17.433735934437674 | 1.7137314707201177 | 13.269911458745717 | -20.029065033923484 | 35.464535464535466 | 2.6474810598465117 | 0.0853751234080037 | 0.9106373008873275 |
| `m05_joint_td_raw` / `ed541a7c-01e2-447e-a771-783517728d47` | 611 | 99 | 977 | 276.8452761073675 | 18.53267625984543 | 1.9654752296486733 | 15.18830605272333 | -18.227528148718594 | 35.107471852610026 | 2.588566883296248 | 0.08010078018569426 | 0.959386586639004 |

| Model | CAGR % | Growth/slate | Growth 95% CI | P(ROI > 0) |
|---|---:|---:|---:|---:|
| `m12_hybrid_td_raw` | 112.09780416548551 | 0.013099723125920542 | [0.0018787573615103965, 0.024832451397318187] | 0.994 |
| `m05_joint_td_raw` | 115.79302457007606 | 0.013400651606223786 | [0.003431352446593555, 0.023626104817110104] | 0.99675 |

The database holds a separate 30-row `portfolio_t25_calibrated` block. A post-write query showed
both finalist blocks and all pre-existing `grid` subsets still present; the replacement key now
includes stage and subset.

### 3. Screen validity and leaderboard separation

`screen` now writes one `harness_checks` diagnostic per candidate. The W1 backfill recorded:

```text
SCREEN_VALIDITY rows=12 limited=12 ranking_only=0
```

All 12 are correctly `limited`: `TimeDecayDynamics` learns attack/defence σ and
`MultiScaleGRW` learns its innovation scales. `experiments/LEADERBOARD.md` now has:

1. a posterior-grid headline table (20 runs) with Δ columns;
2. a separate warning-labelled MAP screen table (12 deterministic screen UUIDs, no Δ columns);
3. posterior-grid transition cohorts only.

No MAP row is present in `experiments/LEADERBOARD.csv`, which remains the posterior-grid view.

**Proposed better cheap screen (not built).** Use a short single-chain NUTS run on every fourth
canonical fold, retaining the declared priors and integrating scale uncertainty. It costs more
than MAP but evaluates every model class under the same posterior operation. It should remain a
triage diagnostic: the full multi-chain 40-fold grid and convergence audit still decide promotion.
An alternative worth testing later is MAP with each learned walk scale fixed to a reference
posterior median, but that introduces a reference-run dependency and is less universal.

### 4. Stale screen rows purged

The only DELETE in Phase 4 was through the new fail-closed tool and targeted only
`harness_scores`:

```text
BEFORE run_id=fcab115f-21f6-4506-8b32-fbd0e2440bc5 stage=screen rows=654
AFTER  run_id=fcab115f-21f6-4506-8b32-fbd0e2440bc5 stage=screen rows=0 deleted=654
```

A direct after-query showed the correct deterministic UUID
`630d12ec-ef5c-5163-b967-78f0ece987a0` still has 654 rows. The script refuses a missing or
non-UUID run ID, defaults explicitly to stage `screen`, prints the before count, and cannot touch
another table.

### 5. Documentation, provenance and register

- Added [`experiment_harness_guide.md`](../guides/experiment_harness_guide.md): EDA/component/
  experiment lanes, `candidates.jl`, checks/stages, immutable beast checkouts, external logs,
  scorecard/control identity, MAP validity, leaderboard/register use and register workflow.
- Added [`experiments/TEMPLATE_README.md`](../../experiments/TEMPLATE_README.md).
- Linked the guide from `AGENTS.md` §1 and §5. §6 and
  [`model_generations_guide.md`](../guides/model_generations_guide.md) now point to the versioned
  leaderboard and identify suites 01–03 as older-scorer historical records.
- `UNIFIED_PARADIGM_REPORT.md` now names its two synthetic bench rows and UUIDs:
  `m00_baseline` `013af743-853d-481a-abc7-03a233fc8420` and
  `m05_production_wealth` `16f9e83d-4865-4298-9b4e-ae33f9852f0d`, both
  `git_commit = synthetic-no-mcmc`. The other six report-time rows were real MCMC runs.
- The Scottish Lower index points first to `REVIEW.md`, `LEADERBOARD.md` and `REGISTER.md`.
- The register now contains TODO 030 as `validated`, `harness_validation` with reproduced g2 run
  `75190012-c833-4b6c-8489-49bdb676a0d3`, and TODO 031 W1 as `screened` / grid pending.

### 6. Precompile warning provenance

A clean load at historical commit `6c1da1cd` reproduced all nine warnings exactly:

```bash
git worktree add --detach /tmp/BF_6c1da1cd 6c1da1cd
julia --project -e 'using BayesianFootball'
```

The undeclared imports are `TurnoverCommission`, `IndependentKelly`, `StaticFamilyTrust`,
`ShrinkToMarketTrust`, `MatchDrawdown`, `FixedFraction`, `PerMatchCap`, `WeeklySlate`, and
`MatchSlate`. Commit `4b6b5906` added those exports and is an ancestor of `6c1da1cd`; none of the
names has a declaration under `src/Portfolio/`. The warning is therefore pre-existing, not
introduced by the harness branch. It was recorded rather than expanded into an out-of-scope
Portfolio API cleanup.

### 7. Tests and repository checks

Standalone suites:

```text
harness_portfolio_tests.jl: 18/18 pass
harness_runner_tests.jl:    54/54 pass
harness_scoring_tests.jl:   29/29 pass
scoped_splitter_tests.jl:   48/48 pass
```

The parallel runner included the new portfolio suite:

```text
22 / 23 suites passed in 256.9s
```

The only failure was the known T007 signature in `features_tests.jl`:
`UndefVarError: SplitClockProbe not defined in Main`. All four harness suites passed; this is no
worse than Phase 3's one known failure.

Repository checks:

```text
./scripts/todo.sh check
OK: 31 task(s); metadata, template and registry agree; AGENTS.md 20004 bytes (< 22000).

git diff --check
(no output)
```

TODO 030 remains `ACTIVE`, as requested, with dated `@pi` work-log entries.

## Human decisions

Do **not** delete these improvised completed runs automatically. They are listed for human review:

| Namespace | Name | UUID | Git provenance |
|---|---|---|---|
| `harness_clean_smoke` | `g2_clean_smoke` | `dd8a9005-ad02-42cf-8e7c-5f34bd2e70e6` | `unknown` |
| `harness_clean_smoke` | `g2_clean_smoke2` | `07d182a1-fbf3-469a-8d7e-9bb730a99c94` | `19091737` |

The human decides whether those `runs` rows and their inference artifacts should be deleted.
The pre-existing undeclared Portfolio exports likewise need a separate cleanup decision if the
warnings are to be removed rather than merely documented.

## Blocked

None.

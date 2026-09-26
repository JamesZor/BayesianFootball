# NN — Experiment title

> **Status:** planned | screened | active | completed | blocked  
> **Experiment:** `stable_experiment_namespace`  
> **TODO:** NNN  
> **Scorecard:** `vX.Y` (state the reference price instant)  
> **Control:** `control_candidate_name` — run UUID: `pending`

## Question

State the decision the experiment is intended to make. Name what is held fixed,
the control, and what evidence would change the decision.

## Hypothesis and scope

- **Hypothesis:**
- **Target population / seasons:**
- **Training scope:**
- **Selection metrics:**
- **Non-goals / exclusions:**

## Candidates

The executable source of truth is [`candidates.jl`](candidates.jl). Summarise
its declared arms here; do not maintain an alternate recipe by hand.

| Candidate | Role | Model / scope difference | Hypothesis |
|---|---|---|---|
| `control_candidate_name` | control | | |
| `candidate_name` | candidate | | |

## Decision rule

Specify the scorecard version, price reference, selection subset and markets,
control run, required checks, and how trade-offs will be resolved. MAP screen
ranking is not a promotion rule for models with learned hierarchical or
random-walk scales.

## Execution record

| Stage | Command / checkout | Check outcome | Run UUID | Log |
|---|---|---|---|---|
| screen | | | not persisted | |
| smoke | | | | |
| grid | | | | |
| portfolio | | | reuses grid run | |

Record the immutable beast checkout SHA, with logs under
`/root/BF_runs/logs/<sha>/`. A grid row is not accepted if its hard checks fail;
a convergence `review` flag must be discussed before promotion.

## Results

Quote only rows from the stated scorecard version and reference. For paired
deltas, include both the candidate UUID and `control_run_id`.

| Candidate | Target LogLoss | 1X2 LogLoss | ECE | Compression | Δ vs control [95% CI] | Run UUID |
|---|---:|---:|---:|---:|---:|---|
| | | | | | | |

Link to [`../../LEADERBOARD.md`](../../LEADERBOARD.md) for cross-experiment
comparisons and [`../../REGISTER.md`](../../REGISTER.md) for the decision
inventory.

## Decision

- **Decision:** pending | retain | promote | reject | blocked
- **Rationale:**
- **Follow-up:**

## Caveats and reproducibility

Document filtration limitations, data or market coverage, scorecard changes,
MAP-screen limitations, convergence reviews, persistence thinning, and anything
that makes the result non-comparable. Include exact commands needed to reproduce
scoring or portfolio evaluation; never include credentials.

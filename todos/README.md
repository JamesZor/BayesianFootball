# TODO Tracking

`todos/` is the repository-local tracker for actionable work. Task files are
canonical; this registry is a human-maintained index of their metadata. It does
not replace experiment records, issue trackers, or `docs/tickets/`.

## Registry

<!-- TASKS:START -->
| ID | Title | Status | Priority | Assignee | Last Updated |
|---|---|---|---|---|---|
| [001](001_sampling_budget_benchmark.md) | Benchmark MCMC Sampling Budget and Target Acceptance on Fold 1 | COMPLETED | P1 | claude | 2026-09-10 |
| [002](002_complete_experiment_08_goal_decomposition_production_grid.md) | Complete Experiment 08 Goal Decomposition Production Grid | IN_PROGRESS | P1 | claude | 2026-09-10 |
| [003](003_benchmark_julia_gc_tuning_and_heap_size_hint_on_multi_core_sampling.md) | Benchmark Julia GC Tuning and Heap Size Hint on Multi-Core Sampling | COMPLETED | P1 | claude | 2026-09-10 |
| [004](004_scope_turing_jl_v0_46_and_dynamicppl_v0_42_upgrade.md) | Scope Turing.jl v0.46 and DynamicPPL v0.42 Upgrade | ACTIVE | P1 | claude | 2026-09-10 |
| [005](005_rerun_scottish_lower_experiment_01_poisson_grid_with_reversediff_and_verify_post.md) | Rerun Scottish Lower Experiment 01 Poisson grid with ReverseDiff and verify posterior parity | COMPLETED | P1 | pi | 2026-09-10 |
| [006](006_rerun_scottish_lower_historical_paradigms_with_reversediff_and_unified_benchmark.md) | Rerun Scottish Lower historical paradigms with ReverseDiff and unified benchmark | COMPLETED | P1 | pi | 2026-09-10 |
| [007](007_prototype_gaussian_random_walk_state_space_dynamics_with_reversediff.md) | Prototype Gaussian Random Walk State Space Dynamics with ReverseDiff | COMPLETED | P2 | pi | 2026-09-11 |
| [008](008_hierarchical_scottish_pitch_type_and_match_timing_home_advantage.md) | Hierarchical Scottish Pitch Type and Match Timing Home Advantage | BACKLOG | P2 | unassigned | 2026-09-10 |
| [009](009_hierarchical_multi_tournament_pooling_across_scottish_segments.md) | Hierarchical Multi Tournament Pooling Across Scottish Segments | BACKLOG | P2 | unassigned | 2026-09-10 |
| [010](010_design_and_prototype_autonomous_agent_bayesian_model_search_loop.md) | Design and Prototype Autonomous Agent Bayesian Model Search Loop | BACKLOG | P2 | unassigned | 2026-09-10 |
| [011](011_fuse_goal_decomposition_with_proxy_xg_and_player_lineup_dynamics.md) | Fuse Goal Decomposition with Proxy xG and Player Lineup Dynamics | BACKLOG | P2 | unassigned | 2026-09-10 |
| [012](012_standardise_model_attribution_and_capture_ratio_in_portfolio_engine.md) | Standardise Model Attribution and Capture Ratio in Portfolio Engine | BACKLOG | P2 | unassigned | 2026-09-11 |
| [013](013_fuse_multiscalegrw_with_player_lineup_dynamics.md) | Fuse MultiScaleGRW with Player Lineup Dynamics | IN_PROGRESS | P1 | claude | 2026-09-11 |
<!-- TASKS:END -->

## Commands and Task Files

Use `NNN_lowercase_slug.md` task names. The CLI is implemented:

```bash
./scripts/todo.sh list
./scripts/todo.sh new "Task title"
./scripts/todo.sh view 001
./scripts/todo.sh check
```

`list` is the default command; colors appear only on a terminal (`NO_COLOR=1`
disables them). `new` creates the next ID using the UTC date with `BACKLOG`, `P2`,
and `unassigned`, and updates the index automatically. Begin manually created
tasks from [`template.md`](template.md), replace its `{{ID}}`, `{{TITLE}}`, and
`{{DATE}}` placeholders, and preserve required metadata and sections. IDs are
`001`–`999`; use the largest existing ID plus one, never recycle completed IDs.
Keep metadata on single-line table rows without literal `|` inside values.
`check` flags a stale registry index or invalid task contract. When creating or
closing a task, update the canonical metadata, matching README row, `Updated`
date, and dated work log together. Close only after acceptance criteria are met:
record verification/findings, set `COMPLETED`, and retain the file in this index.
If work cannot proceed, set `BLOCKED` with the dependency and next action instead
of closing it. Run `check` before handing off or committing.

`new` serializes allocation in this worktree with `todos/.todo-new.lock`. That
lock does not cover separate clones or manual edits: coordinate ID reservations
and resolve collisions before merge. If the lock is stale, first verify no writer
is running, inspect any staging files and reconcile task/index changes left by an
interrupted writer, then remove those staging files and
`rmdir todos/.todo-new.lock`. Existing locks are never auto-cleared. Humans and
agents update README rows with metadata; there is no sync command.

Run the isolated CLI regression tests with `bash test/test_todo.sh`; they create
throwaway tasks in a temporary copy, not in this registry.

## Workflow and Ownership

Statuses move `BACKLOG -> ACTIVE -> IN_PROGRESS -> COMPLETED`, or become
`BLOCKED`. `BACKLOG` is unprioritized/unclaimed work; `ACTIVE` is the prioritized
ready queue; `IN_PROGRESS` is claimed and executing; `COMPLETED` is accepted
with findings recorded; `BLOCKED` awaits an external dependency or decision.
Unblock into `ACTIVE` for a new claim or `IN_PROGRESS` when the owner resumes;
reopen a completed task only with a logged reason and new acceptance criteria.

Priority means `P0` immediate operational/data-integrity risk, `P1` high-value or
near-term work, `P2` normal planned work, and `P3` deferred/nice-to-have work.
Assignee values are `unassigned`, `human`, `antigravity`, `claude`, or `pi`.
Claiming changes metadata and the registry row to `IN_PROGRESS`, records the
assignee plus session/worktree in a dated work-log entry, and is announced to
collaborators. Do not silently steal a task; coordinate a hand-off or unassign it
first.

Add concrete proposed approaches and trade-offs to **Ideas & Candidate Solutions**
before substantial implementation; preserve rejected ideas where they explain a
decision. `docs/tickets/` remains the separate `Txxx` namespace for durable
defects and tracked engineering tickets. Cross-link related TODOs and tickets;
do not migrate or renumber either namespace.

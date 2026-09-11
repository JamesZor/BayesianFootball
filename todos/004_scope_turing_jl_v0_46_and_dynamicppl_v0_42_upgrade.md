# 004 — Scope Turing.jl v0.46 and DynamicPPL v0.42 Upgrade

| Field | Value |
|---|---|
| ID | 004 |
| Title | Scope Turing.jl v0.46 and DynamicPPL v0.42 Upgrade |
| Status | ACTIVE |
| Priority | P1 |
| Assignee | claude |
| Created | 2026-09-10 |
| Updated | 2026-09-10 |
| Related Files / Commits / PRs | `Project.toml`, `src/models/`, `src/samplers/`, `todos/003_benchmark_julia_gc_tuning_and_heap_size_hint_on_multi_core_sampling.md` |

## Context & Problem Statement

Task 003's gradient-path allocation probe on Fold 1 $m_{00}$ revealed that while the compiled ReverseDiff tape evaluates in 0 bytes (0.20 ms) and `LogDensityProblems.logdensity_and_gradient` takes only 5.4 KB, `Turing.sample(NUTS)` allocates 18.55 MB per leapfrog step (~1.1 GiB per iteration, ~742 GiB per 1,000-draw chain). This leads to 68,000+ GC pauses and 40%–60% stop-the-world GC overhead across 16 threads.

The root cause is that the repository is pinned to **Turing.jl v0.41.4** and **DynamicPPL.jl v0.38.10**, which rely on legacy mutable `VarInfo` state containers that box variables and dynamically track names during HMC leapfrog integration. Modern upstream releases (**Turing.jl v0.46** / **DynamicPPL.jl v0.42.12**) overhauled this architecture with `OnlyAccsVarInfo` and `VectorValueAccumulator`, eliminating the `VarInfo` allocation explosion.

Before executing any upgrade, we must thoroughly **scope out the upgrade with NO CODE CHANGES**. We need to map every API break, deprecated method, and internal dependency across the codebase.

## Acceptance Criteria

- [ ] Audit all direct and indirect dependencies that bound `Turing` and `DynamicPPL` in `Project.toml`.
- [ ] Catalog breaking changes between DynamicPPL v0.38 $\to$ v0.42 and Turing v0.41 $\to$ v0.46 (specifically `VarInfo`, `evaluate!!`, `LogDensityFunction`, `AbstractPPL`, `ADTypes`).
- [ ] Identify all repository files in `src/` (e.g. `src/models/`, `src/samplers/`, `src/training/`) that use `DynamicPPL`, `VarInfo`, or `Turing` internals.
- [ ] Verify compatibility of custom likelihoods (`JointGammaPoissonObservation`, `PlayerLineupPillar`, `CountModelBuilder`).
- [ ] Deliver a structured Upgrade Scoping Report in markdown with recommended migration steps, risks, and a verification ladder.
- [ ] **STRICT CONSTRAINT**: No code changes to `src/` or production runners during this scoping phase.

## Ideas & Candidate Solutions

- **Option A: Test bump in isolated branch/worktree**: Use Julia Pkg test environment in a worktree to run `Pkg.up("Turing", "DynamicPPL")` in dry-run/preview mode to see package resolver dependency conflicts.
- **Option B: Targeted internal adapter**: If upgrading Turing breaks custom components, explore wrapping model evaluations with `OnlyAccsVarInfo` directly if supported in intermediate versions.
- **Option C: Staged upgrade path**: Upgrade DynamicPPL to v0.40 first, then v0.42, or jump straight to Turing v0.46 / DynamicPPL v0.42.

## Work Log & Progress

- [2026-09-10 @antigravity] Created Task 004, spun up dedicated git worktree `/home/james/bet_project/.worktrees/BayesianFootball-turing-upgrade` on branch `chore/turing-dynamicppl-upgrade-scope`, and launched dedicated Claude CLI session `claude_upgrade`.
- [2026-09-10 @claude] Claimed Task 004 in tmux `claude_upgrade:0`, operating in worktree `/home/james/bet_project/.worktrees/BayesianFootball-turing-upgrade`. Read-only scoping investigation underway.

## Verification & Findings

- Scoping findings, dependency resolver diffs, and migration surface will be recorded here and in a scoping report document.


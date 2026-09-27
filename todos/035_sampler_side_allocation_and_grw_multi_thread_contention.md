# 035 — Sampler-side allocation and GRW multi-thread contention

| Field | Value |
|---|---|
| ID | 035 |
| Title | Sampler-side allocation and GRW multi-thread contention |
| Status | BACKLOG |
| Priority | P2 |
| Assignee | unassigned |
| Created | 2026-09-27 |
| Updated | 2026-09-27 |
| Related Files / Commits / PRs | TODO 034, `docs/architecture/zero_alloc_engine_report.md` §8–9, `src/harness/perf_monitor.jl` |

## Context & Problem Statement

After TODO 034 the compiled tape allocates 0 bytes, but the smoke performance monitor still flags
`td_base` (GC share 31.6%, in-situ/bare efficiency 3.1). `Profile.Allocs` attributes about
28 KB per leapfrog to AdvancedHMC (`∂H∂r`, `step`, `build_tree` allocate fresh parameter-length
vectors) and about 2 KB per gradient to the DynamicPPL/DifferentiationInterface wrappers.

Separately, GRW cells run 7.3× slower per step at 16 threads than the bare tape, with only 7% GC.
The suspected cause is memory bandwidth or contention on a ~14,800-instruction tape.

All 28 legacy hand-written engines in `src/models/pregame/engines/` keep the scalar-clamp defect.


## Acceptance Criteria

- [ ] Sampler-side allocation reduced (in-place integrator state, a thin `LogDensityProblems`
      wrapper around the compiled tape, or GC settings), with before/after smoke performance reports.
- [ ] The cause of GRW 16-thread contention identified (hardware counters or tape layout) and either
      fixed or documented.
- [ ] Legacy engines: fix the scalar clamp, or record a decision not to.


## Ideas & Candidate Solutions

- `LinuxPerf.jl` for cache and bandwidth counters; JET `@report_opt`; flame graphs in a separate tools environment (TODO 034 rev2 item 7).

## Work Log & Progress

- [2026-09-27 @claude] Created from the TODO 034 block, with the human's decision (options 1 + 3).

## Verification & Findings

Not run yet. Record commands, pass/fail or skips, wall time, benchmark metrics,
R-hat / bulk and tail ESS / divergences / predictive scores as applicable, and
links to artifacts and immutable run IDs. Distinguish hypotheses from findings.

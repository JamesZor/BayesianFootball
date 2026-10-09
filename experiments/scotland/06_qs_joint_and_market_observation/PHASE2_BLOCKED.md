# Phase 2 — historical recipe-loading failure (2026-10-09)

Manager subsequently authorised the one-line loader fix and fresh smoke retry;
see DECISIONS_LOG.md/PROGRESS.md for current status. Original failure follows.

Source `73147ce0`, fresh owned beast REPL `%374`, launcher `%373`, 16 threads,
BLAS 1. First grw_joint entry point failed at 17:04:45 UTC, include wall
14.675230026245117 s:

```text
LoadError: UndefVarError: include not defined in Main.Wave2Candidates
candidates.jl:5
r03_smoke.jl:15 — Base.include(W2_SMOKE_CANDIDATES, candidates.jl)
```

The runner and generic harness load candidates into a module constructed with
`Module(:name)`. Unlike `module ... end`, this has no local `include` helper.
The new candidates file uses an unqualified include at line 5. This is a loader
bug, not evidence against the model/likelihood or the accepted Phase 1 results.

Proposed surgical fix: change that one line to
`Base.include(@__MODULE__, joinpath(@__DIR__, "l01_market_table.jl"))`.
The table adapter still includes wave 1 into Main.QSMarketArm; no original
provenance guard, saved manifest, table, priors, thresholds or seeds change.
Await manager authorisation before the fix/retry; new evidence names must retain
the failed attempt rather than overwrite it.

No candidate built, datastore preflight, canonical registration, smoke hard
check, model fit, performance probe or database write started. Therefore no
smoke UUID, convergence statistics or grid ETA exists. The launcher stopped;
other three arms did not run. No grid executed or authorised.

Phase 1 remains PASS at 153239a7 (evidence e86a03fe): all amended component,
AD/FD, recovery, allocation, builder and harness gates accepted. No rerun needed.

Evidence: logs/phase2_smoke_grw_joint.summary.txt and phase2_launcher.log;
raw log on beast /root/BF_runs/logs/qs_experiment_w2/phase2_smoke_grw_joint.log.
Initial shell-only no-source launch preserved separately as
logs/phase2_launcher0_no_source.log. All owned idle panes closed after retrieval;
other sessions untouched. See PROGRESS.md/DECISIONS_LOG.md for current state.

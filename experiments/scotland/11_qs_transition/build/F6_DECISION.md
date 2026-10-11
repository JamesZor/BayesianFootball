# F6: cadence and clock are currently tied

`DataScope` exposes one `dynamics_col` (`src/Data/splitting/scoped.jl:8–42`).
`_scope_grouped` uses that same column to build fold periods (`:47–59`).
`create_id_boundaries` derives the held-out rows from those periods (`:209–242`),
and the scoped feature builder uses `splitter.scope` to create the target-state clock
(`src/features/builder.jl:175–187`). `Harness.fit_config` always constructs
`ScopedWalkForwardCV(candidate.scope)` (`src/harness/candidate.jl:103–122`).

Therefore changing the recipe to `:match_week` changes BOTH refit cadence and QS
innovation cadence; that is not the requested one-factor weekly-refit arm.
No F6 src change or falsely labelled weekly arm has been made.

Cheapest separable alternative for manager review: add an opt-in fold-cadence field
(defaulting to the existing dynamics clock), use it only for boundary/held-out selection,
and keep the existing biweekly state indices in the feature builder. Weekly folds inside
one biweek must forecast from the same state rather than automatically draw a full new
biweek innovation; that forecast rule needs a dedicated test. Count the actual nonempty
weekly folds on the frozen W2 panel before approving its budget. Do not assume 120 folds.

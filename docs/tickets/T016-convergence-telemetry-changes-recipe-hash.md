# T016 — Convergence telemetry changes the inference recipe hash

- **Status:** open
- **Raised:** 2026-09-24 during TODO 028
- **Severity:** medium
- **Area:** `src/training/inference/{engine,db_storage}.jl`

## Evidence and root cause

`engine.jl:567` appends `"convergence:FAIL"` to the output FitConfig tags when
its diagnostics fail. `db_storage.jl:97-114` strips `time:`, `folds_failed:` and
`latents:` telemetry before computing `configs.config_hash`, but does not strip
`convergence:FAIL`.

Therefore the same submitted recipe acquires a different database deduplication
hash depending on its convergence outcome. A pre-sampling lookup using the
submitted configuration misses an already persisted failed-convergence run.
The row can still have `runs.status = 'completed'`: completed execution is not a
passing convergence audit.

## Minimal reproduction (no database or MCMC)

After loading BayesianFootball:

```julia
I = BayesianFootball.Training.Inference
@assert I._db_recipe_tags(["smoke"]) == ["smoke"]
@assert I._db_recipe_tags(["smoke", "convergence:FAIL"]) ==
        ["smoke", "convergence:FAIL"]
```

These unequal tag sequences enter the canonical hash string for otherwise
identical configurations. A regression test should additionally construct Fits
that differ only in this telemetry and compare their `config_hash` values.

## Blast radius

- Preflight deduplication can unnecessarily repeat failed MCMC recipes.
- Sampling outcome, rather than only the approved recipe, affects run identity.
- Removing the tag in new code alone will not find old rows stored with the
  outcome-dependent hash.

## Proposed fix and trade-offs

Classify convergence telemetry consistently with other runtime tags. Define a
shared public pre-execution recipe-hash function rather than reproducing the
canonicalisation in runners. Provide a compatibility lookup or migration plan
for existing failure-tag hashes; do not rewrite historical UUIDs or silently
collapse distinct historical runs.

TODO 028 uses an experiment-local compatibility lookup for both unmarked and
failure-marked recipes until this is fixed. A reused failed run must remain
non-promotable; deduplication is not permission to ignore diagnostics.

## Acceptance criteria

- Runtime/convergence telemetry does not alter an inference recipe's identity.
- Model, splitter, sampler and scientific-tag changes still alter the hash.
- Preflight locates both legacy failure-marked and canonical stored recipes.
- Existing UUIDs and artifacts remain accessible without destructive migration.
- Tests cover preflight versus save-time identity for passing and failing audits.

## Scope guard

Do not change model equations, convergence thresholds, sampling budgets, or
historical run diagnostics. No database schema/data mutation is authorized by
this ticket alone.

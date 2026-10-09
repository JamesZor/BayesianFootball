# Phase 2 — persistence parity BLOCKED (2026-10-09)

Source ac30c4d7. Fresh sequential beast REPLs %376–%378, 16 pinned threads,
BLAS 1. Manager-authorised loader correction works. All earlier failures retained.

## Partial smoke results (NOT grid results)

| Arm | Hard checks | Max Rhat | Min bulk / tail ESS | Divergences | Launcher wall | Probe grid ETA |
|---|---|---:|---:|---:|---:|---:|
| grw_joint | 6/6 pass | 1.05414 | 182.89 / 69.66 | 0 | 218 s | 1.937 h |
| qs_joint | 6/6 pass | 1.07449 | 133.14 / 79.94 | 0 | 227 s | 1.955 h |
| grw_marketobs | 5/6 pass; fit_parity FAIL | 1.13787 | 16.10 / 57.47 | 0 | 271 s | 4.219 h |
| qs_marketobs | not started | — | — | — | — | — |

All three convergence/performance diagnostics are flagged review (stored status
fail, severity review). No thresholds changed or flag hidden. Joint sampling-adapter
walls 170.21/177.45 s exclude package/recipe startup; launcher entry walls above
include process startup (failed include alone: 263.02 s). Largest-fold 16x(50+50) probe extrapolation is NOT a guarantee;
no complete four-arm ETA yet. Market-observation probe GC share 36.2%, GC-stall
share 56.4%, efficiency ratio 4.741; tape hard check still 0 B. Full reports in
logs/grw_joint.md, qs_joint.md and grw_marketobs.md.

Saved smoke fit UUIDs (sampler completed does NOT imply gate accepted):
- grw_joint: e169aa8e-2302-4c9c-9e39-6c3b1e93218b
- qs_joint: 032bd044-1f10-4e71-9876-f586a04eaa48
- grw_marketobs: 8ff19fe2-8a83-4a50-b964-8b8ac617e316 (parity failed)

The market-observation failure is `HarnessCheckError("fit_parity", "FitConfig changed on reload")`
at checks.jl:51/stages.jl:193. Gradient, tape_allocation, filtration, latents and
score_grid_coherence passed. Failure rows have null run_id because the harness
catch writes before success stamping; the read-only audit identifies the saved fit.

## Read-only diagnosis, before any correction/retry

Audit source 58665779, fresh REPL %379. No sampling or database mutation.
The helper `_structural_equal` has no AbstractDict case: it descends into the
Dict hash-table implementation fields instead of comparing active key/value entries.
MarketRatesFeature is the first relevant injected dictionary in these recipes.

Confirmed in results/parity_audit.csv and parity_config_fields.csv:
- 1,430 entries; every original key/value tuple equals the in-memory serialized
  clone AND the actual database-reloaded feature dictionary (`isequal == true`).
- `_structural_equal` returns false for both dictionaries and their containing model.
- A serialization-only clone of the full smoke FitConfig reproduces the failure;
  all non-model config fields pass unchanged. No sampler/inversion drift needed
  to reproduce it. The dictionary's internal storage is not recipe semantics.
- The gate short-circuited on config; later chain/latent/diagnostic parity checks
  have NOT been accepted for this failed arm. Do not claim whole-fit parity.

Proposed fix for manager review: a type-preserving AbstractDict branch comparing
exact key sets and recursively comparing each value; regression tests must reject
changed rates, flags and keys, while accepting a faithful serialization round trip.
No gate weakening/bypass or scientific-input change has been implemented.

Also flag eligibility policy: `has_passing_smoke` currently rejects ANY historical
failed hard row for the same recipe hash. A faithful passing retry would remain
blocked by this retained failure. Manager must specify how a passing complete
attempt supersedes an earlier failed attempt; do not delete or relabel history.

## Stop state

No sampling retry, parity fix, eligibility fix, fourth smoke or grid started.
Phase 1 remains PASS under all amendments; do not rerun it. Frozen table SHA256
and original saved manifest SHA256 rechecked unchanged. All owned panes closed.
TODO 041 BLOCKED on manager parity/eligibility review. Evidence: results/smoke_*,
results/parity_*, logs/phase2_launcher_v2.log and market-observation summary.

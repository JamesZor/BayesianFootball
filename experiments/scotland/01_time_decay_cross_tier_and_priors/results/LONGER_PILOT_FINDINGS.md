# Longer-chain pilot — acceptance 0.65

## Result

The user-approved longer pilot resolves the short smoke's **R-hat and ESS
failures for all five candidates**. Three candidates pass the full unchanged
six-part convergence audit. The two all-SPFL candidates still fail the strict
zero-divergence requirement.

Run directory: [`longer_pilot/20260925_001951/`](longer_pilot/20260925_001951/),
using the remote host's execution timestamp. Complete machine-readable metrics,
recipe hashes, source/data digests and prior evidence are in
[`pilot_results.json`](longer_pilot/20260925_001951/pilot_results.json).

## Configuration and comparability

- One fold, same **20 held-out fixtures** beginning 2024-08-03.
- Same likelihood data: 720 lower-tier observations for m00/B1/B2;
  1,476 all-SPFL observations for A1/A2.
- **4 chains × 1,000 retained draws**, **500 adaptation iterations**, **0.65
  target acceptance**, max depth 10.
- All model equations, prior inputs and convergence thresholds unchanged.
- mcmc-beast, Julia 1.12.6, 16 pinned Julia threads, BLAS threads = 1;
  candidates run sequentially with four concurrent chains per candidate.
- Exactly one longer-pilot attempt per candidate; no seed retries. Seeds
  2807–2811 are recorded in the recipe.

The earlier 0.85 proposal came from WORK_PACKAGE_PROMPT Step 4. The user's
2026-09-24 correction to **0.65** is now explicit in this pilot and the prepared
production recipe. No production grid was executed.

## Unchanged six-part convergence audit

| Candidate | Max R-hat | Min bulk ESS | Min tail ESS | Min BFMI | Divergences / 4,000 | Full audit |
|---|---:|---:|---:|---:|---:|---|
| m00 control | 1.00375 | 1,513 | 1,873 | 0.739 | 0 | PASS |
| m01 league offsets | 1.00839 | 1,335 | 1,672 | 0.680 | 22 (0.550%) | FAIL |
| m02 hierarchical tiers | 1.00566 | 1,271 | 1,465 | 0.605 | 1 (0.025%) | FAIL |
| m03 structural prior | 1.00471 | 1,378 | 1,878 | 0.708 | 0 | PASS |
| m04 market prior | 1.00424 | 1,299 | 1,541 | 0.718 | 0 | PASS |

All treedepth-cap rates are zero, and no convergence diagnostics abstained.
Thresholds remain R-hat <1.05, bulk/tail ESS >100, divergence rate <eps(),
BFMI >0.30, and treedepth-cap rate <0.05.

Low R-hat and large ESS do not override divergences. These findings support
insufficient sampling as the cause of the earlier R-hat/ESS failures, but do
not establish safe production sampling for A1/A2 across 40 folds.

## Persistence and remaining gates

All candidates extracted **20 × 4,000 CountLatents draws** and passed exact
PostgreSQL save/load parity. The runner's **55/55** assertions concern
filtration, sampler dimensions and persistence—not five passing convergence
audits.

| Candidate | Immutable pilot Fit UUID |
|---|---|
| m00 | `e1e154ec-7693-4879-ad4d-e48693beb309` |
| m01 | `52142683-3291-4c15-acf1-6aeaf38c6c94` |
| m02 | `f97fc57d-7953-4f5f-86ad-ddc4c895eb4d` |
| m03 | `365d3e08-d2fb-4c75-a680-026bd9387fea` |
| m04 | `9c809ce5-cfc4-47ca-9ea3-ec7373f46095` |

Namespace: `scotland_cross_tier_time_decay`. Include the prototype loader before
loading these Fits. They are separate pilot recipes; the original 2×100 smoke
records have not been replaced or retrospectively marked as passed.

Remaining independent blockers:

- **Tape allocations persist for every candidate**: 64,152 / 131,032 / 135,832 /
  65,688 / 65,688 bytes for m00–m04. A1/A2 replay latency also remains above
  0.05 ms. Gradient correctness probes pass.
- The same per-draw 12×12 grid mass floor (0.999) fails for m00, m01, m03 and
  m04 in this larger draw panel; m02 passes. The first failing masses are
  0.9989267, 0.9989849, 0.9988990 and 0.9989913, respectively. More draws expose
  additional tail cases; they do not repair finite-grid truncation.
- No portfolio execution or seven-gate promotion manifest was produced.

**Production remains blocked**, despite the substantial improvement in R-hat
and ESS. Next work is divergence investigation for A1/A2, tape allocation
removal, and an explicit score-grid tail policy—not additional seed retries or
silently loosening thresholds.

## Reproduction

```bash
# In /root/BayesianFootball-scotland-cross-tier on mcmc-beast:
JULIA_PKG_PRECOMPILE_AUTO=0 julia --startup-file=no --project -t 16 \
  experiments/scotland/01_time_decay_cross_tier_and_priors/r06_longer_chain_pilot.jl
```

Exact recipes are registered and checked before sampling; a repeat invocation
reuses saved Fits rather than repeating their chains. Source digest:
`4593e4ec88f3980b0ea23b564f0644a9ddef65cd652e5d7a382e4f0ac909a7d1`.
The authorized isolated rsync checkout remains uncommitted and has no `.git`;
source/data hashes are recorded instead of pretending it is a clean Git run.

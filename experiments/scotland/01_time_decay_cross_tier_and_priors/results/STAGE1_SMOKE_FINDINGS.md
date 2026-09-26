# TODO 028 — Stage 1 smoke findings: BLOCKED

> **Follow-up:** the separately approved 4×1,000 pilot at acceptance 0.65 now
> passes R-hat/ESS for all five, with three full convergence passes. A1/A2
> retain divergences; see [LONGER_PILOT_FINDINGS.md](LONGER_PILOT_FINDINGS.md).
> The original smoke evidence below is preserved unchanged.

## Executed, not promoted

All five candidates completed **2 chains × 100 retained draws**, with 100
adaptation iterations, on `mcmc-beast` (Julia 1.12.6, 16 pinned Julia threads,
BLAS threads = 1). The latest corrected execution is
[`smoke/20260924_235900/`](smoke/20260924_235900/).
Its timestamp is the remote host's timestamp.

**No candidate passed all seven gates. No production grid was launched.**
No thresholds or sampling budgets were relaxed, and no portfolio was executed
from these unconverged fits. Stage 2/3 and predictive/portfolio superiority
claims remain unavailable.

## Exact gate outcomes

P = passed; F = failed; B = not run because blocked by failed convergence.

| Candidate | G1 tape | G2 sampling | G3 audit | G4 CountLatents | G5 grid | G6 PostgreSQL parity | G7 portfolio |
|---|---|---|---|---|---|---|---|
| m00 control | F | P | F | P | P | P | B |
| m01 league offsets | F | P | F | P | P | P | B |
| m02 hierarchical tiers | F | P | F | P | P | P | B |
| m03 structural prior | F | P | F | P | F | P | B |
| m04 market prior | F | P | F | P | F | P | B |

### Gate 1: correct gradients, nonzero allocation

Linked unconstrained log density includes the Jacobian, matching Turing's NUTS
Hamiltonian (`getlogjoint_internal`). Compiled/fresh ReverseDiff and ForwardDiff
agree at the reference point; the compiled/fresh check also passes perturbations
up to ±0.8 per coordinate multiplier. Performance still fails:

| Candidate | Minimum warmed replay (ms) | Allocated bytes |
|---|---:|---:|
| m00 | 0.031238 | 64,152 |
| m01 | 0.065361 | 131,032 |
| m02 | 0.057898 | 135,832 |
| m03 | 0.031098 | 65,688 |
| m04 | 0.030978 | 65,688 |

The work-package gate requires **<0.05 ms AND zero bytes**. All five fail the
allocation requirement; A1/A2 additionally fail latency. These are measured
failures, not an assertion that the engines are allocation-free.

### Gate 3: strict six-part audit

Values were independently reloaded from PostgreSQL and re-audited without
sampling by `audit_saved_smoke.jl`. Full data and per-club prior evidence:
[`saved_fit_audit.json`](smoke/20260924_235900/saved_fit_audit.json).

| Candidate | Max R-hat | Min bulk ESS | Min tail ESS | Min BFMI |
|---|---:|---:|---:|---:|
| m00 | 1.0537 | 126.9 | 73.1 | 0.640 |
| m01 | 1.0642 | 33.6 | 57.7 | 0.776 |
| m02 | 1.0993 | 91.1 | 46.1 | 0.972 |
| m03 | 1.0560 | 35.6 | 24.6 | 0.640 |
| m04 | 1.0649 | 85.5 | 64.1 | 0.746 |

All five have **0 divergences** and **0 treedepth-cap rate**. Nevertheless,
R-hat and tail ESS fail for every candidate; bulk ESS also fails for m01–m04.
The thresholds remain R-hat <1.05, bulk/tail ESS >100, divergence rate <eps(),
BFMI >0.30, and treedepth-cap rate <0.05. This short run does not establish
whether longer runs would converge.

### Gates 4–7

All five now extract **20 held-out fixtures × 200 posterior draws** as
CountLatents and pass exact PostgreSQL save/load parity. The 12×12 grid's
implemented mass check requires at least 0.999 mass per draw, in addition to
finite probabilities and coherent 1X2/totals/BTTS partitions:

- m03 first failing grid: fixture index 1, mass **0.9953675057**.
- m04 first failing grid: fixture index 5, mass **0.9980655404**.

These are the first reported failing draws, not necessarily the minimum over
the entire posterior. No grid-size or tolerance change was made to turn them
into passes. Gate 7 abstains rather than pricing/persisting portfolios with
failed convergence. Therefore there are **no portfolio UUIDs** for this run.

## Persisted diagnostic Fit addresses

Namespace: `scotland_cross_tier_time_decay`. These are completed **smoke** fits
with failed convergence, not canonical production fits.

| Candidate | Run UUID |
|---|---|
| m00 | `bbd9b8c1-7fa8-409a-95dc-51581d0f4fb8` |
| m01 | `624bbbc6-d900-486b-9f47-fb69c0336b87` |
| m02 | `58546775-5771-4bd1-8948-a0b1da2f352c` |
| m03 | `d32da218-d9ec-451a-b9fe-04234d93b0d6` |
| m04 | `7fc0643b-14c5-424c-9ed1-32411ab3e076` |

The JSON audit also records each `configs.config_hash`. Include the prototype
loader before deserialization. Failed-convergence preflight compatibility is
tracked in [T016](../../../../docs/tickets/T016-convergence-telemetry-changes-recipe-hash.md).

## Filtration and prior evidence

All models predict the same 20 lower-tier fixtures beginning **2024-08-03**.
m00/B1/B2 fit exactly **720** lower-tier observations; A1/A2 fit **1,476**
all-SPFL observations whose lower-tier subset is identical. All fitted dates
strictly precede the cutoff. The full lower calendar has independently passed
**40 folds / 710 unique OOS IDs**, with no overlap or duplicate OOS IDs.

- A2 historical anchor counts, tiers 1–4: **12 / 10 / 10 / 11**; **301** fitted
  fixtures pair clubs with different historical anchor tiers. This is evidence
  of connecting fixtures, not proof of well-identified posterior tier steps.
- B1 identifies **Arbroath and Inverness** as cold relegated entrants from
  prior-season upper-tier evidence; both receive the specified structural prior.
- B2 accepts the fixed five-fixture prior-season evidence for **Inverness**.
  **Arbroath falls back** because at least one fixed last-five market inversion
  is missing/refused. No sixth fixture is substituted. Exact IDs are in JSON.
- B2 requires complete home/draw/away probabilities before applying the existing
  inversion's convergence/residual/rate-bound gates, avoiding the known T015
  totals-only identification loophole.

## Execution and provenance

Remote directory: `/root/BayesianFootball-scotland-cross-tier`, an authorized
isolated rsync copy. No commit/push or modification of the existing Julia REPL.
The remote copy has no `.git`; Fit Git metadata is therefore unknown, not a
claim of a clean committed run. Local base HEAD:
`b168635368497e66dbad05f47068d6f9b27e4a7d`, branch
`feat/scotland-cross-tier-models`, with uncommitted changes.

Corrected smoke source digest (l01, l02, r01, segments joined with U+001E):
`35e0f2bffb3d8e93f21e2ab7eb6d65798213f5b06571f2e9a5aa88f01004f220`.

```bash
# On the isolated remote copy; runs only the strict smoke, never production:
CT01_PREPARE_ONLY=false JULIA_PKG_PRECOMPILE_AUTO=0 \
  julia --startup-file=no --project -t 16 \
  experiments/scotland/01_time_decay_cross_tier_and_priors/r01_smoke_test.jl
```

Earlier executions remain in `results/smoke/`: an initial reporter-constructor
failure, an incorrect DynamicPPL getter, and a first full smoke exposing an A2
extraction transpose and a diagnostic namespace error. These implementation
errors were corrected, with A2 two-chain extraction regression coverage, before
the final execution above. Earlier runs are not promotion evidence.

Production was tested in **prepare-only** mode against the final smoke directory:
its sampler API tests pass **7/7**, and it refuses the missing promotion manifest
before sampling. No manifest is emitted for a failed smoke candidate.
Local focused tests: data module **55/55**; loader deterministic contracts
**15/15**. These do not replace the failed real-data gates or a full independent
log-density/filtration regression suite.

## Required next decisions/work

1. Profile and remove tape replay allocations without changing the equations;
   also bring A1/A2 below the latency bar.
2. Obtain researcher approval for any longer one-fold convergence validation
   protocol; do not relabel these exact 2×100 chains as converged or simply
   retry random seeds until green.
3. Resolve the prior candidates' score-grid tail coverage with a documented
   grid/pricing policy, not a relaxed tolerance chosen after observing failures.
4. Only after all required smoke gates pass, validate all-candidate production
   preparation and launch the native 40-fold queue. Stage 2/3 remain deferred.

# Momentum MultiScale GRW — Scottish Lower Phase 1

**TODO [022](../../../todos/022_prototype_momentum_multiscale_grw_dynamics.md)** ·
namespace `scottish_lower_momentum_grw` · **IN PROGRESS**

## Scientific question

Does damped team velocity improve favourite-tail forecasts and proper scores
relative to pure-Poisson TimeDecay(180) and first-order MultiScaleGRW? The matched
cohort is 40 pooled-56/57 match-biweek folds, seasons 24/25–25/26, 710 fixtures.
No player, wealth, proxy-xG or smile terms are included.

See [DESIGN.md](DESIGN.md) for equations, priors, boundary conditions, stationarity
analysis and AD implementation. Velocity is stable for phi < 1; the level retains
a unit root. First-order GRW has zero expected increments, not a level that resets
to zero. Momentum is a hypothesis, **not a guarantee of decompression**.

User-approved forecasting: zero velocity at the target-season boundary;
conditional-mean next-biweek states, without future innovation noise, matching
the first-order control's convention. The terminal unobserved velocity innovation
is integrated out; K≤1 folds have no velocity parameters.

## Stage 0 — measured verification (2026-09-21)

Executed on `mcmc-beast`, `/root/BF_momentum_grw`, 16 pinned threads, BLAS=1.
`test_momentum.jl`: **200/200 assertions passed**. Tests cover independent scalar
recurrences, sigma_v=0 nesting, phi endpoints, no-target/single-target shapes,
centering, multi-chain draw ordering, Turing-returned states, OOS reconstruction
and compiled/ForwardDiff gradient parity. Isolated momentum replay: **0 B**.

Full **linked-space** gradient replay, warmed minimum of 100 calls:

| Arm | Fold | Parameters | Tape instructions | Gradient ms | Allocated B |
|---|---:|---:|---:|---:|---:|
| TimeDecay | 1 | 50 | 196 | 0.0370 | 0 |
| TimeDecay | 20 | 54 | 196 | 0.0506 | 0 |
| TimeDecay | 40 | 50 | 196 | 0.0509 | 0 |
| First-order GRW | 1 | 98 | 610 | 0.0532 | 0 |
| First-order GRW | 20 | 1058 | 8284 | 0.3220 | 0 |
| First-order GRW | 40 | 974 | 7644 | 0.3007 | 0 |
| Momentum GRW | 1 | 98 | 610 | 0.0531 | 0 |
| Momentum GRW | 20 | 1962 | 15580 | 0.5662 | 0 |
| Momentum GRW | 40 | 1806 | 14364 | 0.5247 | 0 |

Original and optimized engines have identical sampled-site layouts and prior
initializations. Density discrepancies ≤ **4.7e-10**; gradient relative errors ≤
**3.6e-15**, comparing compiled, fresh ReverseDiff, ForwardDiff, and the original
engine, at linked-space displacements 0, 0.003, ±0.8 and ±3. No allocation
threshold was waived. Compilation/setup allocations are not replay allocations.

The source controls allocate 35–52 KB per replay. The prototype removes scalar
broadcast scratch, `fill` scratch, and scalarizing keyword centering without
changing priors, sample sites, weights or clamp limits. All modifications remain
local; [T002](../../../docs/tickets/T002-scalar-taped-likelihood.md) records the
shared-engine findings. The custom scalar-lift instruction is version-sensitive.

Prepare-only smoke preflight passed: exact 40/710 cohort, ordered filtration on
folds 1/20/40, all nine AD gates, canonical recipe registration and run-hash
lookup. No matching completed smoke runs existed at that preflight.

## Stage 1 — smoke protocol

Three arms, each 4 chains × (400 warmup + 400 retained), target acceptance 0.90,
max tree depth 10. Native queue, at most 16 concurrent chain tasks. Gates:

- zero divergences, max R-hat ≤1.05, bulk/tail ESS ≥200;
- six-part audit also requires BFMI ≥0.30 and tree-depth saturation ≤5%;
- exact OOS coverage, finite positive rate draws and coherent market partitions;
- fit/chain/latent database round-trip; portfolio persistence and identical
  re-priced bet ledger after loading the fit;
- report phi/sigma_v posterior summaries against their priors (identification is
  measured, not presumed; no velocity posterior exists on fold 1).

**User-approved score-grid clarification:** production scores truncate each side
to 0–11 goals without renormalization. Check all 1X2/OU2.5/BTTS partitions against
`cdf(Poisson(lambda_h),11)*cdf(Poisson(lambda_a),11)` within **1e-12** on every draw;
report omitted tail mass separately. This is not a claim that truncated mass is 1.

Only a passing, source-matched smoke certificate permits production. Failed fits
may be persisted for diagnostics, but cannot enter portfolio promotion.

## Results

**No sampling, predictive-performance or portfolio results are claimed yet.**
Stage 2 (40-fold production), Stage 3 (scoring/decompression/portfolio comparison)
and completion sign-off remain gated on Stage 1.

## Reproduction

```bash
cd /root/BF_momentum_grw
# Load environment without printing credentials.
set -a; . ./.env; set +a
J=/root/.juliaup/bin/julia
D=experiments/scottish_lower/10_momentum_multiscale_grw
$J --project -t 16 --startup-file=no "$D/test_momentum.jl"
$J --project -t 16 --startup-file=no "$D/r00_momentum_preflight.jl"
MMG_PREPARE_ONLY=true $J --project -t 16 --startup-file=no "$D/r10_momentum_smoke.jl"
$J --project -t 16 --startup-file=no "$D/r10_momentum_smoke.jl"
```

Artifacts are under `results/smoke/<source SHA256>/`; recipes/checkpoint directories
include source identity. The `MomentumGRW` module must be included before loading
prototype fits from PostgreSQL. DataStore caches and binary fits are not committed.

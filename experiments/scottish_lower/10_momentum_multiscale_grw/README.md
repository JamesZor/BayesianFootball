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
`test_momentum.jl`: **202/202 assertions passed** (including the strict-zero audit adapter regression). Tests cover independent scalar
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

### Measured Stage 1 outcome — PASS

All nine arm/fold combinations passed; 50 OOS fixtures, 1,600 draws per fixture.
All full-tape AD gates stayed allocation-free. No convergence metric abstained.

| Arm | Max R-hat | Min bulk ESS | Min tail ESS | Divergences | Min BFMI |
|---|---:|---:|---:|---:|---:|
| TimeDecay | 1.0118 | 552.8 | 578.4 | 0/4800 | 0.682 |
| First-order GRW | 1.0252 | 399.0 | 533.8 | 0/4800 | 0.668 |
| Momentum GRW | 1.0209 | 450.4 | 359.9 | 0/4800 | 0.749 |

No tree-depth saturation. The advisory R-hat≤1.01 bar was not met by every fold;
the specified acceptance bar remains ≤1.05. Maximum score-partition error was
1.78e-15; worst omitted tail mass was 0.000510 / 0.000384 / 0.001766 respectively.

**Audit correction, not resampling:** the initial runner mistakenly passed zero
to the shared audit's strict `<` divergence threshold, causing `0 < 0` failures.
It now uses the smallest positive Float64, plus an explicit zero-count check.
`r12_reaudit_smoke.jl` re-audited the exact original chains, preserved the original
UUIDs, and proved unchanged draw arrays. Both zero- and one-divergence cases are
regression-tested. No convergence requirement was relaxed.

**User-approved common portfolio panel:** 44/50 fixtures. Five lacked closing
quotes (`12477130`, `12476630`, `12476458`, `12476570`, `14032714`); `12473327`
had no usable selection. Every arm had exactly these refusals, with no construction
errors. All 50 remain in latent/grid checks. Fit/chain/latent database round-trips,
portfolio persistence and re-priced ledger equality passed for every arm on the
same 44-fixture panel. Production will also report a common tradeable panel rather
than silently exclude different fixtures per arm.

| Arm | Accepted smoke run UUID | Portfolio UUID |
|---|---|---|
| TimeDecay | `a05fb858-8033-4d4f-a805-509c5b5daab4` | `0adf69d9-9bc5-41ba-84b5-40b2bceb9e53` |
| First-order GRW | `02c1d10a-515f-4592-ba97-c895f8b38895` | `b2ace66e-57ca-4466-b9b6-68a307ce01f8` |
| Momentum GRW | `112bb865-c0e5-470a-b53a-619909367ced` | `9a1fcfde-fb0b-4f2f-9cdc-9ee6d4af4b83` |

Certificate source: `6869de6297b580cdc7aaf3052ba31c81c1facc0bc826d495e770bfbbb4a73c6f`.
Auditable CSVs, including original UUID lineage, are in [verification/stage1/](verification/stage1/).

Momentum persistence is **weakly identified**: posterior phi SDs 0.199–0.219 versus
prior SD 0.224. Attack sigma_v means are 0.00935/0.01182 on folds 20/40 versus prior
0.015; defence means 0.01064/0.01024 versus prior 0.012. On this small smoke panel,
supremacy slopes are 0.249/0.358/0.388. Only **three** quoted favourites meet ≥0.70;
their model probabilities are 0.518/0.511/0.521 versus market 0.750. These are
mechanical diagnostics, not evidence of a full-cohort performance win.

## Stage 2 — production protocol

The all-fold prepare-only run passed: 40 folds, 710 fixtures, source-matched smoke
certificate, canonical recipes and no existing completed production matches.
Budget stays 4×(800 warmup + 800 retained), 16 concurrent chain tasks.

**User-approved storage contract:** all 3,200 retained draws per fold are audited
and saved in full local fits; PostgreSQL stores every fourth draw (800 per fold),
with latents reconstructed from those exact draws. The same stride applies to all
arms. Full-draw diagnostics are retained. This avoids the current hex-encoded
single-artifact limit; the momentum smoke artifact alone is 49 MB. Stored recipe
descriptions explicitly record the persistence stride. Stage 3 compares the
matched persisted panels. Storage preflight passed on all three smoke fits:
reconstructed thinned rates equal the exact original draw columns; full-draw
diagnostics are preserved.

**Stage 2 launched** at commit `a6cca1bc` in beast tmux session
`momentum_production`. Log: `results/stage2_production.log`; exit status is written
to `results/stage2_production.exit` on termination. Production output is under
`results/production/6869de6297b580cdc7aaf3052ba31c81c1facc0bc826d495e770bfbbb4a73c6f/`.
Do not launch a duplicate queue. Completion, convergence and database round-trips
are still pending. No full-cohort performance result is claimed yet.

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

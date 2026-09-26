# Scottish cross-tier time-decay experiment — TODO 028

## Status

**BLOCKED after the real five-candidate smoke on mcmc-beast.** All five complete
2×100 sampling, CountLatents extraction and PostgreSQL round-trip parity, but
all fail tape allocation and strict convergence gates. B1/B2 also fail the
implemented score-grid tail-mass check. Portfolios were not run, and the
40-fold production grid was **not launched**.

Original smoke table and Fit UUIDs:
[`results/STAGE1_SMOKE_FINDINGS.md`](results/STAGE1_SMOKE_FINDINGS.md).

**Longer-pilot follow-up:** the user-approved **4×1,000 / 500 adaptation /
acceptance 0.65** pilot resolves R-hat and ESS failures for all five. m00/B1/B2
pass the full convergence audit; A1/A2 fail only its zero-divergence gate
(22 and 1 divergences). Tape allocation and grid-tail blockers remain, so
production is still blocked. All pilot Fits round-trip through PostgreSQL.
See [`results/LONGER_PILOT_FINDINGS.md`](results/LONGER_PILOT_FINDINGS.md).

**No predictive superiority is established.** Stage 2 (joint Gamma–Poisson) and
Stage 3 (lineups) remain deferred. The work-package narrative is motivation,
not evidence that these candidates improve pricing.

## Data verification

Fresh remote SQL collection passed: **4,324** unique fixtures across all four
tiers; odds cover 99.86%, lineups 100%, BBC match commentary 98.36%, and BBC
shot-event commentary 52.41%. Checked domains have no out-of-scope match IDs.
See [`results/data_preflight.md`](results/data_preflight.md) for per-tier counts
and limitations. Local `test/data_tests.jl`: **55/55 passed**;
`test/cross_tier_loader_tests.jl`: **15/15 passed**.
The fresh snapshot reproduces the required **40 folds / 710 distinct lower-tier
held-out fixtures** (360 in 24/25, 350 in 25/26), with no repeated OOS IDs or
train/OOS overlap and strict training-before-prediction cutoffs throughout.
See [`results/fold_inventory.md`](results/fold_inventory.md).

## Frozen Stage 1 decisions (2026-09-24)

All candidates use independent Poisson goals, shared club attack/defence effects,
and 180-day exponential likelihood decay. No Gaussian random walk is used.
Prediction-only club identities enter the prior vocabulary without their
outcomes, consistently across the control and prior candidates.
The comparison uses the Scottish Lower held-out calendar bins with two historical
seasons; only A1/A2 expand likelihood rows to the full SPFL pyramid. All five
read one ScottishAll snapshot for comparability; m00 uses no upper-tier outcomes
or prior evidence, whereas B1/B2 use upper-tier fixtures only as historical
prior evidence.

- **m00:** lower-tier control, zero-centred team-effect priors.
- **m01 / A1:** all-SPFL shared club effects and zero-sum fixture-league scoring
  offsets applied equally to both log rates. These describe scoring environment,
  not a direct opponent-strength advantage.
- **m02 / A2:** ordered nonnegative tier steps, League Two reference zero. A club's
  prior anchor is its **first observed tier in the fitted training window**, held
  fixed across that window (user-approved). Stronger-tier prior locations enter
  attack positively and opponent-defence contribution negatively.
- **m03 / B1:** cold relegated entrants receive a +0.90 attack prior location and
  a corresponding negative defensive location. Cold means fewer than five
  lower-scope fitted matches. **No promoted-club offset** (user-approved).
- **m04 / B2:** the prior location is half the mean **market-implied expected goal
  difference**, oriented to the club, over its last five prior-season
  higher-division fixtures (user-approved). Use closing-market Poisson inversion;
  missing evidence falls back explicitly and must be reported. This quantity is
  relative to higher-tier opponents and can remain negative for relegated clubs.
  It must not be tuned to force an expected live-slate result.

All prior evidence must precede the fold cutoff. Held-out outcomes must never
enter feature construction. Prediction identities may be used without outcomes.
Historical source coverage, fallback counts and the actual evaluation-panel size
must be reported rather than assumed from the work-package's historical totals.

## Execution and promotion

`r01_smoke_test.jl` owns the one-fold seven-gate workflow;
`l01_cross_tier_loader.jl` owns model/feature/split definitions and
`l02_smoke_helpers.jl` owns gate/persistence machinery. Include these prototype
types before reloading PostgreSQL artifacts.

Smoke sampling is exactly **2 chains × 100 retained draws**. The specified
convergence and tape thresholds are not relaxed to turn failures into passes.
Failed or unavailable gates block production. Reports under `results/` will
separate executed checks from skipped or blocked checks.

Execution uses an authorized isolated rsync copy at
`mcmc-beast:/root/BayesianFootball-scotland-cross-tier`, excluding `.cache/`,
`data/` and credentials. No commit/push or changes to the existing remote Julia
session were authorized. Experiment persistence uses namespace
`scotland_cross_tier_time_decay`; operational data are read from `betdb`.

`r02_production_grid.jl` defaults to prepare-only. It requires
`CT02_SMOKE_MANIFEST_DIR` pointing to one smoke execution with all five valid
promotion manifests. Production settings are 1,000 retained draws × four chains,
500 MCMC adaptation iterations, acceptance target **0.65**, and 16 native queued
tasks. The user's 2026-09-24 correction supersedes the work package's Step-4
value of 0.85; no grid has run under either setting. Splitter `warmup_period = 0` is a **calendar setting**, not zero MCMC
adaptation. Execution requires `CT02_RUN_PRODUCTION=true`;
`CT02_PREPARE_ONLY=false` enables sampling only after its gates pass. The
prepare-only production guard was exercised and correctly refused the failed
smoke execution's missing promotion manifests.

BTTS is checked as a score-grid partition in smoke Gate 5; the portfolio book
in Gate 7 deliberately contains **1X2 + O/U 2.5**, matching the work package.
Experiment-local preflight checks both normal and failure-stamped recipe hashes
to mitigate [T016](../../../docs/tickets/T016-convergence-telemetry-changes-recipe-hash.md);
reusing a failed-convergence run never makes it promotable.

# Unified V2 architecture — layer detail

> Extracted from `AGENTS.md` (formerly §2). `AGENTS.md` keeps the layer table, the module
> map and a short calibration summary; this file keeps the per-layer narrative.
> MatchDay (L5) is covered in full in
> [`../guides/matchday_console_guide.md`](../guides/matchday_console_guide.md).

### L0 — Data (`src/Data/`)

SQL → `DataStore` via `LibPQ`. Every data domain (Matches, Odds, Betfair,
Stats, Lineups, Incidents, BBC commentary) passes a strict
**Fetch → Process → QA** three-step contract defined in
`src/Data/fetchers/interfaces.jl`. Tournament segments are singletons
(`ScottishLower`, `Ireland`, …) in `src/Data/fetchers/segments.jl`. The
`Markets` submodule does vig removal, fair odds, and closing-line-movement
math. Cached locally as `.jls` files in `.cache/`.

### L1 — Bayesian engines (`src/models/pregame/`)

Component-driven, assembled from mathematical building blocks:

- **Components** (`components/`): `Interception`, `Dispersion`, `HomeAdvantage`,
  `Dynamics`, `Kappa`, `Copula`, `DixonColes`, covariates
  (`ProductionWealthCovariate`, `DistanceCovariate`, `BenchDepthCovariate`),
  and the `PlayerLineupPillar` that composes RAPM teamsheet ratings beside team
  attack/defence.
- **Observations**: `PoissonObservation`, `NegBinObservation`,
  `JointGammaPoissonObservation` (the two-arm proxy-xG + goals likelihood).
- **Team-level engines** (`engines/team_level/`): goals, xg, copula-goals,
  market variants — split `standard/` and `time_decay/`
- **Player-level engines** (`engines/player_level/`): outfield xg, hierarchical
  player, Dixon-Coles variants — split `standard/` and `time_decay/`

Notable exports: `DynamicGoalsModel`, `DynamicXGModel`,
`DynamicCopulaGoalsTimeDecayModel`,
`DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel` — the local-intensity
per-strike totals "smile" pillar, which prices O/U via its own intensity
`λ_tot·φ(K)` while 1X2/BTTS/CS use the goals grid
(`src/predictions/score_computation/smile_poisson.jl`).

Every model must implement `Features.required_features(model)` returning a
`Vector{Symbol}` declaring the data features it needs.

### L5 — MatchDay (`src/MatchDay/`)

The operational layer. It prices a whole simultaneous fixture slate at a stated
instant, records the planned stake vector in a paper ledger, and serves it to an
operator. **The slate is the execution atom**: `Portfolio` solves one joint
problem for every fixture that settles together, so the stake vector is only
valid *as a vector*, and reservation is one transaction for the whole of it.

```
fixtures → identity → lineups → book → features → inference → gate → stake_sheet
```

Every stage is a seam with an abstract type (`AbstractFixtureSource`,
`AbstractIdentityResolver`, `AbstractLineupSource`, `AbstractBookSource`,
`AbstractGate`, …), which is what lets the replay console swap only the sources
that read a clock or a network while keeping the gates, the instrument rule, the
stake rounding, the market set and the portfolio policy identical to the live
path. Posteriors are never sampled here: `MD.canonical_fit` loads a completed run
out of `mcmc_experiments`. See [`matchday_console_guide.md`](../guides/matchday_console_guide.md).

### Calibration (`src/Calibration/`) — the L2 calibrator tier

**Generative rate calibration.** The tradeable book is inverted back to
`(lambda_mkt_h, lambda_mkt_a)` by Nelder-Mead on
`Features.DoublePoissonMarketFeature`, every posterior log-rate draw is pooled
with it, and the calibrated container is priced through the **same** score-grid
kernels, evaluator and portfolio the raw one goes through:

```julia
book, refusals = point_in_time_book(ds; config = PointInTimeBookConfig(as_of_minutes = -25.0))
cal = GenerativeRateCalibrator(name = "scot_lower_t25_inv",
                               law  = InverseGaussianLaw(w_base = 0.25, sigma = 0.35),
                               book_as_of_minutes = -25.0)
cf  = calibrate_fit(cal, fit, book)                        # -> CalibratedFit
result, books, rep = run_portfolio_simulation(spec, policy, cf, book, ds)
```

1X2, every totals line and BTTS are then three partitions of **one** 12×12 score
tensor, so derivative coherence is structural rather than audited.
`cf.fit` is a real `Training.Fit` carrying the calibrated latents, which is what
lets L3 and L4 consume it with **no change to `src/Portfolio/`**.

Three location laws (`InverseGaussianLaw`, `StandardGaussianLaw`,
`StaticGeometricLaw`) and four dispersion maps (`PoolDispersion` — the default and
the validated production transform — `PreservedDispersion`, `ConjugateDispersion`,
`SupremacyDispersion`). **Which law wins depends on the sharpness of the book being
pooled with, not on the league**: the standard form wins at the Betfair close and
the inverse form at T−25, and a spec transferred between instants gives up
0.0015–0.0020 LogLoss. A calibrator therefore records `book_as_of_minutes` and
`calibrate_fit` refuses a book from a different instant.

Persistence is `config_registry` (`config_type = 'calibrator'`) plus
`calibration_runs` / `calibration_artifacts` in `mcmc_experiments`; a portfolio run
is linked through `portfolio_runs.metadata`, not a foreign key.

Design record: [`docs/architecture/rfc_layer2_calibration_v2.md`](rfc_layer2_calibration_v2.md).
Evidence, including two published conclusions the stream's own later phases
retracted: [`current_development/calibration_generative_eda/README.md`](../../current_development/calibration_generative_eda/README.md).

> **`BasicLogitShift` is DEPRECATED** (`build_l2_training_df` → `train_calibrators`
> → `apply_calibrators`). It fits one GLM offset per selection and applies it
> independently, so `P(over 2.5) + P(under 2.5) != 1` and the shifted board is not
> a scoreline distribution at all. It still runs and warns once per session.

### Meta model (`current_development/MetaModels/`)

In active development. Blends L1 predictions with market-implied probabilities
via a dynamic Gaussian random-walk mixture: `Q_i = θ_t·p_L1_i + (1-θ_t)·m_i`.
Two engine types: `ConvexMixtureMetaModel`, `AffineCalibrationMetaModel`. See
`../archive/meta_model_design.md`.

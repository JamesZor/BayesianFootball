# Extension recipes

> Extracted from `AGENTS.md` (formerly §9). Each recipe is "add a type + its methods";
> none should require editing an existing file beyond the one named.

**New league / segment** — edit only `src/Data/fetchers/segments.jl`: define a
struct subtyping `DataTournemantSegment` and implement
`tournament_ids(::MyLeague) = [id1, id2]`.

**New model component** — three steps in `src/models/pregame/components/`:
1. Define a `Config` struct subtyping the relevant abstract (e.g.
   `AbstractDynamicsConfig`).
2. Write the Turing `@model` builder returning the expected NamedTuple.
3. Write `extract_parameters` to pull variables from `MCMCChains.Chains`.

**New feature extractor** — add
`add_feature!(F_data::Dict, ::Val{:feature_name}, ordered_ids, team_map, ds::DataStore)`
in `src/features/extractors/`, then return the symbol from the model's
`required_features`.

**New backtesting metric** — subtype `AbstractWealthMetric` or
`AbstractDistributionalMetric` in `src/backtesting/metrics/`. See
`hurdle_roi.jl` for the distributional pattern, `implentations/` for wealth
metrics.

**New calibration weight law** — subtype `AbstractCalibrationWeightLaw` in
`src/Calibration/types.jl` and implement `calibration_weight(law, delta)`,
`is_identity_law(law)` and `law_label(law)`. **New dispersion map** — subtype
`AbstractDispersionMap` and implement `residual_map(map, w_h, w_a)` returning the
2×2 map row-major. Either is "add a struct + one method"; no existing file changes.

**New L2 shift model (deprecated path)** — subtype `AbstractLayerTwoModel`,
implement `fit_calibrator(model, data, config)` and
`apply_calibration(fitted_model, new_data)` in `src/Calibration/shift_models/`.
Prefer a weight law or a dispersion map: a selection-level shift cannot be coherent
across derivative markets.

**New MatchDay source or gate** — subtype the relevant seam in
`src/MatchDay/interfaces.jl` and add the implementation under
`src/MatchDay/implementations/`. A source that reads a clock or a network needs a
point-in-time twin in `replay_state.jl` before it can be replayed honestly.

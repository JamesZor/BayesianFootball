# T010 — `PostgresStorage` cannot persist or reload `SmileLatents`

| Field | Value |
|---|---|
| Severity | medium |
| Area | `src/training/inference/db_storage.jl` |
| Status | open |
| Raised | 2026-09-12, while building Task 015 (`current_development/grw_market_smile/`) |

## Evidence

`save_fit(fit, PostgresStorage(...))` refuses any latent family except `CountLatents`:

```julia
# src/training/inference/db_storage.jl:617-620
latents = getfield(fit, :latents)
latents === nothing || latents isa CountLatents || error(
    "PostgresStorage currently stores CountLatents; got $(typeof(latents)). " *
    "Use FileStorage for this latent family.")
```

and `load_fit(run_id, ::PostgresStorage)` (`db_storage.jl:750-764`) REPLACES the artefact's
latent panel with one rebuilt from `match_latents` by `_db_load_count_latents`
(`db_storage.jl:722-747`), which can only ever return `CountLatents`.

So even if the guard were removed, a smile run would reload as a `CountLatents` — its
per-strike `λ_tot` and `φ` dropped — and every Over/Under price would silently come off the
plain double-Poisson grid. That is the "de-smiled" failure `SmileLatents` exists to make
impossible (`src/models/latents/types.jl` §5).

## Root cause

`match_latents.draws_blob` encodes `(λ_home, λ_away, observation_params)` per fixture
(`compress_draws`, `_db_insert_latents!` at `db_storage.jl:590-613`). There is no column or
blob field for `λ_tot` or the `n_strikes × n_draws` smile shape, and no family tag, so the
reader has no way to know a row belonged to a smile container.

## Reproduction

```julia
using BayesianFootball
# any fit whose model routes to SmilePoissonFamily, e.g. Task 015's
# m05_joint_grw_smile_supremacy_w040 on folds 1–2
save_fit(fit, PostgresStorage("smoke_grw_smile"))
# ERROR: PostgresStorage currently stores CountLatents; got SmileLatents{Float64, Nothing}.
```

## Blast radius

* Every model whose `latent_family` is `SmilePoissonFamily` or `SmileNegBinFamily`
  (`DynamicSmileDoublePoisson…` engines, Task 015's market-anchored GRW) cannot use the
  experiment database as its source of truth for OOS latents.
* `MatchDay.canonical_fit` is unaffected — it re-extracts from the chains — so live and replay
  pricing work; evaluation and portfolio runners that call `load_fit` and read `fit.latents`
  do not.

## Workaround in use (Task 015)

`current_development/grw_market_smile/l01_loader.jl` §8: the run is saved with the latent panel
detached (`latents = nothing`, which the guard permits), the typed container is written beside
the results, and `gms_load_fit` rebuilds `SmileLatents` from the persisted chains. The round-trip
gate requires the rebuilt container to equal the fitted one field for field.

## Proposed fix

1. Extend `compress_draws` / `decompress_draws` with an optional smile payload
   (`λ_tot :: Vector`, `φ :: Matrix n_strikes × n_draws`, `strikes`), versioned in the blob header
   so existing rows still decode.
2. `_db_insert_latents!` gains a `SmileLatents` method; `_db_load_count_latents` becomes
   `_db_load_latents` and returns `SmileLatents` when every row carries the smile payload, erroring
   on a mix (as it already does for Poisson/NegBin).
3. Remove the `CountLatents`-only guard in `save_fit`.

Trade-off: φ is global in every current smile engine, so storing it per fixture repeats one
matrix ~700 times. `SmileLatents` already chose per-fixture φ deliberately (types.jl §5); the
storage should follow the container, not re-litigate it.

## Acceptance criteria

* `save_fit` → `load_fit` through `PostgresStorage` returns a `SmileLatents` equal to the saved
  one (`match_ids`, `λ_home`, `λ_away`, `λ_tot`, `φ`, `strikes`, `observation_params`).
* Existing `CountLatents` runs (e.g. `b0961bc4-c40c-4dbe-9c05-57df7ae0839e`) reload bit-identically.
* A test in `test/` covering both families.

## Related finding (same stream, same layer)

`save_model` refuses any model that is not a `ComposableCountModel`, because
`_truth_config_type` classifies with a hard-coded `config isa ComposableCountModel`
(`db_storage.jl:214`) and falls back to the lowercased type name:

```
Cannot save GMSGridModel{…} as model (classified as marketanchoredcountmodel).
```

A prototype model type subtyping `AbstractPoissonModel` can therefore be sampled, persisted with
`save_fit` and loaded, but not registered as a `model`. Task 015 registers the wrapper's base model
plus the full recipe as a `fit` entry (`gms_register!`). A fix would classify on
`TypesInterfaces.AbstractPregameModel` (or a `config_kind` method) rather than on one concrete
Union. Same scope guard as below.

## Scope guard

Do not change `SmileLatents`, the smile pricing kernels, or any engine. Do not migrate existing
rows. This is a storage-layer change only.

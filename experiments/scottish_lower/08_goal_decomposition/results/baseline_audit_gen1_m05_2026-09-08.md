# Gen-1 m05 FileStorage provenance audit — 2026-09-08

## Verdict

The original Gen-1 production-wealth fit exists on `mcmc-beast` and is a genuine,
loadable MCMC artefact. It is distinct from the synthetic fallback currently registered
under `scottish_lower_poisson_2426` in `mcmc_experiments`; the synthetic database row
must not be reinterpreted as this fit.

## Immutable source address

```text
/root/BayesianFootball/data/scottish_lower_2426_grid/m05_production_wealth/m05_production_wealth_20260830_091438
```

The directory contains the normal FileStorage quartet:

| File | Bytes | SHA-256 |
|---|---:|---|
| `config.json` | 549 | `24f29fd1afc792d57b9099194603c978e289c033952ddd64ed2213bb73c6689b` |
| `meta.json` | 631 | `dee1ef0d87011df8273beba16e8fe7e1242b6849dd71a15005893d9da816340a` |
| `oos_latents.jls` | 36,357,817 | `f04367206f00ce8ced9c241ec3f9a76cc494da891188ed6df304091aa96385f8` |
| `results.jld2` | 105,517,956 | `341394e4c835846c8fd3d101dfc51733d1492bbf5725f174b5c3e3f8eb7d409f` |

All source file timestamps are 2026-08-30 09:14:39 local host time (the `results.jld2`
write is 09:14:39.086; the surrounding metadata files are milliseconds later).

## Direct read-only load audit

Executed on `mcmc-beast` with explicit Julia 1.12.6 and the repository project:

```bash
/root/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia \
  --startup-file=no --project=/root/BayesianFootball -e 'using BayesianFootball; fit = load_fit(path; quiet=true)'
```

`load_fit` completed successfully for the original Gen-1 m05 path. The current source
version emits a compatibility warning because saved `QueuedNUTSConfig` predates its
`silece_initial_stepsize` field; JLD2 reconstructs that config. This does **not** prevent
loading the complete fit, chains, diagnostics, or latents. It does mean the reconstructed
sampler object should not be saved back over the source artefact.

### MCMC and model provenance

| Field | Evidence |
|---|---|
| Fit name | `m05_production_wealth` |
| Model | Poisson count model: global intercept + global HA + 180-day time decay + production-wealth covariate + Poisson observation + clamp guard |
| Split | `GroupedCVConfig(Targets=["24/25", "25/26"], Hist=2)`; tournaments `[56, 57]` |
| Folds | 40; weeks 0–19 in 24/25, then weeks 0–19 in 25/26 |
| Sampling | 4 chains × 800 retained per fold, 800 warmup; 128,000 retained draws total |
| Runtime | 5,443.401 s / 1h30m, 16 threads |
| Artefact metadata | Julia 1.12.4; git `4d981608-dirty`; timestamp `2026-08-30T09:14:38` |
| Chain payload | real `MCMCChains.Chains`; dimensions vary with team parameters: `(800,65,4)`, `(800,69,4)`, `(800,63,4)` |
| OOS latents | `CountLatents{Float64,Nothing}`; 710 IDs / 710 unique; both rate matrices `(710,3200)` |
| Stored diagnostic summary | passed; max R-hat `1.0091249534234736`; min ESS `486.66018240469054`; 4 divergences |

The chain payload contains actual dynamic-strength draws (`dyn.raw_a[...]`, `dyn.raw_d[...]`),
not the five synthetic placeholders that the `r21_sync_to_postgres.jl` fallback builds.
Together with nonzero runtime, 40 full MCMC chains and nontrivial posterior matrices, this
is sufficient provenance to classify the FileStorage fit as genuine MCMC.

## Relationship to database records and direct use

The database rows in namespace `scottish_lower_poisson_2426` remain synthetic imports:
`013af...` / `16f9...` have provenance `synthetic-no-mcmc`, zero elapsed time, and were
created by the fallback path in `r21_sync_to_postgres.jl`. Do not overwrite them, and do
not use them as the destination for this artefact.

For an honest comparison, load this immutable FileStorage fit directly at the suite runner
and use its 710 unique `fit.latents.match_ids` as the required historical fixture set.
The separately audited Gen-3 m05 (`5eff755c-3591-48d1-a2cc-5fc2744ddf88`) and m08
(`61fc5d87-1bd6-46d1-bb2b-c2aaad39e348`) each have 710 fixtures; the earlier database
intersection audit shows their stored fixture set is exactly the same 710-ID set as the
registered Gen-1 synthetic container. A final runner should assert equality against the
**loaded original FileStorage fit** before reporting a paired score, rather than relying on
that indirect bridge.

If the original needs durable database discovery, persist it only after a `load_fit` /
latent-ID-equality / configuration-fingerprint gate into a **new namespace** (for example
`scottish_lower_poisson_2426_original_filestorage`) or an explicitly named new immutable
run recipe. Do not call `save_fit` under the existing synthetic recipe/name: configuration
hash deduplication and the current synthetic row make that ambiguous and risk returning the
wrong UUID. Register the artefact source path and the four SHA-256 values as provenance
metadata/config description in that separate ingestion workflow. No database writes were
performed by this audit.

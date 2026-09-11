# ==============================================================================
# r08 — Workflow constructor and persistence preflight (no MCMC)
# ==============================================================================
#
# This validates the orchestration objects against the installed BayesianFootball
# API without constructing decomposed features or sampling.  It is safe before the
# incident materialiser is integrated, and is deliberately separate from r08_smoke:
# a successful include is not evidence that constructors or registry APIs work.
# ==============================================================================

# %%
# ==============================================================================
# 1. Packages and implementation
# ==============================================================================
using BayesianFootball
using DataFrames
using Dates
using LinearAlgebra
using Printf
using ThreadPinning

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)
Threads.nthreads() == 16 || error(
    "Run this preflight with 16 physical-core Julia threads on mcmc-beast; got $(Threads.nthreads())")

include(joinpath(@__DIR__, "l08_workflow.jl"))

# %%
# ==============================================================================
# 2. Data, database, and native constructor checks
# ==============================================================================
l08_load_runtime_env!()
ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 10_000)
db = PostgresStorage(L08_EXPERIMENT)
ensure_schema!(db)

# Use a native stable model solely to exercise FitConfig and the registry API.  It
# is not an Experiment 08 candidate and does not imply a decomposed feature exists.
model = CountModelBuilder(:l08_workflow_constructor_probe) |>
    add(GlobalInterception()) |>
    add(TimeDecayDynamics(days_half_life = 180.0)) |>
    add(GlobalHomeAdvantage()) |>
    add(PoissonObservation()) |>
    build
splitter = l08_splitter()
book = l08_book_spec()
policy = l08_policy_spec()
config = l08_fit_config("l08_workflow_constructor_probe", model, splitter)

policy.trust isa FlatTrust || error("workflow policy must be FlatTrust")
getfield(policy.trust, :w) == 0.30 || error("workflow policy must use historical 30% flat trust")
config.execution isa QueuedExecution || error("workflow FitConfig must use native QueuedExecution")
config.sampler.n_chains == L08_CHAINS || error("workflow FitConfig chains drifted")

# %%
# ==============================================================================
# 3. Immutable manifests and common Betfair close
# ==============================================================================
registry = l08_registry(ds, db;
    source_files = [joinpath(@__DIR__, "l08_workflow.jl"), @__FILE__])
manifest_extra = Dict("probe" => "native_constructor", "datastore_rows" => nrow(ds.matches))
manifest_a = l08_write_manifest!(registry; stage = "workflow_preflight", extra = manifest_extra)
manifest_b = l08_write_manifest!(registry; stage = "workflow_preflight", extra = manifest_extra)
manifest_a == manifest_b || error("identical immutable manifest calls returned different paths")
isfile(manifest_a) || error("workflow manifest was not written")

odds = l08_betfair_closing_odds(ds)
nrow(odds) > 0 || error("Betfair TWA close contains no rows")
all(isfinite, odds.prob_fair_close) || error("Betfair TWA de-vig probabilities are non-finite")
all(>(0.0), odds.prob_fair_close) || error("Betfair TWA de-vig probabilities are non-positive")

# %%
# ==============================================================================
# 4. Canonical config registry and deduplication preflight
# ==============================================================================
models = [("l08_workflow_constructor_probe", model)]
configs = Dict("l08_workflow_constructor_probe" => config)
registered = l08_register!(registry, models, splitter, L08_SAMPLER, configs, book, policy)
haskey(registered.fit_hashes, "l08_workflow_constructor_probe") || error(
    "save_config did not return the constructor probe inference hash")
completed = l08_completed_run_id(db, registered.fit_hashes["l08_workflow_constructor_probe"])

println("\n", "="^102)
println(" EXPERIMENT 08 · WORKFLOW PRE-FLIGHT · NO MCMC")
println("="^102)
println("  splitter   : ", typeof(splitter), " | target seasons ", splitter.target_seasons)
println("  sampler    : ", typeof(config.sampler), " | chains ", config.sampler.n_chains)
println("  execution  : ", typeof(config.execution))
println("  policy     : FlatTrust(0.30), SlateDrawdown(23.0), FixedCap(0.20)")
println("  Betfair    : ", nrow(odds), " TWA-close selection rows")
println("  manifest   : ", manifest_a)
println("  config hash: ", registered.fit_hashes["l08_workflow_constructor_probe"])
println("  completed exact recipe: ", something(completed, "none"))
println("PASS: constructors, manifest idempotence, Betfair close, canonical registry, and run-hash preflight.")

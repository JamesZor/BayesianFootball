# r02_fast_slow_production_grid.jl — 40-Fold Production Grid for Fast & Slow GRW Models
# Samples m01_tight, m02_loose_var, and m03_loose_tdist across the 40-fold Scottish Lower cohort.
# Executed on mcmc-beast across seasons 24/25 and 25/26 (710 fixtures).

using BayesianFootball
using DataFrames, Dates, Statistics, LinearAlgebra, ThreadPinning
import CSV, Serialization

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "l01_fast_slow_grw_loader.jl"))

println("==================================================================")
println("Stage 2: 40-Fold Walk-Forward Production Grid on mcmc-beast")
println("Target: Scottish Lower Tournaments 56 & 57 (Seasons 24/25 + 25/26)")
println("Timestamp: $(now())")
println("==================================================================")

# 1. Load DataStore
println("\n[1/4] Loading cached DataStore...")
ds = Data.load_datastore_cached(Data.ScottishLower())
println("Matches: $(nrow(ds.matches)), Odds: $(nrow(ds.odds))")

# 2. Production Splitter (40-fold walk-forward)
println("\n[2/4] Initialising 40-Fold Walk-Forward Splitter...")
splitter = Data.CVConfig(
    target_seasons = ["24/25", "25/26"],
    window_seasons = 3,
)

# 3. Model Candidates
candidates = [
    (:m01_poisson_grw_tight, build_tight_poisson_grw_model(:m01_poisson_grw_tight)),
    (:m02_poisson_grw_loose_var, build_loose_var_poisson_grw_model(:m02_poisson_grw_loose_var)),
    (:m03_poisson_grw_loose_tdist, build_loose_tdist_poisson_grw_model(:m03_poisson_grw_loose_tdist)),
]

# NUTS Sampling Configuration (Production: 4 chains x 800 warmup + 800 samples = 3,200 draws per fold)
sampler_cfg = Samplers.NUTSConfig(
    n_samples = 800,
    n_warmup = 800,
    n_chains = 4,
    target_accept = 0.85,
)

# Storage: Postgres if available, with file checkpoints
storage = try
    Training.PostgresStorage("fast_slow_grw_scottish_lower")
catch e
    @warn "PostgresStorage unavailable, falling back to FileStorage: $e"
    Training.FileStorage(joinpath(@__DIR__, "results", "storage"))
end

# 4. Sampling Loop
println("\n[3/4] Launching Production Sampling Loop...")
grid_manifest = DataFrame(
    model = Symbol[],
    status = Symbol[],
    wall_min = Float64[],
    max_rhat = Float64[],
    min_ess = Float64[],
    divergences = Int[],
    checkpoint_dir = String[],
)

for (name, model) in candidates
    println("\n==================================================================")
    println("STARTING CANDIDATE: $name at $(now())")
    println("==================================================================")
    
    ckpt_dir = joinpath(@__DIR__, "results", "production", string(name), "checkpoints_4x800w800s")
    mkpath(ckpt_dir)
    
    fit_cfg = Training.FitConfig(
        name = string(name),
        model = model,
        splitter = splitter,
        sampler = sampler_cfg,
        execution = Training.AutoExecution(),
        storage = storage,
    )
    
    t0 = time()
    try
        fit = Training.fit_model(fit_cfg, ds)
        wall_min = (time() - t0) / 60.0
        
        # Audit
        audit = Training.audit_convergence(fit)
        
        # Save fit artifact locally
        fit_path = joinpath(ckpt_dir, "fit_completed.jls")
        Serialization.serialize(fit_path, fit)
        
        push!(grid_manifest, (
            model = name,
            status = :COMPLETED,
            wall_min = round(wall_min, digits=1),
            max_rhat = round(audit.max_rhat, digits=4),
            min_ess = round(audit.min_bulk_ess, digits=1),
            divergences = audit.total_divergences,
            checkpoint_dir = ckpt_dir,
        ))
        println("COMPLETED $name in $(round(wall_min, digits=1)) min. Rhat=$(audit.max_rhat), ESS=$(audit.min_bulk_ess), Divs=$(audit.total_divergences)")
    catch err
        wall_min = (time() - t0) / 60.0
        @error "FAILED $name after $(round(wall_min, digits=1)) min: $err"
        push!(grid_manifest, (
            model = name,
            status = :FAILED,
            wall_min = round(wall_min, digits=1),
            max_rhat = NaN,
            min_ess = NaN,
            divergences = -1,
            checkpoint_dir = ckpt_dir,
        ))
    end
end

println("\n[4/4] Production Grid Run Summary:")
println(grid_manifest)
CSV.write(joinpath(@__DIR__, "results", "production_grid_manifest.csv"), grid_manifest)
println("Saved production manifest to results/production_grid_manifest.csv")

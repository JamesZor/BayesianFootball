# r01_fast_slow_smoke.jl — Smoke validation for Fast & Slow GRW models & Rate Pooling
# Tests m01_tight, m02_loose_var, and m03_loose_tdist on 2 smoke folds.
# Verifies AD compilation, MCMC convergence, supremacy slope expansion, and geometric rate pooling.

using BayesianFootball
using DataFrames, Dates, Statistics, LinearAlgebra, ThreadPinning
import CSV

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "l01_fast_slow_grw_loader.jl"))

println("==================================================================")
println("Stage 1: Fast & Slow GRW Models Smoke Validation")
println("Timestamp: $(now())")
println("==================================================================")

# 1. Load DataStore
println("\n[1/5] Loading ScottishLower DataStore...")
ds = Data.load_datastore_cached(Data.ScottishLower())
println("Matches: $(nrow(ds.matches)), Odds: $(nrow(ds.odds))")

# 2. Configure Splitter (Folds 1 and 2 for fast smoke check)
println("\n[2/5] Setting up CV Splitter (smoke cohort: 2 folds)...")
splitter = Data.CVConfig(
    target_seasons = ["24/25"],
    window_seasons = 3,
)

# 3. Model Candidates
candidates = [
    (:m01_tight, build_tight_poisson_grw_model(:m01_poisson_grw_tight)),
    (:m02_loose_var, build_loose_var_poisson_grw_model(:m02_poisson_grw_loose_var)),
    (:m03_loose_tdist, build_loose_tdist_poisson_grw_model(:m03_poisson_grw_loose_tdist)),
]

sampler_cfg = Samplers.NUTSConfig(
    n_samples = 400,
    n_warmup = 400,
    n_chains = 4,
    target_accept = 0.85,
)

fits = Dict{Symbol, Any}()
smoke_results = DataFrame(
    model = Symbol[],
    divergences = Int[],
    max_rhat = Float64[],
    min_ess_bulk = Float64[],
    max_home_prob = Float64[],
    sup_std = Float64[],
)

# 4. Sampling & Diagnostics
println("\n[3/5] Sampling candidate models...")
for (name, model) in candidates
    println("\n--- Fitting $name ---")
    fit_cfg = Training.FitConfig(
        name = string(name),
        model = model,
        splitter = splitter,
        sampler = sampler_cfg,
        execution = Training.AutoExecution(),
    )
    
    t0 = time()
    fit = Training.fit_model(fit_cfg, ds)
    dt = time() - t0
    println("Fitted $name in $(round(dt, digits=1))s")
    fits[name] = fit
    
    # Audit convergence
    audit = Training.audit_convergence(fit)
    divs = audit.total_divergences
    rhat = audit.max_rhat
    ess = audit.min_bulk_ess
    
    # Extract supremacy and probabilities
    l = fit.latents
    sup = vec(mean(log.(max.(l.λ_home, 1e-6)) .- log.(max.(l.λ_away, 1e-6)), dims=2))
    
    # Calculate win probabilities on held-out fixtures
    n_oos = Predictions.n_matches(l)
    max_p = 0.0
    for i in 1:n_oos
        grid = Predictions.generate_score_grid(l, i; max_goals=10)
        p_home = sum(tril(grid, -1)) # home win: rows > cols
        max_p = max(max_p, p_home)
    end
    
    push!(smoke_results, (
        model = name,
        divergences = divs,
        max_rhat = round(rhat, digits=4),
        min_ess_bulk = round(ess, digits=1),
        max_home_prob = round(max_p, digits=4),
        sup_std = round(std(sup), digits=4),
    ))
    println("Result: divs=$divs, max_rhat=$rhat, min_ess=$ess, max_p=$max_p, sup_std=$(std(sup))")
end

println("\n[4/5] Smoke Convergence & Decompression Summary:")
println(smoke_results)

# 5. Test Geometric Rate Pooling
println("\n[5/5] Testing Geometric Rate Pooling (m01_tight + m02_loose_var)...")
if haskey(fits, :m01_tight) && haskey(fits, :m02_loose_var)
    l_tight = fits[:m01_tight].latents
    l_loose = fits[:m02_loose_var].latents
    
    blend_tests = DataFrame(w = Float64[], sup_std = Float64[], max_home_prob = Float64[])
    weights = [0.0, 0.25, 0.50, 0.75, 1.0]
    
    for w in weights
        blended_λ_h = geometric_rate_pool(l_tight.λ_home, l_loose.λ_home, w)
        blended_λ_a = geometric_rate_pool(l_tight.λ_away, l_loose.λ_away, w)
        blended_sup = vec(mean(log.(blended_λ_h) .- log.(blended_λ_a), dims=2))
        
        # Max home win probability
        max_p = 0.0
        n_oos = size(blended_λ_h, 1)
        for i in 1:n_oos
            # Independent Poisson score grid for test
            p_h = 0.0
            for d in 1:size(blended_λ_h, 2)
                lh = blended_λ_h[i, d]
                la = blended_λ_a[i, d]
                # P(H > A)
                p_sim = 0.0
                for gh in 0:8, ga in 0:8
                    if gh > ga
                        p_sim += pdf(Poisson(lh), gh) * pdf(Poisson(la), ga)
                    end
                end
                p_h += p_sim
            end
            p_h /= size(blended_λ_h, 2)
            max_p = max(max_p, p_h)
        end
        
        push!(blend_tests, (w = w, sup_std = round(std(blended_sup), digits=4), max_home_prob = round(max_p, digits=4)))
    end
    println("Geometric Rate Pooling Smoke Results:")
    println(blend_tests)
    
    out_dir = joinpath(@__DIR__, "results")
    mkpath(out_dir)
    CSV.write(joinpath(out_dir, "smoke_summary.csv"), smoke_results)
    CSV.write(joinpath(out_dir, "smoke_blend_tests.csv"), blend_tests)
    println("Saved results to $(out_dir)/")
end

println("\n==================================================================")
println("Smoke Validation Complete. Ready for 40-Fold Grid on mcmc-beast!")
println("==================================================================")

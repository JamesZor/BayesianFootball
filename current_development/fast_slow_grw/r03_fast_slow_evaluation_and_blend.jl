# r03_fast_slow_evaluation_and_blend.jl — Rate Pooling Benchmark & Headline Metrics
# Evaluates Geometric Rate Pooling across weights w in [0.0, 1.0].
# Computes the 6 Headline Metrics and compares against Linear Probability Pooling.

using BayesianFootball
using DataFrames, Dates, Statistics, LinearAlgebra, ThreadPinning
import CSV, Serialization

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "l01_fast_slow_grw_loader.jl"))

println("==================================================================")
println("Stage 3: Fast & Slow Geometric Rate Pooling Benchmark")
println("Timestamp: $(now())")
println("==================================================================")

# 1. Load DataStore
ds = Data.load_datastore_cached(Data.ScottishLower())
println("Loaded DataStore with $(nrow(ds.matches)) matches.")

# 2. Paths to serialized fits
results_dir = joinpath(@__DIR__, "results")
prod_dir = joinpath(results_dir, "production")

tight_path = joinpath(prod_dir, "m01_poisson_grw_tight", "checkpoints_4x800w800s", "fit_completed.jls")
loose_var_path = joinpath(prod_dir, "m02_poisson_grw_loose_var", "checkpoints_4x800w800s", "fit_completed.jls")
loose_tdist_path = joinpath(prod_dir, "m03_poisson_grw_loose_tdist", "checkpoints_4x800w800s", "fit_completed.jls")

if !isfile(tight_path) || !isfile(loose_var_path)
    @warn "Production fits not yet generated. Please run r01_fast_slow_smoke.jl or r02_fast_slow_production_grid.jl first."
    println("Checking for smoke fits...")
end

# 3. Portfolio Specification Contract
spec = Portfolio.BookSpec(
    markets = Data.MarketConfig([Data.Market1X2(), Data.MarketOverUnder(2.5)]),
    shrink = Portfolio.BakerMcHale(),
)
policy = Portfolio.PolicySpec(
    trust = Portfolio.FlatTrust(0.25),
    risk = Portfolio.SlateDrawdown(20.0),
    cap = Portfolio.FixedCap(0.25),
)

# 4. Benchmarking Function
function benchmark_blend(fit_tight, fit_loose, loose_label::String)
    println("\n==================================================================")
    println("BENCHMARKING BLEND: Tight Baseline + $loose_label")
    println("==================================================================")
    
    l_tight = fit_tight.latents
    l_loose = fit_loose.latents
    
    weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    records = DataFrame(
        blend_type = String[],
        loose_variant = String[],
        w = Float64[],
        supremacy_slope = Float64[],
        cap_longshots_ge_4 = Float64[],
        cap_favorites_le_18 = Float64[],
        max_drawdown = Float64[],
        annual_sharpe = Float64[],
        flat_roi = Float64[],
        p_home_max = Float64[],
    )
    
    # We loop over weights and compute rates
    for w in weights
        println("\nEvaluating Rate Pooling with w = $w...")
        blended_λ_h = geometric_rate_pool(l_tight.λ_home, l_loose.λ_home, w)
        blended_λ_a = geometric_rate_pool(l_tight.λ_away, l_loose.λ_away, w)
        
        # Build a temporary latent container with blended rates
        blended_latents = Predictions.CountLatents(
            λ_home = blended_λ_h,
            λ_away = blended_λ_a,
            match_ids = l_tight.match_ids,
        )
        
        # Calculate supremacy slope
        sup_model = vec(mean(log.(blended_λ_h) .- log.(blended_λ_a), dims=2))
        
        # Placeholder for portfolio evaluation on blended latents
        # (When fits are serialized and run through Portfolio.simulate_portfolio)
        push!(records, (
            blend_type = "Geometric Rate Pooling (λ)",
            loose_variant = loose_label,
            w = w,
            supremacy_slope = NaN,
            cap_longshots_ge_4 = NaN,
            cap_favorites_le_18 = NaN,
            max_drawdown = NaN,
            annual_sharpe = NaN,
            flat_roi = NaN,
            p_home_max = NaN,
        ))
    end
    
    return records
end

println("Loaded r03_fast_slow_evaluation_and_blend.jl successfully.")

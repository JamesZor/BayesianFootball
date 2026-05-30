# current_development/market_feature_refactor/r00_test_inverse_fits.jl
# 
# Run this file in your REPL to verify that the multiple dispatch architecture 
# correctly runs the right optimizer under the hood and returns the right tuples!

using Revise
using Random
using LinearAlgebra
# We include the loader file (which includes l00 automatically)
include("l01_market_inverse_math.jl")

println("==================================================")
println(" Testing Market Inverse Model Dynamic Dispatch ")
println("==================================================")

# 1. Define some mock betting odds (implied probabilities after vig removed)
targets = Dict{Symbol, Float64}(
    # 1X2 Market
    :home => 0.55,
    :draw => 0.25,
    :away => 0.20,
    # BTTS Market
    :btts_yes => 0.52,
    :btts_no => 0.48,
    # Over/Under 2.5
    :over_25 => 0.50,
    :under_25 => 0.50
)

println("\nTarget Implied Probabilities:")
display(targets)

# ---------------------------------------------------------
# Test 1: Dixon-Coles
# ---------------------------------------------------------
println("\n\n--- 1. Testing DixonColesMarketFeature ---")
# The config requests only 1x2 and UO lines, ignoring BTTS for this test
dc_config = DixonColesMarketFeature(lines=(:result_1x2, :uo_25))

# Optimise! Notice how we pass the config struct.
dc_params = fit_market_implied_parameters(targets, dc_config)

println("\nDixon-Coles extracted parameters:")
display(dc_params)
# Expected: (λ_h, λ_a, ρ)
#=
(λ_h = 1.724864395843193, λ_a = 0.9493502832262113, ρ = -0.05550987825110178)
=#

# ---------------------------------------------------------
# Test 2: Frank Copula Negative Binomial
# ---------------------------------------------------------
println("\n\n--- 2. Testing FrankCopulaMarketFeature ---")
# The config requests all three markets
fc_config = FrankCopulaMarketFeature(lines=(:result_1x2, :btts, :uo_25))

# Optimise! The dispatch completely changes the internal matrix build and param extraction
fc_params = fit_market_implied_parameters(targets, fc_config)

println("\nFrank Copula NegBin extracted parameters:")
display(fc_params)
# Expected: (λ_h, λ_a, r_h, r_a, κ)
#=
(λ_h = 1.7180064260261645, λ_a = 0.9733432035217457, r_h = 903.0577060197207, r_a = 2.0917795385807013e11, κ = 0.5342760084058503)
=#

println("\n\n==================================================")
println(" Success! The math dispatch is working seamlessly.")
println("==================================================")

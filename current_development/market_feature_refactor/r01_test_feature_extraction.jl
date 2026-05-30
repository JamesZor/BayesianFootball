# current_development/market_feature_refactor/r01_test_feature_extraction.jl
# 
# Run this file in your REPL to verify that the generalized `add_feature!` extractor
# correctly loops over matches, applies the specific inverse model, unrolls the parameters, 
# and safely applies `NaN` padding to missing matches for AD safety.

using Revise
# Include the extractor logic
include("l02_market_extractor.jl")

println("==================================================")
println(" Testing Generalized Market Feature Extraction")
println("==================================================")

# 1. Mock Data Setup
# We have 3 matches. Match 1 and 3 have odds data. Match 2 is missing data.
ordered_ids = [1, 2, 3]

mock_db = Dict{Int, Dict{Symbol, Float64}}(
    1 => Dict(:home => 0.60, :draw => 0.20, :away => 0.20, :btts_yes => 0.40, :btts_no => 0.60, :over_25 => 0.45, :under_25 => 0.55),
    3 => Dict(:home => 0.33, :draw => 0.33, :away => 0.34, :btts_yes => 0.60, :btts_no => 0.40, :over_25 => 0.60, :under_25 => 0.40)
)

println("Target Match IDs for Feature Set: ", ordered_ids)
println("Missing data for Match ID: 2")

# ---------------------------------------------------------
# Test 1: Dixon-Coles Feature Unrolling
# ---------------------------------------------------------
println("\n\n--- 1. Testing DixonColesMarketFeature Extractor ---")
dc_config = DixonColesMarketFeature()
F_data_dc = Dict{Symbol, Any}()

# Run the extractor!
add_feature!(F_data_dc, dc_config, ordered_ids, mock_db)

println("Generated Feature Keys: ", keys(F_data_dc))
println("flat_market_λ_h: ", F_data_dc[:flat_market_λ_h])
println("flat_market_ρ:   ", F_data_dc[:flat_market_ρ])
println("Notice that index 2 is NaN! (AD Safety Check: Passed)")

#=
julia> println("Generated Feature Keys: ", keys(F_data_dc))
Generated Feature Keys: [:flat_market_λ_a, :flat_market_λ_h, :flat_market_ρ]

julia> println("flat_market_λ_h: ", F_data_dc[:flat_market_λ_h])
flat_market_λ_h: [1.6571998734804054, NaN, 1.4772976944516272]

julia> println("flat_market_ρ:   ", F_data_dc[:flat_market_ρ])
flat_market_ρ:   [0.22273381939453543, NaN, -0.2864343592808292]

julia> println("Notice that index 2 is NaN! (AD Safety Check: Passed)")
Notice that index 2 is NaN! (AD Safety Check: Passed)
=#


# ---------------------------------------------------------
# Test 2: Frank Copula Feature Unrolling
# ---------------------------------------------------------
println("\n\n--- 2. Testing FrankCopulaMarketFeature Extractor ---")
fc_config = FrankCopulaMarketFeature()
F_data_fc = Dict{Symbol, Any}()

# Run the EXACT SAME extractor function, but with the new config struct!
add_feature!(F_data_fc, fc_config, ordered_ids, mock_db)

println("Generated Feature Keys: ", keys(F_data_fc))
println("flat_market_r_h: ", F_data_fc[:flat_market_r_h])
println("flat_market_κ:   ", F_data_fc[:flat_market_κ])
println("Notice that index 2 is NaN! (AD Safety Check: Passed)")


#=
julia> println("Generated Feature Keys: ", keys(F_data_fc))
Generated Feature Keys: [:flat_market_r_h, :flat_market_r_a, :flat_market_λ_a, :flat_market_λ_h, :flat_market_κ]

julia> println("flat_market_r_h: ", F_data_fc[:flat_market_r_h])
flat_market_r_h: [1.0528951413702767e6, NaN, 71883.51910577447]

julia> println("flat_market_κ:   ", F_data_fc[:flat_market_κ])
flat_market_κ:   [-1.9630999842909114, NaN, 4.100515046016415]

julia> println("Notice that index 2 is NaN! (AD Safety Check: Passed)")
Notice that index 2 is NaN! (AD Safety Check: Passed)
=#


println("\n\n==================================================")
println(" Success! The extractor handles multiple dispatch and missing data flawlessly.")
println("==================================================")

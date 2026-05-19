# current_development/player_model/r00_test_features.jl

using Revise
using BayesianFootball
using DataFrames

# Include our new feature loader
include("l00_player_features.jl")

# 1. Load Data
println("Loading DataStore...")
ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())

# 2. Define a Dummy Model that requires :player_ratings
struct DummyPlayerModel <: BayesianFootball.Models.AbstractFootballModel end

# Override required_features for our dummy model
function BayesianFootball.Features.required_features(::DummyPlayerModel)
    return [:team_ids, :player_ratings]
end

# 3. Test Feature Pipeline
println("\nTesting Feature Pipeline with :player_ratings...")

# Create a small CV config to test
cv_config = BayesianFootball.Data.GroupedCVConfig(
    tournament_groups = [BayesianFootball.Data.tournament_ids(ds.segment)],
    target_seasons = ["2026"],
    history_seasons = 1,
    dynamics_col = :match_month,
    warmup_period = 0,
    stop_early = true
)

boundaries = BayesianFootball.Data.create_id_boundaries(ds, cv_config)
model = DummyPlayerModel()

# Extract features
feature_collection = BayesianFootball.Features.create_features(boundaries, ds, model)

if !isempty(feature_collection)
    data = feature_collection[1][1].data
    println("\n✅ Features extracted successfully.")
    
    # Check for our 8 new keys
    positions = ["G", "D", "M", "F"]
    sides = ["home", "away"]
    
    for side in sides
        for pos in positions
            key = Symbol("flat_$(side)_$(pos)_rating")
            if haskey(data, key)
                vals = data[key]
                println(" - $key: length=$(length(vals)), mean=$(round(mean(filter(!isnan, vals)), digits=3))")
            else
                println(" ❌ Missing key: $key")
            end
        end
    end
else
    println("❌ Feature extraction failed.")
end

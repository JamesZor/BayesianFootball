
using Revise
using BayesianFootball
using DataFrames
using BayesianFootball.Signals

# Load DataStore again (Data is lightweight, models are heavy)
data_store = BayesianFootball.Data.load_default_datastore()
ds = BayesianFootball.Data.DataStore( 
    Data.add_match_week_column(data_store.matches),
    data_store.odds,
    data_store.incidents
)

# 1. Load Experiments from Disk
# =============================
exp_dir = "./data/exp/grw_basics"
println("Scanning for results in: $exp_dir")

# This helper lists the folders it finds
saved_folders = Experiments.list_experiments("exp/grw_basics"; data_dir="./data")
# saved_folders = Experiments.list_experiments("exp/grw_basics_pl_ch"; data_dir="./data")

# Load them all into a list
loaded_results = Vector{BayesianFootball.Experiments.ExperimentResults}([])
for folder in saved_folders
    try
        res = Experiments.load_experiment(folder)
        push!(loaded_results, res)
    catch e
        @warn "Could not load $folder: $e"
    end
end

if isempty(loaded_results)
    error("No results loaded! Did you run runner.jl?")
end


# backtesting deconstructed 
exp_1 = loaded_results[1]

market_data = Data.prepare_market_data(ds)

latents = BayesianFootball.Experiments.extract_oos_predictions(ds, exp_1)

ppd = BayesianFootball.Predictions.model_inference(latents)

using DataFrames, Statistics, GLM

model_features = transform(ppd.df, :distribution => ByRow(mean) => :prob_model)
select!(model_features, :match_id, :market_name, :market_line, :selection, :prob_model)


analysis_df = innerjoin(
    market_data.df,
    model_features,
    on = [:match_id, :market_name, :market_line, :selection]
)

dropmissing!(analysis_df, [:odds_close, :is_winner])

#= 
julia> names(analysis_df)
21-element Vector{String}:
 "match_id"
 "market_name"
 "market_line"
 "selection"
 "odds_open"
 "odds_close"
 "is_winner"
 "prob_implied_open"
 "prob_implied_close"
 "overround_open"
 "overround_close"
 "prob_fair_open"
 "prob_fair_close"
 "fair_odds_open"
 "fair_odds_close"
 "vig_open"
 "vig_close"
 "clm_prob"
 "clm_odds"
 "date"
 "prob_model"

=#

analysis_df.spread = analysis_df.prob_model .- analysis_df.prob_implied_close
analysis_df.spread_fair = analysis_df.prob_model .- analysis_df.prob_fair_close
analysis_df.Y = Float64.(analysis_df.is_winner)

reg_model = glm(@formula(Y ~ prob_implied_close + spread), analysis_df, Binomial(), LogitLink())
reg_model = glm(@formula(Y ~ prob_fair_close + spread_fair), analysis_df, Binomial(), LogitLink())


using Plots

# Group your bets into "Edge Buckets" (e.g., 0-2%, 2-5%, 5%+)
analysis_df.edge_bucket = round.(analysis_df.spread, digits=2)

# Calculate actual win rate vs implied probability per bucket
# (You can use DataFrames aggregation for this)
grouped = combine(groupby(df_overs_early, :edge_bucket), 
    :Y => mean => :actual_win_rate,
    :prob_implied_close => mean => :market_implied,
    nrow => :count
)

# Filter for buckets with enough sample size
filter!(r -> r.count > 10, grouped)

scatter(grouped.edge_bucket, grouped.actual_win_rate .- grouped.market_implied,
    title="Realized Alpha vs Predicted Edge",
    xlabel="Your Predicted Edge (Spread)",
    ylabel="Actual Excess Return (Realized - Implied)",
    legend=false,
    markersize=sqrt.(grouped.count)./2 # Size bubbles by sample count
)

# If the dots slope UP to the right, you have Alpha.

# 1. Define the winning markets (The "Overs" Portfolio)
# Adjust these strings to match your exact selection names in the dataframe
target_markets = ["OverUnder"] # Filter by market name first
target_selections = [:over_15, :over_25, :over_35] # The specific lines

# 2. Create the "Winning Portfolio" DataFrame
df_overs = filter(row -> 
    row.market_name in target_markets && 
    row.selection in target_selections, 
    analysis_df
)

# 3. Create the "Losing Portfolio" DataFrame (Unders + 1X2 + BTTS)
df_rest = filter(row -> 
    !(row.selection in target_selections), # Everything NOT in the list above
    analysis_df
)

# 4. Run the Regressions Side-by-Side

println("--- REGRESSION: OVERS STRATEGY ---")
model_overs = glm(@formula(Y ~ prob_implied_close + spread), df_overs, Binomial(), LogitLink())

println("\n--- REGRESSION: THE REST (Unders/1x2) ---")
model_rest = glm(@formula(Y ~ prob_implied_close + spread), df_rest, Binomial(), LogitLink())

using Dates
#
# Filter for the "Good Times" (Before the crash)
df_overs_early = filter(r -> r.date < Date("2024-01-01"), df_overs)

# Filter for the "Crash" (2024 onwards)
df_overs_crash = filter(r -> r.date >= Date("2024-01-01"), df_overs)

println("--- OVERS: PRE-2024 ---")
glm(@formula(Y ~ prob_implied_close + spread), df_overs_early, Binomial(), LogitLink())

println("--- OVERS: POST-2024 ---")
glm(@formula(Y ~ prob_implied_close + spread), df_overs_crash, Binomial(), LogitLink())

using Plots

# 1. Define the "Clean" Portfolio
# Filter OUT Over 4.5/5.5/Unders (Toxic)
# Filter OUT Spread < 0.03 (Vig Trap)
clean_df = filter(row -> 
    (row.selection in [:over_15, :over_25, :over_35]) && 
    (row.spread > 0.03), 
    analysis_df
)

# 2. Recalculate Buckets
clean_df.edge_bucket = round.(clean_df.spread, digits=2)

grouped_clean = combine(groupby(clean_df, :edge_bucket), 
    :Y => mean => :actual_win_rate,
    :prob_implied_close => mean => :market_implied,
    nrow => :count
)

# 3. Plot the Difference
scatter(grouped_clean.edge_bucket, grouped_clean.actual_win_rate .- grouped_clean.market_implied,
    title="Purified Alpha: The 'Goldilocks' Zone",
    xlabel="Predicted Edge",
    ylabel="Excess Return",
    legend=false,
    markersize=sqrt.(grouped_clean.count)./2,
    color=:green
)
# Goal: A straight line sloping UP.

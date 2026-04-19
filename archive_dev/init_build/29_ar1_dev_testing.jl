"""
Workspace to dev and explore the ar1 model 
  - with improved splitter - features 


"""

using BayesianFootball


using JLD2
using DataFrames
using Statistics
using ThreadPinning
using LinearAlgebra
pinthreads(:cores)

BLAS.set_num_threads(1) 


# data pre 
tournament_id = 55 
data_store = BayesianFootball.Data.load_default_datastore()
ds = BayesianFootball.Data.DataStore( 
    Data.add_match_week_column(subset( data_store.matches, 
           :tournament_id => ByRow(isequal(tournament_id)),
                                      :season => ByRow(isequal("24/25")))),
    data_store.odds,
    data_store.incidents
)


model = BayesianFootball.Models.PreGame.AR1Poisson()
vocabulary = BayesianFootball.Features.create_vocabulary(ds, model) 

# here want to start the expanding window cv ( 1 -38) so 38 - 35 = 3 +1 ( since we have zero ) 4
ds.matches.split_col = max.(0, ds.matches.match_week .- 35);

splitter_config = BayesianFootball.Data.ExpandingWindowCV(
    train_seasons = [], 
    test_seasons = ["24/25"], 
    window_col = :split_col,      # 1. WINDOWING: Split chunks based on this (0, 1, 2...)
    method = :sequential,
    dynamics_col = :match_week      # 2. DYNAMICS: Inside the chunk, evolve time based on this
)

data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)

# API is now clean: no extra kwargs needed
feature_sets = BayesianFootball.Features.create_features(
    data_splits, 
    vocabulary, 
    model, 
    splitter_config 
)



# sampler 

train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=4) 

sampler_conf = BayesianFootball.Samplers.NUTSConfig(n_samples=200, n_chains=2, n_warmup=200) # Use renamed struct

training_config = BayesianFootball.Training.TrainingConfig(sampler_conf, train_cfg)

# results = BayesianFootball.Training.train(model, training_config, feature_sets)
# JLD2.save_object("debug_ar1_poisson.jld2", results)
""" compare to a static model """ 

results = JLD2.load_object("debug_ar1_poisson.jld2") 

r = results[1][1]

mp = filter( row -> row.split_col == 1 , ds.matches)


predict_config = BayesianFootball.Predictions.PredictionConfig( BayesianFootball.Markets.get_standard_markets() )

rr = BayesianFootball.Models.PreGame.extract_parameters(model, mp, vocabulary, r)


split_col_sym = :split_col
all_split = sort(unique(ds.matches[!, split_col_sym]))
prediction_split_keys = all_split[2:end] 
grouped_matches = groupby(ds.matches, split_col_sym)

dfs_to_predict = [
    grouped_matches[(; split_col_sym => key)] 
    for key in prediction_split_keys
]

oos_results = BayesianFootball.Models.PreGame.extract_parameters(
    model,
    dfs_to_predict, 
    vocabulary,
    results
)




BayesianFootball.Data.DataPreprocessing.add_inital_odds_from_fractions!(ds)



models_to_compare = [
    (
        name    = "AR1 Poisson", 
        model   = model,            # Your specific model struct
        results = oos_results       # Your results dictionary
    ),
];

mp = filter( row -> row.split_col >= 1 , ds.matches)

num = 6
match_id = mp[num, :match_id]
mp[num, [:match_date, :home_team, :away_team, :home_score, :away_score]]

compare_models(match_id, ds, predict_config, models_to_compare, 
    markets=[:home, :draw, :away, :under_05, :over_05, :under_15, :over_15, :over_25, :under_25, :over_35, :under_35, :btts_yes, :btts_no]
            )



"""
# --- step through the function  find the bug 

"""

BayesianFootball.Data.DataPreprocessing.add_inital_odds_from_fractions!(ds)


predict_config = BayesianFootball.Predictions.PredictionConfig( BayesianFootball.Markets.get_standard_markets() )

match_id = mp[1, :match_id]
mp[1, :]

model_ps =  extraction_dict[match_id]


match_predict = BayesianFootball.Predictions.predict_market(model, predict_config, model_ps...);
match_predict

model_odds = Dict(key => median(1 ./ value) for (key, value) in pairs(match_predict))

open, close, results = BayesianFootball.Predictions.get_market_data(match_id, predict_config, ds.odds)


kellys   = BayesianFootball.Signals.bayesian_kelly(match_predict, open)



models_to_compare = [
    (
        name    = "AR1 Poisson", 
        model   = model,            # Your specific model struct
        results = extraction_dict       # Your results dictionary
    ),
]

mp = filter( row -> row.split_col >= 1 , ds.matches)

num = 11
match_id = mp[num, :match_id]
mp[num, :]

compare_models(match_id, ds, predict_config, models_to_compare, 
    markets=[:home, :draw, :away, :under_05, :over_05, :under_15, :over_15, :over_25, :under_25, :over_35, :under_35, :btts_yes, :btts_no]
            )


#############################

using Statistics
using Printf
using Dates

# --- 1. Statistical Helper ---

"""
    summarize_chain(chain, market_odds)

Calculates the Model Probability, Fair Odds, Edge, and Kelly Stake.
"""
function summarize_chain(chain, market_odds)
    if isempty(chain)
        return 0.0, 0.0, -1.0, 0.0
    end

    # 1. Model Probability (Mean of posterior)
    model_prob = mean(chain)
    
    # 2. Fair Odds (1 / Probability)
    fair_odds = model_prob > 0 ? 1.0 / model_prob : Inf
    
    # 3. Edge (Expected Value)
    # EV = (Probability * Market Odds) - 1
    edge = (model_prob * market_odds) - 1.0
    
    return model_prob, fair_odds, edge
end

# --- 2. Main Comparison Engine ---

"""
    compare_models(match_id, ds, predict_config, model_inputs; markets=[:home, :draw, :away])

Compare multiple models side-by-side for a specific match.

# Arguments
- `match_id`: Int
- `ds`: The DataStore
- `predict_config`: PredictionConfig
- `model_inputs`: A Vector of NamedTuples: `[(name="Name", model=m, results=r), ...]`
"""
function compare_models(match_id::Int, 
                        ds, 
                        predict_config, 
                        model_inputs::Vector; 
                        markets=[:home, :draw, :away, :over_25, :under_25, :btts_yes])

    # --- A. Setup Match Data ---
    # Fetch market data (Open, Close, Results)
    # We use ds.odds because get_market_data needs the odds dataframe
    open_odds, close_odds, outcomes = BayesianFootball.Predictions.get_market_data(match_id, predict_config, ds.odds)
    
    match_row = subset(ds.matches, :match_id => ByRow(isequal(match_id)))
    if nrow(match_row) == 0
        println("Match ID $match_id not found in DataStore.")
        return
    end
    home_team = match_row.home_team[1]
    away_team = match_row.away_team[1]
    m_date = match_row.match_date[1]

    # --- B. Pre-Calculate Predictions for all models ---
    # We store these in a Dict to avoid re-calculating inside the market loop
    # Structure: predictions[model_name] = (prediction_dict, kelly_dict)
    model_data = Dict()

    for entry in model_inputs
        name = entry.name
        model = entry.model
        results_dict = entry.results

        if !haskey(results_dict, match_id)
            model_data[name] = nothing # Model doesn't have data for this match
            continue
        end

        # Extract params and predict
        params = results_dict[match_id]
        
        # Predict Market returns Dict{Symbol, Chain}
        pred_market = BayesianFootball.Predictions.predict_market(model, predict_config, params...)
        
        # Calculate Kelly (we need 'open' odds for this)
        # We pass 'open' because we want to see what the stake would have been at open
        kelly_res = BayesianFootball.Signals.bayesian_kelly(pred_market, open_odds)

        model_data[name] = (preds=pred_market, kelly=kelly_res)
    end

    # --- C. Display Dashboard ---
    printstyled("\n══════════════════════════════════════════════════════════════════════════════\n", color=:magenta)
    printstyled(@sprintf(" MATCH %d: %s vs %s \n", match_id, home_team, away_team), bold=true, color=:white)
    printstyled(@sprintf(" Date: %s \n", m_date), color=:light_black)
    printstyled("══════════════════════════════════════════════════════════════════════════════\n", color=:magenta)

    for market_sym in markets
        # 1. Check if market data exists
        if !haskey(open_odds, market_sym)
            continue
        end

        o_price = open_odds[market_sym]
        c_price = haskey(close_odds, market_sym) ? close_odds[market_sym] : 0.0
        
        has_result = haskey(outcomes, market_sym)
        is_win = has_result ? outcomes[market_sym] : false
        res_str = has_result ? (is_win ? "WIN" : "LOSS") : "PENDING"
        res_col = has_result ? (is_win ? :green : :red) : :yellow

        # 2. Market Header
        printstyled("──────────────────────────────────────────────────────────────────────────────\n", color=:light_black)
        printstyled(@sprintf(" %-10s ", string(market_sym)), bold=true, color=:cyan)
        printstyled("Result: ", color=:light_black)
        printstyled("$res_str ", bold=true, color=res_col)
        printstyled(@sprintf("| Open: %.2f | Close: %.2f\n", o_price, c_price), color=:light_black)
        println()

        # 3. Table Header
        printstyled(@sprintf(" %-15s | %-8s | %-8s | %-8s | %-8s\n", "Model", "Prob", "Fair", "Edge", "Kelly"), color=:light_blue)
        println(" " * "-"^65)

        # 4. Loop through models and print rows
        for entry in model_inputs
            name = entry.name
            
            if model_data[name] === nothing
                printstyled(@sprintf(" %-15s | %-30s\n", name, "No Data for Match"), color=:light_black)
                continue
            end

            (preds, kelly_dict) = model_data[name]

            if haskey(preds, market_sym)
                chain = preds[market_sym]
                
                prob, fair, edge = summarize_chain(chain, o_price)
                kelly_stake = get(kelly_dict, market_sym, 0.0)

                # Formatting Colors
                edge_col = edge > 0 ? :green : :light_black
                kelly_col = kelly_stake > 0 ? :yellow : :light_black
                
                # Print Row
                printstyled(@sprintf(" %-15s | %5.1f%%   | %6.2f   | ", name, prob*100, fair), color=:white)
                printstyled(@sprintf("%+5.1f%%", edge*100), color=edge_col)
                printstyled(@sprintf("   | %5.1f%%\n", kelly_stake*100), color=kelly_col)
            else
                printstyled(@sprintf(" %-15s | N/A\n", name), color=:light_black)
            end
        end
        println()
    end
    printstyled("══════════════════════════════════════════════════════════════════════════════\n", color=:magenta)
end


######

""""

plotting the paths 

"""
r = results[4][1]

using Plots 
using Statistics
using DataFrames

# --- 1. Helper Functions (Ensure these are defined) ---

function reconstruct_ar1_path(Z_init, Z_steps, σ_vec, ρ_vec)
    n_teams, n_steps, n_samples = size(Z_steps)
    n_rounds = n_steps + 1 
    path = zeros(Float64, n_teams, n_rounds, n_samples)
    
    # Reshape for broadcasting
    σ_b = reshape(σ_vec, 1, n_samples)
    ρ_b = reshape(ρ_vec, 1, n_samples)
    
    # t=1
    path[:, 1, :] .= Z_init .* σ_b 
    
    # t=2..T
    for t in 2:n_rounds
        prev = view(path, :, t-1, :)
        innov = view(Z_steps, :, t-1, :)
        path[:, t, :] .= (prev .* ρ_b) .+ (innov .* σ_b)
    end
    return path
end

function unwrap_ntuple_ar1(tuple_of_arrays)
    n_features = length(tuple_of_arrays)
    n_samples = length(tuple_of_arrays[1])
    out = Matrix{Float64}(undef, n_features, n_samples)
    for (i, arr) in enumerate(tuple_of_arrays)
        out[i, :] .= vec(parent(arr))
    end
    return out
end

# --- 2. Extract Full Paths ---

# Params from chains (assuming 'r' is your chain object)
params = get(r, [:home_adv, :σ_att, :σ_def, :ρ_att, :ρ_def, 
                 :z_att_init, :z_def_init, :z_att_steps, :z_def_steps])

# Unwrap and Reshape
n_teams = vocabulary.mappings[:n_teams]
n_samples = length(params.home_adv)

# Init
Z_att_init = unwrap_ntuple_ar1(params.z_att_init)
Z_def_init = unwrap_ntuple_ar1(params.z_def_init)

# Steps
Z_att_steps_raw = unwrap_ntuple_ar1(params.z_att_steps)
Z_def_steps_raw = unwrap_ntuple_ar1(params.z_def_steps)
n_innovations = div(size(Z_att_steps_raw, 1), n_teams)

Z_att_steps = reshape(Z_att_steps_raw, n_teams, n_innovations, n_samples)
Z_def_steps = reshape(Z_def_steps_raw, n_teams, n_innovations, n_samples)

# Hyperparams
σ_att_vec = vec(params.σ_att)
σ_def_vec = vec(params.σ_def)
ρ_att_vec = vec(params.ρ_att)
ρ_def_vec = vec(params.ρ_def)

# Reconstruct Raw Paths (Teams x Rounds x Samples)
raw_att = reconstruct_ar1_path(Z_att_init, Z_att_steps, σ_att_vec, ρ_att_vec)
raw_def = reconstruct_ar1_path(Z_def_init, Z_def_steps, σ_def_vec, ρ_def_vec)

# Center Constraints (Zero Sum per round)
# Subtract the mean across teams (dim 1) for every sample/round
att_paths = raw_att .- mean(raw_att, dims=1)
def_paths = raw_def .- mean(raw_def, dims=1)

# --- 3. Aggregate for Plotting (Mean across samples) ---

# Shape: (Teams x Rounds)
att_mean = dropdims(mean(att_paths, dims=3), dims=3)
def_mean = dropdims(mean(def_paths, dims=3), dims=3)

# Map back to Team Names
# Invert the mapping: ID -> Name
id_to_team = Dict(v => k for (k, v) in vocabulary.mappings[:team_map])
team_names = [id_to_team[i] for i in 1:n_teams]

# Create a DataFrame for easy inspection
n_rounds = size(att_mean, 2)
rounds = 1:n_rounds

# Helper to plot specific teams
function plot_team_strengths(teams_to_plot::Vector{String})
    p = plot(layout=(2,1), size=(1000, 1000), legend=:outerright)
    
    for team in teams_to_plot
        team_id = vocabulary.mappings[:team_map][team]
        
        # Plot Attack
        plot!(p[1], rounds, att_mean[team_id, :], 
              label=team, lw=2, title="Attack Strength (Higher is Better)")
              
        # Plot Defense (Inverted logic usually: Negative strength often means good defense in Poisson, 
        # BUT check your model formulation. 
        # In `log_λs = home_adv + att_h + def_a`, a HIGH def_a increases opponent goals.
        # So LOWER def parameter is better defense.
        plot!(p[2], rounds, def_mean[team_id, :], 
              label=team, lw=2, title="Defense Parameter (Lower is Better)")
    end
    
    display(p)
end

"""
10-element Vector{InlineStrings.String31}:
 "falkirk-fc"
 "partick-thistle"
 "livingston"
 "hamilton-academical"
 "airdrieonians"
 "queens-park-fc"
 "ayr-united"
 "dunfermline-athletic"
 "raith-rovers"
 "greenock-morton"
"""


plot_team_strengths(["falkirk-fc", "livingston", "hamilton-academical"])

champ =[ 
 "falkirk-fc",
 "partick-thistle", 
 "livingston",
 "hamilton-academical",
 "airdrieonians",
 "queens-park-fc",
 "ayr-united",
 "dunfermline-athletic",
 "raith-rovers",
 "greenock-morton"
]

plot_team_strengths(champ)


###########

"""
not working bit 

"""

df_to_predict = mp 
chains = r 

params = get(chains, [:home_adv, :σ_att, :σ_def, :ρ_att, :ρ_def, 
                      :z_att_init, :z_def_init, :z_att_steps, :z_def_steps])


# Vectors (Samples,)
home_adv_vec = vec(params.home_adv)
σ_att_vec    = vec(params.σ_att)
σ_def_vec    = vec(params.σ_def)
ρ_att_vec    = vec(params.ρ_att)
ρ_def_vec    = vec(params.ρ_def)


function unwrap_ntuple_ar1(tuple_of_arrays)
    n_features = length(tuple_of_arrays)
    n_samples = length(tuple_of_arrays[1])
    out = Matrix{Float64}(undef, n_features, n_samples)
    for (i, arr) in enumerate(tuple_of_arrays)
        out[i, :] .= vec(parent(arr))
    end
    return out
end


# Arrays (Features x Samples) -> Unwrap
Z_att_init_raw = unwrap_ntuple_ar1(params.z_att_init)
Z_def_init_raw = unwrap_ntuple_ar1(params.z_def_init)
Z_att_steps_raw = unwrap_ntuple_ar1(params.z_att_steps)
Z_def_steps_raw = unwrap_ntuple_ar1(params.z_def_steps)


# Reshape for reconstruction
n_teams = vocabulary.mappings[:n_teams]
n_samples = length(home_adv_vec)


Z_att_init = Z_att_init_raw # Already (Features, Samples) where Features=Teams
Z_def_init = Z_def_init_raw 



n_innovations = div(size(Z_att_steps_raw, 1), n_teams) # Should be n_rounds - 1

Z_att_steps = reshape(Z_att_steps_raw, n_teams, n_innovations, n_samples)
Z_def_steps = reshape(Z_def_steps_raw, n_teams, n_innovations, n_samples)


function reconstruct_ar1_path(Z_init, Z_steps, σ_vec, ρ_vec)
    # Dimensions
    n_teams, n_steps, n_samples = size(Z_steps)
    # Z_steps is actually n_rounds-1 long in the time dimension
    n_rounds = n_steps + 1 
    
    # Output container: (Teams, Rounds, Samples)
    path = zeros(Float64, n_teams, n_rounds, n_samples)
    
    # Reshape vectors for broadcasting: (1, 1, Samples)
    σ_b = reshape(σ_vec, 1, 1, n_samples)
    ρ_b = reshape(ρ_vec, 1, 1, n_samples)
    
    # t=1
    path[:, 1, :] .= Z_init .* σ_b # Scale init
    
    # t=2..T
    for t in 2:n_rounds
        prev = path[:, t-1, :]
        innov = Z_steps[:, t-1, :]
        
        # AR1 Update: ρ * prev + σ * innov
        path[:, t, :] .= (prev .* ρ_b) .+ (innov .* σ_b)
    end
    
    return path
end


Z_att_init
Z_att_steps
σ_att_vec
ρ_att_vec
"""
number of samples in the chain 400,
number of teams 10 
number of week / rounds 35, with 34 steps 

julia> Z_att_init
10×400 Matrix{Float64}:

julia> Z_att_steps
10×34×400 Array{Float64, 3}:

julia> σ_att_vec                                                                                                                                            
400-element reshape(::AxisArrays.AxisMatrix{Float64, Matrix{Float64}, Tuple{AxisArrays.Axis{:iter, StepRange{Int64, Int64}}, AxisArrays.Axis{:chain, UnitRan
ge{Int64}}}}, 400) with eltype Float64:

julia> ρ_att_vec                                                                                                                                            
400-element reshape(::AxisArrays.AxisMatrix{Float64, Matrix{Float64}, Tuple{AxisArrays.Axis{:iter, StepRange{Int64, Int64}}, AxisArrays.Axis{:chain, UnitRan
ge{Int64}}}}, 400) with eltype Float64:


"""


raw_att = reconstruct_ar1_path(Z_att_init, Z_att_steps, σ_att_vec, ρ_att_vec)


"""
julia> raw_att = reconstruct_ar1_path(Z_att_init, Z_att_steps, σ_att_vec, ρ_att_vec)
ERROR: DimensionMismatch: cannot broadcast array to have fewer non-singleton dimensions
Stacktrace:
 [1] check_broadcast_shape
   @ ./broadcast.jl:554 [inlined]
 [2] check_broadcast_shape (repeats 2 times)
   @ ./broadcast.jl:560 [inlined]
 [3] check_broadcast_axes
   @ ./broadcast.jl:562 [inlined]
 [4] check_broadcast_axes
   @ ./broadcast.jl:566 [inlined]
 [5] instantiate
   @ ./broadcast.jl:316 [inlined]
 [6] materialize!
   @ ./broadcast.jl:905 [inlined]
 [7] materialize!
   @ ./broadcast.jl:902 [inlined]
 [8] reconstruct_ar1_path(Z_init::Matrix{…}, Z_steps::Array{…}, σ_vec::Base.ReshapedArray{…}, ρ_vec::Base.ReshapedArray{…})
   @ Main ./REPL[90]:15
 [9] top-level scope
   @ REPL[91]:1
Some type information was truncated. Use `show(err)` to see complete types.


"""

function extract_parameters(
  model::AR1Poisson,
  df_to_predict::AbstractDataFrame,
  vocabulary::Vocabulary,
  chains::Chains)

    # --- STEP 1: Fast Parameter Retrieval ---
    params = get(chains, [:home_adv, :σ_att, :σ_def, :ρ_att, :ρ_def, 
                          :z_att_init, :z_def_init, :z_att_steps, :z_def_steps])

    # Vectors (Samples,)
    home_adv_vec = vec(params.home_adv)
    σ_att_vec    = vec(params.σ_att)
    σ_def_vec    = vec(params.σ_def)
    ρ_att_vec    = vec(params.ρ_att)
    ρ_def_vec    = vec(params.ρ_def)

    # Arrays (Features x Samples) -> Unwrap
    Z_att_init_raw = unwrap_ntuple_ar1(params.z_att_init)
    Z_def_init_raw = unwrap_ntuple_ar1(params.z_def_init)
    Z_att_steps_raw = unwrap_ntuple_ar1(params.z_att_steps)
    Z_def_steps_raw = unwrap_ntuple_ar1(params.z_def_steps)

    # Reshape for reconstruction
    n_teams = vocabulary.mappings[:n_teams]
    n_samples = length(home_adv_vec)
    
    # Reshape Init: (Teams, Samples)
    Z_att_init = Z_att_init_raw # Already (Features, Samples) where Features=Teams
    Z_def_init = Z_def_init_raw 

    # Reshape Steps: (Teams, Steps, Samples)
    n_innovations = div(size(Z_att_steps_raw, 1), n_teams) # Should be n_rounds - 1
    Z_att_steps = reshape(Z_att_steps_raw, n_teams, n_innovations, n_samples)
    Z_def_steps = reshape(Z_def_steps_raw, n_teams, n_innovations, n_samples)

    # --- STEP 2: Reconstruct Paths ---
    # Need to reshape Init to (Teams, 1, Samples) for the helper if we wanted to be generic,
    # but let's just pass matching shapes.
    
    # We expand Init to (Teams, Samples) -> (Teams, Samples) used inside helper
    raw_att = reconstruct_ar1_path(Z_att_init, Z_att_steps, σ_att_vec, ρ_att_vec)
    raw_def = reconstruct_ar1_path(Z_def_init, Z_def_steps, σ_def_vec, ρ_def_vec)

    # Center: Subtract mean across teams (dim 1)
    # mean(raw_att, dims=1) results in (1, Rounds, Samples)
    final_att = raw_att .- mean(raw_att, dims=1)
    final_def = raw_def .- mean(raw_def, dims=1)

    # --- STEP 3: Prediction Loop ---
    # We use the LAST time step for prediction (assuming prediction is for the *next* round? 
    # Or based on the specific match round in df_to_predict?)
    
    # NOTE: Usually `extract_parameters` for backtesting predicts the match *at its specific time*.
    # However, if df_to_predict contains matches from the FUTURE relative to training, 
    # we usually project the AR1 forward or take the last known state.
    # For now, let's assume we use the latent strength at the **End of the Chain** (Last Round trained)
    # for all predictions, OR we match the round ID if available.
    
    # GRWPoisson implementation usually assumes we are extracting the *final* strengths 
    # or the specific strengths if the match is in-sample. 
    # Looking at your GRW code: it calculates `final_att` which seems to be the SUM of all steps.
    # This implies it uses the strength at time T (end of training) for the prediction.
    
    # So we slice the last time step:
    att_final_T = final_att[:, end, :] # (Teams, Samples)
    def_final_T = final_def[:, end, :] # (Teams, Samples)
    
    ExtractionValue = NamedTuple{(:λ_h, :λ_a), Tuple{Vector{Float64}, Vector{Float64}}}
    extraction_dict = Dict{Int64, ExtractionValue}()
    sizehint!(extraction_dict, nrow(df_to_predict))
    
    team_map = vocabulary.mappings[:team_map]

    for row in eachrow(df_to_predict)
        h_team = row.home_team
        a_team = row.away_team
        h_id = get(team_map, h_team, 0)
        a_id = get(team_map, a_team, 0)

        # Skip if team not found (or handle error)
        if h_id == 0 || a_id == 0 
            continue 
        end

        # Use views for speed
        att_h = @view att_final_T[h_id, :]
        def_a = @view def_final_T[a_id, :]
        att_a = @view att_final_T[a_id, :]
        def_h = @view def_final_T[h_id, :]

        λ_h = exp.(att_h .+ def_a .+ home_adv_vec)
        λ_a = exp.(att_a .+ def_h)

        extraction_dict[Int(row.match_id)] = (; λ_h, λ_a)
    end

    return extraction_dict
end


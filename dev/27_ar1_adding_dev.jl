# dev/27_ar_1_adding_dev.jl

"""
Dev work regarding the adding of the AR1 model.


added file: src/models/pregame/models-src/ar1-poisson.jl
needs the following functions 
  - ar1_poisson_model
  - kdef struct model - ar1-poisson
  - build_turing_model
  - extract_paramters 
  - reconstruct / trend  -( want) 

    
= src/types-interfaces - TypesInterfaces 
  - add abstract struct type 

= src/models/pregame/pregame-module.jl 
  - export model - export AR1Poisson 
  - wrapper functions extract_paramters - could move into the model as the file is getting large 
  - add file path to model src file 

"""


using BayesianFootball
using DataFrames
using Statistics

using JLD2

using ThreadPinning
using LinearAlgebra
pinthreads(:cores)
BLAS.set_num_threads(1) 


data_store = BayesianFootball.Data.load_default_datastore()

ds = BayesianFootball.Data.DataStore( 
    subset( data_store.matches, 
           :tournament_id => ByRow(isequal(55)),
           :season => ByRow(isequal("24/25"))),
    data_store.odds,
    data_store.incidents
)
  
names(ds.matches)
"""
julia> names(ds.matches)
19-element Vector{String}:
 "tournament_id"
 "season_id"
 "season"
 "match_id"
 "tournament_slug"
 "home_team"
 "away_team"
 "home_score"
 "away_score"
 "home_score_ht"
 "away_score_ht"
 "match_date"
 "round"
 "winner_code"
 "has_xg"
 "has_stats"
 "match_hour"
 "match_dayofweek"
 "match_month"
"""

df_matches = Data.add_match_week_column(ds.matches)


combine( groupby(df_matches, :match_week), 
        nrow 
        )

"""
julia> combine( groupby(df_matches, :match_week), 
               nrow 
               )
38×2 DataFrame
 Row │ match_week  nrow  
     │ Int64       Int64 
─────┼───────────────────
   1 │          1      5
   2 │          2      5
   3 │          3      1
   4 │          4      5
   5 │          5      5
   6 │          6      5
   7 │          7      4
   8 │          8      5
   9 │          9      5
  10 │         10      2
  11 │         11      5
  12 │         12      5
  13 │         13      8
  14 │         14      5
  15 │         15      5
  16 │         16      2
  17 │         17      5
  18 │         18      5
  19 │         19      6
  20 │         20      5
  21 │         21      4
  22 │         22      2
  23 │         23      2
  24 │         24      5
  25 │         25      6
  26 │         26      1
  27 │         27      5
  28 │         28      6
  29 │         29      8
  30 │         30      6
  31 │         31      6
  32 │         32      4
  33 │         33      5
  34 │         34      7
  35 │         35      5
  36 │         36      5
  37 │         37      5
  38 │         38      5

"""


combine( groupby(df_matches, :round), 
        nrow 
        )



"""
SyntheticData run 
"""

using BayesianFootball
using DataFrames
using Statistics

using JLD2

using ThreadPinning
using LinearAlgebra
pinthreads(:cores)
BLAS.set_num_threads(1) 

model = BayesianFootball.Models.PreGame.AR1Poisson()

ds, true_params = BayesianFootball.SyntheticData.generate_synthetic_data_with_params(
    n_teams=10, 
    n_seasons=1,
    legs_per_season=4,
    in_season_volatility=0.05 # Made it volatile so you can see movement in plots!
)

vocabulary = BayesianFootball.Features.create_vocabulary(ds, model) 

splitter_config = BayesianFootball.Data.StaticSplit(
    train_seasons = ["2020/21"], 
    round_col = :round
)

data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)

feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model, splitter_config)

# fs_modded = feature_sets[2:end]

train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=1) 

sampler_conf = BayesianFootball.Samplers.NUTSConfig(n_samples=500, n_chains=6, n_warmup=500) # Use renamed struct

training_config = BayesianFootball.Training.TrainingConfig(sampler_conf, train_cfg)

results = BayesianFootball.Training.train(model, training_config, feature_sets)


JLD2.save_object("debug_ar1_poisson.jld2", results)

# plot 
using MCMCChains

"""
    reconstruct_ar1_tensors(chain, n_teams, n_rounds)

Reconstructs the (Team x Time x Sample) tensors for an AR1 process.
Correctly handles multiple chains by calculating total samples from chain dimensions.
"""
function reconstruct_ar1_tensors(chain, n_teams, n_rounds)
    # 1. Determine Total Samples (Iterations * Chains)
    # size(chain) returns (n_samples, n_vars, n_chains)
    n_samples_per_chain = size(chain, 1)
    n_chains = size(chain, 3)
    n_total_samples = n_samples_per_chain * n_chains
    
    println("Reconstructing tensors for $n_total_samples samples ($n_samples_per_chain x $n_chains)...")

    n_steps = n_rounds - 1

    # --- 2. Extract Hyperparameters (Total_Samples,) ---
    # vec(Array(...)) flattens the (Samples x Chains) matrix into a single vector
    σ_att = vec(Array(chain[:σ_att]))
    σ_def = vec(Array(chain[:σ_def]))
    ρ_att = vec(Array(chain[:ρ_att]))
    ρ_def = vec(Array(chain[:ρ_def]))

    # --- 3. Extract & Reshape Latents (Teams, Time, Total_Samples) ---
    function extract_tensor(param_name, time_dim)
        # Array(group(...)) returns a matrix of shape (Total_Samples, N_Params)
        flat = Array(group(chain, param_name))
        
        # Permute to (N_Params, Total_Samples) for reshaping
        flat_T = permutedims(flat, (2, 1))
        
        # Reshape to (Teams, Time, Total_Samples)
        # Assumes Turing/MCMCChains orders parameters as: Team 1, Team 2... 
        return reshape(flat_T, n_teams, time_dim, n_total_samples)
    end

    Z_att_init  = extract_tensor(:z_att_init, 1)          
    Z_def_init  = extract_tensor(:z_def_init, 1)
    Z_att_steps = extract_tensor(:z_att_steps, n_steps)   
    Z_def_steps = extract_tensor(:z_def_steps, n_steps)

    # --- 4. AR1 Reconstruction Loop ---
    att_tensor = zeros(n_teams, n_rounds, n_total_samples)
    def_tensor = zeros(n_teams, n_rounds, n_total_samples)

    # Reshape scalars for broadcasting: (1, 1, Total_Samples)
    r_shape = (1, 1, n_total_samples)
    σ_a_b = reshape(σ_att, r_shape)
    σ_d_b = reshape(σ_def, r_shape)
    ρ_a_b = reshape(ρ_att, r_shape)
    ρ_d_b = reshape(ρ_def, r_shape)

    # t=1: Scaled Init
    att_tensor[:, 1:1, :] = Z_att_init .* σ_a_b
    def_tensor[:, 1:1, :] = Z_def_init .* σ_d_b

    # t=2..T: AR1 Update
    for t in 2:n_rounds
        # Previous state
        prev_att = att_tensor[:, t-1:t-1, :]
        prev_def = def_tensor[:, t-1:t-1, :]
        
        # Innovation (index is t-1)
        innov_att = Z_att_steps[:, t-1:t-1, :]
        innov_def = Z_def_steps[:, t-1:t-1, :]
        
        # Update: ρ * prev + σ * innov
        att_tensor[:, t:t, :] = (prev_att .* ρ_a_b) .+ (innov_att .* σ_a_b)
        def_tensor[:, t:t, :] = (prev_def .* ρ_d_b) .+ (innov_def .* σ_d_b)
    end

    # --- 5. Centering (Zero-Sum Constraint) ---
    # Subtract the mean across teams at every time step
    att_tensor .-= mean(att_tensor, dims=1)
    def_tensor .-= mean(def_tensor, dims=1)

    return att_tensor, def_tensor
end

# 1. Get the Chain and Dimensions
chain = results[1][1]
n_teams = vocabulary.mappings[:n_teams]
n_rounds = length(unique(ds.matches.round)) # Or derive from true_params

# 2. Reconstruct the Dynamic Tensors
# Returns (Teams, Rounds, Samples)
att_tensor, def_tensor = reconstruct_ar1_tensors(chain, n_teams, n_rounds)

# 3. Prepare "Static" Matrices (Optional)
# The plotting function asks for `alpha_matrix` (Sample x Team) to show a static baseline.
# Since we didn't train a static model here, we can use the TIME-AVERAGE of our AR1 model 
# to see how the average strength compares to the moving strength.
# Mean over time (dim 2) -> (Teams, 1, Samples) -> Permute to (Samples, Teams)
alpha_matrix = permutedims(dropdims(mean(att_tensor, dims=2), dims=2), (2, 1))
beta_matrix  = permutedims(dropdims(mean(def_tensor, dims=2), dims=2), (2, 1))

# 4. Plot
# Replace `6` with the team index you want to visualize
team_id_to_plot = 3

p1 = BayesianFootball.SyntheticData.plot_dynamic_trajectory(
    team_id_to_plot, 
    true_params, 
    att_tensor, 
    def_tensor; 
    stat_att=alpha_matrix, 
    stat_def=beta_matrix
)

display(p1)


############ ----
using StatsPlots

density(chain[:σ_def])
density(chain[:ρ_att])
density(chain[:ρ_def])

"""
init build 
"""

using Revise
using BayesianFootball
using DataFrames
using Statistics



model = BayesianFootball.Models.PreGame.AR1Poisson()


ds, true_params = BayesianFootball.SyntheticData.generate_synthetic_data_with_params(
    n_teams=10, 
    n_seasons=1,
    legs_per_season=4,
    in_season_volatility=0.05 # Made it volatile so you can see movement in plots!
)

vocabulary = BayesianFootball.Features.create_vocabulary(ds, model) 

splitter_config = BayesianFootball.Data.StaticSplit(
    train_seasons = ["2020/21"], 
    round_col = :round
)

data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)

feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model, splitter_config)

turing_model = BayesianFootball.Models.PreGame.build_turing_model(model, feature_sets[1][1])
"""

julia> turing_model = BayesianFootball.Models.PreGame.build_turing_model(model, feature_sets[1][1])
this is a place holder for the AR1Model build_turing_model
"""

ldf_1 = Turing.LogDensityFunction(turing_model)

# testing the extract_parameters 
using JLD2

# using another models chains as dummy data 
results = JLD2.load_object("training_results.jld2")

mp = filter( row -> row.round == 10 , ds.matches)
predict_config = BayesianFootball.Predictions.PredictionConfig( BayesianFootball.Markets.get_standard_markets() )

rr = BayesianFootball.Models.PreGame.extract_parameters(model, mp, vocabulary, results[1][1])
"""
julia> rr = BayesianFootball.Models.PreGame.extract_parameters(model, mp, vocabulary, results[1][1])
this is a place holder for the AR1Model extract_parameters level 1 
"""


# testing the wrapper 
split_col_sym = :round
all_split = sort(unique(ds.matches[!, split_col_sym]))
prediction_split_keys = all_split[3:end] 
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
"""
julia> oos_results = BayesianFootball.Models.PreGame.extract_parameters(
           model,
           dfs_to_predict, 
           vocabulary,
           results
       )
this is a place holder for the AR1Model extract_parameters level 2 - the wrapper. 
"""

using BayesianFootball
using DataFrames
using Dates
using ReverseDiff, Memoization
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

include("/home/james/bet_project/models_julia/workspace/basic_state_space/setup.jl")

using .AR1NegativeBinomial

const EXPERIMENT_GROUP_NAME = "ar1_poisson_test"
const SAVE_PATH = "./experiments"
# Make sure this path is correct for your system
const DATA_PATH = "/home/james/bet_project/football/uk_football_data_20_26" 

data_files = DataFiles(DATA_PATH)
data_store = DataStore(data_files)
add_global_round_column!(data_store.matches)


sample_config = BayesianFootball.ModelSampleConfig(500, true) # 1500 steps, show progress bar
model_def = AR1NegativeBinomialModel()
run_name = "ar1_negbi_2425_to_2526"



cv_config = BayesianFootball.TimeSeriesSplitsConfig(
    ["24/25", "25/26"],
    [],
    :round
)

mapping_funcs = BayesianFootball.MappingFunctions(BayesianFootball.create_list_mapping)

config = ExperimentConfig(run_name, model_def, cv_config, sample_config, mapping_funcs)

global_mapping = BayesianFootball.MappedData(data_store, mapping_funcs)
train_df = filter(row -> row.season in config.cv_config.base_seasons, data_store.matches) 

training_morphism = BayesianFootball.compose_training_morphism(
    config.model_def,
    config.sample_config,
    global_mapping
)


# train the model
start_time = now()
trained_chains = training_morphism(train_df, "test sample")
end_time = now()
run_duration_seconds = Dates.value(end_time - start_time) / 1000

# save 
result = ExperimentResult(
    [trained_chains], # Stored in a vector to match the expected format [cite: 86]
    global_mapping,
    hash(config),
    run_duration_seconds
)

run_manager = prepare_run(EXPERIMENT_GROUP_NAME, config, SAVE_PATH)
save(run_manager, result)
println("Model saved successfully for run: $(run_name)")



################################################################################
# Predict stuff
################################################################################
using BayesianFootball
using DataFrames
using Dates
using ReverseDiff, Memoization
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

include("/home/james/bet_project/models_julia/workspace/basic_state_space/setup.jl")

using .AR1NegativeBinomial

include("/home/james/bet_project/models_julia/workspace/basic_state_space/prediction.jl")

using .AR1NegBiPrediction



file_path = "/home/james/bet_project/models_julia/experiments/ar1_poisson_test/ar1_negbi_2425_to_2526_20250926-173118"
loaded_model = load_model(file_path)
mapping = loaded_model.result.mapping
chains = loaded_model.result.chains_sequence[1]


posterior_samples = BayesianFootball.extract_posterior_samples(
    loaded_model.config.model_def,
    chains.ft,
    mapping
);

last_training_round = posterior_samples.n_rounds
next_round = last_training_round + 1

team_name_home = "west-bromwich-albion"
team_name_away = "leicester-city"

match_to_predict = DataFrame(
    home_team=team_name_home,
    away_team=team_name_away,
    tournament_id=2,
    global_round = next_round, # Use the calculated next_round
    home_score_ht=0, away_score_ht=0, home_score=0, away_score=0
)

# --- 4. Run the prediction ---
features = BayesianFootball.create_master_features(match_to_predict, mapping)

predictions = predict_ar1_neg_bin_match_lines(
    loaded_model.config.model_def,
    chains,
    features,
    mapping
);


using Statistics
println("\n--- Predicted Odds for Next Round ---")
println("Home Win: ", round(mean( 1 ./ predictions.ft.home), digits=2))
println("Away Win: ", round(mean( 1 ./ predictions.ft.away), digits=2))
println("Draw:     ", round(mean( 1 ./ predictions.ft.draw), digits=2))


println("Home Win: ", round(mean( 1 ./ predictions.ht.home), digits=2))
println("Away Win: ", round(mean( 1 ./ predictions.ht.away), digits=2))
println("Draw:     ", round(mean( 1 ./ predictions.ht.draw), digits=2))

round(mean( 1 ./ predictions.ft.under_05), digits=2)
round(mean( 1 ./ predictions.ft.under_15), digits=2)
round(mean( 1 ./ predictions.ft.under_25), digits=2)
round(mean( 1 ./ predictions.ft.under_35), digits=2)

round(mean( 1 ./ predictions.ft.btts), digits=2)

round(mean( 1 ./ predictions.ht.under_05), digits=2)
round(mean( 1 ./ predictions.ht.under_15), digits=2)
round(mean( 1 ./ predictions.ht.under_25), digits=2)



p_cs = Dict( k => mean(v) for (k,v) in predictions.ft.correct_score)
sort(collect(p_cs), by = x -> x[2], rev=true)



o_cs = Dict( k => mean(1 ./ v) for (k,v) in predictions.ft.correct_score)

cs = Dict( k => mean(1 ./ v) for (k,v) in match_predictions.ft.correct_score)
"""
julia> println("Home Win: ", round(mean( 1 ./ predictions.ft.home), digits=2))
Home Win: 2.21

julia> println("Away Win: ", round(mean( 1 ./ predictions.ft.away), digits=2))
Away Win: 4.0

julia> println("Draw:     ", round(mean( 1 ./ predictions.ft.draw), digits=2))
Draw:     3.83

julia> sort(collect(p_cs), by = x -> x[2], rev=true)
19-element Vector{Pair{Any, Float64}}:
           (1, 0) => 0.13063348977732775
           (0, 0) => 0.11383430019747878
           (1, 1) => 0.10829051032357198
           (0, 1) => 0.09452173243776779
           (2, 0) => 0.08760501646850043
 "other_home_win" => 0.07606088360011792
           (2, 1) => 0.072490060006086
           (1, 2) => 0.05252229440057993
           (0, 2) => 0.04590781959094237
           (3, 0) => 0.04506541876372435
           (3, 1) => 0.03721699692809202
           (2, 2) => 0.035106860638355296
 "other_away_win" => 0.025257785083086886
           (1, 3) => 0.019557482340891323
           (3, 2) => 0.017995500089477926
           (0, 3) => 0.017113161261752705
           (2, 3) => 0.013057880883876521
           (3, 3) => 0.006685336713987912
     "other_draw" => 0.0010386240777138715

julia> o_cs = Dict( k => mean(1 ./ v) for (k,v) in predictions.ft.correct_score)
Dict{Any, Float64} with 19 entries:
  (1, 2)           => 19.9275
  (3, 1)           => 29.0851
  (0, 2)           => 24.1837
  (1, 3)           => 59.4072
  (0, 3)           => 72.3275
  "other_home_win" => 17.6606
  (3, 2)           => 62.0144
  (2, 0)           => 11.9154
  (3, 3)           => 183.682
  "other_draw"     => 1612.67
  (2, 1)           => 13.9739
  (2, 2)           => 29.8686
  (2, 3)           => 88.7567
  (1, 0)           => 7.91761
  (0, 0)           => 9.57202
  (1, 1)           => 9.30074
  (0, 1)           => 11.2609
  (3, 0)           => 24.8467
  "other_away_win" => 58.3645

"""



using Plots

team1_name = "west-bromwich-albion"
team2_name = "leicester-city"


team1_id = loaded_model.result.mapping.team[team1_name]
team2_id = loaded_model.result.mapping.team[team2_name]

log_α_centered = posterior_samples.log_α_centered
log_β_centered = posterior_samples.log_β_centered
n_rounds = posterior_samples.n_rounds

# --- 3. Calculate the posterior mean AND STANDARD DEVIATION over time ---
# Mean calculations
team1_attack_mean = vec(mean(log_α_centered[:, team1_id, :], dims=1))
team1_defense_mean = vec(mean(log_β_centered[:, team1_id, :], dims=1))
team2_attack_mean = vec(mean(log_α_centered[:, team2_id, :], dims=1))
team2_defense_mean = vec(mean(log_β_centered[:, team2_id, :], dims=1))

# Standard deviation calculations
team1_attack_std = vec(std(log_α_centered[:, team1_id, :], dims=1))
team1_defense_std = vec(std(log_β_centered[:, team1_id, :], dims=1))
team2_attack_std = vec(std(log_α_centered[:, team2_id, :], dims=1))
team2_defense_std = vec(std(log_β_centered[:, team2_id, :], dims=1))


p = plot(
    layout=(1, 2),
    size=(1400, 500),
    legend=:outertopright,
    link=:y,
    xlabel="Global Round"
)

# Subplot 1: Attacking Strength
plot!(p[1], 1:n_rounds, team1_attack_mean,
    ribbon = 1 .* team1_attack_std, # <-- ADDED RIBBON
    fillalpha = 0.2,                # Make ribbon transparent
    # label = team1_name,
    title = "Attacking Strength (log α)",
    ylabel = "Parameter Value",
    lw = 2
)
plot!(p[1], 1:n_rounds, team2_attack_mean,
    ribbon = 1 .* team2_attack_std, # <-- ADDED RIBBON
    fillalpha = 0.2,
    # label = team2_name,
    lw = 2
)

# Subplot 2: Defensive Strength
plot!(p[2], 1:n_rounds, team1_defense_mean,
    ribbon = 1 .* team1_defense_std, # <-- ADDED RIBBON
    fillalpha = 0.2,
    label = team1_name,
    title = "Defensive Strength (log β)",
    lw = 2
)
plot!(p[2], 1:n_rounds, team2_defense_mean,
    ribbon = 1 .* team2_defense_std, # <-- ADDED RIBBON
    fillalpha = 0.2,
    label = team2_name,
    lw = 2
)


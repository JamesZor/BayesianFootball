using BayesianFootball
using DataFrames
using Dates
using Statistics, StatsBase, StatsPlots, Distributions
using CSV 


include("/home/james/bet_project/models_julia/workspace/basic_state_space/setup.jl")
include("/home/james/bet_project/models_julia/workspace/basic_state_space/prediction.jl")
include( "/home/james/bet_project/models_julia/workspace/basic_state_space/matchday_utils_ssm.jl")
include( "/home/james/bet_project/models_julia/workspace/basic_state_space/analysis_functions.jl")


using .MatchDayUtilsSSM

using .AR1NegativeBinomial
using .AR1NegBiPrediction
using .AR1StateSpace
using .AR1Prediction
using .AnalysisSSM

all_model_paths = Dict(
  "ssm_bineg" => "/home/james/bet_project/models_julia/experiments/ar1_poisson_test/ar1_negbi_2425_to_2526_20250926-173118",
  "ssm_poiss" => "/home/james/bet_project/models_julia/experiments/ar1_poisson_test/ar1_poisson_2425_to_2526_20250926-135921",
  "ssm_full_bineg" => "/home/james/bet_project/models_julia/experiments/dynamic_negbin_models/dyn_negbin_2425_to_2526_20250928-132059"

)



todays_matches = get_todays_matches(["scotland", "england"]; cli_path=CLI_PATH)

patterns_to_exclude = [" U21", r"\bB\b"]
# Filter out rows where ANY of the patterns are found
filter!(todays_matches) do row
    !any(p -> occursin(p, row.event_name), patterns_to_exclude)
end
todays_matches


odds_df = fetch_all_market_odds(
    todays_matches,
    MARKET_LIST;
    cli_path=CLI_PATH
)

save_odds_to_csv(odds_df, "data/")

odds_df = CSV.read("/home/james/bet_project/models_julia/data/market_odds_2025-09-27.csv", DataFrame, header=1)

loaded_models_all = load_models_from_paths(all_model_paths)


match_to_analyze = todays_matches[1, :]

# (comparison_df, prediction_matrices, market_book) = generate_match_analysis(
#     match_to_analyze,
#     odds_df, # Your wide DataFrame of all market odds
#     loaded_models_all,
#     MARKET_LIST # Use your comprehensive or specific market list here
# );
#

m1 = loaded_models_all["ssm_bineg"]
m1 = loaded_model
mapping = m1.result.mapping
chain = m1.result.chains_sequence[1]


posterior_samples = BayesianFootball.extract_posterior_samples(
    m1.config.model_def,
    chain.ft,
    mapping
);

last_training_round = posterior_samples.n_rounds
next_round = last_training_round + 1

team_name_home = "coventry-city"
team_name_away = "birmingham-city"

team_name_home = match_to_analyze.home_team
team_name_away = match_to_analyze.away_team

match_to_predict = DataFrame(
    home_team=team_name_home,
    away_team=team_name_away,
    tournament_id=2,
    global_round = next_round, # Use the calculated next_round
    home_score_ht=0, away_score_ht=0, home_score=0, away_score=0
)

features = BayesianFootball.create_master_features(match_to_predict, mapping)

predictions = predict_ar1_neg_bin_match_lines(
    m1.config.model_def,
    chains,
    features,
    mapping
);

println("Home Win: ", round(mean( 1 ./ predictions.ft.home), digits=2))
println("Away Win: ", round(mean( 1 ./ predictions.ft.away), digits=2))
println("Draw:     ", round(mean( 1 ./ predictions.ft.draw), digits=2))


println("Home Win: ", round(median( 1 ./ predictions.ft.home), digits=2))
println("Away Win: ", round(median( 1 ./ predictions.ft.away), digits=2))
println("Draw:     ", round(median( 1 ./ predictions.ft.draw), digits=2))


temp = Dict( "ssm_bineg" => predictions)
create_odds_dataframe1(temp)

filter(row -> row.event_name== match_to_analyze.event_name, odds_df)


"""
    generate_match_analysis(match_to_analyze, odds_df, loaded_models_all)

Generates model predictions and fetches market odds for a single match,
returning a combined DataFrame for comparison.
"""
function generate_match_analysis1(
    match_to_analyze::DataFrameRow,
    odds_df::DataFrame,
    loaded_models_all::Dict
)
    # --- 1. Generate Predictions for All Models ---
    all_model_predictions = Dict{String, Any}()
    for (model_name, model_data) in loaded_models_all
        mapping = model_data.result.mapping
        chains = model_data.result.chains_sequence[1]
        posterior_samples = BayesianFootball.extract_posterior_samples(
            model_data.config.model_def, chains.ft, mapping
        )
        next_round = posterior_samples.n_rounds + 1

        match_to_predict = DataFrame(
            home_team=match_to_analyze.home_team,
            away_team=match_to_analyze.away_team,
            tournament_id=2, 
            global_round = next_round,
            home_score_ht=0, away_score_ht=0, home_score=0, away_score=0
        )
        features = BayesianFootball.create_master_features(match_to_predict, mapping)

        predictions = if occursin("negbi", model_name) || occursin("bineg", model_name)
            AR1NegBiPrediction.predict_ar1_neg_bin_match_lines(
                model_data.config.model_def, chains, features, mapping
            )
        elseif occursin("poiss", model_name)
             AR1Prediction.predict_ar1_match_lines( # Assuming this is the correct function
                model_data.config.model_def, chains, features, mapping
            )
        else
            println("Warning: No prediction function found for model type in '$model_name'.")
            nothing
        end
        
        if !isnothing(predictions)
            all_model_predictions[model_name] = predictions
        end
    end

    isempty(all_model_predictions) && return DataFrame()
    model_odds_df = create_odds_dataframe1(all_model_predictions)

    # --- 2. Fetch and Reformat Market Odds ---
    market_odds_row_df = filter(row -> row.event_name == match_to_analyze.event_name, odds_df)
    market_df_reformatted = DataFrame()

    if !isempty(market_odds_row_df)
        market_row = market_odds_row_df[1, :]
        
        market_map = Dict(
            :Home => "ft_1x2_home_back", :Draw => "ft_1x2_draw_back", :Away => "ft_1x2_away_back",
            :O25 => "ft_ou_25_over_back", :U25 => "ft_ou_25_under_back",
            :BTTS_Yes => "ft_btts_yes_back", :BTTS_No => "ft_btts_no_back",
            :O15 => "ft_ou_15_over_back", :U15 => "ft_ou_15_under_back"
        )
        
        market_data = Dict{Symbol, Any}(:Model => "Market", :Time => "FT")
        for (target_col, source_col) in market_map
            if hasproperty(market_row, Symbol(source_col))
                market_data[target_col] = market_row[Symbol(source_col)]
            end
        end
        market_df_reformatted = DataFrame([market_data])
    else
        println("Info: No market odds found for $(match_to_analyze.event_name)")
    end

    # --- 3. Combine and Finalize ---
    combined_df = vcat(model_odds_df, market_df_reformatted, cols=:union)
    combined_df[!, :event_name] .= match_to_analyze.event_name
    
    # **FIX**: This is the corrected, simplified line that caused the error.
    # The `:` automatically selects all remaining columns.
    return select(combined_df, :event_name, :Model, :Time, :Home, :Draw, :Away, :O25, :U25, :)
end




"""
    display_all_match_comparisons(todays_matches, odds_df, loaded_models_all)

Iterates through matches grouped by kickoff time, generates analysis for each,
and displays a consolidated comparison table for each time slot.
"""
function display_all_match_comparisons(
    todays_matches::DataFrame,
    odds_df::DataFrame,
    loaded_models_all::Dict
)
    grouped_matches = groupby(todays_matches, :time)

    for kickoff_group in grouped_matches
        kickoff_time = kickoff_group.time[1]
        println("\n" * "="^80)
        println(" KICK-OFF: $(kickoff_time)")
        println("="^80)

        # Generate analysis for each match in the current time group
        list_of_dfs = [
            generate_match_analysis1(match_row, odds_df, loaded_models_all) 
            for match_row in eachrow(kickoff_group)
        ]
        
        # Combine all non-empty DataFrames for this kickoff time
        kickoff_comparison_df = vcat(filter(!isempty, list_of_dfs)..., cols=:union)

        if !isempty(kickoff_comparison_df)
            # Display the full comparison table for this time slot
            show(kickoff_comparison_df, allrows=true, eltypes=false, summary=false)
            println("\n")
        else
            println("No valid data to display for this kickoff time.")
        end
    end
end


# --- EXECUTION ---
# Now, you can run the analysis with a single function call:
a1 = filter( row -> row.time=="11:30", todays_matches)
display_all_match_comparisons(a1, odds_df, loaded_models_all);



function process_single_match(match_to_analyze::DataFrameRow, loaded_models_all::Dict, odds_df::DataFrame)
    println("\n" * "="^80)
    println("▶️  Processing Match: $(match_to_analyze.event_name)")
    println("="^80)

    try
        all_model_predictions = Dict{String, Any}()

        # Loop through your models (ssm_bineg, ssm_poiss, etc.)
        for (model_name, model_data) in loaded_models_all
            mapping = model_data.result.mapping
            chains = model_data.result.chains_sequence[1]
            posterior_samples = BayesianFootball.extract_posterior_samples(
                model_data.config.model_def, chains.ft, mapping
            )
            next_round = posterior_samples.n_rounds + 1

            match_to_predict = DataFrame(
                home_team=match_to_analyze.home_team,
                away_team=match_to_analyze.away_team,
                tournament_id=2,
                global_round = next_round,
                home_score_ht=0, away_score_ht=0, home_score=0, away_score=0
            )

            # This will now work without crashing on unknown teams
            features = BayesianFootball.create_master_features(match_to_predict, mapping)

            predictions = if occursin("negbi", model_name) || occursin("bineg", model_name)
                AR1NegBiPrediction.predict_ar1_neg_bin_match_lines(model_data.config.model_def, chains, features, mapping)
            elseif occursin("poiss", model_name)
                AR1Prediction.predict_ar1_match_lines(model_data.config.model_def, chains, features, mapping)
            end
            
            all_model_predictions[model_name] = predictions
        end

        # --- Display Model Odds ---
        println("\nMODEL ODDS:")
        model_odds = create_odds_dataframe1(all_model_predictions)
        filter!(row->row.Time=="FT", model_odds)
        show(model_odds; allrows=true, summary=false)
        println("\n")

        # --- Display Market Odds ---
        println("MARKET ODDS:")
        market_odds = filter(row -> row.event_name == match_to_analyze.event_name, odds_df)
        
        if isempty(market_odds)
            println("--> No market odds found.")
        else
            # Select only the most relevant market columns for a clean view
            cols_to_show = ["ft_1x2_home_back", "ft_1x2_draw_back", "ft_1x2_away_back", "ft_ou_05_under_back", "ft_ou_15_under_back", "ft_ou_25_under_back", "ft_btts_yes_back"]
            existing_cols = filter(c -> hasproperty(market_odds, Symbol(c)), cols_to_show)
            show(select(market_odds, existing_cols); summary=false)
        end
        println("\n")

    catch e
        # This block catches the error and prevents the program from crashing
        if isa(e, KeyError)
            println("\n‼️  ERROR: SKIPPING MATCH. Team not found in model's training data: $(e.key)")
        else
            println("\n‼️  An unexpected error occurred while processing this match: $e")
        end
    end
end

aa = todays_matches[3, :];
process_single_match(aa, loaded_models_all, odds_df)



for tt in eachrow(a1)
    process_single_match(tt, loaded_models_all, odds_df)
end


####
# plot analysis 
#### 

function print_1x2(predictions)
println("Home Win: ", round(mean( 1 ./ predictions.ft.home), digits=2),"  |  " ,  round(median( 1 ./ predictions.ft.home), digits=2))
println("away Win: ", round(mean( 1 ./ predictions.ft.away), digits=2),"  |  " ,  round(median( 1 ./ predictions.ft.away), digits=2))
println("draw Win: ", round(mean( 1 ./ predictions.ft.draw), digits=2),"  |  " ,  round(median( 1 ./ predictions.ft.draw), digits=2))
end


function print_under(predictions)
  println("under 05, mean: ", round(mean( 1 ./ predictions.ft.under_05),digits=2), " | ", round(median( 1 ./ predictions.ft.under_05), digits=2) )
  println("under 15, mean: ", round(mean( 1 ./ predictions.ft.under_15),digits=2), " | ", round(median( 1 ./ predictions.ft.under_15), digits=2) )
  println("under 25, mean: ", round(mean( 1 ./ predictions.ft.under_25),digits=2), " | ", round(median( 1 ./ predictions.ft.under_25), digits=2) )
  println("under 35, mean: ", round(mean( 1 ./ predictions.ft.under_35),digits=2), " | ", round(median( 1 ./ predictions.ft.under_35), digits=2) )
end

using Plots
using Statistics


function plot_attack_defence(team1_name, team2_name, loaded_model, posterior_samples)

team1_id = loaded_model.result.mapping.team[team1_name]
team2_id = loaded_model.result.mapping.team[team2_name]

# --- 2. Get the full time-series of the parameters ---
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


# --- 4. Create the 1x2 plot with ribbons ---
p = plot(
    layout=(1, 2),
    # size=(1400, 500),
    size=(900, 500),
    legend=:bottomleft,
    # link=:y,
    xlabel="Global Round"
)

# Subplot 1: Attacking Strength
plot!(p[1], 1:n_rounds, team1_attack_mean,
    ribbon = 1 .* team1_attack_std, # <-- ADDED RIBBON
    fillalpha = 0.2,                # Make ribbon transparent
    label = team1_name,
    title = "Attacking Strength (log α)",
    ylabel = "Parameter Value",
    lw = 2
)
plot!(p[1], 1:n_rounds, team2_attack_mean,
    ribbon = 1 .* team2_attack_std, # <-- ADDED RIBBON
    fillalpha = 0.2,
    label = team2_name,
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


end 

function plot_1x2(predictions)
p = plot(
    layout=(1, 1),
    # size=(1400, 500),
    size=(900, 500),
    legend=:topleft,
    # link=:y,
    ylabel ="likelihood",
    xlabel="probability"
)
  density!(p, predictions.ft.home, label="home")
  density!(p, predictions.ft.away, label="away")
  density!(p, predictions.ft.draw, label="draw")


end 

function plot_xg(predictions)
p = plot(
    layout=(1, 1),
    # size=(1400, 500),
    size=(900, 500),
    legend=:topleft,
    # link=:y,
    title =" predicted goals",
    ylabel ="likelihood",
    xlabel="probability"
)
  density!(p, predictions.ft.λ_h, label="home")
  density!(p, predictions.ft.λ_a, label="away")
end 

function plot_xg_t(predictions)
p = plot(
    layout=(1, 1),
    # size=(1400, 500),
    size=(900, 500),
    legend=:topleft,
    # link=:y,
    title =" predicted goals",
    ylabel ="likelihood",
    xlabel="probability"
)
  density!(p, predictions.ft.λ_h, label="home")
  density!(p, predictions.ft.λ_a, label="away")
  density!(p, predictions.ft.λ_h .+  predictions.ft.λ_a, label="total")
end 






m1 = loaded_models_all["ssm_bineg"]
mapping = m1.result.mapping
chains = m1.result.chains_sequence[1]
posterior_samples = BayesianFootball.extract_posterior_samples(
    m1.config.model_def,
    chains.ft,
    mapping
);

last_training_round = posterior_samples.n_rounds
next_round = last_training_round + 1

league_id_to_predict = mapping.league["1"]

#
#
# game 1 
todays_matches
match_to_analyze = todays_matches[2, :]

team_name_home = match_to_analyze.home_team
team_name_away = match_to_analyze.away_team
league_id_to_predict = mapping.league["54"]

match_to_predict = DataFrame(
    home_team=team_name_home,
    away_team=team_name_away,
    tournament_id=league_id_to_predict,
    global_round = next_round, # Use the calculated next_round
    home_score_ht=0, away_score_ht=0, home_score=0, away_score=0
)

features = BayesianFootball.create_master_features(match_to_predict, mapping)

predictions = predict_ar1_neg_bin_match_lines(
    m1.config.model_def,
    chain,
    features,
    mapping
);

print_1x2(predictions)
print_under(predictions)

plot_1x2(predictions)
plot_xg(predictions)
plot_xg_t(predictions)



plot_attack_defence(team_name_home, team_name_away, m1, posterior_samples)


p_cs = Dict( k => mean(v) for (k,v) in predictions.ft.correct_score)
sort(collect(p_cs), by = x -> x[2], rev=true)

o_cs = Dict( k => mean(1 ./ v) for (k,v) in predictions.ft.correct_score)


mean( 1 ./ predictions.ft.under_05)
mean( 1 ./ predictions.ft.under_15)
mean( 1 ./ predictions.ft.under_25)
mean( 1 ./ predictions.ft.under_35)




match_to_analyze


"""
***************************************
Aston Villa v Fulham
*********************************
full negbi 
Home Win: 2.43  |  2.31
away Win: 3.88  |  3.64
draw Win: 3.68  |  3.6
===============
under 05, mean: 11.0 | 9.75
under 15, mean: 3.21 | 2.99
under 25, mean: 1.74 | 1.67
under 35, mean: 1.28 | 1.25


Livingston v Rangers
Home Win: 2.42  |  2.33
away Win: 3.93  |  3.57
draw Win: 3.7  |  3.6

"""




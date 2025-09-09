using BayesianFootball


data_files = DataFiles("/home/james/bet_project/football_data/scot_nostats_20_to_24")
data_store = DataStore(data_files)
r1 = load_experiment("/home/james/bet_project/models_julia/experiments/maher_basic_test_20250820_231427.jld2")
result = r1[1]
experiment_config = r1[2]
mapping = result.mapping
target_matches = filter(row -> row.season == experiment_config.cv_config.target_season, data_store.matches)

# set up
matches_prediction =  BayesianFootball.predict_target_season(target_matches, result, mapping)
matches_odds = BayesianFootball.process_matches_odds(data_store, target_matches)
matches_results = BayesianFootball.process_matches_results(data_store, target_matches)
matches_activelines = BayesianFootball.process_matches_active_minutes(data_store, target_matches)
# kelly bit 
kelly_config = BayesianFootball.Kelly.Config(0.02, 0.05)  # 2% commission, 5% probability value threshold
matches_kelly = process_matches_kelly(matches_prediction, matches_odds, kelly_config)


matches_prediction_1 =  BayesianFootball.predict_target_season_fix(target_matches, result.chains_sequence[8], mapping)
matches_kelly_1 = process_matches_kelly(matches_prediction_1, matches_odds, kelly_config)



### performace card 
include("/home/james/bet_project/models_julia/notebooks/08_eval/performace_card_setup.jl")



## main 1 
c_values = 0.01:0.01:0.99
evaluation_cube = build_evaluation_cube(matches_kelly, matches_odds, matches_results, c_values)
println("   - Cube built successfully with $(length(evaluation_cube.match_ids)) matches.")




# C. Define the analysis groups for segmented performance review
analysis_groups = [
    (name="Total Portfolio", filters=Dict()),
    (name="Premiership", filters=Dict(:tournament_slug => "premiership")),
    (name="Championship", filters=Dict(:tournament_slug => "championship")),
    (name="Season 23/24", filters=Dict(:season => "23/24")),
    (name="Season 24/25", filters=Dict(:season => "24/25")),
    (name="Premiership 24/25", filters=Dict(:tournament_slug => "premiership", :season => "24/25")),
]



analysis_groups = [
    (name="Total Portfolio", filters=Dict()),
    (name="Premiership", filters=Dict(:tournament_slug => "premiership")),
    (name="Championship", filters=Dict(:tournament_slug => "championship")),
]

# --- 4. RUN ANALYSIS & GENERATE SCORECARDS ---
println("Calculating performance scorecards for $(length(analysis_groups)) groups...")
all_results_df = DataFrame()

for group in analysis_groups
    # A. Filter match_ids for the current group
    filtered_df = target_matches
    for (col, val) in group.filters
        filtered_df = filter(row -> row[col] == val, filtered_df)
    end
    num_potential_matches = nrow(filtered_df)
    group_match_ids = Set(filtered_df.match_id)
    
    # B. Get the corresponding row indices from the cube (this is very fast)
    row_indices = [evaluation_cube.match_id_map[id] for id in evaluation_cube.match_ids if id in group_match_ids]
    
    if isempty(row_indices)
        println("   - Skipping group '$(group.name)', no matching matches found in the cube.")
        continue
    end

    # C. Slice the cube data
    cube_slice = (
        stakes = evaluation_cube.stakes[row_indices, :, :],
        outcomes = evaluation_cube.outcomes[row_indices, :],
        c_values = evaluation_cube.c_values,
        market_map = evaluation_cube.market_map
    )

    # D. Generate the performance object and convert to a DataFrame
    performance_struct = PerformanceAnalytics.calculate_market_performance(
        group.name,
        num_potential_matches,
        cube_slice
    )
    
    # E. Convert the nested struct to a tidy DataFrame and append
    group_df = PerformanceAnalytics.performance_to_dataframe(performance_struct)
    # Add a column to identify which group this result belongs to
    group_df.group_name = fill(group.name, nrow(group_df))
    append!(all_results_df, group_df)
end



# Reorder columns for better readability
final_cols = ["group_name", "analysis_name", "max_roi", "c_at_max_roi", "max_sharpe", "num_bets", "total_profit", "max_drawdown", "num_cube_matches"]
display_df = select(all_results_df, final_cols...)

display_df

df_t = filter(row -> row.group_name =="Total Portfolio", all_results_df)
df_c = filter(row -> row.group_name =="Championship", all_results_df)
df_p = filter(row -> row.group_name =="Premiership", all_results_df)

# For better display, you might want to format the numbers
# For now, we print the selected columns.
show(stdout, display_df, allcols=true, truncate=100)
println("\n")

### compare 

c_values = 0.01:0.01:0.99

evaluation_cube = build_evaluation_cube(matches_kelly, matches_odds, matches_results, c_values)
evaluation_cube_1 = build_evaluation_cube(matches_kelly_1, matches_odds, matches_results, c_values)

analysis_groups = [
    (name="Total Portfolio", filters=Dict()),
]


#1 
all_results_df = DataFrame()
for group in analysis_groups
    # A. Filter match_ids for the current group
    filtered_df = target_matches
    for (col, val) in group.filters
        filtered_df = filter(row -> row[col] == val, filtered_df)
    end
    num_potential_matches = nrow(filtered_df)
    group_match_ids = Set(filtered_df.match_id)
    
    # B. Get the corresponding row indices from the cube (this is very fast)
    row_indices = [evaluation_cube.match_id_map[id] for id in evaluation_cube.match_ids if id in group_match_ids]
    
    if isempty(row_indices)
        println("   - Skipping group '$(group.name)', no matching matches found in the cube.")
        continue
    end

    # C. Slice the cube data
    cube_slice = (
        stakes = evaluation_cube.stakes[row_indices, :, :],
        outcomes = evaluation_cube.outcomes[row_indices, :],
        c_values = evaluation_cube.c_values,
        market_map = evaluation_cube.market_map
    )

    # D. Generate the performance object and convert to a DataFrame
    performance_struct = PerformanceAnalytics.calculate_market_performance(
        group.name,
        num_potential_matches,
        cube_slice
    )
    
    # E. Convert the nested struct to a tidy DataFrame and append
    group_df = PerformanceAnalytics.performance_to_dataframe(performance_struct)
    # Add a column to identify which group this result belongs to
    group_df.group_name = fill(group.name, nrow(group_df))
    append!(all_results_df, group_df)
end

all_results_df_1 = DataFrame()
for group in analysis_groups
    # A. Filter match_ids for the current group
    filtered_df = target_matches
    for (col, val) in group.filters
        filtered_df = filter(row -> row[col] == val, filtered_df)
    end
    num_potential_matches = nrow(filtered_df)
    group_match_ids = Set(filtered_df.match_id)
    
    # B. Get the corresponding row indices from the cube (this is very fast)
    row_indices = [evaluation_cube_1.match_id_map[id] for id in evaluation_cube_1.match_ids if id in group_match_ids]
    
    if isempty(row_indices)
        println("   - Skipping group '$(group.name)', no matching matches found in the cube.")
        continue
    end

    # C. Slice the cube data
    cube_slice = (
        stakes = evaluation_cube_1.stakes[row_indices, :, :],
        outcomes = evaluation_cube_1.outcomes[row_indices, :],
        c_values = evaluation_cube_1.c_values,
        market_map = evaluation_cube_1.market_map
    )

    # D. Generate the performance object and convert to a DataFrame
    performance_struct = PerformanceAnalytics.calculate_market_performance(
        group.name,
        num_potential_matches,
        cube_slice
    )
    
    # E. Convert the nested struct to a tidy DataFrame and append
    group_df = PerformanceAnalytics.performance_to_dataframe(performance_struct)
    # Add a column to identify which group this result belongs to
    group_df.group_name = fill(group.name, nrow(group_df))
    append!(all_results_df_1, group_df)
end



####
c_values = 0.01:0.01:0.99
analysis_groups = [(name="Total Portfolio", filters=Dict())]

# extra
matches_prediction_1 =  BayesianFootball.predict_target_season_fix(target_matches, result.chains_sequence[8], mapping)
matches_kelly_1 = process_matches_kelly(matches_prediction_1, matches_odds, kelly_config)
#

# matches_kelly comes from your first model
evaluation_cube_model_A = build_evaluation_cube(matches_kelly, matches_odds, matches_results, c_values)

# matches_kelly_1 comes from your second model
evaluation_cube_model_B = build_evaluation_cube(matches_kelly_1, matches_odds, matches_results, c_values)

models_to_compare = Dict(
    "Baseline Model" => evaluation_cube_model_A,
    "model B" => evaluation_cube_model_B
    # You can easily add more models here:
    # "New Feature Model" => evaluation_cube_model_C,
)

# --- 3. RUN ANALYSIS FOR ALL MODELS ---
all_models_df = DataFrame()

for (model_name, eval_cube) in models_to_compare
    println("Running analysis for: $(model_name)...")
    
    # Call our reusable function for the current model
    model_results_df = run_analysis(eval_cube, analysis_groups, target_matches)
    
    # Add a column to identify which model these results belong to
    model_results_df.model_name = fill(model_name, nrow(model_results_df))
    
    # Append to the final master DataFrame
    append!(all_models_df, model_results_df)
end

comparison_total_profit = filter(row -> row.analysis_name == "Total Portfolio", all_models_df)


display(select(comparison_total_profit, :model_name, :total_profit, :max_roi, :max_drawdown))






# Example 2: See which model performed better on the 'ft_home' market
comparison_ft_home = filter(row -> occursin("ft_home", row.analysis_name), all_models_df)
display(select(comparison_ft_home, :model_name, :total_profit, :max_roi, :num_winning_bets, :num_losing_bets))



# Example 2: See which model performed better on the 'ft_home' market
comparison_ft_home = filter(row -> occursin("ft_over_05", row.analysis_name), all_models_df)
display(select(comparison_ft_home, :model_name, :total_staked, :total_profit, :max_roi, :max_sharpe, :max_ir, :num_winning_bets, :num_losing_bets))

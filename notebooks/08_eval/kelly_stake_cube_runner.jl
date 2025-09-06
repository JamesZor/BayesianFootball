# include("/home/james/bet_project/models_julia/notebooks/08_eval/test_eval_setup.jl")
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



#### dev bit 
include("/home/james/bet_project/models_julia/notebooks/08_eval/kelly_stake_cube_setup.jl")

c_values = 0.01:0.05:0.99

evaluation_cube = build_evaluation_cube(matches_kelly, matches_odds, matches_results, c_values)

evaluation_cube.stakes[:, 1, :]



## performance 
# Select the c-value you want to analyze
c_to_analyze = 0.5

# Calculate the per-match performance
performance_results = calculate_performance_by_match(evaluation_cube, c_to_analyze)
id = rand(keys(performance_results))
# Now you can inspect the results for a specific match
pr = performance_results[id]


## performance cube 

performance_cube_results = calculate_performance_cube_by_match(evaluation_cube)

id = rand(keys(matches_prediction))
one_match_performance = performance_cube_results[id]

# Get the profit vector for the FT Home Win market for this single match
# This vector shows the profit for c=0.1, c=0.2, ..., c=0.9
ft_home_profit_curve = one_match_performance.ft.home

# You can now see how the profit for that specific bet
# changes as your model's "confidence" (the c-value) increases.
println("Profit curve for FT Home Win:")
display(ft_home_profit_curve)
one_match_performance.ft.correct_score
one_match_performance.ht.correct_score

mr =matches_results[id]


## summary performance

# Calculate the performance summary
total_summary = summarize_performance_roi(evaluation_cube)

# --- Analysis Example ---
# Access the total ROI curve for the Full-Time Home Win market
ft_home_roi_curve = total_summary.ft.home
ht_home_roi_curve = total_summary.ht.home

println("Total ROI curve for FT Home Win across all matches:")
display(ft_home_roi_curve)

# Find the best c-value and the maximum ROI for this market
max_roi, best_idx = findmax(ft_home_roi_curve)
best_c_value = evaluation_cube.c_values[best_idx]

max_roi, best_idx = findmax(ht_home_roi_curve)
best_c_value = evaluation_cube.c_values[best_idx]

println("\nMaximum ROI for FT Home Win is $(round(max_roi*100, digits=2))% at c = $best_c_value")

# You can now plot this to visualize performance
using Plots

plot(evaluation_cube.c_values, ft_home_roi_curve,
     xlabel="c-value (Risk Appetite)",
     ylabel="Total ROI",
     title="Performance of FT Home Win Market",
     legend=false,
     markershape=:circle)

plot!(evaluation_cube.c_values, ht_home_roi_curve,
     xlabel="c-value (Risk Appetite)",
     ylabel="Total ROI",
     title="Performance of FT Home Win Market",
     legend=false,
     markershape=:circle)

### ft 1x2 line
ft_home_roi_curve = total_summary.ft.home
ft_away_roi_curve = total_summary.ft.away
ft_draw_roi_curve = total_summary.ft.draw

plot(evaluation_cube.c_values, ft_home_roi_curve,
     xlabel="c-value (Risk Appetite)",
     ylabel="Total ROI",
     title="Performance of FT Home Win Market",
     legend=true,
     label="home",
    )

plot!(evaluation_cube.c_values, ft_away_roi_curve,
     xlabel="c-value (Risk Appetite)",
     ylabel="Total ROI",
     title="Performance of FT Home Win Market",
     label="away",
     legend=true)

plot!(evaluation_cube.c_values, ft_draw_roi_curve,
     xlabel="c-value (Risk Appetite)",
     ylabel="Total ROI",
     title="Performance of FT Home Win Market",
     label="draw",
     legend=true)


### ht 1x2 line
ht_home_roi_curve = total_summary.ht.home
ht_away_roi_curve = total_summary.ht.away
ht_draw_roi_curve = total_summary.ht.draw

plot(evaluation_cube.c_values, ht_home_roi_curve,
     xlabel="c-value (Risk Appetite)",
     ylabel="Total ROI",
     title="Performance of ht Home Win Market",
     legend=true,
     label="home",
    )

plot!(evaluation_cube.c_values, ht_away_roi_curve,
     xlabel="c-value (Risk Appetite)",
     ylabel="Total ROI",
     title="Performance of ht Home Win Market",
     label="away",
     legend=true)

plot!(evaluation_cube.c_values, ht_draw_roi_curve,
     xlabel="c-value (Risk Appetite)",
     ylabel="Total ROI",
     title="Performance of ht Home Win Market",
     label="draw",
     legend=true)


### ft under over 
plot(evaluation_cube.c_values, total_summary.ft.under_05,
     xlabel="c-value (risk appetite)",
     ylabel="total roi",
     title="performance of ft under win market",
     legend=true,
     label="under 0.5",
     colour=:blue,
     ylims=(-0.2,0.1)
    )
plot!(evaluation_cube.c_values, total_summary.ft.over_05,
     xlabel="c-value (risk appetite)",
     ylabel="total roi",
     title="performance of ft under win market",
     legend=true,
     ls=:dash,
     label="over 0.5",
     colour=:blue
    )
# under over 15
plot!(evaluation_cube.c_values, total_summary.ft.under_15,
     xlabel="c-value (risk appetite)",
     ylabel="total roi",
     title="performance of ft under win market",
     legend=true,
     label="under 15",
     colour=:green
    )
plot!(evaluation_cube.c_values, total_summary.ft.over_15,
     xlabel="c-value (risk appetite)",
     ylabel="total roi",
     title="performance of ft under win market",
     legend=true,
     ls=:dash,
     label="over 15",
     colour=:green
    )

# under over 25
plot!(evaluation_cube.c_values, total_summary.ft.under_25,
     xlabel="c-value (risk appetite)",
     ylabel="total roi",
     title="performance of ft under win market",
     legend=true,
     label="under 25",
     colour=:red
    )
plot!(evaluation_cube.c_values, total_summary.ft.over_25,
     xlabel="c-value (risk appetite)",
     ylabel="total roi",
     title="performance of ft under win market",
     legend=true,
     ls=:dash,
     label="over 25",
     colour=:red
    )

# under over 35
plot!(evaluation_cube.c_values, total_summary.ft.under_35,
     xlabel="c-value (risk appetite)",
     ylabel="total roi",
     title="performance of ft under win market",
     legend=true,
     label="under 35",
     colour=:pink
    )
plot!(evaluation_cube.c_values, total_summary.ft.over_35,
     xlabel="c-value (risk appetite)",
     ylabel="total roi",
     title="performance of ft under win market",
     legend=true,
     ls=:dash,
     label="over 35",
     colour=:pink
    )



### ht under over 

plot(evaluation_cube.c_values, total_summary.ht.under_05,
     xlabel="c-value (risk appetite)",
     ylabel="total roi",
     title="performance of ht under win market",
     legend=true,
     label="under 0.5",
     colour=:blue,
     # ylims=(-0.2,0.1)
    )
plot!(evaluation_cube.c_values, total_summary.ht.over_05,
     xlabel="c-value (risk appetite)",
     ylabel="total roi",
     title="performance of ht under win market",
     legend=true,
     ls=:dash,
     label="over 0.5",
     colour=:blue
    )

# under over 15
plot!(evaluation_cube.c_values, total_summary.ht.under_15,
     xlabel="c-value (risk appetite)",
     ylabel="total roi",
     title="performance of ht under win market",
     legend=true,
     label="under 15",
     colour=:green
    )
plot!(evaluation_cube.c_values, total_summary.ht.over_15,
     xlabel="c-value (risk appetite)",
     ylabel="total roi",
     title="performance of ht under win market",
     legend=true,
     ls=:dash,
     label="over 15",
     colour=:green
    )

# under over 25
plot!(evaluation_cube.c_values, total_summary.ht.under_25,
     xlabel="c-value (risk appetite)",
     ylabel="total roi",
     title="performance of ht under win market",
     legend=true,
     label="under 25",
     colour=:red
    )
plot!(evaluation_cube.c_values, total_summary.ht.over_25,
     xlabel="c-value (risk appetite)",
     ylabel="total roi",
     title="performance of ht under win market",
     legend=true,
     ls=:dash,
     label="over 25",
     colour=:red
    )


# ft btts 
plot(evaluation_cube.c_values, total_summary.ft.btts_yes,
     xlabel="c-value (risk appetite)",
     ylabel="total roi",
     title="performance of ht under win market",
     legend=true,
     label="btts yes",
     # colour=:blue,
     # ylims=(-0.2,0.1)
    )
plot!(evaluation_cube.c_values, total_summary.ft.btts_no,
     xlabel="c-value (risk appetite)",
     ylabel="total roi",
     title="performance of ft btts win market",
     legend=true,
     # ls=:dash,
     label="btts no",
     # colour=:blue
    )


##### cs

function plot_correct_score_performance(cs_summary::Dict, c_values::AbstractVector)
    # Define colors for each outcome type
    colors = (home_win=:blue, draw=:green, away_win=:red, other=:purple)
    
    # Initialize the plot
    p = plot(
        title="Full-Time Correct Score Performance",
        xlabel="c-value (Risk Appetite)",
        ylabel="Total ROI",
        legend=:outertopright,
        # size=(800, 600) # Adjust size for better legend visibility
    )

    # Predefined line styles to cycle through for each color group
    linestyles = [:solid, :dash, :dot, :dashdot, :dashdotdot]
    style_idx = Dict(k => 1 for k in keys(colors))

    # Sort the keys for a consistent plotting order
    sorted_keys = sort(collect(keys(cs_summary)), by=string)

    for key in sorted_keys
        roi_curve = cs_summary[key]
        
        # Determine the outcome type and assign a color
        outcome_color_key, label = if key isa Tuple
            h, a = key
            if h > a
                (:home_win, "$h - $a")
            elseif h < a
                (:away_win, "$h - $a")
            else
                (:draw, "$h - $a")
            end
        else # It's a string key like "other_home_win"
            (:other, string(key))
        end
        
        outcome_color = colors[outcome_color_key]
        
        # Select a line style for the current group
        current_style = linestyles[style_idx[outcome_color_key]]
        
        # Plot the curve for this specific score
        plot!(p, c_values, roi_curve,
            label=label,
            color=outcome_color,
            linestyle=current_style
        )
        
        # Cycle the line style for the next plot of the same color
        style_idx[outcome_color_key] = mod1(style_idx[outcome_color_key] + 1, length(linestyles))
    end
    
    return p
end

ft_cs_summary = total_summary.ft.correct_score

correct_score_plot = plot_correct_score_performance(ft_cs_summary, c_values)
correct_score_plot = plot_correct_score_performance(total_summary.ht.correct_score, c_values)



#### other metrics 
# Calculate the Information Ratio summary
ir_summary = summarize_performance_ir(evaluation_cube)

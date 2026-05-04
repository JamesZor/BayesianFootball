# current_development/dev_xg_models/r05_goal_engine_model.jl

using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
pinthreads(:cores)



# Create an alias so we don't have to type the full path every time
const PreGame = BayesianFootball.Models.PreGame

println("Building model components...")

# ==========================================
# 1. INSTANTIATE THE COMPONENTS (The Lego Blocks)
# ==========================================
# Here you can easily swap GlobalDispersion for HomeAwayDispersion
# just by changing the struct you call!
inter_cfg = PreGame.GlobalInterception()
disp_cfg  = PreGame.HomeAwayDispersion() 
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
dyn_cfg   = PreGame.MultiScaleGRW()
kap_cfg   = PreGame.GlobalKappa()

# ==========================================
# 2. BUILD THE MASTER MODEL
# ==========================================
println("Assembling DynamicGoalsModel...")

model = PreGame.DynamicGoalsModel(
    interception_config  = inter_cfg,
    dynamics_config      = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg
)

println("Success! Model instantiated:")
display(model)



model_xg = PreGame.DynamicXGModel(
    interception_config  = inter_cfg,
    dynamics_config      = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
    kappa_config         = kap_cfg
)

ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())


cfgs = create_CVsplit_training_config(ds,get_target_seasons_string(ds.segment))


# TODO:
save_dir::String = "./data/dev_xg_models/"



# 1 . function to combine model + cfgs to get an experiment_confg etc, 
# Create tasks for both models
task_goals = build_experiment_task(ds, model, "Dev_Goals_v1", save_dir, cfgs)
task_xg    = build_experiment_task(ds, model_xg, "Dev_XG_v1", save_dir, cfgs)


resultsgoals = Experiments.run_experiment(task_goals.ds, task_goals.config)
resultsxg = Experiments.run_experiment(task_xg.ds, task_xg.config)

resultsxg = run_experiment_task.(task_goals)


todays_matches = fetch_todays_matches(ds)




target_ppd = compute_todays_matches_pdds(ds, resultsgoals, todays_matches)
target_ppd1 = compute_todays_matches_pdds(ds, resultsxg, todays_matches)



using DataFrames, Statistics
selections_to_calibrate = [:home, :away, :over_15, :over_25, :under_25, :over_35, :under_35]

# Create a helper function so you don't have to write the logic twice
function calculate_mean_odds(df, selections)
    # 1. Subset the dataframe to only keep the target selections
    df_filtered = subset(df, :selection => ByRow(in(selections)))
    
    # 2. Select the columns we want to keep, and compute the mean odds row-by-row
    df_odds = select(df_filtered, 
        :match_id, 
        :selection, 
        :distribution => ByRow(d -> mean(1 ./ d)) => :mean_odds
    )
    
    return df_odds
end

# Apply the function to both of your model outputs
odds_goals = calculate_mean_odds(target_ppd.df, selections_to_calibrate)
odds_xg    = calculate_mean_odds(target_ppd1.df, selections_to_calibrate)

# Join them together for a direct side-by-side comparison
comparison_df = innerjoin(
    rename(odds_goals, :mean_odds => :odds_goals_model),
    rename(odds_xg, :mean_odds => :odds_xg_model),
    on = [:match_id, :selection]
)

# Calculate the difference between the two models
comparison_df.odds_diff = comparison_df.odds_goals_model .- comparison_df.odds_xg_model

# Sort for readability
sort!(comparison_df, [:match_id, :selection])


# 1. Join the team names onto your comparison dataframe
comparison_df = leftjoin(
    comparison_df, 
    select(todays_matches, :match_id, :home_team, :away_team), 
    on = :match_id
)

# 2. Reorder the columns so the team names are at the beginning
select!(comparison_df, 
    :match_id, 
    :home_team, 
    :away_team, 
    :selection, 
    :odds_goals_model, 
    :odds_xg_model, 
    :odds_diff
)

# notes on the features - used to check we no getting empty features etc, 
boundaries_with_meta = BayesianFootball.Data.create_id_boundaries(ds, cfgs.cv_cfg)
feature_collection = BayesianFootball.Features.create_features(boundaries_with_meta, ds, model)






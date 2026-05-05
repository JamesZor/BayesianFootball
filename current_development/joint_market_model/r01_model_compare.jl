# current_development/joint_market_model/r01_model_compare.jl
include("./l00_inverse_problem.jl")

using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
pinthreads(:cores)


# for running the models
const PreGame = BayesianFootball.Models.PreGame


# ==========================================
# 1. LOAD THE ROBUST COMPONENTS
# ==========================================
inter_cfg = PreGame.GlobalInterception()
# Note: Using HomeAwayDispersion based on your previous grid (unless you built a custom TeamDispersion!)
disp_cfg  = PreGame.HomeAwayDispersion() 
# "ha global"
ha_cfg    = PreGame.GlobalHomeAdvantage() 

dyn_cfg   = PreGame.MultiScaleGRW(
    z₀ = Normal(0, 1), 
      zₛ = Normal(0,1),   # The fat-tailed robust winner
    zₖ = Normal(0, 1)
)

model = PreGame.DynamicGoalsModel(
    interception_config  = inter_cfg,
    dynamics_config      = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
)

save_dir::String = "./data/dev_inverse_model/"



cfgs = create_CVsplit_training_config(ds, get_target_seasons_string(ds.segment))
task_base    = build_experiment_task(ds, model,    "basic",    save_dir, cfgs)



# res_base    = Experiments.run_experiment(task_base.ds,    task_base.config)

Experiments.save_experiment(res_base)


latents = Experiments.extract_oos_predictions(ds, res_base)


matches = subset(ds.matches, :season => ByRow(isequal("2025")))



using DataFrames
using Statistics
using StatsPlots # Used for the EDA visualisations

# 1. Isolate the 2025 Odds Data
matches_2025 = subset(ds.matches, :season => ByRow(isequal("2025")))
odds_2025 = subset(ds.odds, :match_id => ByRow(in(matches_2025.match_id)))

# 2. Extract Market Parameters for all 2025 Matches
# Group the odds dataframe by match_id
grouped_odds = groupby(odds_2025, :match_id)

# Apply your fit_market_implied_parameters function to each group 
# and collect the NamedTuples directly into a new DataFrame
println("Fitting market parameters for $(length(grouped_odds)) matches...")
market_df = DataFrame([fit_market_implied_parameters(sub_df) for sub_df in grouped_odds])

# 3. Merge Market Data with Model Latents
analysis_df = innerjoin(market_df, latents.df, on=:match_id)

# 4. Calculate the Log Differences
# We calculate the mean of your posterior arrays first, then calculate the log difference
analysis_df = transform(analysis_df,
    :λ_h => ByRow(mean) => :model_λ_h_mean,
    :λ_a => ByRow(mean) => :model_λ_a_mean
)

analysis_df = transform(analysis_df,
    [:λ_home, :model_λ_h_mean] => ByRow((mkt, mod) -> log(mkt) - log(mod)) => :log_diff_h,
    [:λ_away, :model_λ_a_mean] => ByRow((mkt, mod) -> log(mkt) - log(mod)) => :log_diff_a
)

# Optional: View a summary of the residuals
println("\n--- Summary of Home Log Differences ---")
describe(analysis_df[:, [:log_diff_h]])
println("\n--- Summary of Away Log Differences ---")
describe(analysis_df[:, [:log_diff_a]])

# 5. Exploratory Data Analysis (EDA) Visualisation
# Plot histograms with overlaid Kernel Density Estimates (KDE) to check for Normality

p1 = histogram(analysis_df.log_diff_h, 
               normalize=:pdf, 
               bins=20, 
               title="Home λ Log Difference\n(Market vs Model)", 
               label="Empirical", 
               color=:steelblue, 
               alpha=0.7)
density!(p1, analysis_df.log_diff_h, label="KDE", color=:red, linewidth=2)

p2 = histogram(analysis_df.log_diff_a, 
               normalize=:pdf, 
               bins=20, 
               title="Away λ Log Difference\n(Market vs Model)", 
               label="Empirical", 
               color=:seagreen, 
               alpha=0.7)
density!(p2, analysis_df.log_diff_a, label="KDE", color=:red, linewidth=2)

# Display side-by-side
display(plot(p1, p2, layout=(1,2), size=(900, 400), margin=5Plots.mm))

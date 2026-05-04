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


##### ------
# current_development/dev_xg_models/r05_goal_engine_model.jl

using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
using Distributions
pinthreads(:cores)



# Create an alias so we don't have to type the full path every time
const PreGame = BayesianFootball.Models.PreGame
# ==========================================
# 1. LOAD THE ROBUST COMPONENTS
# ==========================================
inter_cfg = PreGame.GlobalInterception()

# Note: Using HomeAwayDispersion based on your previous grid (unless you built a custom TeamDispersion!)
disp_cfg  = PreGame.HomeAwayDispersion() 

# "ha global"
ha_cfg    = PreGame.GlobalHomeAdvantage() 

# "kappa team"
kap_cfg   = PreGame.HierarchicalTeamKappa() 

# "tdist and normal for grw" - Robust Season Drift!
dyn_cfg   = PreGame.MultiScaleGRW(
    z₀ = Normal(0, 1), 
    zₛ = TDist(4.0),   # The fat-tailed robust winner
    zₖ = Normal(0, 1)
)


dyn_cfg1   = PreGame.MultiScaleGRW(
    z₀ = Normal(0, 1), 
      zₛ = Normal(0,1),   # The fat-tailed robust winner
    zₖ = Normal(0, 1)
)



production_model = PreGame.DynamicXGModel(
    interception_config  = inter_cfg,
    dynamics_config      = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
    kappa_config         = kap_cfg
)

production_model1 = PreGame.DynamicXGModel(
    interception_config  = inter_cfg,
    dynamics_config      = dyn_cfg1,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
    kappa_config         = kap_cfg
)



ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())

cfgs = create_CVsplit_training_config(ds,get_target_seasons_string(ds.segment))




task_1 = build_experiment_task(ds, production_model, "zdist", save_dir, cfgs)
task_2    = build_experiment_task(ds, production_model1, "normal", save_dir, cfgs)



results_tdist = Experiments.run_experiment(task_1.ds, task_1.config)
results_normal = Experiments.run_experiment(task_2.ds, task_2.config)


c_t = results_tdist.training_results[1][1]
c_n = results_normal.training_results[1][1]


describe(c_t)
describe(c_n)



target_ppd_t= compute_todays_matches_pdds(ds, results_tdist, todays_matches)
target_ppd_n= compute_todays_matches_pdds(ds, results_normal, todays_matches)

selections_to_calibrate = [:home, :away, :over_15, :over_25, :under_25, :over_35, :under_35]



odds_t = calculate_mean_odds(target_ppd.df, selections_to_calibrate)
odds_n    = calculate_mean_odds(target_ppd1.df, selections_to_calibrate)

comparison_df = innerjoin(
    rename(odds_t, :mean_odds => :odds_tdist_model),
    rename(odds_n, :mean_odds => :odds_normal_model),
    on = [:match_id, :selection]
)


comparison_df.odds_diff = comparison_df.odds_tdist_model .- comparison_df.odds_normal_model;

comparison_df = leftjoin(
    comparison_df, 
    select(todays_matches, :match_id, :home_team, :away_team), 
    on = :match_id
)




# -------

function create_CVsplit_training_config(ds::Data.DataStore, target_seasons::Vector{<:String})

    # 1. Define the shared parts (CV and Training)
    cv_config = Data.GroupedCVConfig(
        tournament_groups = [Data.tournament_ids(ds.segment)],
        target_seasons = target_seasons,
        history_seasons = 1,
        dynamics_col = :match_week,
        warmup_period = 13,
        stop_early = false
    )

    sampler_conf = Samplers.NUTSConfig(
    1000, # n steps
    8,    # n chains
    300,  # warm up steps
    0.65, # acceptance rate
    10,   # Max depth
    Samplers.UniformInit(-1, 1), # init step up 
    :perchain # show progress bar
    )

    train_cfg = BayesianFootball.Training.Independent(
        parallel=true, max_concurrent_splits=4
    )
    training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)


    return (; cv_cfg=cv_config, training_cfg=training_config)

end



using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
using Distributions
pinthreads(:cores)

const PreGame = BayesianFootball.Models.PreGame

save_dir::String = "./data/dev_xg_models/"
# ==========================================
# 1. LOAD STATIC COMPONENTS
# ==========================================
inter_cfg = PreGame.GlobalInterception()
disp_cfg  = PreGame.HomeAwayDispersion() 
ha_cfg    = PreGame.GlobalHomeAdvantage() 
kap_cfg   = PreGame.HierarchicalTeamKappa() 

# ==========================================
# 2. DEFINE THE "GRID" OF DYNAMIC PRIORS
# ==========================================
# ==========================================
# TEST 1: The Baseline (Tight Leash)
# ==========================================
dyn_base = PreGame.MultiScaleGRW(
    z₀ = Normal(0, 1), zₛ = TDist(4.0), zₖ = Normal(0, 1),
    # Original macro priors
    α_σ₀ = Gamma(2, 0.06), α_σₛ = Gamma(2, 0.03),
    β_σ₀ = Gamma(2, 0.10), β_σₛ = Gamma(2, 0.055),
    # Original tight micro priors (Too slow for Pinnacle?)
    α_σₖ = Gamma(2, 0.015), 
    β_σₖ = Gamma(2, 0.012)
)

# ==========================================
# TEST 2: The "Agile" Model (2x Weekly Variance)
# ==========================================
dyn_agile = PreGame.MultiScaleGRW(
    z₀ = Normal(0, 1), zₛ = TDist(4.0), zₖ = Normal(0, 1),
    # Keep macro priors exactly the same
    α_σ₀ = Gamma(2, 0.06), α_σₛ = Gamma(2, 0.03),
    β_σ₀ = Gamma(2, 0.10), β_σₛ = Gamma(2, 0.055),
    # DOUBLED week-to-week variance to catch short-term form
    α_σₖ = Gamma(2, 0.030), 
    β_σₖ = Gamma(2, 0.024)
)

# ==========================================
# TEST 3: The "Highly Reactive" Model (3x Weekly Variance)
# ==========================================
dyn_reactive = PreGame.MultiScaleGRW(
    z₀ = Normal(0, 1), zₛ = TDist(4.0), zₖ = Normal(0, 1),
    # Keep macro priors exactly the same
    α_σ₀ = Gamma(2, 0.06), α_σₛ = Gamma(2, 0.03),
    β_σ₀ = Gamma(2, 0.10), β_σₛ = Gamma(2, 0.055),
    # TRIPLED week-to-week variance
    α_σₖ = Gamma(2, 0.045), 
    β_σₖ = Gamma(2, 0.036)
)

# Build the 3 Models
model_base    = PreGame.DynamicXGModel(
  interception_config=inter_cfg,
  dynamics_config=dyn_base,
  dispersion_config=disp_cfg,
  homeadvantage_config=ha_cfg,
  kappa_config=kap_cfg
)
model_agile = PreGame.DynamicXGModel(
  interception_config=inter_cfg,
  dynamics_config=dyn_agile,
  dispersion_config=disp_cfg,
  homeadvantage_config=ha_cfg,
  kappa_config=kap_cfg
)

model_reactive   = PreGame.DynamicXGModel(
  interception_config=inter_cfg,
  dynamics_config=dyn_reactive,
  dispersion_config=disp_cfg,
  homeadvantage_config=ha_cfg,
  kappa_config=kap_cfg
)

# ==========================================
# 3. PREP DATA & TASKS
# ==========================================
ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())
cfgs = create_CVsplit_training_config(ds, get_target_seasons_string(ds.segment))

# Assumes save_dir is defined in your environment
task_base    = build_experiment_task(ds, model_base,    "grid_base",    save_dir, cfgs)
task_aglie = build_experiment_task(ds, model_agile, "grid_aglie", save_dir, cfgs)
task_reactive   = build_experiment_task(ds, model_reactive,   "grid_reactive",   save_dir, cfgs)

# ==========================================
# 4. RUN INFERENCE
# ==========================================
res_base    = Experiments.run_experiment(task_base.ds,    task_base.config)
res_aglie = Experiments.run_experiment(task_aglie.ds, task_aglie.config)
res_reactive   = Experiments.run_experiment(task_reactive.ds,   task_reactive.config)

# ==========================================
# 5. GENERATE ODDS & COMPARE
# ==========================================
# Run PPDs

todays_matches = fetch_todays_matches(ds)
ppd_base    = compute_todays_matches_pdds(ds, res_base,    todays_matches)
ppd_aglie = compute_todays_matches_pdds(ds, res_aglie, todays_matches)
ppd_reactive   = compute_todays_matches_pdds(ds, res_reactive,   todays_matches)

selections_to_calibrate = [:home, :away, :over_15, :over_25, :under_25, :over_35, :under_35]

# Calculate mean odds for all three
odds_base    = calculate_mean_odds(ppd_base.df,    selections_to_calibrate)
odds_aglie = calculate_mean_odds(ppd_aglie.df, selections_to_calibrate)
odds_reactive   = calculate_mean_odds(ppd_reactive.df,   selections_to_calibrate)

# Chain joins to build the master comparison table
comparison_df = innerjoin(
    rename(odds_base,    :mean_odds => :odds_base),
    rename(odds_aglie, :mean_odds => :odds_aglie),
    on = [:match_id, :selection]
)

comparison_df = innerjoin(
    comparison_df,
    rename(odds_reactive,   :mean_odds => :odds_reactive),
    on = [:match_id, :selection]
)

# Add Team Names
comparison_df = leftjoin(
    comparison_df, 
    select(todays_matches, :match_id, :home_team, :away_team), 
    on = :match_id,
  makeunique=true
)

# Reorder columns nicely
select!(comparison_df, 
    :match_id, :home_team, :away_team, :selection, 
    :odds_base, :odds_aglie, :odds_reactive
)

sort!(comparison_df, [:match_id, :selection])




# ==========================================
# 1. THE KAPPA GRID
# ==========================================
# Grid 1: The Anchor (Your current baseline)
kap_base = PreGame.HierarchicalTeamKappa(
    κ_base = Normal(1.0, 0.2),
    σ_κ    = truncated(Normal(0, 0.1), lower=0.0) 
)

# Grid 2: Moderate Freedom (2.5x wider variance)
kap_loose = PreGame.HierarchicalTeamKappa(
    κ_base = Normal(1.0, 0.2),
    σ_κ    = truncated(Normal(0, 0.25), lower=0.0) # Teams can easily reach 1.25 to 1.50 multiplier
)

# Grid 3: The "Unleashed" Kappa (5x wider variance)
kap_unleashed = PreGame.HierarchicalTeamKappa(
    κ_base = Normal(1.0, 0.2),
    σ_κ    = truncated(Normal(0, 0.5), lower=0.0) # Sampler is free to put elite teams wherever the data dictates
)

# ==========================================
# 2. BUILD THE MODELS
# ==========================================
# Note: Revert to your standard Weekly Normal GRW (dyn_base) from the previous test, 
# since we proved the Random Walk wasn't the issue!

model_base      = PreGame.DynamicXGModel(
  interception_config=inter_cfg,
  dynamics_config=dyn_base,
  dispersion_config=disp_cfg,
  homeadvantage_config=ha_cfg,
  kappa_config=kap_base
)

model_loose     = PreGame.DynamicXGModel(
  interception_config=inter_cfg,
  dynamics_config=dyn_base,
  dispersion_config=disp_cfg,
  homeadvantage_config=ha_cfg,
  kappa_config=kap_loose
)

model_unleashed = PreGame.DynamicXGModel(
  interception_config=inter_cfg,
  dynamics_config=dyn_base,
  dispersion_config=disp_cfg,
  homeadvantage_config=ha_cfg,
  kappa_config=kap_unleashed
)

# ==========================================
# 3. RUN INFERENCE & COMPARE (Same as before)
# ==========================================
task_base      = build_experiment_task(ds, model_base,      "kappa_base",      save_dir, cfgs)
task_loose     = build_experiment_task(ds, model_loose,     "kappa_loose",     save_dir, cfgs)
task_unleashed = build_experiment_task(ds, model_unleashed, "kappa_unleashed", save_dir, cfgs)

res_base      = Experiments.run_experiment(task_base.ds,      task_base.config)
res_loose     = Experiments.run_experiment(task_loose.ds,     task_loose.config)
res_unleashed = Experiments.run_experiment(task_unleashed.ds, task_unleashed.config)

todays_matches = fetch_todays_matches(ds)

kappa_comparison = generate_odds_comparison(ds, [
    "base"      => res_base,
    "loose"     => res_loose,
    "unleashed" => res_unleashed
], todays_matches)


display(kappa_comparison)




#=
julia> display(kappa_comparison)
35×7 DataFrame
 Row │ match_id  home_team        away_team             selection  odds_base  odds_loose  odds_unleashed 
     │ Int64     String?          String?               Symbol     Float64    Float64     Float64        
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ 15238077  bohemian         shelbourne            away         3.44557     3.43645         3.43565
   2 │ 15238077  bohemian         shelbourne            home         2.37266     2.38298         2.38048
   3 │ 15238077  bohemian         shelbourne            over_15      1.50773     1.51033         1.50504
   4 │ 15238077  bohemian         shelbourne            over_25      2.49135     2.49959         2.48326
   5 │ 15238077  bohemian         shelbourne            over_35      4.96357     4.9895          4.94014
   6 │ 15238077  bohemian         shelbourne            under_25     1.70549     1.70323         1.71061
   7 │ 15238077  bohemian         shelbourne            under_35     1.26981     1.26886         1.27204
   8 │ 15238078  derry-city       galway-united         away         4.35281     4.43001         4.44572
   9 │ 15238078  derry-city       galway-united         home         1.98667     1.97219         1.96913
  10 │ 15238078  derry-city       galway-united         over_15      1.43348     1.43345         1.432
  11 │ 15238078  derry-city       galway-united         over_25      2.26134     2.26103         2.25688
  12 │ 15238078  derry-city       galway-united         over_35      4.26955     4.26796         4.25659
  13 │ 15238078  derry-city       galway-united         under_25     1.83354     1.83433         1.83749
  14 │ 15238078  derry-city       galway-united         under_35     1.32606     1.32649         1.32781
  15 │ 15238079  shamrock-rovers  drogheda-united       away         5.74752     5.80637         5.81093
  16 │ 15238079  shamrock-rovers  drogheda-united       home         1.81604     1.80984         1.80768
  17 │ 15238079  shamrock-rovers  drogheda-united       over_15      1.5957      1.5931          1.58904
  18 │ 15238079  shamrock-rovers  drogheda-united       over_25      2.76854     2.76022         2.74767
  19 │ 15238079  shamrock-rovers  drogheda-united       over_35      5.82581     5.79989         5.76188
  20 │ 15238079  shamrock-rovers  drogheda-united       under_25     1.59508     1.59934         1.60511
  21 │ 15238079  shamrock-rovers  drogheda-united       under_35     1.2222      1.22411         1.2266
  22 │ 15238080  sligo-rovers     st-patricks-athletic  away         2.38724     2.39473         2.39852
  23 │ 15238080  sligo-rovers     st-patricks-athletic  home         3.37126     3.37079         3.3672
  24 │ 15238080  sligo-rovers     st-patricks-athletic  over_15      1.47786     1.48165         1.47927
  25 │ 15238080  sligo-rovers     st-patricks-athletic  over_25      2.39686     2.40914         2.40192
  26 │ 15238080  sligo-rovers     st-patricks-athletic  over_35      4.67066     4.71032         4.68884
  27 │ 15238080  sligo-rovers     st-patricks-athletic  under_25     1.75013     1.74705         1.7505
  28 │ 15238080  sligo-rovers     st-patricks-athletic  under_35     1.28952     1.28823         1.2897
  29 │ 15238081  waterford-fc     dundalk-fc            away         2.31886     2.3164          2.31942
  30 │ 15238081  waterford-fc     dundalk-fc            home         3.13236     3.1387          3.1339
  31 │ 15238081  waterford-fc     dundalk-fc            over_15      1.28314     1.28345         1.28276
  32 │ 15238081  waterford-fc     dundalk-fc            over_25      1.81519     1.81606         1.81421
  33 │ 15238081  waterford-fc     dundalk-fc            over_35      3.00813     3.0106          3.00592
  34 │ 15238081  waterford-fc     dundalk-fc            under_25     2.25798     2.25912         2.26159
  35 │ 15238081  waterford-fc     dundalk-fc            under_35     1.51282     1.51334         1.51432
=#

# ==========================================
# 1. SETUP THE HOME ADVANTAGE CONFIGS
# ==========================================
# Grid 1: The Baseline (Global Home Advantage)
ha_global = PreGame.GlobalHomeAdvantage()

# Grid 2: The Fortress Model (Hierarchical Team Home Advantage)
ha_hierarchical = PreGame.HierarchicalTeamHomeAdvantage(
    γ_base = Normal(0.2, 0.2),
    σ_γ    = truncated(Normal(0, 0.1), lower=0.0) 
)

# Optional Grid 3: The "Extreme Fortress" Model (Looser variance)
ha_extreme = PreGame.HierarchicalTeamHomeAdvantage(
    γ_base = Normal(0.2, 0.2),
    σ_γ    = truncated(Normal(0, 0.25), lower=0.0) # Allows massive home advantages
)

# ==========================================
# 2. BUILD THE MODELS
# ==========================================
# Note: Keep Kappa loose and Dynamics weekly based on our previous findings!
model_global       = PreGame.DynamicXGModel(
  interception_config=inter_cfg,
  dynamics_config=dyn_base,
  dispersion_config=disp_cfg,
  homeadvantage_config=ha_global,
  kappa_config=kap_loose
)

model_hierarchical = PreGame.DynamicXGModel(
  interception_config=inter_cfg,
  dynamics_config=dyn_base,
  dispersion_config=disp_cfg,
  homeadvantage_config=ha_hierarchical,
  kappa_config=kap_loose
)
model_extreme      = PreGame.DynamicXGModel(
  interception_config=inter_cfg,
  dynamics_config=dyn_base,
  dispersion_config=disp_cfg,
  homeadvantage_config=ha_extreme,
  kappa_config=kap_loose
)

# ==========================================
# 3. RUN EXPERIMENTS
# ==========================================
task_global       = build_experiment_task(ds, model_global,       "ha_global",       save_dir, cfgs)
task_hierarchical = build_experiment_task(ds, model_hierarchical, "ha_hierarchical", save_dir, cfgs)
task_extreme      = build_experiment_task(ds, model_extreme,      "ha_extreme",      save_dir, cfgs)

res_global       = Experiments.run_experiment(task_global.ds,       task_global.config)
res_hierarchical = Experiments.run_experiment(task_hierarchical.ds, task_hierarchical.config)
res_extreme      = Experiments.run_experiment(task_extreme.ds,      task_extreme.config)

# ==========================================
# 4. COMPARE
# ==========================================
ha_comparison = generate_odds_comparison(ds, [
    "global"       => res_global,
    "hierarchical" => res_hierarchical,
    "extreme"      => res_extreme
], todays_matches)

display(ha_comparison)


#=
julia> ha_comparison = generate_odds_comparison(ds, [
           "global"       => res_global,
           "hierarchical" => res_hierarchical,
           "extreme"      => res_extreme
       ], todays_matches)
Processing model: global...
Running Inference on 5 matches...
Processing model: hierarchical...
Running Inference on 5 matches...
Processing model: extreme...
Running Inference on 5 matches...
Comparison generated successfully!
35×7 DataFrame
 Row │ match_id  home_team        away_team             selection  odds_global  odds_hierarchical  odds_extreme 
     │ Int64     String?          String?               Symbol     Float64      Float64            Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ 15238077  bohemian         shelbourne            away           3.43176            3.4768        3.49966
   2 │ 15238077  bohemian         shelbourne            draw           3.63122            3.666         3.67474
   3 │ 15238077  bohemian         shelbourne            home           2.38213            2.35112       2.33514
   4 │ 15238077  bohemian         shelbourne            over_15        1.50576            1.4923        1.48992
   5 │ 15238077  bohemian         shelbourne            over_25        2.48517            2.4438        2.43662
   6 │ 15238077  bohemian         shelbourne            over_35        4.945              4.82097       4.80026
   7 │ 15238077  bohemian         shelbourne            under_25       1.70987            1.73218       1.73797
   8 │ 15238078  derry-city       galway-united         away           4.4188             4.52913       4.59658
   9 │ 15238078  derry-city       galway-united         draw           3.95005            4.00185       4.02071
  10 │ 15238078  derry-city       galway-united         home           1.97543            1.94614       1.92961
  11 │ 15238078  derry-city       galway-united         over_15        1.43363            1.42377       1.42222
  12 │ 15238078  derry-city       galway-united         over_25        2.26149            2.23226       2.22765
  13 │ 15238078  derry-city       galway-united         over_35        4.26909            4.1863        4.1734
  14 │ 15238078  derry-city       galway-united         under_25       1.83401            1.85929       1.86445
  15 │ 15238079  shamrock-rovers  drogheda-united       away           5.81298            5.78946       5.82045
  16 │ 15238079  shamrock-rovers  drogheda-united       draw           3.82338            3.8395        3.86102
  17 │ 15238079  shamrock-rovers  drogheda-united       home           1.80794            1.80824       1.80168
  18 │ 15238079  shamrock-rovers  drogheda-united       over_15        1.58824            1.58065       1.57414
  19 │ 15238079  shamrock-rovers  drogheda-united       over_25        2.74473            2.72129       2.70044
  20 │ 15238079  shamrock-rovers  drogheda-united       over_35        5.75133            5.68038       5.61446
  21 │ 15238079  shamrock-rovers  drogheda-united       under_25       1.60585            1.61644       1.62498
  22 │ 15238080  sligo-rovers     st-patricks-athletic  away           2.39021            2.3706        2.36961
  23 │ 15238080  sligo-rovers     st-patricks-athletic  draw           3.67532            3.64695       3.63042
  24 │ 15238080  sligo-rovers     st-patricks-athletic  home           3.37654            3.45202       3.47358
  25 │ 15238080  sligo-rovers     st-patricks-athletic  over_15        1.48048            1.49951       1.50933
  26 │ 15238080  sligo-rovers     st-patricks-athletic  over_25        2.40524            2.46536       2.49601
  27 │ 15238080  sligo-rovers     st-patricks-athletic  over_35        4.69765            4.88414       4.97809
  28 │ 15238080  sligo-rovers     st-patricks-athletic  under_25       1.74834            1.72157       1.70698
  29 │ 15238081  waterford-fc     dundalk-fc            away           2.32233            2.31234       2.3094
  30 │ 15238081  waterford-fc     dundalk-fc            draw           4.1528             4.15135       4.14545
  31 │ 15238081  waterford-fc     dundalk-fc            home           3.12915            3.15639       3.16746
  32 │ 15238081  waterford-fc     dundalk-fc            over_15        1.28352            1.28524       1.28763
  33 │ 15238081  waterford-fc     dundalk-fc            over_25        1.81622            1.82136       1.82823
  34 │ 15238081  waterford-fc     dundalk-fc            over_35        3.01085            3.02528       3.04371
  35 │ 15238081  waterford-fc     dundalk-fc            under_25       2.25884            2.25296       2.2436
=#
# ==========================================
# 1. SETUP THE CONFIGS
# ==========================================
# Grid 1: The Baseline (Your current tight dispersion)
inter_base = PreGame.GlobalInterception(μ = Normal(0.2, 0.1))
disp_base  = PreGame.HomeAwayDispersion(log_r = Normal(3.1, 0.4), δ_r_home = Normal(0.0, 0.5))

# Grid 2: Fatter Tails (More variance, same xG)
inter_base = PreGame.GlobalInterception(μ = Normal(0.2, 0.1))
disp_fat   = PreGame.HomeAwayDispersion(
    log_r = Normal(1.6, 0.4), # exp(1.6) ≈ 5.0 (Significant overdispersion)
    δ_r_home = Normal(0.0, 0.5)
)

# Grid 3: Higher Baseline Scoring (More xG overall)
inter_high = PreGame.GlobalInterception(μ = Normal(0.35, 0.1)) # Higher base rate
disp_base  = PreGame.HomeAwayDispersion(log_r = Normal(3.1, 0.4), δ_r_home = Normal(0.0, 0.5))

# Grid 4: The Combo (Higher Baseline + Fatter Tails)
inter_high = PreGame.GlobalInterception(μ = Normal(0.35, 0.1))
disp_fat   = PreGame.HomeAwayDispersion(
    log_r = Normal(1.6, 0.4), 
    δ_r_home = Normal(0.0, 0.5)
)

# ==========================================
# 2. BUILD THE MODELS
# ==========================================
# Using the Weekly Normal GRW and Loose Kappa from previous findings
model_base  = PreGame.DynamicXGModel(
  interception_config=inter_base,
  dynamics_config=dyn_base,
  dispersion_config=disp_base,
  homeadvantage_config=ha_hierarchical,
  kappa_config=kap_loose
)

model_fat   = PreGame.DynamicXGModel(
  interception_config=inter_base,
  dynamics_config=dyn_base,
  dispersion_config=disp_fat,
  homeadvantage_config=ha_hierarchical,
  kappa_config=kap_loose
)

model_high  = PreGame.DynamicXGModel(
  interception_config=inter_high,
  dynamics_config=dyn_base,
  dispersion_config=disp_base,
  homeadvantage_config=ha_hierarchical,
  kappa_config=kap_loose
)


model_combo = PreGame.DynamicXGModel(
  interception_config=inter_high,
  dynamics_config=dyn_base,
  dispersion_config=disp_fat,
  homeadvantage_config=ha_hierarchical,
  kappa_config=kap_loose
)

# ==========================================
# 3. RUN EXPERIMENTS & COMPARE
# ==========================================
task_base  = build_experiment_task(ds, model_base,  "ou_base",  save_dir, cfgs)
task_fat   = build_experiment_task(ds, model_fat,   "ou_fat",   save_dir, cfgs)
task_high  = build_experiment_task(ds, model_high,  "ou_high",  save_dir, cfgs)
task_combo = build_experiment_task(ds, model_combo, "ou_combo", save_dir, cfgs)

res_base  = Experiments.run_experiment(task_base.ds,  task_base.config)
res_fat   = Experiments.run_experiment(task_fat.ds,   task_fat.config)
res_high  = Experiments.run_experiment(task_high.ds,  task_high.config)
res_combo = Experiments.run_experiment(task_combo.ds, task_combo.config)

# Run the comparison!
ou_comparison = generate_odds_comparison(ds, [
    "base"  => res_base,
    "fat"   => res_fat,
    "high"  => res_high,
    "combo" => res_combo
], todays_matches)

display(ou_comparison)

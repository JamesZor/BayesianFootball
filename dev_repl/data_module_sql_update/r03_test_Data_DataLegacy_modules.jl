using Revise
using BayesianFootball

using DataFrames
using ThreadPinning
pinthreads(:cores)

# 
include("./l03_test_Data_DataLegacy_modules.jl")
#

# Load from sql
# server is busy... hence blacked out

# working local
# ds = get_datastore_local_ip()


ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.ScottishLower())
# ds_legacy = get_datastore_legacy() - issues witht he load extra data..
ds_legacy = BayesianFootball.DataLegacy.load_default_datastore()



# ------ 2 . Running the experiments  --------
save_dir::String = "./data/exp/data_module_test"

es_new = DSExperimentSettings(ds, "new", save_dir)
es_legacy = DSExperimentSettings(ds_legacy, "legacy", save_dir)


all_tasks = create_list_experiment_tasks([es_new, es_legacy])

results = run_experiment_task.(all_tasks)


# ------ 3 . Load the experiments  --------

saved_folders = Experiments.list_experiments(save_dir; data_dir="")

loaded_results = loaded_experiment_files(saved_folders);

# ------ 4. Simple Backtesting -----




# ------------------------------------------
# TODO:
# This needs to be moved to an eda / statistics sections. 
# ---- Task 2 -- explore dataset 

ds = get_datastore_local_ip()
ds = get_datastore_local_ip(segment=Data.Ireland())


for g in groupby(ds.matches, :season)
  println("$(first(g.season)) $(first(g.season_id)) - $(nrow(g))")
end

#=
2021 35611 - 180
2022 40193 - 180
2023 47946 - 180
2024 57285 - 180
2025 69981 - 180
2026 87682 - 53

=#


stat = subset(ds.statistics,
              :season_id => ByRow(isequal(69981)), 
              :period => ByRow(isequal("ALL"))
              )

stat

describe(stat)


names(stat)
describe(stat[:, [:match_id, :period, :expectedGoals_home, :expectedGoals_away]])


ds.statistics[:, [:match_id, :period, :expectedGoals_home, :expectedGoals_away]]
describe(ds.statistics[:, [:match_id, :expectedGoals_home, :expectedGoals_away]])



using StatsPlots

# This automatically handles the columns by name from the 'stat' DataFrame
@df stat histogram([:expectedGoals_home :expectedGoals_away], 
    label=["Home xG" "Away xG"], 
    alpha=0.2, 
    bins=30, 
    title="xG Distribution: Home vs Away",
    xlabel="Expected Goals",
    ylabel="Frequency")

#=
To compare which distribution (Gamma vs. Log-Normal) fits the data better, the standard approach is to compare their Akaike Information Criterion (AIC) or Log-Likelihoods.
Then, you can use a Kolmogorov-Smirnov (KS) test to formally check if the data significantly deviates from the fitted distributions
=#

using Distributions
using HypothesisTests
using DataFrames

# 1. Clean the data (remove the 8 missing values)
# collect(skipmissing()) ensures we pass a clean Array of Float64 to the fit functions
xg_home = collect(skipmissing(stat.expectedGoals_home));
xg_away = collect(skipmissing(stat.expectedGoals_away));

# ==========================================
# 2. Fit the Distributions
# ==========================================

# Fit Gamma
gamma_home = fit(Gamma, xg_home)
gamma_away = fit(Gamma, xg_away)

# Fit Log-Normal
lognorm_home = fit(LogNormal, xg_home)
lognorm_away = fit(LogNormal, xg_away)

println("--- Fitted Distributions (Home) ---")
println("Gamma: ", gamma_home)
println("LogNormal: ", lognorm_home)
#=
julia> println("Gamma: ", gamma_home)
Gamma: Gamma{Float64}(α=3.2606750919949286, θ=0.49313871235643186)

julia> println("LogNormal: ", lognorm_home)
LogNormal: LogNormal{Float64}(μ=0.313859726264117, σ=0.6011675078068176)
=#



# ==========================================
# 3. Compare Fits using AIC
# ==========================================
# AIC = 2k - 2ln(L), where k is the number of parameters (2 for both distributions)
# Lower AIC indicates a better fit.

function calculate_aic(fitted_dist, data)
    k = length(params(fitted_dist))
    return 2 * k - 2 * loglikelihood(fitted_dist, data)
end

println("\n--- AIC Comparison (Lower is better) ---")
println("Home - Gamma AIC:      ", calculate_aic(gamma_home, xg_home))
println("Home - LogNormal AIC:  ", calculate_aic(lognorm_home, xg_home))
println("Away - Gamma AIC:      ", calculate_aic(gamma_away, xg_away))
println("Away - LogNormal AIC:  ", calculate_aic(lognorm_away, xg_away))

#=

julia> println("Home - Gamma AIC:      ", calculate_aic(gamma_home, xg_home))
Home - Gamma AIC:      414.26700851330713

julia> println("Home - LogNormal AIC:  ", calculate_aic(lognorm_home, xg_home))
Home - LogNormal AIC:  424.0273074212293

julia> println("Away - Gamma AIC:      ", calculate_aic(gamma_away, xg_away))
Away - Gamma AIC:      364.46021709305296

julia> println("Away - LogNormal AIC:  ", calculate_aic(lognorm_away, xg_away))
Away - LogNormal AIC:  399.5573313101946

=#

# ==========================================
# 4. Formal Hypothesis Testing (Kolmogorov-Smirnov)
# ==========================================
# The KS test evaluates the null hypothesis that the data comes from the fitted distribution.
# A p-value > 0.05 means we fail to reject the null (i.e., it's a decent fit).

ks_gamma_home = ExactOneSampleKSTest(xg_home, gamma_home)
ks_lognorm_home = ExactOneSampleKSTest(xg_home, lognorm_home)

println("\n--- KS Test Results (Home xG) ---")
println("Testing Gamma Fit:")
println("p-value: ", pvalue(ks_gamma_home))

println("\nTesting LogNormal Fit:")
println("p-value: ", pvalue(ks_lognorm_home))


#=
julia> ks_gamma_home = ExactOneSampleKSTest(xg_home, gamma_home)
Exact one sample Kolmogorov-Smirnov test
----------------------------------------
Population details:
    parameter of interest:   Supremum of CDF differences
    value under h_0:         0.0
    point estimate:          0.0489804

Test summary:
    outcome with 95% confidence: fail to reject h_0
    two-sided p-value:           0.7849

Details:
    number of observations:   172


julia> ks_lognorm_home = ExactOneSampleKSTest(xg_home, lognorm_home)
Exact one sample Kolmogorov-Smirnov test
----------------------------------------
Population details:
    parameter of interest:   Supremum of CDF differences
    value under h_0:         0.0
    point estimate:          0.0691788

Test summary:
    outcome with 95% confidence: fail to reject h_0
    two-sided p-value:           0.3660

Details:
    number of observations:   172


julia> println("\n--- KS Test Results (Home xG) ---")

--- KS Test Results (Home xG) ---

julia> println("Testing Gamma Fit:")
Testing Gamma Fit:

julia> println("p-value: ", pvalue(ks_gamma_home))
p-value: 0.7849391173832432

julia> println("\nTesting LogNormal Fit:")

Testing LogNormal Fit:

julia> println("p-value: ", pvalue(ks_lognorm_home))
p-value: 0.3660235508551567


=#


using DataFrames
using Distributions
using Statistics

# 1. Join the matches and stats DataFrames on match_id
# We only need the team names and the xG columns
df_joined = innerjoin(ds.matches[:, [:match_id, :home_team, :away_team]], 
                      stat[:, [:match_id, :expectedGoals_home, :expectedGoals_away]], 
                      on = :match_id)

# 2. Extract Home xG and Away xG into separate DataFrames, renaming columns to match
home_df = select(df_joined, :home_team => :team, :expectedGoals_home => :xG)
away_df = select(df_joined, :away_team => :team, :expectedGoals_away => :xG)

# 3. Stack them vertically into a single "long" DataFrame
team_xg_long = vcat(home_df, away_df)

# 4. Clean the data
# - Drop any missing xG values
# - Filter out any exact 0.0 values (The Gamma distribution strictly requires x > 0. A 0.0 will crash the fitter).
dropmissing!(team_xg_long, :xG)
filter!(:xG => >(0.0), team_xg_long)

# 5. Group by Team and apply the Gamma fit
team_gamma_parameters = combine(groupby(team_xg_long, :team)) do group
    # Convert to standard Float64 array for the fitter
    vals = Float64.(group.xG) 
    
    # Fit the distribution
    fitted = fit(Gamma, vals)
    
    # Return a named tuple to populate the new columns
    (;
        matches_played = length(vals),
        mean_xG = mean(vals),
        shape_alpha = shape(fitted),
        scale_theta = scale(fitted)
    )
end

# Sort by highest mean xG for easy reading
sort!(team_gamma_parameters, :mean_xG, rev=true)

# Display the final table
team_gamma_parameters

#=
julia> team_gamma_parameters
10×5 DataFrame
 Row │ team                  matches_played  mean_xG  shape_alpha  scale_theta 
     │ String?               Int64           Float64  Float64      Float64     
─────┼─────────────────────────────────────────────────────────────────────────
   1 │ bohemian                          35  1.83943      3.32278     0.553581
   2 │ galway-united                     35  1.54029      3.73681     0.412193
   3 │ st-patricks-athletic              34  1.53059      3.81127     0.401595
   4 │ shamrock-rovers                   33  1.5103       3.77853     0.399706
   5 │ shelbourne                        33  1.50697      2.47872     0.607962
   6 │ derry-city                        36  1.38111      1.8558      0.744214
   7 │ sligo-rovers                      35  1.31914      2.65299     0.497228
   8 │ waterford-fc                      35  1.31571      2.59702     0.506625
   9 │ drogheda-united                   35  1.20743      2.55923     0.471793
  10 │ cork-city                         33  1.06121      2.26345     0.468847
=#

using DataFrames
using Statistics

# Assuming you still have 'df_joined' from the previous step:
# df_joined = innerjoin(ds.matches[:, [:match_id, :home_team, :away_team]], 
#                       stat[:, [:match_id, :expectedGoals_home, :expectedGoals_away]], 
#                       on = :match_id)

# 1. Create Home records 
# If they are the home team, their xG is the home_xG, and their xGA is the away_xG
home_records = select(df_joined, 
    :home_team => :team, 
    :expectedGoals_home => :xG, 
    :expectedGoals_away => :xGA
)

# 2. Create Away records 
# If they are the away team, their xG is the away_xG, and their xGA is the home_xG
away_records = select(df_joined, 
    :away_team => :team, 
    :expectedGoals_away => :xG, 
    :expectedGoals_home => :xGA
)

# 3. Stack them into one long DataFrame
team_perf_long = vcat(home_records, away_records)

# 4. Clean missing values (drop rows where either xG or xGA is missing)
dropmissing!(team_perf_long, [:xG, :xGA])

# 5. Group by team and calculate the per-game averages
team_summary = combine(groupby(team_perf_long, :team)) do group
    mean_xg = mean(group.xG)
    mean_xga = mean(group.xGA)
    
    (;
        matches = nrow(group),
        xG_For = round(mean_xg, digits=2),
        xG_Against = round(mean_xga, digits=2),
        xG_Diff = round(mean_xg - mean_xga, digits=2) # The crucial metric!
    )
end

# Sort by Expected Goal Difference (Best to Worst)
sort!(team_summary, :xG_Diff, rev=true)

team_summary


#=


julia> team_summary
10×5 DataFrame
 Row │ team                  matches  xG_For   xG_Against  xG_Diff 
     │ String?               Int64    Float64  Float64     Float64 
─────┼─────────────────────────────────────────────────────────────
   1 │ shamrock-rovers            33     1.51        0.93     0.58
   2 │ bohemian                   35     1.84        1.29     0.55
   3 │ st-patricks-athletic       34     1.53        1.06     0.47
   4 │ shelbourne                 33     1.51        1.13     0.38
   5 │ derry-city                 36     1.38        1.23     0.15
   6 │ galway-united              35     1.54        1.6     -0.06
   7 │ drogheda-united            35     1.21        1.46    -0.26
   8 │ sligo-rovers               35     1.32        1.81    -0.49
   9 │ waterford-fc               35     1.32        1.87    -0.55
  10 │ cork-city                  33     1.06        1.81    -0.75
=#


using DataFrames
using Distributions
using Statistics

# Assuming 'team_perf_long' is already created from the previous step,
# containing columns: :team, :xG, :xGA

team_full_gamma = combine(groupby(team_perf_long, :team)) do group
    # Extract and filter values > 0.0 to prevent DomainErrors in the Gamma fit
    xg_vals = Float64.(filter(x -> x > 0.0, group.xG))
    xga_vals = Float64.(filter(x -> x > 0.0, group.xGA))
    
    # Fit the distributions
    fit_xg = fit(Gamma, xg_vals)
    fit_xga = fit(Gamma, xga_vals)
    
    # Return the metrics
    (;
        matches = nrow(group),
        xG_mean = round(mean(xg_vals), digits=2),
        xG_alpha = round(shape(fit_xg), digits=3),
        xG_theta = round(scale(fit_xg), digits=3),
        xGA_mean = round(mean(xga_vals), digits=2),
        xGA_alpha = round(shape(fit_xga), digits=3),
        xGA_theta = round(scale(fit_xga), digits=3)
    )
end

# Sort by highest attacking mean xG
sort!(team_full_gamma, :xG_mean, rev=true)

team_full_gamma

#=
julia> team_full_gamma
10×8 DataFrame
 Row │ team                  matches  xG_mean  xG_alpha  xG_theta  xGA_mean  xGA_alpha  xGA_theta 
     │ String?               Int64    Float64  Float64   Float64   Float64   Float64    Float64   
─────┼────────────────────────────────────────────────────────────────────────────────────────────
   1 │ bohemian                   35     1.84     3.323     0.554      1.29      3.55       0.364
   2 │ galway-united              35     1.54     3.737     0.412      1.6       3.511      0.456
   3 │ st-patricks-athletic       34     1.53     3.811     0.402      1.06      1.932      0.547
   4 │ shelbourne                 33     1.51     2.479     0.608      1.13      2.824      0.4
   5 │ shamrock-rovers            33     1.51     3.779     0.4        0.93      2.679      0.347
   6 │ derry-city                 36     1.38     1.856     0.744      1.23      2.717      0.455
   7 │ sligo-rovers               35     1.32     2.653     0.497      1.81      3.44       0.526
   8 │ waterford-fc               35     1.32     2.597     0.507      1.87      5.313      0.351
   9 │ drogheda-united            35     1.21     2.559     0.472      1.46      2.887      0.507
  10 │ cork-city                  33     1.06     2.263     0.469      1.81      2.825      0.64

=#


using Distributions
using DataFrames

# A function to simulate a match using the means derived from your Gamma parameters
function simulate_match(home_team_name, away_team_name, df; n_sims=10000)
    # 1. Extract team stats from your team_full_gamma DataFrame
    home_stats = subset(df, :team => ByRow(==(home_team_name)))[1, :]
    away_stats = subset(df, :team => ByRow(==(away_team_name)))[1, :]
    
    # 2. Calculate the "Match Expected Goals" (Blending Attack and Defense)
    # Home expectation = (Home Attack Mean + Away Defense Mean) / 2
    home_lambda = (home_stats.xG_mean + away_stats.xGA_mean) / 2
    
    # Away expectation = (Away Attack Mean + Home Defense Mean) / 2
    away_lambda = (away_stats.xG_mean + home_stats.xGA_mean) / 2
    
    # 3. Run the Monte Carlo Simulation using Poisson distributions
    home_simulated_goals = rand(Poisson(home_lambda), n_sims)
    away_simulated_goals = rand(Poisson(away_lambda), n_sims)
    
    # 4. Tally the results
    home_wins = sum(home_simulated_goals .> away_simulated_goals)
    draws = sum(home_simulated_goals .== away_simulated_goals)
    away_wins = sum(home_simulated_goals .< away_simulated_goals)
    
    println("--- Match Simulation: $home_team_name vs $away_team_name ---")
    println("Expected Goals: $home_team_name ($(round(home_lambda, digits=2))) - $(round(away_lambda, digits=2))) $away_team_name")
    println("Home Win Probability: ", round((home_wins / n_sims) * 100, digits=1), "%")
    println("Draw Probability:     ", round((draws / n_sims) * 100, digits=1), "%")
    println("Away Win Probability: ", round((away_wins / n_sims) * 100, digits=1), "%")
end

# Example usage:
simulate_match("shamrock-rovers", "bohemian", team_full_gamma)


s = subset(ds.matches, :home_team => ByRow(isequal("shamrock-rovers")), :away_team => ByRow(isequal("bohemian")), :season_id => ByRow(isequal( 69981 )))

o = subset(ds.odds, :match_id => ByRow(in(s.match_id)), :market_name => ByRow(isequal("1X2")))







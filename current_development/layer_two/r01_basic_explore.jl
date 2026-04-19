# current_development/layer_two/r01_basic_explore.jl


using Revise
using BayesianFootball

using DataFrames
using ThreadPinning
pinthreads(:cores)


include("./l01_basic_explore.jl")


# load the datastore - which is all the details regarding a tournament
ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.ScottishLower())
#=
julia> ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.ScottishLower())
[ Info: Building DataStore for BayesianFootball.Data.ScottishLower...
DataStore [BayesianFootball.Data.ScottishLower]
 (Container for football data)
=========
[Matches] (1950 rows × 20 cols)
  Columns: tournament_id, season_id, season, match_id, tournament_slug, home_team + 14 more

[Statistics] (Empty)

[Odds] (38551 rows × 20 cols)
  Columns: match_id, market_name, market_line, selection, odds_open, odds_close + 14 more

[Lineups] (69513 rows × 16 cols)
  Columns: tournament_id, season_id, match_id, team_side, player_id, player_name + 10 more

[Incidents] (30205 rows × 17 cols)
  Columns: id, match_id, incident_type, time, is_home, added_time + 11 more
=#


# load the past ran experiment in which we fit a model to expanding window CV 
# So a new model for each split, 
exp_dir = "exp/ablation_study"
saved_folders = BayesianFootball.Experiments.list_experiments(exp_dir)
exp = BayesianFootball.Experiments.load_experiment(saved_folders[2])


#=
julia> exp = BayesianFootball.Experiments.load_experiment(saved_folders[2])
Loading: 06_02_ablation_month_r_20260327_143844
BayesianFootball.Experiments.ExperimentResults(BayesianFootball.Experiments.ExperimentConfig("06_02_ablation_month_r", BayesianFootball.Models.PreGame.AblationStudy_NB_baseline_month_r(Distributions.Normal{Float64}(μ=0.2, σ=0.2), Distributions.Normal{Float64}(μ=0.12,
 σ=0
.5), Distributions.Normal{Float64}(μ=2.5, σ=0.5), Distributions.Gamma{Float64}(α=2.0, θ=0.05), Distributions.Normal{Float64}(μ=0.0, σ=1.0), Distributions.Gamma{Float64}(α=2.0, θ=0.15), Distributions.Gamma{Float64}(α=2.0, θ=0.04), Distributions.Gamma{Float64}(α=2.0, θ
=0.0
15), Distributions.Normal{Float64}(μ=0.0, σ=1.0), Distributions.Normal{Float64}(μ=0.0, σ=1.0), Distributions.Normal{Float64}(μ=0.0, σ=1.0)), BayesianFootball.Data.GroupedCVConfig([[56, 57]], ["22/23", "23/24", "24/25", "25/26"], 2, :match_month, 0, nothing, true), Tr
aini
ngConfig(strategy=Independent, checkpointing=false), ["time:7h 31m"], "", "./data/exp/ablation_study"), Tuple{MCMCChains.Chains{Float64, AxisArrays.AxisArray{Float64, 3, Array{Float64, 3}, Tuple{AxisArrays.Axis{:iter, StepRange{Int64, Int64}}, AxisArrays.Axis{:var, V
ecto
r{Symbol}}, AxisArrays.Axis{:chain, UnitRange{Int64}}}}, Missing, @NamedTuple{parameters::Vector{Symbol}, internals::Vector{Symbol}}, @NamedTuple{varname_to_symbol::OrderedCollections.OrderedDict{AbstractPPL.VarName, Symbol}, start_time::Vector{Float64}, stop_time::V
ecto
r{Float64}, samplerstate::Vector{Nothing}}}, BayesianFootball.Data.GroupedSplitMetaData}[(MCMC chain (300×186×2 Array{Float64, 3}), GroupedSplit(Tourns: [56, 57], Season: 22/23, Week: 1, Hist: 2)), (MCMC chain (300×236×2 Array{Float64, 3}), GroupedSplit(Tourns: [56, 
57],
 Season: 22/23, Week: 2, Hist: 2)), (MCMC chain (300×286×2 Array{Float64, 3}), GroupedSplit(Tourns: [56, 57], Season: 22/23, Week: 3, Hist: 2)), (MCMC chain (300×336×2 Array{Float64, 3}), GroupedSplit(Tourns: [56, 57], Season: 22/23, Week: 4, Hist: 2)), (MCMC chain (
300×
386×2 Array{Float64, 3}), GroupedSplit(Tourns: [56, 57], Season: 22/23, Week: 5, Hist: 2)), (MCMC chain (300×436×2 Array{Float64, 3}), GroupedSplit(Tourns: [56, 57], Season: 22/23, Week: 6, Hist: 2)), (MCMC chain (300×486×2 Array{Float64, 3}), GroupedSplit(Tourns: [5
6, 5
7], Season: 22/23, Week: 7, Hist: 2)), (MCMC chain (300×536×2 Array{Float64, 3}), GroupedSplit(Tourns: [56, 57], Season: 22/23, Week: 8, Hist: 2)), (MCMC chain (300×586×2 Array{Float64, 3}), GroupedSplit(Tourns: [56, 57], Season: 22/23, Week: 9, Hist: 2)), (MCMC chai
n (3
00×586×2 Array{Float64, 3}), GroupedSplit(Tourns: [56, 57], Season: 23/24, Week: 1, Hist: 2))  …  (MCMC chain (300×536×2 Array{Float64, 3}), GroupedSplit(Tourns: [56, 57], Season: 24/25, Week: 7, Hist: 2)), (MCMC chain (300×536×2 Array{Float64, 3}), GroupedSplit(Tour
ns: 
[56, 57], Season: 24/25, Week: 8, Hist: 2)), (MCMC chain (300×536×2 Array{Float64, 3}), GroupedSplit(Tourns: [56, 57], Season: 24/25, Week: 9, Hist: 2)), (MCMC chain (300×496×2 Array{Float64, 3}), GroupedSplit(Tourns: [56, 57], Season: 25/26, Week: 1, Hist: 2)), (MCM
C ch
ain (300×496×2 Array{Float64, 3}), GroupedSplit(Tourns: [56, 57], Season: 25/26, Week: 2, Hist: 2)), (MCMC chain (300×496×2 Array{Float64, 3}), GroupedSplit(Tourns: [56, 57], Season: 25/26, Week: 3, Hist: 2)), (MCMC chain (300×496×2 Array{Float64, 3}), GroupedSplit(T
ourn
s: [56, 57], Season: 25/26, Week: 4, Hist: 2)), (MCMC chain (300×496×2 Array{Float64, 3}), GroupedSplit(Tourns: [56, 57], Season: 25/26, Week: 5, Hist: 2)), (MCMC chain (300×496×2 Array{Float64, 3}), GroupedSplit(Tourns: [56, 57], Season: 25/26, Week: 6, Hist: 2)), (
MCMC
 chain (300×496×2 Array{Float64, 3}), GroupedSplit(Tourns: [56, 57], Season: 25/26, Week: 7, Hist: 2))], nothing, "./data/exp/ablation_study/06_02_ablation_month_r_20260327_143844")

julia> 

julia> exp.
config
save_path
training_results
vocabulary


julia> exp.training_results
TrainingResults: 34 

julia> exp.training_results[1][1]
Chains MCMC chain (300×186×2 Array{Float64, 3}):

Iterations        = 51:1:350
Number of chains  = 2
Samples per chain = 300
Wall duration     = 1004.24 seconds
Compute duration  = 1988.46 seconds
parameters        = μ, γ, log_r, log_r_month.σ, log_r_month.z[1], log_r_month.z[2], log_r_month.z[3], log_r_month.z[4], log_r_month.z[5], log_r_month.z[6], log_r_month.z[7], log_r_month.z[8], log_r_month.z[9], log_r_month.z[10], log_r_month.z[11], log_r_month.z[12], 
α.σ₀
, α.σₛ, α.σₖ, α.z_init[1], α.z_init[2], α.z_init[3], α.z_init[4], α.z_init[5], α.z_init[6], α.z_init[7], α.z_init[8], α.z_init[9], α.z_init[10], α.z_init[11], α.z_init[12], α.z_init[13], α.z_init[14], α.z_init[15], α.z_init[16], α.z_init[17], α.z_init[18], α.z_init[1
9], 
α.z_init[20], α.z_init[21], α.z_init[22], α.z_init[23], α.z_init[24], α.z_init[25], α.z_season_steps[1, 1], α.z_season_steps[2, 1], α.z_season_steps[3, 1], α.z_season_steps[4, 1], α.z_season_steps[5, 1], α.z_season_steps[6, 1], α.z_season_steps[7, 1], α.z_season_step
s[8,
 1], α.z_season_steps[9, 1], α.z_season_steps[10, 1], α.z_season_steps[11, 1], α.z_season_steps[12, 1], α.z_season_steps[13, 1], α.z_season_steps[14, 1], α.z_season_steps[15, 1], α.z_season_steps[16, 1], α.z_season_steps[17, 1], α.z_season_steps[18, 1], α.z_season_st
eps[
19, 1], α.z_season_steps[20, 1], α.z_season_steps[21, 1], α.z_season_steps[22, 1], α.z_season_steps[23, 1], α.z_season_steps[24, 1], α.z_season_steps[25, 1], α.z_season_steps[1, 2], α.z_season_steps[2, 2], α.z_season_steps[3, 2], α.z_season_steps[4, 2], α.z_season_st
eps[
5, 2], α.z_season_steps[6, 2], α.z_season_steps[7, 2], α.z_season_steps[8, 2], α.z_season_steps[9, 2], α.z_season_steps[10, 2], α.z_season_steps[11, 2], α.z_season_steps[12, 2], α.z_season_steps[13, 2], α.z_season_steps[14, 2], α.z_season_steps[15, 2], α.z_season_ste
ps[1
6, 2], α.z_season_steps[17, 2], α.z_season_steps[18, 2], α.z_season_steps[19, 2], α.z_season_steps[20, 2], α.z_season_steps[21, 2], α.z_season_steps[22, 2], α.z_season_steps[23, 2], α.z_season_steps[24, 2], α.z_season_steps[25, 2], β.σ₀, β.σₛ, β.σₖ, β.z_init[1], β.z_
init
[2], β.z_init[3], β.z_init[4], β.z_init[5], β.z_init[6], β.z_init[7], β.z_init[8], β.z_init[9], β.z_init[10], β.z_init[11], β.z_init[12], β.z_init[13], β.z_init[14], β.z_init[15], β.z_init[16], β.z_init[17], β.z_init[18], β.z_init[19], β.z_init[20], β.z_init[21], β.z
_ini
t[22], β.z_init[23], β.z_init[24], β.z_init[25], β.z_season_steps[1, 1], β.z_season_steps[2, 1], β.z_season_steps[3, 1], β.z_season_steps[4, 1], β.z_season_steps[5, 1], β.z_season_steps[6, 1], β.z_season_steps[7, 1], β.z_season_steps[8, 1], β.z_season_steps[9, 1], β.
z_se
ason_steps[10, 1], β.z_season_steps[11, 1], β.z_season_steps[12, 1], β.z_season_steps[13, 1], β.z_season_steps[14, 1], β.z_season_steps[15, 1], β.z_season_steps[16, 1], β.z_season_steps[17, 1], β.z_season_steps[18, 1], β.z_season_steps[19, 1], β.z_season_steps[20, 1]
, β.
z_season_steps[21, 1], β.z_season_steps[22, 1], β.z_season_steps[23, 1], β.z_season_steps[24, 1], β.z_season_steps[25, 1], β.z_season_steps[1, 2], β.z_season_steps[2, 2], β.z_season_steps[3, 2], β.z_season_steps[4, 2], β.z_season_steps[5, 2], β.z_season_steps[6, 2], 
β.z_
season_steps[7, 2], β.z_season_steps[8, 2], β.z_season_steps[9, 2], β.z_season_steps[10, 2], β.z_season_steps[11, 2], β.z_season_steps[12, 2], β.z_season_steps[13, 2], β.z_season_steps[14, 2], β.z_season_steps[15, 2], β.z_season_steps[16, 2], β.z_season_steps[17, 2],
 β.z
_season_steps[18, 2], β.z_season_steps[19, 2], β.z_season_steps[20, 2], β.z_season_steps[21, 2], β.z_season_steps[22, 2], β.z_season_steps[23, 2], β.z_season_steps[24, 2], β.z_season_steps[25, 2]
internals         = n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size, lp, logprior, loglikelihood

Use `describe(chains)` for summary statistics and quantiles.

=#



# ---

# The model is trained on the following data splits ( which breack the ds into windows )
# This stop look ahead / data leakage into the model.
splits = Data.create_data_splits(ds, exp.config.splitter)
#=
julia> splits[1]
(580×20 SubDataFrame
 Row │ tournament_id  season_id  season   match_id  tournament_slug  home_team          away_team          home_score  away_score  home_score_ht  away_score_ht  winner_code  round   has_xg   has_stats  match_hour  match_month  match_dayofweek  match_date  match_week 
     │ Int32?         Int32?     String?  Int32?    String?          String?            String?            Int32?      Int32?      Int32?         Int32?         Int32?       Int32?  Bool?    Bool?      Int64       Int64        Int64            Dates.Date  Int64      
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │            56      29270  20/21     8824033  missing          cove-rangers       east-fife                   3           1              1              0            1       1  missing    missing          14            1                5  2020-10-17           1
   2 │            56      29270  20/21     8824035  missing          airdrieonians      peterhead                   2           0              1              0            1       1  missing    missing          14            1                5  2020-10-17           1
   3 │            56      29270  20/21     8824032  missing          montrose           falkirk-fc                  1           3              0              1            2       1  missing    missing          14            1                5  2020-10-17           1
   4 │            56      29270  20/21     8824034  missing          forfar-athletic    dumbarton                   0           0              0              0            3       1  missing    missing          14            1                5  2020-10-17           1
   5 │            56      29270  20/21     8824045  missing          clyde-fc           partick-thistle             1           0              0              0            1       1  missing    missing          14            1                5  2020-10-17           1
   6 │            56      29270  20/21     8824073  missing          falkirk-fc         forfar-athletic             1           1              1              0            3       2  missing    missing          14            1                5  2020-10-24           2
=#

# Furthermore, we take the raw Data splits and create them into features for the model. 
# Mostly apply some rules and turning them into vectorise data, for better performance 
feature_sets = Features.create_features(splits, exp.config.model, exp.config.splitter)
#=
julia> feature_sets[4][1]
FeatureSet with 22 entries:
  :n_target_steps   => 4
  :round_away_ids   => [[12, 20, 15, 10, 19, 13, 4, 14, 23, 22  …  15, 8, 12, 7, 19, 25, 14, 13, 9, 23], [15, 3, 22, 18, 10, 25, 16, 13, 9, 24  …  15, 8, 20, 7, 18, 13, 17, 23, 14, 2], [18, 7, 13, 1, 3, 24, 4, 12, 2, 16  …  15, 11, 3, 7, 21, 24, 10, 16, 2, 25], [1, 1
7, …
  :matches_df       => 701×20 DataFrame…
  :n_months         => 12
  :flat_away_ids    => [12, 20, 15, 10, 19, 13, 4, 14, 23, 22  …  16, 12, 25, 5, 4, 21, 20, 1, 13, 3]
  :round_is_plastic => [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0  …  0, 1, 1, 1, 0, 0, 0, 0, 0, 1], [1, 0, 0, 1, 1, 0, 1, 0, 1, 1  …  0, 0, 1, 0, 1, 0, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 1, 1  …  1, 1, 1, 1, 0, 1, 1, 0, 1, 0], [0, 1, 1, 1, 1, 0, 0, 1, 0, 1  …  1, 0, 0, 0, 1,
 1,…
  :round_month_ids  => [[10, 10, 10, 10, 10, 10, 10, 10, 10, 10  …  4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [7, 7, 7, 7, 7, 7, 7, 7, 7, 7  …  4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [7, 7, 7, 7, 7, 7, 7, 7, 7, 7  …  8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8  …  9, 9
, 9…
  :time_indices     => [1, 1, 1, 1, 1, 1, 1, 1, 1, 1  …  6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
  :flat_is_plastic  => [1, 1, 1, 1, 1, 0, 0, 0, 0, 0  …  0, 0, 0, 1, 0, 0, 1, 1, 1, 1]
  :flat_months      => [10, 10, 10, 10, 10, 10, 10, 10, 10, 10  …  11, 11, 11, 11, 11, 12, 12, 12, 12, 12]
  :flat_home_goals  => [3, 2, 1, 0, 1, 1, 0, 1, 1, 0  …  2, 3, 1, 1, 1, 1, 3, 1, 0, 0]
  :round_home_ids   => [[8, 1, 18, 16, 7, 6, 9, 25, 2, 24  …  20, 18, 1, 16, 10, 2, 22, 24, 6, 4], [8, 20, 12, 1, 7, 14, 4, 2, 17, 23  …  22, 12, 1, 10, 3, 24, 4, 16, 25, 9], [15, 21, 17, 20, 11, 10, 25, 14, 23, 5  …  13, 17, 1, 18, 20, 4, 23, 12, 5, 14], [11, 18, 13
, 1…
  :flat_home_ids    => [8, 1, 18, 16, 7, 6, 9, 25, 2, 24  …  10, 24, 2, 23, 25, 11, 17, 15, 7, 18]
  :flat_is_midweek  => [0, 0, 0, 0, 0, 0, 0, 0, 0, 0  …  0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  :n_history_steps  => 2
  :team_map         => Dict("edinburgh-city-fc"=>13, "queens-park-fc"=>22, "albion-rovers"=>2, "clyde-fc"=>7, "queen-of-the-south"=>21, "brechin-city"=>6, "stirling-albion"=>24, "dunfermline-athletic"=>11, "forfar-athletic"=>16, "peterhead"=>20…)
  :n_rounds         => 6
  :flat_away_goals  => [1, 0, 3, 0, 0, 5, 3, 4, 3, 0  …  2, 1, 0, 1, 1, 1, 0, 1, 2, 0]
  :n_teams          => 25
  :round_home_goals => SubArray{Int64, 1, Vector{Int64}, Tuple{Vector{Int64}}, false}[[3, 2, 1, 0, 1, 1, 0, 1, 1, 0  …  1, 1, 2, 1, 0, 0, 0, 0, 0, 5], [1, 2, 1, 0, 0, 1, 0, 2, 2, 0  …  1, 2, 1, 2, 4, 5, 1, 0, 2, 0], [0, 1, 0, 0, 1, 2, 1, 1, 1, 2  …  0, 0, 2, 2, 1, 1,
 1,…
  :round_away_goals => SubArray{Int64, 1, Vector{Int64}, Tuple{Vector{Int64}}, false}[[1, 0, 3, 0, 0, 5, 3, 4, 3, 0  …  0, 0, 0, 3, 2, 2, 0, 1, 0, 1], [1, 0, 1, 3, 3, 1, 2, 0, 0, 1  …  1, 3, 1, 1, 1, 0, 2, 0, 0, 1], [0, 4, 2, 2, 0, 0, 0, 3, 0, 0  …  3, 0, 0, 1, 4, 2,
 3,…
  :round_is_midweek => [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0  …  1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0  …  0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0  …  0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0  …  0, 0, 1, 1, 0,
 0,…
=#



# That is move or less what the extract_oos_predictions function does, 
# it allows use to predict the next set of games 
# here we extract the models parameters as distributions - so vector that estimates the distribution 
# of the parameters of the model for that game - inference 
latents = Experiments.extract_oos_predictions(ds, exp)
#=
julia> latents = Experiments.extract_oos_predictions(ds, exp)
Progress: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:00
1226×4 DataFrame
  Row │ match_id  r                                  λ_a                                λ_h                               
      │ Any       Any                                Any                                Any                               
──────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    1 │ 10388081  [16.4946, 27.4114, 28.7736, 32.1…  [0.878594, 1.0832, 0.886278, 1.1…  [1.74702, 2.2618, 1.16424, 1.505…
    2 │ 10388169  [16.4946, 27.4114, 28.7736, 32.1…  [1.78895, 1.53821, 1.04986, 1.23…  [1.64656, 1.03065, 1.42405, 1.19…
    3 │ 10388096  [16.4946, 27.4114, 28.7736, 32.1…  [0.759399, 0.730802, 1.35093, 1.…  [1.49915, 1.21782, 1.87237, 1.14…
    4 │ 10388088  [15.7069, 26.4051, 28.1576, 32.9…  [1.45048, 1.79832, 1.25669, 1.59…  [1.22016, 1.51087, 1.18414, 1.53…
    5 │ 10387962  [16.4946, 27.4114, 28.7736, 32.1…  [2.56881, 2.39578, 1.51223, 1.51…  [1.3577, 0.989402, 1.53376, 1.51…
    6 │ 10387886  [15.0775, 27.6305, 27.3532, 33.2…  [1.34817, 2.52898, 0.922688, 1.4…  [1.50519, 1.03016, 1.79419, 1.37…
    7 │ 10387855  [16.4946, 27.4114, 28.7736, 32.1…  [0.976525, 0.756354, 1.08315, 1.…  [0.9729, 1.5198, 1.02336, 0.9273…
    8 │ 10388129  [16.4946, 27.4114, 28.7736, 32.1…  [1.28105, 1.80844, 0.632939, 1.6…  [1.91887, 1.21399, 1.90701, 1.02…
    9 │ 10388093  [15.7069, 26.4051, 28.1576, 32.9…  [1.82357, 1.31442, 2.15204, 1.21…  [2.44414, 3.44426, 0.934655, 2.0…
   10 │ 10387926  [15.7069, 26.4051, 28.1576, 32.9…  [1.02071, 2.22053, 0.956407, 1.1…  [1.83002, 1.99108, 3.45349, 1.84…
   11 │ 10387889  [15.0775, 27.6305, 27.3532, 33.2…  [0.86781, 0.836015, 0.791552, 0.…  [1.83389, 1.74369, 1.15628, 1.43…

Note: the model is a Negative Binomial discrete distribution 
=#


# From the models parameters, we can then make prediction/ inference 
# about the games/ well the number of markets for the games, so each match we have different 
# market line in which we can predict, so we have a Predictive Posterior Distribution 
ppd = Predictions.model_inference(latents)
#=
julia> ppd = Predictions.model_inference(latents)
Running Inference on 1226 matches...
33102×5 DataFrame
   Row │ match_id  market_name  market_line  selection  distribution                      
       │ Int64     String       Float64      Symbol     Array…                            
───────┼──────────────────────────────────────────────────────────────────────────────────
     1 │ 10388081  1X2                  0.0  away       [0.194111, 0.176082, 0.28007, 0.…
     2 │ 10388081  1X2                  0.0  home       [0.571166, 0.632279, 0.420725, 0…
     3 │ 10388081  1X2                  0.0  draw       [0.23472, 0.191621, 0.299206, 0.…
     4 │ 10388081  BTTS                 0.0  btts_yes   [0.465842, 0.579878, 0.396286, 0…
     5 │ 10388081  BTTS                 0.0  btts_no    [0.534155, 0.420104, 0.603714, 0…
     6 │ 10388081  OverUnder            0.5  under_05   [0.0807378, 0.0393342, 0.133427,…
     7 │ 10388081  OverUnder            0.5  over_05    [0.919259, 0.960648, 0.866573, 0…
     8 │ 10388081  OverUnder            1.5  over_15    [0.724369, 0.837476, 0.602554, 0…
     9 │ 10388081  OverUnder            1.5  under_15   [0.275628, 0.162506, 0.397446, 0…
    10 │ 10388081  OverUnder            2.5  over_25    [0.481339, 0.640714, 0.336722, 0…
    11 │ 10388081  OverUnder            2.5  under_25   [0.518658, 0.359268, 0.663278, 0…
    12 │ 10388081  OverUnder            3.5  under_35   [0.727221, 0.572993, 0.844816, 0…
    13 │ 10388081  OverUnder            3.5  over_35    [0.272775, 0.426989, 0.155184, 0…
    14 │ 10388081  OverUnder            4.5  under_45   [0.865677, 0.750521, 0.939386, 0…
    15 │ 10388081  OverUnder            4.5  over_45    [0.134319, 0.249461, 0.0606137, …
    16 │ 10388081  OverUnder            5.5  under_55   [0.941459, 0.870767, 0.979461, 0…
    17 │ 10388081  OverUnder            5.5  over_55    [0.0585376, 0.129215, 0.0205393,…
=#



#=
#TODO: 

Following the Gelman / and other papers, we looking to add a layer two to the overall process here. 
Layer two, will be the process in shifting the ppd of the market lines, based on data (feature_sets/ data split) 
to make an estimated guess / inference about a suitable shift to improve the metric (roi, loss etc ) of the different market lines. 
hence for each market line selection ( over_25, home, under_35 etc) we should compute a shift for each selection 
which we will then use, and feed into the next layer which deal with the signal - so stake size, but this is based off the ppd. 
So we want to remove the model baisis from market line. 

overall we looking at making an abstract type / funcitons to deal with this as there are different layer two 
models that we want to test and experiment. 
so for the layer two model we should be looking at have a set of interfaces.jl that define the 
funcitons we need to build for each model, so we can expanded and build more (abstract) 
in such  a way that we dont have have data leakage/ so possible use the data split, feature split. 

I guess we are tracking the shifts over time, 

also we need to play around with which metrics to use as roi would be ideal, but i guess that it will be non linear and 
hard to optimise. 

- not sure if randomforrest boosttrees might be usefull, 

anyway, im working at develop a module for my package to do the layer two, 
but im creating a development area in here in current_development/layer_two, to have files 
that i can run in the repl, so i dont need to play around with the project code until i get some thing im 
happy with. i tend to have a loader file that i just parse into the repl and a runner file that allow me to run the funciotns and 
write out code until i make it into a funciton. 

i have a rough code that i used for match day inference, which is a basic version in what we are build. 

can we make sure we are on the same page before coding this up, 
as the first stage is to play and design the strucutre 



=#

using Revise
using BayesianFootball
using DataFrames



ds = Data.load_extra_ds()
transform!(ds.matches, :match_week => ByRow(w -> cld(w, 4)) => :match_month)


model = BayesianFootball.Models.PreGame.GRWNegativeBinomial()


cv_config = BayesianFootball.Data.CVConfig(
    # tournament_ids = [56,57],
    tournament_ids = [56],
    target_seasons = ["24/25"],
    history_seasons = 2,
    dynamics_col = :match_month,
    # warmup_period = 36,
    warmup_period = 8,
    stop_early = true
)


cv_config_g = BayesianFootball.Data.GroupedCVConfig(
    tournament_groups = [[56, 57]],
    target_seasons = ["24/25"],
    history_seasons = 2,
    dynamics_col = :match_month,
    warmup_period = 8,
    stop_early = true
)

model = BayesianFootball.Models.PreGame.MSNegativeBinomial()

splits = BayesianFootball.Data.create_data_splits(ds, cv_config)
splits_grouped = BayesianFootball.Data.create_data_splits(ds, cv_config_g)

feature_sets = BayesianFootball.Features.create_features(
    splits, model, cv_config_g
)



f1 = feature_sets[1][1]
f2 = feature_sets[2][1]
feature_sets[2][2]
#=
Split(Tourn: 56, Season: 22/23, Week: 1, Hist: 1)
julia> f = feature_sets[1][1]
FeatureSet with 13 entries:
  :round_away_ids   => [[11, 5, 9, 14, 2, 7, 1, 12, 3, 4  …  8, 10, 11, 12, 1, 2, 9, 6, 3, 13], [4, 12, 14, 5, 9, 11, 7, 3, 1, 2, 7, 14, 5, 2, 4, 1, 2, 3, 9, 12], [3, 4, 7, 5, 11, 12, 11, 14, 4, 1, 5, 3, 7, 9, 2, 1, 9, 14, 5, 11], [12, 3, 2, 4, 7, 11, 12, 9, 1, 3, 4, 5,…
  :matches_df       => 200×33 DataFrame…
  :flat_away_ids    => [11, 5, 9, 14, 2, 7, 1, 12, 3, 4  …  1, 5, 2, 14, 7, 12, 11, 3, 4, 9]
  :time_indices     => [1, 1, 1, 1, 1, 1, 1, 1, 1, 1  …  9, 9, 9, 9, 9, 10, 10, 10, 10, 10]
  :flat_home_goals  => [0, 0, 1, 1, 2, 3, 2, 2, 2, 2  …  0, 1, 1, 2, 1, 1, 4, 2, 2, 1]
  :round_home_ids   => [[1, 3, 4, 7, 12, 2, 5, 9, 11, 14  …  2, 3, 6, 9, 13, 1, 8, 10, 11, 12], [3, 7, 9, 11, 2, 4, 5, 12, 14, 1, 1, 3, 9, 11, 12, 4, 5, 7, 11, 14], [1, 2, 9, 12, 14, 2, 3, 5, 7, 9, 1, 4, 11, 12, 14, 2, 3, 4, 7, 12], [1, 5, 9, 11, 14, 2, 4, 5, 7, 14, 1, …
  :flat_home_ids    => [1, 3, 4, 7, 12, 2, 5, 9, 11, 14  …  3, 4, 9, 11, 12, 1, 2, 5, 7, 14]
  :team_map         => Dict("queens-park-fc"=>14, "dumbarton"=>5, "clyde-fc"=>3, "queen-of-the-south"=>13, "alloa-athletic"=>2, "kelty-hearts-fc"=>10, "montrose"=>11, "cove-rangers"=>4, "falkirk-fc"=>9, "airdrieonians"=>1…)
  :n_rounds         => 10
  :flat_away_goals  => [3, 3, 1, 1, 0, 1, 2, 1, 2, 0  …  5, 0, 2, 1, 0, 1, 1, 1, 3, 1]
  :n_teams          => 14
  :round_home_goals => SubArray{Int64, 1, Vector{Int64}, Tuple{Vector{Int64}}, false}[[0, 0, 1, 1, 2, 3, 2, 2, 2, 2  …  2, 3, 1, 3, 1, 2, 0, 0, 2, 1], [2, 3, 0, 1, 2, 1, 5, 3, 0, 2, 3, 2, 1, 0, 0, 1, 1, 0, 2, 3], [2, 1, 2, 5, 1, 2, 0, 0, 4, 0, 3, 3, 4, 0, 3, 2, 1, 3, 2,…
  :round_away_goals => SubArray{Int64, 1, Vector{Int64}, Tuple{Vector{Int64}}, false}[[3, 3, 1, 1, 0, 1, 2, 1, 2, 0  …  4, 0, 0, 1, 1, 0, 3, 0, 1, 4], [1, 0, 1, 2, 0, 1, 0, 2, 0, 1, 0, 2, 2, 2, 1, 0, 1, 2, 2, 2], [1, 3, 1, 0, 1, 4, 5, 3, 2, 3, 2, 0, 1, 0, 4, 1, 3, 3, 1,…


Split(Tourn: 56, Season: 22/23, Week: 2, Hist: 1)
FeatureSet with 13 entries:
  :round_away_ids   => [[11, 5, 9, 14, 2, 7, 1, 12, 3, 4  …  8, 10, 11, 12, 1, 2, 9, 6, 3, 13], [4, 12, 14, 5, 9, 11, 7, 3, 1, 2  …  3, 11, 9, 10, 8, 12, 3, 1, 8, 2], [3, 4, 7, 5, 11, 12, 11, 14, 4, 1, 5, 3, 7, 9, 2, 1, 9, 14, 5, 11], [12, 3, 2, 4, 7, 11, 12, 9, 1, 3, 4…
  :matches_df       => 220×33 DataFrame…
  :flat_away_ids    => [11, 5, 9, 14, 2, 7, 1, 12, 3, 4  …  1, 5, 2, 14, 7, 12, 11, 3, 4, 9]
  :time_indices     => [1, 1, 1, 1, 1, 1, 1, 1, 1, 1  …  9, 9, 9, 9, 9, 10, 10, 10, 10, 10]
  :flat_home_goals  => [0, 0, 1, 1, 2, 3, 2, 2, 2, 2  …  0, 1, 1, 2, 1, 1, 4, 2, 2, 1]
  :round_home_ids   => [[1, 3, 4, 7, 12, 2, 5, 9, 11, 14  …  2, 3, 6, 9, 13, 1, 8, 10, 11, 12], [3, 7, 9, 11, 2, 4, 5, 12, 14, 1  …  1, 2, 6, 12, 13, 6, 9, 10, 11, 13], [1, 2, 9, 12, 14, 2, 3, 5, 7, 9, 1, 4, 11, 12, 14, 2, 3, 4, 7, 12], [1, 5, 9, 11, 14, 2, 4, 5, 7, 14,…
  :flat_home_ids    => [1, 3, 4, 7, 12, 2, 5, 9, 11, 14  …  3, 4, 9, 11, 12, 1, 2, 5, 7, 14]
  :team_map         => Dict("queens-park-fc"=>14, "dumbarton"=>5, "clyde-fc"=>3, "queen-of-the-south"=>13, "alloa-athletic"=>2, "kelty-hearts-fc"=>10, "montrose"=>11, "cove-rangers"=>4, "falkirk-fc"=>9, "airdrieonians"=>1…)
  :n_rounds         => 10
  :flat_away_goals  => [3, 3, 1, 1, 0, 1, 2, 1, 2, 0  …  5, 0, 2, 1, 0, 1, 1, 1, 3, 1]
  :n_teams          => 14
  :round_home_goals => SubArray{Int64, 1, Vector{Int64}, Tuple{Vector{Int64}}, false}[[0, 0, 1, 1, 2, 3, 2, 2, 2, 2  …  2, 3, 1, 3, 1, 2, 0, 0, 2, 1], [2, 3, 0, 1, 2, 1, 5, 3, 0, 2  …  5, 2, 1, 2, 4, 2, 2, 1, 1, 1], [2, 1, 2, 5, 1, 2, 0, 0, 4, 0, 3, 3, 4, 0, 3, 2, 1, 3,…
  :round_away_goals => SubArray{Int64, 1, Vector{Int64}, Tuple{Vector{Int64}}, false}[[3, 3, 1, 1, 0, 1, 2, 1, 2, 0  …  4, 0, 0, 1, 1, 0, 3, 0, 1, 4], [1, 0, 1, 2, 0, 1, 0, 2, 0, 1  …  0, 1, 1, 1, 1, 2, 0, 0, 2, 1], [1, 3, 1, 0, 1, 4, 5, 3, 2, 3, 2, 0, 1, 0, 4, 1, 3, 3,…

=#


f1[:round_home_goals][2]



f2[:round_home_goals][3]



export MSNegativeBinomial 

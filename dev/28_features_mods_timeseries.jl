"""

To update the feature sets to handle the dynamics models better. 

"""
"""
consider the more complex case for tournament_id = 54 - Scottish pl . 

"""

using BayesianFootball
using DataFrames
using Statistics


using ThreadPinning
using LinearAlgebra
pinthreads(:cores)

data_store = BayesianFootball.Data.load_default_datastore()

ds = BayesianFootball.Data.DataStore( 
    Data.add_match_week_column(subset( data_store.matches, 
           :tournament_id => ByRow(isequal(54)),
                                      :season => ByRow(isequal("24/25")))),
    data_store.odds,
    data_store.incidents
)



# check we match weeks 
combine( groupby( ds.matches, :match_week), 
        nrow => :n_matches,
        :round => (x -> [sort(unique(x))]) => :round_list
        )



model = BayesianFootball.Models.PreGame.AR1Poisson()
vocabulary = BayesianFootball.Features.create_vocabulary(ds, model) 


splitter_config = BayesianFootball.Data.StaticSplit(
    train_seasons = ["24/25"], 
    round_col = :match_week
)

data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)
feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model, splitter_config)


feature_sets[1][1].data





""" 
simpler case for the tournament_id = 55 -> Scottish Champs 

"""


using BayesianFootball
using DataFrames
using Statistics


using ThreadPinning
using LinearAlgebra
pinthreads(:cores)

data_store = BayesianFootball.Data.load_default_datastore()

df_matches = Data.add_match_week_column(ds.matches)
ds = BayesianFootball.Data.DataStore( 
    Data.add_match_week_column(subset( data_store.matches, 
           :tournament_id => ByRow(isequal(55)),
                                      :season => ByRow(isequal("24/25")))),
    data_store.odds,
    data_store.incidents
)


model = BayesianFootball.Models.PreGame.AR1Poisson()
vocabulary = BayesianFootball.Features.create_vocabulary(ds, model) 


splitter_config = BayesianFootball.Data.StaticSplit(
    train_seasons = ["24/25"], 
    round_col = :match_week
)

data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)
feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model, splitter_config)


feature_sets[1][1].data
"""
--- older version 


julia> feature_sets[1][1].data
Dict{Symbol, Any} with 12 entries:
  :round_away_ids   => [[10, 5, 1, 2, 3], [8, 4, 9, 6, 7], [5, 6, 1, 3, 10], [8, 2, 9, 10, 4], [3, 6, 9, 7, 5], [7, 2, 1, 4, 8], [9, 6, 4, 10, 5], [1, 7, 2…
  :matches_df       => 180×19 DataFrame…
  :flat_away_ids    => [10, 5, 1, 2, 3, 8, 4, 9, 6, 7  …  9, 10, 5, 8, 4, 2, 6, 3, 7, 1]
  :flat_home_goals  => [2, 0, 2, 0, 1, 1, 5, 0, 1, 0  …  2, 2, 1, 0, 0, 0, 0, 1, 3, 2]
  :flat_away_goals  => [1, 0, 0, 2, 0, 1, 0, 2, 0, 0  …  1, 2, 1, 3, 0, 1, 1, 5, 1, 0]
  :round_home_ids   => [[9, 6, 8, 7, 4], [10, 2, 1, 3, 5], [8, 9, 7, 2, 4], [3, 1, 5, 6, 7], [1, 2, 4, 8, 10], [3, 10, 6, 5, 9], [3, 7, 8, 1, 2], [4, 10, 9…
  :flat_home_ids    => [9, 6, 8, 7, 4, 10, 2, 1, 3, 5  …  6, 2, 3, 7, 1, 4, 8, 10, 9, 5]
  :team_map         => Dict{String31, Int64}("raith-rovers"=>3, "queens-park-fc"=>10, "ayr-united"=>2, "falkirk-fc"=>9, "airdrieonians"=>4, "greenock-morto…
  :n_rounds         => 36
  :n_teams          => 10
  :round_home_goals => SubArray{Int64, 1, Vector{Int64}, Tuple{Vector{Int64}}, false}[[2, 0, 2, 0, 1], [1, 5, 0, 1, 0], [1, 2, 1, 2, 0], [0, 1, 2, 3, 2], […
  :round_away_goals => SubArray{Int64, 1, Vector{Int64}, Tuple{Vector{Int64}}, false}[[1, 0, 0, 2, 0], [1, 0, 2, 0, 0], [1, 1, 0, 0, 2], [1, 1, 3, 0, 2], […

julia> feature_sets[1][1].data[:matches_df][:, [:match_id, :match_date, :home_team, :away_team, :home_score, :away_score]]
180×6 DataFrame
 Row │ match_id  match_date  home_team             away_team             home_score  away_score 
     │ Int64     Dates.Date  String31              String31              Int64       Int64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────
   1 │ 12477141  2024-08-02  falkirk-fc            queens-park-fc                 2           1
   2 │ 12477136  2024-08-03  partick-thistle       greenock-morton                0           0
   3 │ 12477137  2024-08-03  livingston            dunfermline-athletic           2           0
   4 │ 12477138  2024-08-03  hamilton-academical   ayr-united                     0           2
   5 │ 12477142  2024-08-03  airdrieonians         raith-rovers                   1           0
   6 │ 12476942  2024-08-09  queens-park-fc        livingston                     1           1
   7 │ 12476945  2024-08-09  ayr-united            airdrieonians                  5           0
   8 │ 12476940  2024-08-10  dunfermline-athletic  falkirk-fc                     0           2
   9 │ 12476943  2024-08-10  raith-rovers          partick-thistle                1           0
  10 │ 12476944  2024-08-10  greenock-morton       hamilton-academical            0           0
  11 │ 12473119  2024-08-17  ayr-united            hamilton-academical            3           2
  12 │ 12476936  2024-08-24  livingston            greenock-morton                1           1
  13 │ 12476937  2024-08-24  falkirk-fc            partick-thistle                2           1
  14 │ 12476938  2024-08-24  hamilton-academical   dunfermline-athletic           1           0
  15 │ 12476939  2024-08-24  ayr-united            raith-rovers                   2           0
  16 │ 12476941  2024-08-24  airdrieonians         queens-park-fc                 0           2
  17 │ 12476930  2024-08-31  raith-rovers          livingston                     0           1
  18 │ 12476932  2024-08-31  dunfermline-athletic  ayr-united                     1           1
  19 │ 12476933  2024-08-31  greenock-morton       falkirk-fc                     2           3
  20 │ 12476934  2024-08-31  partick-thistle       queens-park-fc                 3           0
  21 │ 12476935  2024-08-31  hamilton-academical   airdrieonians                  2           2
  22 │ 12476928  2024-09-13  dunfermline-athletic  raith-rovers                   2           0
  23 │ 12476929  2024-09-14  ayr-united            partick-thistle                1           1
  24 │ 12476931  2024-09-14  airdrieonians         falkirk-fc                     0           2
  25 │ 12476927  2024-09-14  livingston            hamilton-academical            3           0
  26 │ 12476926  2024-09-14  queens-park-fc        greenock-morton                1           0
  27 │ 12476921  2024-09-21  raith-rovers          hamilton-academical            3           3
  28 │ 12476922  2024-09-21  queens-park-fc        ayr-united                     1           1
  29 │ 12476923  2024-09-21  partick-thistle       dunfermline-athletic           1           0
  30 │ 12476924  2024-09-21  greenock-morton       airdrieonians                  2           0
  31 │ 12476916  2024-09-28  raith-rovers          falkirk-fc                     1           0
  32 │ 12476917  2024-09-28  hamilton-academical   partick-thistle                1           0
  33 │ 12476918  2024-09-28  livingston            airdrieonians                  2           1
  34 │ 12476919  2024-09-28  dunfermline-athletic  queens-park-fc                 1           2
  35 │ 12476920  2024-09-28  ayr-united            greenock-morton                1           0


--- improved version 
julia> feature_sets[1][1].data
Dict{Symbol, Any} with 13 entries:
  :matches_df       => 180×19 DataFrame…
  :n_teams          => 10
  :team_map         => Dict{String31, Int64}("raith-rovers"=>3, "queens-park-fc"=>10, "ayr-united"=>2, "falkirk-fc"=>9, "airdrieonians"=>4, "greenock-morto…
  :n_rounds         => 36
  :flat_home_ids    => [9, 6, 8, 7, 4, 10, 2, 1, 3, 5  …  6, 2, 3, 7, 1, 4, 8, 10, 9, 5]
  :flat_away_ids    => [10, 5, 1, 2, 3, 8, 4, 9, 6, 7  …  9, 10, 5, 8, 4, 2, 6, 3, 7, 1]
  :flat_home_goals  => [2, 0, 2, 0, 1, 1, 5, 0, 1, 0  …  2, 2, 1, 0, 0, 0, 0, 1, 3, 2]
  :flat_away_goals  => [1, 0, 0, 2, 0, 1, 0, 2, 0, 0  …  1, 2, 1, 3, 0, 1, 1, 5, 1, 0]
  :time_indices     => [1, 1, 1, 1, 1, 2, 2, 2, 2, 2  …  35, 35, 35, 35, 35, 36, 36, 36, 36, 36]
  :round_home_ids   => [[9, 6, 8, 7, 4], [10, 2, 1, 3, 5], [8, 9, 7, 2, 4], [3, 1, 5, 6, 7], [1, 2, 4, 8, 10], [3, 10, 6, 5, 9], [3, 7, 8, 1, 2], [4, 10, 9…
  :round_away_ids   => [[10, 5, 1, 2, 3], [8, 4, 9, 6, 7], [5, 6, 1, 3, 10], [8, 2, 9, 10, 4], [3, 6, 9, 7, 5], [7, 2, 1, 4, 8], [9, 6, 4, 10, 5], [1, 7, 2…
  :round_home_goals => SubArray{Int64, 1, Vector{Int64}, Tuple{Vector{Int64}}, false}[[2, 0, 2, 0, 1], [1, 5, 0, 1, 0], [1, 2, 1, 2, 0], [0, 1, 2, 3, 2], […
  :round_away_goals => SubArray{Int64, 1, Vector{Int64}, Tuple{Vector{Int64}}, false}[[1, 0, 0, 2, 0], [1, 0, 2, 0, 0], [1, 1, 0, 0, 2], [1, 1, 3, 0, 2], […


--- imporved version on match_week 
julia> feature_sets[1][1].data
Dict{Symbol, Any} with 13 entries:
  :matches_df       => 180×20 DataFrame…
  :n_teams          => 10
  :n_rounds         => 38
  :team_map         => Dict{String31, Int64}("raith-rovers"=>3, "queens-park-fc"=>10, "ayr-united"=>2, "falkirk-fc"=>9, "airdrieonians"=>4, "greenock-morto…
  :flat_home_ids    => [9, 6, 8, 7, 4, 10, 2, 1, 3, 5  …  6, 2, 3, 7, 1, 4, 8, 10, 9, 5]
  :flat_away_ids    => [10, 5, 1, 2, 3, 8, 4, 9, 6, 7  …  9, 10, 5, 8, 4, 2, 6, 3, 7, 1]
  :flat_away_goals  => [1, 0, 0, 2, 0, 1, 0, 2, 0, 0  …  1, 2, 1, 3, 0, 1, 1, 5, 1, 0]
  :flat_home_goals  => [2, 0, 2, 0, 1, 1, 5, 0, 1, 0  …  2, 2, 1, 0, 0, 0, 0, 1, 3, 2]
  :time_indices     => [1, 1, 1, 1, 1, 2, 2, 2, 2, 2  …  37, 37, 37, 37, 37, 38, 38, 38, 38, 38]
  :round_home_ids   => [[9, 6, 8, 7, 4], [10, 2, 1, 3, 5], [2], [8, 9, 7, 2, 4], [3, 1, 5, 6, 7], [1, 2, 4, 8, 10], [3, 10, 6, 5], [3, 7, 8, 1, 2], [4, 10,…
  :round_away_ids   => [[10, 5, 1, 2, 3], [8, 4, 9, 6, 7], [7], [5, 6, 1, 3, 10], [8, 2, 9, 10, 4], [3, 6, 9, 7, 5], [7, 2, 1, 4], [9, 6, 4, 10, 5], [1, 7,…
  :round_home_goals => SubArray{Int64, 1, Vector{Int64}, Tuple{Vector{Int64}}, false}[[2, 0, 2, 0, 1], [1, 5, 0, 1, 0], [3], [1, 2, 1, 2, 0], [0, 1, 2, 3, …
  :round_away_goals => SubArray{Int64, 1, Vector{Int64}, Tuple{Vector{Int64}}, false}[[1, 0, 0, 2, 0], [1, 0, 2, 0, 0], [2], [1, 1, 0, 0, 2], [1, 1, 3, 0,


julia> feature_sets[1][1].data[:time_indices]
180-element Vector{Int64}:
  1
  1
  1
  1
  1
  2
  2
  2
  2
  2
  3
  4
  4
  4
  4
  4
  5
  5
  5
  5
  5
  6
  6
  6
  6
  6
  7
  7
  7
  7
  8
  8
  8
  8
  8


# --- improved version for tournament_id = 54 - mid week games more , cause double weeks 
# should be ok to leave 

julia> feature_sets[1][1].data
Dict{Symbol, Any} with 13 entries:
  :round_away_ids   => [[9, 8, 7, 11, 10], [12, 1, 3, 2, 4, 6, 5], [8, 7, 6, 4, 1, 11], [2, 12, 5, 10, 3, 9], [11, 7, 6, 1, 2, 9], [1, 6], [4, 5, 8, 12, 3,…
  :matches_df       => 198×20 DataFrame…
  :flat_away_ids    => [9, 8, 7, 11, 10, 12, 1, 3, 2, 4  …  10, 5, 4, 3, 6, 8, 1, 11, 7, 9]
  :time_indices     => [1, 1, 1, 1, 1, 2, 2, 2, 2, 2  …  30, 30, 30, 30, 31, 31, 31, 31, 31, 31]
  :flat_home_goals  => [0, 0, 2, 4, 3, 1, 3, 1, 2, 0  …  0, 2, 1, 0, 1, 3, 0, 5, 4, 2]
  :round_home_ids   => [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11, 12], [9, 10, 3, 5, 2, 12], [6, 8, 7, 11, 1, 4], [5, 8, 10, 4, 12, 3], [5, 8], [6, 2, 1, 7, 11…
  :flat_home_ids    => [1, 2, 3, 4, 5, 6, 7, 8, 9, 10  …  9, 7, 6, 1, 3, 5, 2, 4, 10, 12]
  :team_map         => Dict{String31, Int64}("motherwell"=>2, "st-johnstone"=>6, "hibernian"=>10, "dundee-united"=>3, "st-mirren"=>5, "heart-of-midlothian"…
  :n_rounds         => 31
  :flat_away_goals  => [0, 0, 2, 0, 0, 2, 1, 1, 1, 2  …  2, 0, 0, 1, 0, 2, 0, 1, 0, 2]
  :n_teams          => 12
  :round_home_goals => SubArray{Int64, 1, Vector{Int64}, Tuple{Vector{Int64}}, false}[[0, 0, 2, 4, 3], [1, 3, 1, 2, 0, 0, 3], [6, 2, 2, 0, 3, 2], [1, 0, 2,…
  :round_away_goals => SubArray{Int64, 1, Vector{Int64}, Tuple{Vector{Int64}}, false}[[0, 0, 2, 0, 0], [2, 1, 1, 1, 2, 3, 1], [0, 2, 0, 3, 1, 0], [2, 1, 2,


julia> feature_sets[1][1].data[:time_indices]
198-element Vector{Int64}:
  1
  1
  1
  1
  1
  2
  2
  2
  2
  2
  2
  2
  3
  3
  3
  3
  3
  3
  4
  4
  4
  4
  4
  4
  5
  5
  5
  5
  5
  5
  6
  6
  7
  7
  7
  7
  7
  7

"""


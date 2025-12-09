"""

To update the feature sets to handle the dynamics models better. 

"""


"""
addressing the time window and dynamics of feature sets 

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
           :tournament_id => ByRow(isequal(55)),
                                      :season => ByRow(isequal("24/25")))),
    data_store.odds,
    data_store.incidents
)


model = BayesianFootball.Models.PreGame.AR1Poisson()
vocabulary = BayesianFootball.Features.create_vocabulary(ds, model) 

# here want to start the expanding window cv ( 1 -38) so 38 - 35 = 3 +1 ( since we have zero ) 4
ds.matches.split_col = max.(0, ds.matches.match_week .- 35);
ds.matches.split_col ;

# CONFIG: One place to rule them all
splitter_config = BayesianFootball.Data.ExpandingWindowCV(
    train_seasons = [], 
    test_seasons = ["24/25"], 
    window_col = :split_col,      # 1. WINDOWING: Split chunks based on this (0, 1, 2...)
    method = :sequential,
    dynamics_col = :match_week      # 2. DYNAMICS: Inside the chunk, evolve time based on this
)

data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)

# API is now clean: no extra kwargs needed
feature_sets = BayesianFootball.Features.create_features(
    data_splits, 
    vocabulary, 
    model, 
    splitter_config 
)


split_col_index = 1
feature_sets[split_col_index][1].data
unique(feature_sets[split_col_index][1].data[:time_indices])

# static splits 


splitter_config = BayesianFootball.Data.StaticSplit(
    train_seasons = ["24/25"], 
    window_col = :match_week
)

data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)

# API is now clean: no extra kwargs needed
feature_sets = BayesianFootball.Features.create_features(
    data_splits, 
    vocabulary, 
    model, 
    splitter_config 
)

split_col_index = 1
feature_sets[split_col_index][1].data
unique(feature_sets[split_col_index][1].data[:time_indices])


"""
# ----

julia> split_col_index = 1
1

julia> feature_sets[split_col_index][1].data
Dict{Symbol, Any} with 13 entries:
  :round_away_ids   => [[6, 10, 8, 7, 9], [3, 5, 1, 2, 4], [4], [10, 2, 8, 9, 6], [3, 7, 1, 6, 5], [9, 2, 1, 4, 10], [4, 7, 8, 5], [1…
  :matches_df       => 165×21 DataFrame…
  :flat_away_ids    => [6, 10, 8, 7, 9, 3, 5, 1, 2, 4  …  6, 10, 4, 2, 7, 1, 8, 5, 4, 3]
  :time_indices     => [1, 1, 1, 1, 1, 2, 2, 2, 2, 2  …  34, 34, 34, 34, 34, 35, 35, 35, 35, 35]
  :flat_home_goals  => [2, 0, 2, 0, 1, 1, 5, 0, 1, 0  …  3, 5, 0, 2, 1, 1, 0, 1, 1, 1]
  :round_home_ids   => [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [7], [3, 1, 4, 7, 5], [9, 8, 10, 2, 4], [8, 7, 5, 3, 6], [9, 6, 2, 10], [9…
  :flat_home_ids    => [1, 2, 3, 4, 5, 6, 7, 8, 9, 10  …  3, 1, 8, 5, 9, 7, 6, 9, 2, 10]
  :team_map         => Dict{String31, Int64}("queens-park-fc"=>6, "raith-rovers"=>9, "ayr-united"=>7, "falkirk-fc"=>1, "hamilton-acad…
  :n_rounds         => 35
  :flat_away_goals  => [1, 0, 0, 2, 0, 1, 0, 2, 0, 0  …  0, 0, 1, 1, 0, 1, 1, 1, 2, 2]
  :n_teams          => 10
  :round_home_goals => SubArray{Int64, 1, Vector{Int64}, Tuple{Vector{Int64}}, false}[[2, 0, 2, 0, 1], [1, 5, 0, 1, 0], [3], [1, 2, 1…
  :round_away_goals => SubArray{Int64, 1, Vector{Int64}, Tuple{Vector{Int64}}, false}[[1, 0, 0, 2, 0], [1, 0, 2, 0, 0], [2], [1, 1, 0…

julia> unique(feature_sets[split_col_index][1].data[:time_indices])
35-element Vector{Int64}:
  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35


# ------

julia> split_col_index = 2
2

julia> feature_sets[split_col_index][1].data
Dict{Symbol, Any} with 13 entries:
  :round_away_ids   => [[6, 10, 8, 7, 9], [3, 5, 1, 2, 4], [4], [10, 2, 8, 9, 6], [3, 7, 1, 6, 5], [9, 2, 1, 4, 10], [4, 7, 8, 5], [1…
  :matches_df       => 170×21 DataFrame…
  :flat_away_ids    => [6, 10, 8, 7, 9, 3, 5, 1, 2, 4  …  1, 8, 5, 4, 3, 7, 10, 6, 2, 9]
  :time_indices     => [1, 1, 1, 1, 1, 2, 2, 2, 2, 2  …  35, 35, 35, 35, 35, 36, 36, 36, 36, 36]
  :flat_home_goals  => [2, 0, 2, 0, 1, 1, 5, 0, 1, 0  …  1, 0, 1, 1, 1, 5, 0, 0, 0, 1]
  :round_home_ids   => [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [7], [3, 1, 4, 7, 5], [9, 8, 10, 2, 4], [8, 7, 5, 3, 6], [9, 6, 2, 10], [9…
  :flat_home_ids    => [1, 2, 3, 4, 5, 6, 7, 8, 9, 10  …  7, 6, 9, 2, 10, 3, 5, 4, 8, 1]
  :team_map         => Dict{String31, Int64}("queens-park-fc"=>6, "raith-rovers"=>9, "ayr-united"=>7, "falkirk-fc"=>1, "hamilton-acad…
  :n_rounds         => 36
  :flat_away_goals  => [1, 0, 0, 2, 0, 1, 0, 2, 0, 0  …  1, 1, 1, 2, 2, 0, 1, 0, 0, 3]
  :n_teams          => 10
  :round_home_goals => SubArray{Int64, 1, Vector{Int64}, Tuple{Vector{Int64}}, false}[[2, 0, 2, 0, 1], [1, 5, 0, 1, 0], [3], [1, 2, 1…
  :round_away_goals => SubArray{Int64, 1, Vector{Int64}, Tuple{Vector{Int64}}, false}[[1, 0, 0, 2, 0], [1, 0, 2, 0, 0], [2], [1, 1, 0…

julia> unique(feature_sets[split_col_index][1].data[:time_indices])
36-element Vector{Int64}:
  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35
 36


# ------

julia> split_col_index = 3
3

julia> feature_sets[split_col_index][1].data
Dict{Symbol, Any} with 13 entries:
  :round_away_ids   => [[6, 10, 8, 7, 9], [3, 5, 1, 2, 4], [4], [10, 2, 8, 9, 6], [3, 7, 1, 6, 5], [9, 2, 1, 4, 10], [4, 7, 8, 5], [1…
  :matches_df       => 175×21 DataFrame…
  :flat_away_ids    => [6, 10, 8, 7, 9, 3, 5, 1, 2, 4  …  7, 10, 6, 2, 9, 1, 6, 10, 3, 5]
  :time_indices     => [1, 1, 1, 1, 1, 2, 2, 2, 2, 2  …  36, 36, 36, 36, 36, 37, 37, 37, 37, 37]
  :flat_home_goals  => [2, 0, 2, 0, 1, 1, 5, 0, 1, 0  …  5, 0, 0, 0, 1, 2, 2, 1, 0, 0]
  :round_home_ids   => [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [7], [3, 1, 4, 7, 5], [9, 8, 10, 2, 4], [8, 7, 5, 3, 6], [9, 6, 2, 10], [9…
  :flat_home_ids    => [1, 2, 3, 4, 5, 6, 7, 8, 9, 10  …  3, 5, 4, 8, 1, 2, 7, 9, 4, 8]
  :team_map         => Dict{String31, Int64}("queens-park-fc"=>6, "raith-rovers"=>9, "ayr-united"=>7, "falkirk-fc"=>1, "hamilton-acad…
  :n_rounds         => 37
  :flat_away_goals  => [1, 0, 0, 2, 0, 1, 0, 2, 0, 0  …  0, 1, 0, 0, 3, 1, 2, 1, 3, 0]
  :n_teams          => 10
  :round_home_goals => SubArray{Int64, 1, Vector{Int64}, Tuple{Vector{Int64}}, false}[[2, 0, 2, 0, 1], [1, 5, 0, 1, 0], [3], [1, 2, 1…
  :round_away_goals => SubArray{Int64, 1, Vector{Int64}, Tuple{Vector{Int64}}, false}[[1, 0, 0, 2, 0], [1, 0, 2, 0, 0], [2], [1, 1, 0…

julia> unique(feature_sets[split_col_index][1].data[:time_indices])
37-element Vector{Int64}:
  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35
 36
 37

"""



"""
feature_sets for expanding window cv 

 - want the column for the match_week intervals, and column to create the feature cv splits 
 - different, so we can do a split mid way through the seasons etc .. 

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


# here want to start the expanding window cv ( 1 -38) so 38 - 35 = 3 +1 ( since we have zero ) 4
ds.matches.split_col = max.(0, ds.matches.match_week .- 35);
ds.matches.split_col 

model = BayesianFootball.Models.PreGame.AR1Poisson()
vocabulary = BayesianFootball.Features.create_vocabulary(ds, model) 


splitter_config = BayesianFootball.Data.ExpandingWindowCV([], ["24/25"], :split_col, :sequential) #
# splitter_config = BayesianFootball.Data.StaticSplit(
#     train_seasons = ["24/25"], 
#     round_col = :match_week
# )

data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)
feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model, splitter_config)

split_col_index = 3
feature_sets[split_col_index][1].data
unique(feature_sets[split_col_index][1].data[:time_indices])


"""
julia> data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)
Creating TimeSeriesSplits (Expanding Window)...
4-element Vector{Tuple{SubDataFrame, String}}:
 (165×21 SubDataFrame
                       16 columns and 150 rows omitted, "24/25/Round_0")

 (170×21 SubDataFrame
                       16 columns and 155 rows omitted, "24/25/Round_1")

 (175×21 SubDataFrame
                       16 columns and 160 rows omitted, "24/25/Round_2")

 (180×21 SubDataFrame
                       16 columns and 165 rows omitted, "24/25/Round_3")


julia> feature_sets[1][1].data
Dict{Symbol, Any} with 13 entries:
  :round_away_ids   => [[6, 10, 8, 7, 9, 3, 5, 1, 2, 4  …  6, 10, 4, 2, 7, 1, 8, 5, 4, 3]]
  :matches_df       => 165×21 DataFrame…
  :flat_away_ids    => [6, 10, 8, 7, 9, 3, 5, 1, 2, 4  …  6, 10, 4, 2, 7, 1, 8, 5, 4, 3]
  :time_indices     => [1, 1, 1, 1, 1, 1, 1, 1, 1, 1  …  1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  :flat_home_goals  => [2, 0, 2, 0, 1, 1, 5, 0, 1, 0  …  3, 5, 0, 2, 1, 1, 0, 1, 1, 1]
  :round_home_ids   => [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10  …  3, 1, 8, 5, 9, 7, 6, 9, 2, 10]]
  :flat_home_ids    => [1, 2, 3, 4, 5, 6, 7, 8, 9, 10  …  3, 1, 8, 5, 9, 7, 6, 9, 2, 10]
  :team_map         => Dict{String31, Int64}("queens-park-fc"=>6, "raith-rovers"=>9, "ayr-united"=>7, "falkirk-fc"=>1, "hamilton-academical"=>4, "partick-t…
  :n_rounds         => 1
  :flat_away_goals  => [1, 0, 0, 2, 0, 1, 0, 2, 0, 0  …  0, 0, 1, 1, 0, 1, 1, 1, 2, 2]
  :n_teams          => 10
  :round_home_goals => SubArray{Int64, 1, Vector{Int64}, Tuple{Vector{Int64}}, false}[[2, 0, 2, 0, 1, 1, 5, 0, 1, 0  …  3, 5, 0, 2, 1, 1, 0, 1, 1, 1]]
  :round_away_goals => SubArray{Int64, 1, Vector{Int64}, Tuple{Vector{Int64}}, false}[[1, 0, 0, 2, 0, 1, 0, 2, 0, 0  …  0, 0, 1, 1, 0, 1, 1, 1, 2, 2]]


julia> unique(feature_sets[1][1].data[:time_indices])
1-element Vector{Int64}:
 1

----
julia> split_col_index = 2
2
julia> feature_sets[split_col_index][1].data
Dict{Symbol, Any} with 13 entries:
  :round_away_ids   => [[6, 10, 8, 7, 9, 3, 5, 1, 2, 4  …  6, 10, 4, 2, 7, 1, 8, 5, 4, 3], [7, 10, 6, 2, 9]]
  :matches_df       => 170×21 DataFrame…
  :flat_away_ids    => [6, 10, 8, 7, 9, 3, 5, 1, 2, 4  …  1, 8, 5, 4, 3, 7, 10, 6, 2, 9]
  :time_indices     => [1, 1, 1, 1, 1, 1, 1, 1, 1, 1  …  1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
  :flat_home_goals  => [2, 0, 2, 0, 1, 1, 5, 0, 1, 0  …  1, 0, 1, 1, 1, 5, 0, 0, 0, 1]
  :round_home_ids   => [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10  …  3, 1, 8, 5, 9, 7, 6, 9, 2, 10], [3, 5, 4, 8, 1]]
  :flat_home_ids    => [1, 2, 3, 4, 5, 6, 7, 8, 9, 10  …  7, 6, 9, 2, 10, 3, 5, 4, 8, 1]
  :team_map         => Dict{String31, Int64}("queens-park-fc"=>6, "raith-rovers"=>9, "ayr-united"=>7, "falkirk-fc"=>1, "hamilton-academical"=>4, "partick-t…
  :n_rounds         => 2
  :flat_away_goals  => [1, 0, 0, 2, 0, 1, 0, 2, 0, 0  …  1, 1, 1, 2, 2, 0, 1, 0, 0, 3]
  :n_teams          => 10
  :round_home_goals => SubArray{Int64, 1, Vector{Int64}, Tuple{Vector{Int64}}, false}[[2, 0, 2, 0, 1, 1, 5, 0, 1, 0  …  3, 5, 0, 2, 1, 1, 0, 1, 1, 1], [5, …
  :round_away_goals => SubArray{Int64, 1, Vector{Int64}, Tuple{Vector{Int64}}, false}[[1, 0, 0, 2, 0, 1, 0, 2, 0, 0  …  0, 0, 1, 1, 0, 1, 1, 1, 2, 2], [0, …

julia> unique(feature_sets[split_col_index][1].data[:time_indices])
2-element Vector{Int64}:
 1
 2

---
julia> split_col_index = 3
3

julia> feature_sets[split_col_index][1].data
Dict{Symbol, Any} with 13 entries:
  :round_away_ids   => [[6, 10, 8, 7, 9, 3, 5, 1, 2, 4  …  6, 10, 4, 2, 7, 1, 8, 5, 4, 3], [7, 10, 6, 2, 9], [1, 6, 10, 3, 5]]
  :matches_df       => 175×21 DataFrame…
  :flat_away_ids    => [6, 10, 8, 7, 9, 3, 5, 1, 2, 4  …  7, 10, 6, 2, 9, 1, 6, 10, 3, 5]
  :time_indices     => [1, 1, 1, 1, 1, 1, 1, 1, 1, 1  …  2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
  :flat_home_goals  => [2, 0, 2, 0, 1, 1, 5, 0, 1, 0  …  5, 0, 0, 0, 1, 2, 2, 1, 0, 0]
  :round_home_ids   => [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10  …  3, 1, 8, 5, 9, 7, 6, 9, 2, 10], [3, 5, 4, 8, 1], [2, 7, 9, 4, 8]]
  :flat_home_ids    => [1, 2, 3, 4, 5, 6, 7, 8, 9, 10  …  3, 5, 4, 8, 1, 2, 7, 9, 4, 8]
  :team_map         => Dict{String31, Int64}("queens-park-fc"=>6, "raith-rovers"=>9, "ayr-united"=>7, "falkirk-fc"=>1, "hamilton-academical"=>4, "partick-t…
  :n_rounds         => 3
  :flat_away_goals  => [1, 0, 0, 2, 0, 1, 0, 2, 0, 0  …  0, 1, 0, 0, 3, 1, 2, 1, 3, 0]
  :n_teams          => 10
  :round_home_goals => SubArray{Int64, 1, Vector{Int64}, Tuple{Vector{Int64}}, false}[[2, 0, 2, 0, 1, 1, 5, 0, 1, 0  …  3, 5, 0, 2, 1, 1, 0, 1, 1, 1], [5, …
  :round_away_goals => SubArray{Int64, 1, Vector{Int64}, Tuple{Vector{Int64}}, false}[[1, 0, 0, 2, 0, 1, 0, 2, 0, 0  …  0, 0, 1, 1, 0, 1, 1, 1, 2, 2], [0, …

julia> unique(feature_sets[split_col_index][1].data[:time_indices])
3-element Vector{Int64}:
 1
 2
 3

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
  ⋮
 26
 26
 27
 27
 27
 27
 27
 27
 27
 27
 27
 27
 27
 27
 28
 28
 28
 28
 28
 28
 29
 29
 29
 29
 29
 29
 30
 30
 30
 30
 30
 30
 31
 31
 31
 31
 31
 31

"""


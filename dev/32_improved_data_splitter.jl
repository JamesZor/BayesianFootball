# dev/32_improved_data_splitter.jl

"""

We define season data as Sⁱ where i defines the years/ season index so that
S²⁰ imples that we have the seaosn "20/21" . 

futher more, we add to this notation with S^{i}_{n:m} or S^{i}_{n} with n,m postive intergers 
that denote the match weeks ( or the time dynamics of the season ) with n being the start and m being the end 
of the indexing so: 
s^{20} = { S^{20}_{1} , S^{20}_{2} , S^{20}_{3} , ..., S^{20}_{end} } = S^{20}_{1:end} 
with S^{20}_{1} being a collection of features of the games played in that week. 
So just looking at the 20/21 season. 

Hence we want to be able to construct a set of sets in order to conduct expanding window cross validations 
/ walk forward test on a season, to simulate the data avabiles in real time series manner, with the n 
indicating the warm up number of time dynamics to be considers before the first predictions p_1 which is 
index at the n+1. Then it follows that m is the last training batch need either for end -1 so we can predict 
p_end as we the last week of the season is not need since the there are no game after that - only the next season, or 
at the end in the case we want to make a predict at m+1. 
Abusing some set notation we have 

S^{20}_{n:end} = { { S^{20}_{1} }, 
                  { S^{20}_{1} + S^{20}_{2} },
                  { S^{20}_{1} + S^{20}_{2} + S^{20}_{3} }, 
 ..., 
                  { S^{20}_{1} + S^{20}_{2} + ... +  S^{20}_{end} }
                }
noting that the operator "+" is more of a contacation of the data sets. 

S^{20:21}_{n:end} = { { S^{20}_{1} }, 
                  { S^{20}_{1} + S^{20}_{2} },
                  { S^{20}_{1} + S^{20}_{2} + S^{20}_{3} }, 
 ..., 
                  { S^{20}_{1} + S^{20}_{2} + ... +  S^{20}_{end} }

                  { S^{21}_{1} }, 
                  { S^{21}_{1} + S^{21}_{2} },
                  { S^{21}_{1} + S^{21}_{2} + S^{21}_{3} }, 
 ..., 
                  { S^{21}_{1} + S^{21}_{2} + ... +  S^{21}_{end} }
                }

In the data split we want to indicate if the training / data split includes historical years, 
as in if we include the previous season in the model, i.e we split 20/21 seaosn based on the time 
dynamics but include the 19/20 seaosns for the model, we can denotes this t which is T-t for the seaosn 
to include,  S^{21:t:21}_{n:end} or  S^{20:t:21}_{n:end} 

normal : 
S^{20:0:20}_{n:end} = { { S^{20}_{1} }, 
                  { S^{20}_{1} + S^{20}_{2} },
                  { S^{20}_{1} + S^{20}_{2} + S^{20}_{3} }, 
 ..., 
                  { S^{20}_{1} + S^{20}_{2} + ... +  S^{20}_{end} } 
}

the case to include one year previous t=1 

S^{21:1:21}_{n:end} = { { S^{20} + S^{21}_{1} }, 
                  { S^{20} + S^{21}_{1} + S^{21}_{2} },
                  { S^{21}_{1} + S^{21}_{2} + S^{21}_{3} }, 
 ..., 
                  { S^{20} + S^{21}_{1} + S^{21}_{2} + ... +  S^{21}_{end} } 
}

the case for t=1 but repeat for the seaosn 20/21 and 21/22: 


S^{20:1:21}_{n:end} = 
                  { { S^{19} + S^{20}_{1} }, 
                  { S^{19} + S^{20}_{1} + S^{20}_{2} },
                  { S^{20}_{1} + S^{20}_{2} + S^{20}_{3} }, 
 ..., 
                  { S^{19} + S^{20}_{1} + S^{20}_{2} + ... +  S^{20}_{end} },

                  { { S^{20} + S^{21}_{1} }, 
                  { S^{20} + S^{21}_{1} + S^{21}_{2} },
                  { S^{21}_{1} + S^{21}_{2} + S^{21}_{3} }, 
 ..., 
                  { S^{20} + S^{21}_{1} + S^{21}_{2} + ... +  S^{21}_{end} } 
}


The aim is to allow use to create this kind of data splits for the training process in our models. 

Following the data.create_data_splits api function structure 
we have a mapping from create_data_splits: D × C -> D' 
were D is the data_store, and C is a config , D' is like a powerset of the D, / or a view with will be like the S. 

here for the data_split config 
data_split_config :
  tournament_id = vector /list  -> [1] for just tournament_id =1 or [1],[2] to repeat the process for tournament_id 
                1 and 2 seperately, or [1,2] to do 1,2 at the same time. 

  dynamics_col = symbol to indicate which dataframe column to use for the time dynamics so moslty either month or week 

  warnup_period_dynamics = n -> where to begin as doing an ar1 process on 1 week wont fit well so we allow the seaosn to play be for training 

  seasons => list of seasons to be split up examples: ["20/21"], ["20/21", "21/22"]  etc repeats for seaosn and tournament_id 

  end_daynamics: m -> if we need to run to the need to make a predcit m+1 or stop a week before the last week, as no games after the last week, end or end -1 

  season_hist: i -> the number of past seaosn to include, examples: 0 imples that it will be just "20/21" or the seaosns indicated 
                    in the seaons parameter, if 1 then use the last season so ["20/21"] we need to include ["19/20"] S^{20:1:20}_{n:end}, 
                    noting we need to check/ ensure we dont go back to far as currelty we have the data for 20/21 til 24/25 amd some of 25/26. 
                    for the time being so need ot check in data store and warn users that it cant be done if i is too large. 


D' is similar to the out now, a vector of tuples with index 1 is the S^{20}_{n} ( the training data split) 
and the [2] is currelty a string, however this string should be a split_meta_data struct, which inherts from abstract meta struct 
as we need one for data split and then the feature_split - tho they will be the same i think. 
the point of the split_meta_data is to contain relavent information regarding the split for the features split process which 
which will mostly be the split_meta_data information, and which is need in the predictions part of the process 
so we can ensure we are predicting the correct season, and time step 
so we need to carry this information 
split_meta_data: 
    -tournament_ids 
    -current_seaosn_fold:
    -current_time_step:


in the mapping create_data_splits, here it would be nice to filter / subset the need data from the data_store 
struct, as it will help reduce boiler code when writitng experients. 



"""

using BayesianFootball

using DataFrames
using Statistics
using ThreadPinning
pinthreads(:cores)




data_store = BayesianFootball.Data.load_default_datastore()

cv_config = BayesianFootball.Data.CVConfig(
    tournament_ids = [55],
    target_seasons = ["24/25"],
    history_seasons = 0, # Will auto-include "23/24" if available
    dynamics_col = :match_week,
    warmup_period = 5
)


tournament_id = 55
ds = BayesianFootball.Data.DataStore( 
    Data.add_match_week_column(subset( data_store.matches, 
           :tournament_id => ByRow(isequal(tournament_id)),
                                      :season => ByRow(isequal("24/25")))),
    data_store.odds,
    data_store.incidents
)


cv_config = BayesianFootball.Data.CVConfig(
    tournament_ids = [55],
    target_seasons = ["24/25"],
    history_seasons = 0, # Will auto-include "23/24" if available
    dynamics_col = :match_week,
    warmup_period = 5
)


splits = Data.create_data_splits(ds, cv_config)


"""

julia> splits = Data.create_data_splits(ds, cv_config)
34-element Vector{Tuple{SubDataFrame, BayesianFootball.Data.SplitMetaData}}:
 (21×20 SubDataFrame
 Row │ tournament_id  season_id  season   match_id  tournament_slug  home_team ⋯
     │ Int64          Int64      String7  Int64     String15         String31  ⋯
─────┼──────────────────────────────────────────────────────────────────────────
   1 │            55      62411  24/25    12477141  championship     falkirk-f ⋯
   2 │            55      62411  24/25    12477136  championship     partick-t
   3 │            55      62411  24/25    12477137  championship     livingsto
   4 │            55      62411  24/25    12477138  championship     hamilton-
   5 │            55      62411  24/25    12477142  championship     airdrieon ⋯
   6 │            55      62411  24/25    12476942  championship     queens-pa
   7 │            55      62411  24/25    12476945  championship     ayr-unite
   8 │            55      62411  24/25    12476940  championship     dunfermli
  ⋮  │       ⋮            ⋮         ⋮        ⋮             ⋮                   ⋱
  15 │            55      62411  24/25    12476939  championship     ayr-unite ⋯
  16 │            55      62411  24/25    12476941  championship     airdrieon
  17 │            55      62411  24/25    12476930  championship     raith-rov
  18 │            55      62411  24/25    12476932  championship     dunfermli
  19 │            55      62411  24/25    12476933  championship     greenock- ⋯
  20 │            55      62411  24/25    12476934  championship     partick-t
  21 │            55      62411  24/25    12476935  championship     hamilton-
                                                   15 columns and 6 rows omitted, Split(Tourn: 55, Season: 24/25, Week: 5, Hist: 0))
 (26×20 SubDataFrame
 Row │ tournament_id  season_id  season   match_id  tournament_slug  home_team ⋯
     │ Int64          Int64      String7  Int64     String15         String31  ⋯
─────┼──────────────────────────────────────────────────────────────────────────
   1 │            55      62411  24/25    12477141  championship     falkirk-f ⋯
   2 │            55      62411  24/25    12477136  championship     partick-t
   3 │            55      62411  24/25    12477137  championship     livingsto
   4 │            55      62411  24/25    12477138  championship     hamilton-
   5 │            55      62411  24/25    12477142  championship     airdrieon ⋯
   6 │            55      62411  24/25    12476942  championship     queens-pa
   7 │            55      62411  24/25    12476945  championship     ayr-unite
   8 │            55      62411  24/25    12476940  championship     dunfermli
  ⋮  │       ⋮            ⋮         ⋮        ⋮             ⋮                   ⋱
  20 │            55      62411  24/25    12476934  championship     partick-t ⋯
  21 │            55      62411  24/25    12476935  championship     hamilton-
  22 │            55      62411  24/25    12476928  championship     dunfermli
  23 │            55      62411  24/25    12476929  championship     ayr-unite
  24 │            55      62411  24/25    12476931  championship     airdrieon ⋯
  25 │            55      62411  24/25    12476927  championship     livingsto
  26 │            55      62411  24/25    12476926  championship     queens-pa
                                                  15 columns and 11 rows omitted, Split(Tourn: 55, Season: 24/25, Week: 6, Hist: 0))

julia> splits[1]
(21×20 SubDataFrame
 Row │ tournament_id  season_id  season   match_id  tournament_slug  home_team             away_team             home_score  away_sc ⋯
     │ Int64          Int64      String7  Int64     String15         String31              String31              Int64?      Int64?  ⋯
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │            55      62411  24/25    12477141  championship     falkirk-fc            queens-park-fc                 2          ⋯
   2 │            55      62411  24/25    12477136  championship     partick-thistle       greenock-morton                0
   3 │            55      62411  24/25    12477137  championship     livingston            dunfermline-athletic           2
   4 │            55      62411  24/25    12477138  championship     hamilton-academical   ayr-united                     0
   5 │            55      62411  24/25    12477142  championship     airdrieonians         raith-rovers                   1          ⋯
   6 │            55      62411  24/25    12476942  championship     queens-park-fc        livingston                     1
   7 │            55      62411  24/25    12476945  championship     ayr-united            airdrieonians                  5
   8 │            55      62411  24/25    12476940  championship     dunfermline-athletic  falkirk-fc                     0
   9 │            55      62411  24/25    12476943  championship     raith-rovers          partick-thistle                1          ⋯
  10 │            55      62411  24/25    12476944  championship     greenock-morton       hamilton-academical            0
  11 │            55      62411  24/25    12473119  championship     ayr-united            hamilton-academical            3
  12 │            55      62411  24/25    12476936  championship     livingston            greenock-morton                1
  13 │            55      62411  24/25    12476937  championship     falkirk-fc            partick-thistle                2          ⋯
  14 │            55      62411  24/25    12476938  championship     hamilton-academical   dunfermline-athletic           1
  15 │            55      62411  24/25    12476939  championship     ayr-united            raith-rovers                   2
  16 │            55      62411  24/25    12476941  championship     airdrieonians         queens-park-fc                 0
  17 │            55      62411  24/25    12476930  championship     raith-rovers          livingston                     0          ⋯
  18 │            55      62411  24/25    12476932  championship     dunfermline-athletic  ayr-united                     1
  19 │            55      62411  24/25    12476933  championship     greenock-morton       falkirk-fc                     2
  20 │            55      62411  24/25    12476934  championship     partick-thistle       queens-park-fc                 3
  21 │            55      62411  24/25    12476935  championship     hamilton-academical   airdrieonians                  2          ⋯
                                                                                                                    12 columns omitted, Split(Tourn: 55, Season: 24/25, Week: 5, Hist: 0))


"""
using DataFrames, Dates

"""
    sunday_of_week(dt::Date)

Returns the date of the Sunday following the given date (or the date itself if it is Sunday).
Used to group matches occurring in the same week (Mon-Sun).
"""
function sunday_of_week(dt::Date)::Date
    day_num = dayofweek(dt)
    return dt + Day(7 - day_num)
end

"""
    add_match_week_column(matches_df::AbstractDataFrame)

Adds a ':match_week' column (Int) that resets for every season and tournament.
The first week of matches in a season becomes Week 1, the next Week 2, etc.

Groups by: [:tournament_id, :season]
"""
function add_match_week_column(matches_df::AbstractDataFrame)::DataFrame
    df = copy(matches_df) # Work on a copy to avoid mutating the original
    
    # 1. Ensure global sort order first (Tournament -> Season -> Date)
    # This ensures that when we group, the data is relatively ordered, 
    # though the transform logic below explicitly handles date sorting too.
    sort!(df, [:tournament_id, :season, :match_date])

    # 2. Define the per-season logic
    # We take the vector of dates for a specific season, map them to Week Ending Sundays,
    # and then index those Sundays 1..N
    transform!(groupby(df, [:tournament_id, :season]), :match_date => (dates -> begin
        # A. Map distinct dates to their "Week Ending Sunday"
        #    (Matches Mon-Sun will share the same sunday_date)
        week_dates = sunday_of_week.(dates)
        
        # B. Find the unique weeks and sort them chronologically
        unique_weeks = sort(unique(week_dates))
        
        # C. Create a map: SundayDate -> Index (1, 2, 3...)
        week_map = Dict(w => i for (i, w) in enumerate(unique_weeks))
        
        # D. Map the original dates row-by-row to their Week Index
        return [week_map[w] for w in week_dates]
    end) => :match_week)

    return df
end

ds = BayesianFootball.Data.DataStore( 
    Data.add_match_week_column(data_store.matches),
    data_store.odds,
    data_store.incidents
)


unique(ds.matches.match_week)

cv_config = BayesianFootball.Data.CVConfig(
    tournament_ids = [55],
    target_seasons = ["24/25"],
    history_seasons = 0, # Will auto-include "23/24" if available
    dynamics_col = :match_week,
    warmup_period = 35,
    # stop_early = true  # Splits go 1..5, 1..6, ..., 1..37
    stop_early = false # Splits go 1..5, ..., 1..38 (The default)
)


splits = Data.create_data_splits(ds, cv_config)

"""

cv_config = BayesianFootball.Data.CVConfig(
    tournament_ids = [55],
    target_seasons = ["23/24","24/25"],
    history_seasons = 0, # Will auto-include "23/24" if available
    dynamics_col = :match_week,
    warmup_period = 35
)


splits = Data.create_data_splits(ds, cv_config)

julia> splits = Data.create_data_splits(ds, cv_config)
6-element Vector{Tuple{SubDataFrame, BayesianFootball.Data.SplitMetaData}}:
 (175×20 SubDataFrame
 Row │ tournament_id  season_id  season   match_id  tournament_slug  home_team ⋯
     │ Int64          Int64      String7  Int64     String15         String31  ⋯
─────┼──────────────────────────────────────────────────────────────────────────
   1 │            55      52606  23/24    11395859  championship     arbroath  ⋯
   2 │            55      52606  23/24    11395855  championship     partick-t
   3 │            55      52606  23/24    11395856  championship     greenock-
   4 │            55      52606  23/24    11395857  championship     inverness
   5 │            55      52606  23/24    11395858  championship     dunfermli ⋯
   6 │            55      52606  23/24    11395850  championship     raith-rov
   7 │            55      52606  23/24    11395851  championship     queens-pa
   8 │            55      52606  23/24    11395852  championship     dundee-un
  ⋮  │       ⋮            ⋮         ⋮        ⋮             ⋮                   ⋱
 169 │            55      52606  23/24    11395681  championship     dunfermli ⋯
 170 │            55      52606  23/24    11395682  championship     arbroath
 171 │            55      52606  23/24    11395677  championship     airdrieon
 172 │            55      52606  23/24    11395673  championship     partick-t
 173 │            55      52606  23/24    11395674  championship     dunfermli ⋯
 174 │            55      52606  23/24    11395675  championship     arbroath
 175 │            55      52606  23/24    11395676  championship     greenock-
                                                 15 columns and 160 rows omitted, Split(Tourn: 55, Season: 23/24, Week: 35, Hist: 0))
 (180×20 SubDataFrame
 Row │ tournament_id  season_id  season   match_id  tournament_slug  home_team ⋯
     │ Int64          Int64      String7  Int64     String15         String31  ⋯
─────┼──────────────────────────────────────────────────────────────────────────
   1 │            55      52606  23/24    11395859  championship     arbroath  ⋯
   2 │            55      52606  23/24    11395855  championship     partick-t
   3 │            55      52606  23/24    11395856  championship     greenock-
   4 │            55      52606  23/24    11395857  championship     inverness
   5 │            55      52606  23/24    11395858  championship     dunfermli ⋯
   6 │            55      52606  23/24    11395850  championship     raith-rov
   7 │            55      52606  23/24    11395851  championship     queens-pa
   8 │            55      52606  23/24    11395852  championship     dundee-un
  ⋮  │       ⋮            ⋮         ⋮        ⋮             ⋮                   ⋱
 174 │            55      52606  23/24    11395675  championship     arbroath  ⋯
 175 │            55      52606  23/24    11395676  championship     greenock-
 176 │            55      52606  23/24    11395668  championship     raith-rov
 177 │            55      52606  23/24    11395669  championship     queens-pa
 178 │            55      52606  23/24    11395670  championship     inverness ⋯
 179 │            55      52606  23/24    11395671  championship     dundee-un
 180 │            55      52606  23/24    11395672  championship     ayr-unite
                                                 15 columns and 165 rows omitted, Split(Tourn: 55, Season: 23/24, Week: 36, Hist: 0))
 (165×20 SubDataFrame
 Row │ tournament_id  season_id  season   match_id  tournament_slug  home_team ⋯
     │ Int64          Int64      String7  Int64     String15         String31  ⋯
─────┼──────────────────────────────────────────────────────────────────────────
   1 │            55      62411  24/25    12477141  championship     falkirk-f ⋯
   2 │            55      62411  24/25    12477136  championship     partick-t
   3 │            55      62411  24/25    12477137  championship     livingsto
   4 │            55      62411  24/25    12477138  championship     hamilton-
   5 │            55      62411  24/25    12477142  championship     airdrieon ⋯
   6 │            55      62411  24/25    12476942  championship     queens-pa
   7 │            55      62411  24/25    12476945  championship     ayr-unite
   8 │            55      62411  24/25    12476940  championship     dunfermli
  ⋮  │       ⋮            ⋮         ⋮        ⋮             ⋮                   ⋱
 159 │            55      62411  24/25    12476817  championship     airdrieon ⋯
 160 │            55      62411  24/25    12476868  championship     raith-rov
 161 │            55      62411  24/25    12476808  championship     ayr-unite
 162 │            55      62411  24/25    12476805  championship     queens-pa
 163 │            55      62411  24/25    12476806  championship     raith-rov ⋯
 164 │            55      62411  24/25    12476810  championship     partick-t
 165 │            55      62411  24/25    12476811  championship     greenock-
                                                 15 columns and 150 rows omitted, Split(Tourn: 55, Season: 24/25, Week: 35, Hist: 0))
 (170×20 SubDataFrame
 Row │ tournament_id  season_id  season   match_id  tournament_slug  home_team ⋯
     │ Int64          Int64      String7  Int64     String15         String31  ⋯
─────┼──────────────────────────────────────────────────────────────────────────
   1 │            55      62411  24/25    12477141  championship     falkirk-f ⋯
   2 │            55      62411  24/25    12477136  championship     partick-t
   3 │            55      62411  24/25    12477137  championship     livingsto
   4 │            55      62411  24/25    12477138  championship     hamilton-
   5 │            55      62411  24/25    12477142  championship     airdrieon ⋯
   6 │            55      62411  24/25    12476942  championship     queens-pa
   7 │            55      62411  24/25    12476945  championship     ayr-unite
   8 │            55      62411  24/25    12476940  championship     dunfermli
  ⋮  │       ⋮            ⋮         ⋮        ⋮             ⋮                   ⋱
 164 │            55      62411  24/25    12476810  championship     partick-t ⋯
 165 │            55      62411  24/25    12476811  championship     greenock-
 166 │            55      62411  24/25    12476802  championship     livingsto
 167 │            55      62411  24/25    12476804  championship     airdrieon
 168 │            55      62411  24/25    12476801  championship     hamilton- ⋯
 169 │            55      62411  24/25    12476803  championship     dunfermli
 170 │            55      62411  24/25    12476807  championship     falkirk-f
                                                 15 columns and 155 rows omitted, Split(Tourn: 55, Season: 24/25, Week: 36, Hist: 0))
 (175×20 SubDataFrame
 Row │ tournament_id  season_id  season   match_id  tournament_slug  home_team ⋯
     │ Int64          Int64      String7  Int64     String15         String31  ⋯
─────┼──────────────────────────────────────────────────────────────────────────
   1 │            55      62411  24/25    12477141  championship     falkirk-f ⋯
   2 │            55      62411  24/25    12477136  championship     partick-t
   3 │            55      62411  24/25    12477137  championship     livingsto
   4 │            55      62411  24/25    12477138  championship     hamilton-
   5 │            55      62411  24/25    12477142  championship     airdrieon ⋯
   6 │            55      62411  24/25    12476942  championship     queens-pa
   7 │            55      62411  24/25    12476945  championship     ayr-unite
   8 │            55      62411  24/25    12476940  championship     dunfermli
  ⋮  │       ⋮            ⋮         ⋮        ⋮             ⋮                   ⋱
 169 │            55      62411  24/25    12476803  championship     dunfermli ⋯
 170 │            55      62411  24/25    12476807  championship     falkirk-f
 171 │            55      62411  24/25    12473064  championship     partick-t
 172 │            55      62411  24/25    12473055  championship     ayr-unite
 173 │            55      62411  24/25    12473060  championship     raith-rov ⋯
 174 │            55      62411  24/25    12473062  championship     hamilton-
 175 │            55      62411  24/25    12473066  championship     dunfermli
                                                 15 columns and 160 rows omitted, Split(Tourn: 55, Season: 24/25, Week: 37, Hist: 0))
 (180×20 SubDataFrame
 Row │ tournament_id  season_id  season   match_id  tournament_slug  home_team ⋯
     │ Int64          Int64      String7  Int64     String15         String31  ⋯
─────┼──────────────────────────────────────────────────────────────────────────
   1 │            55      62411  24/25    12477141  championship     falkirk-f ⋯
   2 │            55      62411  24/25    12477136  championship     partick-t
   3 │            55      62411  24/25    12477137  championship     livingsto
   4 │            55      62411  24/25    12477138  championship     hamilton-
   5 │            55      62411  24/25    12477142  championship     airdrieon ⋯
   6 │            55      62411  24/25    12476942  championship     queens-pa
   7 │            55      62411  24/25    12476945  championship     ayr-unite
   8 │            55      62411  24/25    12476940  championship     dunfermli
  ⋮  │       ⋮            ⋮         ⋮        ⋮             ⋮                   ⋱
 174 │            55      62411  24/25    12473062  championship     hamilton- ⋯
 175 │            55      62411  24/25    12473066  championship     dunfermli
 176 │            55      62411  24/25    12473058  championship     airdrieon
 177 │            55      62411  24/25    12473059  championship     livingsto
 178 │            55      62411  24/25    12473061  championship     queens-pa ⋯
 179 │            55      62411  24/25    12473063  championship     falkirk-f
 180 │            55      62411  24/25    12473065  championship     greenock-
                                                 15 columns and 165 rows omitted, Split(Tourn: 55, Season: 24/25, Week: 38, Hist: 0))
"""



cv_config = BayesianFootball.Data.CVConfig(
    tournament_ids = [55, 56],
    target_seasons = ["21/22"],
    history_seasons = 0, # Will auto-include "23/24" if available
    dynamics_col = :match_week,
    warmup_period = 34
)

config = CVConfig(
    # ...
    stop_early = true  # Splits go 1..5, 1..6, ..., 1..37
)

splits = Data.create_data_splits(ds, cv_config)
unique(splits[end][1].tournament_id)

model = bayesianfootball.models.pregame.ar1poisson()
vocabulary = bayesianfootball.features.create_vocabulary(ds, model) 



feature_sets = BayesianFootball.Features.create_features(
    splits, 
    vocabulary, 
    model, 
    cv_config 
)


# ---------------------------------------------------------------------------
"""

using this with the feautes api / functions 


"""


data_store = BayesianFootball.Data.load_default_datastore()

ds = BayesianFootball.Data.DataStore( 
    Data.add_match_week_column(data_store.matches),
    data_store.odds,
    data_store.incidents
)


cv_config = BayesianFootball.Data.CVConfig(
    tournament_ids = [55],
    target_seasons = ["24/25"],
    history_seasons = 0, # Will auto-include "23/24" if available
    dynamics_col = :match_week,
    warmup_period = 35,
    stop_early = true  # Splits go 1..5, 1..6, ..., 1..37
    # stop_early = false # Splits go 1..5, ..., 1..38 (The default)
)


splits = Data.create_data_splits(ds, cv_config)

model   = BayesianFootball.Models.PreGame.StaticPoisson()
vocabulary = BayesianFootball.Features.create_vocabulary(ds, model) 



feature_sets = BayesianFootball.Features.create_features(
    splits, 
    vocabulary, 
    model, 
    cv_config 
)


feature_sets[1][1].data[:team_map]


"""
julia> feature_sets = BayesianFootball.Features.create_features(
           splits, 
           vocabulary, 
           model, 
           cv_config 
       )
3-element Vector{Tuple{FeatureSet, BayesianFootball.Data.SplitMetaData}}:
 (FeatureSet(Dict{Symbol, Any}(:round_away_ids => [[26, 16, 19, 18, 17], [9, 27, 15, 24, 8], [8], [16, 24, 19, 17, 26], [9, 18, 15, 26, 27], [17, 24, 15, 8, 16], [8, 18, 19, 27], [15, 24, 27, 26, 16], [19, 8, 18, 17, 9], [9, 15]  …  [24], [8, 27, 19, 24, 18], [8, 15, 27, 8, 9, 26], [16, 19, 17, 9, 16, 15, 26, 27], [18, 15, 17, 18, 19, 8], [24, 19, 18, 16, 15, 9], [17, 26, 27, 8], [15, 17, 24, 17, 19], [27, 9, 26, 16, 8, 24, 18], [15, 19, 27, 8, 9]], :matches_df => 165×20 DataFrame
 Row │ tournament_id  season_id  season   match_id  tournament_slug  home_team ⋯
     │ Int64          Int64      String7  Int64     String15         String31  ⋯
─────┼──────────────────────────────────────────────────────────────────────────
   1 │            55      62411  24/25    12477141  championship     falkirk-f ⋯
   2 │            55      62411  24/25    12477136  championship     partick-t
   3 │            55      62411  24/25    12477137  championship     livingsto
   4 │            55      62411  24/25    12477138  championship     hamilton-
   5 │            55      62411  24/25    12477142  championship     airdrieon ⋯
   6 │            55      62411  24/25    12476942  championship     queens-pa
   7 │            55      62411  24/25    12476945  championship     ayr-unite
   8 │            55      62411  24/25    12476940  championship     dunfermli
  ⋮  │       ⋮            ⋮         ⋮        ⋮             ⋮                   ⋱
 159 │            55      62411  24/25    12476817  championship     airdrieon ⋯
 160 │            55      62411  24/25    12476868  championship     raith-rov
 161 │            55      62411  24/25    12476808  championship     ayr-unite
 162 │            55      62411  24/25    12476805  championship     queens-pa
 163 │            55      62411  24/25    12476806  championship     raith-rov ⋯
 164 │            55      62411  24/25    12476810  championship     partick-t
 165 │            55      62411  24/25    12476811  championship     greenock-
                                                 15 columns and 150 rows omitted, :flat_away_ids => [26, 16, 19, 18, 17, 9, 27, 15, 24, 8  …  26, 16, 8, 24, 18, 15, 19, 27, 8, 9], :time_indices => [1, 1, 1, 1, 1, 2, 2, 2, 2, 2  …  34, 34, 34, 34, 34, 35, 35, 35, 35, 35], :flat_home_goals => [2, 0, 2, 0, 1, 1, 5, 0, 1, 0  …  3, 5, 0, 2, 1, 1, 0, 1, 1, 1], :round_home_ids => [[15, 24, 9, 8, 27], [26, 18, 19, 17, 16], [18], [9, 15, 8, 18, 27], [17, 19, 16, 24, 8], [19, 18, 27, 9, 26], [17, 26, 24, 16], [17, 8, 9, 19, 18], [27, 26, 15, 16, 24], [15, 24]  …  [15], [26, 16, 9, 17, 15], [16, 16, 24, 17, 18, 19], [9, 8, 27, 24, 19, 8, 17, 18], [26, 27, 9, 16, 15, 27], [26, 17, 24, 8, 26, 27], [24, 16, 15, 18], [9, 16, 16, 8, 18], [26, 19, 9, 15, 19, 27, 17], [18, 26, 17, 24, 16]], :flat_home_ids => [15, 24, 9, 8, 27, 26, 18, 19, 17, 16  …  9, 15, 19, 27, 17, 18, 26, 17, 24, 16], :team_map => Dict{InlineStrings.String31, Int64}("cumnock" => 83, "broomhill-fc" => 88, "musselburgh-athletic-fc" => 112, "coldstream-fc" => 97, "nairn-county-fc" => 58, "carnoustie-panmure" => 80, "inverurie-loco-works-fc" => 95, "dundee-fc" => 13, "east-kilbride" => 46, "dunbar-united" => 85…), :n_rounds => 35, :flat_away_goals => [1, 0, 0, 2, 0, 1, 0, 2, 0, 0  …  0, 0, 1, 1, 0, 1, 1, 1, 2, 2]…)), Split(Tourn: 55, Season: 24/25, Week: 35, Hist: 0))
 (FeatureSet(Dict{Symbol, Any}(:round_away_ids => [[26, 16, 19, 18, 17], [9, 27, 15, 24, 8], [8], [16, 24, 19, 17, 26], [9, 18, 15, 26, 27], [17, 24, 15, 8, 16], [8, 18, 19, 27], [15, 24, 27, 26, 16], [19, 8, 18, 17, 9], [9, 15]  …  [8, 27, 19, 24, 18], [8, 15, 27, 8, 9, 26], [16, 19, 17, 9, 16, 15, 26, 27], [18, 15, 17, 18, 19, 8], [24, 19, 18, 16, 15, 9], [17, 26, 27, 8], [15, 17, 24, 17, 19], [27, 9, 26, 16, 8, 24, 18], [15, 19, 27, 8, 9], [18, 16, 26, 24, 17]], :matches_df => 170×20 DataFrame



julia> feature_sets[1][1].data
Dict{Symbol, Any} with 13 entries:
  :round_away_ids   => [[26, 16, 19, 18, 17], [9, 27, 15, 24, 8], [8], [16, 24, 19, 17, 26], [9, 18, 15, 26, 27], [17, 24, 15, 8, 16]…
  :matches_df       => 165×20 DataFrame…
  :flat_away_ids    => [26, 16, 19, 18, 17, 9, 27, 15, 24, 8  …  26, 16, 8, 24, 18, 15, 19, 27, 8, 9]
  :time_indices     => [1, 1, 1, 1, 1, 2, 2, 2, 2, 2  …  34, 34, 34, 34, 34, 35, 35, 35, 35, 35]
  :flat_home_goals  => [2, 0, 2, 0, 1, 1, 5, 0, 1, 0  …  3, 5, 0, 2, 1, 1, 0, 1, 1, 1]
  :round_home_ids   => [[15, 24, 9, 8, 27], [26, 18, 19, 17, 16], [18], [9, 15, 8, 18, 27], [17, 19, 16, 24, 8], [19, 18, 27, 9, 26],…
  :flat_home_ids    => [15, 24, 9, 8, 27, 26, 18, 19, 17, 16  …  9, 15, 19, 27, 17, 18, 26, 17, 24, 16]
  :team_map         => Dict{String31, Int64}("cumnock"=>83, "broomhill-fc"=>88, "musselburgh-athletic-fc"=>112, "coldstream-fc"=>97, …
  :n_rounds         => 35
  :flat_away_goals  => [1, 0, 0, 2, 0, 1, 0, 2, 0, 0  …  0, 0, 1, 1, 0, 1, 1, 1, 2, 2]
  :n_teams          => 131
  :round_home_goals => SubArray{Int64, 1, Vector{Int64}, Tuple{Vector{Int64}}, false}[[2, 0, 2, 0, 1], [1, 5, 0, 1, 0], [3], [1, 2, 1…
  :round_away_goals => SubArray{Int64, 1, Vector{Int64}, Tuple{Vector{Int64}}, false}[[1, 0, 0, 2, 0], [1, 0, 2, 0, 0], [2], [1, 1, 0…

julia> feature_sets[1][1].data
Dict{Symbol, Any} with 13 entries:
  :round_away_ids   => [[26, 16, 19, 18, 17], [9, 27, 15, 24, 8], [8], [16, 24, 19, 17, 26], [9, 18, 15, 26, 27], [17, 24, 15, 8, 16]…
  :matches_df       => 165×20 DataFrame…
  :flat_away_ids    => [26, 16, 19, 18, 17, 9, 27, 15, 24, 8  …  26, 16, 8, 24, 18, 15, 19, 27, 8, 9]
  :time_indices     => [1, 1, 1, 1, 1, 2, 2, 2, 2, 2  …  34, 34, 34, 34, 34, 35, 35, 35, 35, 35]
  :flat_home_goals  => [2, 0, 2, 0, 1, 1, 5, 0, 1, 0  …  3, 5, 0, 2, 1, 1, 0, 1, 1, 1]
  :round_home_ids   => [[15, 24, 9, 8, 27], [26, 18, 19, 17, 16], [18], [9, 15, 8, 18, 27], [17, 19, 16, 24, 8], [19, 18, 27, 9, 26],…
  :flat_home_ids    => [15, 24, 9, 8, 27, 26, 18, 19, 17, 16  …  9, 15, 19, 27, 17, 18, 26, 17, 24, 16]
  :team_map         => Dict{String31, Int64}("cumnock"=>83, "broomhill-fc"=>88, "musselburgh-athletic-fc"=>112, "coldstream-fc"=>97, …
  :n_rounds         => 35
  :flat_away_goals  => [1, 0, 0, 2, 0, 1, 0, 2, 0, 0  …  0, 0, 1, 1, 0, 1, 1, 1, 2, 2]
  :n_teams          => 131
  :round_home_goals => SubArray{Int64, 1, Vector{Int64}, Tuple{Vector{Int64}}, false}[[2, 0, 2, 0, 1], [1, 5, 0, 1, 0], [3], [1, 2, 1…
  :round_away_goals => SubArray{Int64, 1, Vector{Int64}, Tuple{Vector{Int64}}, false}[[1, 0, 0, 2, 0], [1, 0, 2, 0, 0], [2], [1, 1, 0…



"""



# ---------------------------------

"""
  # out of sample process

"""
using Revise
using BayesianFootball

using DataFrames
using Statistics
using ThreadPinning
pinthreads(:cores)

data_store = BayesianFootball.Data.load_default_datastore()

ds = BayesianFootball.Data.DataStore( 
    Data.add_match_week_column(data_store.matches),
    data_store.odds,
    data_store.incidents
)


cv_config = BayesianFootball.Data.CVConfig(
    tournament_ids = [55],
    target_seasons = ["24/25"],
    history_seasons = 0, # Will auto-include "23/24" if available
    dynamics_col = :match_week,
    warmup_period = 35,
    stop_early = true  # Splits go 1..5, 1..6, ..., 1..37
    # stop_early = false # Splits go 1..5, ..., 1..38 (The default)
)


# 1. Setup & Train
splits = Data.create_data_splits(ds, cv_config)

model = BayesianFootball.Models.PreGame.StaticPoisson()
vocabulary = BayesianFootball.Features.create_vocabulary(ds, model) 

# 2. Create Features (NOW AUTOMATICALLY ADAPTS VOCAB)
feature_sets = BayesianFootball.Features.create_features(
    splits, vocabulary, model, cv_config
)

# train 


train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=2) 
sampler_conf = BayesianFootball.Samplers.NUTSConfig(n_samples=100, n_chains=1, n_warmup=100) # Use renamed struct
training_config = BayesianFootball.Training.TrainingConfig(sampler_conf, train_cfg)

results = BayesianFootball.Training.train(model, training_config, feature_sets)

# test the get next match 
meta = results[1][2]

df = Data.get_next_matches(ds, meta, cv_config)


# extract parameters walk through

# function extract_parameters(
#     model::AbstractFootballModel,
#     data_store::DataStore,
#     feature_sets::Vector,      # Contains the Local Maps
#     results::Vector,           # Contains the Posteriors
#     config::CVConfig           # Contains the Column definitions
# )
#


# Allocate memory 
oos_predictions = Dict{Int, Any}
cv_config

n_folds  = min(length(results), length(feature_sets))

zip

z = zip(results, feature_sets)

zz = first(z)
# process each fold step
chain = zz[1][1]
(fset, meta) = zz[2]
oos_df = Data.get_next_matches(ds, meta, cv_config)

if isempty(oos_df); println("is empty - replace with continue"); end 

v = Features.Vocabulary(fset.data)

p = Models.PreGame.extract_parameters(model, oos_df, v, chain)

jH



df = Data.get_next_matches(ds, meta, cv_config)

function process_fold_extract_parameters(
            model,
            cv_config,
            ds,
            zipped_fold 
          )

    chain = zipped_fold[1][1]
    (fset, meta) = zipped_fold[2]

    oos_df = Data.get_next_matches(ds, meta, cv_config)

    if isempty(oos_df); println("is empty - replace with continue"); end 

    v = Features.Vocabulary(fset.data)

    p = Models.PreGame.extract_parameters(model, oos_df, v, chain)

    return p

end 

process_fold_extract_parameters(model, cv_config, ds, zz) 

for zz in z
    process_fold_extract_parameters(model, cv_config, ds, zz) 
end


function process_fold_extract_parameters(model, cv_config, ds, zipped_fold)
    # Unpack the "Zip" structure: ((chain, meta), (fset, meta))
    # We take the chain from the first tuple, and features/meta from the second
    chain = zipped_fold[1][1]
    (fset, meta) = zipped_fold[2]

    # 1. Get Out-Of-Sample (OOS) data for this specific fold
    oos_df = Data.get_next_matches(ds, meta, cv_config)

    # 2. Handle empty weeks (e.g., end of season)
    if isempty(oos_df)
        return Dict{Int, Any}() # Return empty dict so 'merge' just skips it
    end

    # 3. Reconstruct Local Vocabulary
    # (The FeatureSet stores the Dict, we wrap it back into a Vocabulary struct)
    v = Features.Vocabulary(fset.data)

    # 4. Extract parameters (Low-level function)
    return Models.PreGame.extract_parameters(model, oos_df, v, chain)
end


# Create the iterator
all_folds = zip(results, feature_sets)

# MAP: Run process_fold on every item
# REDUCE: Merge all resulting dictionaries into one
all_oos_results = reduce(
    merge, 
    process_fold_extract_parameters(model, cv_config, ds, fold) for fold in all_folds
)

using BenchmarkTools

@benchmark all_oos_results = reduce(
    merge, 
    process_fold_extract_parameters(model, cv_config, ds, fold) for fold in all_folds
)

# 4. Extract (NOW AUTOMATICALLY FINDS OOS GAMES)
oos_results = BayesianFootball.Models.PreGame.extract_parameters(
    model,
    ds,              # Source of truth
    feature_sets,    # Metadata + Maps
    results,         # Posteriors
    cv_config        # Config used for splitting
)



"""

testing if the tournament_id and seaosn_id stuff works

"""



using Revise
using BayesianFootball

using DataFrames
using Statistics
using ThreadPinning
pinthreads(:cores)

data_store = BayesianFootball.Data.load_default_datastore()

ds = BayesianFootball.Data.DataStore( 
    Data.add_match_week_column(data_store.matches),
    data_store.odds,
    data_store.incidents
)


cv_config = BayesianFootball.Data.CVConfig(
    tournament_ids = [55],
    target_seasons = ["21/22", "22/23"],
    history_seasons = 0, # Will auto-include "23/24" if available
    dynamics_col = :match_week,
    warmup_period = 34,
    stop_early = true  # Splits go 1..5, 1..6, ..., 1..37
    # stop_early = false # Splits go 1..5, ..., 1..38 (The default)
)

# 1. Setup & Train
splits = Data.create_data_splits(ds, cv_config)

model = BayesianFootball.Models.PreGame.AR1Poisson()
vocabulary = BayesianFootball.Features.create_vocabulary(ds, model) 

# 2. Create Features (NOW AUTOMATICALLY ADAPTS VOCAB)
feature_sets = BayesianFootball.Features.create_features(
    splits, vocabulary, model, cv_config
)

# train 


train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=2) 
sampler_conf = BayesianFootball.Samplers.NUTSConfig(n_samples=100, n_chains=1, n_warmup=100) # Use renamed struct
training_config = BayesianFootball.Training.TrainingConfig(sampler_conf, train_cfg)

results = BayesianFootball.Training.train(model, training_config, feature_sets)

# test the get next match 
meta = results[3][2]
df = Data.get_next_matches(ds, meta, cv_config)


# 4. Extract (NOW AUTOMATICALLY FINDS OOS GAMES)
oos_results = BayesianFootball.Models.PreGame.extract_parameters(
    model,
    ds,              # Source of truth
    feature_sets,    # Metadata + Maps
    results,         # Posteriors
    cv_config        # Config used for splitting
)


predict_config = BayesianFootball.Predictions.PredictionConfig( BayesianFootball.Markets.get_standard_markets() )

BayesianFootball.Data.DataPreprocessing.add_inital_odds_from_fractions!(ds)


models_to_compare = [
    (
        name    = "AR1 Poisson", 
        model   = model,            # Your specific model struct
        results = oos_results       # Your results dictionary
    ),
]

match_id = collect(keys(oos_results))[2]
subset( ds.matches, :match_id => ByRow(isequal(match_id)))

# dev/31 for the functions
compare_models(match_id, ds, predict_config, models_to_compare, 
    markets=[:home, :draw, :away, :under_05, :over_05, :under_15, :over_15, :over_25, :under_25, :over_35, :under_35, :btts_yes, :btts_no]
            )






















using BayesianFootball
using DataFrames
using BayesianFootball.Signals

# Load DataStore again (Data is lightweight, models are heavy)
data_store = BayesianFootball.Data.load_default_datastore()
ds = BayesianFootball.Data.DataStore( 
    Data.add_match_week_column(data_store.matches),
    data_store.odds,
    data_store.incidents
)

l2 = subset(ds.matches, :tournament_id => ByRow(isequal(57)), :season => ByRow(isequal("25/26")))
l1 = subset(ds.matches, :tournament_id => ByRow(isequal(56)), :season => ByRow(isequal("25/26")))


### load 


saved_folders = Experiments.list_experiments("exp/market_runs"; data_dir="./data")

# Load them all into a list
loaded_results = Vector{BayesianFootball.Experiments.ExperimentResults}([])
for folder in saved_folders
    try
        res = Experiments.load_experiment(folder)
        push!(loaded_results, res)
    catch e
        @warn "Could not load $folder: $e"
    end
end


m2 = loaded_results[1]
m1 = loaded_results[2]




# 2. Re-create the feature sets using the saved model config
# feature_collection is a Vector of tuples: (FeatureSet, SplitMetaData)
feature_collection1 = BayesianFootball.Features.create_features(
    BayesianFootball.Data.create_data_splits(ds, m1.config.splitter),
    m1.config.model, 
    m1.config.splitter
)


feature_collection2 = BayesianFootball.Features.create_features(
    BayesianFootball.Data.create_data_splits(ds, m2.config.splitter),
    m2.config.model, 
    m2.config.splitter
)


# We usually want the LAST split (the most recent training data)
last_split_idx = length(m1.training_results)

# Get the Chain from the results
chain1 = m1.training_results[last_split_idx][1]
# Get the corresponding FeatureSet from our regenerated list
# Note: [1] gets the FeatureSet, [2] would be the Metadata
feature_set1 = feature_collection1[last_split_idx][1]
#=
julia> feature_set1.data[:team_map]
Dict{String, Int64} with 10 entries:
  "alloa-athletic"               => 1
  "kelty-hearts-fc"              => 6
  "montrose"                     => 7
  "cove-rangers"                 => 2
  "queen-of-the-south"           => 9
  "hamilton-academical"          => 4
  "peterhead"                    => 8
  "east-fife"                    => 3
  "inverness-caledonian-thistle" => 5
  "stenhousemuir"                => 10
=#
chain2 = m2.training_results[last_split_idx][1]
feature_set2 = feature_collection2[last_split_idx][1]

#=
julia> feature_set2.data[:team_map]
Dict{String, Int64} with 10 entries:
  "edinburgh-city-fc" => 5
  "dumbarton"         => 3
  "east-kilbride"     => 4
  "stranraer"         => 9
  "the-spartans-fc"   => 10
  "clyde-fc"          => 2
  "elgin-city"        => 6
  "forfar-athletic"   => 7
  "stirling-albion"   => 8
  "annan-athletic"    => 1
=#


match_to_predict1 = DataFrame(
    match_id = [1, 2, 3],
    match_week = [999, 999, 999], 
    home_team = ["cove-rangers", "inverness-caledonian-thistle", "peterhead"], 
    away_team = ["queen-of-the-south", "east-fife", "alloa-athletic"]
)

match_to_predict2 = DataFrame(
    match_id = [1, 2, 3],
    match_week = [999, 999, 999], 
    home_team = ["edinburgh-city-fc", "forfar-athletic", "stranraer"], 
    away_team = ["east-kilbride", "dumbarton", "annan-athletic"]
)

raw_preds1 = BayesianFootball.Models.PreGame.extract_parameters(
    m1.config.model, 
    match_to_predict1, 
    feature_set1, 
    chain1
)


raw_preds2 = BayesianFootball.Models.PreGame.extract_parameters(
    m2.config.model, 
    match_to_predict2, 
    feature_set2, 
    chain2
)



# Convert to DataFrame (Output is now Int64)
function raw_preds_to_df(raw_preds::Dict)
    ids = collect(keys(raw_preds))
    cols = Dict{Symbol, Vector{Any}}(:match_id => ids)
    for k in keys(raw_preds[ids[1]]); cols[k] = [raw_preds[i][k] for i in ids]; end
    return DataFrame(cols)
end


ppd1 = BayesianFootball.Predictions.model_inference(
  BayesianFootball.Experiments.LatentStates(raw_preds_to_df(raw_preds1), m1.config.model)
)

ppd2 = BayesianFootball.Predictions.model_inference(
  BayesianFootball.Experiments.LatentStates(raw_preds_to_df(raw_preds2), m2.config.model)
)


baker = BayesianKelly()

my_signals = [baker]

### HACK:
unique_ids = unique(match_to_predict1.match_id)
str_to_int = Dict(id => i for (i, id) in enumerate(unique_ids))

df_odds.match_id_str = df_odds.match_id 
# Only keep odds for matches we are actually modeling
df_odds_clean = filter(row -> haskey(str_to_int, row.match_id), df_odds)
df_odds_clean.match_id = [str_to_int[x] for x in df_odds_clean.match_id]
###

results = BayesianFootball.Signals.process_signals(
    ppd1, 
    df_odds, 
    my_signals; 
    odds_column=:odds
)


results = BayesianFootball.Signals.process_signals(
    ppd2, 
    df_odds, 
    my_signals; 
    odds_column=:odds
)




# 1. Create a dictionary mapping the Odds String ID to your Model's Int ID.
# Based on your match_to_predict1 and df_odds output:
# "e870eea3" (Cove Rangers) seems to match match_id 1
# "c45c3481" (Dunfermline) matches whatever ID Dunfermline is in your system

id_mapping = Dict(
    "e870eea3" => 1, 
     "a7b3c237"=> 2,
    "b2004323" => 3,
)

# 2. Create a clean copy of the odds dataframe to modify
df_odds_clean = copy(df_odds)

# 3. Map the String IDs to Int64s
# We use a default of -1 for matches not in your model
df_odds_clean.match_id = [get(id_mapping, mid, -1) for mid in df_odds.match_id]

# 4. Filter out any matches that didn't get mapped (were -1)
filter!(row -> row.match_id != -1, df_odds_clean)

# 5. Ensure the column is strictly Int64 (to match ppd1)
df_odds_clean.match_id = identity.(df_odds_clean.match_id)

# 6. Now run the function with the cleaned dataframe
results = BayesianFootball.Signals.process_signals(
    ppd1, 
    df_odds_clean, 
    my_signals; 
    odds_column=:odds
)


#= 
21×10 DataFrame
 Row │ match_id  date        market_name  selection  is_winner  signal_name    signal_params  odds_type  odds     stake      
     │ Int64     Date        String       Symbol     Missing    String         String         String     Float64  Float64    
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │        2  2026-02-07  1X2          away         missing  BayesianKelly  none           odds         11.0   0.00379639
   2 │        2  2026-02-07  1X2          home         missing  BayesianKelly  none           odds          1.29  0.0
   3 │        2  2026-02-07  1X2          draw         missing  BayesianKelly  none           odds          6.2   0.00476902
   4 │        2  2026-02-07  OverUnder    over_15      missing  BayesianKelly  none           odds          1.22  0.0
   5 │        2  2026-02-07  OverUnder    under_15     missing  BayesianKelly  none           odds          5.3   0.0273631
   6 │        2  2026-02-07  OverUnder    over_25      missing  BayesianKelly  none           odds          1.62  0.0
   7 │        2  2026-02-07  OverUnder    under_25     missing  BayesianKelly  none           odds          2.46  0.0257875
   8 │        3  2026-02-07  1X2          away         missing  BayesianKelly  none           odds          2.58  0.168477
   9 │        3  2026-02-07  1X2          home         missing  BayesianKelly  none           odds          2.78  0.0
  10 │        3  2026-02-07  1X2          draw         missing  BayesianKelly  none           odds          3.25  0.0
  11 │        3  2026-02-07  OverUnder    over_15      missing  BayesianKelly  none           odds          1.34  0.00841164
  12 │        3  2026-02-07  OverUnder    under_15     missing  BayesianKelly  none           odds          3.7   0.0
  13 │        3  2026-02-07  OverUnder    over_25      missing  BayesianKelly  none           odds          2.02  0.0309181
  14 │        3  2026-02-07  OverUnder    under_25     missing  BayesianKelly  none           odds          1.87  0.0
  15 │        1  2026-02-07  1X2          away         missing  BayesianKelly  none           odds          2.72  0.0029868
  16 │        1  2026-02-07  1X2          home         missing  BayesianKelly  none           odds          2.64  0.0
  17 │        1  2026-02-07  1X2          draw         missing  BayesianKelly  none           odds          3.6   0.0
  18 │        1  2026-02-07  OverUnder    over_15      missing  BayesianKelly  none           odds          1.3   0.0
  19 │        1  2026-02-07  OverUnder    under_15     missing  BayesianKelly  none           odds          3.9   0.0278019
  20 │        1  2026-02-07  OverUnder    over_25      missing  BayesianKelly  none           odds          1.93  0.0
  21 │        1  2026-02-07  OverUnder    under_25     missing  BayesianKelly  none           odds          1.99  0.0257828

3×4 DataFrame
 Row │ match_id  match_week  home_team                     away_team          
     │ Int64     Int64       String                        String             
─────┼────────────────────────────────────────────────────────────────────────
   1 │        1         999  cove-rangers                  queen-of-the-south
   2 │        2         999  inverness-caledonian-thistle  east-fife
   3 │        3         999  peterhead                     alloa-athletic


# cancelled 
#   peterhead    alloa-athletic
# 1        OverUnder    over_25      missing  BayesianKelly  none           odds          2.02  0.0309181 
# 1        OverUnder    over_15      missing  BayesianKelly  none           odds          1.34  0.00841164

# 
forfar-athletic    dumbarton
3       OverUnder    over_15      missing  BayesianKelly  none           odds          1.27  0.0409278
2       OverUnder    over_25      missing  BayesianKelly  none           odds          1.83  0.0661395

=#


f* w = amount 



id_mapping = Dict(
"1718fd03" => 1,
"9e59ef31" => 2,
"4c7a06f0" => 3,
)

# 2. Create a clean copy of the odds dataframe to modify
df_odds_clean = copy(df_odds)

# 3. Map the String IDs to Int64s
# We use a default of -1 for matches not in your model
df_odds_clean.match_id = [get(id_mapping, mid, -1) for mid in df_odds.match_id]

# 4. Filter out any matches that didn't get mapped (were -1)
filter!(row -> row.match_id != -1, df_odds_clean)

# 5. Ensure the column is strictly Int64 (to match ppd1)
df_odds_clean.match_id = identity.(df_odds_clean.match_id)

# 6. Now run the function with the cleaned dataframe
results2 = BayesianFootball.Signals.process_signals(
    ppd2, 
    df_odds_clean, 
    my_signals; 
    odds_column=:odds
)
#=

 Row │ match_id  match_week  home_team          away_team      
     │ Int64     Int64       String             String         
─────┼─────────────────────────────────────────────────────────
   1 │        1         999  edinburgh-city-fc  east-kilbride
   2 │        2         999  forfar-athletic    dumbarton
   3 │        3         999  stranraer          annan-athletic
       )
21×10 DataFrame
 Row │ match_id  date        market_name  selection  is_winner  signal_name    signal_params  odds_type  odds     stake      
     │ Int64     Date        String       Symbol     Missing    String         String         String     Float64  Float64    
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │        2  2026-02-07  1X2          away         missing  BayesianKelly  none           odds          2.98  0.0
   2 │        2  2026-02-07  1X2          home         missing  BayesianKelly  none           odds          2.38  0.00971221
   3 │        2  2026-02-07  1X2          draw         missing  BayesianKelly  none           odds          3.5   0.0
   4 │        2  2026-02-07  OverUnder    over_15      missing  BayesianKelly  none           odds          1.27  0.0409278
   5 │        2  2026-02-07  OverUnder    under_15     missing  BayesianKelly  none           odds          4.2   0.0
   6 │        2  2026-02-07  OverUnder    over_25      missing  BayesianKelly  none           odds          1.83  0.0661395
   7 │        2  2026-02-07  OverUnder    under_25     missing  BayesianKelly  none           odds          2.1   0.0
   8 │        3  2026-02-07  1X2          away         missing  BayesianKelly  none           odds          2.78  0.0
   9 │        3  2026-02-07  1X2          home         missing  BayesianKelly  none           odds          2.48  0.0227864
  10 │        3  2026-02-07  1X2          draw         missing  BayesianKelly  none           odds          3.45  0.0
  11 │        3  2026-02-07  OverUnder    over_15      missing  BayesianKelly  none           odds          1.29  0.0
  12 │        3  2026-02-07  OverUnder    under_15     missing  BayesianKelly  none           odds          4.0   0.0
  13 │        3  2026-02-07  OverUnder    over_25      missing  BayesianKelly  none           odds          1.86  0.0
  14 │        3  2026-02-07  OverUnder    under_25     missing  BayesianKelly  none           odds          2.0   0.0
  15 │        1  2026-02-07  1X2          away         missing  BayesianKelly  none           odds          1.93  0.0
  16 │        1  2026-02-07  1X2          home         missing  BayesianKelly  none           odds          3.55  0.0318824
  17 │        1  2026-02-07  1X2          draw         missing  BayesianKelly  none           odds          4.3   0.0
  18 │        1  2026-02-07  OverUnder    over_15      missing  BayesianKelly  none           odds          1.13  0.0
  19 │        1  2026-02-07  OverUnder    under_15     missing  BayesianKelly  none           odds          8.0   0.0316616
  20 │        1  2026-02-07  OverUnder    over_25      missing  BayesianKelly  none           odds          1.39  0.0
  21 │        1  2026-02-07  OverUnder    under_25     missing  BayesianKelly  none           odds          3.25  0.041917

=#


ppd2.df.mean_odds = mean( 1 ./ ppd2.df.distribution)

ppd2.df.mean_odds = mean.(1 ./ ppd2.df.distribution)
ppd2
ppd2.df.mean_odds = [mean(1 ./ row) for row in ppd2.df.distribution]

ppd2

ppd1.df.mean_odds = [mean(1 ./ row) for row in ppd1.df.distribution]



ppd1

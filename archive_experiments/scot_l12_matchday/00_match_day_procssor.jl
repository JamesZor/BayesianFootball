
using Revise
using BayesianFootball
using DataFrames
using BayesianFootball.Signals

using DataFramesMeta


# load the data store for league 1  and league 2
ds = Data.load_extra_ds()


df_56 = subset(ds.matches, :tournament_id => ByRow(isequal(56)), :season => ByRow(isequal("25/26")))
df_57 =subset(ds.matches, :tournament_id => ByRow(isequal(57)), :season => ByRow(isequal("25/26")))


# loaded the sampled models files 

saved_folders = Experiments.list_experiments("exp/scot_l12_matchday"; data_dir="./data")
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


m1 = loaded_results[1]
m2 = loaded_results[2]



println("Preparing Model 1 (L1)...")
feats_m1 = BayesianFootball.Features.create_features(
    BayesianFootball.Data.create_data_splits(ds, m1.config.splitter),
    m1.config.model, m1.config.splitter
)

# Get the last split (latest data)
chain_m1 = m1.training_results[end][1]
fset_m1  = feats_m1[end][1]



feats_m2 = BayesianFootball.Features.create_features(
    BayesianFootball.Data.create_data_splits(ds, m2.config.splitter),
    m2.config.model, m2.config.splitter
)
chain_m2 = m2.training_results[end][1]
fset_m2  = feats_m2[end][1]

#=
julia> subset(bets, :selection => ByRow(in(bbb)), :stake => ByRow( >(0)))
7×7 DataFrame
 Row │ match_id  event_name                 selection  odds     model_odds  edge_calc   stake       
     │ Int64     String?                    Symbol     Float64  Float64?    Float64     Float64     
─────┼──────────────────────────────────────────────────────────────────────────────────────────────
   1 │        5  Stenhousemuir v Montrose   over_25       2.2      2.1007   0.047272    0.00431379
   2 │        7  East Fife v Cove Rangers   over_25       1.92     1.90542  0.00765276  0.000184714
   3 │        8  Kelty Hearts v Peterhead   over_05       1.09     1.04931  0.0387796   0.275483
   4 │        8  Kelty Hearts v Peterhead   over_15       1.41     1.20741  0.167792    0.274137
   5 │        8  Kelty Hearts v Peterhead   over_25       2.24     1.54705  0.447915    0.250142
   6 │        1  Spartans v Edinburgh City  over_15       1.24     1.23645  0.00286709  0.00030854
   7 │        1  Spartans v Edinburgh City  over_25       1.71     1.64038  0.0424406   0.00887971

 14-Feb-26
14:47:26 	Stenhousemuir v Montrose
Over 2.5 Goals - Over/Under 2.5 Goals
Betfair Bet ID 1:418486995962 | Matched: 14-Feb-26 14:47:26 	Back 	2.22 	1.00 	-- 	--
	1.22
	Matched
14-Feb-26
14:46:41 	Kelty Hearts v Peterhead
Over 2.5 Goals - Over/Under 2.5 Goals
Betfair Bet ID 1:418486823972 | Matched: 14-Feb-26 14:46:41 	Back 	2.26 	1.00 	-- 	--
	1.26
	Matched
14-Feb-26
14:46:20 	Kelty Hearts v Peterhead
Over 1.5 Goals - Over/Under 1.5 Goals
Betfair Bet ID 1:418486749053 | Matched: 14-Feb-26 14:46:20 	Back 	1.43 	4.00 	-- 	--
	1.72
	Matched
14-Feb-26
14:45:15 	Spartans v Edinburgh City
Over 2.5 Goals - Over/Under 2.5 Goals
Betfair Bet ID 1:418486469181 | Matched: 14-Feb-26 14:45:15 	Back 	1.73 	1.00 	-- 	--
	0.73
	Matched
14-Feb-26
14:45:00 	Spartans v Edinburgh City
Over 1.5 Goals - Over/Under 1.5 Goals
Betfair Bet ID 1:418486389069 | Matched: 14-Feb-26 14:45:00 	Back 	1.25 	4.00 	-- 	--
	1.00
	Matched 


final scores:

Stenhousemuir 2 -2 Montrose, 

Kelty Hearts 1 - 1 Peterhead 

Spartans 0 - 0 Edinburgh

=#

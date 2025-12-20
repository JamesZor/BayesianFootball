using Revise 
using BayesianFootball

using DataFrames
using Statistics
using ThreadPinning
using LinearAlgebra
pinthreads(:cores)

# BLAS.set_num_threads(1) 


data_store = BayesianFootball.Data.load_default_datastore()

ds = BayesianFootball.Data.DataStore( 
    Data.add_match_week_column(data_store.matches),
    data_store.odds,
    data_store.incidents
)

cv_config = BayesianFootball.Data.CVConfig(
    tournament_ids = [55],
    target_seasons = ["22/23"],
    history_seasons = 0, # Will auto-include "23/24" if available
    dynamics_col = :match_week,
  warmup_period = 33,
    stop_early = false
)

splits = Data.create_data_splits(ds, cv_config)

model = BayesianFootball.Models.PreGame.StaticPoisson()


vocabulary = BayesianFootball.Features.create_vocabulary(ds, model) 


feature_sets = BayesianFootball.Features.create_features(
    splits, model, cv_config
)



train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=1) 

sampler_conf = Samplers.NUTSConfig(
                100,
                2,
                100,
                0.65,
                10,
  Samplers.UniformInit(-0.05, 0.05)
)


training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)

results = Training.train(model, training_config, feature_sets)

#=

This was the deconstructed  method get getting the predictions out of the folds 

it very messy and i dont remember  it that well, so there could be some issues here 

im struggling to think here  - and understand how it is all connected,
as it does feel like there are redundant parts in this process. - possible 
around the feature_sets and the vocabulary - adapt_vocabulary, 
the extract extract_parameters dont need the vocabulary ( adapt_vocabulary) since 
the features has that information in it since the struct for vocabulary was a dynamic one that took 
a dict / named vector,
=#

predict_config = BayesianFootball.Predictions.PredictionConfig( BayesianFootball.Markets.get_standard_markets() )

BayesianFootball.Data.add_inital_odds_from_fractions!(ds)



feature_sets
fs = feature_sets[1]

df_test_1 = BayesianFootball.Data.get_next_matches(ds, fs[2], cv_config)

(df_train, meta_check) = splits[1]

local_vocab = BayesianFootball.Features.adapt_vocabulary(vocabulary, df_train)

model_preds_1 = BayesianFootball.Models.PreGame.extract_parameters(
     model,
    df_test_1,
    local_vocab, 
  results[1][1]
);

match_id = collect(keys(model_preds_1))[1]
probs = BayesianFootball.Predictions.predict_market(
  model, predict_config, model_preds_1[match_id]...
                )

using Statistics
model_odds = Dict(key => mean(1 ./ value) for (key, value) in pairs(probs))

                open_odds, close_odds, outcomes = BayesianFootball.Predictions.get_market_data(
                     match_id, predict_config, ds.odds
                )
open_odds



#= 
 Using the more update to date method of the experiment module to run sampling 


=#

experiment_conf = Experiments.ExperimentConfig(
                    name = "test_static_poisson",
                    model = model,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./data/junk"
)

exp_results = Experiments.run_experiment(ds, experiment_conf)


#=
exp_results:
BayesianFootball.Experiments.ExperimentResults(BayesianFootball.Experiments.ExperimentConfig("test_static_poisson", BayesianFootball.Models.PreGame.StaticPoisson{Distributions.Normal{Float64}}(Distributions.Normal{Float64}(μ=0.0, σ=0.5)), CVConfig(Targets=["22/23"], Hist=0), TrainingConfig(strategy=Independent, checkpointing=false), String[], "", "./data/junk"), Tuple{Any, Any}[(MCMC chain (100×35×2 Array{Float64, 3}), Split(Tourn: 55, Season: 22/23, Week: 33, Hist: 0)), (MCMC chain (100×35×2 Array{Float64, 3}), Split(Tourn: 55, Season: 22/23, Week: 34, Hist: 0)), (MCMC chain (100×35×2 Array{Float64, 3}), Split(Tourn: 55, Season: 22/23, Week: 35, Hist: 0))], Vocabulary(Dict{Symbol, Any}(:n_teams => 131, :team_map => Dict{InlineStrings.String31, Int64}("cumnock" => 83, "broomhill-fc" => 88, "musselburgh-athletic-fc" => 112, "coldstream-fc" => 97, "nairn-county-fc" => 58, "carnoustie-panmure" => 80, "inverurie-loco-works-fc" => 95, "dundee-fc" => 13, "east-kilbride" => 46, "dunbar-united" => 85…))), "./data/junk/test_static_poisson_20251217_140143")

We need some interfacing module here to address the extract_parameters 

so for the folds in the splitter, we want to get the get_next_matches 
and extract the model parameters extract_parameters style
I think we should return a DataFrames with match_id and col for the the model types 
const PoissonRates = NamedTuple{(:λ_h, :λ_a), Tuple{Vector{Float64}, Vector{Float64}}}
const DixonColesRates = NamedTuple{(:λ_h, :λ_a, :ρ), Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}

etc 

Moreover, im not sure were this functionality should exist, since we have the extract_parameters
for the chains etc, i suppose we just extend them for this case right, since they are similar. 

But then again, we need to sort out the predictions module which takes the 
# src/prediction/pregame/predict-abstractpoisson.jl
# src/predictions/pregame/predict-dixoncoles.jl

# src/predictions/prediction-module.jl
 and some config regarding the markets to process - and some methods on how to process them

but it might be useful to have a module to process / preprocess the market/ odds, 
like get the open and close odds, get the them as probabilities - 
also remove the vig / overround - but use the market config so we know the associated  markets 
as 1x2 need to be processed different from an under/ over market as there are 3 lines in 1x2 compared to the 
two. 

in sense we get a dataframe, with the mathc_id, market_symbol (:home, away, under_25 etc ) 
market_type ( 1x2, under/over, .. ) , market_choice, outcome(bool), open_odds, close_odds, 
open_prob, close_prob, fair_open_odds, fair_close_odds, fair_open_prob, fair_close_prob, 
line_move, etc .. 

Furthermore, having a prediction module, or interfacing, which base on the config ( which market to look at )
we can compute the predict posterior Distributions for the market lines, use the extract_parameters dataframe,
which will give a dataframe, that is one-to-one to the markets data frame, as we will have 
match_id, model , market_choice, market_type, .. , model's predictive posterior Distributions ( a vector of Float64 which is a the chain / estmate of the prob ) 

Then further along the line we use model prediction dataframe with the market dataframe to run 
strategies on it, as i have a couple kelly / util funcitons on. which will give use a stake sizing 
for that match_id, model, market_choice, market_type, stake, etc. 
so we can then take a collection of model and get a dataframe like this 
but don't worry about this know - it more information to help relate the previous parts and to 
make sense of it all.

=#

"""
# --- address the vocabulary, extract_parameters issue
"""


#=
 # src/features/vocabulary
create_vocabulary  -> Vocabulary - Which is a wrapper for a Dict

 # src/TypesInterfaces 
"""
    Vocabulary (Your 'G')
"""
struct Vocabulary
    mappings::Dict{Symbol, Any}
end


in the function create_features 
function create_features(
    data_split::AbstractDataFrame,
    vocabulary::Vocabulary,
    model::AbstractFootballModel,
    splitter_config::AbstractSplitter
)::FeatureSet

basic just pulls from the vocabulary 


function create_features(
    data_splits::Vector{<:Tuple{<:AbstractDataFrame, M}},
    vocabulary::Vocabulary,
    model::AbstractFootballModel,
    splitter_config::AbstractSplitter
)::Vector{Tuple{FeatureSet, M}} where M

    return [
        (
            create_features(
                data, 
                adapt_vocabulary(vocabulary, data), # <--- ADD THIS CALL
                model, 
                splitter_config
            ), 
            meta
        ) 
        for (data, meta) in data_splits
    ]
end



here is the wrapper of the create_features, 

Note that we can move vocabulary into the features create_features 


in the models we have 

# --- 3. Builder ---
function build_turing_model(model::StaticPoisson, feature_set::FeatureSet)
    return static_poisson_model_train(
        feature_set.data[:n_teams]::Int,
        feature_set.data[:flat_home_ids],     # Pre-flattened
        feature_set.data[:flat_away_ids],     # Pre-flattened
        feature_set.data[:flat_home_goals],   # Pre-flattened
        feature_set.data[:flat_away_goals],   # Pre-flattened
        model
        )
end


function extract_parameters(
    model::StaticPoisson, 
    df::AbstractDataFrame, 
    vocabulary::Vocabulary, 
    chains::Chains
)::Dict{Int, PoissonRates}
    # ... (Specific extraction logic for Poisson) ...
    extraction_dict = Dict{Int64, PoissonRates}()
    
    home_adv_vec = vec(chains["home_adv"])
    team_map = vocabulary.mappings[:team_map]

As we see in the extract_parameters, we need to remove the Vocabulary, 
and replace it with the feature_set, 



### ---- Notes 
# 1. Define the struct as a subtype of AbstractDict
struct FeatureSet <: AbstractDict{Symbol, Any}
    data::Dict{Symbol, Any}
end

# 2. Add a helper constructor (optional, for convenience)
FeatureSet(pairs::Pair...) = FeatureSet(Dict{Symbol, Any}(pairs...))

# 3. Implement the Dictionary Interface
# This allows using [] to retrieve data
Base.getindex(fs::FeatureSet, key::Symbol) = getindex(fs.data, key)

# This allows using [] = value to set data
Base.setindex!(fs::FeatureSet, value, key::Symbol) = setindex!(fs.data, value, key)

# These allow iteration (for loops) and checking size
Base.length(fs::FeatureSet) = length(fs.data)
Base.iterate(fs::FeatureSet, state...) = iterate(fs.data, state...)

#--------



=#

using Revise 
using BayesianFootball

using DataFrames
using Statistics
using ThreadPinning
using LinearAlgebra
pinthreads(:cores)

# BLAS.set_num_threads(1) 


data_store = BayesianFootball.Data.load_default_datastore()

ds = BayesianFootball.Data.DataStore( 
    Data.add_match_week_column(data_store.matches),
    data_store.odds,
    data_store.incidents
)

cv_config = BayesianFootball.Data.CVConfig(
    tournament_ids = [55],
    target_seasons = ["22/23"],
    history_seasons = 0, # Will auto-include "23/24" if available
    dynamics_col = :match_week,
  warmup_period = 33,
    stop_early = false
)

splits = Data.create_data_splits(ds, cv_config)

model = BayesianFootball.Models.PreGame.StaticPoisson()



feature_sets = BayesianFootball.Features.create_features(splits, model, cv_config)


train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=1) 

sampler_conf = Samplers.NUTSConfig(
                100,
                2,
                100,
                0.65,
                10,
  Samplers.UniformInit(-0.05, 0.05)
)


training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)

results = Training.train(model, training_config, feature_sets)




feature_sets
fs = feature_sets[1]

df_test_1 = BayesianFootball.Data.get_next_matches(ds, fs[2], cv_config)

Data.get_next_matches(ds, fs, cv_config)

model_preds_1 = BayesianFootball.Models.PreGame.extract_parameters(
     model,
    df_test_1,
    fs[1],
  results[1][1]
)

model_preds_1 = BayesianFootball.Models.PreGame.extract_parameters(
     model,
    df_test_1,
     feature_sets[1],
  results[1]
)


model_preds_1 = BayesianFootball.Models.PreGame.extract_parameters(
     model,
    df_test_1,
     feature_sets[1],
  results[1]
)


DataFrame(model_preds_1)

df = DataFrame(
    match_id = collect(keys(model_preds_1)), 
    latentstate = collect(values(model_preds_1))
)

df.latentstate[1][:λ_h]

predict_config = BayesianFootball.Predictions.PredictionConfig( BayesianFootball.Markets.get_standard_markets() )

BayesianFootball.Data.add_inital_odds_from_fractions!(ds)



match_id = collect(keys(model_preds_1))[3]
probs = BayesianFootball.Predictions.predict_market(
  model, predict_config, model_preds_1[match_id]...
                )

using Statistics
model_odds = Dict(key => mean(1 ./ value) for (key, value) in pairs(probs))

open_odds, close_odds, outcomes = BayesianFootball.Predictions.get_market_data(
     match_id, predict_config, ds.odds
)
open_odds
outcomes


"""
### 2025-12-18 11:30
"""
#=
Step 1: The "Experiment Bridge" (src/experiments/post_process.jl)

We need a function that takes ExperimentResults and returns the LatentsDataFrame.

    Task: Implement extract_oos_predictions(ds, exp_results).

    Design: It automates the get_next_matches -> extract_parameters loop.

=#

# ----------------------------------------------
# 1. The set up 
# ----------------------------------------------
using Revise 
using BayesianFootball

using DataFrames
using Statistics
using ThreadPinning
using LinearAlgebra
pinthreads(:cores)


data_store = BayesianFootball.Data.load_default_datastore()

ds = BayesianFootball.Data.DataStore( 
    Data.add_match_week_column(data_store.matches),
    data_store.odds,
    data_store.incidents
)

cv_config = BayesianFootball.Data.CVConfig(
    tournament_ids = [55],
    target_seasons = ["22/23"],
    history_seasons = 0, # Will auto-include "23/24" if available
    dynamics_col = :match_week,
  warmup_period = 33,
    stop_early = false
)

splits = BayesianFootball.Data.create_data_splits(ds, cv_config)
model = BayesianFootball.Models.PreGame.StaticPoisson()
vocabulary = BayesianFootball.Features.create_vocabulary(ds, model) 
feature_sets = BayesianFootball.Features.create_features(
    splits, model, cv_config
)
train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=1) 
sampler_conf = Samplers.NUTSConfig(
                100,
                2,
                100,
                0.65,
                10,
  Samplers.UniformInit(-0.05, 0.05)
)
training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)

experiment_conf = Experiments.ExperimentConfig(
                    name = "test_static_poisson",
                    model = model,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./data/junk"
)

exp_results = Experiments.run_experiment(ds, experiment_conf)

# -----------------------------------------------------------------------------------------------
# 2. The extraction of the LatentsDataFrame 
# -----------------------------------------------------------------------------------------------

# 2. Bridge: Get the Latents DataFrame
# This handles all the vocabulary recreation, OOS matching, and extraction
latents = Experiments.extract_oos_predictions(ds, exp_results)

# 3. Use it (Method Forwarding works!)
println("Extracted $(nrow(latents)) matches.")
# Access columns directly
first_lambda_home = latents[1, :λ_h]

latents.model

latents


#= 
Step 2: The Market Processor (src/predictions/data_prep.jl)

We organize the logic to get clean odds and outcomes.

    Task: Implement prepare_market_data(ds, prediction_config).

    Design: Focus on handling the specific columns (open/close) and removing the overround.

=#




odds = ds.odds
names(odds)
"""
julia> names(odds)
12-element Vector{String}:
 "tournament_id"
 "season_id"
 "match_id"
 "market_id"
 "market_name"
 "market_group"
 "choice_name"
 "choice_group"
 "initial_fractional_value"
 "final_fractional_value"
 "winning"
"""




#=
Needing to sort out the predictions markets part of the code since it is mess, and doesn't flow right. 
Since market sub module deal with more of the data processing side rather the predictions I believe 
that it should be contained in the data module - As a sub module dedicated to processing the market odds 
( As in future we might have different sources for odds and thus a module to manage them in an abstract manner).

dir layout suggestion:
  data/markets/
      - types.jl -> house the abstract types - AbstractMarket, marketconfig ( wrapper - holy traits of dict of vector, that contains the AbstractMarket
                  used to define which markets we want to process as each market will have a slightly different way) 

      - interfaces.jl -> contains the abstract function declaration and fall back function for the common interaction functions for the concrete 
                    abstractmarket struct / types, etc market_keys, names, process_market_type  etc 

      types/ -> dir to house the files for the concrete abstractmarket types/ struct and interface function implementations.
            types/x12.jl -> for the implementations of 1x2 (home, draw, away) market. 
            types/under_over.jl -> for the implementations of under over market type
            types/btts.jl -> for the implementations of under over market type
            etc 

    - prepare_market_odds.jl -> File to handle the processing of the ds to marketsdata struct 

    -common.jl/helpers.jl  -> Some file to contain the helper and common funcitons like the devig, and odds parse (fractions to decimal odds) 
        

Example of the work flow 

market_odds = BayesianFootball.Data.Markets.get_market_data(ds::Data_store)::Data.Markets.MarketData  

=#
# ---- 

using DataFrames
using Test

# 1. Load the Package (assuming you are in the project env)
# If you are developing locally, you might need: include("src/BayesianFootball.jl")
using BayesianFootball
using BayesianFootball.Data
# We need to access the exported types from the submodule
using BayesianFootball.Data: MarketConfig, Market1X2, MarketOverUnder, MarketBTTS, prepare_market_data

# ==============================================================================
# 2. SETUP: Create Mock Data
# ==============================================================================
println(">> Creating Mock Data...")

# Matches the structure of your uploaded CSV
# We add a mix of 1X2, Over/Under, and BTTS to test all paths.
mock_odds = DataFrame(
    match_id = [101, 101, 101, 101, 101, 102, 102],
    market_name = ["Full time", "Full time", "Full time", "Match goals", "Match goals", "Both teams to score", "Both teams to score"],
    market_group = ["1X2", "1X2", "1X2", "Over/Under", "Over/Under", "BTTS", "BTTS"],
    choice_name = ["1", "X", "2", "Over", "Under", "Yes", "No"],
    choice_group = ["", "", "", "2.5", "2.5", "", ""], # Note 2.5 is string here to test safe parsing
    initial_fractional_value = ["2/1", "3/1", "2/1", "1/1", "4/5", "10/11", "10/11"], # Open odds
    final_fractional_value   = ["2/1", "2/1", "3/1", "4/5", "1/1", "1/1", "4/5"],     # Close odds (Line moved!)
    winning = [true, false, false, true, false, false, true]
)

# Mock DataStore (we only care about odds for this test)
struct MockDataStore
    odds::DataFrame
end
ds = MockDataStore(mock_odds)

# ==============================================================================
# 3. SETUP: Configure Markets
# ==============================================================================
println(">> Configuring Markets...")

config = MarketConfig([
    Market1X2(),
    MarketOverUnder(2.5),
    MarketBTTS()
])

# ==============================================================================
# 4. EXECUTE: Run the Pipeline
# ==============================================================================
println(">> Running prepare_market_data...")

market_data = prepare_market_data(ds, config)
df = market_data.df

# ==============================================================================
# 5. VERIFY: Assertions & Inspection
# ==============================================================================
println(">> Verifying Results...")

@testset "Market Data Refactor Tests" begin

    # A. Check Structure
    @test nrow(df) == 7
    @test "clm_prob" in names(df)
    @test "vig_open" in names(df)

    # B. Check 1X2 Logic
    # Filter for Match 101, 1X2
    rows_1x2 = subset(df, :match_id => ByRow(==(101)), :market_name => ByRow(==("1X2")))
    @test nrow(rows_1x2) == 3
    # Check symbol mapping "1" -> :home
    @test :home in rows_1x2.selection
    @test :draw in rows_1x2.selection
    
    # C. Check Odds Parsing (Fractional to Decimal)
    # "2/1" -> 3.0
    home_row = filter(r -> r.selection == :home, rows_1x2)[1, :]
    @test home_row.odds_open ≈ 3.0
    
    # D. Check Enrichment (Vig & CLM)
    # Open Odds: Home(3.0), Draw(4.0), Away(3.0) -> Implied: 0.33 + 0.25 + 0.33 = 0.916... ??
    # Wait, 2/1=3.0. 1/3=0.333. Sum = 1.0 (Approx). 
    # Let's check calculation of CLM
    # Open: Home 3.0, Close: Home 3.0. CLM Odds = 0.0
    @test home_row.clm_odds ≈ 0.0

    # Check Draw: Open 3/1 (4.0), Close 2/1 (3.0). Odds dropped.
    draw_row = filter(r -> r.selection == :draw, rows_1x2)[1, :]
    @test draw_row.clm_odds ≈ -1.0 # 3.0 - 4.0

    # E. Check Over/Under Logic
    rows_ou = subset(df, :market_name => ByRow(==("OverUnder")))
    @test nrow(rows_ou) == 2
    @test rows_ou[1, :market_line] == 2.5
    @test :over_25 in rows_ou.selection # Check dynamic symbol generation

    println("   ALL TESTS PASSED!")
end

# ==============================================================================
# 6. VISUAL INSPECTION
# ==============================================================================
println("\n>> Preview of Processed Data:")
show(df[!, [:match_id, :market_name, :selection, :odds_open, :odds_close, :prob_fair_open, :clm_prob]], allrows=true)



# -------
predict_config = BayesianFootball.Predictions.PredictionConfig( BayesianFootball.Markets.get_standard_markets() )
#=
julia> predict_config = BayesianFootball.Predictions.PredictionConfig( BayesianFootball.Markets.get_standard_markets() )
BayesianFootball.Predictions.PredictionConfig(Set(BayesianFootball.Markets.AbstractMarket[BayesianFootball.Markets.Market1X2(), BayesianFootball.Markets.MarketOverUnder(0.5), BayesianFootball.Markets.MarketOverUnder(2.5), BayesianFootball.Markets.MarketOverUnder(1.5), BayesianFootball.Markets.MarketBTTS(), BayesianFootball.Markets.MarketOverUnder(3.5), BayesianFootball.Markets.MarketOverUnder(4.5)]))
=#


a = first(predict_config.markets)
#=
julia> first(predict_config.markets)
BayesianFootball.Markets.Market1X2()

=#
Predictions._process_market_type(ds.odds, a)

#=
julia> Predictions._process_market_type(ds.odds, a)
ERROR: MethodError: no method matching _process_market_type(::DataFrame, ::BayesianFootball.Markets.Market1X2)
The function `_process_market_type` exists, but no method is defined for this combination of argument types.

Closest candidates are:
  _process_market_type(::DataFrame, ::BayesianFootball.Predictions.Markets.Market1X2)
   @ BayesianFootball ~/bet_project/BayesianFootball/src/predictions/market_data.jl:55
  _process_market_type(::DataFrame, ::BayesianFootball.Predictions.Markets.MarketOverUnder)
   @ BayesianFootball ~/bet_project/BayesianFootball/src/predictions/market_data.jl:71
  _process_market_type(::DataFrame, ::BayesianFootball.Predictions.Markets.MarketBTTS)
   @ BayesianFootball ~/bet_project/BayesianFootball/src/predictions/market_data.jl:87

Stacktrace:
 [1] top-level scope
   @ REPL[29]:1

=#



  

Predictions.prepare_market_data(ds, predict_config) 
#=

julia> Predictions.prepare_market_data(ds, predict_config) 
ERROR: MethodError: no method matching _process_market_type(::DataFrame, ::BayesianFootball.Markets.Market1X2)
The function `_process_market_type` exists, but no method is defined for this combination of argument types.

Closest candidates are:
  _process_market_type(::DataFrame, ::BayesianFootball.Predictions.Markets.Market1X2)
   @ BayesianFootball ~/bet_project/BayesianFootball/src/predictions/market_data.jl:55
  _process_market_type(::DataFrame, ::BayesianFootball.Predictions.Markets.MarketOverUnder)
   @ BayesianFootball ~/bet_project/BayesianFootball/src/predictions/market_data.jl:71
  _process_market_type(::DataFrame, ::BayesianFootball.Predictions.Markets.MarketBTTS)
   @ BayesianFootball ~/bet_project/BayesianFootball/src/predictions/market_data.jl:87

Stacktrace:
 [1] prepare_market_data(ds::BayesianFootball.Data.DataStore, config::BayesianFootball.Predictions.PredictionConfig)
   @ BayesianFootball.Predictions ~/bet_project/BayesianFootball/src/predictions/market_data.jl:17
 [2] top-level scope
   @ REPL[21]:1
=#





#=
predict_config = BayesianFootball.Predictions.PredictionConfig( BayesianFootball.Markets.get_standard_markets() )


# TODO: 
prepare_market_data(ds; predictions_confit = DEFAULT_PREDICTION _CONFIG) 


# return 
 struct MarketData
    df::DataFrame
    market_confg:: prediction_config  - tell us what markets are processed
end


Base.getindex(ls::MarketData, args...) = getindex(ls.df, args...)
Base.setindex!(ls::MarketData, val, args...) = setindex!(ls.df, val, args...)
Base.size(ls::MarketData) = size(ls.df)
Base.size(ls::MarketData, i) = size(ls.df, i)
Base.show(io::IO, ls::MarketData) = show(io, ls.df)

# 2. DataFrames specific methods (Use DataFrames.nrow, not Base.nrow)
DataFrames.nrow(ls::MarketData) = nrow(ls.df)
DataFrames.ncol(ls::MarketData) = ncol(ls.df)


in the df we need the cols 
  -match_id
  - market_group ( 1x2, under /over, 
  - market_choice ( home, under_15, ) 
  - symbol ( :home, :under_15 ... ) 
  - open_odds - initial_fractional_value is a string like "1/2"  -> decimal odds  
  - close_odds - final_fractional_value is a string like "1/2"  -> decimal odds  
  - open_prob - open_odds to probability 1/ odds 
  - close_odds - close_odds to probability 1/odds 
  - clm - closing line movement, close_odds - open_odds - track the movement of the line 
  - outcome - if the line was winning or not bool  -> winning 
  - fair_open_odds -> open odds with the vig remove for the market group. 
  - fair_close_odds -> close odds with the vig removed for the market group 
  - fair_open_prob  -> open prob with the vig removed for the market group 
  - fair_close_prob -> close prob with the vig removed 
  - open_vig -> amount of vig for the open market group 
  - closed_vig -> amount of vig for the close for the market group 

since we repeat the process for open and close odds, we can abstract the process and 
run it for the open ( initial_fractional_value) and then the ( final_fractional_value). 


=#



#=


This is so we can then process the latentstate struct for different models, 
as an example suppose we have  poisson model out of sample latentstate struct named 

latents

then we want 
postrior_predictive_distributions = BayesianFootball.Predictions.model_inference(market_data.prediction_config , latents) 
or 
postrior_predictive_distributions = BayesianFootball.Predictions.model_inference(prediction_config , latents) 
postrior_predictive_distributions = BayesianFootball.Predictions.model_inference(latents; =DEFAULT_PREDICTION) 
use kwags so we dont need to handle the prediction_config passing around etc 

postrior_predictive_distributions is a wrapper like latentstate 

struct PPD 
  df::AbstractDataFrame 
  model::AbstractFootballModel 
end 

with the goal to then have strategy  module or something 


BayesianFootball.Strategy.compute_stakes( postrior_predictive_distributions, market_data; strategies_config = DEFAULT_Stratergis ) 

dont worry about this last step yet this is more for an idea of what is needed 



=#




#=

Step 3: The Probabilities Engine (src/predictions/probabilities.jl)

We implement the math to convert λ to P(HomeWin).

    Task: Implement predict_probabilities(model, latents_df, market_config).

    Design: Use the PoissonRates and DixonColesRates types we defined earlier to dispatch to the correct probability formulas (Skellam distribution, etc.).

Step 4: The Strategy Runner (Future)

Once we have ModelProb and MarketProb in the same dataframe, calculating Kelly Criterion is a simple vectorized column operation: edge = model_prob - market_prob.

=#






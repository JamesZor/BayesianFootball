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

predict_config = BayesianFootball.Predictions.PredictionConfig( BayesianFootball.Markets.get_standard_markets() )



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

Step 3: The Probabilities Engine (src/predictions/probabilities.jl)

We implement the math to convert λ to P(HomeWin).

    Task: Implement predict_probabilities(model, latents_df, market_config).

    Design: Use the PoissonRates and DixonColesRates types we defined earlier to dispatch to the correct probability formulas (Skellam distribution, etc.).

Step 4: The Strategy Runner (Future)

Once we have ModelProb and MarketProb in the same dataframe, calculating Kelly Criterion is a simple vectorized column operation: edge = model_prob - market_prob.

=#






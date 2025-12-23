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
  warmup_period = 30,
    stop_early = false
)

splits = BayesianFootball.Data.create_data_splits(ds, cv_config)
model = BayesianFootball.Models.PreGame.StaticPoisson()
vocabulary = BayesianFootball.Features.create_vocabulary(ds, model) 
feature_sets = BayesianFootball.Features.create_features(
    splits, model, cv_config
)
train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=2) 
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

# pd = Data.Markets.MarketConfig( Data.Markets.Market1X2())
pd = Data.Markets.DEFAULT_MARKET_CONFIG

a = Predictions.model_inference(latents; market_config=pd)

market_data = Data.prepare_market_data(ds)


# signals 

# 1. Flat Stake: Bet 5% of bankroll if Expected Value > 0
using BayesianFootball.Signals
flat_strat = FlatStake(0.05)

# 2. Conservative Kelly: Quarter Kelly (0.25)
kelly_strat = KellyCriterion(0.25)

# 3. Bayesian/Shrinkage Kelly: Uses the Baker-McHale analytical approximation
shrink_strat = AnalyticalShrinkageKelly()

baker = BayesianKelly()

# Combine them into a vector
my_signals = [flat_strat, kelly_strat, shrink_strat, baker]

# TUI Check: Let's look at one to see your new Base.show implementation
display(kelly_strat)
# Output should be:
# Signal Strategy: KellyCriterion
# ├─ Parameters: fraction=0.25
# └─ Logic: Classic Kelly: f = p - (1-p)/(odds-1), using mean probability.

println("Market Data Columns: ", names(market_data))

# Run the pipeline
results = process_signals(
    a,              # Your PPD (Predictions)
    market_data.df,    # Your Market Odds
    my_signals;     # The strategies defined above
    odds_column = :odds_close # Explicitly tell it to use closing odds
)



using DataFrames, Statistics, PrettyTables

"""
    audit_signals(results; breakdown=[:signal_name, :signal_params, :market_name, :selection])

Calculates performance metrics (ROI, Total Profit, Hit Rate) for the processed signals.
"""
function audit_signals(results; breakdown=[:signal_name, :market_name, :selection])
    # 1. Unpack the DataFrame
    df = results isa DataFrame ? results : results.df
    
    # 2. Filter for Active Bets (Stake > 0) & Settled Outcomes
    # We filter out tiny floating point stakes (effectively 0)
    active_bets = filter(row -> row.stake > 1e-6, df)
    
    if nrow(active_bets) == 0
        @warn "No active bets found."
        return DataFrame()
    end
    
    # Ensure we have the outcome column
    if !("is_winner" in names(active_bets))
        error("Column 'is_winner' not found. Ensure your market_data includes outcomes.")
    end
    
    # 3. Calculate Profit/Loss per bet
    # If Winner: Stake * (DecimalOdds - 1)
    # If Loser: -Stake
    # If Missing (Cancelled/Future): 0.0
    active_bets.pnl = map(eachrow(active_bets)) do r
        if ismissing(r.is_winner) return 0.0 end
        r.is_winner ? r.stake * (r.odds - 1.0) : -r.stake
    end

    # 4. Group and Aggregate
    gdf = groupby(active_bets, breakdown)
    
    stats = combine(gdf, 
        nrow => :count,
        :is_winner => (w -> mean(skipmissing(w))) => :win_rate,
        :stake => sum => :total_staked,
        :pnl => sum => :total_profit,
        [:pnl, :stake] => ((p, s) -> sum(p) / sum(s)) => :roi
    )
    
    # 5. Formatting
    # Convert ROI to percentage for readability
    stats.roi_pct = round.(stats.roi * 100, digits=2)
    stats.win_rate_pct = round.(stats.win_rate * 100, digits=1)
    
    # Clean up and Sort
    select!(stats, Not([:roi, :win_rate])) # Drop raw ratios, keep percentages
    sort!(stats, :roi_pct, rev=true)
    
    return stats
end

stats = audit_signals(results)


strat_stats = audit_signals(results, breakdown=[:signal_name, :signal_params])
display(strat_stats)


using DataFrames, Statistics, Plots

"""
    plot_equity_curve(results::SignalsResult; initial_bankroll=100.0)

Plots the cumulative profit/loss over time for each signal strategy.
Assumes a fixed stake percentage (non-compounding) for simplicity, 
but visualizes the growth trajectory.
"""
function plot_equity_curve(results::SignalsResult; initial_bankroll=100.0)
    df = results.df
    
    # 1. Ensure we have a time dimension (using match_id as proxy if date missing)
    # Ideally, join with match_date from your data. For now, we sort by match_id.
    sort!(df, :match_id)
    
    # 2. Calculate PnL per bet
    # If winner: stake * (odds - 1)
    # If loser: -stake
    df.pnl = map(r -> r.is_winner ? r.stake * (r.odds - 1.0) : -r.stake, eachrow(df))
    
    # 3. Accumulate PnL by Signal
    signals = unique(df.signal_name)
    p = plot(title="Strategy Equity Curves", xlabel="Bets Placed", ylabel="Profit (Units)", legend=:topleft)
    
    for sig in signals
        # Get bets for this signal
        sig_data = filter(r -> r.signal_name == sig, df)
        
        if nrow(sig_data) > 0
            # Calculate cumulative sum of PnL
            cumulative_pnl = cumsum(sig_data.pnl)
            
            # Add to plot
            plot!(p, cumulative_pnl, label=sig, linewidth=2, alpha=0.8)
        end
    end
    
    display(p)
end

plot_equity_curve(results)

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

latents = Experiments.extract_oos_predictions(ds, exp_results)

then we want 
postrior_predictive_distributions = BayesianFootball.Predictions.model_inference(latents ; market_confg = DEFAULT_MARKET_CONFIG) 
use kwags so we dont need to handle the passing around etc. 

model_inference logic overview. 
  For each match we have we need to process the following

  - compute the score matrix posterior type dispatch on abstractpoisson, abstract dixoncoles, abstractnegativebinomal etc, 
    Here the score matrix is a score_matrix: S_home × S_away × N , where S_home, S_away are equal and define the max score line around 12, 
    and N is the number of samples from all the chains, - so if we have 2 chains with 500 steps then N = 2 × 500 
    + Here score_matrix should be a struct, which is a wrapper for an abstract matrix / array as 

    struct ScoreMatrix 
        data::AbstractMatrix
        type:: ( abstractpoisson, abstractnegativebinomal, etc / abstracrpregamemodel) 
    end

      inference_score_matrix(latents )  use dispatch of latents.model, 


  then for each market in the market config. 
        inference_market_type( score_matrix, market) 

  So have separate files to handle the process of the market types, 
        - 1x2, under_over, btts 

hence we should have 


struct PPD 
  df::AbstractDataFrame 
  model::AbstractFootballModel 
  marketconfig::Data.Market.MarketConfig 
end 


here the PPD df has the columns, match_id, market_name, market_line, selection, distribution ( vector floats - the chains  for the distributions) 
here PPD process is similar to that of the markets_data 

folder structure:

src/predictions/
    - types.jl # place all the types and show functions 
    - interface.jl #  interfacing base functions to be use by the different markets in the implementations of the score matrix 

    model-type-implementations/  # better name 
        poisson.jl # compute the score matrix for poisson 
        dixoncoles # compute the score matrix for dixoncoles model 
        negativebinomal # compute the score matrix for negativebinomal 
        etc... 

  inferences.jl # main methods for the processing 
  
    markets-inference/ 
        inferences-1x2.jl # compute the 1x2 lines from the score matrix 
        under_over.jl # compute the under over line functions 
        btts.jl 


    - utils / helpers .jl 

=#




#=

Step 3: The Probabilities Engine (src/predictions/probabilities.jl)

We implement the math to convert λ to P(HomeWin).

    Task: Implement predict_probabilities(model, latents_df, market_config).

    Design: Use the PoissonRates and DixonColesRates types we defined earlier to dispatch to the correct probability formulas (Skellam distribution, etc.).

Step 4: The Strategy Runner (Future)

Once we have ModelProb and MarketProb in the same dataframe, calculating Kelly Criterion is a simple vectorized column operation: edge = model_prob - market_prob.

=#



"""
# - Signals module dev 
"""

market_data = Data.prepare_market_data(ds)
latents = Experiments.extract_oos_predictions(ds, exp_results)
ppd = Predictions.model_inference(latents)

#=
The signal module of the package is for generating a market making signals, as in whether to back or lay a market line 
given the predictive posterior distribution from the given model. A signal should return a parentage of wealth to 
stake, given the market lines PPD. 
We can achieve this in many ways, the simplest signal would be a fixed percentage stake, i.e 10% of wealth for the market line.
moreover we can add more complexity, such as computing the point estimate expected value: E(PPD)* market_odds. 
and then place a bet when EV is great than some thing. 
Or more commonly we can use a kelly Criterion for the fractions to bet. 
However since we are dealing with PPD, we dont want to use a point estimate, instead we want to use 
the full distributions. 
One that we can look at is the: 
"""
Calculates the Optimal Shrinkage Factor 'k' using the Baker & McHale (2013) 
Bootstrap/Resampling method (Eq. 2).

This simulates the penalty of acting on noisy probability estimates.
"""
function bayesian_kelly(chain_probs::AbstractVector, offered_odds::Number)
    b = offered_odds - 1.0
    
    # 1. We treat the Mean of the posterior as the "Ground Truth" for this simulation
    p_true = mean(chain_probs)
    
    # If the mean suggests no bet, we can't shrink what doesn't exist.
    s_mean = kelly_fraction(offered_odds, p_true)
    if s_mean <= 1e-6
        return 0.0
    end

    # 2. We generate the "Naive" bets we would have made for every sample in the chain.
    # Ideally, we calculate s*(q) for every q.
    # This represents the variability of our decision making process.
    naive_bets = [kelly_fraction(offered_odds, q) for q in chain_probs]

    # 3. Objective Function: 
    # Find k such that if we shrink ALL our naive bets by k, 
    # we maximize growth against the "p_true".
    function objective(k)
        utility_sum = 0.0
        n = length(naive_bets)
        
        for s_q in naive_bets
            # The bet we actually place is the Naive Bet * Shrinkage k
            actual_stake = k * s_q
            
            # Constraint check
            actual_stake = k * s_q
            if actual_stake >= 0.999 return Inf end
            if actual_stake < 1e-4 actual_stake = 0.0 end
            
            
            # Utility evaluated against the Mean (p_true)
            u = p_true * log(1.0 + b * actual_stake) + (1.0 - p_true) * log(1.0 - actual_stake)
            utility_sum += u
        end
        
        return -(utility_sum / n)
    end

    res = optimize(objective, 0.0, 1.0)
    best_k = Optim.minimizer(res)
    
    return s_mean * best_k
end

or an estimate of it as 
"""Baker-McHale Eq 5 (Analytical Approx)"""
function calc_analytical_shrinkage(chain_probs::AbstractVector, offered_odds::Number)
    p_mean = mean(chain_probs)
    p_var = var(chain_probs)
    b = offered_odds - 1.0
    s_star = ((b + 1) * p_mean - 1) / b
    if s_star <= 0 return 0.0 end
    term = ((b + 1) / b)^2
    k_factor = (s_star^2) / (s_star^2 + term * p_var)
    return s_star * k_factor
end

function calc_analytical_shrinkage(probabilities::NamedTuple, odds::NamedTuple)
    common_keys = keys(probabilities) ∩ keys(odds) 
    return NamedTuple(
            k => calc_analytical_shrinkage(probabilities[k], odds[k])
            for k in common_keys 
          )
end


more we have the following: 


"""
.3.1
Dixon and Coles approach
Once the vector of betting probabilities π has been obtained with one among
the methods described in Section 7.1—either with basic normalization, Shin
procedure, or regression analysis—we should try to assess how and when it
is convenient to bet on some events using the model probabilities. The ratio-
nale behind betting is that if our model reflects approximately well the real
chances of occurrence of a given event, then these probabilities should be used
to challenge the bookmakers and eventually beat them: we have already seen
in the previous sections that whenever oi > pi for some binary events, then
the bookmaker obtains a positive expected profit. Regarding football and the
three-way outcomes, Dixon and Coles (1997) suggested to fix a margin toler-
ance δ, such that one would bet on a match/event i if and only if:
pj
i/πj
i > δ,
j ∈∆i,
where pj
i, πj
i denote the probabilities for event i and occurrence j under the
model and the bookmaker, respectively—we assume here to deal with a unique
bookmaker, for simplicity Actually neither pj
i nor πj
i correspond to the true
probabilities for match i, which are usually unknown in football and in general
in sports; however, we could obtain a positive return if our estimated proba-
bilities are sufficiently more accurate than those derived from the bookmakers,
meaning their ratio exceeds a fixed tolerance δ. If the model probabilities pi
are accurate, then the expected gain from a unit bet for match i and outcome
j is given by
E(Gbe) = pj
i/πj
i −1.
(7.13)
As remarked by Dixon and Coles (1997), the choice of δ strongly depends
on the amount of risk aversion undertaken by the bettor. They even propose
to estimate δ or, alternatively, to monitor the return by varying the values of δ
through a sensitivity analysis. If we increase δ, this means we adopt a stricter
betting regime, but with fewer bets. Thus, the amount of δ strictly depends
on the risk aversion of the single bettors: we give a practical example on the
choice of δ in the case-study reported in Section 7.4.
"""



"""
7.3.4
Expected profit optimization
According to a common sense the choice of the matches for which placing
one or more bets depends on our utility for betting: we could decide which
matches to bet on and obtain a favourable game to play as the posterior
expected profit is positive. Epstein (2012) proposed to bet on outcomes with
a positive expected profit but place the bets so we obtain a low variance of the
profit. According to a similar perspective, Rue and Salvesen (2000) proposed
to bet in order to maximize the expected profit while keeping the variance of
the profit lower than some threshold.
An equivalent formulation is to maximise the expected profit minus the
variance of the profit, which determine how we should place our bets up to
a multiplicative constant. This constant can be found if we choose a specific
value or an upper limit for the variance of the profit.
Let E(Gj
i) and (σj
i )2 be the expected profit and the variance for
betting an unit amount on outcome j in match i, where j
∈∆i
=
{“Home win”, “Draw”, “Away win”}, respectively: we can detect these val-
ues from the probabilities pj
i and inverse odds oj
i, as previously explained. For
simplicity, suppose to not place more than one bet for each match, and let
βj
i be the corresponding bet. Let U(·) denote a proper bettor utility function,
then the optimal bet is given by setting the condition:
argmax
βj
i ≥0
U({βj
i }),
where
U({βj
i }) = E(profit) −Var(profit) =
�
i∈β
βj
i (E(Gj
i) −βj
i (σj
i )2).
(7.16)
The analytical solution is given by βj
i = max{0, E(Gj
i)/(2(σj
i )2)}, where addi-
tionally we choose the outcome j with maximal βj
i E(Gj
i) for match i in order
to not place more than one bet for each match.
As an imaginary example, consider again the odds from the example con-
sidered in the previous sections, Arsenal vs Manchester United considered in
the previous sections. For this match the expected profit for the single out-
comes can be computed as pj/πj −1, where pj and πj are the bettor and
the betting probabilities for the outcome j, respectively. Suppose the bettor
model probabilities for the home win, the draw and away win are: 0.41, 0.28,
0.31, whereas the bookmaker probabilities obtained through basic normaliza-
tion are: 0.382, 0.319, 0.299. We then get the following expected profits from
the bettor’s perspective:
E(GH
be) = 0.41/0.382 −1 = 0.073
E(GD
be) = 0.28/0.319 −1 = −0.122
E(GL
be) = 0.31/0.299 −1 = 0.037.
"""


Hence there are different signals / strategies to consider and thus we need to compute some empirical / 
experimentail results to consider which one is better, in order to achieve this we need the signal module in the 
package to handle the construction and processing of different types signals/ strategies. 
Similar to the other modules in this package we shall have an abstractsignals struct, and thus when creating a singls, 
we shall have a separate files that has the implement abstract functions that are shared between them using the 
julia dispatch method, thus we have a basic abstract implementation in an interfaces files. 

hence we can construct a config / vector of signals ( concrete types from the abstractsignals) 
and process each market line from the  PPD dataframe ( Predictions.model_inference(latents)) .

Given we have the following: 

market_data = Data.prepare_market_data(ds)
latents = Experiments.extract_oos_predictions(ds, exp_results)
ppd = Predictions.model_inference(latents)


signals = BayesianFootball.Signals.process(ppd, market_data; signals_config=<struct wrapper for the vector of signals types for dispatch>) 

here the returned value signals is a wrapper of dataframe and a few configs. 

struct 
df::dataframe,
model::AbstractFootballModel,
singals::singalsCOnfig ( which signals are contained) 
end

with the df being a long format dataframe, 
  -match_id
  - market_group ( 1x2, under /over, 
  - market_choice ( home, under_15, ) 
  - symbol ( :home, :under_15 ... ) 
  - signal --> name of the signal : kelly, ev, fix stake 
  - stake -> amount to stake - percentage 


The dir should look like this: 

src/signals/ 
    -types.jl -> Houses the abstract types: abstractsignals, and abstractconfigs and the wrapper " holy constant" logic 
    - interfaces.jl -> contains the base abstract functions for the concrete signals logic, such that we can use the julia dispatch logic in the processing 
                        so shared function / contract for the signals. 
    - utils.jl /helpers.jl -> For common helper functions that are needed between files and funcitons here . 

    -process_signals.jl -> The main process logic to handle the dispatch / and process of the each market line for each signals and to 
                        form the returned wrapper struct. Using the dataframe.jl funcitons for the speed/ such as subset and transfrom and threads for speed up. 

    implementations/ 
        flat_stake.jl -> logic for a strategies / signal - defining the logic in the interfaces.jl 
        kelly.jl 
        baker-mchale.jl 
        baker-mchale-srink.jl 
        etc, i think you get the idea 












=#

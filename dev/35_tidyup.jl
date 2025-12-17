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
    splits, vocabulary, model, cv_config
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

Note that we can move vocabulary into the features



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

exp_results


"""
# ----------------------------
# training 
"""



"""
Test 1: Enable Checkpointing & Verify Files

First, we modify your configuration to write to a local tmp folder. Run this script:

"""

# 1. Setup Config with Checkpointing ENABLED
# We set cleanup=false so we can inspect the files manually after the run
training_config = BayesianFootball.Training.TrainingConfig(
    sampler = sampler_conf,
    strategy = train_cfg,
    checkpoint_dir = "./data/tmp_checkpoints",   # <--- NEW: Local folder
    cleanup_checkpoints = false             # <--- NEW: Keep files for inspection
)

println("Test 1: Running Training with Checkpoints...")
results_run_1 = BayesianFootball.Training.train(model, training_config, feature_sets)

# 2. Verification
println("\nVerifying Checkpoints...")
if isdir("./data/tmp_checkpoints")
    files = readdir("./data/tmp_checkpoints")
    println("   Found $(length(files)) files in checkpoint dir: $files")
    
    # Check if we have the expected number of .jls files (should match splits length)
    expected = length(feature_sets)
    actual = count(f -> endswith(f, ".jls"), files)
    
    if expected == actual
        println("   ✅ SUCCESS: Created $actual checkpoint files for $expected splits.")
    else
        println("   ❌ FAIL: Expected $expected checkpoints, found $actual.")
    end
else
    println("   ❌ FAIL: Checkpoint directory was not created.")
end


"""
Test 2: The "Resume" Capability

Now we simulate a crash/restart. We will re-run the exact same command.

    Expected Behavior: The system should see the files in ./tmp_checkpoints, print a message like "Found X checkpoints. Resuming...", and finish instantly without running the sampler again.

"""

println("\n🧪 Test 2: Resuming (Should skip all work)...")

# We use the SAME config (pointing to the existing ./tmp_checkpoints)
@time results_run_2 = BayesianFootball.Training.train(model, training_config, feature_sets)

# Verification
# If the time is near 0.0 seconds, it worked.
# You should see logs: "✅ All splits already completed via checkpoints."


"""
Test 3: Resilience (Simulate Partial Failure)

Now we delete one file to simulate a crash that happened partway through.

    Expected Behavior: The system should re-train only the missing split.

"""


println("\n Test 3: Partial Resume (Simulating crash)...")

# 1. Delete the checkpoint for Split #1 (or any index)
split_to_delete = 2
file_to_delete = joinpath("./data/tmp_checkpoints", "split_$(lpad(split_to_delete, 3, '0')).jls")
rm(file_to_delete; force=true)

println("   Deleted checkpoint: $file_to_delete")

# 2. Run Training Again
println("   Re-running training...")
results_run_3 = BayesianFootball.Training.train(model, training_config, feature_sets)

# Verification
# Look at the logs. You should see:
# "Starting Independent training for 1 splits..." (instead of total splits)
# And it should verify that only Split 1 was processed.


println("\n Test 4: Cleanup...")

# 1. Update config to enable cleanup
cleanup_config = BayesianFootball.Training.TrainingConfig(
    sampler = sampler_conf,
    strategy = train_cfg,
    checkpoint_dir = "./data/tmp_checkpoints",
    cleanup_checkpoints = true  # <--- NEW: Delete after success
)

# 2. Run (will likely resume Split 1 if Test 3 finished, or just finish instantly)
BayesianFootball.Training.train(model, cleanup_config, feature_sets)

# 3. Verify directory is gone/empty
if !isdir("./tmp_checkpoints") || isempty(readdir("./data/tmp_checkpoints"))
    println("   ✅ SUCCESS: Checkpoint directory cleaned up.")
else
    println("   ❌ FAIL: Checkpoint directory still exists/not empty.")
end

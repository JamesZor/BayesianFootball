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
  warmup_period = 20,
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



using Distributions
model_2 = Models.PreGame.StaticPoisson(prior= Cauchy(0))

experiment_conf_2 = Experiments.ExperimentConfig(
                    name = "test_static_poisson_cauchy",
                    model = model_2,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./data/junk"
)

exp_results_2 = Experiments.run_experiment(ds, experiment_conf_2)



using BayesianFootball.Signals
flat_strat = FlatStake(0.05)
# 2. Conservative Kelly: Quarter Kelly (0.25)
kelly_strat = KellyCriterion(0.25)

# 3. Bayesian/Shrinkage Kelly: Uses the Baker-McHale analytical approximation
shrink_strat = AnalyticalShrinkageKelly()

baker = BayesianKelly()

my_signals = [flat_strat, kelly_strat, shrink_strat, baker]

using BayesianFootball.BackTesting


# ledger = BayesianFootball.BackTesting.run_backtest(ds, [exp_results], my_signals; market_config = Data.Markets.DEFAULT_MARKET_CONFIG)


ledger = BayesianFootball.BackTesting.run_backtest(ds, [exp_results, exp_results_2], my_signals; market_config = Data.Markets.DEFAULT_MARKET_CONFIG)


a = BayesianFootball.BackTesting.generate_tearsheet(ledger)



symbols = [:model_name, :model_parameters, :signal_name, :signal_params, :selection]

a = BayesianFootball.BackTesting.generate_tearsheet(ledger, symbols)

sort(a, :SharpeRatio, rev=true)


sorted_df = sort(a, [:selection, order(:SharpeRatio, rev=true)])


"""
    best_strategies_per_selection(breakdown_df::DataFrame; 
                                  metric::Symbol=:SharpeRatio, 
                                  top_n::Int=3)

Finds the top performing strategies for each unique market selection.
"""
function best_strategies_per_selection(breakdown_df::DataFrame; 
                                       metric,
                                       top_n::Int=3)
    
    # 1. Group by Selection
    gdf = groupby(breakdown_df, :selection)
    
    # 2. Select Top N per group
    best_df = combine(gdf) do sub_df
        # Sort this group by the requested metric
        sorted = sort(sub_df, metric, rev=true)
        
        # Take top N (or fewer if not enough rows)
        n = min(top_n, nrow(sorted))
        return first(sorted, n)
    end
    
    # 3. Final Sort for presentation (Group alphabetically by selection)
    sort!(best_df, [:selection, order(metric, rev=true)])
    
    return best_df
end

best_stats = best_strategies_per_selection(a, metric=:CumulativeWealth)
best_stats = best_strategies_per_selection(a, metric=:CalmarRatio)


BayesianFootball.BackTesting.summarize_models(ledger)

a = BayesianFootball.BackTesting.summarize_markets(ledger; compare_models=true)

b = BayesianFootball.BackTesting.detailed_breakdown(ledger)


sort(b, :roi_pct, rev=true)

c = subset(b, :roi_pct => ByRow( x -> x > 0))


sort(c, :roi_pct, rev=true)

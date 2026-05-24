# current_development/copula_negbin/r01_frank_copula_src.jl

using Pkg; Pkg.activate(".")
using Revise
using BayesianFootball
using DataFrames
using Turing
using MCMCChains

println("--- Testing Hierarchical Frank Copula NegBin Model from SRC ---")

# 1. Setup Data
ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.Ireland())

# 2. Configure the Model
inter_cfg = BayesianFootball.Models.PreGame.GlobalInterception()
disp_cfg  = BayesianFootball.Models.PreGame.HomeAwayDispersion()
ha_cfg    = BayesianFootball.Models.PreGame.HierarchicalTeamHomeAdvantage()
dyn_cfg   = BayesianFootball.Models.PreGame.TimeDecayDynamics(days_half_life=90.0)
cop_cfg   = BayesianFootball.Models.PreGame.HierarchicalFrankCopulaConfig()

copula_model = BayesianFootball.Models.PreGame.DynamicCopulaGoalsTimeDecayModel(
    interception_config=inter_cfg, 
    dynamics_config=dyn_cfg, 
    dispersion_config=disp_cfg, 
    homeadvantage_config=ha_cfg,
    copula_config=cop_cfg
)

# 3. CV Configuration
cv_config = BayesianFootball.Data.GroupedCVConfig(
    tournament_groups = [BayesianFootball.Data.tournament_ids(ds.segment)],
    target_seasons = ["2026"],
    history_seasons = 2,
    dynamics_col = :match_month,
    warmup_period = 0,
    stop_early = true # Just 1 fold for smoke testing
)

# 4. Sampler Config (Using NUTS to properly explore the Hierarchical Copula)
sampler_config = BayesianFootball.Samplers.NUTSConfig(
            500,  # samples
            4,    # chains
            200,  # warmup
            0.65, # accept_rate
            10,   # max_depth
            Samplers.UniformInit(-2, 2),
            true  # show_progress
        )

samples=train_cfg = BayesianFootball.Training.Independent(
    parallel = true,
    max_concurrent_splits = 4
)

training_config = BayesianFootball.Training.TrainingConfig(sampler_config, train_cfg, nothing, false)

# 5. Experiment Setup
config = BayesianFootball.Experiments.ExperimentConfig(
    name = "hierarchical_copula_model_test",
    model = copula_model, 
    splitter = cv_config,
    training_config = training_config,
    save_dir = "./data/hierarchical_copula_test/"
)

task = BayesianFootball.Experiments.ExperimentTask(ds, config)

# 6. Run Experiment
println(">>> Running Hierarchical Copula Experiment...")
results = BayesianFootball.Experiments.run_experiment(task)

# 7. Extract Results
println(">>> Extracting Chains...")
chains = BayesianFootball.Experiments.Diagnostics.extract_chains(ds, results)

println(">>> Chain Summary:")
display(describe(chains.df))

println("\nAll done! Model works dynamically from src!")


describe(results.training_results[3][1])



conv_diag_all = BayesianFootball.Experiments.Diagnostics.check_convergence(chains)
stab_diag_all = BayesianFootball.Experiments.Diagnostics.check_stability(chains)

chains.df


const PreGame = BayesianFootball.Models.PreGame
const Features = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Diagnostics = BayesianFootball.Experiments.Diagnostics
const Evaluation = BayesianFootball.Evaluation
const BackTesting = BayesianFootball.BackTesting
const Signals = BayesianFootball.Signals



metrics = [
    Evaluation.RQR(),
    Evaluation.LogLoss(), 
    Evaluation.CRPS(), 
    Evaluation.GLMEdge()
]
master_eval_df = Evaluation.evaluate_experiments(metrics, [results], ds)



#=
julia> master_eval_df = Evaluation.evaluate_experiments(metrics, [results], ds)
============================================================
 🚀 Running Batch Evaluation...
============================================================
Progress: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:02
┌ Warning: Error evaluating $(typeof(metric)) for $model_name: $e
└ @ BayesianFootball.Evaluation ~/BayesianFootball/src/evaluation/batch_runner.jl:40
Running Inference on 84 matches...
┌ Warning: Error evaluating $(typeof(metric)) for $model_name: $e
└ @ BayesianFootball.Evaluation ~/BayesianFootball/src/evaluation/batch_runner.jl:40
┌ Warning: Error evaluating $(typeof(metric)) for $model_name: $e
└ @ BayesianFootball.Evaluation ~/BayesianFootball/src/evaluation/batch_runner.jl:40
Running Inference on 84 matches...
┌ Warning: Error evaluating $(typeof(metric)) for $model_name: $e
└ @ BayesianFootball.Evaluation ~/BayesianFootball/src/evaluation/batch_runner.jl:40
❌ Failed (Partial or complete failure)
0×0 DataFrame
=#



latents = Experiments.extract_oos_predictions(ds, results)




#=
julia> latents = Experiments.extract_oos_predictions(ds, results)
84×8 DataFrame
 Row │ match_id  r_a                                r_h                                true_xg_a                          true_xg_h                          κ_frank                            λ_a                                λ_h                               
     │ Any       Any                                Any                                Any                                Any                                Any                                Any                                Any                               
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ 15238017  [16.565, 18.9806, 20.4116, 23.60…  [24.0251, 16.2526, 32.5992, 25.7…  [0.912269, 1.38725, 0.959899, 1.…  [1.27256, 1.38802, 1.29578, 1.69…  [0.187081, 0.807924, 1.04765, 1.…  [0.912269, 1.38725, 0.959899, 1.…  [1.27256, 1.38802, 1.29578, 1.69…
   2 │ 15238008  [16.565, 18.9806, 20.4116, 23.60…  [24.0251, 16.2526, 32.5992, 25.7…  [1.33318, 1.41043, 0.937831, 0.9…  [1.17264, 1.73396, 1.15257, 1.25…  [0.509763, 0.394063, 0.716542, 3…  [1.33318, 1.41043, 0.937831, 0.9…  [1.17264, 1.73396, 1.15257, 1.25…
   3 │ 15238007  [16.565, 18.9806, 20.4116, 23.60…  [24.0251, 16.2526, 32.5992, 25.7…  [0.913676, 1.35779, 1.05806, 0.9…  [1.235, 1.55036, 2.03978, 1.1855…  [0.311461, 1.0577, 0.976246, 1.4…  [0.913676, 1.35779, 1.05806, 0.9…  [1.235, 1.55036, 2.03978, 1.1855…
   4 │ 15238023  [16.565, 18.9806, 20.4116, 23.60…  [24.0251, 16.2526, 32.5992, 25.7…  [1.11968, 1.35508, 0.796747, 1.2…  [1.36573, 1.22445, 1.17864, 1.33…  [0.38719, 1.28823, 1.0489, 2.185…  [1.11968, 1.35508, 0.796747, 1.2…  [1.36573, 1.22445, 1.17864, 1.33…
   5 │ 15238018  [16.565, 18.9806, 20.4116, 23.60…  [24.0251, 16.2526, 32.5992, 25.7…  [1.37143, 1.47404, 1.12754, 1.28…  [1.11732, 1.68439, 1.69145, 1.34…  [0.524245, 0.862879, 0.850199, 2…  [1.37143, 1.47404, 1.12754, 1.28…  [1.11732, 1.68439, 1.69145, 1.34…
   6 │ 15238026  [16.565, 18.9806, 20.4116, 23.60…  [24.0251, 16.2526, 32.5992, 25.7…  [0.923496, 0.985372, 1.15366, 0.…  [1.51355, 1.5784, 1.53687, 1.380…  [0.669208, 1.22892, 1.01884, 2.4…  [0.923496, 0.985372, 1.15366, 0.…  [1.51355, 1.5784, 1.53687, 1.380…
   7 │ 15238009  [16.565, 18.9806, 20.4116, 23.60…  [24.0251, 16.2526, 32.5992, 25.7…  [1.16342, 1.07419, 1.05586, 1.30…  [1.26277, 1.22414, 1.11143, 0.82…  [0.479526, 1.03516, 0.966908, 3.…  [1.16342, 1.07419, 1.05586, 1.30…  [1.26277, 1.22414, 1.11143, 0.82…
   8 │ 15238021  [16.565, 18.9806, 20.4116, 23.60…  [24.0251, 16.2526, 32.5992, 25.7…  [1.46534, 0.900864, 0.974121, 1.…  [1.27133, 1.05489, 1.14298, 1.42…  [0.769838, 1.42386, 0.875111, 3.…  [1.46534, 0.900864, 0.974121, 1.…  [1.27133, 1.05489, 1.14298, 1.42…
   9 │ 15238020  [16.565, 18.9806, 20.4116, 23.60…  [24.0251, 16.2526, 32.5992, 25.7…  [1.04325, 1.02065, 1.16418, 0.88…  [1.29036, 1.36362, 1.82067, 1.03…  [0.419898, 1.21548, 1.00863, 2.8…  [1.04325, 1.02065, 1.16418, 0.88…  [1.29036, 1.36362, 1.82067, 1.03…
=#


ledger = BackTesting.run_backtest(
    ds, 
    results, 
    [Signals.BayesianKelly()]; 
    market_config = BayesianFootball.Data.Markets.DEFAULT_MARKET_CONFIG
)

tearsheet = BackTesting.generate_tearsheet(ledger)

println("\n>>> Backtest Comparison Summary:")
cols_to_show = [:model_name, :selection, :opportunities, :activity_pct, :bets_placed, :turnover, :profit, :roi_pct, :win_rate_pct]
show(tearsheet[:, cols_to_show], allrows=true)



#=
julia> ledger = BackTesting.run_backtest(
           ds, 
           results, 
           [Signals.BayesianKelly()]; 
           market_config = BayesianFootball.Data.Markets.DEFAULT_MARKET_CONFIG
       )
Running Inference on 84 matches...
ERROR: TaskFailedException

    nested task error: MethodError: no method matching extract_params(::BayesianFootball.Models.PreGame.DynamicCopulaGoalsTimeDecayModel, ::DataFrameRow{…})
    The function `extract_params` exists, but no method is defined for this combination of argument types.
    
    Closest candidates are:
      extract_params(::BayesianFootball.TypesInterfaces.AbstractPoissonModel, ::Any)
       @ BayesianFootball ~/BayesianFootball/src/predictions/score_computation/poisson.jl:6
      extract_params(::BayesianFootball.TypesInterfaces.AbstractBivariatePoissonModel, ::Any)
       @ BayesianFootball ~/BayesianFootball/src/predictions/score_computation/bivariate_poisson.jl:6
      extract_params(::BayesianFootball.TypesInterfaces.AbstractDixonColesModel, ::Any)
       @ BayesianFootball ~/BayesianFootball/src/predictions/score_computation/dixoncoles.jl:8
      ...
    
    Stacktrace:
     [1] predict_row(model::BayesianFootball.Models.PreGame.DynamicCopulaGoalsTimeDecayModel, row::DataFrameRow{…}, markets::Vector{…})
       @ BayesianFootball.Predictions ~/BayesianFootball/src/predictions/inference.jl:16
     [2] macro expansion
       @ ~/BayesianFootball/src/predictions/inference.jl:49 [inlined]
     [3] #14
       @ ./threadingconstructs.jl:276 [inlined]
     [4] #12
       @ ./threadingconstructs.jl:243 [inlined]
     [5] (::Base.Threads.var"#threading_run##0#threading_run##1"{BayesianFootball.Predictions.var"#12#13"{…}, Int64})()
       @ Base.Threads ./threadingconstructs.jl:177

...and 15 more exceptions.

Stacktrace:
  [1] threading_run(fun::BayesianFootball.Predictions.var"#12#13"{BayesianFootball.Predictions.var"#14#15"{Vector{…}, Vector{…}, Vector{…}, BayesianFootball.Models.PreGame.DynamicCopulaGoalsTimeDecayModel, UnitRange{…}}}, static::Bool)
    @ Base.Threads ./threadingconstructs.jl:196
  [2] macro expansion
    @ ./threadingconstructs.jl:213 [inlined]
  [3] model_inference(latents::BayesianFootball.Experiments.LatentStates; market_config::BayesianFootball.Data.Markets.MarketConfig)
    @ BayesianFootball.Predictions ~/BayesianFootball/src/predictions/inference.jl:48
  [4] model_inference
    @ ~/BayesianFootball/src/predictions/inference.jl:32 [inlined]
  [5] _process_single_experiment(exp_res::BayesianFootball.Experiments.ExperimentResults, ds::BayesianFootball.Data.DataStore, signals::Vector{…}, market_df::DataFrame, market_config::BayesianFootball.Data.Markets.MarketConfig; odds_column::Symbol)
    @ BayesianFootball.BackTesting ~/BayesianFootball/src/backtesting/processor.jl:56
  [6] _process_single_experiment
    @ ~/BayesianFootball/src/backtesting/processor.jl:42 [inlined]
  [7] worker_fn
    @ ~/BayesianFootball/src/backtesting/processor.jl:29 [inlined]
  [8] mapreduce_first
    @ ./reduce.jl:413 [inlined]
  [9] _mapreduce(f::BayesianFootball.BackTesting.var"#worker_fn#2"{BayesianFootball.Data.Markets.MarketConfig, Symbol, BayesianFootball.Data.DataStore, Vector{…}}, op::typeof(vcat), ::IndexLinear, A::Vector{BayesianFootball.Experiments.ExperimentResults})
    @ Base ./reduce.jl:424
 [10] _mapreduce_dim
    @ ./reducedim.jl:334 [inlined]
 [11] mapreduce
    @ ./reducedim.jl:326 [inlined]
 [12] #run_backtest#1
    @ ~/BayesianFootball/src/backtesting/processor.jl:35 [inlined]
 [13] run_backtest
    @ ~/BayesianFootball/src/backtesting/processor.jl:13 [inlined]
 [14] #run_backtest#7
    @ ~/BayesianFootball/src/backtesting/processor.jl:87 [inlined]
 [15] top-level scope
    @ REPL[33]:1
Some type information was truncated. Use `show(err)` to see complete types.
=#


tearsheet = BackTesting.generate_tearsheet(ledger)


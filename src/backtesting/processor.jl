"""
    run_backtest(ds::DataStore, experiments, signals; market_config=nothing)

The Master Orchestrator. 
Sequentially processes each experiment to avoid thread oversubscription, 
relying on the low-level modules (Inference/Signals) to handle parallelism.

# Arguments
- `ds`: The DataStore containing odds and match outcomes.
- `experiments`: A single `ExperimentResults` or a `Vector` of them.
- `signals`: A vector of signal strategies to test on these models.
"""
function run_backtest(
    ds::DataStore, 
    experiments::Vector{ExperimentResults}, 
    signals::Vector{<:AbstractSignal};
    market_config = nothing,
    odds_column::Symbol =:odds_close
    )

    # 1. Prepare Environment (Once)
    # We prep the market data once so we don't re-calculate it for every model
    market_data = Data.prepare_market_data(ds)
    
    # 2. Define the Worker (Closure)
    # This captures the common data (ds, signals, market_data) 
    # and takes only the specific experiment 'exp' as input.
    worker_fn(exp) = _process_single_experiment(exp, ds, signals, market_data.df, market_config; 
                                                odds_column=odds_column)

    # 3. Execute
    # mapreduce is efficient here: it processes one experiment, gets the DataFrame, 
    # and VCATs it into the growing master ledger.
    master_df = mapreduce(worker_fn, vcat, experiments)

    return BacktestLedger(master_df)
end

# --- Internal Worker ---

function _process_single_experiment(
    exp_res::ExperimentResults, 
    ds::DataStore, 
    signals::Vector{<:AbstractSignal}, 
    market_df::DataFrame,
    market_config;
    odds_column::Symbol=:odds_close
)
    # A. Experiment Bridge: Get Latents
    # (This might trigger heavy computation/allocations)
    latents = extract_oos_predictions(ds, exp_res)

    # B. Prediction View: Get Probabilities
    # (This likely uses Threads inside to compute scores for chains)
    ppd = model_inference(latents; market_config=market_config)

    # C. Signal Agent: Get Decisions
    # (This might use Threads to process thousands of rows)
    sig_result = process_signals(ppd, market_df, signals; odds_column=odds_column)
    
    # D. Enrichment
    # Add metadata so we can distinguish this model's bets in the big ledger
    df = sig_result.df
    
    m_name = model_name(exp_res.config.model)
    m_params = model_parameters(exp_res.config.model)

    df.model_name = fill(m_name, nrow(df))
    df.model_parameters = fill(m_params, nrow(df))
    
    return df
end

# Convenience method for a single experiment input
run_backtest(ds::DataStore, exp::ExperimentResults, sigs; kwargs...) = 
    run_backtest(ds, [exp], sigs; kwargs...)


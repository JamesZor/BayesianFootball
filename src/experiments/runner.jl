# src/experiments/runner.jl

module ExperimentRunner

using ..ExperimentTypes
using ..Data
using ..Data.Splitting # Your TimeSeriesSplits iterator
using ..Features
using ..Models
using ..Sampling
using Turing, DataFrames
using JLD2, JSON3, ProgressMeter # For I/O and progress bars

export run_experiment

# --- I/O Helpers (can be in a separate io.jl file) ---

function save_artifacts(path, chains, goal_preds, config)
    mkpath(path)
    jldsave(joinpath(path, "chains.jld2"); chains)
    jldsave(joinpath(path, "goal_predictions.jld2"); goal_predictions=goal_preds)
    JSON3.write(joinpath(path, "run_config.json"), config)
    println("✅ Artifacts saved to $path")
end

# --- The Main Dispatcher ---

"""
The main entry point for the Trainer.
It takes an Experiment config and the DataStore.
"""
function run_experiment(exp::Experiment, data_store::DataStore; base_save_path::String)
    exp_path = joinpath(base_save_path, exp.name)
    mkpath(exp_path)
    
    # Save the master config
    JSON3.write(joinpath(exp_path, "master_experiment_config.json"), exp)
    
    println("--- 🚀 RUNNING EXPERIMENT: $(exp.name) ---")
    
    # This function dispatches based on the splitter type
    _run_splitter(exp, exp.splitter, data_store.matches, exp_path)
end

# --- Runner for StaticSplit (runs ONCE) ---

function _run_splitter(exp::Experiment, splitter::StaticSplit, all_matches_df::DataFrame, exp_path::String)
    println("Running as a single static split...")
    
    # 1. Get the training data
    train_df = filter(row -> row.season in splitter.train_seasons, all_matches_df)
    
    # 2. Run the core pipeline ONCE
    # The results are saved in the main experiment path
    _run_pipeline_step(exp, train_df, exp_path, split_name="static_run")
end

# --- Runner for ExpandingWindowCV (runs in a LOOP) ---

function _run_splitter(exp::Experiment, splitter::ExpandingWindowCV, all_matches_df::DataFrame, exp_path::String)
    println("Running as expanding window CV...")
    
    # 1. Create your TimeSeriesSplits iterator
    cv_iterator = Data.Splitting.TimeSeriesSplits(
        all_matches_df, 
        splitter.base_seasons, 
        splitter.target_seasons, 
        splitter.round_col, 
        splitter.ordering
    )
    
    # 2. Loop over each split
    println("Found $(length(cv_iterator)) CV splits.")
    @showprogress "Running CV Splits..." for (split_idx, (train_view, round_info)) in enumerate(cv_iterator)
        
        println("\n>>> Starting CV Split $split_idx: $round_info")
        
        # 3. Create a unique sub-folder for this split's results
        split_save_path = joinpath(exp_path, "split_$(lpad(split_idx, 3, '0'))_$(replace(round_info, "/" => "-"))")
        
        # 4. Run the core pipeline for THIS split
        _run_pipeline_step(
            exp, 
            DataFrame(train_view), # Use the data for this split
            split_save_path,
            split_name=round_info
        )
    end
end


# --- The Core Pipeline (f -> g -> h_goals) ---

"""
This is the core function of the Trainer. It runs the pipeline
for a SINGLE DataFrame and saves the two key artifacts.
"""
function _run_pipeline_step(exp::Experiment, train_df::DataFrame, save_path::String; split_name::String)
    
    # --- 1. Morphism f: (D_i, M) -> F_i ---
    # (Uses the refactored Features.jl)
    features = Features.create_features(exp.model, train_df)
    
    # --- 2. Build TRAINING Model ---
    # (Uses the refactored Models.jl)
    turing_model = Models.build_turing_model(exp.model, features)
    
    # --- 3. Morphism g: (F_i, M, Config_s) -> C_params ---
    println("Sampling parameters...")
    chains_params = Sampling.train(turing_model, exp.sampler_config)
    
    # --- 4. Build PREDICTION Model ---
    # We predict on the same data we just trained on.
    # (Uses the *other* method of our refactored Models.jl)
    turing_pred_model = Models.build_turing_model(exp.model, chains_params, features)
    
    # --- 5. Morphism h_goals: (M, C_params, F_i) -> C_goals ---
    println("Generating goal predictions...")
    chains_goals = Turing.predict(turing_pred_model)
    
    # --- 6. Save Artifacts ---
    save_artifacts(save_path, chains_params, chains_goals, exp)
    
    println(">>> Finished CV Split: $split_name")
end

end # module ExperimentRunner

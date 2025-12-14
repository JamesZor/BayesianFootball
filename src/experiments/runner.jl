# src/experiments/runner.jl

# src/experiments/runner.jl

using JLD2
using Dates
using ..Data
using ..Features
using ..Training

export run_experiment

"""
    run_experiment(data_store, config::ExperimentConfig; force_name=nothing)

Orchestrates the full experiment pipeline:
1. Creates Global Vocabulary (Model + Data)
2. Generates Data Splits (Splitter + Data)
3. Builds Feature Sets (Splits + Vocab + Model)
4. Executes Training (FeatureSets + TrainingConfig)
5. Saves Artifacts (Config, Vocabulary, Results)

Returns an `ExperimentResults` struct.
"""
function run_experiment(data_store, config::ExperimentConfig; force_name=nothing)
    # 1. Setup Paths & Logging
    exp_name = isnothing(force_name) ? config.name : force_name
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    # Folder structure: ./experiments/ExpName_Timestamp/
    save_path = joinpath(config.save_dir, "$(exp_name)_$(timestamp)")
    mkpath(save_path)

    println("════════════════════════════════════════════════════════════")
    println("🚀 STARTING EXPERIMENT: $exp_name")
    println("📂 Save Path: $save_path")
    println("════════════════════════════════════════════════════════════")

    # 2. Create Vocabulary (Global)
    println("\n[1/4] 📖 Creating Vocabulary...")
    # NOTE: The vocabulary creation depends on the Model and the DataStore
    vocabulary = Features.create_vocabulary(data_store, config.model)
    # println("      Vocabulary size: $(length(vocabulary.teams)) teams")

    # 3. Create Data Splits
    println("\n[2/4] ✂️  Generating Data Splits...")
    # This uses the specific AbstractSplitter in the config (ExpandingWindow, Static, etc.)
    data_splits = Data.create_data_splits(data_store, config.splitter)
    println("      Generated $(length(data_splits)) splits.")

    # 4. Create Feature Sets
    println("\n[3/4] 🏗️  Building Feature Sets...")
    # Uses the refactored Features module to auto-generate required features
    feature_sets = Features.create_features(
        data_splits, 
        vocabulary, 
        config.model, 
        config.splitter 
    )

    # 5. Execute Training
    println("\n[4/4] 🏋️  Executing Training Strategy...")
    # Dispatch to the Training module (handles Parallel vs Serial internally)
    training_results = Training.train(
        config.model, 
        config.training_config, 
        feature_sets
    )

    # 6. Save Artifacts
    println("\n💾 Saving Artifacts...")
    
    # A. Save the Results Object (JLD2)
    # We construct the ExperimentResults struct
    results = ExperimentResults(
        config,
        training_results,
        vocabulary,
        save_path
    )
    
    # jldsave(joinpath(save_path, "results.jld2"); results)
    #
    # # B. Save Config as JSON (for human readability/metadata)
    # # We might need to handle struct serialization carefully, but JSON3 usually handles it well
    # # or we convert it to a Dict first if needed.
    # open(joinpath(save_path, "config.json"), "w") do io
    #     JSON3.pretty(io, config)
    # end
    #
    println("✅ Experiment Complete!")
    println("════════════════════════════════════════════════════════════")

    return results
end

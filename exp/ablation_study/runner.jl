# exp/ablation_study/runner.jl

using Revise
using BayesianFootball

using DataFrames
using Statistics
using ThreadPinning
using LinearAlgebra
using Logging

# Performance tuning for MCMC
pinthreads(:cores)
threadinfo()
BLAS.set_num_threads(1)

# Include the models configuration file
# include("models.jl")

function run_all()
    # 1. Get Configs
    ds, configs = get_ablation_configs()
    
    println("============================================================")
    println(" Starting Experiment Suite: Forward-Step Ablation Study")
    println("    > Found $(length(configs)) configurations.")
    println("    > Save Dir: $(configs[1].save_dir)")
    println("============================================================")

    # 2. Loop and Run
    for (i, conf) in enumerate(configs)
        println("\n\n>>> RUNNING MODEL [$i/$(length(configs))]: $(conf.name)")
        
        # Disable Info logs to keep the Progress Bar clean during sampling
        disable_logging(Logging.Info) 
        
        try
            # Run the cross-validation
            results = Experiments.run_experiment(ds, conf)
            
            # Re-enable logging to confirm save
            disable_logging(Logging.Debug) 
            
            # SAVE the artifacts and backtest results
            Experiments.save_experiment(results)
            println("✅ Model $(conf.name) completed and saved successfully.")
            
        catch e
            disable_logging(Logging.Debug)
            @error "Failed to run $(conf.name): $e"
            # We continue to the next model even if one fails
        end
    end
    
    println("\n Ablation Study completely finished.")
end

# Execute the ablation pipeline
run_all()

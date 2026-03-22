# exp/market_scot/runner.jl
using BayesianFootball

using DataFrames
using Statistics
using ThreadPinning
using LinearAlgebra
using Logging
pinthreads(:cores)

# Load the definitions from the sibling file
include("./exp/market_scot/models.jl")

function run_all()
    # 1. Get Configs
    ds, configs = get_grw_basics_configs()
    
    println("🚀 Starting Experiment Suite: GRW Basics")
    println("   > Found $(length(configs)) configurations.")
    println("   > Save Dir: $(configs[1].save_dir)")
    println("="^60)

    # 2. Loop and Run
    for (i, conf) in enumerate(configs)
        println("\n\n>>> RUNNING MODEL [$i/$(length(configs))]: $(conf.name)")
        
        # Disable Info logs to keep the Progress Bar clean
        # disable_logging(Logging.Info) 
        
        try
            # Run
            results = Experiments.run_experiment(ds, conf)
            
            # Re-enable logging to confirm save
            # disable_logging(Logging.Debug) 
            
            # SAVE (Important!)
            Experiments.save_experiment(results)
            
        catch e
            # disable_logging(Logging.Debug)
            @error "Failed to run $(conf.name): $e"
            # We continue to the next model even if one fails
        end
    end
    
    println("\n✅ All experiments finished.")
end

# Execute
run_all()

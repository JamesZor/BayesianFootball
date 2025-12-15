# src/training/training-module.jl

module Training

# --- Dependencies ---
using Turing # For Chains type
using ProgressMeter # Optional for progress bar
using Base.Threads # For threading

# Import types/modules from other parts of BayesianFootball
using ..Samplers: AbstractSamplerConfig, run_sampler # Use the renamed sampler module
using ..Models: AbstractFootballModel # Abstract model type
using ..Features: FeatureSet # FeatureSet type
# We need access to the model building API
using ..Models.PreGame: build_turing_model # Assuming PreGame for now

# --- Exports ---
# Configs
export TrainingConfig, AbstractExecutionStrategy, Independent, SequentialPriorUpdate 
# Main function
export train

# --- Configuration Structs ---

# Strategy defines *how* multiple splits are handled
abstract type AbstractExecutionStrategy end

"""
    Independent(; parallel=false, max_concurrent_splits=Threads.nthreads() ÷ 2)

Training strategy where each data split is processed independently.

# Arguments
- `parallel::Bool`: Set `true` to use multi-threading.
- `max_concurrent_splits::Int`: Limits the number of splits processed concurrently when `parallel=true`. 
                                Defaults to half the available Julia threads (heuristic for physical cores).
"""
struct Independent <: AbstractExecutionStrategy
    parallel::Bool 
    max_concurrent_splits::Int
end
# Adjust default constructor
Independent(; parallel::Bool=false, max_concurrent_splits::Int = max(1, Threads.nthreads() ÷ 2)) = Independent(parallel, max_concurrent_splits)


"""
    SequentialPriorUpdate(; prior_update_method=:MvNormalFit, ...)

Placeholder: Training strategy where the posterior from one split informs the prior for the next.
"""
struct SequentialPriorUpdate <: AbstractExecutionStrategy
    prior_update_method::Symbol # e.g., :MvNormalFit
    # ... other parameters ...
end
# Basic constructor for placeholder
SequentialPriorUpdate(; prior_update_method=:MvNormalFit) = SequentialPriorUpdate(prior_update_method)

"""
    TrainingConfig(sampler, strategy)

Overall configuration for the training process.

# Arguments
- `sampler::AbstractSamplerConfig`: Specifies the core sampling algorithm (e.g., `NUTSConfig`).
- `strategy::AbstractExecutionStrategy`: Defines how multiple splits are handled (e.g., `Independent`, `SequentialPriorUpdate`).
"""
struct TrainingConfig
    sampler::AbstractSamplerConfig
    strategy::AbstractExecutionStrategy
end


# --- Core Training Functions (Overloaded) ---

"""
    train(model, config, feature_set)

Trains the model on a single FeatureSet using the sampler specified in the config.
Internal function, usually called by the multi-split version.
"""
function train(
    model::AbstractFootballModel, 
    config::TrainingConfig, 
    feature_set::FeatureSet
)#::Turing.Chains # Return type depends on sampler, might return Optimization result etc.

    println("Dispatching to train(model, config, ::FeatureSet)...")
    
    # Build the Turing model instance
    # NOTE: This currently assumes PreGame. We might need to make this more generic later
    # if we add InGame models requiring different build functions.
    turing_model = build_turing_model(model, feature_set) 
    
    # Run the sampler using the config's sampler part
    result = run_sampler(turing_model, config.sampler)
    
    return result 
end


"""
    train(model, config, feature_sets_with_metadata)

Trains the model according to the strategy specified in the `TrainingConfig`.
Handles iterating over multiple data splits, potentially in parallel or sequentially.
"""
function train(
    model::AbstractFootballModel, 
    config::TrainingConfig, 
    feature_sets_with_metadata::Vector{Tuple{FeatureSet, String}}
)#::Vector{Tuple{Any, String}} 

    println("Dispatching to train(model, config, ::Vector{Tuple{FeatureSet, String}})...")
    num_splits = length(feature_sets_with_metadata)
    all_results = Vector{Tuple{Any, String}}(undef, num_splits) 

    # --- Strategy Dispatch ---
    strategy = config.strategy # Local variable for easier access

    if strategy isa Independent
        if strategy.parallel && num_splits > 1 # Only use threading if parallel is true and there's work to parallelize
            # --- Independent Parallel Execution with Semaphore ---

            # Determine actual concurrency limit
            concurrency = min(num_splits, strategy.max_concurrent_splits, Threads.nthreads()) 
            if concurrency < strategy.max_concurrent_splits
               @warn "Limiting concurrency to $concurrency (min of num_splits, max_concurrent_splits, and nthreads)."
            end
            println("Starting Independent training for $num_splits splits using up to $concurrency concurrent tasks...")

            semaphore = Base.Semaphore(concurrency)
            tasks = Vector{Task}(undef, num_splits)

            # Use @sync to wait for all spawned tasks
            @sync begin 
                for i in 1:num_splits
                    tasks[i] = Threads.@spawn begin
                        # Acquire semaphore: blocks if > concurrency tasks are running
                        Base.acquire(semaphore) 
                        try
                            feature_set_i, metadata = feature_sets_with_metadata[i]
                            println("--- Starting Split $i: $metadata on Thread $(Threads.threadid()) ---")

                            # Call the single FeatureSet method
                            result_i = train(model, config, feature_set_i) 

                            all_results[i] = (result_i, metadata)
                            println("Finished Split $i on Thread $(Threads.threadid()).")
                        finally
                            # IMPORTANT: Release semaphore even if an error occurs
                            Base.release(semaphore) 
                        end
                    end # end @spawn
                end # end for
            end # end @sync

            println("\nThreaded training complete!")

        else 
            # --- Independent Sequential Execution ---
            println("🚀 Starting Independent training for $num_splits splits sequentially...")
            @showprogress "Training splits..." for i in 1:num_splits
                feature_set_i, metadata = feature_sets_with_metadata[i]
                println("\n--- Training Split $i: $metadata ---")
                
                # Call the single FeatureSet method
                result_i = train(model, config, feature_set_i) 
                
                all_results[i] = (result_i, metadata)
                println("✅ Finished Split $i.")
            end
             println("\nSequential training complete!")
        end
        return all_results


    elseif config.strategy isa SequentialPriorUpdate
        # --- Sequential Prior Update Execution (Placeholder) ---
        println("🚀 Starting Sequential training with prior updates (Placeholder)...")
        @warn "SequentialPriorUpdate strategy is not yet fully implemented."
        
        current_prior_info = nothing 
        for i in 1:num_splits
             feature_set_i, metadata = feature_sets_with_metadata[i]
             println("\n--- Training Split $i: $metadata ---")
             
             # TODO: Modify build_turing_model to accept prior_info
             # turing_model = build_turing_model(model, feature_set_i, prior_info=current_prior_info) 
             turing_model = build_turing_model(model, feature_set_i) # Using default for now
             
             result_i = run_sampler(turing_model, config.sampler)
             all_results[i] = (result_i, metadata)

             if i < num_splits && result_i isa Turing.Chains
                # TODO: Implement construct_prior_from_chains
                 # current_prior_info = construct_prior_from_chains(result_i, config.strategy) 
                 println("--- Would construct prior for next step here ---")
             elseif !(result_i isa Turing.Chains)
                 @warn "Sampler did not return Chains. Cannot update priors for next step. Stopping sequential update."
                 break # Cannot continue SMC without chains
             end
             println("✅ Finished Split $i.")
        end
        println("\nSequential training complete (Placeholder)!")
        return all_results
    
    else
        error("Unsupported execution strategy: $(typeof(config.strategy))")
    end
end


"""
    train(model, config, feature_sets)

Generic wrapper for training on a vector of (FeatureSet, Metadata) tuples.
Accepts ANY metadata type 'M' (String, SplitMetaData, etc.).
"""
function train(
    model::AbstractFootballModel, 
    config::TrainingConfig, 
    feature_sets_with_metadata::Vector{<:Tuple{FeatureSet, M}}
)::Vector{Tuple{Any, M}} where M

    println("Dispatching to train(model, config, ::Vector{Tuple{FeatureSet, $M}})...")
    num_splits = length(feature_sets_with_metadata)
    
    # Pre-allocate generic result vector
    all_results = Vector{Tuple{Any, M}}(undef, num_splits) 

    strategy = config.strategy 

    if strategy isa Independent
        if strategy.parallel && num_splits > 1 
            # --- Independent Parallel Execution ---
            concurrency = min(num_splits, strategy.max_concurrent_splits, Threads.nthreads()) 
            println("Starting Independent training for $num_splits splits using up to $concurrency concurrent tasks...")

            semaphore = Base.Semaphore(concurrency)
            tasks = Vector{Task}(undef, num_splits)

            @sync begin 
                for i in 1:num_splits
                    tasks[i] = Threads.@spawn begin
                        Base.acquire(semaphore) 
                        try
                            feature_set_i, metadata = feature_sets_with_metadata[i]
                            println("--- Starting Split $i on Thread $(Threads.threadid()) ---")

                            result_i = train(model, config, feature_set_i) 

                            all_results[i] = (result_i, metadata)
                            println("Finished Split $i on Thread $(Threads.threadid()).")
                        finally
                            Base.release(semaphore) 
                        end
                    end 
                end 
            end 
            println("\nThreaded training complete!")

        else 
            # --- Independent Sequential Execution ---
            println("🚀 Starting Independent training for $num_splits splits sequentially...")
            @showprogress "Training splits..." for i in 1:num_splits
                feature_set_i, metadata = feature_sets_with_metadata[i]
                
                result_i = train(model, config, feature_set_i) 
                
                all_results[i] = (result_i, metadata)
            end
             println("\nSequential training complete!")
        end
        return all_results

    elseif config.strategy isa SequentialPriorUpdate
        # --- Sequential Prior Update Execution (Placeholder) ---
        println("🚀 Starting Sequential training with prior updates (Placeholder)...")
        @warn "SequentialPriorUpdate strategy is not yet fully implemented."
        
        current_prior_info = nothing 
        for i in 1:num_splits
             feature_set_i, metadata = feature_sets_with_metadata[i]
             println("\n--- Training Split $i: $metadata ---")
             
             # TODO: Modify build_turing_model to accept prior_info
             # turing_model = build_turing_model(model, feature_set_i, prior_info=current_prior_info) 
             turing_model = build_turing_model(model, feature_set_i) # Using default for now
             
             result_i = run_sampler(turing_model, config.sampler)
             all_results[i] = (result_i, metadata)

             if i < num_splits && result_i isa Turing.Chains
                # TODO: Implement construct_prior_from_chains
                 # current_prior_info = construct_prior_from_chains(result_i, config.strategy) 
                 println("--- Would construct prior for next step here ---")
             elseif !(result_i isa Turing.Chains)
                 @warn "Sampler did not return Chains. Cannot update priors for next step. Stopping sequential update."
                 break # Cannot continue SMC without chains
             end
             println("✅ Finished Split $i.")
        end
        println("\nSequential training complete (Placeholder)!")
        return all_results
    
    else
        error("Unsupported execution strategy: $(typeof(config.strategy))")
    end
end


# --- Placeholder for SMC prior construction ---
# function construct_prior_from_chains(chains::Turing.Chains, strategy::SequentialPriorUpdate)
#     # Placeholder logic
#     println("    (Placeholder) Constructing prior using method: $(strategy.prior_update_method)")
#     return Dict{Symbol, Any}() 
# end


end # module Training

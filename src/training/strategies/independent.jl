# src/training/strategies/independent.jl

using ProgressMeter
using Base.Threads

export train_independent

"""
    train_independent(model, config, feature_sets)

Executes training either sequentially or in parallel, respecting checkpoints.
"""
function train_independent(model, config, feature_sets)
    num_splits = length(feature_sets)
    strategy = config.strategy
    
    # 1. Resume / Load State
    # 'results' will contain loaded data for finished splits, 'nothing' for pending ones
    results = get_checkpoint_status(config.checkpoint_dir, num_splits)
    
    # Identify pending indices
    pending_indices = findall(isnothing, results)
    
    if isempty(pending_indices)
        println("✅ All splits already completed via checkpoints.")
        return Vector{Tuple{Any, Any}}(results) # Cast to expected type logic below handles tuples
    end
    
    # 2. Define the Work Unit (The "Kernel")
    function process_split(i)
        feature_set, metadata = feature_sets[i]
        
        # A. Run the core training logic
        # Note: We rely on the generic 'train' overload in the parent module to call the sampler
        turing_result = Training.train(model, config, feature_set) 
        
        # B. Checkpoint immediately
        if !isnothing(config.checkpoint_dir)
            # We save the tuple (result, metadata) to maintain consistency
            save_split_checkpoint(config.checkpoint_dir, i, (turing_result, metadata))
        end
        
        return (turing_result, metadata)
    end

    # 3. Execution Logic (Parallel vs Sequential)
    if strategy.parallel && length(pending_indices) > 1
        
        concurrency = min(length(pending_indices), strategy.max_concurrent_splits, Threads.nthreads())
        println("Starting Parallel Training: $(length(pending_indices)) splits remaining (Concurrency: $concurrency)...")
        
        semaphore = Base.Semaphore(concurrency)
        
        # Use a lock for thread-safe assignment to the 'results' vector
        results_lock = ReentrantLock()
        
        @sync for i in pending_indices
            Threads.@spawn begin
                Base.acquire(semaphore)
                try
                    res = process_split(i)
                    lock(results_lock) do
                        results[i] = res
                    end
                    println("   [Thread $(Threads.threadid())] Finished Split $i")
                catch e
                    @error "Error in Split $i: $e"
                    # We do not rethrow to allow other splits to finish/save
                finally
                    Base.release(semaphore)
                end
            end
        end
        
    else
        # Sequential Fallback
        println("🚀 Starting Sequential Training: $(length(pending_indices)) splits remaining...")
        
        @showprogress for i in pending_indices
            results[i] = process_split(i)
        end
    end
    
    # 4. Cleanup (Optional)
    if !isnothing(config.checkpoint_dir) && config.cleanup_checkpoints
        cleanup_checkpoints(config.checkpoint_dir, num_splits)
    end
    
    # Ensure type stability for return
    return TrainingResults(results)
end

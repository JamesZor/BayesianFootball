# src/training/strategies/independent.jl

using Base.Threads
using Dates
using Printf

export train_independent

"""
    train_independent(model, config, feature_sets)

Executes training either sequentially or in parallel, respecting checkpoints.
"""
function train_independent(model, config, feature_sets)
    num_splits = length(feature_sets)
    strategy = config.strategy
    
    # 1. Resume / Load State
    results = get_checkpoint_status(config.checkpoint_dir, num_splits)
    pending_indices = findall(isnothing, results)
    
    if isempty(pending_indices)
        println("✅ All splits already completed via checkpoints.")
        return TrainingResults(Vector{Tuple{Any, Any}}(results)) 
    end
    
    # Setup Progress Tracking Variables
    total_splits = length(pending_indices)
    completed_counter = Atomic{Int}(0)
    start_time_global = time()

    # 2. Define the Work Unit
    function process_split(i)
        feature_set, metadata = feature_sets[i]
        
        # Run training
        # Note: We rely on the generic 'train' overload in the parent module
        turing_result = Training.train(model, config, feature_set) 
        
        # Checkpoint
        if !isnothing(config.checkpoint_dir)
            save_split_checkpoint(config.checkpoint_dir, i, (turing_result, metadata))
        end
        
        return (turing_result, metadata)
    end
    
    # Helper for logging progress
    function log_progress(current_idx)
        # Atomic add returns the OLD value, so we add 1 to get current
        c = atomic_add!(completed_counter, 1) + 1
        
        elapsed = time() - start_time_global
        avg_time = elapsed / c
        remaining = total_splits - c
        eta_seconds = avg_time * remaining
        
        # Format strings
        pct = round((c / total_splits) * 100, digits=1)
        eta_str = _format_seconds(eta_seconds)
        elapsed_str = _format_seconds(elapsed)
        
        # Print a clear block that won't get missed
        printstyled("\n" * "="^60 * "\n", color=:green)
        printstyled("[PROGRESS] Split $c / $total_splits ($pct%)\n", color=:green, bold=true)
        println("   > Finished Index: $current_idx")
        println("   > Elapsed:        $elapsed_str")
        println("   > Est. Remaining: $eta_str")
        printstyled("="^60 * "\n", color=:green)
    end

    # 3. Execution Logic (Parallel vs Sequential)
    if strategy.parallel && length(pending_indices) > 1
        
        concurrency = min(length(pending_indices), strategy.max_concurrent_splits, Threads.nthreads())
        println("Starting Parallel Training: $total_splits splits remaining (Concurrency: $concurrency)...")
        
        semaphore = Base.Semaphore(concurrency)
        results_lock = ReentrantLock()
        
        @sync for i in pending_indices
            Threads.@spawn begin
                Base.acquire(semaphore)
                try
                    res = process_split(i)
                    
                    lock(results_lock) do
                        results[i] = res
                    end
                    
                    # Log AFTER the work is done and locked
                    log_progress(i)
                    
                catch e
                    @error "Error in Split $i: $e"
                finally
                    Base.release(semaphore)
                end
            end
        end
        
    else
        # Sequential Fallback
        println("🚀 Starting Sequential Training: $total_splits splits remaining...")
        
        for i in pending_indices
            results[i] = process_split(i)
            log_progress(i)
        end
    end
    
    # 4. Cleanup
    if !isnothing(config.checkpoint_dir) && config.cleanup_checkpoints
        cleanup_checkpoints(config.checkpoint_dir, num_splits)
    end
    
    return TrainingResults(results)
end

function _format_seconds(total_seconds)
    minutes, seconds = divrem(total_seconds, 60)
    hours, minutes = divrem(minutes, 60)
    return @sprintf("%02d:%02d:%02d", hours, minutes, seconds)
end

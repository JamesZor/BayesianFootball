# src/training/strategies/independent.jl

using Base.Threads
using Dates
using Printf
using ProgressMeter
using MCMCChains

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

    total_splits = length(pending_indices)

    # Determine if we are using the queued configuration
    is_queued = typeof(config.sampler).name.name == :QueuedNUTSConfig
    
    if strategy.parallel && is_queued
        _train_queued(model, config, feature_sets, pending_indices, results)
    else
        _train_standard(model, config, feature_sets, pending_indices, results)
    end

    # 4. Cleanup
    if !isnothing(config.checkpoint_dir) && config.cleanup_checkpoints
        cleanup_checkpoints(config.checkpoint_dir, num_splits)
    end
    
    return TrainingResults(results)
end

function _train_standard(model, config, feature_sets, pending_indices, results)
    strategy = config.strategy
    total_splits = length(pending_indices)
    completed_counter = Atomic{Int}(0)
    start_time_global = time()

    function process_split(i)
        feature_set, metadata = feature_sets[i]
        turing_result = Training.train(model, config, feature_set) 
        if !isnothing(config.checkpoint_dir)
            save_split_checkpoint(config.checkpoint_dir, i, (turing_result, metadata))
        end
        return (turing_result, metadata)
    end
    
    function log_progress(current_idx)
        c = atomic_add!(completed_counter, 1) + 1
        elapsed = time() - start_time_global
        avg_time = elapsed / c
        remaining = total_splits - c
        eta_seconds = avg_time * remaining
        pct = round((c / total_splits) * 100, digits=1)
        eta_str = _format_seconds(eta_seconds)
        elapsed_str = _format_seconds(elapsed)
        progress_msg = @sprintf(
            "[PROGRESS] Split %4d / %-4d (%5.1f%%)  |  Index: %-5d  |  Elapsed: %8s  |  ETA: %8s\n",
            c, total_splits, pct, current_idx, elapsed_str, eta_str
        )
        printstyled(progress_msg, color=:green, bold=true)
    end

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
                    log_progress(i)
                catch e
                    @error "Error in Split $i" exception=(e, catch_backtrace())
                finally
                    Base.release(semaphore)
                end
            end
        end
    else
        println("Starting Sequential Training: $total_splits splits remaining...")
        for i in pending_indices
            results[i] = process_split(i)
            log_progress(i)
        end
    end
end

function _train_queued(model, config, feature_sets, pending_indices, results)
    strategy = config.strategy
    n_chains = config.sampler.n_chains
    total_chains_to_run = length(pending_indices) * n_chains

    println("Starting Queued Training: $(length(pending_indices)) splits x $n_chains chains = $total_chains_to_run tasks remaining...")
    
    # Pre-allocate array of chains for each split
    split_chains = Dict{Int, Vector{Any}}()
    for i in pending_indices
        split_chains[i] = Vector{Any}(undef, n_chains)
    end
    
    # We use a progress meter
    prog = Progress(total_chains_to_run, desc="Sampling Chains: ")
    
    # Create the flattened list of tasks: (split_index, chain_id)
    tasks = Tuple{Int, Int}[]
    for i in pending_indices
        for c in 1:n_chains
            push!(tasks, (i, c))
        end
    end

    # Set up concurrency based on new CPU setting
    concurrency = min(length(tasks), strategy.max_concurrent_tasks)
    semaphore = Base.Semaphore(concurrency)
    results_lock = ReentrantLock()

    # Track how many chains finished per split to know when to combine and checkpoint
    chains_finished_per_split = Dict{Int, Int}(i => 0 for i in pending_indices)

    @sync for (i, c_id) in tasks
        Threads.@spawn begin
            Base.acquire(semaphore)
            try
                feature_set, metadata = feature_sets[i]
                
                # Run single chain
                chain_res = Training.train(model, config, feature_set; chain_id=c_id)
                
                lock(results_lock) do
                    split_chains[i][c_id] = chain_res
                    chains_finished_per_split[i] += 1
                    
                    # If all chains for this split are done, combine and save
                    if chains_finished_per_split[i] == n_chains
                        combined_chain = cat(split_chains[i]...; dims=3)
                        results[i] = (combined_chain, metadata)
                        
                        if !isnothing(config.checkpoint_dir)
                            save_split_checkpoint(config.checkpoint_dir, i, results[i])
                        end
                        
                        # Free up memory
                        delete!(split_chains, i)
                    end
                end
                
            catch e
                @error "Error in Split $i, Chain $c_id" exception=(e, catch_backtrace())
            finally
                next!(prog)
                Base.release(semaphore)
            end
        end
    end
end

function _format_seconds(total_seconds)
    minutes, seconds = divrem(total_seconds, 60)
    hours, minutes = divrem(minutes, 60)
    return @sprintf("%02d:%02d:%02d", hours, minutes, seconds)
end

# Experiment-local durable queue and prescribed one-attempt convergence reruns.
module Wave2Grid
using BayesianFootball, CSV, DataFrames, Dates, Random, Serialization
const H = BayesianFootball.Harness
const T = BayesianFootball.Training
const I = T.Inference
const ORDER = ["grw_joint", "qs_joint", "grw_marketobs", "qs_marketobs"]
# Frozen BEFORE grid: unique task-local seed per arm/fold/attempt/chain.
chain_seed(arm, fold, attempt, chain) = 202700000 + 100000arm + 100fold + 10attempt + chain
post_seed(arm, attempt) = 202700000 + 100000arm + 9000 + attempt

"Refuse changed recipe/seed manifests, rather than reusing incompatible checkpoints."
function manifest!(dir, candidate, arm, attempt)
    mkpath(dir)
    text = "recipe=$(H.recipe_hash(candidate))\narm=$arm\nattempt=$attempt\nsampler=$(candidate.sampler)\nseed=202700000+100000arm+100fold+10attempt+chain\n"
    path = joinpath(dir, "wave2_manifest.txt")
    if isfile(path)
        read(path, String) == text || error("checkpoint manifest changed: $dir")
    else
        any(startswith("split_"), readdir(dir)) && error("unmanifested checkpoints: $dir")
        write(path, text)
    end
end

"Same core sample_fold/budget, explicit reproducible chain seeds, atomic fold landing."
function sample_pending!(candidate, inputs, dir, arm, attempt; sample = T.sample_fold)
    n = length(inputs.feature_sets)
    nc = candidate.sampler.n_chains
    restored = I.load_checkpoints(dir, n)
    pending = findall(isnothing, restored)
    for i in setdiff(1:n, pending)
        size(restored[i], 3) == nc || error("incomplete checkpoint fold $i")
    end
    buffers = Dict(i => Vector{Any}(nothing, nc) for i in pending)
    landed = Dict(i => 0 for i in pending)
    sem, lk = Base.Semaphore(16), ReentrantLock()
    @sync for i in pending, c in 1:nc
        Threads.@spawn begin
            Base.acquire(sem)
            try
                seed = chain_seed(arm, i, attempt, c)
                Random.seed!(seed)
                fs, meta = inputs.feature_sets[i]
                part = sample(candidate.model, candidate.sampler, fs, i; chain_id = c)
                part === nothing && error("empty chain: fold=$i chain=$c")
                lock(lk) do
                    buffers[i][c] = part
                    landed[i] += 1
                    if landed[i] == nc
                        chain = cat(buffers[i]...; dims = 3)
                        size(chain, 3) == nc || error("wrong chain count for fold $i")
                        I.save_checkpoint(dir, i, (chain, meta))
                        delete!(buffers, i)
                        println("[FOLD_CHECKPOINT] arm=$(candidate.name) fold=$i attempt=$attempt at=$(now())")
                        flush(stdout)
                    end
                end
            finally
                Base.release(sem)
            end
        end
    end
    all(!isnothing, I.load_checkpoints(dir, n)) || error("missing folds after queue")
    return nothing
end

function diagnostics(fit, candidate, attempt, run_id)
    return DataFrame([(; candidate, attempt, run_id = string(run_id), fold = d.fold,
        max_rhat = d.max_rhat, min_ess_bulk = d.min_ess_bulk, min_ess_tail = d.min_ess_tail,
        divergence_rate = d.divergence_rate, n_divergent = d.n_divergent,
        rhat_pass = d.max_rhat <= 1.05, tail_ess_400 = d.min_ess_tail >= 400) for d in fit.diagnostics.folds])
end

"Resume by immutable UUID receipt, otherwise use the unchanged harness grid lifecycle."
function grid_stage(candidate, ds, db, experiment, arm, attempt, output; control = nothing)
    receipt = joinpath(output, "run_$(candidate.name).txt")
    if isfile(receipt)
        id = strip(read(receipt, String))
        fit = T.load_fit(db, id)
        H._structural_equal(fit.config.model, candidate.model) || error("receipt model changed")
        H._structural_equal(fit.config.sampler, candidate.sampler) || error("receipt sampler changed")
        H._structural_equal(fit.config.splitter, Data.ScopedWalkForwardCV(candidate.scope)) ||
            error("receipt scope changed")
        return (; fit, run_id = id)
    end
    Random.seed!(post_seed(arm, attempt))
    result = H.grid(candidate; ds, experiment, db, control)
    all(r -> r.status == "pass", filter(r -> r.severity == "hard", result.records)) ||
        error("grid hard checks failed: $(candidate.name)")
    write(receipt, string(result.run_id) * "\n")
    CSV.write(joinpath(output, "diagnostics_$(candidate.name).csv"),
        diagnostics(result.fit, candidate.name, attempt, result.run_id))
    return result
end

"One arm, then mandatory rerun of every Rhat>1.05 fold; never choose the better result."
function run(candidate, ds, db, metadata; output)
    mkpath(output)
    arm = only(findall(==(candidate.name), ORDER))
    H.has_passing_smoke(db, candidate) || error("no passing smoke: $(candidate.name)")
    config = H.fit_config(candidate; stage = :grid, experiment = metadata.experiment)
    tags = ["harness", "wave2", "todo:041"]
    T.save_model(db, candidate.name, candidate.model; tags, description = candidate.hypothesis)
    T.save_config(db, candidate.name * "_grid_recipe", config; tags)
    # Query completed scientific config before any sampling. A receipt also protects
    # convergence:FAIL runs whose stamped config tags differ from the template.
    existing = H.find_completed_run(db, config)
    inputs = H._fold_inputs(candidate, ds; stage = :grid)
    length(inputs.boundaries) == 40 || error("expected 40 folds")
    H._filtration_check(candidate, inputs)
    primary_dir = joinpath("data", "checkpoints", metadata.experiment, candidate.name)
    manifest!(primary_dir, candidate, arm, 0)
    if existing === nothing && !isfile(joinpath(output, "run_$(candidate.name).txt"))
        sample_pending!(candidate, inputs, primary_dir, arm, 0)
    end
    primary = grid_stage(candidate, ds, db, metadata.experiment, arm, 0, output)
    length(primary.fit.folds) == 40 || error("incomplete primary grid")
    bad = [d.fold for d in primary.fit.diagnostics.folds if d.max_rhat > 1.05]
    CSV.write(joinpath(output, "rerun_selection_$(candidate.name).csv"),
        DataFrame(fold = collect(1:40), rerun = in.(1:40, Ref(bad))))
    final = primary
    if !isempty(bad)
        rerun_candidate = H.Candidate(name = candidate.name * "__rhat_rerun",
            model = candidate.model, sampler = candidate.sampler, scope = candidate.scope,
            role = candidate.role, hypothesis = candidate.hypothesis * "; one prescribed Rhat rerun")
        rerun_config = H.fit_config(rerun_candidate; stage = :grid, experiment = metadata.experiment)
        T.save_config(db, rerun_candidate.name * "_grid_recipe", rerun_config; tags)
        rerun_dir = joinpath("data", "checkpoints", metadata.experiment, rerun_candidate.name)
        manifest!(rerun_dir, rerun_candidate, arm, 1)
        for i in setdiff(1:40, bad)
            target = I.checkpoint_path(rerun_dir, i)
            if !isfile(target)
                # Copy chains from the persisted initial fit, even if an existing DB
                # grid was loaded without local checkpoints. Preserve the original.
                fold = only(filter(f -> f.fold == i, primary.fit.folds))
                I.save_checkpoint(rerun_dir, i, (fold.chain, inputs.feature_sets[i][2]))
            end
        end
        if !isfile(joinpath(output, "run_$(rerun_candidate.name).txt")) &&
                H.find_completed_run(db, rerun_config) === nothing
            sample_pending!(rerun_candidate, inputs, rerun_dir, arm, 1)
        end
        final = grid_stage(rerun_candidate, ds, db, metadata.experiment, arm, 1, output)
    end
    # A bad rerun is retained, never replaced by the initial chain or rerun twice.
    all(d -> isfinite(d.max_rhat) && d.divergence_rate <= 0.001,
        final.fit.diagnostics.folds) || error("final divergence/nonfinite diagnostic gate")
    row = (; candidate = candidate.name, primary_run_id = string(primary.run_id),
        final_run_id = string(final.run_id), rerun_folds = join(bad, ";"),
        max_rhat = final.fit.diagnostics.max_rhat, min_ess_tail = final.fit.diagnostics.min_ess_tail,
        git_sha = T.git_commit_id())
    CSV.write(joinpath(output, "grid_summary_$(candidate.name).csv"), DataFrame([row]))
    println("[GRID_ARM_DONE] ", repr(row))
    flush(stdout)
    return final
end
end # module

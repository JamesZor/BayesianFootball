# Phase 5: wave-1 r06_reproduce comparison path adapted to the frozen wave-2 queue.
# ONE fold (fold 1), ONE selected new arm per fresh beast REPL. No reference fits,
# DB writes, grid checkpoint changes, or new experiment runs.

# %% 1. Runtime and frozen recipes
using BayesianFootball, CSV, DataFrames, LinearAlgebra, MCMCChains, Random, SHA
using ThreadPinning, UUIDs
strip(read(`hostname`, String)) == "mcmc-beast" && Threads.nthreads() == 16 ||
    error("reproduction requires fresh 16-threaded beast REPL")
pinthreads(:cores)
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "l05_grid.jl"))
const R7C = Module(:Wave2ReproCandidates)
Base.include(R7C, joinpath(@__DIR__, "candidates.jl"))

# %% 2. One new arm; separate fresh checkpoint directory, attempt-0 frozen seeds
r7_name = ENV["QSX2_ONLY"]
r7_arm = only(findall(==(r7_name), Wave2Grid.ORDER))
r7_candidate = only(filter(c -> c.name == r7_name, R7C.CANDIDATES))
r7_out = joinpath("/root/BF_runs/qs_experiment_w2_out/phase5", r7_name)
isdir(r7_out) && !isempty(readdir(r7_out)) && error("reproduction output must be fresh")
mkpath(r7_out)
r7_ids = CSV.read(joinpath(@__DIR__, "results", "phase3", "RUNS.csv"), DataFrame)
r7_id = UUID(string(only(filter(r -> r.label == r7_name, r7_ids)).run_id))
r7_db = Training.PostgresStorage(R7C.EXPERIMENT)
r7_saved = Training.load_fit(r7_db, r7_id)
Harness._structural_equal(r7_saved.config.model, r7_candidate.model) || error("model changed")
Harness._structural_equal(r7_saved.config.sampler, r7_candidate.sampler) || error("budget changed")
r7_ds = Main.QSMarketArm.checked_datastore()
r7_inputs = Harness._fold_inputs(r7_candidate, r7_ds; stage=:grid)
Harness._filtration_check(r7_candidate, r7_inputs)
r7_one = (; feature_sets=r7_inputs.feature_sets[1:1], oos=r7_inputs.oos[1:1])
r7_config = Harness.fit_config(r7_candidate; stage=:grid, experiment=R7C.EXPERIMENT)
r7_dir = joinpath(r7_out, "checkpoints")
Wave2Grid.manifest!(r7_dir, r7_candidate, r7_arm, 0)

# %% 3. ONLY authorised reproduction sampling; unchanged queue and sampler
r7_started = time()
Wave2Grid.sample_pending!(r7_candidate, r7_one, r7_dir, r7_arm, 0)
Random.seed!(Wave2Grid.post_seed(r7_arm, 0))
r7_fit = Training.fit_model(r7_config; feature_sets=r7_one.feature_sets,
    oos_fixtures=r7_one.oos, checkpoint_dir=r7_dir, quiet=true)

# %% 4. Wave-1 chain/latent comparison, strengthened to raw Float64 bytes and SHA256
r7_fold = only(filter(f -> f.fold == 1, r7_saved.folds))
names(r7_fit.folds[1].chain) == names(r7_fold.chain) || error("parameter names differ")
r7_fresh_chain = cat(Array(r7_fit.folds[1].chain, append_chains=false)...; dims=3)
r7_saved_chain = cat(Array(r7_fold.chain, append_chains=false)...; dims=3)
r7_rowmap = Dict(id => i for (i, id) in enumerate(r7_saved.latents.match_ids))
r7_rows = [r7_rowmap[id] for id in r7_fit.latents.match_ids]
r7_fresh_lat = hcat(r7_fit.latents.λ_home, r7_fit.latents.λ_away)
r7_saved_lat = hcat(r7_saved.latents.λ_home[r7_rows, :], r7_saved.latents.λ_away[r7_rows, :])
function r7_bytes(a)
    eltype(a) === Float64 || error("comparison must be Float64")
    return reinterpret(UInt8, vec(a))
end
r7_evidence = NamedTuple[]
for (channel, a, b) in (("chains", r7_fresh_chain, r7_saved_chain),
                       ("latents", r7_fresh_lat, r7_saved_lat))
    size(a) == size(b) || error("$channel shape differs")
    fresh_sha, saved_sha = bytes2hex(sha256(r7_bytes(a))), bytes2hex(sha256(r7_bytes(b)))
    identical = r7_bytes(a) == r7_bytes(b)
    push!(r7_evidence, (; arm=r7_name, fold=1, attempt=0, run_id=string(r7_id), channel,
        identical, fresh_sha, saved_sha, n_values=length(a), max_abs=maximum(abs.(a .- b)),
        seconds=time()-r7_started,
        chain_seeds=join([Wave2Grid.chain_seed(r7_arm, 1, 0, c) for c in 1:4], ";"),
        post_seed=Wave2Grid.post_seed(r7_arm, 0)))
end
CSV.write(joinpath(r7_out, "reproduction.csv"), DataFrame(r7_evidence))
all(r -> r.identical, r7_evidence) || error("REPRODUCTION_BYTE_MISMATCH $r7_name; no retry")
println("PHASE5_REPRODUCTION_ARM_PASS arm=$r7_name fold=1 chains_and_latents_bytes=true")

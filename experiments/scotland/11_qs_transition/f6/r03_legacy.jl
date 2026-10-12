# Original wave-2 fold, original seeds; F6 source, four threads, no DB writes.
# %% 1. Runtime
using BayesianFootball, CSV, DataFrames, LinearAlgebra, MCMCChains, Random, SHA
using ThreadPinning, UUIDs
strip(read(`hostname`, String)) == "mcmc-beast" && Threads.nthreads() == 4 || error("beast4 only")
pinthreads(:cores)
BLAS.set_num_threads(1)

# %% 2. Frozen recipe / reference
const F6W2DIR = normpath(joinpath(@__DIR__, "..", "..", "06_qs_joint_and_market_observation"))
include(joinpath(F6W2DIR, "l05_grid.jl"))
const F6Legacy = Module(:F6LegacyCandidates)
Base.include(F6Legacy, joinpath(F6W2DIR, "candidates.jl"))
legacy_candidate = only(filter(c -> c.name == "qs_marketobs", F6Legacy.CANDIDATES))
legacy_id = UUID("81fdf817-f82d-4f66-9bfd-5f456c4fe29f")
legacy_saved = Training.load_fit(Training.PostgresStorage(F6Legacy.EXPERIMENT), legacy_id)
Harness._structural_equal(legacy_saved.config.model, legacy_candidate.model) || error("model changed")
Harness._structural_equal(legacy_saved.config.sampler, legacy_candidate.sampler) || error("sampler changed")

# %% 3. Inputs / durable original seed schedule
legacy_ds = Main.QSMarketArm.checked_datastore()
legacy_inputs = Harness._fold_inputs(legacy_candidate, legacy_ds; stage=:grid)
Harness._filtration_check(legacy_candidate, legacy_inputs)
legacy_one = (; feature_sets=legacy_inputs.feature_sets[1:1], oos=legacy_inputs.oos[1:1])
legacy_config = Harness.fit_config(legacy_candidate; stage=:grid, experiment=F6Legacy.EXPERIMENT)
legacy_dir = joinpath(@__DIR__, "legacy_checkpoints_f6")
isdir(legacy_dir) && error("legacy checkpoint directory must be fresh")
Wave2Grid.manifest!(legacy_dir, legacy_candidate, 4, 0)

# %% 4. Reproduction (four chains concurrently, at most four worker threads)
legacy_start = time()
Wave2Grid.sample_pending!(legacy_candidate, legacy_one, legacy_dir, 4, 0)
Random.seed!(Wave2Grid.post_seed(4, 0))
legacy_fit = Training.fit_model(legacy_config; feature_sets=legacy_one.feature_sets,
    oos_fixtures=legacy_one.oos, checkpoint_dir=legacy_dir, quiet=true)

# %% 5. Raw-byte identity
legacy_fold = only(filter(f -> f.fold == 1, legacy_saved.folds))
names(legacy_fit.folds[1].chain) == names(legacy_fold.chain) || error("parameter names differ")
legacy_fresh_chain = cat(Array(legacy_fit.folds[1].chain, append_chains=false)...; dims=3)
legacy_saved_chain = cat(Array(legacy_fold.chain, append_chains=false)...; dims=3)
legacy_rowmap = Dict(id => i for (i, id) in enumerate(legacy_saved.latents.match_ids))
legacy_rows = [legacy_rowmap[id] for id in legacy_fit.latents.match_ids]
legacy_fresh_lat = hcat(legacy_fit.latents.λ_home, legacy_fit.latents.λ_away)
legacy_saved_lat = hcat(legacy_saved.latents.λ_home[legacy_rows, :], legacy_saved.latents.λ_away[legacy_rows, :])
legacy_bytes(a) = reinterpret(UInt8, vec(a))
legacy_evidence = NamedTuple[]
for (channel, a, b) in (("chains", legacy_fresh_chain, legacy_saved_chain),
                       ("latents", legacy_fresh_lat, legacy_saved_lat))
    size(a) == size(b) || error("$channel shape differs")
    push!(legacy_evidence, (; arm="qs_marketobs", fold=1, run_id=string(legacy_id), channel,
        git_sha=Training.git_commit_id(), identical=legacy_bytes(a)==legacy_bytes(b),
        fresh_sha=bytes2hex(sha256(legacy_bytes(a))), saved_sha=bytes2hex(sha256(legacy_bytes(b))),
        n_values=length(a), max_abs=maximum(abs.(a .- b)), seconds=time()-legacy_start,
        chain_seeds=join([Wave2Grid.chain_seed(4, 1, 0, c) for c in 1:4], ";"),
        post_seed=Wave2Grid.post_seed(4, 0)))
end
CSV.write(joinpath(@__DIR__, "legacy_reproduction.csv"), DataFrame(legacy_evidence))
all(r -> r.identical, legacy_evidence) || error("REPRODUCTION_BYTE_MISMATCH; stop F6")
println("F6_LEGACY_PASS chains_and_latents_bytes=true seconds=$(time()-legacy_start)")

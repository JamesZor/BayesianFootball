# Read-only real feature inputs, FAKE chains: validate queue persistence/seeding before grid.
using Test, BayesianFootball, MCMCChains, Random, LinearAlgebra, ThreadPinning
pinthreads(:cores); BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "l05_grid.jl"))
const QC = Module(:Wave2QueueCandidates)
Base.include(QC, joinpath(@__DIR__, "candidates.jl"))
candidate = first(QC.CANDIDATES)
ds = Main.QSMarketArm.checked_datastore()
full = Harness._fold_inputs(candidate, ds; stage = :grid)
inputs = (; feature_sets = full.feature_sets[1:2])
fake_sample(model, sampler, fs, fold; chain_id) = MCMCChains.Chains(randn(12, 1, 1), [:dummy])
@testset "Wave 2 durable seeded queue (no MCMC)" begin
    mktempdir() do dir
        Wave2Grid.manifest!(dir, candidate, 1, 0)
        Wave2Grid.sample_pending!(candidate, inputs, dir, 1, 0; sample = fake_sample)
        first_run = Wave2Grid.I.load_checkpoints(dir, 2)
        @test all(c -> size(c, 3) == 4, first_run)
        Wave2Grid.sample_pending!(candidate, inputs, dir, 1, 0;
            sample = (args...; kwargs...) -> error("completed checkpoint must not sample"))
        @test Array(first_run[1]) == Array(Wave2Grid.I.load_checkpoints(dir, 2)[1])
        rm(Wave2Grid.I.checkpoint_path(dir, 2))
        Wave2Grid.sample_pending!(candidate, inputs, dir, 1, 0; sample = fake_sample)
        @test Array(first_run[2]) == Array(Wave2Grid.I.load_checkpoints(dir, 2)[2])
        @test_throws ErrorException Wave2Grid.manifest!(dir, candidate, 1, 1)
        @test length(unique(Wave2Grid.chain_seed(a, f, k, c)
            for a in 1:4 for f in 1:40 for k in 0:1 for c in 1:4)) == 1280
    end
end
println("GRID_QUEUE_VALIDATION_PASS: no actual sampling")

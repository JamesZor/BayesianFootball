# Read-only real feature inputs, FAKE chains: validate queue persistence/seeding before grid.
using Test, BayesianFootball, MCMCChains, Random, LinearAlgebra, ThreadPinning
pinthreads(:cores); BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "l05_grid.jl"))
const QC = Module(:Wave3QueueCandidates)
Base.include(QC, joinpath(@__DIR__, "candidates.jl"))
candidate = first(QC.CANDIDATES)
ds = Main.QSMarketArm.checked_datastore()
full = Harness._fold_inputs(candidate, ds; stage = :grid)
inputs = (; feature_sets = full.feature_sets[1:2])
fake_sample(model, sampler, fs, fold; chain_id) = MCMCChains.Chains(randn(12, 1, 1), [:dummy])
@testset "Wave 3 durable seeded queue (no MCMC)" begin
    @test length(full.feature_sets) == 40
    @test [c.name for c in QC.CANDIDATES] == Wave3Grid.ORDER
    mktempdir() do dir
        Wave3Grid.manifest!(dir, candidate, 1, 0)
        Wave3Grid.sample_pending!(candidate, inputs, dir, 1, 0; sample = fake_sample)
        first_run = Wave3Grid.I.load_checkpoints(dir, 2)
        @test all(c -> size(c, 3) == 4, first_run)
        Wave3Grid.sample_pending!(candidate, inputs, dir, 1, 0;
            sample = (args...; kwargs...) -> error("completed checkpoint must not sample"))
        @test Array(first_run[1]) == Array(Wave3Grid.I.load_checkpoints(dir, 2)[1])
        rm(Wave3Grid.I.checkpoint_path(dir, 2))
        Wave3Grid.sample_pending!(candidate, inputs, dir, 1, 0; sample = fake_sample)
        @test Array(first_run[2]) == Array(Wave3Grid.I.load_checkpoints(dir, 2)[2])
        @test_throws ErrorException Wave3Grid.manifest!(dir, candidate, 1, 1)
        seeds = [Wave3Grid.chain_seed(a, f, k, c) for a in 1:3 for f in 1:40 for k in 0:1 for c in 1:4]
        @test length(unique(seeds)) == 960
        # Disjoint from every wave-2 chain and post seed (max 202700000 + 400000 + 9001).
        @test minimum(seeds) > 202700000 + 100000 * 4 + 9001
    end
    # Control preflight fails closed without the control's summary and receipt.
    mktempdir() do dir
        @test_throws Exception Wave3Grid.checked_control(nothing, QC.EXPERIMENT, dir)
    end
    @test_throws ErrorException Wave3Grid.check_control_ref(QC.CANDIDATES[2], QC.EXPERIMENT, nothing)
end
println("GRID_QUEUE_VALIDATION_PASS: no actual sampling")

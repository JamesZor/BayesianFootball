# test/sampler_adtype_tests.jl
#
# Regression: the NUTS samplers must differentiate with the compiled ReverseDiff tape.
#
# Turing reads the AD backend from the sampler object (`spl.adtype`) and silently ignores
# an `adtype` keyword on `sample`. `run_sampler` once passed it there, so every chain ran
# `NUTS`'s default AutoForwardDiff (Experiment 08 TODO 003). Two layers are checked: the
# sampler object `run_sampler` builds, and the number type the model is actually evaluated
# with when sampled through `run_sampler` — the layer the original bug slipped past.

using Test
using BayesianFootball
using BayesianFootball.Samplers: nuts_algorithm, run_sampler
import ForwardDiff
import ReverseDiff
import Turing

# Every element type `x` takes while the model is evaluated. ForwardDiff evaluates on
# `ForwardDiff.Dual`; recording the compiled ReverseDiff tape evaluates on
# `ReverseDiff.TrackedReal`. A lock because `NUTSConfig` samples under `MCMCThreads()`.
const ADTYPE_SEEN = Set{DataType}()
const ADTYPE_LOCK = ReentrantLock()

Turing.@model function adtype_probe_model()
    x ~ Turing.Normal(0, 1)
    lock(() -> push!(ADTYPE_SEEN, typeof(x)), ADTYPE_LOCK)
end

@testset "NUTS sampler AD backend" begin
    @testset "nuts_algorithm carries AutoReverseDiff(compile = true): $(nameof(C))" for C in (NUTSConfig, QueuedNUTSConfig)
        alg = nuts_algorithm(C(n_warmup = 123, accept_rate = 0.9, max_depth = 8))
        @test alg isa Turing.NUTS
        @test alg.adtype isa Turing.AutoReverseDiff
        @test alg.adtype == Turing.AutoReverseDiff(compile = true)
        # The config still reaches the sampler unchanged.
        @test alg.n_adapts == 123
        @test alg.δ == 0.9
        @test alg.max_depth == 8
    end

    @testset "run_sampler evaluates the model on ReverseDiff, never ForwardDiff: $label" for (label, sample_it) in (
        "QueuedNUTSConfig" => () -> run_sampler(adtype_probe_model(),
            QueuedNUTSConfig(n_samples = 20, n_warmup = 20, n_chains = 1), 1),
        "NUTSConfig" => () -> run_sampler(adtype_probe_model(),
            NUTSConfig(n_samples = 20, n_warmup = 20, n_chains = 1, show_progress = false)),
    )
        empty!(ADTYPE_SEEN)
        chain = sample_it()
        @test size(chain, 1) == 20
        @test any(T -> T <: ReverseDiff.TrackedReal, ADTYPE_SEEN)
        @test !any(T -> T <: ForwardDiff.Dual, ADTYPE_SEEN)
    end
end

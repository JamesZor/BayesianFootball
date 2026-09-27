#!/usr/bin/env julia
#
# The smoke's sampling-performance probe, standalone, on a candidate's LARGEST fold.
#
#   julia --project -t 16 scripts/sampling_perf_probe.jl <candidates.jl> --only NAME
#         [--chains 16] [--warmup 50] [--samples 50] [--report FILE.md]
#         [--single-chain 150,150] [--overhead] [--no-profile] [--no-allocs]
#
# Default: `--chains` defaults to Threads.nthreads(); 50 + 50 iterations, exactly the smoke's probe
# (src/harness/perf_monitor.jl, experiments/claude_zero_alloc_addendum_perf_monitor.md). Prints
# the metrics, the review flags and writes the markdown report.
#
# `--overhead` runs the probe twice, with and without the statistical profiler, and prints the wall
# time ratio (rev2 item 6: must stay within ~10 %). `--no-profile` / `--no-allocs` switch the
# profiler and the Profile.Allocs pass off.
#
# `--single-chain W,S` instead runs ONE chain of W warm-up + S draws under `@timed` with
# `discard_adapt = false` and prints GC %, GiB allocated, leapfrogs, KB and ms per leapfrog (the
# zero-alloc work package's §2.4 single-chain benchmark).
#
# Like scripts/tape_allocation_audit.jl this includes the harness helper files BY PATH, so it runs
# against an older checkout: `julia --project=<old> <this checkout>/scripts/sampling_perf_probe.jl`.

using BayesianFootball
using LinearAlgebra
using Printf
using ThreadPinning

LinearAlgebra.BLAS.set_num_threads(1)
try
    pinthreads(:cores)
catch err
    @warn "pinthreads(:cores) failed" exception = err
end

module PerfProbe
using BayesianFootball
import DynamicPPL, LogDensityProblems, Logging, Profile, Random, ReverseDiff, Turing
include(joinpath(@__DIR__, "..", "src", "harness", "tape_profile.jl"))
include(joinpath(@__DIR__, "..", "src", "harness", "perf_monitor.jl"))
end

function parse_args(args)
    file = nothing
    only = nothing
    chains = Threads.nthreads()
    warmup = 50
    samples = 50
    report = nothing
    single = nothing
    overhead = false
    profile = true
    allocs = true
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--only"
            only = args[i + 1]; i += 2
        elseif arg == "--chains"
            chains = parse(Int, args[i + 1]); i += 2
        elseif arg == "--warmup"
            warmup = parse(Int, args[i + 1]); i += 2
        elseif arg == "--samples"
            samples = parse(Int, args[i + 1]); i += 2
        elseif arg == "--report"
            report = args[i + 1]; i += 2
        elseif arg == "--single-chain"
            single = parse.(Int, split(args[i + 1], ",")); i += 2
        elseif arg == "--overhead"
            overhead = true; i += 1
        elseif arg == "--no-profile"
            profile = false; i += 1
        elseif arg == "--no-allocs"
            allocs = false; i += 1
        elseif startswith(arg, "-")
            error("unknown option $arg")
        else
            file = arg; i += 1
        end
    end
    (file === nothing || only === nothing) &&
        error("usage: sampling_perf_probe.jl <candidates.jl> --only NAME [options]")
    return (; file, only, chains, warmup, samples, report, single, overhead, profile, allocs)
end

function largest_fold(candidate)
    raw = BayesianFootball.Data.load_datastore_cached(
        BayesianFootball.Data.ScottishPyramid(); max_age_hours = 10_000)
    ds = BayesianFootball.Data.apply_scope(raw, candidate.scope)
    splitter = BayesianFootball.Data.ScopedWalkForwardCV(candidate.scope)
    bounds = BayesianFootball.Data.create_id_boundaries(ds, splitter)
    fs = first(BayesianFootball.Features.create_features([bounds[end]], ds, candidate.model,
                                                         splitter))[1]
    return (; fs, index = length(bounds))
end

function single_chain(candidate, fs, warmup, samples)
    config = PerfProbe._probe_sampler(candidate.sampler, 1, warmup, samples)
    PerfProbe._probe_chain(candidate.model, fs, PerfProbe._probe_sampler(candidate.sampler, 1, 2, 2), 1)
    stats = @timed PerfProbe._probe_chain(candidate.model, fs, config, 1)
    leapfrogs = stats.value
    @printf("SINGLE candidate=%s warmup=%d samples=%d wall_s=%.2f gc_pct=%.1f gib=%.3f leapfrogs=%d kb_per_leapfrog=%.2f ms_per_leapfrog=%.4f\n",
            candidate.name, warmup, samples, stats.time, 100 * stats.gctime / stats.time,
            stats.bytes / 2^30, leapfrogs, stats.bytes / leapfrogs / 1024,
            1e3 * stats.time / leapfrogs)
end

function main()
    opts = parse_args(ARGS)
    mod = Module(:ProbeCandidates)
    Base.include(mod, abspath(opts.file))
    candidates = Base.invokelatest(getproperty, mod, :CANDIDATES)
    candidate = only(filter(c -> c.name == opts.only, candidates))
    println("git: ", BayesianFootball.Training.git_commit_id(), "  threads: ", Threads.nthreads(),
            "  candidate: ", candidate.name)
    fold = largest_fold(candidate)

    if opts.single !== nothing
        single_chain(candidate, fold.fs, opts.single[1], opts.single[2])
        println("PROBE_DONE")
        return
    end

    metrics = PerfProbe.sampling_performance_probe(candidate.model, fold.fs, candidate.sampler;
        n_chains = opts.chains, n_warmup = opts.warmup, n_samples = opts.samples,
        profile = opts.profile, allocs = opts.allocs)
    flags = PerfProbe.sampling_performance_flags(metrics)
    for key in keys(metrics)
        value = getproperty(metrics, key)
        value isa AbstractVector && eltype(value) <: NamedTuple && continue
        println("METRIC ", rpad(String(key), 32), " ", value)
    end
    for row in metrics.allocating_instructions
        println("ALLOC  #", row.index, " ", row.bytes, " B  ", row.description)
    end
    for t in metrics.profile_threads
        println("THREAD ", t.thread, " ", t.pool, " samples=", t.samples, " util=", round(t.utilisation, digits = 3))
    end
    for f in metrics.top_frames
        println("FRAME  ", f.self, "  ", f.frame)
    end
    for site in vcat(metrics.non_tape_gradient_sites, metrics.non_tape_step_sites)
        println("SITE   ", site.source, "  ", round(site.est_bytes_per_unit, digits = 1), " B  ", site.site)
    end
    if opts.overhead
        bare = PerfProbe.sampling_performance_probe(candidate.model, fold.fs, candidate.sampler;
            n_chains = opts.chains, n_warmup = opts.warmup, n_samples = opts.samples,
            profile = false, allocs = false)
        @printf("OVERHEAD wall_with_profiler=%.2f wall_without=%.2f ratio=%.3f window_with=%.2f window_without=%.2f\n",
                metrics.wall_seconds, bare.wall_seconds, metrics.wall_seconds / bare.wall_seconds,
                metrics.window_seconds, bare.window_seconds)
    end
    status = isempty(flags) ? "pass" : "review"
    for f in flags
        println("FLAG   ", f.flag, " value=", f.value, " threshold=", f.threshold)
    end
    println("SUMMARY ", PerfProbe.perf_summary((; status, metrics, flags, report = nothing)))
    if opts.report !== nothing
        PerfProbe.write_sampling_performance_report(opts.report, metrics, flags;
            title = "Sampling performance — $(candidate.name)",
            context = (; candidate = candidate.name,
                         git_sha = BayesianFootball.Training.git_commit_id(),
                         fold = "$(fold.index) (largest)"))
        println("REPORT ", opts.report)
    end
    println("PROBE_DONE")
end

main()

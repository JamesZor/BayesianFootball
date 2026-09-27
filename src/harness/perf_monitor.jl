# src/harness/perf_monitor.jl
#
# Load-realistic sampling-performance probe for the smoke stage.
#
# Specification: experiments/claude_zero_alloc_addendum_perf_monitor.md (§2.6) as revised by
# experiments/claude_zero_alloc_addendum_perf_monitor_rev2.md (the measurement tools). Rationale:
# docs/turing_ad_performance_guide.md §10.5 — a gradient that looks fast in isolation can still
# sample slowly under 16-way concurrency (GC pauses, idle sampler threads, a slow in-situ
# gradient), and the old smoke's 2 folds × 2 chains could not reproduce that contention.
#
# WHICH TOOL MEASURES WHAT (rev2):
#
#   * GC                 — `Base.gc_num()` diffed with `Base.GC_Diff` (+ the safepoint fields).
#   * utilisation, stalls — the statistical profiler (`Profile`) over a sub-window of the
#                          16-chain probe, per thread; `/proc/self/task` only as a fallback.
#   * tape allocation    — the per-instruction tape audit (tape_profile.jl). `Profile.Allocs`
#                          cannot attribute it: the compiled tape runs closures detached from model
#                          source, so every tape allocation stops at ReverseDiff's `macro expansion`.
#   * non-tape allocation — `Profile.Allocs` on full sampler steps (`logdensity_and_gradient`
#                          through Turing's `LogDensityFunction`, and NUTS steps), where stack traces
#                          do reach real source.
#
# Like tape_profile.jl this file is include-able on its own (it needs `BayesianFootball`'s
# `Models` and `Samplers`, Turing, DynamicPPL, LogDensityProblems, ReverseDiff, Random, Logging,
# Profile and the tape_profile.jl helpers in scope), so `scripts/sampling_perf_probe.jl` can point
# it at an older checkout — which is how the pre-fix engine's report is produced.

"""
Thresholds for the `sampling_performance` REVIEW check (never hard). Change them here only; the
addendum states the first three, and `max_tape_bytes` mirrors the hard `tape_allocation` limit so a
standalone probe (which skips the hard check) still names that cause.

  * `max_gc_share`            — GC time (`GC_Diff.total_time`) ÷ wall, full-concurrency window
  * `min_sampler_utilisation` — mean per-thread utilisation of the sampler threads (profiler;
                                `/proc` fallback), same window
  * `max_efficiency_ratio`    — in-situ ms per leapfrog ÷ warmed bare-tape gradient latency
  * `max_tape_bytes`          — bytes per compiled gradient
"""
const SAMPLING_PERF_THRESHOLDS = (; max_gc_share = 0.15, min_sampler_utilisation = 0.75,
                                   max_efficiency_ratio = 2.0, max_tape_bytes = 1024)

const PERF_UNAVAILABLE = "unavailable"
# ------------------------------------------------------------------------------
# Static tape metrics
# ------------------------------------------------------------------------------

function _min_gradient_ms(tape, gradient, θ, reps::Int)
    best = Inf
    for _ in 1:reps
        best = min(best, @elapsed ReverseDiff.gradient!(gradient, tape, θ))
    end
    return 1e3 * best
end

"Linked prior draw → recorded + compiled tape, and the objective it was recorded from."
function _linked_compiled_tape(model, fs; seed::Int)
    turing_model = Models.PreGame.build_turing_model(model, fs)
    Random.seed!(seed)
    varinfo = DynamicPPL.link!!(DynamicPPL.VarInfo(turing_model), turing_model)
    θ = copy(varinfo[:])
    density = DynamicPPL.LogDensityFunction(
        turing_model, DynamicPPL.getlogjoint_internal, varinfo)
    objective = x -> LogDensityProblems.logdensity(density, x)
    raw = ReverseDiff.GradientTape(objective, θ)
    return (; raw, tape = ReverseDiff.compile(raw), θ)
end

"""
    tape_metrics(model, feature_set; seed, latency_reps = 100) -> NamedTuple

Where NUTS evaluates the gradient — a linked prior draw on the compiled tape — measure bytes per
gradient (minimum over 20 warmed replays), warmed-minimum latency, instruction count, and every
allocating instruction. `setup_seconds` is a second, warm record + compile of the same tape: what
each chain pays before its first leapfrog.
"""
function tape_metrics(model, fs; seed::Int = 20260911, latency_reps::Int = 100)
    compiled = _linked_compiled_tape(model, fs; seed)
    gradient = similar(compiled.θ)
    bytes = compiled_gradient_bytes(compiled.tape, gradient, compiled.θ)
    gradient_ms = _min_gradient_ms(compiled.tape, gradient, compiled.θ, latency_reps)
    rows = bytes > 0 ? tape_allocation_profile(compiled.raw) : TapeAllocationRow[]
    setup_seconds = @elapsed _linked_compiled_tape(model, fs; seed)
    return (; allocated_bytes = bytes, gradient_ms,
              tape_instructions = length(compiled.raw.tape),
              n_parameters = length(compiled.θ), setup_seconds,
              allocating_instructions = rows)
end

# ------------------------------------------------------------------------------
# /proc CPU accounting (Linux). Anything missing degrades to PERF_UNAVAILABLE.
# ------------------------------------------------------------------------------

function _clock_ticks_per_second()
    ticks = try
        Int(ccall(:sysconf, Clong, (Cint,), 2))   # _SC_CLK_TCK on Linux
    catch
        -1
    end
    return ticks > 0 ? ticks : 100
end

"utime + stime of one `/proc/.../stat` file in seconds, or `nothing` if it cannot be read."
function _proc_stat_cpu_seconds(path::AbstractString)
    text = try
        read(path, String)
    catch
        return nothing
    end
    close_paren = findlast(')', text)
    close_paren === nothing && return nothing
    # Fields after the parenthesised command name start at field 3 (state); utime and stime
    # are fields 14 and 15 of proc(5), i.e. positions 12 and 13 here.
    fields = split(text[(close_paren + 1):end])
    length(fields) >= 13 || return nothing
    utime = tryparse(Int, fields[12])
    stime = tryparse(Int, fields[13])
    (utime === nothing || stime === nothing) && return nothing
    return (utime + stime) / _clock_ticks_per_second()
end

"Process CPU seconds and per-OS-thread CPU seconds, or `nothing` when `/proc` is unavailable."
function _proc_cpu_snapshot(proc_root::AbstractString)
    process = _proc_stat_cpu_seconds(joinpath(proc_root, "self", "stat"))
    process === nothing && return nothing
    threads = Dict{Int,Float64}()
    task_dir = joinpath(proc_root, "self", "task")
    entries = try
        readdir(task_dir)
    catch
        String[]
    end
    for entry in entries
        tid = tryparse(Int, entry)
        tid === nothing && continue
        seconds = _proc_stat_cpu_seconds(joinpath(task_dir, entry, "stat"))
        seconds === nothing || (threads[tid] = seconds)
    end
    return (; process, threads)
end

_os_thread_id() = try
    Int(ccall(:gettid, Cint, ()))
catch
    -1
end

"""
    sampler_thread_ids() -> Vector{Int}

The Linux thread ids of Julia's default-pool worker threads — the threads the queued sampler's
chain tasks run on. Found by running one iteration per thread with `Threads.@threads :static`
(iteration `i` is pinned to the pool's `i`-th thread) and asking each for `gettid()`. Every other
OS thread of the process — GC mark/sweep threads, the interactive pool, libuv, BLAS — is "other".
Empty when `gettid` is unavailable (non-Linux).
"""
function sampler_thread_ids()
    n = Threads.nthreads(:default)
    tids = fill(-1, n)
    Threads.@threads :static for i in 1:n
        tids[i] = _os_thread_id()
    end
    return sort!(unique!(filter(>(0), tids)))
end

"CPU deltas between two snapshots: process, sampler threads, everything else."
function _cpu_split(before, after, sampler_tids)
    (before === nothing || after === nothing) && return nothing
    sampler = 0.0
    for tid in sampler_tids
        haskey(after.threads, tid) || continue
        sampler += after.threads[tid] - get(before.threads, tid, 0.0)
    end
    process = after.process - before.process
    return (; process, sampler = isempty(sampler_tids) ? nothing : sampler,
              other = isempty(sampler_tids) ? nothing : process - sampler)
end


# ------------------------------------------------------------------------------
# GC counters (rev2 item 1)
# ------------------------------------------------------------------------------

"""
GC activity between two `Base.gc_num()` snapshots over `seconds` of wall time: `GC_Diff` for time,
bytes, pauses and full sweeps, plus the cumulative time-to-safepoint (how long mutator threads
took to stop for a collection). `max_pause_ms` / `max_time_to_safepoint_ms` are the PROCESS
maxima at the end of the window — `gc_num` keeps no per-window maximum.
"""
function _gc_window(before, after, seconds)
    diff = Base.GC_Diff(after, before)
    to_safepoint = (after.total_time_to_safepoint - before.total_time_to_safepoint) / 1e9
    return (; gc_share = _ratio(diff.total_time / 1e9, seconds),
              gc_seconds = diff.total_time / 1e9, gc_allocd_bytes = diff.allocd,
              gc_pauses = diff.pause, gc_full_sweeps = diff.full_sweep,
              gc_time_to_safepoint_seconds = to_safepoint,
              gc_max_pause_ms = after.max_pause / 1e6,
              gc_max_time_to_safepoint_ms = after.max_time_to_safepoint / 1e6)
end

# ------------------------------------------------------------------------------
# Statistical profiler: per-thread utilisation, GC stalls, JIT, hot frames (rev2 item 2)
# ------------------------------------------------------------------------------

const PROFILE_GC_STALL_FRAME = r"^(jl_safepoint_wait_gc|_?i?jl_gc_collect|i?jl_gc_small_alloc.*|ijl_gc_managed_malloc)$"
const PROFILE_JIT_FRAME = r"jl_compile|jl_generate_fptr|jl_emit|codegen|jl_type_infer|typeinf|jl_add_to_ee|sizedOptimize|LLVM|llvm"

_frame_label(frame) = string(frame.func, " ", basename(string(frame.file)), ":", frame.line)
_frame_pool(tid::Int) = try
    string(Threads.threadpooldescription(tid))
catch
    "unknown"
end

"""
    profile_thread_summary(data, lidict; top = 15) -> NamedTuple

Parse raw `Profile.fetch(include_meta = true)` blocks — `[leaf ip … root ip, threadid, taskid,
cpu clock, sleep state, 0, 0]` — into per-thread utilisation (1 − sleeping share, the figure
`Profile.print(groupby = :thread)` prints), and over the SAMPLER threads (Julia's `default` pool,
which is where the queued chain tasks run; GC threads report as `foreign: gc`): the share of awake
samples inside a GC-stall frame, the share inside JIT/LLVM compilation, and the `top` leaf frames by
self count.
"""
function profile_thread_summary(data::AbstractVector{<:Unsigned}, lidict; top::Int = 15)
    total = Dict{Int,Int}()
    sleeping = Dict{Int,Int}()
    sampler_awake = 0
    sampler_gc = 0
    sampler_jit = 0
    self_counts = Dict{String,Int}()
    pools = Dict{Int,String}()
    pool(tid) = get!(() -> _frame_pool(tid), pools, tid)
    block_start = firstindex(data)
    for i in eachindex(data)
        Profile.is_block_end(data, i) || continue
        tid = Int(data[i - Profile.META_OFFSET_THREADID])
        asleep = data[i - Profile.META_OFFSET_SLEEPSTATE] - 1 == 1
        ips = block_start:(i - Profile.nmeta - 2)
        block_start = i + 1
        total[tid] = get(total, tid, 0) + 1
        asleep && (sleeping[tid] = get(sleeping, tid, 0) + 1)
        (asleep || pool(tid) != "default" || isempty(ips)) && continue
        sampler_awake += 1
        in_gc = false
        in_jit = false
        for k in ips
            for frame in get(lidict, data[k], ())
                name = string(frame.func)
                in_gc |= occursin(PROFILE_GC_STALL_FRAME, name)
                in_jit |= occursin(PROFILE_JIT_FRAME, name) || occursin("llvm", string(frame.file))
            end
        end
        sampler_gc += in_gc
        sampler_jit += in_jit
        leaf = get(lidict, data[first(ips)], ())
        isempty(leaf) || (key = _frame_label(first(leaf)); self_counts[key] = get(self_counts, key, 0) + 1)
    end
    threads = [(; thread = tid, pool = pool(tid), samples = n,
                  utilisation = 1 - get(sleeping, tid, 0) / n)
               for (tid, n) in sort!(collect(total); by = first)]
    sampler = filter(t -> t.pool == "default", threads)
    utilisations = [t.utilisation for t in sampler]
    ranked = first(sort!(collect(self_counts); by = last, rev = true), top)
    return (; threads, n_sampler_threads = length(sampler),
              utilisation_mean = isempty(utilisations) ? PERF_UNAVAILABLE : sum(utilisations) / length(utilisations),
              utilisation_min = isempty(utilisations) ? PERF_UNAVAILABLE : minimum(utilisations),
              gc_stall_share = _ratio(sampler_gc, sampler_awake),
              jit_share = _ratio(sampler_jit, sampler_awake),
              top_frames = [(; frame, self = n, share = n / max(sampler_awake, 1)) for (frame, n) in ranked])
end

# ------------------------------------------------------------------------------
# Allocation outside the tape: Profile.Allocs on full sampler steps (rev2 item 4)
# ------------------------------------------------------------------------------

const _PERF_BASE_FILE = r"^(\./|@Base|@Core|client\.jl|boot\.jl)|/share/julia/(base|stdlib)/|/usr/share/julia/"

"First frame outside Julia's Base/stdlib, labelled with its package (or `src/…` for this repo)."
function _first_source_frame(stacktrace)
    for frame in stacktrace
        file = string(frame.file)
        (isempty(file) || occursin(_PERF_BASE_FILE, file) || frame.from_c) && continue
        pkg = match(r"/packages/([^/]+)/", file)
        owner = pkg !== nothing ? pkg.captures[1] :
                occursin(r"/src/", file) ? "BayesianFootball" : "?"
        return string(owner, ": ", _frame_label(frame))
    end
    return "(no source frame)"
end

function _alloc_sites(result, sample_rate, per, source; top::Int = 10)
    bytes = Dict{String,Float64}()
    counts = Dict{String,Int}()
    types = Dict{String,String}()
    for a in result.allocs
        key = _first_source_frame(a.stacktrace)
        bytes[key] = get(bytes, key, 0.0) + a.size
        counts[key] = get(counts, key, 0) + 1
        haskey(types, key) || (types[key] = first(string(a.type), 60))
    end
    ranked = first(sort!(collect(bytes); by = last, rev = true), top)
    return [(; source, site, type = types[site], samples = counts[site],
               est_bytes_per_unit = b / sample_rate / per) for (site, b) in ranked]
end

"""
    non_tape_allocations(model, fs, sampler; n_gradients = 50, n_steps = 20, sample_rate = 0.05)

`Profile.Allocs` on full sampler steps, where stack traces reach real source (rev2 item 4):

  * `n_gradients` × `LogDensityProblems.logdensity_and_gradient` through Turing's
    `LogDensityFunction` with `AutoReverseDiff(compile = true)` — the call NUTS makes per leapfrog;
    estimates are bytes per GRADIENT;
  * `n_steps` NUTS transitions from `AbstractMCMC.steps`, after the chain's setup step (tape
    recording and compilation are excluded); estimates are bytes per LEAPFROG.

Returns the top 10 sites of each by estimated bytes, with the first non-Base frame. A zero-allocation
tape leaves only wrapper and sampler allocation here; the tape's own allocation has no source frame
and is attributed by `tape_allocation_profile` instead.
"""
function non_tape_allocations(model, fs, sampler; n_gradients::Int = 50, n_steps::Int = 20,
                              sample_rate::Float64 = 0.05, seed::Int = 20260911)
    turing_model = Models.PreGame.build_turing_model(model, fs)
    Random.seed!(seed)
    varinfo = DynamicPPL.link!!(DynamicPPL.VarInfo(turing_model), turing_model)
    θ = copy(varinfo[:])
    density = DynamicPPL.LogDensityFunction(turing_model, DynamicPPL.getlogjoint_internal,
                                            varinfo; adtype = Turing.AutoReverseDiff(compile = true))
    for _ in 1:5
        LogDensityProblems.logdensity_and_gradient(density, θ)
    end
    Profile.Allocs.clear()
    Profile.Allocs.@profile sample_rate = sample_rate for _ in 1:n_gradients
        LogDensityProblems.logdensity_and_gradient(density, θ)
    end
    gradient_sites = _alloc_sites(Profile.Allocs.fetch(), sample_rate, n_gradients, "gradient")
    Profile.Allocs.clear()

    config = _probe_sampler(sampler, 1, max(n_steps, 10), n_steps)
    init = Samplers.get_init_params(turing_model, config.initialisation, 1)[1]
    steps = Turing.AbstractMCMC.steps(Random.default_rng(), turing_model,
                                      Samplers.nuts_algorithm(config);
                                      initial_params = init, nadapts = config.n_warmup)
    transition, state = Logging.with_logger(Logging.ConsoleLogger(stderr, Logging.Warn)) do
        first_step = iterate(steps)                          # setup: tape record + compile
        for _ in 1:3
            first_step = iterate(steps, first_step[2])       # JIT the step itself
        end
        first_step
    end
    leapfrogs = Ref(0)
    Profile.Allocs.@profile sample_rate = sample_rate for _ in 1:n_steps
        transition, state = iterate(steps, state)
        leapfrogs[] += transition.stat.n_steps
    end
    step_sites = _alloc_sites(Profile.Allocs.fetch(), sample_rate, max(leapfrogs[], 1), "nuts_step")
    Profile.Allocs.clear()
    return (; gradient_sites, step_sites, sample_rate, n_gradients, n_steps,
              step_leapfrogs = leapfrogs[])
end

# ------------------------------------------------------------------------------
# The probe
# ------------------------------------------------------------------------------

_sampler_field(sampler, name::Symbol, default) =
    hasproperty(sampler, name) ? getproperty(sampler, name) : default

function _probe_sampler(sampler, n_chains::Int, n_warmup::Int, n_samples::Int)
    init = _sampler_field(sampler, :initialisation, nothing)
    return Samplers.QueuedNUTSConfig(
        n_samples = n_samples, n_warmup = n_warmup, n_chains = n_chains,
        accept_rate = _sampler_field(sampler, :accept_rate, 0.65),
        max_depth = _sampler_field(sampler, :max_depth, 10),
        initialisation = init, show_progress = false, silence_initial_stepsize = true)
end

"""
One chain exactly as `Samplers.run_sampler(::QueuedNUTSConfig, chain_id)` runs it — own model,
own initial point, `Samplers.nuts_algorithm` (compiled ReverseDiff) — except that warm-up is KEPT
(`discard_adapt = false`, `n_warmup + n_samples` iterations) so every leapfrog is counted.
Returns the chain's total leapfrogs (the initial transition reports no `n_steps`).
"""
function _probe_chain(model, fs, config, chain_id::Int)
    turing_model = Models.PreGame.build_turing_model(model, fs)
    init = Samplers.get_init_params(turing_model, config.initialisation, config.n_chains)
    logger = Logging.ConsoleLogger(stderr, Logging.Warn)
    chain = Logging.with_logger(logger) do
        Turing.sample(turing_model, Samplers.nuts_algorithm(config),
                      config.n_warmup + config.n_samples;
                      progress = false, discard_adapt = false,
                      initial_params = init[chain_id])
    end
    return round(Int, sum(skipmissing(vec(Array(chain[:n_steps])))))
end

_counters(proc_root) = (; t = time_ns(), gc = Base.gc_num(), bytes = Base.gc_bytes(),
                          cpu = _proc_cpu_snapshot(proc_root))

_ratio(num, den) = (num === nothing || den === nothing || den <= 0) ? PERF_UNAVAILABLE : num / den

"Start the profiler with a fresh buffer; `false` if it cannot run here."
function _profiler_start(n::Int, delay::Float64)
    try
        Profile.clear()
        Profile.init(n = n, delay = delay)
        Profile.start_timer()
        return true
    catch err
        @warn "statistical profiler unavailable; utilisation falls back to /proc" exception = err
        return false
    end
end

"""
    sampling_performance_probe(model, feature_set, sampler; n_chains = Threads.nthreads(),
        n_warmup = 50, n_samples = 50, profile = true, profile_n = 10^7, profile_delay = 0.002,
        allocs = true, proc_root = "/proc") -> NamedTuple

Run `n_chains` NUTS chains of `n_warmup + n_samples` iterations concurrently — one task per chain
behind a semaphore of `Threads.nthreads()`, as `QueuedExecution` schedules the grid — with the
candidate's acceptance target, tree depth and initialisation, and measure:

  * static (the tape audit): bytes per compiled gradient, bare latency, instructions, allocating
    instructions, warm tape record + compile time;
  * wall time and total leapfrogs (warm-up included);
  * in-situ ms per leapfrog = (Σ chain busy time − chains × tape setup) ÷ leapfrogs, and the
    efficiency ratio against the bare latency;
  * bytes per leapfrog over the whole run;
  * GC (`GC_Diff`) over the FULL-CONCURRENCY window — start until the first chain finishes, while
    every thread has a chain — so the straggler tail every short probe has is not mistaken for poor
    utilisation; whole-run GC share beside it;
  * with `profile = true`, the statistical profiler (`Profile.init(n = profile_n, delay =
    profile_delay)`) from the start of the window until the first chain finishes or the buffer is
    90 % full — a sub-window, which bounds the profiler's overhead: per-thread utilisation of the
    sampler threads (mean, min), GC-stall and JIT shares, top 15 frames. Without it, or if the
    profiler cannot start, utilisation comes from `/proc/self/task` and `utilisation_source` says so;
  * with `allocs = true`, `non_tape_allocations` (single-threaded, before the window).

A one-chain, few-iteration run first compiles the sampler for this model type, outside the window.
"""
function sampling_performance_probe(model, fs, sampler;
                                    n_chains::Int = Threads.nthreads(),
                                    n_warmup::Int = 50, n_samples::Int = 50,
                                    profile::Bool = true, profile_n::Int = 10^7,
                                    profile_delay::Float64 = 0.002,
                                    allocs::Bool = true,
                                    proc_root::AbstractString = "/proc",
                                    seed::Int = 20260911)
    n_chains >= 1 || error("sampling_performance_probe needs n_chains ≥ 1; got $n_chains")
    static = tape_metrics(model, fs; seed)
    config = _probe_sampler(sampler, n_chains, n_warmup, n_samples)
    _probe_chain(model, fs, _probe_sampler(sampler, 1, 2, 2), 1)      # JIT, outside the window
    outside = allocs ? non_tape_allocations(model, fs, sampler; seed) : nothing

    n_threads = Threads.nthreads()
    active = min(n_chains, n_threads)
    sampler_tids = sampler_thread_ids()
    busy = zeros(n_chains)
    steps = zeros(Int, n_chains)
    first_done = Threads.Atomic{Bool}(false)
    window_end = Ref{Any}(nothing)
    semaphore = Base.Semaphore(n_threads)

    start = _counters(proc_root)
    profiling = profile && _profiler_start(profile_n, profile_delay)
    tasks = map(1:n_chains) do chain_id
        Threads.@spawn begin
            Base.acquire(semaphore)
            try
                began = time_ns()
                steps[chain_id] = _probe_chain(model, fs, config, chain_id)
                busy[chain_id] = (time_ns() - began) / 1e9
                Threads.atomic_xchg!(first_done, true) || (window_end[] = _counters(proc_root))
            finally
                Base.release(semaphore)
            end
        end
    end
    profile_seconds = 0.0
    if profiling
        while !first_done[] && Profile.len_data() < 0.9 * Profile.maxlen_data() &&
              !all(istaskdone, tasks)
            sleep(0.05)
        end
        Profile.stop_timer()
        profile_seconds = (time_ns() - start.t) / 1e9
    end
    foreach(fetch, tasks)
    stop = _counters(proc_root)
    mid = something(window_end[], stop)

    wall = (stop.t - start.t) / 1e9
    window = (mid.t - start.t) / 1e9
    leapfrogs = sum(steps)
    adjusted_busy = max(sum(busy) - n_chains * static.setup_seconds, 0.0)
    ms_per_leapfrog = leapfrogs > 0 ? 1e3 * adjusted_busy / leapfrogs : PERF_UNAVAILABLE
    efficiency = ms_per_leapfrog isa Real ? ms_per_leapfrog / static.gradient_ms : PERF_UNAVAILABLE

    gc_window = _gc_window(start.gc, mid.gc, window)
    gc_run = _gc_window(start.gc, stop.gc, wall)

    profiled = nothing
    if profiling
        data = Profile.fetch(include_meta = true, limitwarn = false)
        profiled = profile_thread_summary(data, Profile.getdict(data))
        Profile.clear()
    end
    cpu_window = _cpu_split(start.cpu, mid.cpu, sampler_tids)
    proc_util = cpu_window === nothing ? PERF_UNAVAILABLE : _ratio(cpu_window.sampler, window * active)
    utilisation, utilisation_min, source =
        profiled !== nothing && profiled.utilisation_mean isa Real ?
            (profiled.utilisation_mean, profiled.utilisation_min, "profile") :
        proc_util isa Real ? (proc_util, PERF_UNAVAILABLE, "proc") :
            (PERF_UNAVAILABLE, PERF_UNAVAILABLE, PERF_UNAVAILABLE)

    return (;
        n_rows = length(fs.data[:flat_home_ids]),
        n_parameters = static.n_parameters,
        tape_instructions = static.tape_instructions,
        tape_bytes_per_gradient = static.allocated_bytes,
        tape_gradient_ms = static.gradient_ms,
        tape_setup_seconds = static.setup_seconds,
        allocating_instructions = static.allocating_instructions,
        threads = n_threads, chains = n_chains, n_warmup, n_samples,
        wall_seconds = wall, window_seconds = window,
        leapfrogs = leapfrogs, leapfrogs_per_chain = collect(steps),
        ms_per_leapfrog, efficiency_ratio = efficiency,
        bytes_per_leapfrog = leapfrogs > 0 ? (stop.bytes - start.bytes) / leapfrogs : PERF_UNAVAILABLE,
        gc_window...,
        gc_share_run = gc_run.gc_share, gc_pauses_run = gc_run.gc_pauses,
        sampler_utilisation = utilisation, sampler_utilisation_min = utilisation_min,
        utilisation_source = source,
        profile_seconds = profiling ? profile_seconds : PERF_UNAVAILABLE,
        profile_threads = profiled === nothing ? NamedTuple[] : profiled.threads,
        gc_stall_share = profiled === nothing ? PERF_UNAVAILABLE : profiled.gc_stall_share,
        jit_share = profiled === nothing ? PERF_UNAVAILABLE : profiled.jit_share,
        top_frames = profiled === nothing ? NamedTuple[] : profiled.top_frames,
        proc_sampler_utilisation = proc_util,
        process_utilisation = cpu_window === nothing ? PERF_UNAVAILABLE :
            _ratio(cpu_window.process, window * n_threads),
        other_cpu_cores = cpu_window === nothing ? PERF_UNAVAILABLE : _ratio(cpu_window.other, window),
        sampler_thread_ids_found = length(sampler_tids),
        non_tape_gradient_sites = outside === nothing ? NamedTuple[] : outside.gradient_sites,
        non_tape_step_sites = outside === nothing ? NamedTuple[] : outside.step_sites,
    )
end

# ------------------------------------------------------------------------------
# Verdict and diagnosis
# ------------------------------------------------------------------------------

_perf_value(x) = x isa Real ? Float64(x) : nothing

"""
    sampling_performance_flags(metrics; thresholds = SAMPLING_PERF_THRESHOLDS) -> Vector

Every threshold the metrics breach, as `(; flag, value, threshold, diagnosis)`. Empty means pass.
A metric that is `"unavailable"` cannot raise its flag. The diagnosis maps the flag, read together
with the other metrics, to its likely cause and the AD-guide section with the fix.
"""
function sampling_performance_flags(m; thresholds = SAMPLING_PERF_THRESHOLDS)
    flags = NamedTuple{(:flag, :value, :threshold, :diagnosis),Tuple{String,Float64,Float64,String}}[]
    tape_bytes = m.tape_bytes_per_gradient
    gc = _perf_value(m.gc_share)
    util = _perf_value(m.sampler_utilisation)
    eff = _perf_value(m.efficiency_ratio)
    stall = _perf_value(get(m, :gc_stall_share, nothing))
    jit = _perf_value(get(m, :jit_share, nothing))
    allocating = tape_bytes > thresholds.max_tape_bytes
    high_gc = gc !== nothing && gc > thresholds.max_gc_share
    stall_note = stall === nothing ? "" :
        " The profiler put $(round(100stall, digits = 1))% of awake sampler-thread samples in GC-stall frames."

    if allocating
        push!(flags, (; flag = "tape_allocation", value = Float64(tape_bytes),
            threshold = Float64(thresholds.max_tape_bytes),
            diagnosis = "The compiled tape allocates $(tape_bytes) B per gradient. A tracked " *
                "scalar (or a plain Real) inside a fused broadcast sends ReverseDiff to " *
                "tracker_∇broadcast: lift it with tape_scalar or reduce first and multiply after " *
                "(sum(x .* w) * ν); clamp bounds go in as Refs. See the allocating-instruction " *
                "table and docs/turing_ad_performance_guide.md §10.5."))
    end
    if high_gc
        push!(flags, (; flag = "gc_share", value = gc, threshold = thresholds.max_gc_share,
            diagnosis = (allocating ?
                "GC takes $(round(100gc, digits = 1))% of wall time and the tape allocates: " *
                "fix the allocating instructions first (AD guide §10.5)." :
                "GC takes $(round(100gc, digits = 1))% of wall time with a zero-allocation " *
                "tape: the garbage comes from the non-tape path — see the Profile.Allocs sites " *
                "(DynamicPPL `LogDensityFunction` wrappers, the NUTS tree, chain storage).") *
                stall_note))
    end
    if util !== nothing && util < thresholds.min_sampler_utilisation
        push!(flags, (; flag = "sampler_utilisation", value = util,
            threshold = thresholds.min_sampler_utilisation,
            diagnosis = high_gc || (stall !== nothing && stall > 0.10) ?
                "Sampler threads were busy $(round(100util, digits = 1))% of the " *
                "full-concurrency window while GC was high: threads wait at safepoints for " *
                "collections (see the hot frames). Remove the allocation (AD guide §10.5)." *
                stall_note :
                "Sampler threads were busy $(round(100util, digits = 1))% of the " *
                "full-concurrency window with low GC: too few chains for the threads, or serial " *
                "work — tape recording/compilation" *
                (jit === nothing ? "" : ", JIT ($(round(100jit, digits = 1))% of samples)") *
                ", initialisation, a lock — dominating a short probe. Check chains ≥ threads " *
                "and tape_setup_seconds against the chain time."))
    end
    if eff !== nothing && eff > thresholds.max_efficiency_ratio
        push!(flags, (; flag = "efficiency_ratio", value = eff,
            threshold = thresholds.max_efficiency_ratio,
            diagnosis = "A leapfrog in situ costs $(round(eff, digits = 2))× the bare compiled " *
                "gradient. With high GC this is collection pauses; with low GC suspect " *
                "per-leapfrog work outside the tape (DynamicPPL/AdvancedHMC overhead — see the " *
                "Profile.Allocs sites and hot frames) or memory-bandwidth contention from a large " *
                "tape across all threads — compare tape_instructions and AD guide §10.1."))
    end
    return flags
end

# ------------------------------------------------------------------------------
# Report
# ------------------------------------------------------------------------------

_fmt(x::Integer) = string(x)
_fmt(x::Real) = isfinite(x) ? string(round(x, sigdigits = 4)) : string(x)
_fmt(x) = string(x)
_pct(x) = x isa Real ? string(round(100x, digits = 1), "%") : string(x)

"""
    write_sampling_performance_report(path, metrics, flags; title, context = NamedTuple()) -> path

A markdown report: context, verdict, metrics table, per-thread profile, hot frames, the
allocating-instruction table (tape), the allocation sites outside the tape, and the diagnosis of
every flag. Creates the directory.
"""
function write_sampling_performance_report(path::AbstractString, m, flags;
                                           title::AbstractString = "Sampling performance",
                                           context = NamedTuple(),
                                           thresholds = SAMPLING_PERF_THRESHOLDS)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# ", title, "\n")
        for (key, value) in pairs(context)
            println(io, "- **", key, "**: ", value)
        end
        println(io, "- **probe**: ", m.chains, " chains × (", m.n_warmup, " warm-up + ",
                m.n_samples, " draws) on ", m.threads, " threads; ", m.n_rows, " rows, ",
                m.n_parameters, " parameters\n")

        println(io, "## Verdict\n")
        println(io, isempty(flags) ? "**pass** — no threshold breached." :
                    "**review** — " * join((f.flag for f in flags), ", ") * ".")
        println(io)

        println(io, "## Metrics\n")
        println(io, "| metric | value | review if |")
        println(io, "|---|---|---|")
        rows = (
            ("bytes per compiled gradient (tape audit)", _fmt(m.tape_bytes_per_gradient), "> $(thresholds.max_tape_bytes) (hard check)"),
            ("bare gradient latency (warmed min)", _fmt(m.tape_gradient_ms) * " ms", ""),
            ("tape instructions", _fmt(m.tape_instructions), ""),
            ("tape record + compile", _fmt(m.tape_setup_seconds) * " s", ""),
            ("wall time", _fmt(m.wall_seconds) * " s", ""),
            ("full-concurrency window", _fmt(m.window_seconds) * " s", ""),
            ("leapfrogs (warm-up included)", _fmt(m.leapfrogs), ""),
            ("in-situ ms per leapfrog", _fmt(m.ms_per_leapfrog), ""),
            ("efficiency ratio", _fmt(m.efficiency_ratio), "> $(thresholds.max_efficiency_ratio)"),
            ("bytes per leapfrog", _fmt(m.bytes_per_leapfrog), ""),
            ("GC share (window, GC_Diff)", _pct(m.gc_share), "> $(_pct(thresholds.max_gc_share))"),
            ("GC share (whole run)", _pct(m.gc_share_run), ""),
            ("GC pauses / full sweeps (window)", "$(m.gc_pauses) / $(m.gc_full_sweeps)", ""),
            ("GC allocated (window)", _fmt(m.gc_allocd_bytes / 2^30) * " GiB", ""),
            ("time to safepoint (window, summed)", _fmt(m.gc_time_to_safepoint_seconds) * " s", ""),
            ("max GC pause / max time to safepoint (process)", _fmt(m.gc_max_pause_ms) * " ms / " * _fmt(m.gc_max_time_to_safepoint_ms) * " ms", ""),
            ("sampler-thread utilisation, mean ($(m.utilisation_source))", _pct(m.sampler_utilisation), "< $(_pct(thresholds.min_sampler_utilisation))"),
            ("sampler-thread utilisation, min", _pct(m.sampler_utilisation_min), ""),
            ("sampler samples in GC-stall frames", _pct(m.gc_stall_share), ""),
            ("sampler samples in JIT/LLVM frames", _pct(m.jit_share), ""),
            ("profiled sub-window", _fmt(m.profile_seconds) * " s", ""),
            ("/proc sampler-thread CPU ÷ (window × threads) (fallback)", _pct(m.proc_sampler_utilisation), ""),
            ("/proc process CPU ÷ (window × threads)", _pct(m.process_utilisation), ""),
            ("/proc non-sampler CPU (cores)", _fmt(m.other_cpu_cores), ""),
        )
        for (name, value, rule) in rows
            println(io, "| ", name, " | ", value, " | ", rule, " |")
        end
        println(io)

        println(io, "## Threads (statistical profiler)\n")
        if isempty(m.profile_threads)
            println(io, "Profiler not run; utilisation source: ", m.utilisation_source, ".\n")
        else
            println(io, "| thread | pool | samples | utilisation |")
            println(io, "|---|---|---|---|")
            for t in m.profile_threads
                println(io, "| ", t.thread, " | ", t.pool, " | ", t.samples, " | ", _pct(t.utilisation), " |")
            end
            println(io)
            println(io, "### Top frames by self samples (sampler threads, awake)\n")
            println(io, "| frame | self | share |")
            println(io, "|---|---|---|")
            for f in m.top_frames
                println(io, "| `", f.frame, "` | ", f.self, " | ", _pct(f.share), " |")
            end
            println(io)
        end

        println(io, "## Allocating tape instructions (tape audit)\n")
        if isempty(m.allocating_instructions)
            println(io, "None — the compiled tape replays without allocating.\n")
        else
            println(io, "| # | bytes per replay | instruction (T = tracked array, TReal = tracked scalar, Real = plain scalar) |")
            println(io, "|---|---|---|")
            for row in m.allocating_instructions
                println(io, "| ", row.index, " | ", row.bytes, " | `", row.description, "` |")
            end
            println(io)
        end

        println(io, "## Allocation outside the tape (Profile.Allocs)\n")
        sites = vcat(collect(m.non_tape_gradient_sites), collect(m.non_tape_step_sites))
        if isempty(sites)
            println(io, "Not measured.\n")
        else
            println(io, "Estimated bytes per gradient (`gradient`: `logdensity_and_gradient` through ",
                    "Turing's `LogDensityFunction`) or per leapfrog (`nuts_step`: NUTS transitions ",
                    "after setup). Tape allocation has no source frame here — see the tape audit.\n")
            println(io, "| source | first non-Base frame | type | samples | est. bytes per unit |")
            println(io, "|---|---|---|---|---|")
            for s in sites
                println(io, "| ", s.source, " | `", s.site, "` | `", s.type, "` | ", s.samples,
                        " | ", _fmt(s.est_bytes_per_unit), " |")
            end
            println(io)
        end

        println(io, "## Diagnosis\n")
        if isempty(flags)
            println(io, "Nothing to diagnose. See docs/turing_ad_performance_guide.md §10.5 for what each metric means.")
        else
            for f in flags
                println(io, "- **", f.flag, "** (", _fmt(f.value), " vs ", _fmt(f.threshold), "): ",
                        f.diagnosis)
            end
        end
    end
    return path
end

"""
    perf_summary(perf) -> String

The smoke's performance probe as `[SUMMARY]` fields:
`gc=<%> util=<%> ms_leapfrog=<x> eff=<ratio> perf=<pass|review>`.
"""
function perf_summary(perf)
    perf === nothing && return "gc=n/a util=n/a ms_leapfrog=n/a eff=n/a perf=n/a"
    m = perf.metrics
    m === nothing && return "gc=n/a util=n/a ms_leapfrog=n/a eff=n/a perf=$(perf.status)"
    number(x, digits) = x isa Real ? string(round(x, digits = digits)) : "n/a"
    percent(x) = x isa Real ? string(round(100x, digits = 1), "%") : "n/a"
    return "gc=$(percent(m.gc_share)) util=$(percent(m.sampler_utilisation)) " *
           "ms_leapfrog=$(number(m.ms_per_leapfrog, 3)) eff=$(number(m.efficiency_ratio, 2)) " *
           "perf=$(perf.status)"
end

# src/harness/perf_monitor.jl
#
# Load-realistic sampling-performance probe for the smoke stage.
#
# Specification: experiments/claude_zero_alloc_addendum_perf_monitor.md (§2.6). Rationale:
# docs/turing_ad_performance_guide.md §10.5 — a gradient that looks fast in isolation can still
# sample slowly under 16-way concurrency (GC pauses, idle sampler threads, a slow in-situ
# gradient), and the old smoke's 2 folds × 2 chains could not reproduce that contention.
#
# Like tape_profile.jl this file is include-able on its own (it needs `BayesianFootball`'s
# `Models` and `Samplers`, Turing, DynamicPPL, LogDensityProblems, ReverseDiff, Random, Logging and
# the tape_profile.jl helpers in scope), so `scripts/sampling_perf_probe.jl` can point it at an
# older checkout — which is how the pre-fix engine's report is produced.

"""
Thresholds for the `sampling_performance` REVIEW check (never hard). Change them here only; the
addendum above states the first three, and `max_tape_bytes` mirrors the hard `tape_allocation`
limit so a standalone probe (which skips the hard check) still names that cause.

  * `max_gc_share`            — GC pause time ÷ wall time over the full-concurrency window
  * `min_sampler_utilisation` — sampler-thread CPU ÷ (wall × active sampler threads), same window
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
    return round(Int, sum(Array(chain[:n_steps])))
end

_counters(proc_root) = (; t = time_ns(), gc_ns = Base.gc_time_ns(), bytes = Base.gc_bytes(),
                          cpu = _proc_cpu_snapshot(proc_root))

_ratio(num, den) = (num === nothing || den === nothing || den <= 0) ? PERF_UNAVAILABLE : num / den

"""
    sampling_performance_probe(model, feature_set, sampler; n_chains = Threads.nthreads(),
                               n_warmup = 50, n_samples = 50, proc_root = "/proc") -> NamedTuple

Run `n_chains` NUTS chains of `n_warmup + n_samples` iterations concurrently — one task per chain
behind a semaphore of `Threads.nthreads()`, as `QueuedExecution` schedules the grid — with the
candidate's acceptance target, tree depth and initialisation, and measure:

  * static: bytes per compiled gradient, bare latency, instructions, allocating instructions;
  * wall time and total leapfrogs (warm-up included);
  * in-situ ms per leapfrog = (Σ chain busy time − chains × tape setup) ÷ leapfrogs, and the
    efficiency ratio against the bare latency;
  * bytes per leapfrog (`Base.gc_bytes` delta) over the whole run;
  * GC share (`Base.gc_time_ns` delta ÷ wall) and CPU utilisation over the FULL-CONCURRENCY
    window — from the start until the first chain finishes, while every thread has a chain — so
    the tail of stragglers a 16-chain probe always has is not mistaken for poor utilisation.
    Whole-run figures are reported beside them.

`sampler_utilisation` is sampler-thread CPU ÷ (window × active threads); `process_utilisation` is
all process CPU ÷ (window × nthreads), which GC threads inflate. `other_cpu_cores` is non-sampler
CPU expressed in cores. CPU figures are `"unavailable"` without `/proc`.

A one-chain, few-iteration run first compiles the sampler for this model type, outside the window.
"""
function sampling_performance_probe(model, fs, sampler;
                                    n_chains::Int = Threads.nthreads(),
                                    n_warmup::Int = 50, n_samples::Int = 50,
                                    proc_root::AbstractString = "/proc",
                                    seed::Int = 20260911)
    n_chains >= 1 || error("sampling_performance_probe needs n_chains ≥ 1; got $n_chains")
    static = tape_metrics(model, fs; seed)
    config = _probe_sampler(sampler, n_chains, n_warmup, n_samples)
    _probe_chain(model, fs, _probe_sampler(sampler, 1, 2, 2), 1)      # JIT, outside the window

    n_threads = Threads.nthreads()
    active = min(n_chains, n_threads)
    sampler_tids = sampler_thread_ids()
    busy = zeros(n_chains)
    steps = zeros(Int, n_chains)
    first_done = Threads.Atomic{Bool}(false)
    window_end = Ref{Any}(nothing)
    semaphore = Base.Semaphore(n_threads)

    start = _counters(proc_root)
    @sync for chain_id in 1:n_chains
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
    stop = _counters(proc_root)
    mid = something(window_end[], stop)

    wall = (stop.t - start.t) / 1e9
    window = (mid.t - start.t) / 1e9
    leapfrogs = sum(steps)
    adjusted_busy = max(sum(busy) - n_chains * static.setup_seconds, 0.0)
    ms_per_leapfrog = leapfrogs > 0 ? 1e3 * adjusted_busy / leapfrogs : PERF_UNAVAILABLE
    efficiency = ms_per_leapfrog isa Real ? ms_per_leapfrog / static.gradient_ms : PERF_UNAVAILABLE

    cpu_window = _cpu_split(start.cpu, mid.cpu, sampler_tids)
    cpu_run = _cpu_split(start.cpu, stop.cpu, sampler_tids)
    util(cpu, seconds, threads) = cpu === nothing ? PERF_UNAVAILABLE :
        _ratio(cpu.sampler, seconds * threads)
    proc_util(cpu, seconds) = cpu === nothing ? PERF_UNAVAILABLE :
        _ratio(cpu.process, seconds * n_threads)
    other_cores(cpu, seconds) = cpu === nothing ? PERF_UNAVAILABLE : _ratio(cpu.other, seconds)

    return (;
        n_rows = length(fs.data[:flat_home_ids]),
        n_parameters = static.n_parameters,
        tape_instructions = static.tape_instructions,
        tape_bytes_per_gradient = static.allocated_bytes,
        tape_gradient_ms = static.gradient_ms,
        tape_setup_seconds = static.setup_seconds,
        allocating_instructions = static.allocating_instructions,
        threads = n_threads, chains = n_chains, n_warmup, n_samples,
        sampler_thread_ids_found = length(sampler_tids),
        wall_seconds = wall, window_seconds = window,
        leapfrogs = leapfrogs, leapfrogs_per_chain = collect(steps),
        ms_per_leapfrog, efficiency_ratio = efficiency,
        bytes_per_leapfrog = leapfrogs > 0 ? (stop.bytes - start.bytes) / leapfrogs : PERF_UNAVAILABLE,
        gc_share = _ratio((mid.gc_ns - start.gc_ns) / 1e9, window),
        gc_share_run = _ratio((stop.gc_ns - start.gc_ns) / 1e9, wall),
        sampler_utilisation = util(cpu_window, window, active),
        sampler_utilisation_run = util(cpu_run, wall, active),
        process_utilisation = proc_util(cpu_window, window),
        other_cpu_cores = other_cores(cpu_window, window),
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
    util === nothing && (util = _perf_value(m.process_utilisation))
    eff = _perf_value(m.efficiency_ratio)
    allocating = tape_bytes > thresholds.max_tape_bytes
    high_gc = gc !== nothing && gc > thresholds.max_gc_share

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
            diagnosis = allocating ?
                "GC pauses take $(round(100gc, digits = 1))% of wall time and the tape allocates: " *
                "fix the allocating instructions first (AD guide §10.5)." :
                "GC pauses take $(round(100gc, digits = 1))% of wall time with a zero-allocation " *
                "tape: the garbage comes from the non-tape path — DynamicPPL `logdensity` / " *
                "LogDensityFunction wrappers, the sampler's own per-iteration state, or " *
                "chain construction. Compare bytes per leapfrog with the tape's 0 B."))
    end
    if util !== nothing && util < thresholds.min_sampler_utilisation
        push!(flags, (; flag = "sampler_utilisation", value = util,
            threshold = thresholds.min_sampler_utilisation,
            diagnosis = high_gc ?
                "Sampler threads were busy $(round(100util, digits = 1))% of the " *
                "full-concurrency window while GC was high: threads are stopped in collections " *
                "(and GC threads show as other_cpu_cores). Remove the allocation (AD guide §10.5)." :
                "Sampler threads were busy $(round(100util, digits = 1))% of the " *
                "full-concurrency window with low GC: too few chains for the threads, or serial " *
                "work — tape recording/compilation, initialisation, a lock — dominating a short " *
                "probe. Check chains ≥ threads and tape_setup_seconds against the chain time."))
    end
    if eff !== nothing && eff > thresholds.max_efficiency_ratio
        push!(flags, (; flag = "efficiency_ratio", value = eff,
            threshold = thresholds.max_efficiency_ratio,
            diagnosis = "A leapfrog in situ costs $(round(eff, digits = 2))× the bare compiled " *
                "gradient. With high GC this is collection pauses; with low GC suspect " *
                "per-leapfrog work outside the tape (DynamicPPL/AdvancedHMC overhead) or memory " *
                "bandwidth contention from a large tape across all threads — compare " *
                "tape_instructions and the latency bar in AD guide §10.1."))
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

A markdown report: context, verdict, metrics table, allocating-instruction table, and the
diagnosis of every flag. Creates the directory.
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
            ("bytes per compiled gradient", _fmt(m.tape_bytes_per_gradient), "> $(thresholds.max_tape_bytes) (hard check)"),
            ("bare gradient latency (warmed min)", _fmt(m.tape_gradient_ms) * " ms", ""),
            ("tape instructions", _fmt(m.tape_instructions), ""),
            ("tape record + compile", _fmt(m.tape_setup_seconds) * " s", ""),
            ("wall time", _fmt(m.wall_seconds) * " s", ""),
            ("full-concurrency window", _fmt(m.window_seconds) * " s", ""),
            ("leapfrogs (warm-up included)", _fmt(m.leapfrogs), ""),
            ("in-situ ms per leapfrog", _fmt(m.ms_per_leapfrog), ""),
            ("efficiency ratio", _fmt(m.efficiency_ratio), "> $(thresholds.max_efficiency_ratio)"),
            ("bytes per leapfrog", _fmt(m.bytes_per_leapfrog), ""),
            ("GC share (window)", _pct(m.gc_share), "> $(_pct(thresholds.max_gc_share))"),
            ("GC share (whole run)", _pct(m.gc_share_run), ""),
            ("sampler-thread utilisation (window)", _pct(m.sampler_utilisation), "< $(_pct(thresholds.min_sampler_utilisation))"),
            ("sampler-thread utilisation (whole run)", _pct(m.sampler_utilisation_run), ""),
            ("process CPU ÷ (window × threads)", _pct(m.process_utilisation), ""),
            ("non-sampler CPU (cores)", _fmt(m.other_cpu_cores), ""),
            ("sampler threads identified", _fmt(m.sampler_thread_ids_found), ""),
        )
        for (name, value, rule) in rows
            println(io, "| ", name, " | ", value, " | ", rule, " |")
        end
        println(io)

        println(io, "## Allocating tape instructions\n")
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

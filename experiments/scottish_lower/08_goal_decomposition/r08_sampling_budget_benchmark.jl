# ==============================================================================
# r08 — Experiment 08 Fold 1 NUTS sampling-budget benchmark (TODO 001)
# ==============================================================================
#
# WHAT THIS IS. A matched comparison of pre-declared NUTS configurations on the
# identical Fold 1 data, filtration and model arms, through the exact production
# sampling path (`sample_fold` → `Samplers.run_sampler(::QueuedNUTSConfig, chain_id)`):
#
#   A   4 chains × 1,000 warmup × 1,000 retained · δ = 0.95 · max_depth 10 (control)
#   B   4 chains ×   500 warmup ×   500 retained · δ = 0.95 · max_depth 10
#   C   4 chains ×   500 warmup ×   500 retained · δ = 0.90 · max_depth 10
#   D8  B with max_depth 8 — optional follow-up, only via L08_BENCH_CONFIGS.
#
# A is asserted field-for-field equal to `L08_SAMPLER`. The metric (Turing's default
# diagonal) and the uniform initialisation are shared by every configuration.
#
# DESIGN
#   * Each (model, configuration) cell gets `L08_BENCH_REPS` independent 4-chain fits.
#   * All chains of all cells enter ONE flat 16-slot FIFO queue, interleaved
#     rep-major in a seeded shuffle, so every configuration is sampled throughout the
#     run: thermal drift and the queue tail are shared, not confounded with config.
#   * Seeds are common across cells — (rep, chain) gives the same seed for every
#     configuration and model — so initial values are paired.
#   * Per-chain start/stop timestamps are kept; the mean number of concurrently
#     running chains over each chain's lifetime is reported so load is auditable.
#
# PRIMARY METRIC (declared before sampling and written to the manifest):
#   min ESS per wall-second, where min ESS = min over parameters of min(bulk, tail)
#   ESS of the pooled 4-chain fit and wall-seconds = the slowest of its 4 chains.
#   Secondary: min ESS per core-second (sum of the 4 chains' elapsed). A flat
#   16-slot production queue pays core-seconds, so grid projections use it.
#   Hard gates (a configuration is safe only if EVERY replicate passes):
#   max R-hat ≤ 1.05, min bulk and tail ESS ≥ 400, zero divergences, BFMI ≥ 0.30,
#   depth-capped rate < 5%. The committed production gate (ESS ≥ 200, otherwise
#   identical) is reported alongside; divergence counts are always explicit.
#
# WHAT THIS IS NOT. It writes no rows to any database and persists no Fit. Fold 1
# is one of 40 folds; §7 audits the existing production-grid checkpoints so the
# fold-1 ESS margin can be compared with the worst fold. Sampling is disabled
# unless L08_RUN_BENCH=true; otherwise the runner preflights only.
# ==============================================================================

# %%
# ==============================================================================
# 1. Packages and implementation
# ==============================================================================
using BayesianFootball
using CSV
using DataFrames
using Dates
using LinearAlgebra
using MCMCChains
using Printf
using Random
using Serialization
using Statistics
using ThreadPinning
import Turing

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)
Threads.nthreads() == 16 || error(
    "Experiment 08 sampling benchmark requires 16 physical-core Julia threads on mcmc-beast; got $(Threads.nthreads())")

include(joinpath(@__DIR__, "l08_workflow.jl"))
include(joinpath(@__DIR__, "l08_incident_data.jl"))
include(joinpath(@__DIR__, "l08_decomposed_models.jl"))
include(joinpath(@__DIR__, "l08_model_checks.jl"))

const R08B_INF = BayesianFootball.Training.Inference

# %%
# ==============================================================================
# 2. Configuration
# ==============================================================================
const R08B_RUN = lowercase(get(ENV, "L08_RUN_BENCH", "false")) in ("1", "true", "yes")
const R08B_FOLD = 1
const R08B_REPS = parse(Int, get(ENV, "L08_BENCH_REPS", "4"))
const R08B_SLOTS = Threads.nthreads()
const R08B_MODELS = String.(split(get(ENV, "L08_BENCH_MODELS",
    "m00_recombined_control,m01_decomposed_baseline"), ","))
const R08B_CONFIG_NAMES = String.(split(get(ENV, "L08_BENCH_CONFIGS", "A,B,C"), ","))
const R08B_SEED_BASE = 20_260_910
const R08B_TAG = get(ENV, "L08_BENCH_TAG", Dates.format(Dates.now(), "yyyymmdd_HHMMSS"))
const R08B_OUT = joinpath(@__DIR__, "results", "sampling_budget_fold1", R08B_TAG)
const R08B_GRID_CHECKPOINTS = joinpath(@__DIR__, "results", "m00_recombined_control", "checkpoints")
const R08B_SOURCE_FILES = [
    joinpath(@__DIR__, "l08_workflow.jl"),
    joinpath(@__DIR__, "l08_decomposed_models.jl"),
    joinpath(@__DIR__, "l08_incident_data.jl"),
    joinpath(@__DIR__, "l08_model_checks.jl"),
    @__FILE__,
]

const R08B_CONFIGS = Dict(
    "A"  => (warmup = 1_000, retained = 1_000, delta = 0.95, max_depth = 10),
    "B"  => (warmup =   500, retained =   500, delta = 0.95, max_depth = 10),
    "C"  => (warmup =   500, retained =   500, delta = 0.90, max_depth = 10),
    "D8" => (warmup =   500, retained =   500, delta = 0.95, max_depth =  8),
)
const R08B_GATES = (max_rhat = 1.05, min_ess_strict = 400.0, min_ess_production = 200.0,
                    min_bfmi = 0.30, max_treedepth_rate = 0.05)

all(name -> haskey(R08B_CONFIGS, name), R08B_CONFIG_NAMES) || error(
    "unknown configuration in $(R08B_CONFIG_NAMES); known: $(sort!(collect(keys(R08B_CONFIGS))))")
R08B_REPS >= 1 || error("L08_BENCH_REPS must be positive")

r08b_sampler(cfg) = QueuedNUTSConfig(
    n_samples = cfg.retained,
    n_warmup = cfg.warmup,
    n_chains = L08_CHAINS,
    accept_rate = cfg.delta,
    max_depth = cfg.max_depth,
)
const R08B_SAMPLERS = Dict(name => r08b_sampler(R08B_CONFIGS[name]) for name in R08B_CONFIG_NAMES)

# Configuration A must be the production recipe itself, not a near copy.
let a = r08b_sampler(R08B_CONFIGS["A"])
    all(getfield(a, f) == getfield(L08_SAMPLER, f) for f in fieldnames(QueuedNUTSConfig)) || error(
        "configuration A differs from L08_SAMPLER; the control would not be the production sampler")
end

# %%
# ==============================================================================
# 3. Data, models, and the one canonical fold
# ==============================================================================
l08_load_runtime_env!()
ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 10_000)
registry = l08_registry(ds, nothing; output_dir = R08B_OUT, source_files = R08B_SOURCE_FILES)
splitter = l08_splitter()
incident_registry, incident_snapshot_hash = GoalDecompositionIncidentData.load_registry(joinpath(@__DIR__, "results"))
model_entries = l08_models(incident_registry, incident_snapshot_hash;
                            registry_hash = incident_snapshot_hash)
models = [(String(entry.name), entry.model) for entry in model_entries
          if String(entry.name) in R08B_MODELS]
Set(first.(models)) == Set(R08B_MODELS) || error(
    "requested models $(R08B_MODELS) are not all Experiment 08 candidates $(L08_CANDIDATE_NAMES)")

boundaries = Data.create_id_boundaries(ds, splitter)
length(boundaries) == L08_EXPECTED_FOLDS || error(
    "splitter made $(length(boundaries)) boundaries rather than canonical $(L08_EXPECTED_FOLDS)")
fold_boundary = boundaries[R08B_FOLD:R08B_FOLD]

prepared = Dict{String,Any}()
for (name, model) in models
    original_features = Features.create_features(fold_boundary, ds, model, splitter)
    oos = [Data.get_next_matches(ds, feature, splitter) for feature in original_features]
    all(frame -> nrow(frame) > 0, oos) || error("$name fold $R08B_FOLD has no OOS fixture")
    features = l08_declare_prediction_teams(original_features, oos)
    fs = first(first(features))
    String(fs.data[:goal_decomposition_data_hash]) == model.data_hash || error(
        "$name FeatureSet does not carry the frozen incident snapshot hash")
    n_teams = Int(fs.data[:n_teams])
    n_referees = Int(fs.data[:n_referees])
    prepared[name] = (; model, fs,
        n_training = length(fs.data[:goal_decomposition_training_ids]),
        n_oos = nrow(only(oos)), n_teams, n_referees,
        expected_params = l08_expected_params(Symbol(name), n_teams, n_referees))
end

# Only the metric type parameter: the adtype in `NUTS`'s type is Turing's default, whereas
# `run_sampler` passes `AutoReverseDiff(compile = true)` to `sample` for every chain.
metric_type = string(last(typeof(Turing.NUTS(10, 0.95)).parameters))
git_commit = try
    readchomp(`git -C $(@__DIR__) rev-parse HEAD`)
catch
    "unavailable"
end

println("\n", "="^108)
println(" EXPERIMENT 08 · FOLD $R08B_FOLD NUTS SAMPLING-BUDGET BENCHMARK (TODO 001)")
println("="^108)
println("  mode      : ", R08B_RUN ? "SAMPLING AUTHORISED" : "PREFLIGHT ONLY")
println("  host      : ", gethostname(), " · Julia ", VERSION, " · ", Threads.nthreads(),
        " default + ", Threads.nthreads(:interactive), " interactive threads · BLAS ",
        LinearAlgebra.BLAS.get_num_threads())
println("  models    : ", join(R08B_MODELS, ", "))
for name in R08B_CONFIG_NAMES
    cfg = R08B_CONFIGS[name]
    @printf("  config %-3s: %d chains × %d warmup × %d retained · δ = %.2f · max_depth %d\n",
        name, L08_CHAINS, cfg.warmup, cfg.retained, cfg.delta, cfg.max_depth)
end
println("  replicates: ", R08B_REPS, " independent 4-chain fits per cell")
println("  metric    : ", metric_type)
println("  snapshot  : ", registry.snapshot_hash)
println("  incidents : ", incident_snapshot_hash)
println("  git       : ", git_commit)
println("  output    : ", R08B_OUT)
for (name, _) in models
    p = prepared[name]
    @printf("  %-34s training %5d · OOS %3d · teams %3d · referees %3d · params %3d\n",
        name, p.n_training, p.n_oos, p.n_teams, p.n_referees, p.expected_params)
end

# %%
# ==============================================================================
# 4. Pre-declaration manifest
# ==============================================================================
manifest_path = l08_write_manifest!(registry;
    stage = R08B_RUN ? "sampling_budget_authorised" : "sampling_budget_preflight",
    extra = Dict(
        "todo" => "001_sampling_budget_benchmark",
        "fold" => R08B_FOLD,
        "models" => R08B_MODELS,
        "replicates" => R08B_REPS,
        "queue_slots" => R08B_SLOTS,
        "seed_base" => R08B_SEED_BASE,
        "seed_rule" => "seed = seed_base + 1000 * rep + chain, common to every model and configuration",
        "host" => gethostname(),
        "git_commit" => git_commit,
        "blas_threads" => LinearAlgebra.BLAS.get_num_threads(),
        "interactive_threads" => Threads.nthreads(:interactive),
        "thread_pinning" => "pinthreads(:cores)",
        "metric" => metric_type,
        "initialisation" => string(L08_SAMPLER.initialisation),
        "incident_snapshot_hash" => incident_snapshot_hash,
        "configurations" => Dict(name => Dict(string(k) => v for (k, v) in pairs(R08B_CONFIGS[name]))
                                 for name in R08B_CONFIG_NAMES),
        "primary_metric" => "min over parameters of min(bulk ESS, tail ESS) of the pooled 4-chain fit, divided by the fit's wall-seconds (slowest chain)",
        "secondary_metric" => "the same min ESS divided by core-seconds (sum of the 4 chains' elapsed)",
        "hard_gates" => "every replicate: max R-hat <= 1.05; min bulk and tail ESS >= 400; zero divergences; min BFMI >= 0.30; depth-capped rate < 0.05",
        "reported_gate" => "committed production gate: identical but ESS >= 200",
    ))
println("  manifest  : ", manifest_path)

# %%
# ==============================================================================
# 5. Helpers: per-chain internals, concurrency, fit rows
# ==============================================================================
"Retained-draw sampler internals of one (possibly pooled) chain."
function r08b_internals(chain)
    internal(name) = R08B_INF._inf_internal(chain, name)
    td, ns, ss, ar = internal(:tree_depth), internal(:n_steps), internal(:step_size), internal(:acceptance_rate)
    div = internal(:numerical_error)
    return (
        mean_tree_depth = td === nothing ? NaN : mean(td),
        mean_leapfrog = ns === nothing ? NaN : mean(ns),
        total_leapfrog = ns === nothing ? 0 : Int(sum(ns)),
        step_size_min = ss === nothing ? NaN : minimum(ss[end, :]),
        step_size_max = ss === nothing ? NaN : maximum(ss[end, :]),
        mean_acceptance = ar === nothing ? NaN : mean(ar),
        n_divergent = div === nothing ? -1 : count(>(0), div),
    )
end

"Mean number of chains running (self included) over each record's lifetime."
function r08b_mean_concurrency(starts::Vector{Float64}, stops::Vector{Float64})
    return map(eachindex(starts)) do i
        overlap = sum(max(0.0, min(stops[i], stops[j]) - max(starts[i], starts[j]))
                      for j in eachindex(starts))
        overlap / (stops[i] - starts[i])
    end
end

function r08b_fit_row(model_name, config_name, rep, chains, elapsed, concurrency)
    cfg = R08B_CONFIGS[config_name]
    pooled = length(chains) == 1 ? only(chains) : cat(chains...; dims = 3)
    audit = audit_fold(R08B_FOLD, pooled; max_depth = cfg.max_depth)
    internals = r08b_internals(pooled)
    wall_s = maximum(elapsed)
    core_s = sum(elapsed)
    min_ess = min(audit.min_ess_bulk, audit.min_ess_tail)
    gate_rhat = audit.max_rhat <= R08B_GATES.max_rhat
    gate_ess400 = audit.min_ess_bulk >= R08B_GATES.min_ess_strict && audit.min_ess_tail >= R08B_GATES.min_ess_strict
    gate_ess200 = audit.min_ess_bulk >= R08B_GATES.min_ess_production && audit.min_ess_tail >= R08B_GATES.min_ess_production
    gate_div0 = audit.n_divergent == 0
    gate_bfmi = audit.min_bfmi >= R08B_GATES.min_bfmi
    gate_depth = audit.treedepth_rate < R08B_GATES.max_treedepth_rate
    return (
        model = model_name, config = config_name, rep = rep,
        warmup = cfg.warmup, retained = cfg.retained, delta = cfg.delta, max_depth = cfg.max_depth,
        n_chains = audit.n_chains, n_draws = audit.n_draws, n_params = audit.n_params,
        wall_s, core_s, mean_concurrency = mean(concurrency),
        max_rhat = audit.max_rhat, worst_rhat_param = audit.worst_rhat_param,
        min_ess_bulk = audit.min_ess_bulk, worst_bulk_param = audit.worst_ess_bulk_param,
        min_ess_tail = audit.min_ess_tail, worst_tail_param = audit.worst_ess_tail_param,
        min_ess, ess_per_wall_s = min_ess / wall_s, ess_per_core_s = min_ess / core_s,
        ess_per_draw = min_ess / (audit.n_draws * audit.n_chains),
        n_divergent = audit.n_divergent, n_transitions = audit.n_transitions,
        divergence_rate = audit.divergence_rate,
        max_tree_depth = audit.max_tree_depth, n_depth_capped = audit.n_depth_capped,
        treedepth_rate = audit.treedepth_rate, min_bfmi = audit.min_bfmi,
        internals.mean_tree_depth, internals.mean_leapfrog, internals.total_leapfrog,
        internals.step_size_min, internals.step_size_max, internals.mean_acceptance,
        gate_rhat, gate_ess400, gate_ess200, gate_div0, gate_bfmi, gate_depth,
        pass_strict = gate_rhat && gate_ess400 && gate_div0 && gate_bfmi && gate_depth,
        pass_production = gate_rhat && gate_ess200 && gate_div0 && gate_bfmi && gate_depth,
    )
end

# %%
# ==============================================================================
# 6. Sampling: JIT warm-up, then one flat interleaved 16-slot queue
# ==============================================================================
if !R08B_RUN
    println("\nPREFLIGHT ONLY passed.  Set L08_RUN_BENCH=true only on an idle mcmc-beast.")
else
    # Compile Turing/ReverseDiff method instances once, outside the timed queue, and
    # verify the structural parameter contract on a real chain. Per-chain tape
    # compilation (`AutoReverseDiff(compile = true)`) is still paid inside every timed
    # chain, exactly as in production.
    warm = QueuedNUTSConfig(n_samples = 20, n_warmup = 20, n_chains = L08_CHAINS, accept_rate = 0.95)
    for (name, _) in models
        p = prepared[name]
        Random.seed!(R08B_SEED_BASE)
        t = @elapsed chain = sample_fold(p.model, warm, p.fs, R08B_FOLD; chain_id = 1)
        n_params = length(MCMCChains.names(chain, :parameters))
        n_params == p.expected_params || error(
            "$name chain has $n_params parameters; structural contract expects $(p.expected_params)")
        @printf("  warm-up %-34s %5.1f s · %d parameters (contract %d)\n", name, t, n_params, p.expected_params)
    end

    cells = [(model = name, config = config) for config in R08B_CONFIG_NAMES for (name, _) in models]
    order_rng = MersenneTwister(R08B_SEED_BASE)
    tasks = NamedTuple[]
    for rep in 1:R08B_REPS, chain in 1:L08_CHAINS
        for cell in shuffle(order_rng, cells)
            push!(tasks, (; cell..., rep, chain, seed = R08B_SEED_BASE + 1_000 * rep + chain))
        end
    end

    println("\n", "-"^108)
    println(" QUEUE · ", length(tasks), " chains · ", R08B_SLOTS, " slots · started ", Dates.now())
    println("-"^108)
    queue = Channel{Int}(length(tasks))
    foreach(i -> put!(queue, i), eachindex(tasks))
    close(queue)
    chains = Vector{Any}(nothing, length(tasks))
    records = Vector{Any}(nothing, length(tasks))
    origin = time()
    print_lock = ReentrantLock()
    done = Threads.Atomic{Int}(0)
    @sync for _ in 1:R08B_SLOTS
        Threads.@spawn for i in queue
            task = tasks[i]
            p = prepared[task.model]
            Random.seed!(task.seed)
            status = "ok"
            t0 = time()
            try
                chains[i] = sample_fold(p.model, R08B_SAMPLERS[task.config], p.fs, R08B_FOLD;
                                        chain_id = task.chain)
            catch e
                status = first(sprint(showerror, e), 300)
                @error "$(task.model) $(task.config) rep $(task.rep) chain $(task.chain) failed" exception = (e, catch_backtrace())
            end
            t1 = time()
            records[i] = (; task..., thread = Threads.threadid(),
                          start_s = t0 - origin, stop_s = t1 - origin, elapsed_s = t1 - t0, status)
            n = Threads.atomic_add!(done, 1) + 1
            lock(print_lock) do
                @printf("  [%s] %3d/%d  %-34s %-3s rep %d chain %d  %7.1f s  thread %2d  %s\n",
                    Dates.format(Dates.now(), "HH:MM:SS"), n, length(tasks), task.model, task.config,
                    task.rep, task.chain, t1 - t0, Threads.threadid(), status)
                flush(stdout)
            end
        end
    end
    queue_wall = time() - origin
    @printf("  queue wall %.1f s (%.1f min)\n", queue_wall, queue_wall / 60)

    # Raw chains first: every derived number below can be recomputed from this file.
    raw_path = joinpath(R08B_OUT, "chains.jls")
    serialize(raw_path, Dict((t.model, t.config, t.rep, t.chain) => chains[i] for (i, t) in enumerate(tasks)))

    chain_df = DataFrame(identity.(records))
    chain_df.mean_concurrency = r08b_mean_concurrency(chain_df.start_s, chain_df.stop_s)
    per_chain = [chains[i] === nothing ? nothing : r08b_internals(chains[i]) for i in eachindex(tasks)]
    for col in (:mean_tree_depth, :mean_leapfrog, :step_size_min, :mean_acceptance, :n_divergent)
        chain_df[!, col] = [x === nothing ? missing : getfield(x, col) for x in per_chain]
    end
    rename!(chain_df, :step_size_min => :step_size)
    CSV.write(joinpath(R08B_OUT, "chains.csv"), chain_df)

    fit_rows = NamedTuple[]
    for cell in cells, rep in 1:R08B_REPS
        idx = [i for (i, t) in enumerate(tasks)
               if t.model == cell.model && t.config == cell.config && t.rep == rep]
        sort!(idx; by = i -> tasks[i].chain)
        all(i -> chains[i] !== nothing, idx) || (@error "$(cell.model) $(cell.config) rep $rep lost a chain"; continue)
        push!(fit_rows, r08b_fit_row(cell.model, cell.config, rep, chains[idx],
                                     chain_df.elapsed_s[idx], chain_df.mean_concurrency[idx]))
    end
    fits = DataFrame(identity.(fit_rows))
    CSV.write(joinpath(R08B_OUT, "fits.csv"), fits)

    cell_summary = combine(groupby(fits, [:model, :config]),
        nrow => :reps,
        :pass_strict => sum => :pass_strict,
        :pass_production => sum => :pass_production,
        :wall_s => median => :wall_s_median,
        :wall_s => maximum => :wall_s_max,
        :core_s => median => :core_s_median,
        :max_rhat => maximum => :max_rhat_worst,
        :min_ess_bulk => median => :min_ess_bulk_median,
        :min_ess_bulk => minimum => :min_ess_bulk_worst,
        :min_ess_tail => median => :min_ess_tail_median,
        :min_ess_tail => minimum => :min_ess_tail_worst,
        :ess_per_wall_s => median => :ess_per_wall_s_median,
        :ess_per_wall_s => minimum => :ess_per_wall_s_min,
        :ess_per_wall_s => maximum => :ess_per_wall_s_max,
        :ess_per_core_s => median => :ess_per_core_s_median,
        :n_divergent => sum => :divergences,
        :n_transitions => sum => :transitions,
        :n_depth_capped => sum => :depth_capped,
        :max_tree_depth => maximum => :max_tree_depth,
        :mean_tree_depth => mean => :mean_tree_depth,
        :mean_leapfrog => mean => :mean_leapfrog,
        :min_bfmi => minimum => :min_bfmi,
        :mean_concurrency => mean => :mean_concurrency)
    control = Dict(row.model => row for row in eachrow(cell_summary) if row.config == "A")
    cell_summary.core_s_vs_A = [haskey(control, r.model) ? r.core_s_median / control[r.model].core_s_median : NaN
                           for r in eachrow(cell_summary)]
    cell_summary.ess_per_wall_vs_A = [haskey(control, r.model) ? r.ess_per_wall_s_median / control[r.model].ess_per_wall_s_median : NaN
                                 for r in eachrow(cell_summary)]
    CSV.write(joinpath(R08B_OUT, "summary.csv"), cell_summary)

    println("\n", "="^108)
    println(" PER-FIT DIAGNOSTICS")
    println("="^108)
    @printf(" %-26s %-3s %3s %8s %8s %7s %7s %7s %8s %5s %5s %6s %5s %s\n",
        "model", "cfg", "rep", "wall s", "core s", "R-hat", "bulk", "tail", "ESS/s", "div", "cap", "BFMI", "load", "gates")
    for r in eachrow(fits)
        @printf(" %-26s %-3s %3d %8.1f %8.1f %7.4f %7.0f %7.0f %8.3f %5d %5d %6.3f %5.1f %s\n",
            r.model, r.config, r.rep, r.wall_s, r.core_s, r.max_rhat, r.min_ess_bulk, r.min_ess_tail,
            r.ess_per_wall_s, r.n_divergent, r.n_depth_capped, r.min_bfmi, r.mean_concurrency,
            r.pass_strict ? "STRICT" : (r.pass_production ? "prod-only" : "FAIL"))
    end
    println("\n", "="^108)
    println(" CELL SUMMARY (medians over replicates; ratios against A of the same model)")
    println("="^108)
    @printf(" %-26s %-3s %6s %6s %8s %7s %7s %7s %8s %7s %7s %5s\n",
        "model", "cfg", "strict", "prod", "wall s", "core×A", "R-hat", "minESS", "ESS/s", "ESS/s×A", "depth", "div")
    for r in eachrow(cell_summary)
        @printf(" %-26s %-3s %4d/%d %4d/%d %8.1f %7.3f %7.4f %7.0f %8.3f %7.3f %7.2f %5d\n",
            r.model, r.config, r.pass_strict, r.reps, r.pass_production, r.reps, r.wall_s_median,
            r.core_s_vs_A, r.max_rhat_worst, min(r.min_ess_bulk_worst, r.min_ess_tail_worst),
            r.ess_per_wall_s_median, r.ess_per_wall_vs_A, r.mean_tree_depth, r.divergences)
    end
end

# %%
# ==============================================================================
# 7. Context: the failed production grid's 40 m00 fold checkpoints (config A)
# ==============================================================================
# Read-only. Fold 1 is not the hardest fold; this measures how far below fold 1 the
# worst fold's ESS sits under A, which bounds what a fold-1 saving can promise.
if isdir(R08B_GRID_CHECKPOINTS)
    grid_chains = R08B_INF.load_checkpoints(R08B_GRID_CHECKPOINTS, L08_EXPECTED_FOLDS)
    grid_rows = NamedTuple[]
    for (fold, chain) in enumerate(grid_chains)
        chain isa MCMCChains.Chains || continue
        a = audit_fold(fold, chain; max_depth = 10)
        x = r08b_internals(chain)
        push!(grid_rows, (fold = fold, n_draws = a.n_draws, n_chains = a.n_chains,
            max_rhat = a.max_rhat, worst_rhat_param = a.worst_rhat_param,
            min_ess_bulk = a.min_ess_bulk, worst_bulk_param = a.worst_ess_bulk_param,
            min_ess_tail = a.min_ess_tail, worst_tail_param = a.worst_ess_tail_param,
            n_divergent = a.n_divergent, max_tree_depth = a.max_tree_depth,
            n_depth_capped = a.n_depth_capped, min_bfmi = a.min_bfmi,
            x.mean_tree_depth, x.mean_leapfrog, x.step_size_min, x.step_size_max))
    end
    if !isempty(grid_rows)
        grid_df = DataFrame(identity.(grid_rows))
        CSV.write(joinpath(R08B_OUT, "production_grid_m00_A_fold_audit.csv"), grid_df)
        grid_df.min_ess = min.(grid_df.min_ess_bulk, grid_df.min_ess_tail)
        worst = grid_df[argmin(grid_df.min_ess), :]
        f1 = grid_df[grid_df.fold .== 1, :]
        println("\n", "="^108)
        println(" PRODUCTION GRID m00 · CONFIG A · ", nrow(grid_df), " FOLD CHECKPOINTS")
        println("="^108)
        @printf("  fold 1 min ESS %.0f · worst fold %d min ESS %.0f (%s) · ratio %.3f\n",
            isempty(f1) ? NaN : f1.min_ess[1], worst.fold, worst.min_ess,
            worst.min_ess_bulk <= worst.min_ess_tail ? worst.worst_bulk_param : worst.worst_tail_param,
            isempty(f1) ? NaN : worst.min_ess / f1.min_ess[1])
        @printf("  max R-hat %.4f (fold %d) · divergences %d (folds %s) · max depth %d · capped %d · mean leapfrog %.1f–%.1f\n",
            maximum(grid_df.max_rhat), grid_df.fold[argmax(grid_df.max_rhat)], sum(grid_df.n_divergent),
            join(grid_df.fold[grid_df.n_divergent .> 0], ","), maximum(grid_df.max_tree_depth),
            sum(grid_df.n_depth_capped), minimum(grid_df.mean_leapfrog), maximum(grid_df.mean_leapfrog))
    end
else
    println("\n  no production-grid checkpoints at $R08B_GRID_CHECKPOINTS; §7 skipped")
end

# %%
# ==============================================================================
# 8. Final report
# ==============================================================================
println("\nOutputs in ", R08B_OUT)
println("Finished: ", Dates.now())

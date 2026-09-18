# ==============================================================================
# r06 — Smoke gate: contextual home advantage ladder, folds 1–2
# ==============================================================================
#
# WHAT THIS IS AND IS NOT
#
# A mechanical gate, not a result. Can each rung of the Phase 2 ladder be compiled,
# sampled, audited, extracted and persisted end to end? Coefficients are printed so a
# broken design (a term that is all zeros, a sign flip) is visible early, but nothing
# here is a proper score and two folds say nothing about H1–H5.
#
# GATES (WORK_PACKAGE_PHASE_2_TURF_TIMING.md §5) — every one, every rung, before r07:
#
#   G1  ReverseDiff tape compiles; compiled == fresh RD ≤ 1e-8; RD == ForwardDiff ≤ 1e-6;
#       compiled tape correct at three perturbed points; folds 1 AND 2. The stadium RE is
#       non-centred by construction (`HierarchicalTeamHomeAdvantage`). Allocation and
#       gradient time are reported against the flat Exp 06 twin, not gated — Phase 1
#       showed a literal zero-allocation tape is unreachable on this ReverseDiff stack.
#   G2  0 divergences · max R̂ ≤ 1.05 over every site and every ha.* / contextual site ·
#       bulk and tail ESS ≥ 400.
#   G3  CountLatents finite and positive; save_fit → load_fit round-trip exact through
#       PostgresStorage("smoke_contextual_ha").
#
# Also reported: the contextual design (how many training fixtures switch each term on),
# the midweek definition's (weekday, UTC hour) cells, held-out fixtures with an unmapped
# home club (T003).
#
# USAGE (mcmc-beast, from /root/BF_hier_ha)
#
#   julia --project -t 16 current_development/hierarchical_home_advantage/r06_contextual_smoke.jl
#   R06_MODELS=m05_joint_td_turf_asym  julia ... r06_contextual_smoke.jl
# ==============================================================================

# %%
# ===================================================================
# 1. Packages and implementation
# ===================================================================
using ThreadPinning
using LinearAlgebra
pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

using BayesianFootball
using CSV
using DataFrames
using Dates
using Printf

include(joinpath(@__DIR__, "l04_contextual_loader.jl"))

# %%
# ===================================================================
# 2. Configuration
# ===================================================================
const R06_CONFIG = let env(k, d) = parse(Int, get(ENV, k, string(d)))
    base = CtxConfig()
    CtxConfig(smoke_samples = env("R06_SAMPLES", base.smoke_samples),
              smoke_warmup = env("R06_WARMUP", base.smoke_warmup),
              smoke_chains = env("R06_CHAINS", base.smoke_chains))
end
const R06_SELECTED = let raw = strip(get(ENV, "R06_MODELS", ""))
    isempty(raw) ? copy(CTX_MODEL_NAMES) : String.(strip.(split(raw, ",")))
end
all(in(CTX_MODEL_NAMES), R06_SELECTED) ||
    error("R06_MODELS names an unknown model: $(setdiff(R06_SELECTED, CTX_MODEL_NAMES))")
const R06_SUFFIX = "_smoke_" * Dates.format(now(), "yyyymmddHHMMSS")
const R06_OUT_DIR = joinpath(R06_CONFIG.save_root, "smoke",
    "$(R06_CONFIG.smoke_chains)x$(R06_CONFIG.smoke_warmup)w$(R06_CONFIG.smoke_samples)s")
const R06_GIT = try readchomp(`git rev-parse --short HEAD`) catch; "unsynced-rsync" end

println("\n" * "="^96)
println("  r06 SMOKE GATE — contextual home advantage, folds 1–$(R06_CONFIG.smoke_folds)")
println("  models     : ", join(R06_SELECTED, ", "))
println("  sampler    : QueuedNUTS  $(R06_CONFIG.smoke_chains) chains × ",
        "$(R06_CONFIG.smoke_warmup) warmup + $(R06_CONFIG.smoke_samples) retained, ",
        "δ = $(R06_CONFIG.accept_rate), max depth $(R06_CONFIG.max_depth)")
println("  experiment : ", R06_CONFIG.smoke_experiment, "   (name suffix ", R06_SUFFIX, ")")
println("  git        : ", R06_GIT, "   threads: ", Threads.nthreads())
println("="^96)

# %%
# ===================================================================
# 3. Runtime, data, splits
# ===================================================================
mkpath(R06_OUT_DIR)
r06_db = gph_database(R06_CONFIG.smoke_experiment)
r06_ds = gph_load_data()
r06_splitter = gph_splitter(R06_CONFIG.target_seasons)
println("\n  matches in store : ", nrow(r06_ds.matches),
        "   latest kickoff: ", maximum(r06_ds.matches.match_date))

# The midweek definition, cell by cell over the whole store: which (weekday, UTC hour)
# kickoffs it switches on. A Saturday or a 14:00 Friday here would mean the UTC reading
# of `match_hour` is wrong.
let b = ctx_bridge(r06_ds, ContextualMatchFeature())
    cells = combine(groupby(DataFrame(
        dow = Dates.dayname.(r06_ds.matches.match_date),
        hour_utc = r06_ds.matches.match_hour,
        midweek = [ctx_midweek(b, Date(r.match_date), _ctx_hour(r)) for r in eachrow(r06_ds.matches)]),
        [:dow, :hour_utc, :midweek]), nrow => :n)
    sort!(cells, [:midweek, order(:n, rev = true)])
    CSV.write(joinpath(R06_OUT_DIR, "r06_midweek_cells.csv"), cells)
    println("\n=== MIDWEEK CELLS (weekday × UTC hour) ===")
    show(stdout, MIME"text/plain"(), cells; allrows = true)
    println()
end

# %%
# ===================================================================
# 4. Models
# ===================================================================
r06_models = filter(m -> first(m) in R06_SELECTED, ctx_models())
r06_flat = ctx_flat_twin()
r06_sampler = ctx_smoke_sampler(R06_CONFIG)
r06_configs = ctx_fit_configs(R06_CONFIG, r06_models, r06_splitter, r06_sampler;
                              name_suffix = R06_SUFFIX)
for (name, model) in r06_models
    println("  ", rpad(name, 26), " HA: ", nameof(typeof(model.home_advantage)),
            " | covariates: ", join(string.(predictor_name.(model.covariates)), ", "))
end

# %%
# ===================================================================
# 5. Features and G1
# ===================================================================
r06_gradients = NamedTuple[]
r06_inputs = Dict{String,Any}()
r06_designs = DataFrame[]
r06_unmapped = DataFrame[]

for (name, model) in r06_models
    println("\n--- features + gradient audit: ", name)
    inputs = gph_fold_inputs(r06_ds, r06_splitter, model; limit = R06_CONFIG.smoke_folds)
    r06_inputs[name] = inputs
    filtration = gph_filtration_report(r06_ds, inputs)
    all(filtration.ordered) || error("$name: a fold's last training kickoff is not before its first OOS kickoff")

    design = ctx_design_summary(inputs)
    design.model = fill(name, nrow(design))
    push!(r06_designs, design)
    show(stdout, MIME"text/plain"(), design; allcols = true)
    println()

    unmapped = hha_unmapped_home_report(inputs)
    unmapped.model = fill(name, nrow(unmapped))
    push!(r06_unmapped, unmapped)

    for (fold, fs) in enumerate(inputs.feature_sets)
        audit = gph_gradient_audit(model, fs; replays = R06_CONFIG.gradient_replays)
        twin = gph_gradient_audit(r06_flat, fs; replays = R06_CONFIG.gradient_replays)
        push!(r06_gradients, (; model = name, fold,
                                n_teams = Int(first(fs).data[:n_teams]),
                                audit...,
                                flat_n_parameters = twin.n_parameters,
                                flat_tape_instructions = twin.tape_instructions,
                                flat_gradient_ms = twin.gradient_ms,
                                flat_allocated_bytes = twin.allocated_bytes))
        @printf("  fold %d  θ=%4d (+%d)  tape=%5d (+%d)  grad=%.3f ms (flat %.3f)  alloc=%7d B (flat %d)  RD/FD=%.1e  perturbed=%.1e\n",
                fold, audit.n_parameters, audit.n_parameters - twin.n_parameters,
                audit.tape_instructions, audit.tape_instructions - twin.tape_instructions,
                audit.gradient_ms, twin.gradient_ms, audit.allocated_bytes, twin.allocated_bytes,
                audit.compiled_forward_error, audit.worst_perturbed_error)
    end
end
println("\nR06_G1 PASS  (every tape compiled and matched ForwardDiff; gph_gradient_audit throws otherwise)")

# %%
# ===================================================================
# 6. Sampling
# ===================================================================
r06_fits = Dict{String,Any}()
for (name, _) in r06_models
    println("\n--- sampling: ", name)
    r06_fits[name] = ctx_sample(r06_configs[name], r06_inputs[name], R06_CONFIG)
end

# %%
# ===================================================================
# 7. G2 / G3 and coefficients
# ===================================================================
r06_rows = NamedTuple[]
r06_sites = DataFrame[]
r06_coefs = DataFrame[]

for (name, _) in r06_models
    fit = r06_fits[name]
    d = fit.diagnostics
    failures = String[]

    d.n_divergent == 0 || push!(failures, "G2 divergences=$(d.n_divergent)")
    d.max_rhat <= R06_CONFIG.max_rhat || push!(failures,
        @sprintf("G2 max R̂ %.4f at %s (fold %d)", d.max_rhat,
                 d.folds[d.worst_rhat_fold].worst_rhat_param, d.worst_rhat_fold))
    min(d.min_ess_bulk, d.min_ess_tail) >= R06_CONFIG.min_ess || push!(failures,
        @sprintf("G2 min ESS bulk %.0f / tail %.0f", d.min_ess_bulk, d.min_ess_tail))

    sites = ctx_site_report(fit)
    sites.model = fill(name, nrow(sites))
    push!(r06_sites, sites)
    worst = sites[argmax(sites.max_rhat), :]
    worst.max_rhat <= R06_CONFIG.max_rhat || push!(failures,
        @sprintf("G2 site %s R̂ %.4f", worst.site, worst.max_rhat))
    min(minimum(sites.min_ess_bulk), minimum(sites.min_ess_tail)) >= R06_CONFIG.min_ess ||
        push!(failures, "G2 contextual/ha site ESS < $(R06_CONFIG.min_ess)")

    for fold in eachindex(fit.folds)
        c = ctx_coefficients(fit, fold)
        c.model = fill(name, nrow(c))
        push!(r06_coefs, c)
    end

    latent = try
        gph_latent_audit(fit)
    catch err
        push!(failures, "G3 latents " * sprint(showerror, err)); nothing
    end
    run_id = try
        gph_save_and_verify(r06_db, fit)
    catch err
        push!(failures, "G3 round-trip " * sprint(showerror, err)); nothing
    end

    push!(r06_rows, (; model = name, folds = length(fit.folds),
                       oos = fit.latents === nothing ? 0 : n_matches(fit.latents),
                       max_rhat = d.max_rhat, max_site_rhat = worst.max_rhat,
                       min_ess_bulk = d.min_ess_bulk, min_ess_tail = d.min_ess_tail,
                       min_site_ess = min(minimum(sites.min_ess_bulk), minimum(sites.min_ess_tail)),
                       n_divergent = d.n_divergent, min_bfmi = d.min_bfmi,
                       treedepth_rate = d.treedepth_rate,
                       wall_min = fit.metadata.elapsed_seconds / 60,
                       gate_pass = isempty(failures), gate_failures = join(failures, "; "),
                       run_id = run_id === nothing ? "" : string(run_id)))
    @printf("  %-26s R̂=%.4f  site R̂=%.4f  ESS bulk=%6.1f tail=%6.1f  div=%d  run=%s  %s\n",
            name, d.max_rhat, worst.max_rhat, d.min_ess_bulk, d.min_ess_tail, d.n_divergent,
            run_id === nothing ? "FAIL" : string(run_id),
            isempty(failures) ? "PASS" : "FAIL: " * join(failures, "; "))
end

r06_coef_frame = vcat(r06_coefs...)
println("\n=== CONTEXTUAL COEFFICIENTS (posterior vs prior) ===")
show(stdout, MIME"text/plain"(),
     select(r06_coef_frame, :model, :fold, :site, :mean, :sd, :q05, :q95, :p_positive,
            :prior_p_positive, :contraction); allrows = true, allcols = true)
println()

# %%
# ===================================================================
# 8. Report
# ===================================================================
r06_summary = DataFrame(r06_rows)
r06_gradient_frame = DataFrame(r06_gradients)
r06_design_frame = vcat(r06_designs...)
r06_site_frame = vcat(r06_sites...)
r06_unmapped_frame = vcat(r06_unmapped...)
CSV.write(joinpath(R06_OUT_DIR, "r06_smoke_gates.csv"), r06_summary)
CSV.write(joinpath(R06_OUT_DIR, "r06_gradient_audit.csv"), r06_gradient_frame)
CSV.write(joinpath(R06_OUT_DIR, "r06_design_summary.csv"), r06_design_frame)
CSV.write(joinpath(R06_OUT_DIR, "r06_sites.csv"), r06_site_frame)
CSV.write(joinpath(R06_OUT_DIR, "r06_coefficients.csv"), r06_coef_frame)
CSV.write(joinpath(R06_OUT_DIR, "r06_unmapped_home.csv"), r06_unmapped_frame)

open(joinpath(R06_OUT_DIR, "r06_smoke_report.md"), "w") do io
    println(io, "# r06 smoke gate — Task 008 Phase 2\n")
    println(io, "Generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"), " at `", R06_GIT,
            "` on ", gethostname(), " with ", Threads.nthreads(), " threads.\n")
    println(io, "Sampler: QueuedNUTS, ", R06_CONFIG.smoke_chains, " chains × ",
            R06_CONFIG.smoke_warmup, " warmup + ", R06_CONFIG.smoke_samples,
            " retained, δ = ", R06_CONFIG.accept_rate, ". Folds 1–", R06_CONFIG.smoke_folds, ".\n")
    println(io, "## G1 — gradient audit vs the flat Exp 06 twin\n")
    print(io, gph_markdown_table(select(r06_gradient_frame,
        :model, :fold, :n_teams, :n_parameters, :flat_n_parameters, :tape_instructions,
        :flat_tape_instructions, :gradient_ms, :flat_gradient_ms, :allocated_bytes,
        :flat_allocated_bytes, :compiled_forward_error, :worst_perturbed_error);
        formats = Dict(:gradient_ms => v -> gph_num(v; digits = 3),
                       :flat_gradient_ms => v -> gph_num(v; digits = 3),
                       :compiled_forward_error => v -> @sprintf("%.1e", v),
                       :worst_perturbed_error => v -> @sprintf("%.1e", v))))
    println(io, "\n## G2/G3 — sampling, latents, persistence\n")
    print(io, gph_markdown_table(select(r06_summary, :model, :folds, :oos, :max_rhat,
        :max_site_rhat, :min_ess_bulk, :min_ess_tail, :min_site_ess, :n_divergent, :min_bfmi,
        :wall_min, :gate_pass, :run_id)))
    println(io, "\n## Contextual design (training fixtures with the term switched on)\n")
    print(io, gph_markdown_table(r06_design_frame))
    println(io, "\n## Coefficients\n")
    print(io, gph_markdown_table(select(r06_coef_frame, :model, :fold, :site, :mean, :sd,
        :q05, :q95, :p_positive, :prior_p_positive, :contraction)))
    failed = filter(:gate_pass => !, r06_summary)
    if nrow(failed) > 0
        println(io, "\n### Failures\n")
        for r in eachrow(failed)
            println(io, "* `", r.model, "` — ", r.gate_failures)
        end
    end
end

println("\nR06_VERDICT ", all(r06_summary.gate_pass) ? "PASS" : "FAIL", "  (",
        count(r06_summary.gate_pass), "/", nrow(r06_summary), " models)")
println("R06_DONE report=", joinpath(R06_OUT_DIR, "r06_smoke_report.md"))

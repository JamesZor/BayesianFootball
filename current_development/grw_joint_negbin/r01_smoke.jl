# ==============================================================================
# r01 — Smoke gate: JointGammaNegBinObservation ladder, folds 1–2
# ==============================================================================
#
# WHAT THIS IS AND IS NOT
#
# A mechanical gate, not a result. It answers one question per ladder model: can this
# composition be compiled, sampled, audited, priced, extracted and persisted end to
# end? Nothing it prints is a proper score.
#
# The genuinely new object is `JointGammaNegBinObservation` — a masked Gamma proxy-xG
# arm beside a negative-binomial goals arm on one shared latent, routed to the
# double-negative-binomial score grid. Two of its failure modes are invisible to a
# traceplot, so two gates exist specifically for them (G0 and G6).
#
# GATES — every one must pass for every model before `r02` may be launched:
#
#   G0  engine log density == the independent `equations.jl` reference log-joint, at a
#       prior draw and three displaced points. Catches the two algebra slips the
#       hand-expanded NegBin density can hide: κ leaking into the Gamma arm, and `r`
#       written against η instead of ζ = η + log κ. Runs on a TimeDecay build, which
#       is the dynamics the reference covers; the observation block is identical.
#   G1  ReverseDiff tape compiles; compiled == fresh RD ≤ 1e-8; RD == ForwardDiff
#       ≤ 1e-6; compiled tape correct at three perturbed points (folds 1 AND 2 —
#       fold 1 has no target steps, fold 2 does, so both GRW branches are taped)
#   G2  zero divergences across all chains
#   G3  max R̂ < 1.05 across all sites, and bulk/tail ESS ≥ 400
#   G4  CountLatents extract with finite, strictly positive λ draws AND finite,
#       strictly positive `r_h`/`r_a` draws, and finite non-zero per-fixture variance
#   G5  save_fit → load_fit round-trip through PostgresStorage("smoke_grw_joint_negbin")
#       reproduces fold count, latents, `observation_params` and every chain value
#   G6  the dispersion reaches the pricing tensor: the model's own 12×12 grid differs
#       from the double-Poisson grid at the SAME λ draws, and differs MORE on the tail
#       markets (O/U 3.5, BTTS) than on 1X2
#
# WHY THE GATE RUNS AT THE PRODUCTION SAMPLER. Task 013 measured the work package's
# sketched smoke budget (2 × (50 + 100)) and found it fails R̂ and ESS as a BUDGET
# ARTEFACT — 50 warmup draws do not adapt a step size, and ESS ≥ 400 cannot exist in
# 200 draws. A gate that fails for arithmetic reasons tests nothing. This runs at
# 4 × (500 + 1000), the budget `r02` will use, so a pass here is a statement about the
# production configuration.
#
# PERSISTENCE CAVEAT. Each invocation writes four runs into `smoke_grw_joint_negbin`,
# named with a per-invocation suffix so `save_fit`'s config-hash deduplication can never
# hand back an older smoke run's UUID for the round-trip comparison.
#
# USAGE (mcmc-beast, from /root/BF_grw_joint_negbin)
#
#   /root/.juliaup/bin/julia --project -t 16 current_development/grw_joint_negbin/r01_smoke.jl
#   R01_SAMPLES=100 R01_WARMUP=50 R01_CHAINS=2 julia ... r01_smoke.jl   # fast plumbing pass
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

include(joinpath(@__DIR__, "l01_loader.jl"))

# %%
# ===================================================================
# 2. Configuration
# ===================================================================
const R01_CONFIG = let env(k, d) = parse(Int, get(ENV, k, string(d)))
    base = GJNConfig()
    GJNConfig(smoke_samples = env("R01_SAMPLES", base.smoke_samples),
              smoke_warmup = env("R01_WARMUP", base.smoke_warmup),
              smoke_chains = env("R01_CHAINS", base.smoke_chains))
end
# ESS ≥ 400 is gated only when the budget can arithmetically reach it.
const R01_GATE_ESS = R01_CONFIG.smoke_chains * R01_CONFIG.smoke_samples >= 2 * R01_CONFIG.min_ess
const R01_SUFFIX = "_smoke_" * Dates.format(now(), "yyyymmddHHMMSS")
const R01_OUT_DIR = joinpath(R01_CONFIG.save_root, "smoke",
    "$(R01_CONFIG.smoke_chains)x$(R01_CONFIG.smoke_warmup)w$(R01_CONFIG.smoke_samples)s")
const R01_GIT = try readchomp(`git rev-parse --short HEAD`) catch; "unknown" end

println("\n" * "="^96)
println("  r01 SMOKE GATE — JointGammaNegBinObservation ladder, folds 1–$(R01_CONFIG.smoke_folds)")
println("  sampler    : QueuedNUTS  $(R01_CONFIG.smoke_chains) chains × ",
        "$(R01_CONFIG.smoke_warmup) warmup + $(R01_CONFIG.smoke_samples) retained, ",
        "δ = $(R01_CONFIG.accept_rate), max depth $(R01_CONFIG.max_depth)")
println("  experiment : ", R01_CONFIG.smoke_experiment, "   (name suffix ", R01_SUFFIX, ")")
println("  git        : ", R01_GIT, "   threads: ", Threads.nthreads())
println("="^96)

# %%
# ===================================================================
# 3. Runtime and output directory
# ===================================================================
mkpath(R01_OUT_DIR)
r01_db = gjn_database(R01_CONFIG.smoke_experiment)

# %%
# ===================================================================
# 4. Data snapshot and temporal splits
# ===================================================================
r01_ds = gjn_load_data()
r01_splitter = gjn_splitter(R01_CONFIG.target_seasons)
println("\n  matches in store : ", nrow(r01_ds.matches),
        "   latest kickoff: ", maximum(r01_ds.matches.match_date))

# %%
# ===================================================================
# 5. Likelihood parity against the equations reference (G0)
# ===================================================================
# Run FIRST and on its own models. If the density is wrong, every number produced
# after this point is a well-converged posterior for the wrong model.
r01_parity = NamedTuple[]
for (label, model) in gjn_parity_models()
    inputs = gjn_fold_inputs(r01_ds, r01_splitter, model; limit = 1)
    res = gjn_parity_check(model, inputs.feature_sets[1])
    pass = res.worst_rel <= 1.0e-9
    push!(r01_parity, (; arm = label, points = res.n_points,
                         worst_abs = res.worst_abs, worst_rel = res.worst_rel, pass))
    @printf("  G0 %-26s points=%d  worst_abs=%.2e  worst_rel=%.2e  %s\n",
            label, res.n_points, res.worst_abs, res.worst_rel, pass ? "PASS" : "FAIL")
end
all(r -> r.pass, r01_parity) ||
    error("G0 FAILED: the engine log density disagrees with the equations reference")

# %%
# ===================================================================
# 6. Engine / model construction
# ===================================================================
r01_models = gjn_models()
r01_sampler = gjn_smoke_sampler(R01_CONFIG)
r01_configs = gjn_fit_configs(R01_CONFIG, r01_models, r01_splitter, r01_sampler;
                              name_suffix = R01_SUFFIX)
for (name, model) in r01_models
    println("  ", rpad(name, 34), " covariates: ",
            join(string.(nameof.(typeof.(model.covariates))), ", "),
            " | obs: ", nameof(typeof(model.observation)),
            " | family: ", latent_family(model))
end

# %%
# ===================================================================
# 7. Feature construction and gradient audit (G1)
# ===================================================================
r01_gradients = NamedTuple[]
r01_inputs = Dict{String,Any}()

for (name, model) in r01_models
    println("\n--- features + gradient audit: ", name)
    inputs = gjn_fold_inputs(r01_ds, r01_splitter, model; limit = R01_CONFIG.smoke_folds)
    r01_inputs[name] = inputs
    filtration = gjn_filtration_report(r01_ds, inputs)
    show(stdout, MIME"text/plain"(), filtration; allcols = true)
    println()
    all(filtration.ordered) ||
        error("$name: a fold's last training kickoff is not before its first OOS kickoff")

    for (fold, fs) in enumerate(inputs.feature_sets)
        audit = gjn_gradient_audit(model, fs; replays = R01_CONFIG.gradient_replays)
        push!(r01_gradients, (; model = name, fold,
                                n_target = Int(first(fs).data[:n_target_steps]),
                                audit...))
        @printf("  fold %d  θ=%4d  tape=%5d  grad=%.3f ms  alloc=%7d B  RD/FD=%.1e  perturbed=%.1e\n",
                fold, audit.n_parameters, audit.tape_instructions, audit.gradient_ms,
                audit.allocated_bytes, audit.compiled_forward_error, audit.worst_perturbed_error)
    end
end

# %%
# ===================================================================
# 8. Training — two folds per model
# ===================================================================
r01_fits = Dict{String,Any}()
for (name, _) in r01_models
    println("\n--- sampling: ", name)
    r01_fits[name] = gjn_sample(r01_configs[name], r01_inputs[name], R01_CONFIG)
end

# %%
# ===================================================================
# 9. Convergence (G2, G3), latents (G4), round-trip (G5), score grid (G6)
# ===================================================================
r01_rows = NamedTuple[]
r01_grids = DataFrame[]

for (name, _) in r01_models
    fit = r01_fits[name]
    d = fit.diagnostics
    failures = String[]

    d.n_divergent == 0 || push!(failures, "G2 divergences=$(d.n_divergent)")
    d.max_rhat < R01_CONFIG.max_rhat || push!(failures,
        @sprintf("G3 max R̂ %.4f (fold %d)", d.max_rhat, d.worst_rhat_fold))
    R01_GATE_ESS && min(d.min_ess_bulk, d.min_ess_tail) < R01_CONFIG.min_ess && push!(failures,
        @sprintf("G3 min ESS bulk %.0f / tail %.0f", d.min_ess_bulk, d.min_ess_tail))

    latent = try
        gjn_latent_audit(fit)
    catch err
        push!(failures, "G4 " * sprint(showerror, err))
        nothing
    end

    grid = try
        g = gjn_grid_gate(fit)
        push!(r01_grids, insertcols(g, 1, :model => name))
        # The dispersion must MOVE the tensor, and must move the tail more than the result.
        maximum(abs.(g.d_over35)) > 1.0e-6 || error(
            "the NegBin grid is indistinguishable from the double-Poisson grid on O/U 3.5; " *
            "r never reached compute_score_grid!")
        mean(abs.(g.d_over35)) > mean(abs.(g.d_home)) || error(
            "dispersion moves 1X2 at least as much as O/U 3.5, which is not how a negative " *
            "binomial redistributes mass — check the r_h/r_a wiring")
        all(m -> 0.90 <= m <= 1.0 + 1e-9, g.mass) || error(
            "score tensor mass outside [0.90, 1.0]: the tail has escaped the 12×12 grid")
        g
    catch err
        push!(failures, "G6 " * sprint(showerror, err))
        nothing
    end

    run_id = try
        gjn_save_and_verify(r01_db, fit)
    catch err
        push!(failures, "G5 " * sprint(showerror, err))
        nothing
    end

    disp = gjn_dispersion_summary(fit)
    row = gjn_convergence_row(name, fit, R01_CONFIG; run_id)
    push!(r01_rows, (; row...,
                       mean_r = disp.mean_r, median_r = disp.median_r,
                       r_q05 = disp.q05, r_q95 = disp.q95,
                       latent_mean_lambda_h = latent === nothing ? NaN : latent.mean_lambda_h,
                       latent_min_sd = latent === nothing ? NaN : latent.min_sd,
                       grid_d_over35 = grid === nothing ? NaN : mean(abs.(grid.d_over35)),
                       grid_d_home = grid === nothing ? NaN : mean(abs.(grid.d_home)),
                       gate_pass = isempty(failures),
                       gate_failures = join(failures, "; ")))
    @printf("  %-34s R̂=%.4f  ESS=%6.1f/%6.1f  div=%d  r̂=%.1f  Δou35=%.4f Δ1x2=%.4f  %s\n",
            name, d.max_rhat, d.min_ess_bulk, d.min_ess_tail, d.n_divergent,
            disp.median_r,
            grid === nothing ? NaN : mean(abs.(grid.d_over35)),
            grid === nothing ? NaN : mean(abs.(grid.d_home)),
            isempty(failures) ? "PASS" : "FAIL: " * join(failures, "; "))
end

# %%
# ===================================================================
# 10. Final report
# ===================================================================
r01_summary = DataFrame(r01_rows)
r01_gradient_frame = DataFrame(r01_gradients)
r01_parity_frame = DataFrame(r01_parity)
r01_grid_frame = isempty(r01_grids) ? DataFrame() : vcat(r01_grids...)

CSV.write(joinpath(R01_OUT_DIR, "r01_smoke_gates.csv"), r01_summary)
CSV.write(joinpath(R01_OUT_DIR, "r01_gradient_audit.csv"), r01_gradient_frame)
CSV.write(joinpath(R01_OUT_DIR, "r01_parity.csv"), r01_parity_frame)
nrow(r01_grid_frame) > 0 && CSV.write(joinpath(R01_OUT_DIR, "r01_grid_gate.csv"), r01_grid_frame)

open(joinpath(R01_OUT_DIR, "r01_smoke_report.md"), "w") do io
    println(io, "# r01 smoke gate — Task 014 (JointGammaNegBinObservation)\n")
    println(io, "Generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"), " at `", R01_GIT,
            "` on ", gethostname(), " with ", Threads.nthreads(), " threads.\n")
    println(io, "Sampler: QueuedNUTS, ", R01_CONFIG.smoke_chains, " chains × ",
            R01_CONFIG.smoke_warmup, " warmup + ", R01_CONFIG.smoke_samples,
            " retained, δ = ", R01_CONFIG.accept_rate, ". Folds 1–", R01_CONFIG.smoke_folds, ".\n")

    println(io, "## Likelihood parity vs `equations.jl` (G0)\n")
    print(io, gjn_markdown_table(r01_parity_frame;
        formats = Dict(:worst_abs => v -> @sprintf("%.2e", v),
                       :worst_rel => v -> @sprintf("%.2e", v))))

    println(io, "\n## Gradient audit (G1)\n")
    print(io, gjn_markdown_table(select(r01_gradient_frame,
        :model, :fold, :n_target, :n_parameters, :tape_instructions, :gradient_ms,
        :allocated_bytes, :compiled_forward_error, :worst_perturbed_error);
        formats = Dict(:gradient_ms => v -> gjn_num(v; digits = 3),
                       :compiled_forward_error => v -> @sprintf("%.1e", v),
                       :worst_perturbed_error => v -> @sprintf("%.1e", v))))

    println(io, "\n## Sampling, latents and persistence (G2–G5)\n")
    print(io, gjn_markdown_table(select(r01_summary,
        :model, :folds, :oos, :draws, :max_rhat, :min_ess_bulk, :min_ess_tail,
        :n_divergent, :median_r, :latent_min_sd, :gate_pass, :run_id)))

    if nrow(r01_grid_frame) > 0
        println(io, "\n## Score grid: NegBin vs double-Poisson at the same λ (G6)\n")
        println(io, "`d_*` is this model's grid minus the double-Poisson grid built from the ",
                "SAME posterior λ draws. A negative binomial moves mass to 0 and to 4+, so the ",
                "tail markets must move more than 1X2.\n")
        print(io, gjn_markdown_table(select(r01_grid_frame,
            :model, :fixture, :lambda_h, :lambda_a, :r, :mass,
            :d_home, :d_over25, :d_over35, :d_btts)))
    end

    failed = filter(:gate_pass => !, r01_summary)
    if nrow(failed) > 0
        println(io, "\n### Failures\n")
        for r in eachrow(failed)
            println(io, "* `", r.model, "` — ", r.gate_failures)
        end
    end
end

r01_verdict = all(r01_summary.gate_pass) && all(r01_parity_frame.pass)
println("\nR01_VERDICT ", r01_verdict ? "PASS" : "FAIL", "  (",
        count(r01_summary.gate_pass), "/", nrow(r01_summary), " models)")
println("R01_DONE report=", joinpath(R01_OUT_DIR, "r01_smoke_report.md"))

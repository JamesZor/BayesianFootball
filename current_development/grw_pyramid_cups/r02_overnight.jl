# ==============================================================================
# r02 — 40-fold production grid for the pyramid GRW arms
# ==============================================================================
#
# Same contract as grw_player_hybrid/r02_production_grid.jl: canonical 40 folds /
# 710 held-out 56/57 fixtures, QueuedNUTS 4 × (500 warmup + 1000), δ = 0.80, the
# six-part audit on every retained draw, persist every 2nd draw to
# mcmc_experiments `scottish_pyramid_grw_cups`, reload-and-compare. A failing arm is
# not persisted; its checkpoints stay under results/<arm>/checkpoints for a resume.
# A persisted recipe is loaded, never resampled.
#
# USAGE (mcmc-beast, from /root/BF_grw_pyramid_cups)
#   PCX_ARMS=g1_grw_all_spfl julia --project -t 16 current_development/grw_pyramid_cups/r02_overnight.jl
# ==============================================================================

using ThreadPinning
using LinearAlgebra
pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

using BayesianFootball, CSV, DataFrames, Dates, Printf
include(joinpath(@__DIR__, "l01_loader.jl"))

const R02_CFG = PCXConfig()
const R02_ARMS = let raw = strip(get(ENV, "PCX_ARMS", ""))
    isempty(raw) ? copy(PCX_ARMS) : String.(strip.(split(raw, ",")))
end
all(in(PCX_ARMS), R02_ARMS) || error("unknown arm in PCX_ARMS")
const R02_GIT = try readchomp(`git rev-parse --short HEAD`) catch; "unknown" end
mkpath(R02_CFG.save_root)

println("="^90)
println("  r02 PRODUCTION — pyramid GRW arms: ", join(R02_ARMS, ", "))
println("  sampler 4 × (", R02_CFG.gph.warmup, " + ", R02_CFG.gph.samples, "), δ = ",
        R02_CFG.gph.accept_rate, " | threads ", Threads.nthreads(), " | git ", R02_GIT)
println("="^90)

db = gph_database(R02_CFG.experiment)
ds = pcx_load_data(; max_age_hours = parse(Int, get(ENV, "PCX_CACHE_HOURS", "12")))
sampler = gph_production_sampler(R02_CFG.gph)

rows = NamedTuple[]
for arm in R02_ARMS
    println("\n", "-"^90, "\nARM ", arm, "  ", now(), "\n", "-"^90)
    local fc = pcx_fit_config(arm, R02_CFG, sampler)
    pcx_register!(db, arm, fc)
    existing = gph_completed_run(db, fc)
    if existing !== nothing
        println("  already persisted as ", existing, " — skipping")
        println("R02_ARM_DONE ", arm, " reused ", existing)
        continue
    end
    inputs = gph_fold_inputs(ds, fc.splitter, fc.model)
    wid = pcx_widening_report(ds, inputs)
    CSV.write(joinpath(R02_CFG.save_root, "r02_widening_$(arm).csv"), wid)
    sum(wid.n_oos) == R02_CFG.expected_oos || error("$arm: $(sum(wid.n_oos)) OOS")
    @printf("  folds %d | train %d–%d (upper %d–%d, cup %d–%d) | OOS %d | teams %d–%d\n",
            nrow(wid), minimum(wid.n_train), maximum(wid.n_train), minimum(wid.n_upper),
            maximum(wid.n_upper), minimum(wid.n_cup), maximum(wid.n_cup), sum(wid.n_oos),
            minimum(wid.n_teams), maximum(wid.n_teams))

    t0 = time()
    ckpt = joinpath(R02_CFG.save_root, arm, "checkpoints")
    fit = gph_sample(fc, inputs, R02_CFG.gph; checkpoint_dir = ckpt)
    d = fit.diagnostics
    @printf("  audit: R̂ %.4f | ESS bulk %.0f tail %.0f | div %d/%d | depth %.4f | BFMI %.3f | %s\n",
            d.max_rhat, d.min_ess_bulk, d.min_ess_tail, d.n_divergent, d.n_transitions,
            d.treedepth_rate, d.min_bfmi, d.passed ? "PASS" : "FAIL: " * join(d.failures, "; "))
    if !d.passed
        push!(rows, (; arm, passed = false, max_rhat = d.max_rhat, min_ess_bulk = d.min_ess_bulk,
                       min_ess_tail = d.min_ess_tail, divergences = d.n_divergent,
                       wall_min = (time() - t0) / 60, run_id = ""))
        println("R02_ARM_FAIL ", arm, " — not persisted; checkpoints in ", ckpt)
        continue
    end
    persisted = gph_thin_for_persistence(fit, inputs, R02_CFG.gph.persist_stride)
    gph_assert_coverage(arm, persisted; folds = R02_CFG.expected_folds, oos = R02_CFG.expected_oos)
    la = gph_latent_audit(persisted)
    run_id = gph_save_and_verify(db, persisted)
    push!(rows, (; arm, passed = true, max_rhat = d.max_rhat, min_ess_bulk = d.min_ess_bulk,
                   min_ess_tail = d.min_ess_tail, divergences = d.n_divergent,
                   wall_min = (time() - t0) / 60, run_id = string(run_id)))
    CSV.write(joinpath(R02_CFG.save_root, "r02_runs_$(arm).csv"), DataFrame(rows[end:end]))
    println("R02_ARM_DONE ", arm, " ", run_id, "  wall ", round((time() - t0) / 60, digits = 1), " min")
    fit = nothing; persisted = nothing; inputs = nothing; GC.gc()
end
isempty(rows) || CSV.write(joinpath(R02_CFG.save_root, "r02_runs_$(join(R02_ARMS, "+")).csv"), DataFrame(rows))
println("R02_DONE ", now())

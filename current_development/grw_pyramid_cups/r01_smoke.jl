# ==============================================================================
# r01 — Smoke gate for the three pyramid GRW arms (folds 1–2)
# ==============================================================================
#
# Gates per arm, all must pass before r02 may launch the arm:
#   G0 filtration : canonical held-out fixtures unchanged; every added row precedes
#                   its fold's first held-out kickoff; 56/57 clock cross-check passes
#                   (asserted inside pcx_align_time!)
#   G1 tape       : compiled ReverseDiff gradient finite and equal to ForwardDiff
#   G2 sampling   : 4 chains × (300 warmup + 300) on 2 folds complete; divergence
#                   rate < 1%; R̂ ≤ 1.10 (smoke budget, not the production gate)
#   G3 latents    : held-out λ finite and positive for every fixture
#
# Persistence/portfolio round-trips are exercised by r02 on the production run
# (gph_save_and_verify reloads and compares), not here.
#
# USAGE (mcmc-beast, from /root/BF_grw_pyramid_cups)
#   julia --project -t 16 current_development/grw_pyramid_cups/r01_smoke.jl
# ==============================================================================

using ThreadPinning
using LinearAlgebra
pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

using BayesianFootball, CSV, DataFrames, Dates, Printf
include(joinpath(@__DIR__, "l01_loader.jl"))

const R01_CFG = PCXConfig()
const R01_ARMS = let raw = strip(get(ENV, "PCX_ARMS", ""))
    isempty(raw) ? copy(PCX_ARMS) : String.(strip.(split(raw, ",")))
end
const R01_OUT = joinpath(R01_CFG.save_root, "smoke")
mkpath(R01_OUT)

println("="^90, "\n  r01 SMOKE — pyramid GRW arms: ", join(R01_ARMS, ", "), "\n", "="^90)
ds = pcx_load_data(; max_age_hours = parse(Int, get(ENV, "PCX_CACHE_HOURS", "12")))

# canonical reference: the 56/57-only splitter's held-out sets
canon = gph_splitter(R01_CFG.target_seasons)
canon_b = Data.create_id_boundaries(ds, canon)
length(canon_b) == R01_CFG.expected_folds || error("canonical splitter gave $(length(canon_b)) folds")
canon_oos = [Set(Int.(Data.get_next_matches(ds, m, canon).match_id)) for (_, m) in canon_b]
sum(length, canon_oos) == R01_CFG.expected_oos || error("canonical OOS = $(sum(length, canon_oos))")

smoke_sampler = QueuedNUTSConfig(n_samples = 300, n_warmup = 300, n_chains = 4,
                                 accept_rate = R01_CFG.gph.accept_rate, max_depth = R01_CFG.gph.max_depth,
                                 show_progress = false)
rows = NamedTuple[]
for arm in R01_ARMS
    println("\n", "-"^90, "\nARM ", arm, "\n", "-"^90)
    local fc = pcx_fit_config(arm, R01_CFG, smoke_sampler)
    t0 = time()
    # --- G0: all 40 folds' boundaries, 2 folds' features ---------------------
    all_b = Data.create_id_boundaries(ds, fc.splitter)
    length(all_b) == R01_CFG.expected_folds || error("$arm: $(length(all_b)) folds")
    for (i, (b, m)) in enumerate(all_b)
        oos = Data.get_next_matches(ds, m, fc.splitter)
        Set(Int.(oos.match_id)) == canon_oos[i] || error("$arm fold $i: held-out set differs from canonical")
        isempty(oos) && continue
        cutoff = minimum(Date.(oos.match_date))
        tr = Set(vcat(Int.(b.history_match_ids), Int.(b.target_match_ids)))
        isempty(intersect(tr, canon_oos[i])) || error("$arm fold $i: train/OOS overlap")
        maximum(Date.(ds.matches.match_date[in.(Int.(ds.matches.match_id), Ref(tr))])) < cutoff ||
            error("$arm fold $i: a training row reaches the held-out bin")
    end
    inputs = gph_fold_inputs(ds, fc.splitter, fc.model; limit = 2)
    wid = pcx_widening_report(ds, inputs)
    CSV.write(joinpath(R01_OUT, "widening_$(arm)_folds1-2.csv"), wid)
    show(wid; allrows = true); println()
    # --- G1: gradient audit on fold 2 ----------------------------------------
    ga = gph_gradient_audit(fc.model, inputs.feature_sets[2]; replays = 50)
    println("  G1 gradient: ", ga)
    # --- G2: sampling ---------------------------------------------------------
    fit = gph_sample(fc, inputs, R01_CFG.gph)
    d = fit.diagnostics
    div_rate = d.n_divergent / max(d.n_transitions, 1)
    g2 = div_rate < 0.01 && d.max_rhat <= 1.10
    @printf("  G2 sampling: R̂ %.4f | ESS bulk %.0f tail %.0f | div %d/%d (%.4f) | depth %.3f | BFMI %.3f -> %s\n",
            d.max_rhat, d.min_ess_bulk, d.min_ess_tail, d.n_divergent, d.n_transitions, div_rate,
            d.treedepth_rate, d.min_bfmi, g2 ? "PASS" : "FAIL")
    # --- G3: latents -----------------------------------------------------------
    la = gph_latent_audit(fit)
    g3 = la.n_matches == sum(nrow, inputs.oos) && isfinite(la.mean_lambda_h) && la.min_sd > 0
    @printf("  G3 latents: %d fixtures | mean λ_h %.3f λ_a %.3f -> %s\n", la.n_matches,
            la.mean_lambda_h, la.mean_lambda_a, g3 ? "PASS" : "FAIL")
    g1 = isfinite(ga.log_density) && ga.compiled_forward_error < 1e-8 && ga.worst_perturbed_error < 1e-6
    push!(rows, (arm = arm, g0 = true, g1 = g1, g2 = g2, g3 = g3, max_rhat = d.max_rhat,
                 min_ess_bulk = d.min_ess_bulk, divergences = d.n_divergent,
                 n_teams_fold2 = wid.n_teams[end], n_cup_fold2 = wid.n_cup[end],
                 wall_min = (time() - t0) / 60))
    println("R01_ARM ", arm, " ", (g1 && g2 && g3) ? "PASS" : "FAIL")
end
summary = DataFrame(rows)
CSV.write(joinpath(R01_OUT, "r01_smoke_summary.csv"), summary)
show(summary; allrows = true); println()
println("R01_DONE ", all(summary.g1 .& summary.g2 .& summary.g3) ? "PASS" : "FAIL")

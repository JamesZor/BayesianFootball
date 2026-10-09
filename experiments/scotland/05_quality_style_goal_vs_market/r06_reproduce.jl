# Phase 6 reproduction, in a FRESH beast REPL (-t 16). Byte-for-byte comparison against the
# persisted grid results; nothing is written to mcmc_experiments.
#   Goal arms: fold 1 of each arm, refitted from the arm's frozen seed (Random.seed!(4001..4004)
#   before the harness's own input construction). Fold 1's four chain tasks are the first ones
#   spawned, so their task RNGs are the grid's if no RNG is consumed in between.
#   Market arm: folds 5 (originally run on 2 threads) and 33 (16 threads), frozen chain seeds.
using BayesianFootball, CSV, DataFrames, Dates, LinearAlgebra, MCMCChains, Random
using Serialization, ThreadPinning, UUIDs
strip(read(`hostname`, String)) == "mcmc-beast" && Threads.nthreads() == 16 ||
    error("reproduction only on a 16-threaded beast REPL")
pinthreads(:cores)
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "market_arm.jl"))
const R6C = Module(:QSReproCandidates)
Base.include(R6C, joinpath(@__DIR__, "candidates.jl"))
const R6_OUT = joinpath(get(ENV, "QSX_OUT", "/root/BF_runs/qs_experiment_out"), "repro")
isdir(R6_OUT) && !isempty(readdir(R6_OUT)) && error("reproduction output not fresh: $R6_OUT")
mkpath(R6_OUT)
const R6_SEEDS = Dict("control_grw" => 4_001, "qs_market_r" => 4_002,
                      "qs_weak_r" => 4_003, "control_td" => 4_004)

r6_ds = QSMarketArm.checked_datastore()
r6_db = Training.PostgresStorage(R6C.EXPERIMENT)
r6_rows = NamedTuple[]

"Exact equality of values, and the largest absolute difference if not."
function r6_compare(a::AbstractArray, b::AbstractArray)
    size(a) == size(b) || return (; identical = false, max_abs = Inf, detail = "shape $(size(a)) vs $(size(b))")
    same = isequal(a, b)
    gap = maximum(abs.(Float64.(a) .- Float64.(b)); init = 0.0)
    return (; identical = same, max_abs = gap, detail = "")
end

# %% Goal arms, fold 1
for c in R6C.CANDIDATES
    started = time()
    saved = Training.load_fit(r6_db, Training.Inference._run_uuid(r6_db, c.name))
    cfg = Harness.fit_config(c; stage = :grid, experiment = R6C.EXPERIMENT)
    Random.seed!(R6_SEEDS[c.name])
    inputs = Harness._fold_inputs(c, r6_ds; stage = :grid)
    fit = Training.fit_model(cfg; feature_sets = inputs.feature_sets[1:1],
                             oos_fixtures = inputs.oos[1:1], quiet = true)
    fresh_chain = Array(fit.folds[1].chain, append_chains = false)
    saved_fold = only(filter(f -> f.fold == 1, collect(saved.folds)))
    saved_chain = Array(saved_fold.chain, append_chains = false)
    names(fit.folds[1].chain) == names(saved_fold.chain) || error("$(c.name): chain parameter names differ")
    chains = r6_compare(cat(fresh_chain...; dims = 3), cat(saved_chain...; dims = 3))
    ids = fit.latents.match_ids
    rows = [findfirst(==(id), saved.latents.match_ids) for id in ids]
    any(isnothing, rows) && error("$(c.name): fold-1 fixtures missing from saved latents")
    lat = r6_compare(hcat(fit.latents.λ_home, fit.latents.λ_away),
                     hcat(saved.latents.λ_home[rows, :], saved.latents.λ_away[rows, :]))
    push!(r6_rows, (; arm = c.name, fold = 1, chains_identical = chains.identical,
        chains_max_abs = chains.max_abs, latents_identical = lat.identical,
        latents_max_abs = lat.max_abs, n_values = length(cat(fresh_chain...; dims = 3)),
        seconds = round(time() - started; digits = 1)))
    println("REPRO goal arm=$(c.name) fold=1 chains_identical=$(chains.identical) max_abs=$(chains.max_abs) ",
            "latents_identical=$(lat.identical) max_abs=$(lat.max_abs)")
    flush(stdout)
end

# %% Market arm, folds 5 and 33
r6_market = QSMarketArm.prepare_panel(r6_ds, joinpath(dirname(R6_OUT), "market_panel"))
r6_folds = QSMarketArm.goal_folds(r6_ds)
r6_fixtures = Dict(Int(r.match_id) => r for r in eachrow(r6_folds.scoped.matches))
const R6_GRID = joinpath(dirname(R6_OUT), "market_grid")
for number in (5, 33)
    started = time()
    fold = r6_folds.folds[number]
    r = QSMarketArm.run_fold(r6_market.panel, fold, r6_fixtures, joinpath(R6_OUT, "market"))
    work = joinpath(R6_GRID, "fold_$(lpad(number, 2, '0'))")
    saved_fit = Serialization.deserialize(joinpath(work, "fit.jls"))
    saved_lat = Serialization.deserialize(joinpath(work, "latents.jls"))
    draws = r6_compare(r.fit.udraws, saved_fit.udraws)
    lat = r6_compare(hcat(r.latents.λ_home, r.latents.λ_away),
                     hcat(saved_lat.λ_home, saved_lat.λ_away))
    probs = read(joinpath(R6_OUT, "market", "fold_$(lpad(number, 2, '0'))", "probabilities.csv")) ==
            read(joinpath(work, "probabilities.csv"))
    push!(r6_rows, (; arm = "market_c0", fold = number, chains_identical = draws.identical,
        chains_max_abs = draws.max_abs, latents_identical = lat.identical && probs,
        latents_max_abs = lat.max_abs, n_values = length(r.fit.udraws),
        seconds = round(time() - started; digits = 1)))
    println("REPRO market fold=$number draws_identical=$(draws.identical) max_abs=$(draws.max_abs) ",
            "latents_identical=$(lat.identical) probabilities_csv_bytes_identical=$probs")
    flush(stdout)
end

r6 = DataFrame(r6_rows)
CSV.write(joinpath(R6_OUT, "reproduction.csv"), r6)
println("REPRO all_identical=$(all(r6.chains_identical .& r6.latents_identical))")
println("REPRODUCTION_DONE")

# Phase 4 market arm on all 40 folds, in its own 2-thread beast REPL pinned to cores 14–15
# (beside a 14-thread goal grid on cores 0–13: combined load 16 threads).
# Resume: a fold with probabilities.csv is reused; a partial fold directory is an error.
# A fold failing the C0 gate is rerun ONCE with frozen alternative seeds (+50) in fold_XX_rerun.
using BayesianFootball, CSV, DataFrames, Dates, LinearAlgebra, Serialization, ThreadPinning
strip(read(`hostname`, String)) == "mcmc-beast" || error("market grid only on mcmc-beast")
m4_cores = let r = split(get(ENV, "QSX_CORES", "14:15"), ":")
    parse(Int, r[1]):parse(Int, r[2])
end
length(m4_cores) == Threads.nthreads() || error("core range $(m4_cores) != $(Threads.nthreads()) threads")
pinthreads(collect(m4_cores))
BLAS.set_num_threads(1)
isdefined(Main, :QSMarketArm) || include(joinpath(@__DIR__, "market_arm.jl"))
const MAG = QSMarketArm
const M4_OUT = joinpath(get(ENV, "QSX_OUT", "/root/BF_runs/qs_experiment_out"), "market_grid")
mkpath(M4_OUT)

m4_ds = MAG.checked_datastore()
m4_market = MAG.prepare_panel(m4_ds, joinpath(dirname(M4_OUT), "market_panel"))
m4_folds = MAG.goal_folds(m4_ds)
m4_fixtures = Dict(Int(r.match_id) => r for r in eachrow(m4_folds.scoped.matches))

"Run (or reload) one fold; returns (latents, row)."
function m4_fold(fold; rerun = false)
    out = rerun ? joinpath(M4_OUT, "rerun") : M4_OUT
    work = joinpath(out, "fold_$(lpad(fold.number, 2, '0'))")
    if isfile(joinpath(work, "probabilities.csv"))
        diag = CSV.read(joinpath(work, "convergence.csv"), DataFrame)
        return (Serialization.deserialize(joinpath(work, "latents.jls")), diag, missing, true)
    elseif isfile(joinpath(work, "convergence.csv")) && !rerun
        diag = CSV.read(joinpath(work, "convergence.csv"), DataFrame)
        all(diag.gate_pass) && error("fold $(fold.number) converged but has no prices: inspect $work")
        return (nothing, diag, missing, true)
    end
    started = time()
    seeds = rerun ? MAG.SEEDS .+ 100fold.number .+ 50 : nothing
    r = rerun ? MAG.run_fold(m4_market.panel, fold, m4_fixtures, out; seeds) :
                MAG.run_fold(m4_market.panel, fold, m4_fixtures, out)
    return (r.latents, r.diag, time() - started, false)
end

m4_rows = NamedTuple[]
m4_parts = Any[]
for fold in m4_folds.folds
    latents, diag, seconds, reused = m4_fold(fold)
    reran = false
    if latents === nothing
        println("QS_MARKET_RERUN fold=$(fold.number) first_rhat=$(maximum(diag.rhat))")
        latents, diag, seconds, reused = m4_fold(fold; rerun = true)
        reran = true
        latents === nothing && println("QS_MARKET_FOLD_FAILED fold=$(fold.number) rhat=$(maximum(diag.rhat))")
    end
    latents === nothing || push!(m4_parts, latents)
    push!(m4_rows, (; fold = fold.number, reran, reused, seconds,
        rhat_max = maximum(diag.rhat), ess_bulk_min = minimum(diag.ess_bulk),
        ess_tail_min = minimum(diag.ess_tail), gate = all(diag.gate_pass),
        heldout = length(fold.ids), at = string(now())))
    CSV.write(joinpath(M4_OUT, "grid_summary.csv"), DataFrame(m4_rows))
    flush(stdout)
end
m4_failed = [r.fold for r in m4_rows if !r.gate]
if isempty(m4_failed)
    m4_latents = MAG.combine_latents(m4_parts)
    length(m4_latents.match_ids) == 710 || error("market arm covers $(length(m4_latents.match_ids)) fixtures")
    Serialization.serialize(joinpath(M4_OUT, "market_latents.jls"), m4_latents)
end
println("MARKET_GRID folds=$(length(m4_rows)) failed=$(m4_failed) reruns=$(count(r -> r.reran, m4_rows))")
println("MARKET_GRID_DONE")

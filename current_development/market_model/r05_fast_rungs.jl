# Revision 8 fast-rung production: C0/H1/H2, full books only, both protocols.
# Question: split HA / quality-linked HA versus C0; C0 versus published B2 R6.
# Exact Gaussian Kalman likelihood; unchanged priors, seeds, budgets and targets.
# 10a is retrospective. 10b fits theta/static HA on training only, forecasts
# before ALL updates each week, and scores the prescribed test season.
# Fresh immutable output directory; no resume. No recovery dependency for these
# exact Gaussian rungs. C1 promotion still requires successful synthetic recovery.
# H1 individual HA coefficients are prior-dependent (mu confounds their split).
# No DB writes, new packages, cache refresh, thin pooling, grid changes or Phase D.
# Run by include in a fresh OWNED persistent beast REPL, not one-shot Julia.

# %%
# ===================================================================
# 1. Packages and implementations
# ===================================================================
using BayesianFootball, LinearAlgebra, CSV, DataFrames, SHA, TOML, ThreadPinning
if !isdefined(@__MODULE__, :PooledMarket)
    include(joinpath(@__DIR__,"l05_pooled.jl"))
end
if !isdefined(@__MODULE__, :FullBookWorkflow)
    include(joinpath(@__DIR__,"l05_fullbook_workflow.jl"))
end
if !isdefined(@__MODULE__, :FastFullBookReports)
    include(joinpath(@__DIR__,"l05_fast_reports.jl"))
end
const C08F = FullBookWorkflow
const C08P = PooledMarket
const C08R = FastFullBookReports

# %%
# ===================================================================
# 2. Scientific configuration and immutable output
# ===================================================================
const C08_CONFIG = C08P.MM.scottish_lower_2425_2526()
const C08_SEEDS = TOML.parsefile(joinpath(@__DIR__,"fullbook_seeds.toml"))
const C08_OUT = get(ENV,"C08_FAST_OUT",joinpath(@__DIR__,"results","C","v8_fast_1"))
const C08_RUNGS = (:C0,:H1,:H2)
const C08_PROTOCOLS = ("10b","10a")
strip(read(`hostname`,String)) == "mcmc-beast" || error("sampling requires mcmc-beast")
Threads.nthreads() == 16 || error("production requires 16 Julia threads")
pinthreads(:cores)
BLAS.set_num_threads(1)
isdir(C08_OUT) && !isempty(readdir(C08_OUT)) && error("output must be fresh: $C08_OUT")
mkpath(C08_OUT)

# %%
# ===================================================================
# 3. Pinned snapshot, isolated targets, honest train/test filtration
# ===================================================================
c08_cache = joinpath(pwd(),".cache","datastore_ScottishLower.jls")
open(io -> bytes2hex(SHA.sha256(io)),c08_cache) ==
    "c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4" || error("wrong pinned cache")
c08_ds = BayesianFootball.Data.load_datastore_cached(C08_CONFIG.segment; max_age_hours=10^6)
c08_panel = C08P.CM.TB.phase_b_panel(c08_ds; config=C08_CONFIG).panel
c08_markets = C08P.fullbook_markets(c08_ds,c08_panel,C08_CONFIG)
c08_train,c08_train_markets = C08P.restrict_fullbook(c08_panel,c08_markets,C08_CONFIG.honest_train)
c08_test = DataFrames.filter(r -> r.season in C08_CONFIG.honest_test,c08_panel.matches)
isempty(intersect(Set(c08_train.obs_match),Set(c08_test.match_id))) || error("train/test overlap")
maximum(c08_train.matches.match_date) < minimum(c08_test.match_date) || error("chronological cut violated")
println("C8 fast: full=$(length(c08_markets)), train=$(length(c08_train_markets)), test=$(nrow(c08_test)); forecasts pre-week")

# %%
# ===================================================================
# 4. Gates 2–4 including all inherited B/B2/TODO023 gates
# ===================================================================
c08_gates = C08P.fullbook_engine_gates()
CSV.write(joinpath(C08_OUT,"engine_gates_c.csv"),c08_gates)
all(c08_gates.pass) || error("engine gates failed; no sampling")

# %%
# ===================================================================
# 5. Prescribed 4 x (2000 warmup + 3000 retained), thin 1; Gate 5
# ===================================================================
c08_fits = Dict{Tuple{Symbol,String},C08P.MID.ArmFit}()
c08_diag = DataFrame[]
for protocol in C08_PROTOCOLS, rung in C08_RUNGS
    panel = protocol == "10a" ? c08_panel : c08_train
    fit = C08F.train_rung(C08P.FullBookRung(rung),panel,nothing,C08_OUT;
        protocol,seeds=C08_SEEDS[String(rung)][protocol])
    c08_fits[(rung,protocol)] = fit
    push!(c08_diag,CSV.read(joinpath(C08_OUT,"convergence_$(rung)_$(protocol).csv"),DataFrame))
    CSV.write(joinpath(C08_OUT,"convergence_c.csv"),vcat(c08_diag...))
end

# %%
# ===================================================================
# 6. Accepted forecasts, coverage, paired scores, smoothed paths/ratios
# ===================================================================
c08_raw = C08F.evaluate_fits(c08_panel,c08_markets,C08_CONFIG,c08_fits,C08_OUT;
    prediction_seed=C08_SEEDS["inference"]["prediction"])
C08R.compare_r6(c08_raw,C08_OUT)
C08R.smoothed_total(c08_fits,c08_panel,C08_OUT; seed=C08_SEEDS["inference"]["smoothing"])

# %%
# ===================================================================
# 7. Static HA posterior mixture; learned H2 kappa; figures
# ===================================================================
CSV.write(joinpath(C08_OUT,"home_advantage_rungs.csv"),C08R.home_advantage(c08_fits,c08_panel,c08_train))
C08R.figures(C08_OUT)
println("R05_FAST_RUNGS_DONE")

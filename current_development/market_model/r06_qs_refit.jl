# Fresh-beast-REPL fixed-seed QS refits, to a SEPARATE immutable output directory.
# Set QS_REFIT_LEAGUES to a comma-separated explicit list. If measured 66-fit
# wall is <=3 hours, leave unset to refit all 11; otherwise use
# eng_premier,sco_league_two. No old posterior is loaded for these comparisons.

# %%
# ===================================================================
# 1. Packages, budget gate and byte comparison
# ===================================================================
using BayesianFootball, DataFrames, CSV, LinearAlgebra, ThreadPinning
if !isdefined(@__MODULE__, :QualityStyleEDA)
    include(joinpath(@__DIR__,"l06_qs_eda.jl"))
end
const Q06C = QualityStyleEDA
strip(read(`hostname`,String)) == "mcmc-beast" || error("refit only on beast")
Threads.nthreads() == 16 || error("requires 16 Julia threads")
pinthreads(:cores)
BLAS.set_num_threads(1)
const Q06C_ROOT = joinpath(@__DIR__,"results","QS")
const Q06C_REPRO = joinpath(Q06C_ROOT,"reproduction","refits")
const Q06C_CONFIGS = CSV.read(joinpath(Q06C_ROOT,"configs.csv"),DataFrame)
const Q06C_SELECTION = strip(get(ENV,"QS_REFIT_LEAGUES",""))
const Q06C_ENTRIES = isempty(Q06C_SELECTION) ? Q06C.LEAGUES :
    filter(e -> e[1] in split(Q06C_SELECTION,','),Q06C.LEAGUES)
length(Q06C_ENTRIES) == (isempty(Q06C_SELECTION) ? length(Q06C.LEAGUES) :
    length(split(Q06C_SELECTION,','))) || error("unknown or duplicate refit league")
all(Q06C.PM.fullbook_engine_gates().pass) || error("engine gates failed before refit")
function q06c_compare(league,filename,rows)
    canonical = joinpath(Q06C_ROOT,"fits",league,"$filename.csv")
    isfile(canonical) || error("missing canonical $canonical")
    tmp = tempname()
    try
        CSV.write(tmp,rows)
        read(tmp) == read(canonical) || error("non-identical fixed-seed refit: $canonical")
    finally
        isfile(tmp) && rm(tmp)
    end
    println("REFIT_BYTE_IDENTICAL ",league," ",filename)
end

# %%
# ===================================================================
# 2. Same snapshot, market panel, frozen priors/seeds and six fits per league
# ===================================================================
function q06c_refit(league,tournament,segment)
    Q06C.snapshot_hash(segment) ==
        Q06C.SNAPSHOT_HASHES[string(nameof(typeof(segment)))] || error("wrong pinned snapshot")
    row = only(eachrow(filter(:league=>==(league),Q06C_CONFIGS)))
    config = Q06C.MM.MarketModelConfig(; name=league,segment,tournaments=[tournament],
        seasons=String.(split(row.seasons,";")),
        honest_train=String.(split(row.honest_train,";")),
        honest_test=[String(row.honest_test)])
    config.price_window == (-20.0,0.0) || error("wrong price window")
    out = joinpath(Q06C_REPRO,league)
    isdir(out) && !isempty(readdir(out)) && error("non-fresh refit output: $out")
    mkpath(out)
    ds = BayesianFootball.Data.load_datastore_cached(segment; max_age_hours=10^6)
    panel = Q06C.TB.phase_b_panel(ds; config,
        rates_path=joinpath(Q06C_ROOT,"rates_$(league).csv")).panel
    train = Q06C.MID.restrict_panel(panel,
        in.(panel.obs_season,Ref(config.honest_train)))
    train_ids = Set(train.obs_match)
    maximum(filter(:match_id=>in(train_ids),panel.matches).match_date) <
        minimum(filter(:season=>in(config.honest_test),panel.matches).match_date) ||
        error("nonchronological train/test split")
    start = time()
    fits = Q06C.train_league(panel,config,out)
    wall = time()-start
    for protocol in ("10b","10a"), rung in ("R2","R6","C0")
        diag = CSV.read(joinpath(out,"convergence_$(rung)_$(protocol).csv"),DataFrame)
        all((diag.rhat .<= 1.05) .& (diag.ess_bulk .>= 200) .&
            (diag.ess_tail .>= 200)) || error("unconverged refit $league $rung $protocol")
    end
    parameters = Q06C.parameter_rows(config,fits)
    for (protocol,p) in (("10a",panel),("10b",train))
        append!(parameters,Q06C.gamma_row(config,fits[("C0",protocol)],p,protocol))
    end
    q06c_compare(league,"parameters",parameters)
    q06c_compare(league,"identifiability",Q06C.style_noise_correlation(config,fits))
    forecast = Q06C.linear_forecasts(panel,config,fits)
    q06c_compare(league,"forecast_fixtures",forecast)
    q06c_compare(league,"forecast_coverage",Q06C.forecast_coverage(forecast))
    q06c_compare(league,"paired_scores",Q06C.paired_forecasts(forecast))
    levels,teams = Q06C.level_geometry(panel,config,fits)
    q06c_compare(league,"levels",levels)
    q06c_compare(league,"level_teams",teams)
    goals = Q06C.goal_fixtures(ds,panel,config,fits[("C0","10b")])
    q06c_compare(league,"goal_fixtures",goals)
    q06c_compare(league,"goal_summary",Q06C.goal_summary(goals))
    println("QS_REFIT ",league," wall=",wall)
    flush(stdout)
    return (; league,wall,n_fixtures=Q06C.MID.n_fixtures(panel),
        sha=Q06C.snapshot_hash(segment))
end
q06c_rows = NamedTuple[]
for (league,tournament,segment) in Q06C_ENTRIES
    push!(q06c_rows,q06c_refit(league,tournament,segment))
    CSV.write(joinpath(Q06C_ROOT,"reproduction","refit_timing.csv"),DataFrame(q06c_rows))
    GC.gc()
end
println("R06_REFIT_DONE")

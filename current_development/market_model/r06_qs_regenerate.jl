# Fresh-beast-REPL reproducibility: recompute EVERY draw-derived scientific CSV
# for all 11 leagues without sampling or overwriting any canonical result.
# Sampling provenance/gate/newton CSVs are historical metadata, not derivable
# from posterior draws; verify their saved gates and preserve them unchanged.

# %%
# ===================================================================
# 1. Packages, pinned source and byte-comparison seam
# ===================================================================
using BayesianFootball, DataFrames, CSV, Serialization, LinearAlgebra, ThreadPinning
if !isdefined(@__MODULE__, :QualityStyleEDA)
    include(joinpath(@__DIR__,"l06_qs_eda.jl"))
end
const Q06B = QualityStyleEDA
strip(read(`hostname`,String)) == "mcmc-beast" || error("regeneration only on beast")
Threads.nthreads() == 16 || error("requires 16 Julia threads")
pinthreads(:cores)
BLAS.set_num_threads(1)
const Q06B_ROOT = joinpath(@__DIR__,"results","QS")
const Q06B_CONFIGS = CSV.read(joinpath(Q06B_ROOT,"configs.csv"),DataFrame)
const Q06B_SELECTION = strip(get(ENV,"QS_REGEN_LEAGUES",""))
const Q06B_ENTRIES = isempty(Q06B_SELECTION) ? Q06B.LEAGUES :
    filter(e -> e[1] in split(Q06B_SELECTION,','),Q06B.LEAGUES)
length(Q06B_ENTRIES) == (isempty(Q06B_SELECTION) ? length(Q06B.LEAGUES) :
    length(split(Q06B_SELECTION,','))) || error("unknown or duplicate regeneration league")
function q06b_compare(path,rows)
    isfile(path) || error("missing canonical CSV: $path")
    tmp = tempname()
    try
        CSV.write(tmp,rows)
        read(tmp) == read(path) || error("non-identical regenerated CSV: $path")
    finally
        isfile(tmp) && rm(tmp)
    end
    return println("BYTE_IDENTICAL ",path)
end

# %%
# ===================================================================
# 2. Reconstruct all posterior-derived tables from immutable saved draws
# ===================================================================
for (league,tournament,segment) in Q06B_ENTRIES
    Q06B.snapshot_hash(segment) ==
        Q06B.SNAPSHOT_HASHES[string(nameof(typeof(segment)))] ||
        error("wrong frozen snapshot for $league")
    row = only(eachrow(filter(:league=>==(league),Q06B_CONFIGS)))
    config = Q06B.MM.MarketModelConfig(; name=league,segment,
        tournaments=[tournament],seasons=String.(split(row.seasons,";")),
        honest_train=String.(split(row.honest_train,";")),
        honest_test=[String(row.honest_test)])
    out = joinpath(Q06B_ROOT,"fits",league)
    fits = Dict{Tuple{String,String},Q06B.MID.ArmFit}()
    for protocol in ("10b","10a"), rung in ("R2","R6","C0")
        diag_path = joinpath(out,"convergence_$(rung)_$(protocol).csv")
        diag = CSV.read(diag_path,DataFrame)
        nrow(diag) > 0 && all((diag.rhat .<= 1.05) .&
            (diag.ess_bulk .>= 200) .& (diag.ess_tail .>= 200)) ||
            error("failed immutable convergence gate: $diag_path")
        saved = Serialization.deserialize(joinpath(out,"$(rung)_$(protocol).jls"))
        fits[(rung,protocol)] = rung == "C0" ? saved.fit : saved
        if rung != "C0"
            q06b_compare(diag_path,Q06B.MID.convergence_table(fits[(rung,protocol)]))
        end
    end
    ds = BayesianFootball.Data.load_datastore_cached(segment; max_age_hours=10^6)
    panel = Q06B.TB.phase_b_panel(ds; config,
        rates_path=joinpath(Q06B_ROOT,"rates_$(league).csv")).panel
    train = Q06B.MID.restrict_panel(panel,
        in.(panel.obs_season,Ref(config.honest_train)))
    parameters = Q06B.parameter_rows(config,fits)
    for (protocol,p) in (("10a",panel),("10b",train))
        append!(parameters,Q06B.gamma_row(config,fits[("C0",protocol)],p,protocol))
    end
    q06b_compare(joinpath(out,"parameters.csv"),parameters)
    q06b_compare(joinpath(out,"identifiability.csv"),
        Q06B.style_noise_correlation(config,fits))
    pred = Q06B.linear_forecasts(panel,config,fits)
    q06b_compare(joinpath(out,"forecast_fixtures.csv"),pred)
    q06b_compare(joinpath(out,"forecast_coverage.csv"),Q06B.forecast_coverage(pred))
    q06b_compare(joinpath(out,"paired_scores.csv"),Q06B.paired_forecasts(pred))
    levels,teams = Q06B.level_geometry(panel,config,fits)
    q06b_compare(joinpath(out,"levels.csv"),levels)
    q06b_compare(joinpath(out,"level_teams.csv"),teams)
    goals = Q06B.goal_fixtures(ds,panel,config,fits[("C0","10b")])
    q06b_compare(joinpath(out,"goal_fixtures.csv"),goals)
    q06b_compare(joinpath(out,"goal_summary.csv"),Q06B.goal_summary(goals))
    println("QS_REGENERATED ",league)
    flush(stdout)
end
println("R06_REGENERATE_DONE")

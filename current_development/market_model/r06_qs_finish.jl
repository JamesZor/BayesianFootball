# Deterministic completion after all six converged QS fits are saved.
# Never samples or modifies a saved posterior. Used if reporting fails after
# the last fit; requires every fit and convergence CSV before writing scores.
# Include in a fresh owned persistent beast REPL, not one-shot Julia.

# %%
# ===================================================================
# 1. Packages, implementation and explicit league
# ===================================================================
using BayesianFootball, DataFrames, CSV, Serialization, LinearAlgebra, ThreadPinning
if !isdefined(@__MODULE__, :QualityStyleEDA)
    include(joinpath(@__DIR__,"l06_qs_eda.jl"))
end
const Q06D = QualityStyleEDA
strip(read(`hostname`,String)) == "mcmc-beast" || error("finish only on beast")
Threads.nthreads() == 16 || error("requires 16 Julia threads")
pinthreads(:cores)
BLAS.set_num_threads(1)
const Q06D_NAME = get(ENV,"QS_LEAGUE","")
const Q06D_ENTRY = only(filter(e -> e[1] == Q06D_NAME,Q06D.LEAGUES))
const Q06D_OUT = get(ENV,"QS_FIT_OUT",joinpath(@__DIR__,"results","QS","fits",Q06D_NAME))
q06d_name,q06d_tournament,q06d_segment = Q06D_ENTRY
Q06D.snapshot_hash(q06d_segment) == Q06D.SNAPSHOT_HASHES[string(nameof(typeof(q06d_segment)))] ||
    error("wrong pinned snapshot")
q06d_row = only(eachrow(filter(:league=>==(Q06D_NAME),
    CSV.read(joinpath(@__DIR__,"results","QS","configs.csv"),DataFrame))))
const Q06D_CONFIG = Q06D.MM.MarketModelConfig(; name=Q06D_NAME,segment=q06d_segment,
    tournaments=[q06d_tournament],seasons=String.(split(q06d_row.seasons,";")),
    honest_train=String.(split(q06d_row.honest_train,";")),
    honest_test=[String(q06d_row.honest_test)])

# %%
# ===================================================================
# 2. Verify every persisted chain and rebuild the exact immutable panel
# ===================================================================
q06d_fits = Dict{Tuple{String,String},Q06D.MID.ArmFit}()
for protocol in ("10b","10a"), rung in ("R2","R6","C0")
    diag = CSV.read(joinpath(Q06D_OUT,"convergence_$(rung)_$(protocol).csv"),DataFrame)
    all((diag.rhat .<= 1.05) .& (diag.ess_bulk .>= 200) .&
        (diag.ess_tail .>= 200)) || error("unconverged $rung $protocol")
    saved = Serialization.deserialize(joinpath(Q06D_OUT,"$(rung)_$(protocol).jls"))
    q06d_fits[(rung,protocol)] = rung == "C0" ? saved.fit : saved
end
q06d_ds = BayesianFootball.Data.load_datastore_cached(q06d_segment; max_age_hours=10^6)
q06d_panel = Q06D.TB.phase_b_panel(q06d_ds; config=Q06D_CONFIG,
    rates_path=joinpath(@__DIR__,"results","QS","rates_$(Q06D_NAME).csv")).panel

# %%
# ===================================================================
# 3. Posterior summaries and pre-week fixture scores (no refit)
# ===================================================================
q06d_parameters = Q06D.parameter_rows(Q06D_CONFIG,q06d_fits)
q06d_train = Q06D.MID.restrict_panel(q06d_panel,
    in.(q06d_panel.obs_season,Ref(Q06D_CONFIG.honest_train)))
for (protocol,panel) in (("10a",q06d_panel),("10b",q06d_train))
    append!(q06d_parameters,Q06D.gamma_row(Q06D_CONFIG,q06d_fits[("C0",protocol)],panel,protocol))
end
param_path = joinpath(Q06D_OUT,"parameters.csv")
if isfile(param_path)
    original = read(param_path)
    tmp = tempname(Q06D_OUT)
    CSV.write(tmp,q06d_parameters)
    read(tmp) == original || error("saved posterior parameter CSV differs from fresh REPL")
    rm(tmp)
else
    CSV.write(param_path,q06d_parameters)
end
q06d_pred = Q06D.linear_forecasts(q06d_panel,Q06D_CONFIG,q06d_fits)
CSV.write(joinpath(Q06D_OUT,"forecast_fixtures.csv"),q06d_pred)
CSV.write(joinpath(Q06D_OUT,"forecast_coverage.csv"),Q06D.forecast_coverage(q06d_pred))
CSV.write(joinpath(Q06D_OUT,"paired_scores.csv"),Q06D.paired_forecasts(q06d_pred))

# %%
# ===================================================================
# 4. Descriptive level geometry and exploratory 10b goals
# ===================================================================
q06d_levels,q06d_teams = Q06D.level_geometry(q06d_panel,Q06D_CONFIG,q06d_fits)
CSV.write(joinpath(Q06D_OUT,"levels.csv"),q06d_levels)
CSV.write(joinpath(Q06D_OUT,"level_teams.csv"),q06d_teams)
q06d_goal = Q06D.goal_fixtures(q06d_ds,q06d_panel,Q06D_CONFIG,q06d_fits[("C0","10b")])
CSV.write(joinpath(Q06D_OUT,"goal_fixtures.csv"),q06d_goal)
CSV.write(joinpath(Q06D_OUT,"goal_summary.csv"),Q06D.goal_summary(q06d_goal))
println("R06_FINISH_DONE")

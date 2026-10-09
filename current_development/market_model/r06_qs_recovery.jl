# C0 synthetic recovery at two predeclared leagues, three generation seeds each.
# Truth is the league's 10a posterior medians; panel shape and 4x(2000+3000)
# budget match production. Recovery is exploratory, not a selection mechanism.
# Include only in an owned persistent 16-thread beast Julia REPL.

# %%
# ===================================================================
# 1. Packages, frozen output and sample prerequisites
# ===================================================================
using BayesianFootball, CSV, DataFrames, LinearAlgebra, Serialization, ThreadPinning
if !isdefined(@__MODULE__, :QualityStyleEDA)
    include(joinpath(@__DIR__,"l06_qs_eda.jl"))
end
const Q06S = QualityStyleEDA
strip(read(`hostname`,String)) == "mcmc-beast" || error("recovery sampling only on beast")
Threads.nthreads() == 16 || error("recovery requires 16 Julia threads")
pinthreads(:cores)
BLAS.set_num_threads(1)
const Q06S_NAME = get(ENV,"QS_LEAGUE","")
Q06S_NAME in ("eng_premier","sco_league_two") || error("recovery restricted to sharp and thin leagues")
const Q06S_ENTRY = only(filter(e -> e[1] == Q06S_NAME,Q06S.LEAGUES))
const Q06S_OUT = joinpath(@__DIR__,"results","QS","fits",Q06S_NAME)
isfile(joinpath(Q06S_OUT,"c0_recovery.csv")) && error("recovery already reported; no overwrite")
q06s_convergence = CSV.read(joinpath(Q06S_OUT,"convergence_C0_10a.csv"),DataFrame)
all(q06s_convergence.gate_pass) || error("unconverged truth fit")
q06s_fit = Serialization.deserialize(joinpath(Q06S_OUT,"C0_10a.jls")).fit
q06s_fit.arm.name == :C0 || error("wrong recovery reference model")

# %%
# ===================================================================
# 2. Same snapshot, eligible seasons, fixture grid and full-book templates
# ===================================================================
q06s_name,q06s_tournament,q06s_segment = Q06S_ENTRY
Q06S.snapshot_hash(q06s_segment) ==
    Q06S.SNAPSHOT_HASHES[string(nameof(typeof(q06s_segment)))] || error("wrong pinned cache")
q06s_row = only(eachrow(filter(:league=>==(Q06S_NAME),
    CSV.read(joinpath(@__DIR__,"results","QS","configs.csv"),DataFrame))))
const Q06S_CONFIG = Q06S.MM.MarketModelConfig(; name=Q06S_NAME,
    segment=q06s_segment,tournaments=[q06s_tournament],
    seasons=String.(split(q06s_row.seasons,";")),
    honest_train=String.(split(q06s_row.honest_train,";")),
    honest_test=[String(q06s_row.honest_test)])
q06s_ds = BayesianFootball.Data.load_datastore_cached(q06s_segment; max_age_hours=10^6)
q06s_panel = Q06S.TB.phase_b_panel(q06s_ds; config=Q06S_CONFIG,
    rates_path=joinpath(@__DIR__,"results","QS","rates_$(Q06S_NAME).csv")).panel
q06s_markets = Q06S.PM.fullbook_markets(q06s_ds,q06s_panel,Q06S_CONFIG)
println("QS recovery $Q06S_NAME: ",Q06S.MID.n_fixtures(q06s_panel),
    " fixtures; ",q06s_panel.n_weeks," weeks; ",Q06S.MID.n_teams(q06s_panel)," teams")

# %%
# ===================================================================
# 3. Three seeds; preserve recovered and missed truths alike
# ===================================================================
q06s_recovery = Q06S.c0_recovery(q06s_panel,q06s_markets,Q06S_CONFIG,q06s_fit,Q06S_OUT)
CSV.write(joinpath(Q06S_OUT,"c0_recovery.csv"),q06s_recovery)
println("R06_RECOVERY_DONE")

# Cross-league quality/style EDA: one league per immutable beast run.
# R2 independent alpha/beta, R6 correlated alpha/beta, C0 independent quality/style.
# Each rung uses the Phase B2/C priors, seeds and 4x(2000+3000) budget, unchanged.
# 10a fits every eligible season; 10b theta fits earlier seasons ONLY and scores
# the last complete season pre-week (all fixtures in a week predicted before updates).
# Goals are an exploratory outcome check, not a model fitted to scores.
# Include in an owned persistent mcmc-beast REPL with 16 threads; no SQL writes.

# %%
# ===================================================================
# 1. Packages and stream-local implementation
# ===================================================================
using BayesianFootball, DataFrames, CSV, LinearAlgebra, ThreadPinning
if !isdefined(@__MODULE__, :QualityStyleEDA)
    include(joinpath(@__DIR__, "l06_qs_eda.jl"))
end
const Q06F = QualityStyleEDA

# %%
# ===================================================================
# 2. League, immutable output and pinned fresh cache
# ===================================================================
strip(read(`hostname`,String)) == "mcmc-beast" || error("sampling only on beast")
Threads.nthreads() == 16 || error("requires 16 threads")
pinthreads(:cores)
BLAS.set_num_threads(1)
const Q06F_NAME = get(ENV,"QS_LEAGUE","")
const Q06F_ENTRY = only(filter(e -> e[1] == Q06F_NAME,Q06F.LEAGUES))
const Q06F_OUT = get(ENV,"QS_FIT_OUT",joinpath(@__DIR__,"results","QS","fits",Q06F_NAME))
isdir(Q06F_OUT) && !isempty(readdir(Q06F_OUT)) && error("output is not fresh: $Q06F_OUT")
mkpath(Q06F_OUT)
q06f_name,q06f_tournament,q06f_segment = Q06F_ENTRY
Q06F.snapshot_hash(q06f_segment) == Q06F.SNAPSHOT_HASHES[string(nameof(typeof(q06f_segment)))] ||
    error("snapshot SHA mismatch: $Q06F_NAME")
q06f_configs = CSV.read(joinpath(@__DIR__,"results","QS","configs.csv"),DataFrame)
q06f_config_row = only(eachrow(filter(:league=>==(Q06F_NAME),q06f_configs)))
q06f_seasons = split(q06f_config_row.seasons,";")
q06f_train = split(q06f_config_row.honest_train,";")
q06f_test = [String(q06f_config_row.honest_test)]
const Q06F_CONFIG = Q06F.MM.MarketModelConfig(; name=Q06F_NAME,segment=q06f_segment,
    tournaments=[q06f_tournament],seasons=String.(q06f_seasons),
    honest_train=String.(q06f_train),honest_test=q06f_test)
Q06F_CONFIG.price_window == (-20.0,0.0) || error("price window differs")

# %%
# ===================================================================
# 3. Isolated inverted full-book targets; inspect split before training
# ===================================================================
q06f_ds = BayesianFootball.Data.load_datastore_cached(q06f_segment; max_age_hours=10^6)
q06f_rates = joinpath(@__DIR__,"results","QS","rates_$(Q06F_NAME).csv")
q06f_panel = Q06F.TB.phase_b_panel(q06f_ds; config=Q06F_CONFIG,rates_path=q06f_rates).panel
q06f_train = Q06F.MID.restrict_panel(q06f_panel,
    in.(q06f_panel.obs_season,Ref(Q06F_CONFIG.honest_train)))
q06f_test = filter(:season=>in(Q06F_CONFIG.honest_test),q06f_panel.matches)
q06f_train_ids = Set(q06f_train.obs_match)
q06f_train_dates = filter(:match_id=>in(q06f_train_ids),q06f_panel.matches).match_date
maximum(q06f_train_dates) < minimum(q06f_test.match_date) ||
    error("honest split is not chronological")
println("QS $Q06F_NAME: full=",Q06F.MID.n_fixtures(q06f_panel),
    " train=",Q06F.MID.n_fixtures(q06f_train)," test=",nrow(q06f_test),
    " weeks=",q06f_panel.n_weeks," teams=",Q06F.MID.n_teams(q06f_panel))
flush(stdout)

# %%
# ===================================================================
# 4. Inherited engine/state gates before model sampling
# ===================================================================
q06f_gates = Q06F.PM.fullbook_engine_gates()
CSV.write(joinpath(Q06F_OUT,"engine_gates.csv"),q06f_gates)
all(q06f_gates.pass) || error("engine/state gates failed")

# %%
# ===================================================================
# 5. Six production fits; reject any Rhat >1.05 or bulk/tail ESS <200
# ===================================================================
q06f_started = time()
q06f_fits = Q06F.train_league(q06f_panel,Q06F_CONFIG,Q06F_OUT)
q06f_parameters = Q06F.parameter_rows(Q06F_CONFIG,q06f_fits)
for (protocol,panel) in (("10a",q06f_panel),("10b",q06f_train))
    append!(q06f_parameters,Q06F.gamma_row(Q06F_CONFIG,q06f_fits[("C0",protocol)],panel,protocol))
end
CSV.write(joinpath(Q06F_OUT,"parameters.csv"),q06f_parameters)
CSV.write(joinpath(Q06F_OUT,"identifiability.csv"),
    Q06F.style_noise_correlation(Q06F_CONFIG,q06f_fits))

# %%
# ===================================================================
# 6. Pre-week linear densities, paired by fixture; descriptive RTS levels
# ===================================================================
q06f_predictions = Q06F.linear_forecasts(q06f_panel,Q06F_CONFIG,q06f_fits)
CSV.write(joinpath(Q06F_OUT,"forecast_fixtures.csv"),q06f_predictions)
CSV.write(joinpath(Q06F_OUT,"forecast_coverage.csv"),Q06F.forecast_coverage(q06f_predictions))
CSV.write(joinpath(Q06F_OUT,"paired_scores.csv"),Q06F.paired_forecasts(q06f_predictions))
q06f_levels,q06f_teams = Q06F.level_geometry(q06f_panel,Q06F_CONFIG,q06f_fits)
CSV.write(joinpath(Q06F_OUT,"levels.csv"),q06f_levels)
CSV.write(joinpath(Q06F_OUT,"level_teams.csv"),q06f_teams)

# %%
# ===================================================================
# 7. Exploratory goals, using C0's same pre-week conditional state
# ===================================================================
q06f_goal = Q06F.goal_fixtures(q06f_ds,q06f_panel,Q06F_CONFIG,q06f_fits[("C0","10b")])
CSV.write(joinpath(Q06F_OUT,"goal_fixtures.csv"),q06f_goal)
CSV.write(joinpath(Q06F_OUT,"goal_summary.csv"),Q06F.goal_summary(q06f_goal))
println("QS $Q06F_NAME end-to-end seconds=",time()-q06f_started)
println("R06_FIT_DONE")

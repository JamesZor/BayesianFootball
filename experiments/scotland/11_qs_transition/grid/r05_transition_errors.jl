# D23 / DESIGN §9: export paired per-game errors; no fit or gate changes.
using BayesianFootball, CSV, DataFrames
const ERROR_DIR = @__DIR__
ds = Data.load_datastore_cached(Data.ScottishPyramid(); max_age_hours=10000)
tiers = Harness.club_season_tiers(ds)
runs = CSV.read(joinpath(ERROR_DIR, "runs.csv"), DataFrame)
control_obs = CSV.read(joinpath(ERROR_DIR, "observations_qs_control.csv"), DataFrame)
panel = unique(Int.(control_obs.match_id))
panel_set = Set(panel)
# The observation panel is priced; count every league game BEFORE intersecting it.
games = Dict{Tuple{Int,String},Int}()
counts = Dict{Tuple{String,String},Int}()
matches = filter(r -> Int(r.tournament_id) in (54,55,56,57), ds.matches)
sort!(matches, [:match_date,:match_id])
for r in eachrow(matches), club in (String(r.home_team),String(r.away_team))
    key = (club,String(r.season))
    get(tiers,key,0) == Int(r.tournament_id) || continue
    n = get(counts,key,0)+1
    counts[key] = n
    games[(Int(r.match_id),club)] = n
end
output = DataFrame(arm=String[],run_id=String[],direction=String[],match_id=Int[],
                   club=String[],season=String[],game=Int[],delta=Float64[])
for run in eachrow(runs)
    obs = CSV.read(joinpath(ERROR_DIR,"observations_$(run.label).csv"),DataFrame)
    # Pooled observations also contain monitored Championship fixtures. Restrict
    # all arms to the frozen control's priced lower-tier panel before pairing.
    obs = filter(r -> Int(r.match_id) in panel_set, obs)
    Set(Int.(obs.match_id)) == panel_set || error("Unequal priced target panel $(run.label)")
    obs.selection = Symbol.(obs.selection)
    for direction in (:relegated_into_L1,:promoted_into_L1,:entered_spfl)
        cohort = Harness.transition_cohort(ds,panel,tiers;first_n=20,direction)
        values = Harness._transition_bias_observations(obs,ds,cohort.clubs)
        for r in eachrow(values)
            push!(output,(String(run.label),String(run.run_id),string(direction),r.match_id,
                          r.club,r.season,games[(r.match_id,r.club)],r.delta))
        end
    end
end
sort!(output,[:arm,:direction,:club,:season,:game,:match_id])
CSV.write(joinpath(ERROR_DIR,"transition_errors.csv"),output)
println("S3B_TRANSITION_ERRORS_DONE rows=$(nrow(output))")

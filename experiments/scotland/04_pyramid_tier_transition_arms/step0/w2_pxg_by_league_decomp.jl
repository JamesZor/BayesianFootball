# W2 step 0b: why does goals/pxG differ by league? Shot counts vs pxG per shot vs event mix.
using BayesianFootball, DataFrames, Statistics, Printf

ds = Data.load_datastore_cached(Data.ScottishAll(); max_age_hours = 10_000)
m = ds.matches
tid = Dict(Int(r.match_id) => Int(r.tournament_id) for r in eachrow(m))
season = Dict(Int(r.match_id) => String(r.season) for r in eachrow(m))
goals = Dict(Int(r.match_id) => (ismissing(r.home_score) ? missing : r.home_score + r.away_score) for r in eachrow(m))
panel = Set(["23/24", "24/25", "25/26"])

shots = Features.build_shots(ds)
model = Features.fit_shot_xg(shots; k = 25.0)
shots.xg = Float64.(Features.predict_xg(model, shots))
shots = shots[[haskey(season, i) && season[i] in panel for i in shots.match_id], :]
shots.tid = [tid[i] for i in shots.match_id]

println("== commentary shots by league (23/24–25/26 matches with commentary) ==")
@printf("%-4s %6s %9s %9s %9s %9s %9s %9s\n", "tid", "matches", "shots/m", "xg/shot", "xg/m", "cmtGoal/m", "goals/m", "parsed%")
for t in (54, 55, 56, 57)
    s = shots[shots.tid .== t, :]
    ids = unique(s.match_id); n = length(ids)
    g = mean(skipmissing([goals[i] for i in ids]))
    @printf("%-4d %6d %9.2f %9.4f %9.3f %9.3f %9.3f %9.1f\n", t, n, nrow(s) / n, mean(s.xg),
            sum(s.xg) / n, sum(s.is_goal) / n, g, 100 * mean(s.parsed))
end

println("\n== event mix per match by league ==")
mix = combine(groupby(shots, [:tid, :event_type]), nrow => :k)
nm = Dict(t => length(unique(shots.match_id[shots.tid .== t])) for t in (54, 55, 56, 57))
mix.per_match = [r.k / nm[r.tid] for r in eachrow(mix)]
w = unstack(mix[:, [:tid, :event_type, :per_match]], :event_type, :tid, :per_match)
show(stdout, MIME("text/plain"), w; allrows = true, allcols = true)

println("\n\n== zone / context mix (share of shots) by league ==")
for col in (:zone, :context)
    z = combine(groupby(shots, [:tid, col]), nrow => :k)
    tot = Dict(t => sum(z.k[z.tid .== t]) for t in (54, 55, 56, 57))
    z.share = [round(100 * r.k / tot[r.tid]; digits = 1) for r in eachrow(z)]
    show(stdout, MIME("text/plain"), unstack(z[:, [:tid, col, :share]], col, :tid, :share); allrows = true)
    println()
end

println("\n== independent shot counts (statistics table) ==")
st = ds.statistics
println("statistics columns: ", first(names(st), min(40, ncol(st))))
println("bbc columns: ", names(ds.bbc))
println("W2_PXG_DECOMP_DONE")

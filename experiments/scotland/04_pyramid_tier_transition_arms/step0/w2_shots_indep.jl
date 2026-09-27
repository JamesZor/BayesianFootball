using BayesianFootball, DataFrames, Statistics, Printf
ds = Data.load_datastore_cached(Data.ScottishAll(); max_age_hours = 10_000)
panel = Set(["23/24", "24/25", "25/26"])
season = Dict(Int(r.match_id) => String(r.season) for r in eachrow(ds.matches))
shots = Features.build_shots(ds)
cmt = combine(groupby(shots, :match_id), nrow => :cmt_shots, :event_type => (e -> count(==("attempt_blocked"), e)) => :cmt_blocked)
b = dropmissing(ds.bbc[:, [:match_id, :tournament_id, :shots_h, :shots_a, :sot_h, :sot_a]])
b = b[[get(season, Int(i), "") in panel for i in b.match_id], :]
j = innerjoin(b, cmt, on = :match_id)
st = ds.statistics
hasblk = "blockedScoringAttempt_home" in names(st) && "totalShotsOnGoal_home" in names(st)
@printf("%-4s %6s %11s %9s %10s %11s\n", "tid", "n", "bbcShots/m", "bbcSoT/m", "cmtShots/m", "cmtBlock/m")
for t in (54, 55, 56, 57)
    s = j[j.tournament_id .== t, :]
    @printf("%-4d %6d %11.2f %9.2f %10.2f %11.2f\n", t, nrow(s), mean(s.shots_h .+ s.shots_a),
            mean(s.sot_h .+ s.sot_a), mean(s.cmt_shots), mean(s.cmt_blocked))
end
if hasblk
    x = st[(st.period .== "ALL") .& [get(season, Int(i), "") in panel for i in st.match_id], :]
    println("\nsofascore (period ALL): per match totals")
    for t in (54, 55, 56, 57)
        s = x[x.tournament_id .== t, :]
        tot = collect(skipmissing(s.totalShotsOnGoal_home .+ s.totalShotsOnGoal_away))
        blk = collect(skipmissing(s.blockedScoringAttempt_home .+ s.blockedScoringAttempt_away))
        @printf("%-4d n=%4d totalShots/m=%6.2f  blocked/m=%5.2f (n_blk=%d)\n", t, length(tot), isempty(tot) ? NaN : mean(tot), isempty(blk) ? NaN : mean(blk), length(blk))
    end
end
println("W2_SHOTS_INDEP_DONE")

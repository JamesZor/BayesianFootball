# W2 step 0 (TODO 031): does finishing (goals / pxG) differ by SPFL league?
# Read-only. Gate: drop the league-δ cells if every league's κ_league/κ_pooled is within ±3%.
using BayesianFootball, DataFrames, Statistics, Random, Printf

ds = Data.load_datastore_cached(Data.ScottishAll(); max_age_hours = 10_000)
m = ds.matches
println("match columns: ", names(m))
hs, as = :home_score in propertynames(m) ? (:home_score, :away_score) : error("no score columns")

obs = Features.pxg_match_observations(ds, Features.PxGFeature(k = 25.0, fallback = :none))
println("pxG-covered matches: ", length(obs))

rows = DataFrame(tid = Int[], season = String[], goals = Float64[], pxg = Float64[])
for r in eachrow(m)
    (ismissing(r[hs]) || ismissing(r[as])) && continue
    o = get(obs, Int(r.match_id), nothing)
    o === nothing && continue
    (isfinite(o.h) && isfinite(o.a)) || continue
    push!(rows, (Int(r.tournament_id), String(r.season), r[hs] + r[as], o.h + o.a))
end

function boot_ratio(g, x; B = 5000, rng = MersenneTwister(31))
    n = length(g); rs = Vector{Float64}(undef, B)
    for b in 1:B
        idx = rand(rng, 1:n, n); rs[b] = sum(g[idx]) / sum(x[idx])
    end
    return quantile(rs, (0.025, 0.975))
end

for (label, seasons) in (("W2 panel 23/24-25/26", ["23/24", "24/25", "25/26"]),
                         ("all pxG seasons", unique(rows.season)))
    sub = rows[in.(rows.season, Ref(seasons)), :]
    pooled = sum(sub.goals) / sum(sub.pxg)
    @printf("\n== %s: pooled goals/pxG = %.4f (n = %d) ==\n", label, pooled, nrow(sub))
    @printf("%-4s %6s %9s %9s %8s %18s %10s %18s\n",
            "tid", "n", "goals/m", "pxg/m", "ratio", "95% CI", "rel", "rel 95% CI")
    for t in (54, 55, 56, 57)
        s = sub[sub.tid .== t, :]
        nrow(s) == 0 && continue
        r = sum(s.goals) / sum(s.pxg); lo, hi = boot_ratio(s.goals, s.pxg)
        @printf("%-4d %6d %9.3f %9.3f %8.4f   [%.4f, %.4f] %+9.1f%%   [%+.1f%%, %+.1f%%]\n",
                t, nrow(s), mean(s.goals), mean(s.pxg), r, lo, hi,
                100 * (r / pooled - 1), 100 * (lo / pooled - 1), 100 * (hi / pooled - 1))
    end
end

println("\n== goals per match by league and season (all finished league matches) ==")
g = combine(groupby(dropmissing(m[:, [:tournament_id, :season, hs, as]]), [:tournament_id, :season]),
            [hs, as] => ((h, a) -> mean(h .+ a)) => :goals_pm, nrow => :n)
sort!(g, [:season, :tournament_id])
show(stdout, MIME("text/plain"), g[in.(g.season, Ref(["21/22","22/23","23/24","24/25","25/26"])), :]; allrows = true)
println("\nW2_KAPPA_CHECK_DONE")

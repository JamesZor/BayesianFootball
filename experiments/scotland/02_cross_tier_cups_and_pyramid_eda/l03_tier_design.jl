# l03 — Long-format tier design shared by r03 (tier-step GLMs) and r05 (walk-forward pricing).
#
#   log μ_ij = β0 + comp_c + h·Home_ij + h_cup·Home_ij·Cup + g_T(i) + g_T(j) + ½(θ_T(i) − θ_T(j))
#
# Requires _common.jl.

"""
    long_design(fx; split_old_firm=false, t5split=false)

Returns (y, X, colnames, clusters) for senior-only fixtures.  With
`split_old_firm`, Celtic and Rangers become their own level "T0" (so T1 = the
other Premiership clubs; Rangers 2012–16 are T0 too).  With `t5split`, T5 splits into T5a / T6+ (r01's
heuristic marker).
"""
function long_design(fx; split_old_firm = false, t5split = false)
    d = senior_only(fx)
    # Old Firm → "T0" in every season they appear (incl. Rangers' 2012–16 climb through T4→T2),
    # so the ordinary tier means are never carried by a club outside the normal pyramid.
    lev(cat, team, sub) = split_old_firm && cat in SENIOR && team in OLD_FIRM ? "T0" :
                          t5split && cat == "T5" ? (coalesce(sub, "T6+") == "T5a" ? "T5" : "T6") : cat
    ht = lev.(d.home_cat, d.home_team, d.home_t5sub); at = lev.(d.away_cat, d.away_team, d.away_t5sub)
    levels_ = sort(unique(vcat(ht, at)))
    ref = "T1"; others = filter(!=(ref), levels_)
    n = nrow(d)
    att = vcat(ht, at); dfn = vcat(at, ht)
    home = vcat(Float64.(.!d.neutral), zeros(n))
    cup = Float64.(vcat(d.is_cup, d.is_cup))
    y = Float64.(vcat(d.home_goals, d.away_goals))
    comp = vcat(d.competition, d.competition)
    cols = Dict{String, Vector{Float64}}()
    cols["(Intercept)"] = ones(2n)
    for c in ("Scottish Cup", "League Cup", "Challenge Cup"); cols["comp: " * c] = Float64.(comp .== c); end
    cols["home"] = home; cols["home×cup"] = home .* cup
    for k in others
        cols["g_" * k] = Float64.(att .== k) .+ Float64.(dfn .== k)
        cols["θ_" * k] = 0.5 .* (Float64.(att .== k) .- Float64.(dfn .== k))
    end
    names_ = vcat("(Intercept)", ["comp: " * c for c in ("Scottish Cup", "League Cup", "Challenge Cup")],
                  "home", "home×cup", ["g_" * k for k in others], ["θ_" * k for k in others])
    keep = [nm for nm in names_ if any(!=(0), cols[nm])]   # drop cups absent from a window
    X = reduce(hcat, (cols[nm] for nm in keep))
    cs(team, fs) = string(team, "@", fs)
    att_cs = vcat(cs.(d.home_team, d.fs), cs.(d.away_team, d.fs))
    def_cs = vcat(cs.(d.away_team, d.fs), cs.(d.home_team, d.fs))
    match = vcat(d.match_id, d.match_id)
    return (; y, X, names = keep, att_cs, def_cs, match, levels = levels_, nmatch = n,
              att, dfn, home, is_cup = Bool.(cup))
end

"""
    design_row(names, att_level, def_level, home, competition)

One long-format row (club at level `att_level` attacking a club at `def_level`) aligned
with the columns `names` of a fitted `long_design`.  Levels absent from `names` (the
reference T1, or a level unseen in training) contribute zero.
"""
function design_row(names, att, dfn, home::Bool, competition)
    cup = competition in ("Scottish Cup", "League Cup", "Challenge Cup")
    map(names) do nm
        nm == "(Intercept)" ? 1.0 :
        startswith(nm, "comp: ") ? Float64(nm == "comp: " * competition) :
        nm == "home" ? Float64(home) :
        nm == "home×cup" ? Float64(home && cup) :
        startswith(nm, "g_") ? Float64(att == nm[3:end]) + Float64(dfn == nm[3:end]) :
        startswith(nm, "θ_") ? 0.5 * (Float64(att == nm[nextind(nm, 1, 2):end]) - Float64(dfn == nm[nextind(nm, 1, 2):end])) : 0.0
    end
end

# r04 — Unified Dixon–Coles network ratings across every Scottish fixture.
#
#   include("experiments/scotland/02_cross_tier_cups_and_pyramid_eda/r04_dixon_coles_network_ratings.jl")
#
#   log λ_home = μ + comp_c + h·Home + h_cup·Home·Cup + a_{i,s} + d_{j,s}
#   log λ_away = μ + comp_c +                            a_{j,s} + d_{i,s}
#   a_{i,s} = A_i + a′_{i,s},   d_{i,s} = D_i + d′_{i,s}    (club + club-season deviation)
#   P(x, y) = τ_ρ(x, y; λ_h, λ_a) · Pois(x; λ_h) · Pois(y; λ_a)     (Dixon–Coles low-score term)
#
# Penalised MLE (= MAP under Gaussian priors):  A, D ~ N(0, σ_club²),  a′, d′ ~ N(0, σ_season²).
# No tier information enters the fit — tier structure in the ratings is measured, not
# imposed.  B-teams and guest clubs are ordinary nodes (they never share a parent's rating).
# Sign convention matches the L1 engines: a = attack (α), d = concession (β, + ⇒ leakier),
# net strength θ = a − d.
#
# Outputs: results/r04_*.csv|md, results/figures/r04_*.png, data/r04_club_season_ratings.csv

# ── 1. Setup ────────────────────────────────────────────────────────────────
include(joinpath(@__DIR__, "_common.jl"))
using Optim, Distributions, SpecialFunctions, Plots
gr(); default(fontfamily = "sans-serif", dpi = 150, framestyle = :axes, grid = :y, gridalpha = 0.15)

fx_all = load_fixtures(:long)
fx = fx_all[fx_all.in_primary_window, :]

# ── 2. Dixon–Coles core (build_network, dc_fg!, fit_dc, rating_table) ─────
include(joinpath(@__DIR__, "l04_dixon_coles_core.jl"))

# ── 3. Fits: main spec + sensitivity grid (primary), main spec (long) ───────
# σ_season is the season-to-season movement of a club's rating; σ_club the spread of
# permanent club levels.  "independent" (σ_club → 0, σ_season = 1) links seasons only
# through cup ties — the purest cup-identified (and noisiest) variant.
const MAIN = "main σc=1.0 σs=0.20"
grid = [(MAIN, 1.0, 0.20), ("σc=1.0 σs=0.35", 1.0, 0.35),
        ("σc=1.0 σs=0.10", 1.0, 0.10), ("independent σc=0.01 σs=1.0", 0.01, 1.0)]
runs = vcat([("primary", fx, g...) for g in grid], [("long", fx_all, grid[1]...)])
R_parts = DataFrame[]; glob_rows = NamedTuple[]
for (win, d, lab, σc, σs) in runs
    N = build_network(d)
    local p, res = fit_dc(N, σc, σs)
    push!(R_parts, rating_table(N, p, lab, win))
    push!(glob_rows, (window = win, spec = lab, matches = N.D.nm, club_seasons = N.D.ncs, nll = Optim.minimum(res),
                      iterations = Optim.iterations(res), converged = Optim.converged(res),
                      mu = p[1], comp_SC = p[2], comp_LC = p[3], comp_CC = p[4], home = p[5], home_cup = p[6], rho = p[7]))
end
R_all = vcat(R_parts...); glob = DataFrame(glob_rows)
CSV.write(joinpath(DATA, "r04_club_season_ratings.csv"), R_all)
save_csv("r04_global_parameters.csv", glob); save_md("r04_global_parameters.md", glob; digits = 3)
R = R_all[R_all.window .== "primary", :]                # sensitivity grid, primary window
Rm = R[R.spec .== MAIN, :]
Rlong = R_all[R_all.window .== "long", :]

# ── 4. Tier distributions and steps from the ratings ────────────────────────
# Only club-seasons with ≥ 10 fixtures (so every league club-season, but not a
# one-off amateur side) enter the tier summaries; T5 uses ≥ 2 because non-league
# clubs play at most a handful of Scottish/Challenge Cup ties.
elig(t) = (t.cat .!= "T5" .&& t.n .>= 10) .| (t.cat .== "T5" .&& t.n .>= 2)
function tier_summary(t; exclude_of = false)
    t = t[elig(t) .& in.(t.cat, Ref(SENIOR)), :]
    exclude_of && (t = t[.!t.old_firm, :])
    sort(combine(groupby(t, [:window, :spec, :cat]), nrow => :club_seasons,
                 :theta => mean => :theta_mean, :theta => std => :theta_sd,
                 :theta => (x -> quantile(x, 0.1)) => :theta_p10, :theta => median => :theta_p50,
                 :theta => (x -> quantile(x, 0.9)) => :theta_p90,
                 :a => mean => :a_mean, :a => std => :a_sd, :d => mean => :d_mean, :d => std => :d_sd), [:window, :spec, :cat])
end
ts = vcat(insertcols!(tier_summary(R_all), 3, :old_firm => "incl"),
          insertcols!(tier_summary(R_all; exclude_of = true), 3, :old_firm => "excl"))
save_csv("r04_tier_rating_distributions.csv", ts)
save_md("r04_tier_rating_distributions.md", ts[startswith.(ts.spec, "main"), Not(:spec)]; digits = 3)

# steps between consecutive tier means (θ, a, d) — comparable with r03's τ, ΔA, ΔD
step_rows = NamedTuple[]
for g in groupby(ts, [:window, :spec, :old_firm]), k in 1:4
    u = g[g.cat .== "T$k", :]; l = g[g.cat .== "T$(k+1)", :]
    (nrow(u) == 1 && nrow(l) == 1) || continue
    push!(step_rows, (window = g.window[1], spec = g.spec[1], old_firm = g.old_firm[1], step = "T$k→T$(k+1)",
                      tau = u.theta_mean[1] - l.theta_mean[1], d_attack = u.a_mean[1] - l.a_mean[1],
                      d_concede = u.d_mean[1] - l.d_mean[1],
                      pooled_within_sd = sqrt((u.theta_sd[1]^2 + l.theta_sd[1]^2) / 2)))
end
rsteps = DataFrame(step_rows)
save_csv("r04_rating_tier_steps.csv", rsteps); save_md("r04_rating_tier_steps.md", rsteps; digits = 3)

# ── 5. Overlap between adjacent tiers ───────────────────────────────────────
# AUC = P(θ of a random upper-tier club-season > θ of a random lower-tier one), within
# the same football season; plus the share of lower-tier club-seasons that out-rate the
# upper tier's median / bottom club.
ov_rows = NamedTuple[]
E = Rm[elig(Rm) .& in.(Rm.cat, Ref(SENIOR)), :]
for k in 1:4, of in (false, true)
    up_all = E[(E.cat .== "T$k") .& (of ? .!E.old_firm : trues(nrow(E))), :]
    lo_all = E[E.cat .== "T$(k+1)", :]
    wins = 0.0; pairs = 0; above_med = 0; above_min = 0; nlo = 0
    for s in unique(E.fs)
        u = up_all.theta[up_all.fs .== s]; l = lo_all.theta[lo_all.fs .== s]
        (isempty(u) || isempty(l)) && continue
        wins += sum((ui > li) + 0.5(ui == li) for ui in u, li in l); pairs += length(u) * length(l)
        above_med += count(>(median(u)), l); above_min += count(>(minimum(u)), l); nlo += length(l)
    end
    push!(ov_rows, (upper = "T$k" * (of && k == 1 ? " excl. Old Firm" : ""), lower = "T$(k+1)",
                    auc_upper_beats_lower = wins / pairs,
                    pct_lower_above_upper_median = 100above_med / nlo,
                    pct_lower_above_upper_bottom = 100above_min / nlo, lower_club_seasons = nlo))
end
ov = unique(DataFrame(ov_rows), [:upper, :lower])
save_csv("r04_adjacent_tier_overlap.csv", ov); save_md("r04_adjacent_tier_overlap.md", ov; digits = 3)

# top-of-lower vs bottom-of-upper, each season (the "bridge" clubs)
br_rows = NamedTuple[]
for s in sort(unique(E.fs)), k in 1:3
    u = E[(E.fs .== s) .& (E.cat .== "T$k"), :]; l = E[(E.fs .== s) .& (E.cat .== "T$(k+1)"), :]
    (nrow(u) == 0 || nrow(l) == 0) && continue
    iu = argmin(u.theta); il = argmax(l.theta)
    push!(br_rows, (season = fs_label(s), pair = "T$k/T$(k+1)", bottom_upper = u.team[iu], theta_bottom_upper = u.theta[iu],
                    top_lower = l.team[il], theta_top_lower = l.theta[il], gap = u.theta[iu] - l.theta[il]))
end
bridges = DataFrame(br_rows)
save_csv("r04_bridge_clubs_by_season.csv", bridges); save_md("r04_bridge_clubs_by_season.md", bridges; digits = 3)

# ── 6. Promotion / relegation transitions (feeds Option B in r06) ───────────
# For each club whose tier changed between fs−1 and fs: its rating in fs relative to the
# mean of the NON-moving clubs of its new tier (the zero point of a lower-league-only
# model), and the change in its own rating across the move.
function transitions(t)
    t = t[in.(t.cat, Ref(SENIOR)) .& (t.n .>= 10), :]
    key = Dict((r.team, r.fs) => r for r in eachrow(t))
    rows = NamedTuple[]
    for r in eachrow(t)
        prev = get(key, (r.team, r.fs - 1), nothing)
        prev === nothing && continue
        tp, tn = parse(Int, prev.cat[2:end]), parse(Int, r.cat[2:end])
        tp == tn && continue
        stayers = t[(t.fs .== r.fs) .& (t.cat .== r.cat), :]
        stayers = stayers[[get(key, (x, r.fs - 1), nothing) !== nothing && key[(x, r.fs - 1)].cat == r.cat for x in stayers.team], :]
        nrow(stayers) < 3 && continue
        push!(rows, (team = r.team, season = fs_label(r.fs), move = tn > tp ? "relegated" : "promoted",
                     from = prev.cat, to = r.cat,
                     d_theta_vs_new_tier = r.theta - mean(stayers.theta),
                     d_a_vs_new_tier = r.a - mean(stayers.a), d_d_vs_new_tier = r.d - mean(stayers.d),
                     d_theta_vs_old_tier = prev.theta - mean(t.theta[(t.fs .== r.fs - 1) .& (t.cat .== prev.cat)]),
                     own_theta_change = r.theta - prev.theta, own_a_change = r.a - prev.a, own_d_change = r.d - prev.d))
    end
    DataFrame(rows)
end
tr = vcat((insertcols!(transitions(g), 1, :window => g.window[1], :spec => g.spec[1]) for g in groupby(R_all, [:window, :spec]))...)
save_csv("r04_transition_clubs.csv", tr)
trs = sort(combine(groupby(tr, [:window, :spec, :move, :to]), nrow => :n,
                   :d_theta_vs_new_tier => mean => :d_theta_mean, :d_theta_vs_new_tier => std => :d_theta_sd,
                   :d_a_vs_new_tier => mean => :d_a_mean, :d_a_vs_new_tier => std => :d_a_sd,
                   :d_d_vs_new_tier => mean => :d_d_mean, :d_d_vs_new_tier => std => :d_d_sd,
                   :own_theta_change => mean => :own_theta_change), [:window, :spec, :move, :to])
save_csv("r04_transition_summary.csv", trs); save_md("r04_transition_summary.md", trs; digits = 3)

# ── 7. Figure: θ by tier, per club-season (main spec) ───────────────────────
p = plot(size = (760, 460), xlabel = "", ylabel = "Net strength θ = a − d (log goal-rate)", legend = :topright,
         title = "Club-season Dixon–Coles ratings by tier, 2021/22–2026/27", titlefontsize = 10,
         xticks = (1:7, ["T1", "T2", "T3", "T4", "T5+", "B-teams", "Guests"]))
rng = MersenneTwister(29)
for (k, c) in enumerate(["T1", "T2", "T3", "T4", "T5", "B", "GUEST"])
    t = Rm[(Rm.cat .== c) .& (c in ("T5", "B", "GUEST") ? (Rm.n .>= 2) : (Rm.n .>= 10)), :]
    isempty(t) && continue
    x = k .+ 0.28 .* (rand(rng, nrow(t)) .- 0.5)
    of = t.old_firm
    scatter!(p, x[.!of], t.theta[.!of]; ms = 3, color = colorant"#2a78d6", alpha = 0.55, msw = 0, label = k == 1 ? "club-season" : "")
    any(of) && scatter!(p, x[of], t.theta[of]; ms = 4, color = colorant"#eb6834", msw = 0, label = "Celtic / Rangers")
    plot!(p, [k - 0.25, k + 0.25], fill(mean(t.theta[.!of]), 2); color = :black, lw = 2, label = k == 1 ? "tier mean (excl. OF)" : "")
end
savefig(p, joinpath(FIGS, "r04_club_season_theta_by_tier.png"))

# ── 8. Console summary ─────────────────────────────────────────────────────
println("Tier distributions (main):"); show(ts[startswith.(ts.spec, "main"), [:window, :old_firm, :cat, :club_seasons, :theta_mean, :theta_sd, :a_mean, :d_mean]]; allrows = true); println()
println("Rating steps:"); show(rsteps[(rsteps.old_firm .== "excl"), :]; allrows = true); println()
println("Overlap:"); show(ov; allrows = true); println()
println("Transitions:"); show(trs; allrows = true); println()

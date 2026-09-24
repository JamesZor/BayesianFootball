# r02 — Empirical supremacy matrices across the Scottish pyramid.
#
#   include("experiments/scotland/02_cross_tier_cups_and_pyramid_eda/r02_descriptive_supremacy_eda.jl")
#
# Inputs : data/r01_pyramid_fixtures.csv
# Outputs: results/r02_*.csv|md, results/figures/r02_*.png
#
# Everything here is descriptive: no team strength adjustment (that is r03/r04).

# ── 1. Setup ────────────────────────────────────────────────────────────────
include(joinpath(@__DIR__, "_common.jl"))
using Plots, GLM, StatsModels, Distributions
gr(); default(fontfamily = "sans-serif", dpi = 150, framestyle = :axes, grid = :y, gridalpha = 0.15)
const C1, C2, C3 = colorant"#2a78d6", colorant"#eb6834", colorant"#1baf7a"   # categorical slots 1–3

fx_all = load_fixtures(:long)
fx = fx_all[fx_all.in_primary_window, :]

"Multiplicative de-vig of a decimal 1X2 triplet (r05 adds Shin)."
function devig!(d)
    s = 1 ./ d.odds_home .+ 1 ./ d.odds_draw .+ 1 ./ d.odds_away
    d.p_home_mkt = (1 ./ d.odds_home) ./ s
    d.p_draw_mkt = (1 ./ d.odds_draw) ./ s
    d.p_away_mkt = (1 ./ d.odds_away) ./ s
    d
end
devig!(fx); devig!(fx_all)

# ── 2. ΔTier matrix, home perspective (senior first teams, non-neutral) ─────
# ΔTier = Tier_away − Tier_home: +k ⇒ the home side is k divisions higher.
function supremacy_row(g)
    gd = g.home_goals .- g.away_goals
    ds = g.home_shots .- g.away_shots; dt = g.home_sot .- g.away_sot; dx = g.home_pxg .- g.away_pxg
    (N = nrow(g), home_pct = 100mean(gd .> 0), draw_pct = 100mean(gd .== 0), away_pct = 100mean(gd .< 0),
     mean_gd = mean(gd), se_gd = nrow(g) > 1 ? std(gd) / sqrt(nrow(g)) : NaN,
     total_goals = mean(g.home_goals .+ g.away_goals),
     n_shots = nnz_(ds), d_shots = nanmean(ds), d_sot = nanmean(dt),
     n_pxg = nnz_(dx), d_pxg = nanmean(dx),
     n_mkt = nnz_(g.p_home_mkt), mkt_home_pct = 100nanmean(g.p_home_mkt), mkt_away_pct = 100nanmean(g.p_away_mkt))
end
sen = senior_only(fx)
sen_nn = sen[.!sen.neutral, :]
dmat = sort(combine(groupby(sen_nn, :tier_delta), supremacy_row), :tier_delta)
save_csv("r02_tier_delta_matrix.csv", dmat); save_md("r02_tier_delta_matrix.md", dmat; digits = 2)

# Same table on the 2008+ long window (goals only — BBC/odds coverage starts 2020/21).
sen_long = senior_only(fx_all); sen_long = sen_long[.!sen_long.neutral, :]
dmat_long = sort(combine(groupby(sen_long, :tier_delta),
                         g -> (r = supremacy_row(g); (N = r.N, home_pct = r.home_pct, draw_pct = r.draw_pct,
                               away_pct = r.away_pct, mean_gd = r.mean_gd, se_gd = r.se_gd, total_goals = r.total_goals))),
                 :tier_delta)
save_csv("r02_tier_delta_matrix_long.csv", dmat_long); save_md("r02_tier_delta_matrix_long.md", dmat_long; digits = 2)

# Full home-tier × away-tier matrix (includes same-tier league baseline on the diagonal).
pair = sort(combine(groupby(sen_nn, [:home_tier, :away_tier]), supremacy_row), [:home_tier, :away_tier])
save_csv("r02_home_away_tier_matrix.csv", pair)

# ── 3. Oriented by the higher-tier club: gap × competition × venue ──────────
function oriented_row(g)
    (N = nrow(g), hi_win_pct = 100mean(g.res_hi .== 1), draw_pct = 100mean(g.res_hi .== 0),
     upset_pct = 100mean(g.res_hi .== -1), mean_gd_hi = mean(g.gd_hi),
     se_gd_hi = nrow(g) > 1 ? std(g.gd_hi) / sqrt(nrow(g)) : NaN,
     total_goals = mean(g.goals_hi .+ g.goals_lo),
     n_shots = nnz_(g.shots_hi), d_shots_hi = nanmean(g.shots_hi .- g.shots_lo),
     d_sot_hi = nanmean(g.sot_hi .- g.sot_lo), n_pxg = nnz_(g.pxg_hi), d_pxg_hi = nanmean(g.pxg_hi .- g.pxg_lo),
     n_mkt = nnz_(g.p_mkt_hi), mkt_hi_win_pct = 100nanmean(g.p_mkt_hi))
end
ox = oriented_cross_tier(fx)
ox_long = oriented_cross_tier(fx_all)
by_pair = sort(combine(groupby(ox, [:hi_tier, :lo_tier]), oriented_row), [:hi_tier, :lo_tier])
by_gap_comp = sort(combine(groupby(ox, [:gap, :competition]), oriented_row), [:gap, :competition])
by_gap_venue = sort(combine(groupby(ox, [:gap, :venue]), oriented_row), [:gap, :venue])
by_pair_long = sort(combine(groupby(ox_long, [:hi_tier, :lo_tier]), oriented_row), [:hi_tier, :lo_tier])
by_gap_comp_long = sort(combine(groupby(ox_long, [:gap, :competition]), oriented_row), [:gap, :competition])
for (n, t) in (("r02_oriented_by_tier_pair", by_pair), ("r02_oriented_by_gap_competition", by_gap_comp),
               ("r02_oriented_by_gap_venue", by_gap_venue), ("r02_oriented_by_tier_pair_long", by_pair_long),
               ("r02_oriented_by_gap_competition_long", by_gap_comp_long))
    save_csv(n * ".csv", t); save_md(n * ".md", t; digits = 2)
end

# ── 4. Does the Challenge Cup compress supremacy? ───────────────────────────
# OLS: gd_hi ~ gap (factor) + venue + competition, HC1 errors.  The competition
# coefficient is the Challenge Cup / League Cup margin shift vs the Scottish Cup
# at equal tier gap and venue.  Run on both windows (long = 5× the power).
function hc1(m)
    X = modelmatrix(m); e = residuals(m); n, k = size(X)
    B = inv(X'X); V = B * (X' * Diagonal(e .^ 2) * X) * B .* (n / (n - k))
    DataFrame(term = coefnames(m), coef = coef(m), se = sqrt.(diag(V)),
              z = coef(m) ./ sqrt.(diag(V)), p = 2 .* ccdf.(Normal(), abs.(coef(m) ./ sqrt.(diag(V)))))
end
rot = DataFrame[]
for (lab, d) in (("primary", ox), ("long", ox_long))
    d = copy(d); d.gapf = string.(d.gap)
    d.comp = [c == "Scottish Cup" ? "0_SC" : c == "League Cup" ? "1_LC" : "2_CC" for c in d.competition]
    m = lm(@formula(gd_hi ~ gapf + venue + comp), d)
    t = hc1(m); insertcols!(t, 1, :window => lab, :n => nrow(d)); push!(rot, t)
end
rot = vcat(rot...)
save_csv("r02_challenge_cup_rotation_ols.csv", rot); save_md("r02_challenge_cup_rotation_ols.md", rot; digits = 3)

# ── 5. B-teams and guest clubs (kept out of every senior tier estimate) ─────
function vs_senior(fx, cat)
    rows = NamedTuple[]
    for r in eachrow(fx)
        if r.home_cat == cat && r.away_cat in SENIOR
            push!(rows, (opp = r.away_cat, gd_senior = r.away_goals - r.home_goals, senior_home = false, neutral = r.neutral))
        elseif r.away_cat == cat && r.home_cat in SENIOR
            push!(rows, (opp = r.home_cat, gd_senior = r.home_goals - r.away_goals, senior_home = true, neutral = r.neutral))
        end
    end
    d = DataFrame(rows)
    sort(combine(groupby(d, :opp), nrow => :N, :gd_senior => (x -> 100mean(x .> 0)) => :senior_win_pct,
                 :gd_senior => (x -> 100mean(x .== 0)) => :draw_pct, :gd_senior => (x -> 100mean(x .< 0)) => :senior_loss_pct,
                 :gd_senior => mean => :mean_gd_senior, :gd_senior => (x -> std(x) / sqrt(length(x))) => :se,
                 :senior_home => mean => :share_senior_home), :opp)
end
btab = vs_senior(fx_all[fx_all.match_date .>= Date(2016, 7, 1), :], "B")
gtab = vs_senior(fx_all, "GUEST")
save_csv("r02_b_teams_vs_senior.csv", btab); save_md("r02_b_teams_vs_senior.md", btab; digits = 2)
save_csv("r02_guests_vs_senior.csv", gtab); save_md("r02_guests_vs_senior.md", gtab; digits = 2)

# ── 6. Same-tier baseline: home advantage in league vs cup ──────────────────
same = sen_nn[coalesce.(sen_nn.tier_delta .== 0, false), :]
ha = sort(combine(groupby(same, [:is_cup, :home_tier]), supremacy_row), [:is_cup, :home_tier])
save_csv("r02_same_tier_home_advantage.csv", ha); save_md("r02_same_tier_home_advantage.md", ha[:, [:is_cup, :home_tier, :N, :home_pct, :draw_pct, :away_pct, :mean_gd, :se_gd, :total_goals]]; digits = 2)

# ── 7. Figures ──────────────────────────────────────────────────────────────
# (a) mean home GD, home tier × away tier (long window for coverage; N annotated)
pl = sort(combine(groupby(sen_long, [:home_tier, :away_tier]), nrow => :N,
                  [:home_goals, :away_goals] => ((h, a) -> mean(h .- a)) => :gd), [:home_tier, :away_tier])
Z = fill(NaN, 5, 5); Nn = zeros(Int, 5, 5)
for r in eachrow(pl); Z[r.home_tier, r.away_tier] = r.gd; Nn[r.home_tier, r.away_tier] = r.N; end
lim = maximum(abs, filter(isfinite, Z))
div = cgrad([colorant"#e34948", colorant"#f2f2f0", colorant"#2a78d6"])
p = heatmap(1:5, 1:5, Z; c = div, clims = (-lim, lim), yflip = true, aspect_ratio = 1, size = (620, 540),
            xlabel = "Away club tier", ylabel = "Home club tier", colorbar_title = "mean home goal diff",
            title = "Home goal difference by tier pairing, 2008/09–2026/27\n(senior first teams, non-neutral)",
            titlefontsize = 10, xticks = (1:5, ["T1", "T2", "T3", "T4", "T5+"]), yticks = (1:5, ["T1", "T2", "T3", "T4", "T5+"]))
for i in 1:5, j in 1:5
    Nn[i, j] > 0 && annotate!(p, j, i, text(@sprintf("%+.2f\nn=%d", Z[i, j], Nn[i, j]), 7, :black))
end
savefig(p, joinpath(FIGS, "r02_home_away_tier_gd_heatmap.png"))

# (b) higher-tier goal margin by gap, per competition (long window), 95% CI
p2 = plot(size = (720, 420), xlabel = "Tier gap (divisions)", ylabel = "Higher-tier goal difference",
          title = "Higher-tier margin by tier gap and cup, 2008/09–2026/27", titlefontsize = 10, legend = :topleft)
for (k, (comp, col)) in enumerate((("Scottish Cup", C1), ("League Cup", C2), ("Challenge Cup", C3)))
    t = by_gap_comp_long[(by_gap_comp_long.competition .== comp) .& (by_gap_comp_long.N .>= 8), :]
    x = t.gap .+ (k - 2) * 0.12
    scatter!(p2, x, t.mean_gd_hi; yerror = 1.96 .* t.se_gd_hi, color = col, msc = col, ms = 5, label = comp)
end
hline!(p2, [0]; color = :gray, lw = 1, label = "")
savefig(p2, joinpath(FIGS, "r02_margin_by_gap_competition.png"))

println("ΔTier matrix (primary, non-neutral):"); show(dmat[:, [:tier_delta, :N, :home_pct, :draw_pct, :away_pct, :mean_gd, :d_shots, :d_sot, :n_pxg, :d_pxg, :mkt_home_pct]]; allrows = true); println()
println("Oriented by pair (long):"); show(by_pair_long[:, [:hi_tier, :lo_tier, :N, :hi_win_pct, :draw_pct, :upset_pct, :mean_gd_hi, :se_gd_hi]]; allrows = true); println()
println("Rotation OLS:"); show(rot[occursin.("comp", rot.term), :]; allrows = true); println()
println("B-teams:"); show(btab); println()
println("Guests:"); show(gtab); println()

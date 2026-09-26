# ==============================================================================
# r02_market_feature_attribution.jl — why does the market favour some teams so hard?
# ==============================================================================
#
# WHAT THIS IS AND IS NOT
# -----------------------
# Phase 2 of TODO 023. Phase 1 showed HOW the market's team ratings move (a
# no-momentum random walk). This runner asks WHAT the market's supremacy is made
# of, and why it is sharper than a goals model:
#
#   Δ_mkt  = log λ_mkt,h − log λ_mkt,a          (inverted Betfair close)
#   Δ_goal = E[log λ_h − log λ_a]               (TODO 021 m01: pure-Poisson GRW,
#                                                walk-forward, 40 folds, OOS)
#   Gap    = Δ_mkt − Δ_goal                      (the market conviction gap)
#
# regressed on features every one of which is known BEFORE the close:
#
#   goal history   Δ_goal, goal-form supremacy, points-per-game gap
#   wealth         production (age-weighted) wealth Δ, raw log-sum wealth Δ
#   lineup         shots-RAPM XI differential (starters + 0.10 × bench)
#   proxy xG       commentary proxy-xG form supremacy
#   rest/schedule  rest-days differential, log travel distance
#
# and then the SAME features enter the Phase 1 state-space model with Student-t
# observation noise, to see how much team-rating spread and fixture-level noise
# they absorb.
#
# It is NOT a causal study: the market may price a thing because it correlates
# with what it really prices. Shapley shares split SHARED variance evenly among
# collinear groups; that is a convention, stated, not a finding.
#
# FILTRATION CONTRACT
# -------------------
# * Δ_goal is out-of-sample: each fixture scored by the fold trained strictly
#   before it (run 2b42d3bf-28d7-47ac-8706-88798c9031ac).
# * Lineup / wealth / travel columns are the point-in-time design of the
#   feature-compression EDA (commit c3bdb53a, pinned as inputs/…csv). The lineup
#   is the PLAYED XI, which the market knows at the close (teamsheets are out
#   ~1 h before kick-off); ratings are history-fitted shots-RAPM.
# * Form, rest and ppg use strictly earlier calendar days only.
#
# USAGE (mcmc-beast, -t 16):
#     include("current_development/market_inverse_dynamics/r02_market_feature_attribution.jl")
# MID2_SMOKE=1 for a short pass.
# ==============================================================================

# %%
# ===================================================================
# 1. Packages and implementation
# ===================================================================
using BayesianFootball
using DataFrames, Dates, Statistics, LinearAlgebra, Printf, Random, UUIDs
using CSV, Serialization
using ThreadPinning

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "l01_market_inverse_loader.jl"))
const MID = MarketInverseDynamics

# %%
# ===================================================================
# 2. Configuration
# ===================================================================
const MID2_SMOKE      = get(ENV, "MID2_SMOKE", "0") == "1"
const MID2_SEASONS    = ["24/25", "25/26"]
const MID2_GOAL_RUN   = (experiment = "fast_slow_grw_scottish_lower",
                         run_id = UUID("2b42d3bf-28d7-47ac-8706-88798c9031ac"),
                         name = "m01_poisson_grw_tight")
const MID2_EDA_PANEL  = joinpath(@__DIR__, "inputs", "fce_fixture_panel_c3bdb53a.csv")
const MID2_FAV_CUT    = 0.60        # market favourite: max(p_home, p_away) ≥ this
const MID2_BOOT       = MID2_SMOKE ? 50 : 1_000
const MID2_CHAINS     = 4
const MID2_WARMUP     = MID2_SMOKE ? 100 : 1_000
const MID2_SAMPLES    = MID2_SMOKE ? 100 : 2_000
const MID2_THIN       = MID2_SMOKE ? 1 : 4   # σ_obs / ν mix slowly through the ω mixture
const MID2_PATHS      = MID2_SMOKE ? 8 : 100
const MID2_SEED       = 20260923

# feature => group. Order fixes the column order everywhere below.
const MID2_FEATURES = [
    "delta_goal"          => "goal history",
    "goal_form_sup"       => "goal history",
    "delta_ppg"           => "goal history",
    "delta_prod_wealth"   => "wealth",
    "delta_wealth_logsum" => "wealth",
    "delta_lineup"        => "lineup (RAPM)",
    "pxg_form_sup"        => "proxy xG",
    "delta_rest"          => "rest & schedule",
    "log_dist_z"          => "rest & schedule",
]
const MID2_GROUPS = ["goal history", "wealth", "lineup (RAPM)", "proxy xG", "rest & schedule"]

# %%
# ===================================================================
# 3. Output directory
# ===================================================================
const MID2_OUT = joinpath(@__DIR__, "results", MID2_SMOKE ? "phase2_smoke" : "phase2")
mkpath(MID2_OUT)
println("threads = $(Threads.nthreads()), output → $MID2_OUT")

# %%
# ===================================================================
# 4. Data — the panel, the goal model, the features
# ===================================================================
ds = Data.load_datastore_cached(Data.ScottishLower())
panel, book, inversion = MID.build_market_panel(ds; seasons = MID2_SEASONS)

goal_fit = BayesianFootball.load_fit(BayesianFootball.Training.PostgresStorage(MID2_GOAL_RUN.experiment),
                                     MID2_GOAL_RUN.run_id)
goal_sup = MID.latent_supremacy(goal_fit.latents)
eda = CSV.read(MID2_EDA_PANEL, DataFrame)
eda = select(eda, :match_id, :delta_lineup, :delta_prod_wealth, :delta_wealth_logsum,
             :log_dist_z, :p_home_mkt, :p_away_mkt, :sup_model_m05, :sup_model)
form = MID.phase2_form_features(ds)
sched = MID.phase2_schedule_features(ds)

fx = DataFrame(match_id = panel.matches.match_id, match_date = panel.matches.match_date,
               season = panel.matches.season, home_team = panel.matches.home_team,
               away_team = panel.matches.away_team,
               delta_mkt = log.(panel.matches.lambda_mkt_h) .- log.(panel.matches.lambda_mkt_a))
for (other, how) in ((goal_sup, :left), (eda, :left), (form, :left), (sched, :left))
    global fx = leftjoin(fx, other; on = :match_id, order = :left)
end
nrow(fx) == MID.n_fixtures(panel) || error("feature frame has $(nrow(fx)) rows for $(MID.n_fixtures(panel)) fixtures")
fx.match_id == [panel.obs_match[2m-1] for m in 1:MID.n_fixtures(panel)] ||
    error("feature frame is not in the panel's fixture order")

coverage = DataFrame(feature = first.(MID2_FEATURES),
                     missing = [count(ismissing, fx[!, Symbol(f)]) for f in first.(MID2_FEATURES)],
                     exact_zero = [count(x -> !ismissing(x) && x == 0.0, fx[!, Symbol(f)]) for f in first.(MID2_FEATURES)])
show(stdout, MIME"text/plain"(), coverage)
println()
CSV.write(joinpath(MID2_OUT, "feature_coverage.csv"), coverage)
any(coverage.missing .> 0) && println("NOTE: missing feature values are imputed to 0 (the neutral differential)")
for (f, _) in MID2_FEATURES
    fx[!, Symbol(f)] = Float64.(coalesce.(fx[!, Symbol(f)], 0.0))
end
fx.p_fav = max.(coalesce.(fx.p_home_mkt, NaN), coalesce.(fx.p_away_mkt, NaN))

# Book quality. A totals-only book pins the goal LEVEL but not who scores them, so
# its inverted supremacy is unidentified; a 3-selection 1X2-only book pins it
# weakly. "well identified" = a 1X2 market plus ≥ 5 quoted selections.
has_1x2 = Set(Int.(book.match_id[book.market_name .== "1X2"]))
fx = leftjoin(fx, select(inversion, :match_id, :n_targets); on = :match_id, order = :left)
fx.has_1x2 = in.(fx.match_id, Ref(has_1x2))
fx.well_identified = fx.has_1x2 .& (fx.n_targets .>= 5)
println("book quality: $(count(.!fx.has_1x2)) fixtures without 1X2, " *
        "$(count(fx.well_identified)) of $(nrow(fx)) well identified (1X2 + ≥ 5 selections)")
fx.gap = fx.delta_mkt .- fx.delta_goal

# Standardised design: divide by the panel sd, do NOT centre (0 = level teams).
feat_names = first.(MID2_FEATURES)
Xraw = Matrix{Float64}(fx[:, Symbol.(feat_names)])
feat_sd = vec(std(Xraw; dims = 1))
Xz = Xraw ./ feat_sd'
groups = [findall(==(g), last.(MID2_FEATURES)) for g in MID2_GROUPS]
CSV.write(joinpath(MID2_OUT, "fixture_features.csv"), fx)

# %%
# ===================================================================
# 5. Engine gates (Phase 1 + the Phase 2 feature / Student-t path)
# ===================================================================
gates = MID.mid_gates()
all(gates.pass) || error("engine gates failed")
CSV.write(joinpath(MID2_OUT, "engine_gates.csv"), gates)
println("engine gates: $(nrow(gates)) / $(nrow(gates)) pass")

# %%
# ===================================================================
# 6. The conviction gap — how compressed is the goal model?
# ===================================================================
slope_goal = MID.ols_fit(fx.delta_goal, reshape(fx.delta_mkt, :, 1))
slope_m05 = MID.ols_fit(fx.sup_model_m05, reshape(fx.delta_mkt, :, 1))
slope_m12 = MID.ols_fit(fx.sup_model, reshape(fx.delta_mkt, :, 1))
compression = DataFrame(model = ["m01 pure-Poisson GRW (Δ_goal)", "m05 joint + wealth (time decay)",
                                 "m12 joint + wealth + lineup"],
                        slope_on_market = [slope_goal.coef[2], slope_m05.coef[2], slope_m12.coef[2]],
                        r2 = [slope_goal.r2, slope_m05.r2, slope_m12.r2],
                        sd_model = [std(fx.delta_goal), std(fx.sup_model_m05), std(fx.sup_model)],
                        sd_market = std(fx.delta_mkt))
show(stdout, MIME"text/plain"(), compression)
println()
CSV.write(joinpath(MID2_OUT, "compression.csv"), compression)

bands = [(0.0, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 1.0)]
fx.orient = sign.(fx.delta_mkt)
gap_bands = DataFrame([(band = @sprintf("%.2f–%.2f", lo, hi),
                        n = count(r -> lo <= r.p_fav < hi, eachrow(fx)),
                        mkt_conviction = mean(abs.(fx.delta_mkt[lo .<= fx.p_fav .< hi])),
                        goal_conviction = mean((fx.orient .* fx.delta_goal)[lo .<= fx.p_fav .< hi]),
                        gap = mean((fx.orient .* fx.gap)[lo .<= fx.p_fav .< hi]))
                       for (lo, hi) in bands])
show(stdout, MIME"text/plain"(), gap_bands)
println()
CSV.write(joinpath(MID2_OUT, "gap_by_favourite_band.csv"), gap_bands)

# %%
# ===================================================================
# 7. Attribution — OLS, exact Shapley splits, bootstrap bands
# ===================================================================
# 7a. Coefficients (standardised features, HC1 SEs), market supremacy and gap.
coef_rows = NamedTuple[]
for (target, y) in (("delta_mkt", fx.delta_mkt), ("gap", fx.gap))
    b, se = MID.ols_hc1(y, Xz)
    for (k, f) in enumerate(feat_names)
        push!(coef_rows, (target = target, feature = f, group = last(MID2_FEATURES[k]),
                          coef_per_sd = b[k+1], se = se[k+1], t = b[k+1] / se[k+1]))
    end
end
coefs = DataFrame(coef_rows)
CSV.write(joinpath(MID2_OUT, "ols_coefficients.csv"), coefs)
show(stdout, MIME"text/plain"(), coefs; allrows = true)
println()

# 7b. Shapley R² over the five groups (+ unexplained), with bootstrap 90% bands.
function mid2_group_r2(rows, y)
    φ = MID.shapley_r2(y[rows], Xz[rows, :], groups)
    return vcat(φ, 1 - sum(φ))
end
r2_rows = NamedTuple[]
for (target, y) in (("delta_mkt", fx.delta_mkt), ("gap", fx.gap))
    est = mid2_group_r2(1:nrow(fx), y)
    lo, hi = MID.bootstrap_ci(rows -> mid2_group_r2(rows, y), nrow(fx); reps = MID2_BOOT)
    for (g, name) in enumerate(vcat(MID2_GROUPS, ["unexplained"]))
        push!(r2_rows, (target = target, group = name, share = est[g], lo = lo[g], hi = hi[g]))
    end
end
r2_shares = DataFrame(r2_rows)
CSV.write(joinpath(MID2_OUT, "shapley_r2_groups.csv"), r2_shares)
show(stdout, MIME"text/plain"(), r2_shares; allrows = true)
println()

# 7c. Per-feature Shapley R² (9 features, 512 coalitions) — the fine print.
fine_rows = NamedTuple[]
for (target, y) in (("delta_mkt", fx.delta_mkt), ("gap", fx.gap))
    φ = MID.shapley_r2(y, Xz, [[k] for k in eachindex(feat_names)])
    for (k, f) in enumerate(feat_names)
        push!(fine_rows, (target = target, feature = f, group = last(MID2_FEATURES[k]), share = φ[k],
                          r2_alone = MID.ols_fit(y, Xz[:, [k]]).r2))
    end
end
fine = DataFrame(fine_rows)
CSV.write(joinpath(MID2_OUT, "shapley_r2_features.csv"), fine)

# 7d. Favourite conviction: the mean oriented supremacy of market favourites
#     (p_fav ≥ MID2_FAV_CUT), split exactly into intercept (home advantage) +
#     each group's Shapley contribution + residual. Same for the gap.
fav = coalesce.(fx.p_fav .>= MID2_FAV_CUT, false)
conv_rows = NamedTuple[]
for (target, y) in (("delta_mkt", fx.delta_mkt), ("gap", fx.gap))
    sc = MID.shapley_conviction(y, Xz, groups, fav, fx.orient)
    function stat(rows)
        r = MID.shapley_conviction(y[rows], Xz[rows, :], groups, fav[rows], fx.orient[rows])
        return vcat(r.base, r.φ, r.residual, r.total)
    end
    lo, hi = MID.bootstrap_ci(stat, nrow(fx); reps = MID2_BOOT)
    est = vcat(sc.base, sc.φ, sc.residual, sc.total)
    names_ = vcat(["intercept (home adv. / league mean)"], MID2_GROUPS, ["unexplained residual", "TOTAL"])
    for (i, nm) in enumerate(names_)
        push!(conv_rows, (target = target, component = nm, log_rate = est[i],
                          share_of_total = est[i] / sc.total, lo = lo[i], hi = hi[i]))
    end
end
conviction = DataFrame(conv_rows)
CSV.write(joinpath(MID2_OUT, "favourite_conviction_attribution.csv"), conviction)
println("favourites (p_fav ≥ $MID2_FAV_CUT): $(count(fav)) fixtures")
show(stdout, MIME"text/plain"(), conviction; allrows = true)
println()

# 7e. Honest check: fit on 24/25, R² on 25/26, single groups and all together.
s1 = fx.season .== MID2_SEASONS[1]
s2 = .!s1
function mid2_oos_r2(cols, y)
    fit = MID.ols_fit(y[s1], Xz[s1, cols])
    ŷ = hcat(ones(count(s2)), Xz[s2, cols]) * fit.coef
    return 1 - sum((y[s2] .- ŷ) .^ 2) / sum((y[s2] .- mean(y[s2])) .^ 2)
end
oos_rows = NamedTuple[]
for (target, y) in (("delta_mkt", fx.delta_mkt), ("gap", fx.gap))
    for (g, name) in enumerate(MID2_GROUPS)
        push!(oos_rows, (target = target, predictors = name, oos_r2 = mid2_oos_r2(groups[g], y)))
    end
    push!(oos_rows, (target = target, predictors = "all five", oos_r2 = mid2_oos_r2(1:length(feat_names), y)))
    push!(oos_rows, (target = target, predictors = "all but goal history",
                     oos_r2 = mid2_oos_r2(reduce(vcat, groups[2:end]), y)))
end
oos_r2 = DataFrame(oos_rows)

# 7f. Robustness: the same Shapley splits on well-identified books only.
wi = findall(fx.well_identified)
wi_rows = NamedTuple[]
for (target, y) in (("delta_mkt", fx.delta_mkt), ("gap", fx.gap))
    est = mid2_group_r2(wi, y)
    lo, hi = MID.bootstrap_ci(rows -> mid2_group_r2(wi[rows], y), length(wi); reps = MID2_BOOT)
    for (g, name) in enumerate(vcat(MID2_GROUPS, ["unexplained"]))
        push!(wi_rows, (target = target, group = name, share = est[g], lo = lo[g], hi = hi[g]))
    end
    sc = MID.shapley_conviction(y[wi], Xz[wi, :], groups, fav[wi], fx.orient[wi])
    for (nm, v) in zip(vcat(["intercept"], MID2_GROUPS, ["unexplained residual", "TOTAL"]),
                       vcat(sc.base, sc.φ, sc.residual, sc.total))
        push!(wi_rows, (target = target * " | favourites", group = nm, share = v / sc.total,
                        lo = NaN, hi = NaN))
    end
end
wi_tab = DataFrame(wi_rows)
CSV.write(joinpath(MID2_OUT, "shapley_well_identified.csv"), wi_tab)
println("well-identified subset: $(length(wi)) fixtures, $(count(fav[wi])) favourites")
show(stdout, MIME"text/plain"(), wi_tab; allrows = true)
println()
CSV.write(joinpath(MID2_OUT, "oos_r2_season_split.csv"), oos_r2)
show(stdout, MIME"text/plain"(), oos_r2; allrows = true)
println()

# %%
# ===================================================================
# 8. Feature-augmented state-space model (Student-t observation noise)
# ===================================================================
all_cols = collect(eachindex(feat_names))
nongoal = reduce(vcat, groups[2:end])
ss_arms = [
    ("s0_grw1_gauss",   Int[],     false),
    ("s1_grw1_t",       Int[],     true),
    ("s2_t_goal",       groups[1], true),
    ("s3_t_wealth",     groups[2], true),
    ("s4_t_lineup",     groups[3], true),
    ("s5_t_pxg",        groups[4], true),
    ("s6_t_rest",       groups[5], true),
    ("s7_t_nongoal",    nongoal,   true),
    ("s8_t_all",        all_cols,  true),
]
ss_panels = Dict(name => MID.with_features(panel, Xz[:, cols], feat_names[cols]) for (name, cols, _) in ss_arms)
ss_tasks = Dict(name => Threads.@spawn(
                    MID.fit_arm(MID.FeatureGRW(length(cols), st, name), ss_panels[name];
                                n_chains = MID2_CHAINS, n_warmup = MID2_WARMUP,
                                n_samples = MID2_SAMPLES, n_paths = MID2_PATHS,
                                seed = MID2_SEED, thin = MID2_THIN))
                for (name, cols, st) in ss_arms)
ss_fits = Dict(name => fetch(t) for (name, t) in ss_tasks)
foreach(a -> @printf("fitted %-16s in %7.1f s\n", a[1], ss_fits[a[1]].seconds), ss_arms)
serialize(joinpath(MID2_OUT, "ss_fits.jls"), (; ss_fits, fx, feat_names, feat_sd))

# %%
# ===================================================================
# 9. Convergence
# ===================================================================
ss_conv = vcat([MID.convergence_table(ss_fits[a[1]]) for a in ss_arms]...)
ss_conv.gate_pass = (ss_conv.rhat .<= 1.05) .& (ss_conv.ess_bulk .>= 200) .& (ss_conv.ess_tail .>= 200)
CSV.write(joinpath(MID2_OUT, "ss_posterior_summary.csv"), ss_conv)
println("state-space convergence: $(count(ss_conv.gate_pass)) / $(nrow(ss_conv)) parameters pass")
show(stdout, MIME"text/plain"(), ss_conv[.!ss_conv.gate_pass, :]; allrows = true)
println()

# %%
# ===================================================================
# 10. What the features absorb — σ_obs, rating spread, coefficients
# ===================================================================
absorb_rows = NamedTuple[]
for (name, cols, st) in ss_arms
    f = ss_fits[name]
    sp = MID.rating_spread(f, ss_panels[name])
    so = vec(f.draws[:, 1, :])
    push!(absorb_rows, (arm = name, K = length(cols), student = st,
                        sigma_obs = median(so),
                        sigma_att = median(vec(f.draws[:, 2, :])),
                        sigma_def = median(vec(f.draws[:, 3, :])),
                        nu = st ? median(vec(f.draws[:, 4, :])) : Inf,
                        spread_att = sp.att, spread_att_lo = sp.att_lo, spread_att_hi = sp.att_hi,
                        spread_def = sp.def, spread_def_lo = sp.def_lo, spread_def_hi = sp.def_hi,
                        n_downweighted = count(<(0.5), f.aux_mean)))
end
absorb = DataFrame(absorb_rows)
CSV.write(joinpath(MID2_OUT, "ss_absorption.csv"), absorb)
show(stdout, MIME"text/plain"(), absorb; allrows = true)
println()

w_rows = NamedTuple[]
for (name, cols, st) in ss_arms, (k, c) in enumerate(cols)
    f = ss_fits[name]
    j = length(MID.param_names(f.arm)) + k
    w = vec(f.draws[:, j, :])
    push!(w_rows, (arm = name, feature = feat_names[c], group = last(MID2_FEATURES[c]),
                   w_median = median(w), w_q025 = quantile(w, 0.025), w_q975 = quantile(w, 0.975),
                   supremacy_per_sd = 2 * median(w),
                   supremacy_per_unit = 2 * median(w) / feat_sd[c],
                   excludes_zero_95 = quantile(w, 0.025) > 0 || quantile(w, 0.975) < 0))
end
wtab = DataFrame(w_rows)
CSV.write(joinpath(MID2_OUT, "ss_feature_coefficients.csv"), wtab)
show(stdout, MIME"text/plain"(), wtab; allrows = true)
println()

# Where the favourite conviction sits in the full state-space arm: team ratings vs features.
dec = MID.supremacy_decomposition(ss_fits["s8_t_all"], ss_panels["s8_t_all"])
dec.orient = fx.orient
dec.fav = fav
dec_cols = [c for c in names(dec) if startswith(c, "sup_")]
ss_fav = DataFrame(component = dec_cols,
                   mean_oriented_fav = [mean(dec.orient[fav] .* dec[fav, c]) for c in dec_cols],
                   mean_oriented_all = [mean(dec.orient .* dec[!, c]) for c in dec_cols])
CSV.write(joinpath(MID2_OUT, "ss_favourite_decomposition.csv"), ss_fav)
show(stdout, MIME"text/plain"(), ss_fav; allrows = true)
println()

# Fixtures the Student-t model refuses to believe (posterior mean ω < 0.5).
ωbar = vec(ss_fits["s8_t_all"].aux_mean)
outl = fx[ωbar .< 0.5, [:match_id, :match_date, :home_team, :away_team, :delta_mkt, :delta_goal]]
outl.omega = ωbar[ωbar .< 0.5]
sort!(outl, :omega)
CSV.write(joinpath(MID2_OUT, "ss_downweighted_fixtures.csv"), outl)
println("Student-t down-weights $(nrow(outl)) fixtures (ω < 0.5)")

# %%
# ===================================================================
# 11. Final report
# ===================================================================
println("\n=== ATTRIBUTION (Shapley R², market supremacy) ===")
show(stdout, MIME"text/plain"(), filter(:target => ==("delta_mkt"), r2_shares); allrows = true)
println()
println("MID2_DONE")

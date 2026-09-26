# r06 — Translate r03–r05 into prior specifications for the TODO 028 L1 models.
#
#   include("experiments/scotland/02_cross_tier_cups_and_pyramid_eda/r06_prior_calibration_recommendations.jl")
#
# Units: log goal-rate, on the L1 engines' own scale
#   log λ_home = intercept + γ_home + α_home + β_away     (α = attack, β = concession)
# so a tier offset on α adds to the club's scoring rate and one on β to its opponents'.
# Net strength θ = α − β;  a tier step τ splits as  Δα = +share·τ,  Δβ = −(1 − share)·τ.
#
# Inputs : results/r03_*.csv, results/r04_*.csv, results/r05_first5_market_view_summary.csv
# Outputs: results/r06_prior_recommendations.json, results/r06_*.md

# ── 1. Setup ────────────────────────────────────────────────────────────────
include(joinpath(@__DIR__, "_common.jl"))
using Distributions
rd(name) = CSV.read(joinpath(RESULTS, name), DataFrame)
steps03 = rd("r03_tier_steps.csv"); tests03 = rd("r03_linearity_tests.csv"); era03 = rd("r03_tier_steps_by_era.csv")
coef03 = rd("r03_glm_coefficients.csv")
dist04 = rd("r04_tier_rating_distributions.csv"); steps04 = rd("r04_rating_tier_steps.csv"); trans04 = rd("r04_transition_summary.csv")
mkt05 = rd("r05_first5_market_view_summary.csv")
const SPFL_STEPS = ["T1→T2", "T2→T3", "T3→T4"]
const MAIN = "main σc=1.0 σs=0.20"

# ── 2. Option A — tier-step size ────────────────────────────────────────────
# Evidence ladder for one SPFL step (net θ), all with Celtic/Rangers held out of T1
# (they are a club effect, not a tier effect — r03 §4):
ev = DataFrame(source = String[], step = String[], tau = Float64[], se = Float64[])
for r in eachrow(steps03[(steps03.fit .== "primary_oldfirm_split") .& (steps03.family .== "poisson") .& in.(steps03.step, Ref(SPFL_STEPS)), :])
    push!(ev, ("r03 GLM primary, OF split", r.step, r.tau, r.se))
end
for r in eachrow(era03[era03.old_firm_split .& in.(era03.step, Ref(SPFL_STEPS)), :])
    push!(ev, ("r03 GLM era " * r.era * ", OF split", r.step, r.tau, r.se))
end
for r in eachrow(steps04[(steps04.old_firm .== "excl") .& (steps04.spec .== MAIN) .& in.(steps04.step, Ref(SPFL_STEPS)), :])
    push!(ev, ("r04 DC ratings " * r.window, r.step, r.tau, NaN))
end
save_csv("r06_step_evidence.csv", ev); save_md("r06_step_evidence.md", ev; digits = 3)

# Pooled step under the (not rejected, p = 0.92) linear constraint, primary window.
lin = tests03[(tests03.fit .== "primary_oldfirm_split") .& (tests03.family .== "poisson") .&
              (tests03.hypothesis .== "H0 SPFL: τ12=τ23=τ34"), :][1, :]
μ_step, se_step = lin.linear_step, lin.linear_step_se
# Between-era heterogeneity of a single step (method of moments over the 3 eras × 3 steps):
e = era03[era03.old_firm_split .& in.(era03.step, Ref(SPFL_STEPS)), :]
τ2_era = max(var(e.tau) - mean(e.se .^ 2), 0.0)
σ_step = sqrt(se_step^2 + τ2_era)
# Attack share of a step (primary, OF split, SPFL steps, inverse-variance weighted)
ps = steps03[(steps03.fit .== "primary_oldfirm_split") .& (steps03.family .== "poisson") .& in.(steps03.step, Ref(SPFL_STEPS)), :]
share_att = sum(ps.d_attack) / sum(ps.tau)
# HalfNormal(s) matched on the mean: E = s·√(2/π)
s_halfnormal = μ_step / sqrt(2 / π)
hn = truncated(Normal(0, s_halfnormal), 0, Inf); tn = truncated(Normal(μ_step, σ_step), 0, Inf)
@printf("pooled step %.3f ± %.3f (sampling), era τ = %.3f ⇒ prior σ %.3f; attack share %.2f\n",
        μ_step, se_step, sqrt(τ2_era), σ_step, share_att)

# T4 → non-league (only matters if T5 clubs are ever modelled — Challenge/Scottish Cup nodes)
t45 = steps03[(steps03.fit .== "primary_oldfirm_split") .& (steps03.family .== "poisson") .& (steps03.step .== "T4→T5"), :][1, :]
of01 = steps03[(steps03.fit .== "primary_oldfirm_split") .& (steps03.family .== "poisson") .& (steps03.step .== "T0→T1"), :][1, :]

# League scoring offsets (A1's δ_league): g_T from the OF-split primary GLM, made zero-sum over T1..T4
g = Dict("T1" => 0.0)
for k in ("T2", "T3", "T4")
    r = coef03[(coef03.fit .== "primary_oldfirm_split") .& (coef03.family .== "poisson") .& (coef03.term .== "g_" * k), :]
    g[k] = r.coef[1]
end
gbar = mean(values(g))
δ_league = Dict(k => 2 * (g[k] - gbar) for k in ("T1", "T2", "T3", "T4"))   # 2g: both sides' rates move

# Within-tier club dispersion (the σ of club effects around their tier mean), primary, OF excluded
wd = dist04[(dist04.window .== "primary") .& (dist04.spec .== MAIN) .& (dist04.old_firm .== "excl") .& in.(dist04.cat, Ref(("T1", "T2", "T3", "T4"))), :]

optA = DataFrame(parameter = String[], family = String[], mean = Float64[], sd = Float64[], note = String[])
push!(optA, ("d_j (net θ step, each of T1→T2, T2→T3, T3→T4)", "TruncatedNormal(μ, σ; 0, ∞)", μ_step, σ_step,
             "recommended; linearity not rejected post-2020 once Celtic/Rangers are club effects"))
push!(optA, ("d_j (net θ step) — weak alternative", "HalfNormal(s)", mean(hn), std(hn), @sprintf("s = %.3f matches the mean; puts %.0f%% mass below 0.2", s_halfnormal, 100cdf(hn, 0.2))))
push!(optA, ("attack part of each step, Δα", "TruncatedNormal", share_att * μ_step, share_att * σ_step, @sprintf("share %.2f of τ", share_att)))
push!(optA, ("concession part of each step, Δβ (sign: lower tier concedes more)", "TruncatedNormal", (1 - share_att) * μ_step, (1 - share_att) * σ_step, @sprintf("share %.2f of τ", 1 - share_att)))
push!(optA, ("Old Firm over rest of T1 (club effect, not a tier)", "Normal", of01.tau, of01.se, "do NOT fold into τ_1; give Celtic/Rangers ordinary team effects"))
push!(optA, ("T4 → non-league step (only if T5 nodes enter)", "TruncatedNormal", t45.tau, sqrt(t45.se^2 + τ2_era), "Highland/Lowland/EoS/WoS pooled; heterogeneous"))
for r in eachrow(wd)
    push!(optA, ("club σ_θ within $(r.cat)", "Normal(0, σ) around tier mean", 0.0, r.theta_sd, @sprintf("σ_α %.3f, σ_β %.3f", r.a_sd, r.d_sd)))
end
for k in ("T1", "T2", "T3", "T4")
    push!(optA, ("δ_league $k (A1, zero-sum)", "point / Normal(δ, 0.05)", δ_league[k], 0.05, "tier goal level is flat across the SPFL (|δ| ≤ 0.1)"))
end
save_csv("r06_option_a_priors.csv", optA); save_md("r06_option_a_priors.md", optA; digits = 3)

# ── 3. Option B — relegated / promoted cold-start offset ────────────────────
# Offset of a transitioning club vs the mean of its new tier's stayers.
#   structural: r04 DC club-season ratings, long window (n largest) and primary
#   market:     r05 closing-line view of the club over its first five league fixtures
ob_rows = NamedTuple[]
for (lab, win) in (("DC ratings 08/09–26/27", "long"), ("DC ratings 21/22–26/27", "primary"))
    for r in eachrow(trans04[(trans04.window .== win) .& (trans04.spec .== MAIN), :])
        push!(ob_rows, (source = lab, move = r.move, to = r.to, n = r.n, d_theta = r.d_theta_mean, sd_theta = r.d_theta_sd,
                        d_alpha = r.d_a_mean, sd_alpha = r.d_a_sd, d_beta = r.d_d_mean, sd_beta = r.d_d_sd))
    end
end
for r in eachrow(mkt05[mkt05.move .!= "stayed", :])
    push!(ob_rows, (source = "closing market, first 5 league games", move = r.move, to = r.to, n = r.n,
                    d_theta = r.mkt_theta_mean, sd_theta = r.mkt_theta_sd, d_alpha = NaN, sd_alpha = NaN, d_beta = NaN, sd_beta = NaN))
end
ob = sort(DataFrame(ob_rows), [:move, :to, :source])
save_csv("r06_option_b_evidence.csv", ob); save_md("r06_option_b_evidence.md", ob; digits = 3)

# Recommendation: precision-weight the long-window structural estimate (largest n) with the
# market view (point-in-time); the between-club sd is the prior σ_0 (the uncertainty about
# a *specific* new club, not about the mean).
function recommend(move, to)
    s = ob[(ob.source .== "DC ratings 08/09–26/27") .& (ob.move .== move) .& (ob.to .== to), :][1, :]
    m = ob[(ob.source .== "closing market, first 5 league games") .& (ob.move .== move) .& (ob.to .== to), :]
    w_s = s.n / s.sd_theta^2
    θ = nrow(m) == 1 ? (w_s * s.d_theta + m.n[1] / m.sd_theta[1]^2 * m.d_theta[1]) / (w_s + m.n[1] / m.sd_theta[1]^2) : s.d_theta
    share = s.d_alpha / (s.d_alpha - s.d_beta)       # attack share of the net offset
    (move = move, into = to, n_structural = s.n, mu_theta = θ, mu_alpha = share * θ, sigma0_alpha = s.sd_alpha,
     mu_beta = -(1 - share) * θ, sigma0_beta = s.sd_beta, se_of_mean_theta = s.sd_theta / sqrt(s.n))
end
optB = DataFrame([recommend("relegated", "T3"), recommend("relegated", "T4"), recommend("relegated", "T2"),
                  recommend("promoted", "T3"), recommend("promoted", "T2"), recommend("promoted", "T1")])
save_csv("r06_option_b_priors.csv", optB); save_md("r06_option_b_priors.md", optB; digits = 3)

# How plausible are the preliminary offsets? z of each candidate vs the mean offset for a club
# relegated into League One, on the α scale (as B1 is written) and on net θ.
rT3 = optB[(optB.move .== "relegated") .& (optB.into .== "T3"), :][1, :]
cand = DataFrame(candidate = [0.90, 0.75, 0.65, 0.40, 0.20])
cand.z_as_alpha = (cand.candidate .- rT3.mu_alpha) ./ rT3.se_of_mean_theta
cand.z_as_theta = (cand.candidate .- rT3.mu_theta) ./ rT3.se_of_mean_theta
cand.pct_clubs_above_as_theta = [100ccdf(Normal(rT3.mu_theta, sqrt(rT3.sigma0_alpha^2 + rT3.sigma0_beta^2)), c) for c in cand.candidate]
save_csv("r06_candidate_offsets.csv", cand); save_md("r06_candidate_offsets.md", cand; digits = 2)

# ── 4. JSON artefact for TODO 028 ───────────────────────────────────────────
rec = Dict(
    "units" => "log goal-rate on the L1 scale: log λ_h = int + γ + α_h + β_a; θ = α − β",
    "source" => "TODO 029, experiments/scotland/02_cross_tier_cups_and_pyramid_eda",
    "option_A2_tier_steps" => Dict(
        "parameterisation" => "τ_4 = 0, τ_r = Σ_{j≥r} d_j; apply share_attack·τ to α and −(1−share_attack)·τ to β",
        "d_j_recommended" => Dict("family" => "TruncatedNormal", "mu" => μ_step, "sigma" => σ_step, "lower" => 0.0),
        "d_j_weak_alternative" => Dict("family" => "HalfNormal", "s" => s_halfnormal),
        "share_attack" => share_att,
        "sampling_se_of_pooled_step" => se_step, "between_era_sd" => sqrt(τ2_era),
        "linearity_test_primary_of_split" => Dict("wald_chi2" => lin.wald_chi2, "df" => lin.restrictions, "p" => lin.p),
        "old_firm_club_effect_over_T1" => Dict("mu" => of01.tau, "se" => of01.se),
        "club_sd_within_tier" => Dict(r.cat => Dict("theta" => r.theta_sd, "alpha" => r.a_sd, "beta" => r.d_sd) for r in eachrow(wd))),
    "option_A1_league_offsets" => Dict("delta_league_zero_sum" => δ_league, "sd" => 0.05),
    "option_B1_cold_start" => Dict(string(r.move, "_into_", r.into) =>
        Dict("alpha0" => Dict("mu" => r.mu_alpha, "sigma" => r.sigma0_alpha),
             "beta0" => Dict("mu" => r.mu_beta, "sigma" => r.sigma0_beta),
             "theta_mu" => r.mu_theta, "n_structural" => r.n_structural) for r in eachrow(optB)),
    "option_B1_verdict_on_0p90" => Dict("relegated_into_T3_alpha_mu" => rT3.mu_alpha, "relegated_into_T3_theta_mu" => rT3.mu_theta,
                                        "z_0p90_as_alpha" => cand.z_as_alpha[1], "z_0p90_as_theta" => cand.z_as_theta[1]))
save_json("r06_prior_recommendations.json", rec)

println("Option A:"); show(optA; allrows = true); println()
println("Option B:"); show(optB; allrows = true); println()
println("Candidates:"); show(cand); println()

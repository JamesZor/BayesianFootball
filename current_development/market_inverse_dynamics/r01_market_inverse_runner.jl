# ==============================================================================
# r01_market_inverse_runner.jl — what does the market think teams are, week to week?
# ==============================================================================
#
# WHAT THIS IS AND IS NOT
# -----------------------
# An exploratory state-space study of the INVERTED BETFAIR CLOSE (TODO 023). The
# target is log λ_mkt — the market's price, not goals — decomposed as
#
#     log λ_home = μ + γ_home + α[home, week] + β[away, week] + ε
#     log λ_away = μ          + α[away, week] + β[home, week] + ε
#
# and the question is which dynamic law for α, β the market's repricing follows:
#
#   a0  static ratings (control: is there dynamics at all?)
#   a1  1st-order GRW, constant σ                       (DESIGN Arm 1)
#   a1b a1 + a one-off season-boundary jump            (control for the summer)
#   a2  damped-velocity momentum GRW                    (DESIGN Arm 2)
#   a3  stochastic-volatility GRW                       (DESIGN Arm 3)
#   a4  2-state regime-switching GRW                    (DESIGN Arm 4)
#
# It is NOT a betting study and NOT a goals model: nothing here prices a
# fixture from information the market did not already have. What it can do is
# tell the goals models (Gen 1–4, TODO 021) what shape of dynamics the market's
# own ratings have — the compression finding says that is where the gap lives.
#
# FILTRATION / COMPARABILITY CONTRACT
# -----------------------------------
# * Every arm sees the SAME 1,246 observations (623 accepted fixtures × 2).
# * One-step-ahead predictions are PRE-WEEK: an observation in week t is predicted
#   from observations strictly before week t, so two fixtures in the same week
#   never inform each other (cf. the LastHistorical same-day leak).
# * Two protocols for θ (the handful of global hyper-parameters):
#     §10a  θ = posterior median from the full panel — "quasi out-of-sample": the
#           states are filtered honestly, the 3–5 global scales are not;
#     §10b  θ fitted on 24/25 only, predictions scored on 25/26 only — honest.
#
# PERSISTENCE CAVEAT
# ------------------
# Results are CSV + PNG under `results/`; the posterior draws are serialized to
# `results/fits.jls` (git-ignored). Nothing is written to `mcmc_experiments`:
# these are not CountModel fits and have no place in its config registry.
#
# USAGE (mcmc-beast, warm REPL, -t 16):
#     include("current_development/market_inverse_dynamics/r01_market_inverse_runner.jl")
# Set MID_SMOKE=1 in ENV for a 3-minute pass with short chains.
# ==============================================================================

# %%
# ===================================================================
# 1. Packages and implementation
# ===================================================================
using BayesianFootball
using DataFrames, Dates, Statistics, LinearAlgebra, Printf, Random
using CSV, Serialization
using ThreadPinning
ENV["GKSwstype"] = "100"            # headless GR
using Plots

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "l01_market_inverse_loader.jl"))
const MID = MarketInverseDynamics

# %%
# ===================================================================
# 2. Configuration
# ===================================================================
const MID_SMOKE      = get(ENV, "MID_SMOKE", "0") == "1"
const MID_SEASONS    = ["24/25", "25/26"]
const MID_TOURNAMENTS = [56, 57]                 # Scottish League One / League Two
const MID_STEP_DAYS  = 7                         # weekly state grid
const MID_CHAINS     = 4
const MID_WARMUP     = MID_SMOKE ? 150 : 2_000
const MID_SAMPLES    = MID_SMOKE ? 150 : 3_000
const MID_PATHS      = MID_SMOKE ? 20 : 200      # retained FFBS path draws per arm
const MID_PARTICLES  = MID_SMOKE ? 500 : 20_000  # RBPF particles (SV, regime); fewer collapse to ESS ≈ 1
const MID_SEED       = 20260922
const MID_Z_CUT      = 2.5                       # anomaly threshold (DESIGN §5.4)
const MID_RHAT_MAX   = 1.05                      # TODO 023 Stage 2 gates
const MID_ESS_MIN    = 200
const MID_PLOT_TEAMS = ["hamilton-academical", "inverness-caledonian-thistle",
                        "dumbarton", "peterhead"]

const MID_ARMS = [MID.StaticArm(), MID.GRW1(), MID.GRW1Break(), MID.MomentumGRW(),
                  MID.StochVolGRW(), MID.RegimeGRW()]
# Thinning per arm: the Gaussian arms sample θ from its collapsed posterior and mix
# fast; momentum trades σ against σ_v; the SV / regime Gibbs chains carry 2N latent
# volatility paths and are strongly autocorrelated in (γ_h, σ_h) / (Δ, p11). A sweep
# costs ~40 ms, so they buy ESS with length rather than a new sampler.
const MID_THIN = Dict("a0_static" => 1, "a1_grw1" => 1, "a1b_grw1_break" => 1,
                      "a2_momentum" => 4, "a3_stochvol" => 20, "a4_regime" => 20)

# %%
# ===================================================================
# 3. Runtime and output directory
# ===================================================================
const MID_OUT = joinpath(@__DIR__, "results", MID_SMOKE ? "smoke" : "production")
const MID_FIG = joinpath(MID_OUT, "figures")
mkpath(MID_FIG)
println("threads = $(Threads.nthreads()), output → $MID_OUT")

# %%
# ===================================================================
# 4. Data: invert the close, lay it out on a weekly grid
# ===================================================================
ds = Data.load_datastore_cached(Data.ScottishLower())
panel, book, inversion = MID.build_market_panel(ds; seasons = MID_SEASONS,
                                                tournaments = MID_TOURNAMENTS,
                                                step_days = MID_STEP_DAYS)

refusal_reasons = BayesianFootball.Calibration.inversion_refusals(inversion)
println("fixtures: $(nrow(panel.matches) + nrow(panel.refusals)) in panel, " *
        "$(nrow(panel.matches)) accepted, $(nrow(panel.refusals)) refused")
foreach(r -> println("   refused  $(lpad(r[2], 3))  $(r[1])"), refusal_reasons)
println("teams = $(length(panel.teams)), weeks = $(panel.n_weeks), observations = $(MID.n_obs(panel))")
CSV.write(joinpath(MID_OUT, "refusals.csv"), panel.refusals)
CSV.write(joinpath(MID_OUT, "market_targets.csv"), panel.matches)

# %%
# ===================================================================
# 5. Engine gates — exactness on a toy panel (must all pass)
# ===================================================================
gates = MID.mid_gates()
show(stdout, MIME"text/plain"(), gates; allrows = true)
println()
all(gates.pass) || error("engine gates failed — do not trust any fit below")
CSV.write(joinpath(MID_OUT, "engine_gates.csv"), gates)

# %%
# ===================================================================
# 6. Model fitting — every arm, 4 chains in parallel
# ===================================================================
# All arms at once: 6 × 4 chain tasks share the 16 cores, the slow Gibbs arms
# dominating the wall clock either way.
fit_tasks = Dict(MID.arm_name(arm) => Threads.@spawn(
                     MID.fit_arm(arm, panel; n_chains = MID_CHAINS, n_warmup = MID_WARMUP,
                                 n_samples = MID_SAMPLES, n_paths = MID_PATHS, seed = MID_SEED,
                                 thin = MID_THIN[MID.arm_name(arm)], ess_steps = 5))
                 for arm in MID_ARMS)
fits = Dict{String, MID.ArmFit}(a => fetch(t) for (a, t) in fit_tasks)
foreach(a -> @printf("fitted %-16s in %7.1f s\n", a, fits[a].seconds),
        [MID.arm_name(arm) for arm in MID_ARMS])
arm_order = [MID.arm_name(a) for a in MID_ARMS]
serialize(joinpath(MID_OUT, "fits.jls"), (; fits, panel))

# %%
# ===================================================================
# 7. Convergence diagnostics
# ===================================================================
conv = vcat([MID.convergence_table(fits[a]) for a in arm_order]...)
conv.gate_pass = (conv.rhat .<= MID_RHAT_MAX) .& (conv.ess_bulk .>= MID_ESS_MIN) .&
                 (conv.ess_tail .>= MID_ESS_MIN)
show(stdout, MIME"text/plain"(), conv; allrows = true)
println()
CSV.write(joinpath(MID_OUT, "posterior_summary.csv"), conv)
println("convergence gates: $(count(conv.gate_pass)) / $(nrow(conv)) parameters pass " *
        "(R̂ ≤ $MID_RHAT_MAX, bulk & tail ESS ≥ $MID_ESS_MIN)")

# %%
# ===================================================================
# 8. Posterior profiles — persistence, volatility scales, regime durations
# ===================================================================
regime = fits["a4_regime"]
p11 = vec(regime.draws[:, 5, :])
p22 = vec(regime.draws[:, 6, :])
calm_weeks = 1 ./ (1 .- p11)
turb_weeks = 1 ./ (1 .- p22)
occupancy = (1 .- p11) ./ ((1 .- p11) .+ (1 .- p22))       # stationary P(turbulent)
@printf("regime: calm lasts %.1f weeks [%.1f, %.1f], turbulence %.2f weeks [%.2f, %.2f], stationary P(turbulent) = %.3f\n",
        median(calm_weeks), quantile(calm_weeks, 0.05), quantile(calm_weeks, 0.95),
        median(turb_weeks), quantile(turb_weeks, 0.05), quantile(turb_weeks, 0.95),
        median(occupancy))

# Where does the turbulent state sit in the calendar? (posterior mean over teams)
turbulence_by_week = DataFrame(week = 1:panel.n_weeks, week_start = panel.week_start,
                               p_turbulent_att = vec(mean(regime.aux_mean[1:length(panel.teams), :]; dims = 1)),
                               p_turbulent_def = vec(mean(regime.aux_mean[length(panel.teams)+1:end, :]; dims = 1)),
                               n_obs = diff(panel.week_ptr))
CSV.write(joinpath(MID_OUT, "regime_turbulence_by_week.csv"), turbulence_by_week)
sv = fits["a3_stochvol"]
vol_by_week = DataFrame(week = 1:panel.n_weeks, week_start = panel.week_start,
                        sigma_att = vec(mean(exp.(sv.aux_mean[1:length(panel.teams), :]); dims = 1)),
                        sigma_def = vec(mean(exp.(sv.aux_mean[length(panel.teams)+1:end, :]); dims = 1)))
CSV.write(joinpath(MID_OUT, "sv_volatility_by_week.csv"), vol_by_week)

let
    t2 = 2:panel.n_weeks
    plt = plot(panel.week_start[t2], turbulence_by_week.p_turbulent_att[t2]; label = "P(turbulent) attack",
               ylabel = "posterior P(turbulent), team mean", legend = :topleft, lw = 2,
               title = "Where the volatility lives (a4 regime, a3 SV)", size = (1000, 450))
    plot!(plt, panel.week_start[t2], turbulence_by_week.p_turbulent_def[t2]; label = "P(turbulent) defence", lw = 2)
    plot!(twinx(plt), panel.week_start[t2], vol_by_week.sigma_att[t2]; label = "SV σ attack", color = :black,
          ls = :dash, ylabel = "SV weekly σ, team mean", legend = :topright)
    savefig(plt, joinpath(MID_FIG, "volatility_calendar.png"))
end

# %%
# ===================================================================
# 9. EDA — does the GRW1 walk look like a GRW1 walk?
# ===================================================================
# Posterior draws of the weekly innovations, in-season weeks only. Under a correct
# GRW1 the standardised innovations are iid N(0,1): acf1 ≈ 0 (no momentum),
# excess kurtosis ≈ 0 (no jumps / regimes), acf1 of squares ≈ 0 (no clustering).
innov = MID.innovation_diagnostics(fits["a1_grw1"], panel)
innov_summary = combine(groupby(innov, :component),
                        :acf1 => median => :acf1, :acf1_sq => median => :acf1_sq,
                        :excess_kurtosis => median => :excess_kurtosis,
                        :acf1 => (x -> quantile(x, 0.05)) => :acf1_q05,
                        :acf1 => (x -> quantile(x, 0.95)) => :acf1_q95)
show(stdout, MIME"text/plain"(), innov_summary)
println()
CSV.write(joinpath(MID_OUT, "grw1_innovation_diagnostics.csv"), innov_summary)

# %%
# ===================================================================
# 10. One-step-ahead evaluation
# ===================================================================
# 10a — full-panel θ (plug-in posterior median); states filtered honestly.
function mid_theta_and_P(f::MID.ArmFit)
    θ = MID.median_theta(f)
    P = f.arm isa MID.RegimeGRW ? (median(vec(f.draws[:, 5, :])), median(vec(f.draws[:, 6, :]))) :
                                  (0.9, 0.8)
    return θ, P
end

season_open = falses(MID.n_obs(panel))            # first 3 weeks of each season
for s in unique(panel.obs_season)
    w0 = minimum(panel.obs_week[panel.obs_season .== s])
    season_open .|= (panel.obs_season .== s) .& (panel.obs_week .< w0 + 3)
end
first_season_start = minimum(panel.obs_week)
warm = panel.obs_week .>= first_season_start + 3   # drop the cold-start weeks from scoring

preds = Dict{String, Any}()
metric_rows = NamedTuple[]
for a in arm_order
    θ, P = mid_theta_and_P(fits[a])
    pr = MID.onestep_predictions(fits[a].arm, panel, θ; n_particles = MID_PARTICLES,
                                 seed = MID_SEED, P = P)
    preds[a] = pr
    fitted = MID.fitted_logrates(panel, fits[a])
    ins = panel.obs_y .- fitted
    for (label, mask) in (("all (warm)", warm), ("in-season", warm .& .!season_open),
                          ("season-open", warm .& season_open))
        pm = MID.prediction_metrics(panel, pr; mask = mask)
        push!(metric_rows, (arm = a, protocol = "10a full-panel θ", subset = label, pm...,
                            insample_rmse = sqrt(mean(ins[mask] .^ 2)),
                            insample_mae = mean(abs.(ins[mask])),
                            loglik_total = pr.loglik,
                            min_ess = isempty(pr.ess) ? NaN : minimum(pr.ess[2:end]),
                            median_ess = isempty(pr.ess) ? NaN : median(pr.ess[2:end])))
    end
end

# 10b — honest: θ fitted on 24/25 observations only, scored on 25/26 only.
first_season = panel.obs_season .== MID_SEASONS[1]
panel_s1 = MID.restrict_panel(panel, first_season)
s1_tasks = Dict(MID.arm_name(arm) => Threads.@spawn(
                    MID.fit_arm(arm, panel_s1; n_chains = MID_CHAINS,
                                n_warmup = cld(MID_WARMUP, 2), n_samples = cld(MID_SAMPLES, 2),
                                n_paths = 4, seed = MID_SEED + 1,
                                thin = MID_THIN[MID.arm_name(arm)], ess_steps = 5))
                for arm in MID_ARMS)
fits_s1 = Dict{String, MID.ArmFit}(a => fetch(t) for (a, t) in s1_tasks)
second = .!first_season
for a in arm_order
    θ, P = mid_theta_and_P(fits_s1[a])
    pr = MID.onestep_predictions(fits_s1[a].arm, panel, θ; n_particles = MID_PARTICLES,
                                 seed = MID_SEED, P = P)
    for (label, mask) in (("25/26 all", second), ("25/26 in-season", second .& .!season_open),
                          ("25/26 season-open", second .& season_open))
        pm = MID.prediction_metrics(panel, pr; mask = mask)
        push!(metric_rows, (arm = a, protocol = "10b θ from 24/25", subset = label, pm...,
                            insample_rmse = NaN, insample_mae = NaN, loglik_total = pr.loglik,
                            min_ess = isempty(pr.ess) ? NaN : minimum(pr.ess[2:end]),
                            median_ess = isempty(pr.ess) ? NaN : median(pr.ess[2:end])))
    end
end
metrics = DataFrame(metric_rows)
show(stdout, MIME"text/plain"(), select(metrics, :arm, :protocol, :subset, :n, :rmse, :mae,
                                        :mean_logpd, :cover90, :insample_rmse, :min_ess, :median_ess);
     allrows = true)
println()
CSV.write(joinpath(MID_OUT, "prediction_metrics.csv"), metrics)
conv_s1 = vcat([MID.convergence_table(fits_s1[a]) for a in arm_order]...)
CSV.write(joinpath(MID_OUT, "posterior_summary_s1.csv"), conv_s1)

# %%
# ===================================================================
# 11. Trajectories — what each arm thinks a team is, week by week
# ===================================================================
paths = vcat([MID.team_paths(fits[a], panel) for a in arm_order if a != "a0_static"]...)
CSV.write(joinpath(MID_OUT, "team_paths.csv"), paths)
active = MID.active_weeks(panel)
for team in MID_PLOT_TEAMS
    ti = findfirst(==(team), panel.teams)
    ti === nothing && (println("skip plot: $team not in panel"); continue)
    plt = plot(layout = (2, 1), size = (1100, 750), legend = :outerright)
    for (k, comp) in enumerate(("att", "def"))
        for a in ("a1_grw1", "a1b_grw1_break", "a2_momentum", "a3_stochvol", "a4_regime")
            d = filter(r -> r.arm == a && r.team == team, paths)
            y = d[!, Symbol(comp, "_mean")]
            y = [active[ti, w] ? y[i] : NaN for (i, w) in enumerate(d.week)]
            plot!(plt[k], d.week_start, y; label = a, lw = a == "a1_grw1" ? 2.5 : 1.5)
            if a == "a1_grw1"
                lo = [active[ti, w] ? v : NaN for (v, w) in zip(d[!, Symbol(comp, "_lo")], d.week)]
                hi = [active[ti, w] ? v : NaN for (v, w) in zip(d[!, Symbol(comp, "_hi")], d.week)]
                plot!(plt[k], d.week_start, lo; fillrange = hi, fillalpha = 0.15, lw = 0,
                      label = "a1 90%", color = :grey)
            end
        end
        title!(plt[k], "$team — $(comp == "att" ? "attack α" : "defence β (+ = concedes more)")")
    end
    savefig(plt, joinpath(MID_FIG, "trajectory_$(team).png"))
end

# Market supremacy vs the a1 decomposition, as a sanity plot: fitted vs observed.
let
    fitted = MID.fitted_logrates(panel, fits["a1_grw1"])
    plt = scatter(panel.obs_y, fitted; ms = 2, alpha = 0.5, label = "",
                  xlabel = "log λ_mkt (inverted close)", ylabel = "a1 smoothed fit",
                  title = "In-sample fit, a1 GRW1", size = (600, 600))
    plot!(plt, [-1.2, 1.5], [-1.2, 1.5]; label = "y = x", color = :black)
    savefig(plt, joinpath(MID_FIG, "insample_fit_a1.png"))
end

# %%
# ===================================================================
# 12. Anomaly / shock detection
# ===================================================================
# a1 one-step surprises and a1 smoothed residuals, both at |z| ≥ 2.5.
θ1, _ = mid_theta_and_P(fits["a1_grw1"])
anomalies = MID.anomaly_catalog(panel, preds["a1_grw1"], MID.fitted_logrates(panel, fits["a1_grw1"]),
                                exp(θ1[1]); z_cut = MID_Z_CUT)
CSV.write(joinpath(MID_OUT, "anomaly_catalog.csv"), anomalies)
println("anomalies: $(nrow(anomalies)) observations " *
        "($(count(abs.(anomalies.z_surprise) .>= MID_Z_CUT)) one-step surprises, " *
        "$(count(abs.(anomalies.z_idio) .>= MID_Z_CUT)) idiosyncratic)")

# Team-weeks the regime arm calls turbulent with posterior probability ≥ 0.5.
N = length(panel.teams)
shock_rows = NamedTuple[]
for i in 1:N, (comp, off) in (("att", 0), ("def", N)), t in 2:panel.n_weeks
    pt = regime.aux_mean[off+i, t]
    pt >= 0.5 && active[i, t] && push!(shock_rows, (team = panel.teams[i], component = comp, week = t,
                                                   week_start = panel.week_start[t], p_turbulent = pt))
end
regime_shocks = sort!(DataFrame(shock_rows), [:week, :team])
CSV.write(joinpath(MID_OUT, "regime_shock_weeks.csv"), regime_shocks)
println("regime: $(nrow(regime_shocks)) in-season team-weeks with P(turbulent) ≥ 0.5")

# %%
# ===================================================================
# 13. Final report
# ===================================================================
headline = filter(r -> r.subset in ("all (warm)", "25/26 all"), metrics)
println("\n=== HEADLINE ===")
show(stdout, MIME"text/plain"(), select(headline, :arm, :protocol, :rmse, :mae, :mean_logpd,
                                        :cover90, :loglik_total); allrows = true)
println()
println("MID_DONE")

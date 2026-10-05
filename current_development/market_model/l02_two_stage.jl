module TwoStageMarket

import BayesianFootball
import CSV
import DataFrames
import Dates
import Distributions
import LinearAlgebra
import MCMCChains
import Random
import Statistics
import SpecialFunctions
import Turing
import ReverseDiff
import ForwardDiff
import DynamicPPL
import LogDensityProblems
import Serialization

if !isdefined(parentmodule(@__MODULE__), :MarketModel)
    Base.include(parentmodule(@__MODULE__), joinpath(@__DIR__, "l01_market_model.jl"))
end
const MM = parentmodule(@__MODULE__).MarketModel
const MID = MM.MID
const DF = DataFrames
const LA = LinearAlgebra
const ST = Statistics
const DS = Distributions
const EXCLUSION = "1X2-only book (Phase B exclusion: totals biased ~0.10 low)"

"""
Build the 517-fixture, 1,034-observation Phase B panel from Phase A's pinned CSV.
Only accepted books with ≥5 selections survive; no inversion is changed.
The bridge's legacy `sse` column holds KL, NOT squared error.
"""
function phase_b_panel(ds; config=MM.scottish_lower_2425_2526(),
                       rates_path=joinpath(@__DIR__, "results", "A", "rates.csv"))
    rates = CSV.read(rates_path, DF.DataFrame)
    rates = DF.filter(r -> r.season in config.seasons && r.tournament in config.tournaments, rates)
    MM.check_expected(config, :panel, DF.nrow(rates))
    MM.check_expected(config, :accepted, count(rates.accepted))
    excluded = rates.accepted .& (rates.n_selections .< config.min_selections_ladder)
    MM.check_expected(config, :one_x2_only, count(excluded))
    # The bridge's legacy sse column holds KL here, not squared error.
    frame = DF.DataFrame(match_id=Int.(rates.match_id), accepted=rates.accepted .& .!excluded,
                         lambda_mkt_h=rates.lambda_h, lambda_mkt_a=rates.lambda_a,
                         sse=rates.kl, n_targets=rates.n_selections,
                         reason=[excluded[i] ? EXCLUSION : coalesce(rates.reason[i], "") for i in 1:DF.nrow(rates)])
    panel = MID.build_market_panel(ds, frame; config, step_days=7)
    MM.check_expected(config, :ladder, MID.n_fixtures(panel))
    MM.check_expected(config, :obs, MID.n_obs(panel))
    return (; panel, rates, frame, exclusions=rates[excluded, :])
end

"""
R0–R4 on TODO 023's exact conditional Kalman engine.
R0 pins all attack/defence states to zero. R1 has zero in-season steps and one
learned break jump. R2 is weekly GRW. R3 adds static home-only δ_i states.
R4 additionally has shared fixture ω_m ~ Gamma(ν/2, scale 2/ν).

Priors: μ ~ Normal(log(1.35),0.5); γ ~ Normal(0.15,0.25);
initial unconstrained attack/defence ~ Normal(0,0.5), zero-centred in observation;
σ_obs ~ HalfNormal(0.20); weekly σ_att,σ_def ~ HalfNormal(0.10);
σ_break ~ HalfNormal(0.30); σ_γ ~ HalfNormal(0.10);
δ_i | σ_γ ~ Normal(0,σ_γ), static, home observations only (not centred);
ν ~ Gamma(2,scale 10), as TODO 023 Phase 2. All scales use log transforms with
Jacobians. `ha_sd` is conditional on θ, never a separate estimated parameter.
"""
struct Rung <: MID.AbstractArm
    level::Int
    ha_sd::Float64
    function Rung(level::Int, ha_sd::Float64=0.0)
        0 <= level <= 4 || error("rung must be R0–R4")
        new(level, ha_sd)
    end
end
MID.arm_name(a::Rung) = "R$(a.level)"
MID.n_features(a::Rung) = 0 # HA states exist only in the conditional HomeRung layout

"Conditional layout with one static HA coefficient per team."
struct HomeRung <: MID.AbstractArm
    rung::Rung
    N::Int
end
MID.n_features(a::HomeRung) = a.N
MID.arm_name(a::HomeRung) = MID.arm_name(a.rung)
MID.param_names(a::Rung) = a.level == 0 ? ["sigma_obs"] :
    a.level == 1 ? ["sigma_obs", "sigma_break"] :
    a.level == 2 ? ["sigma_obs", "sigma_att", "sigma_def"] :
    a.level == 3 ? ["sigma_obs", "sigma_att", "sigma_def", "sigma_gamma"] :
                  ["sigma_obs", "sigma_att", "sigma_def", "sigma_gamma", "nu"]
MID.n_features(a::Rung, N::Int) = a.level >= 3 ? N : 0
function MID.initial_state(a::Rung, N)
    a.level >= 3 && error("use conditional HomeRung layout")
    m, P = MID.initial_state(MID.StaticArm(), N)
    if a.level == 0
        P[3:end, 3:end] .= 0.0
    end
    return m, P
end
function MID.initial_state(a::HomeRung, N)
    m0, P0 = MID.initial_state(MID.GRW1(), N)
    m = vcat(m0, zeros(N))
    P = zeros(length(m), length(m))
    P[1:length(m0), 1:length(m0)] .= P0
    for i in (length(m0)+1):length(m)
        P[i, i] = a.rung.ha_sd^2
    end
    return m, P
end

"Home-only indicator design; unlike TODO 023's antisymmetric with_features bridge."
function home_panel(p::MID.MarketPanel)
    N = MID.n_teams(p)
    X = zeros(MID.n_obs(p), N)
    for j in eachindex(p.obs_y)
        X[j, p.obs_att[j]] = p.obs_home[j]
    end
    return MID.MarketPanel(p.matches, p.refusals, p.teams, p.n_weeks, p.week_start,
        p.obs_week, p.obs_home, p.obs_att, p.obs_def, p.obs_y, p.obs_match,
        p.obs_season, p.week_ptr, X, ["ha_" * t for t in p.teams])
end
layout(a::Rung, p, θ) = a.level >= 3 ? HomeRung(Rung(a.level, exp(θ[4])), MID.n_teams(p)) : a
"First dated observed week of each configured later season; no calendar literal."
function season_break_weeks(p, config=nothing)
    seasons = config === nothing ? sort(unique(p.obs_season); by=s->minimum(p.obs_week[p.obs_season .== s])) : config.seasons
    present = [s for s in seasons if s in p.obs_season]
    return [minimum(p.obs_week[p.obs_season .== s]) for s in present[2:end]]
end

function rung_schedule(a::Rung, p, θ; config=nothing)
    N = MID.n_teams(p)
    T = p.n_weeks
    if a.level == 0
        return MID.schedule(MID.StaticArm(), θ, N, T)
    elseif a.level == 1
        return MID.schedule(MID.GRW1Break(), [θ[1], -Inf, -Inf, θ[2]], N, T, season_break_weeks(p, config))
    end
    return MID.schedule(MID.GRW1(), θ, N, T)
end
function rung_filter(a::Rung, p, θ; store=false, predict=false, omega=Float64[])
    return MID.run_filter(layout(a, p, θ), p, rung_schedule(a, p, θ);
                          σ_obs=exp(θ[1]), store, predict, ω=omega)
end
function MID.log_prior(a::Rung, θ)
    lp = MID.halfnormal_lp(exp(θ[1]), 0.20) + θ[1]
    if a.level == 1
        lp += MID.halfnormal_lp(exp(θ[2]), 0.30) + θ[2]
    elseif a.level >= 2
        lp += MID.halfnormal_lp(exp(θ[2]), 0.10) + θ[2]
        lp += MID.halfnormal_lp(exp(θ[3]), 0.10) + θ[3]
    end
    if a.level >= 3
        lp += MID.halfnormal_lp(exp(θ[4]), 0.10) + θ[4]
    end
    if a.level == 4
        lp += 2θ[5] - 0.1exp(θ[5])
    end
    return lp
end
MID.init_centre(a::Rung) = a.level == 0 ? [log(0.2)] :
    a.level == 1 ? log.([0.15, 0.1]) : a.level == 2 ? log.([0.08, 0.03, 0.03]) :
    a.level == 3 ? log.([0.08, 0.03, 0.03, 0.05]) : log.([0.05, 0.03, 0.03, 0.05, 5.0])
function rung_logpost(a, p, θ, omega)
    maximum(abs, θ) > MID.THETA_BOUND && return -Inf
    lp = rung_filter(a, p, θ; omega=a.level == 4 ? omega : Float64[]).loglik + MID.log_prior(a, θ)
    return a.level == 4 ? lp + MID.log_mixing_density(omega, exp(θ[5])) : lp
end

"Exact deterministic FFBS step when the entire R0/R1 state has zero process noise."
function MID.backward_step!(xt, xnext, a::Rung, N, sch, filt, F, t, rng)
    if all(iszero, sch.D[:, t+1])
        copyto!(xt, xnext)
    else
        mean, cov = MID.backward_moments(xnext, a, N, sch, filt, F, t)
        MID.draw_psd!(xt, rng, mean, cov)
        xt[1:2] .= xnext[1:2] # league intercept / HA are static even at the break
    end
    return nothing
end

"Partially collapsed Gibbs: collapsed θ | ω, FFBS paths | θ,ω, conjugate ω | paths,θ."
function rung_chain(a, p, chain, seed, warmup, samples, thin)
    rng = Random.Xoshiro(seed + 1000chain)
    θ = MID.init_theta(a, rng)
    omega = ones(MID.n_fixtures(p))
    U = zeros(samples, length(θ))
    hist = zeros(warmup, length(θ))
    widths = ones(length(θ))
    omega_sum = zeros(length(omega))
    for it in 1:(warmup+samples*thin)
        target = z -> rung_logpost(a, p, z, omega)
        θ, _ = MID.slice_sweep(target, θ, target(θ), widths, rng)
        if it <= warmup
            hist[it, :] .= θ
            if it >= 50 && it % 25 == 0
                widths .= clamp.(3 .* vec(ST.std(hist[max(1,it-199):it, :]; dims=1)), 0.02, 3.0)
            end
        end
        if a.level == 4
            filt = rung_filter(a, p, θ; store=true, omega)
            X = MID.ffbs(layout(a, p, θ), p, rung_schedule(a, p, θ), filt, rng)
            residual = p.obs_y .- MID.fitted_logrates(p, X; K=MID.n_teams(p))
            for m in eachindex(omega)
                ss = residual[2m-1]^2 + residual[2m]^2
                omega[m] = rand(rng, DS.Gamma((exp(θ[5])+2)/2, 2/(exp(θ[5])+ss/exp(2θ[1]))))
            end
        end
        if it > warmup && (it-warmup) % thin == 0
            s = (it-warmup) ÷ thin
            U[s, :] .= θ
            omega_sum .+= omega
        end
    end
    return (; U, omega=omega_sum ./ samples)
end
function fit_rung(a::Rung, p; warmup=a.level == 4 ? 1000 : 2000,
                  samples=a.level == 4 ? 2000 : 3000, thin=a.level == 4 ? 4 : 1,
                  chains=4, seed=3902)
    p = a.level >= 3 ? home_panel(p) : p
    start = time()
    tasks = [Threads.@spawn rung_chain(a, p, c, seed, warmup, samples, thin) for c in 1:chains]
    out = fetch.(tasks)
    U = cat([o.U for o in out]...; dims=3)
    D = exp.(U)
    omega = ST.mean(o.omega for o in out)
    fit = MID.ArmFit(a, MID.param_names(a), D, U, Matrix{Float64}[], Vector{Float64}[],
                    reshape(omega, 1, :), time()-start)
    return fit
end

"R0/R1/R3 batch joint-Gaussian gates and R1/R3 reductions, no fitted parameters."
function engine_gates()
    rows = DF.DataFrame(MID.mid_gates())
    rng = Random.Xoshiro(3902)
    p0 = MID.toy_panel(rng)
    seasons = [w < 5 ? "toy-first" : "toy-second" for w in p0.obs_week]
    p = MID.MarketPanel(p0.matches, p0.refusals, p0.teams, p0.n_weeks, p0.week_start,
        p0.obs_week, p0.obs_home, p0.obs_att, p0.obs_def, p0.obs_y, p0.obs_match,
        seasons, p0.week_ptr, p0.obs_X, p0.feature_names)
    for (a, θ) in ((Rung(0), log.([0.07])), (Rung(1), log.([0.07, 0.1])),
                   (Rung(3), log.([0.07, 0.05, 0.03, 0.1])))
        pp = a.level >= 3 ? home_panel(p) : p
        sch = rung_schedule(a, pp, θ)
        arm = layout(a, pp, θ)
        f = rung_filter(a, pp, θ; store=true)
        batch = MID.batch_posterior(arm, pp, sch, exp(θ[1]))
        smooth = MID.smoothed_mean(arm, pp, sch, f)
        push!(rows, ("B loglik $(MID.arm_name(a))", abs(f.loglik-batch.loglik), 1e-9, false))
        push!(rows, ("B RTS $(MID.arm_name(a))", maximum(abs.(vec(smooth).-batch.post_mean)), 1e-8, false))
    end
    θ = log.([0.07, 0.05, 0.03])
    push!(rows, ("B R1 zero-break reduction", abs(rung_filter(Rung(1), p, [θ[1], -60.0]).loglik-
        MID.kalman_loglik(MID.StaticArm(), p, θ[1:1])), 1e-9, false))
    push!(rows, ("B R3 zero-HA reduction", abs(rung_filter(Rung(3), home_panel(p), vcat(θ,-60.0)).loglik-
        MID.kalman_loglik(MID.GRW1(), p, θ)), 1e-9, false))
    rows.pass = rows.value .<= rows.tol
    return rows
end

"Poisson log-link IRLS, Fisher information SE (no dispersion rescaling)."
function poisson_glm(y, x; tol=1e-10, maxiter=100)
    X = hcat(ones(length(x)), x)
    b = [log(max(ST.mean(y), 0.01)), 0.0]
    for it in 1:maxiter
        eta = X*b
        mu = exp.(eta)
        info = X' * (X .* mu)
        step = info \ (X' * (y .- mu))
        b .+= step
        if maximum(abs, step) < tol
            se = sqrt.(LA.diag(inv(X' * (X .* exp.(X*b)))))
            return (; coef=b, se, fitted=exp.(X*b), iterations=it)
        end
    end
    error("Poisson IRLS failed after $maxiter iterations")
end

"Seeded fixture-level fold assignment; both consecutive observation rows use the same fold."
function fixture_folds(p; k=10, seed=3903)
    permutation = Random.randperm(Random.Xoshiro(seed), MID.n_fixtures(p))
    folds = zeros(Int, length(permutation))
    for (i, m) in enumerate(permutation)
        folds[m] = mod1(i, k)
    end
    return folds
end

"Train both full-panel and honest 24/25-only θ fits at the prescribed budgets."
function train_ladder(panel, out; seed=3902, config=MM.scottish_lower_2425_2526())
    convergence = DF.DataFrame[]
    fits = Dict{Tuple{Int,String},MID.ArmFit}()
    sha = strip(read(`git rev-parse HEAD`,String))
    for protocol in ("10a","10b"), level in 0:4
        p = protocol == "10a" ? panel : MID.restrict_panel(panel,in.(panel.obs_season,Ref(config.honest_train)))
        fit = fit_rung(Rung(level),p; seed=seed+100level+(protocol == "10b" ? 1 : 0))
        fits[(level,protocol)] = fit
        Serialization.serialize(joinpath(out,"R$(level)_$(protocol).jls"),fit)
        diag = MID.convergence_table(fit)
        diag.protocol .= protocol
        diag.sha .= sha
        diag.gate_pass = (diag.rhat .<= 1.05) .& (diag.ess_bulk .>= 200) .& (diag.ess_tail .>= 200)
        push!(convergence,diag)
        CSV.write(joinpath(out,"convergence.csv"),vcat(convergence...))
        open(joinpath(out,"RUN_PROVENANCE.md"),"a") do io
            println(io,"- $(Dates.now(Dates.UTC)): R$level $protocol; SHA `$sha`; $(fit.seconds) seconds; 4 chains; warmup=$(level == 4 ? 1000 : 2000), retained=$(level == 4 ? 2000 : 3000), thin=$(level == 4 ? 4 : 1).")
        end
        println("R$level $protocol: $(fit.seconds) seconds; gates $(count(diag.gate_pass))/$(DF.nrow(diag))")
        flush(stdout)
        all(diag.gate_pass) || error("R$level $protocol convergence fails; budget may be raised only with recorded evidence")
    end
    return fits
end

include("l02_two_stage_predictions.jl")
include("l02_two_stage_families.jl")
include("l02_two_stage_eda.jl")

end # module

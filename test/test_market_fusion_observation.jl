using Test, BayesianFootball, CSV, DataFrames, Distributions, DynamicPPL
using ForwardDiff, LinearAlgebra, LogDensityProblems, MCMCChains, Random, ReverseDiff, Serialization, SpecialFunctions, Statistics

const MFO_PG = BayesianFootball.Models.PreGame
const MFO_API = MFO_PG.Builder
const MFO_W2 = joinpath(@__DIR__, "..", "experiments", "scotland", "06_qs_joint_and_market_observation")
isdefined(Main, :Wave2MarketTable) || include(joinpath(MFO_W2, "l01_market_table.jl"))
isdefined(Main, :Wave2LikelihoodAudit) || include(joinpath(MFO_W2, "l02_likelihood_audit.jl"))
isdefined(Main, :MarketFusionFDAudit) || include(joinpath(@__DIR__, "helpers", "market_fusion_fd_audit.jl"))

const MFO_OUTPUT = get(ENV, "QSX3_TEST_OUTPUT", "/root/BF_runs/qs_experiment_w3_out/phase1")

# The wave-2 joint configuration (04_.../candidates.jl l.17–25, no league deltas).
mfo_joint() = JointGammaPoissonObservation(
    feature = Features.MatchProxyXGFeature(k = 25.0, fallback = :none),
    shape_prior = truncated(Normal(4.0, 1.5), 0.5, Inf),
    log_kappa_prior = Normal(0.0, 0.2), kappa = SharedKappa())
mfo_fusion(feature; bias = true, kw...) = bias ?
    JointMarketFusionObservation(; joint = mfo_joint(), market = MarketRateObservation(feature = feature), kw...) :
    JointMarketFusionObservation(; joint = mfo_joint(), market = MarketRateObservation(feature = feature),
        kappa_D_prior = nothing, delta_D_prior = nothing, delta_M_prior = nothing, kw...)
mfo_model(dynamics, observation; guard = ClampGuard()) =
    CountModelBuilder(:market_fusion_test) |> add(GlobalInterception()) |> add(dynamics) |>
    add(GlobalHomeAdvantage()) |> add(observation) |> add(guard) |> build
mfo_weak_qs() = QualityStyleGRW(r₀ = LogNormal(0, 1), rₛ = LogNormal(0, 1), rₖ = LogNormal(0, 1))

function mfo_density(tm; linked = false, seed = 20261110, selector = DynamicPPL.getlogjoint_internal)
    Random.seed!(seed)
    vi = DynamicPPL.VarInfo(tm)
    linked && (vi = DynamicPPL.link!!(vi, tm))
    density = DynamicPPL.LogDensityFunction(tm, selector, vi)
    return (; f = x -> LogDensityProblems.logdensity(density, x), theta = copy(vi[:]), vi)
end
"The production seam: `_observe` returns ll; only the parent accumulates it."
mfo_loglik(sub, theta) = mfo_density(Wave2LikelihoodAudit.accumulated(sub);
    selector = DynamicPPL.getloglikelihood).f(theta)

"A real late fold, so every scale, mask and full book participates in AD and recovery."
function mfo_real_features(model)
    ds = Wave2MarketTable.QSMarketArm.checked_datastore()
    scope = Wave2MarketTable.QSMarketArm.goal_scope()
    scoped = Data.apply_scope(ds, scope)
    splitter = Data.ScopedWalkForwardCV(scope)
    boundaries = Data.create_id_boundaries(scoped, splitter)
    @test length(boundaries) == 40
    fs = first(Features.create_features(boundaries[40:40], scoped, model, splitter))
    return fs isa Tuple ? first(fs) : fs
end

"Hand-written per-match fusion density: Poisson goals and Gamma xG (Distributions) + biased Normal."
function mfo_hand(eh, ea, yh, ya, wts, jd, md, p)
    zh, za = eh .+ p.log_κ, ea .+ p.log_κ
    total = 0.0
    for i in eachindex(eh)
        total += wts[i] * (logpdf(Poisson(exp(zh[i])), yh[i]) + logpdf(Poisson(exp(za[i])), ya[i]))
        total += jd.mask_weights[i] * (logpdf(Gamma(p.ν, exp(eh[i]) / p.ν), jd.pxg_h[i]) +
                                       logpdf(Gamma(p.ν, exp(ea[i]) / p.ν), jd.pxg_a[i]))
        D, M = zh[i] - za[i], (zh[i] + za[i]) / 2
        mh = p.δ_M + p.δ_D / 2 + M + p.κ_D * D / 2
        ma = p.δ_M - p.δ_D / 2 + M - p.κ_D * D / 2
        total += md.mask_weights[i] * (logpdf(Normal(mh, p.σ_obs), md.log_h[i]) +
                                       logpdf(Normal(ma, p.σ_obs), md.log_a[i]))
    end
    return total
end
"Hand-written biased market arm alone, per match."
mfo_hand_market(zh, za, md, p) = sum(md.mask_weights[i] *
    (logpdf(Normal(p.δ_M + p.δ_D / 2 + (zh[i] + za[i]) / 2 + p.κ_D * (zh[i] - za[i]) / 2, p.σ_obs), md.log_h[i]) +
     logpdf(Normal(p.δ_M - p.δ_D / 2 + (zh[i] + za[i]) / 2 - p.κ_D * (zh[i] - za[i]) / 2, p.σ_obs), md.log_a[i]))
    for i in eachindex(zh))

"Independent synthetic walk using the fold's accumulator geometry; no engine likelihood calls."
function mfo_walk(scales, n, design, rng)
    initial = scales[1] .* randn(rng, n, 1) * design.initial_accumulator
    season = scales[2] .* randn(rng, n, size(design.season_accumulator, 1)) * design.season_accumulator
    micro = scales[3] .* randn(rng, n, size(design.target_accumulator, 1)) * design.target_accumulator
    raw = initial + season + micro
    return raw .- mean(raw; dims = 1)
end

"Goals, proxy xG and market rates from one latent, with the pre-declared truths."
function mfo_synthetic(fs, dynamics, bias::Bool, rng)
    n = fs.data[:n_teams]
    design = MFO_API.dynamics_design(dynamics, fs, length(fs.data[:flat_home_ids]))
    truth = Dict{String,Float64}("obs.ν" => 4.0, "obs.log_κ" => 0.05, "obs.σ_obs" => 0.07)
    if dynamics isa QualityStyleGRW
        tau, ratio = [0.16, 0.085, 0.027], [0.4, 0.4, 0.3]
        q = mfo_walk(tau ./ sqrt.(1 .+ ratio.^2), n, design, rng)
        s = mfo_walk(ratio .* tau ./ sqrt.(1 .+ ratio.^2), n, design, rng)
        alpha, beta = q + s, s - q
        for (j, suffix) in enumerate(("₀", "ₛ", "ₖ"))
            truth["dyn.τ" * suffix], truth["dyn.r" * suffix] = tau[j], ratio[j]
        end
    else
        a, b = [0.12, 0.06, 0.03], [0.20, 0.11, 0.024]
        alpha, beta = mfo_walk(a, n, design, rng), mfo_walk(b, n, design, rng)
        for (j, suffix) in enumerate(("₀", "ₛ", "ₖ"))
            truth["dyn.α.σ" * suffix], truth["dyn.β.σ" * suffix] = a[j], b[j]
        end
    end
    κ_D, δ_D, δ_M = bias ? (0.85, 0.03, -0.05) : (1.0, 0.0, 0.0)
    if bias
        truth["obs.κ_D"], truth["obs.δ_D"], truth["obs.δ_M"] = κ_D, δ_D, δ_M
    end
    h, a = design.home_state_indices, design.away_state_indices
    eta_h = log(1.2) .+ 0.15 .+ alpha[h] .+ beta[a]
    eta_a = log(1.2) .+ alpha[a] .+ beta[h]
    zeta_h, zeta_a = eta_h .+ truth["obs.log_κ"], eta_a .+ truth["obs.log_κ"]
    ν, σ = truth["obs.ν"], truth["obs.σ_obs"]
    data = copy(fs.data)
    data[:flat_home_goals] = [rand(rng, Poisson(exp(z))) for z in zeta_h]
    data[:flat_away_goals] = [rand(rng, Poisson(exp(z))) for z in zeta_a]
    data[:flat_pxg_home] = [rand(rng, Gamma(ν, exp(e) / ν)) for e in eta_h]
    data[:flat_pxg_away] = [rand(rng, Gamma(ν, exp(e) / ν)) for e in eta_a]
    D, M = zeta_h .- zeta_a, (zeta_h .+ zeta_a) ./ 2
    data[:flat_market_log_h] = (δ_M + δ_D / 2) .+ M .+ κ_D .* D ./ 2 .+ σ .* randn(rng, length(h))
    data[:flat_market_log_a] = (δ_M - δ_D / 2) .+ M .- κ_D .* D ./ 2 .+ σ .* randn(rng, length(a))
    # Real pxG availability and full-book patterns are kept; uncovered synthetic values are masked.
    return FeatureSet(data), truth
end

@testset "JointMarketFusionObservation" begin
    table = DataFrame(match_id = [2, 1, 3], log_lambda_h = [0.2, 0.3, NaN],
        log_lambda_a = [-0.1, 0.1, NaN], full_book = [true, true, false])
    tiny_feature = Features.MarketRatesFeature(table)
    # Hand designs: 4 matches, market weights [1, 0.5, 0, 0], pxG weights [1, 0, 0.5, 1].
    eh, ea = [0.4, 0.1, 0.8, -0.4], [0.0, -0.2, -0.3, 0.7]
    yh, ya = [1, 0, 3, 2], [0, 2, 1, 1]
    wts = [1.0, 0.5, 1.0, 1.0]
    lfh, lfa = SpecialFunctions.loggamma.(yh .+ 1.0), SpecialFunctions.loggamma.(ya .+ 1.0)
    pxg_h, pxg_a = [1.3, 0.7, 2.1, 0.9], [0.6, 1.1, 0.4, 1.8]
    jd = MFO_API.JointGammaPoissonDesign(pxg_h, pxg_a, log.(pxg_h), log.(pxg_a),
        [1.0, 0.0, 0.5, 1.0], [1, 2, 1, 2], [2, 1, 2, 1], 3, ones(Int, 4), ones(Int, 4), ones(Int, 4), 1, 1)
    md = MFO_API.MarketRateDesign([0.3, 0.2, 0.0, 0.0], [0.1, -0.1, 0.0, 0.0], [1.0, 0.5, 0.0, 0.0], 1.5)
    fusion_design(m) = MFO_API.MarketFusionDesign(jd, m, m.log_h .- m.log_a, m.log_h .+ m.log_a)
    fd = fusion_design(md)
    p = (; ν = 3.5, log_κ = 0.07, σ_obs = 0.12, κ_D = 0.8, δ_D = 0.04, δ_M = -0.06)
    theta_on = [p.ν, p.log_κ, p.σ_obs, p.κ_D, p.δ_D, p.δ_M]
    theta_off = theta_on[1:3]
    biased, unbiased = mfo_fusion(tiny_feature), mfo_fusion(tiny_feature; bias = false)
    observe(o, design) = MFO_API._observe(o, eh, ea, yh, ya, wts, lfh, lfa, 2, 12, design)

    @testset "1. wiring, identity and masking" begin
        model = mfo_model(MultiScaleGRW(), biased)
        @test model isa MFO_API.PoissonCountModel
        @test MFO_API.observation_family(biased) == :poisson
        @test MFO_API.observation_wired(biased)
        @test MFO_API.observation_prefixes(biased) == [:obs]
        @test MFO_API.observation_features(biased) == [biased.joint.feature, tiny_feature]
        @test MFO_API.market_bias_off(unbiased) && !MFO_API.market_bias_off(biased)
        @test unbiased isa MFO_API.UnbiasedMarketFusion && !(biased isa MFO_API.UnbiasedMarketFusion)
        all_sites = Symbol.(["obs.ν", "obs.log_κ", "obs.σ_obs", "obs.κ_D", "obs.δ_D", "obs.δ_M"])
        for (kw, sites) in (((;), all_sites),
                            ((; kappa_D_prior = nothing), all_sites[[1, 2, 3, 5, 6]]),
                            ((; delta_D_prior = nothing), all_sites[[1, 2, 3, 4, 6]]),
                            ((; delta_M_prior = nothing), all_sites[1:5]))
            o = mfo_fusion(tiny_feature; kw...)
            @test MFO_API._sites_observation(o) == sites
            # θ layout: the VarInfo names the submodel declares, in order.
            vi = DynamicPPL.VarInfo(observe(o, fd))
            @test Symbol.(string.(keys(vi))) == sites
            @test all(s -> s in MFO_API.cb_varinfo_sites(mfo_model(MultiScaleGRW(), o)), sites)
        end
        @test MFO_API._sites_observation(unbiased) == all_sites[1:3]
        @test Symbol.(string.(keys(DynamicPPL.VarInfo(observe(unbiased, fd))))) == all_sites[1:3]
        # Validation: the joint's η floor, the market σ rule and a positive κ_D are enforced.
        @test_throws ErrorException mfo_model(MultiScaleGRW(), biased; guard = NoGuard())
        @test_throws ErrorException mfo_model(MultiScaleGRW(), mfo_fusion(tiny_feature; kappa_D_prior = Normal(1.0, 0.2)))
        @test_throws ErrorException mfo_model(MultiScaleGRW(), JointMarketFusionObservation(joint = mfo_joint(),
            market = MarketRateObservation(feature = tiny_feature, sigma_prior = Normal())))
        @test mfo_model(MultiScaleGRW(), unbiased) isa MFO_API.PoissonCountModel
        @test_throws Exception JointMarketFusionObservation(joint = JointGammaPoissonObservation(kappa = HierarchicalKappa()),
            market = MarketRateObservation(feature = tiny_feature))
        # Recipe identity: the market digest and every prior; each bias switch changes it.
        shown = string(model)
        @test occursin(Features.market_rates_digest(tiny_feature), shown)
        @test occursin("kappa_D_prior=" * repr(LogNormal(0.0, 0.2)), shown)
        @test occursin("MatchProxyXGFeature(k=25.0,fallback=none,floor=0.05,dummy=1.0)", shown)
        variants = [mfo_fusion(tiny_feature; bias = false), mfo_fusion(tiny_feature; kappa_D_prior = nothing),
            mfo_fusion(tiny_feature; delta_D_prior = nothing), mfo_fusion(tiny_feature; delta_M_prior = nothing),
            mfo_fusion(tiny_feature; delta_M_prior = Normal(0.0, 0.11)),
            JointMarketFusionObservation(joint = mfo_joint(), market = MarketRateObservation(feature = tiny_feature,
                sigma_prior = truncated(Normal(0.0, 0.21), 0.0, Inf))),
            JointMarketFusionObservation(joint = JointGammaPoissonObservation(feature = Features.MatchProxyXGFeature(k = 20.0),
                kappa = SharedKappa()), market = MarketRateObservation(feature = tiny_feature)),
            mfo_fusion(Features.MarketRatesFeature(DataFrame(match_id = [2, 1, 3], log_lambda_h = [0.2, 0.31, NaN],
                log_lambda_a = [-0.1, 0.1, NaN], full_book = [true, true, false])))]
        shown_variants = [string(mfo_model(MultiScaleGRW(), v)) for v in variants]
        @test allunique(vcat(shown, shown_variants))
        # The UNCHANGED harness comparator on a complete config, without sampling.
        candidate = BayesianFootball.Harness.Candidate(name = "fusion_parity", model = model,
            scope = Wave2MarketTable.QSMarketArm.goal_scope(),
            sampler = Samplers.QueuedNUTSConfig(n_samples = 200, n_warmup = 200, n_chains = 2))
        config = BayesianFootball.Harness.fit_config(candidate; stage = :smoke, experiment = "fusion_parity")
        io = IOBuffer(); Serialization.serialize(io, config); seekstart(io)
        restored = Serialization.deserialize(io)
        @test BayesianFootball.Harness._structural_equal(config, restored)
        @test string(config.model) == string(restored.model)
        changed = deepcopy(restored); changed.model.observation.market.feature.log_lambda_h[1] += 1
        @test !BayesianFootball.Harness._structural_equal(config, changed)
        @test string(config.model) != string(changed.model)

        # Masking: matches without a full book add exactly 0 market log density.
        moved = MFO_API.MarketRateDesign([0.3, 0.2, 50.0, -50.0], [0.1, -0.1, -50.0, 50.0],
            md.mask_weights, md.weight_sum)
        @test mfo_loglik(observe(biased, fusion_design(moved)), theta_on) == mfo_loglik(observe(biased, fd), theta_on)
        @test mfo_loglik(observe(unbiased, fusion_design(moved)), theta_off) == mfo_loglik(observe(unbiased, fd), theta_off)
        no_book = MFO_API.MarketRateDesign(md.log_h, md.log_a, zeros(4), 0.0)
        joint_ll = mfo_loglik(MFO_API._observe(mfo_joint(), eh, ea, yh, ya, wts, lfh, lfa, 2, 12, jd), theta_on[1:2])
        @test mfo_loglik(observe(biased, fusion_design(no_book)), theta_on) == joint_ll
        @test mfo_loglik(observe(unbiased, fusion_design(no_book)), theta_off) == joint_ll
        zh, za = eh .+ p.log_κ, ea .+ p.log_κ
        @test MFO_API._biased_market_rate_ll(zh, za, p.σ_obs, p.κ_D, p.δ_D, p.δ_M, fusion_design(no_book)) == 0.0
        # Masked rows of the market do not change the goals or xG arms either.
        @test MFO_API._biased_market_rate_ll(zh, za, p.σ_obs, p.κ_D, p.δ_D, p.δ_M, fd) ≈
            mfo_hand_market(zh, za, md, p) atol = 1e-12 rtol = 0

        # Prediction uses the TRUE rates of the joint goals arm; bias never reaches the grid.
        nt = (; ν = [4.0, 5.0], κ = [1.1, 0.9], σ_obs = [0.07, 0.08], κ_D = [0.8, 0.9], δ_D = [0.0, 0.1], δ_M = [0.1, 0.0])
        fused_rates = MFO_API._cb_rates(biased, [1.0, 1.2], [0.9, 1.1], nt, 1, 2, 1)
        joint_rates = MFO_API._cb_rates(mfo_joint(), [1.0, 1.2], [0.9, 1.1], nt, 1, 2, 1)
        @test fused_rates == joint_rates
        @test fused_rates.λ_h == [1.1, 0.9] .* [1.0, 1.2]
        # Extraction: draws by site name; an off term reports its fixed no-bias value.
        values = reshape([4.0 0.05 0.07 0.85 0.03 -0.05; 4.5 0.04 0.08 0.9 0.02 -0.04], 2, 6, 1)
        chain = MCMCChains.Chains(values, string.(all_sites))
        extracted = MFO_API._cb_extract_observation(biased, chain, 2)
        @test extracted.ν == [4.0, 4.5] && extracted.κ == exp.([0.05, 0.04])
        @test extracted.σ_obs == [0.07, 0.08] && extracted.κ_D == [0.85, 0.9]
        @test extracted.δ_D == [0.03, 0.02] && extracted.δ_M == [-0.05, -0.04]
        off_chain = MCMCChains.Chains(values[:, 1:3, :], string.(all_sites[1:3]))
        off = MFO_API._cb_extract_observation(unbiased, off_chain, 2)
        @test off.κ_D == [1.0, 1.0] && off.δ_D == [0.0, 0.0] && off.δ_M == [0.0, 0.0]
    end

    @testset "2. likelihood equals the hand-written density at the production seam" begin
        expected = mfo_hand(eh, ea, yh, ya, wts, jd, md, p)
        sub = observe(biased, fd)
        @test mfo_loglik(sub, theta_on) ≈ expected atol = 1e-12 rtol = 0
        priors = logpdf(biased.joint.shape_prior, p.ν) + logpdf(biased.joint.log_kappa_prior, p.log_κ) +
            logpdf(biased.market.sigma_prior, p.σ_obs) + logpdf(biased.kappa_D_prior, p.κ_D) +
            logpdf(biased.delta_D_prior, p.δ_D) + logpdf(biased.delta_M_prior, p.δ_M)
        # Returned-only would drop the likelihood: the seam is what accumulates it.
        @test mfo_density(sub).f(theta_on) ≈ priors atol = 1e-12 rtol = 0
        accumulated = Wave2LikelihoodAudit.accumulated(sub)
        @test mfo_density(accumulated).f(theta_on) ≈ expected + priors atol = 1e-12 rtol = 0
        # Linked space (NUTS's coordinates): the same likelihood at the same constrained point.
        vi = DynamicPPL.unflatten(DynamicPPL.VarInfo(accumulated), theta_on)
        linked_vi = DynamicPPL.link!!(vi, accumulated)
        @test length(linked_vi[:]) == 6
        linked_likelihood = DynamicPPL.LogDensityFunction(accumulated, DynamicPPL.getloglikelihood, linked_vi)
        @test LogDensityProblems.logdensity(linked_likelihood, linked_vi[:]) ≈ expected atol = 1e-12 rtol = 0
        unbiased_expected = mfo_hand(eh, ea, yh, ya, wts, jd, md, (; p..., κ_D = 1.0, δ_D = 0.0, δ_M = 0.0))
        @test mfo_loglik(observe(unbiased, fd), theta_off) ≈ unbiased_expected atol = 1e-12 rtol = 0
        println("FUSION_SEAM expected=", expected, " engine=", mfo_loglik(sub, theta_on),
            " abs_error=", abs(mfo_loglik(sub, theta_on) - expected))
    end

    @testset "3. bias off is exactly the joint plus MarketRateObservation" begin
        joint_ll = mfo_loglik(MFO_API._observe(mfo_joint(), eh, ea, yh, ya, wts, lfh, lfa, 2, 12, jd), theta_off[1:2])
        zh, za = eh .+ p.log_κ, ea .+ p.log_κ
        market_obs = MarketRateObservation(feature = tiny_feature)
        market_ll = mfo_loglik(MFO_API._observe(market_obs, zh, za, yh, ya, wts, lfh, lfa, 2, 12, md), [p.σ_obs])
        @test mfo_loglik(observe(unbiased, fd), theta_off) == joint_ll + market_ll
        @test MFO_API._fusion_market_ll(unbiased, p, zh, za, fd) == MFO_API._market_rate_ll(zh, za, p.σ_obs, md)
        # The biased path at the no-bias VALUES agrees to rounding.
        @test MFO_API._biased_market_rate_ll(zh, za, p.σ_obs, 1.0, 0.0, 0.0, fd) ≈
            MFO_API._market_rate_ll(zh, za, p.σ_obs, md) atol = 1e-12 rtol = 0
        @test mfo_loglik(observe(biased, fd), [theta_off; 1.0; 0.0; 0.0]) ≈ joint_ll + market_ll atol = 1e-12 rtol = 0
    end

    feature = Wave2MarketTable.feature()
    frozen = CSV.read(Wave2MarketTable.TABLE_PATH, DataFrame)
    @test length(feature.match_ids) == 1430 && count(feature.full_book) == 1107
    @test feature.match_ids == frozen.match_id
    arms = (("grw_bias", MultiScaleGRW(), true), ("qs_bias", mfo_weak_qs(), true), ("qs_nobias", mfo_weak_qs(), false))

    @testset "1–3 on the real fold-40 design" begin
        model = mfo_model(mfo_weak_qs(), mfo_fusion(feature))
        fs = mfo_real_features(model)
        n = length(fs.data[:flat_home_ids])
        weights = MFO_API.dynamics_match_weights(model.dynamics, Vector{Float64}(fs.data[:dates]))
        d = MFO_API.observation_design(model.observation, fs, n, weights)
        joint_d = MFO_API.observation_design(mfo_joint(), fs, n, weights)
        market_d = MFO_API.observation_design(MarketRateObservation(feature = feature), fs, n, weights)
        for field in fieldnames(MFO_API.JointGammaPoissonDesign)
            @test getfield(d.joint, field) == getfield(joint_d, field)
        end
        for field in fieldnames(MFO_API.MarketRateDesign)
            @test getfield(d.market, field) == getfield(market_d, field)
        end
        @test d.supremacy_obs == market_d.log_h .- market_d.log_a
        @test d.total_obs == market_d.log_h .+ market_d.log_a
        @test count(>(0), d.market.mask_weights) == count(==(1.0), fs.data[:flat_market_available])
        rng = MersenneTwister(20261111)
        zh, za = 0.2 .+ 0.3 .* randn(rng, n), 0.05 .+ 0.3 .* randn(rng, n)
        hand = mfo_hand_market(zh, za, d.market, p)
        engine = MFO_API._biased_market_rate_ll(zh, za, p.σ_obs, p.κ_D, p.δ_D, p.δ_M, d)
        relative = abs(engine - hand) / abs(hand)
        @test relative <= 1e-12
        @test MFO_API._fusion_market_ll(mfo_fusion(feature; bias = false), p, zh, za, d) ==
            MFO_API._market_rate_ll(zh, za, p.σ_obs, d.market)
        println("FUSION_REAL_FOLD matches=", n, " full_books=", count(>(0), d.market.mask_weights),
            " biased_market_relative_error=", relative)
    end

    mkpath(MFO_OUTPUT)
    @testset "4. exact AD and FD convergence on fold 40" begin
        for (index, (arm, dynamics, bias)) in enumerate(arms)
            model = mfo_model(dynamics, mfo_fusion(feature; bias))
            fs = mfo_real_features(model)
            density = mfo_density(MFO_PG.build_turing_model(model, fs); linked = true)
            f, theta = density.f, density.theta
            @test isfinite(f(theta))
            tape = ReverseDiff.compile(ReverseDiff.GradientTape(f, theta))
            rng = MersenneTwister(20261112)
            points, gradients, exact_checks = Vector{Float64}[], Vector{Float64}[], NamedTuple[]
            for (number, delta) in enumerate((0.0, 0.001, -0.002))
                point = theta + delta .* randn(rng, length(theta))
                compiled = similar(point); ReverseDiff.gradient!(compiled, tape, point)
                forward = ForwardDiff.gradient(f, point)
                fresh = ReverseDiff.gradient(f, point)
                forward_error = norm(compiled - forward) / max(norm(compiled), norm(forward), 1.0)
                fresh_error = norm(compiled - fresh) / max(norm(compiled), norm(fresh), 1.0)
                @test forward_error <= 1e-10
                @test fresh_error <= 1e-10
                push!(points, point); push!(gradients, compiled)
                push!(exact_checks, (; arm, point = number, forward_error, fresh_error))
            end
            CSV.write(joinpath(MFO_OUTPUT, "exact_ad_$arm.csv"), DataFrame(exact_checks))
            audit = MarketFusionFDAudit.audit(f, points, gradients, arm, MFO_OUTPUT; seed = 20261120 + index)
            for row in audit.comparisons
                @test row.high_precision_ad_relative_error <= 1e-10
            end
            for row in audit.rows
                row.precision == "BigFloat128" && (@test row.trend_ok)
            end
            for row in audit.extrapolations
                @test row.passed
            end
            println("FUSION_AD_EVIDENCE arm=", arm, " fold=40 parameters=", length(theta),
                " forward_max=", maximum(row.forward_error for row in exact_checks),
                " fresh_max=", maximum(row.fresh_error for row in exact_checks))
        end
    end

    @testset "6. synthetic recovery, three pre-declared seeds per arm" begin
        intervals = NamedTuple[]
        for seed in 1:3, (index, (arm, dynamics, bias)) in enumerate(arms)
            model = mfo_model(dynamics, mfo_fusion(feature; bias))
            fs = mfo_real_features(model)
            data_seed = 20261113 + index + 10(seed - 1)
            sampler_seed = 20261117 + index + 10(seed - 1)
            synthetic, truth = mfo_synthetic(fs, dynamics, bias, MersenneTwister(data_seed))
            Random.seed!(sampler_seed)
            config = Samplers.NUTSConfig(n_samples = 200, n_warmup = 200, n_chains = 2,
                accept_rate = 0.65, max_depth = 10, show_progress = false, silence_initial_stepsize = true)
            started = time()
            chain = Samplers.run_sampler(MFO_PG.build_turing_model(model, synthetic), config)
            wall = time() - started
            for name in sort!(collect(keys(truth)))
                draws = vec(Array(chain[Symbol(name)]))
                lo, median, hi = quantile(draws, [0.05, 0.5, 0.95])
                covered = lo <= truth[name] <= hi
                push!(intervals, (; arm, fold = 40, seed, data_seed, sampler_seed, parameter = name,
                    truth = truth[name], lo, median, hi, covered, wall_seconds = wall))
                println("RECOVERY arm=", arm, " seed=", seed, " parameter=", name, " truth=", truth[name],
                    " interval=[", lo, ",", hi, "] covered=", covered)
            end
            CSV.write(joinpath(MFO_OUTPUT, "synthetic_recovery_three_seeds.csv"), DataFrame(intervals))
        end
        @test length(intervals) == 3 * (12 + 12 + 9)
        for (arm, parameter) in unique([(row.arm, row.parameter) for row in intervals])
            group = filter(row -> row.arm == arm && row.parameter == parameter, intervals)
            @test length(group) == 3
            @test any(row.covered for row in group)  # No parameter misses ALL three seeds.
        end
        coverage = count(row.covered for row in intervals) / length(intervals)
        @test coverage >= 0.80
        println("RECOVERY_GATE covered=", count(row.covered for row in intervals),
            "/", length(intervals), " pooled_coverage=", coverage)
    end
end
println("MARKET_FUSION_OBSERVATION_TESTS_DONE")

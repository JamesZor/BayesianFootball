using Test, BayesianFootball, CSV, DataFrames, Distributions, DynamicPPL
using ForwardDiff, LinearAlgebra, LogDensityProblems, MCMCChains, Random, ReverseDiff, Statistics

const MRO_PG = BayesianFootball.Models.PreGame
const MRO_API = MRO_PG.Builder
const MRO_DIR = joinpath(@__DIR__, "..", "experiments", "scotland", "06_qs_joint_and_market_observation")
isdefined(Main, :Wave2MarketTable) || include(joinpath(MRO_DIR, "l01_market_table.jl"))
isdefined(Main, :Wave2LikelihoodAudit) || include(joinpath(MRO_DIR, "l02_likelihood_audit.jl"))
isdefined(Main, :MarketRateFDAudit) || include(joinpath(@__DIR__, "helpers", "market_rate_fd_audit.jl"))

mro_model(dynamics, feature; prior = truncated(Normal(0.0, 0.20), 0.0, Inf)) =
    CountModelBuilder(:market_rate_test) |> add(GlobalInterception()) |> add(dynamics) |>
    add(GlobalHomeAdvantage()) |> add(MarketRateObservation(feature = feature, sigma_prior = prior)) |> build
mro_weak_qs() = QualityStyleGRW(r₀ = LogNormal(0, 1), rₛ = LogNormal(0, 1), rₖ = LogNormal(0, 1))

function mro_density(tm; linked = false, seed = 20261010)
    Random.seed!(seed)
    vi = DynamicPPL.VarInfo(tm)
    linked && (vi = DynamicPPL.link!!(vi, tm))
    density = DynamicPPL.LogDensityFunction(tm, DynamicPPL.getlogjoint_internal, vi)
    return (; f = x -> LogDensityProblems.logdensity(density, x), theta = copy(vi[:]))
end

"A real late fold, so every initial/macro/micro scale and ratio participates in AD/recovery."
function mro_real_features(model)
    ds = Wave2MarketTable.QSMarketArm.checked_datastore()
    scope = Wave2MarketTable.QSMarketArm.goal_scope()
    scoped = Data.apply_scope(ds, scope)
    splitter = Data.ScopedWalkForwardCV(scope)
    boundaries = Data.create_id_boundaries(scoped, splitter)
    @test length(boundaries) == 40
    fs = first(Features.create_features(boundaries[40:40], scoped, model, splitter))
    return fs isa Tuple ? first(fs) : fs
end

"Independent synthetic walk using the fold's accumulator geometry; no engine likelihood calls."
function mro_walk(scales, n, design, rng)
    initial = scales[1] .* randn(rng, n, 1) * design.initial_accumulator
    season = scales[2] .* randn(rng, n, size(design.season_accumulator, 1)) * design.season_accumulator
    micro = scales[3] .* randn(rng, n, size(design.target_accumulator, 1)) * design.target_accumulator
    raw = initial + season + micro
    return raw .- mean(raw; dims = 1)
end

function mro_synthetic(fs, dynamics, rng)
    n = fs.data[:n_teams]
    design = MRO_API.dynamics_design(dynamics, fs, length(fs.data[:flat_home_ids]))
    truth = Dict{String,Float64}("obs.σ_obs" => 0.07)
    if dynamics isa QualityStyleGRW
        tau, ratio = [0.16, 0.085, 0.027], [0.4, 0.4, 0.3]
        q = mro_walk(tau ./ sqrt.(1 .+ ratio.^2), n, design, rng)
        s = mro_walk(ratio .* tau ./ sqrt.(1 .+ ratio.^2), n, design, rng)
        alpha, beta = q + s, s - q
        for (j, suffix) in enumerate(("₀", "ₛ", "ₖ"))
            truth["dyn.τ" * suffix], truth["dyn.r" * suffix] = tau[j], ratio[j]
        end
    else
        a, b = [0.12, 0.06, 0.03], [0.20, 0.11, 0.024]
        alpha, beta = mro_walk(a, n, design, rng), mro_walk(b, n, design, rng)
        for (j, suffix) in enumerate(("₀", "ₛ", "ₖ"))
            truth["dyn.α.σ" * suffix], truth["dyn.β.σ" * suffix] = a[j], b[j]
        end
    end
    h, a = design.home_state_indices, design.away_state_indices
    eta_h = log(1.2) .+ 0.15 .+ alpha[h] .+ beta[a]
    eta_a = log(1.2) .+ alpha[a] .+ beta[h]
    data = copy(fs.data)
    data[:flat_market_log_h] = eta_h + truth["obs.σ_obs"] .* randn(rng, length(h))
    data[:flat_market_log_a] = eta_a + truth["obs.σ_obs"] .* randn(rng, length(a))
    # Preserve the real full-book availability pattern; uncovered synthetic values are masked.
    return FeatureSet(data), truth
end

@testset "MarketRateObservation" begin
    @testset "injected feature, builder and likelihood" begin
        table = DataFrame(match_id = [2, 1, 3], log_lambda_h = [0.2, 0.3, NaN],
            log_lambda_a = [-0.1, 0.1, NaN], full_book = [true, true, false])
        feature = Features.MarketRatesFeature(table)
        empty = DataFrame()
        ds = Data.DataStore(Data.ScottishLower(), empty, empty, empty, empty, empty, empty, empty, empty)
        data = Dict{Symbol,Any}()
        Features.add_feature!(data, feature, [1, 2, 3, 99], Dict(), ds)
        @test data[:flat_market_log_h] == [0.3, 0.2, 0.0, 0.0]
        @test data[:flat_market_available] == [1.0, 1.0, 0.0, 0.0]
        @test eltype(data[:flat_market_log_a]) == Float64
        @test_throws ErrorException Features.MarketRatesFeature(vcat(table, table[1:1, :]))
        bad = deepcopy(table); bad.full_book[3] = true
        @test_throws ErrorException Features.MarketRatesFeature(bad)
        model = mro_model(MultiScaleGRW(), feature)
        @test model isa MRO_API.PoissonCountModel
        @test MRO_API.observation_family(model.observation) == :poisson
        @test MRO_API.observation_wired(model.observation)
        @test MRO_API.observation_prefixes(model.observation) == [:obs]
        @test MRO_API.observation_features(model.observation) == [feature]
        @test Symbol("obs.σ_obs") in MRO_API.cb_varinfo_sites(model)
        @test_throws ErrorException mro_model(MultiScaleGRW(), feature; prior = Normal())
        @test quantile(model.observation.sigma_prior, 0.5) ≈ 0.20 * quantile(Normal(), 0.75)
        fs = FeatureSet(data)
        design = MRO_API.observation_design(model.observation, fs, 4, [1.0, 0.5, 1.0, 1.0])
        @test design.mask_weights == [1.0, 0.5, 0.0, 0.0]
        bad_data = deepcopy(data); bad_data[:flat_market_available][1] = 0.5
        @test_throws ErrorException MRO_API.observation_design(model.observation, FeatureSet(bad_data), 4, ones(4))
        @test_throws ErrorException MRO_API.observation_design(model.observation, fs, 3, ones(3))
        eh, ea, sigma = [0.4, 0.1, 8.0, -4.0], [0.0, -0.2, -3.0, 7.0], 0.12
        # Independent hand-written Normal log density, including BOTH normalising constants.
        expected = sum(design.mask_weights[i] * (-2log(sigma) - log(2pi) -
            ((design.log_h[i] - eh[i])^2 + (design.log_a[i] - ea[i])^2) / (2sigma^2)) for i in 1:4)
        via_logpdf = sum(design.mask_weights[i] * (logpdf(Normal(eh[i], sigma), design.log_h[i]) +
            logpdf(Normal(ea[i], sigma), design.log_a[i])) for i in 1:4)
        @test expected ≈ via_logpdf atol = 1e-12 rtol = 0
        @test MRO_API._market_rate_ll(eh, ea, sigma, design) ≈ expected atol = 1e-12 rtol = 0
        zeros_design = MRO_API.MarketRateDesign(design.log_h, design.log_a, zeros(4), 0.0)
        @test MRO_API._market_rate_ll(eh, ea, sigma, zeros_design) == 0.0
        changed = MRO_API.MarketRateDesign([0.3, 0.2, 50.0, -50.0],
            [0.1, -0.1, -50.0, 50.0], design.mask_weights, design.weight_sum)
        @test MRO_API._market_rate_ll(eh, ea, sigma, changed) == MRO_API._market_rate_ll(eh, ea, sigma, design)
        tm = MRO_API._observe(model.observation, eh, ea, zeros(Int, 4), zeros(Int, 4),
            ones(4), zeros(4), zeros(4), 2, 12, design)
        # _observe returns ll; only the production parent accumulates it. Mirror that seam.
        @test mro_density(tm).f([sigma]) == logpdf(model.observation.sigma_prior, sigma)
        accumulated = Wave2LikelihoodAudit.accumulated(tm)
        density = mro_density(accumulated)
        @test length(density.theta) == 1
        @test density.f([sigma]) ≈ expected + logpdf(model.observation.sigma_prior, sigma) atol = 1e-12 rtol = 0
        likelihood = DynamicPPL.LogDensityFunction(accumulated, DynamicPPL.getloglikelihood)
        @test LogDensityProblems.logdensity(likelihood, [sigma]) ≈ expected atol = 1e-12 rtol = 0
        @test mro_density(accumulated; linked = true).f([log(sigma)]) ≈
            expected + logpdf(model.observation.sigma_prior, sigma) + log(sigma) atol = 1e-12 rtol = 0
        zero_tm = MRO_API._observe(model.observation, eh, ea, fill(999, 4), fill(999, 4),
            ones(4), zeros(4), zeros(4), 2, 12, zeros_design)
        @test mro_density(zero_tm).f([sigma]) == logpdf(model.observation.sigma_prior, sigma)
        rates = MRO_API._cb_rates(model.observation, [1.0], [2.0], (; σ_obs = [sigma]), 1, 2, 1)
        @test rates.λ_h == [1.0] && rates.λ_a == [2.0]
    end

    feature = Wave2MarketTable.feature()
    for (arm, dynamics) in (("grw", MultiScaleGRW()), ("qs", mro_weak_qs()))
        model = mro_model(dynamics, feature)
        fs = mro_real_features(model)
        @testset "$arm real-fold AD" begin
            density = mro_density(MRO_PG.build_turing_model(model, fs); linked = true)
            f, theta = density.f, density.theta
            @test isfinite(f(theta))
            tape = ReverseDiff.compile(ReverseDiff.GradientTape(f, theta))
            rng = MersenneTwister(20261012)
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
                push!(points, point)
                push!(gradients, compiled)
                push!(exact_checks, (; arm, point = number, forward_error, fresh_error))
            end
            output = "/root/BF_runs/qs_experiment_w2_out/phase1"
            mkpath(output)
            CSV.write(joinpath(output, "exact_ad_$arm.csv"), DataFrame(exact_checks))
            audit = MarketRateFDAudit.audit(f, points, gradients, arm, output)
            for row in audit.comparisons
                @test row.high_precision_ad_relative_error <= 1e-10
            end
            for row in audit.rows
                row.precision == "BigFloat128" && (@test row.trend_ok)
            end
            for row in audit.extrapolations
                @test row.passed
            end
            println("MARKET_AD_EVIDENCE arm=", arm, " fold=40 parameters=", length(theta),
                " forward_max=", maximum(row.forward_error for row in exact_checks),
                " fresh_max=", maximum(row.fresh_error for row in exact_checks))
        end
    end

    @testset "synthetic recovery, frozen seeds and smoke budget" begin
        # Manager amendment: retain the original seed, run TWO further fixed seeds per arm.
        # The original interval CSV is immutable; do not refit it or overwrite its evidence.
        original = CSV.read(joinpath(MRO_DIR, "results", "synthetic_recovery.csv"), DataFrame)
        @test nrow(original) == 14
        intervals = NamedTuple[]
        for row in eachrow(original)
            index = row.arm == "grw" ? 1 : 2
            push!(intervals, (; arm = row.arm, fold = row.fold, seed = 1,
                data_seed = 20261013 + index, sampler_seed = 20261015 + index,
                parameter = row.parameter, truth = row.truth, lo = row.lo, median = row.median,
                hi = row.hi, covered = row.covered, wall_seconds = row.wall_seconds))
        end
        for seed in 2:3
            for (index, (arm, dynamics)) in enumerate((("grw", MultiScaleGRW()), ("qs", mro_weak_qs())))
                model = mro_model(dynamics, feature)
                fs = mro_real_features(model)
                data_seed = 20261013 + index + 10(seed - 1)
                sampler_seed = 20261015 + index + 10(seed - 1)
                synthetic, truth = mro_synthetic(fs, dynamics, MersenneTwister(data_seed))
                Random.seed!(sampler_seed)
                config = Samplers.NUTSConfig(n_samples = 200, n_warmup = 200, n_chains = 2,
                    accept_rate = 0.65, max_depth = 10, show_progress = false, silence_initial_stepsize = true)
                started = time()
                chain = Samplers.run_sampler(MRO_PG.build_turing_model(model, synthetic), config)
                for name in sort!(collect(keys(truth)))
                    # Select the immutable first seed, not the other new-seed intervals.
                    baseline = only(filter(row -> row.arm == arm && row.parameter == name && row.seed == 1, intervals))
                    draws = vec(Array(chain[Symbol(name)]))
                    lo, median, hi = quantile(draws, [0.05, 0.5, 0.95])
                    covered = lo <= truth[name] <= hi
                    push!(intervals, (; arm, fold = 40, seed, data_seed, sampler_seed,
                        parameter = name, truth = truth[name], lo, median, hi, covered,
                        wall_seconds = time() - started))
                    @test baseline.truth == truth[name]
                end
            end
        end
        output = "/root/BF_runs/qs_experiment_w2_out/phase1"
        mkpath(output)
        CSV.write(joinpath(output, "synthetic_recovery_three_seeds.csv"), DataFrame(intervals))
        for row in intervals
            println("RECOVERY arm=", row.arm, " seed=", row.seed, " parameter=", row.parameter,
                " truth=", row.truth, " interval=[", row.lo, ",", row.hi, "] covered=", row.covered)
        end
        @test length(intervals) == 42
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
println("MARKET_RATE_OBSERVATION_TESTS_DONE")

using Test
using BayesianFootball
using DataFrames
using Dates
using Distributions
using DynamicPPL
using LogDensityProblems
using Random
using ReverseDiff

# ==============================================================================
# Zero-allocation compiled gradients in the builder engine
# ==============================================================================
#
# Every composable recipe must replay its compiled ReverseDiff tape without touching the heap.
# A tracked scalar (or any plain `Real`) inside a FUSED broadcast sends ReverseDiff to
# `tracker_∇broadcast`, whose reverse pass allocates O(rows) per call. That is invisible in
# latency — the W2 `td_base` gradient took 0.24 ms — and at 16 threads it halved NUTS throughput
# through the garbage collector. docs/turing_ad_performance_guide.md §10.5.
#
# The matrix below crosses every dynamics, observation, guard and predictor family the builder
# composes, prunes the combinations the builder itself rejects, and asserts 0 bytes per compiled
# gradient (the minimum over 20 warmed replays) for each survivor, naming the offending
# instructions when one allocates.
#
# Self-contained on purpose: the profiling helpers are included from src/harness/tape_profile.jl
# by path, so this file runs unchanged against an older checkout —
#   julia --project=<old checkout> <this checkout>/test/tape_allocation_tests.jl
# — which is how it is shown to FAIL on the pre-fix engine (8effca72).
# ==============================================================================

module TapeAllocationTestProfile
import ReverseDiff
include(joinpath(@__DIR__, "..", "src", "harness", "tape_profile.jl"))
end

const TA_PG = BayesianFootball.Models.PreGame
const TA_F = BayesianFootball.Features

# A pooled synthetic pyramid: four SPFL tiers, a club that changes tier, a promoted club, the Old
# Firm, and a senior cup tie — enough for PyramidTiers events and CompetitionKappa's tier and
# competition maps to be non-trivial.
function _ta_rows()
    rows = NamedTuple[]
    add(id, tournament, season, date, home, away; biweek = 1, score = (1, 0)) = push!(rows, (;
        match_id = id, tournament_id = tournament, season = String(season),
        match_date = Date(date), match_hour = 15, match_week = biweek,
        match_biweek = biweek, match_month = month(Date(date)),
        home_team = String(home), away_team = String(away),
        home_score = score[1], away_score = score[2], neutral_venue = false))
    add(1, 55, "21/22", "2022-04-01", "changer", "champ")
    add(2, 54, "21/22", "2022-04-02", "celtic", "top"; score = (2, 1))
    add(3, 56, "21/22", "2022-04-03", "l1", "l1b"; score = (0, 0))
    add(4, 57, "21/22", "2022-04-04", "promoted", "l2"; score = (3, 1))
    add(5, 56, "22/23", "2023-04-01", "changer", "l1"; score = (1, 2))
    add(6, 55, "22/23", "2023-04-02", "champ", "champ2")
    add(7, 54, "22/23", "2023-04-03", "celtic", "top"; score = (4, 0))
    add(8, 57, "22/23", "2023-04-04", "promoted", "l2"; score = (2, 2))
    add(9, 56, "23/24", "2023-08-05", "changer", "promoted"; score = (0, 1))
    add(10, 57, "23/24", "2023-08-05", "entered", "l2")
    add(11, 55, "23/24", "2023-08-06", "champ", "champ2"; score = (1, 1))
    add(12, 54, "23/24", "2023-08-06", "celtic", "top"; biweek = 2, score = (2, 0))
    add(13, 73, "2023", "2023-08-12", "changer", "celtic"; biweek = 2, score = (0, 3))
    return DataFrame(rows)
end

function _ta_pooled_store()
    empty = DataFrame()
    raw = BayesianFootball.Data.DataStore(BayesianFootball.Data.ScottishPyramid(),
        _ta_rows(), empty, empty, empty, empty, empty, empty, empty)
    scope = BayesianFootball.Data.DataScope(
        name = "spfl_cups", train_tournaments = [54, 55, 56, 57],
        cups = :senior_spfl_ties, target_tournaments = [56, 57],
        monitor_tournaments = [54, 55], clock_tournaments = [56, 57],
        target_seasons = ["23/24"], history_seasons = 2, dynamics_col = :match_biweek)
    return BayesianFootball.Data.apply_scope(raw, scope)
end

const TA_BOUNDARY = BayesianFootball.Data.SplitBoundary(1, 1, collect(1:8), collect(9:13))

_ta_proxy_feature() = MatchProxyXGFeature(k = 25.0, fallback = :none)
_ta_joint(kappa) = JointGammaPoissonObservation(
    feature = _ta_proxy_feature(), shape_prior = truncated(Normal(4.0, 1.5), 0.5, Inf),
    log_kappa_prior = Normal(0.0, 0.2), kappa = kappa)

const TA_DYNAMICS = [
    ("td", () -> TimeDecayDynamics(days_half_life = 180.0)),
    ("grw", () -> MultiScaleGRW()),
    ("grw_step", () -> MultiScaleGRW(target_season_step = true)),
]

const TA_OBSERVATIONS = [
    ("poisson", () -> PoissonObservation()),
    ("negbin", () -> NegativeBinomialObservation(dispersion = GlobalDispersion())),
    ("joint_shared", () -> _ta_joint(SharedKappa())),
    ("joint_hier", () -> _ta_joint(HierarchicalKappa())),
    ("joint_competition", () -> _ta_joint(CompetitionKappa())),
    ("joint_negbin", () -> JointGammaNegBinObservation(
        dispersion = GlobalDispersion(), feature = _ta_proxy_feature())),
]

const TA_GUARDS = [("clamp", () -> ClampGuard()), ("noguard", () -> NoGuard())]

# Predictors whose design comes from the pyramid store go through `create_features`; the wealth and
# lineup columns need tables this store does not have, so they are injected as flat design vectors
# (`_ta_inject`) — the engine reads nothing else.
const TA_PREDICTORS = [
    ("none", () -> ()),
    ("tiers_nocarry_nojump", () -> (PyramidTiers(carry = NoCarry(), jump = NoJump()),)),
    ("tiers_carry_jump", () -> (PyramidTiers(carry = EstimatedCarry(), jump = EstimatedJump()),)),
    ("tiers_fixed_priors", () -> (PyramidTiers(tier_term = false, anchor = :full_pyramid,
                                               carry = NoCarry(),
                                               jump = FixedTransitionPriors()),)),
    ("wealth", () -> (ProductionWealthCovariate(),)),
    ("lineup_bench", () -> (PlayerLineupPillar(
        feature = TA_F.XGPlusMinusFeature(),
        aggregation = BenchWeightedPlayerAggregation(w_bench = 0.10),
        w_bench_prior = truncated(Normal(0.10, 0.05), 0.0, 1.0)),)),
    ("lineup_positional", () -> (PlayerLineupPillar(
        feature = TA_F.XGPlusMinusFeature(),
        aggregation = PositionalPlayerAggregation()),)),
]

_ta_needs_injection(term) = term isa PlayerLineupPillar || term isa ProductionWealthCovariate

"Flat design vectors for the wealth and lineup predictors, periodic so no column is constant."
function _ta_inject(fs)
    n = length(fs.data[:flat_home_ids])
    x = Float64[0.4 * sin(i) for i in 1:n]
    y = Float64[0.3 * cos(i) for i in 1:n]
    extra = Dict{Symbol,Any}(
        :flat_delta_production_wealth => x,
        :production_wealth_oos_bridge_by_match_id => Dict{Int,Float64}(),
        :flat_home_outfield_rating => x .+ 0.3, :flat_away_outfield_rating => y .+ 0.1,
        :flat_home_bench_rating => 0.3 .* y, :flat_away_bench_rating => 0.2 .* x,
        :flat_home_D_rating => 0.2 .* x, :flat_home_M_rating => 0.4 .* y,
        :flat_home_F_rating => 0.4 .* x .+ 0.3, :flat_away_D_rating => 0.3 .* y,
        :flat_away_M_rating => 0.4 .* x, :flat_away_F_rating => 0.3 .* y .+ 0.1,
        :flat_home_bench_D_rating => 0.05 .* x, :flat_home_bench_M_rating => 0.10 .* y,
        :flat_home_bench_F_rating => 0.15 .* x, :flat_away_bench_D_rating => 0.04 .* y,
        :flat_away_bench_M_rating => 0.08 .* x, :flat_away_bench_F_rating => 0.08 .* y,
        :flat_home_minute_weighted_rating => 0.85 .* x, :flat_away_minute_weighted_rating => 0.75 .* y,
    )
    return FeatureSet(merge(copy(fs.data), extra))
end

"Build, or return `nothing` when the builder's own validation rejects the combination."
function _ta_build(name, dynamics, predictors, observation, guard)
    try
        return build_count_model(Symbol(name), GlobalInterception(), dynamics,
                                 GlobalHomeAdvantage(), predictors..., observation, guard)
    catch err
        err isa ErrorException && occursin("CountModelBuilder validation failed", err.msg) &&
            return nothing
        rethrow()
    end
end

"Bytes per compiled gradient at a linked prior draw — the space NUTS evaluates the tape in."
function _ta_compiled_bytes(model, fs; seed::Int = 20260927)
    turing_model = TA_PG.build_turing_model(model, fs)
    Random.seed!(seed)
    varinfo = DynamicPPL.link!!(DynamicPPL.VarInfo(turing_model), turing_model)
    θ = copy(varinfo[:])
    density = DynamicPPL.LogDensityFunction(turing_model, DynamicPPL.getlogjoint_internal, varinfo)
    f = x -> LogDensityProblems.logdensity(density, x)
    raw = ReverseDiff.GradientTape(f, θ)
    tape = ReverseDiff.compile(raw)
    gradient = similar(θ)
    bytes = TapeAllocationTestProfile.compiled_gradient_bytes(tape, gradient, θ)
    all(isfinite, gradient) || error("non-finite compiled gradient")
    rows = bytes == 0 ? NamedTuple[] : TapeAllocationTestProfile.tape_allocation_profile(raw)
    return (; bytes, rows, n_parameters = length(θ))
end

@testset "Zero-allocation compiled gradients" begin
    pooled = _ta_pooled_store()
    feature_cache = Dict{String,Any}()
    built = 0
    pruned = String[]
    offenders = String[]

    for (dyn_name, dyn) in TA_DYNAMICS, (obs_name, obs) in TA_OBSERVATIONS,
        (guard_name, guard) in TA_GUARDS, (pred_name, pred) in TA_PREDICTORS

        name = join((dyn_name, obs_name, guard_name, pred_name), "__")
        predictors = pred()
        model = _ta_build(name, dyn(), predictors, obs(), guard())
        if model === nothing
            push!(pruned, name)
            continue
        end
        built += 1

        # Features come from the model minus any injected predictor; cache by what they need.
        feature_predictors = Tuple(p for p in predictors if !_ta_needs_injection(p))
        feature_model = _ta_build(name * "__features", dyn(), feature_predictors, obs(), guard())
        key = string(TA_F.required_features(feature_model))
        base_fs = get!(feature_cache, key) do
            TA_F.create_features(TA_BOUNDARY, pooled, feature_model, :match_biweek)
        end
        fs = any(_ta_needs_injection, predictors) ? _ta_inject(base_fs) : base_fs

        result = _ta_compiled_bytes(model, fs)
        @test result.bytes == 0
        if result.bytes != 0
            listing = join(("#$(r.index) $(r.bytes) B $(r.description)" for r in result.rows), "; ")
            push!(offenders, "$name: $(result.bytes) B/gradient — $listing")
        end
    end

    isempty(offenders) || @info "Allocating compiled tapes" offenders
    @info "tape allocation matrix" built pruned = length(pruned)
    # The builder rejects NoGuard for every NegBin and joint-Gamma observation (a finite η floor is
    # required) — 3 dynamics × 5 observations × 7 predictor sets. Nothing else may be pruned: a
    # combination that silently stopped building would silently stop being tested.
    @test length(pruned) == 3 * 5 * 7
    @test all(n -> occursin("__noguard__", n) && !occursin("__poisson__", n), pruned)
    @test built == 3 * 6 * 2 * 7 - 3 * 5 * 7
end

using Test
using BayesianFootball
using MCMCChains
using Statistics
using DataFrames
using Dates
using Distributions
using DynamicPPL
using LogDensityProblems
using ReverseDiff
using ForwardDiff
using LinearAlgebra

const PyramidGRW = BayesianFootball.Models.PreGame
const PyramidBuilder = PyramidGRW.Builder

function _pyramid_empty_target_chain(n_teams::Int, n_draws::Int)
    columns = String[]
    for side in ("α", "β")
        append!(columns, ["dyn.$side.σ₀", "dyn.$side.σₛ"])
        append!(columns, ["dyn.$side.z_init[$team]" for team in 1:n_teams])
        append!(columns, ["dyn.$side.z_season[$team,1]" for team in 1:n_teams])
    end
    values = zeros(n_draws, length(columns), 1)
    for (column, name) in enumerate(columns)
        values[:, column, 1] .= occursin("σₛ", name) ? 0.4 : 0.0
    end
    return Chains(values, Symbol.(columns))
end

@testset "Pyramid-tier MultiScaleGRW" begin
    @testset "macro accumulator geometry" begin
        for (n_history, n_target) in ((2, 0), (2, 3), (3, 2))
            old = PyramidGRW.grw_accumulators(n_history, n_target)
            flagged = PyramidGRW.grw_accumulators(
                n_history, n_target; target_season_step = true)
            @test old.initial == flagged.initial
            @test old.target == flagged.target
            if n_target == 0
                @test flagged.season == old.season
            else
                @test size(flagged.season, 1) == size(old.season, 1) + 1
                @test flagged.season[1:size(old.season, 1), :] == old.season
                @test flagged.season[end, :] ==
                      Float64[state >= n_history + 1 for state in 1:(n_history + n_target)]
            end
        end

        @test PyramidGRW.grw_accumulators(2, 3).season == Float64[0 1 1 1 1;]
        @test PyramidGRW.grw_accumulators(2, 3; target_season_step = true).season ==
              Float64[0 1 1 1 1; 0 0 1 1 1]
        @test PyramidGRW.grw_accumulators(3, 2).season ==
              Float64[0 1 1 1 1; 0 0 1 1 1]
    end

    @testset "empty-target OOS boundary innovation" begin
        n_teams, n_draws = 3, 4_000
        chain = _pyramid_empty_target_chain(n_teams, n_draws)
        config = MultiScaleGRW(target_season_step = true)
        feature_set = FeatureSet(Dict{Symbol,Any}(:grw_oos_seed => 8128))
        draw_a = PyramidBuilder._cb_extract_dynamics(
            chain, config, "dyn", n_teams, feature_set)
        draw_b = PyramidBuilder._cb_extract_dynamics(
            chain, config, "dyn", n_teams, feature_set)
        @test draw_a.oos_α == draw_b.oos_α
        @test draw_a.oos_β == draw_b.oos_β
        @test size(draw_a.oos_α) == (n_draws, n_teams)
        @test abs(mean(draw_a.oos_α)) < 0.015
        @test isapprox(std(vec(draw_a.oos_α)), 0.4; rtol = 0.04)

        # The OOS hook applies that one innovation to each side's carried state.
        effects = PyramidBuilder._cb_oos_dynamics(
            config, draw_a, Dict(), 1, 1, 2, n_draws)
        @test effects.att_h == draw_a.oos_α[:, 1]
        @test effects.def_a == draw_a.oos_β[:, 2]
    end

    @testset "default-off regression surface" begin
        default = MultiScaleGRW()
        explicit_off = MultiScaleGRW(target_season_step = false)
        @test default == explicit_off
        @test string(default) == string(explicit_off)
        @test PyramidGRW.grw_accumulators(3, 2) ==
              PyramidGRW.grw_accumulators(3, 2; target_season_step = false)
    end
end

function _pyramid_rows()
    rows = NamedTuple[]
    add(id, tournament, season, date, home, away; biweek = 1) = push!(rows, (;
        match_id = id, tournament_id = tournament, season = String(season),
        match_date = Date(date), match_hour = 15, match_week = biweek,
        match_biweek = biweek, match_month = month(Date(date)),
        home_team = String(home), away_team = String(away),
        home_score = 1, away_score = 0, neutral_venue = false))
    add(1, 55, "21/22", "2022-04-01", "changer", "champ")
    add(2, 54, "21/22", "2022-04-02", "celtic", "top")
    add(3, 56, "21/22", "2022-04-03", "l1", "l1b")
    add(4, 57, "21/22", "2022-04-04", "promoted", "l2")
    add(5, 56, "22/23", "2023-04-01", "changer", "l1")
    add(6, 55, "22/23", "2023-04-02", "champ", "champ2")
    add(7, 54, "22/23", "2023-04-03", "celtic", "top")
    add(8, 57, "22/23", "2023-04-04", "promoted", "l2")
    add(9, 56, "23/24", "2023-08-05", "changer", "promoted")
    add(10, 57, "23/24", "2023-08-05", "entered", "l2")
    add(11, 55, "23/24", "2023-08-06", "champ", "champ2")
    add(12, 54, "23/24", "2023-08-06", "celtic", "top")
    add(13, 73, "2023", "2023-08-12", "changer", "celtic"; biweek = 2)
    return DataFrame(rows)
end

function _pyramid_store()
    empty = DataFrame()
    return BayesianFootball.Data.DataStore(BayesianFootball.Data.ScottishPyramid(),
        _pyramid_rows(), empty, empty, empty, empty, empty, empty, empty)
end

function _pyramid_scope(; lower = false)
    return BayesianFootball.Data.DataScope(
        name = lower ? "lower" : "spfl_cups",
        train_tournaments = lower ? [56, 57] : [54, 55, 56, 57],
        cups = lower ? :none : :senior_spfl_ties,
        target_tournaments = [56, 57], monitor_tournaments = lower ? Int[] : [54, 55],
        clock_tournaments = [56, 57], target_seasons = ["23/24"],
        history_seasons = 2, dynamics_col = :match_biweek)
end

function _pyramid_model(; tiers = PyramidTiers(carry = EstimatedCarry(), jump = EstimatedJump()))
    return CountModelBuilder(:pyramid_synthetic) |> add(GlobalInterception()) |>
        add(TimeDecayDynamics(days_half_life = 180.0)) |> add(GlobalHomeAdvantage()) |>
        add(tiers) |> add(JointGammaPoissonObservation()) |> build
end

@testset "Pyramid tier features and reductions" begin
    raw = _pyramid_store()
    lower = BayesianFootball.Data.apply_scope(raw, _pyramid_scope(lower = true))
    @test DataFrames.metadata(lower.matches, "full_club_season_tiers", nothing) !== nothing
    ordered = Int[3, 4, 5, 8]
    team_map = Dict(name => i for (i, name) in enumerate(sort(unique(vcat(
        lower.matches.home_team, lower.matches.away_team)))))
    F_full = Dict{Symbol,Any}(:target_season => "23/24")
    BayesianFootball.Features.add_feature!(F_full,
        PyramidTierFeature(anchor = :full_pyramid, jump = EstimatedJump()),
        ordered, team_map, lower)
    F_lower = Dict{Symbol,Any}(:target_season => "23/24")
    BayesianFootball.Features.add_feature!(F_lower,
        PyramidTierFeature(anchor = :first_in_window, jump = EstimatedJump()),
        ordered, team_map, lower)
    changer_row = findfirst(==(5), ordered)
    @test F_full[:pyramid_anchor_home][changer_row] == -1.0
    @test F_lower[:pyramid_anchor_home][changer_row] == -2.0

    events = F_full[:pyramid_events]
    changer_event = findfirst(e -> e.club == "changer" && e.season == "22/23", events)
    promoted_event = findfirst(e -> e.club == "promoted" && e.season == "23/24", events)
    @test changer_event !== nothing
    @test promoted_event !== nothing
    target = F_full[:pyramid_oos_bridge][9]
    @test target.home.incidence[changer_event] == 1.0
    @test target.away.incidence[promoted_event] == 1.0
    @test all(e -> e.club != "entered", events)

    fixed = Dict{Symbol,Any}(:target_season => "23/24")
    BayesianFootball.Features.add_feature!(fixed,
        PyramidTierFeature(anchor = :full_pyramid, jump = FixedTransitionPriors()),
        ordered, team_map, lower)
    fixed_events = fixed[:pyramid_events]
    @test Set((e.from, e.to) for e in fixed_events) == Set([(55, 56), (57, 56)])
    rel = only(filter(e -> (e.from, e.to) == (55, 56), fixed_events))
    @test (rel.α_mean, rel.α_sd, rel.β_mean, rel.β_sd) == (0.12, 0.20, -0.16, 0.26)

    pooled = BayesianFootball.Data.apply_scope(raw, _pyramid_scope())
    all_ids = Int.(pooled.matches.match_id)
    pooled_map = Dict(name => i for (i, name) in enumerate(sort(unique(vcat(
        pooled.matches.home_team, pooled.matches.away_team)))))
    F_pooled = Dict{Symbol,Any}(:target_season => "23/24")
    BayesianFootball.Features.add_feature!(F_pooled,
        PyramidTierFeature(anchor = :first_in_window, jump = NoJump()),
        all_ids, pooled_map, pooled)
    cup = F_pooled[:pyramid_oos_bridge][13]
    @test cup.home.anchor + cup.home.delta == -2.0
    @test cup.away.anchor + cup.away.delta == 0.0
    @test cup.away.old_firm == 1.0

    source = (; home = (; anchor = -1.0, delta = -1.0, old_firm = 0.0,
                         incidence = Float64[]),
                away = (; anchor = -1.0, delta = 0.0, old_firm = 0.0,
                         incidence = Float64[]), events = NamedTuple[])
    bridge = Dict(99 => source)
    row = (; match_id = 99)
    base_draw = (; step = [1.0], carry = [0.0], old_firm = [0.0],
                  jump = nothing, attack_share = 0.48)
    carried_draw = merge(base_draw, (; carry = [1.0]))
    no_carry = BayesianFootball.predictor_oos(
        PyramidTiers(carry = NoCarry(), jump = NoJump()), base_draw, bridge, row)
    fixed_zero = BayesianFootball.predictor_oos(
        PyramidTiers(carry = FixedCarry(0), jump = NoJump()), base_draw, bridge, row)
    fixed_one = BayesianFootball.predictor_oos(
        PyramidTiers(carry = FixedCarry(1), jump = NoJump()), carried_draw, bridge, row)
    @test no_carry == fixed_zero
    @test fixed_one.h - no_carry.h == [-0.48]
    @test fixed_one.a - no_carry.a == [0.52]

    old_source = merge(source, (; home = merge(source.home, (; old_firm = 1.0))))
    old_draw = merge(base_draw, (; old_firm = [1.15]))
    old = BayesianFootball.predictor_oos(PyramidTiers(jump = NoJump()), old_draw,
        Dict(99 => old_source), row)
    plain = BayesianFootball.predictor_oos(PyramidTiers(jump = NoJump()), old_draw,
        bridge, row)
    @test old.h - plain.h == [0.48 * 1.15]
    @test old.a - plain.a == [-0.52 * 1.15]
end

@testset "Pyramid and league-delta defaults and recipe identity" begin
    tiers = PyramidTiers()
    @test tiers.tier_term
    @test tiers.anchor == :first_in_window
    @test tiers.attack_share == 0.48
    @test tiers.carry isa NoCarry
    @test tiers.jump isa NoJump
    @test tiers.step_prior.untruncated == Normal(0.47, 0.16)
    @test tiers.old_firm_prior.untruncated == Normal(1.15, 0.10)
    @test EstimatedCarry().prior == Beta(1.5, 4.0)
    @test EstimatedJump().scale_prior == Gamma(2.0, 0.08)
    @test CompetitionKappa().intercept_scale == 0.10
    @test CompetitionKappa().kappa_scale == 0.25
    @test DEFAULT_TRANSITION_PRIORS[2] ==
          (from = 56, to = 57, α_mean = 0.03, α_sd = 0.20,
           β_mean = -0.04, β_sd = 0.24)

    base = CountModelBuilder(:base) |> add(GlobalInterception()) |>
        add(TimeDecayDynamics(days_half_life = 180.0)) |>
        add(GlobalHomeAdvantage()) |> add(JointGammaPoissonObservation()) |> build
    same = CountModelBuilder(:same) |> add(GlobalInterception()) |>
        add(TimeDecayDynamics(days_half_life = 180.0)) |>
        add(GlobalHomeAdvantage()) |> add(JointGammaPoissonObservation()) |> build
    @test string(base) == string(same)
    c0 = Harness.Candidate(name = "a", model = base, scope = _pyramid_scope(), role = :control)
    c1 = Harness.Candidate(name = "b", model = same, scope = _pyramid_scope(), role = :candidate)
    @test Harness.recipe_hash(c0) == Harness.recipe_hash(c1)
    @test string(_pyramid_model(tiers = PyramidTiers(carry = NoCarry()))) !=
          string(_pyramid_model(tiers = PyramidTiers(carry = EstimatedCarry())))
end

function _league_delta_model(dynamics)
    observation = JointGammaPoissonObservation(
        feature = BayesianFootball.Features.MatchProxyXGFeature(k = 25.0, fallback = :none),
        shape_prior = truncated(Normal(4.0, 1.5), 0.5, Inf),
        log_kappa_prior = Normal(0.0, 0.2), kappa = CompetitionKappa())
    return CountModelBuilder(:league_delta_synthetic) |> add(GlobalInterception()) |>
        add(dynamics) |> add(GlobalHomeAdvantage()) |>
        add(PyramidTiers(carry = EstimatedCarry(), jump = EstimatedJump())) |>
        add(observation) |> build
end

@testset "Pyramid and league deltas are compiled-AD safe" begin
    pooled = BayesianFootball.Data.apply_scope(_pyramid_store(), _pyramid_scope())
    boundary = BayesianFootball.Data.SplitBoundary(1, 1, collect(1:8), collect(9:13))
    for dynamics in (TimeDecayDynamics(days_half_life = 180.0),
                     MultiScaleGRW(target_season_step = true))
        model = _league_delta_model(dynamics)
        fs = BayesianFootball.Features.create_features(
            boundary, pooled, model, :match_biweek)
        design = PyramidBuilder.cb_design(model, fs)
        @test design.observation_data.n_tiers == 4
        @test design.observation_data.n_competitions == 5
        audit = BayesianFootball.Harness._gradient_audit(model, fs; replays = 3)
        @test audit.reversediff_forward_error <= 1.0e-6
        @test audit.worst_perturbed_error <= 1.0e-8
        @test isfinite(audit.log_density)
    end
end

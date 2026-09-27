# Wave 2: pyramid tiers, transition jumps and league observation deltas.
using BayesianFootball
using Distributions

const EXPERIMENT = "scottish_pyramid_tier_transition_w2"
const TODO = 31
const QUESTION = "Can explicit pyramid tiers and transition events fix cross-tier compression and league-change bias?"
const DIMENSION = "pyramid_tiers_transition_observation"
const STATUS = "spec agreed, not run"
const DECISION = "spec agreed, not run"
const README = joinpath(@__DIR__, "README.md")
# Candidate has one experiment-wide role/control and cannot encode a different
# paired control per row. The manager scores GRW rows vs grw_base and lower rows
# vs td_lower_base with scripts/score_runs.jl after their UUIDs exist.
const CONTROL = "td_base"

function make_joint_observation(; league_deltas::Bool = false)
    mode = league_deltas ? CompetitionKappa() : SharedKappa()
    return JointGammaPoissonObservation(
        feature = Features.MatchProxyXGFeature(k = 25.0, fallback = :none),
        shape_prior = truncated(Normal(4.0, 1.5), 0.5, Inf),
        log_kappa_prior = Normal(0.0, 0.2),
        kappa = mode,
    )
end

const W2_TARGET_SEASONS = ["23/24", "24/25", "25/26"]
const SPFL_CUPS_SCOPE = Data.DataScope(
    name = "spfl_cups",
    train_tournaments = [54, 55, 56, 57],
    cups = :senior_spfl_ties,
    target_tournaments = [56, 57],
    monitor_tournaments = [54, 55],
    clock_tournaments = [56, 57],
    target_seasons = W2_TARGET_SEASONS,
    history_seasons = 2,
    dynamics_col = :match_biweek,
)
const LOWER_SCOPE = Data.DataScope(
    name = "lower",
    train_tournaments = [56, 57],
    cups = :none,
    target_tournaments = [56, 57],
    monitor_tournaments = Int[],
    clock_tournaments = [56, 57],
    target_seasons = W2_TARGET_SEASONS,
    history_seasons = 2,
    dynamics_col = :match_biweek,
)

const DEFAULT_SAMPLER = Samplers.QueuedNUTSConfig(
    n_samples = 1_000,
    n_warmup = 500,
    n_chains = 4,
    accept_rate = 0.65,
    max_depth = 10,
    initialisation = nothing,
    show_progress = false,
    silence_initial_stepsize = true,
)

function w2_model(name::String, dynamics;
                  tiers = nothing, league_deltas::Bool = false)
    builder = CountModelBuilder(Symbol(name)) |>
        add(GlobalInterception()) |>
        add(dynamics) |>
        add(GlobalHomeAdvantage())
    tiers === nothing || add!(builder, tiers)
    add!(builder, make_joint_observation(; league_deltas))
    return build(builder)
end

estimated_tiers(; anchor = :first_in_window) = PyramidTiers(
    anchor = anchor,
    carry = EstimatedCarry(),
    jump = EstimatedJump(),
)

# Exact persisted m12 model recipe (run 132df5c2-c742-4e95-8693-3aeb2b2cbaef),
# reconstructed from configs.model_config and Experiment 06's canonical loader.
function m12_td_reference()
    lineup = PlayerLineupPillar(
        feature = Features.ShotsPlusMinusFeature(
            w_sim = 0.0, λ = 1000.0, half_life_days = 730.0, fit_on = :history),
        aggregation = BenchWeightedPlayerAggregation(w_bench = 0.10),
        w_att_prior = Normal(0.0, 0.3),
        w_def_prior = Normal(0.0, 0.3),
    )
    wealth = ProductionWealthCovariate(
        feature = ProductionWealthFeature(curve = RichardsSigmoid(23.0, 0.80, 2.0)),
        prior = truncated(Normal(0.10, 0.05), lower = 0.0),
    )
    return CountModelBuilder(:m12_joint_hybrid_synergy) |>
        add(GlobalInterception()) |>
        add(TimeDecayDynamics(days_half_life = 180.0)) |>
        add(GlobalHomeAdvantage()) |>
        add(lineup) |>
        add(wealth) |>
        add(make_joint_observation()) |>
        build
end

function build_w2_candidates()
    specs = [
        ("td_base", SPFL_CUPS_SCOPE,
         w2_model("td_base", TimeDecayDynamics(days_half_life = 180.0)), :control,
         "TD pooled-cups joint control."),
        ("td_a2", SPFL_CUPS_SCOPE,
         w2_model("td_a2", TimeDecayDynamics(days_half_life = 180.0);
                  tiers = PyramidTiers(carry = NoCarry(), jump = NoJump())), :candidate,
         "Tier anchor shrinkage without carry or jump; paired base td_base."),
        ("td_a2_carry", SPFL_CUPS_SCOPE,
         w2_model("td_a2_carry", TimeDecayDynamics(days_half_life = 180.0);
                  tiers = PyramidTiers(carry = EstimatedCarry(), jump = NoJump())), :candidate,
         "Tier anchor plus estimated carry; paired base td_base."),
        ("td_a2_carry_jump", SPFL_CUPS_SCOPE,
         w2_model("td_a2_carry_jump", TimeDecayDynamics(days_half_life = 180.0);
                  tiers = estimated_tiers()), :candidate,
         "Tier anchor, carry and persistent transition jump; paired base td_base."),
        ("td_ldelta", SPFL_CUPS_SCOPE,
         w2_model("td_ldelta", TimeDecayDynamics(days_half_life = 180.0);
                  league_deltas = true), :candidate,
         "Attacking-tier intercept and match-competition kappa deltas; paired base td_base."),
        ("td_a2_carry_jump_ldelta", SPFL_CUPS_SCOPE,
         w2_model("td_a2_carry_jump_ldelta", TimeDecayDynamics(days_half_life = 180.0);
                  tiers = estimated_tiers(), league_deltas = true), :candidate,
         "Full TD pyramid plus league deltas; paired base td_base."),
        ("grw_base", SPFL_CUPS_SCOPE,
         w2_model("grw_base", MultiScaleGRW()), :candidate,
         "GRW pooled-cups joint control; semantic control for GRW rows."),
        ("grw_step", SPFL_CUPS_SCOPE,
         w2_model("grw_step", MultiScaleGRW(target_season_step = true)), :candidate,
         "GRW target-season macro step; paired base grw_base."),
        ("grw_step_a2_carry_jump", SPFL_CUPS_SCOPE,
         w2_model("grw_step_a2_carry_jump", MultiScaleGRW(target_season_step = true);
                  tiers = estimated_tiers()), :candidate,
         "GRW step plus pyramid carry/jump; paired base grw_base."),
        ("grw_step_ldelta", SPFL_CUPS_SCOPE,
         w2_model("grw_step_ldelta", MultiScaleGRW(target_season_step = true);
                  league_deltas = true), :candidate,
         "GRW step plus league deltas; paired base grw_base."),
        ("grw_step_a2_carry_jump_ldelta", SPFL_CUPS_SCOPE,
         w2_model("grw_step_a2_carry_jump_ldelta", MultiScaleGRW(target_season_step = true);
                  tiers = estimated_tiers(), league_deltas = true), :candidate,
         "Full GRW pyramid plus league deltas; paired base grw_base."),
        ("td_lower_base", LOWER_SCOPE,
         w2_model("td_lower_base", TimeDecayDynamics(days_half_life = 180.0)), :candidate,
         "Lower-scope TD joint control; semantic control for lower rows."),
        ("td_lower_a2full_carry_jump", LOWER_SCOPE,
         w2_model("td_lower_a2full_carry_jump", TimeDecayDynamics(days_half_life = 180.0);
                  tiers = estimated_tiers(anchor = :full_pyramid)), :candidate,
         "Lower scope with full-pyramid anchor, carry and jump; paired base td_lower_base."),
        ("td_lower_b1lit", LOWER_SCOPE,
         w2_model("td_lower_b1lit", TimeDecayDynamics(days_half_life = 180.0);
                  tiers = PyramidTiers(tier_term = false, anchor = :full_pyramid,
                                       carry = NoCarry(),
                                       jump = FixedTransitionPriors())), :candidate,
         "B1-literal transition priors with no tier term; paired base td_lower_base."),
        ("m12_td_ref", LOWER_SCOPE, m12_td_reference(), :candidate,
         "Exact production m12 recipe rerun on the W2 panel; reported, not gating."),
    ]
    return Harness.Candidate[
        Harness.Candidate(name = name, model = model, scope = scope,
                          sampler = DEFAULT_SAMPLER, role = role, hypothesis = hypothesis)
        for (name, scope, model, role, hypothesis) in specs
    ]
end

const CANDIDATES = build_w2_candidates()

# Loader only: recorded regular goals, penalty attempts/conversion and own-goal receipts.
# BBC referee identities are mapped from fitted fixtures only. Historical appointment
# availability is assumed, not established by publication timestamps in this snapshot.
import BayesianFootball
import Turing
using Turing: @model
import DynamicPPL
import Distributions
import MCMCChains
import DataFrames
import SpecialFunctions
import Statistics
import LinearAlgebra
import LogExpFunctions

if !isdefined(@__MODULE__, :GoalDecompositionIncidentData)
    include(joinpath(@__DIR__, "l08_incident_data.jl"))
end
const GD = BayesianFootball
const GD_PG = GD.Models.PreGame
const GD_Features = GD.Features

"Concrete, immutable prior parameters; all four arms share the same values."
Base.@kwdef struct GoalDecompositionPriors
    regular_log_mean::Float64
    penalty_log_mean::Float64
    own_log_mean::Float64
    conversion_alpha::Float64
    conversion_beta::Float64
    regular_log_sd::Float64 = 0.50
    penalty_log_sd::Float64 = 0.50
    own_log_sd::Float64 = 0.60
    regular_home_sd::Float64 = 0.25
    penalty_home_sd::Float64 = 0.30
    regular_team_sd::Float64 = 0.35
    penalty_team_sd::Float64 = 0.25
    referee_sd::Float64 = 0.35
    own_pressure_sd::Float64 = 0.50
end

abstract type AbstractDecomposedGoalsModel <: GD.TypesInterfaces.AbstractPoissonModel end

# Prototype model families are not `ComposableCountModel`s, so register their
# owned abstract type explicitly as a canonical model recipe.
GD.Training.Inference._truth_config_type(::AbstractDecomposedGoalsModel) = "model"

"Total-score-only control with exactly m01's latent structure and priors."
Base.@kwdef struct RecombinedControlModel{R} <: AbstractDecomposedGoalsModel
    priors::GoalDecompositionPriors
    registry::R
    registry_hash::String
    data_hash::String
    days_half_life::Float64 = 180.0
end

"Global penalty rate plus partially pooled referee effects, flat own-goal rate."
Base.@kwdef struct DecomposedBaselineModel{R} <: AbstractDecomposedGoalsModel
    priors::GoalDecompositionPriors
    registry::R
    registry_hash::String
    data_hash::String
    days_half_life::Float64 = 180.0
end

"Baseline plus separately shrunk team penalty drawing/conceding effects."
Base.@kwdef struct DecomposedTeamPenaltiesModel{R} <: AbstractDecomposedGoalsModel
    priors::GoalDecompositionPriors
    registry::R
    registry_hash::String
    data_hash::String
    days_half_life::Float64 = 180.0
end

"Baseline plus own-goal pressure; deliberately NOT the team-penalty arm."
Base.@kwdef struct DecomposedPressureOwnGoalsModel{R} <: AbstractDecomposedGoalsModel
    priors::GoalDecompositionPriors
    registry::R
    registry_hash::String
    data_hash::String
    days_half_life::Float64 = 180.0
end

function Base.show(io::IO, model::AbstractDecomposedGoalsModel)
    print(io, nameof(typeof(model)), "(snapshot=", model.data_hash,
          ", half_life=", model.days_half_life, ", priors=", model.priors, ")")
end

decomposed_prior_manifest(model::AbstractDecomposedGoalsModel) =
    (; model = string(nameof(typeof(model))), priors = model.priors,
       days_half_life = model.days_half_life, registry_hash = model.registry_hash,
       referee_missing_policy = "zero_effect", prior_scope = "usable_pre_24_25",
       new_team_policy = "declared_slate_identity_prior_only_hierarchical")

"Weak pre-study empirical-Bayes centres; never use 24/25 or later outcomes."
function l08_pre_target_prior_anchor(registry)
    historical = DataFrames.subset(registry.matches,
        :usable_for_components => DataFrames.ByRow(identity),
        :season => DataFrames.ByRow(s -> String(s) in ("20/21", "21/22", "22/23", "23/24")))
    n_sides = 2 * DataFrames.nrow(historical)
    n_sides > 0 || error("no usable pre-24/25 fixtures for the frozen prior anchor")
    regular = sum(historical.non_penalty_non_own_goal_home) + sum(historical.non_penalty_non_own_goal_away)
    awarded = sum(historical.penalty_awarded_home) + sum(historical.penalty_awarded_away)
    converted = sum(historical.penalty_goal_home) + sum(historical.penalty_goal_away)
    own = sum(historical.own_goal_home) + sum(historical.own_goal_away)
    regular > 0 && own > 0 && 0 < converted < awarded || error("unsupported pre-study prior anchor")
    return (; n_matches = DataFrames.nrow(historical), n_sides, regular, awarded,
        converted, own, regular_rate = regular / n_sides, penalty_award_rate = awarded / n_sides,
        own_rate = own / n_sides, conversion_rate = converted / awarded,
        beta_alpha = 10converted / awarded, beta_beta = 10(awarded - converted) / awarded)
end

function l08_models(registry, data_hash::AbstractString; registry_hash::AbstractString)
    actual = GoalDecompositionIncidentData.registry_snapshot_hash(registry)
    actual == registry_hash == data_hash || error("model requires the exact full registry SHA-256")
    anchor = l08_pre_target_prior_anchor(registry)
    priors = GoalDecompositionPriors(regular_log_mean = log(anchor.regular_rate),
        penalty_log_mean = log(anchor.penalty_award_rate), own_log_mean = log(anchor.own_rate),
        conversion_alpha = anchor.beta_alpha, conversion_beta = anchor.beta_beta)
    recipe = (; priors, registry, registry_hash = String(registry_hash), data_hash = String(data_hash))
    return [
        (; name = :m00_recombined_control, model = RecombinedControlModel(; recipe...)),
        (; name = :m01_decomposed_baseline, model = DecomposedBaselineModel(; recipe...)),
        (; name = :m02_decomposed_team_penalties, model = DecomposedTeamPenaltiesModel(; recipe...)),
        (; name = :m03_decomposed_pressure_own_goals, model = DecomposedPressureOwnGoalsModel(; recipe...)),
    ]
end

function l08_expected_params(name::Symbol, n_teams::Integer, n_referees::Integer)
    common = 8 + 2Int(n_teams) + Int(n_referees) # 7 old scalars + referee scale
    name in (:m00_recombined_control, :m01_decomposed_baseline) && return common
    name === :m02_decomposed_team_penalties && return common + 2 + 2Int(n_teams)
    name === :m03_decomposed_pressure_own_goals && return common + 1
    error("unknown Experiment 08 candidate $name")
end

struct GoalDecompositionFeature{R} <: GD_Features.AbstractFeatureConfig
    registry::R
    snapshot_hash::String
end

function GD_Features.required_features(model::AbstractDecomposedGoalsModel)
    return GD_Features.AbstractFeatureConfig[GD_Features.TeamIDsFeature(), GD_Features.GoalsFeature(),
        GD_Features.DatesFeature(), GoalDecompositionFeature(model.registry, model.registry_hash)]
end

function GD_Features.add_feature!(F_data::Dict, feature::GoalDecompositionFeature,
                                 ordered_ids, team_map::Dict, ds::GD.Data.DataStore)
    return l08_add_component_features!(F_data, feature, ordered_ids)
end

function l08_add_component_features!(F_data::Dict, feature::GoalDecompositionFeature, ordered_ids)
    frame = GoalDecompositionIncidentData.model_feature_view(feature.registry, feature.snapshot_hash)
    rows = Dict(Int(row.match_id) => row for row in eachrow(frame))
    ids = Int.(ordered_ids)
    missing_ids = filter(id -> !haskey(rows, id), ids)
    isempty(missing_ids) || error("incident registry lacks fitted match IDs $missing_ids")
    count_column(column) = Float64[getproperty(rows[id], column) for id in ids]
    for (target, source) in (
        (:flat_open_play_home, :flat_non_penalty_non_own_goal_home),
        (:flat_open_play_away, :flat_non_penalty_non_own_goal_away),
        (:flat_penalty_goals_home, :flat_penalty_goals_home),
        (:flat_penalty_goals_away, :flat_penalty_goals_away),
        (:flat_penalty_awarded_home, :flat_penalty_awarded_home),
        (:flat_penalty_awarded_away, :flat_penalty_awarded_away),
        (:flat_own_goals_credited_home, :flat_own_goals_credited_home),
        (:flat_own_goals_credited_away, :flat_own_goals_credited_away))
        F_data[target] = count_column(source)
    end
    F_data[:incident_complete] = Float64[rows[id].component_usable_mask for id in ids]
    F_data[:goal_decomposition_data_hash] = feature.snapshot_hash
    F_data[:goal_decomposition_training_ids] = ids
    F_data[:goal_decomposition_quarantine_reason] = String[rows[id].component_quarantine_reason for id in ids]
    # No global referee vocabulary: an official appears only if a FITTED fixture names them.
    known = sort!(unique(String[rows[id].referee_id for id in ids if rows[id].referee_id != "UNKNOWN"]))
    isempty(known) && error("no known training referees; this study's referee hierarchy requires observed groups")
    referee_map = Dict(ref => index for (index, ref) in enumerate(known))
    F_data[:referee_map] = referee_map
    F_data[:n_referees] = length(known)
    F_data[:flat_referee_ids] = Int[get(referee_map, rows[id].referee_id, 1) for id in ids]
    F_data[:flat_referee_known] = Float64[haskey(referee_map, rows[id].referee_id) for id in ids]
    return nothing
end

"""
Declare upcoming fixture identities without reading their outcomes. A relegated or
promoted team absent from fitted history receives explicit hierarchical latent
parameters, NOT a zero/league-mean plug-in. The user approved this policy after the
canonical audit found six affected fixtures in folds 1 and 21. Historical row IDs
and counts are unchanged; unexpected teams outside this declaration still refuse.
"""
function l08_declare_prediction_teams(feature_sets, oos)
    length(feature_sets) == length(oos) || error("fixture/feature fold count mismatch")
    items = copy(feature_sets.items)
    for (fold, (fs, metadata)) in enumerate(feature_sets)
        d = copy(fs.data)
        teams = copy(d[:team_map])
        identities = DataFrames.select(oos[fold], :match_id, :home_team, :away_team)
        isempty(intersect(Set(d[:goal_decomposition_training_ids]), Set(identities.match_id))) ||
            error("declaration includes fitted fixtures")
        newcomers = sort!(setdiff(unique(vcat(String.(identities.home_team), String.(identities.away_team))), collect(keys(teams))))
        for team in newcomers
            teams[team] = length(teams) + 1
        end
        d[:team_map] = teams
        d[:n_teams] = length(teams)
        d[:goal_decomposition_prior_only_teams] = sort!(union(get(d, :goal_decomposition_prior_only_teams, String[]), newcomers))
        d[:goal_decomposition_declared_oos_ids] = Int.(identities.match_id)
        items[fold] = (typeof(fs)(d), metadata)
    end
    return typeof(feature_sets)(items)
end

@inline _gd_logpoisson(y, eta, logfact) = y .* eta .- exp.(eta) .- logfact
# Evaluate from logits, not log(logistic(x)): finite even when Float64 logistic rounds to 0/1.
@inline _gd_logbinomial(c, a, logbinom, logit) = -c .* LogExpFunctions.log1pexp.(-logit) .-
    (a .- c) .* LogExpFunctions.log1pexp.(logit) .+ logbinom
_gd_component_mask(::RecombinedControlModel, mask) = zero.(mask)
_gd_component_mask(::AbstractDecomposedGoalsModel, mask) = mask

"Independent distribution-object reference used OUTSIDE the AD model."
function decomposed_side_loglikelihood(regular::Int, awarded::Int, converted::Int, own::Int,
                                      eta_regular::Real, eta_penalty::Real, eta_own::Real, k::Real)
    return Distributions.logpdf(Distributions.Poisson(exp(eta_regular)), regular) +
        Distributions.logpdf(Distributions.Poisson(exp(eta_penalty)), awarded) +
        Distributions.logpdf(Distributions.Binomial(awarded, k), converted) +
        Distributions.logpdf(Distributions.Poisson(exp(eta_own)), own)
end

# Exact changes of variables, not replacement priors: log-scale standard normals
# are corrected to HalfNormal(sd); a logit standard normal is corrected to Beta.
# This avoids Vector{TrackedReal} produced by length-one product distributions.
_gd_halfnormal_correction(raw) = sum(raw) - sum(exp.(raw) .* exp.(raw)) / 2 +
    sum(raw .* raw) / 2 + log(2.0)
_gd_beta_logit_correction(raw, alpha, beta) =
    -alpha * sum(LogExpFunctions.log1pexp.(-raw)) - beta * sum(LogExpFunctions.log1pexp.(raw)) -
    SpecialFunctions.logbeta(alpha, beta) + sum(raw .* raw) / 2 + log(2pi) / 2

@model function _gd_penalty_shifts(::AbstractDecomposedGoalsModel, n_teams::Int, projection, scale)
    return nothing
end
@model function _gd_penalty_shifts(model::DecomposedTeamPenaltiesModel, n_teams::Int, projection, scale)
    sigma_att_raw ~ Turing.filldist(Distributions.Normal(), 1)
    sigma_def_raw ~ Turing.filldist(Distributions.Normal(), 1)
    raw_att ~ Turing.filldist(Distributions.Normal(), n_teams)
    raw_def ~ Turing.filldist(Distributions.Normal(), n_teams)
    sigma_att = exp.(sigma_att_raw) .* scale
    sigma_def = exp.(sigma_def_raw) .* scale
    Turing.@addlogprob! _gd_halfnormal_correction(sigma_att_raw) + _gd_halfnormal_correction(sigma_def_raw)
    return (; att = (projection * raw_att) .* sigma_att,
              def = (projection * raw_def) .* sigma_def)
end
_gd_penalty_eta(::AbstractDecomposedGoalsModel, eta, _, _, _, _) = eta
_gd_penalty_eta(::DecomposedTeamPenaltiesModel, eta, p, h, a, ::Val{:home}) = eta .+ p.att[h] .+ p.def[a]
_gd_penalty_eta(::DecomposedTeamPenaltiesModel, eta, p, h, a, ::Val{:away}) = eta .+ p.att[a] .+ p.def[h]

@model function _gd_own_pressure(::AbstractDecomposedGoalsModel)
    return 0.0
end
@model function _gd_own_pressure(model::DecomposedPressureOwnGoalsModel)
    pressure ~ Turing.filldist(Distributions.Normal(0, model.priors.own_pressure_sd), 1)
    return pressure
end
_gd_own_eta(::AbstractDecomposedGoalsModel, mu_own, _, _, _) = mu_own
_gd_own_eta(::DecomposedPressureOwnGoalsModel, mu_own, eta_regular, mu_regular, pressure) =
    mu_own .+ pressure .* (eta_regular .- mu_regular)

# Scalar priors are length-one arrays deliberately: mixing TrackedReal scalars into
# fused broadcasts selects ReverseDiff's allocating Tracker fallback. Vector priors
# plus constant centering projections keep every hot broadcast on TrackedArray.
# All masks, counts and factorial constants are constructed before taping. No
# parameter-dependent guards/clamps or Binomial objects occur in this engine.
@model function decomposed_goals_engine(z, model::AbstractDecomposedGoalsModel)
    mu_regular ~ Turing.filldist(Distributions.Normal(model.priors.regular_log_mean, model.priors.regular_log_sd), 1)
    ha_regular ~ Turing.filldist(Distributions.Normal(0, model.priors.regular_home_sd), 1)
    sigma_regular_raw ~ Turing.filldist(Distributions.Normal(), 1)
    raw_attack ~ Turing.filldist(Distributions.Normal(), z.n_teams)
    raw_defence ~ Turing.filldist(Distributions.Normal(), z.n_teams)
    mu_penalty ~ Turing.filldist(Distributions.Normal(model.priors.penalty_log_mean, model.priors.penalty_log_sd), 1)
    ha_penalty ~ Turing.filldist(Distributions.Normal(0, model.priors.penalty_home_sd), 1)
    mu_own ~ Turing.filldist(Distributions.Normal(model.priors.own_log_mean, model.priors.own_log_sd), 1)
    conversion_raw ~ Turing.filldist(Distributions.Normal(), 1)
    sigma_referee_raw ~ Turing.filldist(Distributions.Normal(), 1)
    raw_referee ~ Turing.filldist(Distributions.Normal(), z.n_referees)
    penalty ~ DynamicPPL.to_submodel(_gd_penalty_shifts(model, z.n_teams, z.team_projection, z.penalty_scale), false)
    own_pressure ~ DynamicPPL.to_submodel(_gd_own_pressure(model), false)

    sigma_regular = exp.(sigma_regular_raw) .* z.regular_scale
    sigma_referee = exp.(sigma_referee_raw) .* z.referee_scale
    conversion = LogExpFunctions.logistic.(conversion_raw)
    Turing.@addlogprob! _gd_halfnormal_correction(sigma_regular_raw) + _gd_halfnormal_correction(sigma_referee_raw)
    Turing.@addlogprob! _gd_beta_logit_correction(conversion_raw, model.priors.conversion_alpha, model.priors.conversion_beta)

    attack = (z.team_projection * raw_attack) .* sigma_regular
    defence = (z.team_projection * raw_defence) .* sigma_regular
    referee = (z.referee_projection * raw_referee) .* sigma_referee
    referee_shift = referee[z.referee_ids] .* z.referee_known
    eta_regular_h = mu_regular .+ ha_regular .+ attack[z.h] .+ defence[z.a]
    eta_regular_a = mu_regular .+ attack[z.a] .+ defence[z.h]
    eta_penalty_h = _gd_penalty_eta(model, mu_penalty .+ ha_penalty .+ referee_shift, penalty, z.h, z.a, Val(:home))
    eta_penalty_a = _gd_penalty_eta(model, mu_penalty .+ referee_shift, penalty, z.h, z.a, Val(:away))
    eta_own_h = _gd_own_eta(model, mu_own, eta_regular_h, mu_regular, own_pressure)
    eta_own_a = _gd_own_eta(model, mu_own, eta_regular_a, mu_regular, own_pressure)

    classified_h = _gd_logpoisson(z.regular_h, eta_regular_h, z.lf_regular_h) .+
        _gd_logpoisson(z.awarded_h, eta_penalty_h, z.lf_awarded_h) .+
        _gd_logbinomial(z.converted_h, z.awarded_h, z.lbc_h, conversion_raw) .+
        _gd_logpoisson(z.own_h, eta_own_h, z.lf_own_h)
    classified_a = _gd_logpoisson(z.regular_a, eta_regular_a, z.lf_regular_a) .+
        _gd_logpoisson(z.awarded_a, eta_penalty_a, z.lf_awarded_a) .+
        _gd_logbinomial(z.converted_a, z.awarded_a, z.lbc_a, conversion_raw) .+
        _gd_logpoisson(z.own_a, eta_own_a, z.lf_own_a)
    total_h = exp.(eta_regular_h) .+ conversion .* exp.(eta_penalty_h) .+ exp.(eta_own_h)
    total_a = exp.(eta_regular_a) .+ conversion .* exp.(eta_penalty_a) .+ exp.(eta_own_a)
    total_ll_h = z.total_h .* log.(total_h) .- total_h .- z.lf_total_h
    total_ll_a = z.total_a .* log.(total_a) .- total_a .- z.lf_total_a
    Turing.@addlogprob! sum(classified_h .* z.component_weights) + sum(classified_a .* z.component_weights)
    Turing.@addlogprob! sum(total_ll_h .* z.total_weights) + sum(total_ll_a .* z.total_weights)
end

function gd_design(model::AbstractDecomposedGoalsModel, fs)
    d = fs.data
    get(d, :goal_decomposition_data_hash, nothing) == model.data_hash || error("missing/mismatched incident snapshot hash")
    count_vector(key) = begin
        values = Float64.(d[key])
        all(x -> isfinite(x) && x >= 0 && isinteger(x), values) || error("$key contains invalid counts")
        values  # Float64 counts avoid mixed Int/Float boxing in the compiled broadcast kernel.
    end
    h, a = Int.(d[:flat_home_ids]), Int.(d[:flat_away_ids])
    regular_h, regular_a = count_vector(:flat_open_play_home), count_vector(:flat_open_play_away)
    awarded_h, awarded_a = count_vector(:flat_penalty_awarded_home), count_vector(:flat_penalty_awarded_away)
    converted_h, converted_a = count_vector(:flat_penalty_goals_home), count_vector(:flat_penalty_goals_away)
    own_h, own_a = count_vector(:flat_own_goals_credited_home), count_vector(:flat_own_goals_credited_away)
    total_h, total_a = count_vector(:flat_home_goals), count_vector(:flat_away_goals)
    complete = copy(Float64.(d[:incident_complete]))
    all(x -> x in (0.0, 1.0), complete) || error("component mask must be exactly binary")
    conserved = (regular_h .+ converted_h .+ own_h .== total_h) .& (regular_a .+ converted_a .+ own_a .== total_a)
    mismatches = findall((complete .== 1.0) .& .!conserved)
    isempty(mismatches) || error("registry/cache score mismatch on usable fitted rows $mismatches; refresh or audit, never silently downgrade the component mask")
    complete = _gd_component_mask(model, complete)
    all(converted_h .<= awarded_h) && all(converted_a .<= awarded_a) || error("converted count exceeds attempts")
    weights = 0.5 .^ (Float64.(d[:dates]) ./ model.days_half_life)
    referee_ids = Int.(d[:flat_referee_ids])
    referee_known = Float64.(d[:flat_referee_known])
    n_teams, n_referees = Int(d[:n_teams]), Int(d[:n_referees])
    all(x -> x in (0.0, 1.0), referee_known) || error("referee mask must be binary")
    all(x -> 1 <= x <= n_referees, referee_ids) || error("invalid fitted referee index")
    all(x -> 1 <= x <= n_teams, vcat(h, a)) || error("invalid fitted team index")
    all(x -> isfinite(x) && 0 < x <= 1, weights) || error("invalid chronological likelihood weights")
    vectors = (h, a, regular_h, regular_a, awarded_h, awarded_a, converted_h, converted_a,
        own_h, own_a, total_h, total_a, complete, weights, referee_ids, referee_known)
    all(v -> length(v) == length(h), vectors) || error("component feature lengths differ")
    lf(y) = SpecialFunctions.loggamma.(y .+ 1.0)
    lbc(c, n) = lf(n) .- lf(c) .- lf(n .- c)
    team_projection = Matrix{Float64}(LinearAlgebra.I, n_teams, n_teams) .- (1 / n_teams)
    referee_projection = Matrix{Float64}(LinearAlgebra.I, n_referees, n_referees) .- (1 / n_referees)
    component_weights = weights .* complete
    total_weights = weights .* (1.0 .- complete)
    return (; h, a, regular_h, regular_a, awarded_h, awarded_a, converted_h, converted_a,
        own_h, own_a, total_h, total_a, complete, weights, referee_ids, referee_known, n_teams, n_referees,
        team_projection, referee_projection, component_weights, total_weights,
        regular_scale = [model.priors.regular_team_sd], referee_scale = [model.priors.referee_sd],
        penalty_scale = [model.priors.penalty_team_sd],
        lf_regular_h = lf(regular_h), lf_regular_a = lf(regular_a),
        lf_awarded_h = lf(awarded_h), lf_awarded_a = lf(awarded_a),
        lf_own_h = lf(own_h), lf_own_a = lf(own_a), lf_total_h = lf(total_h), lf_total_a = lf(total_a),
        lbc_h = lbc(converted_h, awarded_h), lbc_a = lbc(converted_a, awarded_a))
end

GD_PG.build_turing_model(model::AbstractDecomposedGoalsModel, fs) = decomposed_goals_engine(gd_design(model, fs), model)

function GD_PG.extract_parameters(model::AbstractDecomposedGoalsModel,
                                 fixtures::DataFrames.AbstractDataFrame, fs, chain::MCMCChains.Chains)
    d = fs.data
    get(d, :goal_decomposition_data_hash, nothing) == model.data_hash || error("extraction snapshot mismatch")
    teams = d[:team_map]
    unknown = ["$(r.match_id): $(r.home_team) v $(r.away_team)" for r in eachrow(fixtures)
               if !haskey(teams, r.home_team) || !haskey(teams, r.away_team)]
    isempty(unknown) || error("REFUSED unseen-team fixtures: " * join(unknown, "; "))
    draw(name) = vec(Array(chain[Symbol(name)]))
    scalar_draw(name) = draw("$name[1]")
    matrix_draws(name, n) = hcat([draw("$name[$i]") for i in 1:n]...)
    centered(x) = x .- Statistics.mean(x; dims = 2)
    mu_r, ha_r = scalar_draw("mu_regular"), scalar_draw("ha_regular")
    mu_p, ha_p, mu_o = scalar_draw("mu_penalty"), scalar_draw("ha_penalty"), scalar_draw("mu_own")
    k = LogExpFunctions.logistic.(scalar_draw("conversion_raw"))
    sigma_regular = model.priors.regular_team_sd .* exp.(scalar_draw("sigma_regular_raw"))
    sigma_referee = model.priors.referee_sd .* exp.(scalar_draw("sigma_referee_raw"))
    attack = centered(matrix_draws("raw_attack", d[:n_teams])) .* sigma_regular
    defence = centered(matrix_draws("raw_defence", d[:n_teams])) .* sigma_regular
    referee = centered(matrix_draws("raw_referee", d[:n_referees])) .* sigma_referee
    penalty_att = model isa DecomposedTeamPenaltiesModel ? centered(matrix_draws("raw_att", d[:n_teams])) .* (model.priors.penalty_team_sd .* exp.(scalar_draw("sigma_att_raw"))) : nothing
    penalty_def = model isa DecomposedTeamPenaltiesModel ? centered(matrix_draws("raw_def", d[:n_teams])) .* (model.priors.penalty_team_sd .* exp.(scalar_draw("sigma_def_raw"))) : nothing
    pressure = model isa DecomposedPressureOwnGoalsModel ? scalar_draw("pressure") : 0.0
    referee_ids = Dict(Int(r.match_id) => String(r.referee_id) for r in eachrow(model.registry.matches))
    output = Dict{Int, NamedTuple}()
    for row in eachrow(fixtures)
        hi, ai = teams[row.home_team], teams[row.away_team]
        ref_id = get(referee_ids, Int(row.match_id), "UNKNOWN")
        ri = get(d[:referee_map], ref_id, 0)
        ref_shift = ri == 0 ? 0.0 : referee[:, ri]
        eta_rh = mu_r .+ ha_r .+ attack[:, hi] .+ defence[:, ai]
        eta_ra = mu_r .+ attack[:, ai] .+ defence[:, hi]
        eta_ph = mu_p .+ ha_p .+ ref_shift
        eta_pa = mu_p .+ ref_shift
        if penalty_att !== nothing
            eta_ph = eta_ph .+ penalty_att[:, hi] .+ penalty_def[:, ai]
            eta_pa = eta_pa .+ penalty_att[:, ai] .+ penalty_def[:, hi]
        end
        eta_oh = mu_o .+ pressure .* (eta_rh .- mu_r)
        eta_oa = mu_o .+ pressure .* (eta_ra .- mu_r)
        lambda_h = exp.(eta_rh) .+ k .* exp.(eta_ph) .+ exp.(eta_oh)
        lambda_a = exp.(eta_ra) .+ k .* exp.(eta_pa) .+ exp.(eta_oa)
        all(x -> isfinite(x) && x > 0, lambda_h) && all(x -> isfinite(x) && x > 0, lambda_a) ||
            error("non-finite/nonpositive total-rate draw for match $(row.match_id)")
        output[Int(row.match_id)] = (; λ_h = lambda_h, λ_a = lambda_a)
    end
    return output
end

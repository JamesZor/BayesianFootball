# ==============================================================================
# 01 — Scottish cross-tier time-decay models (Stage 1)
# ==============================================================================
#
# Definitions only.  Stage 1 deliberately contains only the Poisson control and
# all-SPFL league-environment candidate.  A2/B1/B2 remain unimplemented until
# their identifiability and prior contracts are approved.
#
# Filtration contract for `CrossTierLowerCV`:
#   * held-out fixtures are exactly the Scottish Lower (56/57) calendar bins;
#   * every fitted all-SPFL fixture kicks off strictly before that held-out bin;
#   * all-SPFL rows are restricted to the same selected seasons as the lower
#     control boundary, so pooling does not silently widen the history window.
#
# The model is a time-decayed, independent-Poisson goals model.  Time decay is a
# likelihood weight, not a state evolution: no GRW term is present.
# ==============================================================================

using Turing

if !isdefined(@__MODULE__, :CrossTierPoissonModel)

import BayesianFootball
import BayesianFootball.Data
import BayesianFootball.Features
import BayesianFootball.Models.PreGame
import BayesianFootball.Training
import BayesianFootball.Calibration
import DataFrames
import Dates
import Distributions
import MCMCChains
import SpecialFunctions
import Statistics

const CT_TI = BayesianFootball.TypesInterfaces

"Common Stage-1 model fields; concrete types select their likelihood design."
abstract type CrossTierPoissonModel <: CT_TI.AbstractPoissonModel end

"Scottish Lower control: flat, zero-sum attack/defence priors and 180-day decay."
Base.@kwdef struct CTM00Control <: CrossTierPoissonModel
    days_half_life::Float64 = 180.0
    intercept_prior::Distributions.ContinuousUnivariateDistribution = Distributions.Normal(0.2, 0.2)
    home_prior::Distributions.ContinuousUnivariateDistribution = Distributions.Normal(0.2, 0.2)
    attack_scale_prior::Distributions.ContinuousUnivariateDistribution = Distributions.Gamma(2.0, 0.15)
    defence_scale_prior::Distributions.ContinuousUnivariateDistribution = Distributions.Gamma(2.0, 0.15)
end

"All-SPFL A1: shared club ratings plus zero-sum, symmetric tier scoring offsets."
Base.@kwdef struct CTM01LeagueOffsets <: CrossTierPoissonModel
    days_half_life::Float64 = 180.0
    intercept_prior::Distributions.ContinuousUnivariateDistribution = Distributions.Normal(0.2, 0.2)
    home_prior::Distributions.ContinuousUnivariateDistribution = Distributions.Normal(0.2, 0.2)
    attack_scale_prior::Distributions.ContinuousUnivariateDistribution = Distributions.Gamma(2.0, 0.15)
    defence_scale_prior::Distributions.ContinuousUnivariateDistribution = Distributions.Gamma(2.0, 0.15)
    league_scale_prior::Distributions.ContinuousUnivariateDistribution = Distributions.truncated(Distributions.Normal(0.0, 0.20), lower = 0.0)
end

"A2 feature: each club's earliest observed all-SPFL tier in the fitted training window."
struct CTFirstObservedTierFeature <: Features.AbstractFeatureConfig end

"All-SPFL A2: ordered tier-strength prior locations, anchored at League Two."
Base.@kwdef struct CTM02HierarchicalTiers <: CrossTierPoissonModel
    days_half_life::Float64 = 180.0
    intercept_prior::Distributions.ContinuousUnivariateDistribution = Distributions.Normal(0.2, 0.2)
    home_prior::Distributions.ContinuousUnivariateDistribution = Distributions.Normal(0.2, 0.2)
    attack_scale_prior::Distributions.ContinuousUnivariateDistribution = Distributions.Gamma(2.0, 0.15)
    defence_scale_prior::Distributions.ContinuousUnivariateDistribution = Distributions.Gamma(2.0, 0.15)
    tier_step_prior::Distributions.ContinuousUnivariateDistribution = Distributions.truncated(Distributions.Normal(0.0, 0.35), lower = 0.0)
end

"Lower-only likelihood boundaries with all-SPFL history available for cold-start priors."
Base.@kwdef struct CrossTierPriorCV <: Data.AbstractSplitter
    target_seasons::Vector{String} = ["24/25", "25/26"]
    history_seasons::Int = 2
    dynamics_col::Symbol = :match_biweek
    warmup_period::Int = 0
    end_dynamics::Union{Int, Nothing} = nothing
end

"Fitted-history-only B1 prior locations and a missing-history fallback audit count."
struct CTStructuralPriorFeature <: Features.AbstractFeatureConfig
    relegated_attack_offset::Float64
end

"Fitted-history-only B2 last-five higher-tier market locations and fallback audit count."
struct CTMarketPriorFeature <: Features.AbstractFeatureConfig end

"B1: no promoted offset; cold relegated clubs receive fixed ±0.90 prior locations."
Base.@kwdef struct CTM03StructuralOffset <: CrossTierPoissonModel
    days_half_life::Float64 = 180.0
    intercept_prior::Distributions.ContinuousUnivariateDistribution = Distributions.Normal(0.2, 0.2)
    home_prior::Distributions.ContinuousUnivariateDistribution = Distributions.Normal(0.2, 0.2)
    attack_scale_prior::Distributions.ContinuousUnivariateDistribution = Distributions.Gamma(2.0, 0.15)
    defence_scale_prior::Distributions.ContinuousUnivariateDistribution = Distributions.Gamma(2.0, 0.15)
    relegated_attack_offset::Float64 = 0.90
end

"B2: cold-team initial locations from last-five prior-season higher-tier market goal difference."
Base.@kwdef struct CTM04MarketDerived <: CrossTierPoissonModel
    days_half_life::Float64 = 180.0
    intercept_prior::Distributions.ContinuousUnivariateDistribution = Distributions.Normal(0.2, 0.2)
    home_prior::Distributions.ContinuousUnivariateDistribution = Distributions.Normal(0.2, 0.2)
    attack_scale_prior::Distributions.ContinuousUnivariateDistribution = Distributions.Gamma(2.0, 0.15)
    defence_scale_prior::Distributions.ContinuousUnivariateDistribution = Distributions.Gamma(2.0, 0.15)
end

"A lower-boundary wrapper that pools historical all-SPFL rows without changing OOS fixtures."
Base.@kwdef struct CrossTierLowerCV <: Data.AbstractSplitter
    target_seasons::Vector{String} = ["24/25", "25/26"]
    history_seasons::Int = 2
    dynamics_col::Symbol = :match_biweek
    warmup_period::Int = 0
    end_dynamics::Union{Int, Nothing} = nothing
end

"The standard lower-only grouped splitter used by m00."
function scottish_lower_splitter(; target_seasons::Vector{String} = ["24/25", "25/26"],
                                history_seasons::Int = 2,
                                dynamics_col::Symbol = :match_biweek,
                                warmup_period::Int = 0,
                                end_dynamics::Union{Int, Nothing} = nothing)
    return Data.GroupedCVConfig(
        tournament_groups = [[56, 57]],
        target_seasons = target_seasons,
        history_seasons = history_seasons,
        dynamics_col = dynamics_col,
        warmup_period = warmup_period,
        end_dynamics = end_dynamics,
    )
end

"Construct cross-tier boundaries with lower-only held-out fixtures and a strict kickoff cutoff."
function Data.create_id_boundaries(ds::Data.DataStore, splitter::CrossTierLowerCV)
    lower = scottish_lower_splitter(
        target_seasons = splitter.target_seasons,
        history_seasons = splitter.history_seasons,
        dynamics_col = splitter.dynamics_col,
        warmup_period = splitter.warmup_period,
        end_dynamics = splitter.end_dynamics,
    )
    lower_boundaries = Data.create_id_boundaries(ds, lower)
    out = Vector{Tuple{Data.SplitBoundary, Data.GroupedSplitMetaData}}()

    for (boundary, meta) in lower_boundaries
        heldout = Data.get_next_matches(ds, (boundary, meta), lower)
        DataFrames.nrow(heldout) > 0 || continue
        any(ismissing, heldout.match_date) && error("CrossTierLowerCV held-out fixtures have missing match_date")
        cutoff = minimum(Dates.Date.(heldout.match_date))

        lower_fitted = Set(vcat(boundary.history_match_ids, boundary.target_match_ids))
        fitted_rows = [Int(id) in lower_fitted for id in ds.matches.match_id]
        selected_seasons = Set(String.(ds.matches[fitted_rows, :season]))
        has_missing_dates = any(ismissing, ds.matches.match_date)
        has_missing_dates && error("CrossTierLowerCV source matches have missing match_date")
        eligible = (in.(ds.matches.tournament_id, Ref([54, 55, 56, 57])) .&
                    in.(String.(ds.matches.season), Ref(selected_seasons)) .&
                    (Dates.Date.(ds.matches.match_date) .< cutoff))
        history_ids = Int.(ds.matches[eligible, :match_id])
        isempty(history_ids) && error("CrossTierLowerCV built an empty all-SPFL history before $cutoff")

        # The all-tier history is complete by cutoff; putting it entirely in the history
        # arm prevents the feature builder from treating any source outcome as OOS data.
        pooled = Data.SplitBoundary(boundary.fold_id, boundary.target_step, history_ids, Int[])
        push!(out, (pooled, meta))
    end
    isempty(out) && error("CrossTierLowerCV produced no predictive lower-tier boundaries")
    return out
end

function Data.create_id_boundaries(ds::Data.DataStore, splitter::CrossTierPriorCV)
    lower = scottish_lower_splitter(target_seasons = splitter.target_seasons,
                                    history_seasons = splitter.history_seasons,
                                    dynamics_col = splitter.dynamics_col,
                                    warmup_period = splitter.warmup_period,
                                    end_dynamics = splitter.end_dynamics)
    return Data.create_id_boundaries(ds, lower)
end

function Data.get_next_matches(ds::Data.DataStore, meta::Data.GroupedSplitMetaData,
                               splitter::CrossTierPriorCV)
    lower = scottish_lower_splitter(target_seasons = splitter.target_seasons,
                                    history_seasons = splitter.history_seasons,
                                    dynamics_col = splitter.dynamics_col,
                                    warmup_period = splitter.warmup_period,
                                    end_dynamics = splitter.end_dynamics)
    return Data.get_next_matches(ds, meta, lower)
end

"Add next-bin fixture identities to the fitted vocabulary without adding their outcomes."
function ct_attach_prediction_roster!(feature_set, ds::Data.DataStore, meta,
                                      splitter::Union{CrossTierLowerCV, CrossTierPriorCV}, model)
    fixtures = Data.get_next_matches(ds, meta, splitter)
    DataFrames.nrow(fixtures) > 0 || error("cross-tier fold has no held-out lower fixtures")
    meta.target_season in splitter.target_seasons ||
        error("cross-tier fold target season $(meta.target_season) is outside splitter contract")
    any(ismissing, fixtures.match_date) && error("cross-tier held-out fixtures have missing match_date")
    roster = sort!(unique(vcat(String.(fixtures.home_team), String.(fixtures.away_team))))
    team_map = feature_set.data[:team_map]
    for club in roster
        haskey(team_map, club) || (team_map[club] = length(team_map) + 1)
    end
    feature_set.data[:n_teams] = length(team_map)
    feature_set.data[:prediction_cutoff] = minimum(Dates.Date.(fixtures.match_date))
    feature_set.data[:prediction_target_season] = meta.target_season
    # Rebuild anchor/prior features after roster expansion. Unknown clubs may have
    # zero likelihood rows, but never receive an invented tier or silent zero prior.
    for config in Features.required_features(model)
        config isa Union{CTFirstObservedTierFeature, CTStructuralPriorFeature,
                         CTMarketPriorFeature} || continue
        Features.add_feature!(feature_set.data, config,
                              feature_set.data[:ordered_match_ids], team_map, ds)
    end
    return feature_set
end

"Summarise fixed prior inputs for the smoke filtration audit."
function ct_feature_report(feature_set)
    data = feature_set.data
    return (; n_teams = Int(data[:n_teams]),
              cold_count = count(==(1.0), get(data, :cold_team_mask, Float64[])),
              relegated_count = get(data, :relegated_cold_count, 0),
              prior_lower_only_count = get(data, :prior_lower_only_cold_count, 0),
              structural_fallback_count = get(data, :structural_prior_fallback_count, 0),
              market_fallback_count = get(data, :market_prior_fallback_count, 0),
              market_candidate_match_ids = get(data, :market_prior_candidate_match_ids, Int[]),
              prior_audit = get(data, :prior_audit, Dict{String, NamedTuple}()),
              tier_anchor_counts = get(data, :tier_anchor_counts, Int[]),
              cross_anchor_fixture_count = get(data, :cross_anchor_fixture_count, 0))
end

"Exact smoke API: builds one vector feature collection fold, then attaches the OOS roster/prior data."
function cross_tier_smoke_prepare_one_fold(candidate, ds::Data.DataStore; fold::Int = 1)
    splitter = candidate.splitter
    model = candidate.model
    boundaries = Data.create_id_boundaries(ds, splitter)
    fold <= length(boundaries) || error("requested fold $fold but splitter produced $(length(boundaries)) folds")
    boundary, meta = boundaries[fold]
    feature_set = Features.create_features(boundary, ds, model, splitter.dynamics_col)
    ct_attach_prediction_roster!(feature_set, ds, meta, splitter, model)
    oos = Data.get_next_matches(ds, meta, splitter)
    cutoff = feature_set.data[:prediction_cutoff]
    fitted_ids = Int.(vcat(boundary.history_match_ids, boundary.target_match_ids))
    heldout_ids = Int.(oos.match_id)
    isempty(intersect(fitted_ids, heldout_ids)) || error("smoke fold fitted/held-out IDs overlap")
    return (; boundary, meta, feature_set, oos, cutoff, fitted_ids, heldout_ids,
              prior_metrics = ct_feature_report(feature_set))
end

function Features.create_features(splits::Vector{<:Tuple{Data.SplitBoundary,<:Any}},
                                  ds::Data.DataStore, model::CrossTierPoissonModel,
                                  splitter::Union{CrossTierLowerCV, CrossTierPriorCV})
    items = [(let fs = Features.create_features(boundary, ds, model, splitter.dynamics_col)
                  ct_attach_prediction_roster!(fs, ds, meta, splitter, model)
                  (fs, meta)
              end) for (boundary, meta) in splits]
    return CT_TI.FeatureCollection(items)
end

function Data.get_next_matches(ds::Data.DataStore,
                               meta::Data.GroupedSplitMetaData,
                               splitter::CrossTierLowerCV)
    lower = scottish_lower_splitter(
        target_seasons = splitter.target_seasons,
        history_seasons = splitter.history_seasons,
        dynamics_col = splitter.dynamics_col,
        warmup_period = splitter.warmup_period,
        end_dynamics = splitter.end_dynamics,
    )
    return Data.get_next_matches(ds, meta, lower)
end

"The features shared by both Stage-1 models."
function Features.required_features(::CTM00Control)
    return Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(), Features.GoalsFeature(), Features.DatesFeature(),
    ]
end

function Features.required_features(::CTM01LeagueOffsets)
    return Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(), Features.GoalsFeature(), Features.DatesFeature(),
        Features.LeagueFeature(),
    ]
end

function Features.required_features(model::CTM03StructuralOffset)
    return Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(), Features.GoalsFeature(), Features.DatesFeature(),
        CTStructuralPriorFeature(model.relegated_attack_offset),
    ]
end

function Features.required_features(::CTM04MarketDerived)
    return Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(), Features.GoalsFeature(), Features.DatesFeature(),
        CTMarketPriorFeature(),
    ]
end

function Features.required_features(::CTM02HierarchicalTiers)
    return Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(), Features.GoalsFeature(), Features.DatesFeature(),
        Features.LeagueFeature(), CTFirstObservedTierFeature(),
    ]
end

"Frozen lower-scope cold indicator: a club has fewer than five fitted 56/57 likelihood rows."
"Historical lower-row counts, including a zero count for a prediction-only entrant."
function ct_cold_mask(F_data::Dict, team_map::Dict)
    counts = zeros(Int, length(team_map))
    fitted_ids = union(F_data[:history_match_ids], F_data[:target_match_ids])
    # The training likelihood contains BOTH split arms; target rows are fitted
    # observations, while only the next bin is held out.
    home_ids = Vector{Int}(F_data[:flat_home_ids])
    away_ids = Vector{Int}(F_data[:flat_away_ids])
    for i in eachindex(home_ids)
        Int(F_data[:ordered_match_ids][i]) in fitted_ids || continue
        counts[home_ids[i]] += 1
        counts[away_ids[i]] += 1
    end
    return counts .< 5
end

"The season immediately before `prediction_target_season`, or `nothing` at series start."
function ct_prior_season(ds::Data.DataStore, target_season::String)
    seasons = sort(unique(String.(ds.matches.season)))
    index = findfirst(==(target_season), seasons)
    return isnothing(index) || index == 1 ? nothing : seasons[index - 1]
end

"Store B1 locations from prior-season, higher-tier evidence strictly before the prediction cutoff."
function Features.add_feature!(F_data::Dict, feature::CTStructuralPriorFeature,
                               ordered_ids, team_map::Dict, ds::Data.DataStore)
    haskey(F_data, :prediction_cutoff) || return nothing
    cutoff = F_data[:prediction_cutoff]
    cold = ct_cold_mask(F_data, team_map)
    prior = zeros(Float64, length(team_map))
    relegated = 0
    prior_lower_only = 0
    audit = Dict{String, NamedTuple}()
    prior_season = ct_prior_season(ds, F_data[:prediction_target_season])
    for (club, index) in team_map
        cold[index] || continue
        if isnothing(prior_season)
            audit[club] = (; status = :fallback, prior_season = nothing,
                           evidence_match_ids = Int[], reason = :no_prior_season)
            continue
        end
        prior_rows = DataFrames.subset(ds.matches,
            [:season, :tournament_id, :match_date, :home_team, :away_team] => DataFrames.ByRow(
                (s, t, d, h, a) -> s == prior_season && !ismissing(d) && Dates.Date(d) < cutoff &&
                                  (h == club || a == club)))
        if isempty(prior_rows)
            audit[club] = (; status = :fallback, prior_season, evidence_match_ids = Int[],
                           reason = :no_prior_season_fixture)
            continue
        end
        evidence_ids = Int.(prior_rows.match_id)
        tiers = unique(Int.(prior_rows.tournament_id))
        # Relegation is an immediately prior-season upper-tier membership. A club
        # observed only in lower tiers is a promotion/stayer and gets exactly zero.
        if any(t -> t in (54, 55), tiers)
            prior[index] = feature.relegated_attack_offset
            relegated += 1
            audit[club] = (; status = :relegated, prior_season, evidence_match_ids = evidence_ids,
                           reason = :upper_tier_prior_season)
        elseif any(t -> t in (56, 57), tiers)
            prior_lower_only += 1
            audit[club] = (; status = :lower_only, prior_season, evidence_match_ids = evidence_ids,
                           reason = :no_promoted_offset)
        else
            audit[club] = (; status = :fallback, prior_season, evidence_match_ids = evidence_ids,
                           reason = :non_spfl_prior_season)
        end
    end
    F_data[:cold_team_mask] = Float64.(cold)
    F_data[:team_prior_attack_location] = prior
    F_data[:relegated_cold_count] = relegated
    F_data[:prior_lower_only_cold_count] = prior_lower_only
    F_data[:prior_audit] = audit
    F_data[:structural_prior_fallback_count] = count(cold .& (prior .== 0.0))
    return nothing
end

"Fit one complete closing 1X2 independent-Poisson inversion; totals-only books are refused."
function ct_market_goal_difference(odds::DataFrames.AbstractDataFrame)::Union{Nothing, Float64}
    DataFrames.nrow(odds) == 0 && return nothing
    one_x_two = DataFrames.subset(odds,
        [:selection, :prob_fair_close] => DataFrames.ByRow(
            (selection, probability) -> selection in (:home, :draw, :away) && !ismissing(probability)))
    present = Set(Symbol.(one_x_two.selection))
    present == Set((:home, :draw, :away)) || return nothing
    probabilities = Dict(Symbol(row.selection) => Float64(row.prob_fair_close) for row in eachrow(one_x_two))
    all(p -> isfinite(p) && 0.0 < p < 1.0, values(probabilities)) || return nothing
    # Calibration's accepted gate contributes convergence, residual and rate-bounds;
    # the complete-1X2 guard above closes T015's totals-only loophole.
    complete = DataFrames.subset(odds, :prob_fair_close => DataFrames.ByRow(!ismissing))
    match_id = Int(first(complete.match_id))
    rate = get(Calibration.invert_market_rates(complete; match_ids = [match_id]), match_id, nothing)
    (rate === nothing || !rate.accepted) && return nothing
    difference = rate.lambda_home - rate.lambda_away
    isfinite(difference) || error("market inversion returned non-finite goal difference")
    return difference
end

"B2 last-five prior-season, higher-tier closing-market goal-difference locations."
function Features.add_feature!(F_data::Dict, ::CTMarketPriorFeature,
                               ordered_ids, team_map::Dict, ds::Data.DataStore)
    haskey(F_data, :prediction_cutoff) || return nothing
    cutoff = F_data[:prediction_cutoff]
    cold = ct_cold_mask(F_data, team_map)
    prior = zeros(Float64, length(team_map))
    fallback = 0
    prior_season = ct_prior_season(ds, F_data[:prediction_target_season])
    audit = Dict{String, NamedTuple}()
    if isnothing(prior_season)
        for (club, index) in team_map
            cold[index] || continue
            audit[club] = (; status = :fallback, prior_season = nothing,
                           evidence_match_ids = Int[], reason = :no_prior_season)
        end
        F_data[:cold_team_mask] = Float64.(cold)
        F_data[:team_prior_attack_location] = prior
        F_data[:market_prior_fallback_count] = count(cold)
        F_data[:prior_audit] = audit
        return nothing
    end
    candidate_ids = Int[]
    for (club, index) in team_map
        cold[index] || continue
        candidates = DataFrames.subset(ds.matches,
            [:season, :tournament_id, :match_date, :home_team, :away_team] => DataFrames.ByRow(
                (s, t, d, h, a) -> s == prior_season && t in (54, 55) && !ismissing(d) &&
                                  Dates.Date(d) < cutoff && (h == club || a == club)))
        sort!(candidates, [:match_date, :match_id], rev = true)
        if DataFrames.nrow(candidates) < 5
            fallback += 1
            audit[club] = (; status = :fallback, prior_season, evidence_match_ids = Int[],
                           reason = :fewer_than_five_higher_tier_fixtures)
            continue
        end
        # The last five FIXTURES are fixed before inversion. Missing market data in
        # any one of those five is a declared fallback, never a silent sixth-match swap.
        last_five = first(candidates, min(5, DataFrames.nrow(candidates)))
        values = Float64[]
        for row in eachrow(last_five)
            push!(candidate_ids, Int(row.match_id))
            market_rows = DataFrames.subset(ds.odds, :match_id => DataFrames.ByRow(==(row.match_id)))
            gd = ct_market_goal_difference(market_rows)
            isnothing(gd) && continue
            push!(values, row.home_team == club ? gd : -gd)
        end
        evidence_ids = Int.(last_five.match_id)
        if DataFrames.nrow(last_five) == 5 && length(values) == 5
            prior[index] = Statistics.mean(values) / 2.0
            audit[club] = (; status = :market, prior_season, evidence_match_ids = evidence_ids,
                           reason = :accepted_complete_1x2_inversions)
        else
            fallback += 1
            audit[club] = (; status = :fallback, prior_season, evidence_match_ids = evidence_ids,
                           reason = :missing_or_refused_complete_1x2_inversion)
        end
    end
    F_data[:cold_team_mask] = Float64.(cold)
    F_data[:team_prior_attack_location] = prior
    F_data[:market_prior_fallback_count] = fallback
    F_data[:market_prior_candidate_match_ids] = sort!(unique(candidate_ids))
    F_data[:prior_audit] = audit
    return nothing
end

"Build fixed, historical tier anchors before the Turing boundary; 54=1 through 57=4."
function Features.add_feature!(F_data::Dict, ::CTFirstObservedTierFeature,
                               ordered_ids, team_map::Dict, ds::Data.DataStore)
    ids = Set(Int.(ordered_ids))
    rows = DataFrames.subset(ds.matches, :match_id => DataFrames.ByRow(id -> Int(id) in ids))
    DataFrames.nrow(rows) == length(ids) || error("A2 tier-anchor feature cannot resolve every fitted match")
    any(ismissing, rows.match_date) && error("A2 tier-anchor feature found missing match_date")
    sort!(rows, [:match_date, :match_id])
    tournament_tier = Dict(54 => 1, 55 => 2, 56 => 3, 57 => 4)
    anchors = zeros(Int, length(team_map))
    seen = falses(length(team_map))
    for row in eachrow(rows)
        tier = get(tournament_tier, Int(row.tournament_id), 0)
        tier > 0 || error("A2 tier-anchor feature found non-SPFL tournament $(row.tournament_id)")
        for club in (row.home_team, row.away_team)
            index = get(team_map, club, 0)
            index > 0 || error("A2 tier-anchor feature cannot map club $club")
            if !seen[index]
                anchors[index] = tier
                seen[index] = true
            end
        end
    end
    # A roster-only club has no all-SPFL fitted history. A2 has no approved prior
    # for it, so fail rather than assigning League Two by accident.
    all(seen) || error("A2 tier-anchor feature left a club without all-SPFL fitted history")
    F_data[:first_observed_tier_by_team] = anchors
    F_data[:tier_anchor_counts] = [count(==(tier), anchors) for tier in 1:4]
    home = Vector{Int}(F_data[:flat_home_ids])
    away = Vector{Int}(F_data[:flat_away_ids])
    F_data[:cross_anchor_fixture_count] = count(anchors[home] .!= anchors[away])
    return nothing
end

@model function ct_m00_engine(home_ids::Vector{Int}, away_ids::Vector{Int},
                              home_goals::Vector{Int}, away_goals::Vector{Int},
                              weights::Vector{Float64}, log_fact_h::Vector{Float64},
                              log_fact_a::Vector{Float64}, n_teams::Int,
                              config::CTM00Control)
    μ ~ config.intercept_prior
    γ ~ config.home_prior
    σ_a ~ config.attack_scale_prior
    σ_d ~ config.defence_scale_prior
    raw_a ~ Turing.filldist(Distributions.Normal(0.0, 1.0), n_teams)
    raw_d ~ Turing.filldist(Distributions.Normal(0.0, 1.0), n_teams)

    α_scaled = raw_a .* σ_a
    β_scaled = raw_d .* σ_d
    α = α_scaled .- Statistics.mean(α_scaled)
    β = β_scaled .- Statistics.mean(β_scaled)
    η_h = clamp.(μ .+ γ .+ α[home_ids] .+ β[away_ids], -10.0, 10.0)
    η_a = clamp.(μ .+      α[away_ids] .+ β[home_ids], -10.0, 10.0)
    ll_h = home_goals .* η_h .- exp.(η_h) .- log_fact_h
    ll_a = away_goals .* η_a .- exp.(η_a) .- log_fact_a
    Turing.@addlogprob! sum(ll_h .* weights) + sum(ll_a .* weights)
end

@model function ct_m03_engine(home_ids::Vector{Int}, away_ids::Vector{Int},
                              prior_attack_location::Vector{Float64},
                              home_goals::Vector{Int}, away_goals::Vector{Int},
                              weights::Vector{Float64}, log_fact_h::Vector{Float64},
                              log_fact_a::Vector{Float64}, n_teams::Int,
                              config::Union{CTM03StructuralOffset, CTM04MarketDerived})
    μ ~ config.intercept_prior
    γ ~ config.home_prior
    σ_a ~ config.attack_scale_prior
    σ_d ~ config.defence_scale_prior
    raw_a ~ Turing.filldist(Distributions.Normal(0.0, 1.0), n_teams)
    raw_d ~ Turing.filldist(Distributions.Normal(0.0, 1.0), n_teams)

    # Locations are fitted-history-only data. The residuals remain zero-sum, so a
    # cold entrant can have an informative location even with no likelihood rows.
    α_scaled = raw_a .* σ_a
    β_scaled = raw_d .* σ_d
    α = prior_attack_location .+ α_scaled .- Statistics.mean(α_scaled)
    β = .-prior_attack_location .+ β_scaled .- Statistics.mean(β_scaled)
    η_h = clamp.(μ .+ γ .+ α[home_ids] .+ β[away_ids], -10.0, 10.0)
    η_a = clamp.(μ .+      α[away_ids] .+ β[home_ids], -10.0, 10.0)
    ll_h = home_goals .* η_h .- exp.(η_h) .- log_fact_h
    ll_a = away_goals .* η_a .- exp.(η_a) .- log_fact_a
    Turing.@addlogprob! sum(ll_h .* weights) + sum(ll_a .* weights)
end

@model function ct_m02_engine(home_ids::Vector{Int}, away_ids::Vector{Int},
                              anchor_tiers::Vector{Int}, home_goals::Vector{Int},
                              away_goals::Vector{Int}, weights::Vector{Float64},
                              log_fact_h::Vector{Float64}, log_fact_a::Vector{Float64},
                              n_teams::Int, config::CTM02HierarchicalTiers)
    μ ~ config.intercept_prior
    γ ~ config.home_prior
    σ_a ~ config.attack_scale_prior
    σ_d ~ config.defence_scale_prior
    d_1 ~ config.tier_step_prior
    d_2 ~ config.tier_step_prior
    d_3 ~ config.tier_step_prior
    raw_a ~ Turing.filldist(Distributions.Normal(0.0, 1.0), n_teams)
    raw_d ~ Turing.filldist(Distributions.Normal(0.0, 1.0), n_teams)

    # τ₄=0; τ₃=d₃; τ₂=d₂+d₃; τ₁=d₁+d₂+d₃.  The anchor is fixed pre-model data.
    # Binary masks preserve a vectorised tracked array; a literal vector of tracked
    # scalars would instead produce Array{TrackedReal} and scalarise the tape.
    tier_1 = Float64.(anchor_tiers .== 1)
    tier_2 = Float64.(anchor_tiers .== 2)
    tier_3 = Float64.(anchor_tiers .== 3)
    τ = tier_1 .* (d_1 + d_2 + d_3) .+ tier_2 .* (d_2 + d_3) .+ tier_3 .* d_3
    α_scaled = raw_a .* σ_a
    β_scaled = raw_d .* σ_d
    α = τ .+ α_scaled .- Statistics.mean(α_scaled)
    β = .-τ .+ β_scaled .- Statistics.mean(β_scaled)
    η_h = clamp.(μ .+ γ .+ α[home_ids] .+ β[away_ids], -10.0, 10.0)
    η_a = clamp.(μ .+      α[away_ids] .+ β[home_ids], -10.0, 10.0)
    ll_h = home_goals .* η_h .- exp.(η_h) .- log_fact_h
    ll_a = away_goals .* η_a .- exp.(η_a) .- log_fact_a
    Turing.@addlogprob! sum(ll_h .* weights) + sum(ll_a .* weights)
end

@model function ct_m01_engine(home_ids::Vector{Int}, away_ids::Vector{Int},
                              league_ids::Vector{Int}, home_goals::Vector{Int},
                              away_goals::Vector{Int}, weights::Vector{Float64},
                              log_fact_h::Vector{Float64}, log_fact_a::Vector{Float64},
                              n_teams::Int, n_leagues::Int, config::CTM01LeagueOffsets)
    μ ~ config.intercept_prior
    γ ~ config.home_prior
    σ_a ~ config.attack_scale_prior
    σ_d ~ config.defence_scale_prior
    σ_league ~ config.league_scale_prior
    raw_a ~ Turing.filldist(Distributions.Normal(0.0, 1.0), n_teams)
    raw_d ~ Turing.filldist(Distributions.Normal(0.0, 1.0), n_teams)
    raw_league ~ Turing.filldist(Distributions.Normal(0.0, 1.0), n_leagues)

    α_scaled = raw_a .* σ_a
    β_scaled = raw_d .* σ_d
    δ_scaled = raw_league .* σ_league
    α = α_scaled .- Statistics.mean(α_scaled)
    β = β_scaled .- Statistics.mean(β_scaled)
    δ = δ_scaled .- Statistics.mean(δ_scaled)
    base = μ .+ δ[league_ids]
    η_h = clamp.(base .+ γ .+ α[home_ids] .+ β[away_ids], -10.0, 10.0)
    η_a = clamp.(base .+      α[away_ids] .+ β[home_ids], -10.0, 10.0)
    ll_h = home_goals .* η_h .- exp.(η_h) .- log_fact_h
    ll_a = away_goals .* η_a .- exp.(η_a) .- log_fact_a
    Turing.@addlogprob! sum(ll_h .* weights) + sum(ll_a .* weights)
end

function ct_design(model::CrossTierPoissonModel, feature_set)
    data = feature_set.data
    home_ids = Vector{Int}(data[:flat_home_ids])
    away_ids = Vector{Int}(data[:flat_away_ids])
    home_goals = Vector{Int}(data[:flat_home_goals])
    away_goals = Vector{Int}(data[:flat_away_goals])
    days_ago = Vector{Float64}(data[:dates])
    n = length(home_ids)
    all(length(v) == n for v in (away_ids, home_goals, away_goals, days_ago)) ||
        error("cross-tier design vectors are not aligned")
    all(days_ago .>= 0.0) || error("cross-tier dates contain a future fixture")
    weights = 0.5 .^ (days_ago ./ model.days_half_life)
    all(isfinite, weights) || error("cross-tier time-decay weights are non-finite")
    log_fact_h = SpecialFunctions.loggamma.(Float64.(home_goals) .+ 1.0)
    log_fact_a = SpecialFunctions.loggamma.(Float64.(away_goals) .+ 1.0)
    return (; home_ids, away_ids, home_goals, away_goals, weights, log_fact_h, log_fact_a,
              n_teams = Int(data[:n_teams]))
end

function PreGame.build_turing_model(model::CTM00Control, feature_set)
    z = ct_design(model, feature_set)
    return ct_m00_engine(z.home_ids, z.away_ids, z.home_goals, z.away_goals,
                         z.weights, z.log_fact_h, z.log_fact_a, z.n_teams, model)
end

function ct_build_prior_model(model::Union{CTM03StructuralOffset, CTM04MarketDerived}, feature_set)
    z = ct_design(model, feature_set)
    location = Vector{Float64}(feature_set.data[:team_prior_attack_location])
    length(location) == z.n_teams || error("cold-start prior locations do not align with team vocabulary")
    all(isfinite, location) || error("cold-start prior locations are non-finite")
    return ct_m03_engine(z.home_ids, z.away_ids, location, z.home_goals, z.away_goals,
                         z.weights, z.log_fact_h, z.log_fact_a, z.n_teams, model)
end

PreGame.build_turing_model(model::CTM03StructuralOffset, feature_set) =
    ct_build_prior_model(model, feature_set)
PreGame.build_turing_model(model::CTM04MarketDerived, feature_set) =
    ct_build_prior_model(model, feature_set)

function PreGame.build_turing_model(model::CTM02HierarchicalTiers, feature_set)
    z = ct_design(model, feature_set)
    anchor_tiers = Vector{Int}(feature_set.data[:first_observed_tier_by_team])
    length(anchor_tiers) == z.n_teams || error("A2 tier anchors do not align with team vocabulary")
    all((1 .<= anchor_tiers) .& (anchor_tiers .<= 4)) ||
        error("A2 tier anchors must be integers in 1:4")
    return ct_m02_engine(z.home_ids, z.away_ids, anchor_tiers, z.home_goals, z.away_goals,
                         z.weights, z.log_fact_h, z.log_fact_a, z.n_teams, model)
end

function PreGame.build_turing_model(model::CTM01LeagueOffsets, feature_set)
    z = ct_design(model, feature_set)
    data = feature_set.data
    league_ids = Vector{Int}(data[:flat_league_ids])
    n_leagues = Int(data[:n_leagues])
    length(league_ids) == length(z.home_ids) || error("league IDs are not aligned with fixtures")
    all((1 .<= league_ids) .& (league_ids .<= n_leagues)) ||
        error("league IDs must be in 1:$n_leagues")
    return ct_m01_engine(z.home_ids, z.away_ids, league_ids, z.home_goals, z.away_goals,
                         z.weights, z.log_fact_h, z.log_fact_a, z.n_teams, n_leagues, model)
end

"Flatten selected chain parameters to draws × widths in the same iter-major, chain-second order as `vec`."
function ct_chain_matrix(chain::MCMCChains.Chains, symbols::Vector{Symbol})
    n_iter = size(chain, 1)
    n_chains = size(chain, 3)
    width = length(symbols)
    out = Matrix{Float64}(undef, n_iter * n_chains, width)
    for (column, symbol) in enumerate(symbols)
        out[:, column] .= vec(Array(chain[symbol]))
    end
    return out
end

function ct_extract_rates(model::CrossTierPoissonModel, df::DataFrames.AbstractDataFrame,
                          feature_set, chain::MCMCChains.Chains)
    data = feature_set.data
    team_map = data[:team_map]
    n_teams = Int(data[:n_teams])
    n_draws = size(chain, 1) * size(chain, 3)
    μ = vec(Array(chain[Symbol("μ")]))
    γ = vec(Array(chain[Symbol("γ")]))
    σ_a = vec(Array(chain[Symbol("σ_a")]))
    σ_d = vec(Array(chain[Symbol("σ_d")]))
    raw_a = ct_chain_matrix(chain, [Symbol("raw_a[$i]") for i in 1:n_teams])
    raw_d = ct_chain_matrix(chain, [Symbol("raw_d[$i]") for i in 1:n_teams])
    α_scaled = raw_a .* σ_a
    β_scaled = raw_d .* σ_d
    α = α_scaled .- Statistics.mean(α_scaled, dims = 2)
    β = β_scaled .- Statistics.mean(β_scaled, dims = 2)
    return ct_extract_rates(model, df, data, team_map, n_draws, μ, γ, α, β, chain)
end

function ct_extract_rates(::Union{CTM00Control, CTM03StructuralOffset, CTM04MarketDerived},
                          df, data, team_map, n_draws, μ, γ, α, β, chain)
    out = Dict{Int, NamedTuple}()
    for row in eachrow(df)
        h = get(team_map, row.home_team, 0)
        a = get(team_map, row.away_team, 0)
        α_h = h > 0 ? α[:, h] : zeros(n_draws)
        β_h = h > 0 ? β[:, h] : zeros(n_draws)
        α_a = a > 0 ? α[:, a] : zeros(n_draws)
        β_a = a > 0 ? β[:, a] : zeros(n_draws)
        out[Int(row.match_id)] = (; λ_h = exp.(μ .+ γ .+ α_h .+ β_a),
                                  λ_a = exp.(μ .+ α_a .+ β_h),
                                  true_xg_h = exp.(μ .+ γ .+ α_h .+ β_a),
                                  true_xg_a = exp.(μ .+ α_a .+ β_h))
    end
    return out
end

function ct_extract_rates(::CTM02HierarchicalTiers, df, data, team_map, n_draws, μ, γ, α, β, chain)
    n_teams = Int(data[:n_teams])
    anchors = Vector{Int}(data[:first_observed_tier_by_team])
    d_1 = vec(Array(chain[Symbol("d_1")]))
    d_2 = vec(Array(chain[Symbol("d_2")]))
    d_3 = vec(Array(chain[Symbol("d_3")]))
    τ_by_tier = hcat(d_1 .+ d_2 .+ d_3, d_2 .+ d_3, d_3, zeros(n_draws))
    # Both residual matrices and tier locations are draws × teams.
    τ = τ_by_tier[:, anchors]
    # `α`/`β` supplied by the common extractor are residuals. Restore fixed A2 locations.
    α = α .+ τ
    β = β .- τ
    return ct_extract_rates(CTM00Control(), df, data, team_map, n_draws, μ, γ, α, β, chain)
end

function ct_extract_rates(::Union{CTM03StructuralOffset, CTM04MarketDerived},
                          df, data, team_map, n_draws, μ, γ, α, β, chain)
    location = Vector{Float64}(data[:team_prior_attack_location])
    return ct_extract_rates(CTM00Control(), df, data, team_map, n_draws, μ, γ,
                            α .+ permutedims(location), β .- permutedims(location), chain)
end

function ct_extract_rates(::CTM01LeagueOffsets, df, data, team_map, n_draws, μ, γ, α, β, chain)
    n_leagues = Int(data[:n_leagues])
    σ_league = vec(Array(chain[Symbol("σ_league")]))
    raw_league = ct_chain_matrix(chain, [Symbol("raw_league[$i]") for i in 1:n_leagues])
    δ_scaled = raw_league .* σ_league
    δ = δ_scaled .- Statistics.mean(δ_scaled, dims = 2)
    league_lookup = data[:league_lookup]
    out = Dict{Int, NamedTuple}()
    for row in eachrow(df)
        h = get(team_map, row.home_team, 0)
        a = get(team_map, row.away_team, 0)
        league = get(league_lookup, Int(row.match_id), 0)
        league > 0 || error("no all-SPFL league index for OOS match $(row.match_id)")
        α_h = h > 0 ? α[:, h] : zeros(n_draws)
        β_h = h > 0 ? β[:, h] : zeros(n_draws)
        α_a = a > 0 ? α[:, a] : zeros(n_draws)
        β_a = a > 0 ? β[:, a] : zeros(n_draws)
        η_h = clamp.(μ .+ δ[:, league] .+ γ .+ α_h .+ β_a, -10.0, 10.0)
        η_a = clamp.(μ .+ δ[:, league] .+ α_a .+ β_h, -10.0, 10.0)
        out[Int(row.match_id)] = (; λ_h = exp.(η_h), λ_a = exp.(η_a),
                                  true_xg_h = exp.(η_h), true_xg_a = exp.(η_a))
    end
    return out
end

function PreGame.extract_parameters(model::CrossTierPoissonModel,
                                    df::DataFrames.AbstractDataFrame,
                                    feature_set, chain::MCMCChains.Chains)
    return ct_extract_rates(model, df, feature_set, chain)
end

"Build the Stage-1 candidate manifest consumed by the smoke runner."
function cross_tier_smoke_candidates(; target_seasons::Vector{String} = ["24/25", "25/26"],
                                     history_seasons::Int = 2,
                                     sampler = BayesianFootball.NUTSConfig(n_samples = 100, n_chains = 2),
                                     execution = BayesianFootball.AutoExecution())
    lower_split = scottish_lower_splitter(target_seasons = target_seasons,
                                          history_seasons = history_seasons)
    all_split = CrossTierLowerCV(target_seasons = target_seasons,
                                 history_seasons = history_seasons)
    prior_split = CrossTierPriorCV(target_seasons = target_seasons,
                                   history_seasons = history_seasons)
    m00 = CTM00Control()
    m01 = CTM01LeagueOffsets()
    m02 = CTM02HierarchicalTiers()
    m03 = CTM03StructuralOffset()
    m04 = CTM04MarketDerived()
    return [
        (; name = "m00_control", model = m00, segment = Data.ScottishAll(),
           splitter = prior_split,
           fit_config = Training.FitConfig(name = "scotland_cross_tier_time_decay_m00",
                                           model = m00, splitter = prior_split, sampler = sampler,
                                           execution = execution),
           build_one_fold = (ds -> first(Data.create_id_boundaries(ds, prior_split)))),
        (; name = "m01_all_spfl_league_offsets", model = m01, segment = Data.ScottishAll(),
           splitter = all_split,
           fit_config = Training.FitConfig(name = "scotland_cross_tier_time_decay_m01",
                                           model = m01, splitter = all_split, sampler = sampler,
                                           execution = execution),
           build_one_fold = (ds -> first(Data.create_id_boundaries(ds, all_split)))),
        (; name = "m02_all_spfl_hierarchical_tiers", model = m02, segment = Data.ScottishAll(),
           splitter = all_split,
           fit_config = Training.FitConfig(name = "scotland_cross_tier_time_decay_m02",
                                           model = m02, splitter = all_split, sampler = sampler,
                                           execution = execution),
           build_one_fold = (ds -> first(Data.create_id_boundaries(ds, all_split)))),
        (; name = "m03_prior_structural_offset", model = m03, segment = Data.ScottishAll(),
           splitter = prior_split,
           fit_config = Training.FitConfig(name = "scotland_cross_tier_time_decay_m03",
                                           model = m03, splitter = prior_split, sampler = sampler,
                                           execution = execution),
           build_one_fold = (ds -> first(Data.create_id_boundaries(ds, prior_split)))),
        (; name = "m04_prior_market_derived", model = m04, segment = Data.ScottishAll(),
           splitter = prior_split,
           fit_config = Training.FitConfig(name = "scotland_cross_tier_time_decay_m04",
                                           model = m04, splitter = prior_split, sampler = sampler,
                                           execution = execution),
           build_one_fold = (ds -> first(Data.create_id_boundaries(ds, prior_split)))),
    ]
end

end # include guard

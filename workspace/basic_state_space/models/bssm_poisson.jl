# workspace/basic_state_space/models/bssm_poisson.jl

module BSSMPoisson

using Turing
using LinearAlgebra
using Distributions
using DataFrames
using MCMCChains
using Statistics
using BayesianFootball

export BSSMPoissonModel

#=
--------------------------------------------------------------------------------
1.  MODEL DEFINITION
--------------------------------------------------------------------------------
=#

struct BSSMPoissonModel <: AbstractStateSpaceModel end

function BayesianFootball.get_required_features(::BSSMPoissonModel)
    return (
        :home_team_ids, :away_team_ids, :n_teams,
        :global_round
    )
end

function BayesianFootball.build_turing_model(
    ::BSSMPoissonModel,
    features::NamedTuple,
    goals_home::Vector{Int},
    goals_away::Vector{Int}
)
    temp_df = DataFrame(
        global_round=features.global_round,
        home_id=features.home_team_ids,
        away_id=features.away_team_ids,
        gh=goals_home,
        ga=goals_away
    )
    grouped = groupby(temp_df, :global_round)
    n_rounds = length(grouped)

    home_ids_by_round = [collect(g.home_id) for g in grouped]
    away_ids_by_round = [collect(g.away_id) for g in grouped]
    goals_home_by_round = [collect(g.gh) for g in grouped]
    goals_away_by_round = [collect(g.ga) for g in grouped]

    @model function bssm_model(
        home_ids::Vector{Vector{Int}},
        away_ids::Vector{Vector{Int}},
        goals_h::Vector{Vector{Int}},
        goals_a::Vector{Vector{Int}},
        n_teams::Int,
        n_rounds::Int
    )
        μ_att ~ Normal(0, 1)
        μ_def ~ Normal(0, 1)
        σ_att ~ Gamma(2, 0.5)
        σ_def ~ Gamma(2, 0.5)
        home_adv ~ Normal(0.3, 0.5)

        σ_evo_att ~ Gamma(2, 0.5)
        σ_evo_def ~ Gamma(2, 0.5)

        log_attack_t0 ~ MvNormal(fill(μ_att, n_teams), σ_att^2 * I)
        log_defense_t0 ~ MvNormal(fill(μ_def, n_teams), σ_def^2 * I)

        log_attack = Matrix{Real}(undef, n_teams, n_rounds)
        log_defense = Matrix{Real}(undef, n_teams, n_rounds)

        log_attack[:, 1] ~ MvNormal(log_attack_t0, σ_evo_att^2 * I)
        log_defense[:, 1] ~ MvNormal(log_defense_t0, σ_evo_def^2 * I)

        for t in 2:n_rounds
            log_attack[:, t] ~ MvNormal(log_attack[:, t-1], σ_evo_att^2 * I)
            log_defense[:, t] ~ MvNormal(log_defense[:, t-1], σ_evo_def^2 * I)
        end

        for t in 1:n_rounds
            n_matches_in_round = length(home_ids[t])
            for m in 1:n_matches_in_round
                h_team = home_ids[t][m]
                a_team = away_ids[t][m]
                log_λ_home = log_attack[h_team, t] + log_defense[a_team, t] + home_adv
                log_λ_away = log_attack[a_team, t] + log_defense[h_team, t]
                goals_h[t][m] ~ LogPoisson(log_λ_home)
                goals_a[t][m] ~ LogPoisson(log_λ_away)
            end
        end
    end

    return bssm_model(
        home_ids_by_round,
        away_ids_by_round,
        goals_home_by_round,
        goals_away_by_round,
        features.n_teams,
        n_rounds
    )
end

#=
--------------------------------------------------------------------------------
2.  POSTERIOR EXTRACTION (CORRECTED FUNCTION)
--------------------------------------------------------------------------------
=#

function BayesianFootball.extract_posterior_samples(
    ::BSSMPoissonModel,
    chain::Chains,
    ::MappedData
)
    # This correctly iterates through all parameters in the chain,
    # extracts their samples as vectors, and returns them in a NamedTuple.
    # This is the format the `_predict_match_ft` function expects.
    param_names = names(chain)
    samples_dict = Dict{Symbol, Vector}()
    for p_name in param_names
        samples_dict[Symbol(p_name)] = vec(Array(chain[p_name]))
    end
    return (; samples_dict...) # Convert Dict to NamedTuple
end

#=
--------------------------------------------------------------------------------
3.  PREDICTION LOGIC
--------------------------------------------------------------------------------
=#

function get_goal_rates(
    model_def::BSSMPoissonModel,
    posterior_samples::NamedTuple,
    sample_idx::Int,
    features::NamedTuple
)
    home_team = features.home_team_ids
    away_team = features.away_team_ids
    last_round = features.last_round_trained

    home_adv = posterior_samples.home_adv[sample_idx]
    
    att_h_sym = Symbol("log_attack[$(home_team),$(last_round)]")
    def_a_sym = Symbol("log_defense[$(away_team),$(last_round)]")
    att_a_sym = Symbol("log_attack[$(away_team),$(last_round)]")
    def_h_sym = Symbol("log_defense[$(home_team),$(last_round)]")
    
    att_h = posterior_samples[att_h_sym][sample_idx]
    def_a = posterior_samples[def_a_sym][sample_idx]
    att_a = posterior_samples[att_a_sym][sample_idx]
    def_h = posterior_samples[def_h_sym][sample_idx]

    λ_home = exp(att_h + def_a + home_adv)
    λ_away = exp(att_a + def_h)

    return λ_home, λ_away
end

#=
--------------------------------------------------------------------------------
4.  FULL PREDICTION IMPLEMENTATION
--------------------------------------------------------------------------------
=#

function BayesianFootball._predict_match_ft(
    model_def::BSSMPoissonModel,
    chain::Chains,
    features::NamedTuple,
    posterior_samples::NamedTuple
)
    n_samples = size(chain, 1) * size(chain, 3)
    λ_home=zeros(n_samples); λ_away=zeros(n_samples); home_win_probs=zeros(n_samples); draw_probs=zeros(n_samples); away_win_probs=zeros(n_samples)
    under_05=zeros(n_samples); under_15=zeros(n_samples); under_25=zeros(n_samples); under_35=zeros(n_samples); btts=zeros(n_samples)
    cs_keys = vcat([(h, a) for h in 0:3 for a in 0:3], ["other_home_win", "other_away_win", "other_draw"])
    correct_score = Dict{Union{Tuple{Int,Int}, String}, Vector{Float64}}(key => zeros(n_samples) for key in cs_keys)

    for i in 1:n_samples
        λ_h, λ_a = get_goal_rates(model_def, posterior_samples, i, features)
        λ_home[i]=λ_h; λ_away[i]=λ_a
        p = BayesianFootball.compute_xScore(λ_h, λ_a, 10)
        hda = BayesianFootball.calculate_1x2(p)
        cs = BayesianFootball.calculate_correct_score_dict_ft(p)
        
        home_win_probs[i]=hda.hw
        draw_probs[i]=hda.dr
        away_win_probs[i]=hda.aw
        
        for (key, value) in cs; correct_score[key][i]=value; end
        
        under_05[i]=BayesianFootball.calculate_under_prob(p,0)
        under_15[i]=BayesianFootball.calculate_under_prob(p,1)
        under_25[i]=BayesianFootball.calculate_under_prob(p,2)
        under_35[i]=BayesianFootball.calculate_under_prob(p,3)
        btts[i] = BayesianFootball.calculate_btts(p)
    end

    return BayesianFootball.Predictions.MatchFTPredictions(λ_home, λ_away, home_win_probs, draw_probs, away_win_probs, correct_score, under_05, under_15, under_25, under_35, btts)
end

function BayesianFootball._predict_match_ht(
    model_def::BSSMPoissonModel,
    chain::Chains,
    features::NamedTuple,
    posterior_samples::NamedTuple
)
    n_samples = size(chain, 1) * size(chain, 3)
    λ_home=zeros(n_samples); λ_away=zeros(n_samples); home_win_probs=zeros(n_samples); draw_probs=zeros(n_samples); away_win_probs=zeros(n_samples)
    under_05=zeros(n_samples); under_15=zeros(n_samples); under_25=zeros(n_samples)
    cs_keys = vcat([(h, a) for h in 0:2 for a in 0:2], ["any_unquoted"])
    correct_score = Dict{Union{Tuple{Int,Int}, String}, Vector{Float64}}(key => zeros(n_samples) for key in cs_keys)

    for i in 1:n_samples
        λ_h, λ_a = get_goal_rates(model_def, posterior_samples, i, features)
        λ_home[i]=λ_h; λ_away[i]=λ_a
        p = BayesianFootball.compute_xScore(λ_h, λ_a, 10)
        hda = BayesianFootball.calculate_1x2(p)
        cs = BayesianFootball.calculate_correct_score_dict_ht(p)
        
        home_win_probs[i]=hda.hw
        draw_probs[i]=hda.dr
        away_win_probs[i]=hda.aw
        
        for (key, value) in cs; correct_score[key][i]=value; end
        
        under_05[i]=BayesianFootball.calculate_under_prob(p,0)
        under_15[i]=BayesianFootball.calculate_under_prob(p,1)
        under_25[i]=BayesianFootball.calculate_under_prob(p,2)
    end

    return BayesianFootball.Predictions.MatchHTPredictions(λ_home, λ_away, home_win_probs, draw_probs, away_win_probs, correct_score, under_05, under_15, under_25)
end

end # end module

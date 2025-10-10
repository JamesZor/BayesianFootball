module AR1NegativeBinomialHA

using Turing
using LinearAlgebra
using Distributions
using DataFrames
using MCMCChains
using Statistics
using BayesianFootball

export AR1NegativeBinomialHAModel

#=
--------------------------------------------------------------------------------
1.  MODEL DEFINITION
--------------------------------------------------------------------------------
=#

struct AR1NegativeBinomialHAModel <: AbstractStateSpaceModel end

function BayesianFootball.get_required_features(::AR1NegativeBinomialHAModel)
    return (
        :home_team_ids, :away_team_ids, :n_teams,
        :global_round, :league_ids, :n_leagues
    )
end

function BayesianFootball.build_turing_model(
    ::AR1NegativeBinomialHAModel,
    features::NamedTuple,
    goals_home::Vector{Int},
    goals_away::Vector{Int}
)
    temp_df = DataFrame(
        global_round=features.global_round, home_id=features.home_team_ids,
        away_id=features.away_team_ids, league_id=features.league_ids,
        gh=goals_home, ga=goals_away
    )
    grouped = groupby(temp_df, :global_round)
    n_rounds = length(grouped)

    home_ids_by_round = [g.home_id for g in grouped]
    away_ids_by_round = [g.away_id for g in grouped]
    league_ids_by_round = [g.league_id for g in grouped]
    home_goals_by_round = [g.gh for g in grouped]
    away_goals_by_round = [g.ga for g in grouped]
    
    return ar1_neg_bin_ha_model(
        home_ids_by_round, away_ids_by_round, league_ids_by_round,
        home_goals_by_round, away_goals_by_round,
        features.n_teams, features.n_leagues, n_rounds
    )
end

#=
--------------------------------------------------------------------------------
2.  TURING @MODEL IMPLEMENTATION
--------------------------------------------------------------------------------
=#

@model function ar1_neg_bin_ha_model(
    home_team_ids::Vector{<:AbstractVector}, away_team_ids::Vector{<:AbstractVector}, 
    league_ids::Vector{<:AbstractVector}, home_goals::Vector{<:AbstractVector}, 
    away_goals::Vector{<:AbstractVector}, n_teams::Int, n_leagues::Int, n_rounds::Int
)
    ρ_attack ~ Beta(10, 1.5); ρ_defense ~ Beta(10, 1.5)
    μ_log_σ_attack ~ Normal(-2.5, 0.5); τ_log_σ_attack ~ Truncated(Normal(0, 0.2), 0, Inf)
    z_log_σ_attack ~ MvNormal(zeros(n_teams), I)
    log_σ_attack = μ_log_σ_attack .+ z_log_σ_attack * τ_log_σ_attack; σ_attack = exp.(log_σ_attack)

    μ_log_σ_defense ~ Normal(-2.5, 0.5); τ_log_σ_defense ~ Truncated(Normal(0, 0.2), 0, Inf)
    z_log_σ_defense ~ MvNormal(zeros(n_teams), I)
    log_σ_defense = μ_log_σ_defense .+ z_log_σ_defense * τ_log_σ_defense; σ_defense = exp.(log_σ_defense)

    ρ_home ~ Beta(10, 1.5); μ_log_σ_home ~ Normal(-3.0, 0.5); σ_home = exp(μ_log_σ_home)
    ρ_phi ~ Beta(10, 1.5); μ_log_σ_phi ~ Normal(-3.0, 0.5); σ_phi = exp(μ_log_σ_phi)
    
    initial_α_z ~ MvNormal(zeros(n_teams), I); log_α_raw_t0 = initial_α_z * sqrt(0.5)
    initial_β_z ~ MvNormal(zeros(n_teams), I); log_β_raw_t0 = initial_β_z * sqrt(0.5)
    initial_home_z ~ MvNormal(zeros(n_leagues), I); log_home_adv_raw_t0 = log(1.3) .+ initial_home_z .* sqrt(0.1)
    initial_phi_z ~ Normal(0, 1); log_phi_raw_t0 = log(10.0) + initial_phi_z * sqrt(0.1)

    z_α ~ MvNormal(zeros(n_teams * n_rounds), I); z_α_mat = reshape(z_α, n_teams, n_rounds)
    z_β ~ MvNormal(zeros(n_teams * n_rounds), I); z_β_mat = reshape(z_β, n_teams, n_rounds)
    z_home ~ MvNormal(zeros(n_leagues * n_rounds), I); z_home_mat = reshape(z_home, n_leagues, n_rounds)
    z_phi ~ MvNormal(zeros(n_rounds), I)

    log_α_raw = Matrix{Real}(undef, n_teams, n_rounds); log_β_raw = Matrix{Real}(undef, n_teams, n_rounds)
    log_home_adv_raw = Matrix{Real}(undef, n_leagues, n_rounds); log_phi_raw = Vector{Real}(undef, n_rounds)

    for t in 1:n_rounds
        if t == 1
            log_α_raw[:, 1] = log_α_raw_t0 .+ z_α_mat[:, 1] .* σ_attack
            log_β_raw[:, 1] = log_β_raw_t0 .+ z_β_mat[:, 1] .* σ_defense
            log_home_adv_raw[:, 1] = log_home_adv_raw_t0 .+ z_home_mat[:, 1] .* σ_home
            log_phi_raw[1] = log_phi_raw_t0 + z_phi[1] * σ_phi
        else
            log_α_raw[:, t] = ρ_attack * log_α_raw[:, t-1] .+ z_α_mat[:, t] .* σ_attack
            log_β_raw[:, t] = ρ_defense * log_β_raw[:, t-1] .+ z_β_mat[:, t] .* σ_defense
            log_home_adv_raw[:, t] = ρ_home * log_home_adv_raw[:, t-1] .+ z_home_mat[:, t] .* σ_home
            log_phi_raw[t] = ρ_phi * log_phi_raw[t-1] + z_phi[t] * σ_phi
        end

        log_α_t = log_α_raw[:, t] .- mean(log_α_raw[:, t])
        log_β_t = log_β_raw[:, t] .- mean(log_β_raw[:, t])
        
        home_ids = home_team_ids[t]; away_ids = away_team_ids[t]; l_ids = league_ids[t]
        
        if !isempty(home_ids)
            log_home_adv_t = log_home_adv_raw[l_ids, t]
            log_λs = log_α_t[home_ids] .+ log_β_t[away_ids] .+ log_home_adv_t
            log_μs = log_α_t[away_ids] .+ log_β_t[home_ids]
            
            ϕ_t = exp(log_phi_raw[t])
            ps_home = Turing.logistic.(log(ϕ_t) .- log_λs)
            ps_away = Turing.logistic.(log(ϕ_t) .- log_μs)

            home_goals[t] .~ NegativeBinomial.(ϕ_t, ps_home)
            away_goals[t] .~ NegativeBinomial.(ϕ_t, ps_away)
        end
    end
end
#=
--------------------------------------------------------------------------------
3.  PREDICTION LOGIC (REFACTORED)
--------------------------------------------------------------------------------
=#

function get_neg_bin_params(
    ::AR1NegativeBinomialHAModel,
    samples::NamedTuple,
    i::Int,
    features::NamedTuple
)
    home_idx=features.home_team_ids[1]; away_idx=features.away_team_ids[1];
    league_idx=features.league_ids[1]
    current_round = features.global_round[1]

    local log_α_t, log_β_t, log_home_adv_i, ϕ_i

    if current_round <= samples.n_rounds
        log_α_t = samples.log_α_centered[i, :, current_round]
        log_β_t = samples.log_β_centered[i, :, current_round]
        log_home_adv_i = samples.log_home_adv_raw[i, league_idx, current_round]
        ϕ_i = exp(samples.log_phi_raw[i, current_round])
    else
        n_teams=size(samples.log_α_raw,2);
        n_leagues=size(samples.log_home_adv_raw,2)

        last_α_raw=samples.log_α_raw[i,:,samples.n_rounds];
        last_β_raw=samples.log_β_raw[i,:,samples.n_rounds]
        last_home_adv_raw=samples.log_home_adv_raw[i,:,samples.n_rounds]; last_phi_raw=samples.log_phi_raw[i,samples.n_rounds]

        future_α_raw = samples.ρ_attack[i].*last_α_raw .+ randn(n_teams).*samples.σ_attack[i,:]
        future_β_raw = samples.ρ_defense[i].*last_β_raw .+ randn(n_teams).*samples.σ_defense[i,:]
        future_home_adv_raw = samples.ρ_home[i].*last_home_adv_raw .+ randn(n_leagues).*samples.σ_home[i]
        future_phi_raw = samples.ρ_phi[i]*last_phi_raw + randn()*samples.σ_phi[i]

        log_α_t = future_α_raw .- mean(future_α_raw);
        log_β_t = future_β_raw .- mean(future_β_raw)
        log_home_adv_i = future_home_adv_raw[league_idx];
        ϕ_i = exp(future_phi_raw)
    end

    log_λ_home = log_α_t[home_idx] + log_β_t[away_idx] + log_home_adv_i
    log_λ_away = log_α_t[away_idx] + log_β_t[home_idx]

    p_home = Turing.logistic(log(ϕ_i) - log_λ_home)
    p_away = Turing.logistic(log(ϕ_i) - log_λ_away)

    return (p_home=p_home, p_away=p_away, ϕ=ϕ_i, λ_home=exp(log_λ_home), λ_away=exp(log_λ_away))
end

function _compute_xScore_neg_bin(ϕ::Number, p_home::Number, p_away::Number, max_goals::Int)
    p = zeros(max_goals + 1, max_goals + 1)
    for h in 0:max_goals, a in 0:max_goals
        p[h+1, a+1] = pdf(NegativeBinomial(ϕ, p_home), h) * pdf(NegativeBinomial(ϕ, p_away), a)
    end
    return p
end


function BayesianFootball.predict(
    model_def::AR1NegativeBinomialHAModel,
    chain::Chains,
    features::NamedTuple,
    mapping::MappedData
)
    posterior_samples = extract_posterior_samples(model_def, chain, mapping)
    
    n_samples = size(chain, 1) * size(chain, 3)

    # Pre-allocate FT arrays
    ft_λ_home=zeros(n_samples); ft_λ_away=zeros(n_samples); ft_home_win=zeros(n_samples); ft_draw=zeros(n_samples); ft_away_win=zeros(n_samples)
    ft_under_05=zeros(n_samples); ft_under_15=zeros(n_samples); ft_under_25=zeros(n_samples); ft_under_35=zeros(n_samples);
    ft_btts=zeros(n_samples)
    ft_cs_keys = vcat([(h, a) for h in 0:3 for a in 0:3], ["other_home_win", "other_away_win", "other_draw"])
    ft_correct_score = Dict{Union{Tuple{Int,Int}, String}, Vector{Float64}}(key => zeros(n_samples) for key in ft_cs_keys)

    # Pre-allocate HT arrays
    ht_λ_home=zeros(n_samples); ht_λ_away=zeros(n_samples); ht_home_win=zeros(n_samples); ht_draw=zeros(n_samples); ht_away_win=zeros(n_samples)
    ht_under_05=zeros(n_samples); ht_under_15=zeros(n_samples); ht_under_25=zeros(n_samples)
    ht_cs_keys = vcat([(h, a) for h in 0:2 for a in 0:2], ["any_unquoted"])
    ht_correct_score = Dict{Union{Tuple{Int,Int}, String}, Vector{Float64}}(key => zeros(n_samples) for key in ht_cs_keys)


    # --- SINGLE LOOP FOR ALL PREDICTIONS ---
    for i in 1:n_samples
        # Call params function ONCE
        params = get_neg_bin_params(model_def, posterior_samples, i, features)
        
        # Calculate score probability matrix ONCE
        p = _compute_xScore_neg_bin(params.ϕ, params.p_home, params.p_away, 10)

        # --- Full Time Predictions ---
        ft_λ_home[i] = params.λ_home
        ft_λ_away[i] = params.λ_away

        hda_ft = BayesianFootball.calculate_1x2(p)
        ft_home_win[i] = hda_ft.hw
        ft_draw[i] = hda_ft.dr
        ft_away_win[i] = hda_ft.aw

        cs_ft = BayesianFootball.calculate_correct_score_dict_ft(p)
        for (key, value) in cs_ft; ft_correct_score[key][i]=value; end

        ft_under_05[i] = BayesianFootball.calculate_under_prob(p, 0)
        ft_under_15[i] = BayesianFootball.calculate_under_prob(p, 1)
        ft_under_25[i] = BayesianFootball.calculate_under_prob(p, 2)
        ft_under_35[i] = BayesianFootball.calculate_under_prob(p, 3)
        ft_btts[i] = BayesianFootball.calculate_btts(p)

        # --- Half Time Predictions ---
        ht_λ_home[i] = params.λ_home
        ht_λ_away[i] = params.λ_away
        
        # NOTE: Using the same `p` matrix for HT as for FT. 
        # If HT logic should be different, it needs adjustment here.
        hda_ht = BayesianFootball.calculate_1x2(p)
        ht_home_win[i] = hda_ht.hw
        ht_draw[i] = hda_ht.dr
        ht_away_win[i] = hda_ht.aw

        cs_ht = BayesianFootball.calculate_correct_score_dict_ht(p)
        for (key, value) in cs_ht; ht_correct_score[key][i]=value; end

        ht_under_05[i] = BayesianFootball.calculate_under_prob(p, 0)
        ht_under_15[i] = BayesianFootball.calculate_under_prob(p, 1)
        ht_under_25[i] = BayesianFootball.calculate_under_prob(p, 2)
    end

    ft_preds = BayesianFootball.Predictions.MatchFTPredictions(
        ft_λ_home, ft_λ_away, ft_home_win, ft_draw, ft_away_win, 
        ft_correct_score, ft_under_05, ft_under_15, ft_under_25, ft_under_35, ft_btts
    )

    ht_preds = BayesianFootball.Predictions.MatchHTPredictions(
        ht_λ_home, ht_λ_away, ht_home_win, ht_draw, ht_away_win,
        ht_correct_score, ht_under_05, ht_under_15, ht_under_25
    )

    return BayesianFootball.Predictions.MatchPredictions(ft_preds, ht_preds)
end

end # end module

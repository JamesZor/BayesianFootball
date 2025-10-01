# workspace/basic_state_space/analysis/evaluation.jl

module ModelEvaluation

using MCMCChains
using Statistics
using Distributions
using PrettyTables
using Printf
using Turing

export generate_test_set, generate_predictions, evaluate_predictions, display_evaluation_results

#=
--------------------------------------------------------------------------------
1.  TEST SET GENERATION
--------------------------------------------------------------------------------
=#

"""
    generate_test_set(data)

Generates a future test set with ground truth outcomes based on the synthetic data.
"""
function generate_test_set(data)
    n_teams = data.n_teams
    n_rounds = data.n_rounds
    
    # --- Project true strengths one step into the future ---
    attack_slopes = data.true_log_α[:, n_rounds] - data.true_log_α[:, n_rounds-1]
    defense_slopes = data.true_log_β[:, n_rounds] - data.true_log_β[:, n_rounds-1]
    
    true_log_α_future = data.true_log_α[:, n_rounds] + attack_slopes
    true_log_β_future = data.true_log_β[:, n_rounds] + defense_slopes

    # Handle both static and dynamic home advantage from the synthetic data
    if :true_log_home_adv in keys(data)
        home_adv_slope = data.true_log_home_adv[n_rounds] - data.true_log_home_adv[n_rounds-1]
        true_log_home_adv_future = data.true_log_home_adv[n_rounds] + home_adv_slope
    else
        true_log_home_adv_future = log(1.3) # Fallback for older data format
    end

    true_log_α_future .-= mean(true_log_α_future)
    true_log_β_future .-= mean(true_log_β_future)

    # --- Create all possible fixtures and simulate true outcomes ---
    test_set = []
    for i in 1:n_teams, j in 1:n_teams
        if i == j continue end
        
        log_λ_true = true_log_α_future[i] + true_log_β_future[j] + true_log_home_adv_future
        log_μ_true = true_log_α_future[j] + true_log_β_future[i]
        
        push!(test_set, (
            home_team=i, away_team=j, 
            true_home_goals=rand(LogPoisson(log_λ_true)), 
            true_away_goals=rand(LogPoisson(log_μ_true))
        ))
    end
    
    return test_set
end

#=
--------------------------------------------------------------------------------
2.  PREDICTION GENERATION (One function per model type)
--------------------------------------------------------------------------------
=#

"""
Generates predictions for the static Maher model.
"""
function generate_predictions(chain_static::Chains, test_set)
    log_α_s = Array(group(chain_static, :log_α_raw)) .- mean(Array(group(chain_static, :log_α_raw)), dims=2)
    log_β_s = Array(group(chain_static, :log_β_raw)) .- mean(Array(group(chain_static, :log_β_raw)), dims=2)
    log_home_adv_s = vec(Array(chain_static["log_home_adv"]))

    n_samples = size(chain_static, 1) * size(chain_static, 3)
    predictions = Dict()

    for match in test_set
        i, j = match.home_team, match.away_team
        simulated_outcomes = []
        for s in 1:n_samples
            log_λ = log_α_s[s, i] + log_β_s[s, j] + log_home_adv_s[s]
            log_μ = log_α_s[s, j] + log_β_s[s, i]
            push!(simulated_outcomes, (hg=rand(LogPoisson(log_λ)), ag=rand(LogPoisson(log_μ))))
        end
        predictions[match] = simulated_outcomes
    end
    return predictions
end

"""
Generates predictions for the AR1 model with static home advantage.
"""
function generate_predictions(chain_dynamic::Chains, data, test_set)
    raw_params = get_raw_parameters(chain_dynamic, data) # Assumes this helper exists
    n_samples = size(chain_dynamic, 1) * size(chain_dynamic, 3)
    ρ_attack_s = vec(Array(chain_dynamic["ρ_attack"]))
    ρ_defense_s = vec(Array(chain_dynamic["ρ_defense"]))
    log_home_adv_s = vec(Array(chain_dynamic["log_home_adv"])) # Static HA
    predictions = Dict()

    for match in test_set
        i, j = match.home_team, match.away_team
        simulated_outcomes = []
        for s in 1:n_samples
            last_α_raw = raw_params.log_α_raw[s, :, end]
            last_β_raw = raw_params.log_β_raw[s, :, end]
            
            future_α_raw = ρ_attack_s[s] .* last_α_raw .+ randn(data.n_teams) .* raw_params.σ_attack[s, :]
            future_β_raw = ρ_defense_s[s] .* last_β_raw .+ randn(data.n_teams) .* raw_params.σ_defense[s, :]

            future_log_α = future_α_raw .- mean(future_α_raw)
            future_log_β = future_β_raw .- mean(future_β_raw)
            
            log_λ = future_log_α[i] + future_log_β[j] + log_home_adv_s[s]
            log_μ = future_log_α[j] + future_log_β[i]
            
            push!(simulated_outcomes, (hg=rand(LogPoisson(log_λ)), ag=rand(LogPoisson(log_μ))))
        end
        predictions[match] = simulated_outcomes
    end
    return predictions
end

"""
Generates predictions for the AR1 model with dynamic home advantage.
"""
function generate_predictions_ha(chain_ha::Chains, data, test_set)
    raw_params = get_raw_parameters_ha(chain_ha, data) # Assumes HA-specific helper exists
    n_samples = size(chain_ha, 1) * size(chain_ha, 3)
    ρ_attack_s = vec(Array(chain_ha["ρ_attack"]))
    ρ_defense_s = vec(Array(chain_ha["ρ_defense"]))
    ρ_home_s = vec(Array(chain_ha["ρ_home"])) # Dynamic HA rho
    predictions = Dict()

    for match in test_set
        i, j = match.home_team, match.away_team
        simulated_outcomes = []
        for s in 1:n_samples
            # Propagate team strengths
            last_α_raw = raw_params.log_α_raw[s, :, end]
            last_β_raw = raw_params.log_β_raw[s, :, end]
            future_α_raw = ρ_attack_s[s] * last_α_raw .+ randn(data.n_teams) .* raw_params.σ_attack[s, :]
            future_β_raw = ρ_defense_s[s] * last_β_raw .+ randn(data.n_teams) .* raw_params.σ_defense[s, :]
            future_log_α = future_α_raw .- mean(future_α_raw)
            future_log_β = future_β_raw .- mean(future_β_raw)

            # Propagate home advantage
            last_ha_raw = raw_params.log_home_adv_raw[s, :, end]
            future_ha_raw = ρ_home_s[s] * last_ha_raw .+ randn() * raw_params.σ_home[s]

            log_λ = future_log_α[i] + future_log_β[j] + future_ha_raw[1] # [1] since only 1 league
            log_μ = future_log_α[j] + future_log_β[i]
            
            push!(simulated_outcomes, (hg=rand(LogPoisson(log_λ)), ag=rand(LogPoisson(log_μ))))
        end
        predictions[match] = simulated_outcomes
    end
    return predictions
end


#=
--------------------------------------------------------------------------------
3.  EVALUATION AND DISPLAY
--------------------------------------------------------------------------------
=#
# ... (simulations_to_probabilities and evaluate_predictions functions from your prompt) ...
function simulations_to_probabilities(simulations)
    n_sims = length(simulations)
    p_hw = count(s -> s.hg > s.ag, simulations) / n_sims
    p_d = count(s -> s.hg == s.ag, simulations) / n_sims
    p_aw = 1 - p_hw - p_d
    score_counts = Dict()
    for s in simulations
        score = (s.hg, s.ag)
        score_counts[score] = get(score_counts, score, 0) + 1
    end
    p_cs = Dict(score => count / n_sims for (score, count) in score_counts)
    return (p_1x2 = [p_hw, p_d, p_aw], p_cs = p_cs)
end

function evaluate_predictions(predictions, test_set)
    log_scores, brier_scores = [], []
    for match in test_set
        true_score = (match.true_home_goals, match.true_away_goals)
        probs = simulations_to_probabilities(predictions[match])
        p_true_score = get(probs.p_cs, true_score, 1e-9)
        push!(log_scores, log(p_true_score))
        outcome_vec = true_score[1] > true_score[2] ? [1,0,0] : (true_score[1] == true_score[2] ? [0,1,0] : [0,0,1])
        push!(brier_scores, sum((probs.p_1x2 - outcome_vec).^2))
    end
    return (log_score = mean(log_scores), brier_score = mean(brier_scores))
end


"""
Displays a formatted table comparing the evaluation scores of multiple models.
"""
function display_evaluation_results(all_scores::Dict)
    header = ["Metric", "Static Maher", "AR1 (Static HA)", "AR1 (Dynamic HA)"]
    data = Matrix{String}(undef, 2, 4)
    metrics = ["Log Score", "Brier Score"]
    
    for i in 1:2
        data[i, 1] = metrics[i]
        data[i, 2] = @sprintf("%.4f", all_scores["Static Maher"][i])
        data[i, 3] = @sprintf("%.4f", all_scores["AR1"][i])
        data[i, 4] = @sprintf("%.4f", all_scores["AR1-HA"][i])
    end
    
    println("\n--- Predictive Performance Evaluation ---")
    pretty_table(data, header=header)
end

end # end module

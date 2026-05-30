import re

with open("src/models/pregame/engines/player_level/time_decay/outfield_xg_dixon_coles.jl", "r") as f:
    code = f.read()

# 1. Rename struct
code = code.replace("DynamicDixonColesXGOutfieldPlayerTimeDecayModel", "DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel")

# 2. Rename builder
code = code.replace("build_dixon_coles_xg_market_player_engine", "build_double_poisson_xg_market_player_engine")

# 3. Change default market feature
code = code.replace("market_feature_config::M = Features.DixonColesMarketFeature()", "market_feature_config::M = Features.DoublePoissonMarketFeature()")

# 4. Strip out rho from model
# Find ρ_raw ~ Normal(0, 1.0) and ρ = 0.3 * tanh(ρ_raw)
code = re.sub(r'# Dixon-Coles Correlation Parameter\s*ρ_raw ~ Normal\(0, 1\.0\)\s*ρ = 0\.3 \* tanh\(ρ_raw\) # Bounded tightly\n', '', code)

# 5. Fix Market builder arguments
code = code.replace("market_log_λ_h::Vector{Float64},\n    market_log_λ_a::Vector{Float64},\n    market_ρ::Vector{Float64},", "market_log_λ_h::Vector{Float64},\n    market_log_λ_a::Vector{Float64},")

code = re.sub(r'# --- Dixon Coles Index Groupings ---\s*idx_00::Vector\{Int\},\s*idx_10::Vector\{Int\},\s*idx_01::Vector\{Int\},\s*idx_11::Vector\{Int\},', '', code)

# 6. Fix Likelihood block
old_lik = """    # Calculate Tau correction
    τ_term = ones(eltype(λ_goals_h), length(home_goals))
    if !isempty(idx_00) τ_term[idx_00] = 1.0 .- (λ_goals_h[idx_00] .* λ_goals_a[idx_00] .* ρ) end
    if !isempty(idx_10) τ_term[idx_10] = 1.0 .+ (λ_goals_a[idx_10] .* ρ) end
    if !isempty(idx_01) τ_term[idx_01] = 1.0 .+ (λ_goals_h[idx_01] .* ρ) end
    if !isempty(idx_11) τ_term[idx_11] .= 1.0 - ρ end

    # AD-Safe hard rejection for invalid Tau boundaries (prevents HMC flattening)
    if any(τ_term .<= 0.0)
        Turing.@addlogprob! -Inf
        return
    end

    # Combine into final likelihood vector for all matches
    log_lik_goals = log_lik_indep_h .+ log_lik_indep_a .+ log.(τ_term)   # Apply Match Weights globally to the combined goals likelihood"""

new_lik = """    # Combine into final likelihood vector for all matches
    log_lik_goals = log_lik_indep_h .+ log_lik_indep_a"""
code = code.replace(old_lik, new_lik)

# 7. Fix Market lik block
old_mlik = """        log_lik_market_h = logpdf.(Normal.(market_rate_h, σ_market), market_log_λ_h[idx_market])
        log_lik_market_a = logpdf.(Normal.(market_rate_a, σ_market), market_log_λ_a[idx_market])
        log_lik_market_ρ = logpdf(Normal(ρ, σ_market), mean(market_ρ[idx_market]))

        Turing.@addlogprob! config.market_weight * (sum(log_lik_market_h .* match_weights[idx_market]) + sum(log_lik_market_a .* match_weights[idx_market]) + log_lik_market_ρ)"""

new_mlik = """        log_lik_market_h = logpdf.(Normal.(market_rate_h, σ_market), market_log_λ_h[idx_market])
        log_lik_market_a = logpdf.(Normal.(market_rate_a, σ_market), market_log_λ_a[idx_market])

        Turing.@addlogprob! config.market_weight * (sum(log_lik_market_h .* match_weights[idx_market]) + sum(log_lik_market_a .* match_weights[idx_market]))"""
code = code.replace(old_mlik, new_mlik)

# 8. Fix build_turing_model mappings
code = code.replace("market_ρ     = Vector{Float64}(coalesce.(data[:flat_market_ρ], NaN))\n", "")

old_groupings = """    # Dixon-Coles groupings for unrolled likelihood
    idx_00, idx_10, idx_01, idx_11 = Int[], Int[], Int[], Int[]
    for i in eachindex(home_goals)
        h, a = home_goals[i], away_goals[i]
        if h == 0 && a == 0 push!(idx_00, i)
        elseif h == 1 && a == 0 push!(idx_10, i)
        elseif h == 0 && a == 1 push!(idx_01, i)
        elseif h == 1 && a == 1 push!(idx_11, i)
        end
    end"""
code = code.replace(old_groupings, "")

old_call = """        market_log_h, market_log_a, market_ρ, idx_market,
        idx_00, idx_10, idx_01, idx_11,
        n_teams, n_seasons, n_months,"""
new_call = """        market_log_h, market_log_a, idx_market,
        n_teams, n_seasons, n_months,"""
code = code.replace(old_call, new_call)

# 9. Fix extract_parameters
code = code.replace('ρ_vec = 0.3 * tanh(mean(chain["ρ_raw"]))', 'ρ_vec = 0.0')
code = code.replace('ρ_vec = Array(chain[:ρ])', 'ρ_vec = zeros(n_samples)')

with open("src/models/pregame/engines/player_level/time_decay/outfield_xg_double_poisson.jl", "w") as f:
    f.write(code)


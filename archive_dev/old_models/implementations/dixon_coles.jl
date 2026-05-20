# src/models/pregame/implementations/dixon_coles.jl

using Turing, Distributions, LinearAlgebra, Statistics
using ..MyDistributions 

export DixonColesNCP

# --- 1. Struct ---
Base.@kwdef struct DixonColesNCP <: AbstractStaticDixonColesModel 
    # Priors for the global parameters
    μ::Distribution   = Normal(0, 10) 
    γ::Distribution   = Normal(log(1.3), 0.2) 
    
    # Hierarchical Variance Priors
    σ_att::Distribution = Truncated(Normal(0, 0.5), 0, Inf) 
    σ_def::Distribution = Truncated(Normal(0, 0.5), 0, Inf)
    
    # Non-Centered Parameterization base
    Δₛ::Distribution  = Normal(0, 1)    
    
    # Rho (Independence) Prior
    # We use a Normal on the raw space, which will be tanh-transformed.
    ρ_raw::Distribution = Normal(0, 1.0) 
end 

function Base.show(io::IO, ::MIME"text/plain", m::DixonColesNCP)
    printstyled(io, "Static Dixon-Coles (Vectorized Log-Space)\n", color=:magenta, bold=true)
    println(io, "  ├── Att/Def Variance: $(m.σ_att)")
    println(io, "  └── Rho (Raw): $(m.ρ_raw)")
end

# --- 2. Model Definition ---
@model function dixon_coles_model_train(
    n_teams, 
    home_ids, away_ids,
    # Grouped Indices
    idx_00, idx_10, idx_01, idx_11, idx_else,
    # Data for the 'else' group
    scores_else_x, scores_else_y,
    model::DixonColesNCP
)

    # A. Global Parameters
    μ  ~ model.μ        
    γ  ~ model.γ        
    
    # Variances
    σₐ ~ model.σ_att  
    σ_d ~ model.σ_def  
    
    # B. Rho (Dependence) - Tanh Parameterization
    # Maps (-Inf, Inf) -> (-0.3, 0.3)
    # 0.3 is chosen as a safe bound where 1/(λμ) is usually > 0.3
    ρ_raw ~ model.ρ_raw
    ρ = 0.3 * tanh(ρ_raw)

    # C. Team Parameters (NCP)
    # We use sum-to-zero centering for identifiability
    z_att ~ filldist(model.Δₛ, n_teams) 
    z_def ~ filldist(model.Δₛ, n_teams)

    att_raw = z_att .* σₐ 
    def_raw = z_def .* σ_d 
    att = att_raw .- mean(att_raw)
    def = def_raw .- mean(def_raw)

    # D. Log-Rate Calculation (Vectorized for ALL matches)
    # θ = log(λ)
    θ_home_all = μ .+ γ .+ att[home_ids] .+ def[away_ids]
    θ_away_all = μ      .+ att[away_ids] .+ def[home_ids]
    
    # E. Vectorized Likelihood Execution by Group
    
    # 1. Group 0-0
    if !isempty(idx_00)
        # We slice the global θ vectors using the group indices
        Turing.@addlogprob! logpdf(
            DixonColesLogGroup(θ_home_all[idx_00], θ_away_all[idx_00], ρ, :s00),
            # Pass a dummy matrix of correct size (2 x N_subset)
            zeros(2, length(idx_00)) 
        )
    end

    # 2. Group 1-0
    if !isempty(idx_10)
        Turing.@addlogprob! logpdf(
            DixonColesLogGroup(θ_home_all[idx_10], θ_away_all[idx_10], ρ, :s10),
            zeros(2, length(idx_10))
        )
    end

    # 3. Group 0-1
    if !isempty(idx_01)
        Turing.@addlogprob! logpdf(
            DixonColesLogGroup(θ_home_all[idx_01], θ_away_all[idx_01], ρ, :s01),
            zeros(2, length(idx_01))
        )
    end
    
    # 4. Group 1-1
    if !isempty(idx_11)
        Turing.@addlogprob! logpdf(
            DixonColesLogGroup(θ_home_all[idx_11], θ_away_all[idx_11], ρ, :s11),
            zeros(2, length(idx_11))
        )
    end

    # 5. Group Else (Standard Independent Poisson)
    # For scores >= 2, Dixon-Coles correction τ=1, so log(τ)=0.
    # We just sum standard Poisson log-likelihoods.
    if !isempty(idx_else)
        θ_h_else = θ_home_all[idx_else]
        θ_a_else = θ_away_all[idx_else]
        
        # Vectorized Poisson LogPdf: k*θ - exp(θ) - log(k!)
        # Using Turing's logpdf for Poisson is efficient enough here
        Turing.@addlogprob! sum(logpdf.(Poisson.(exp.(θ_h_else)), scores_else_x))
        Turing.@addlogprob! sum(logpdf.(Poisson.(exp.(θ_a_else)), scores_else_y))
    end
end

# --- 3. Builder ---
function build_turing_model(model::DixonColesNCP, feature_set::FeatureSet)
    flat_home = feature_set[:flat_home_goals]
    flat_away = feature_set[:flat_away_goals]
    home_ids = feature_set[:flat_home_ids]
    away_ids = feature_set[:flat_away_ids]
    
    # --- Pre-processing: Group Matches by Score ---
    # This happens once, before training, to enable the vectorization
    
    idx_00 = Int[]
    idx_10 = Int[]
    idx_01 = Int[]
    idx_11 = Int[]
    idx_else = Int[]

    for i in eachindex(flat_home)
        h, a = flat_home[i], flat_away[i]
        
        if h == 0 && a == 0
            push!(idx_00, i)
        elseif h == 1 && a == 0
            push!(idx_10, i)
        elseif h == 0 && a == 1
            push!(idx_01, i)
        elseif h == 1 && a == 1
            push!(idx_11, i)
        else
            push!(idx_else, i)
        end
    end

    # Extract scores for the 'else' group (needed for standard Poisson calc)
    scores_else_x = flat_home[idx_else]
    scores_else_y = flat_away[idx_else]

    return dixon_coles_model_train(
        feature_set[:n_teams]::Int,
        home_ids,
        away_ids,
        idx_00, idx_10, idx_01, idx_11, idx_else,
        scores_else_x, scores_else_y,
        model
    )
end

# --- 4. Extractor ---
function extract_parameters(
    model::DixonColesNCP, 
    df_to_predict::AbstractDataFrame,
    feature_set::FeatureSet, 
    chains::Chains
)::Dict{Int, NamedTuple}

    extraction_dict = Dict{Int, NamedTuple}()
    team_map = feature_set[:team_map]

    # Extract Global Params
    μ = vec(chains[:μ])
    γ = vec(chains[:γ])
    
    # Extract Rho
    # If the chain saved 'ρ', use it. If only 'ρ_raw', transform it.
    if :ρ in names(chains)
        ρ_vec = vec(chains[:ρ])
    else
        ρ_raw_vec = vec(chains[:ρ_raw])
        ρ_vec = 0.3 .* tanh.(ρ_raw_vec)
    end
    
    # Extract Variance & Team Params
    σₐ_vec = vec(chains[:σₐ])
    σ_d_vec = vec(chains[:σ_d])
    z_att_mat = Array(group(chains, :z_att))
    z_def_mat = Array(group(chains, :z_def))

    # Reconstruct NCP & Center
    att_mat = z_att_mat .* σₐ_vec
    def_mat = z_def_mat .* σ_d_vec
    att_c = att_mat .- mean(att_mat, dims=2)
    def_c = def_mat .- mean(def_mat, dims=2)

    for row in eachrow(df_to_predict)
        h_id = team_map[row.home_team]
        a_id = team_map[row.away_team]

        α_h = att_c[:, h_id]
        β_a = def_c[:, a_id]
        α_a = att_c[:, a_id]
        β_h = def_c[:, h_id]

        # Calculate Log-Rates (Theta)
        θ_1 = μ .+ γ .+ α_h .+ β_a # Home Log-Rate
        θ_2 = μ      .+ α_a .+ β_h # Away Log-Rate
        θ_3 = ρ_vec                # Rho (Dependence)

        # Return Thetas + Rho. 
        # Note: We return θ_3 as Rho for consistency with Bivariate interface, 
        # or you can name it rho explicitly.
        extraction_dict[row.match_id] = (; θ_1, θ_2, θ_3)
    end
    return extraction_dict
end

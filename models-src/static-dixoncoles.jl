using DataFrames
using Turing
using LinearAlgebra
using Base.Threads
using ...MyDistributions: DixonColes

export StaticDixonColes, build_turing_model, predict


struct StaticDixonColes <: AbstractDixonColesModel end 

@model function dixon_coles_turing_model(n_teams, home_ids, away_ids, 
                                         observed_data, ::Type{T} = Float64) where {T}
    # Priors
    log_α_raw ~ MvNormal(zeros(n_teams), 0.5 * I)
    log_β_raw ~ MvNormal(zeros(n_teams), 0.5 * I)
    home_adv  ~ Normal(log(1.3), 0.2)
    ρ         ~ Uniform(-0.3, 0.3)

    # Identifiability
    log_α := log_α_raw .- mean(log_α_raw)
    log_β := log_β_raw .- mean(log_β_raw)

    # Rates
    log_λs = home_adv .+ log_α[home_ids] .+ log_β[away_ids]
    log_μs = log_α[away_ids] .+ log_β[home_ids]
    λs = exp.(log_λs)
    μs = exp.(log_μs)

    observed_data ~ arraydist(DixonColes.(λs, μs, ρ))

end


function build_turing_model(model::StaticDixonColes, feature_set::FeatureSet)
    data = TuringHelpers.prepare_data(model, feature_set)
    # Stack home/away goals into 2xN matrix for arraydist
    obs_matrix = stack([data.flat_home_goals, data.flat_away_goals], dims=1)
    
    return dixon_coles_turing_model(
        data.n_teams, 
        data.flat_home_ids, 
        data.flat_away_ids, 
        obs_matrix
    )
end

"""
    extract_parameters(model::DixonColesModel, df_predict, vocab, chains)

Extracts the posterior samples for Home Expected Goals (λ_h), Away Expected Goals (λ_a), 
and the Correlation Coefficient (ρ) for each match in the dataframe.
"""
function extract_parameters(model::StaticDixonColes, df_to_predict::AbstractDataFrame, vocabulary::Vocabulary, chains::Chains)
    
    # 1. Define the Output Type
    # We return a NamedTuple containing vectors of samples for each parameter
    ValueType = NamedTuple{(:λ_h, :λ_a, :ρ), Tuple{AbstractVector{Float64}, AbstractVector{Float64}, AbstractVector{Float64}}}
    
    extraction_dict = Dict{Int64, ValueType}()

    # 2. Extract Global Parameters (Vectorized over all chain samples)
    # These are the same for every match
    home_adv_vec = vec(chains[Symbol("home_adv")])
    rho_vec      = vec(chains[Symbol("ρ")]) 

    # 3. Iterate over each match to calculate specific parameters
    for row in eachrow(df_to_predict)
        # Look up Team IDs
        h_id = vocabulary.mappings[:team_map][row.home_team]
        a_id = vocabulary.mappings[:team_map][row.away_team]

        # Extract Team Strengths from Chains
        # "log_α[i]" corresponds to the attack strength of team i
        log_att_h = vec(chains[Symbol("log_α[$h_id]")])
        log_def_a = vec(chains[Symbol("log_β[$a_id]")])
        
        log_att_a = vec(chains[Symbol("log_α[$a_id]")])
        log_def_h = vec(chains[Symbol("log_β[$h_id]")])

        # 4. Calculate Expected Goals (λ)
        # We perform element-wise arithmetic on the vectors to preserve the joint posterior distribution
        
        # λ_home = exp(Home Advantage + Home Attack + Away Defense)
        λ_h = exp.(home_adv_vec .+ log_att_h .+ log_def_a)
        
        # λ_away = exp(Away Attack + Home Defense)
        λ_a = exp.(log_att_a .+ log_def_h)

        # 5. Store in Dictionary
        # Note: 'rho_vec' is passed by reference, so this is memory efficient
        extraction_dict[Int(row.match_id)] = (; λ_h, λ_a, ρ = rho_vec)
    end

    return extraction_dict
end

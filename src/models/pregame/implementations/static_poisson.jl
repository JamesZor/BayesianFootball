# src/models/pregame/implementations/static_poisson.jl

export StaticPoisson

# --- 1. Struct ---
Base.@kwdef struct StaticPoisson{D<:Distribution} <: AbstractStaticPoissonModel
    prior::D = Normal(0, 0.5) 
end

function Base.show(io::IO, ::MIME"text/plain", m::StaticPoisson)
    printstyled(io, "StaticPoisson Model\n", color=:cyan, bold=true)
    println(io, "  Prior: $(m.prior)")
end

# --- 2. Model Definition ---
@model function static_poisson_model_train(n_teams, home_ids, away_ids, home_goals, away_goals, model::StaticPoisson)
    # ... (same logic as before) ...
    log_α_raw ~ filldist(model.prior, n_teams) 
    log_β_raw ~ filldist(model.prior, n_teams) 
    home_adv ~ Normal(log(1.3), 0.2)

    log_α = log_α_raw .- mean(log_α_raw)
    log_β = log_β_raw .- mean(log_β_raw)

    home_goals ~ arraydist(LogPoisson.(home_adv .+ log_α[home_ids] .+ log_β[away_ids]))
    away_goals ~ arraydist(LogPoisson.(log_α[away_ids] .+ log_β[home_ids]))
end

# --- 3. Builder ---
function build_turing_model(model::StaticPoisson, feature_set::FeatureSet)
    # Using dictionary syntax feature_set[:key]
    return static_poisson_model_train(
        feature_set[:n_teams]::Int,
        feature_set[:flat_home_ids],
        feature_set[:flat_away_ids],
        feature_set[:flat_home_goals],
        feature_set[:flat_away_goals],
        model
    )
end


# --- 4. The Worker (Single Split) ---
"""
    extract_parameters(model, feature_set, chains)

Extracts parameters using the mappings contained within the feature_set.
"""
function extract_parameters(
    model::StaticPoisson, 
    df_to_predict::AbstractDataFrame,
    feature_set::FeatureSet,  # <-- Replaces vocabulary
    chains::Chains
)::Dict{Int, PoissonRates}

    extraction_dict = Dict{Int64, PoissonRates}()
    
    # Retrieve the exact map and data used for this specific training run
    team_map = feature_set[:team_map]
    # We retrieve the dataframe stored inside the features to iterate over matches

    home_adv_vec = vec(chains["home_adv"])

    for row in eachrow(df_to_predict)
        # 1. Get string names
        h_name = row.home_team
        a_name = row.away_team
        
        # 2. Convert to ID using the LOCAL map
        h_id = team_map[h_name]
        a_id = team_map[a_name]

        # 3. Extract chains for these specific IDs
        alpha_h = vec(chains["log_α_raw[$h_id]"]) 
        beta_a  = vec(chains["log_β_raw[$a_id]"])
        alpha_a = vec(chains["log_α_raw[$a_id]"])
        beta_h  = vec(chains["log_β_raw[$h_id]"])

        λ_h = exp.(alpha_h .+ beta_a .+ home_adv_vec)
        λ_a = exp.(alpha_a .+ beta_h)
        
        extraction_dict[row.match_id] = (; λ_h, λ_a)
    end
    return extraction_dict
end


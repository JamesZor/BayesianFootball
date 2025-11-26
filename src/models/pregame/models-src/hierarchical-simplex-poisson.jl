# --- Model File: hierarchical_simplex_poisson.jl ---
using Turing
using DataFrames
using LinearAlgebra
using Distributions

export HierarchicalSimplexPoisson, build_turing_model

# 1. DEFINE A CONCRETE STRUCT FOR THE MODEL
struct HierarchicalSimplexPoisson <: AbstractPregameModel end 

# 2. DEFINE THE TURING MODEL LOGIC
@model function hierarchical_simplex_turing_model(n_teams, home_ids, away_ids, home_goals, away_goals)
    
    # --- Priors for Scaling Factors (How much do teams vary?) ---
    # We constrain these to be positive to resolve sign ambiguity with the simplex
    home_att_scale ~ truncated(Normal(0, 10), 0, Inf)
    away_att_scale ~ truncated(Normal(0, 10), 0, Inf)
    home_def_scale ~ truncated(Normal(0, 10), 0, Inf)
    away_def_scale ~ truncated(Normal(0, 10), 0, Inf)

    # --- Priors for Simplexes ---
    # A Dirichlet(1.0) is uniform over the simplex.
    # We use this to get a "shape" vector that sums to 1.
    home_att_raw ~ Dirichlet(n_teams, 1.0)
    away_att_raw ~ Dirichlet(n_teams, 1.0)
    home_def_raw ~ Dirichlet(n_teams, 1.0)
    away_def_raw ~ Dirichlet(n_teams, 1.0)

    # --- Transformed Parameters (Enforce sum-to-zero) ---
    # 1. Center the simplex (0 to 1) around 0 by subtracting mean (1/N)
    # 2. Scale it by the magnitude parameter
    # Result: A vector that sums to 0, scaled to the appropriate magnitude.
    centering = 1.0 / n_teams
    home_att = home_att_scale .* (home_att_raw .- centering)
    away_att = away_att_scale .* (away_att_raw .- centering)
    home_def = home_def_scale .* (home_def_raw .- centering)
    away_def = away_def_scale .* (away_def_raw .- centering)

    # --- Calculate Goal Rates ---
    # Home Goals = Home Team's Home Attack + Away Team's Away Defense
    log_λs = home_att[home_ids] .+ away_def[away_ids]
    
    # Away Goals = Away Team's Away Attack + Home Team's Home Defense
    log_μs = away_att[away_ids] .+ home_def[home_ids]

    # --- Likelihood ---
    home_goals ~ arraydist(LogPoisson.(log_λs))
    away_goals ~ arraydist(LogPoisson.(log_μs))
end

# 3. BUILDER FUNCTION
function build_turing_model(model::HierarchicalSimplexPoisson, feature_set::FeatureSet)
    data = TuringHelpers.prepare_data(model, feature_set)
    
    # Stack home/away goals into 2xN matrix for arraydist

    return hierarchical_simplex_turing_model(
        data.n_teams, 
        data.flat_home_ids, 
        data.flat_away_ids, 
        data.flat_home_goals, 
        data.flat_away_goals
    )
end

# 4. PREDICTION BUILDER
function build_turing_model(model::HierarchicalSimplexPoisson, n_teams::Int, home_ids::Vector{Int}, away_ids::Vector{Int})
    return hierarchical_simplex_turing_model(n_teams, home_ids, away_ids, missing)
end

# 5. PARAMETER EXTRACTION
function extract_parameters(model::HierarchicalSimplexPoisson, df_to_predict::AbstractDataFrame, vocabulary::Vocabulary, chains::Chains)
    ValueType = NamedTuple{(:λ_h, :λ_a), Tuple{AbstractVector{Float64}, AbstractVector{Float64}}}
    extraction_dict = Dict{Int64, ValueType}()

    # There is no global intercept in this specific parameterization
    # (Goal rates are purely sum of team strengths)
    
    # Pre-fetch the scaling factors (scalar chains)
    h_att_scale = vec(chains[:home_att_scale])
    a_att_scale = vec(chains[:away_att_scale])
    h_def_scale = vec(chains[:home_def_scale])
    a_def_scale = vec(chains[:away_def_scale])

    n_teams = vocabulary.mappings[:n_teams]
    centering = 1.0 / n_teams

    for row in eachrow(df_to_predict)
        h_id = vocabulary.mappings[:team_map][row.home_team]
        a_id = vocabulary.mappings[:team_map][row.away_team]

        # Extract Raw Simplex Values (these sum to 1 across teams)
        # "home_att_raw[h_id]" is a vector of samples for that specific team
        raw_h_att = vec(chains[Symbol("home_att_raw[$h_id]")])
        raw_a_def = vec(chains[Symbol("away_def_raw[$a_id]")])
        
        raw_a_att = vec(chains[Symbol("away_att_raw[$a_id]")])
        raw_h_def = vec(chains[Symbol("home_def_raw[$h_id]")])

        # Transform to Real Scale (Centering & Scaling)
        # We replicate the model logic here on the posterior samples
        real_h_att = h_att_scale .* (raw_h_att .- centering)
        real_a_def = a_def_scale .* (raw_a_def .- centering)
        
        real_a_att = a_att_scale .* (raw_a_att .- centering)
        real_h_def = h_def_scale .* (raw_h_def .- centering)

        # Calculate Lambdas
        λ_h = exp.(real_h_att .+ real_a_def)
        λ_a = exp.(real_a_att .+ real_h_def)

        extraction_dict[Int(row.match_id)] = (; λ_h, λ_a)
    end

    return extraction_dict
end

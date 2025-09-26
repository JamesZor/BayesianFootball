# workspace/basic_state_space/setup

using DataFrames
using Dates

"""
    add_global_round_column!(matches_df::DataFrame)

Adds a `:global_round` column in-place to the DataFrame.

This function sorts matches chronologically and groups them into the minimum
number of time steps (`global_round`) such that no team plays more than
once within a single time step. This is essential for sequential state-space models.

# Arguments
- `matches_df::DataFrame`: The DataFrame of matches. Must contain `:date`, 
  `:home_team`, and `:away_team` columns.

# Returns
- The modified `DataFrame` with the new `:global_round` column.
"""
function add_global_round_column!(matches_df::DataFrame)
    # 1. Ensure data is in chronological order.
    sort!(matches_df, :match_date)

    # 2. Pre-allocate a vector to hold the new column's data for efficiency.
    num_matches = nrow(matches_df)
    global_rounds = Vector{Int}(undef, num_matches)
    
    # 3. Initialize state variables.
    global_round_counter = 1
    # A Set provides fast lookups to check if a team has played in the current round.
    teams_in_current_round = Set{String}()

    # 4. Iterate through each match to assign a global round.
    for (i, row) in enumerate(eachrow(matches_df))
        home_team = row.home_team
        away_team = row.away_team

        # If either team has already been assigned a match in this round,
        # we must start a new round.
        if home_team in teams_in_current_round || away_team in teams_in_current_round
            global_round_counter += 1
            empty!(teams_in_current_round) # Clear the set for the new round.
        end

        # Assign the current round number to the match.
        global_rounds[i] = global_round_counter
        
        # Add the teams from this match to the set for the current round.
        push!(teams_in_current_round, home_team)
        push!(teams_in_current_round, away_team)
    end


    # 5. Add the newly created vector as a column to the DataFrame.
    matches_df.global_round = global_rounds
    
    println("✅ Successfully added `:global_round` column. Found $(global_round_counter) unique time steps.")

    return matches_df
end


################################################################################
# AR1 model poisson  - alpha, beta - static home 
################################################################################

module AR1StateSpace

using Turing
using LinearAlgebra
using Distributions
using DataFrames
using BayesianFootball


# Export the new model type so it can be used in runner scripts
export AR1PoissonModel

#=
--------------------------------------------------------------------------------
1.  MODEL DEFINITION STRUCT
--------------------------------------------------------------------------------
This struct identifies our new model. The training pipeline will use this
type to dispatch to the correct feature and model-building implementations.
=#
struct AR1PoissonModel <: AbstractModelDefinition end

#=
--------------------------------------------------------------------------------
2.  FRAMEWORK INTERFACE IMPLEMENTATION
--------------------------------------------------------------------------------
These functions tell the BayesianFootball framework how to handle the
AR1PoissonModel.
=#

"""
    get_required_features(::AR1PoissonModel)

Tells the training morphism which core features to extract. We need the new
`:global_round` column to structure the data correctly.
"""
function BayesianFootball.get_required_features(::AR1PoissonModel)
    return (
        :home_team_ids,
        :away_team_ids,
        :n_teams,
        :global_round # <-- The crucial new feature
    )
end

"""
    build_turing_model(::AR1PoissonModel, features, goals_home, goals_away)

Constructs the Turing model instance.

This is the key integration point. It takes the standard "flat" feature vectors
from the framework and transforms them into the nested `Vector{Vector{Int}}`
format required by the sequential AR(1) model. This transformation is done
by grouping the data by the `:global_round` column.
"""
function BayesianFootball.build_turing_model(
    ::AR1PoissonModel,
    features::NamedTuple,
    goals_home::Vector{Int},
    goals_away::Vector{Int}
)
    # Create a temporary DataFrame for easy grouping
    temp_df = DataFrame(
        global_round = features.global_round,
        home_id = features.home_team_ids,
        away_id = features.away_team_ids,
        gh = goals_home,
        ga = goals_away
    )
    
    # Group by the time-step identifier
    grouped = groupby(temp_df, :global_round)
    n_rounds = length(grouped)

    # Pre-allocate nested vectors for the model arguments
    home_team_ids_by_round = Vector{Vector{Int}}(undef, n_rounds)
    away_team_ids_by_round = Vector{Vector{Int}}(undef, n_rounds)
    home_goals_by_round = Vector{Vector{Int}}(undef, n_rounds)
    away_goals_by_round = Vector{Vector{Int}}(undef, n_rounds)

    # Populate the nested vectors
    for (i, round_df) in enumerate(grouped)
        home_team_ids_by_round[i] = round_df.home_id
        away_team_ids_by_round[i] = round_df.away_id
        home_goals_by_round[i] = round_df.gh
        away_goals_by_round[i] = round_df.ga
    end
    
    # Instantiate and return the Turing model with the correctly structured data
    return ar1_poisson_model(
        home_team_ids_by_round,
        away_team_ids_by_round,
        home_goals_by_round,
        away_goals_by_round,
        features.n_teams,
        n_rounds
    )
end


#=
--------------------------------------------------------------------------------
3.  TURING @MODEL DEFINITION
--------------------------------------------------------------------------------
This is the dynamic state-space model, ported directly from the logic in
`02_ar1_play.jl`. It uses a non-centered parameterization for efficiency.
[cite: 57, 58, 59]
=#
@model function ar1_poisson_model(
    home_team_ids::Vector{<:Vector}, 
    away_team_ids::Vector{<:Vector}, 
    home_goals::Vector{<:Vector}, 
    away_goals::Vector{<:Vector}, 
    n_teams::Int, 
    n_rounds::Int
)
    # --- Priors for AR(1) process ---
    # Persistence parameters
    ρ_attack ~ Beta(10, 1.5)
    ρ_defense ~ Beta(10, 1.5)

    # Volatility parameters (non-centered)
    μ_log_σ_attack ~ Normal(-2.5, 0.5) 
    τ_log_σ_attack ~ Truncated(Normal(0, 0.2), 0, Inf)
    μ_log_σ_defense ~ Normal(-2.5, 0.5) 
    τ_log_σ_defense ~ Truncated(Normal(0, 0.2), 0, Inf)

    z_log_σ_attack ~ MvNormal(zeros(n_teams), I)
    z_log_σ_defense ~ MvNormal(zeros(n_teams), I)
    log_σ_attack = μ_log_σ_attack .+ z_log_σ_attack * τ_log_σ_attack
    log_σ_defense = μ_log_σ_defense .+ z_log_σ_defense * τ_log_σ_defense
    σ_attack = exp.(log_σ_attack)
    σ_defense = exp.(log_σ_defense)
    
    # Home advantage
    log_home_adv ~ Normal(log(1.3), 0.2)

    # --- Latent State Variables ---
    # Initial states (t=0)
    initial_α_z ~ MvNormal(zeros(n_teams), I)
    initial_β_z ~ MvNormal(zeros(n_teams), I)
    log_α_raw_t0 = initial_α_z * sqrt(0.5)
    log_β_raw_t0 = initial_β_z * sqrt(0.5)

    # Innovations for all subsequent time steps
    z_α ~ MvNormal(zeros(n_teams * n_rounds), I)
    z_β ~ MvNormal(zeros(n_teams * n_rounds), I)
    z_α_mat = reshape(z_α, n_teams, n_rounds)
    z_β_mat = reshape(z_β, n_teams, n_rounds)

    # Matrix to store the full time-series of team strengths
    log_α_raw = Matrix{Real}(undef, n_teams, n_rounds)
    log_β_raw = Matrix{Real}(undef, n_teams, n_rounds)

    # --- Main Time-Series Loop ---
    for t in 1:n_rounds
        # 1. Evolve the latent state according to the AR(1) process
        if t == 1
            log_α_raw[:, 1] = log_α_raw_t0 .+ z_α_mat[:, 1] .* σ_attack
            log_β_raw[:, 1] = log_β_raw_t0 .+ z_β_mat[:, 1] .* σ_defense
        else
            log_α_raw[:, t] = ρ_attack * log_α_raw[:, t-1] .+ z_α_mat[:, t] .* σ_attack
            log_β_raw[:, t] = ρ_defense * log_β_raw[:, t-1] .+ z_β_mat[:, t] .* σ_defense
        end

        # 2. Apply sum-to-zero constraint for identifiability
        log_α_t = log_α_raw[:, t] .- mean(log_α_raw[:, t])
        log_β_t = log_β_raw[:, t] .- mean(log_β_raw[:, t])
    
        # 3. Connect to data via likelihood
        home_ids = home_team_ids[t]
        away_ids = away_team_ids[t]
        
        # Check if the round is empty (can happen in CV splits)
        if !isempty(home_ids)
            log_λs = log_α_t[home_ids] .+ log_β_t[away_ids] .+ log_home_adv
            log_μs = log_α_t[away_ids] .+ log_β_t[home_ids]
            home_goals[t] .~ LogPoisson.(log_λs)
            away_goals[t] .~ LogPoisson.(log_μs)
        end
    end
end

end # end module AR1StateSpace




# workspace/neg_bin_ar1/setup.jl

module AR1NegativeBinomial

using Turing
using LinearAlgebra
using Distributions
using DataFrames
using BayesianFootball
using Statistics # For mean()

# Export the new model type so it can be used in runner scripts
export AR1NegativeBinomialModel

#=
--------------------------------------------------------------------------------
1.  MODEL DEFINITION STRUCT
--------------------------------------------------------------------------------
This struct identifies our new Negative Binomial model.
=#
struct AR1NegativeBinomialModel <: AbstractModelDefinition end

#=
--------------------------------------------------------------------------------
2.  FRAMEWORK INTERFACE IMPLEMENTATION
--------------------------------------------------------------------------------
These functions tell the BayesianFootball framework how to handle the new model.
They are almost identical to the Poisson version, just dispatched on the new type.
=#

"""
    get_required_features(::AR1NegativeBinomialModel)

Declares the data features required by this model.
"""
function BayesianFootball.get_required_features(::AR1NegativeBinomialModel)
    return (
        :home_team_ids,
        :away_team_ids,
        :n_teams,
        :global_round
    )
end

"""
    build_turing_model(::AR1NegativeBinomialModel, features, goals_home, goals_away)

Constructs the Turing model instance by grouping the flat data vectors by
the `:global_round` feature.
"""
function BayesianFootball.build_turing_model(
    ::AR1NegativeBinomialModel,
    features::NamedTuple,
    goals_home::Vector{Int},
    goals_away::Vector{Int}
)
    # This data preparation is identical to the Poisson AR(1) version
    temp_df = DataFrame(
        global_round = features.global_round,
        home_id = features.home_team_ids,
        away_id = features.away_team_ids,
        gh = goals_home,
        ga = goals_away
    )
    
    grouped = groupby(temp_df, :global_round)
    n_rounds = length(grouped)

    # Pre-allocate nested vectors for the model arguments
    home_team_ids_by_round = Vector{Vector{Int}}(undef, n_rounds)
    away_team_ids_by_round = Vector{Vector{Int}}(undef, n_rounds)
    home_goals_by_round = Vector{Vector{Int}}(undef, n_rounds)
    away_goals_by_round = Vector{Vector{Int}}(undef, n_rounds)

    # Populate the nested vectors
    for (i, round_df) in enumerate(grouped)
        home_team_ids_by_round[i] = round_df.home_id
        away_team_ids_by_round[i] = round_df.away_id
        home_goals_by_round[i] = round_df.gh
        away_goals_by_round[i] = round_df.ga
    end
    #
    # home_team_ids_by_round = [g.home_id for g in grouped]
    # away_team_ids_by_round = [g.away_id for g in grouped]
    # home_goals_by_round = [g.gh for g in grouped]
    # away_goals_by_round = [g.ga for g in grouped]
    
    # Call the new Negative Binomial Turing model
    return ar1_neg_bin_model(
        home_team_ids_by_round,
        away_team_ids_by_round,
        home_goals_by_round,
        away_goals_by_round,
        features.n_teams,
        n_rounds
    )
end

#=
--------------------------------------------------------------------------------
3.  TURING @MODEL DEFINITION
--------------------------------------------------------------------------------
This is the dynamic state-space model with a Negative Binomial likelihood to
account for overdispersion in goal scoring.
=#
@model function ar1_neg_bin_model(
    home_team_ids::Vector{<:AbstractVector}, 
    away_team_ids::Vector{<:AbstractVector}, 
    home_goals::Vector{<:AbstractVector}, 
    away_goals::Vector{<:AbstractVector}, 
    n_teams::Int, 
    n_rounds::Int
)
    # --- Priors for AR(1) process (Identical to Poisson model) ---
    ρ_attack ~ Beta(10, 1.5)
    ρ_defense ~ Beta(10, 1.5)

    μ_log_σ_attack ~ Normal(-2.5, 0.5) 
    τ_log_σ_attack ~ Truncated(Normal(0, 0.2), 0, Inf)
    μ_log_σ_defense ~ Normal(-2.5, 0.5) 
    τ_log_σ_defense ~ Truncated(Normal(0, 0.2), 0, Inf)

    z_log_σ_attack ~ MvNormal(zeros(n_teams), I)
    z_log_σ_defense ~ MvNormal(zeros(n_teams), I)
    log_σ_attack = μ_log_σ_attack .+ z_log_σ_attack * τ_log_σ_attack
    log_σ_defense = μ_log_σ_defense .+ z_log_σ_defense * τ_log_σ_defense
    σ_attack = exp.(log_σ_attack)
    σ_defense = exp.(log_σ_defense)
    
    log_home_adv ~ Normal(log(1.3), 0.2)

    # --- NEW: Prior for the Negative Binomial dispersion parameter ---
    # A Gamma prior is a good choice for a positive, continuous parameter.
    # This prior is weakly informative and centered around a plausible value.
    ϕ ~ Gamma(2, 0.1)

    # --- Latent State Variables (Identical to Poisson model) ---
    initial_α_z ~ MvNormal(zeros(n_teams), I)
    initial_β_z ~ MvNormal(zeros(n_teams), I)
    log_α_raw_t0 = initial_α_z * sqrt(0.5)
    log_β_raw_t0 = initial_β_z * sqrt(0.5)

    z_α ~ MvNormal(zeros(n_teams * n_rounds), I)
    z_β ~ MvNormal(zeros(n_teams * n_rounds), I)
    z_α_mat = reshape(z_α, n_teams, n_rounds)
    z_β_mat = reshape(z_β, n_teams, n_rounds)

    log_α_raw = Matrix{Real}(undef, n_teams, n_rounds)
    log_β_raw = Matrix{Real}(undef, n_teams, n_rounds)

    # --- Main Time-Series Loop (Identical to Poisson model) ---
    for t in 1:n_rounds
        # 1. Evolve the latent state
        if t == 1
            log_α_raw[:, 1] = log_α_raw_t0 .+ z_α_mat[:, 1] .* σ_attack
            log_β_raw[:, 1] = log_β_raw_t0 .+ z_β_mat[:, 1] .* σ_defense
        else
            log_α_raw[:, t] = ρ_attack * log_α_raw[:, t-1] .+ z_α_mat[:, t] .* σ_attack
            log_β_raw[:, t] = ρ_defense * log_β_raw[:, t-1] .+ z_β_mat[:, t] .* σ_defense
        end

        # 2. Apply sum-to-zero constraint
        log_α_t = log_α_raw[:, t] .- mean(log_α_raw[:, t])
        log_β_t = log_β_raw[:, t] .- mean(log_β_raw[:, t])
    
        # 3. Connect to data via likelihood
        home_ids = home_team_ids[t]
        away_ids = away_team_ids[t]
        
        if !isempty(home_ids)
            log_λs = log_α_t[home_ids] .+ log_β_t[away_ids] .+ log_home_adv
            log_μs = log_α_t[away_ids] .+ log_β_t[home_ids]
            
            # --- NEW: Negative Binomial Likelihood ---
            # We use the mean-dispersion parameterization: NegativeBinomial(μ, ϕ)
            # where μ is the mean and ϕ is the dispersion parameter.
            home_goals[t] .~ NegativeBinomial.(exp.(log_λs), ϕ)
            away_goals[t] .~ NegativeBinomial.(exp.(log_μs), ϕ)
        end
    end
end

end # end module AR1NegativeBinomial



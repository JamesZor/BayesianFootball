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
            
            # Define the log-odds for the success probability `p`
            logit_ps_home = log(ϕ) .- log_λs
            logit_ps_away = log(ϕ) .- log_μs

            # Use the logistic function (inv_logit) to ensure p is between 0 and 1
            ps_home = Turing.logistic.(logit_ps_home)
            ps_away = Turing.logistic.(logit_ps_away)

            # Use the (r, p) parameterization, which is more numerically stable
            home_goals[t] .~ NegativeBinomial.(ϕ, ps_home)
            away_goals[t] .~ NegativeBinomial.(ϕ, ps_away)

        end
    end
end

end # end module AR1NegativeBinomial

# workspace/neg_bin_ar1_vectorized/setup.jl

module AR1NegBinVectorized

using SpecialFunctions
using Turing
using LinearAlgebra
using Distributions
using DataFrames
using BayesianFootball
using Statistics         # For mean()
using LogExpFunctions    # For logaddexp()
using SparseArrays       # For efficient selection matrices

# Export the new model type so it can be used in runner scripts
export AR1NegBinVectorizedModel

#=
--------------------------------------------------------------------------------
1.  NUMERICALLY STABLE LIKELIHOOD FUNCTION
--------------------------------------------------------------------------------
As discussed, this custom function calculates the log-probability of the
Negative Binomial likelihood in a more numerically stable way than the
default, especially when working with log-rates.
=#
"""
A numerically stable and efficient log-PMF for the Negative Binomial 
distribution parameterized by the log-mean (log_μ) and log-dispersion (log_ϕ).
"""
function logpdf_negbin_lograte(k::Int, log_μ::Real, log_ϕ::Real)
    # This is the most numerically sensitive term, now correctly uses log_ϕ
    log_μ_plus_ϕ = logaddexp(log_μ, log_ϕ)

    ϕ = exp(log_ϕ)

    # Log-PMF formula
    return (
        loggamma(k + ϕ) - loggamma(k + 1) - loggamma(ϕ) +
        k * log_μ +
        ϕ * log_ϕ - # This is equivalent to ϕ * log(ϕ)
        (k + ϕ) * log_μ_plus_ϕ
    )
end

#=
--------------------------------------------------------------------------------
2.  MODEL DEFINITION STRUCT
--------------------------------------------------------------------------------
=#
struct AR1NegBinVectorizedModel <: AbstractModelDefinition end

#=
--------------------------------------------------------------------------------
3.  FRAMEWORK INTERFACE IMPLEMENTATION
--------------------------------------------------------------------------------
=#

"""
    get_required_features(::AR1NegBinVectorizedModel)

Declares the data features required by this model.
"""
function BayesianFootball.get_required_features(::AR1NegBinVectorizedModel)
    return (
        :home_team_ids,
        :away_team_ids,
        :n_teams,
        :global_round,
        :n_leagues,
        :league_ids
    )
end

"""
    build_turing_model(::AR1NegBinVectorizedModel, features, goals_home, goals_away)

Constructs the Turing model instance. This function now has the crucial role
of pre-calculating the selection matrices (S_t) for each time step, which
is the key to the vectorized model's performance.
"""
function BayesianFootball.build_turing_model(
    ::AR1NegBinVectorizedModel,
    features::NamedTuple,
    goals_home::Vector{Int},
    goals_away::Vector{Int}
)
    # Create a temporary DataFrame for easy grouping
    temp_df = DataFrame(
        global_round = features.global_round,
        home_id = features.home_team_ids,
        away_id = features.away_team_ids,
        league_id = features.league_ids,
        gh = goals_home,
        ga = goals_away
    )
    
    grouped = groupby(temp_df, :global_round)
    n_rounds = length(grouped)
    n_teams = features.n_teams
    n_leagues = features.n_leagues

    # Define the size and structure of the state vector
    # [α_1...α_N, β_1...β_N, ha_1...ha_L, log_ϕ]
    state_size = 2 * n_teams + n_leagues + 1
    
    # Pre-allocate containers for data grouped by round
    goals_by_round = Vector{Vector{Int}}(undef, n_rounds)
    selection_matrices = Vector{SparseMatrixCSC{Int, Int}}(undef, n_rounds)

    for (t, round_df) in enumerate(grouped)
        n_matches_in_round = nrow(round_df)
        
        # 1. Store goals for this round, alternating home and away
        goals_t = Vector{Int}(undef, 2 * n_matches_in_round)
        for (i, row) in enumerate(eachrow(round_df))
            goals_t[2i-1] = row.gh
            goals_t[2i] = row.ga
        end
        goals_by_round[t] = goals_t

        # 2. Construct the selection matrix S_t for this round
        # Using a sparse matrix is highly efficient for memory
        St = spzeros(Int, 2 * n_matches_in_round, state_size)
        for (k, match) in enumerate(eachrow(round_df))
            home_id = match.home_id
            away_id = match.away_id
            league_id = match.league_id

            # Indices in the state vector
            idx_α_home = home_id
            idx_α_away = away_id
            idx_β_home = n_teams + home_id
            idx_β_away = n_teams + away_id
            idx_home_adv = 2 * n_teams + league_id
            
            # Row for log_λ (home team's expected goals)
            St[2k-1, idx_α_home] = 1
            St[2k-1, idx_β_away] = 1
            St[2k-1, idx_home_adv] = 1

            # Row for log_μ (away team's expected goals)
            St[2k, idx_α_away] = 1
            St[2k, idx_β_home] = 1
        end
        selection_matrices[t] = St
    end
    
    # Instantiate and return the Turing model with the pre-computed structures
    return ar1_neg_bin_vectorized_model(
        goals_by_round,
        selection_matrices,
        n_teams,
        n_leagues,
        n_rounds
    )
end


#=
--------------------------------------------------------------------------------
4.  TURING @MODEL DEFINITION
--------------------------------------------------------------------------------
This is the high-performance, vectorized state-space model. The main `for` loop
now contains only efficient matrix-vector operations.
=#

@model function ar1_neg_bin_vectorized_model(
    goals_by_round::Vector{<:Vector},
    selection_matrices::Vector{<:SparseMatrixCSC},
    n_teams::Int,
    n_leagues::Int,
    n_rounds::Int,
    ::Type{T}=Float64
) where {T}

    # --- Priors ---
    ρ_attack ~ Beta(10, 1.5)
    ρ_defense ~ Beta(10, 1.5)
    μ_log_σ_attack ~ Normal(-2.5, 0.5) 
    τ_log_σ_attack ~ Truncated(Normal(0, 0.2), 0, Inf)
    μ_log_σ_defense ~ Normal(-2.5, 0.5) 
    τ_log_σ_defense ~ Truncated(Normal(0, 0.2), 0, Inf)
    ρ_home_adv ~ Beta(10, 1.5)
    μ_log_σ_home_adv ~ Normal(-3.0, 0.5)
    τ_log_σ_home_adv ~ Truncated(Normal(0, 0.2), 0, Inf)
    ρ_log_ϕ ~ Beta(10, 1.5)
    σ_log_ϕ ~ Truncated(Normal(-1, 0.5), 0, Inf)

    # --- Construct State-Space Matrices Φ and H ---
    z_log_σ_attack ~ MvNormal(zeros(T, n_teams), I)
    σ_attack = exp.(μ_log_σ_attack .+ z_log_σ_attack .* τ_log_σ_attack)
    z_log_σ_defense ~ MvNormal(zeros(T, n_teams), I)
    σ_defense = exp.(μ_log_σ_defense .+ z_log_σ_defense .* τ_log_σ_defense)
    z_log_σ_home_adv ~ MvNormal(zeros(T, n_leagues), I)
    σ_home_adv = exp.(μ_log_σ_home_adv .+ z_log_σ_home_adv .* τ_log_σ_home_adv)
    phi_diag = vcat(fill(ρ_attack, n_teams), fill(ρ_defense, n_teams), fill(ρ_home_adv, n_leagues), ρ_log_ϕ)
    Φ_mat = Diagonal(T.(phi_diag))
    H_diag = vcat(σ_attack.^2, σ_defense.^2, σ_home_adv.^2, σ_log_ϕ^2)
    H_mat = Diagonal(T.(H_diag))

    # --- Latent State ---
    z = Vector{Any}(undef, n_rounds)

    # Initial State (t=1)
    α_t1 ~ MvNormal(zeros(T, n_teams), 0.5 * I)
    β_t1 ~ MvNormal(zeros(T, n_teams), 0.5 * I)
    ha_t1 ~ MvNormal(fill(T(log(1.3)), n_leagues), 0.1 * I)
    log_ϕ_t1 ~ Normal(T(log(10)), 0.5)
    z[1] = vcat(α_t1, β_t1, ha_t1, log_ϕ_t1)
    
    # --- Main Time-Series Loop ---
    for t in 2:n_rounds
        z[t] ~ MvNormal(Φ_mat * z[t-1], H_mat)
    end

    # --- Likelihood Loop ---
    for t in 1:n_rounds
        z_t = z[t]
        z_constrained = copy(z_t)
        z_constrained[1:n_teams] .-= mean(z_t[1:n_teams])
        z_constrained[n_teams+1:2*n_teams] .-= mean(z_t[n_teams+1:2*n_teams])

        St = selection_matrices[t]
        if !isempty(St)
            log_rates = St * z_constrained

            log_ϕ_t = z_constrained[end] # Get log_ϕ directly
            goals_t = goals_by_round[t]
            log_prob = sum(logpdf_negbin_lograte.(goals_t, log_rates, log_ϕ_t)) # Pass log_ϕ_t
            Turing.@addlogprob! log_prob
        end
    end
end
end # end module AR1NegBinVectorized


module TestModel

using Turing
using LinearAlgebra
using Distributions
using DataFrames
using BayesianFootball
using Statistics
using SparseArrays

export SimplePoissonModel

struct SimplePoissonModel <: AbstractModelDefinition end

function BayesianFootball.get_required_features(::SimplePoissonModel)
    return (
        :home_team_ids, :away_team_ids, :n_teams,
        :global_round, :n_leagues, :league_ids
    )
end

function BayesianFootball.build_turing_model(
    ::SimplePoissonModel,
    features::NamedTuple,
    goals_home::Vector{Int},
    goals_away::Vector{Int}
)
    temp_df = DataFrame(
        global_round = features.global_round, home_id = features.home_team_ids,
        away_id = features.away_team_ids, league_id = features.league_ids,
        gh = goals_home, ga = goals_away
    )
    
    grouped = groupby(temp_df, :global_round)
    n_rounds = length(grouped)
    n_teams = features.n_teams
    n_leagues = features.n_leagues

    state_size = 2 * n_teams + n_leagues + 1
    
    goals_by_round = Vector{Vector{Int}}(undef, n_rounds)
    # 💡 CRITICAL FIX: Ensure the matrix type is Float64
    selection_matrices = Vector{SparseMatrixCSC{Float64, Int}}(undef, n_rounds)

    for (t, round_df) in enumerate(grouped)
        n_matches_in_round = nrow(round_df)
        
        goals_t = Vector{Int}(undef, 2 * n_matches_in_round)
        for (i, row) in enumerate(eachrow(round_df)); goals_t[2i-1]=row.gh; goals_t[2i]=row.ga; end
        goals_by_round[t] = goals_t

        # 💡 CRITICAL FIX: Create the sparse matrix with Float64 elements
        St = spzeros(Float64, 2 * n_matches_in_round, state_size)
        for (k, match) in enumerate(eachrow(round_df))
            home_id, away_id, league_id = match.home_id, match.away_id, match.league_id
            
            idx_α_home = home_id; idx_α_away = away_id
            idx_β_home = n_teams + home_id; idx_β_away = n_teams + away_id
            idx_home_adv = 2 * n_teams + league_id
            
            # 💡 CRITICAL FIX: Assign Float64 values (1.0)
            St[2k-1, idx_α_home] = 1.0; St[2k-1, idx_β_away] = 1.0; St[2k-1, idx_home_adv] = 1.0
            St[2k, idx_α_away] = 1.0; St[2k, idx_β_home] = 1.0
        end
        selection_matrices[t] = St
    end
    
    return minimal_test_turing_model(
        goals_by_round, selection_matrices, n_teams, n_leagues, n_rounds
    )
end

@model function minimal_test_turing_model(
    goals_by_round::Vector{<:Vector},
    selection_matrices::Vector{<:SparseMatrixCSC},
    n_teams::Int, n_leagues::Int, n_rounds::Int, ::Type{T}=Float64
) where {T}
    α ~ MvNormal(zeros(T, n_teams), 1.0 * I)
    β ~ MvNormal(zeros(T, n_teams), 1.0 * I)
    ha ~ MvNormal(zeros(T, n_leagues), 1.0 * I)
    dummy_param = 0.0
    state_vector = vcat(α, β, ha, dummy_param)

    for t in 1:n_rounds
        St = selection_matrices[t]
        goals_t = goals_by_round[t]
        if !isempty(St)
            log_rates = St * state_vector
            for i in 1:length(goals_t)
                goals_t[i] ~ Poisson(exp(log_rates[i]))
            end
        end
    end
end

end # end module



module AR1NegBinVectorized

using SpecialFunctions
using Turing
using LinearAlgebra
using Distributions
using DataFrames
using BayesianFootball
using Statistics
using LogExpFunctions

export AR1NegBinVectorizedModel

# (This helper function is unchanged)
function logpdf_negbin_lograte(k::Int, log_μ::Real, log_ϕ::Real)
    log_μ_plus_ϕ = logaddexp(log_μ, log_ϕ)
    ϕ = exp(log_ϕ)
    return (
        loggamma(k + ϕ) - loggamma(k + 1) - loggamma(ϕ) +
        k * log_μ +
        ϕ * log_ϕ -
        (k + ϕ) * log_μ_plus_ϕ
    )
end

struct AR1NegBinVectorizedModel <: AbstractModelDefinition end

# 💡 We no longer need the sparse matrix features, matching the working model
function BayesianFootball.get_required_features(::AR1NegBinVectorizedModel)
    return (
        :home_team_ids, :away_team_ids, :n_teams,
        :global_round, :n_leagues, :league_ids
    )
end

# 💡 This function now prepares nested vectors, not sparse matrices
function BayesianFootball.build_turing_model(
    ::AR1NegBinVectorizedModel,
    features::NamedTuple,
    goals_home::Vector{Int},
    goals_away::Vector{Int}
)
    temp_df = DataFrame(
        global_round = features.global_round,
        home_id = features.home_team_ids,
        away_id = features.away_team_ids,
        league_id = features.league_ids,
        gh = goals_home,
        ga = goals_away
    )
    
    grouped = groupby(temp_df, :global_round)
    n_rounds = length(grouped)

    home_ids_by_round = Vector{Vector{Int}}(undef, n_rounds)
    away_ids_by_round = Vector{Vector{Int}}(undef, n_rounds)
    league_ids_by_round = Vector{Vector{Int}}(undef, n_rounds)
    home_goals_by_round = Vector{Vector{Int}}(undef, n_rounds)
    away_goals_by_round = Vector{Vector{Int}}(undef, n_rounds)

    for (i, round_df) in enumerate(grouped)
        home_ids_by_round[i] = round_df.home_id
        away_ids_by_round[i] = round_df.away_id
        league_ids_by_round[i] = round_df.league_id
        home_goals_by_round[i] = round_df.gh
        away_goals_by_round[i] = round_df.ga
    end
    
    return ar1_neg_bin_vectorized_model(
        home_ids_by_round, away_ids_by_round,
        league_ids_by_round,
        home_goals_by_round, away_goals_by_round,
        features.n_teams, features.n_leagues, n_rounds
    )
end


@model function ar1_neg_bin_vectorized_model(
    home_team_ids::Vector{<:Vector}, away_team_ids::Vector{<:Vector},
    league_ids::Vector{<:Vector},
    home_goals::Vector{<:Vector}, away_goals::Vector{<:Vector},
    n_teams::Int, n_leagues::Int, n_rounds::Int, ::Type{T}=Float64
) where {T}

    # Priors (Hierarchical structure is unchanged)
    ρ_attack ~ Beta(10, 1.5); ρ_defense ~ Beta(10, 1.5)
    μ_log_σ_attack ~ Normal(-2.5, 0.5); τ_log_σ_attack ~ Truncated(Normal(0, 0.2), 0, Inf)
    μ_log_σ_defense ~ Normal(-2.5, 0.5); τ_log_σ_defense ~ Truncated(Normal(0, 0.2), 0, Inf)
    ρ_home_adv ~ Beta(10, 1.5);
    μ_log_σ_home_adv ~ Normal(-3.0, 0.5); τ_log_σ_home_adv ~ Truncated(Normal(0, 0.2), 0, Inf)
    ρ_log_ϕ ~ Beta(10, 1.5); σ_log_ϕ ~ Truncated(Normal(-1, 0.5), 0, Inf)

    # State-Space Matrices (Φ and H are still used for the AR(1) evolution)
    z_log_σ_attack ~ MvNormal(zeros(T, n_teams), I); σ_attack = exp.(μ_log_σ_attack .+ z_log_σ_attack .* τ_log_σ_attack)
    z_log_σ_defense ~ MvNormal(zeros(T, n_teams), I); σ_defense = exp.(μ_log_σ_defense .+ z_log_σ_defense .* τ_log_σ_defense)
    z_log_σ_home_adv ~ MvNormal(zeros(T, n_leagues), I); σ_home_adv = exp.(μ_log_σ_home_adv .+ z_log_σ_home_adv .* τ_log_σ_home_adv)
    phi_diag = vcat(fill(ρ_attack, n_teams), fill(ρ_defense, n_teams), fill(ρ_home_adv, n_leagues), ρ_log_ϕ)
    Φ_mat = Diagonal(T.(phi_diag))
    H_diag = vcat(σ_attack.^2, σ_defense.^2, σ_home_adv.^2, σ_log_ϕ^2)
    H_mat = Diagonal(T.(H_diag))

    # Latent State (AR(1) evolution logic is unchanged)
    z = Vector{Any}(undef, n_rounds)
    α_t1 ~ MvNormal(zeros(T, n_teams), 0.5 * I); β_t1 ~ MvNormal(zeros(T, n_teams), 0.5 * I)
    ha_t1 ~ MvNormal(fill(T(log(1.3)), n_leagues), 0.1 * I); log_ϕ_t1 ~ Normal(T(log(10)), 0.5)
    z[1] = vcat(α_t1, β_t1, ha_t1, log_ϕ_t1)
    
    for t in 2:n_rounds; z[t] ~ MvNormal(Φ_mat * z[t-1], H_mat); end

    # Likelihood Loop (Now using direct indexing)
    for t in 1:n_rounds
        z_t = z[t]
        
        # Deconstruct the state vector for this time step
        α_t = @view z_t[1:n_teams]
        β_t = @view z_t[n_teams+1:2*n_teams]
        ha_t = @view z_t[2*n_teams+1:2*n_teams+n_leagues]
        log_ϕ_t = z_t[end]

        # Apply sum-to-zero constraint
        α_t_constrained = α_t .- mean(α_t)
        β_t_constrained = β_t .- mean(β_t)
        
        home_ids = home_team_ids[t]
        away_ids = away_team_ids[t]
        leagues = league_ids[t]
        
        if !isempty(home_ids)
            # 💡 VECTORIZED LIKELIHOOD CALCULATION
            log_λs = α_t_constrained[home_ids] .+ β_t_constrained[away_ids] .+ ha_t[leagues]
            log_μs = α_t_constrained[away_ids] .+ β_t_constrained[home_ids]

            # Calculate both log-probabilities
            log_prob_home = sum(logpdf_negbin_lograte.(home_goals[t], log_λs, log_ϕ_t))
            log_prob_away = sum(logpdf_negbin_lograte.(away_goals[t], log_μs, log_ϕ_t))

            # Add them together in a single call
            Turing.@addlogprob! log_prob_home + log_prob_away




        end
    end
end

end # end module

#src/models/pregame/models-src/ar1-poisson.jl
using DataFrames
using Turing
using LinearAlgebra
using Statistics

export AR1Poisson, build_turing_model, extract_parameters


struct AR1Poisson <: AbstractDynamicPoissonModel end 



# ==============================================================================
# turing model
# ==============================================================================


@model function ar1_poisson_model_train(n_teams, n_rounds, 
                                        flat_home_ids, flat_away_ids, 
                                        flat_home_goals, flat_away_goals, 
                                        time_indices)



end 




function build_turing_model(model::AR1Poisson, feature_set::FeatureSet)
  println("this is a place holder for the AR1Model build_turing_model")

end 


function extract_parameters(
  model::AR1Poisson,
  df_to_predict::AbstractDataFrame,
  vocabulary::Vocabulary,
  chains::Chains)

  println("this is a place holder for the AR1Model extract_parameters level 1 ")

end
    

"""
OVERLOADED METHOD: Wrapper
Iterates through results and prediction dataframes, calling the inner extraction logic for each fold.
"""
function extract_parameters(
    model::AR1Poisson,
    dfs_to_predict::AbstractVector, 
    vocabulary::Vocabulary,
    results_vector::AbstractVector
)

  println("this is a place holder for the AR1Model extract_parameters level 2 - the wrapper. ")

end



# # src/models/pregame/models-src/ar1-poisson.jl
#
# using DataFrames, Turing, LinearAlgebra, Statistics
#
# export AR1Poisson, build_turing_model
#
# struct AR1Poisson <: AbstractDynamicPoissonModel end 
#
# @model function ar1_poisson_model_train(n_teams, n_rounds, 
#                                         flat_home_ids, flat_away_ids, 
#                                         flat_home_goals, flat_away_goals, 
#                                         time_indices)
#
#     # --- 1. Hyperparameters ---
#     ρ_att ~ Beta(10, 2) 
#     ρ_def ~ Beta(10, 2)
#     σ_att ~ Truncated(Normal(0, 0.05), 0, Inf)
#     σ_def ~ Truncated(Normal(0, 0.05), 0, Inf)
#     home_adv ~ Normal(log(1.3), 0.2)
#
#     # --- 2. Random Walk Innovation ---
#     # T=1
#     z_att_init ~ filldist(Normal(0, 1), n_teams)
#     z_def_init ~ filldist(Normal(0, 1), n_teams)
#     # T=2..N
#     z_att_steps ~ filldist(Normal(0, 1), n_teams, n_rounds - 1)
#     z_def_steps ~ filldist(Normal(0, 1), n_teams, n_rounds - 1)
#
#     # --- 3. The Loop (Optimized for Type Stability) ---
#
#     # TRICK: We need to tell Julia that 'att_seq' will contain the same type of numbers
#     # as 'z_att_init' (which might be Float64 or ReverseDiff.TrackedReal).
#     # We use `eltype` to detect this type dynamically.
#
#     T = eltype(z_att_init) 
#
#     # Initialize vector of vectors with explicit type T
#     # (Using Vector of Vectors is friendly to ReverseDiff)
#     att_seq = Vector{Vector{T}}(undef, n_rounds)
#     def_seq = Vector{Vector{T}}(undef, n_rounds)
#
#     # Initialize t=1
#     att_seq[1] = z_att_init .* σ_att
#     def_seq[1] = z_def_init .* σ_def
#
#     # Iterate t=2..Round
#     for t in 2:n_rounds
#         # Standard AR(1) update
#         att_seq[t] = (att_seq[t-1] .* ρ_att) .+ (z_att_steps[:, t-1] .* σ_att)
#         def_seq[t] = (def_seq[t-1] .* ρ_def) .+ (z_def_steps[:, t-1] .* σ_def)
#     end
#
#     # Reduce to Matrix (Efficient Stack)
#     att_raw = reduce(hcat, att_seq)
#     def_raw = reduce(hcat, def_seq)
#
#     # Center (Identifiability)
#     att = att_raw .- mean(att_raw, dims=1)
#     def = def_raw .- mean(def_raw, dims=1)
#
#     # Track for extraction
#     att_hist := att
#     def_hist := def
#     rho_att_track := ρ_att
#
#     # --- 4. Likelihood (V1 Cartesian Approach) ---
#     att_h = view(att, CartesianIndex.(flat_home_ids, time_indices))
#     def_a = view(def, CartesianIndex.(flat_away_ids, time_indices))
#     att_a = view(att, CartesianIndex.(flat_away_ids, time_indices))
#     def_h = view(def, CartesianIndex.(flat_home_ids, time_indices))
#
#     log_λs = home_adv .+ att_h .+ def_a
#     log_μs = att_a .+ def_h
#
#     flat_home_goals ~ arraydist(LogPoisson.(log_λs))
#     flat_away_goals ~ arraydist(LogPoisson.(log_μs))
# end
#
# # ... (Include your build_turing_model function here) ...

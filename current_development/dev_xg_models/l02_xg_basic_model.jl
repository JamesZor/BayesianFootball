# current_development/dev_xg_models/l02_xg_basic_model.jl



#=
The vanilla baseline model:
  log[λ] = μ + αᵢ + βⱼ + γ
  Yᵢ ~ NegativeBinomial(λ, r)
=#

# export AblationStudy_NB_baseLine


# we need an Abstract model

abstract type AbstractXGNegativeBinomial <: BayesianFootball.AbstractPregameModel end 

using Distributions
# ==============================================================================
# 1. STRUCT DEFINITION (The Priors)
# ==============================================================================

Base.@kwdef struct XG_baseLine_test <: AbstractXGNegativeBinomial
    # --- Global Baseline (Intercept) ---
    # Represents the average log-goal rate for an away team.
    μ::Distribution = Normal(0.2, 0.2)

    # Standard priors for team strength
    γ::Distribution   = Normal(0.12, 0.5)
    
    # Dispersion parameter (Negative Binomial)
    log_r::Distribution = Normal(2.5, 0.5)

    # xG to NB parameter 
    κ::Distribution = Gamma(3,1)
    ν_xg::Distribution =  truncated(Normal(10.0, 5.0), lower=1.0) 

    σ₀::Distribution = Gamma(2, 0.15)   # Mean = 0.30 (Initial spread of teams)
    σₛ::Distribution = Gamma(2, 0.04)   # Mean = 0.08 (Macro season jump)
    σₖ::Distribution = Gamma(2, 0.015)   # Mean = 0.03 (Micro monthly jump)

    z₀::Distribution = Normal(0,1)
    zₛ::Distribution = Normal(0,1)
    zₖ::Distribution = Normal(0,1)
end


function BayesianFootball.Features.required_features(model::XG_baseLine_test)
    return [:team_ids, :goals, :xg] 
end


using Turing

@model function model_train_11(
          n_teams, n_rounds, n_history, n_target,
          home_ids_flat, away_ids_flat, 
          home_goals_flat, away_goals_flat,
          clean_home_xg, clean_away_xg, # No missing values allowed here!
          idx_xg, idx_no_xg,            # The splitting vectors
          time_indices, 
          model::XG_baseLine_test,
          ::Type{T} = Float64 ) where {T} 

    # --- A. HYPERPARAMETERS ---
    μ ~ model.μ 
    γ ~ model.γ 
    log_r ~ model.log_r 
    r = exp(log_r)
    
    # Global scaling factor between xG and Goals (Optional, but from your draft)
    κ ~ model.κ 

    # xG Gamma Shape parameter (controls how tightly xG clusters around λ)
    ν_xg ~ model.ν_xg 

    # --- B. LATENT STATES (GRW) ---
    α ~ to_submodel(BayesianFootball.Models.PreGame.grw_two_step_component(n_teams, n_rounds, n_history, n_target, model.z₀, model.zₛ, model.zₖ, model.σ₀, model.σₛ, model.σₖ))
    β ~ to_submodel(BayesianFootball.Models.PreGame.grw_two_step_component(n_teams, n_rounds, n_history, n_target, model.z₀, model.zₛ, model.zₖ, model.σ₀, model.σₛ, model.σₖ))

    # --- C. LIKELIHOOD PIPELINE ---
    αₕ = view(α, CartesianIndex.(home_ids_flat, time_indices))
    αₐ = view(α, CartesianIndex.(away_ids_flat, time_indices))
    βₐ = view(β, CartesianIndex.(away_ids_flat, time_indices))
    βₕ = view(β, CartesianIndex.(home_ids_flat, time_indices))

    # Base Expectancy
    λₕ = exp.(μ .+ αₕ .+ βₐ .+ γ) .+ 1e-6
    λₐ = exp.(μ .+ αₐ .+ βₕ) .+ 1e-6

    # ---------------------------------------------------------
    # GROUP 1: Matches WITH xG Data
    # ---------------------------------------------------------
    if !isempty(idx_xg)
        # Slice the λ arrays for xG matches
        λₕ_xg = λₕ[idx_xg]
        λₐ_xg = λₐ[idx_xg]
        
        # 1. xG Likelihood (Mean = λ, Shape = ν)
        clean_home_xg ~ arraydist(Gamma.(ν_xg, λₕ_xg ./ ν_xg))
        clean_away_xg ~ arraydist(Gamma.(ν_xg, λₐ_xg ./ ν_xg))

        # 2. Goals Likelihood (Using κ multiplier)
        home_goals_flat[idx_xg] ~ arraydist(BayesianFootball.MyDistributions.RobustNegativeBinomial.(r, κ .* λₕ_xg))
        away_goals_flat[idx_xg] ~ arraydist(BayesianFootball.MyDistributions.RobustNegativeBinomial.(r, κ .* λₐ_xg))
    end

    # ---------------------------------------------------------
    # GROUP 2: Matches WITHOUT xG Data (History)
    # ---------------------------------------------------------
    if !isempty(idx_no_xg)
        λₕ_no = λₕ[idx_no_xg]
        λₐ_no = λₐ[idx_no_xg]

        # Goals Likelihood
        # NOTE: We MUST use κ here too! Otherwise the definition of λ 
        # changes halfway through the timeline, causing massive GRW drift!
        home_goals_flat[idx_no_xg] ~ arraydist(BayesianFootball.MyDistributions.RobustNegativeBinomial.(r, κ .* λₕ_no))
        away_goals_flat[idx_no_xg] ~ arraydist(BayesianFootball.MyDistributions.RobustNegativeBinomial.(r, κ .* λₐ_no))
    end
end

function BayesianFootball.Models.PreGame.build_turing_model(model::XG_baseLine_test, feature_set::Features.FeatureSet)
    data = feature_set.data
    
    raw_h_xg = data[:flat_home_xg]
    raw_a_xg = data[:flat_away_xg]
    
    # Pre-processing: Group Matches by xG Availability
    idx_xg = Int[]
    idx_no_xg = Int[]
    clean_h_xg = Float64[]
    clean_a_xg = Float64[]

    for i in eachindex(raw_h_xg)
        if ismissing(raw_h_xg[i]) || ismissing(raw_a_xg[i])
            push!(idx_no_xg, i)
        else
            push!(idx_xg, i)
            push!(clean_h_xg, Float64(raw_h_xg[i]))
            push!(clean_a_xg, Float64(raw_a_xg[i]))
        end
    end

    return model_train_11(
        data[:n_teams]::Int,
        data[:n_rounds]::Int,
        data[:n_history_steps]::Int,
        data[:n_target_steps]::Int,
        data[:flat_home_ids],    
        data[:flat_away_ids],     
        data[:flat_home_goals],
        data[:flat_away_goals],
        clean_h_xg,  # Sliced arrays to prevent Missing crashes
        clean_a_xg,
        idx_xg,      # Indices for the Turing `if` blocks
        idx_no_xg,
        data[:time_indices],
        model
    )
end




function create_experiment_tasks(ds::Data.DataStore, label::String, save_dir::String, target_seasons::Vector{<:String} )

    # 1. Define the shared parts (CV and Training)
        cv_config = Data.GroupedCVConfig(
        tournament_groups = [Data.tournament_ids(ds.segment)],
        target_seasons = target_seasons,
        history_seasons = 1,
        dynamics_col = :match_month,
        warmup_period = 0,
        stop_early = true
    )

    sampler_conf = Samplers.NUTSConfig(
    500, # Number of samples for each chain
    4,   # Number of chains
    150, # Number of warm up steps 
    0.65,# Accept rate  [0,1]
    10,  # Max tree depth
    Samplers.UniformInit(-1, 1), # Interval for starting a chain 
    false,   # show_progress (We use the Global Logger instead)
    # false, # Display progress bar setting
  )
    train_cfg = BayesianFootball.Training.Independent(
    parallel=true,
    max_concurrent_splits=4
  )
    training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)

    # 2. Build the list of Configs
    configs = [
        # Experiments.ExperimentConfig(
        #     name = "$(label)_01_baseline",
        #     model = Models.PreGame.AblationStudy_NB_baseLine(),
        #     splitter = cv_config,
        #     training_config = training_config,
        #     save_dir = save_dir
        # ),

        Experiments.ExperimentConfig(
            name = "$(label)_xg_basic_runner",
          model = XG_baseLine_test(),
            splitter = cv_config,
            training_config = training_config,
            save_dir = save_dir
    ),
    ]

    # 3. THE "SMART" BIT: 
    # Wrap every config with the DS into an ExperimentTask
    # We use Ref(ds) so it doesn't try to "iterate" the DataStore
    return ExperimentTask.(Ref(ds), configs)
end



# function train(model::AbstractFootballModel, config::TrainingConfig, feature_set::FeatureSet)
#     # This logic remains the same: Build -> Run Sampler
#     turing_model = build_turing_model(model, feature_set) 
#     return run_sampler(turing_model, config.sampler)
# end
#

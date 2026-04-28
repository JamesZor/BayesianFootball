include("./l02_xg_basic_model.jl")
using Distributions
using Turing


# ==============================================================================
# 1. STRUCT DEFINITION (The Priors)
# ==============================================================================
Base.@kwdef struct XG_kappa_team_test <: AbstractXGNegativeBinomial
    # --- Global Baselines ---
    μ::Distribution = Normal(0.17, 0.1)      
    γ::Distribution = Normal(0.27, 0.1)      
    log_r::Distribution = Normal(3.1, 0.4)   
    
    # --- The xG Conversion Parameters (Kappa) ---
    κ_global::Distribution = truncated(Normal(0.84, 0.1), lower=0.1)
    
    # Hierarchical components for Team Kappa Deltas
    # Gamma(2, 0.025) gives a mean standard deviation of 0.05, letting the model learn the true spread
    δ_κ_σ::Distribution = Gamma(2, 0.025) 
    δ_κ_z::Distribution = Normal(0, 1)    

    # --- xG Gamma Shape ---
    ν_xg::Distribution = truncated(Normal(3.0, 0.5), lower=0.5) 

    # --- Latent State Variances (ATTACK & DEFENSE) ---
    α_σ₀::Distribution = Gamma(2, 0.06)    
    α_σₛ::Distribution = Gamma(2, 0.03)    
    α_σₖ::Distribution = Gamma(2, 0.015)   

    β_σ₀::Distribution = Gamma(2, 0.10)    
    β_σₛ::Distribution = Gamma(2, 0.055)   
    β_σₖ::Distribution = Gamma(2, 0.012)   

    z₀::Distribution = Normal(0,1)
    zₛ::Distribution = Normal(0,1)
    zₖ::Distribution = Normal(0,1)
end


function BayesianFootball.Features.required_features(model::XG_kappa_team_test)
    return [:team_ids, :goals, :xg] 
end

@model function model_train_12(
          n_teams, n_rounds, n_history, n_target,
          home_ids_flat, away_ids_flat, 
          home_goals_flat, away_goals_flat,
          clean_home_xg, clean_away_xg, 
          idx_xg, idx_no_xg,            
          time_indices, 
          model::XG_kappa_team_test,
          ::Type{T} = Float64 ) where {T} 

    # --- A. HYPERPARAMETERS ---
    μ ~ model.μ 
    γ ~ model.γ 
    log_r ~ model.log_r 
    r = exp(log_r)
    ν_xg ~ model.ν_xg 

    # 1. Sample Global Kappa
    κ_global ~ model.κ_global 

    # 2. Hierarchical Team Deltas (Using your new helper!)
    δ_κ ~ to_submodel(BayesianFootball.Models.PreGame.hierarchical_zero_centered_component(
        n_teams, 
        model.δ_κ_σ, 
        model.δ_κ_z
    ))
    
    # 3. Final Team Kappa Array
    κ_team = κ_global .+ δ_κ

    # 4. Map the kappas to the flat match indices
    κ_h_flat = κ_team[home_ids_flat]
    κ_a_flat = κ_team[away_ids_flat]

    # --- B. LATENT STATES (GRW) ---
    # Using your separated alpha/beta variances from the previous iteration
    α ~ to_submodel(BayesianFootball.Models.PreGame.grw_two_step_component(n_teams, n_rounds, n_history, n_target, model.z₀, model.zₛ, model.zₖ, model.α_σ₀, model.α_σₛ, model.α_σₖ))
    β ~ to_submodel(BayesianFootball.Models.PreGame.grw_two_step_component(n_teams, n_rounds, n_history, n_target, model.z₀, model.zₛ, model.zₖ, model.β_σ₀, model.β_σₛ, model.β_σₖ))

    # --- C. LIKELIHOOD PIPELINE ---
    αₕ = view(α, CartesianIndex.(home_ids_flat, time_indices))
    αₐ = view(α, CartesianIndex.(away_ids_flat, time_indices))
    βₐ = view(β, CartesianIndex.(away_ids_flat, time_indices))
    βₕ = view(β, CartesianIndex.(home_ids_flat, time_indices))

    # Base Expectancy
    # λₕ = exp.(μ .+ αₕ .+ βₐ .+ γ)
    # λₐ = exp.(μ .+ αₐ .+ βₕ)

    λₕ = exp.(μ .+ αₕ .+ βₐ .+ γ) .+ 1e-6
    λₐ = exp.(μ .+ αₐ .+ βₕ) .+ 1e-6


    # ---------------------------------------------------------
    # GROUP 1: Matches WITH xG Data
    # ---------------------------------------------------------
    if !isempty(idx_xg)
        λₕ_xg = λₕ[idx_xg]
        λₐ_xg = λₐ[idx_xg]
        
        clean_home_xg ~ arraydist(Gamma.(ν_xg, λₕ_xg ./ ν_xg))
        clean_away_xg ~ arraydist(Gamma.(ν_xg, λₐ_xg ./ ν_xg))

        home_goals_flat[idx_xg] ~ arraydist(BayesianFootball.MyDistributions.RobustNegativeBinomial.(r, κ_h_flat[idx_xg] .* λₕ_xg))
        away_goals_flat[idx_xg] ~ arraydist(BayesianFootball.MyDistributions.RobustNegativeBinomial.(r, κ_a_flat[idx_xg] .* λₐ_xg))
    end

    # ---------------------------------------------------------
    # GROUP 2: Matches WITHOUT xG Data (History)
    # ---------------------------------------------------------
    if !isempty(idx_no_xg)
        λₕ_no = λₕ[idx_no_xg]
        λₐ_no = λₐ[idx_no_xg]

        home_goals_flat[idx_no_xg] ~ arraydist(BayesianFootball.MyDistributions.RobustNegativeBinomial.(r, κ_h_flat[idx_no_xg] .* λₕ_no))
        away_goals_flat[idx_no_xg] ~ arraydist(BayesianFootball.MyDistributions.RobustNegativeBinomial.(r, κ_a_flat[idx_no_xg] .* λₐ_no))
    end
end

function BayesianFootball.Models.PreGame.build_turing_model(model::XG_kappa_team_test, feature_set::Features.FeatureSet)
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

    return model_train_12(
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


function BayesianFootball.Models.PreGame.extract_parameters(
    model::XG_kappa_team_test, 
    df::AbstractDataFrame, 
    feature_set::Features.FeatureSet,
    chain::Chains
)
    n_teams = feature_set.data[:n_teams]
    n_rounds = feature_set.data[:n_rounds]
    n_history = feature_set.data[:n_history_steps]
    n_target = feature_set.data[:n_target_steps]
    team_map = feature_set.data[:team_map]
    
    # 1. Reconstruct Latent States
    α = BayesianFootball.Models.PreGame.reconstruct_multiscale_submodel(chain, "α", n_teams, n_history, n_target)
    β = BayesianFootball.Models.PreGame.reconstruct_multiscale_submodel(chain, "β", n_teams, n_history, n_target)
    
    # 2. Extract Globals
    μ_v = vec(Array(chain[:μ]))
    γ_v = vec(Array(chain[:γ]))
    log_r_v = vec(Array(chain[:log_r]))
    r_v = exp.(log_r_v)
    n_samples = length(μ_v) 

    # Extract global Kappa
    κ_global_v = vec(Array(chain[:κ_global]))
    
    # Extract hierarchical deltas using your new helper!
    # Returns matrix of size [Samples, Teams]
    δ_κ_matrix = BayesianFootball.Models.PreGame.reconstruct_hierarchical_centered(chain, "δ_κ")
    
    results = Dict{Int64, NamedTuple}()
    sizehint!(results, nrow(df))

    for row in eachrow(df)
        mid = Int(row.match_id)
        
        # --- RESOLVE TIME INDEX ---
        t_idx = n_rounds
        
        # --- SAFEGUARD: Unseen Teams ---
        h_idx = get(team_map, row.home_team, -1)
        a_idx = get(team_map, row.away_team, -1)

        α_h = h_idx > 0 ? α[h_idx, t_idx, :] : zeros(n_samples)
        α_a = a_idx > 0 ? α[a_idx, t_idx, :] : zeros(n_samples)
        β_h = h_idx > 0 ? β[h_idx, t_idx, :] : zeros(n_samples)
        β_a = a_idx > 0 ? β[a_idx, t_idx, :] : zeros(n_samples)

        # 3. Calculate True Underlying xG
        true_xg_h = exp.(μ_v .+ γ_v .+ α_h .+ β_a)
        true_xg_a = exp.(μ_v .+        α_a .+ β_h)

        # 4. Calculate Team-Specific Kappa for this match
        κ_h_v = h_idx > 0 ? κ_global_v .+ δ_κ_matrix[:, h_idx] : κ_global_v
        κ_a_v = a_idx > 0 ? κ_global_v .+ δ_κ_matrix[:, a_idx] : κ_global_v

        # 5. Calculate Final Goal Expectancy
        λ_goals_h = κ_h_v .* true_xg_h
        λ_goals_a = κ_a_v .* true_xg_a

        results[mid] = (;
            λ_h = λ_goals_h,       
            λ_a = λ_goals_a,       
            r = r_v,
            true_xg_h = true_xg_h, 
            true_xg_a = true_xg_a, 
            κ_h = κ_h_v,           
            κ_a = κ_a_v            
        )
    end

    return results
end




# ------ utils -------


function create_experiment_tasks2(es::DSExperimentSettings)
    return create_experiment_tasks2(es.ds, es.label, es.save_dir, es.target_season)
end


function create_experiment_tasks2(ds::Data.DataStore, label::String, save_dir::String, target_seasons::Vector{<:String} )

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
            name = "$(label)_xg_kappa_team",
            model = XG_kappa_team_test(),
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


include("./l02_xg_basic_model.jl")
using Distributions
using Turing


# --- KAPPA CONFIGS ---
abstract type AbstractKappaConfig end

Base.@kwdef struct GlobalKappa <: AbstractKappaConfig
    κ_global::Distribution = truncated(Normal(0.84, 0.1), lower=0.1)
end

Base.@kwdef struct HierarchicalTeamKappa <: AbstractKappaConfig
    κ_global::Distribution = truncated(Normal(0.84, 0.1), lower=0.1)
    δ_κ_σ::Distribution = Gamma(2, 0.025)
    δ_κ_z::Distribution = Normal(0, 1)
end

# --- DISPERSION (R) CONFIGS ---
abstract type AbstractDispersionConfig end

Base.@kwdef struct GlobalDispersion <: AbstractDispersionConfig
    log_r::Distribution = Normal(3.1, 0.4)
end

Base.@kwdef struct HomeAwayDispersion <: AbstractDispersionConfig
    log_r::Distribution = Normal(3.1, 0.4)        # Away / Base baseline
    δ_r_home::Distribution = Normal(0.0, 0.5)     # Home deviation
end

# --- MAIN MODEL STRUCT ---
Base.@kwdef struct XG_Master_Model{K<:AbstractKappaConfig, D<:AbstractDispersionConfig} <: AbstractXGNegativeBinomial
    μ::Distribution = Normal(0.17, 0.1)
    γ::Distribution = Normal(0.27, 0.1)
    ν_xg::Distribution = truncated(Normal(3.0, 0.5), lower=0.5) 
    
    # Inject Modules
    kappa_config::K
    disp_config::D

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



# ==========================================
# KAPPA SUBMODELS
# ==========================================
@model function build_kappa_base(n_teams, config::GlobalKappa)
    κ_global ~ config.κ_global
    return fill(κ_global, n_teams) 
end

@model function build_kappa_base(n_teams, config::HierarchicalTeamKappa)
    κ_global ~ config.κ_global
    δ_κ ~ to_submodel(BayesianFootball.Models.PreGame.hierarchical_zero_centered_component(n_teams, config.δ_κ_σ, config.δ_κ_z))
    return κ_global .+ δ_κ
end

"""
    get_match_kappa(κ::AbstractVector, team_ids, time_ids)

Fallback for 1D Kappa arrays (Global or Static Team). 
Ignores the time indices entirely.
"""
function get_match_kappa(κ::AbstractVector, team_ids, time_ids)
    return κ[team_ids]
end

"""
    get_match_kappa(κ::AbstractMatrix, team_ids, time_ids)

Handler for 2D Kappa arrays (Form-based Time drift). 
Uses Cartesian indexing to match teams to their specific time step.
"""
function get_match_kappa(κ::AbstractMatrix, team_ids, time_ids)
    return view(κ, CartesianIndex.(team_ids, time_ids))
end

# ==========================================
# DISPERSION SUBMODELS
# ==========================================
@model function build_dispersion(config::GlobalDispersion)
    log_r ~ config.log_r
    r = exp(log_r)
    return (; h = r, a = r) # Same exact R for both
end

@model function build_dispersion(config::HomeAwayDispersion)
    log_r ~ config.log_r
    δ_r_home ~ config.δ_r_home
    
    r_a = exp(log_r)
    r_h = exp(log_r + δ_r_home)
    return (; h = r_h, a = r_a) # Separated R
end


function BayesianFootball.Features.required_features(model::XG_Master_Model)
    return [:team_ids, :goals, :xg] 
end


@model function model_train_11(
          n_teams, n_rounds, n_history, n_target,
          home_ids_flat, away_ids_flat, home_goals_flat, away_goals_flat,
          clean_home_xg, clean_away_xg, idx_xg, idx_no_xg, # <-- RESTORED THESE!
          time_indices, 
          model::XG_Master_Model,
          ::Type{T} = Float64 ) where {T}

    # --- A. HYPERPARAMETERS ---
    μ ~ model.μ 
    γ ~ model.γ 
    ν_xg ~ model.ν_xg

    # Build Modules via Dispatch
    κ_base ~ to_submodel(build_kappa_base(n_teams, model.kappa_config))
    disp   ~ to_submodel(build_dispersion(model.disp_config))

    
    # --- B. LATENT STATES (GRW) ---
    α ~ to_submodel(BayesianFootball.Models.PreGame.grw_two_step_component(n_teams, n_rounds, n_history, n_target, model.z₀, model.zₛ, model.zₖ, model.α_σ₀, model.α_σₛ, model.α_σₖ))
    β ~ to_submodel(BayesianFootball.Models.PreGame.grw_two_step_component(n_teams, n_rounds, n_history, n_target, model.z₀, model.zₛ, model.zₖ, model.β_σ₀, model.β_σₛ, model.β_σₖ))

    αₕ = view(α, CartesianIndex.(home_ids_flat, time_indices))
    αₐ = view(α, CartesianIndex.(away_ids_flat, time_indices))
    βₕ = view(β, CartesianIndex.(home_ids_flat, time_indices))
    βₐ = view(β, CartesianIndex.(away_ids_flat, time_indices))
  

    # --- C. LIKELIHOOD PIPELINE ---
    λₕ = exp.(μ .+ γ .+ αₕ .+ βₐ) .+ 1e-6
    λₐ = exp.(μ .+      αₐ .+ βₕ) .+ 1e-6

    # Map parameters to matches safely
    κ_h_flat = get_match_kappa(κ_base, home_ids_flat, time_indices)
    κ_a_flat = get_match_kappa(κ_base, away_ids_flat, time_indices)

    # ---------------------------------------------------------
    # GROUP 1: Matches WITH xG Data
    # ---------------------------------------------------------
    if !isempty(idx_xg)
        λₕ_xg = λₕ[idx_xg]
        λₐ_xg = λₐ[idx_xg]
        
        clean_home_xg ~ arraydist(Gamma.(ν_xg, λₕ_xg ./ ν_xg))
        clean_away_xg ~ arraydist(Gamma.(ν_xg, λₐ_xg ./ ν_xg))

        home_goals_flat[idx_xg] ~ arraydist(BayesianFootball.MyDistributions.RobustNegativeBinomial.(disp.h, κ_h_flat[idx_xg] .* λₕ_xg))
        away_goals_flat[idx_xg] ~ arraydist(BayesianFootball.MyDistributions.RobustNegativeBinomial.(disp.a, κ_a_flat[idx_xg] .* λₐ_xg))
    end

    # ---------------------------------------------------------
    # GROUP 2: Matches WITHOUT xG Data (History)
    # ---------------------------------------------------------
    if !isempty(idx_no_xg)
        λₕ_no = λₕ[idx_no_xg]
        λₐ_no = λₐ[idx_no_xg]

        home_goals_flat[idx_no_xg] ~ arraydist(BayesianFootball.MyDistributions.RobustNegativeBinomial.(disp.h, κ_h_flat[idx_no_xg] .* λₕ_no))
        away_goals_flat[idx_no_xg] ~ arraydist(BayesianFootball.MyDistributions.RobustNegativeBinomial.(disp.a, κ_a_flat[idx_no_xg] .* λₐ_no))
    end
end

function BayesianFootball.Models.PreGame.build_turing_model(model::XG_Master_Model, feature_set::Features.FeatureSet)
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
        clean_h_xg,  # Pass the restored arguments
        clean_a_xg,  
        idx_xg,      
        idx_no_xg,   
        data[:time_indices],
        model
    )
end



# --- KAPPA EXTRACTORS ---

function extract_kappa(chain, config::GlobalKappa, n_teams)
    # Prefix: κ_base.
    k_glob = vec(Array(chain[Symbol("κ_base.κ_global")]))
    return repeat(k_glob, 1, n_teams) 
end

function extract_kappa(chain, config::HierarchicalTeamKappa, n_teams)
    k_glob = vec(Array(chain[Symbol("κ_base.κ_global")]))
    # Prefix: κ_base.δ_κ
    delta_k = reconstruct_hierarchical_centered(chain, "κ_base.δ_κ") 
    return k_glob .+ delta_k
end
# --- DISPERSION EXTRACTORS ---
function extract_dispersion(chain, config::GlobalDispersion)
    # Scoped under disp because of: disp ~ to_submodel(...)
    r_val = exp.(vec(Array(chain[Symbol("disp.log_r")])))
    return (; h = r_val, a = r_val)
end

function extract_dispersion(chain, config::HomeAwayDispersion)
    # Prefix: disp.
    log_r = vec(Array(chain[Symbol("disp.log_r")]))
    delta_r = vec(Array(chain[Symbol("disp.δ_r_home")]))
    
    r_a = exp.(log_r)
    r_h = exp.(log_r .+ delta_r)
    return (; h = r_h, a = r_a)
end



function reconstruct_hierarchical_centered(chain::Chains, prefix::String)
    # Extract σ vector: [Samples]
    σ_vec = vec(Array(chain[Symbol("$prefix.σ")])) 
    
    # Extract z matrix: [Samples, Teams]
    # 'group' finds all variables matching the prefix.z[i] pattern
    z_raw = Array(group(chain, Symbol("$prefix.z"))) 
    
    # Multiply z by σ (broadcasting σ across columns)
    raw_val = z_raw .* reshape(σ_vec, :, 1)
    
    # Subtract the mean of each row to enforce sum-to-zero
    centered = raw_val .- mean(raw_val, dims=2)
    
    return centered
end

function BayesianFootball.Models.PreGame.extract_parameters(
    model::XG_Master_Model, 
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

    μ_v = vec(Array(chain[:μ]))
    γ_v = vec(Array(chain[:γ]))     
    n_samples = length(μ_v)          

    # 2. Dispatch Extractors (These "just work" no matter the config!)
    κ_matrix = extract_kappa(chain, model.kappa_config, n_teams)
    disp_tuple = extract_dispersion(chain, model.disp_config)
    
    results = Dict{Int64, NamedTuple}()
    
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
        
        # Pull team-specific kappa (or fallback to global if unseen team)
        κ_h_v = h_idx > 0 ? κ_matrix[:, h_idx] : κ_matrix[:, 1]
        κ_a_v = a_idx > 0 ? κ_matrix[:, a_idx] : κ_matrix[:, 1]

        λ_goals_h = κ_h_v .* true_xg_h
        λ_goals_a = κ_a_v .* true_xg_a

      results[mid] = (;
            λ_h = λ_goals_h,       
            λ_a = λ_goals_a,       
            # ---> FIX: ADD THESE KEYS FOR LEGACY COMPATIBILITY <---
            r = disp_tuple.h,     # Fallback key
            r_h = disp_tuple.h,   # Expected by Home-specific logic
            r_a = disp_tuple.a,   # Expected by Away-specific logic
            # -----------------------------------------------------
            true_xg_h = true_xg_h, 
            true_xg_a = true_xg_a
        )
    end
    return results
end


# ---------------------------
function create_experiment_tasks_grid(ds::Data.DataStore, label::String, save_dir::String, target_seasons::Vector{<:String})

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
        1000, # Number of samples for each chain
        4,    # Number of chains
        200,  # Number of warm up steps 
        0.65, # Accept rate  [0,1]
        10,   # Max tree depth
        Samplers.UniformInit(-1, 1), 
        false
    )
    
    train_cfg = BayesianFootball.Training.Independent(
        parallel=true,
        max_concurrent_splits=4
    )
    training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)

    # 2. Define the Grid Spaces (Add new modules here in the future!)
    kappa_options = [
        GlobalKappa(),
        HierarchicalTeamKappa()
    ]

    disp_options = [
        GlobalDispersion(),
        HomeAwayDispersion()
    ]

    # 3. Build the list of Configs dynamically
    configs = Experiments.ExperimentConfig[]

    for k in kappa_options
        for d in disp_options
            # Automatically grab the clean struct names for labeling
            # e.g., "GlobalKappa" and "HomeAwayDispersion"
            k_name = string(nameof(typeof(k)))
            d_name = string(nameof(typeof(d)))
            
            # Create a highly descriptive experiment name
            experiment_name = "$(label)_$(k_name)_$(d_name)"
            
            # Instantiate the Master Model with this specific combination
            model_instance = XG_Master_Model(kappa_config=k, disp_config=d)

            push!(configs, Experiments.ExperimentConfig(
                name = experiment_name,
                model = model_instance,
                splitter = cv_config,
                training_config = training_config,
                save_dir = save_dir
            ))
        end
    end

    # 4. THE "SMART" BIT: 
    # Wrap every config with the DS into an ExperimentTask
    return ExperimentTask.(Ref(ds), configs)
end



function create_experiment_tasks_grid(es::DSExperimentSettings)
    return create_experiment_tasks_grid(es.ds, es.label, es.save_dir, es.target_season)
end


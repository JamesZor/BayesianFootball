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

function BayesianFootball.Models.PreGame.extract_parameters(
    model::XG_baseLine_test, 
    df::AbstractDataFrame, 
    feature_set::Features.FeatureSet,
    chain::Chains
)
    n_teams = feature_set.data[:n_teams]
    n_rounds = feature_set.data[:n_rounds]
    n_history = feature_set.data[:n_history_steps]
    n_target = feature_set.data[:n_target_steps]
    team_map = feature_set.data[:team_map]
    
    # 1. Reconstruct Latent States (The Random Walks)
    α = BayesianFootball.Models.PreGame.reconstruct_multiscale_submodel(chain, "α", n_teams, n_history, n_target)
    β = BayesianFootball.Models.PreGame.reconstruct_multiscale_submodel(chain, "β", n_teams, n_history, n_target)
    
    # 2. Extract Globals
    μ_v = vec(Array(chain[:μ]))
    γ_v = vec(Array(chain[:γ]))         # Global Home Advantage
    κ_v = vec(Array(chain[:κ]))         # Goal Conversion Bias
    log_r_v = vec(Array(chain[:log_r]))
    r_v = exp.(log_r_v)
    n_samples = length(μ_v) 
    
    results = Dict{Int64, NamedTuple}()
    sizehint!(results, nrow(df))

    for row in eachrow(df)
        mid = Int(row.match_id)
        
        # --- RESOLVE TIME INDEX ---
        # For predicting unseen future matches, project the most recent latent state
        t_idx = n_rounds
        
        # --- SAFEGUARD: Unseen Teams ---
        h_idx = get(team_map, row.home_team, -1)
        a_idx = get(team_map, row.away_team, -1)

        α_h = h_idx > 0 ? α[h_idx, t_idx, :] : zeros(n_samples)
        α_a = a_idx > 0 ? α[a_idx, t_idx, :] : zeros(n_samples)
        β_h = h_idx > 0 ? β[h_idx, t_idx, :] : zeros(n_samples)
        β_a = a_idx > 0 ? β[a_idx, t_idx, :] : zeros(n_samples)

        # 3. Calculate True Underlying xG (The pure Latent State)
        true_xg_h = exp.(μ_v .+ γ_v .+ α_h .+ β_a)
        true_xg_a = exp.(μ_v .+        α_a .+ β_h)

        # 4. Calculate Final Goal Expectancy (Applying the κ scaling factor)
        λ_goals_h = κ_v .* true_xg_h
        λ_goals_a = κ_v .* true_xg_a

        # Store BOTH distributions in the NamedTuple!
        results[mid] = (;
            λ_h = λ_goals_h,       # MUST be passed to your Poisson/NegBin odds compiler
            λ_a = λ_goals_a,       # MUST be passed to your Poisson/NegBin odds compiler
            r = r_v,
            true_xg_h = true_xg_h, # Great for secondary backtesting and diagnostic dashboards
            true_xg_a = true_xg_a, 
            κ = κ_v
        )
    end

    return results
end

# ==============================================================================
# 5. SCORE COMPUTATION WRAPPERS (Odds Compiler Hooks)
# ==============================================================================
using Distributions

# 1. Extract Params for the xG Model
function BayesianFootball.Predictions.extract_params(model::AbstractXGNegativeBinomial, row)
    if hasproperty(row, :r)
        return (
            λ_h = row.λ_h, # This is the κ-scaled λ_goals_h
            λ_a = row.λ_a, # This is the κ-scaled λ_goals_a
            r_h = row.r,   # Shared dispersion
            r_a = row.r 
        )
    else
        throw(ArgumentError("Row does not contain expected shape parameter :r"))
    end 
end

# 2. Get Column Symbols
function BayesianFootball.Predictions.get_latent_column_symbols(::AbstractXGNegativeBinomial, df::DataFrames.AbstractDataFrame)
    cols = [:match_id, :λ_h, :λ_a]
    if :r in propertynames(df)
        push!(cols, :r)
    end
    return cols
end

# 3. Compute Score Matrix
function BayesianFootball.Predictions.compute_score_matrix(
    model::AbstractXGNegativeBinomial, 
    params; 
    max_goals::Int=12
)
    λ_h, λ_a = params.λ_h, params.λ_a
    r_h, r_a = params.r_h, params.r_a
    n_samples = length(λ_h)

    outcomes_grid = [[h, a] for h in 0:max_goals-1, a in 0:max_goals-1]
    S = zeros(Float64, max_goals, max_goals, n_samples)

    @inbounds for k in 1:n_samples
        # Rely on the package's DoubleNegativeBinomial distribution
        dist = BayesianFootball.MyDistributions.DoubleNegativeBinomial(λ_h[k], λ_a[k], r_h[k], r_a[k])
        S_k = pdf.(Ref(dist), outcomes_grid)
        S[:, :, k] = S_k
    end
    
    # Wrap it in the package's ScoreMatrix type
    return BayesianFootball.Predictions.ScoreMatrix(S)
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

# ----
using DataFrames
using Statistics

function dev_compute_xg_residuals(
    ds::BayesianFootball.Data.DataStore, 
    exp::BayesianFootball.Experiments.ExperimentResults
)
    println("1. Extracting OOS Latent States...")
    latents_raw = BayesianFootball.Experiments.extract_oos_predictions(ds, exp)
    
    println("2. Joining with Actual xG Data from ds.statistics...")
    
    # SAFEGUARD: Ensure the statistics table has what we need
    if !hasproperty(ds, :statistics)
        error("DataStore does not contain a `statistics` DataFrame.")
    end
    
    avail_cols = propertynames(ds.statistics)
    if !(:expectedGoals_home in avail_cols) || !(:expectedGoals_away in avail_cols)
        println("\n❌ ERROR: Could not find xG columns in ds.statistics!")
        println("Available columns: ", avail_cols)
        error("Missing :expectedGoals_home or :expectedGoals_away")
    end

    # Join with the statistics table and rename them to our internal standard
    joined = innerjoin(
        latents_raw.df,
        select(ds.statistics, :match_id, :expectedGoals_home => :home_xg, :expectedGoals_away => :away_xg),
        on = :match_id
    )
    
    # Filter out matches where xG is missing
    dropmissing!(joined, [:home_xg, :away_xg])
    
    n_matches = nrow(joined)
    if n_matches == 0
        error("No non-missing xG data found in the out-of-sample predictions.")
    end
    
    println("3. Computing Residuals for $n_matches matches...")
    
    # Extract the Mean Expected xG from the posterior distributions
    exp_xg_h = mean.(joined.true_xg_h)
    exp_xg_a = mean.(joined.true_xg_a)
    
    # Calculate Continuous Residuals (Observed - Expected)
    res_h = Float64.(joined.home_xg) .- exp_xg_h
    res_a = Float64.(joined.away_xg) .- exp_xg_a
    res_all = vcat(res_h, res_a)
    
    # Print a quick diagnostic summary
    println("\n=== xG Continuous Residuals Summary ===")
    println("Home xG Bias (Mean Error):  ", round(mean(res_h), digits=4))
    println("Away xG Bias (Mean Error):  ", round(mean(res_a), digits=4))
    println("Total xG Bias:              ", round(mean(res_all), digits=4))
    println("Total xG MAE (Abs Error):   ", round(mean(abs.(res_all)), digits=4))
    println("Total xG RMSE:              ", round(sqrt(mean(res_all.^2)), digits=4))
    println("=======================================\n")
    
    # Return a detailed DataFrame
    return DataFrame(
        match_id = vcat(joined.match_id, joined.match_id),
        team_type = vcat(fill("Home", n_matches), fill("Away", n_matches)),
        observed_xg = vcat(joined.home_xg, joined.away_xg),
        expected_xg = vcat(exp_xg_h, exp_xg_a),
        residual = res_all
    )
end

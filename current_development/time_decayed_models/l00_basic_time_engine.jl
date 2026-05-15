
using BayesianFootball
using Revise
using DataFrames
using ThreadPinning
pinthreads(:cores)

using Turing, Distributions, Dates



# --- 2. Running the experiment
struct DSExperimentSettings 
  ds::Data.DataStore
  label::String
  save_dir::String
  target_season::Vector{<:String}
end

struct ExperimentTask
    ds::Data.DataStore
    config::Experiments.ExperimentConfig
end


get_target_seasons_string(::Data.Ireland)       = ["2026"]

"""
  Wrapper verison of the func to allow for the DSExperimentSettings type to 
  be used as a parameter.
"""
function create_experiment_tasks(es::DSExperimentSettings)
    return create_experiment_tasks(es.ds, es.label, es.save_dir, es.target_season)
end



function create_experiment_tasks(ds::Data.DataStore, model, label::String, save_dir::String, target_seasons::Vector{<:String} )

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

        Experiments.ExperimentConfig(
            name = "$(label)_",
            model = model, 
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


function run_experiment_task(task::ExperimentTask)
    conf = task.config
    println("Running: $(conf.name)")

    try
        # 2. Execute
        results = Experiments.run_experiment(task.ds, conf)

        # 3. Re-enable logging to save and confirm
        Experiments.save_experiment(results)
        
        return true # Success flag

    catch e
        @error "❌ Failed [$(conf.name)]: $e"
        # If you want to see the stacktrace for debugging:
        # Base.showerror(stdout, e, catch_backtrace())
        return false # Failure flag
    end
end


function loaded_experiment_files(saved_folders::Vector{String})
  loaded_results = Vector{BayesianFootball.Experiments.ExperimentResults}([])
  for folder in saved_folders
      try
          res = Experiments.load_experiment(folder)
          push!(loaded_results, res)
      catch e
          @warn "Could not load $folder: $e"
      end
  end

  if isempty(loaded_results)
      error("No results loaded! Did you run runner.jl?")
  end

  return loaded_results

end


# Time Decay Dynamics  abstraction 
# ==========================================
# 1.1 CONFIGURATION
# ==========================================
Base.@kwdef struct TimeDecayDynamics <: BayesianFootball.Models.PreGame.AbstractDynamicsConfig
  days_half_life::Real = 180
  σ_att::ContinuousUnivariateDistribution = Gamma(2.0, 0.15)
  σ_def::ContinuousUnivariateDistribution = Gamma(2.0, 0.15)
end

# ==========================================
# 1.2. TURING SUBMODEL
# ==========================================
@model function build_dynamics(config::TimeDecayDynamics, n_teams::Int)
    # Global variance for attack and defense spread
    σ_a ~ config.σ_att
    σ_d ~ config.σ_def
    
    # Non-centered parameterization (the Z-scores)
    raw_a ~ filldist(Normal(0, 1), n_teams)
    raw_d ~ filldist(Normal(0, 1), n_teams)
    
    # Scale them
    α_scaled = raw_a .* σ_a
    β_scaled = raw_d .* σ_d
    
    # Zero-sum constraint (ensures league average is exactly 0)
    α = α_scaled .- mean(α_scaled)
    β = β_scaled .- mean(β_scaled)
    
    return (; α, β)
end

# ==========================================
# 1.3. EXTRACTOR
# ==========================================
function extract_dynamics(chain::Chains, ::TimeDecayDynamics, prefix::String, n_teams::Int)
    n_samples = size(chain, 1) * size(chain, 3)
    
    # 1. Extract the global standard deviations (Vector of length n_samples)
    σ_a = vec(Array(chain[Symbol("$prefix.σ_a")]))
    σ_d = vec(Array(chain[Symbol("$prefix.σ_d")]))
    
    # 2. Extract the raw Z-scores
    raw_a_matrix = zeros(n_samples, n_teams)
    raw_d_matrix = zeros(n_samples, n_teams)
    
    for i in 1:n_teams
        raw_a_matrix[:, i] = vec(Array(chain[Symbol("$prefix.raw_a[$i]")]))
        raw_d_matrix[:, i] = vec(Array(chain[Symbol("$prefix.raw_d[$i]")]))
    end
    
    # 3. Reconstruct the scaled parameters
    # The dot operator matches the vector of sigmas to each row (sample) of the matrix
    α_scaled = raw_a_matrix .* σ_a 
    β_scaled = raw_d_matrix .* σ_d
    
    # 4. Apply the Zero-Sum constraint
    # dims=2 means we calculate the mean across the teams (columns) for each sample (row)
    α_matrix = α_scaled .- mean(α_scaled, dims=2)
    β_matrix = β_scaled .- mean(β_scaled, dims=2)
    
    return (; α = α_matrix, β = β_matrix)
end

Base.@kwdef struct DynamicGoalsTimeDecayModel{
  I<:BayesianFootball.Models.PreGame.AbstractInterceptionConfig,
  T<:BayesianFootball.Models.PreGame.AbstractDynamicsConfig, 
  D<:BayesianFootball.Models.PreGame.AbstractDispersionConfig, 
  H<:BayesianFootball.Models.PreGame.AbstractHomeAdvantageConfig
  } <: BayesianFootball.AbstractNegBinModel
    interception_config::I
    dynamics_config::T
    dispersion_config::D
    homeadvantage_config::H
end

# ==========================================
# 2.1. Goal engine time decay
# ==========================================
# src/Models/PreGame/engines/goals_engine.jl


# ==========================================
# 2.2. Features
# ==========================================
function BayesianFootball.Features.required_features(model::DynamicGoalsTimeDecayModel)
    return [:team_ids, :goals, :dates] 
end

# src/features/extractors 
function BayesianFootball.Features.add_feature!(F_data::Dict, ::Val{:dates}, ordered_ids, team_map::Dict, ds)
    # 1. Create a fast lookup dictionary: Match ID -> Date
    date_lookup = Dict(row.match_id => row.match_date for row in eachrow(ds.matches))
    
    # 2. Extract the dates in the exact sequence of ordered_ids
    subset_dates = [date_lookup[id] for id in ordered_ids]
    
    # 3. Find the newest date in this specific subset
    newest_in_subset = maximum(subset_dates)
    
    # 4. Calculate deltas and convert to integers
    F_data[:dates] = (newest_in_subset .- subset_dates) .|> Dates.value
end



@model function build_weighted_goals_engine(
    home_team_indices::Vector{Int},
    away_team_indices::Vector{Int},
    season_indices::Vector{Int},
    time_indices::Vector{Int},
    home_goals::Vector{Int},
    away_goals::Vector{Int},
    match_weights::Vector{Float64}, # <--- NEW: Pre-calculated array of weights [0.0 to 1.0]
    n_teams::Int,
    n_seasons::Int,
    config::DynamicGoalsTimeDecayModel
)
    # 1. LOAD COMPONENTS (Back to your simple global ones!)
    inter ~ to_submodel(BayesianFootball.Models.PreGame.build_interception(config.interception_config, n_seasons))
    disp  ~ to_submodel(BayesianFootball.Models.PreGame.build_dispersion(config.dispersion_config))
    ha    ~ to_submodel(BayesianFootball.Models.PreGame.build_home_advantage(config.homeadvantage_config, n_teams))
    dyn   ~ to_submodel(build_dynamics(config.dynamics_config, n_teams))

    # 2. VECTORIZED INDEXING 
    att_h = view(dyn.α, home_team_indices)
    def_h = view(dyn.β, home_team_indices)
    att_a = view(dyn.α, away_team_indices)
    def_a = view(dyn.β, away_team_indices)
    inter_match = view(inter, season_indices) # <--- NEW
    home_adv = view(ha, home_team_indices)
    

    # 3. VECTORIZED RATES (λ)
    λ_h = exp.(inter_match .+ home_adv .+ att_h .+ def_a)
    λ_a = exp.(inter_match .+             att_a .+ def_h)

    # ==========================================
    # 4. TIME-DECAYED LIKELIHOOD
    # ==========================================
    # Instead of: home_goals ~ arraydist(...)
    # We calculate the logpdf manually, multiply by weights, and add to the target.
    # Note: Using dot syntax here makes it insanely fast and Zygote-friendly.
    
    log_lik_h = logpdf.(BayesianFootball.MyDistributions.RobustNegativeBinomial.(disp.h, λ_h), home_goals)
    log_lik_a = logpdf.(BayesianFootball.MyDistributions.RobustNegativeBinomial.(disp.a, λ_a), away_goals)

    Turing.@addlogprob! sum(log_lik_h .* match_weights)
    Turing.@addlogprob! sum(log_lik_a .* match_weights)
end



function calculate_match_weights(deltas::Vector{<:Real}, half_life_days::Real)
    weights = 0.5 .^ (deltas ./ half_life_days)
    return weights
end

function BayesianFootball.Models.PreGame.build_turing_model(model::DynamicGoalsTimeDecayModel, feature_set::Features.FeatureSet)
    data = feature_set.data
    
    n_teams    = Int(data[:n_teams])
    n_seasons  = Int(data[:n_seasons]) # <--- Ensure your pipeline provides this
    n_history  = Int(data[:n_history_steps])
    n_target   = Int(data[:n_target_steps])

    date_deltas = Vector{Int}(data[:dates])
    match_weights = calculate_match_weights(date_deltas, model.dynamics_config.days_half_life)
    
    home_ids   = Vector{Int}(data[:flat_home_ids])
    away_ids   = Vector{Int}(data[:flat_away_ids])
    season_ids = Vector{Int}(data[:season_indices]) # <--- NEW
    time_idxs  = Vector{Int}(data[:time_indices])
    home_goals = Vector{Int}(data[:flat_home_goals])
    away_goals = Vector{Int}(data[:flat_away_goals])

    return build_weighted_goals_engine(
        home_ids,
        away_ids,
        season_ids,
        time_idxs,
        home_goals,
        away_goals,
        match_weights,
        n_teams,
        n_seasons,
        model
    )
end


function BayesianFootball.Models.PreGame.extract_parameters(
    model::DynamicGoalsTimeDecayModel, 
    df::AbstractDataFrame, 
    feature_set::BayesianFootball.Features.FeatureSet,
    chain::Chains
)
    # 1. Unpack Metadata
    data = feature_set.data
    n_teams   = Int(data[:n_teams])
    n_rounds  = Int(data[:n_rounds])
    n_history = Int(data[:n_history_steps])
    n_target  = Int(data[:n_target_steps])
    n_teams   = Int(data[:n_teams])
    n_seasons = Int(data[:n_seasons])
    team_map  = data[:team_map]

    # inter_mat is [Samples, Seasons]
    inter_mat = BayesianFootball.Models.PreGame.extract_interception(chain, model.interception_config, n_seasons)
    disp_nt   = BayesianFootball.Models.PreGame.extract_dispersion(chain, model.dispersion_config)
    ha_mat    = BayesianFootball.Models.PreGame.extract_home_advantage(chain, model.homeadvantage_config, n_teams)
    dyn_nt  = extract_dynamics(chain, model.dynamics_config, "dyn", n_teams)

    n_samples = size(chain, 1) * size(chain, 3) # total draws across all chains
    results = Dict{Int, NamedTuple}()

#     # ==========================================
#     # 3. FIXTURE LOOP (Calculate λ for each match)
#     # ==========================================
    for row in eachrow(df)
        mid = Int(row.match_id)

        # If forecasting future matches, default to the most recent time step (n_rounds)
        # If backtesting, use the exact time_index from the DataFrame
        t_idx = hasproperty(row, :time_index) ? Int(row.time_index) : n_rounds

        h_idx = get(team_map, row.home_team, -1)
        a_idx = get(team_map, row.away_team, -1)

        # Safely extract dynamic states (Fallback to 0 if a team isn't in the mapping)
        # Note: dyn_nt.α is [Teams, Time, Samples] based on your _reconstruct_trajectory logic
        α_h = h_idx > 0 ? dyn_nt.α[:, h_idx] : zeros(n_samples)
        β_h = h_idx > 0 ? dyn_nt.β[:, h_idx] : zeros(n_samples)
        α_a = a_idx > 0 ? dyn_nt.α[:, a_idx] : zeros(n_samples)
        β_a = a_idx > 0 ? dyn_nt.β[:, a_idx] : zeros(n_samples)

        # ha_mat is [Samples, Teams]
        γ_h = h_idx > 0 ? ha_mat[:, h_idx] : zeros(n_samples)

        s_idx = hasproperty(row, :season_idx) ? Int(row.season_idx) : n_seasons
        inter_match = inter_mat[:, s_idx] 
        # ==========================================
        # 4. FINAL LIKELIHOOD MATH
        # ==========================================
        λ_goals_h = exp.(inter_match .+ γ_h .+ α_h .+ β_a)
        λ_goals_a = exp.(inter_match .+        α_a .+ β_h)

        # Pack it exactly how model_inference() expects it!
        results[mid] = (;
            λ_h = λ_goals_h,
            λ_a = λ_goals_a,
            r_h = disp_nt.h,  # Handles Global vs Home/Away automatically
            r_a = disp_nt.a,

            # Since this is a Raw Goals model, True xG = Expected Goals
            true_xg_h = λ_goals_h, 
            true_xg_a = λ_goals_a
        )
    end

    return results
end

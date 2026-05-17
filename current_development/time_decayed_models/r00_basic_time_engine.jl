using BayesianFootball
using Revise
using DataFrames
using ThreadPinning
pinthreads(:cores)

using Dates

include("./l00_basic_time_engine.jl")

const PreGame = BayesianFootball.Models.PreGame

ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())
save_dir::String = "./data/test_time_decay_models/"

## models 
inter_cfg = PreGame.GlobalInterception()
disp_cfg  = PreGame.HomeAwayDispersion() 
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
dyn_cfg   = TimeDecayDynamics()
model = DynamicGoalsTimeDecayModel(
    interception_config  = inter_cfg,
    dynamics_config      = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg
)
training_task = create_experiment_tasks(ds, model, "test_1_timedecay_week", save_dir, ["2025"])

# results = run_experiment_task.(training_task)

# baseline model for grw 
dyn_cfg_grw  = PreGame.MultiScaleGRW(
    z₀ = Normal(0, 1), 
      zₛ = Normal(0,1),   # The fat-tailed robust winner
    zₖ = Normal(0, 1)
)

model_g = PreGame.DynamicGoalsModel(
    interception_config  = inter_cfg,
    dynamics_config      = dyn_cfg_grw,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
)

training_task = create_experiment_tasks(ds, model_g, "test_1", save_dir, ["2025"])

results = run_experiment_task.(training_task)




saved_folders = Experiments.list_experiments(save_dir; data_dir="")

loaded_results = loaded_experiment_files(saved_folders);

loaded_results_ = loaded_results[]



ledger = BayesianFootball.BackTesting.run_backtest(
    ds2, 
  loaded_results, 
  [BayesianFootball.Signals.BayesianKelly()]; 
    market_config = Data.Markets.DEFAULT_MARKET_CONFIG
)

tearsheet = BayesianFootball.BackTesting.generate_tearsheet(ledger)


function parse_dynamics_param(s::String)
    # We look for the literal prefix, then capture one or more digits/dots
    # until we hit the first comma.
    pattern = r"dynamics_config=TimeDecayDynamics\(([\d\.]+)"
    
    m = match(pattern, s)
    
    if m !== nothing
        # captures[1] contains the string "400"
        return parse(Int, m.captures[1]) 
    end
    
    return nothing # or throw an error if not found
end

tearsheet.model_parameters .=parse_dynamics_param.(tearsheet.model_parameters)


model_names = unique(tearsheet.selection)

model_names = model_names

for m_name in model_names
    println("\nStats for: $m_name")
    sub = subset(tearsheet, :selection => ByRow(isequal(m_name)))
    show(sub)
end






expr = loaded_results[1]

ch = expr.training_results[2][1]

describe(ch)

latents = BayesianFootball.Experiments.extract_oos_predictions(ds, expr)
ppd = BayesianFootball.Predictions.model_inference(ds, expr)





#=
julia> master_ll_df  = sort!(DataFrame(ll_rows), :logloss_overall_model_ll)
1×5 DataFrame
 Row │ model    logloss_overall_model_ll  logloss_overall_market_ll  logloss_overall_diff_ll  logloss_overall_n_obs 
     │ String   Float64                   Float64                    Float64                  Int64                 
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ test_1_                  0.504499                    18.8217                 -18.3172                   4006

julia> master_glm_df = sort!(DataFrame(glm_rows), :glmedge_spread_fair_p_value)
1×14 DataFrame
 Row │ model    glmedge_intercept_coef  glmedge_intercept_std_error  glmedge_intercept_z_score  glmedge_intercept_p_value  glmedge_prob_fair_coef  glmedge_prob_fair_std_error  glmedge_prob_fair_z_score  glmedge_prob_fair_p_value  glmedge_spread_fair_coef  glmedge_spread_fair_std_error  glmedge_spread_fair_z_score  glmedge_spread_fair_p_value  glmedge_n_obs 
     │ String   Float64                 Float64                      Float64                    Float64                    Float64                 Float64                      Float64                    Float64                    Float64                   Float64                        Float64                      Float64                      Int64         
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ test_1_                -2.89577                    0.0980509                   -29.5333               1.07653e-191                 5.89543                     0.188244                     31.318               2.65458e-215                  0.546279                       0.295422                      1.84915                     0.064436           4006

julia> master_rqr_df = DataFrame(rqr_rows)
1×19 DataFrame
 Row │ model    rqr_home_mean  rqr_home_std  rqr_home_skewness  rqr_home_kurtosis  rqr_home_shapiro_w  rqr_home_shapiro_p  rqr_away_mean  rqr_away_std  rqr_away_skewness  rqr_away_kurtosis  rqr_away_shapiro_w  rqr_away_shapiro_p  rqr_all_mean  rqr_all_std  rqr_all_skewness  rqr_all_kurtosis  rqr_all_shapiro_w  rqr_all_shapiro_p 
     │ String   Float64        Float64       Float64            Float64            Float64             Float64             Float64        Float64       Float64            Float64            Float64             Float64             Float64       Float64      Float64           Float64           Float64            Float64           
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ test_1_      0.0503466      0.969159          -0.326276           0.567423            0.975616          0.00301614     -0.0351357      0.887975          0.0450923          -0.153538            0.996119            0.927876    0.00760542     0.929145         -0.152731          0.255438           0.993545           0.127475
=#


cv_config = BayesianFootball.Data.CVConfig(
    tournament_ids = [79], 
    target_seasons = ["2025"],
    history_seasons = 2,   
    dynamics_col = :match_month,
    warmup_period = 0, # Using the calculated variable
    stop_early = true
)


const PreGame = BayesianFootball.Models.PreGame

inter_cfg = PreGame.GlobalInterception()
# Note: Using HomeAwayDispersion based on your previous grid (unless you built a custom TeamDispersion!)
disp_cfg  = PreGame.HomeAwayDispersion() 
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
dyn_cfg   = TimeDecayDynamics()

## models 

model = DynamicGoalsTimeDecayModel(
    interception_config  = inter_cfg,
    dynamics_config      = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg
)

boundaries_with_meta = BayesianFootball.Data.create_id_boundaries(ds, cv_config)

feature_collection = BayesianFootball.Features.create_features(boundaries_with_meta, ds, model)


f1 = feature_collection[1][1]
f2 = feature_collection[4][1]

calculate_match_weights(f2[:dates], 180)




# --- add - notes
# 1. The Expected Goals (xG) Time Decay Model
## ==========================================
# 1. THE STRUCT
# ==========================================
Base.@kwdef struct DynamicXGTimeDecayModel{
    I<:BayesianFootball.Models.PreGame.AbstractInterceptionConfig,
    T<:TimeDecayDynamics, 
    D<:BayesianFootball.Models.PreGame.AbstractDispersionConfig,
    H<:BayesianFootball.Models.PreGame.AbstractHomeAdvantageConfig,
    K<:BayesianFootball.Models.PreGame.AbstractKappaConfig
} <: BayesianFootball.AbstractXGModel
    ν_xg::ContinuousUnivariateDistribution = Gamma(10.0, 1.0) # Degrees of freedom for xG noise
    interception_config::I
    dynamics_config::T
    dispersion_config::D
    homeadvantage_config::H
    kappa_config::K
end

function BayesianFootball.Features.required_features(model::DynamicXGTimeDecayModel)
    # Don't forget to ask for :dates!
    return [:team_ids, :goals, :xg, :dates] 
end

# ==========================================
# 2. THE TURING ENGINE
# ==========================================
@model function build_weighted_xg_engine(
    home_team_indices::Vector{Int},
    away_team_indices::Vector{Int},
    season_indices::Vector{Int},
    time_indices::Vector{Int},
    home_goals::Vector{Int},
    away_goals::Vector{Int},
    home_xg::Vector{Float64},
    away_xg::Vector{Float64},
    idx_xg::Vector{Int},
    idx_no_xg::Vector{Int},
    match_weights::Vector{Float64}, # <--- NEW: Time weights
    n_teams::Int,
    n_seasons::Int,
    config::DynamicXGTimeDecayModel
)
    # 1. LOAD COMPONENTS
    ν_xg ~ config.ν_xg
    inter ~ to_submodel(BayesianFootball.Models.PreGame.build_interception(config.interception_config, n_seasons))
    disp  ~ to_submodel(BayesianFootball.Models.PreGame.build_dispersion(config.dispersion_config))
    ha    ~ to_submodel(BayesianFootball.Models.PreGame.build_home_advantage(config.homeadvantage_config, n_teams))
    kap   ~ to_submodel(BayesianFootball.Models.PreGame.build_kappa(config.kappa_config, n_teams))
    dyn   ~ to_submodel(build_dynamics(config.dynamics_config, n_teams))

    # 2. VECTORIZED INDEXING
    att_h = view(dyn.α, home_team_indices)
    def_h = view(dyn.β, home_team_indices)
    att_a = view(dyn.α, away_team_indices)
    def_a = view(dyn.β, away_team_indices)

    γ_h_flat = view(ha, home_team_indices)
    κ_h_flat = view(kap, home_team_indices)
    κ_a_flat = view(kap, away_team_indices)
    inter_match = view(inter, season_indices) 

    # 3. STABLE RATE GENERATION (True xG)
    log_λₕ = clamp.(inter_match .+ γ_h_flat .+ att_h .+ def_a, -20.0, 20.0) 
    log_λₐ = clamp.(inter_match .+             att_a .+ def_h, -20.0, 20.0)

    λₕ = exp.(log_λₕ) .+ 1e-6
    λₐ = exp.(log_λₐ) .+ 1e-6

    if any(isnan, λₕ) || any(isnan, λₐ) || any(isinf, λₕ) || any(isinf, λₐ)
        Turing.@addlogprob! -Inf
        return
    end

    # ==========================================
    # 4. TIME-DECAYED LIKELIHOOD PIPELINE
    # ==========================================
    
    # Group 1: Matches WITH xG Data
    if !isempty(idx_xg)
        # xG Likelihoods (Gamma)
        ll_xg_h = logpdf.(Gamma.(ν_xg, λₕ[idx_xg] ./ ν_xg), home_xg[idx_xg])
        ll_xg_a = logpdf.(Gamma.(ν_xg, λₐ[idx_xg] ./ ν_xg), away_xg[idx_xg])
        
        # Goals Likelihoods (RobustNegativeBinomial)
        ll_g_h_xg = logpdf.(BayesianFootball.MyDistributions.RobustNegativeBinomial.(disp.h, κ_h_flat[idx_xg] .* λₕ[idx_xg]), home_goals[idx_xg])
        ll_g_a_xg = logpdf.(BayesianFootball.MyDistributions.RobustNegativeBinomial.(disp.a, κ_a_flat[idx_xg] .* λₐ[idx_xg]), away_goals[idx_xg])

        # Sum them up, apply time weights, and add to target
        Turing.@addlogprob! sum((ll_xg_h .+ ll_xg_a .+ ll_g_h_xg .+ ll_g_a_xg) .* match_weights[idx_xg])
    end

    # Group 2: Matches WITHOUT xG Data
    if !isempty(idx_no_xg)
        ll_g_h_no = logpdf.(BayesianFootball.MyDistributions.RobustNegativeBinomial.(disp.h, κ_h_flat[idx_no_xg] .* λₕ[idx_no_xg]), home_goals[idx_no_xg])
        ll_g_a_no = logpdf.(BayesianFootball.MyDistributions.RobustNegativeBinomial.(disp.a, κ_a_flat[idx_no_xg] .* λₐ[idx_no_xg]), away_goals[idx_no_xg])
        
        Turing.@addlogprob! sum((ll_g_h_no .+ ll_g_a_no) .* match_weights[idx_no_xg])
    end
end

# ==========================================
# 3. THE BUILDER
# ==========================================
function BayesianFootball.Models.PreGame.build_turing_model(config::DynamicXGTimeDecayModel, feature_set::BayesianFootball.Features.FeatureSet)
    data = feature_set.data
    
    n_teams   = Int(data[:n_teams])
    n_seasons = Int(data[:n_seasons]) 

    # Handle Time Weights
    date_deltas = Vector{Int}(data[:dates])
    match_weights = calculate_match_weights(date_deltas, config.dynamics_config.days_half_life)
    
    home_ids   = Vector{Int}(data[:flat_home_ids])
    away_ids   = Vector{Int}(data[:flat_away_ids])
    season_ids = Vector{Int}(data[:season_indices]) 
    time_idxs  = Vector{Int}(data[:time_indices])
    home_goals = Vector{Int}(data[:flat_home_goals])
    away_goals = Vector{Int}(data[:flat_away_goals])

    # Clean xG Data
    home_xg = Vector{Float64}(coalesce.(data[:flat_home_xg], NaN))
    away_xg = Vector{Float64}(coalesce.(data[:flat_away_xg], NaN))

    idx_xg    = findall(x -> !isnan(x), home_xg)
    idx_no_xg = findall(isnan, home_xg)

    return build_weighted_xg_engine(
        home_ids, away_ids, season_ids, time_idxs, 
        home_goals, away_goals, 
        home_xg, away_xg, 
        idx_xg, idx_no_xg,
        match_weights,
        n_teams, n_seasons, config
    )
end

# ==========================================
# 4. THE EXTRACTOR (Reusing your xg_engine logic)
# ==========================================
# Note: You can reuse your exact extract_parameters from xg_engine.jl, 
# just change the signature to:
# extract_parameters(model::DynamicXGTimeDecayModel, ...) 
# and use the static `dyn_nt.α[:, h_idx]` indexing we fixed earlier!
#
#
#
# 2. The Market + Goals Time Decay Model
## ==========================================
# 1. THE STRUCT
# ==========================================
Base.@kwdef struct DynamicMarketGoalsTimeDecayModel{
    I<:BayesianFootball.Models.PreGame.AbstractInterceptionConfig,
    T<:TimeDecayDynamics, 
    D<:BayesianFootball.Models.PreGame.AbstractDispersionConfig,
    H<:BayesianFootball.Models.PreGame.AbstractHomeAdvantageConfig
} <: BayesianFootball.AbstractGoalsModel
    market_σ::ContinuousUnivariateDistribution = Exponential(0.2)
    interception_config::I
    dynamics_config::T
    dispersion_config::D
    homeadvantage_config::H
end

function BayesianFootball.Features.required_features(model::DynamicMarketGoalsTimeDecayModel)
    return [:team_ids, :goals, :market_lambda, :dates] 
end

# ==========================================
# 2. THE TURING ENGINE
# ==========================================
@model function build_weighted_market_goals_engine(
    home_team_indices::Vector{Int},
    away_team_indices::Vector{Int},
    season_indices::Vector{Int},
    time_indices::Vector{Int},
    home_goals::Vector{Int},
    away_goals::Vector{Int},
    market_log_λ_h::Vector{Float64},
    market_log_λ_a::Vector{Float64},
    idx_market::Vector{Int},
    idx_no_market::Vector{Int},
    match_weights::Vector{Float64}, # <--- NEW
    n_teams::Int,
    n_seasons::Int,         
    config::DynamicMarketGoalsTimeDecayModel 
)
    # 1. LOAD COMPONENTS
    σ_market ~ config.market_σ
    inter ~ to_submodel(BayesianFootball.Models.PreGame.build_interception(config.interception_config, n_seasons))
    disp  ~ to_submodel(BayesianFootball.Models.PreGame.build_dispersion(config.dispersion_config))
    ha    ~ to_submodel(BayesianFootball.Models.PreGame.build_home_advantage(config.homeadvantage_config, n_teams))
    dyn   ~ to_submodel(build_dynamics(config.dynamics_config, n_teams))

    # 2. VECTORIZED INDEXING
    att_h = view(dyn.α, home_team_indices)
    def_h = view(dyn.β, home_team_indices)
    att_a = view(dyn.α, away_team_indices)
    def_a = view(dyn.β, away_team_indices)

    home_adv = view(ha, home_team_indices)
    inter_match = view(inter, season_indices) 

    # 3. RATE GENERATION (Log Scale)
    log_λ_h = clamp.(inter_match .+ home_adv .+ att_h .+ def_a, -10.0, 10.0) 
    log_λ_a = clamp.(inter_match .+             att_a .+ def_h, -10.0, 10.0)

    λ_h = exp.(log_λ_h) .+ 1e-6
    λ_a = exp.(log_λ_a) .+ 1e-6

    if any(isnan, λ_h) || any(isnan, λ_a) || any(isinf, λ_h) || any(isinf, λ_a)
        Turing.@addlogprob! -Inf
        return
    end

    # ==========================================
    # 4. TIME-DECAYED LIKELIHOOD PIPELINE
    # ==========================================
    
    # A. Goal Likelihood (Always runs)
    ll_g_h = logpdf.(BayesianFootball.MyDistributions.RobustNegativeBinomial.(disp.h, λ_h), home_goals)
    ll_g_a = logpdf.(BayesianFootball.MyDistributions.RobustNegativeBinomial.(disp.a, λ_a), away_goals)
    
    Turing.@addlogprob! sum((ll_g_h .+ ll_g_a) .* match_weights)

    # B. Market Likelihood (Only for matches where we have market odds)
    if !isempty(idx_market)
        ll_m_h = logpdf.(Normal.(log_λ_h[idx_market], σ_market), market_log_λ_h[idx_market])
        ll_m_a = logpdf.(Normal.(log_λ_a[idx_market], σ_market), market_log_λ_a[idx_market])
        
        # Apply the same time decay weights to the market data!
        Turing.@addlogprob! sum((ll_m_h .+ ll_m_a) .* match_weights[idx_market])
    end
end

# ==========================================
# 3. THE BUILDER
# ==========================================
function BayesianFootball.Models.PreGame.build_turing_model(config::DynamicMarketGoalsTimeDecayModel, feature_set::BayesianFootball.Features.FeatureSet)
    data = feature_set.data
    
    n_teams   = Int(data[:n_teams])
    n_seasons = Int(data[:n_seasons])

    # Handle Time Weights
    date_deltas = Vector{Int}(data[:dates])
    match_weights = calculate_match_weights(date_deltas, config.dynamics_config.days_half_life)
    
    home_ids   = Vector{Int}(data[:flat_home_ids])
    away_ids   = Vector{Int}(data[:flat_away_ids])
    time_idxs  = Vector{Int}(data[:time_indices])
    home_goals = Vector{Int}(data[:flat_home_goals])
    away_goals = Vector{Int}(data[:flat_away_goals])
    season_ids = Vector{Int}(data[:season_indices])

    # Extract Market Lambdas
    market_log_h = Vector{Float64}(coalesce.(log.(data[:flat_market_λ_home]), NaN))
    market_log_a = Vector{Float64}(coalesce.(log.(data[:flat_market_λ_away]), NaN))

    idx_market    = findall(x -> !isnan(x), market_log_h)
    idx_no_market = findall(isnan, market_log_h)

    return build_weighted_market_goals_engine(
        home_ids, away_ids, season_ids, time_idxs, 
        home_goals, away_goals,
        market_log_h, market_log_a, 
        idx_market, idx_no_market,
        match_weights,
        n_teams, n_seasons, config
    )
end

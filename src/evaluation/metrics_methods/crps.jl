# src/evaluation/metrics_methods/crps.jl 

export CRPS, CRPSResults, CRPSComponent

# --- The Trigger ---
struct CRPS <: AbstractScoringRule end 

struct CRPSComponent <: AbstractMetricComponent 
    mean::Float64 
end 

struct CRPSResults <: AbstractEvaluationResult 
    home::CRPSComponent
    away::CRPSComponent
    all::CRPSComponent
end

# The Translator needs to know the name of the RESULT struct
function get_metric_method_name(::CRPSResults)::String
    return "crps"
end

function get_metric_method_name(::CRPS)::String
    return "crps"
end


# ==============================================================================
# MATH & HELPERS
# ==============================================================================

function compute_crps(y::Int, λ::Float64, r_disp::Float64; max_goals=30)
    # Convert mean (λ) and dispersion (r) to NegBinomial (r, p)
    p = r_disp / (r_disp + λ)
    dist = NegativeBinomial(r_disp, p)
    
    crps_value = 0.0
    # Sum over possible goal counts
    for x in 0:max_goals
        F_x = cdf(dist, x)            # Model's cumulative probability up to x goals
        indicator = x >= y ? 1.0 : 0.0 # 1.0 if the actual score was less than or equal to x
        
        crps_value += (F_x - indicator)^2
    end
    
    return crps_value
end

# 2. Extract Dispersion (Handles hierarchical vs global r)
function get_r(df)
    if hasproperty(df, :r)
        return mean.(df.r), mean.(df.r)
    elseif hasproperty(df, :r_h)
        return mean.(df.r_h), mean.(df.r_a) 
    else
        throw(ArgumentError("Row does not contain expected shape parameters (:r or :r_h)"))
    end 
end

# ==============================================================================
# MAIN COMPUTE METHOD
# ==============================================================================

function compute_metric(metric::CRPS, exp::ExperimentResults, ds::DataStore)::CRPSResults
    # 1. Extract latents (Expected Goals, Dispersions)
    latents_raw = Experiments.extract_oos_predictions(ds, exp)
    
    latent_cols = Predictions.get_latent_column_symbols(exp.config.model, latents_raw.df)
    
    # Ensure :match_id is included so we can join!
    if :match_id ∉ latent_cols
        push!(latent_cols, :match_id)
    end

    joined = innerjoin(
        select(latents_raw.df, latent_cols),
        select(ds.matches, :match_id, :match_month, :match_date, :home_score, :away_score, :tournament_id, :season, :home_team, :away_team),
        on = :match_id
    )

    # 2. Extract Expected values (Means of posterior samples)
    exp_home = mean.(joined.λ_h)
    exp_away = mean.(joined.λ_a)
    exp_r_h, exp_r_a = get_r(joined)

    # 3. Compute CRPS vectors
    crps_home = compute_crps.(joined.home_score, exp_home, exp_r_h)
    crps_away = compute_crps.(joined.away_score, exp_away, exp_r_a)
    
    # For football, the "Match CRPS" is generally defined as the average of the Home and Away CRPS
    crps_all = (crps_home .+ crps_away) ./ 2.0 

    # 4. Pack the results into the strict structs and return
    # (Notice we pass the mean of the arrays, as CRPS is evaluated as an aggregate average over the dataset)
    return CRPSResults(
        CRPSComponent(mean(crps_home)),
        CRPSComponent(mean(crps_away)),
        CRPSComponent(mean(crps_all))
    )
end

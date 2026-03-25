# src/evaluation/metrics_methods/glm_edge.jl 

export GLMEdge, GLMEdgeResult, GLMCoefComponent

# --- The Trigger ---
struct GLMEdge <: AbstractScoringRule end

# --- The Components ---
struct GLMCoefComponent <: AbstractMetricComponent
    coef::Float64
    std_error::Float64
    z_score::Float64
    p_value::Float64
end

struct GLMEdgeResult <: AbstractEvaluationResult
    intercept::GLMCoefComponent
    prob_fair::GLMCoefComponent
    spread_fair::GLMCoefComponent
    n_obs::Int
end

# --- Translator Mappings ---
function get_metric_method_name(::GLMEdgeResult)::String
    return "glmedge"
end

function get_metric_method_name(::GLMEdge)::String
    return "glmedge"
end

# ==============================================================================
# MAIN COMPUTE METHOD
# ==============================================================================

function compute_metric(metric::GLMEdge, exp::ExperimentResults, ds::DataStore)::GLMEdgeResult
    
    # 1. Extract Latents
    latents_raw = Experiments.extract_oos_predictions(ds, exp)
    
    # 2. Prepare Market Data
    # (Assuming Data.prepare_market_data exists in your Data module as you showed)
    market_data = Data.prepare_market_data(ds)

    # 3. Model Inference (PPD - Posterior Predictive Distribution)
    # Generates betting probabilities based on latents
    ppd = Predictions.model_inference(latents_raw)
    
    model_features = transform(ppd.df, :distribution => ByRow(mean) => :prob_model)
    select!(model_features, :match_id, :market_name, :market_line, :selection, :prob_model)

    # 4. Merge with Market Data
    analysis_df = innerjoin(
        market_data.df,
        model_features,
        on = [:match_id, :market_name, :market_line, :selection]
    )
    
    # 5. Clean & Calculate Spread
    dropmissing!(analysis_df, [:odds_close, :is_winner])
    
    analysis_df.spread = analysis_df.prob_model .- analysis_df.prob_implied_close
    analysis_df.spread_fair = analysis_df.prob_model .- analysis_df.prob_fair_close
    analysis_df.Y = Float64.(analysis_df.is_winner)
    
    # 6. Run the Logistic Regression
    reg_model = glm(@formula(Y ~ prob_fair_close + spread_fair), analysis_df, Binomial(), LogitLink())
    
    # 7. Extract Coefficients Safely
    ct = coeftable(reg_model)
    
    # Helper function to extract a row from the GLM Coefficient Table
    function extract_coef(name::String)
        idx = findfirst(==(name), ct.rownms)
        if isnothing(idx)
            return GLMCoefComponent(NaN, NaN, NaN, NaN)
        end
        return GLMCoefComponent(
            ct.cols[1][idx], # Coef
            ct.cols[2][idx], # Std. Error
            ct.cols[3][idx], # Z
            ct.cols[4][idx]  # Pr(>|z|)
        )
    end

    # 8. Pack and Return
    return GLMEdgeResult(
        extract_coef("(Intercept)"),
        extract_coef("prob_fair_close"),
        extract_coef("spread_fair"),
        nrow(analysis_df)
    )
end

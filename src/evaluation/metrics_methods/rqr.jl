# src/evaluation/metrics_methods/rqr.jl 

export RQR, RQRResult

# --- The Triggers ---
struct RQR <: AbstractScoringRule end

# A reusable component for any distribution
struct DistributionStats <: AbstractMetricComponent
    mean::Float64
    std::Float64
    skewness::Float64
    kurtosis::Float64
    shapiro_w::Float64
    shapiro_p::Float64
end

# The specific result payloads
struct RQRResult <: AbstractEvaluationResult
    home::DistributionStats
    away::DistributionStats
    all::DistributionStats 
end

# The Translator needs to know the name of the RESULT struct
function get_metric_method_name(::RQRResult)::String
    return "rqr"
end

function get_metric_method_name(::RQR)::String
    return "rqr"
end

# ==============================================================================
# MATH & HELPERS
# ==============================================================================

# 1. Core RQR Calculation
function compute_rqr(y::Int, λ::Float64, r_disp::Float64)
    p = r_disp / (r_disp + λ)
    dist = NegativeBinomial(r_disp, p)
    
    cdf_lower = y > 0 ? cdf(dist, y - 1) : 0.0
    cdf_upper = cdf(dist, y)
    
    u = rand(Uniform(cdf_lower, cdf_upper))
    
    # Clamp u slightly to avoid Inf/-Inf in extreme edge cases
    u = clamp(u, 1e-7, 1.0 - 1e-7) 
    return quantile(Normal(0, 1), u)
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

# 3. Helper to pack arrays into our Component DTO
function _summarize_stats(x::Vector{Float64})::DistributionStats
    sw = ShapiroWilkTest(x)
    return DistributionStats(
        mean(x),
        std(x),
        skewness(x),
        kurtosis(x),
        sw.W,
        pvalue(sw)
    )
end

# ==============================================================================
# MAIN COMPUTE METHOD
# ==============================================================================

function compute_metric(metric::RQR, exp::ExperimentResults, ds::DataStore)::RQRResult
    # 1. Extract latents (Expected Goals, Dispersions)
    latents_raw = Experiments.extract_oos_predictions(ds, exp)
    
    latent_cols = Predictions.get_latent_column_symbols(exp.config.model, latents_raw.df)

    joined = innerjoin(
        select(latents_raw.df, latent_cols),
        select(ds.matches, :match_id, :match_month, :match_date, :home_score, :away_score, :tournament_id, :season, :home_team, :away_team),
        on = :match_id
    )

    # 2. Extract Expected values (Means of posterior samples)
    exp_home = mean.(joined.λ_h)
    exp_away = mean.(joined.λ_a)
    exp_r_h, exp_r_a = get_r(joined)

    # 3. Compute RQR vectors
    rqr_home = compute_rqr.(joined.home_score, exp_home, exp_r_h)
    rqr_away = compute_rqr.(joined.away_score, exp_away, exp_r_a)
    rqr_all  = vcat(rqr_home, rqr_away) # Pooled together for total calibration check

    # 4. Pack the results into the strict structs and return
    return RQRResult(
        _summarize_stats(rqr_home),
        _summarize_stats(rqr_away),
        _summarize_stats(rqr_all)
    )
end

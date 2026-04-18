
using Revise
using BayesianFootball

using DataFrames
using ThreadPinning
pinthreads(:cores)


using Distributions
using HypothesisTests

function get_goals(ds::Data.DataStore)
    ds.matches.goals_total =  ds.matches.home_score .+ ds.matches.away_score;

    goals = Dict{String, AbstractVector{<:Integer}}(
            "home" => collect(skipmissing(ds.matches.home_score)),
            "away" => collect(skipmissing(ds.matches.away_score)),
            "total"=> collect(skipmissing(ds.matches.goals_total)),
    )
  return goals

end

function simple_describe(goals::Vector{<:Integer}, label::String)
    m = mean(goals)
    s = std(goals)
    n = length(goals)
    println("$(label) ::  Mean:$m | std:$s | n:$n")
end 

function simple_describe(goals::Dict{String, AbstractVector{<:Integer}})
  for (label, vec) in goals
    simple_describe(vec, label) 
  end 
end 



# ==========================================
# 2. Fit the Distributions
# ==========================================

using Optim
using Distributions, Statistics, StatsBase, Printf

function fit_mle(::Type{MyDistributions.RobustNegativeBinomial}, data)
    # Start with MoM estimates as a guess
    m = mean(data)
    v = var(data)
    r_guess = v > m ? m^2 / (v - m) : 10.0
    
    # Minimize the Negative Log-Likelihood
    # Use log-transformed parameters so the optimizer can't pick negative values
    func(params) = -sum(logpdf(MyDistributions.RobustNegativeBinomial(exp(params[1]), exp(params[2])), data))
    
    res = optimize(func, [log(r_guess), log(m)])
    
    return MyDistributions.RobustNegativeBinomial(exp(res.minimizer[1]), exp(res.minimizer[2]))
end


# function fit_goal_distributions(data::AbstractVector{<:Integer})
#     # 1. Fit Poisson (standard MLE)
#     p_dist = fit(Poisson, data)
#
#     nb_dist = fit_mle(MyDistributions.RobustNegativeBinomial, data)
#
#     # # 2. Fit Robust Negative Binomial (Method of Moments)
#     # μ = mean(data)
#     # σ2 = var(data)
#     #
#     # if σ2 > μ
#     #     # Solving for r using the variance formula: σ² = μ + μ²/r
#     #     r = μ^2 / (σ2 - μ)
#     #     nb_dist = MyDistributions.RobustNegativeBinomial(r, μ)
#     # else
#     #     # If variance <= mean, NB is not the right tool
#     #     nb_dist = nothing 
#     # end
#     #
#     return (poisson = p_dist, nb = nb_dist)
# end
#
function compute_metrics(dist, data::AbstractVector{<:Integer})
    isnothing(dist) && return nothing

    # Log-Likelihood & AIC
    ll = loglikelihood(dist, data)
    k_params = length(params(dist))
    aic = 2*k_params - 2*ll
    
    # Chi-Squared Goodness of Fit
    obs_counts = counts(data, 0:7)
    n = length(data)
    
    expected = [pdf(dist, i) * n for i in 0:6]
    push!(expected, (1 - cdf(dist, 6)) * n) 
    
    expected_safe = max.(expected, 1e-6)
    chi_sq = sum((obs_counts .- expected_safe).^2 ./ expected_safe)
    
    # --- NEW: Degrees of Freedom & P-Value ---
    num_bins = length(obs_counts) # This is 8
    df = num_bins - 1 - k_params
    
    # ccdf gives us the area to the right of the chi_sq value (the p-value)
    p_val = ccdf(Chisq(df), chi_sq)
    
    return (log_likelihood = ll, aic = aic, chi_sq = chi_sq, df = df, p_value = p_val)
end


# function analyze_goal_models(goals_dict::Dict{String, <:AbstractVector{<:Integer}})
#     for (label, data) in goals_dict
#         println("\n" * "═"^48)
#         println(" MODEL COMPARISON: $(uppercase(label)) ")
#         println("═"^48)
#
#         fits = fit_goal_distributions(data)
#
#         p_stats = compute_metrics(fits.poisson, data)
#         nb_stats = compute_metrics(fits.nb, data)
#
#         @printf("%-18s | %-12s | %-12s\n", "Metric", "Poisson", "Robust NB")
#         println("-"^48)
#
#         # Helper function to print safely handling N/A
#         print_row(name, p_val, nb_stats, metric, fmt) = begin
#             if isnothing(nb_stats)
#                 @printf("%-18s | %-12s | %-12s\n", name, Printf.format(fmt, p_val), "N/A")
#             else
#                 nb_val = getproperty(nb_stats, metric)
#                 @printf("%-18s | %-12s | %-12s\n", name, Printf.format(fmt, p_val), Printf.format(fmt, nb_val))
#             end
#         end
#
#         # Float format (2 decimals)
#         fmt_float = Printf.Format("%.2f")
#         # Int format
#         fmt_int = Printf.Format("%d")
#         # P-value format (4 decimals)
#         fmt_pval = Printf.Format("%.4f")
#
#         print_row("Log likelihood", p_stats.log_likelihood, nb_stats, :log_likelihood, fmt_float)
#         print_row("AIC", p_stats.aic, nb_stats, :aic, fmt_float)
#         print_row("Chi sq", p_stats.chi_sq, nb_stats, :chi_sq, fmt_float)
#         print_row("Degrees of freedom", p_stats.df, nb_stats, :df, fmt_int)
#         print_row("P-value", p_stats.p_value, nb_stats, :p_value, fmt_pval)
#
#         if isnothing(nb_stats)
#             println("\n[!] RobustNB skipped: No overdispersion detected.")
#         end
#     end
# end
#
#
#
# --- weibull 

# weibull 
function fit_mle(::Type{MyDistributions.WeibullCount}, data)
    # Start with Poisson assumptions as a guess:
    # When c = 1.0, WeibullCount reduces to Poisson, and λ = mean 
    c_guess = 1.0
    λ_guess = mean(data)
    
    # Minimize the Negative Log-Likelihood
    function objective(params)
        c_val = exp(params[1])
        λ_val = exp(params[2])
        dist = MyDistributions.WeibullCount(c_val, λ_val)
        
        # Sum logpdf over all data points
        return -sum(logpdf(dist, x) for x in data)
    end
    
    # Run the optimization
    res = optimize(objective, [log(c_guess), log(λ_guess)])
    
    return MyDistributions.WeibullCount(exp(res.minimizer[1]), exp(res.minimizer[2]))
end



function fit_goal_distributions(data::AbstractVector{<:Integer})
    # 1. Fit Poisson (standard MLE)
    p_dist = fit(Poisson, data)

    # 2. Fit Robust Negative Binomial
    nb_dist = fit_mle(MyDistributions.RobustNegativeBinomial, data)
    
    # 3. Fit Weibull Count
    wc_dist = fit_mle(MyDistributions.WeibullCount, data)
    
    return (poisson = p_dist, nb = nb_dist, wc = wc_dist)
end

function analyze_goal_models(goals_dict::Dict{String, <:AbstractVector{<:Integer}})
    for (label, data) in goals_dict
        println("\n" * "═"^66)
        println(" MODEL COMPARISON: $(uppercase(label)) ")
        println("═"^66)
        
        fits = fit_goal_distributions(data)
        
        p_stats = compute_metrics(fits.poisson, data)
        nb_stats = compute_metrics(fits.nb, data)
        wc_stats = compute_metrics(fits.wc, data)
        
        @printf("%-18s | %-12s | %-12s | %-12s\n", "Metric", "Poisson", "Robust NB", "Weibull Cnt")
        println("-"^66)
        
        # Updated helper function for 3 columns
        print_row(name, p_val, nb_stats, wc_stats, metric, fmt) = begin
            p_str  = Printf.format(fmt, p_val)
            nb_str = isnothing(nb_stats) ? "N/A" : Printf.format(fmt, getproperty(nb_stats, metric))
            wc_str = isnothing(wc_stats) ? "N/A" : Printf.format(fmt, getproperty(wc_stats, metric))
            
            @printf("%-18s | %-12s | %-12s | %-12s\n", name, p_str, nb_str, wc_str)
        end

        fmt_float = Printf.Format("%.2f")
        fmt_int = Printf.Format("%d")
        fmt_pval = Printf.Format("%.4f")

        print_row("Log likelihood", p_stats.log_likelihood, nb_stats, wc_stats, :log_likelihood, fmt_float)
        print_row("AIC", p_stats.aic, nb_stats, wc_stats, :aic, fmt_float)
        print_row("Chi sq", p_stats.chi_sq, nb_stats, wc_stats, :chi_sq, fmt_float)
        print_row("Degrees of fr.", p_stats.df, nb_stats, wc_stats, :df, fmt_int)
        print_row("P-value", p_stats.p_value, nb_stats, wc_stats, :p_value, fmt_pval)
        
        if isnothing(nb_stats)
            println("\n[!] RobustNB skipped: No overdispersion detected.")
        end
    end
end

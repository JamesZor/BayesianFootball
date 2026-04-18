
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
            # "total"=> collect(skipmissing(ds.matches.goals_total)),
            "total" => vcat(collect(skipmissing(ds.matches.home_score)), collect(skipmissing(ds.matches.away_score)))
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



# ---
using Optim
using Distributions, Statistics, StatsBase, Printf

# --- 1. Dixon-Coles Log-Likelihood Function ---
# We write a clean MLE objective using the exact logic from your uploaded files.
function dixon_coles_loglikelihood(λ_val, μ_val, ρ_val, home_goals, away_goals)
    ll = 0.0
    for (h, a) in zip(home_goals, away_goals)
        # Base independent Poisson log-likelihood
        log_indep = logpdf(Poisson(λ_val), h) + logpdf(Poisson(μ_val), a)
        
        # Dixon-Coles Tau correction for low scores
        τ = 1.0
        if h == 0 && a == 0
            τ = 1.0 - (λ_val * μ_val * ρ_val)
        elseif h == 1 && a == 0
            τ = 1.0 + (μ_val * ρ_val)
        elseif h == 0 && a == 1
            τ = 1.0 + (λ_val * ρ_val)
        elseif h == 1 && a == 1
            τ = 1.0 - ρ_val
        end
        
        # Safe log for τ to prevent domain errors during optimization
        log_τ = τ > 0 ? log(τ) : -Inf
        ll += log_indep + log_τ
    end
    return ll
end

# --- 2. Fit the Bivariate Models ---
function fit_bivariate_models(home_data, away_data)
    # Model 1: Independent Poisson (Baseline)
    # For independent Poisson, the MLE is just the means!
    λ_indep = mean(home_data)
    μ_indep = mean(away_data)
    ll_indep = sum(logpdf.(Poisson(λ_indep), home_data)) + sum(logpdf.(Poisson(μ_indep), away_data))
    
    # Model 2: Dixon-Coles
    # We optimize λ, μ, and ρ simultaneously
    function objective_dc(params)
        λ_val = exp(params[1])
        μ_val = exp(params[2])
        ρ_val = tanh(params[3]) # scale to roughly [-1, 1] safely
        return -dixon_coles_loglikelihood(λ_val, μ_val, ρ_val, home_data, away_data)
    end
    
    # Initial guess: Means for rates, 0.0 for rho
    res_dc = optimize(objective_dc, [log(λ_indep), log(μ_indep), 0.0])
    
    λ_dc = exp(res_dc.minimizer[1])
    μ_dc = exp(res_dc.minimizer[2])
    ρ_dc = tanh(res_dc.minimizer[3])
    ll_dc = -res_dc.minimum
    
    return (
        indep = (λ = λ_indep, μ = μ_indep, ρ = 0.0, ll = ll_indep, k = 2),
        dc    = (λ = λ_dc, μ = μ_dc, ρ = ρ_dc, ll = ll_dc, k = 3)
    )
end

# --- 3. Analyze and Print ---
function analyze_bivariate_models(ds::BayesianFootball.Data.DataStore)
    home_data = collect(skipmissing(ds.matches.home_score))
    away_data = collect(skipmissing(ds.matches.away_score))
    
    fits = fit_bivariate_models(home_data, away_data)
    
    println("\n" * "═"^60)
    println(" BIVARIATE MODEL COMPARISON (HOME vs AWAY) ")
    println("═"^60)
    
    @printf("%-20s | %-15s | %-15s\n", "Metric", "Indep. Poisson", "Dixon-Coles")
    println("-"^60)
    
    fmt_float = Printf.Format("%.4f")
    fmt_ll    = Printf.Format("%.2f")
    
    # Calculate AIC: 2k - 2 * LogLikelihood
    aic_indep = 2 * fits.indep.k - 2 * fits.indep.ll
    aic_dc    = 2 * fits.dc.k - 2 * fits.dc.ll
    
    @printf("%-20s | %-15s | %-15s\n", "λ (Home Rate)", Printf.format(fmt_float, fits.indep.λ), Printf.format(fmt_float, fits.dc.λ))
    @printf("%-20s | %-15s | %-15s\n", "μ (Away Rate)", Printf.format(fmt_float, fits.indep.μ), Printf.format(fmt_float, fits.dc.μ))
    @printf("%-20s | %-15s | %-15s\n", "ρ (Dependence)", "N/A", Printf.format(fmt_float, fits.dc.ρ))
    println("-"^60)
    @printf("%-20s | %-15s | %-15s\n", "Log likelihood", Printf.format(fmt_ll, fits.indep.ll), Printf.format(fmt_ll, fits.dc.ll))
    @printf("%-20s | %-15s | %-15s\n", "AIC", Printf.format(fmt_ll, aic_indep), Printf.format(fmt_ll, aic_dc))
    
    if aic_dc < aic_indep
        println("\n[RESULT] Dixon-Coles improved the fit! The ρ parameter is capturing dependence.")
    else
        println("\n[RESULT] Independent Poisson wins. Correlation (ρ) did not justify the extra parameter.")
    end
end



# -----

using Optim
using Distributions, Statistics, StatsBase, Printf

# --- 1. Log-Likelihood Functions ---

function dixon_coles_poisson_loglikelihood(λ_val, μ_val, ρ_val, home_goals, away_goals)
    ll = 0.0
    for (h, a) in zip(home_goals, away_goals)
        log_indep = logpdf(Poisson(λ_val), h) + logpdf(Poisson(μ_val), a)
        
        τ = 1.0
        if h == 0 && a == 0; τ = 1.0 - (λ_val * μ_val * ρ_val)
        elseif h == 1 && a == 0; τ = 1.0 + (μ_val * ρ_val)
        elseif h == 0 && a == 1; τ = 1.0 + (λ_val * ρ_val)
        elseif h == 1 && a == 1; τ = 1.0 - ρ_val
        end
        
        ll += log_indep + (τ > 0 ? log(τ) : -Inf)
    end
    return ll
end

function dixon_coles_nb_loglikelihood(λ_val, μ_val, r_h_val, r_a_val, ρ_val, home_goals, away_goals)
    ll = 0.0
    for (h, a) in zip(home_goals, away_goals)
        # Using your custom RobustNegativeBinomial (Dispersion, Mean)
        log_indep = logpdf(MyDistributions.RobustNegativeBinomial(r_h_val, λ_val), h) + 
                    logpdf(MyDistributions.RobustNegativeBinomial(r_a_val, μ_val), a)
        
        τ = 1.0
        if h == 0 && a == 0; τ = 1.0 - (λ_val * μ_val * ρ_val)
        elseif h == 1 && a == 0; τ = 1.0 + (μ_val * ρ_val)
        elseif h == 0 && a == 1; τ = 1.0 + (λ_val * ρ_val)
        elseif h == 1 && a == 1; τ = 1.0 - ρ_val
        end
        
        ll += log_indep + (τ > 0 ? log(τ) : -Inf)
    end
    return ll
end

# --- 2. Fit the Bivariate Models ---

function fit_bivariate_models(home_data, away_data)
    # Model 1: Independent Poisson
    λ_indep = mean(home_data)
    μ_indep = mean(away_data)
    ll_indep_pois = sum(logpdf.(Poisson(λ_indep), home_data)) + sum(logpdf.(Poisson(μ_indep), away_data))
    
    # Model 2: Dixon-Coles Poisson
    res_dc_pois = optimize(
        p -> -dixon_coles_poisson_loglikelihood(exp(p[1]), exp(p[2]), tanh(p[3]), home_data, away_data), 
        [log(λ_indep), log(μ_indep), 0.0]
    )
    
    # Model 3: Independent Negative Binomial
    # Use moment matching for initial guesses for r_h and r_a
    v_h, v_a = var(home_data), var(away_data)
    r_h_guess = v_h > λ_indep ? λ_indep^2 / (v_h - λ_indep) : 10.0
    r_a_guess = v_a > μ_indep ? μ_indep^2 / (v_a - μ_indep) : 10.0
    
    res_indep_nb = optimize(
        p -> -(sum(logpdf.(MyDistributions.RobustNegativeBinomial(exp(p[3]), exp(p[1])), home_data)) + 
               sum(logpdf.(MyDistributions.RobustNegativeBinomial(exp(p[4]), exp(p[2])), away_data))),
        [log(λ_indep), log(μ_indep), log(r_h_guess), log(r_a_guess)]
    )
    
    # Model 4: Dixon-Coles Negative Binomial
    res_dc_nb = optimize(
        p -> -dixon_coles_nb_loglikelihood(exp(p[1]), exp(p[2]), exp(p[3]), exp(p[4]), tanh(p[5]), home_data, away_data),
        [res_indep_nb.minimizer[1], res_indep_nb.minimizer[2], res_indep_nb.minimizer[3], res_indep_nb.minimizer[4], 0.0]
    )
    
    return (
        indep_pois = (ll = ll_indep_pois, k = 2),
        dc_pois    = (ll = -res_dc_pois.minimum, k = 3),
        indep_nb   = (ll = -res_indep_nb.minimum, k = 4),
        dc_nb      = (ll = -res_dc_nb.minimum, ρ = tanh(res_dc_nb.minimizer[5]), k = 5)
    )
end

# --- 3. Analyze and Print ---

function analyze_bivariate_models(ds::BayesianFootball.Data.DataStore)
    home_data = collect(skipmissing(ds.matches.home_score))
    away_data = collect(skipmissing(ds.matches.away_score))
    
    fits = fit_bivariate_models(home_data, away_data)
    
    println("\n" * "═"^80)
    println(" BIVARIATE MODEL COMPARISON (HOME vs AWAY) ")
    println("═"^80)
    
    @printf("%-16s | %-14s | %-14s | %-14s | %-14s\n", "Metric", "Indep Poisson", "DC Poisson", "Indep NB", "DC NB")
    println("-"^80)
    
    fmt_ll = Printf.Format("%.2f")
    
    aic_indep_pois = 2 * fits.indep_pois.k - 2 * fits.indep_pois.ll
    aic_dc_pois    = 2 * fits.dc_pois.k    - 2 * fits.dc_pois.ll
    aic_indep_nb   = 2 * fits.indep_nb.k   - 2 * fits.indep_nb.ll
    aic_dc_nb      = 2 * fits.dc_nb.k      - 2 * fits.dc_nb.ll
    
    @printf("%-16s | %-14s | %-14s | %-14s | %-14s\n", "Log likelihood", 
        Printf.format(fmt_ll, fits.indep_pois.ll), Printf.format(fmt_ll, fits.dc_pois.ll),
        Printf.format(fmt_ll, fits.indep_nb.ll),   Printf.format(fmt_ll, fits.dc_nb.ll))
        
    @printf("%-16s | %-14s | %-14s | %-14s | %-14s\n", "AIC", 
        Printf.format(fmt_ll, aic_indep_pois), Printf.format(fmt_ll, aic_dc_pois),
        Printf.format(fmt_ll, aic_indep_nb),   Printf.format(fmt_ll, aic_dc_nb))
        
    println("-"^80)
    @printf("DC NB ρ (Dependence): %.4f\n", fits.dc_nb.ρ)
end



# --- weibull add 
using Optim
using Distributions, Statistics, StatsBase, Printf

# --- 1. Log-Likelihood Functions ---

function dixon_coles_nb_loglikelihood(λ_val, μ_val, r_h_val, r_a_val, ρ_val, home_goals, away_goals)
    ll = 0.0
    for (h, a) in zip(home_goals, away_goals)
        log_indep = logpdf(MyDistributions.RobustNegativeBinomial(r_h_val, λ_val), h) + 
                    logpdf(MyDistributions.RobustNegativeBinomial(r_a_val, μ_val), a)
        
        τ = 1.0
        if h == 0 && a == 0; τ = 1.0 - (λ_val * μ_val * ρ_val)
        elseif h == 1 && a == 0; τ = 1.0 + (μ_val * ρ_val)
        elseif h == 0 && a == 1; τ = 1.0 + (λ_val * ρ_val)
        elseif h == 1 && a == 1; τ = 1.0 - ρ_val
        end
        
        ll += log_indep + (τ > 0 ? log(τ) : -Inf)
    end
    return ll
end

function dixon_coles_weibull_loglikelihood(c_h, λ_h, c_a, λ_a, ρ_val, home_goals, away_goals)
    ll = 0.0
    for (h, a) in zip(home_goals, away_goals)
        log_indep = logpdf(MyDistributions.WeibullCount(c_h, λ_h), h) + 
                    logpdf(MyDistributions.WeibullCount(c_a, λ_a), a)
        
        # We use the scale parameters (λ) as proxies for the rate in the DC correction
        τ = 1.0
        if h == 0 && a == 0; τ = 1.0 - (λ_h * λ_a * ρ_val)
        elseif h == 1 && a == 0; τ = 1.0 + (λ_a * ρ_val)
        elseif h == 0 && a == 1; τ = 1.0 + (λ_h * ρ_val)
        elseif h == 1 && a == 1; τ = 1.0 - ρ_val
        end
        
        ll += log_indep + (τ > 0 ? log(τ) : -Inf)
    end
    return ll
end

# --- 2. Fit the Bivariate Models ---

function fit_heavyweight_models(home_data, away_data)
    λ_indep = mean(home_data)
    μ_indep = mean(away_data)
    v_h, v_a = var(home_data), var(away_data)
    
    # 1. Independent Negative Binomial
    r_h_guess = v_h > λ_indep ? λ_indep^2 / (v_h - λ_indep) : 10.0
    r_a_guess = v_a > μ_indep ? μ_indep^2 / (v_a - μ_indep) : 10.0
    
    res_indep_nb = optimize(
        p -> -(sum(logpdf.(MyDistributions.RobustNegativeBinomial(exp(p[3]), exp(p[1])), home_data)) + 
               sum(logpdf.(MyDistributions.RobustNegativeBinomial(exp(p[4]), exp(p[2])), away_data))),
        [log(λ_indep), log(μ_indep), log(r_h_guess), log(r_a_guess)]
    )
    
    # 2. Dixon-Coles Negative Binomial
    res_dc_nb = optimize(
        p -> -dixon_coles_nb_loglikelihood(exp(p[1]), exp(p[2]), exp(p[3]), exp(p[4]), tanh(p[5]), home_data, away_data),
        [res_indep_nb.minimizer[1], res_indep_nb.minimizer[2], res_indep_nb.minimizer[3], res_indep_nb.minimizer[4], 0.0]
    )

    println("[INFO] Optimizing Weibull Models... this may take a moment.")
    
    # 3. Independent Weibull Count
    res_indep_wb = optimize(
        p -> -(sum(logpdf.(MyDistributions.WeibullCount(exp(p[1]), exp(p[2])), home_data)) + 
               sum(logpdf.(MyDistributions.WeibullCount(exp(p[3]), exp(p[4])), away_data))),
        [log(1.0), log(λ_indep), log(1.0), log(μ_indep)]
    )
    
    # 4. Dixon-Coles Weibull Count
    res_dc_wb = optimize(
        p -> -dixon_coles_weibull_loglikelihood(exp(p[1]), exp(p[2]), exp(p[3]), exp(p[4]), tanh(p[5]), home_data, away_data),
        [res_indep_wb.minimizer[1], res_indep_wb.minimizer[2], res_indep_wb.minimizer[3], res_indep_wb.minimizer[4], 0.0]
    )
    
    return (
        indep_nb = (ll = -res_indep_nb.minimum, k = 4),
        dc_nb    = (ll = -res_dc_nb.minimum, ρ = tanh(res_dc_nb.minimizer[5]), k = 5),
        indep_wb = (ll = -res_indep_wb.minimum, k = 4),
        dc_wb    = (ll = -res_dc_wb.minimum, ρ = tanh(res_dc_wb.minimizer[5]), k = 5)
    )
end

# --- 3. Analyze and Print ---

function analyze_heavyweight_models(ds::BayesianFootball.Data.DataStore)
    home_data = collect(skipmissing(ds.matches.home_score))
    away_data = collect(skipmissing(ds.matches.away_score))
    
    fits = fit_heavyweight_models(home_data, away_data)
    
    println("\n" * "═"^80)
    println(" BIVARIATE HEAVYWEIGHTS (HOME vs AWAY) ")
    println("═"^80)
    
    @printf("%-16s | %-14s | %-14s | %-14s | %-14s\n", "Metric", "Indep NB", "DC NB", "Indep Weibull", "DC Weibull")
    println("-"^80)
    
    fmt_ll = Printf.Format("%.2f")
    
    aic_indep_nb = 2 * fits.indep_nb.k - 2 * fits.indep_nb.ll
    aic_dc_nb    = 2 * fits.dc_nb.k    - 2 * fits.dc_nb.ll
    aic_indep_wb = 2 * fits.indep_wb.k - 2 * fits.indep_wb.ll
    aic_dc_wb    = 2 * fits.dc_wb.k    - 2 * fits.dc_wb.ll
    
    @printf("%-16s | %-14s | %-14s | %-14s | %-14s\n", "Log likelihood", 
        Printf.format(fmt_ll, fits.indep_nb.ll), Printf.format(fmt_ll, fits.dc_nb.ll),
        Printf.format(fmt_ll, fits.indep_wb.ll), Printf.format(fmt_ll, fits.dc_wb.ll))
        
    @printf("%-16s | %-14s | %-14s | %-14s | %-14s\n", "AIC", 
        Printf.format(fmt_ll, aic_indep_nb), Printf.format(fmt_ll, aic_dc_nb),
        Printf.format(fmt_ll, aic_indep_wb), Printf.format(fmt_ll, aic_dc_wb))
        
    println("-"^80)
    @printf("DC NB ρ:      %.4f\n", fits.dc_nb.ρ)
    @printf("DC Weibull ρ: %.4f\n", fits.dc_wb.ρ)
end




# ----- team rating 

using DataFrames, Statistics, GLM

# 1. Drop players with missing ratings
valid_lineups = dropmissing(ds.lineups, :rating)

# 2. Sum the ratings for the Home and Away teams for each match
team_ratings = combine(groupby(valid_lineups, [:match_id, :team_side]), :rating => sum => :total_rating)

# 3. Split into Home and Away dataframes
home_ratings = filter(row -> row.team_side == "home", team_ratings)
away_ratings = filter(row -> row.team_side == "away", team_ratings)

rename!(home_ratings, :total_rating => :home_rating)
rename!(away_ratings, :total_rating => :away_rating)
select!(home_ratings, Not(:team_side))
select!(away_ratings, Not(:team_side))

# 4. Join the aggregated ratings back to the main matches table
df = innerjoin(ds.matches, home_ratings, on=:match_id)
df = innerjoin(df, away_ratings, on=:match_id)

# Let's also calculate the difference in ratings (Home - Away)
df.rating_diff = df.home_rating .- df.away_rating


# Did the team with the higher rating win?
df.highest_rated_won = [
    (r.home_rating > r.away_rating && r.winner_code == 1) || 
    (r.away_rating > r.home_rating && r.winner_code == 2) 
    for r in eachrow(df)
]

win_pct = mean(df.highest_rated_won)
println("Team with highest rating won: $(round(win_pct * 100, digits=2))% of matches")


# Create a binary outcome: 1 if Home Win, 0 if Draw or Away Win
df.is_home_win = df.winner_code .== 1

# Fit a Logistic Regression predicting Home Win based on the two rating sums
model = glm(@formula(is_home_win ~ home_rating + away_rating), df, Binomial(), LogitLink())
# Count the total number of matches
total_matches = nrow(df)

# Calculate the percentages
home_win_pct = sum(df.winner_code .== 1) / total_matches
away_win_pct = sum(df.winner_code .== 2) / total_matches
draw_pct     = sum(df.winner_code .== 3) / total_matches

# Print the results clearly
println("Home Win Rate: $(round(home_win_pct * 100, digits=2))%")
println("Draw Rate:     $(round(draw_pct * 100, digits=2))%")
println("Away Win Rate: $(round(away_win_pct * 100, digits=2))%")



# --- 
#=
Phase 1: Building the Player Baseline (No Time Travel!)

To fix the raw ratings, we need to calculate an Exponentially Weighted Moving Average (EWMA) of every player's past performances. Crucially, when calculating the team's strength for a match on a Saturday, we can only use ratings from matches played before that Saturday.
Here is the data engineering blueprint to do this in Julia. We will join the dates, sort chronologically, and calculate the pre-match form for every player.


=#
using DataFrames, Dates, ShiftedArrays

# 1. Join lineups with match dates so we can sort them in time
lineups_with_dates = innerjoin(
    dropmissing(ds.lineups, :rating), 
    select(ds.matches, :match_id, :match_date), 
    on = :match_id
)

# 2. Sort chronologically by date
sort!(lineups_with_dates, :match_date)

# 3. Create a function to calculate an expanding/rolling average WITHOUT the current match
# We use ShiftedArrays.lag to ensure the current match's rating isn't included in the pre-match average
function calc_pre_match_baseline(ratings::AbstractVector)
    n = length(ratings)
    baselines = zeros(Float64, n)
    
    # Track the running sum and count manually to avoid lookahead bias
    running_sum = 0.0
    running_count = 0
    
    for i in 1:n
        if running_count == 0
            # Cold start: No past data. We will set this to missing or a league average later
            baselines[i] = NaN 
        else
            baselines[i] = running_sum / running_count
        end
        # Add the CURRENT match rating to the running tally for the NEXT match
        running_sum += ratings[i]
        running_count += 1
    end
    return baselines
end

# 4. Group by Player and calculate their pre-match baseline
transform!(groupby(lineups_with_dates, :player_id), 
    :rating => calc_pre_match_baseline => :pre_match_rating_baseline
)

# Let's look at the evolution of a single player to verify it worked!
# Grab the player with the most matches in the dataset
top_player_id = first(sort(combine(groupby(lineups_with_dates, :player_id), nrow => :count), :count, rev=true)).player_id
sample_player = filter(row -> row.player_id == top_player_id, lineups_with_dates)

select(sample_player, :match_date, :player_name, :rating, :pre_match_rating_baseline)




#=
The Next Two Fixes: EWMA & Cold Starts

Right now, we are using a Simple Moving Average (SMA). If Evan Caffrey plays 100 games, his 100th baseline will treat his debut 3 years ago with the exact same weight as the match he played last week.

To match the PhD paper (and reality), we need an Exponentially Weighted Moving Average (EWMA). This heavily weights recent form while allowing old history to slowly decay. We also need to fix those NaN values (the "Academy Debut" problem) by filling them with the league-average rating so the model doesn't crash.

Here is the code to apply the EWMA and fill the missing values:
=#


# 1. Calculate the global average rating to fill cold-starts (NaNs)
global_avg_rating = mean(skipmissing(lineups_with_dates.rating))
println("Global Average Rating: ", round(global_avg_rating, digits=3))

# 2. Write an EWMA function (No look-ahead!)
function calc_pre_match_ewma(ratings::AbstractVector; alpha=0.15)
    # alpha = 0.15 means the most recent match accounts for 15% of the new average, 
    # and historical matches account for 85%.
    n = length(ratings)
    baselines = zeros(Float64, n)
    
    # Cold start for match 1
    baselines[1] = NaN 
    
    if n > 1
        # The EWMA going into Match 2 is just the rating from Match 1
        current_ewma = Float64(ratings[1])
        
        for i in 2:n
            baselines[i] = current_ewma
            # Update the EWMA *after* predicting the current match, to be used for the NEXT match
            current_ewma = (alpha * ratings[i]) + ((1.0 - alpha) * current_ewma)
        end
    end
    
    return baselines
end

# 3. Apply the EWMA calculation per player
transform!(groupby(lineups_with_dates, :player_id), 
    :rating => (r -> calc_pre_match_ewma(r, alpha=0.15)) => :pre_match_ewma
)

# 4. Fill the NaNs with the global average
lineups_with_dates.pre_match_ewma = replace(lineups_with_dates.pre_match_ewma, NaN => global_avg_rating)

# Let's look at Evan Caffrey again to see how the EWMA responds faster to his 7.4 spike in May!
sample_player = filter(row -> row.player_id == top_player_id, lineups_with_dates)
select(sample_player, :match_date, :rating, :pre_match_rating_baseline, :pre_match_ewma)




#=
Phase 2: The Positional Roll-Up (Minutes-Weighted)

Now we need to fuse these 11 to 16 individual players back into a Team-Level Match Rating.

If we simply sum the EWMA ratings of everyone who stepped on the pitch, a team that made 5 substitutions would look like they have 16 players' worth of skill. To fix this, we weight every player's contribution by the fraction of the game they played:
Contribution=EWMA×(90Minutes Played​)

Furthermore, the PhD paper noted that attackers and defenders influence the game differently. We are going to calculate the aggregate strength of the Goalkeepers (G), Defenders (D), Midfielders (M), and Forwards (F) separately.

Here is the Julia code to build your final pre-match Team Strength matrix:
=#
# 1. Clean up missing minutes (assume 0 if somehow missing, though usually starters play 90 if missing)
lineups_with_dates.mins = coalesce.(lineups_with_dates.minutes_played, 0)

# 2. Calculate the minute-weighted contribution
# e.g., A 7.0 EWMA player playing 45 minutes contributes 3.5 to the team's total.
lineups_with_dates.weighted_rating = lineups_with_dates.pre_match_ewma .* (lineups_with_dates.mins ./ 90.0)

# 3. Group by match, team, and position, then sum the weighted ratings
position_aggregates = combine(
    groupby(lineups_with_dates, [:match_id, :team_side, :position]),
    :weighted_rating => sum => :line_rating
)

# 4. Unstack (pivot) to get a wide format: one row per team-match, with columns for each position
team_match_strengths = unstack(position_aggregates, [:match_id, :team_side], :position, :line_rating)

# 5. Clean up the column names and handle missing positional lines (fill with 0.0)
rename!(team_match_strengths, 
    names(team_match_strengths) .=> replace.(names(team_match_strengths), "missing" => "Unknown")
)

for col in names(team_match_strengths)
    if col ∉ ["match_id", "team_side"]
        team_match_strengths[!, col] = coalesce.(team_match_strengths[!, col], 0.0)
    end
end

# 6. Let's look at the two teams for a specific match to see their Attack/Defense blocks!
first_match_id = first(team_match_strengths.match_id)
filter(row -> row.match_id == first_match_id, team_match_strengths)





#=
Phase 3: The Player-Driven Likelihood Engine

Now we finally recreate the master equation from Chapter 5 of the PhD paper. We need to construct the Attack (α) and Defense (β) parameters for both teams, but instead of using simple static team averages, the optimizer is going to learn Positional Weights.

For example, it might learn that a Forward's rating heavily drives the Attack score, while a Defender's rating heavily drives the Defense score.

Here is the code to do the final data merge and run the true Player-Based Negative Binomial Model:
=#


using Optim, Distributions, Printf

# ==========================================
# 1. Final Data Merge
# ==========================================
# Separate into Home and Away blocks and rename columns
home_strengths = filter(row -> row.team_side == "home", team_match_strengths)
rename!(home_strengths, :G => :home_G, :D => :home_D, :M => :home_M, :F => :home_F)
select!(home_strengths, Not(:team_side))

away_strengths = filter(row -> row.team_side == "away", team_match_strengths)
rename!(away_strengths, :G => :away_G, :D => :away_D, :M => :away_M, :F => :away_F)
select!(away_strengths, Not(:team_side))

# Merge back onto the main match results
model_df = innerjoin(ds.matches, home_strengths, on=:match_id)
model_df = innerjoin(model_df, away_strengths, on=:match_id)

# Drop any rows that haven't been played yet (missing scores)
dropmissing!(model_df, [:home_score, :away_score])


# ==========================================
# 2. The Log-Likelihood Function
# ==========================================
function player_nb_loglikelihood(params, df)
    # The 11 parameters the optimizer is trying to learn
    w_G_att, w_D_att, w_M_att, w_F_att = params[1], params[2], params[3], params[4]
    w_G_def, w_D_def, w_M_def, w_F_def = params[5], params[6], params[7], params[8]
    home_adv = params[9]
    r_h = exp(params[10]) # Dispersion home (must be positive)
    r_a = exp(params[11]) # Dispersion away (must be positive)

    ll = 0.0
    for r in eachrow(df)
        # Calculate Attack Strengths (Alpha)
        alpha_home = (w_G_att * r.home_G) + (w_D_att * r.home_D) + (w_M_att * r.home_M) + (w_F_att * r.home_F)
        alpha_away = (w_G_att * r.away_G) + (w_D_att * r.away_D) + (w_M_att * r.away_M) + (w_F_att * r.away_F)
        
        # Calculate Defense Strengths (Beta) - lower beta means better defense in Maher models
        beta_home = (w_G_def * r.home_G) + (w_D_def * r.home_D) + (w_M_def * r.home_M) + (w_F_def * r.home_F)
        beta_away = (w_G_def * r.away_G) + (w_D_def * r.away_D) + (w_M_def * r.away_M) + (w_F_def * r.away_F)

        # Calculate Rates (Lambda)
        # Note: We scale down the raw rating sums (which are ~11*6.8=75) by dividing by 11 so 
        # the optimizer doesn't get overwhelmed by massive exponents.
        lambda_home = exp(home_adv + (alpha_home + beta_away) / 11.0)
        lambda_away = exp((alpha_away + beta_home) / 11.0)

        # Accumulate Log-Likelihood using your Robust Negative Binomial
        ll += logpdf(MyDistributions.RobustNegativeBinomial(r_h, lambda_home), r.home_score)
        ll += logpdf(MyDistributions.RobustNegativeBinomial(r_a, lambda_away), r.away_score)
    end
    
    return ll
end

# ==========================================
# 3. Fit the Model
# ==========================================
println("[INFO] Fitting Player-Level Negative Binomial Model. This involves 11 dimensions...")

# Initial Guesses:
# We guess small positive weights for attack, small negative weights for defense (better ratings = fewer goals allowed)
initial_guess = [
    0.01, 0.05, 0.1, 0.2,   # Attacking weights (G, D, M, F)
   -0.2, -0.1, -0.05, -0.01,# Defending weights (G, D, M, F)
    0.3,                    # Home Advantage
    log(10.0), log(10.0)    # Dispersion (r_h, r_a)
]

# Run the optimizer
res_player_model = optimize(p -> -player_nb_loglikelihood(p, model_df), initial_guess, LBFGS(), Optim.Options(iterations=5000))

# ==========================================
# 4. Print the Results
# ==========================================
println("\n" * "═"^50)
println(" PLAYER-BASED MODEL OPTIMIZATION RESULTS ")
println("═"^50)

opt_p = res_player_model.minimizer
@printf("Home Advantage (γ): %.4f\n\n", opt_p[9])

println("--- ATTACKING WEIGHTS (α) ---")
@printf("Goalkeepers:  % .4f\n", opt_p[1])
@printf("Defenders:    % .4f\n", opt_p[2])
@printf("Midfielders:  % .4f\n", opt_p[3])
@printf("Forwards:     % .4f\n\n", opt_p[4])

println("--- DEFENDING WEIGHTS (β) ---")
@printf("Goalkeepers:  % .4f (Negative is good!)\n", opt_p[5])
@printf("Defenders:    % .4f\n", opt_p[6])
@printf("Midfielders:  % .4f\n", opt_p[7])
@printf("Forwards:     % .4f\n\n", opt_p[8])

@printf("Final Log-Likelihood: %.2f\n", -res_player_model.minimum)


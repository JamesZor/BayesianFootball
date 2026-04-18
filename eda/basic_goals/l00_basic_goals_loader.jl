
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

using Revise
using BayesianFootball

using DataFrames
using ThreadPinning
pinthreads(:cores)


using Distributions
using HypothesisTests
using DataFrames


#
include("./l00_basic_goals_loader.jl")


ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())


ds.matches.goals_total =  ds.matches.home_score .+ ds.matches.away_score;
describe(ds.matches[:, [:home_score, :away_score, :goals_total]])


#=
julia> describe(ds.matches[:, [:home_score, :away_score, :goals_total]])
3×7 DataFrame
 Row │ variable     mean     min    median   max    nmissing  eltype                
     │ Symbol       Float64  Int32  Float64  Int32  Int64     Type                  
─────┼──────────────────────────────────────────────────────────────────────────────
   1 │ home_score   1.3914       0      1.0      7         0  Union{Missing, Int32}
   2 │ away_score   1.0829       0      1.0      7         0  Union{Missing, Int32}
   3 │ goals_total  2.47429      0      2.0      9         0  Int32
=#

home_goals = collect(skipmissing(ds.matches.home_score));
away_goals = collect(skipmissing(ds.matches.away_score));
total_goals = collect(skipmissing(ds.matches.goals_total));

goals = Dict{String, AbstractVector{<:Integer}}(
        "home" => collect(skipmissing(ds.matches.home_score)),
        "away" => collect(skipmissing(ds.matches.away_score)),
        "total"=> collect(skipmissing(ds.matches.goals_total)),
)

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

simple_describe(goals)

#=
home ::  Mean:1.391395592864638 | std:1.2277470672736481 | n:953
away ::  Mean:1.0828961175236096 | std:1.0199950602922667 | n:953
tota ::  Mean:2.474291710388248 | std:1.527623843782551 | n:953
=#


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
    func(params) = -sum(logpdf(RobustNegativeBinomial(exp(params[1]), exp(params[2])), data))
    
    res = optimize(func, [log(r_guess), log(m)])
    
    return RobustNegativeBinomial(exp(res.minimizer[1]), exp(res.minimizer[2]))
end

function fit_goal_distributions(data::AbstractVector{<:Integer})
    # 1. Fit Poisson (standard MLE)
    p_dist = fit(Poisson, data)
    
    # 2. Fit Robust Negative Binomial (Method of Moments)
    μ = mean(data)
    σ2 = var(data)
    
    if σ2 > μ
        # Solving for r using the variance formula: σ² = μ + μ²/r
        r = μ^2 / (σ2 - μ)
        nb_dist = RobustNegativeBinomial(r, μ)
    else
        # If variance <= mean, NB is not the right tool
        nb_dist = nothing 
    end
    
    return (poisson = p_dist, nb = nb_dist)
end

function compute_metrics(dist, data::AbstractVector{<:Integer})
    isnothing(dist) && return nothing

    # Log-Likelihood
    ll = loglikelihood(dist, data)
    
    # AIC: 2k - 2LL (k is fetched automatically via nparams)
  k = length(params(dist))
    aic = 2k - 2ll
    
    # Chi-Squared Goodness of Fit
    # Bins 0-6, then a "7+" catch-all
    obs_counts = counts(data, 0:7)
    n = length(data)
    
    expected = [pdf(dist, i) * n for i in 0:6]
    push!(expected, (1 - cdf(dist, 6)) * n) 
    
    # Safeguard against 0 expected values to avoid Inf in Chi-Sq
    expected_safe = max.(expected, 1e-6)
    chi_sq = sum((obs_counts .- expected_safe).^2 ./ expected_safe)
    
    return (log_likelihood = ll, aic = aic, chi_sq = chi_sq)
end


function analyze_goal_models(goals_dict::Dict{String, <:AbstractVector{<:Integer}})
    for (label, data) in goals_dict
        println("\n" * "═"^45)
        println(" MODEL COMPARISON: $(uppercase(label)) ")
        println("═"^45)
        
        fits = fit_goal_distributions(data)
        
        # Calculate stats
        p_stats = compute_metrics(fits.poisson, data)
        nb_stats = compute_metrics(fits.nb, data)
        
        # Table Header
        @printf("%-18s | %-12s | %-12s\n", "Metric", "Poisson", "Robust NB")
        println("-"^48)
        
        # Display Results
        for metric in [:log_likelihood, :aic, :chi_sq]
            p_val = getproperty(p_stats, metric)
            nb_val = isnothing(nb_stats) ? "N/A" : @sprintf("%.2f", getproperty(nb_stats, metric))
            
            @printf("%-18s | %-12.2f | %-12s\n", 
                    uppercasefirst(replace(string(metric), "_" => " ")), 
                    p_val, nb_val)
        end
        
        if isnothing(nb_stats)
            println("\n[!] RobustNB skipped: No overdispersion detected.")
        end
    end
end


analyze_goal_models(goals)

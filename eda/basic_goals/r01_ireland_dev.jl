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
using Distributions, Statistics, StatsBase, Printf

function fit_goal_distributions(data::AbstractVector{<:Integer})
    # Fit Poisson (MLE)
    p_dist = fit(Poisson, data)
    
    # Fit Negative Binomial (Method of Moments)
    μ = mean(data)
    σ2 = var(data)
    
    if σ2 > μ
        p = μ / σ2
        r = μ^2 / (σ2 - μ)
        nb_dist = NegativeBinomial(r, p)
    else
        # Fallback if no overdispersion: NB is technically not applicable
        # We'll return nothing or the Poisson equivalent
        nb_dist = nothing 
    end
    
    return (poisson = p_dist, nb = nb_dist)
end

function compute_metrics(dist, data::AbstractVector{<:Integer}, k_params::Int)
    if isnothing(dist) return nothing end

    # 1. Log-Likelihood
    ll = loglikelihood(dist, data)
    
    # 2. AIC
    aic = 2*k_params - 2*ll
    
    # 3. Chi-Squared Score
    # We'll check bins 0 through 6, then 7+
    obs_counts = counts(data, 0:7)
    n = length(data)
    
    expected = [pdf(dist, i) * n for i in 0:6]
    push!(expected, (1 - cdf(dist, 6)) * n) # The "7+" bin
    
    # Chi-square formula: sum( (O-E)^2 / E )
    chi_sq = sum((obs_counts .- expected).^2 ./ expected)
    
    return (log_likelihood = ll, aic = aic, chi_sq = chi_sq)
end

function analyze_goal_models(goals_dict::Dict{String, <:AbstractVector{<:Integer}})
    results = Dict()

    for (label, data) in goals_dict
        println("\n" * "="^40)
        println(" ANALYSIS: $(uppercase(label)) GOALS ")
        println("="^40)
        
        fits = fit_goal_distributions(data)
        
        # Calculate stats (Poisson has 1 param, NB has 2)
        p_stats = compute_metrics(fits.poisson, data, 1)
        nb_stats = compute_metrics(fits.nb, data, 2)
        
        results[label] = (poisson=p_stats, nb=nb_stats)
        
        # Pretty Print
        @printf("%-15s | %-12s | %-12s\n", "Metric", "Poisson", "NegBinom")
        println("-"^45)
        @printf("%-15s | %-12.2f | %-12.2f\n", "Log-Likelihood", p_stats.log_likelihood, nb_stats.log_likelihood)
        @printf("%-15s | %-12.2f | %-12.2f\n", "AIC", p_stats.aic, nb_stats.aic)
        @printf("%-15s | %-12.2f | %-12.2f\n", "Chi-Squared", p_stats.chi_sq, nb_stats.chi_sq)
    end
    
    return results
end

analyze_goal_models(goals)

using DataFrames
using Statistics
using Distributions
using HypothesisTests
using GLM
using Printf
using Optim
using Dates
using StatsBase
using BayesianFootball

"""
    fit_nb_mle(data)

Fits a Negative Binomial distribution (using RobustNegativeBinomial from the project) via MLE.
"""
function fit_nb_mle(data)
    m = mean(data)
    v = var(data)
    # Start with MoM estimates
    r_guess = v > m ? m^2 / (v - m) : 10.0
    
    func(params) = -sum(logpdf(MyDistributions.RobustNegativeBinomial(exp(params[1]), exp(params[2])), data))
    
    res = optimize(func, [log(r_guess), log(m)])
    return MyDistributions.RobustNegativeBinomial(exp(res.minimizer[1]), exp(res.minimizer[2]))
end

function fit_mle(::Type{MyDistributions.WeibullCount}, data::AbstractVector{<:Integer})
    # Start with Poisson assumptions as a baseline guess.
    # The Poisson model is a special case of Weibull-count with shape parameter c = 1.
    c_guess = 1.0
    λ_guess = max(mean(data), 1e-4) # Prevent log(0) issues
    
    # We optimize in the log-space to enforce strict positivity for c and λ
    function objective(params)
        c_val = exp(params[1])
        λ_val = exp(params[2])
        dist = MyDistributions.WeibullCount(c_val, λ_val)
        
        # Calculate negative log-likelihood safely
        nll = 0.0
        for x in data
            # Use a tiny floor to prevent log(0) = -Inf which breaks Optim
            p = max(pdf(dist, x), 1e-15) 
            nll -= log(p)
        end
        return nll
    end
    
    # Run the optimization using Nelder-Mead (gradient-free, robust for series approximations)
    res = optimize(objective, [log(c_guess), log(λ_guess)], NelderMead(), Optim.Options(iterations=2000, g_tol=1e-6))
    
    if !Optim.converged(res)
        @warn "MLE optimization did not fully converge. Results may be approximate."
    end
    
    best_c = exp(res.minimizer[1])
    best_λ = exp(res.minimizer[2])
    
    return MyDistributions.WeibullCount(best_c, best_λ), -res.minimum
end

"""
    test_overdispersion(goals::Vector{<:Integer}, label::String)

Calculates Mean, Variance, and Index of Dispersion (V/M).
If V/M > 1, the data is overdispersed, justifying Negative Binomial.
"""
function test_overdispersion(goals::Vector{<:Integer}, label::String)
    m = mean(goals)
    v = var(goals)
    dispersion_index = v / m
    
    println("\n" * "═"^50)
    println(" DISTRIBUTION TEST: $(uppercase(label)) ")
    println("═"^50)
    @printf("Mean: %.4f | Variance: %.4f | Index of Dispersion: %.4f\n", m, v, dispersion_index)
    
    # 1. Fit Poisson
    p_dist = fit(Poisson, goals)
    ll_p = loglikelihood(p_dist, goals)
    aic_p = 2 * 1 - 2 * ll_p
    
    # 2. Fit Negative Binomial
    nb_dist = fit_nb_mle(goals)
    ll_nb = loglikelihood(nb_dist, goals)
    aic_nb = 2 * 2 - 2 * ll_nb
    
    @printf("%-15s | %-12s | %-12s\n", "Metric", "Poisson", "NegBinomial")
    println("-"^45)
    @printf("%-15s | %-12.2f | %-12.2f\n", "Log-Likelihood", ll_p, ll_nb)
    @printf("%-15s | %-12.2f | %-12.2f\n", "AIC", aic_p, aic_nb)
    
    if aic_nb < aic_p
        println("\nResult: Overdispersion detected (AIC_NB < AIC_P). Justifies Negative Binomial.")
    else
        println("\nResult: Poisson is sufficient (AIC_P <= AIC_NB).")
    end
    
    return (mean=m, var=v, di=dispersion_index, aic_p=aic_p, aic_nb=aic_nb)
end

"""
    test_home_advantage_mean(df::DataFrame)

Tests if Home Teams score significantly more goals than Away Teams.
Uses Mann-Whitney U test (non-parametric).
"""
function test_home_advantage_mean(df::DataFrame)
    home_goals = collect(skipmissing(df.home_score))
    away_goals = collect(skipmissing(df.away_score))
    
    m_h = mean(home_goals)
    m_a = mean(away_goals)
    diff = m_h - m_a
    
    println("\n--- Home Advantage Test (Mean) ---")
    @printf("Home Mean: %.4f | Away Mean: %.4f | Difference: %.4f\n", m_h, m_a, diff)
    
    # Mann-Whitney U Test
    mwu = MannWhitneyUTest(home_goals, away_goals)
    p_val = pvalue(mwu)
    
    @printf("Mann-Whitney U p-value: %.4e\n", p_val)
    if p_val < 0.05
        println("Result: Statistically significant home advantage on goals.")
    else
        println("Result: No statistically significant home advantage found.")
    end
end

"""
    test_home_advantage_variance(df::DataFrame)

Tests if playing at home/away affects the predictability (variance) of goals.
Uses Levene's Test for equality of variances.
"""
function test_home_advantage_variance(df::DataFrame)
    home_goals = collect(skipmissing(df.home_score))
    away_goals = collect(skipmissing(df.away_score))
    
    v_h = var(home_goals)
    v_a = var(away_goals)
    
    println("\n--- Home Advantage Test (Variance/Chaos) ---")
    @printf("Home Variance: %.4f | Away Variance: %.4f | Ratio: %.4f\n", v_h, v_a, v_h / v_a)
    
    # Variance Test (F-test or Levene's)
    # Levene's is more robust to non-normality. HypothesisTests.jl has VarianceTest (F-test)
    vt = VarianceFTest(home_goals, away_goals)
    p_val = pvalue(vt)
    
    @printf("F-test (Variance) p-value: %.4e\n", p_val)
    if p_val < 0.05
        println("Result: Statistically significant difference in variance (Home vs Away).")
    else
        println("Result: No significant difference in variance.")
    end
end

"""
    test_team_volatility(df::DataFrame)

Calculates Mean and Variance of Goals Conceded per team.
Identifies teams with high 'Index of Dispersion' (Chaos).
"""
function test_team_volatility(df::DataFrame)
    # Home team conceded is away score, Away team conceded is home score
    home_conceded = select(df, :home_team => :team_id, :away_score => :conceded)
    away_conceded = select(df, :away_team => :team_id, :home_score => :conceded)
    
    all_conceded = vcat(home_conceded, away_conceded)
    dropmissing!(all_conceded)
    
    team_stats = combine(groupby(all_conceded, :team_id),
        :conceded => mean => :mean_conceded,
        :conceded => var => :var_conceded,
        nrow => :n
    )
    
    team_stats.dispersion_index = team_stats.var_conceded ./ team_stats.mean_conceded
    
    sort!(team_stats, :dispersion_index, rev=true)
    
    println("\n--- Team-Specific Volatility (Goals Conceded) ---")
    println(first(team_stats, 5)) # Top 5 most chaotic teams
    
    avg_di = mean(team_stats.dispersion_index)
    @printf("\nAverage Team-Level Dispersion Index: %.4f\n", avg_di)
    
    if any(team_stats.dispersion_index .> 1.3)
        println("Result: Significant team-level volatility detected (Some DI > 1.3).")
    end
    
    return team_stats
end

"""
    test_match_level_chaos(df::DataFrame)

Groups matches by tiers and checks if 'Top vs Bottom' matches have higher variance.
"""
function test_match_level_chaos(df::DataFrame)
    # Simple tiering based on win rate in this dataset
    home_wins = combine(groupby(df, :home_team), :winner_code => (x -> sum(x .== 1)/length(x)) => :win_rate)
    away_wins = combine(groupby(df, :away_team), :winner_code => (x -> sum(x .== 2)/length(x)) => :win_rate)
    
    team_performance = innerjoin(home_wins, away_wins, on=:home_team => :away_team, makeunique=true)
    team_performance.avg_win_rate = (team_performance.win_rate .+ team_performance.win_rate_1) ./ 2
    
    # Assign tiers
    q = quantile(team_performance.avg_win_rate, [0.33, 0.66])
    team_performance.tier = [
        r < q[1] ? "Bottom" : (r < q[2] ? "Mid" : "Top")
        for r in team_performance.avg_win_rate
    ]
    
    tier_map = Dict(team_performance.home_team .=> team_performance.tier)
    
    df.total_goals = df.home_score .+ df.away_score
    df.home_tier = [get(tier_map, id, "Unknown") for id in df.home_team]
    df.away_tier = [get(tier_map, id, "Unknown") for id in df.away_team]
    df.matchup = df.home_tier .* " vs " .* df.away_tier
    
    matchup_stats = combine(groupby(df, :matchup),
        :total_goals => var => :variance_total_goals,
        :total_goals => mean => :mean_total_goals,
        nrow => :n
    )
    
    filter!(r -> r.n > 10, matchup_stats)
    sort!(matchup_stats, :variance_total_goals, rev=true)
    
    println("\n--- Match-Level Chaos (Tiered Matchups) ---")
    println(matchup_stats)
    
    return matchup_stats
end

"""
    test_temporal_stability(df::DataFrame)

Analyzes how Goal Mean and Goal Variance shift by month.
Validates 'r.month' (monthly dispersion) and seasonal intercepts.
"""
function test_temporal_stability(df::DataFrame)
    df_temp = copy(df)
    dropmissing!(df_temp, [:home_score, :away_score, :match_date])
    
    df_temp.month = month.(df_temp.match_date)
    df_temp.total_goals = df_temp.home_score .+ df_temp.away_score
    
    monthly_stats = combine(groupby(df_temp, :month),
        :total_goals => mean => :mean_goals,
        :total_goals => var => :var_goals,
        nrow => :n
    )
    sort!(monthly_stats, :month)
    
    println("\n--- Temporal Stability: Monthly Goal Statistics ---")
    println(monthly_stats)
    
    # Kruskal-Wallis for Mean Drift
    groups = [df_temp[df_temp.month .== m, :total_goals] for m in sort(unique(df_temp.month))]
    kw = KruskalWallisTest(groups...)
    
    println("\nKruskal-Wallis (Mean Drift) p-value: ", pvalue(kw))
    if pvalue(kw) < 0.05
        println("Result: Significant seasonal drift in scoring mean detected.")
    else
        println("Result: Scoring mean is relatively stable across months.")
    end
    
    # Check for Variance Spike (Chaos)
    max_var = maximum(monthly_stats.var_goals)
    min_var = minimum(monthly_stats.var_goals)
    println("Variance Range: Max $(round(max_var, digits=3)) vs Min $(round(min_var, digits=3)) (Ratio: $(round(max_var/min_var, digits=2)))")
    
    if max_var / min_var > 1.3
        println("Result: Significant monthly heteroscedasticity. Justifies 'r.month' parameter.")
    end
    
    return monthly_stats
end

"""
    test_form_autocorrelation(df::DataFrame)

Calculates the Autocorrelation Function (ACF) of team Goal Difference.
Helps validate the time-decay half-life.
"""
function test_form_autocorrelation(df::DataFrame)
    # 1. Create long format of results
    home_res = select(df, :match_date, :home_team => :team, [:home_score, :away_score] => ((h, a) -> h .- a) => :gd)
    away_res = select(df, :match_date, :away_team => :team, [:away_score, :home_score] => ((a, h) -> a .- h) => :gd)
    
    all_res = vcat(home_res, away_res)
    dropmissing!(all_res)
    sort!(all_res, :match_date)
    
    # 2. Calculate ACF per team
    team_acfs = []
    for (team_id, sdf) in pairs(groupby(all_res, :team))
        if nrow(sdf) > 20
            # Calculate autocorrelation up to lag 15
            push!(team_acfs, autocor(sdf.gd, 0:15))
        end
    end
    
    if isempty(team_acfs)
        println("\n--- Form Autocorrelation: Insufficient Data ---")
        return nothing
    end
    
    avg_acf = mean(team_acfs)
    
    println("\n--- Form Autocorrelation (Goal Difference) ---")
    @printf("%-8s | %-10s\n", "Lag", "Avg ACF")
    println("-"^22)
    for i in 2:length(avg_acf) # Skip Lag 0 (always 1.0)
        @printf("%-8d | %-10.4f\n", i-1, avg_acf[i])
    end
    
    # Heuristic: Where does it drop below 0.1?
    half_life_lag = findfirst(x -> x < 0.1, avg_acf)
    if isnothing(half_life_lag)
        println("\nResult: Form is highly persistent (ACF stays > 0.1 for 15+ matches).")
    else
        lag = half_life_lag - 1
        println("\nResult: Form decays below 0.1 after approximately $lag matches.")
        println("Note: If $lag is small (e.g. < 5), form is transient. If large (e.g. > 10), form is stable.")
    end
    
    return avg_acf
end

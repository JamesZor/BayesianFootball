#dev/37_bivariatepoisson_mydist.jl
#
#
using BayesianFootball
using Distributions
using Random


# BayesianFootball.MyDistributions.BivariateLogPoisson()
using BayesianFootball.MyDistributions



# ==========================================
# 1. Setup Test Parameters
# ==========================================
# Let's pick simple rates to verify math easily
λ1_true = 2.0
λ2_true = 1.0
λ3_true = 0.5 # Correlation factor

# Convert to log-rates (Theta) which the distribution expects
θ1 = log(λ1_true)
θ2 = log(λ2_true)
θ3 = log(λ3_true)

println("--- 1. Instantiation Check ---")
dist = BivariateLogPoisson(θ1, θ2, θ3)
println("Distribution created: ", dist)
println("Expected Means: X=$(λ1_true + λ3_true), Y=$(λ2_true + λ3_true)")
println("Expected Covariance: $(λ3_true)\n")


# ==========================================
# 2. Sampling Test (Monte Carlo)
# ==========================================
println("--- 2. Sampling Test (N=100,000) ---")
N = 100_000
samples = [rand(dist) for _ in 1:N]

# Extract X and Y vectors
xs = [s[1] for s in samples]
ys = [s[2] for s in samples]

mean_x = mean(xs)
mean_y = mean(ys)
cov_xy = cov(xs, ys)

println("Sample Mean X:  ", round(mean_x, digits=4), " \t(Error: $(round(mean_x - (λ1_true+λ3_true), digits=4)))")
println("Sample Mean Y:  ", round(mean_y, digits=4), " \t(Error: $(round(mean_y - (λ2_true+λ3_true), digits=4)))")
println("Sample Covariance: ", round(cov_xy, digits=4), " \t(Error: $(round(cov_xy - λ3_true, digits=4)))")

if abs(cov_xy - λ3_true) < 0.05
    printstyled("✔ Statistics match theoretical values.\n", color=:green)
else
    printstyled("✖ Statistics divergence! Check rand() logic.\n", color=:red)
end
println()


# ==========================================
# 3. Probability Density Check (Normalization)
# ==========================================

function t3()
    println("--- 3. LogPDF Normalization Check ---")
    # The sum of P(x,y) over all x,y should be 1.0.
    # Since range is infinite, we sum over a large grid (0-20) covering 99.99% of mass.

    prob_sum = 0.0
    for x in 0:20
        for y in 0:20
            # Calculate logpdf and exponentiate back to probability
            lp = logpdf(dist, [x, y])
            prob_sum += exp(lp)
        end
    end

    println("Sum of probabilities over 20x20 grid: ", prob_sum)

    if isapprox(prob_sum, 1.0; atol=0.001)
        printstyled("✔ Distribution is properly normalized (Sums to ~1.0).\n", color=:green)
    else
        printstyled("✖ Distribution is NOT normalized! Check logpdf math.\n", color=:red)
    end
end 
t3()


# ==========================================
# 4. Edge Case Check
# ==========================================
function t4()
    println("\n--- 4. Edge Case Check ---")
    try
        l_zero = logpdf(dist, [0, 0])
        println("logpdf([0,0]) = ", l_zero)
        
        l_neg = logpdf(dist, [-1, 5])
        println("logpdf([-1,5]) = ", l_neg, " (Should be -Inf)")
        
        if l_neg == -Inf
             printstyled("✔ Negative inputs handled correctly.\n", color=:green)
        end
    catch e
        printstyled("✖ Crash on edge cases: $e\n", color=:red)
    end
end

t4()


#####


using DataFrames
using Statistics
using ThreadPinning
using LinearAlgebra
pinthreads(:cores)


data_store = BayesianFootball.Data.load_default_datastore()

ds = BayesianFootball.Data.DataStore( 
    Data.add_match_week_column(data_store.matches),
    data_store.odds,
    data_store.incidents
)


using BayesianFootball
using DataFrames, Statistics, Distributions, StatsPlots
using Random

# 1. Get the Data & Clean Missing Values
# ---------------------------------------------------------
# Assuming 'ds' is already loaded
df = ds.matches

# Filter rows where either score is missing
df_clean = dropmissing(df, [:home_score, :away_score])

home_goals = Vector{Int}(df_clean.home_score)
away_goals = Vector{Int}(df_clean.away_score)

println("Checking fit on $(length(home_goals)) valid matches...")

# 2. Estimate Parameters (Method of Moments)
# ---------------------------------------------------------
# Mean(Home) = λ1 + λ3
# Mean(Away) = λ2 + λ3
# Cov(Home, Away) = λ3

μ_h = mean(home_goals)
μ_a = mean(away_goals)
cov_goals = cov(home_goals, away_goals)

println("\n--- Empirical Statistics ---")
println("Mean Home Goals: ", round(μ_h, digits=4))
println("Mean Away Goals: ", round(μ_a, digits=4))
println("Covariance:      ", round(cov_goals, digits=4))

# Calculate Lambdas
# Clamp covariance to be tiny positive if data shows negative covariance (rare but possible)
λ3_est = max(0.001, cov_goals) 
λ1_est = max(0.001, μ_h - λ3_est)
λ2_est = max(0.001, μ_a - λ3_est)

println("\n--- Estimated Parameters ---")
println("λ1 (Home Indep): ", round(λ1_est, digits=4))
println("λ2 (Away Indep): ", round(λ2_est, digits=4))
println("λ3 (Common):     ", round(λ3_est, digits=4))

# Convert to Log-Rates (Theta) for your custom distribution
θ1 = log(λ1_est)
θ2 = log(λ2_est)
θ3 = log(λ3_est)

# 3. Instantiate Your Custom Distribution
# ---------------------------------------------------------
# Use the module path where you defined it. 
# If you just included the file, it might be in Main.BivariateLogPoisson
dist = BayesianFootball.MyDistributions.BivariateLogPoisson(θ1, θ2, θ3)

# 4. Generate "Fake" Data to Compare
# ---------------------------------------------------------
simulated_matches = [rand(dist) for _ in 1:length(home_goals)]

sim_home = [x[1] for x in simulated_matches]
sim_away = [x[2] for x in simulated_matches]

# 5. Visual Comparison
# ---------------------------------------------------------

# Plot A: Home Goals Distribution
p1 = histogram(home_goals, normalize=:pdf, label="Actual Data", color=:black, alpha=0.4, bins=-0.5:1:8.5)
histogram!(p1, sim_home, normalize=:pdf, label="Bivariate Dist", color=:blue, alpha=0.4, bins=-0.5:1:8.5)
title!(p1, "Home Goals Fit")

# Plot B: Away Goals Distribution
p2 = histogram(away_goals, normalize=:pdf, label="Actual Data", color=:black, alpha=0.4, bins=-0.5:1:8.5)
histogram!(p2, sim_away, normalize=:pdf, label="Bivariate Dist", color=:blue, alpha=0.4, bins=-0.5:1:8.5)
title!(p2, "Away Goals Fit")

# Plot C & D: Heatmaps (The Real Test)
max_g = 5
heatmap_actual = zeros(max_g+1, max_g+1)
heatmap_sim = zeros(max_g+1, max_g+1)

for i in 1:length(home_goals)
    h, a = home_goals[i], away_goals[i]
    if h <= max_g && a <= max_g
        heatmap_actual[a+1, h+1] += 1
    end
end

for i in 1:length(sim_home)
    h, a = sim_home[i], sim_away[i]
    if h <= max_g && a <= max_g
        heatmap_sim[a+1, h+1] += 1
    end
end

# Normalize heatmaps to frequencies
heatmap_actual ./= sum(heatmap_actual)
heatmap_sim ./= sum(heatmap_sim)

p3 = heatmap(0:max_g, 0:max_g, heatmap_actual, title="Actual Score Prob", xlabel="Home", ylabel="Away", c=:viridis)
p4 = heatmap(0:max_g, 0:max_g, heatmap_sim, title="Simulated Score Prob", xlabel="Home", ylabel="Away", c=:viridis)

# Display Grid
plot(p1, p2, p3, p4, layout=(2,2), size=(900, 800))

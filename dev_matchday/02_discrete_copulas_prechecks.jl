
# ----------------------------------------------
# 1. The set up 
# ----------------------------------------------

using Revise
using BayesianFootball

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


# 1. The "Heatmap" (Contingency Table)
# 2. Rank Correlations (Kendall's τ)

using DataFrames, Statistics, StatsBase, FreqTables, Plots

df_played = dropmissing(ds.matches, [:home_score, :away_score])

# 1. Basic Setup
h_scores = Int.(df_played.home_score);
a_scores = Int.(df_played.away_score);

# --- A. Summary Statistics ---
# Pearson is often close to 0, usually slightly negative (as one team scores, they might defend, or kill the game)
p_cor = cor(h_scores, a_scores) 
# Kendall is more robust for discrete data
k_tau = corkendall(h_scores, a_scores) 

println("Pearson Correlation: $(round(p_cor, digits=4))")
println("Kendall's Tau:       $(round(k_tau, digits=4))")

#=
julia> println("Pearson Correlation: $(round(p_cor, digits=4))")
Pearson Correlation: -0.1332

julia> println("Kendall's Tau:       $(round(k_tau, digits=4))")
Kendall's Tau:       -0.0934
=#

# --- B. The Contingency Matrix (Heatmap) ---
# This is the "Truth" of your joint distribution
max_goals = 6 # Cap at 6 for readability
safe_h = min.(h_scores, max_goals)
safe_a = min.(a_scores, max_goals)

# Create a frequency table (Joint Distribution)
joint_counts = freqtable(safe_h, safe_a)
joint_prob = joint_counts ./ sum(joint_counts)

# --- C. Compare vs Independence (The "Copula Signal") ---
# Calculate Marginal Probabilities
marg_h = vec(sum(joint_prob, dims=2)) # Sum rows
marg_a = vec(sum(joint_prob, dims=1)) # Sum cols

# Calculate Expected Probability if they were Independent (P(H)*P(A))
expected_prob = marg_h * marg_a'

# The "Dependence Ratio": Observed / Expected
# > 1.0 means Positive Dependence (Happens more than expected)
# < 1.0 means Negative Dependence (Happens less than expected)
dependence_ratio = joint_prob ./ expected_prob

# --- D. Plotting ---
heatmap(
    0:max_goals, 0:max_goals, dependence_ratio', 
    title="Dependency Ratio (Observed / Independent)",
    xlabel="Home Goals", ylabel="Away Goals",
    color=:diverging_bwr_20_95_c54_n256, # Blue=Low, Red=High
    clims=(0.5, 1.5), # Center color scale on 1.0 (Independence)
    yflip=false
)


#=
julia> dependence_ratio = joint_prob ./ expected_prob
7×7 Named Matrix{Float64}
Dim1 ╲ Dim2 │        0         1         2         3         4         5         6
────────────┼─────────────────────────────────────────────────────────────────────
0           │ 0.827149  0.955234   1.00829   1.36203    1.3598   1.68461   2.43536
1           │ 0.920805   1.02576   1.04695  0.997268   1.17282  0.994135   1.09499
2           │  1.00496   1.04203   1.04161  0.904578  0.806739  0.794001  0.178886
3           │  1.14513   1.02681  0.958394  0.825668  0.585315  0.471333       0.0
4           │   1.5421  0.903191  0.703088  0.534965  0.591609  0.397001       0.0
5           │  1.66339  0.886769   0.83329  0.213986       0.0       0.0       0.0
6           │   2.0814  0.699973  0.504531  0.207299       0.0       0.0       0.0



=# 

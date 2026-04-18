using Revise
using BayesianFootball

using DataFrames
using ThreadPinning
pinthreads(:cores)


using Distributions
using HypothesisTests


#
include("./l00_basic_goals_loader.jl")


ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())


goals = get_goals(ds)

simple_describe(goals)

#=
home ::  Mean:1.391395592864638 | std:1.2277470672736481 | n:953
away ::  Mean:1.0828961175236096 | std:1.0199950602922667 | n:953
tota ::  Mean:2.474291710388248 | std:1.527623843782551 | n:953
=#


# ==========================================
# 2. Fit the Distributions
# ==========================================

analyze_goal_models(goals)


#=
julia> analyze_goal_models(goals)

════════════════════════════════════════════════
 MODEL COMPARISON: HOME 
════════════════════════════════════════════════
Metric             | Poisson      | Robust NB   
------------------------------------------------
Log likelihood     | -1443.32     | -1441.92    
AIC                | 2888.65      | 2885.83     
Chi sq             | 22.13        | 10.09       
Degrees of freedom | 6            | 6           
P-value            | 0.0011       | 0.1207      

════════════════════════════════════════════════
 MODEL COMPARISON: TOTAL 
════════════════════════════════════════════════
Metric             | Poisson      | Robust NB   
------------------------------------------------
Log likelihood     | -1719.99     | -1719.99    
AIC                | 3441.99      | 3441.99     
Chi sq             | 8.19         | 8.19        
Degrees of freedom | 6            | 6           
P-value            | 0.2242       | 0.2242      

════════════════════════════════════════════════
 MODEL COMPARISON: AWAY 
════════════════════════════════════════════════
Metric             | Poisson      | Robust NB   
------------------------------------------------
Log likelihood     | -1274.13     | -1274.13    
AIC                | 2550.26      | 2550.26     
Chi sq             | 12.06        | 12.07       
Degrees of freedom | 6            | 6           
P-value            | 0.0606       | 0.0605
=#



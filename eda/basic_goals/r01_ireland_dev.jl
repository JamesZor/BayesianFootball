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
ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.ScottishLower())
ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Norway())
ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.SouthKorea())


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

analyze_bivariate_models(ds)



#=
════════════════════════════════════════════════════════════
 BIVARIATE MODEL COMPARISON (HOME vs AWAY) 
════════════════════════════════════════════════════════════
Metric               | Indep. Poisson  | Dixon-Coles    
------------------------------------------------------------
λ (Home Rate)        | 1.7249          | 1.7249         
μ (Away Rate)        | 1.3851          | 1.3851         
ρ (Dependence)       | N/A             | -0.0215        
------------------------------------------------------------
Log likelihood       | -7736.57        | -7736.17       
AIC                  | 15477.14        | 15478.34       

[RESULT] Independent Poisson wins. Correlation (ρ) did not justify the extra parameter.
=#



#=
julia> analyze_bivariate_models(ds)

════════════════════════════════════════════════════════════════════════════════
 BIVARIATE MODEL COMPARISON (HOME vs AWAY) 
════════════════════════════════════════════════════════════════════════════════
Metric           | Indep Poisson  | DC Poisson     | Indep NB       | DC NB         
--------------------------------------------------------------------------------
Log likelihood   | -7736.57       | -7736.17       | -7728.50       | -7728.10      
AIC              | 15477.14       | 15478.34       | 15465.00       | 15466.21      
--------------------------------------------------------------------------------
DC NB ρ (Dependence): -0.0215
=#


analyze_heavyweight_models(ds)


ds
ds.lineups
names(ds.lineups)
ds.matches

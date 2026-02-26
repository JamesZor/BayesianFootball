using Revise
using BayesianFootball
using DataFrames

using Statistics
using ThreadPinning
using LinearAlgebra
pinthreads(:cores)


ds = Data.load_extra_ds()
transform!(ds.matches, :match_week => ByRow(w -> cld(w, 4)) => :match_month)


cv_config = BayesianFootball.Data.GroupedCVConfig(
    tournament_groups = [[56, 57]],
    target_seasons = ["24/25"],
    history_seasons = 2,
    dynamics_col = :match_month,
    warmup_period = 9,
    stop_early = true
)



# -------------------------------------
#  MS Gamma
# -------------------------------------
model = BayesianFootball.Models.PreGame.MSNegativeBinomialGamma()

splits = BayesianFootball.Data.create_data_splits(ds, cv_config)

feature_sets = BayesianFootball.Features.create_features(
    splits, model, cv_config
)

splits = BayesianFootball.Data.create_data_splits(ds, cv_config) 
train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=2)  

sampler_conf = Samplers.NUTSConfig(                                                                                                                                                                                                                                     
                       100,                                                                                                                                                                                                                                                    
                       16,                                                                                                                                                                                                                                                     
                       100,                                                                                                                                                                                                                                                    
                       0.65,                                                                                                                                                                                                                                                   
                       10,                                                                                                                                                                                                                                                     
         Samplers.UniformInit(-1, 1),                                                                                                                                                                                                                                      
                       :perchain,                                                                                                                                                                                                                                              
       )

training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false) 

conf_basic = Experiments.ExperimentConfig(                                                                                                                                                                                                                              
                           name = "multi_basic_test",                                                                                                                                                                                                                          
                           model = model,                                                                                                                                                                                                                                      
                           splitter = cv_config,                                                                                                                                                                                                                               
                           training_config = training_config,                                                                                                                                                                                                                  
                           save_dir ="./data/junk"                                                                                                                                                                                                                             
       )  

results_basic = Experiments.run_experiment(ds, conf_basic)    

c = results_basic.training_results[1][1]  
describe(c)



# -------------------------------------
#  MS Rho
# -------------------------------------
model = BayesianFootball.Models.PreGame.MSNegativeBinomialRho()

splits = BayesianFootball.Data.create_data_splits(ds, cv_config)

feature_sets = BayesianFootball.Features.create_features(
    splits, model, cv_config
)

splits = BayesianFootball.Data.create_data_splits(ds, cv_config) 
train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=2)  

sampler_conf = Samplers.NUTSConfig(                                                                                                                                                                                                                                     
                       100,                                                                                                                                                                                                                                                    
                       16,                                                                                                                                                                                                                                                     
                       100,                                                                                                                                                                                                                                                    
                       0.65,                                                                                                                                                                                                                                                   
                       10,                                                                                                                                                                                                                                                     
         Samplers.UniformInit(-1, 1),                                                                                                                                                                                                                                      
                       :perchain,                                                                                                                                                                                                                                              
       )

training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false) 

conf_basic = Experiments.ExperimentConfig(                                                                                                                                                                                                                              
                           name = "multi_basic_test",                                                                                                                                                                                                                          
                           model = model,                                                                                                                                                                                                                                      
                           splitter = cv_config,                                                                                                                                                                                                                               
                           training_config = training_config,                                                                                                                                                                                                                  
                           save_dir ="./data/junk"                                                                                                                                                                                                                             
       )  

results_basic = Experiments.run_experiment(ds, conf_basic)    

c = results_basic.training_results[1][1]  
describe(c)


#=
                                                                                                                                                                                                                                                                                                                                                                                            
Summary Statistics                                                                                                                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                                                                                            
         parameters      mean       std      mcse    ess_bulk    ess_tail      rhat   ess_per_sec                                                                                                                                                                                                                                                                                           
             Symbol   Float64   Float64   Float64     Float64     Float64   Float64       Float64                                                                                                                                                                                                                                                                                           
                                                                                                                                                                                                                                                                                                                                                                                            
                  μ    0.1684    0.0450    0.0011   1632.4615   1405.2831    1.0030        0.0159                                                                                                                                                                                                                                                                                           
                  γ    0.1632    0.0386    0.0009   1679.1658   1368.7012    1.0058        0.0164                                                                                                                                                                                                                                                                                           
              log_r    3.2659    0.3340    0.0086   1595.2555   1122.4088    1.0033        0.0156                                                                                                                                                                                                                                                                                           
           σ_r_team    0.2805    0.2124    0.0045   1739.1827    988.8372    1.0012        0.0170                                                                                                                                                                                                                                                                                           
    z_r_team_raw[1]   -0.0834    1.0110    0.0141   5126.5920   1094.0934    1.0197        0.0500                                                                                                                                                                                                                                                                                           
    z_r_team_raw[2]   -0.1327    1.0493    0.0147   5126.5920   1384.3075    1.0074        0.0500                                                                                                                                                                                                                                                                                           
    z_r_team_raw[3]   -0.0128    0.9669    0.0135   5126.5920   1253.3567    1.0190        0.0500                                                                                                                                                                                                                                                                                           
    z_r_team_raw[4]    0.0169    0.9591    0.0134   5126.5920   1176.8746    1.0196        0.0500                                                                                                                                                                                                                                                                                           
    z_r_team_raw[5]    0.0217    0.9587    0.0134   5126.5920   1026.1600    1.0231        0.0500                                                                                                                                                                                                                                                                                           
    z_r_team_raw[6]   -0.0078    0.9415    0.0131   5126.5920   1219.3576    1.0199        0.0500                                                                                                                                                                                                                           
    z_r_team_raw[7]   -0.0617    1.0524    0.0147   5126.5920    858.0622    1.0296        0.0500                                                                                                                                                                                                                           
    z_r_team_raw[8]    0.0831    0.9602    0.0134   5126.5920   1280.6326    1.0174        0.0500                                                                                                                                                                                                                           
    z_r_team_raw[9]   -0.0563    0.9651    0.0135   5126.5920   1337.2263    1.0121        0.0500                                                                                                                                                                                                                           
   z_r_team_raw[10]    0.0549    1.0124    0.0141   5126.5920   1188.7725    1.0277        0.0500                                                                                                                                                                                                                           
   z_r_team_raw[11]   -0.1911    1.0583    0.0148   5126.5920   1059.4233    1.0186        0.0500                                                                                                                                                                                                                           
   z_r_team_raw[12]   -0.0794    0.9752    0.0136   5126.5920   1264.2771    1.0109        0.0500                                                                                                                                                                                                                           
   z_r_team_raw[13]    0.0207    0.9801    0.0137   5126.5920   1149.4996    1.0280        0.0500                                                                                                                                                                                                                           
   z_r_team_raw[14]    0.0428    1.0021    0.0140   5126.5920   1377.5272    1.0144        0.0500                                                                                                                                                                                                                           
   z_r_team_raw[15]    0.0867    0.9920    0.0139   5126.5920   1179.2921    1.0131        0.0500                                                                                                                                                                                                                           
   z_r_team_raw[16]   -0.0282    0.9997    0.0140   5126.5920   1108.1817    1.0365        0.0500                                                                                                                                                                                                                           
   z_r_team_raw[17]    0.0031    0.9658    0.0140   4686.3875   1153.8143    1.0077        0.0457                                                                                                                                                                                                                           
   z_r_team_raw[18]   -0.0173    0.9585    0.0134   5126.5920   1365.3265    1.0136        0.0500                                                                                                                                                                                                                           
   z_r_team_raw[19]   -0.0375    0.9374    0.0131   5126.5920   1336.9779    1.0089        0.0500                                                                                                                                                                                                                           
   z_r_team_raw[20]   -0.0345    0.9783    0.0137   5126.5920   1121.3152    1.0345        0.0500                                                                                                                                                                                                                           
   z_r_team_raw[21]    0.1262    0.9507    0.0133   5126.5920   1310.0837    1.0132        0.0500                                                                                                                                                                                                                           
   z_r_team_raw[22]    0.0951    0.9977    0.0139   5126.5920   1393.6752    1.0077        0.0500                                                                                                                                                                                                                           
   z_r_team_raw[23]    0.0483    0.9827    0.0137   5126.5920   1357.5560    1.0071        0.0500                                                                                                                                                                                                                           
   z_r_team_raw[24]   -0.0052    0.9381    0.0131   5126.5920   1576.5545    1.0172        0.0500                                                                                                                                                                                                                           
   z_r_team_raw[25]    0.0916    1.0062    0.0141   5126.5920   1465.4381    1.0172        0.0500                                                                                                                                                                                                                           
          σ_r_month    0.3101    0.2405    0.0050   1731.5903    888.1591    1.0036        0.0169                                                                                                                                                                                                                           
   z_r_month_raw[1]    0.1165    0.9411    0.0131   5126.5920   1451.5779    1.0063        0.0500                                                                                                                                                                                                                           
   z_r_month_raw[2]    0.1902    0.9676    0.0135   5126.5920   1335.1800    1.0121        0.0500                                                                                                                                                                                                                           
   z_r_month_raw[3]   -0.1394    1.0254    0.0143   5126.5920   1097.7641    1.0136        0.0500                                                                                                                                                                                                                           
   z_r_month_raw[4]   -0.2332    0.9852    0.0138   5126.5920   1297.7728    1.0165        0.0500                                                                                                                                                                                                                           
   z_r_month_raw[5]    0.0483    0.9331    0.0130   5126.5920   1114.5416    1.0168        0.0500                                                                                                                                                                                                                           
   z_r_month_raw[6]   -0.0851    0.9940    0.0139   5126.5920   1121.8919    1.0208        0.0500                                                                                                                                                                                                                           
   z_r_month_raw[7]   -0.1761    1.0298    0.0144   5126.5920   1099.4774    1.0273        0.0500                                                                                                                                                                                                                           
   z_r_month_raw[8]    0.1163    0.9463    0.0140   4602.8599   1230.9371    1.0171        0.0449                                                                                                                                                                                                                           
   z_r_month_raw[9]    0.0865    0.9367    0.0136   4630.2067   1283.6871    1.0128        0.0452                                                                                                                                                                                                                           
  z_r_month_raw[10]   -0.0059    0.9478    0.0132   5126.5920   1164.8218    1.0189        0.0500                                                                                                                                                                                                                           
  z_r_month_raw[11]    0.1010    0.9446    0.0144   4489.3473   1309.4098    1.0150        0.0438                                                                                                                                                                                                                           
  z_r_month_raw[12]    0.0142    0.9877    0.0138   5126.5920   1367.5617    1.0198        0.0500                                                                                                                                                                                                                           
               σ_δₘ    0.0369    0.0253    0.0009    779.5840   1281.4531    1.0154        0.0076                                                                                                                                                                                                                           
        z_δₘ_raw[1]   -0.3023    0.8643    0.0139   3990.8551   1310.0932    1.0112        0.0390                                                                                                                                                                                                                           
        z_δₘ_raw[2]   -0.2595    0.8743    0.0145   3574.7198   1367.8464    1.0056        0.0349                                                                                                                                                                                                                           
        z_δₘ_raw[3]   -0.0098    0.9035    0.0130   4862.1073   1436.2158    1.0032        0.0475                                                                                                                                                                                                                           
        z_δₘ_raw[4]   -0.3517    0.8878    0.0124   5126.5920   1204.6179    1.0184        0.0500                                                                                                                                                                                                                           
        z_δₘ_raw[5]    0.2446    0.9519    0.0137   4909.2271   1296.2382    1.0293        0.0479                                                                                                                                                                                                                           
        z_δₘ_raw[6]   -0.0072    1.0378    0.0145   5126.5920    906.8014    1.0250        0.0500                                                                                                                                                                                                                           
        z_δₘ_raw[7]   -0.2509    1.0291    0.0150   4771.1725   1059.8602    1.0107        0.0466                                                                                                                                                                                                                           
        z_δₘ_raw[8]   -0.4612    0.8420    0.0130   4138.5820   1329.3872    1.0054        0.0404                                                                                                                                                                                                                           
        z_δₘ_raw[9]    0.4224    0.9117    0.0142   4136.4487   1266.3636    1.0158        0.0404                                                                                                                                                                                                                           
       z_δₘ_raw[10]    0.0446    0.8516    0.0127   4440.8411   1342.2051    1.0205        0.0433                                                                                                                                                                                                                           
       z_δₘ_raw[11]    0.1900    0.9293    0.0141   4440.2098   1303.3198    1.0129        0.0433                                                                                                                                                                                                                           
       z_δₘ_raw[12]    0.7760    0.9130    0.0170   3043.7072   1369.9686    1.0081        0.0297                                                                                                                                                                                                                           
                 δₙ    0.0307    0.0560    0.0014   1668.0161   1054.9346    1.0226        0.0163                                                                                                                                                                                                                           
                 δₚ    0.0598    0.0476    0.0012   1525.5309   1389.2512    1.0040        0.0149                                                                                                                                                                                                                           
               α.σ₀    0.2429    0.0489    0.0019    604.2829   1052.4936    1.0168        0.0059                                                                                                                                                                                                                           
               α.σₛ    0.0434    0.0301    0.0009    942.2126    803.4973    1.0133        0.0092                                                                                                                                                                                                                           
               α.σₖ    0.0213    0.0164    0.0004   1189.8631    964.1570    1.0042        0.0116                                                                                                                                                                                                                           
        α.z_init[1]    1.7327    0.5452    0.0128   1811.9793   1414.0908    1.0013        0.0177                                                                                                                                                                                                                           
        α.z_init[2]   -0.3364    0.5602    0.0127   1957.3967    985.0491    1.0113        0.0191                                                                                                                                                                                                                           
        α.z_init[3]    0.5368    0.4226    0.0092   2082.6472   1599.6763    1.0029        0.0203                                                                                                                                                                                                                           
        α.z_init[4]    0.2248    0.4174    0.0095   1938.2147   1415.5990    1.0055        0.0189                                                                                                                                                                                                                           
        α.z_init[5]    0.5043    0.5716    0.0124   2148.2524   1525.3394    1.0043        0.0210    


=#
using Statistics
using MCMCChains

# A custom helper for your current flat-named chain
function reconstruct_flat_ncp(chain, σ_name::Symbol, z_prefix::Symbol)
    # Extract the sigma parameter (e.g., :σ_δₘ)
    σ_vec = vec(Array(chain[σ_name])) 
    
    # Extract all the z variables (e.g., :z_δₘ_raw)
    z_raw = Array(group(chain, z_prefix)) 
    
    # Scale and center
    raw_val = z_raw .* reshape(σ_vec, :, 1)
    centered = raw_val .- mean(raw_val, dims=2)
    
    return centered
end

# 1. Extract using the exact variable names from your Turing summary
δₘ_samples          = reconstruct_flat_ncp(c, :σ_δₘ, :z_δₘ_raw)
log_r_month_samples = reconstruct_flat_ncp(c, :σ_r_month, :z_r_month_raw)
log_r_team_samples  = reconstruct_flat_ncp(c, :σ_r_team, :z_r_team_raw)
log_r_global        = vec(Array(c[:log_r]))

# 2. Get the Final Posterior Means
δₘ_means          = vec(mean(δₘ_samples, dims=1))
log_r_month_means = vec(mean(log_r_month_samples, dims=1))
log_r_team_means  = vec(mean(log_r_team_samples, dims=1))

# 3. Print the Month Effects
println("\n=== MONTH EFFECTS (δₘ) ===")
println("Positive = Higher Expected Goals, Negative = Lower Expected Goals")
for m in 1:12
    println("Month $m: $(round(δₘ_means[m], digits=4))")
end
println("Sum Check (Should be exactly 0.0): ", round(sum(δₘ_means), digits=10))

#=

=== MONTH EFFECTS (δₘ) ===

julia> println("Positive = Higher Expected Goals, Negative = Lower Expected Goals")
Positive = Higher Expected Goals, Negative = Lower Expected Goals

julia> for m in 1:12
           println("Month $m: $(round(δₘ_means[m], digits=4))")
       end
Month 1: -0.014
Month 2: -0.0113
Month 3: -0.0005
Month 4: -0.015
Month 5: 0.0126
Month 6: 0.0
Month 7: -0.0149
Month 8: -0.0198
Month 9: 0.0182
Month 10: 0.0023
Month 11: 0.0084
Month 12: 0.0339

julia> println("Sum Check (Should be exactly 0.0): ", round(sum(δₘ_means), digits=10))
Sum Check (Should be exactly 0.0): 0.0

=#

# 4. Print the Dispersion (r) for Months
println("\n=== MONTH DISPERSION (r) ===")
println("Higher r = Tighter/More Predictable, Lower r = Wilder/More Chaotic")
for m in 1:12
    r_val = exp(mean(log_r_global) + log_r_month_means[m])
    println("Month $m (r): $(round(r_val, digits=2))")
end


#=

julia> for m in 1:12
           r_val = exp(mean(log_r_global) + log_r_month_means[m])
           println("Month $m (r): $(round(r_val, digits=2))")
       end
Month 1 (r): 27.68
Month 2 (r): 28.45
Month 3 (r): 24.39
Month 4 (r): 23.5
Month 5 (r): 26.72
Month 6 (r): 25.0
Month 7 (r): 24.14
Month 8 (r): 27.94
Month 9 (r): 27.17
Month 10 (r): 26.48
Month 11 (r): 27.3
Month 12 (r): 26.24


=#


# Calculate the final r value for every single team
n_teams = length(log_r_team_means)
team_r_values = zeros(n_teams)

for t in 1:n_teams
    team_r_values[t] = exp(mean(log_r_global) + log_r_team_means[t])
end

# Sort the indices so we can see the extremes
sorted_team_indices = sortperm(team_r_values)

println("\n=== TEAM DISPERSION (r) ===")
println("Higher r = Rigid/Predictable matches (e.g., rigid defenses)")
println("Lower r  = Chaotic/Volatile matches (e.g., heavy attacking/leaky defenses)\n")

println("--- Top 5 Most CHAOTIC Teams (Lowest r) ---")
for i in 1:5
    idx = sorted_team_indices[i]
    println("Team ID $(lpad(idx, 2)): r = $(round(team_r_values[idx], digits=2))")
end

println("\n--- Top 5 Most PREDICTABLE Teams (Highest r) ---")
for i in 1:5
    # Counting backwards from the end of the sorted list
    idx = sorted_team_indices[n_teams - i + 1]
    println("Team ID $(lpad(idx, 2)): r = $(round(team_r_values[idx], digits=2))")
end

#=

julia> team_r_values
25-element Vector{Float64}:
 25.43811101024618
 24.416581338128378
 26.28008322159853
 26.653764859892743
 26.56340475339229
 26.39939907007821
 25.313950165477124
 27.101842073124004
 25.503837107256615
 27.113918989642606
 24.129266249656933
 25.236697234194235
 26.320849964844157
 26.519131615290604
 27.36772785618071
 25.748016718238844
 26.203332919241916
 25.97751343278705
 26.11205731804881
 25.712396412024358
 27.63932944187828
 27.451223307526394
 26.519052741600845
 26.22999025842324
 27.511954294169684

=#

feature_sets[1][1][:team_map]

# 1. Reverse the dictionary so we can look up Names by ID
team_map = feature_sets[1][1][:team_map]
id_to_team = Dict(v => k for (k, v) in team_map)

# 2. Sort the indices based on the r values
sorted_indices = sortperm(team_r_values)

println("=== TEAM DISPERSION (r) ===")
println("Lower r  = Volatile/Chaotic (Wider spread of goals, fatter tails)")
println("Higher r = Rigid/Predictable (Tighter spread of goals, closer to Poisson)\n")

println("--- Top 5 Most CHAOTIC Teams (Lowest r) ---")
for i in 1:5
    idx = sorted_indices[i]
    team_name = id_to_team[idx]
    r_val = team_r_values[idx]
    println("$(rpad(team_name, 30)) | r = $(round(r_val, digits=2))")
end

println("\n--- Top 5 Most PREDICTABLE Teams (Highest r) ---")
for i in 1:5
    idx = sorted_indices[end - i + 1] # Count backwards from the end
    team_name = id_to_team[idx]
    r_val = team_r_values[idx]
    println("$(rpad(team_name, 30)) | r = $(round(r_val, digits=2))")
end

#=
julia> println("=== TEAM DISPERSION (r) ===")
=== TEAM DISPERSION (r) ===

julia> println("Lower r  = Volatile/Chaotic (Wider spread of goals, fatter tails)")
Lower r  = Volatile/Chaotic (Wider spread of goals, fatter tails)

julia> println("Higher r = Rigid/Predictable (Tighter spread of goals, closer to Poisson)\n")
Higher r = Rigid/Predictable (Tighter spread of goals, closer to Poisson)


julia> println("--- Top 5 Most CHAOTIC Teams (Lowest r) ---")
--- Top 5 Most CHAOTIC Teams (Lowest r) ---

julia> for i in 1:5
           idx = sorted_indices[i]
           team_name = id_to_team[idx]
           r_val = team_r_values[idx]
           println("$(rpad(team_name, 30)) | r = $(round(r_val, digits=2))")
       end
east-fife                      | r = 24.13
albion-rovers                  | r = 24.42
edinburgh-city-fc              | r = 25.24
clyde-fc                       | r = 25.31
airdrieonians                  | r = 25.44

julia> println("\n--- Top 5 Most PREDICTABLE Teams (Highest r) ---")

--- Top 5 Most PREDICTABLE Teams (Highest r) ---

julia> for i in 1:5
           idx = sorted_indices[end - i + 1] # Count backwards from the end
           team_name = id_to_team[idx]
           r_val = team_r_values[idx]
           println("$(rpad(team_name, 30)) | r = $(round(r_val, digits=2))")
       end
queen-of-the-south             | r = 27.64
the-spartans-fc                | r = 27.51
stenhousemuir                  | r = 27.45
forfar-athletic                | r = 27.37
dunfermline-athletic           | r = 27.11


=#



# -------------------------------------
#  MS kappa
# -------------------------------------
# Note, we are not pooling and sum to zero the months ... need to update this

model = BayesianFootball.Models.PreGame.MSNegativeBinomialKappa()

splits = BayesianFootball.Data.create_data_splits(ds, cv_config)

feature_sets = BayesianFootball.Features.create_features(
    splits, model, cv_config
)

splits = BayesianFootball.Data.create_data_splits(ds, cv_config) 
train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=2)  

sampler_conf = Samplers.NUTSConfig(                                                                                                                                                                                                                                     
                       200,                                                                                                                                                                                                                                                    
                       16,                                                                                                                                                                                                                                                     
                       100,                                                                                                                                                                                                                                                    
                       0.65,                                                                                                                                                                                                                                                   
                       10,                                                                                                                                                                                                                                                     
         Samplers.UniformInit(-1, 1),                                                                                                                                                                                                                                      
                       :perchain,                                                                                                                                                                                                                                              
       )

training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false) 

conf_basic = Experiments.ExperimentConfig(                                                                                                                                                                                                                              
                           name = "multi_basic_test",                                                                                                                                                                                                                          
                           model = model,                                                                                                                                                                                                                                      
                           splitter = cv_config,                                                                                                                                                                                                                               
                           training_config = training_config,                                                                                                                                                                                                                  
                           save_dir ="./data/junk"                                                                                                                                                                                                                             
       )  

results_basic = Experiments.run_experiment(ds, conf_basic)    



c = results_basic.training_results[1][1]  

typeof(c)


describe(c)

#=
                                                                                                                                                                                                                   
Summary Statistics                                                                                                                                                                                                 
                                                                                                                                                                                                                   
    parameters      mean       std      mcse    ess_bulk    ess_tail      rhat   ess_per_sec                                                                                                                       
        Symbol   Float64   Float64   Float64     Float64     Float64   Float64       Float64                                                                                                                       
                                                                                                                                                                                                                   
             μ    0.1307    0.0505    0.0009   3380.4520   2714.2905    0.9994        0.0180                                                                                                                       
             γ    0.1360    0.0320    0.0006   3237.9329   2561.4171    1.0025        0.0173                                                                                                                       
         log_r    3.3476    0.3157    0.0057   3201.4945   2117.9919    1.0082        0.0171                                                                                                                       
         δₘ[1]   -0.0257    0.0538    0.0010   3126.6653   2759.6589    1.0027        0.0167                                                                                                                       
         δₘ[2]   -0.0499    0.0516    0.0009   3472.3087   2874.4275    1.0018        0.0185                                                                                                                       
         δₘ[3]   -0.0425    0.0504    0.0009   3069.3742   2688.0708    1.0016        0.0164                                                                                                                       
         δₘ[4]   -0.0278    0.0497    0.0009   3231.4724   3103.6598    1.0000        0.0172                                                                                                                       
         δₘ[5]    0.0498    0.0814    0.0015   3077.4184   2140.3797    1.0028        0.0164                                                                                                                       
         δₘ[6]   -0.0006    0.0995    0.0017   3430.7052   2097.9790    1.0063        0.0183                                                                                                                       
         δₘ[7]   -0.0874    0.0843    0.0015   3059.9509   2261.4831    1.0071        0.0163                                                                                                                       
         δₘ[8]   -0.0193    0.0533    0.0009   3246.2276   2930.9046    1.0022        0.0173                                                                                                                       
         δₘ[9]    0.0401    0.0547    0.0010   3247.5775   2536.5448    1.0023        0.0173                                                                                                                       
        δₘ[10]    0.0404    0.0556    0.0010   3036.7051   2431.2697    1.0037        0.0162                                                                                                                       
        δₘ[11]   -0.0058    0.0550    0.0009   3389.1915   2982.0973    0.9993        0.0181                                                                                                                       
        δₘ[12]    0.1217    0.0519    0.0009   3436.3575   2937.9814    1.0040        0.0183                                                                                                                       
            δₙ    0.0093    0.0493    0.0008   3548.2617   2176.7070    1.0029        0.0189                                                                                                                       
            δₚ    0.1039    0.0422    0.0007   3332.2456   3174.9045    1.0025        0.0178                                                                                                                       
          α.σ₀    0.2097    0.0477    0.0013   1366.0218   2277.6677    1.0107        0.0073                                                                                                                       
          α.σₛ    0.0920    0.0304    0.0008   1376.7357   1163.9546    1.0107        0.0073                                                                                                                       
          α.σₖ    0.0230    0.0176    0.0004   2045.4544   1673.3945    1.0023        0.0109                                                                                                                       
   α.z_init[1]    1.6626    0.5307    0.0094   3221.0170   2653.8971    1.0030        0.0172



=#


using Plots
using StatsPlots
using MCMCChains

# 1. Set backend and create directory
plotlyjs()
mkpath("figures")

# Define the chain variable explicitly (change 'c' to whatever your chain is named)
my_chain = c 

println("Generating plots... (This may take a moment)")

# ==============================================================================
# PLOT 1: Global Baselines and Single-Flag Modifiers
# ==============================================================================
base_syms = [Symbol("μ"), Symbol("γ"), Symbol("log_r"), Symbol("δₙ"), Symbol("δₚ")]
base_chain = my_chain[base_syms]

p_base = plot(
    base_chain, 
    size=(1600, 1000), 
    # title="Global Hyperparameters (Baselines & Flags)",
    margin=5Plots.mm
);
savefig(p_base, "figures/mcmc_global_hyperparams.html")
println("Saved: figures/mcmc_global_hyperparams.html")


# ==============================================================================
# PLOT 2: Monthly Modifiers (Using the `group` function)
# ==============================================================================
# `group` automatically grabs all δₘ[1] through δₘ[12]
month_chain = group(my_chain, :δₘ)

p_month = plot(
    month_chain, 
    size=(1200, 1800), # Made taller to fit 12 parameters neatly
    title="Monthly Calendar Modifiers (δₘ)",
    margin=5Plots.mm
);
savefig(p_month, "figures/mcmc_monthly_effects.html")
println("Saved: figures/mcmc_monthly_effects.html")


# ==============================================================================
# PLOT 3: The Multi-Scale Volatilities (Sigmas)
# ==============================================================================
sigma_syms = [
    Symbol("α.σ₀"), Symbol("α.σₛ"), Symbol("α.σₖ"),
    Symbol("β.σ₀"), Symbol("β.σₛ"), Symbol("β.σₖ")
]
sigma_chain = my_chain[sigma_syms]

p_sigmas = plot(
    sigma_chain, 
    size=(1600, 1600), 
    title="Multi-Scale Variances (Macro vs Micro)",
    margin=5Plots.mm
);
savefig(p_sigmas, "figures/mcmc_sigmas.html")
println("Saved: figures/mcmc_sigmas.html")

println("All plots successfully generated!")


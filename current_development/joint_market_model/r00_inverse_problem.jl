# current_development/joint_market_model/r00_inverse_problem.jl

using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
pinthreads(:cores)

ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())


### 1. Explore 
matches = subset(ds.matches, :season => ByRow(isequal("2025")))
odds = subset(ds.odds, :match_id => ByRow(in(matches.match_id)))

rand_matchid = rand( matches.match_id)
rand_odds = subset(odds, :match_id  => ByRow(isequal(rand_matchid)))

########################################
### 2. Basic inverse problem using only the 1x2 lines.
########################################

using Distributions
using Optim

# 1. Target probabilities from your DataFrame (prob_fair_close)
target_H = 0.22363
target_D = 0.284619
target_A = 0.491751

# 2. Define the Forward Model
function match_probabilities(lambda_h::Float64, lambda_a::Float64; max_goals=15)
    # Truncate at max_goals (15 is practically infinite for football)
    dist_h = Poisson(lambda_h)
    dist_a = Poisson(lambda_a)
    
    prob_H = 0.0
    prob_D = 0.0
    prob_A = 0.0
    
    # Calculate the joint probability for every scoreline
    for i in 0:max_goals
        for j in 0:max_goals
            p_ij = pdf(dist_h, i) * pdf(dist_a, j)
            if i > j
                prob_H += p_ij
            elseif i == j
                prob_D += p_ij
            else
                prob_A += p_ij
            end
        end
    end
    return prob_H, prob_D, prob_A
end

# 3. Define the Objective/Loss Function
function loss(theta)
    # theta[1] = lambda_h, theta[2] = lambda_a
    # We use exp() to ensure lambdas remain strictly positive during optimization
    lh, la = exp(theta[1]), exp(theta[2]) 
    
    pH, pD, pA = match_probabilities(lh, la)
    
    # Sum of Squared Errors (SSE)
    return (pH - target_H)^2 + (pD - target_D)^2 + (pA - target_A)^2
end

# 4. Run the Optimizer
# Initial guess: let's guess 1.0 goals for Home, 1.5 goals for Away
# We pass log(1.0) and log(1.5) because our loss function exponentiates the inputs
initial_guess = [log(1.0), log(1.5)] 

result = optimize(loss, initial_guess, NelderMead())

# 5. Extract Results
best_lambda_h = exp(Optim.minimizer(result)[1])
best_lambda_a = exp(Optim.minimizer(result)[2])

println("Optimal Lambda Home: ", round(best_lambda_h, digits=4))
println("Optimal Lambda Away: ", round(best_lambda_a, digits=4))

# Check how well it fits
fit_H, fit_D, fit_A = match_probabilities(best_lambda_h, best_lambda_a)
println("--- Model vs Target ---")
println("Home: ", round(fit_H, digits=4), " | Target: ", round(target_H, digits=4))
println("Draw: ", round(fit_D, digits=4), " | Target: ", round(target_D, digits=4))
println("Away: ", round(fit_A, digits=4), " | Target: ", round(target_A, digits=4))




########################################
### 3. Inverse problem with 1x2, under/over 
########################################

using Distributions
using Optim 

# target probabilities 
const itarget_H = 0.22363
const itarget_D = 0.284619
const itarget_A = 0.491751
const itarget_O25 = 0.513158
const itarget_U25 = 0.486842



function match_probabilities(λ_h::Float64, λ_a::Float64; max_goals=15) 
  dist_h = Poisson(λ_h)
  dist_a = Poisson(λ_a)


  prob_H, prob_D, prob_A = 0.0, 0.0, 0.0 
  prob_U25 = 0.0

  for i in 0:max_goals 
    for j in 0:max_goals 
        p_ij = pdf(dist_h, i) * pdf(dist_a, j) 

  
        # 1x2 logic 
        if i > j prob_H += p_ij 
        elseif i ==j prob_D += p_ij 
        else prob_A += p_ij 
        end 
      
        # Under / Over logic ( total goals < 2.5 means [0,1,2] goals 
        if (i+j)<=2
          prob_U25 += p_ij 
        end 
    end 
  end 

  prob_O25 = 1.0 - prob_U25 
  return prob_H, prob_D, prob_A, prob_O25, prob_U25 
end 
  
# update the loss function ( sum of the squared errors across all markets)
function 𝓛(θ)
  λh, λa = exp.(θ) 
  𝙥H, 𝙥D, 𝙥A, 𝙥O25, 𝙥U25 = match_probabilities(λh, λa)

  # 1x2 Error 
  𝛆1x2 = (𝙥H - itarget_H)^2 + ( 𝙥D - itarget_D)^2 + (𝙥A - itarget_A)^2 
  𝛆ou = (𝙥O25 - itarget_O25)^2 + (𝙥U25 - itarget_U25)^2 

  return 𝛆1x2 + 𝛆ou
end 


# run the Optimisation 
θ_initial = [log(1.5), log(1.0)]

result = optimize(𝓛, θ_initial, NelderMead())

# 4. Results
best_lambda_h = exp(Optim.minimizer(result)[1])
best_lambda_a = exp(Optim.minimizer(result)[2])

println("Optimal Lambda Home: ", round(best_lambda_h, digits=4))
println("Optimal Lambda Away: ", round(best_lambda_a, digits=4))

function display_match_probabilities(lh,la) 
    fit_H, fit_D, fit_A, fit_O25, fit_U25 = match_probabilities(best_lambda_h, best_lambda_a)
    println("--- Model vs Target ---")
    println("Home: ", round(fit_H, digits=4), " | Target: ", round(itarget_H, digits=4))
    println("Draw: ", round(fit_D, digits=4), " | Target: ", round(itarget_D, digits=4))
    println("Away: ", round(fit_A, digits=4), " | Target: ", round(itarget_A, digits=4))
    println("Over25: ", round(fit_O25, digits=4), " | Target: ", round(itarget_O25, digits=4))
    println("Under25: ", round(fit_U25, digits=4), " | Target: ", round(itarget_U25, digits=4))
end

display_match_probabilities(best_lambda_h, best_lambda_a)


########################################
### 4. Inverse problem with 1x2, under/over extend ( 0.5, 1.5, 2.5, 3.5)
########################################
#


#=
julia> rand_odds

selection prob_fair_close  
Symbol    Float64          
───────── ─────────────────
home            0.22363    
draw            0.284619   
away            0.491751   
btts_yes        0.546667   
btts_no         0.453333   
over_05         0.912863   
under_05        0.0871369  
over_15         0.75       
under_15        0.25       
over_25         0.513158   
under_25        0.486842   
over_35         0.3125     
under_35        0.6875     
over_45         0.157895   
under_45        0.842105   
over_55         0.0740741  
under_55        0.925926   
over_65         0.0373936  
under_65        0.962606   
over_75         0.0192685  
under_75        0.980732   
=#


using Distributions
using Optim 

# target probabilities 
const jtarget_H = 0.22363
const jtarget_D = 0.284619
const jtarget_A = 0.491751

const jtarget_O05 = 0.912863
const jtarget_U05 = 0.0871369

const jtarget_O15 = 0.75
const jtarget_U15 = 0.25

const jtarget_O25 = 0.513158
const jtarget_U25 = 0.486842

const jtarget_O35 = 0.3125
const jtarget_U35 = 0.6875


function match_probabilities(λ_h::Float64, λ_a::Float64; max_goals=15) 
  dist_h = Poisson(λ_h)
  dist_a = Poisson(λ_a)


  prob_H, prob_D, prob_A = 0.0, 0.0, 0.0 
  prob_U05, prob_U15, prob_U25, prob_U35 = 0.0, 0.0, 0.0, 0.0

  for i in 0:max_goals 
    for j in 0:max_goals 
        p_ij = pdf(dist_h, i) * pdf(dist_a, j) 

  
        # 1x2 logic 
        if i > j prob_H += p_ij 
        elseif i ==j prob_D += p_ij 
        else prob_A += p_ij 
        end 
      
        # Under / Over logic ( total goals < 2.5 means [0,1,2] goals 
        if (i+j)<=0.5
          prob_U05 += p_ij 
        end 
        if (i+j)<=1.5
          prob_U15 += p_ij 
        end 
        if (i+j)<=2.5
          prob_U25 += p_ij 
        end 
        if (i+j)<=3.5
          prob_U35 += p_ij 
        end 
    end 
  end 

  prob_O05 = 1.0 - prob_U05 
  prob_O15 = 1.0 - prob_U15 
  prob_O25 = 1.0 - prob_U25 
  prob_O35 = 1.0 - prob_U35 
  return prob_H, prob_D, prob_A, prob_O05, prob_U05, prob_O15, prob_U15, prob_O25, prob_U25,prob_O35, prob_U35
end 
  
# update the loss function ( sum of the squared errors across all markets)
function 𝓛(θ)
  λh, λa = exp.(θ) 
  𝙥H, 𝙥D, 𝙥A, 𝙥O05, 𝙥U05, 𝙥O15, 𝙥U15, 𝙥O25, 𝙥U25, 𝙥O35, 𝙥U35 = match_probabilities(λh, λa)

  # 1x2 Error 
  𝛆1x2 = (𝙥H - jtarget_H)^2 + ( 𝙥D - jtarget_D)^2 + (𝙥A - jtarget_A)^2 

  𝛆ou05 = (𝙥O05 - jtarget_O05)^2 + (𝙥U05 - jtarget_U05)^2 
  𝛆ou15 = (𝙥O15 - jtarget_O15)^2 + (𝙥U15 - jtarget_U15)^2 
  𝛆ou25 = (𝙥O25 - jtarget_O25)^2 + (𝙥U25 - jtarget_U25)^2 
  𝛆ou35 = (𝙥O35 - jtarget_O35)^2 + (𝙥U35 - jtarget_U35)^2 

  return 𝛆1x2 + 𝛆ou05 + 𝛆ou15 + 𝛆ou25 + 𝛆ou35
end 


# run the Optimisation 
θ_initial = [log(1.5), log(1.0)]

result = optimize(𝓛, θ_initial, NelderMead())

# 4. Results
best_lambda_h = exp(Optim.minimizer(result)[1])
best_lambda_a = exp(Optim.minimizer(result)[2])

println("Optimal Lambda Home: ", round(best_lambda_h, digits=4))
println("Optimal Lambda Away: ", round(best_lambda_a, digits=4))

function display_match_probabilities(lh,la) 
    fit_H, fit_D, fit_A, fit_O05, fit_U05, fit_O15, fit_U15, fit_O25, fit_U25, fit_O35, fit_U35= match_probabilities(best_lambda_h, best_lambda_a)
    println("--- Model vs Target ---")
    println("Home: ", round(fit_H, digits=4), " | Target: ", round(jtarget_H, digits=4))
    println("Draw: ", round(fit_D, digits=4), " | Target: ", round(jtarget_D, digits=4))
    println("Away: ", round(fit_A, digits=4), " | Target: ", round(jtarget_A, digits=4))

    println("Over05: ", round(fit_O05, digits=4), " | Target: ", round(jtarget_O05, digits=4))
    println("Under05: ", round(fit_U05, digits=4), " | Target: ", round(jtarget_U05, digits=4))
    println("Over15: ", round(fit_O15, digits=4), " | Target: ", round(jtarget_O15, digits=4))
    println("Under15: ", round(fit_U15, digits=4), " | Target: ", round(jtarget_U15, digits=4))
    println("Over25: ", round(fit_O25, digits=4), " | Target: ", round(jtarget_O25, digits=4))
    println("Under25: ", round(fit_U25, digits=4), " | Target: ", round(jtarget_U25, digits=4))
    println("Over35: ", round(fit_O35, digits=4), " | Target: ", round(jtarget_O35, digits=4))
    println("Under35: ", round(fit_U35, digits=4), " | Target: ", round(jtarget_U35, digits=4))
end

display_match_probabilities(best_lambda_h, best_lambda_a)


#=

julia> println("Optimal Lambda Home: ", round(best_lambda_h, digits=4))
Optimal Lambda Home: 1.0812

julia> println("Optimal Lambda Away: ", round(best_lambda_a, digits=4))
Optimal Lambda Away: 1.6493

Double Poisson
julia> display_match_probabilities(best_lambda_h, best_lambda_a)
--- Model vs Target ---
Home: 0.2491 | Target: 0.2236
Draw: 0.2447 | Target: 0.2846
Away: 0.5062 | Target: 0.4918
Over05: 0.9348 | Target: 0.9129
Under05: 0.0652 | Target: 0.0871
Over15: 0.7568 | Target: 0.75
Under15: 0.2432 | Target: 0.25
Over25: 0.5138 | Target: 0.5132
Under25: 0.4862 | Target: 0.4868
Over35: 0.2927 | Target: 0.3125
Under35: 0.7073 | Target: 0.6875
=#


# 
using Distributions
using Optim 

# Target probabilities 
const jtarget_H = 0.22363
const jtarget_D = 0.284619
const jtarget_A = 0.491751

const jtarget_O05 = 0.912863
const jtarget_U05 = 0.0871369

const jtarget_O15 = 0.75
const jtarget_U15 = 0.25

const jtarget_O25 = 0.513158
const jtarget_U25 = 0.486842

const jtarget_O35 = 0.3125
const jtarget_U35 = 0.6875

# 1. Dixon-Coles Correlation Function
function dixon_coles_tau(i::Int, j::Int, λ_h::Float64, λ_a::Float64, ρ::Float64)
    if i == 0 && j == 0
        return 1.0 - (λ_h * λ_a * ρ)
    elseif i == 0 && j == 1
        return 1.0 + (λ_h * ρ)
    elseif i == 1 && j == 0
        return 1.0 + (λ_a * ρ)
    elseif i == 1 && j == 1
        return 1.0 - ρ
    else
        return 1.0
    end
end

# 2. Updated Forward Model (Added ρ)
function match_probabilities(λ_h::Float64, λ_a::Float64, ρ::Float64; max_goals=15) 
    dist_h = Poisson(λ_h)
    dist_a = Poisson(λ_a)

    prob_H, prob_D, prob_A = 0.0, 0.0, 0.0 
    prob_U05, prob_U15, prob_U25, prob_U35 = 0.0, 0.0, 0.0, 0.0

    for i in 0:max_goals 
        for j in 0:max_goals 
            # Calculate Independent Prob
            p_indep = pdf(dist_h, i) * pdf(dist_a, j) 
            
            # Apply Dixon-Coles Correction
            τ = dixon_coles_tau(i, j, λ_h, λ_a, ρ)
            p_ij = p_indep * τ
  
            # 1x2 logic 
            if i > j prob_H += p_ij 
            elseif i == j prob_D += p_ij 
            else prob_A += p_ij 
            end 
      
            # Under / Over logic 
            if (i+j) <= 0.5 prob_U05 += p_ij end 
            if (i+j) <= 1.5 prob_U15 += p_ij end 
            if (i+j) <= 2.5 prob_U25 += p_ij end 
            if (i+j) <= 3.5 prob_U35 += p_ij end 
        end 
    end 

    prob_O05 = 1.0 - prob_U05 
    prob_O15 = 1.0 - prob_U15 
    prob_O25 = 1.0 - prob_U25 
    prob_O35 = 1.0 - prob_U35 
    return prob_H, prob_D, prob_A, prob_O05, prob_U05, prob_O15, prob_U15, prob_O25, prob_U25, prob_O35, prob_U35
end 
  
# 3. Update the Loss Function (Unpack 3 parameters)
function 𝓛(θ)
    λh = exp(θ[1])
    λa = exp(θ[2])
    ρ = θ[3] # Do not exponentiate rho, it can be slightly negative or exactly 0
    
    𝙥H, 𝙥D, 𝙥A, 𝙥O05, 𝙥U05, 𝙥O15, 𝙥U15, 𝙥O25, 𝙥U25, 𝙥O35, 𝙥U35 = match_probabilities(λh, λa, ρ)

    # 1x2 Error 
    𝛆1x2 = (𝙥H - jtarget_H)^2 + (𝙥D - jtarget_D)^2 + (𝙥A - jtarget_A)^2 

    # Over/Under Errors
    𝛆ou05 = (𝙥O05 - jtarget_O05)^2 + (𝙥U05 - jtarget_U05)^2 
    𝛆ou15 = (𝙥O15 - jtarget_O15)^2 + (𝙥U15 - jtarget_U15)^2 
    𝛆ou25 = (𝙥O25 - jtarget_O25)^2 + (𝙥U25 - jtarget_U25)^2 
    𝛆ou35 = (𝙥O35 - jtarget_O35)^2 + (𝙥U35 - jtarget_U35)^2 

    return 𝛆1x2 + 𝛆ou05 + 𝛆ou15 + 𝛆ou25 + 𝛆ou35
end 

# 4. Run the Optimisation (Added initial guess for ρ)
θ_initial = [log(1.5), log(1.0), 0.05]

result = optimize(𝓛, θ_initial, NelderMead())

# Extract Results
best_lambda_h = exp(Optim.minimizer(result)[1])
best_lambda_a = exp(Optim.minimizer(result)[2])
best_rho = Optim.minimizer(result)[3]

println("Optimal Lambda Home: ", round(best_lambda_h, digits=4))
println("Optimal Lambda Away: ", round(best_lambda_a, digits=4))
println("Optimal Rho: ", round(best_rho, digits=4))

# 5. Display Function Updated
function display_match_probabilities(lh, la, rho) 
    fit_H, fit_D, fit_A, fit_O05, fit_U05, fit_O15, fit_U15, fit_O25, fit_U25, fit_O35, fit_U35 = match_probabilities(lh, la, rho)
    
    println("\n--- Model vs Target ---")
    println("Home: ", round(fit_H, digits=4), " | Target: ", round(jtarget_H, digits=4))
    println("Draw: ", round(fit_D, digits=4), " | Target: ", round(jtarget_D, digits=4))
    println("Away: ", round(fit_A, digits=4), " | Target: ", round(jtarget_A, digits=4))

    println("Over05: ", round(fit_O05, digits=4), " | Target: ", round(jtarget_O05, digits=4))
    println("Under05: ", round(fit_U05, digits=4), " | Target: ", round(jtarget_U05, digits=4))
    println("Over15: ", round(fit_O15, digits=4), " | Target: ", round(jtarget_O15, digits=4))
    println("Under15: ", round(fit_U15, digits=4), " | Target: ", round(jtarget_U15, digits=4))
    println("Over25: ", round(fit_O25, digits=4), " | Target: ", round(jtarget_O25, digits=4))
    println("Under25: ", round(fit_U25, digits=4), " | Target: ", round(jtarget_U25, digits=4))
    println("Over35: ", round(fit_O35, digits=4), " | Target: ", round(jtarget_O35, digits=4))
    println("Under35: ", round(fit_U35, digits=4), " | Target: ", round(jtarget_U35, digits=4))
end

display_match_probabilities(best_lambda_h, best_lambda_a, best_rho)




#=
Dixon coles
--- Model vs Target ---
Home: 0.23 | Target: 0.2236
Draw: 0.2747 | Target: 0.2846
Away: 0.4953 | Target: 0.4918
Over05: 0.9193 | Target: 0.9129
Under05: 0.0807 | Target: 0.0871
Over15: 0.7714 | Target: 0.75
Under15: 0.2286 | Target: 0.25
Over25: 0.5129 | Target: 0.5132
Under25: 0.4871 | Target: 0.4868
Over35: 0.2918 | Target: 0.3125
Under35: 0.7082 | Target: 0.6875




#=
julia> println("Optimal Lambda Home: ", round(best_lambda_h, digits=4))
Optimal Lambda Home: 1.0702

julia> println("Optimal Lambda Away: ", round(best_lambda_a, digits=4))
Optimal Lambda Away: 1.6565

julia> println("Optimal Rho: ", round(best_rho, digits=4))
Optimal Rho: -0.1314
=#

=#




### --- Abstracting it out 
using Distributions
using Optim
using LinearAlgebra # For diag, tril, triu

# 1. The Core Correlation Logic
function dixon_coles_tau(i::Int, j::Int, λ_h::Float64, λ_a::Float64, ρ::Float64)
    if i == 0 && j == 0 return 1.0 - (λ_h * λ_a * ρ)
    elseif i == 0 && j == 1 return 1.0 + (λ_h * ρ)
    elseif i == 1 && j == 0 return 1.0 + (λ_a * ρ)
    elseif i == 1 && j == 1 return 1.0 - ρ
    else return 1.0 end
end

# 2. The Abstraction Wrapper
function fit_market_implied_parameters(match_df; max_goals=10)
    
    # Pre-extract target probabilities from the DataFrame into a fast Dictionary
    # We use the 'selection' symbol (e.g., :home, :over_25, :btts_yes) as the key
    targets = Dict(row.selection => row.prob_fair_close for row in eachrow(match_df))

    # Define the objective function inside the wrapper so it has access to 'targets'
    function loss(θ)
        λh, λa = exp(θ[1]), exp(θ[2])
        ρ = θ[3]
        
        # A. Build the Probability Matrix P
        P = zeros(Float64, max_goals + 1, max_goals + 1)
        for i in 0:max_goals
            for j in 0:max_goals
                P[i+1, j+1] = pdf(Poisson(λh), i) * pdf(Poisson(λa), j) * dixon_coles_tau(i, j, λh, λa, ρ)
            end
        end
        
        sse = 0.0
        
        # B. Dynamically calculate errors ONLY for markets that exist in the DataFrame
        
        # --- 1X2 Market ---
        if haskey(targets, :home) sse += (sum(tril(P, -1)) - targets[:home])^2 end
        if haskey(targets, :draw) sse += (sum(diag(P)) - targets[:draw])^2 end
        if haskey(targets, :away) sse += (sum(triu(P, 1)) - targets[:away])^2 end
        
        # --- BTTS Market ---
        if haskey(targets, :btts_yes)
            # BTTS Yes is the sum of the matrix excluding the 0th row and 0th column
            prob_btts = sum(P[2:end, 2:end]) 
            sse += (prob_btts - targets[:btts_yes])^2
            if haskey(targets, :btts_no) sse += ((1.0 - prob_btts) - targets[:btts_no])^2 end
        end
        
        # --- Over/Under Markets ---
        # Iterate through possible lines (0.5 to 8.5) and check if they exist in targets
        for k in 0:8
            over_key = Symbol("over_$(k)5")
            under_key = Symbol("under_$(k)5")
            
            if haskey(targets, over_key) || haskey(targets, under_key)
                # Calculate Under probability by summing where (goals_home + goals_away) <= k
                prob_under = 0.0
                for i in 0:max_goals
                    for j in 0:max_goals
                        if (i + j) <= k
                            prob_under += P[i+1, j+1]
                        end
                    end
                end
                
                prob_over = 1.0 - prob_under
                
                if haskey(targets, over_key) sse += (prob_over - targets[over_key])^2 end
                if haskey(targets, under_key) sse += (prob_under - targets[under_key])^2 end
            end
        end
        
        return sse
    end

    # 3. Run Optimization
    initial_guess = [log(1.5), log(1.0), 0.05]
    result = optimize(loss, initial_guess, NelderMead())
    
    # 4. Return a clean structure
    return (
        match_id = match_df.match_id[1],
        λ_home = exp(Optim.minimizer(result)[1]),
        λ_away = exp(Optim.minimizer(result)[2]),
        ρ = Optim.minimizer(result)[3],
        fit_error = Optim.minimum(result) # Useful for flagging bad fits
    )
end



rand_matchid = rand( matches.match_id)
rand_odds = subset(odds, :match_id  => ByRow(isequal(rand_matchid)))



# Assuming 'rand_odds' is the 21-row DataFrame for match 13250794
market_params = fit_market_implied_parameters(rand_odds)

display_implied_market(market_params)
subset(matches, :match_id => ByRow(isequal(rand_matchid)))[:,[:match_id, :home_team, :away_team, :home_score, :away_score]]





#=
julia> display_implied_market(market_params)
Match ID: 13250702
Market λ_Home: 0.9998
Market λ_Away: 1.2986
Market ρ: -0.0693
Residual Error: 0.002293

julia> subset(matches, :match_id => ByRow(isequal(rand_matchid)))[:,[:match_id, :home_team, :away_team, :home_score, :away_score]]
1×5 DataFrame
 Row │ match_id  home_team  away_team        home_score  away_score 
     │ Int32?    String?    String?          Int32?      Int32?     
─────┼──────────────────────────────────────────────────────────────
   1 │ 13250702  bohemian   shamrock-rovers           2           0
=#

#=
julia> display_implied_market(market_params)
Match ID: 13250813
Market λ_Home: 2.1735
Market λ_Away: 0.765
Market ρ: 0.0301
Residual Error: 0.003559

julia> subset(matches, :match_id => ByRow(isequal(rand_matchid)))[:,[:match_id, :home_team, :away_team, :home_score, :away_score]]
1×5 DataFrame
 Row │ match_id  home_team             away_team     home_score  away_score 
     │ Int32?    String?               String?       Int32?      Int32?     
─────┼──────────────────────────────────────────────────────────────────────
   1 │ 13250813  st-patricks-athletic  sligo-rovers           3           0
=#


#=
julia> display_implied_market(market_params)
Match ID: 13250780
Market λ_Home: 1.5995
Market λ_Away: 1.2127
Market ρ: -0.048
Residual Error: 0.00306

julia> subset(matches, :match_id => ByRow(isequal(rand_matchid)))[:,[:match_id, :home_team, :away_team, :home_score, :away_score]]
1×5 DataFrame
 Row │ match_id  home_team     away_team  home_score  away_score 
     │ Int32?    String?       String?    Int32?      Int32?     
─────┼───────────────────────────────────────────────────────────
   1 │ 13250780  waterford-fc  cork-city           2           0
=#



#=
julia> display_implied_market(market_params)
Match ID: 13250835
Market λ_Home: 1.9418
Market λ_Away: 0.9316
Market ρ: 0.033
Residual Error: 0.002888

 Row │ match_id  home_team      away_team  home_score  away_score 
     │ Int32?    String?        String?    Int32?      Int32?     
─────┼────────────────────────────────────────────────────────────
   1 │ 13250835  galway-united  cork-city           2           1
=#

function display_implied_market( market_params)
  println("Match ID: ", market_params.match_id)
  println("Market λ_Home: ", round(market_params.λ_home, digits=4))
  println("Market λ_Away: ", round(market_params.λ_away, digits=4))
  println("Market ρ: ", round(market_params.ρ, digits=4))
  println("Residual Error: ", round(market_params.fit_error, digits=6))
end



using Revise
using BayesianFootball
using DataFrames
using BayesianFootball.Signals

using Plots
plotlyjs()  # Switch the backend to PlotlyJS

# Load DataStore again (Data is lightweight, models are heavy)
#
ds = Data.load_extra_ds()
transform!(ds.matches, :match_week => ByRow(w -> cld(w, 4)) => :match_month)


# 1. Load Experiments from Disk
# =============================
# This helper lists the folders it finds
saved_folders = Experiments.list_experiments("exp/funnel_basics"; data_dir="./data")
# saved_folders = Experiments.list_experiments("exp/grw_basics_pl_ch"; data_dir="./data")
# Load them all into a list
loaded_results = Vector{BayesianFootball.Experiments.ExperimentResults}([])
for folder in saved_folders
    try
        res = Experiments.load_experiment(folder)
        push!(loaded_results, res)
    catch e
        @warn "Could not load $folder: $e"
    end
end

if isempty(loaded_results)
    error("No results loaded! Did you run runner.jl?")
end

exp_res = loaded_results[1]

market_data = Data.prepare_market_data(ds)
#=
julia> market_data.df
73361×20 DataFrame
   Row │ match_id  market_name  market_line  selection  odds_open  odds_close  is_winner  prob_implied_open  prob_implied_close  overround_open  overround ⋯
       │ Int64     String       Float64      Symbol     Float64    Float64     Bool?      Float64            Float64             Float64         Float64   ⋯
───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
     1 │ 14035482  1X2                  0.0  home         2.5         2.38         false          0.4                 0.420168          1.09412          1 ⋯
     2 │ 14035482  1X2                  0.0  draw         3.4         3.3          false          0.294118            0.30303           1.09412          1
     3 │ 14035482  1X2                  0.0  away         2.5         2.7           true          0.4                 0.37037           1.09412          1
     4 │ 14035484  1X2                  0.0  home         1.67        1.57         false          0.598802            0.636943          1.09273          1
=#


## 

latents_raw = BayesianFootball.Experiments.extract_oos_predictions(ds, exp_res)

mkt_subset = DataFrames.subset(market_data.df, :selection => ByRow(isequal(:over_25)))

joined = innerjoin(
    select(latents_raw.df, :match_id, :λ_h, :λ_a, :r),
    select(mkt_subset, :match_id, :odds_close, :fair_odds_close, :is_winner, :prob_fair_close, :date),
    on = :match_id
)

j1 = joined[1,:]
#=
julia> j1 = joined[1,:]
DataFrameRow
 Row │ match_id  λ_h                                λ_a                                r                                  odds_close  fair_odds_close  is_winner  date       
     │ Any       Any                                Any                                Any                                Float64     Float64          Bool?      Date?      
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ 14035522  [1.07145, 1.05321, 1.83929, 1.32…  [1.46753, 0.756926, 1.01715, 0.9…  [21.8088, 11.0324, 5.84229, 19.7…        1.95          2.05405       true  2025-10-04
=#

params = BayesianFootball.Predictions.extract_params(exp_res.config.model, j1)
score_matrix = BayesianFootball.Predictions.compute_score_matrix(exp_res.config.model, j1)
#= 
julia> score_matrix = BayesianFootball.Predictions.compute_score_matrix(exp_res.config.model, j1)
BayesianFootball.Predictions.ScoreMatrix{Float64}([0.08490487005047988 0.11674485254503969 … 3.4733844792006373e-6 6.332582794571001e-7; 0.08671091910859444 0.1192281839588129 … 3.5472683773021314e-6 6.467285965128448e-7; … ; 1.7746525847899863e-7 2.440161020290135e-7 … 7.259949565105061e-12 1.3236148195151148e-12; 2.403130812671123e-8 3.3043234411044974e-8 … 9.830999401162772e-13 1.7923618313504025e-13;;; 0.17586978093487923 0.12457358650292005 … 3.952141259581116e-8 4.8516842067120095e-9; 0.1690855670828555 0.11976813415825419 … 3.799686577850822e-8 4.6645294662793335e-9; … ; 8.387608712645161e-7 5.941182697579195e-7 … 1.884861303986119e-13 2.3138727134872967e-14; 1.397586827275295e-7 9.89950647561144e-8 … 3.1406535759358184e-14 3.855494617299285e-15;;; 0.07912234952421263 0.06854562832662894 … 1.0358753792210913e-6 2.2122300570426161e-7; 0.11068346889331818 0.09588779865974133 … 1.4490757795088713e-6 3.094666654566543e-7; … ; 0.00012483532917974459 0.00010814790166675483 … 1.6343529322860388e-9 3.4903471528955204e-10; 4.304902191705586e-5 3.729442153696939e-5 … 5.636008305059378e-10 1.2036338756858375e-10;;; … ;;; 0.16173576486491814 0.08639697480622746 … 1.129578171107047e-9 9.283579398596041e-11; 0.1977395729324583 0.10562970357922032 … 1.381032236964329e-9 1.1350186071065678e-10; … ; 4.455196365833755e-6 2.3799033472727377e-6 …
 3.1115520843793616e-14 2.557267975521574e-15; 8.380329053833494e-7 4.4766541199880147e-7 … 5.852902587013276e-15 4.810281153497667e-16;;; 0.13071038881920385 0.13342644166896536 … 1.6580527766444939e-6 3.1386959475996495e-7; 0.11890651137606296 0.1213772894985645 … 1.5083213593748096e-6 2.8552541903572774e-7; … ; 5.238846627430281e-7 5.347705490451319e-7 … 6.645442856910286e-12 1.2579831510068685e-12; 8.83792504673255e-8 9.021569757195159e-8 … 1.1210850412035457e-12 2.1222153632897315e-13;;; 0.10492301856621583 0.11686407787085921 … 6.681798165418368e-7 1.0351384234327103e-7; 0.11275801257714951 0.12559075541715595 … 7.180753011778251e-7 1.1124360789797073e-7; … ; 4.6725839522902317e-7 5.204360514217971e-7 … 2.9756352139709708e-12 4.609828474080982e-13; 6.984391348171021e-8 7.779269654523942e-8 … 4.447860339370733e-13 6.89058697279106e-14])
=#



prob_over_under_25 = BayesianFootball.Predictions.compute_market_probs(
                                      score_matrix, 
                                      BayesianFootball.Data.MarketOverUnder(2.5)
)

#=
Dict{Symbol, Vector{Float64}} with 2 entries:
  :over_25  => [0.46216, 0.273935, 0.520264, 0.396491, 0.432767, 0.768654, 0.57772, 0.513206, 0.252551, 0.364658  …  0.491779, 0.332924, 0.479321, 0.438156, 0.273565, 0.479008, 0.356753, 0.294575, 0.360689, 0.407533]
  :under_25 => [0.53784, 0.726065, 0.479681, 0.603509, 0.567225, 0.231307, 0.422278, 0.486789, 0.747449, 0.635342  …  0.508219, 0.667075, 0.520672, 0.56184, 0.726435, 0.52099, 0.643247, 0.705424, 0.63931, 0.592467]
=#

stake_over_25 = BayesianFootball.Signals.compute_stake(
          BayesianFootball.Signals.BayesianKelly(),
          prob_over_under_25[:over_25],
          j1.odds_close)

function get_pnl(stake::Real, is_winner::Bool, odds::Real)::Real
        pnl = 0.0
        if stake > 0
            pnl = is_winner ? stake * (odds - 1.0) : -stake
        end
        return pnl
end

pnl = get_pnl(stake, j1.is_winner, j1.odds_close)


using SpecialFunctions
#=
SpecialFunctions.beta_inc
help?> beta_inc
search: beta_inc beta_inc_inv gamma_inc @static begin get_pnl beta

  beta_inc(a, b, x, y=1-x)

  Return a tuple (I_{x}(a,b), 1-I_{x}(a,b)) where I_{x}(a,b) is the regularized incomplete beta function given by

  I_{x}(a,b) = \frac{1}{B(a,b)} \int_{0}^{x} t^{a-1}(1-t)^{b-1} \mathrm{d}t,

  where B(a,b) = \Gamma(a)\Gamma(b)/\Gamma(a+b).

  External links: DLMF 8.17.1, Wikipedia

  See also: beta_inc_inv

Note: In Julia's beta_inc(a, b, x), this maps to a=r, b=k+1, x=p.
=#

using Roots, SpecialFunctions

"""
    solve_negbin_lambda_from_over_25(prob_over, r_val)

Finds the mean parameter (λ) of a Negative Binomial distribution given:
- prob_over: The implied probability of the outcome being > 2.5
- r_val: The fixed dispersion parameter (r)

Uses the relationship: P(X > 2) = 1 - I_p(r, 3) where p = r/(r+λ)
"""
function solve_negbin_lambda_from_over_25(prob_over::Float64, r_val::Real)
    # Sanity checks
    if prob_over <= 0.0 || prob_over >= 1.0
        return NaN
    end

    # We want to find λ such that:
    # (1 - CDF(2)) - prob_over = 0
    #
    # Note: beta_inc(a, b, x) returns a tuple: (Ix, 1-Ix)
    # The first element is the CDF (Ix). 
    # The second element is the Survival Function (1-Ix), which matches our Over prob directly.
    
    function objective(λ)
        if λ <= 1e-6 
             # Penalty/Slope to push back into positive range if solver wonders off
            return -1.0 
        end
        
        # 1. Convert Mean(λ) + Dispersion(r) to p
        p = r_val / (r_val + λ)
        
        # 2. Calculate P(X > 2.5) -> P(X >= 3) -> Survival at k=2
        # beta_inc(a, b, x) computes I_x(a, b).
        # We need I_p(r, k+1) where k=2 (Under 2.5 boundary)
        # We access the second element of the tuple [2] which is (1 - Ix)
        
        prob_calc = beta_inc(r_val, 3.0, p)[2]
        
        return prob_calc - prob_over
    end

    try
        # Solve for λ. 
        # Bracket: Goals usually fall between 0.1 and 10.0. 
        # Extending upper bound to 20.0 for extreme high-scoring implied odds.
        return find_zero(objective, (0.001, 25.0))
    catch e
        # Fallback if roots fails (e.g., probability implies impossible lambda)
        return NaN
    end
end


implied_lambda = solve_negbin_lambda_from_over_25(j1.prob_fair_close, mean(params.r))

mean(params.λ_h .+ params.λ_a)
# --------------------------------------------------

using DataFrames, Statistics, Dates, Roots, SpecialFunctions

# ------------------------------------------------------------------
# 1. Helper Functions
# ------------------------------------------------------------------

# Your robust solver (Verified)
function solve_negbin_lambda_from_over_25(prob_over::Float64, r_val::Real)
    if prob_over <= 0.0 || prob_over >= 1.0 return NaN end
    
    function objective(λ)
        if λ <= 1e-6 return -1.0 end
        p = r_val / (r_val + λ)
        # beta_inc return (cdf, survival). We want survival (1-cdf) for Over
        return beta_inc(r_val, 3.0, p)[2] - prob_over
    end

    try
        return find_zero(objective, (0.001, 25.0))
    catch
        return NaN
    end
end

# Simple EMA update
update_ema(old_val, new_val, alpha) = isnan(old_val) ? new_val : (alpha * new_val + (1.0 - alpha) * old_val)

# ------------------------------------------------------------------
# 2. Configuration & Initialization
# ------------------------------------------------------------------

# Sort by date to ensure we don't peek into the future
sort!(joined, :date)

# Hyperparameters (Tweak these)
ALPHA_BIAS = 0.00   # Fast adaptation (approx 20 matches)
ALPHA_TEMP = 0.00   # Slow adaptation for volatility
MIN_TEMP   = 0.2
MAX_TEMP   = 1.5

# Initial State
current_bias = 0.0
current_temp = 1.0
ema_error    = 0.0
ema_sq_error = 1.0 # Initial guess for squared error

# We will store results in a list of NamedTuples first (faster than growing DF)
results_log = []

# ------------------------------------------------------------------
# 3. The Execution Loop
# ------------------------------------------------------------------

println("Starting adaptive backtest on $(nrow(joined)) matches...")

for row in eachrow(joined)
    # --- A. PREDICT (Using YESTERDAY'S State) ---
    
    # 1. Get Raw Samples (Log Space)
    # We sum Home + Away lambda samples to get Total Goals distribution
    raw_total_lambda = row.λ_h .+ row.λ_a
    log_samples = log.(raw_total_lambda)
    
    # 2. Calibrate (Apply Bias & Temp)
    mu_raw = mean(log_samples)
    sigma_raw = std(log_samples)
    centered = log_samples .- mu_raw
    
    # Formula: calibrated = (centered * temp) + mean + bias
    calibrated_log = (centered .* current_temp) .+ mu_raw .+ current_bias
    calibrated_lambdas = exp.(calibrated_log)
    
    # 3. Compute Probability (Over 2.5) using Fixed r
    # We compute prob for every sample and average them (Monte Carlo integration)
    r_val = mean(row.r) # Use mean r if r is a vector, or row.r if scalar
    
    probs_over_samples = map(λ -> begin
        p = r_val / (r_val + λ)
        # beta_inc[2] is Survival (P(X > 2))
        SpecialFunctions.beta_inc(r_val, 3.0, p)[2]
    end, calibrated_lambdas)
    
    model_prob = mean(probs_over_samples)
    
    # --- B. TRADE (Kelly) ---
    stake   = BayesianFootball.Signals.compute_stake(
              BayesianFootball.Signals.BayesianKelly(),
              probs_over_samples,
              row.odds_close)

    pnl = get_pnl(stake, row.is_winner, row.odds_close)
    
    # --- C. MEASURE (The Inverse Problem) ---
    # Solve for Market Lambda
    lambda_mkt = solve_negbin_lambda_from_over_25(row.prob_fair_close, r_val)
    
    # --- D. ADAPT (Update State for TOMORROW) ---
    
    raw_error = 0.0
    
    if !isnan(lambda_mkt)
        # Error = Model_Raw_Mean - Market_Truth
        # If Model says 3.0 goals, Market says 2.5 -> Error is Positive
        raw_error = mu_raw - log(lambda_mkt)
        
        # 1. Update Error Tracker
        ema_error = update_ema(ema_error, raw_error, ALPHA_BIAS)
        
        # 2. Update Variance Tracker (Optional)
        # We compare realized error vs predicted volatility
        sq_error = raw_error^2
        ema_sq_error = update_ema(ema_sq_error, sq_error, ALPHA_TEMP)
        
        # 3. Control Law
        # Bias: Negative feedback. If we are too high (Pos Error), subtract bias.
        current_bias = -ema_error
        
        # Temp: Ratio of Realized Error / Predicted Variance
        # If Error is high, we increase temp to widen distribution (uncertainty)
        # (This is a simplified control law)
        target_temp = sqrt(ema_sq_error) / (sigma_raw + 1e-6)
        current_temp = clamp(target_temp, MIN_TEMP, MAX_TEMP)
    end

    # --- E. LOGGING ---
    push!(results_log, (
        match_id = row.match_id,
        date = row.date,
        
        # Prediction
        prob_model = model_prob,
        prob_mkt = row.prob_fair_close,
        lambda_model = exp(mu_raw), # The raw model center
        lambda_mkt = lambda_mkt,
        
        # State Used (For this prediction)
        bias_used = current_bias,
        temp_used = current_temp,
        
        # Result
        odds = row.odds_close,
        stake = stake,
        pnl = pnl,
        
        # Diagnostics
        raw_error = raw_error,
        ema_error = ema_error
    ))
end

# ------------------------------------------------------------------
# 4. Analysis
# ------------------------------------------------------------------

results_df = DataFrame(results_log)

println("Total PnL: ", sum(results_df.pnl))
println("ROI: ", round( 100*sum(results_df.pnl) / sum(results_df.stake), digits=2))


#=
julia> println("Total PnL: ", sum(results_df.pnl))
Total PnL: 1.160087001917458

julia> println("ROI: ", round( 100*sum(results_df.pnl) / sum(results_df.stake), digits=2))
ROI: 15.49
=#

# Quick check on the Bias Evolution
# using Plots
# plot(results_df.date, results_df.bias_used, title="Bias Correction Over Time", label="Bias")



#---------------------------------------------


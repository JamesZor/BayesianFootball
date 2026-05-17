using Revise
using BayesianFootball
using DataFrames
using BayesianFootball.Signals

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


julia> names(market_data.df)
20-element Vector{String}:
 "match_id"
 "market_name"
 "market_line"
 "selection"
 "odds_open"
 "odds_close"
 "is_winner"
 "prob_implied_open"
 "prob_implied_close"
 "overround_open"
 "overround_close"
 "prob_fair_open"
 "prob_fair_close"
 "fair_odds_open"
 "fair_odds_close"
 "vig_open"
 "vig_close"
 "clm_prob"
 "clm_odds"
 "date"

=#

latents = BayesianFootball.BackTesting.extract_oos_predictions(ds, exp_res)
latents.df 
names(latents.df)
#=
julia> latents.df 
708×4 DataFrame
 Row │ match_id  r                                  λ_a                                λ_h                               
     │ Any       Any                                Any                                Any                               
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ 11395624  [23.6678, 3.60027, 50.7031, 4.15…  [1.26134, 1.96007, 1.10567, 1.00…  [2.41723, 1.90512, 0.830434, 2.4…
   2 │ 11395610  [23.6678, 3.60027, 50.7031, 4.15…  [1.43671, 1.51957, 1.56802, 2.00…  [1.3092, 1.60782, 1.07593, 2.471…
   3 │ 11395618  [23.6678, 3.60027, 50.7031, 4.15…  [1.83271, 1.63113, 1.2208, 1.360…  [1.12939, 1.43397, 1.3722, 0.912…
   4 │ 11395621  [23.6678, 3.60027, 50.7031, 4.15…  [0.789443, 1.15163, 1.06453, 1.1…  [1.61686, 1.74078, 2.34622, 1.31

julia> names(latents.df)
4-element Vector{String}:
 "match_id"
 "r"
 "λ_a"
 "λ_h"
=#

ppd = BayesianFootball.BackTesting.model_inference(latents)
#=
julia> ppd.df
19116×5 DataFrame
   Row │ match_id  market_name  market_line  selection  distribution                      
       │ Int64     String       Float64      Symbol     Array…                            
───────┼──────────────────────────────────────────────────────────────────────────────────
     1 │ 11395624  1X2                  0.0  away       [0.194172, 0.412206, 0.416887, 0…
     2 │ 11395624  1X2                  0.0  home       [0.621307, 0.395342, 0.272958, 0…
     3 │ 11395624  1X2                  0.0  draw       [0.184482, 0.191857, 0.310156, 0…
     4 │ 11395624  BTTS                 0.0  btts_yes   [0.636544, 0.619006, 0.373228, 0…
     5 │ 11395624  BTTS                 0.0  btts_no    [0.363417, 0.380399, 0.626772, 0
=#




over_15_ppd = subset( ppd.df, :selection => ByRow(isequal(:over_15)))
market_data_over_15 = subset( market_data.df, :selection => ByRow(isequal(:over_15)))
#=
ulia> market_data_over_15 = subset( market_data.df, :selection => ByRow(isequal(:over_15)))
3598×20 DataFrame
  Row │ match_id  market_name  market_line  selection  odds_open  odds_close  is_winner  prob_implied_open  prob_implied_close  overround_open  overround_ ⋯
      │ Int64     String       Float64      Symbol     Float64    Float64     Bool?      Float64            Float64             Float64         Float64    ⋯
──────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    1 │ 14035482  OverUnder            1.5  over_15      1.25        1.28571       true           0.8                 0.777778         1.06667          1. ⋯
    2 │ 14035484  OverUnder            1.5  over_15      1.22222     1.25          true           0.818182            0.8              1.06818          1.
    3 │ 14035485  OverUnder            1.5  over_15      1.25        1.28571       true           0.8                 0.777778         1.06667          1.
    4 │ 14035486  OverUnder            1.5  over_15      1.2         1.2           true           0.833333            0.833333         1.0641           1.



=#


using DataFrames, Roots, Statistics

# --- 1. The Inverse Solver for Over/Under 1.5 ---
function solve_lambda_from_under15(prob_under)
    # Target: exp(-λ) * (1 + λ) = prob_under
    # We solve: exp(-λ) * (1 + λ) - prob_under = 0
    f(L) = (exp(-L) * (1 + L)) - prob_under
    
    # Sanity checks for probability
    if prob_under <= 0.001 || prob_under >= 0.999
        return NaN
    end
    
    try
        # Lambda for football usually between 0.5 and 10.0
        return find_zero(f, (0.01, 10.0))
    catch
        return NaN
    end
end

function compute_market_lambda_15(model_df, market_df)
    # --- Step A: Join Model & Market Data ---
    # We join on match_id to ensure we are comparing the same games
    # Renaming columns to avoid collision if necessary
    joined = innerjoin(
        model_df, 
        market_df, 
        on = :match_id, 
        makeunique = true
    )
    
    # Initialize output columns
    joined.lambda_mkt = fill(NaN, nrow(joined))
    joined.lambda_delta = fill(NaN, nrow(joined))
    joined.model_lambda_mean = fill(NaN, nrow(joined))
    
    # --- Step B: Iterate and Solve ---
    for i in 1:nrow(joined)
        # 1. Get Fair Probability of UNDER 1.5
        # The user provided 'fair_odds_close' for the OVER 1.5 selection.
        fair_odds_over = joined.fair_odds_close[i]
        
        # Prob(Over) = 1 / Fair_Odds_Over
        prob_over = 1.0 / fair_odds_over
        
        # Prob(Under) = 1 - Prob(Over)
        prob_under = 1.0 - prob_over
        
        # 2. Solve for Market Lambda
        lam_mkt = solve_lambda_from_under15(prob_under)
        joined.lambda_mkt[i] = lam_mkt
        
        # 3. Get Model Lambda
        # Assuming your PPD dataframe has a column :lambda which is a Vector of samples
        # We take the mean for the point estimate comparison
        # (Adjust column name ':lambda' to whatever your model output uses, e.g. :rate, :mu)
        if hasproperty(joined, :lambda)
            lam_model_vec = joined.lambda[i]
            lam_model_mean = mean(lam_model_vec)
            joined.model_lambda_mean[i] = lam_model_mean
            
            # 4. Calculate Delta (Log Difference)
            # Positive Delta = Model expects MORE goals than Market (Value on Over)
            if !isnan(lam_mkt) && lam_mkt > 0
                joined.distribution[i] = log(lam_model_mean) - log(lam_mkt)
            end
        end
    end
    
    return joined
end

# Usage:
# assuming 'over_15_ppd' has your model samples in a column named :lambda
# assuming 'market_data_over_15' has the 'fair_odds_close' column
final_analysis_15 = compute_market_lambda_15(over_15_ppd, market_data_over_15)

names(final_analysis_15)


#Select and re-order the essential columns
lean_analysis_15 = select(final_analysis_15, 
    # 1. Identifiers
    :date,
    :match_id,
    :selection,
    
    # 2. The "Ground Truth" (Did it win?)
    :is_winner,
    
    # 3. Market Pricing (For PnL calculations)
    :odds_close,       # What you actually bet at
    :fair_odds_close,  # What we used to solve the market lambda
    
    # 4. The Signal (The most important part)
    :lambda_mkt,          # Market's implied goal rate
    :model_lambda_mean,   # Your model's predicted goal rate
    :lambda_delta         # The Edge (Log Difference)
)

# Optional: Preview the clean data
first(lean_analysis_15, 5)

valid_rows = filter(row -> !isnan(row.lambda_delta), final_analysis_15)

cor(valid_rows.lambda_delta, valid_rows.is_winner)



#- 
using Roots, Statistics, DataFrames

function solve_negbin_lambda_from_under15(prob_under, r_val)
    # Objective: Find Lambda such that NegBinCDF(1 | r, lambda) = prob_under
    
    function objective(lam)
        if lam <= 0 return 1.0 end # Penalty for invalid lambda
        
        # Calculate p parameter for NegBin
        # p = r / (r + lambda)
        p = r_val / (r_val + lam)
        
        # P(Y=0) = p^r
        p0 = p^r_val
        
        # P(Y=1) = r * (1-p) * p^r
        p1 = r_val * (1.0 - p) * p0
        
        # CDF(1) = P(0) + P(1)
        cdf_1 = p0 + p1
        
        return cdf_1 - prob_under
    end
    
    try
        # Search for Lambda between 0.01 and 10.0
        return find_zero(objective, (0.001, 10.0))
    catch
        return NaN
    end
end


function compute_negbin_delta(latents_df, market_df)
    # 1. Summarize Latents (Vectors -> Means)
    # We need scalar values to solve the inverse problem efficiently
    
    # Create a clean summary dataframe
    model_summary = select(latents_df, :match_id, 
        # Calculate Total Lambda (Home + Away) per sample, then take mean
        [:λ_h, :λ_a] => ((h, a) -> mean.(h .+ a)) => :model_lambda_mean,
        
        # Calculate Mean Dispersion (r)
        :r => (r -> mean.(r)) => :model_r_mean
    )
    
    # 2. Join with Market Data
    joined = innerjoin(model_summary, market_df, on = :match_id, makeunique = true)
    
    joined.lambda_mkt = fill(NaN, nrow(joined))
    joined.lambda_delta = fill(NaN, nrow(joined))
    
    # 3. Solve Row by Row
    for i in 1:nrow(joined)
        # Market Probability of Under 1.5
        fair_odds_over = joined.fair_odds_close[i]
        prob_over = 1.0 / fair_odds_over
        prob_under = 1.0 - prob_over
        
        # Model Dispersion (r)
        r_val = joined.model_r_mean[i]
        
        # Solve for Market Lambda (using Model's r)
        lam_mkt = solve_negbin_lambda_from_under15(prob_under, r_val)
        joined.lambda_mkt[i] = lam_mkt
        
        # Calculate Delta
        lam_model = joined.model_lambda_mean[i]
        
        if !isnan(lam_mkt) && lam_mkt > 0
            joined.lambda_delta[i] = log(lam_model) - log(lam_mkt)
        end
    end
    
    return joined
end

# Usage:
# assuming 'latents.df' contains the vectors as shown in your snippet
final_signal_df = compute_negbin_delta(latents.df, market_data_over_15)

# Preview
first(select(final_signal_df, :match_id, :is_winner, :model_r_mean, :model_lambda_mean, :lambda_mkt, :lambda_delta), 5)
#=
julia> first(select(final_signal_df, :match_id, :is_winner, :model_r_mean, :model_lambda_mean, :lambda_mkt, :lambda_delta), 5)
5×6 DataFrame
 Row │ match_id  is_winner  model_r_mean  model_lambda_mean  lambda_mkt  lambda_delta 
     │ Any       Bool?      Float64       Float64            Float64     Float64      
─────┼────────────────────────────────────────────────────────────────────────────────
   1 │ 14035522       true       14.2405            2.49201     2.74258   -0.0958109
   2 │ 14035524       true       14.2405            2.66499     2.85745   -0.0697301
   3 │ 14035525       true       14.2405            2.49093     2.49975   -0.00353116
   4 │ 14035527       true       14.2405            2.60225     2.9624    -0.129623
   5 │ 14035528      false       14.2405            2.4333      2.61506   -0.0720384
=#
describe(final_signal_df.lambda_delta)
#=
julia> describe(final_signal_df.lambda_delta)
Summary Stats:
Length:         684
Missing Count:  0
Mean:           -0.022324
Std. Deviation: 0.109522
Minimum:        -0.492550
1st Quartile:   -0.097096
Median:         -0.033112
3rd Quartile:   0.041573
Maximum:        0.418314
Type:           Float64

=#


valid_rows = filter(row -> !isnan(row.lambda_delta), final_signal_df)
cor(valid_rows.lambda_delta, valid_rows.is_winner)
#=
julia> cor(valid_rows.lambda_delta, valid_rows.is_winner)
0.003935007438210145
=#
using DataFrames, Dates, Statistics, Plots, StatsPlots

function analyze_monthly_correlation(df)
    # 1. Ensure Date Format
    # If date is a string, parse it. If it's already Date, this is safe.
    df.dt = try
        Date.(df.date)
    catch
        Date.(df.date, "yyyy-mm-dd") # Adjust format if needed
    end
    
    # 2. Create Year-Month Key (e.g., "2025-10")
    # We use the first day of the month as the anchor for plotting
    df.month_start = floor.(df.dt, Month)
    
    # 3. Group and Aggregate
    monthly_stats = combine(groupby(df, :month_start),
        # Calculate Correlation between Signal (Delta) and Outcome (Win/Loss)
        [:lambda_delta, :is_winner] => ((d, w) -> cor(d, w)) => :signal_correlation,
        
        # Count matches (to ignore months with too little data)
        :match_id => length => :match_count
    )
    
    # Sort by time
    sort!(monthly_stats, :month_start)
    
    return monthly_stats
end

# Run the analysis
monthly_trends = analyze_monthly_correlation(valid_rows) # Use valid_rows from before

# Display the table
display(monthly_stats)

#=
julia> monthly_trends = analyze_monthly_correlation(valid_rows) # Use valid_rows from before
21×3 DataFrame
 Row │ month_start  signal_correlation  match_count 
     │ Date         Float64             Int64       
─────┼──────────────────────────────────────────────
   1 │ 2023-10-01          -0.448976             21
   2 │ 2023-11-01           0.0841837            24
   3 │ 2023-12-01           0.163408             35
   4 │ 2024-01-01          -0.0371449            33
   5 │ 2024-02-01           0.15238              45
   6 │ 2024-03-01          -0.126246             52
   7 │ 2024-04-01          -0.234391             40
   8 │ 2024-05-01         NaN                    10
   9 │ 2024-10-01          -0.259448             25
  10 │ 2024-11-01           0.357889             32
  11 │ 2024-12-01          -0.118502             44
  12 │ 2025-01-01          -0.00959248           31
  13 │ 2025-02-01           0.112541             42
  14 │ 2025-03-01           0.0674527            55
  15 │ 2025-04-01          -0.225815             40
  16 │ 2025-05-01           0.361618             10
  17 │ 2025-10-01          -0.172921             25
  18 │ 2025-11-01           0.260463             40
  19 │ 2025-12-01           0.119182             39
  20 │ 2026-01-01          -0.125261             33
  21 │ 2026-02-01           0.124968              8

=#
using Turing, Distributions, LogExpFunctions, StatsPlots

# 1. Prepare Data for Turing
# We need strictly clean vectors (no missing values)
clean_data = filter(row -> !isnan(row.lambda_delta), final_signal_df)
sort!(clean_data, :date)

# Create a "Week Index" so the Random Walk knows how much time passed
# (Simplified: Just use row index / number of games per week, or just raw sequence)
# Let's use a simple sequential index for the Random Walk
num_matches = nrow(clean_data)
week_idx = 1:num_matches # Ideally this maps to actual weeks, but this works for trend

# Inputs
signals_vec = abs.(clean_data.lambda_delta) # Magnitude of your conviction
outcomes_vec = Int.(clean_data.is_winner)   # 1 = Over Won, 0 = Over Lost

sig_vec = Vector{Float64}(abs.(clean_data.lambda_delta))
out_vec = Vector{Int}(clean_data.is_winner)




@model function regime_switching_tracker_ncp(signals, outcomes)
    n = length(outcomes)
    
    # --- 1. Volatility Priors ---
    # NCP allows us to sample this much faster
    σ_beta ~ Truncated(Normal(0.05, 0.05), 0.001, 0.2)
    
    # --- 2. Latent State (NCP Implementation) ---
    # Step A: The "Raw" Innovations (Standard Normal noise)
    # These are independent, making NUTS sampling very efficient
    z_beta ~ filldist(Normal(0, 1), n)
    
    # Step B: The Initial State
    # Where does the efficacy start at Week 1?
    beta_init ~ Normal(0, 1)
    
    # Step C: Reconstruct the Random Walk
    # We scale the noise by sigma and accumulate it
    # beta[t] = beta_init + sum(noise[1:t] * sigma)
    # Note: We add beta_init to the whole vector
    beta = beta_init .+ cumsum(z_beta .* σ_beta)
    
    # --- 3. Likelihood (Vectorized) ---
    # CRITICAL FIX: Use element-wise multiplication (.*)
    # We interact the Regime (beta) with the Signal magnitude
    # This allows Beta to flip the signal from "Good" to "Bad"
    outcomes ~ BernoulliLogit(beta .* signals)
end

# 2. Run the Tracker
chain_regime = sample(
                    model_regime,
                    NUTS(0.65),
                    MCMCThreads(),
                    500,
                    8,
                    adtype = AutoReverseDiff(compile=true),
)



# 1. Prepare Data (Vectors)
# Ensure they are standard Vectors, not DataFrame columns, for speed

# 2. Sample
println("Sampling NCP Model...")
# You can likely reduce target acceptance to 0.65 or 0.8 with NCP
model_ncp = regime_switching_tracker_ncp(sig_vec, out_vec)
chain_ncp = sample(model_ncp, NUTS(0.65), 1000)

# 3. Extract the Beta (Efficacy) Curve
beta_est = mean(group(chain_ncp, :beta))
plot(beta_est, title="Regime Efficacy (NCP)", ylabel="Beta", xlabel="Match Index")
hline!([0.0], color=:red)



# 3. Plot the "Beta" over time
beta_mean = mean(group(chain_regime, :beta))
plot(beta_mean, title="The Regime Tracker", label="Strategy Efficacy (Beta)", 
     ylabel="Correlation (Positive=Follow, Negative=Fade)", xlabel="Match Sequence",
     color=:blue, lw=2)
hline!([0.0], color=:red, label="Neutral")



# ---- 
using Dates, DataFrames

function add_time_indices(df, resolution=:Monthly)
    # Ensure dates are sorted!
    sort!(df, :date)
    
    # 1. Determine the Start Date
    start_date = minimum(df.date)
    
    # 2. Define the Step Function
    get_index = if resolution == :Monthly
        d -> (year(d) - year(start_date)) * 12 + month(d) - month(start_date) + 1
    elseif resolution == :BiWeekly
        d -> floor(Int, Dates.value(d - start_date) / 14) + 1
    else
        error("Unknown resolution")
    end
    
    # 3. Create the Index Vector
    # This vector maps every match row to a period integer (1, 1, 1, 2, 2...)
    period_indices = map(get_index, df.date)
    
    return period_indices, maximum(period_indices)
end

# Usage:
# Assume 'clean_data' is your filtered dataframe
period_idx, num_periods = add_time_indices(clean_data, :Monthly)

# Extract vectors for the model
signals_vec = Vector{Float64}(abs.(clean_data.lambda_delta))
outcomes_vec = Vector{Int}(clean_data.is_winner)



@model function regime_switching_grouped_ncp(signals, outcomes, period_idx, num_periods)
    # --- 1. Volatility Prior ---
    # How much can the regime change from Month to Month?
    # We expect smoother transitions now that we are grouped.
    σ_beta ~ Truncated(Normal(0.1, 0.1), 0.001, 0.5)
    
    # --- 2. Latent State (The Monthly Regime) ---
    # We only generate 'num_periods' steps (e.g., ~30 for 2.5 years)
    
    # A. Raw Innovations (Standard Normal)
    z_beta ~ filldist(Normal(0, 1), num_periods)
    
    # B. Initial State (Where did we start in Month 1?)
    beta_init ~ Normal(0, 1)
    
    # C. Reconstruct the Walk (Vectorized)
    # beta_steps[t] is the efficacy for Month t
    beta_steps = beta_init .+ cumsum(z_beta .* σ_beta)
    
    # --- 3. Likelihood (Mapped) ---
    # We map the Monthly Beta to the specific Match
    # beta_steps[period_idx] creates a vector of length 'num_matches'
    # where each match gets its corresponding month's beta.
    
    # Element-wise multiplication: Match_Beta * Match_Signal
    match_logits = beta_steps[period_idx] .* signals
    
    outcomes ~ arraydist(BernoulliLogit.(match_logits))
end


model_grouped = regime_switching_grouped_ncp(
    signals_vec, 
    outcomes_vec, 
    period_idx, 
    num_periods
)

chain_regime = sample(
                    model_grouped,
                    NUTS(0.65),
                    MCMCThreads(),
                    300,
                    12,
                    adtype = AutoReverseDiff(compile=true),
)


describe(chain_regime)
#=


julia> describe(chain_regime)
Chains MCMC chain (300×45×12 Array{Float64, 3}):

Iterations        = 151:1:450
Number of chains  = 12
Samples per chain = 300
Wall duration     = 19.96 seconds
Compute duration  = 163.75 seconds
parameters        = σ_beta, z_beta[1], z_beta[2], z_beta[3], z_beta[4], z_beta[5], z_beta[6], z_beta[7], z_beta[8], z_beta[9], z_beta[10], z_beta[11], z_beta[12], z_beta[13], z_beta[14], z_beta[15], z_beta[16], z_beta[17], z_beta[18], z_beta[19], z_beta[20], z_beta[21], z_beta[22], z_beta[23], z_beta[24], z_beta[25], z_beta[26], z_beta[27], z_beta[28], z_beta[29], beta_init
internals         = n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size, lp, logprior, loglikelihood

Summary Statistics

  parameters      mean       std      mcse    ess_bulk    ess_tail      rhat   ess_per_sec 
      Symbol   Float64   Float64   Float64     Float64     Float64   Float64       Float64 

      σ_beta    0.3499    0.0783    0.0016   2274.9872   1602.8949    1.0058       13.8933
   z_beta[1]    1.4307    1.0189    0.0152   4528.4916   2918.0150    1.0008       27.6554
   z_beta[2]    1.2988    0.9533    0.0145   4348.0757   2891.4698    1.0017       26.5536
   z_beta[3]    1.2423    1.0065    0.0152   4418.9550   3032.4006    1.0036       26.9865
   z_beta[4]    1.0053    1.0004    0.0131   5815.9721   2779.0212    1.0018       35.5180
   z_beta[5]    0.8938    1.0154    0.0126   6490.2402   2567.5282    1.0040       39.6358
   z_beta[6]    0.6713    1.0216    0.0129   6306.8460   2900.6439    1.0060       38.5158
   z_beta[7]    0.5159    0.9891    0.0144   4754.6697   2526.1094    1.0021       29.0367
   z_beta[8]    0.5415    0.9970    0.0132   5633.0167   2815.0548    1.0022       34.4007
   z_beta[9]    0.4523    0.9661    0.0125   6018.4521   2969.1624    1.0008       36.7546
  z_beta[10]    0.4623    0.9729    0.0132   5389.0013   2587.9199    1.0043       32.9105
  z_beta[11]    0.4663    1.0273    0.0147   4870.7355   2477.6507    1.0032       29.7455
  z_beta[12]    0.4327    0.9856    0.0143   4749.0470   2640.4088    1.0029       29.0023
  z_beta[13]    0.4666    1.0046    0.0140   5160.2042   2520.5653    1.0042       31.5133
  z_beta[14]    0.4286    1.0037    0.0147   4659.8456   2697.3717    1.0028       28.4576
  z_beta[15]    0.4132    0.9663    0.0149   4191.3368   2887.3981    1.0075       25.5964
  z_beta[16]    0.2348    0.9560    0.0140   4665.1734   2655.4787    1.0005       28.4901
  z_beta[17]    0.2093    0.9877    0.0129   5862.0376   2753.3045    1.0014       35.7994
  z_beta[18]    0.0636    0.9924    0.0151   4293.9168   2905.8338    1.0002       26.2229
  z_beta[19]    0.1927    0.9810    0.0135   5208.5359   2692.0588    1.0064       31.8084
  z_beta[20]    0.0354    1.0095    0.0134   5638.5427   2888.0729    1.0025       34.4345
  z_beta[21]    0.0146    1.0121    0.0134   5656.6656   3022.0490    1.0063       34.5452
  z_beta[22]    0.0305    0.9855    0.0136   5229.8005   2736.0422    1.0056       31.9383
  z_beta[23]    0.0332    0.9993    0.0133   5677.5118   3027.8427    1.0046       34.6725
  z_beta[24]   -0.0084    0.9904    0.0147   4528.5447   2815.2699    1.0018       27.6557
  z_beta[25]    0.0632    0.9723    0.0133   5345.1997   2551.2508    1.0034       32.6430
  z_beta[26]   -0.0298    0.9721    0.0137   5068.3655   3071.4476    1.0030       30.9524
  z_beta[27]   -0.0372    0.9785    0.0124   6264.8531   2921.2616    1.0050       38.2593
  z_beta[28]    0.1140    1.0059    0.0129   6077.8001   2728.3199    1.0025       37.1170
  z_beta[29]    0.0155    0.9972    0.0126   6284.3673   2560.2462    1.0040       38.3785
   beta_init    4.1634    0.8819    0.0155   3263.5771   2499.3712    1.0024       19.9306


Quantiles

  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

      σ_beta    0.1779    0.2996    0.3548    0.4064    0.4853
   z_beta[1]   -0.5270    0.7529    1.4122    2.1343    3.4353
   z_beta[2]   -0.5767    0.6472    1.3124    1.9478    3.1431
   z_beta[3]   -0.7117    0.5635    1.2423    1.9128    3.2627
   z_beta[4]   -0.9312    0.3205    1.0150    1.6577    2.9551
   z_beta[5]   -1.1186    0.2088    0.8954    1.5800    2.8885
   z_beta[6]   -1.3388   -0.0101    0.6724    1.3470    2.7003
   z_beta[7]   -1.3956   -0.1508    0.4969    1.1810    2.4697
   z_beta[8]   -1.4242   -0.1259    0.5496    1.2237    2.4442
   z_beta[9]   -1.4043   -0.2251    0.4608    1.1277    2.3198
  z_beta[10]   -1.4477   -0.2078    0.4674    1.1221    2.3699
  z_beta[11]   -1.5100   -0.2020    0.4704    1.1202    2.5380
  z_beta[12]   -1.4864   -0.2345    0.4453    1.0891    2.3845
  z_beta[13]   -1.5401   -0.2096    0.4574    1.1412    2.4366
  z_beta[14]   -1.5660   -0.2412    0.4180    1.1073    2.3987
  z_beta[15]   -1.4369   -0.2459    0.4134    1.0607    2.3000
  z_beta[16]   -1.6316   -0.4352    0.2311    0.8961    2.0523
  z_beta[17]   -1.7225   -0.4548    0.2093    0.8868    2.1448
  z_beta[18]   -1.9093   -0.5881    0.0615    0.7219    2.0034
  z_beta[19]   -1.7354   -0.4757    0.1774    0.8465    2.0729
  z_beta[20]   -1.9503   -0.6448    0.0308    0.7040    2.0664
  z_beta[21]   -1.9739   -0.6617    0.0180    0.6950    2.0334
  z_beta[22]   -1.9208   -0.6344    0.0177    0.7091    1.9386
  z_beta[23]   -1.9289   -0.6319    0.0249    0.7008    1.9669
  z_beta[24]   -1.9738   -0.6737    0.0021    0.6452    1.9422
  z_beta[25]   -1.8808   -0.5862    0.0697    0.7073    1.9653
  z_beta[26]   -1.9409   -0.6878   -0.0297    0.6359    1.8601
  z_beta[27]   -1.9765   -0.6718   -0.0366    0.6382    1.8188
  z_beta[28]   -1.8256   -0.5751    0.1034    0.8081    2.0696
  z_beta[29]   -1.9035   -0.6757    0.0155    0.6928    1.9542
   beta_init    2.4790    3.5750    4.1673    4.7634    5.8928

=#

using DataFrames, Statistics, MCMCChains, UnicodePlots, Dates

function summarize_regime_robust(chain, dates_vec)
    # 1. Identify Parameter Names
    # We look for parameters starting with "beta_steps" (from the new model)
    # or "beta" (from the old model) that have brackets [i].
    all_names = string.(names(chain))
    
    # Filter for the vector components (e.g. "beta_steps[1]", "beta_steps[2]")
    # We exclude "beta_init" or "sigma_beta" to avoid mixing scalars with the vector.
    param_names = filter(x -> (startswith(x, "z_beta") || startswith(x, "beta")) && occursin("[", x), all_names)
    
    # Sort them naturally (so [10] comes after [9], not after [1])
    # We extract the number inside the brackets to sort
    parse_idx(s) = parse(Int, match(r"\[(\d+)\]", s).captures[1])
    sort!(param_names, by=parse_idx)
    
    if isempty(param_names)
        error("No beta parameters found in chain! Check if model used 'beta' or 'beta_steps'.")
    end

    n_periods = length(param_names)
    println("Found $n_periods time steps in the chain.")

    # 2. Extract and Pool Chains
    # We iterate through each time step t
    means = Float64[]
    lower = Float64[]
    upper = Float64[]
    prob_pos = Float64[]
    
    for name in param_names
        # Extract data for this specific time step t across ALL chains
        # chain[Symbol(name)] returns a 3D AxisArray. 
        # vec() flattens (Iterations * Chains) into a single vector.
        samples_t = vec(chain[Symbol(name)])
        
        push!(means, mean(samples_t))
        push!(lower, quantile(samples_t, 0.05))
        push!(upper, quantile(samples_t, 0.95))
        push!(prob_pos, mean(samples_t .> 0))
    end
    
    # 3. Align Dates
    # Ensure we don't index out of bounds if dates_vec is shorter/longer
    n_plot = min(length(dates_vec), n_periods)
    
    # 4. Create DataFrame
    df = DataFrame(
        :date => dates_vec[1:n_plot],
        :beta_mean => means[1:n_plot],
        :beta_lower => lower[1:n_plot],
        :beta_upper => upper[1:n_plot],
        :prob_edge => prob_pos[1:n_plot]
    )
    
    # 5. Plot (Unicode)
    plt = lineplot(
        df.date, 
        df.beta_mean,
        title = "Strategy Edge (Beta) over Time",
        ylabel = "Signal Efficacy",
        xlabel = "Date",
        color = :blue,
        height = 15,
        width = 60
    )
    
    # Add Zero Line
    lineplot!(plt, df.date, zeros(length(df.date)), color=:red)
    
    println(plt)
    return df
end

# --- Usage ---

# 1. Define the correct dates for the x-axis
# (Use the unique months from your clean_data, NOT period_idx)
unique_months = sort(unique(floor.(clean_data.date, Month)))

# 2. Run with the Corrected Chain
# (Make sure you pass 'chain_corrected' or 'chain_grouped', NOT 'chain_regime' if that was the old one)
regime_df = summarize_regime_robust(chain_regime, unique_months)
#=

julia> regime_df = summarize_regime_robust(chain_regime, unique_months)
Found 29 time steps in the chain.
                     ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀Strategy Edge (Beta) over Time⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ 
                     ┌────────────────────────────────────────────────────────────┐ 
                   2 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                     │⠒⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                     │⠀⠀⠓⠢⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                     │⠀⠀⠀⠀⠈⢢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
   Signal Efficacy   │⠀⠀⠀⠀⠀⠀⠣⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                     │⠀⠀⠀⠀⠀⠀⠀⠈⠱⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⢄⠀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠀⠈⠉⠉⠉⠉⠒⠒⠒⠒⠤⠤⠤⠤⠤⠤⠤⠤⡤⠤⠤⠤⣀⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠢⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠑⠒⠒⠒⠒⠒⠒⠒⠒⠒⠢⡀⠀⠀⡠⡀⠀⠀⠀│ 
                   0 │⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣈⣢⣊⣀⣈⣦⣀⣀│ 
                     └────────────────────────────────────────────────────────────┘ 
                     ⠀2023-10-01⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀2026-02-01⠀ 
                     ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀Date⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ 
21×5 DataFrame
 Row │ date        beta_mean  beta_lower  beta_upper  prob_edge 
     │ Date        Float64    Float64     Float64     Float64   
─────┼──────────────────────────────────────────────────────────
   1 │ 2023-10-01  1.43069     -0.250858     3.09714   0.920556
   2 │ 2023-11-01  1.29877     -0.283886     2.88182   0.9075
   3 │ 2023-12-01  1.2423      -0.399589     2.89748   0.894167
   4 │ 2024-01-01  1.00527     -0.632635     2.64019   0.8375
   5 │ 2024-02-01  0.893754    -0.796853     2.56266   0.811389
   6 │ 2024-03-01  0.671265    -0.985578     2.3575    0.746944
   7 │ 2024-04-01  0.515869    -1.10836      2.14874   0.698056
   8 │ 2024-05-01  0.541489    -1.1047       2.13448   0.706944
   9 │ 2024-10-01  0.452322    -1.13949      1.99811   0.679722
  10 │ 2024-11-01  0.462284    -1.17193      2.06998   0.6825
  11 │ 2024-12-01  0.466266    -1.22479      2.18552   0.6775
  12 │ 2025-01-01  0.43271     -1.21893      2.03091   0.671389
  13 │ 2025-02-01  0.466591    -1.17025      2.09935   0.678333
  14 │ 2025-03-01  0.42864     -1.18481      2.08393   0.661667
  15 │ 2025-04-01  0.413199    -1.16393      2.02274   0.658333
  16 │ 2025-05-01  0.234765    -1.28323      1.76555   0.586944
  17 │ 2025-10-01  0.209306    -1.38147      1.80625   0.588333
  18 │ 2025-11-01  0.0635864   -1.56145      1.71097   0.522222
  19 │ 2025-12-01  0.192704    -1.39926      1.83086   0.580278
  20 │ 2026-01-01  0.0354333   -1.608        1.69462   0.511944
  21 │ 2026-02-01  0.0145608   -1.62713      1.6867    0.508611



=#



# 1. Define the Corrected Model (Separates Base Rate from Edge)
@model function regime_switching_alpha_beta(signals, outcomes, period_idx, num_periods)
    # Volatility Priors
    σ_alpha ~ Truncated(Normal(0.1, 0.1), 0.001, 0.5) 
    σ_beta  ~ Truncated(Normal(0.1, 0.1), 0.001, 0.5)
    
    # A. Base Win Rate (Alpha) - The "Market Average"
    z_alpha ~ filldist(Normal(0, 1), num_periods)
    alpha_init ~ Normal(1.0, 1) # Start around 73% win rate
    alpha_steps = alpha_init .+ cumsum(z_alpha .* σ_alpha)
    
    # B. Your Edge (Beta) - The "Value Add"
    z_beta ~ filldist(Normal(0, 1), num_periods)
    beta_init ~ Normal(0, 1)    # Start Neutral (0.0)
    beta_steps = beta_init .+ cumsum(z_beta .* σ_beta)
    
    # Likelihood
    base_logits = alpha_steps[period_idx]
    signal_logits = beta_steps[period_idx] .* signals
    
    outcomes ~ arraydist(BernoulliLogit.(base_logits .+ signal_logits))
end

# 2. Sample the NEW model
println("Sampling Corrected Model...")
model_corrected = regime_switching_alpha_beta(
    signals_vec, 
    outcomes_vec, 
    period_idx, 
    num_periods
)
chain_corrected = sample(
                    model_corrected,
                    NUTS(0.65),
                    MCMCThreads(),
                    300,
                    12,
                    adtype = AutoReverseDiff(compile=true),
)

# 3. Analyze the CORRECTED chain
# Note: We pass 'chain_corrected' here, NOT 'chain_regime'

describe(chain_corrected)

#=
                                                                                                                                   
      σ_alpha    0.0751    0.0576    0.0025    733.3747    326.2298    1.0102        1.2881                                           
       σ_beta    0.1273    0.0757    0.0013   2916.9764   1757.3493    1.0093        5.1232                                           
   z_alpha[1]    0.0470    0.9876    0.0181   2976.5244   1411.6597    1.0073        5.2278                                           
   z_alpha[2]    0.0470    0.9927    0.0152   4299.4281   2640.1809    1.0144        7.5513                                           
   z_alpha[3]    0.1772    1.0088    0.0195   2637.0145   2456.3972    1.0066        4.6315                                           
   z_alpha[4]   -0.0671    1.0063    0.0210   2255.6507   2120.7215    1.0049        3.9617                                           
   z_alpha[5]   -0.0760    0.9764    0.0167   3438.4715   2116.9352    1.0112        6.0391                                           
   z_alpha[6]   -0.0502    0.9408    0.0150   3966.4377   2763.5232    1.0012        6.9664                                           
   z_alpha[7]   -0.0299    0.9694    0.0244   1589.1758    509.4796    1.0062        2.7911                                           
   z_alpha[8]    0.1403    0.9643    0.0187   2659.5531   2324.6085    1.0046        4.6711                                           
   z_alpha[9]    0.0630    0.9382    0.0173   2943.7171   1997.8238    1.0029        5.1702                                           
  z_alpha[10]    0.0688    0.9679    0.0229   1928.2879    505.0014    1.0085        3.3867                                           
  z_alpha[11]    0.0590    0.9670    0.0189   2629.0667   2210.6500    1.0063        4.6175                                           
  z_alpha[12]   -0.0183    0.9451    0.0163   3342.2397   2442.1838    1.0031        5.8701                                           
  z_alpha[13]    0.0309    0.9748    0.0192   2583.6846   2404.4729    1.0079        4.5378                                           
  z_alpha[14]    0.2090    0.9581    0.0159   3652.8480   1876.9299    1.0032        6.4157                                           
  z_alpha[15]    0.1619    0.9644    0.0178   2931.6155   2003.2078    1.0041        5.1489                                           
  z_alpha[16]   -0.1555    0.9739    0.0218   1975.0081   2109.9412    1.0051        3.4688                                           
  z_alpha[17]    0.0082    0.9479    0.0188   2567.0195   2256.8795    1.0048        4.5086                                           
  z_alpha[18]   -0.0971    0.9578    0.0186   2642.2762   2439.9270    1.0076        4.6408                                           
  z_alpha[19]    0.2924    0.9703    0.0177   2996.8773   1776.9083    1.0074        5.2636                                           
  z_alpha[20]    0.1077    0.9683    0.0177   3015.7437   2159.5873    1.0059        5.2967                                           
  z_alpha[21]    0.0977    0.9316    0.0180   2669.4265   1775.7046    1.0035        4.6884                                           
  z_alpha[22]    0.0565    0.9690    0.0201   2387.1489    573.2541    1.0041        4.1927                                           
  z_alpha[23]    0.1249    0.9902    0.0178   3068.3480   1729.0257    1.0063        5.3891                                           
  z_alpha[24]    0.0549    0.9786    0.0165   3523.8287   2547.6068    1.0113        6.1891                                           
  z_alpha[25]    0.0672    0.9732    0.0205   2287.0993   1155.6311    1.0030        4.0169                                           
  z_alpha[26]    0.2158    1.0053    0.0167   3618.4096   1985.0364    1.0072        6.3552                                           
  z_alpha[27]    0.1438    0.9852    0.0177   3149.7344   2207.6514    1.0049        5.5320                                           
  z_alpha[28]    0.2294    0.9710    0.0168   3368.2205   2034.3882    1.0024        5.9158                                           
  z_alpha[29]   -0.0059    1.0054    0.0301   1388.4721    535.5478    1.0093        2.4386                                           
   alpha_init    1.2156    0.2009    0.0054   1573.0927   1064.8513    1.0027        2.7629                                           
    z_beta[1]    0.0650    1.0137    0.0214   2244.5567   2542.0631    1.0061        3.9422                                           
    z_beta[2]   -0.0081    1.0377    0.0195   2835.6144   1926.7950    1.0064        4.9803                                           
    z_beta[3]   -0.0095    0.9958    0.0171   3394.8923   2280.8507    1.0019        5.9626                                           
    z_beta[4]   -0.0559    0.9638    0.0175   3034.7388   2243.4732    1.0074        5.3301                                           
    z_beta[5]   -0.0457    1.0083    0.0144   4956.0954   2349.3782    1.0079        8.7046                                           
    z_beta[6]   -0.0698    0.9943    0.0165   3606.4525   1990.3002    1.0069        6.3342                                           
    z_beta[7]   -0.0846    0.9626    0.0167   3312.2842   2365.2799    1.0042        5.8175                                           
    z_beta[8]   -0.0610    0.9758    0.0193   2609.8158   2501.1915    1.0074        4.5837                                           
    z_beta[9]   -0.0928    0.9790    0.0198   2451.6129   2081.8597    1.0048        4.3059                                           
   z_beta[10]   -0.0525    0.9901    0.0163   3695.5911   2399.5421    1.0020        6.4907                                           
   z_beta[11]   -0.0730    1.0572    0.0248   1813.2240   1240.4667    1.0026        3.1846                                           
   z_beta[12]   -0.0927    0.9715    0.0162   3653.9986   2399.7610    1.0056        6.4177                                           
   z_beta[13]   -0.0789    1.0237    0.0237   1857.0681   1028.2439    1.0091        3.2617                                           
   z_beta[14]   -0.0158    0.9969    0.0188   2811.0971   2375.0492    1.0021        4.9373                                           
   z_beta[15]   -0.0166    0.9508    0.0163   3405.0743   2332.2502    1.0019        5.9805                                           
   z_beta[16]   -0.0201    0.9929    0.0202   2432.2578   1524.8176    1.0146        4.2719                                           
   z_beta[17]   -0.0367    1.0304    0.0268   1466.2202    450.4462    1.0165        2.5752                                           
   z_beta[18]   -0.1125    0.9505    0.0187   2576.2047   2124.5926    1.0048        4.5247                                           
   z_beta[19]   -0.0044    1.0020    0.0211   2295.0119   1511.4592    1.0046        4.0308                                           
   z_beta[20]   -0.0287    1.0158    0.0180   3164.9979   2292.8402    1.0044        5.5588                                           
   z_beta[21]   -0.0345    1.0052    0.0191   2781.0599   1334.1898    1.0068        4.8845                                           
   z_beta[22]   -0.0179    1.0280    0.0306   1234.4664    359.3331    1.0100        2.1681                       

=#

regime_df = summarize_regime_robust(chain_corrected, unique_months)

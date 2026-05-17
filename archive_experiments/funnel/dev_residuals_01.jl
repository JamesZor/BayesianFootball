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

exp_res.config.model
#=
julia> exp_res.config.model
   GRWNegativeBinomialMu
   -----------------
     μ_init ~ Normal{Float64}(μ=0.2, σ=0.1)
     σ_μ ~ Gamma{Float64}(α=2.0, θ=0.015)
     z_μ_steps ~ Normal{Float64}(μ=0.0, σ=1.0)
     γ ~ Normal{Float64}(μ=0.12, σ=0.5)
     log_r_prior ~ Normal{Float64}(μ=2.5, σ=0.5)
     σ_k ~ Gamma{Float64}(α=2.0, θ=0.08)
     σ_0 ~ Gamma{Float64}(α=2.0, θ=0.08)
     z_init ~ Normal{Float64}(μ=0.0, σ=1.0)
     z_steps ~ Normal{Float64}(μ=0.0, σ=1.0)

=#

market_data = Data.prepare_market_data(ds)

# ----------------------------
# Check the number of goals observed against the number of goals predicted. 
#  For home, away and all. 
#  I guess we need to get a E[ score_matrix] or 
#  

latents_raw = BayesianFootball.Experiments.extract_oos_predictions(ds, exp_res)
#=
1379×4 DataFrame
  Row │ match_id  r                                  λ_a                                λ_h                               
      │ Any       Any                                Any                                Any                               
──────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    1 │ 8824152   [31.448, 5.49576, 17.5147, 10.24…  [1.04826, 1.53016, 0.593666, 1.0…  [1.39129, 1.35173, 1.68627, 1.51…
    2 │ 8824194   [31.448, 5.49576, 17.5147, 10.24…  [1.99893, 0.99796, 1.76385, 1.17…  [2.46117, 1.57326, 2.36236, 0.91…
    3 │ 8824178   [31.448, 5.49576, 17.5147, 10.24…  [1.26069, 1.10189, 0.601548, 1.0…  [1.92468, 1.70173, 2.02289, 1.66…
    4 │ 8824160   [31.448, 5.49576, 17.5147, 10.24…  [1.13303, 1.69631, 1.57255, 0.98…  [0.995991, 1.73927, 0.642108, 1.…

Here the r is ϕ the negative Binomial dispersion parameters and λ_h is the rate parameter for home and λ_a is away. 

=#

names(ds.matches)


joined = innerjoin(
    select(latents_raw.df, :match_id, :λ_h, :λ_a, :r),
    select(ds.matches, :match_id, :match_date, :home_score, :away_score, :tournament_id, :season, :home_team, :away_team),
    on = :match_id
)

sort!(joined, :match_date)

# get models goals E[f]

joined.exp_home = [ mean(λ) for λ in joined.λ_h]
joined.exp_away = [ mean(λ) for λ in joined.λ_a]
joined.exp_all = joined.exp_home .+ joined.exp_away


joined.res_home = joined.home_score .- joined.exp_home
joined.res_away = joined.away_score .- joined.exp_away
joined.res_all = ( joined.home_score .+ joined.away_score ) .- joined.exp_all



sum(joined.res_home) # -60.415
sum(joined.res_away) # -38.07
sum(joined.res_all)  # -98.488

describe(joined.res_home)
describe(joined.res_away)
describe(joined.res_all)


#
joined.cum_res_home = cumsum(joined.home_score) .- cumsum(joined.exp_home)
joined.cum_res_away = cumsum(joined.away_score) .- cumsum(joined.exp_away)
joined.cum_res_all = cumsum(joined.away_score .+ joined.home_score) .- cumsum(joined.exp_all)

OneSampleTTest(joined.res_all)

#=

julia> OneSampleTTest(joined.res_all)
One sample t-test
-----------------
Population details:
    parameter of interest:   Mean
    value under h_0:         0
    point estimate:          -0.0714199
    95% confidence interval: (-0.159, 0.01612)

Test summary:
    outcome with 95% confidence: fail to reject h_0
    two-sided p-value:           0.1097

Details:
    number of observations:   1379
    t-statistic:              -1.6003984856501203
    degrees of freedom:       1378
    empirical standard error: 0.04462633603546454 
=#


using GLM

# Add an index representing time/match order
joined.match_index = 1:nrow(joined)

# Regress RAW residuals against time
time_bias_model = lm(@formula(res_all ~ match_index), joined)
#=
julia> time_bias_model = lm(@formula(res_all ~ match_index), joined)
StatsModels.TableRegressionModel{LinearModel{GLM.LmResp{Vector{Float64}}, GLM.DensePredChol{Float64, CholeskyPivoted{Float64, Matrix{Float64}, Vector{Int64}}}}, Matrix{Float64}}

res_all ~ 1 + match_index

Coefficients:
──────────────────────────────────────────────────────────────────────────────────
                    Coef.   Std. Error      t  Pr(>|t|)    Lower 95%     Upper 95%
──────────────────────────────────────────────────────────────────────────────────
(Intercept)  -0.224978     0.0892058    -2.52    0.0118  -0.399972    -0.0499837
match_index   0.000222548  0.000111983   1.99    0.0471   2.87111e-6   0.000442224
──────────────────────────────────────────────────────────────────────────────────
=#


using Dates, DataFrames, Statistics

# 1. Ensure match_date is a Date object (if it's a string, you'll need to parse it first)
# joined.match_date = Date.(joined.match_date, "yyyy-mm-dd") # uncomment if needed

# 2. Extract the calendar month
joined.month = month.(joined.match_date)

# 3. Create a function to map calendar months to "Season Months"
# Assuming a standard European season starting in August.
# August = Month 1, September = Month 2 ... May = Month 10.
function get_season_month(m::Int)
    if m >= 8 
        return m - 7       # Aug(8)->1, Sep(9)->2, Oct(10)->3, Nov(11)->4, Dec(12)->5
    else
        return m + 5       # Jan(1)->6, Feb(2)->7, Mar(3)->8, Apr(4)->9, May(5)->10, Jun(6)->11, Jul(7)->12
    end
end

joined.season_month = get_season_month.(joined.month)

# 4. Group by Season and Season Month, then calculate the mean residual
monthly_trend = combine(
    groupby(joined, [:season, :season_month]), 
    :res_all => mean => :avg_residual,
    :res_all => sum => :total_residual,
    nrow => :matches_played
)

# 5. Sort chronologically so you can read it top to bottom
sort!(monthly_trend, [:season, :season_month])

#=
julia> sort!(monthly_trend, [:season, :season_month])
41×5 DataFrame
 Row │ season   season_month  avg_residual  total_residual  matches_played 
     │ String7  Int64         Float64       Float64         Int64          
─────┼─────────────────────────────────────────────────────────────────────
   1 │ 20/21               5   -0.195439         -2.93159               15
   2 │ 20/21               6   -1.19232          -4.7693                 4
   3 │ 20/21               8   -0.45489         -14.1016                31
   4 │ 20/21               9   -0.247427        -13.3611                54
   5 │ 21/22               2    0.628467          3.14234                5
   6 │ 21/22               3    0.449254         14.8254                33
   7 │ 21/22               4   -0.5092          -16.2944                32
   8 │ 21/22               5    0.616875         22.2075                36
   9 │ 21/22               6   -0.209088         -8.36354               40
  10 │ 21/22               7   -0.613564        -25.1561                41
  11 │ 21/22               8   -0.629538        -30.2178                48
  12 │ 21/22               9    0.0713687         3.56843               50
  13 │ 22/23               3    0.140224          6.59052               47
  14 │ 22/23               4    0.224728          6.96655               31
  15 │ 22/23               5   -0.138391         -3.73657               27
  16 │ 22/23               6   -0.247674         -8.42091               34
  17 │ 22/23               7   -0.738465        -28.8001                39
  18 │ 22/23               8    0.0598794         2.3353                39
  19 │ 22/23               9   -0.365192        -20.0856                55
  20 │ 22/23              10    0.0860932         0.860932              10
  21 │ 23/24               3    0.183225          3.84773               21
  22 │ 23/24               4   -0.109444         -3.72109               34
  23 │ 23/24               5    0.353054         14.1222                40
  24 │ 23/24               6   -0.333798        -11.6829                35
  25 │ 23/24               7   -0.203332         -9.35326               46
  26 │ 23/24               8    0.0343893         1.85702               54
  27 │ 23/24               9   -0.255017        -10.2007                40
  28 │ 23/24              10    0.586081          5.86081               10
  29 │ 24/25               3   -0.199528         -5.18772               26
  30 │ 24/25               4    0.00956408        0.306051              32
  31 │ 24/25               5    0.487337         21.4428                44
  32 │ 24/25               6   -0.0636564        -1.97335               31
  33 │ 24/25               7    0.265987         11.4374                43
  34 │ 24/25               8   -0.220045        -12.1025                55
  35 │ 24/25               9    0.147846          5.91386               40
  36 │ 24/25              10    0.273739          2.73739               10
  37 │ 25/26               3   -0.221279         -5.75325               26
  38 │ 25/26               4   -0.138354         -5.53417               40
  39 │ 25/26               5    0.234304          9.13784               39
  40 │ 25/26               6    0.464715         15.8003                34
  41 │ 25/26               7   -1.21262          -9.70094                8

=#
using Plots
# using StatsPlots # Optional, but makes grouping DataFrames slightly cleaner if you have it
plotlyjs() 

# 1. Create the figure directory if it doesn't exist
mkpath("figures")

# 2. Build the plot
# We group by season so each season gets its own line.
p = plot(
    monthly_trend.season_month, 
    monthly_trend.avg_residual, 
    group = monthly_trend.season,
    geom = :path,           # Use lines
    markershape = :circle,  # Add dots at the data points
    title = "Model Bias (Average Residuals) by Season Month",
    xlabel = "Month of Season",
    ylabel = "Average Residual (Actual - xG)",
    legend = :outertopright,
    linewidth = 2,
    size = (800, 600)
);

# 3. Add a thick dashed horizontal line at 0 to represent a perfectly calibrated model
StatsPlots.hline!(p, [0], line=(:black, 3, :dash), label="Zero Bias");

# 4. Save the interactive plot as an HTML file
Plots.savefig(p, "figures/monthly_bias_trend.html")




# Here is the code to collapse all the seasons and calculate the true weighted average for each month of the season.
# By grouping by season_month and summing both the total residuals and total matches before dividing, this approach naturally down-weights those weird outlier months that only had 4 or 8 matches.
#

# 1. Group by season_month across ALL seasons
master_trend = combine(
    groupby(monthly_trend, :season_month),
    :total_residual => sum => :sum_residuals,
    :matches_played => sum => :sum_matches 
)


# 2. Calculate the true weighted average residual for each month
master_trend.master_avg_residual = master_trend.sum_residuals ./ master_trend.sum_matches

# 3. Sort chronologically by month
sort!(master_trend, :season_month)

# 4. Create a clean, single-line plot
p2 = plot(
    master_trend.season_month, 
    master_trend.master_avg_residual, 
    geom = :path,
    markershape = :circle,
    markersize = 6,
    color = :blue,
    linewidth = 3,
    title = "Master Bias Trend: Average Residual by Season Month",
    xlabel = "Month of Season (1 = Aug, 10 = May)",
    ylabel = "Average Residual (Actual - xG)",
    legend = false,
    size = (1200, 800)
);

# 5. Add the zero-bias baseline
StatsPlots.hline!(p2, [0], line=(:black, 3, :dash));

# 6. Save the new plot
Plots.savefig(p2, "figures/master_bias_trend.html")

## with ribbons for the std , 

# 1. Group the raw match data by season_month to get the exact variance
master_stats = combine(
    groupby(joined, :season_month),
    :res_all => mean => :mean_res,
    :res_all => std => :std_res,
    nrow => :matches_played
)

# 2. Sort chronologically
sort!(master_stats, :season_month)
#=
julia> sort!(master_stats, :season_month)
9×4 DataFrame
 Row │ season_month  mean_res    std_res  matches_played 
     │ Int64         Float64     Float64  Int64          
─────┼───────────────────────────────────────────────────
   1 │            2   0.628467   1.313                 5
   2 │            3   0.0936122  1.61937             153
   3 │            4  -0.108148   1.4293              169
   4 │            5   0.299712   1.80843             201
   5 │            6  -0.109043   1.60758             178
   6 │            7  -0.34787    1.56238             177
   7 │            8  -0.230086   1.79294             227
   8 │            9  -0.14295    1.648               239
   9 │           10   0.315304   1.46147              30

=#

# 3. Calculate the 2-Sigma margin of error for the MEAN (2 * Standard Error)
master_stats.margin_of_error = 2 .* (master_stats.std_res ./ sqrt.(master_stats.matches_played))

# 4. Create the plot with the ribbon
p3 = plot(
    master_stats.season_month, 
    master_stats.mean_res,
    ribbon = master_stats.margin_of_error, # Plots.jl adds & subtracts this value from the line
    fillalpha = 0.8,                       # Makes the ribbon semi-transparent
    fillcolor = :lightblue,
    geom = :path,
    # markershape = :circle,
    markersize = 6,
    color = :blue,
    linewidth = 3,
    title = "Master Bias Trend with 2σ Confidence Interval",
    xlabel = "Month of Season (1 = Aug, 10 = May)",
    ylabel = "Average Residual (Actual - xG)",
    legend = false,
    size = (1400, 800)
);

# 5. Add the baseline
StatsPlots.hline!(p3, [0], line=(:black, 3, :dash));

# 6. Save it
Plots.savefig(p3, "figures/master_bias_trend_ribbon.html")


# -------
# Residual check for a model. 
# Deconstructed dev version for the Negative Binomial model 
# rᵢ = yᵢ - E[ NegBinomial( λₕ, ϕ ) ]
#
#
using Distributions, Random

# 1. Helper function to compute a single RQR
function compute_rqr(y::Int, λ::Float64, r_disp::Float64)
    # Convert mean (λ) and dispersion (r) to standard NegBinomial (r, p) format
    # Formula: λ = r * (1-p) / p  =>  p = r / (r + λ)
    p = r_disp / (r_disp + λ)
    
    # Instantiate the exact distribution for this specific match prediction
    dist = NegativeBinomial(r_disp, p)
    
    # Get the CDF limits (F(y-1) and F(y))
    # If actual goals = 0, the lower limit must be exactly 0.0
    cdf_lower = y > 0 ? cdf(dist, y - 1) : 0.0
    cdf_upper = cdf(dist, y)
    
    # Draw a random uniform number in the "gap"
    u = rand(Uniform(cdf_lower, cdf_upper))
    
    # Pass through the inverse standard normal CDF (quantile function)
    return quantile(Normal(0, 1), u)
end

# 2. Extract the posterior mean of your dispersion parameter (r) for each match
joined.exp_r = [mean(r_vec) for r_vec in joined.r]

# 3. Compute RQRs for Home and Away
joined.rqr_home = [
    compute_rqr(y, λ, r) 
    for (y, λ, r) in zip(joined.home_score, joined.exp_home, joined.exp_r)
]

joined.rqr_away = [
    compute_rqr(y, λ, r) 
    for (y, λ, r) in zip(joined.away_score, joined.exp_away, joined.exp_r)
]

describe(joined.rqr_home)
describe(joined.rqr_away)
#=
julia> describe(joined.rqr_home)
Summary Stats:
Length:         1379
Missing Count:  0
Mean:           -0.024305
Std. Deviation: 0.973248
Minimum:        -3.198151
1st Quartile:   -0.672029
Median:         -0.009786
3rd Quartile:   0.604862
Maximum:        3.463758
Type:           Float64


julia> describe(joined.rqr_away)
Summary Stats:
Length:         1379
Missing Count:  0
Mean:           -0.021015
Std. Deviation: 1.021371
Minimum:        -2.891136
1st Quartile:   -0.675279
Median:         -0.065088
3rd Quartile:   0.656173
Maximum:        3.537629
Type:           Float64

=#

function normal_stuff_display(rqr)
      sk = skewness(rqr)
      ku = kurtosis(rqr) 

      println("\n--- Moment Analysis ---")
      println("Mean (Bias):      ", round(mean(rqr), digits=4))
      println("Std Dev:          ", round(std(rqr), digits=4))
      println("Skewness:         ", round(sk, digits=4), " (Target: 0.0)")
      println("Excess Kurtosis:  ", round(ku, digits=4), " (Target: 0.0)")

      sw_test = ShapiroWilkTest(rqr)
      println("--- Normality Test Results ---")
      println(sw_test)
end

normal_stuff_display(joined.rqr_home)
normal_stuff_display(joined.rqr_away)

#=
julia> normal_stuff_display(joined.rqr_home)

--- Moment Analysis ---
Mean (Bias):      -0.0243
Std Dev:          0.9732
Skewness:         -0.0025 (Target: 0.0)
Excess Kurtosis:  0.1796 (Target: 0.0)
--- Normality Test Results ---
Shapiro-Wilk normality test
---------------------------
Population details:
    parameter of interest:   Squared correlation of data and expected order statistics of N(0,1) (W)
    value under h_0:         1.0
    point estimate:          0.9989

Test summary:
    outcome with 95% confidence: fail to reject h_0
    one-sided p-value:           0.5725

Details:
    number of observations: 1379
    censored ratio:         0.0
    W-statistic:            0.9989


julia> normal_stuff_display(joined.rqr_away)

--- Moment Analysis ---
Mean (Bias):      -0.021
Std Dev:          1.0214
Skewness:         0.0913 (Target: 0.0)
Excess Kurtosis:  -0.0785 (Target: 0.0)
--- Normality Test Results ---
Shapiro-Wilk normality test
---------------------------
Population details:
    parameter of interest:   Squared correlation of data and expected order statistics of N(0,1) (W)
    value under h_0:         1.0
    point estimate:          0.998411

Test summary:
    outcome with 95% confidence: fail to reject h_0
    one-sided p-value:           0.2298

Details:
    number of observations: 1379
    censored ratio:         0.0
    W-statistic:            0.998411
=#



# 1. Pool Home and Away RQRs into a single long DataFrame
df_home = select(joined, :season_month, :rqr_home => :rqr)
df_away = select(joined, :season_month, :rqr_away => :rqr)
df_pooled = vcat(df_home, df_away)

# 2. Group by season_month and calculate stats
rqr_monthly = combine(
    groupby(df_pooled, :season_month),
    :rqr => mean => :mean_rqr,
    :rqr => std => :std_rqr,
    nrow => :n_obs
)

# 3. Sort chronologically
sort!(rqr_monthly, :season_month)

#=
julia> sort!(rqr_monthly, :season_month)
9×4 DataFrame
 Row │ season_month  mean_rqr    std_rqr   n_obs 
     │ Int64         Float64     Float64   Int64 
─────┼───────────────────────────────────────────
   1 │            2   0.324878   1.01293      10
   2 │            3   0.0394616  1.0182      306
   3 │            4  -0.0125591  0.937772    338
   4 │            5   0.0904728  1.05927     402
   5 │            6  -0.0182346  0.950315    356
   6 │            7  -0.137958   0.954451    354
   7 │            8  -0.0893496  1.00339     454
   8 │            9  -0.0464326  1.02307     478
   9 │           10   0.135718   0.986848     60

=#

# 4. Calculate the 2-Sigma Margin of Error (Standard Error of the Mean)
rqr_monthly.margin_of_error = 2 .* (rqr_monthly.std_rqr ./ sqrt.(rqr_monthly.n_obs))

# 5. Build the plot
p_rqr = plot(
    rqr_monthly.season_month, 
    rqr_monthly.mean_rqr,
    ribbon = rqr_monthly.margin_of_error, 
    fillalpha = 0.4,                       
    fillcolor = :lightgreen,
    geom = :path,
    markershape = :circle,
    markersize = 6,
    color = :darkgreen,
    linewidth = 3,
    title = "RQR Temporal Bias: Mean RQR by Season Month",
    xlabel = "Month of Season (1 = Aug, 10 = May)",
    ylabel = "Mean RQR (Target: 0.0)",
    legend = false,
    size = (1200, 800)
);

# 6. Add the zero-bias baseline
StatsPlots.hline!(p_rqr, [0.0], line=(:black, 3, :dash));

# 7. Save and display
Plots.savefig(p_rqr, "figures/rqr_monthly_trend.html")




rqr_monthly = combine(
    groupby(df_pooled, :season_month),
    :rqr => mean => :mean_rqr,
    :rqr => std => :std_rqr,
    :rqr => skewness => :skewness, 
    :rqr => kurtosis => :kurtosis, 
    :rqr => ( x -> ShapiroWilkTest(x).W )=> :SW_W,
    nrow => :n_obs
)
#=
9×7 DataFrame
 Row │ season_month  mean_rqr    std_rqr   skewness     kurtosis    SW_W      n_obs 
     │ Int64         Float64     Float64   Float64      Float64     Float64   Int64 
─────┼──────────────────────────────────────────────────────────────────────────────
   1 │            2   0.324878   1.01293   -0.130806    -0.89888    0.945509     10
   2 │            3   0.0394616  1.0182    -0.0272789   -0.250736   0.997276    306
   3 │            4  -0.0125591  0.937772  -0.0271587   -0.0298134  0.99421     338
   4 │            5   0.0904728  1.05927   -0.0398672   -0.195765   0.996207    402
   5 │            6  -0.0182346  0.950315  -8.32948e-6   0.120002   0.997722    356
   6 │            7  -0.137958   0.954451   0.0689222    0.0878312  0.996099    354
   7 │            8  -0.0893496  1.00339    0.034835     0.215823   0.996636    454
   8 │            9  -0.0464326  1.02307    0.231012     0.268635   0.995263    478
   9 │           10   0.135718   0.986848  -0.202023     0.163583   0.984603     60
=#


using Revise
using BayesianFootball

using DataFrames
using Statistics
using ThreadPinning
using LinearAlgebra
pinthreads(:cores)
BLAS.set_num_threads(1) 


ds = Data.load_extra_ds()

df = subset(ds.matches, :tournament_id => ByRow(in([56, 57])))

# 1827 rows

### 

using DataFramesMeta

# ---- Check the quality of the data
# check for missing data
describe(df[:, [:HS, :AS, :HST, :AST, :HC, :AC, :HF, :AF, :Referee, :HY, :AY, :HR, :AR]], :nmissing)
#=
13×2 DataFrame
 Row │ variable  nmissing 
     │ Symbol    Int64    
─────┼────────────────────
   1 │ HS               2
   2 │ AS               2
   3 │ HST              2
   4 │ AST              2
   5 │ HC               2
   6 │ AC               2
   7 │ HF               4
   8 │ AF               4
   9 │ Referee          0
  10 │ HY               0
  11 │ AY               0
  12 │ HR               0
  13 │ AR               0

# so we do have missing data 

=#
dropmissing!(df, [:HS, :AS, :HC, :AC, :HF, :AF, :Referee, :HY, :AY, :HR, :AR])
# 1823 rows 

# check if shots on target > shots - which should not happend 
@subset(df, :HST .> :HS .|| :AST .> :AS)

#=
julia> @subset(df, :HST .> :HS .|| :AST .> :AS)
0×33 DataFrame
 Row │ tournament_id  season_id  season   match_id  tournament_slug  home_team  away_team  home_score  ⋯
     │ Int64          Int64      String7  Int64     String15         String31   String31   Int64?      ⋯
─────┴──────────────────────────────────────────────────────────────────────────────────────────────────
                                                                                      25 columns omitted
=# 

# check for goasl > shots on target 
@subset(df, :home_score .> :HST .|| :away_score .> :AST )

#= 

julia> @subset(df, :home_score .> :HST .|| :away_score .> :AST )
4×33 DataFrame
 Row │  season   match_id  tournament_slug  home_team        away_team        home_score  away_score  home_score_ht  away_score_ht  match_date   HS       AS       HST       AST       
     │  String7  Int64     String15         String31         String31         Int64?      Int64?      Int64?         Int64?         Dates.Date   Float64  Float64  Float64?  Float64?  
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │  23/24    11395513  league-one       cove-rangers     stirling-albion           4           2              2              1  2024-03-30       5.0      5.0       3.0       4.0  
   2 │  21/22     9573953  league-two       annan-athletic   kelty-hearts-fc           1           2              1              2  2022-04-30       2.0      9.0       0.0       5.0  
   3 │  22/23    10388128  league-two       annan-athletic   east-fife                 2           2              0              2  2022-09-17      12.0      3.0       5.0       1.0  
   4 │  24/25    12476487  league-two       forfar-athletic  the-spartans-fc           0           2              0              0  2025-03-15      11.0      5.0       3.0       1.0  

=#









# 1. Check the Distribution (Poisson vs. Negative Binomial)
using Statistics, DataFrames

# Assuming df is your dataframe
function check_overdispersion(data, name)
    μ = mean(data)
    σ² = var(data)
    ratio = σ² / μ
    println("$name: Mean = $(round(μ, digits=2)), Var = $(round(σ², digits=2)), Ratio = $(round(ratio, digits=2))")
    if ratio > 1.1
        println("  -> CONFIRMED: Overdispersed (Use Negative Binomial)")
    else
        println("  -> WARNING: Close to Poisson (Maybe r is high)")
    end
end

check_overdispersion(vcat(df.HS, df.AS), "Total Shots")
check_overdispersion(vcat(df.HC , df.AC), "Total Corners")
check_overdispersion(vcat(df.HF , df.AF), "Total Fouls")

#=

julia> check_overdispersion(vcat(df.HS, df.AS), "Total Shots")
Total Shots: Mean = 9.44, Var = 16.96, Ratio = 1.8
  -> CONFIRMED: Overdispersed (Use Negative Binomial)

julia> check_overdispersion(vcat(df.HC , df.AC), "Total Corners")
Total Corners: Mean = 4.87, Var = 7.45, Ratio = 1.53
  -> CONFIRMED: Overdispersed (Use Negative Binomial)

julia> check_overdispersion(vcat(df.HF , df.AF), "Total Fouls")
Total Fouls: Mean = 10.89, Var = 12.98, Ratio = 1.19
  -> CONFIRMED: Overdispersed (Use Negative Binomial)


=#


# 2. Check the "Pressure" Assumption ( Correlation). 
# Here we are assume that Strong teams ( High shots) force weak team to foul More. 
# Hypothesis: Shots difference ( home - away) should correlate with fouls difference (away -home) . 
# If home takes mores shots, away should commit more fouls.

df.shot_diff = df.HS .- df.AS
df.foul_diff = df.AF .- df.HF
df.goal_diff = df.home_score .- df.away_score

shot_foul_cor = cor(df.shot_diff , df.foul_diff)
goal_foul_cor = cor(df.goal_diff, df.foul_diff)
println("Correlation (Shots Diff vs Fouls Diff): $shot_foul_cor")
println("Correlation (Goal Diff vs Fouls Diff): $goal_foul_cor")

#=

julia> println("Correlation (Shots Diff vs Fouls Diff): $shot_foul_cor")
Correlation (Shots Diff vs Fouls Diff): 0.09917053181961116

julia> println("Correlation (Goal Diff vs Fouls Diff): $goal_foul_cor")
Correlation (Goal Diff vs Fouls Diff): -0.004163895703281771

=#

using GLM

model_shots = lm(@formula(foul_diff ~ shot_diff), df)
model_goals = lm(@formula(foul_diff ~ goal_diff), df)
model_shots_goals = lm(@formula(foul_diff ~ shot_diff + goal_diff), df)

#=

julia> model_shots = lm(@formula(foul_diff ~ shot_diff), df)
StatsModels.TableRegressionModel{LinearModel{GLM.LmResp{Vector{Float64}}, GLM.DensePredChol{Float64, CholeskyPivoted{Float64, Matrix{Float64}, Vector{Int64}}}}, Matrix{Float64}}

foul_diff ~ 1 + shot_diff

Coefficients:
─────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error     t  Pr(>|t|)   Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────────
(Intercept)  0.127669    0.111476   1.15    0.2523  -0.0909636   0.346301
shot_diff    0.0743334   0.0173879  4.28    <1e-04   0.0402314   0.108435
─────────────────────────────────────────────────────────────────────────

julia> model_goals = lm(@formula(foul_diff ~ goal_diff), df)
StatsModels.TableRegressionModel{LinearModel{GLM.LmResp{Vector{Float64}}, GLM.DensePredChol{Float64, CholeskyPivoted{Float64, Matrix{Float64}, Vector{Int64}}}}, Matrix{Float64}}

foul_diff ~ 1 + goal_diff

Coefficients:
───────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error      t  Pr(>|t|)   Lower 95%  Upper 95%
───────────────────────────────────────────────────────────────────────────
(Intercept)   0.201481    0.111144    1.81    0.0700  -0.0165016   0.419463
goal_diff    -0.0106385   0.0595618  -0.18    0.8583  -0.127454    0.106177
───────────────────────────────────────────────────────────────────────────

julia> model_shots_goals = lm(@formula(foul_diff ~ shot_diff + goal_diff), df)
StatsModels.TableRegressionModel{LinearModel{GLM.LmResp{Vector{Float64}}, GLM.DensePredChol{Float64, CholeskyPivoted{Float64, Matrix{Float64}, Vector{Int64}}}}, Matrix{Float64}}

foul_diff ~ 1 + shot_diff + goal_diff

Coefficients:
────────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error      t  Pr(>|t|)   Lower 95%   Upper 95%
────────────────────────────────────────────────────────────────────────────
(Intercept)   0.131846    0.111329    1.18    0.2364  -0.0864976   0.35019
shot_diff     0.0977398   0.0197147   4.96    <1e-06   0.0590741   0.136405
goal_diff    -0.168441    0.0672002  -2.51    0.0123  -0.300238   -0.0366447
────────────────────────────────────────────────────────────────────────────


=#



# 3. Check the "corner scaling" Assumption.
# corners are a function of shots 
corner_shot_correlation_home = cor(df.HS, df.HC)
corner_goal_correlation_home = cor(df.home_score, df.HC)

corner_shot_correlation_away = cor(df.AS, df.AC)
corner_goal_correlation_away = cor(df.away_score, df.AC)

corner_shot_correlation_all = cor(vcat(df.HS,df.AS), vcat(df.HC, df.AC))
corner_goal_correlation_all = cor(vcat(df.home_score, df.away_score), vcat(df.HC, df.AC))


phi_shot_home = sum(df.HC) /  sum(df.HS)
phi_shot_away =  sum(df.AC) / sum(df.AS) 
phi_shot_all =  sum(vcat(df.HC, df.AC)) /  sum(vcat(df.HS ,df.AS)) 

phi_goal_home =  sum(df.HC) / sum(df.home_score) 
phi_goal_away =  sum(df.AC) / sum(df.away_score) 
phi_goal_all =  sum(vcat(df.HC, df.AC)) / sum(vcat(df.home_score ,df.away_score)) 


#=

julia> corner_shot_correlation_home = cor(df.HS, df.HC)
0.3790933061206267

julia> corner_goal_correlation_home = cor(df.home_score, df.HC)
0.0893228500958923

julia> corner_shot_correlation_away = cor(df.AS, df.AC)
0.4179223739848645

julia> corner_goal_correlation_away = cor(df.away_score, df.AC)
0.10676205946072222

julia> corner_shot_correlation_all = cor(vcat(df.HS,df.AS), vcat(df.HC, df.AC))
0.4046492634234309

julia> corner_goal_correlation_all = cor(vcat(df.home_score, df.away_score), vcat(df.HC, df.AC))
0.1032694272835954

julia> phi_shot_home = sum(df.HC) /  sum(df.HS)
0.517088642204845

julia> phi_shot_away =  sum(df.AC) / sum(df.AS)
0.5144848484848484

julia> phi_shot_all =  sum(vcat(df.HC, df.AC)) /  sum(vcat(df.HS ,df.AS))
0.5158536234800356

julia> phi_goal_home =  sum(df.HC) / sum(df.home_score)
3.5191663565314477

julia> phi_goal_away =  sum(df.AC) / sum(df.away_score)
3.547430004178855

julia> phi_goal_all =  sum(vcat(df.HC, df.AC)) / sum(vcat(df.home_score ,df.away_score))
3.53248031496063


=#

# 4. 

# Group by Referee
ref_stats = combine(groupby(df, :Referee), 
    [:HF, :AF] => ((h, a) -> mean(h .+ a)) => :avg_fouls,
    [:HF, :AF] => ((h, a) -> length(h)) => :n_games
)

# Filter for refs with enough games (e.g., > 10)
ref_stats = filter(row -> row.n_games > 10, ref_stats)

# Check the spread
sort!(ref_stats, :avg_fouls)
println(first(ref_stats, 5)) # Most lenient refs
println(last(ref_stats, 5))  # Strictest refs




#### mid week games
using Dates

# Helper to categorize games
function get_day_type(date_obj)
    # 1 = Mon, 2 = Tue, 3 = Wed, 4 = Thu, 5 = Fri, 6 = Sat, 7 = Sun
    d = dayofweek(date_obj)
    
    if d in [6, 7] # Saturday, Sunday
        return "Weekend"
    elseif d in [2, 3, 4] # Tue, Wed, Thu
        return "Midweek"
    else 
        # Fridays are tricky. Usually treated as "Weekend" fixtures in modern football,
        # but you can check your specific league schedule.
        return "Weekend" 
    end
end

# Apply to DataFrame (assuming you have a 'Date' column)
df.DayType = get_day_type.(df.match_date)

# Check the split
println(combine(groupby(df, :DayType), nrow => :Count))
#=
julia> println(combine(groupby(df, :DayType), nrow => :Count))
2×2 DataFrame
 Row │ DayType  Count 
     │ String   Int64 
─────┼────────────────
   1 │ Weekend   1686
   2 │ Midweek    156


=#


using HypothesisTests

function analyze_midweek_effect(df, metric_h, metric_a, name)
    # Combine Home and Away to get "Match Total"
    # We care about the "Speed" of the game
    df.total_metric = df[!, metric_h] .+ df[!, metric_a]
    
    # Split
    weekend = filter(row -> row.DayType == "Weekend", df).total_metric
    midweek = filter(row -> row.DayType == "Midweek", df).total_metric
    
    # Means
    μ_we = mean(weekend)
    μ_mw = mean(midweek)
    diff_pct = round(((μ_mw - μ_we) / μ_we) * 100, digits=1)
    
    println("\n=== $name ===")
    println("Weekend Mean: $(round(μ_we, digits=2))")
    println("Midweek Mean: $(round(μ_mw, digits=2)) ($diff_pct%)")
    
    # T-Test (Is the difference real?)
    # Equal variance not assumed (Welch's t-test)
    test = EqualVarianceTTest(weekend, midweek)
    p_val = pvalue(test)
    
    if p_val < 0.05
        println("  -> SIGNIFICANT (p = $(round(p_val, digits=4)))")
        println("  -> There is a real 'Midweek Effect'.")
    else
        println("  -> Not Significant (p = $(round(p_val, digits=4)))")
        println("  -> Likely just random noise.")
    end
end

# Run the Battery
analyze_midweek_effect(df, :home_score, :away_score, "Total Goals")
analyze_midweek_effect(df, :HS, :AS, "Total Shots")
analyze_midweek_effect(df, :HST, :AST, "Shots on Target")
analyze_midweek_effect(df, :HC, :AC, "Total Corners")

#=
julia> analyze_midweek_effect(df, :home_score, :away_score, "Total Goals")

=== Total Goals ===
Weekend Mean: 2.78
Midweek Mean: 2.48 (-10.9%)
  -> SIGNIFICANT (p = 0.0261)
  -> There is a real 'Midweek Effect'.

julia> analyze_midweek_effect(df, :HS, :AS, "Total Shots")

=== Total Shots ===
Weekend Mean: 18.91
Midweek Mean: 18.62 (-1.5%)
  -> Not Significant (p = 0.5056)
  -> Likely just random noise.

julia> analyze_midweek_effect(df, :HST, :AST, "Shots on Target")

=== Shots on Target ===
Weekend Mean: 8.28
Midweek Mean: 7.85 (-5.3%)
  -> Not Significant (p = 0.0788)
  -> Likely just random noise.

julia> analyze_midweek_effect(df, :HC, :AC, "Total Corners")

=== Total Corners ===
Weekend Mean: 9.75
Midweek Mean: 9.67 (-0.8%)
  -> Not Significant (p = 0.7704)
  -> Likely just random noise.


=#

#=
To test for correlation or significant differences across more than two categories,
you should use an ANOVA (Analysis of Variance).

Why ANOVA?

ANOVA tests the null hypothesis that the mean of all groups (months) are equal. If the resulting p-value is less than 0.05,
it tells you that at least one month is statistically significantly different from the others.

A Quick Note on the Data Shape

Goals in soccer typically follow a Poisson distribution (counts of events) rather than a perfect normal bell curve.
Standard ANOVA assumes normal distribution.
Because your sample size is quite large (150+ games a month), standard ANOVA is usually robust enough to handle this.
However, if you want to be completely statistically rigorous, you can swap OneWayANOVATest for KruskalWallisTest in the code above,
which is the non-parametric equivalent that doesn't care about bell curves.

=#

using HypothesisTests
using DataFrames

function test_monthly_seasonality(df)
    # 1. Create the total goals metric
    df.total_goals = df.home_score .+ df.away_score;
    
    # 2. Group the RAW data by Month
    grouped_months = groupby(df, :Month);
    
    # 3. Extract the total_goals column from each month into a list of arrays
    # This creates a vector of vectors, which is what the ANOVA test requires
    month_vectors = [g.total_goals for g in grouped_months]
    
    # 4. Run the One-Way ANOVA Test
    # The '...' splats the vector of vectors into separate arguments for the test
    anova_test = OneWayANOVATest(month_vectors...)
    p_val = pvalue(anova_test)
    
    println("\n=== Monthly Seasonality (Goals) ===")

    println(anova_test)
    
    if p_val < 0.05
        println("  -> SIGNIFICANT (p = $(round(p_val, digits=4)))")
        println("  -> There is a statistically significant difference in goals scored between different months.")
    else
        println("  -> Not Significant (p = $(round(p_val, digits=4)))")
        println("  -> The variance in goals across months is likely just random noise.")
    end
end

# Run the test
test_monthly_seasonality(df)

function test_monthly_seasonality_kw(df)
    # 1. Create the total goals metric
    df.total_goals = df.home_score .+ df.away_score
    
    # 2. Group the RAW data by Month
    grouped_months = groupby(df, :Month)
    
    # 3. Extract the total_goals column from each month into a list of arrays
    month_vectors = [g.total_goals for g in grouped_months]
    
    # 4. Run the Non-Parametric Test (Kruskal-Wallis)
    kw_test = KruskalWallisTest(month_vectors...)
    p_val = pvalue(kw_test)
    
    println("\n=== Monthly Seasonality (Goals) - Kruskal-Wallis ===")

    println(kw_test)
    
    if p_val < 0.05
        println("  -> SIGNIFICANT (p = $(round(p_val, digits=4)))")
        println("  -> There is a real difference in the distribution of goals across months.")
    else
        println("  -> Not Significant (p = $(round(p_val, digits=4)))")
        println("  -> The variance in goals across months is likely just random noise when accounting for non-normal distributions.")
    end
end

# Run the test
test_monthly_seasonality_kw(df)



#=
julia> test_monthly_seasonality(df)

=== Monthly Seasonality (Goals) ===
One-way analysis of variance (ANOVA) test
-----------------------------------------
Population details:
    parameter of interest:   Means
    value under h_0:         "all equal"
    point estimate:          NaN

Test summary:
    outcome with 95% confidence: reject h_0
    p-value:                     0.0088

Details:
    number of observations: [178, 177, 224, 239, 30, 20, 217, 154, 187, 197, 219]
    F statistic:            2.36769
    degrees of freedom:     (10, 1831)

  -> SIGNIFICANT (p = 0.0088)
  -> There is a statistically significant difference in goals scored between different months.




=== Monthly Seasonality (Goals) - Kruskal-Wallis ===
Kruskal-Wallis rank sum test (chi-square approximation)
-------------------------------------------------------
Population details:
    parameter of interest:   Location parameters
    value under h_0:         "all equal"
    point estimate:          NaN

Test summary:
    outcome with 95% confidence: reject h_0
    one-sided p-value:           0.0063

Details:
    number of observation in each group: [178, 177, 224, 239, 30, 20, 217, 154, 187, 197, 219]
    χ²-statistic:                        24.5263
    rank sums:                           [165944.0, 1.5065e5, 190362.0, 211568.0, 31443.5, 13617.5, 1.98864e5, 151340.0, 1.76706e5, 184351.0, 2.22556e5]
    degrees of freedom:                  10
    adjustment for ties:                 0.964595

  -> SIGNIFICANT (p = 0.0063)
  -> There is a real difference in the distribution of goals across months.


julia> # Example run: Comparing December (12) vs February (2)
       compare_two_months(df, 12, 2)

=== Comparing Month 12 vs Month 2 ===
Month 12: 3.09 avg goals (219 games)
Month 2: 2.56 avg goals (177 games)
Approximate Mann-Whitney U test
-------------------------------
Population details:
    parameter of interest:   Location parameter (pseudomedian)
    value under h_0:         0
    point estimate:          1.0

Test summary:
    outcome with 95% confidence: reject h_0
    two-sided p-value:           0.0019

Details:
    number of observations in each group: [219, 177]
    Mann-Whitney-U statistic:             22840.5
    rank sums:                            [46930.5, 31675.5]
    adjustment for ties:                  2.20016e6
    normal approximation (μ, σ):          (3459.0, 1112.19)

  -> SIGNIFICANT (p = 0.0019)
  -> The difference in goals between Month 12 and Month 2 is real.

=#

using HypothesisTests
using Statistics

function compare_two_months(df, month_a::Int, month_b::Int)
    # 1. Ensure total_goals exists
    if !("total_goals" in names(df))
        df.total_goals = df.home_score .+ df.away_score
    end
    
    # 2. Filter data for the two specific months
    data_a = filter(row -> row.Month == month_a, df).total_goals
    data_b = filter(row -> row.Month == month_b, df).total_goals
    
    # 3. Calculate basic stats for context
    mean_a = round(mean(data_a), digits=2)
    mean_b = round(mean(data_b), digits=2)
    n_a = length(data_a)
    n_b = length(data_b)
    
    println("\n=== Comparing Month $month_a vs Month $month_b ===")
    println("Month $month_a: $mean_a avg goals ($n_a games)")
    println("Month $month_b: $mean_b avg goals ($n_b games)")
    
    # 4. Run the Mann-Whitney U Test
    test = MannWhitneyUTest(data_a, data_b)
    p_val = pvalue(test)

    println(test)
    
    # 5. Output Results
    if p_val < 0.05
        println("  -> SIGNIFICANT (p = $(round(p_val, digits=4)))")
        println("  -> The difference in goals between Month $month_a and Month $month_b is real.")
    else
        println("  -> Not Significant (p = $(round(p_val, digits=4)))")
        println("  -> We cannot prove a statistical difference between these two months.")
    end
end

# Example run: Comparing December (12) vs February (2)
compare_two_months(df, 12, 2)



###
using Dates

df.Month = month.(df.match_date)

function analyze_seasonality(df)
    # Group by Month
    monthly = combine(groupby(df, :Month), 
        [:home_score, :away_score] => ((h, a) -> mean(h .+ a)) => :avg_goals,
        [:HS, :AS] => ((h, a) -> mean(h .+ a)) => :avg_shots,
        nrow => :num_rows
    )
    
    sort!(monthly, :Month)
    println("\n=== Monthly Averages ===")
    println(monthly)
end

analyze_seasonality(df)
#=
julia> analyze_seasonality(df)

=== Monthly Averages ===
11×4 DataFrame
 Row │ Month  avg_goals  avg_shots  num_rows 
     │ Int64  Float64    Float64    Int64    
─────┼───────────────────────────────────────
   1 │     1    2.77528    19.1461       178
   2 │     2    2.55932    18.6384       177
   3 │     3    2.58482    18.0848       224
   4 │     4    2.6318     18.6234       239
   5 │     5    3.13333    20.3           30
   6 │     7    2.05       19.55          20
   7 │     8    2.70968    18.9908       217
   8 │     9    2.93506    19.7208       154
   9 │    10    2.81818    18.8503       187
  10 │    11    2.77157    19.5635       197
  11 │    12    3.09132    18.4521       219

=#

### 3g pitches
# 1. Define the Set of Plastic Teams (using your standardized names)
const PLASTIC_TEAMS = Set([
    "airdrieonians",
    "alloa-athletic",
    "annan-athletic",
    "bonnyrigg-rose",
    "clyde-fc",
    "cove-rangers",
    "east-kilbride",
    "edinburgh-city-fc",
    "falkirk-fc",
    "forfar-athletic",
    "hamilton-academical",
    "kelty-hearts-fc",
    "montrose",
    "queen-of-the-south",
    "stenhousemuir",
    "the-spartans-fc"
])

# 2. Function to check a row
function check_surface(team_name_raw)
    # Map raw name to standard name
    std_name = get(Data.SCOT_TEAM_MAPPING, team_name_raw, lowercase(replace(team_name_raw, " " => "-")))
    
    # Check if in plastic set
    return std_name in PLASTIC_TEAMS
end

# 3. Apply to DataFrame
# This creates a Boolean vector (true/false)
df.is_plastic = check_surface.(df.home_team)

# 4. Quick Sanity Check
println("Total Games on Plastic: $(sum(df.is_plastic))")
println("Total Games on Grass: $(nrow(df) - sum(df.is_plastic))")


#=
#
julia> println("Total Games on Plastic: $(sum(df.is_plastic))")
Total Games on Plastic: 1110

julia> println("Total Games on Grass: $(nrow(df) - sum(df.is_plastic))")
Total Games on Grass: 732



=#


# 3. Analyze: Do we see more goals/shots on plastic?
function analyze_surface(df)
    grass = filter(row -> !row.is_plastic, df)
    plastic = filter(row -> row.is_plastic, df)
    
    println("=== Surface Analysis ===")
    println("Grass - Avg Goals: $(round(mean(grass.home_score .+ grass.away_score), digits=2))")
    println("Plastic - Avg Goals: $(round(mean(plastic.home_score .+ plastic.away_score), digits=2))")
    
    # Check Home Advantage specifically
    # Home Win % or Goal Diff
    h_grass = mean(grass.home_score .- grass.away_score)
    h_plastic = mean(plastic.home_score .- plastic.away_score)
    
    println("Grass - Home Adv (Goal Diff): $(round(h_grass, digits=2))")
    println("Plastic - Home Adv (Goal Diff): $(round(h_plastic, digits=2))")
end

analyze_surface(df)

#=

julia> function analyze_surface(df)
           grass = filter(row -> !row.is_plastic, df)
           plastic = filter(row -> row.is_plastic, df)
           
           println("=== Surface Analysis ===")
           println("Grass - Avg Goals: $(round(mean(grass.home_score .+ grass.away_score), digits=2))")
           println("Plastic - Avg Goals: $(round(mean(plastic.home_score .+ plastic.away_score), digits=2))")
           
           # Check Home Advantage specifically
           # Home Win % or Goal Diff
           h_grass = mean(grass.home_score .- grass.away_score)
           h_plastic = mean(plastic.home_score .- plastic.away_score)
           
           println("Grass - Home Adv (Goal Diff): $(round(h_grass, digits=2))")
           println("Plastic - Home Adv (Goal Diff): $(round(h_plastic, digits=2))")
       end
analyze_surface (generic function with 1 method)

julia> analyze_surface(df)
=== Surface Analysis ===
Grass - Avg Goals: 2.57
Plastic - Avg Goals: 2.88
Grass - Home Adv (Goal Diff): 0.16
Plastic - Home Adv (Goal Diff): 0.16


=#

function analyze_surface(df)
    grass = filter(row -> !row.is_plastic, df)
    plastic = filter(row -> row.is_plastic, df)
    
    # Calculate Total Goals and Goal Diff for both groups
    grass_goals = grass.home_score .+ grass.away_score
    plastic_goals = plastic.home_score .+ plastic.away_score
    
    grass_diff = grass.home_score .- grass.away_score
    plastic_diff = plastic.home_score .- plastic.away_score
    
    println("=== Surface Analysis ===")
    println("Grass - Avg Goals: $(round(mean(grass_goals), digits=2))")
    println("Plastic - Avg Goals: $(round(mean(plastic_goals), digits=2))")
    
    println("Grass - Home Adv (Goal Diff): $(round(mean(grass_diff), digits=2))")
    println("Plastic - Home Adv (Goal Diff): $(round(mean(plastic_diff), digits=2))")
    
    println("\n=== Statistical Tests (Mann-Whitney U) ===")
    
    # Test 1: Total Goals
    mwu_goals = MannWhitneyUTest(grass_goals, plastic_goals)
    p_goals = pvalue(mwu_goals)

    println(" Test 1: Total goals ") 
    println(mwu_goals)
    println("Total Goals p-value: $(round(p_goals, digits=4))")
    if p_goals < 0.05
        println("  -> SIGNIFICANT: Plastic pitches genuinely see a different amount of goals.")
    else
        println("  -> Not Significant: The 2.88 vs 2.57 difference might just be random noise.")
    end
    
    # Test 2: Home Advantage (Goal Difference)
    mwu_diff = MannWhitneyUTest(grass_diff, plastic_diff)
    p_diff = pvalue(mwu_diff)

    println(" Test 2: Home Advantage - goal difference ")

    print(mwu_diff) 
    println("\nHome Advantage p-value: $(round(p_diff, digits=4))")
    if p_diff < 0.05
        println("  -> SIGNIFICANT: The home advantage is genuinely different on plastic.")
    else
        println("  -> Not Significant: Home advantage is statistically the same on both surfaces.")
    end
end



analyze_surface(df)

#=
julia> analyze_surface(df)
=== Surface Analysis ===
Grass - Avg Goals: 2.57
Plastic - Avg Goals: 2.88
Grass - Home Adv (Goal Diff): 0.16
Plastic - Home Adv (Goal Diff): 0.16

=== Statistical Tests (Mann-Whitney U) ===
 Test 1: Total goals 
Approximate Mann-Whitney U test
-------------------------------
Population details:
    parameter of interest:   Location parameter (pseudomedian)
    value under h_0:         0
    point estimate:          -1.0

Test summary:
    outcome with 95% confidence: reject h_0
    two-sided p-value:           <1e-04

Details:
    number of observations in each group: [732, 1110]
    Mann-Whitney-U statistic:             363264.0
    rank sums:                            [631542.0, 1.06586e6]
    adjustment for ties:                  2.21274e8
    normal approximation (μ, σ):          (-42996.0, 10971.4)

Total Goals p-value: 0.0001
  -> SIGNIFICANT: Plastic pitches genuinely see a different amount of goals.
 Test 2: Home Advantage - goal difference 
Approximate Mann-Whitney U test
-------------------------------
Population details:
    parameter of interest:   Location parameter (pseudomedian)
    value under h_0:         0
    point estimate:          0.0

Test summary:
    outcome with 95% confidence: fail to reject h_0
    two-sided p-value:           0.8679

Details:
    number of observations in each group: [732, 1110]
    Mann-Whitney-U statistic:             4.0443e5
    rank sums:                            [6.72708e5, 1.02469e6]
    adjustment for ties:                  1.95281e8
    normal approximation (μ, σ):          (-1829.5, 10995.0)

Home Advantage p-value: 0.8679
  -> Not Significant: Home advantage is statistically the same on both surfaces.



=#
function analyze_match_outcomes(df)
    # 1. Figure out who won each game (or if it was a draw)
    # H = Home Win, D = Draw, A = Away Win
    outcomes = map(row -> row.home_score > row.away_score ? "Home Win" : 
                          (row.home_score == row.away_score ? "Draw" : "Away Win"), 
                   eachrow(df))
    
    # Add this temporarily to our dataframe to make grouping easy
    df_temp = copy(df)
    df_temp.Outcome = outcomes
    
    # 2. Split into Grass and Plastic
    grass = filter(row -> !row.is_plastic, df_temp)
    plastic = filter(row -> row.is_plastic, df_temp)
    
    # 3. Helper function to calculate and print the stats
    function print_percentages(subset, name)
        n = nrow(subset)
        hw = count(==("Home Win"), subset.Outcome) / n * 100
        d  = count(==("Draw"), subset.Outcome) / n * 100
        aw = count(==("Away Win"), subset.Outcome) / n * 100
        
        println("$name ($n games):")
        println("  Home Win: $(round(hw, digits=1))%")
        println("  Draw:     $(round(d, digits=1))%")
        println("  Away Win: $(round(aw, digits=1))%\n")
    end
    
    println("\n=== Match Outcomes by Surface ===")
    print_percentages(grass, "Grass Pitches")
    print_percentages(plastic, "Plastic Pitches")
end

# Run the function
analyze_match_outcomes(df)

#=

julia> analyze_match_outcomes(df)

=== Match Outcomes by Surface ===
Grass Pitches (732 games):
  Home Win: 41.1%
  Draw:     25.3%
  Away Win: 33.6%

Plastic Pitches (1110 games):
  Home Win: 41.4%
  Draw:     25.3%
  Away Win: 33.2%


=#


#---
#=

julia> names(df)
41-element Vector{String}:
 "tournament_id"
 "season_id"
 "season"
 "match_id"
 "tournament_slug"
 "home_team"
 "away_team"
 "home_score"
 "away_score"
 "home_score_ht"
 "away_score_ht"
 "match_date"
 "round"
 "winner_code"
 "has_xg"
 "has_stats"
 "match_hour"
 "match_dayofweek"
 "match_month"
 "match_week"
 "HS"
 "AS"
 "HST"
 "AST"
 "HC"
 "AC"
 "HF"
 "AF"
 "Referee"
 "HY"
 "AY"
 "HR"
 "AR"
 "shot_diff"
 "foul_diff"
 "goal_diff"
 "DayType"
 "total_metric"
 "Month"
 "is_plastic"
 "total_goals"

julia> ds.incidents
60809×30 DataFrame
   Row │ tournament_id  season_id  match_id  incident_type  time   is_home  period_text  home_score  away_score  is_live  added_time  time_seconds  reversed_period_time  reversed_period_time_seconds  period_time_seconds  injury_time_length  player_in_name      player_o ⋯
       │ Int64          Int64      Int64     String15       Int64  Bool?    String3?     Int64?      Int64?      Bool?    Float64?    Float64?      Float64?              Float64?                      Float64?             Float64?            String63?           String63 ⋯
───────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
     1 │            54      41957  10387456  period            90  missing  FT                    2           0    false       999.0        5400.0                   1.0                           0.0               2700.0           missing    missing             missing  ⋯
     2 │            54      41957  10387456  injuryTime        90  missing  missing         missing     missing  missing         0.0     missing                     1.0                     missing              missing                   6.0  missing             missing 
     3 │            54      41957  10387456  substitution      89     true  missing         missing     missing  missing   missing       missing               missing                       missing              missing             missing    Riku Danzaki        Blair Sp
     4 │            54      41957  10387456  substitution      89     true  missing         missing     missing  missing   missing       missing               missing                       missing              missing             missing    Bevis Mugabi        Calum Bu


julia> names(ds.incidents)
30-element Vector{String}:
 "tournament_id"
 "season_id"
 "match_id"
 "incident_type"
 "time"
 "is_home"
 "period_text"
 "home_score"
 "away_score"
 "is_live"
 "added_time"
 "time_seconds"
 "reversed_period_time"
 "reversed_period_time_seconds"
 "period_time_seconds"
 "injury_time_length"
 "player_in_name"
 "player_out_name"
 "is_injury"
 "incident_class"
 "player_name"
 "card_type"
 "reason"
 "rescinded"
 "assist1_name"
 "assist2_name"
 "var_confirmed"
 "var_decision"
 "var_reason"
 "penalty_reason"

=#

using DataFrames

function add_scored_first_feature!(df, incidents_df)
    # 1. Filter for only goal events
    # (Sometimes 'incident_type' is "goal", check your unique values if this returns empty!)
    goals_only = filter(row -> row.incident_type == "goal", incidents_df)
    
    # Optional: Filter out VAR rescinded goals if your dataset tracks that
    if "rescinded" in names(goals_only)
        goals_only = filter(row -> ismissing(row.rescinded) || row.rescinded == false, goals_only)
    end

    # 2. Sort by match and then by the time the goal was scored
    sort!(goals_only, [:match_id, :time])
    
    # 3. Group by match_id and take the very first goal event
    first_goals_raw = combine(groupby(goals_only, :match_id), first)
    
    # 4. Clean it up: convert the boolean `is_home` to a readable String
    first_goals = select(first_goals_raw, 
        :match_id, 
        :is_home => ByRow(h -> h == true ? "Home" : "Away") => :scored_first
    )
    
    # 5. Join back to the main dataframe
    # We use leftjoin so we don't lose the 0-0 matches
    df = leftjoin(df, first_goals, on=:match_id)
    
    # 6. Fill in the missing values for 0-0 draws
    df.scored_first = coalesce.(df.scored_first, "None")
    
    println("Successfully added 'scored_first' column!")
    
    # Print a quick summary to verify it worked
    summary = combine(groupby(df, :scored_first), nrow => :count)
    println("\n=== Scored First Summary ===")
    println(summary)
    
    return df
end

# Run the function (Note: using ! in the function name because it modifies df in place)
df = add_scored_first_feature!(df, ds.incidents)

combine(groupby(df, :scored_first), nrow => :count)

subset( df, :scored_first => ByRow(isequal("None")))


function analyze_first_goal_advantage(df)
    # 1. Recreate the Outcome column if it wasn't saved permanently
    if !("Outcome" in names(df))
        df.Outcome = map(row -> row.home_score > row.away_score ? "Home Win" : 
                               (row.home_score == row.away_score ? "Draw" : "Away Win"), 
                         eachrow(df))
    end
    
    # 2. Group by BOTH who scored first and who won, then count
    overlap = combine(groupby(df, [:scored_first, :Outcome]), nrow => :count)
    
    # 3. Pivot the table so it looks like a clean grid
    matrix = unstack(overlap, :scored_first, :Outcome, :count, fill=0)
    
    println("=== Raw Match Outcomes based on First Goal ===")
    println(matrix)
    
    # 4. Calculate the specific Win Percentages
    home_first = filter(row -> row.scored_first == "Home", df)
    home_win_pct = count(==("Home Win"), home_first.Outcome) / nrow(home_first) * 100
    home_draw_pct = count(==("Draw"), home_first.Outcome) / nrow(home_first) * 100
    
    away_first = filter(row -> row.scored_first == "Away", df)
    away_win_pct = count(==("Away Win"), away_first.Outcome) / nrow(away_first) * 100
    away_draw_pct = count(==("Draw"), away_first.Outcome) / nrow(away_first) * 100
    
    println("\n=== The 'First Goal' Power ===")
    println("When HOME scores first:")
    println("  -> They win:  $(round(home_win_pct, digits=1))%")
    println("  -> They draw: $(round(home_draw_pct, digits=1))%")
    
    println("\nWhen AWAY scores first:")
    println("  -> They win:  $(round(away_win_pct, digits=1))%")
    println("  -> They draw: $(round(away_draw_pct, digits=1))%")
end

# Run it!
analyze_first_goal_advantage(df)

#=
=== The 'First Goal' Power ===
When HOME scores first:
  -> They win:  71.3%
  -> They draw: 18.7%

When AWAY scores first:
  -> They win:  66.4%
  -> They draw: 22.0%


=#
using Turing
using Distributions

# We pass in vectors of team IDs (1 to n_teams) and the actual goals scored
@model function football_poisson(home_team_id, away_team_id, home_goals, away_goals, n_teams)
    # 1. Priors (Our starting assumptions before seeing data)
    μ ~ Normal(0, 1)          # Baseline league scoring rate
    home_adv ~ Normal(0, 1)   # Universal home advantage

    # Team-specific parameters: A vector of length 'n_teams'
    # Centered around 0 so average teams are 0, good teams are >0, bad are <0
    att ~ filldist(Normal(0, 1), n_teams) 
    def ~ filldist(Normal(0, 1), n_teams)

    # 2. Likelihood (How the data is generated)
    for i in 1:length(home_goals)
        h = home_team_id[i]
        a = away_team_id[i]

        # Calculate the log-rate for this specific matchup
        log_λ_home = μ + home_adv + att[h] + def[a]
        log_λ_away = μ + att[a] + def[h]

        # Exponentiate to ensure λ > 0, then sample from Poisson
        home_goals[i] ~ Poisson(exp(log_λ_home))
        away_goals[i] ~ Poisson(exp(log_λ_away))
    end
end

using DataFrames
using Turing
using Distributions
using StatsFuns: logistic

# ==========================================
# 1. DATA PREP: Filter and Map
# ==========================================
println("Preparing data for First-Goal Model...")

# Filter out the 0-0 draws (we only care about games where someone scored first)
df_goals = filter(row -> row.scored_first != "None", df)

# Grab every unique team and create ID lookups
all_teams = unique(vcat(df_goals.home_team, df_goals.away_team))
n_teams = length(all_teams)
team_to_id = Dict(team => i for (i, team) in enumerate(all_teams))

# Extract our vectors for the model
h_id = [team_to_id[t] for t in df_goals.home_team]
a_id = [team_to_id[t] for t in df_goals.away_team]

# Create the Binary Target (1 = Home Scored First, 0 = Away Scored First)
y = [row.scored_first == "Home" ? 1 : 0 for row in eachrow(df_goals)]


# ==========================================
# 2. DEFINE THE BERNOULLI MODEL
# ==========================================
@model function first_goal_model(home_id, away_id, y, n_teams)
    # Priors
    μ ~ Normal(0, 1)          # League average log-odds of home scoring first
    home_adv ~ Normal(0, 1)   # The bump the home team gets

    # Team specific parameters (Attack and Defense)
    att ~ filldist(Normal(0, 1), n_teams)
    def ~ filldist(Normal(0, 1), n_teams)

    # Likelihood 
    for i in 1:length(y)
        h = home_id[i]
        a = away_id[i]

        # YOUR FORMULA: Calculate the logit-probability of Home scoring first.
        # (Note: The model will naturally learn negative numbers for good away defenses)
        logit_p = μ + home_adv + att[h] + def[a]
        
        # Squash into a 0.0 to 1.0 probability
        p = logistic(logit_p)
        
        # Sample from Bernoulli
        y[i] ~ Bernoulli(p)
    end
end

# ==========================================
# 3. RUN THE SAMPLER
# ==========================================
model = first_goal_model(h_id, a_id, y, n_teams)

println("Starting NUTS Sampler for First Goal MVP...")
# Running 500 iterations for a quick MVP test
chain = sample(
    model, 
    NUTS(0.65), 
    MCMCThreads(), 
    500, 
    8,
)

describe(chain)


#=
julia> describe(chain)                                                                                                                                                                                                                                                         
Chains MCMC chain (500×76×8 Array{Float64, 3}):                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                               
Iterations        = 251:1:750                                                                                                                                                                                                                                                  
Number of chains  = 8                                                                                                                                                                                                                                                          
Samples per chain = 500                                                                                                                                                                                                                                                        
Wall duration     = 39.32 seconds                                                                                                                                                                                                                                              
Compute duration  = 305.31 seconds                                                                                                                                                                                                                                             
parameters        = μ, home_adv, att[1], att[2], att[3], att[4], att[5], att[6], att[7], att[8], att[9], att[10], att[11], att[12], att[13], att[14], att[15], att[16], att[17], att[18], att[19], att[20], att[21], att[22], att[23], att[24], att[25], att[26], att[27], att[
28], att[29], att[30], def[1], def[2], def[3], def[4], def[5], def[6], def[7], def[8], def[9], def[10], def[11], def[12], def[13], def[14], def[15], def[16], def[17], def[18], def[19], def[20], def[21], def[22], def[23], def[24], def[25], def[26], def[27], def[28], def[2
9], def[30]                                                                                                                                                                                                                                                                    
internals         = n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size, lp, logprior, loglikelihood                                           
                                                                                                                                                                                                                                                                               
Summary Statistics                                                                                                                                                                                                                                                             
                                                                                                                                                                                                                                                                               
  parameters      mean       std      mcse    ess_bulk    ess_tail      rhat   ess_per_sec                                                                                                                                                                                     
      Symbol   Float64   Float64   Float64     Float64     Float64   Float64       Float64                                                                                                                                                                                     
                                                                                                                                                                                                                                                                               
           μ    0.0691    0.7189    0.0138   2745.9736   2258.7901    1.0039        8.9942                                                                                                                                                                                     
    home_adv    0.1130    0.7155    0.0136   2765.8221   2633.2119    1.0032        9.0592                                                                                                                                                                                     
      att[1]    0.1498    0.2985    0.0084   1255.2111   2081.5836    1.0085        4.1113                                                                                                                                                                                     
      att[2]    0.3194    0.3091    0.0084   1351.0922   2391.7748    1.0064        4.4254                                                                                                                                                                                     
      att[3]    0.7498    0.3774    0.0083   2094.6642   2813.3615    1.0028        6.8609                                                                                                                                                                                     
      att[4]   -0.1581    0.3071    0.0081   1425.3321   2466.8914    1.0058        4.6685                                                                                                                                                                                     
      att[5]   -0.7526    0.6404    0.0094   4681.8137   2961.9739    1.0033       15.3348                                                                                                                                                                                     
      att[6]   -0.6574    0.4332    0.0081   2858.1616   2620.8706    1.0033        9.3616                                                                                                                                                                                     
      att[7]   -0.3827    0.2968    0.0081   1329.6553   2092.8305    1.0090        4.3552                                                                                                                                                                                     
      att[8]   -0.7523    0.3577    0.0078   2078.8071   2564.0341    1.0037        6.8089                                                                                                                                                                                     
      att[9]    0.1556    0.2959    0.0080   1377.3051   1965.6637    1.0055        4.5112                                                                                                                                                                                     
     att[10]   -0.1830    0.2975    0.0082   1312.8818   2275.6701    1.0076        4.3002                                                                                                                                                                                     
     att[11]   -0.3493    0.2873    0.0081   1261.4289   2255.2820    1.0075        4.1317                                                                                                                                                                                     
     att[12]    0.8973    0.4831    0.0085   3243.9158   2646.2504    1.0028       10.6251                                                                                                                                                                                     
     att[13]   -0.1362    0.6216    0.0090   4775.3009   2809.5638    1.0023       15.6410                                                                                                                                                                                     
     att[14]   -0.5797    0.2845    0.0078   1337.4373   2401.9298    1.0073        4.3806                                                                                                                                                                                     
     att[15]   -0.8942    0.2971    0.0080   1396.6739   2505.1799    1.0071        4.5747                                                                                                                                                                                     
     att[16]    0.3814    0.3319    0.0080   1729.7544   2161.2529    1.0032        5.6656                                                                                                                                                                                     
     att[17]   -0.4172    0.2901    0.0081   1285.6785   2223.1821    1.0084        4.2111                                                                                                                                                                                     
     att[18]    0.2089    0.2987    0.0084   1277.9938   2055.2603    1.0068        4.1859                                                                                                                                                                                     
     att[19]   -0.3306    0.2901    0.0079   1344.9090   2071.8544    1.0076        4.4051                                                                                                                                                                                     
     att[20]   -0.3486    0.2859    0.0077   1388.5131   2280.1283    1.0052        4.5479                                                                                                                                                                                     
     att[21]    0.1680    0.3020    0.0079   1468.4592   2418.3867    1.0059        4.8098                                                                                                                                                                                     
     att[22]   -0.0614    0.3026    0.0079   1453.1277   2344.1328    1.0052        4.7596                                                                                                                                                                                     
     att[23]    0.5963    0.5186    0.0087   3541.9718   3033.3085    1.0013       11.6014                                                                                                                                                                                     
     att[24]    0.4650    0.3218    0.0083   1504.0599   2131.4937    1.0061        4.9264


=#
function extract_team_ratings(chain, id_to_team, n_teams)
    # 1. Convert the MCMC chain summary into a DataFrame for easy filtering
    chain_summary = DataFrame(summarystats(chain))
    
    # 2. Extract the global league averages
    μ_val = chain_summary[chain_summary.parameters .== :μ, :mean][1]
    home_adv_val = chain_summary[chain_summary.parameters .== :home_adv, :mean][1]
    
    # 3. Create an empty DataFrame to hold our final results
    results = DataFrame(
        Team = String[],
        Attack_Rating = Float64[],
        Defense_Rating = Float64[],
        Prob_First_Goal_Home = Float64[],
        Prob_First_Goal_Away = Float64[]
    )
    
    # 4. Loop through every team ID and extract their specific stats
    for i in 1:n_teams
        att_sym = Symbol("att[$i]")
        def_sym = Symbol("def[$i]")
        
        att_val = chain_summary[chain_summary.parameters .== att_sym, :mean][1]
        def_val = chain_summary[chain_summary.parameters .== def_sym, :mean][1]
        
        # Calculate their actual % chance against an "average" team (where opponent def = 0)
        # We multiply by 100 to make it a readable percentage
        p_home = logistic(μ_val + home_adv_val + att_val) * 100
        p_away = logistic(μ_val + att_val) * 100
        
        push!(results, (
            id_to_team[i], 
            round(att_val, digits=3), 
            round(def_val, digits=3), 
            round(p_home, digits=1), 
            round(p_away, digits=1)
        ))
    end
    
    # Sort by the most lethal home teams
    sort!(results, :Prob_First_Goal_Home, rev=true)
    
    return results
end

# Run the extraction using the id_to_team dictionary we made earlier
team_leaderboard = extract_team_ratings(chain, id_to_team, n_teams)

println("\n=== First Goal Power Rankings ===")
display(first(team_leaderboard, 10)) # Show the top 10

#=

julia> team_leaderboard = extract_team_ratings(chain, id_to_team, n_teams)                                                                                                                                                                                                     
30×5 DataFrame                                                                                                                                                                                                                                                                 
 Row │ Team                          Attack_Rating  Defense_Rating  Prob_First_Goal_Home  Prob_First_Goal_Away                                                                                                                                                                 
     │ String                        Float64        Float64         Float64               Float64                                                                                                                                                                              
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                                                                
   1 │ hamilton-academical                   1.227          -1.075                  80.4                  78.5                                                                                                                                                                 
   2 │ queens-park-fc                        0.897          -0.951                  74.6                  72.4                                                                                                                                                                 
   3 │ arbroath                              0.873          -0.829                  74.2                  72.0                                                                                                                                                                 
   4 │ airdrieonians                         0.75           -0.494                  71.7                  69.4                                                                                                                                                                 
   5 │ inverness-caledonian-thistle          0.621          -1.16                   69.1                  66.6                                                                                                                                                                 
   6 │ dunfermline-athletic                  0.596          -1.007                  68.5                  66.0                                                                                                                                                                 
   7 │ queen-of-the-south                    0.465          -0.248                  65.6                  63.0                                                                                                                                                                 
   8 │ falkirk-fc                            0.381          -0.835                  63.7                  61.1                                                                                                                                                                 
   9 │ cove-rangers                          0.319          -0.483                  62.3                  59.6                                                                                                                                                                 
  10 │ east-kilbride                         0.228           0.3                    60.1                  57.4                                                                                                                                                                 
  11 │ peterhead                             0.209           0.787                  59.7                  56.9                                                                                                                                                                 
  12 │ alloa-athletic                        0.168          -0.292                  58.7                  55.9                                                                                                                                                                 
  13 │ stenhousemuir                         0.156           0.248                  58.4                  55.6                                                                                                                                                                 
  14 │ montrose                              0.15           -0.125                  58.2                  55.5                                                                                                                                                                 
  15 │ kelty-hearts-fc                      -0.061          -0.098                  53.0                  50.2                                                                                                                                                                 
  16 │ partick-thistle                      -0.136          -0.086                  51.1                  48.3                                                                                                                                                                 
  17 │ stranraer                            -0.158           0.426                  50.6                  47.8                                                                                                                                                                 
  18 │ east-fife                            -0.183           0.751                  50.0                  47.2                                                                                                                                                                 
  19 │ the-spartans-fc                      -0.24           -0.008                  48.6                  45.7                                                                                                                                                                 
  20 │ stirling-albion                      -0.331           0.11                   46.3                  43.5                                                                                                                                                                 
  21 │ edinburgh-city-fc                    -0.349          -0.081                  45.8                  43.0                                                                                                                                                                 
  22 │ dumbarton                            -0.349           0.412                  45.8                  43.1                                                                                                                                                                 
  23 │ clyde-fc                             -0.383           0.767                  45.0                  42.2                                                                                                                                                                 
  24 │ elgin-city                           -0.417           0.422                  44.1                  41.4                                                                                                                                                                 
  25 │ bonnyrigg-rose                       -0.488           0.48                   42.4                  39.7                                                                                                                                                                 
  26 │ annan-athletic                       -0.58            0.378                  40.2                  37.5                                                                                                                                                                 
  27 │ cowdenbeath                          -0.657           1.068                  38.3                  35.7                                                                                                                                                                 
  28 │ brechin-city                         -0.753           1.286                  36.1                  33.5                                                                                                                                                                 
  29 │ albion-rovers                        -0.752           0.279                  36.1                  33.6                                                                                                                                                                 
  30 │ forfar-athletic                      -0.894           0.112                  32.9                  30.5        

=#


function check_reality(team_name, model_df, historical_df)
    # 1. Get the model's prediction from your leaderboard
    model_row = filter(row -> row.Team == team_name, model_df)
    if nrow(model_row) == 0
        println("Team not found in leaderboard!")
        return
    end
    predicted_home_prob = model_row.Prob_First_Goal_Home[1]
    
    # 2. Get the actual historical reality from your dataset
    home_games = filter(row -> row.home_team == team_name && row.scored_first != "None", historical_df)
    total_home_games = nrow(home_games)
    
    actual_first_goals = count(==("Home"), home_games.scored_first)
    actual_prob = (actual_first_goals / total_home_games) * 100
    
    # 3. Print the comparison
    println("=== Reality Check: $team_name ===")
    println("Sample Size:   $total_home_games home games")
    println("Model Predicts: $(round(predicted_home_prob, digits=1))% chance to score first")
    println("Actual History: $(round(actual_prob, digits=1))% actually scored first")
    
    diff = abs(predicted_home_prob - actual_prob)
    println("Difference:     $(round(diff, digits=1))%")
end

# Let's test your best and worst teams!
check_reality("hamilton-academical", team_leaderboard, df)
check_reality("forfar-athletic", team_leaderboard, df)



#=
julia> # Let's test your best and worst teams!
       check_reality("hamilton-academical", team_leaderboard, df)
=== Reality Check: hamilton-academical ===
Sample Size:   26 home games
Model Predicts: 80.4% chance to score first
Actual History: 80.8% actually scored first
Difference:     0.4%

julia> check_reality("forfar-athletic", team_leaderboard, df)
=== Reality Check: forfar-athletic ===
Sample Size:   80 home games
Model Predicts: 32.9% chance to score first
Actual History: 40.0% actually scored first
Difference:     7.1%



=#


function build_reality_check_table(model_df, historical_df)
    # 1. Create an empty DataFrame
    reality_df = DataFrame(
        Team = String[],
        Home_Games = Int[],
        Predicted_Prob = Float64[],
        Actual_Prob = Float64[],
        Diff = Float64[]
    )
    
    # 2. Loop through every team in the leaderboard
    for row in eachrow(model_df)
        team = row.Team
        predicted = row.Prob_First_Goal_Home
        
        # Filter for their home games where a goal was actually scored
        home_games = filter(r -> r.home_team == team && r.scored_first != "None", historical_df)
        n_games = nrow(home_games)
        
        if n_games > 0
            # Calculate actual historical success rate
            actual_first_goals = count(==("Home"), home_games.scored_first)
            actual_prob = (actual_first_goals / n_games) * 100
            
            # Calculate the absolute difference
            diff = abs(predicted - actual_prob)
            
            push!(reality_df, (team, n_games, predicted, actual_prob, diff))
        end
    end
    
    # 3. Clean up the formatting for display
    reality_df.Predicted_Prob = round.(reality_df.Predicted_Prob, digits=1)
    reality_df.Actual_Prob = round.(reality_df.Actual_Prob, digits=1)
    reality_df.Diff = round.(reality_df.Diff, digits=1)
    
    # Let's sort by the biggest difference to see where the model disagrees with history most
    sort!(reality_df, :Diff, rev=true)
    
    return reality_df
end

# Run the function
reality_table = build_reality_check_table(team_leaderboard, df)
display(reality_table)

#=
julia> display(reality_table)
30×5 DataFrame
 Row │ Team                          Home_Games  Predicted_Prob  Actual_Prob  Diff    
     │ String                        Int64       Float64         Float64      Float64 
─────┼────────────────────────────────────────────────────────────────────────────────
   1 │ the-spartans-fc                       43            48.6         58.1      9.5
   2 │ bonnyrigg-rose                        53            42.4         50.9      8.5
   3 │ elgin-city                            90            44.1         52.2      8.1
   4 │ stranraer                             84            50.6         58.3      7.7
   5 │ brechin-city                           7            36.1         28.6      7.5
   6 │ forfar-athletic                       80            32.9         40.0      7.1
   7 │ east-kilbride                         12            60.1         66.7      6.6
   8 │ albion-rovers                         40            36.1         42.5      6.4
   9 │ stirling-albion                       88            46.3         52.3      6.0
  10 │ queen-of-the-south                    65            65.6         60.0      5.6
  11 │ queens-park-fc                        24            74.6         79.2      4.6
  12 │ stenhousemuir                         83            58.4         62.7      4.3
  13 │ alloa-athletic                        77            58.7         54.5      4.2
  14 │ montrose                              85            58.2         54.1      4.1
  15 │ cowdenbeath                           24            38.3         41.7      3.4
  16 │ edinburgh-city-fc                     92            45.8         48.9      3.1
  17 │ dumbarton                             86            45.8         48.8      3.0
  18 │ cove-rangers                          69            62.3         59.4      2.9
  19 │ east-fife                             87            50.0         52.9      2.9
  20 │ kelty-hearts-fc                       77            53.0         50.6      2.4
  21 │ clyde-fc                              85            45.0         47.1      2.1
  22 │ inverness-caledonian-thistle          28            69.1         67.9      1.2
  23 │ annan-athletic                        87            40.2         41.4      1.2
  24 │ partick-thistle                        8            51.1         50.0      1.1
  25 │ falkirk-fc                            59            63.7         62.7      1.0
  26 │ arbroath                              16            74.2         75.0      0.8
  27 │ airdrieonians                         45            71.7         71.1      0.6
  28 │ hamilton-academical                   26            80.4         80.8      0.4
  29 │ dunfermline-athletic                  16            68.5         68.8      0.2
  30 │ peterhead                             87            59.7         59.8      0.1


=#



#
function predict_matchup(home_team, away_team, chain, team_to_id)
    # 1. Convert chain to summary stats so we can grab the parameters
    chain_summary = DataFrame(summarystats(chain))
    
    # 2. Grab the global league baseline and home advantage
    μ = chain_summary[chain_summary.parameters .== :μ, :mean][1]
    home_adv = chain_summary[chain_summary.parameters .== :home_adv, :mean][1]
    
    # 3. Lookup the integer IDs for our two teams
    if !haskey(team_to_id, home_team) || !haskey(team_to_id, away_team)
        println("Error: One of those teams isn't in our dictionary!")
        return
    end
    h_id = team_to_id[home_team]
    a_id = team_to_id[away_team]
    
    # 4. Grab Home Attack and Away Defense
    att_h = chain_summary[chain_summary.parameters .== Symbol("att[$h_id]"), :mean][1]
    def_a = chain_summary[chain_summary.parameters .== Symbol("def[$a_id]"), :mean][1]
    
    # 5. The Magic Formula: Calculate the logit, then squash to probability
    logit_p = μ + home_adv + att_h + def_a
    home_prob = logistic(logit_p) * 100
    away_prob = 100.0 - home_prob # Since someone has to score first (excluding 0-0s)
    
    # 6. Print the beautiful results
    println("\n🏆 MATCHUP PREDICTION 🏆")
    println("Home: $home_team")
    println("Away: $away_team")
    println("---------------------------------")
    println("$home_team to score first: $(round(home_prob, digits=1))%")
    println("$away_team to score first: $(round(away_prob, digits=1))%")
end

# Test it out! Let's put your #1 team against your #30 team.
predict_matchup("hamilton-academical", "forfar-athletic", chain, team_to_id)
predict_matchup("hamilton-academical", "kelty-hearts-fc", chain, team_to_id)

#=
🏆 MATCHUP PREDICTION 🏆
Home: hamilton-academical
Away: forfar-athletic
---------------------------------
hamilton-academical to score first: 82.1%
forfar-athletic to score first: 17.9%


🏆 MATCHUP PREDICTION 🏆
Home: hamilton-academical
Away: kelty-hearts-fc
---------------------------------
hamilton-academical to score first: 78.8%
kelty-hearts-fc to score first: 21.2%

=#
kelty-hearts-fc

subset(df, :home_team => ByRow(isequal("hamilton-academical")), :away_team => ByRow(isequal("kelty-hearts-fc"))) 
:away_team => ByRow(isequal("forfar-athletic")))

###
# Sum of Red Cards in a match
df.total_reds = df.HR .+ df.AR

function analyze_red_cards(df)
    clean_games = filter(row -> row.total_reds == 0, df)
    dirty_games = filter(row -> row.total_reds > 0, df)
    
    println("\n=== Red Card Analysis ===")
    println("Clean Games Count: $(nrow(clean_games))")
    println("Red Card Games Count: $(nrow(dirty_games))")
    
    # Compare Variance of the Scoreline (Volatility)
    var_clean = var(clean_games.home_score .- clean_games.away_score)
    var_dirty = var(dirty_games.home_score .- dirty_games.away_score)
    
    println("Score Variance (Clean): $(round(var_clean, digits=2))")
    println("Score Variance (Red Card): $(round(var_dirty, digits=2))")
    
    # Did the red card team lose badly?
    # (Complex to check without minute-by-minute, but we can check correlation)
    cor_red = cor(df.HR .- df.AR, df.home_score .- df.away_score)
    println("Correlation (Home Red Diff vs Goal Diff): $(round(cor_red, digits=3))")
    # We expect a strong NEGATIVE correlation (More Home Reds = Lower Home Score)
end

analyze_red_cards(df)

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
dropmissing!(df, [:HS, :AS, :HC, :AC, :HF, :AF, :Referee])

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

using GLM

model_shots = lm(@formula(foul_diff ~ shot_diff), df)
model_goals = lm(@formula(foul_diff ~ goal_diff), df)
model_shots_goals = lm(@formula(foul_diff ~ shot_diff + goal_diff), df)



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

###
using Dates

df.Month = month.(df.match_date)

function analyze_seasonality(df)
    # Group by Month
    monthly = combine(groupby(df, :Month), 
        [:home_score, :away_score] => ((h, a) -> mean(h .+ a)) => :avg_goals,
        [:HS, :AS] => ((h, a) -> mean(h .+ a)) => :avg_shots
    )
    
    sort!(monthly, :Month)
    println("\n=== Monthly Averages ===")
    println(monthly)
end

analyze_seasonality(df)


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

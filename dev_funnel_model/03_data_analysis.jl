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
dropmissing!(df, [:HS, :AS, :HC, :AC, :HF, :AF])

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


phi_shot_home = sum(df.HS) / sum(df.HC)
phi_shot_away = sum(df.AS) / sum(df.AC)
phi_shot_all = sum(vcat(df.HS ,df.AS)) / sum(vcat(df.HC, df.AC))

phi_goal_home = sum(df.home_score) / sum(df.HC)
phi_goal_away = sum(df.away_score) / sum(df.AC)
phi_goal_all = sum(vcat(df.home_score ,df.away_score)) / sum(vcat(df.HC, df.AC))






# dev_funnel_model/02_feature_set.jl

using Revise
using BayesianFootball

pinthreads(:cores)
ds = Data.load_extra_ds()





# ----------------------------------------------
# 2. Experiment Configs Set up 
# ----------------------------------------------

# --- setup 1 
cv_config = BayesianFootball.Data.CVConfig(
    # tournament_ids = [56,57],
    tournament_ids = [56],
    target_seasons = ["24/25"],
    history_seasons = 0,
    dynamics_col = :match_week,
    # warmup_period = 36,
    warmup_period = 36,
    stop_early = true
)

splits = BayesianFootball.Data.create_data_splits(ds, cv_config)
model = BayesianFootball.Models.PreGame.StaticPoisson() # place holder
# vocabulary = BayesianFootball.Features.create_vocabulary(ds, model) 
feature_sets = BayesianFootball.Features.create_features(
    splits, model, cv_config
)
feature_sets

###

struct test_model <: BayesianFootball.AbstractFunnelModel end

feature_sets = BayesianFootball.Features.create_features(
  splits, test_model(), cv_config
)
feature_sets[1]



# In your REPL or dev script
funnel_model = BayesianFootball.Models.PreGame.SequentialFunnelModel()

feature_sets = BayesianFootball.Features.create_features(
    splits, funnel_model, cv_config
)

# Compile and Sample (Fast run to check for errors)
# It will be slower than Poisson because it has 6 GRW chains instead of 2!
using Turing
turing_model = BayesianFootball.Models.PreGame.build_turing_model(funnel_model, feature_sets[1][1])
chain = sample(turing_model, NUTS(20, 0.65), 10)



## --- 

using Revise
using BayesianFootball

using DataFrames
using Statistics
using ThreadPinning
using LinearAlgebra
pinthreads(:cores)
BLAS.set_num_threads(1) 


ds = Data.load_extra_ds()
#=
julia> names(df)
26-element Vector{String}:
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
=#

transform!(ds.matches, :match_week => ByRow(w -> cld(w, 4)) => :match_month)

df = subset(ds.matches, :tournament_id => ByRow(in(56)), :season => ByRow(isequal("24/25")))
df[:, [:home_team, :away_team, :match_date, :match_week, :match_month]] 


cv_config = BayesianFootball.Data.CVConfig(
    # tournament_ids = [56,57],
    tournament_ids = [56],
    target_seasons = ["24/25"],
    history_seasons = 0,
    dynamics_col = :match_month,
    # warmup_period = 36,
    warmup_period = 2,
    stop_early = true
)


splits = BayesianFootball.Data.create_data_splits(ds, cv_config)
train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=1) 
sampler_conf = Samplers.NUTSConfig(
                100,
                16,
                100,
                0.65,
                10,
  Samplers.UniformInit(-0.05, 0.05),
                :perchain,
)

training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)


funnel_model = BayesianFootball.Models.PreGame.SequentialFunnelModel()

conf_funnel = Experiments.ExperimentConfig(
                    name = "grw funnel_model big",
                    model = funnel_model,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./dev_data/"
)


results_funnel = Experiments.run_experiment(ds, conf_funnel)

Experiments.save_experiment(results_funnel)

(chain_1, meta_1) = results_funnel.training_results[4];

df_test_1 = BayesianFootball.Data.get_next_matches(ds, meta_1, results_funnel.config.splitter);

println("Test Matches Found: ", nrow(df_test_1))
show(df_test_1[:, [:match_date, :home_team, :away_team]], allcols=false)


feature_collection1 = BayesianFootball.Features.create_features(
    BayesianFootball.Data.create_data_splits(ds, results_funnel.config.splitter),
    funnel_model, 
    results_funnel.config.splitter
)

feature_set1 = feature_collection1[4][1]

chain1 = results_funnel.training_results[4][1]

describe(chain1)

model_preds_1 = BayesianFootball.Models.PreGame.extract_parameters(
    results_funnel.config.model,
    df_test_1,
    feature_set1,
    chain1
)


df_test_1

using Statistics, Printf

function summarize_funnel_predictions(preds::Dict, df::DataFrame)
    # Header
    @printf("%-35s | %-12s | %-10s | %-12s | %-10s\n", 
            "Match", "Pred Shots", "Act Shots", "Pred Goals", "Act Goals")
    println("-"^90)
    
    for row in eachrow(df)
        mid = row.match_id
        if haskey(preds, mid)
            p = preds[mid]
            
            # 1. Volume (Shots) - Take Mean of Samples
            # We look at Home Team only for brevity
            pred_shots = mean(p.λ_shots_h)
            
            # 2. Goals (xG) - Take Mean of Samples
            pred_goals = mean(p.exp_goals_h)
            
            # 3. Actuals
            # Use 'get' to handle missing columns safely if checking raw df
            act_shots = get(row, :HS, NaN) 
            act_goals = row.home_score
            
            match_str = "$(row.home_team)"
            
            @printf("%-35s | %-12.2f | %-10.0f | %-12.2f | %-10d\n", 
                    match_str, pred_shots, act_shots, pred_goals, act_goals)
        end
    end
end

# Run it
summarize_funnel_predictions(model_preds_1, df_test_1)


using Distributions, Statistics, LinearAlgebra

function dev_compute_probabilities(rates, max_goals=10)
    # rates is the NamedTuple from model_preds_1[match_id]
    n_samples = length(rates.λ_shots_h)
    
    # 1. Initialize Score Grid (Rows=Home, Cols=Away)
    score_matrix = zeros(Float64, max_goals+1, max_goals+1)
    
    # 2. Monte Carlo Loop (Reconstruct the Match)
    for i in 1:n_samples
        # --- GLOBAL PARAMETERS ---
        r = rates.r_create[i]

        # --- HOME TEAM SIMULATION ---
        # A. Shots (Negative Binomial)
        # Distributions.jl uses (r, p) where p = r / (r + μ)
        λ_h = rates.λ_shots_h[i]
        p_h = r / (r + λ_h)
        # Clamp to avoid numerical errors
        p_h = clamp(p_h, 1e-8, 1.0 - 1e-8)
        
        n_shots_h = rand(NegativeBinomial(r, p_h))
        
        # B. On Target (Binomial)
        θ_h = rates.θ_prec_h[i]
        n_sot_h = rand(Binomial(n_shots_h, θ_h))
        
        # C. Goals (Binomial)
        ϕ_h = rates.ϕ_conv_h[i]
        goals_h = rand(Binomial(n_sot_h, ϕ_h))
        
        # --- AWAY TEAM SIMULATION ---
        λ_a = rates.λ_shots_a[i]
        p_a = r / (r + λ_a)
        p_a = clamp(p_a, 1e-8, 1.0 - 1e-8)
        
        n_shots_a = rand(NegativeBinomial(r, p_a))
        
        θ_a = rates.θ_prec_a[i]
        n_sot_a = rand(Binomial(n_shots_a, θ_a))
        
        ϕ_a = rates.ϕ_conv_a[i]
        goals_a = rand(Binomial(n_sot_a, ϕ_a))

        # --- RECORD OUTCOME ---
        # Clamp to grid size (e.g., 11+ goals go into the 10 bucket)
        idx_h = min(goals_h, max_goals) + 1
        idx_a = min(goals_a, max_goals) + 1
        
        score_matrix[idx_h, idx_a] += 1.0
    end
    
    # 3. Normalize to Probabilities
    prob_matrix = score_matrix ./ sum(score_matrix)
    
    return prob_matrix
end

# --- Helper to view odds ---
function print_implied_odds(grid)
    # Sum triangles for 1x2
    p_home = sum(tril(grid, -1))
    p_draw = sum(diag(grid))
    p_away = sum(triu(grid, 1))
    
    println("\n--- Match Probabilities ---")
    println("Home Win: $(round(p_home * 100, digits=1))%  (Odds: $(round(1/p_home, digits=2)))")
    println("Draw:     $(round(p_draw * 100, digits=1))%  (Odds: $(round(1/p_draw, digits=2)))")
    println("Away Win: $(round(p_away * 100, digits=1))%  (Odds: $(round(1/p_away, digits=2)))")
    
    return (home=p_home, draw=p_draw, away=p_away)
end



df_test_1[1, :match_id]
# 1. Grab rates for a match
mid = 12476645

mid = df_test_1[2, :match_id]
rates = model_preds_1[mid]

# 2. Run Simulation
grid = dev_compute_probabilities(rates)

# 3. Check Results
print_implied_odds(grid)
subset(ds.odds, :match_id => ByRow(isequal(mid)), :market_group => ByRow(isequal("1X2")), :market_name => ByRow(isequal("Full time")))[:, [:match_id, :choice_name, :decimal_odds]]
subset(ds.matches, :match_id => ByRow(isequal(mid)))


# Optional: View the score grid (Correct Score)
# displaying top 3x3
display(grid[1:4, 1:4])



# 1. Get the training features (from the split used for training)
train_fs = feature_sets[1][1] 

# 2. Check the Home Shots column
h_shots = train_fs.data[:flat_home_shots]

println("--- Training Data Inspection ---")
println("Total Rows: ", length(h_shots))
println("Sum of Shots: ", sum(h_shots))
println("Max Shots: ", maximum(h_shots))
println("Count of Zeros: ", count(==(0), h_shots))
println("First 20 values: ", h_shots[1:min(20, end)])



#####

market_data = Data.prepare_market_data(ds)

latents = BayesianFootball.Experiments.extract_oos_predictions(ds, results_funnel)

ppd = BayesianFootball.Predictions.model_inference(latents)


baker = BayesianKelly()
flat_strat = FlatStake(0.05)

my_signals = [baker, flat_strat]

ledger = BayesianFootball.BackTesting.run_backtest(
    ds, 
    results_funnel, 
    my_signals; 
    market_config = Data.Markets.DEFAULT_MARKET_CONFIG
)



tearsheet = BayesianFootball.BackTesting.generate_tearsheet(ledger)




model_names = unique(tearsheet.selection)

for m_name in model_names
    println("\nStats for: $m_name")
    sub = subset(tearsheet, :selection => ByRow(isequal(m_name)))
    show(sub)
end


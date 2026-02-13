# exp/funnel/analysis.jl

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
exp_dir = "./data/exp/funnel_basics"
println("Scanning for results in: $exp_dir")

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




baker = BayesianKelly()
as = AnalyticalShrinkageKelly()
kelly = KellyCriterion(1)
kelly25 = KellyCriterion(1/4)
flat_strat = FlatStake(0.05)

my_signals = [baker]
my_signals = [as]
my_signals = [baker, as, kelly, kelly25, flat_strat]

my_signals = [flat_strat]

# Run backtest on ALL loaded models at once
ledger = BayesianFootball.BackTesting.run_backtest(
    ds, 
  loaded_results, 
    my_signals; 
    market_config = Data.Markets.DEFAULT_MARKET_CONFIG
)

# 3. Analyze
# ==========
tearsheet = BayesianFootball.BackTesting.generate_tearsheet(ledger)

println("\n=== TEARSHEET SUMMARY ===")
println(tearsheet)

# Breakdown by Model (Selection)
println("\n=== BREAKDOWN BY MODEL ===")
model_names = unique(tearsheet.selection)

for m_name in model_names
    println("\nStats for: $m_name")
    sub = DataFrames.subset(tearsheet, :selection => ByRow(isequal(m_name)))
    show(sub)
end

for m in loaded_results 
println(m.config.model)
println("\n")
end


# ------------------------------------------------------------------------
# Rank Strategies 
# ------------------------------------------------------------------------


using DataFrames, Distributions, Statistics, StatsBase

function rank_strategies(ledger::DataFrame)
    group_cols = [:model_name, :model_parameters, :signal_name, :market_name, :selection]

    results = combine(groupby(ledger, group_cols)) do df
        # 1. Filter for active bets
        active_bets = filter(row -> abs(row.stake) > 1e-6, df)
        n_bets = nrow(active_bets)
        bet_freq = n_bets / nrow(df)
        
        # 2. Skip insignificant strategies (less than 10 bets)
        if n_bets < 10
            return (Win_Rate=NaN, Growth_Rate=NaN, Exp_Value=NaN, 
                    Kelly_Risk_Theta=NaN, Edge_Ratio=NaN, Avg_Stake=NaN, Bet_Freq=bet_freq)
        end

        # 3. Prepare vectors
        wins = filter(>(0), active_bets.pnl)
        losses = abs.(filter(<(0), active_bets.pnl)) 

        # 4. Safe Gamma Fitting
        theta_loss = NaN
        edge_ratio = NaN

        if length(losses) >= 5 && var(losses) > 1e-8
            try
                d_loss = fit(Gamma, losses)
                theta_loss = params(d_loss)[2] # Theta (Scale)
                
                if !isempty(wins) && length(wins) >= 5 && var(wins) > 1e-8
                    d_win = fit(Gamma, wins)
                    edge_ratio = mean(d_win) / mean(d_loss)
                else
                    edge_ratio = mean(wins) / mean(losses) # Fallback to simple mean ratio
                end
            catch e
                theta_loss = NaN # Fit failed (likely data issues)
            end
        elseif !isempty(losses)
            # Fallback for constant losses: Variance is 0, so risk is just the magnitude
            theta_loss = mean(losses) 
            edge_ratio = isempty(wins) ? 0.0 : mean(wins) / mean(losses)
        else
            theta_loss = 0.0 # No losses
            edge_ratio = Inf
        end

        # 5. Expected Log-Growth
        g = mean(log.(1.0 .+ active_bets.pnl))

        return (
            Win_Rate = length(wins) / n_bets,
            Growth_Rate = g,
            Exp_Value = mean(active_bets.pnl),
            Kelly_Risk_Theta = theta_loss, 
            Edge_Ratio = edge_ratio,      
            Avg_Stake = mean(active_bets.stake),
            Bet_Freq = bet_freq
        )
    end

    # Remove rows where calculation failed completely (NaN growth)
    filter!(row -> !isnan(row.Growth_Rate), results)
    
    # Sort
    sort!(results, :Growth_Rate, rev=true)
    
    return results
end

# Run the fixed function
strategy_rankings = rank_strategies(ledger.df)

model_names = unique(strategy_rankings.selection)
for m_name in model_names
    println("\nStats for: $m_name")
    sub = DataFrames.subset(strategy_rankings, :selection => ByRow(isequal(m_name)))
  show(sort(sub, :Growth_Rate, rev=true))
end


ids_no_2425 =DataFrames.subset(ds.matches, :season => ByRow(!isequal("24/25"))).match_id
ledger_no_2425 = DataFrames.subset(ledger.df, :match_id => ByRow(in(ids_no_2425)))

strategy_rankings = rank_strategies(ledger_no_2425)

model_names = unique(strategy_rankings.selection)
for m_name in model_names
    println("\nStats for: $m_name")
    sub = DataFrames.subset(strategy_rankings, :selection => ByRow(isequal(m_name)))
  show(sort(sub, :Growth_Rate, rev=true))
end
#= 
Stats for: over_15                                                                                                                                                                                                                                                                                                          
2×12 DataFrame                                                                                                                                                                                                                                                                                                              
 Row │ model_name             model_parameters                   signal_name    market_name  selection  Win_Rate  Growth_Rate  Exp_Value   Kelly_Risk_Theta  Edge_Ratio  Avg_Stake  Bet_Freq                                                                                                                                
     │ String                 String                             String         String       Symbol     Float64   Float64      Float64     Float64           Float64     Float64    Float64                                                                                                                                 
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                              
   1 │ GRWNegativeBinomialMu  μ_init=Normal(μ=0.2, σ=0.1), σ_μ…  BayesianKelly  OverUnder    over_15    0.842105   0.00697924  0.00721854               NaN         NaN  0.035475   0.092233                                                                                                                                
   2 │ SequentialFunnelModel  creation_μ=Normal(μ=2.5, σ=0.3),…  BayesianKelly  OverUnder    over_15    0.829787   0.00276865  0.0030352                NaN         NaN  0.0322606  0.0912621                                                                                                                               
Stats for: over_05                                                                                                                                                                                                                                                                                                          
2×12 DataFrame                                                                                                                                                                                                                                                                                                              
 Row │ model_name             model_parameters                   signal_name    market_name  selection  Win_Rate  Growth_Rate  Exp_Value   Kelly_Risk_Theta  Edge_Ratio  Avg_Stake  Bet_Freq                                                                                                                                
     │ String                 String                             String         String       Symbol     Float64   Float64      Float64     Float64           Float64     Float64    Float64                                                                                                                                 
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                              
   1 │ GRWNegativeBinomialMu  μ_init=Normal(μ=0.2, σ=0.1), σ_μ…  BayesianKelly  OverUnder    over_05      1.0      0.0039936   0.00401291         0.0        Inf         0.0491027  0.0106796                                                                                                                               
   2 │ SequentialFunnelModel  creation_μ=Normal(μ=2.5, σ=0.3),…  BayesianKelly  OverUnder    over_05      0.9375   0.00155642  0.00156633         0.0108675    0.220405  0.0335272  0.015534                                                                                                                                
Stats for: under_55                                                                                                                                                                                                                                                                                                         
2×12 DataFrame                                                                                                                                                                                                                                                                                                              
 Row │ model_name             model_parameters                   signal_name    market_name  selection  Win_Rate  Growth_Rate  Exp_Value   Kelly_Risk_Theta  Edge_Ratio  Avg_Stake  Bet_Freq                                                                                                                                
     │ String                 String                             String         String       Symbol     Float64   Float64      Float64     Float64           Float64     Float64    Float64                                                                                                                                 
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                              
   1 │ GRWNegativeBinomialMu  μ_init=Normal(μ=0.2, σ=0.1), σ_μ…  BayesianKelly  OverUnder    under_55   1.0        0.00334072  0.00336079        0.0         Inf         0.051876   0.0296571                                                                                                                               
   2 │ SequentialFunnelModel  creation_μ=Normal(μ=2.5, σ=0.3),…  BayesianKelly  OverUnder    under_55   0.941176   0.00154694  0.0015563         0.00753706    0.281892  0.0338208  0.0315107                                                                                                                               
Stats for: over_25                                                                                                                                                                                                                                                                                                          
2×12 DataFrame                                                                                                                                                                                                                                                                                                              
 Row │ model_name             model_parameters                   signal_name    market_name  selection  Win_Rate  Growth_Rate  Exp_Value   Kelly_Risk_Theta  Edge_Ratio  Avg_Stake  Bet_Freq                                                                                                                                
     │ String                 String                             String         String       Symbol     Float64   Float64      Float64     Float64           Float64     Float64    Float64                                                                                                                                 
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                               
   1 │ GRWNegativeBinomialMu  μ_init=Normal(μ=0.2, σ=0.1), σ_μ…  BayesianKelly  OverUnder    over_25    0.510345   0.00288415  0.00446997               NaN         NaN  0.0351508  0.266299                                                                                                                                
   2 │ SequentialFunnelModel  creation_μ=Normal(μ=2.5, σ=0.3),…  BayesianKelly  OverUnder    over_25    0.531773   6.99728e-5  0.00120116               NaN         NaN  0.0300949  0.274564                                                                                                                                
Stats for: draw                                                                                                                                                                                                                                                                                                             
2×12 DataFrame                                                                                                                                                                                                                                                                                                              
 Row │ model_name             model_parameters                   signal_name    market_name  selection  Win_Rate  Growth_Rate  Exp_Value  Kelly_Risk_Theta  Edge_Ratio  Avg_Stake  Bet_Freq                                                                                                                                 
     │ String                 String                             String         String       Symbol     Float64   Float64      Float64    Float64           Float64     Float64    Float64                                                                                                                                  
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                               
   1 │ SequentialFunnelModel  creation_μ=Normal(μ=2.5, σ=0.3),…  BayesianKelly  1X2          draw       0.164557   0.00176057  0.0042214               NaN         NaN  0.0112474  0.0701599                                                                                                                                
   2 │ GRWNegativeBinomialMu  μ_init=Normal(μ=0.2, σ=0.1), σ_μ…  BayesianKelly  1X2          draw       0.180952   0.00147508  0.0046709               NaN         NaN  0.0133995  0.0932504                                                                                                                                
Stats for: home                                                                                                                                                                                                                                                                                                             
2×12 DataFrame                                                                                                                                                                                                                                                                                                              
 Row │ model_name             model_parameters                   signal_name    market_name  selection  Win_Rate  Growth_Rate  Exp_Value    Kelly_Risk_Theta  Edge_Ratio  Avg_Stake  Bet_Freq                                                                                                                               
     │ String                 String                             String         String       Symbol     Float64   Float64      Float64      Float64           Float64     Float64    Float64                                                                                                                                
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                              
   1 │ SequentialFunnelModel  creation_μ=Normal(μ=2.5, σ=0.3),…  BayesianKelly  1X2          home       0.358407   0.00021373   0.00373537               NaN         NaN  0.0349147  0.401421                                                                                                                               
   2 │ GRWNegativeBinomialMu  μ_init=Normal(μ=0.2, σ=0.1), σ_μ…  BayesianKelly  1X2          home       0.326146  -0.0095712   -0.00520896               NaN         NaN  0.0428142  0.329485                                                                                                                               
Stats for: over_35                                                                                                                                                                                


=#


##### clv regression 

using DataFrames, Statistics, GLM
# backtesting deconstructed 
exp_1 = loaded_results[1]

market_data = Data.prepare_market_data(ds)

latents = BayesianFootball.Experiments.extract_oos_predictions(ds, exp_1)

ppd = BayesianFootball.Predictions.model_inference(latents)

using DataFrames, Statistics, GLM

model_features = transform(ppd.df, :distribution => ByRow(mean) => :prob_model)
select!(model_features, :match_id, :market_name, :market_line, :selection, :prob_model)


analysis_df = innerjoin(
    market_data.df,
    model_features,
    on = [:match_id, :market_name, :market_line, :selection]
)

dropmissing!(analysis_df, [:odds_close, :is_winner])

#= 
julia> names(analysis_df)
21-element Vector{String}:
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
 "prob_model"

=#

analysis_df.spread = analysis_df.prob_model .- analysis_df.prob_implied_close
analysis_df.spread_fair = analysis_df.prob_model .- analysis_df.prob_fair_close
analysis_df.Y = Float64.(analysis_df.is_winner)

reg_model = glm(@formula(Y ~ prob_implied_close + spread), analysis_df, Binomial(), LogitLink())


# Simple Flat Stake Backtest
# Bet when your model sees > 5% edge (0.05 prob diff)

betting_df = filter(row -> row.spread > 0.05, analysis_df)

# ROI Calculation
# Profit = (Odds - 1) if Win, -1 if Loss
betting_df.profit = [r.is_winner ? (r.odds_close - 1) : -1.0 for r in eachrow(betting_df)]

println("Total Bets: ", nrow(betting_df))
println("Total Profit (Units): ", sum(betting_df.profit))
println("ROI: ", mean(betting_df.profit) * 100, "%")


###
#
target_selections = [:home] # The specific lines


# 2. Create the "Winning Portfolio" DataFrame
df_target = filter(row -> 
    row.selection in target_selections, 
    analysis_df
)

# 3. Create the "Losing Portfolio" DataFrame (Unders + 1X2 + BTTS)
df_rest = filter(row -> 
    !(row.selection in target_selections), # Everything NOT in the list above
    analysis_df
)



model_target = glm(@formula(Y ~ prob_implied_close + spread), df_target, Binomial(), LogitLink())
model_rest = glm(@formula(Y ~ prob_implied_close + spread), df_rest, Binomial(), LogitLink())


function fit_glm_by_market_selection(df::AbstractDataFrame, market::AbstractVector{Symbol} )
  df_target = DataFrames.subset( df,
                                :selection => ByRow(in(market))
                                )
                    
  df_rest = DataFrames.subset( df,
                                :selection => ByRow(!in(market))
                                )
                    
  model_target = glm(@formula(Y ~ prob_implied_close + spread), df_target, Binomial(), LogitLink())
  model_rest = glm(@formula(Y ~ prob_implied_close + spread), df_rest, Binomial(), LogitLink())

  return (
        model_target,
        model_rest
        )
end


a, b = fit_glm_by_market_selection(analysis_df, [:under_55])



using DataFrames, Statistics, Printf

function analyze_aggregate_performance(latents_df, matches_df)
    println("\n============================================================")
    println("          LEAGUE-WIDE AGGREGATE CALIBRATION REPORT          ")
    println("============================================================")
    
    # Storage for aggregation
    total_pred_shots = 0.0; total_act_shots = 0.0
    total_pred_sot   = 0.0; total_act_sot   = 0.0
    total_pred_goals = 0.0; total_act_goals = 0.0
    
    match_count = 0
    
    # Iterate through all matches in the latent predictions
    for row in eachrow(latents_df)
        mid = row.match_id
        
        # Find actuals
        m_idx = findfirst(==(mid), matches_df.match_id)
        if isnothing(m_idx) continue end
        m_row = matches_df[m_idx, :]
        
        # --- Accumulate (Home + Away combined) ---
        
        # 1. Shots
        p_shots = mean(row.λ_shots_h) + mean(row.λ_shots_a)
        a_shots = coalesce(m_row.HS, 0.0) + coalesce(m_row.AS, 0.0)
        
        # 2. Shots on Target (Expected SOT = Shots * Theta)
        # Note: We use the mean of the product (E[S * theta]) directly from samples
        # row.θ_prec_h is the rate, but the actual SOT count is implied.
        # Better: We can approximate E[SOT] ≈ E[Shots] * E[Theta] for summary, 
        # or use the samples if we saved the intermediate SOT integers.
        # Given your latents has theta, let's use: Pred_SOT = Pred_Shots * Pred_Theta
        p_sot_h = mean(row.λ_shots_h .* row.θ_prec_h)
        p_sot_a = mean(row.λ_shots_a .* row.θ_prec_a)
        p_sot   = p_sot_h + p_sot_a
        
        a_sot   = coalesce(m_row.HST, 0.0) + coalesce(m_row.AST, 0.0)
        
        # 3. Goals (xG)
        p_goals = mean(row.exp_goals_h) + mean(row.exp_goals_a)
        a_goals = coalesce(m_row.home_score, 0) + coalesce(m_row.away_score, 0)
        
        # Update Totals
        total_pred_shots += p_shots
        total_act_shots  += a_shots
        
        total_pred_sot   += p_sot
        total_act_sot    += a_sot
        
        total_pred_goals += p_goals
        total_act_goals  += a_goals
        
        match_count += 1
    end
    
    # --- Report Generation ---
    if match_count == 0
        println("No matching records found.")
        return
    end
    
    avg_pred_shots = total_pred_shots / match_count
    avg_act_shots  = total_act_shots / match_count
    
    avg_pred_sot   = total_pred_sot / match_count
    avg_act_sot    = total_act_sot / match_count
    
    avg_pred_goals = total_pred_goals / match_count
    avg_act_goals  = total_act_goals / match_count
    
    function print_stat(name, tot_p, tot_a, avg_p, avg_a)
        diff_pct = ((tot_p - tot_a) / tot_a) * 100
        status = abs(diff_pct) < 5.0 ? "✅ OK" : (diff_pct > 0 ? "⚠️ HIGH" : "⚠️ LOW")
        
        @printf("\n>> %s\n", name)
        @printf("   Total:   Pred %-8.1f  vs  Act %-8.1f  (Diff: %+.1f%%) %s\n", tot_p, tot_a, diff_pct, status)
        @printf("   Per Gm:  Pred %-8.2f  vs  Act %-8.2f\n", avg_p, avg_a)
    end

    println("Matches Analyzed: $match_count")
    
    print_stat("SHOTS (Volume)", total_pred_shots, total_act_shots, avg_pred_shots, avg_act_shots)
    print_stat("SOT (Precision)", total_pred_sot, total_act_sot, avg_pred_sot, avg_act_sot)
    print_stat("GOALS (Conversion)", total_pred_goals, total_act_goals, avg_pred_goals, avg_act_goals)
    
    println("\n============================================================")
end


analyze_aggregate_performance(latents.df, ds.matches)

simulate_smart_portfolio(ledger.df)


using DataFrames, Statistics, Printf

function analyze_edge_buckets(ledger)
    println("\n📊 ROI by Edge Magnitude (The 'Vig' Test)")
    println("==============================================")
    
    # Calculate the raw edge (Model Prob - Implied Prob)
    # We assume 'expected_value' in your ledger represents the edge or similar
    # If not, recalculate: edge = (model_prob - implied_prob)
    
    # Let's use the 'expected_value' column from your tearsheet logic if available,
    # or re-derive it. Assuming 'expected_value' is in the ledger:
    
    # Filter for Overs/Home (your "Smart" portfolio)
    base_df = filter(row -> 
        (contains(string(row.selection), "over") || row.selection == :home) && 
        abs(row.stake) > 1e-6, 
        ledger
    )

    # Define buckets
    thresholds = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15]
    
    @printf("%-12s | %-8s | %-8s | %-8s\n", "Min Edge", "Bets", "Win %", "ROI")
    println("-"^46)

    for t in thresholds
        # Filter bets with edge > t
        # Note: You might need to check column names. 
        # Usually: edge = row.expected_value (if it's decimal edge)
        subset = filter(row -> row.expected_value >= t, base_df)
        
        if nrow(subset) > 0
            roi = (sum(subset.pnl) / sum(subset.stake)) * 100
            win_rate = (count(x -> x > 0, subset.pnl) / nrow(subset)) * 100
            
            @printf("> %-10.2f | %-8d | %-8.1f%% | %-8.2f%%\n", 
                t, nrow(subset), win_rate, roi)
        end
    end
    println("==============================================\n")
end

analyze_edge_buckets(ledger.df)

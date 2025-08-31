include("/home/james/bet_project/models_julia/notebooks/08_eval/test_eval_setup.jl")
using BayesianFootball


data_files = DataFiles("/home/james/bet_project/football_data/scot_nostats_20_to_24")
data_store = DataStore(data_files)
r1 = load_experiment("/home/james/bet_project/models_julia/experiments/maher_basic_test_20250820_231427.jld2")
result = r1[1]
experiment_config = r1[2]
mapping = result.mapping
target_matches = filter(row -> row.season == experiment_config.cv_config.target_season, data_store.matches)

# set up
matches_prediction =  BayesianFootball.predict_target_season(target_matches, result, mapping)
matches_odds = BayesianFootball.process_matches_odds(data_store, target_matches)
matches_results = BayesianFootball.process_matches_results(data_store, target_matches)
matches_activelines = BayesianFootball.process_matches_active_minutes(data_store, target_matches)


###
# messing around, to predict ranger v celtic tomorrow 
mapping.team
t1 = "rangers"
t2 = "celtic"

t1 = "dundee-fc"
t2 = "dundee-united"

t1 = "aberdeen"
t2 = "falkirk-fc"
chain = result.chains_sequence[37]

mp = BayesianFootball.predict_match_ft_ht_chain(t1, t2, chain, mapping);
using Statistics



mean( 1 ./ mp.ft.home) 
mean( 1 ./ mp.ft.away) 
mean( 1 ./ mp.ft.draw )

mean( 1 ./ mp.ht.home) 
mean( 1 ./ mp.ht.away) 
mean( 1 ./ mp.ht.draw )

mean( 1 ./ mp.ft.under_05)
mean( 1 ./ ( 1 .- mp.ft.under_05))
mean( 1 ./ mp.ft.under_15)
mean( 1 ./ ( 1 .- mp.ft.under_15))
mean( 1 ./ mp.ft.under_25)
mean( 1 ./ ( 1 .- mp.ft.under_25))
mean( 1 ./ mp.ft.under_35)
mean( 1 ./ ( 1 .- mp.ft.under_35))

mean( 1 ./ mp.ft.btts)
mean( 1 ./ (1 .- mp.ft.btts))

k_cs = Dict( k => mean(1 ./ v) for (k,v) in mp.ft.correct_score)

## note de construct 
match_id = rand(target_matches.match_id)

mp = matches_prediction[match_id];
mo = matches_odds[match_id]
mr = matches_results[match_id]
ma = matches_activelines[match_id]


mean(mp.ft.home)
mean( 1 ./ mp.ft.home) 
mo.ft.home
k = calculate_bayesian_kelly(mp.ft.home, mo.ft.home);
mean(k)
mr.ft.home


mean(mp.ft.under_35)
mean( 1 ./ mp.ft.under_35) 
mo.ft.under_35
k = calculate_bayesian_kelly(mp.ft.under_35, mo.ft.under_35);
mean(k)
mr.ft.under_35


mean(1 .- mp.ft.under_35)
mean( 1 ./ (1 .- mp.ft.under_35))
mo.ft.over_35
k = calculate_bayesian_kelly(1 .- mp.ft.under_35, mo.ft.over_35);
mean(k)
mr.ft.over_35


#### added main 
config = Kelly.Config(0.02, 0.05)  # 2% commission, 5% probability value threshold
match_kelly = apply_kelly_to_match(mp, mo, config)
mean(match_kelly.ft.home)
mean(match_kelly.ft.draw)
mean(match_kelly.ft.away)


###
matches_kelly = process_matches_kelly(matches_prediction, matches_odds, config)
match_id = rand(target_matches.match_id)
mk = matches_kelly[match_id]
mp = matches_prediction[match_id];
mo = matches_odds[match_id]
mr = matches_results[match_id]
ma = matches_activelines[match_id]

k_cs = Dict( k => mean(v) for (k,v) in mk.ft.correct_score)
sort(collect(k_cs), by = x -> x[2], rev=true)
mr.ft.correct_score

mean(mk.ft.under_35)
mean(mk.ft.over_35)
mr.ft.under_35




##############################
# ROI 
##############################
config = ROI.Config(0.7, 0.01, 0.02) 


match_id = rand(target_matches.match_id)
mk = matches_kelly[match_id]
mp = matches_prediction[match_id];
mo = matches_odds[match_id]
mr = matches_results[match_id]
ma = matches_activelines[match_id]

match_roi = calculate_match_roi(
    mr,  # From your get_match_results function
    mo,     # Odds structure
    mk,    # Kelly chains from previous calculation
    config
)


###


all_roi = process_matches_roi(
    matches_results,
    matches_odds,
    matches_kelly,
    config
)

performance_stats = analyze_roi_performance(all_roi)

#= 
julia> performance_stats.ft_home
(mean = 0.2025069409992728, median = -1.0, std = 2.139775734778602, count = 205, total_roi = 41.513922904850
92)

=#


config_2 = ROI.Config(0.2, 0.01, 0.02) 

all_roi_2 = process_matches_roi(
    matches_results,
    matches_odds,
    matches_kelly,
    config_2
)

performance_stats2 = analyze_roi_performance(all_roi_2)


config_3 = ROI.Config(0.7, 0.01, 0.02) 

all_roi_3 = process_matches_roi(
    matches_results,
    matches_odds,
    matches_kelly,
    config_3
)
performance_stats3 = analyze_roi_performance(all_roi_3)

### 
results1::Vector = []
results_away = []
results_draw = []
c_values1::Vector = []
for c in 0.1:0.01:0.9 
  cfg = ROI.Config(c, 0.01, 0.02) 

  roi = process_matches_roi(
      matches_results,
      matches_odds,
      matches_kelly,
      cfg
  )
  performance = analyze_roi_performance(roi)
  push!(results1, performance.ft_home.total_roi)
  push!(results_away, performance.ft_away.total_roi)
  push!(results_draw, performance.ft_draw.total_roi)
  push!(c_values1, c)
end 

c_values
length(c_values)
length(results)
results1
p_home = plot(c_values1, results1)
p_away = plot(c_values1, results_away)
p_draw = plot(c_values1, results_draw)

plot(p_home, p_away, p_draw)



##############################
# wealth
##############################
wealth_df = calculate_cumulative_wealth(
    all_roi,
    target_matches,
    100.0  # Initial wealth
)

plots = plot_all_market_groups(
    wealth_df,
    initial_wealth = 100.0,
    config_label = "Q50 Strategy"
)


display(plots.main_1x2)      # 1X2 markets
display(plots.under_over_25)  # Under/Over 2.5 goals
display(plots.summary)


wealth_df
plot_wealth_evolution(wealth_df)

plot(wealth_df.ft_home_wealth)
plot(wealth_df.ft_over_15_wealth)

names(wealth_df)


# v7
diagnose_date_issues(wealth_df)
fix_wealth_df_dates!(wealth_df)
p = plot_wealth_simple(
    wealth_df,
    [:ft_home_wealth, :ft_draw_wealth, :ft_away_wealth],
    title = "FT 1X2 Markets",
    initial_wealth = 100.0
)
display(p)
# Get the date range from your data
min_date = minimum(wealth_df.match_date)
max_date = maximum(wealth_df.match_date)

# Use the same date range for multiple plots
p1 = plot_wealth_simple(
    wealth_df,
    [:ft_under_35_wealth, :ft_over_35_wealth],
    title = "FT 1X2 Markets",
    initial_wealth = 100.0,
    date_range = (min_date, max_date)
)

p1 = plot_wealth_simple(
    wealth_df,
    [:ft_correct_score_wealth, :ht_correct_score_wealth],
    title = "FT 1X2 Markets",
    initial_wealth = 100.0,
    date_range = (min_date, max_date)
)

p1 = plot_wealth_simple(
    wealth_df,
    [:ft_btts_yes_wealth, :ft_btts_no_wealth],
    title = "FT 1X2 Markets",
    initial_wealth = 100.0,
    date_range = (min_date, max_date)
)

p2 = plot_wealth_simplev(
    wealth_df,
    [:ht_home_wealth, :ht_draw_wealth, :ht_away_wealth],
    title = "FT 1X2 Markets",
    initial_wealth = 100.0,
    date_range = (min_date, max_date)
)

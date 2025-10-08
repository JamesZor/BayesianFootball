using BayesianFootball
using DataFrames
using Dates
using Statistics, StatsBase, StatsPlots, Distributions, Plots
using CSV 

# utils 

include("/home/james/bet_project/models_julia/workspace/basic_state_space/utils/scot_matchday_utils.jl")

# models
include("/home/james/bet_project/models_julia/workspace/basic_state_space/models/ar1_poisson_ha.jl")
include("/home/james/bet_project/models_julia/workspace/basic_state_space/models/ar1_negative_binomial_ha.jl")
using .AR1NegativeBinomialHA
using .AR1PoissonHA




all_model_paths = Dict(
  "ssm_poiss" => "/home/james/bet_project/models_julia/experiments/scotland_ar1/ar1_poisson_ha_20251004-111854",
  "ssm_neg_bin" => "/home/james/bet_project/models_julia/experiments/scotland_ar1/ar1_neg_bin_ha_20251004-122001",
)


loaded_models_all = load_models_from_paths(all_model_paths)


##


todays_matches = get_todays_matches(["scotland"]; cli_path=CLI_PATH)

odds_df = fetch_all_market_odds(
    todays_matches,
    MARKET_LIST;
    cli_path=CLI_PATH
)


### rough outline
todays_matches
match_to_analyze = todays_matches[5, :]
league_id_to_predict = "56"


m1 = loaded_models_all["ssm_neg_bin"]
m2 = loaded_models_all["ssm_poiss"]
mapping = m1.result.mapping;
chain = m1.result.chains_sequence[1];

posterior_samples = BayesianFootball.extract_posterior_samples(
    m1.config.model_def,
    chain.ft,
    mapping
);

last_training_round = posterior_samples.n_rounds
next_round = last_training_round + 1

match_to_predict = DataFrame(
    home_team=match_to_analyze.home_team,
    away_team=match_to_analyze.away_team,
    tournament_id=league_id_to_predict,
    global_round = next_round,
    home_score_ht=0, away_score_ht=0, home_score=0, away_score=0
)
features = BayesianFootball.create_master_features(match_to_predict, mapping);
predictions = BayesianFootball.predict(m1.config.model_def, chain, features, mapping);

print_1x2(predictions)
print_1x2_ht(predictions)

print_under(predictions)
print_under_ht(predictions)


print_over(predictions)
print_over_ht(predictions)

odds = filter(row -> row.event_name==match_to_analyze.event_name, odds_df);
odds[: , [:ft_1x2_home_back, :ft_1x2_away_back, :ft_1x2_draw_back, :ht_1x2_home_back, :ht_1x2_away_back, :ht_1x2_draw_back, ]]
odds[: , [:ft_ou_05_under_back, :ft_ou_15_under_back, :ft_ou_25_under_back, :ft_ou_35_under_back]]
odds[: , [:ht_ou_05_under_back, :ht_ou_15_under_back, :ht_ou_25_under_back]]



plot_attack_defence(match_to_analyze.home_team, match_to_analyze.away_team, m1, posterior_samples)

plot_1x2(predictions)

plot_xg(predictions)
plot_xg_t(predictions)

  println("under 05, mean: ", round(mean(  predictions.ft.under_05),digits=5), " | ", round(median(  predictions.ft.under_05), digits=5) )
  println("under 15, mean: ", round(mean(  predictions.ft.under_15),digits=5), " | ", round(median(  predictions.ft.under_15), digits=5) )
  println("under 25, mean: ", round(mean(  predictions.ft.under_25),digits=5), " | ", round(median(  predictions.ft.under_25), digits=5) )
  println("under 35, mean: ", round(mean(  predictions.ft.under_35),digits=5), " | ", round(median(  predictions.ft.under_35), digits=5) )



  println("over 15, mean: ", round(mean(1 .-  predictions.ft.under_15),digits=5), " | ", round(median( 1 .- predictions.ft.under_15), digits=5) )


####### method 

# 3. Main Analysis Loop
println("--- Starting Match Analysis Loop ---")
  # Example: Analyze the first match with available odds
  match_to_analyze_row = first(odds_df)
  match_to_analyze_row = odds_df[7,:]
  
  home_team = match_to_analyze_row.home_team
  away_team = match_to_analyze_row.away_team
  
  println("Analyzing Match: $home_team vs $away_team")

  # This league ID might need to be dynamically determined in a real scenario
  league_id_to_predict = 56

  
  posterior_samples = BayesianFootball.extract_posterior_samples(
      m1.config.model_def,
      chain.ft,
      mapping
  );

  last_training_round = posterior_samples.n_rounds
  next_round = last_training_round + 1

  match_to_predict = DataFrame(
      home_team=match_to_analyze_row.home_team,
      away_team=match_to_analyze_row.away_team,
      tournament_id=league_id_to_predict,
      global_round = next_round,
      home_score_ht=0, away_score_ht=0, home_score=0, away_score=0
  )
  features = BayesianFootball.create_master_features(match_to_predict, mapping);
  p_neg = BayesianFootball.predict(m1.config.model_def, chain, features, mapping);
  p_poss = BayesianFootball.predict(m2.config.model_def, chain, features, mapping);

  all_predictions = Dict("ssm_neg_bin" => p_neg,
                         "ssm_poiss" => p_poss)


  # --- Generate and Display Analysis DataFrames ---

  # A) Calculate and display the Expected Value DataFrame
  println("\n--- Expected Value (EV) Analysis ---")
  ev_results_df = calculate_ev_dataframe(all_predictions, match_to_analyze_row)
  println(ev_results_df)

  # B) Create and display the comparison DataFrame for a single model
  println("\n--- Distribution Comparison for ssm_neg_bin ---")
  comparison_df = create_comparison_dataframe(
      all_predictions["ssm_neg_bin"],
      match_to_analyze_row,
      "ssm_neg_bin"
  )

  println(comparison_df)


println("\n--- Analysis Complete ---")

plot_attack_defence(home_team, away_team, m1, posterior_samples)



#### loop 


display_markets = [
    # Full Time 1x2
    :ft_1x2_home, :ft_1x2_draw, :ft_1x2_away,
    
    # Full Time Over/Under
    :ft_ou_05_under, :ft_ou_05_over,
    :ft_ou_15_under, :ft_ou_15_over,
    :ft_ou_25_under, :ft_ou_25_over,
    :ft_ou_35_under, :ft_ou_35_over,
    
    # Half Time 1x2
    :ht_1x2_home, :ht_1x2_draw, :ht_1x2_away,
    
    # Half Time Over/Under
    :ht_ou_05_under, :ht_ou_05_over,
    :ht_ou_15_under, :ht_ou_15_over,
    :ht_ou_25_under, :ht_ou_25_over,
    
]




odds_df[:, [:event_name]]
"""
 Row │ event_name                
     │ String                    
─────┼───────────────────────────
   1 │ Kilmarnock v St Mirren
   2 │ Ross Co v Raith
   3 │ St Johnstone v Ayr
   4 │ Airdrieonians v Morton
   5 │ Hamilton v Inverness CT
   6 │ East Fife v Cove Rangers
   7 │ Stenhousemuir v Montrose
   8 │ Dunfermline v Queens Park
   9 │ Clyde v East Kilbride
  10 │ Kelty Hearts v Peterhead
  11 │ Elgin City FC v Forfar
  12 │ Partick v Arbroath
  13 │ Stranraer v Annan
  14 │ Spartans v Edinburgh City
  15 │ Dundee Utd v Livingston
  16 │ Stirling v Dumbarton
  17 │ Queen of South v Alloa
  18 │ Hearts v Hibernian
"""
league_1_idx = [6, 5, 10, 7, 17 ]
leauge_2_idx = [9, 11, 16, 13]
leauge_c_idx = [4, 8, 12, 2, 3 ]

league_id_to_predict = 57

for idx in leauge_2_idx
  match_to_analyze_row = odds_df[idx,:]
  
  home_team = match_to_analyze_row.home_team
  away_team = match_to_analyze_row.away_team
  println() 
  println("Analyzing Match: $home_team vs $away_team")

  # This league ID might need to be dynamically determined in a real scenario

  
  posterior_samples = BayesianFootball.extract_posterior_samples(
      m1.config.model_def,
      chain.ft,
      mapping
  );

  last_training_round = posterior_samples.n_rounds
  next_round = last_training_round + 1

  match_to_predict = DataFrame(
      home_team=match_to_analyze_row.home_team,
      away_team=match_to_analyze_row.away_team,
      tournament_id=league_id_to_predict,
      global_round = next_round,
      home_score_ht=0, away_score_ht=0, home_score=0, away_score=0
  )
  features = BayesianFootball.create_master_features(match_to_predict, mapping);
  p_neg = BayesianFootball.predict(m1.config.model_def, chain, features, mapping);
  p_poss = BayesianFootball.predict(m2.config.model_def, chain, features, mapping);

  all_predictions = Dict("ssm_neg_bin" => p_neg,
                         "ssm_poiss" => p_poss)


  # --- Generate and Display Analysis DataFrames ---

  # A) Calculate and display the Expected Value DataFrame
  println("\n--- Expected Value (EV) Analysis ---")
  ev_results_df = calculate_ev_dataframe(all_predictions, match_to_analyze_row)
  show(ev_results_df)

  # B) Create and display the comparison DataFrame for a single model
  println("\n--- Distribution Comparison for ssm_neg_bin ---")
  comparison_df = create_comparison_dataframe(
      all_predictions["ssm_neg_bin"],
      match_to_analyze_row,
      "ssm_neg_bin"
  )

  show(comparison_df)



end 



######
# leauge 2

"""


# leauge 2

Analyzing Match: clyde-fc vs east-kilbride
result: ht: 0-1, ft: 2-2 

--- Expected Value (EV) Analysis ---
51×7 DataFrame
 Row │ market            market_back  market_lay  ssm_poiss_mean_ev_pct  ssm_poiss_std_ev_pct  ssm_neg_bin_mean_ev_pct  ssm_neg_bin_std_ev_pct 
     │ String            Float64      Float64     Float64?               Float64?              Float64?                 Float64?               
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ ft_1x2_home              2.7         3.2               3.81781                31.9506                 3.96897                  31.1459
   2 │ ft_1x2_draw              3.9         4.6             -14.9876                 10.3888               -15.3969                   10.1779
   3 │ ft_1x2_away              2.26        2.54            -10.1699                 27.0588               -10.0645                   26.302
   4 │ ft_ou_05_under           1.1       110.0             -95.9543                  2.40253              -95.7712                    2.41446
   5 │ ft_ou_05_over            1.03        1.05             -0.788235                2.24964               -0.959685                  2.26081
   6 │ ft_ou_15_under           5.8         6.6             -11.617                  39.3195                -9.37857                  38.7504
   7 │ ft_ou_15_over            1.18        1.21              0.0186363               7.99949               -0.436774                  7.88369
   8 │ ft_ou_25_under           2.72        3.4              -7.63411                29.947                 -6.52619                  29.2198
   9 │ ft_ou_25_over            1.41        1.58             -6.88085                15.524                 -7.45517                  15.147
  10 │ ft_ou_35_under           1.67        1.72             -8.55487                20.775                 -8.20873                  20.2254
  11 │ ft_ou_35_over            2.22        2.5               0.438213               27.6171                -0.0219233                26.8865
  12 │ ft_btts_yes              1.51        1.6              -1.3411                 13.2092                -2.37509                  12.9958
  13 │ ft_btts_no               2.66        3.1              -7.79647                23.2692                -5.975                    22.8933
  14 │ ft_cs_0_0                5.8       980.0             -78.6682                 12.6679               -77.7027                   12.7308
  15 │ ft_cs_1_0               13.0       980.0             -25.5517                 32.8784               -24.2469                   31.8313
  16 │ ft_cs_0_1                5.3        21.0             -69.0811                 13.4715               -68.4498                   13.2767
  17 │ ft_cs_1_1                4.6        12.0             -58.1813                  9.91346              -58.4245                    9.45301
  18 │ ft_cs_2_0                5.6       980.0             -73.5363                 10.889                -73.3756                   10.4492
  19 │ ft_cs_0_2                5.5        25.0             -73.034                  11.0617               -72.7834                   10.8623
  20 │ ft_cs_2_1                9.6       980.0             -27.9254                 14.111                -29.1397                   13.4521
  21 │ ft_cs_1_2                5.3       980.0             -59.5011                  7.62792              -60.1691                    7.30923
  22 │ ft_cs_2_2               10.5       980.0             -33.7364                  8.76577              -35.5517                    8.43924
  23 │ ft_cs_3_0                6.2       980.0             -82.9553                  8.6078               -82.8086                    8.36306
  24 │ ft_cs_0_3                5.9       980.0             -82.8823                  8.5483               -82.7124                    8.34778
  25 │ ft_cs_3_1                5.7       980.0             -75.0867                  8.54792              -75.4513                    8.20882
  26 │ ft_cs_1_3                5.5       980.0             -75.1294                  8.20666              -75.5301                    7.74761
  27 │ ft_cs_3_2                5.8       980.0             -78.6918                  6.57996              -79.2288                    6.29977
  28 │ ft_cs_2_3                5.7       980.0             -78.7189                  6.31751              -79.2935                    5.916
  29 │ ft_cs_3_3                6.2        55.0             -86.5321                  5.52285              -86.8619                    5.26207
  30 │ ft_cs_other_home         5.0       980.0             -51.5145                 31.7324               -50.5831                   31.3082
  31 │ ft_cs_other_draw         4.1       380.0             -97.7612                  1.80093              -97.7302                    1.79214
  32 │ ft_cs_other_away         8.6        14.0             -12.1672                 57.4662               -10.8526                   56.2718
  33 │ ht_1x2_home              3.1         3.6               3.81743                33.6254                 4.4393                   32.3635
  34 │ ht_1x2_draw              2.54        2.88            -30.6656                  8.92307              -31.0752                    9.15092
  35 │ ht_1x2_away              2.86        3.25             12.1503                 31.3515                12.0349                   30.0777
  36 │ ht_ou_05_under           4.1         4.8             -58.7313                 18.2012               -55.124                    18.6749
  37 │ ht_ou_05_over            1.26        1.33             13.3174                  5.59355               12.2088                    5.73913
  38 │ ht_ou_15_under           1.69        1.85            -45.5027                 16.4886               -43.647                    16.1969
  39 │ ht_ou_15_over            2.18        2.44             47.7017                 21.2693                45.3079                   20.8931
  40 │ ht_ou_25_under           1.21        1.25            -30.3275                 13.6225               -29.9306                   13.2159
  41 │ ht_ou_25_over            5.0         6.0             112.097                  56.2911               110.457                    54.6111
  42 │ ht_cs_0_0                3.6         4.9             -63.764                  15.9816               -60.5967                   16.3975
  43 │ ht_cs_1_0                4.9         6.8             -48.8785                 15.3771               -48.1522                   14.6432
  44 │ ht_cs_0_1                4.6         6.4             -45.9572                 17.4415               -45.6352                   16.4579
  45 │ ht_cs_1_1                6.2       950.0             -24.5883                  6.77446              -29.0264                    6.40694
  46 │ ht_cs_2_0                6.2      1000.0             -63.4758                 14.5111               -63.3678                   13.5254
  47 │ ht_cs_2_1                6.8      1000.0             -53.3856                 12.0897               -56.557                    10.8227
  48 │ ht_cs_2_2                5.8      1000.0             -75.4459                  7.5674               -77.4334                    6.76433
  49 │ ht_cs_0_2                6.0      1000.0             -56.3231                 16.2093               -56.7532                   15.1574
  50 │ ht_cs_1_2                6.8      1000.0             -48.8185                 10.0026               -52.5319                    9.07797
  51 │ ht_cs_other             10.5      1000.0             149.943                  88.8465               160.717                    88.5171
--- Distribution Comparison for ssm_neg_bin ---
51×6 DataFrame
 Row │ market            market_back  market_lay  ssm_neg_bin_mean_odds  ssm_neg_bin_median_odds  ssm_neg_bin_market_quantile 
     │ String            Float64      Float64     Float64?               Float64?                 Float64?                    
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ ft_1x2_home              2.7         3.2                    2.88                     2.59                        0.45
   2 │ ft_1x2_draw              3.9         4.6                    4.68                     4.63                        0.928
   3 │ ft_1x2_away              2.26        2.54                   2.77                     2.59                        0.674
   4 │ ft_ou_05_under           1.1       110.0                   36.73                    28.57                        1.0
   5 │ ft_ou_05_over            1.03        1.05                   1.04                     1.04                        0.61
   6 │ ft_ou_15_under           5.8         6.6                    7.88                     6.68                        0.654
   7 │ ft_ou_15_over            1.18        1.21                   1.19                     1.18                        0.492
   8 │ ft_ou_25_under           2.72        3.4                    3.27                     2.93                        0.604
   9 │ ft_ou_25_over            1.41        1.58                   1.57                     1.52                        0.69
  10 │ ft_ou_35_under           1.67        1.72                   1.93                     1.79                        0.652
  11 │ ft_ou_35_over            2.22        2.5                    2.42                     2.26                        0.528
  12 │ ft_btts_yes              1.51        1.6                    1.58                     1.54                        0.566
  13 │ ft_btts_no               2.66        3.1                    3.02                     2.84                        0.616
  14 │ ft_cs_0_0                5.8       980.0                   36.73                    28.57                        1.0
  15 │ ft_cs_1_0               13.0       980.0                   21.02                    17.62                        0.81
  16 │ ft_cs_0_1                5.3        21.0                   20.59                    17.83                        1.0
  17 │ ft_cs_1_1                4.6        12.0                   11.81                    10.88                        1.0
  18 │ ft_cs_2_0                5.6       980.0                   25.17                    21.93                        1.0
  19 │ ft_cs_0_2                5.5        25.0                   24.12                    20.96                        1.0
  20 │ ft_cs_2_1                9.6       980.0                   14.17                    13.19                        1.0
  21 │ ft_cs_1_2                5.3       980.0                   13.86                    13.14                        1.0
  22 │ ft_cs_2_2               10.5       980.0                   16.65                    15.81                        1.0
  23 │ ft_cs_3_0                6.2       980.0                   47.37                    38.48                        1.0
  24 │ ft_cs_0_3                5.9       980.0                   44.37                    38.31                        1.0
  25 │ ft_cs_3_1                5.7       980.0                   26.72                    23.39                        1.0
  26 │ ft_cs_1_3                5.5       980.0                   25.53                    22.87                        1.0
  27 │ ft_cs_3_2                5.8       980.0                   31.44                    28.09                        1.0
  28 │ ft_cs_2_3                5.7       980.0                   30.72                    27.36                        1.0
  29 │ ft_cs_3_3                6.2        55.0                   58.1                     49.11                        1.0
  30 │ ft_cs_other_home         5.0       980.0                   16.08                    11.94                        0.926
  31 │ ft_cs_other_draw         4.1       380.0                  363.72                   238.97                        1.0
  32 │ ft_cs_other_away         8.6        14.0                   14.97                    11.17                        0.668
  33 │ ht_1x2_home              3.1         3.6                    3.31                     3.02                        0.466
  34 │ ht_1x2_draw              2.54        2.88                   3.75                     3.73                        0.998
  35 │ ht_1x2_away              2.86        3.25                   2.77                     2.57                        0.356
  36 │ ht_ou_05_under           4.1         4.8                   10.93                     9.67                        0.992
  37 │ ht_ou_05_over            1.26        1.33                   1.13                     1.12                        0.03
  38 │ ht_ou_15_under           1.69        1.85                   3.28                     3.06                        0.998
  39 │ ht_ou_15_over            2.18        2.44                   1.54                     1.49                        0.02
  40 │ ht_ou_25_under           1.21        1.25                   1.8                      1.72                        0.996
  41 │ ht_ou_25_over            5.0         6.0                    2.57                     2.39                        0.018
  42 │ ht_cs_0_0                3.6         4.9                   10.93                     9.67                        0.998
  43 │ ht_cs_1_0                4.9         6.8                   10.3                      9.69                        1.0
  44 │ ht_cs_0_1                4.6         6.4                    9.34                     8.69                        0.992
  45 │ ht_cs_1_1                6.2       950.0                    8.81                     8.61                        1.0
  46 │ ht_cs_2_0                6.2      1000.0                   19.8                     17.67                        1.0
  47 │ ht_cs_2_1                6.8      1000.0                   16.93                    15.28                        1.0
  48 │ ht_cs_2_2                5.8      1000.0                   28.71                    26.07                        1.0
  49 │ ht_cs_0_2                6.0      1000.0                   15.84                    14.34                        1.0
  50 │ ht_cs_1_2                6.8      1000.0                   14.95                    14.12                        1.0
  51 │ ht_cs_other             10.5      1000.0                    4.6                      4.14                        0.01

Analyzing Match: elgin-city vs forfar-athletic
result: ht: 0-1 ft: 2-1 

--- Expected Value (EV) Analysis ---
51×7 DataFrame
 Row │ market            market_back  market_lay  ssm_poiss_mean_ev_pct  ssm_poiss_std_ev_pct  ssm_neg_bin_mean_ev_pct  ssm_neg_bin_std_ev_pct 
     │ String            Float64      Float64     Float64?               Float64?              Float64?                 Float64?               
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ ft_1x2_home              2.34        2.64              -0.684624             18.6739                  -0.592379               18.8067
   2 │ ft_1x2_draw              3.0         3.4               -9.05989               9.04468                 -9.31382                 9.06845
   3 │ ft_1x2_away              3.3         3.8              -10.0944               21.4604                  -9.94523                21.5239
   4 │ ft_ou_05_under           8.4        11.0               18.1911               31.8828                  19.5521                 31.5036
   5 │ ft_ou_05_over            1.1         1.14              -5.4774                4.17512                 -5.65563                 4.12547
   6 │ ft_ou_15_under           3.05        3.45              25.5227               22.5242                  25.8067                 22.1012
   7 │ ft_ou_15_over            1.41        1.49             -17.0285               10.4128                 -17.1598                 10.2173
   8 │ ft_ou_25_under           1.64        1.72              11.0531               12.1265                  10.8468                 11.9189
   9 │ ft_ou_25_over            2.38        2.56             -23.1625               17.5982                 -22.8631                 17.2969
  10 │ ft_ou_35_under           1.26        1.31               7.60983               6.40698                  7.31497                 6.36214
  11 │ ft_ou_35_over            4.2         4.9              -38.6994               21.3566                 -37.7166                 21.2071
  12 │ ft_btts_yes              1.9         2.08             -27.3611               11.7021                 -27.9217                 11.3717
  13 │ ft_btts_no               1.94        2.1               19.8319               11.9485                  20.4043                 11.6111
  14 │ ft_cs_0_0                7.8        12.0                9.74884              29.6054                  11.0127                 29.2533
  15 │ ft_cs_1_0                7.2         9.6               11.4216               18.7031                  10.988                  18.1522
  16 │ ft_cs_0_1               10.0        14.0               16.0941               26.2063                  16.0072                 25.962
  17 │ ft_cs_1_1                6.2         8.0              -20.9572                4.65856                -22.2589                  4.78225
  18 │ ft_cs_2_0               11.0        15.5               -2.76891              22.7216                  -3.02668                22.477
  19 │ ft_cs_0_2               19.0        30.0               -5.52602              30.3976                  -5.2512                 30.2616
  20 │ ft_cs_2_1                9.6        13.0              -30.1625               11.7924                 -31.2902                 11.4357
  21 │ ft_cs_1_2               12.0        17.0              -34.5896               14.4392                 -35.4879                 14.0572
  22 │ ft_cs_2_2               17.0        25.0              -47.1846               14.3017                 -47.947                  13.6173
  23 │ ft_cs_3_0               23.0        44.0              -19.6211               31.8849                 -18.4442                 32.1856
  24 │ ft_cs_0_3                6.6       110.0              -90.3049                4.55015                -90.0875                  4.61049
  25 │ ft_cs_3_1               21.0        34.0              -39.641                21.613                  -39.6624                 21.281
  26 │ ft_cs_1_3                6.8        60.0              -89.0696                4.29433                -89.0237                  4.24413
  27 │ ft_cs_3_2               30.0        65.0              -63.2054               15.4563                 -63.196                  14.9657
  28 │ ft_cs_2_3                6.6        95.0              -93.9608                2.55972                -93.9468                  2.47819
  29 │ ft_cs_3_3                6.8       390.0              -97.5458                1.32508                -97.506                   1.29196
  30 │ ft_cs_other_home        16.5      1000.0              -46.2452               35.7726                 -43.1089                 37.6701
  31 │ ft_cs_other_draw         6.0         0.0              -99.8411                0.138986               -99.8275                  0.144322
  32 │ ft_cs_other_away         6.0        65.0              -92.6791                4.43049                -92.2443                  4.67642
  33 │ ht_1x2_home              2.96        3.4              -16.1486               20.0764                 -16.3678                 19.9027
  34 │ ht_1x2_draw              2.06        2.3               -8.02473               9.43081                 -7.72384                 9.4613
  35 │ ht_1x2_away              4.1         4.9               10.797                24.5765                  10.5016                 24.2896
  36 │ ht_ou_05_under           2.66        3.05              -8.52283              16.9845                  -6.85971                16.7521
  37 │ ht_ou_05_over            1.5         1.6               -1.58487               9.57771                 -2.52272                 9.44667
  38 │ ht_ou_15_under           1.36        1.44              -4.11877               9.38104                 -4.11318                 9.24912
  39 │ ht_ou_15_over            3.25        3.8               -4.12795              22.4179                  -4.1413                 22.1027
  40 │ ht_ou_25_under           1.09        1.11              -1.85499               4.23592                 -2.25707                 4.32523
  41 │ ht_ou_25_over           10.0        12.5               -0.412934             38.8617                   3.27586                39.681
  42 │ ht_cs_0_0                2.5         3.05             -14.0252               15.9629                 -12.4621                 15.7444
  43 │ ht_cs_1_0                3.95        5.3              -27.5934               11.9083                 -28.8529                 11.5107
  44 │ ht_cs_0_1                5.2         7.0               -7.54287              15.8972                  -9.11467                15.2761
  45 │ ht_cs_1_1                7.8      1000.0              -25.9606               11.7867                 -29.7625                 11.1652
  46 │ ht_cs_2_0                7.0      1000.0              -63.4834               14.1402                 -63.1014                 13.8309
  47 │ ht_cs_2_1                8.8      1000.0              -76.2185                9.03085                -76.8138                  8.55608
  48 │ ht_cs_2_2                2.34     1000.0              -98.2757                0.877015               -98.2707                  0.853444
  49 │ ht_cs_0_2                8.4      1000.0              -59.4154               14.3331                 -58.9125                 14.0842
  50 │ ht_cs_1_2                3.4      1000.0              -91.2101                3.13195                -91.4218                  2.96984
  51 │ ht_cs_other              8.6      1000.0              -66.1665               15.8567                 -61.8953                 17.6737
--- Distribution Comparison for ssm_neg_bin ---
51×6 DataFrame
 Row │ market            market_back  market_lay  ssm_neg_bin_mean_odds  ssm_neg_bin_median_odds  ssm_neg_bin_market_quantile 
     │ String            Float64      Float64     Float64?               Float64?                 Float64?                    
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ ft_1x2_home              2.34        2.64                   2.44                     2.37                        0.518
   2 │ ft_1x2_draw              3.0         3.4                    3.34                     3.31                        0.86
   3 │ ft_1x2_away              3.3         3.8                    3.9                      3.73                        0.682
   4 │ ft_ou_05_under           8.4        11.0                    7.57                     7.21                        0.294
   5 │ ft_ou_05_over            1.1         1.14                   1.17                     1.16                        0.934
   6 │ ft_ou_15_under           3.05        3.45                   2.51                     2.44                        0.112
   7 │ ft_ou_15_over            1.41        1.49                   1.73                     1.69                        0.96
   8 │ ft_ou_25_under           1.64        1.72                   1.5                      1.48                        0.168
   9 │ ft_ou_25_over            2.38        2.56                   3.25                     3.1                         0.908
  10 │ ft_ou_35_under           1.26        1.31                   1.18                     1.17                        0.116
  11 │ ft_ou_35_over            4.2         4.9                    7.59                     7.0                         0.954
  12 │ ft_btts_yes              1.9         2.08                   2.71                     2.64                        0.992
  13 │ ft_btts_no               1.94        2.1                    1.63                     1.61                        0.044
  14 │ ft_cs_0_0                7.8        12.0                    7.57                     7.21                        0.382
  15 │ ft_cs_1_0                7.2         9.6                    6.66                     6.61                        0.274
  16 │ ft_cs_0_1               10.0        14.0                    9.12                     8.68                        0.26
  17 │ ft_cs_1_1                6.2         8.0                    8.01                     7.86                        1.0
  18 │ ft_cs_2_0               11.0        15.5                   12.03                    11.39                        0.57
  19 │ ft_cs_0_2               19.0        30.0                   22.44                    20.68                        0.604
  20 │ ft_cs_2_1                9.6        13.0                   14.42                    13.84                        1.0
  21 │ ft_cs_1_2               12.0        17.0                   19.67                    18.5                         1.0
  22 │ ft_cs_2_2               17.0        25.0                   35.31                    33.19                        1.0
  23 │ ft_cs_3_0               23.0        44.0                   33.43                    29.64                        0.756
  24 │ ft_cs_0_3                6.6       110.0                   84.99                    72.28                        1.0
  25 │ ft_cs_3_1               21.0        34.0                   39.91                    36.38                        0.956
  26 │ ft_cs_1_3                6.8        60.0                   74.36                    63.77                        1.0
  27 │ ft_cs_3_2               30.0        65.0                   97.42                    85.23                        0.998
  28 │ ft_cs_2_3                6.6        95.0                  133.19                   112.06                        1.0
  29 │ ft_cs_3_3                6.8       390.0                  366.45                   302.97                        1.0
  30 │ ft_cs_other_home        16.5      1000.0                   43.04                    34.55                        0.894
  31 │ ft_cs_other_draw         6.0         0.0                 6882.19                  4640.69                        1.0
  32 │ ft_cs_other_away         6.0        65.0                  119.54                    89.03                        1.0
  33 │ ht_1x2_home              2.96        3.4                    3.75                     3.58                        0.81
  34 │ ht_1x2_draw              2.06        2.3                    2.26                     2.24                        0.804
  35 │ ht_1x2_away              4.1         4.9                    3.9                      3.78                        0.346
  36 │ ht_ou_05_under           2.66        3.05                   2.96                     2.87                        0.646
  37 │ ht_ou_05_over            1.5         1.6                    1.55                     1.54                        0.608
  38 │ ht_ou_15_under           1.36        1.44                   1.43                     1.41                        0.64
  39 │ ht_ou_15_over            3.25        3.8                    3.58                     3.45                        0.612
  40 │ ht_ou_25_under           1.09        1.11                   1.12                     1.11                        0.652
  41 │ ht_ou_25_over           10.0        12.5                   11.25                    10.24                        0.536
  42 │ ht_cs_0_0                2.5         3.05                   2.96                     2.87                        0.782
  43 │ ht_cs_1_0                3.95        5.3                    5.71                     5.57                        0.996
  44 │ ht_cs_0_1                5.2         7.0                    5.9                      5.71                        0.71
  45 │ ht_cs_1_1                7.8      1000.0                   11.41                    11.07                        1.0
  46 │ ht_cs_2_0                7.0      1000.0                   22.06                    19.79                        1.0
  47 │ ht_cs_2_1                8.8      1000.0                   44.16                    39.78                        1.0
  48 │ ht_cs_2_2                2.34     1000.0                  173.72                   147.35                        1.0
  49 │ ht_cs_0_2                8.4      1000.0                   23.22                    21.43                        0.998
  50 │ ht_cs_1_2                3.4      1000.0                   44.85                    42.12                        1.0
  51 │ ht_cs_other              8.6      1000.0                   27.74                    24.68                        0.99


Analyzing Match: stirling-albion vs dumbarton
result: ht: 1-1, ft: 2-2

--- Expected Value (EV) Analysis ---
50×7 DataFrame
 Row │ market            market_back  market_lay  ssm_poiss_mean_ev_pct  ssm_poiss_std_ev_pct  ssm_neg_bin_mean_ev_pct  ssm_neg_bin_std_ev_pct 
     │ String            Float64?     Float64?    Float64?               Float64?              Float64?                 Float64?               
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ ft_1x2_home              3.3         3.8                35.9506               32.9578                  35.5697                32.6536
   2 │ ft_1x2_draw              3.8         4.4               -16.1865                7.60984                -16.6762                 7.49031
   3 │ ft_1x2_away              2.04        2.24              -25.0399               20.074                  -24.5444                19.9395
   4 │ ft_ou_05_under          15.0       980.0               -46.5069               23.6705                 -43.8431                24.029
   5 │ ft_ou_05_over            1.01        1.06               -2.60187               1.59381                 -2.78123                1.61795
   6 │ ft_ou_15_under           4.9         5.8               -25.9091               25.0623                 -23.8547                24.7706
   7 │ ft_ou_15_over            1.21        1.25                2.70407               6.18884                  2.19678                6.11681
   8 │ ft_ou_25_under           2.34        2.6               -20.0734               19.8441                 -19.0258                19.292
   9 │ ft_ou_25_over            1.63        1.75                7.32463              13.823                    6.59491               13.4384
  10 │ ft_ou_35_under           1.55        1.65              -14.1013               14.8652                 -13.7358                14.3468
  11 │ ft_ou_35_over            2.54        2.82               13.2369               24.3597                  12.638                 23.5102
  12 │ ft_btts_yes              1.61        1.7                 5.59924              10.577                    4.36177               10.4095
  13 │ ft_btts_no               2.42        2.64              -16.7268               15.8984                 -14.8668                15.6466
  14 │ ft_cs_0_0               18.5        23.0               -34.0251               29.1936                 -30.7398                29.6358
  15 │ ft_cs_1_0               15.0        19.0               -10.3676               30.8517                  -8.76582               30.1895
  16 │ ft_cs_0_1               11.0        13.5               -38.6321               20.2551                 -37.1484                19.9726
  17 │ ft_cs_1_1                8.2        10.0               -23.5318               14.0445                 -24.0301                13.2068
  18 │ ft_cs_2_0               20.0        32.0                 3.43894              33.3671                   3.81454               32.6952
  19 │ ft_cs_0_2               14.0        16.5               -36.4577               20.8915                 -35.4719                20.6725
  20 │ ft_cs_2_1               13.0        15.5                 4.71834              14.9858                   2.5517                14.5339
  21 │ ft_cs_1_2                9.8        11.5               -25.8826               12.0703                 -26.9961                11.6629
  22 │ ft_cs_2_2               13.5        16.0               -12.058                 8.20459                -14.6602                 8.33691
  23 │ ft_cs_3_0               38.0       110.0                17.2159               46.0811                  17.7305                45.5324
  24 │ ft_cs_0_3               22.0        30.0               -43.7026               24.3108                 -42.4959                24.2315
  25 │ ft_cs_3_1               25.0        42.0                19.9119               31.6462                  17.4192                30.5683
  26 │ ft_cs_1_3               17.5        25.0               -25.6519               22.5007                 -26.3814                21.7876
  27 │ ft_cs_3_2               30.0        46.0                16.0632               26.9263                  12.5258                25.6242
  28 │ ft_cs_2_3               23.0        38.0               -16.1364               21.3265                 -18.2429                20.3047
  29 │ ft_cs_3_3               18.5       120.0               -60.0767               12.8673                 -61.1432                12.1098
  30 │ ft_cs_other_home        17.0        38.0                74.1578               89.775                   76.3634                88.8453
  31 │ ft_cs_other_draw        10.5       560.0               -94.8304                3.23142                -94.7807                 3.12094
  32 │ ft_cs_other_away         9.6        11.5               -17.3361               47.0584                 -15.1313                47.2165
  33 │ ht_1x2_home              1.04      980.0               -69.3597                7.6871                 -69.1582                 7.53022
  34 │ ht_1x2_draw              1.02      980.0               -65.193                 3.68016                -65.1553                 3.69579
  35 │ ht_1x2_away              1.04      980.0               -62.1298                7.63262                -62.3698                 7.3896
  36 │ ht_ou_05_under           3.7         4.3               -28.9953               18.9648                 -25.741                 18.681
  37 │ ht_ou_05_over            1.3         1.36                5.05241               6.6633                   3.90899                6.56359
  38 │ ht_ou_15_under           1.04     1000.0               -47.8101                8.80257                -47.2013                 8.41851
  39 │ ht_ou_15_over            1.04     1000.0               -48.1899                8.80257                -48.7987                 8.41851
  40 │ ht_ou_25_under           1.16        1.2               -11.9941                8.40923                -12.2233                 8.09577
  41 │ ht_ou_25_over            1.1      1000.0               -73.4539                7.97427                -73.2365                 7.67702
  42 │ ht_cs_0_0                2.0         4.9               -61.6191               10.2512                 -59.86                  10.0978
  43 │ ht_cs_1_0                1.26     1000.0               -82.042                 3.24637                -82.0868                 3.06637
  44 │ ht_cs_0_1                1.26     1000.0               -78.908                 4.02103                -79.2337                 3.74837
  45 │ ht_cs_1_1                1.26     1000.0               -84.2856                1.13203                -85.2355                 1.20388
  46 │ ht_cs_2_0                1.26     1000.0               -92.9055                2.39683                -92.8276                 2.29728
  47 │ ht_cs_2_1                1.26     1000.0               -93.774                 1.85651                -94.0792                 1.68037
  48 │ ht_cs_2_2                1.26     1000.0               -97.1641                1.06708                -97.3003                 0.956528
  49 │ ht_cs_0_2                1.26     1000.0               -90.4463                2.65248                -90.5609                 2.48024
  50 │ ht_cs_1_2                1.26     1000.0               -92.8615                1.58827                -93.2779                 1.43674
--- Distribution Comparison for ssm_neg_bin ---
50×6 DataFrame
 Row │ market            market_back  market_lay  ssm_neg_bin_mean_odds  ssm_neg_bin_median_odds  ssm_neg_bin_market_quantile 
     │ String            Float64?     Float64?    Float64?               Float64?                 Float64?                    
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ ft_1x2_home              3.3         3.8                    2.6                      2.46                        0.15
   2 │ ft_1x2_draw              3.8         4.4                    4.6                      4.57                        0.984
   3 │ ft_1x2_away              2.04        2.24                   2.91                     2.73                        0.88
   4 │ ft_ou_05_under          15.0       980.0                   32.32                    29.17                        0.942
   5 │ ft_ou_05_over            1.01        1.06                   1.04                     1.04                        0.996
   6 │ ft_ou_15_under           4.9         5.8                    7.21                     6.72                        0.838
   7 │ ft_ou_15_over            1.21        1.25                   1.19                     1.17                        0.324
   8 │ ft_ou_25_under           2.34        2.6                    3.07                     2.94                        0.838
   9 │ ft_ou_25_over            1.63        1.75                   1.56                     1.52                        0.308
  10 │ ft_ou_35_under           1.55        1.65                   1.85                     1.79                        0.822
  11 │ ft_ou_35_over            2.54        2.82                   2.36                     2.26                        0.306
  12 │ ft_btts_yes              1.61        1.7                    1.56                     1.54                        0.33
  13 │ ft_btts_no               2.42        2.64                   2.95                     2.87                        0.822
  14 │ ft_cs_0_0               18.5        23.0                   32.32                    29.17                        0.852
  15 │ ft_cs_1_0               15.0        19.0                   18.5                     16.94                        0.646
  16 │ ft_cs_0_1               11.0        13.5                   19.59                    17.87                        0.958
  17 │ ft_cs_1_1                8.2        10.0                   11.17                    10.68                        0.978
  18 │ ft_cs_2_0               20.0        32.0                   21.59                    19.92                        0.494
  19 │ ft_cs_0_2               14.0        16.5                   24.3                     22.08                        0.942
  20 │ ft_cs_2_1               13.0        15.5                   12.98                    12.5                         0.366
  21 │ ft_cs_1_2                9.8        11.5                   13.81                    13.25                        1.0
  22 │ ft_cs_2_2               13.5        16.0                   15.99                    15.65                        1.0
  23 │ ft_cs_3_0               38.0       110.0                   38.53                    33.59                        0.392
  24 │ ft_cs_0_3               22.0        30.0                   46.23                    40.83                        0.938
  25 │ ft_cs_3_1               25.0        42.0                   23.04                    21.33                        0.306
  26 │ ft_cs_1_3               17.5        25.0                   26.21                    24.05                        0.87
  27 │ ft_cs_3_2               30.0        46.0                   28.24                    26.55                        0.324
  28 │ ft_cs_2_3               23.0        38.0                   30.24                    28.14                        0.806
  29 │ ft_cs_3_3               18.5       120.0                   53.21                    48.46                        1.0
  30 │ ft_cs_other_home        17.0        38.0                   12.45                    10.68                        0.22
  31 │ ft_cs_other_draw        10.5       560.0                  292.4                    228.64                        1.0
  32 │ ft_cs_other_away         9.6        11.5                   15.37                    12.6                         0.712
  33 │ ht_1x2_home              1.04      980.0                    3.58                     3.41                        1.0
  34 │ ht_1x2_draw              1.02      980.0                    2.96                     2.92                        1.0
  35 │ ht_1x2_away              1.04      980.0                    2.88                     2.79                        1.0
  36 │ ht_ou_05_under           3.7         4.3                    5.34                     5.03                        0.912
  37 │ ht_ou_05_over            1.3         1.36                   1.26                     1.25                        0.26
  38 │ ht_ou_15_under           1.04     1000.0                    2.02                     1.95                        1.0
  39 │ ht_ou_15_over            1.04     1000.0                    2.09                     2.05                        1.0
  40 │ ht_ou_25_under           1.16        1.2                    1.33                     1.31                        0.954
  41 │ ht_ou_25_over            1.1      1000.0                    4.47                     4.25                        1.0
  42 │ ht_cs_0_0                2.0         4.9                    5.34                     5.03                        1.0
  43 │ ht_cs_1_0                1.26     1000.0                    7.26                     7.05                        1.0
  44 │ ht_cs_0_1                1.26     1000.0                    6.29                     6.1                         1.0
  45 │ ht_cs_1_1                1.26     1000.0                    8.6                      8.44                        1.0
  46 │ ht_cs_2_0                1.26     1000.0                   19.61                    18.01                        1.0
  47 │ ht_cs_2_1                1.26     1000.0                   23.27                    21.61                        1.0
  48 │ ht_cs_2_2                1.26     1000.0                   53.39                    49.17                        1.0
  49 │ ht_cs_0_2                1.26     1000.0                   14.4                     13.54                        1.0
  50 │ ht_cs_1_2                1.26     1000.0                   19.69                    19.07                        1.0

Analyzing Match: stranraer vs annan-athletic

results: ht: 0-0, ft: 0-1

--- Expected Value (EV) Analysis ---
50×7 DataFrame
 Row │ market            market_back  market_lay  ssm_poiss_mean_ev_pct  ssm_poiss_std_ev_pct  ssm_neg_bin_mean_ev_pct  ssm_neg_bin_std_ev_pct 
     │ String            Float64?     Float64?    Float64?               Float64?              Float64?                 Float64?               
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ ft_1x2_home              2.68        3.05               2.31294              24.1506                   2.71761                24.174
   2 │ ft_1x2_draw              3.3         3.8              -14.8234                7.41024                -15.2603                  7.53597
   3 │ ft_1x2_away              2.56        2.92              -7.80852              21.5711                  -7.85667                21.4164
   4 │ ft_ou_05_under          11.0        14.5              -17.3218               27.6107                 -15.492                  27.7429
   5 │ ft_ou_05_over            1.08        1.1               -0.1175                2.71087                 -0.297149                2.72385
   6 │ ft_ou_15_under           3.5         4.1               -7.05011              22.3989                  -6.28867                22.243
   7 │ ft_ou_15_over            1.33        1.37              -2.32096               8.51156                 -2.61031                 8.45233
   8 │ ft_ou_25_under           1.86        2.04              -5.00532              15.4968                  -4.9738                 15.359
   9 │ ft_ou_25_over            1.96        2.16              -4.10192              16.33                    -4.13514                16.1847
  10 │ ft_ou_35_under           1.34        1.4               -2.90376               9.89485                 -3.20106                 9.87882
  11 │ ft_ou_35_over            3.45        3.95              -4.98658              25.4755                  -4.22116                25.4343
  12 │ ft_btts_yes              1.76        1.93              -7.6235               11.4837                  -8.39867                11.3148
  13 │ ft_btts_no               2.08        2.32              -1.17223              13.5716                  -0.256123               13.372
  14 │ ft_cs_0_0               11.0        15.5              -17.3218               27.6107                 -15.492                  27.7429
  15 │ ft_cs_1_0                9.6        12.0               -7.66477              21.0902                  -7.34577                20.5797
  16 │ ft_cs_0_1                9.6        12.0               -9.54252              24.3926                  -9.36987                24.1123
  17 │ ft_cs_1_1                6.8         8.2              -18.3258                7.33638                -19.6259                  7.35944
  18 │ ft_cs_2_0               15.5        22.0               -0.525187             26.9762                  -0.427694               26.3049
  19 │ ft_cs_0_2               15.0        21.0               -8.70023              28.0387                  -8.93566                27.5392
  20 │ ft_cs_2_1               10.5        13.5              -16.1793               12.3935                 -17.7191                 12.0003
  21 │ ft_cs_1_2               10.5        13.5              -18.8404               12.2764                 -20.4812                 12.0131
  22 │ ft_cs_2_2               15.5        21.0              -20.6805               13.8055                 -22.4812                 13.1904
  23 │ ft_cs_3_0               34.0        50.0                1.21786              42.9907                   2.66753                42.9832
  24 │ ft_cs_0_3               27.0        48.0              -27.1523               30.4915                 -26.5726                 29.9843
  25 │ ft_cs_3_1               24.0        40.0              -11.4769               30.2666                 -11.9903                 29.6917
  26 │ ft_cs_1_3               23.0        38.0              -21.4888               23.8075                 -22.2782                 23.1084
  27 │ ft_cs_3_2               30.0        55.0              -29.3538               24.218                  -30.0897                 23.4709
  28 │ ft_cs_2_3               30.0        55.0              -32.4651               20.562                  -33.315                  19.8084
  29 │ ft_cs_3_3                7.0       200.0              -92.7787                3.0186                 -92.7796                  2.94491
  30 │ ft_cs_other_home        22.0        34.0               13.0191               71.5858                  18.7701                 75.5823
  31 │ ft_cs_other_draw         7.2         0.0              -99.0282                0.686895               -98.9652                  0.723317
  32 │ ft_cs_other_away        20.0        26.0              -11.785                46.8933                  -8.25806                48.2551
  33 │ ht_1x2_home              3.4         3.9               11.0607               27.7146                  10.0948                 27.0684
  34 │ ht_1x2_draw              2.14        2.4              -14.4925                9.6532                 -14.1333                  9.60851
  35 │ ht_1x2_away              3.25        3.8              -11.0205               21.9245                 -10.6427                 21.5554
  36 │ ht_ou_05_under           2.92        3.35             -18.7932               18.3396                 -16.4136                 17.7012
  37 │ ht_ou_05_over            1.42        1.53               2.50903               8.91856                  1.35184                 8.60814
  38 │ ht_ou_15_under           1.42        1.51             -10.9831               11.4877                 -10.5561                 11.0548
  39 │ ht_ou_15_over            2.96        3.4               10.4436               23.9461                   9.55346                23.0438
  40 │ ht_ou_25_under           1.1         1.14              -6.22846               6.01622                 -6.55938                 5.97354
  41 │ ht_ou_25_over            8.4        10.5               23.9264               45.942                   26.4534                 45.6161
  42 │ ht_cs_0_0                2.88        3.35             -19.9056               18.0883                 -17.5587                 17.4588
  43 │ ht_cs_1_0                4.5         6.2              -16.806                13.694                  -18.4627                 13.0319
  44 │ ht_cs_0_1                4.4         6.0              -27.8846               14.778                  -28.5271                 14.1233
  45 │ ht_cs_1_1                7.2        10.0              -21.6668                9.81685                -26.1779                  8.99723
  46 │ ht_cs_2_0                7.4        24.0              -51.203                18.0974                 -51.6114                 17.2909
  47 │ ht_cs_2_1                8.4      1000.0              -67.4642               11.3033                 -69.0367                 10.1578
  48 │ ht_cs_2_2               10.0      1000.0              -88.0157                5.56066                -88.3282                  5.11644
  49 │ ht_cs_0_2                7.2        22.0              -63.3874               13.4553                 -62.8102                 13.1192
  50 │ ht_cs_1_2                8.4      1000.0              -71.6748                9.23299                -72.6801                  8.44604
--- Distribution Comparison for ssm_neg_bin ---
50×6 DataFrame
 Row │ market            market_back  market_lay  ssm_neg_bin_mean_odds  ssm_neg_bin_median_odds  ssm_neg_bin_market_quantile 
     │ String            Float64?     Float64?    Float64?               Float64?                 Float64?                    
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ ft_1x2_home              2.68        3.05                   2.76                     2.65                        0.474
   2 │ ft_1x2_draw              3.3         3.8                    3.93                     3.89                        0.974
   3 │ ft_1x2_away              2.56        2.92                   2.95                     2.78                        0.63
   4 │ ft_ou_05_under          11.0        14.5                   14.58                    13.5                         0.742
   5 │ ft_ou_05_over            1.08        1.1                    1.08                     1.08                        0.5
   6 │ ft_ou_15_under           3.5         4.1                    3.97                     3.77                        0.628
   7 │ ft_ou_15_over            1.33        1.37                   1.38                     1.36                        0.614
   8 │ ft_ou_25_under           1.86        2.04                   2.01                     1.95                        0.622
   9 │ ft_ou_25_over            1.96        2.16                   2.11                     2.05                        0.606
  10 │ ft_ou_35_under           1.34        1.4                    1.4                      1.37                        0.596
  11 │ ft_ou_35_over            3.45        3.95                   3.88                     3.7                         0.596
  12 │ ft_btts_yes              1.76        1.93                   1.95                     1.92                        0.772
  13 │ ft_btts_no               2.08        2.32                   2.12                     2.09                        0.518
  14 │ ft_cs_0_0               11.0        15.5                   14.58                    13.5                         0.742
  15 │ ft_cs_1_0                9.6        12.0                   10.9                     10.5                         0.686
  16 │ ft_cs_0_1                9.6        12.0                   11.47                    10.63                        0.648
  17 │ ft_cs_1_1                6.8         8.2                    8.54                     8.29                        1.0
  18 │ ft_cs_2_0               15.5        22.0                   16.76                    15.93                        0.54
  19 │ ft_cs_0_2               15.0        21.0                   18.29                    16.87                        0.64
  20 │ ft_cs_2_1               10.5        13.5                   13.07                    12.56                        0.95
  21 │ ft_cs_1_2               10.5        13.5                   13.55                    12.87                        0.982
  22 │ ft_cs_2_2               15.5        21.0                   20.65                    19.79                        0.96
  23 │ ft_cs_3_0               34.0        50.0                   39.71                    35.5                         0.54
  24 │ ft_cs_0_3               27.0        48.0                   44.39                    38.18                        0.8
  25 │ ft_cs_3_1               24.0        40.0                   30.85                    28.13                        0.686
  26 │ ft_cs_1_3               23.0        38.0                   32.74                    29.49                        0.83
  27 │ ft_cs_3_2               30.0        55.0                   48.51                    44.54                        0.89
  28 │ ft_cs_2_3               30.0        55.0                   49.65                    45.96                        0.93
  29 │ ft_cs_3_3                7.0       200.0                  116.15                   102.88                        1.0
  30 │ ft_cs_other_home        22.0        34.0                   27.25                    22.07                        0.51
  31 │ ft_cs_other_draw         7.2         0.0                 1130.69                   857.35                        1.0
  32 │ ft_cs_other_away        20.0        26.0                   28.97                    23.46                        0.624
  33 │ ht_1x2_home              3.4         3.9                    3.29                     3.19                        0.386
  34 │ ht_1x2_draw              2.14        2.4                    2.52                     2.5                         0.922
  35 │ ht_1x2_away              3.25        3.8                    3.87                     3.72                        0.716
  36 │ ht_ou_05_under           2.92        3.35                   3.67                     3.51                        0.822
  37 │ ht_ou_05_over            1.42        1.53                   1.41                     1.4                         0.43
  38 │ ht_ou_15_under           1.42        1.51                   1.61                     1.58                        0.834
  39 │ ht_ou_15_over            2.96        3.4                    2.83                     2.73                        0.354
  40 │ ht_ou_25_under           1.1         1.14                   1.18                     1.17                        0.882
  41 │ ht_ou_25_over            8.4        10.5                    7.59                     6.94                        0.29
  42 │ ht_cs_0_0                2.88        3.35                   3.67                     3.51                        0.844
  43 │ ht_cs_1_0                4.5         6.2                    5.67                     5.52                        0.918
  44 │ ht_cs_0_1                4.4         6.0                    6.43                     6.11                        0.974
  45 │ ht_cs_1_1                7.2        10.0                    9.91                     9.67                        1.0
  46 │ ht_cs_2_0                7.4        24.0                   17.62                    16.11                        0.994
  47 │ ht_cs_2_1                8.4      1000.0                   30.68                    27.68                        1.0
  48 │ ht_cs_2_2               10.0      1000.0                  105.24                    92.7                         1.0
  49 │ ht_cs_0_2                7.2        22.0                   22.19                    20.33                        1.0
  50 │ ht_cs_1_2                8.4      1000.0                   34.07                    31.32                        1.0

#### league 1 


Analyzing Match: east-fife vs cove-rangers
results: ht: 1-0, ft: 2-0

Row │ market            market_back  market_lay  ssm_poiss_mean_ev_pct  ssm_poiss_std_ev_pct  ssm_neg_bin_mean_ev_pct  ssm_neg_bin_std_ev_pct 
     │ String            Float64      Float64     Float64?               Float64?              Float64?                 Float64?               
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ ft_1x2_home              2.02        2.1              -19.9639                17.7616               -20.188                    17.4599
   2 │ ft_1x2_draw              3.65        3.95             -11.4447                 7.47729              -11.8141                    7.57661
   3 │ ft_1x2_away              3.7         4.0               33.6296                31.3536                34.413                    30.7682
   4 │ ft_ou_05_under          12.0        18.0              -31.9994                24.9124               -29.5278                   25.5016
   5 │ ft_ou_05_over            1.07        1.09               0.936612               2.22135                0.71623                   2.27389
   6 │ ft_ou_15_under           3.9         4.6              -15.9088                22.814                -14.4295                   23.018
   7 │ ft_ou_15_over            1.28        1.34               0.400841               7.48768               -0.0846829                 7.55462
   8 │ ft_ou_25_under           2.02        2.24             -10.6961                17.0453               -10.1716                   17.0967
   9 │ ft_ou_25_over            1.8         1.99               0.422262              15.1889                -0.0450828                15.2347
  10 │ ft_ou_35_under           1.38        1.46              -8.83012               11.4619                -8.86522                  11.5305
  11 │ ft_ou_35_over            3.1         3.6                5.1981                25.7477                 5.27695                  25.9018
  12 │ ft_btts_yes              1.73        1.85               0.215177              11.4766                -0.831417                 11.5295
  13 │ ft_btts_no               2.16        2.4               -9.12415               14.3291                -7.81742                  14.3952
  14 │ ft_cs_0_0               13.0        15.5              -26.3327                26.9884               -23.6551                   27.6267
  15 │ ft_cs_1_0                9.0        10.5              -26.6772                19.4989               -26.1111                   19.3715
  16 │ ft_cs_0_1               13.5        18.0                4.59998               29.6628                 6.09107                  29.7816
  17 │ ft_cs_1_1                7.6         8.8              -15.524                 10.286                -16.6011                   10.1558
  18 │ ft_cs_2_0               12.0        14.5              -27.2576                20.2754               -27.5427                   19.6637
  19 │ ft_cs_0_2               23.0        32.0               25.8302                37.8648                26.8838                   37.4512
  20 │ ft_cs_2_1                9.2        11.0              -24.0677                 9.66334              -25.8442                    9.38746
  21 │ ft_cs_1_2               13.5        16.5                5.76168               15.2702                 3.88984                  14.7926
  22 │ ft_cs_2_2               15.0        18.0              -12.8847                12.2411               -15.2902                   11.9724
  23 │ ft_cs_3_0               20.0        25.0              -37.814                 24.0704               -37.7987                   23.2779
  24 │ ft_cs_0_3               50.0       140.0               32.9134                53.9378                35.2092                   52.979
  25 │ ft_cs_3_1               17.5        22.0              -26.0544                20.8384               -27.4446                   20.1346
  26 │ ft_cs_1_3               36.0        65.0               36.8125                40.4181                35.6671                   38.8106
  27 │ ft_cs_3_2               26.0        46.0              -22.8055                22.0716               -24.541                    21.3812
  28 │ ft_cs_2_3               36.0       100.0                1.27202               29.2775                -0.511846                 28.3056
  29 │ ft_cs_3_3               23.0       220.0              -66.9665                12.685                -67.3497                   12.399
  30 │ ft_cs_other_home        12.0        15.0              -17.7318                44.3745               -15.4704                   45.6999
  31 │ ft_cs_other_draw        10.0         0.0              -97.6419                 1.65079              -97.5259                    1.7409
  32 │ ft_cs_other_away        36.0       980.0              108.482                113.809                116.746                   115.644
  33 │ ht_1x2_home              2.66        2.94             -35.404                 16.098                -35.2352                   16.3518
  34 │ ht_1x2_draw              2.22        2.44             -12.1843                 9.52658              -11.7149                    9.69347
  35 │ ht_1x2_away              4.4         5.1               59.1003                30.1924                57.8906                   29.7476
  36 │ ht_ou_05_under           3.15        3.45             -13.6393                18.978                -10.8671                   18.843
  37 │ ht_ou_05_over            1.41        1.45               2.34331                8.49491                1.10244                   8.43451
  38 │ ht_ou_15_under           1.45        1.53              -9.75124               11.1329                -9.25725                  10.9513
  39 │ ht_ou_15_over            2.9         3.3                9.50249               22.2658                 8.5145                   21.9026
  40 │ ht_ou_25_under           1.11        1.14              -5.64736                5.6805                -5.98539                   5.73275
  41 │ ht_ou_25_over            8.2         9.8               22.9805                41.964                 25.4777                   42.35
  42 │ ht_cs_0_0                3.15        3.35             -13.6393                18.978                -10.8671                   18.843
  43 │ ht_cs_1_0                4.2         5.0              -37.8178                11.3211               -38.4007                   10.8646
  44 │ ht_cs_0_1                6.2         7.6               24.1186                17.6662                21.6351                   17.0133
  45 │ ht_cs_1_1                8.6        10.5               -6.53012               11.9399               -12.0851                   11.6315
  46 │ ht_cs_2_0               12.5        15.5              -46.8074                20.3432               -45.9771                   20.4582
  47 │ ht_cs_2_1                8.6        34.0              -73.0424                 9.69009              -74.026                     9.30911
  48 │ ht_cs_2_2               10.5      1000.0              -87.3583                 5.67608              -87.7039                    5.36624
  49 │ ht_cs_0_2                8.8        42.0              -32.7868                19.0105               -33.3651                   18.2326
  50 │ ht_cs_1_2                9.4      1000.0              -60.8639                11.1199               -62.8265                   10.0961
  51 │ ht_cs_other              8.0      1000.0              -48.0349                21.4668               -42.7503                   23.8044
--- Distribution Comparison for ssm_neg_bin ---
51×6 DataFrame
 Row │ market            market_back  market_lay  ssm_neg_bin_mean_odds  ssm_neg_bin_median_odds  ssm_neg_bin_market_quantile 
     │ String            Float64      Float64     Float64?               Float64?                 Float64?                    
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ ft_1x2_home              2.02        2.1                    2.67                     2.57                        0.876
   2 │ ft_1x2_draw              3.65        3.95                   4.17                     4.14                        0.94
   3 │ ft_1x2_away              3.7         4.0                    2.91                     2.78                        0.128
   4 │ ft_ou_05_under          12.0        18.0                   19.65                    17.76                        0.88
   5 │ ft_ou_05_over            1.07        1.09                   1.06                     1.06                        0.352
   6 │ ft_ou_15_under           3.9         4.6                    4.94                     4.61                        0.736
   7 │ ft_ou_15_over            1.28        1.34                   1.29                     1.28                        0.49
   8 │ ft_ou_25_under           2.02        2.24                   2.34                     2.23                        0.712
   9 │ ft_ou_25_over            1.8         1.99                   1.85                     1.81                        0.508
  10 │ ft_ou_35_under           1.38        1.46                   1.54                     1.5                         0.752
  11 │ ft_ou_35_over            3.1         3.6                    3.14                     3.01                        0.454
  12 │ ft_btts_yes              1.73        1.85                   1.77                     1.73                        0.51
  13 │ ft_btts_no               2.16        2.4                    2.4                      2.36                        0.698
  14 │ ft_cs_0_0               13.0        15.5                   19.65                    17.76                        0.818
  15 │ ft_cs_1_0                9.0        10.5                   13.13                    12.4                         0.904
  16 │ ft_cs_0_1               13.5        18.0                   13.87                    12.92                        0.436
  17 │ ft_cs_1_1                7.6         8.8                    9.27                     8.99                        0.982
  18 │ ft_cs_2_0               12.0        14.5                   17.92                    16.61                        0.932
  19 │ ft_cs_0_2               23.0        32.0                   19.93                    18.44                        0.268
  20 │ ft_cs_2_1                9.2        11.0                   12.64                    12.23                        1.0
  21 │ ft_cs_1_2               13.5        16.5                   13.31                    12.84                        0.388
  22 │ ft_cs_2_2               15.0        18.0                   18.12                    17.44                        0.924
  23 │ ft_cs_3_0               20.0        25.0                   37.51                    34.01                        0.946
  24 │ ft_cs_0_3               50.0       140.0                   43.83                    38.48                        0.282
  25 │ ft_cs_3_1               17.5        22.0                   26.41                    24.46                        0.908
  26 │ ft_cs_1_3               36.0        65.0                   29.23                    27.32                        0.186
  27 │ ft_cs_3_2               26.0        46.0                   37.8                     35.04                        0.86
  28 │ ft_cs_2_3               36.0       100.0                   39.71                    37.3                         0.544
  29 │ ft_cs_3_3               23.0       220.0                   82.66                    73.08                        1.0
  30 │ ft_cs_other_home        12.0        15.0                   19.36                    16.24                        0.698
  31 │ ft_cs_other_draw        10.0         0.0                  645.02                   495.93                        1.0
  32 │ ft_cs_other_away        36.0       980.0                   22.43                    18.81                        0.112
  33 │ ht_1x2_home              2.66        2.94                   4.38                     4.19                        0.972
  34 │ ht_1x2_draw              2.22        2.44                   2.54                     2.52                        0.896
  35 │ ht_1x2_away              4.4         5.1                    2.89                     2.79                        0.018
  36 │ ht_ou_05_under           3.15        3.45                   3.7                      3.55                        0.75
  37 │ ht_ou_05_over            1.41        1.45                   1.4                      1.39                        0.432
  38 │ ht_ou_15_under           1.45        1.53                   1.62                     1.59                        0.81
  39 │ ht_ou_15_over            2.9         3.3                    2.8                      2.7                         0.342
  40 │ ht_ou_25_under           1.11        1.14                   1.19                     1.17                        0.858
  41 │ ht_ou_25_over            8.2         9.8                    7.44                     6.83                        0.288
  42 │ ht_cs_0_0                3.15        3.35                   3.7                      3.55                        0.75
  43 │ ht_cs_1_0                4.2         5.0                    7.05                     6.81                        1.0
  44 │ ht_cs_0_1                6.2         7.6                    5.21                     5.05                        0.118
  45 │ ht_cs_1_1                8.6        10.5                    9.98                     9.67                        0.858
  46 │ ht_cs_2_0               12.5        15.5                   26.71                    24.55                        0.974
  47 │ ht_cs_2_1                8.6        34.0                   38.05                    34.32                        1.0
  48 │ ht_cs_2_2               10.5      1000.0                  105.96                    92.37                        1.0
  49 │ ht_cs_0_2                8.8        42.0                   14.28                    13.3                         0.956
  50 │ ht_cs_1_2                9.4      1000.0                   27.5                     25.76                        1.0
  51 │ ht_cs_other              8.0      1000.0                   16.87                    15.22                        0.938

Analyzing Match: hamilton-academical vs inverness-caledonian-thistle

rsults: ht 1-0, ft: 3-2

--- Expected Value (EV) Analysis ---
51×7 DataFrame
 Row │ market            market_back  market_lay  ssm_poiss_mean_ev_pct  ssm_poiss_std_ev_pct  ssm_neg_bin_mean_ev_pct  ssm_neg_bin_std_ev_pct 
     │ String            Float64      Float64     Float64?               Float64?              Float64?                 Float64?               
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ ft_1x2_home              3.75        4.0               41.4735               31.9623                  41.1819                 32.1725
   2 │ ft_1x2_draw              3.7         3.9                2.16277               9.55458                  1.94718                 9.914
   3 │ ft_1x2_away              2.02        2.12             -29.9825               15.8363                 -29.708                  15.9301
   4 │ ft_ou_05_under          10.5        17.0                2.99823              33.3074                   5.61795                34.5443
   5 │ ft_ou_05_over            1.07        1.09              -3.49601               3.39418                 -3.76297                 3.52023
   6 │ ft_ou_15_under           3.45        3.95              10.6729               25.0604                  11.8077                 25.4604
   7 │ ft_ou_15_over            1.34        1.4               -8.98601               9.73362                 -9.42675                 9.88896
   8 │ ft_ou_25_under           1.85        2.02               7.01139              15.8221                   7.22102                15.9295
   9 │ ft_ou_25_over            1.98        2.18             -16.5311               16.934                  -16.7555                 17.0489
  10 │ ft_ou_35_under           1.35        1.39               5.42946               9.32036                  5.26481                 9.40039
  11 │ ft_ou_35_over            3.55        3.9              -22.2404               24.5091                 -21.8075                 24.7195
  12 │ ft_btts_yes              1.81        1.98             -14.7487               12.402                  -15.6947                 12.5134
  13 │ ft_btts_no               2.02        2.22               6.85767              13.8409                   7.91341                13.9652
  14 │ ft_cs_0_0               12.0        14.5               17.7123               38.0656                  20.7062                 39.4792
  15 │ ft_cs_1_0               12.5        15.0               41.9419               29.8292                  42.1745                 29.779
  16 │ ft_cs_0_1                8.0         9.0              -12.6848               22.2182                 -12.1983                 22.1875
  17 │ ft_cs_1_1                7.2         8.4               -9.12238               6.1968                 -10.6333                  6.25564
  18 │ ft_cs_2_0               25.0        38.0               71.8392               45.7739                  71.0984                 45.5673
  19 │ ft_cs_0_2               11.0        12.0              -31.0363               20.5026                 -30.7736                 20.3658
  20 │ ft_cs_2_1               14.5        19.0               10.6584               18.5437                   8.18633                18.2618
  21 │ ft_cs_1_2                9.2        10.5              -33.3517               10.7309                 -34.5369                 10.7462
  22 │ ft_cs_2_2               16.5        19.0              -27.7904               15.9571                 -29.4879                 15.7039
  23 │ ft_cs_3_0               30.0        85.0              -13.047                37.6834                 -12.4981                 38.0633
  24 │ ft_cs_0_3               21.0        23.0              -47.973                21.9871                 -47.0123                 22.22
  25 │ ft_cs_3_1               34.0        70.0                9.26152              40.1328                   7.88432                39.4969
  26 │ ft_cs_1_3               19.0        25.0              -45.6463               17.9596                 -45.828                  17.9187
  27 │ ft_cs_3_2               38.0        60.0              -30.0489               27.5243                 -31.0456                 26.9284
  28 │ ft_cs_2_3               32.0        40.0              -44.7361               20.2313                 -45.2476                 19.9803
  29 │ ft_cs_3_3               17.0       110.0              -87.6605                6.14905                -87.6612                  6.091
  30 │ ft_cs_other_home        14.5       980.0              -43.2629               38.37                   -41.0559                 40.1343
  31 │ ft_cs_other_draw        19.0      1000.0              -98.5158                1.27918                -98.4206                  1.36661
  32 │ ft_cs_other_away        13.5        16.0              -57.2223               25.5812                 -55.0643                 27.0595
  33 │ ht_1x2_home              1.04      980.0              -74.3418                6.45808                -74.3614                  6.29867
  34 │ ht_1x2_draw              1.02      980.0              -54.3659                4.92382                -54.2214                  4.9076
  35 │ ht_1x2_away              1.04      980.0              -68.1871                6.56043                -68.3147                  6.4383
  36 │ ht_ou_05_under           1.2       110.0              -58.4934                8.12169                -57.7757                  7.93379
  37 │ ht_ou_05_over            1.01      110.0              -33.9347                6.83575                -34.5388                  6.67761
  38 │ ht_ou_15_under           1.04      110.0              -26.5351                7.61904                -26.534                   7.32315
  39 │ ht_ou_15_over            1.06     1000.0              -68.8777                7.76556                -68.8788                  7.46398
  40 │ ht_ou_25_under           1.11        1.14              -0.018622              4.64476                 -0.403332                4.53655
  41 │ ht_ou_25_over            1.1      1000.0              -89.0806                4.60292                -88.6994                  4.49568
  42 │ ht_cs_0_0                3.15        3.45               8.95488              21.3194                  10.8388                 20.8262
  43 │ ht_cs_1_0                6.6       950.0                8.67557              19.3245                   7.21613                18.5484
  44 │ ht_cs_0_1                4.3         5.7              -15.7867               12.9545                 -17.4032                 12.2484
  45 │ ht_cs_1_1                5.2        11.0              -51.1706                8.41088                -53.5728                  7.90382
  46 │ ht_cs_2_0                3.75     1000.0              -84.2206                6.5123                 -83.9595                  6.31044
  47 │ ht_cs_2_1                2.16     1000.0              -94.7828                2.20481                -94.8983                  2.03096
  48 │ ht_cs_2_2                2.22     1000.0              -98.3783                0.906758               -98.3781                  0.839616
  49 │ ht_cs_0_2                6.2        16.5              -63.8175               11.8763                 -63.5403                 11.562
  50 │ ht_cs_1_2                7.6      1000.0              -78.57                  7.7797                 -79.1189                  7.17945
  51 │ ht_cs_other              2.1      1000.0              -91.6817                4.17802                -90.6901                  4.39726
--- Distribution Comparison for ssm_neg_bin ---
51×6 DataFrame
 Row │ market            market_back  market_lay  ssm_neg_bin_mean_odds  ssm_neg_bin_median_odds  ssm_neg_bin_market_quantile 
     │ String            Float64      Float64     Float64?               Float64?                 Float64?                    
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ ft_1x2_home              3.75        4.0                    2.81                     2.7                         0.088
   2 │ ft_1x2_draw              3.7         3.9                    3.66                     3.64                        0.432
   3 │ ft_1x2_away              2.02        2.12                   3.04                     2.88                        0.966
   4 │ ft_ou_05_under          10.5        17.0                   11.13                    10.45                        0.488
   5 │ ft_ou_05_over            1.07        1.09                   1.11                     1.11                        0.864
   6 │ ft_ou_15_under           3.45        3.95                   3.26                     3.13                        0.35
   7 │ ft_ou_15_over            1.34        1.4                    1.5                      1.47                        0.828
   8 │ ft_ou_25_under           1.85        2.02                   1.77                     1.73                        0.338
   9 │ ft_ou_25_over            1.98        2.18                   2.49                     2.38                        0.842
  10 │ ft_ou_35_under           1.35        1.39                   1.29                     1.27                        0.262
  11 │ ft_ou_35_over            3.55        3.9                    5.05                     4.65                        0.816
  12 │ ft_btts_yes              1.81        1.98                   2.2                      2.15                        0.902
  13 │ ft_btts_no               2.02        2.22                   1.91                     1.87                        0.302
  14 │ ft_cs_0_0               12.0        14.5                   11.13                    10.45                        0.338
  15 │ ft_cs_1_0               12.5        15.0                    9.22                     8.85                        0.07
  16 │ ft_cs_0_1                8.0         9.0                    9.78                     9.24                        0.708
  17 │ ft_cs_1_1                7.2         8.4                    8.1                      7.92                        1.0
  18 │ ft_cs_2_0               25.0        38.0                   15.79                    14.92                        0.046
  19 │ ft_cs_0_2               11.0        12.0                   17.54                    16.14                        0.93
  20 │ ft_cs_2_1               14.5        19.0                   13.85                    13.09                        0.31
  21 │ ft_cs_1_2                9.2        10.5                   14.5                     13.93                        1.0
  22 │ ft_cs_2_2               16.5        19.0                   24.77                    23.38                        0.976
  23 │ ft_cs_3_0               30.0        85.0                   41.91                    36.62                        0.694
  24 │ ft_cs_0_3               21.0        23.0                   48.16                    41.88                        0.966
  25 │ ft_cs_3_1               34.0        70.0                   36.73                    32.77                        0.454
  26 │ ft_cs_1_3               19.0        25.0                   39.73                    36.27                        0.982
  27 │ ft_cs_3_2               38.0        60.0                   65.6                     57.27                        0.876
  28 │ ft_cs_2_3               32.0        40.0                   67.73                    60.22                        0.972
  29 │ ft_cs_3_3               17.0       110.0                  179.19                   150.83                        1.0
  30 │ ft_cs_other_home        14.5       980.0                   38.61                    29.0                         0.874
  31 │ ft_cs_other_draw        19.0      1000.0                 2319.67                  1561.07                        1.0
  32 │ ft_cs_other_away        13.5        16.0                   42.28                    34.22                        0.958
  33 │ ht_1x2_home              1.04      980.0                    4.31                     4.18                        1.0
  34 │ ht_1x2_draw              1.02      980.0                    2.25                     2.24                        1.0
  35 │ ht_1x2_away              1.04      980.0                    3.43                     3.29                        1.0
  36 │ ht_ou_05_under           1.2       110.0                    2.95                     2.87                        1.0
  37 │ ht_ou_05_over            1.01      110.0                    1.56                     1.54                        1.0
  38 │ ht_ou_15_under           1.04      110.0                    1.43                     1.41                        1.0
  39 │ ht_ou_15_over            1.06     1000.0                    3.62                     3.45                        1.0
  40 │ ht_ou_25_under           1.11        1.14                   1.12                     1.11                        0.488
  41 │ ht_ou_25_over            1.1      1000.0                   11.49                    10.25                        1.0
  42 │ ht_cs_0_0                3.15        3.45                   2.95                     2.87                        0.302
  43 │ ht_cs_1_0                6.6       950.0                    6.35                     6.22                        0.356
  44 │ ht_cs_0_1                4.3         5.7                    5.33                     5.17                        0.936
  45 │ ht_cs_1_1                5.2        11.0                   11.57                    11.1                         1.0
  46 │ ht_cs_2_0                3.75     1000.0                   27.52                    24.81                        1.0
  47 │ ht_cs_2_1                2.16     1000.0                   50.47                    44.7                         1.0
  48 │ ht_cs_2_2                2.22     1000.0                  182.45                   151.73                        1.0
  49 │ ht_cs_0_2                6.2        16.5                   19.02                    17.36                        1.0
  50 │ ht_cs_1_2                7.6      1000.0                   41.52                    37.29                        1.0
  51 │ ht_cs_other              2.1      1000.0                   28.13                    25.04                        1.0



Analyzing Match: kelty-hearts-fc vs peterhead
results: ht: 1-1, ft: 1-2
 
--- Expected Value (EV) Analysis ---
51×7 DataFrame
 Row │ market            market_back  market_lay  ssm_poiss_mean_ev_pct  ssm_poiss_std_ev_pct  ssm_neg_bin_mean_ev_pct  ssm_neg_bin_std_ev_pct 
     │ String            Float64      Float64     Float64?               Float64?              Float64?                 Float64?               
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ ft_1x2_home              4.4         5.1              80.3467                 40.2071                80.1742                   39.8463
   2 │ ft_1x2_draw              3.9         4.3              -1.88966                 9.43792               -2.25657                   9.30419
   3 │ ft_1x2_away              1.77        1.9             -40.0761                 14.6123               -39.8406                   14.4372
   4 │ ft_ou_05_under          13.5        18.0              -7.97277                33.6004                -5.27516                  33.4439
   5 │ ft_ou_05_over            1.06        1.08             -1.22584                 2.63825               -1.43765                   2.62596
   6 │ ft_ou_15_under           4.0         4.7              -1.28456                26.3019                 0.0602584                25.824
   7 │ ft_ou_15_over            1.27        1.31             -4.34215                 8.35087               -4.76913                   8.19913
   8 │ ft_ou_25_under           2.06        2.28             -0.0809338              18.3901                 0.335649                 17.9437
   9 │ ft_ou_25_over            1.79        1.95             -7.82288                15.9797                -8.18486                  15.5918
  10 │ ft_ou_35_under           1.42        1.47             -0.493765               11.8044                -0.552388                 11.5125
  11 │ ft_ou_35_over            3.0         3.4             -10.2244                 24.9388               -10.1006                   24.3222
  12 │ ft_btts_yes              1.79        1.91             -2.62245                12.1526                -3.64622                  11.983
  13 │ ft_btts_no               2.1         2.28             -4.24182                14.2572                -3.04075                  14.0583
  14 │ ft_cs_0_0               15.0        17.0               2.25247                37.3338                 5.24982                  37.1599
  15 │ ft_cs_1_0               16.5        19.0              54.6353                 36.9993                55.5024                   35.9984
  16 │ ft_cs_0_1                9.0        10.0             -23.5883                 22.6087               -22.8338                   22.2685
  17 │ ft_cs_1_1                8.4         9.2              -2.05669                11.033                 -3.37207                  10.5738
  18 │ ft_cs_2_0               28.0        38.0              88.0774                 47.7257                87.7                      46.8978
  19 │ ft_cs_0_2               10.0        12.0             -45.5143                 17.2605               -45.1533                   17.1142
  20 │ ft_cs_2_1               16.5        19.0              37.7103                 18.1034                34.7661                   17.4018
  21 │ ft_cs_1_2                9.4        10.5             -29.7242                 11.0309               -30.9021                   10.8161
  22 │ ft_cs_2_2               17.0        19.0              -9.14259                14.8794               -11.4092                   14.5032
  23 │ ft_cs_3_0               11.5       120.0             -61.5468                 14.8959               -61.3379                   14.79
  24 │ ft_cs_0_3               17.5        21.0             -57.9226                 17.8468               -57.1862                   17.737
  25 │ ft_cs_3_1               23.0        60.0              -4.58371                29.7323                -6.01574                  28.6676
  26 │ ft_cs_1_3               16.0        18.0             -47.2471                 16.7102               -47.5916                   16.1137
  27 │ ft_cs_3_2               48.0        65.0              27.311                  40.8315                24.8893                   39.4632
  28 │ ft_cs_2_3               28.0        36.0             -34.0776                 20.9792               -35.0688                   20.0745
  29 │ ft_cs_3_3               50.0       100.0             -41.6702                 24.6675               -42.2085                   23.8088
  30 │ ft_cs_other_home        42.0        60.0             169.149                 167.254                177.328                   167.765
  31 │ ft_cs_other_draw        12.5      1000.0             -97.8925                  1.68438              -97.8077                    1.63659
  32 │ ft_cs_other_away         9.4        11.0             -58.7852                 25.1299               -57.3446                   24.5707
  33 │ ht_1x2_home              4.8         5.9              21.4905                 33.5841                22.6047                   32.7231
  34 │ ht_1x2_draw              2.3         2.5             -15.9285                  9.34658              -15.5715                    9.49892
  35 │ ht_1x2_away              2.4         2.66             -8.47207                19.2244                -9.40169                  18.4799
  36 │ ht_ou_05_under           3.15        3.55            -26.8668                 17.8593               -24.2019                   17.9472
  37 │ ht_ou_05_over            1.39        1.47              6.72854                 7.88077                5.55257                   7.91955
  38 │ ht_ou_15_under           1.48        1.57            -16.4999                 12.3964               -15.9365                   12.1622
  39 │ ht_ou_15_over            2.76        3.1              20.2836                 23.1176                19.2329                   22.6808
  40 │ ht_ou_25_under           1.12        1.16             -9.47253                 7.25477               -9.84179                   7.15022
  41 │ ht_ou_25_over            7.4         9.4              41.8721                 47.9333                44.3118                   47.2425
  42 │ ht_cs_0_0                3.4         3.6             -21.0626                 19.2767               -18.1861                   19.3716
  43 │ ht_cs_1_0                7.0         8.4              -1.17476                21.093                 -1.49874                  19.7544
  44 │ ht_cs_0_1                4.1         4.8             -21.7548                 13.2303               -23.4728                   12.3796
  45 │ ht_cs_1_1                8.4        10.5              -2.64215                10.207                 -8.18289                  10.0741
  46 │ ht_cs_2_0               10.0        42.0             -54.1907                 18.1284               -53.2832                   17.4056
  47 │ ht_cs_2_1               10.0      1000.0             -62.3678                 13.3018               -63.6619                   12.3654
  48 │ ht_cs_2_2               11.5      1000.0             -81.3076                  8.42756              -81.8413                    7.80231
  49 │ ht_cs_0_2               11.0        13.0              -9.38227                27.2878               -10.9392                   25.4485
  50 │ ht_cs_1_2               23.0        28.0              14.986                  30.4702                 9.07152                  28.2632
  51 │ ht_cs_other              8.2        22.0             -27.9723                 30.3214               -21.7187                   31.3799
--- Distribution Comparison for ssm_neg_bin ---
51×6 DataFrame
 Row │ market            market_back  market_lay  ssm_neg_bin_mean_odds  ssm_neg_bin_median_odds  ssm_neg_bin_market_quantile 
     │ String            Float64      Float64     Float64?               Float64?                 Float64?                    
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ ft_1x2_home              4.4         5.1                    2.57                     2.46                        0.012
   2 │ ft_1x2_draw              3.9         4.3                    4.03                     3.99                        0.602
   3 │ ft_1x2_away              1.77        1.9                    3.13                     2.96                        0.994
   4 │ ft_ou_05_under          13.5        18.0                   16.37                    14.95                        0.62
   5 │ ft_ou_05_over            1.06        1.08                   1.08                     1.07                        0.692
   6 │ ft_ou_15_under           4.0         4.7                    4.31                     4.06                        0.524
   7 │ ft_ou_15_over            1.27        1.31                   1.34                     1.33                        0.714
   8 │ ft_ou_25_under           2.06        2.28                   2.13                     2.05                        0.48
   9 │ ft_ou_25_over            1.79        1.95                   2.01                     1.95                        0.714
  10 │ ft_ou_35_under           1.42        1.47                   1.45                     1.41                        0.482
  11 │ ft_ou_35_over            3.0         3.4                    3.6                      3.42                        0.688
  12 │ ft_btts_yes              1.79        1.91                   1.89                     1.85                        0.638
  13 │ ft_btts_no               2.1         2.28                   2.21                     2.17                        0.598
  14 │ ft_cs_0_0               15.0        17.0                   16.37                    14.95                        0.494
  15 │ ft_cs_1_0               16.5        19.0                   11.29                    10.64                        0.058
  16 │ ft_cs_0_1                9.0        10.0                   12.8                     11.94                        0.846
  17 │ ft_cs_1_1                8.4         9.2                    8.82                     8.53                        0.554
  18 │ ft_cs_2_0               28.0        38.0                   16.0                     15.01                        0.022
  19 │ ft_cs_0_2               10.0        12.0                   20.35                    18.84                        0.992
  20 │ ft_cs_2_1               16.5        19.0                   12.48                    11.88                        0.042
  21 │ ft_cs_1_2                9.4        10.5                   13.99                    13.44                        1.0
  22 │ ft_cs_2_2               17.0        19.0                   19.77                    18.93                        0.768
  23 │ ft_cs_3_0               11.5       120.0                   35.01                    30.87                        1.0
  24 │ ft_cs_0_3               17.5        21.0                   49.27                    43.48                        0.992
  25 │ ft_cs_3_1               23.0        60.0                   27.27                    24.6                         0.58
  26 │ ft_cs_1_3               16.0        18.0                   33.81                    31.82                        0.992
  27 │ ft_cs_3_2               48.0        65.0                   43.11                    38.95                        0.288
  28 │ ft_cs_2_3               28.0        36.0                   47.66                    44.34                        0.954
  29 │ ft_cs_3_3               50.0       100.0                  103.7                     90.6                         0.954
  30 │ ft_cs_other_home        42.0        60.0                   21.86                    17.47                        0.092
  31 │ ft_cs_other_draw        12.5      1000.0                  940.47                   703.94                        1.0
  32 │ ft_cs_other_away         9.4        11.0                   29.3                     25.5                         0.966
  33 │ ht_1x2_home              4.8         5.9                    4.22                     4.0                         0.264
  34 │ ht_1x2_draw              2.3         2.5                    2.76                     2.74                        0.94
  35 │ ht_1x2_away              2.4         2.66                   2.77                     2.65                        0.7
  36 │ ht_ou_05_under           3.15        3.55                   4.42                     4.21                        0.894
  37 │ ht_ou_05_over            1.39        1.47                   1.32                     1.31                        0.246
  38 │ ht_ou_15_under           1.48        1.57                   1.8                      1.75                        0.898
  39 │ ht_ou_15_over            2.76        3.1                    2.4                      2.33                        0.208
  40 │ ht_ou_25_under           1.12        1.16                   1.25                     1.23                        0.942
  41 │ ht_ou_25_over            7.4         9.4                    5.72                     5.34                        0.178
  42 │ ht_cs_0_0                3.4         3.6                    4.42                     4.21                        0.822
  43 │ ht_cs_1_0                7.0         8.4                    7.41                     7.17                        0.542
  44 │ ht_cs_0_1                4.1         4.8                    5.52                     5.33                        0.974
  45 │ ht_cs_1_1                8.4        10.5                    9.27                     9.07                        0.776
  46 │ ht_cs_2_0               10.0        42.0                   24.91                    22.45                        0.996
  47 │ ht_cs_2_1               10.0      1000.0                   31.11                    28.34                        1.0
  48 │ ht_cs_2_2               11.5      1000.0                   76.18                    67.98                        1.0
  49 │ ht_cs_0_2               11.0        13.0                   13.52                    12.47                        0.682
  50 │ ht_cs_1_2               23.0        28.0                   22.74                    21.06                        0.388
  51 │ ht_cs_other              8.2        22.0                   12.35                    11.16                        0.786

Analyzing Match: stenhousemuir vs montrose
result: ht: 2-0, ft: 3-1

--- Expected Value (EV) Analysis ---
51×7 DataFrame
 Row │ market            market_back  market_lay  ssm_poiss_mean_ev_pct  ssm_poiss_std_ev_pct  ssm_neg_bin_mean_ev_pct  ssm_neg_bin_std_ev_pct 
     │ String            Float64      Float64     Float64?               Float64?              Float64?                 Float64?               
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ ft_1x2_home              1.94        2.14             -13.6963               18.102                 -13.898                   17.9791
   2 │ ft_1x2_draw              3.3         3.7              -15.4014                9.54806               -15.5141                   9.63841
   3 │ ft_1x2_away              4.1         4.8               22.4966               31.7406                 23.0615                  31.4729
   4 │ ft_ou_05_under          10.0        13.5              -22.2641               28.3834                -19.6679                  28.8075
   5 │ ft_ou_05_over            1.08        1.11              -0.395473              3.06541                -0.675869                 3.11121
   6 │ ft_ou_15_under           3.2         3.65             -13.2536               23.155                 -11.8274                  23.0549
   7 │ ft_ou_15_over            1.38        1.42               0.590602              9.98559                -0.0244241                9.94241
   8 │ ft_ou_25_under           1.76        1.91              -9.12273              16.7221                 -8.56951                 16.4827
   9 │ ft_ou_25_over            2.1         2.32               1.56689              19.9526                  0.906801                19.6668
  10 │ ft_ou_35_under           1.29        1.34              -6.0821               11.0578                 -6.01695                 10.893
  11 │ ft_ou_35_over            3.85        4.5                4.70238              33.0021                  4.50796                 32.51
  12 │ ft_btts_yes              1.74        2.06             -10.6033               12.4441                -11.7306                  12.2213
  13 │ ft_btts_no               1.96        2.36              -4.69971              14.0175                 -3.42993                 13.7666
  14 │ ft_cs_0_0                9.8        14.0              -23.8189               27.8158                -21.2745                  28.2313
  15 │ ft_cs_1_0                7.2         8.8              -22.5518               18.3472                -22.0295                  17.938
  16 │ ft_cs_0_1               11.5        16.5               -1.3532               29.5292                 -0.0479411               29.3504
  17 │ ft_cs_1_1                6.6         8.4              -21.6764                9.08183               -22.6673                   8.80065
  18 │ ft_cs_2_0               10.0        13.0              -22.3791               18.6158                -22.6347                  18.1344
  19 │ ft_cs_0_2               22.0        36.0                7.73943              36.1787                  8.71535                 35.9047
  20 │ ft_cs_2_1                9.2        12.0              -21.2327                9.02651               -23.0158                   8.96254
  21 │ ft_cs_1_2               13.5        20.0               -8.49259              17.276                 -10.0477                  16.7654
  22 │ ft_cs_2_2               16.5        24.0              -19.2615               15.7263                -21.4704                  15.0591
  23 │ ft_cs_3_0               19.0        30.0              -25.8884               28.4079                -25.7591                  27.9882
  24 │ ft_cs_0_3                8.4       140.0              -83.7951                7.3878                -83.4728                   7.39018
  25 │ ft_cs_3_1               18.5        28.0              -20.4399               24.4961                -21.8835                  23.7352
  26 │ ft_cs_1_3               36.0        80.0               -3.75813              34.449                  -4.44714                 33.4005
  27 │ ft_cs_3_2               28.0       980.0              -31.1467               24.1397                -32.7564                  22.9864
  28 │ ft_cs_2_3               44.0       120.0              -14.9582               31.6776                -16.5106                  30.1043
  29 │ ft_cs_3_3                7.6       300.0              -92.6074                3.62733               -92.7161                   3.44292
  30 │ ft_cs_other_home         7.0        28.0              -52.7793               31.2461                -51.5783                  31.899
  31 │ ft_cs_other_draw         6.8         0.0              -99.1215                0.799525              -99.0906                   0.801654
  32 │ ft_cs_other_away         7.4        90.0              -77.1461               14.3352                -76.3235                  14.4876
  33 │ ht_1x2_home              2.66        2.94             -27.8765               17.8677                -27.9217                  17.6553
  34 │ ht_1x2_draw              2.16        2.36             -15.2946                8.93714               -14.9121                   9.02076
  35 │ ht_1x2_away              4.7         5.3               58.251                32.4636                 57.4985                  31.7338
  36 │ ht_ou_05_under           2.86        3.25             -23.7513               16.9474                -21.2952                  16.6385
  37 │ ht_ou_05_over            1.45        1.54               6.34242               8.5922                  5.0972                   8.43561
  38 │ ht_ou_15_under           1.4         1.47             -14.2724               11.1996                -13.753                   10.7449
  39 │ ht_ou_15_over            3.1         3.55              20.1747               24.799                  19.0245                  23.7922
  40 │ ht_ou_25_under           1.1         1.12              -7.27904               6.18428                -7.55979                  6.01891
  41 │ ht_ou_25_over            9.4        11.0               47.6573               52.8475                 50.0564                  51.4343
  42 │ ht_cs_0_0                2.98        3.2              -20.552                17.6585                -17.9929                  17.3366
  43 │ ht_cs_1_0                4.1         4.7              -35.0391               11.5756                -35.7489                  11.0369
  44 │ ht_cs_0_1                6.6         7.6               23.6144               20.2778                 21.5379                  19.1862
  45 │ ht_cs_1_1                8.8        13.5               -1.77006              11.6186                 -7.49013                 11.0996
  46 │ ht_cs_2_0                7.4      1000.0              -62.8815               13.7185                -62.57                    13.4279
  47 │ ht_cs_2_1                9.8      1000.0              -65.2428               12.2352                -66.6973                  11.2109
  48 │ ht_cs_2_2                2.9      1000.0              -96.1936                1.7578                -96.3154                   1.58301
  49 │ ht_cs_0_2               10.0      1000.0              -31.207                20.8062                -31.3905                  19.8561
  50 │ ht_cs_1_2               11.0      1000.0              -54.7424               13.3413                -56.8573                  11.9871
  51 │ ht_cs_other              9.8      1000.0              -33.9999               29.2176                -27.7488                  30.8579
--- Distribution Comparison for ssm_neg_bin ---
51×6 DataFrame
 Row │ market            market_back  market_lay  ssm_neg_bin_mean_odds  ssm_neg_bin_median_odds  ssm_neg_bin_market_quantile 
     │ String            Float64      Float64     Float64?               Float64?                 Float64?                    
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ ft_1x2_home              1.94        2.14                   2.36                     2.28                        0.788
   2 │ ft_1x2_draw              3.3         3.7                    3.96                     3.87                        0.966
   3 │ ft_1x2_away              4.1         4.8                    3.6                      3.38                        0.226
   4 │ ft_ou_05_under          10.0        13.5                   14.51                    12.86                        0.748
   5 │ ft_ou_05_over            1.08        1.11                   1.09                     1.08                        0.552
   6 │ ft_ou_15_under           3.2         3.65                   3.94                     3.64                        0.688
   7 │ ft_ou_15_over            1.38        1.42                   1.39                     1.38                        0.488
   8 │ ft_ou_25_under           1.76        1.91                   2.0                      1.91                        0.67
   9 │ ft_ou_25_over            2.1         2.32                   2.16                     2.1                         0.504
  10 │ ft_ou_35_under           1.29        1.34                   1.39                     1.35                        0.67
  11 │ ft_ou_35_over            3.85        4.5                    4.06                     3.84                        0.496
  12 │ ft_btts_yes              1.74        2.06                   2.01                     1.97                        0.828
  13 │ ft_btts_no               1.96        2.36                   2.08                     2.03                        0.562
  14 │ ft_cs_0_0                9.8        14.0                   14.51                    12.86                        0.766
  15 │ ft_cs_1_0                7.2         8.8                    9.82                     9.22                        0.888
  16 │ ft_cs_0_1               11.5        16.5                   12.82                    11.63                        0.514
  17 │ ft_cs_1_1                6.6         8.4                    8.68                     8.28                        1.0
  18 │ ft_cs_2_0               10.0        13.0                   13.73                    12.89                        0.896
  19 │ ft_cs_0_2               22.0        36.0                   23.14                    20.68                        0.434
  20 │ ft_cs_2_1                9.2        12.0                   12.14                    11.77                        1.0
  21 │ ft_cs_1_2               13.5        20.0                   15.64                    14.93                        0.708
  22 │ ft_cs_2_2               16.5        24.0                   21.87                    21.2                         0.918
  23 │ ft_cs_3_0               19.0        30.0                   29.73                    26.79                        0.834
  24 │ ft_cs_0_3                8.4       140.0                   64.01                    55.13                        1.0
  25 │ ft_cs_3_1               18.5        28.0                   26.28                    24.23                        0.822
  26 │ ft_cs_1_3               36.0        80.0                   43.16                    39.19                        0.614
  27 │ ft_cs_3_2               28.0       980.0                   47.33                    43.2                         0.91
  28 │ ft_cs_2_3               44.0       120.0                   60.26                    55.63                        0.738
  29 │ ft_cs_3_3                7.6       300.0                  130.31                   113.3                         1.0
  30 │ ft_cs_other_home         7.0        28.0                   21.18                    16.86                        0.938
  31 │ ft_cs_other_draw         6.8         0.0                 1368.47                   981.88                        1.0
  32 │ ft_cs_other_away         7.4        90.0                   44.47                    37.77                        1.0
  33 │ ht_1x2_home              2.66        2.94                   3.93                     3.75                        0.93
  34 │ ht_1x2_draw              2.16        2.36                   2.57                     2.52                        0.958
  35 │ ht_1x2_away              4.7         5.3                    3.11                     3.02                        0.026
  36 │ ht_ou_05_under           2.86        3.25                   3.81                     3.62                        0.9
  37 │ ht_ou_05_over            1.45        1.54                   1.39                     1.38                        0.286
  38 │ ht_ou_15_under           1.4         1.47                   1.65                     1.6                         0.908
  39 │ ht_ou_15_over            3.1         3.55                   2.71                     2.66                        0.23
  40 │ ht_ou_25_under           1.1         1.12                   1.2                      1.18                        0.932
  41 │ ht_ou_25_over            9.4        11.0                    7.04                     6.69                        0.15
  42 │ ht_cs_0_0                2.98        3.2                    3.81                     3.62                        0.852
  43 │ ht_cs_1_0                4.1         4.7                    6.59                     6.38                        1.0
  44 │ ht_cs_0_1                6.6         7.6                    5.58                     5.44                        0.12
  45 │ ht_cs_1_1                8.8        13.5                    9.66                     9.46                        0.714
  46 │ ht_cs_2_0                7.4      1000.0                   22.76                    20.57                        1.0
  47 │ ht_cs_2_1                9.8      1000.0                   33.39                    31.0                         1.0
  48 │ ht_cs_2_2                2.9      1000.0                   95.48                    84.69                        1.0
  49 │ ht_cs_0_2               10.0      1000.0                   15.93                    14.92                        0.928
  50 │ ht_cs_1_2               11.0      1000.0                   27.62                    26.25                        1.0
  51 │ ht_cs_other              9.8      1000.0                   16.06                    15.09                        0.822

Analyzing Match: queen-of-the-south vs alloa-athletic
result: yet to play 

--- Expected Value (EV) Analysis ---
50×7 DataFrame
 Row │ market            market_back  market_lay  ssm_poiss_mean_ev_pct  ssm_poiss_std_ev_pct  ssm_neg_bin_mean_ev_pct  ssm_neg_bin_std_ev_pct 
     │ String            Float64?     Float64?    Float64?               Float64?              Float64?                 Float64?               
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ ft_1x2_home              2.48        2.78               6.8253               22.1261                    6.67112               21.7559
   2 │ ft_1x2_draw              3.4         3.95             -15.9542                8.53452                 -16.1213                 8.58179
   3 │ ft_1x2_away              2.76        3.1              -11.1126               21.175                   -10.8063                20.8381
   4 │ ft_ou_05_under          13.5        18.0              -13.871                33.2809                  -10.4387                33.9833
   5 │ ft_ou_05_over            1.06        1.08              -0.762721              2.61316                  -1.03222                2.66832
   6 │ ft_ou_15_under           3.8         4.6              -10.78                 25.0754                   -8.94917               25.1653
   7 │ ft_ou_15_over            1.29        1.36              -1.28785               8.51244                  -1.90936                8.54296
   8 │ ft_ou_25_under           1.98        2.18              -7.26555              17.9689                   -6.49518               17.87
   9 │ ft_ou_25_over            1.84        2.02              -2.17747              16.6984                   -2.89337               16.6064
  10 │ ft_ou_35_under           1.41        1.47              -3.40298              12.0489                   -3.24926               11.964
  11 │ ft_ou_35_over            3.05        3.5               -3.951                26.0633                   -4.28351               25.8796
  12 │ ft_btts_yes              1.7         1.8               -5.48883              11.5931                   -6.69396               11.5981
  13 │ ft_btts_no               2.24        2.44              -0.532366             15.2756                    1.05558               15.2823
  14 │ ft_cs_0_0               14.0        18.0              -10.681                34.5135                   -7.12161               35.242
  15 │ ft_cs_1_0               10.0        12.5               -7.96673              22.0245                   -6.87931               21.9203
  16 │ ft_cs_0_1               11.0        13.5              -13.1473               26.8599                  -11.8404                26.6712
  17 │ ft_cs_1_1                7.2         9.4              -17.8314               10.0493                  -18.7742                 9.75564
  18 │ ft_cs_2_0               15.0        22.0                3.79742              23.9517                    3.81208               23.3727
  19 │ ft_cs_0_2               16.5        24.0              -17.0959               26.2023                  -16.4813                25.7086
  20 │ ft_cs_2_1               11.0        13.0               -5.52054              10.5682                   -7.63173               10.3374
  21 │ ft_cs_1_2               11.5        14.5              -16.3189               13.3294                  -17.8581                12.873
  22 │ ft_cs_2_2               14.0        17.5              -23.2481               12.0876                  -25.4565                11.8872
  23 │ ft_cs_3_0                5.7        48.0              -79.4028                7.48722                 -79.3344                 7.30222
  24 │ ft_cs_0_3                5.4        55.0              -88.1538                4.83931                 -87.9718                 4.763
  25 │ ft_cs_3_1               22.0        34.0               -1.32432              28.6772                   -3.20758               27.5796
  26 │ ft_cs_1_3                5.5        38.0              -82.4912                5.37627                 -82.674                  5.20785
  27 │ ft_cs_3_2                5.6       980.0              -83.9651                4.94214                 -84.371                  4.7522
  28 │ ft_cs_2_3               32.0       980.0              -23.16                 23.8854                  -24.7411                23.2363
  29 │ ft_cs_3_3                5.7        95.0              -92.8487                2.95241                 -92.9687                 2.8615
  30 │ ft_cs_other_home        18.0       980.0               33.2063               76.6399                   36.1961                77.1118
  31 │ ft_cs_other_draw         6.0         0.0              -98.8586                0.843863                -98.8205                 0.852509
  32 │ ft_cs_other_away        21.0       980.0              -11.1621               49.6282                   -8.27093               51.1775
  33 │ ht_1x2_home              3.1         3.55              -3.19994              23.0388                   -3.71457               21.822
  34 │ ht_1x2_draw              2.26        2.52              -5.2751               10.7006                   -4.80327               10.356
  35 │ ht_1x2_away              3.35        3.9              -10.0173               20.4811                  -10.1606                19.6856
  36 │ ht_ou_05_under           3.15        3.7               -3.83896              20.6324                   -1.36247               19.6031
  37 │ ht_ou_05_over            1.37        1.47              -4.82242               8.97344                  -5.8995                 8.52581
  38 │ ht_ou_15_under           1.47        1.59              -2.92995              11.6911                   -2.53887               11.031
  39 │ ht_ou_15_over            2.7         3.15              -8.29193              21.4735                   -9.01024               20.261
  40 │ ht_ou_25_under           1.12        1.16              -2.1486                5.66722                  -2.43203                5.46956
  41 │ ht_ou_25_over            7.2         8.8               -9.0447               36.4321                   -7.22268               35.1615
  42 │ ht_cs_0_0                3.15        3.6               -3.83896              20.6324                   -1.36247               19.6031
  43 │ ht_cs_1_0                4.9         5.8               -8.46079              13.8073                   -9.91107               13.0497
  44 │ ht_cs_0_1                5.3         6.4              -10.8261               16.397                   -12.0142                15.2988
  45 │ ht_cs_1_1                7.8         9.6              -19.3226               11.2259                  -23.7311                10.1253
  46 │ ht_cs_2_0                7.2        19.5              -55.9578               16.2681                  -55.9694                15.1138
  47 │ ht_cs_2_1                8.6      1000.0              -70.7591               10.848                   -71.9538                 9.65389
  48 │ ht_cs_2_2               10.5      1000.0              -89.5681                5.23511                 -89.7781                 4.7776
  49 │ ht_cs_0_2                7.6      1000.0              -62.9649               12.9717                  -62.6022                12.3901
  50 │ ht_cs_1_2                8.6      1000.0              -74.1165                8.83766                 -74.9992                 8.04106
--- Distribution Comparison for ssm_neg_bin ---
50×6 DataFrame
 Row │ market            market_back  market_lay  ssm_neg_bin_mean_odds  ssm_neg_bin_median_odds  ssm_neg_bin_market_quantile 
     │ String            Float64?     Float64?    Float64?               Float64?                 Float64?                    
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ ft_1x2_home              2.48        2.78                   2.43                     2.35                        0.398
   2 │ ft_1x2_draw              3.4         3.95                   4.1                      4.07                        0.962
   3 │ ft_1x2_away              2.76        3.1                    3.28                     3.11                        0.71
   4 │ ft_ou_05_under          13.5        18.0                   17.5                     15.75                        0.656
   5 │ ft_ou_05_over            1.06        1.08                   1.07                     1.07                        0.598
   6 │ ft_ou_15_under           3.8         4.6                    4.53                     4.21                        0.652
   7 │ ft_ou_15_over            1.29        1.36                   1.33                     1.31                        0.554
   8 │ ft_ou_25_under           1.98        2.18                   2.2                      2.11                        0.634
   9 │ ft_ou_25_over            1.84        2.02                   1.96                     1.9                         0.56
  10 │ ft_ou_35_under           1.41        1.47                   1.48                     1.45                        0.58
  11 │ ft_ou_35_over            3.05        3.5                    3.45                     3.24                        0.586
  12 │ ft_btts_yes              1.7         1.8                    1.85                     1.83                        0.712
  13 │ ft_btts_no               2.24        2.44                   2.27                     2.2                         0.474
  14 │ ft_cs_0_0               14.0        18.0                   17.5                     15.75                        0.632
  15 │ ft_cs_1_0               10.0        12.5                   11.43                    10.66                        0.62
  16 │ ft_cs_0_1               11.0        13.5                   13.77                    12.9                         0.678
  17 │ ft_cs_1_1                7.2         9.4                    9.01                     8.71                        1.0
  18 │ ft_cs_2_0               15.0        22.0                   15.29                    14.47                        0.438
  19 │ ft_cs_0_2               16.5        24.0                   21.99                    20.12                        0.738
  20 │ ft_cs_2_1               11.0        13.0                   12.08                    11.71                        0.73
  21 │ ft_cs_1_2               11.5        14.5                   14.39                    13.83                        0.922
  22 │ ft_cs_2_2               14.0        17.5                   19.33                    18.68                        1.0
  23 │ ft_cs_3_0                5.7        48.0                   31.48                    28.89                        1.0
  24 │ ft_cs_0_3                5.4        55.0                   53.3                     47.25                        1.0
  25 │ ft_cs_3_1               22.0        34.0                   24.93                    23.02                        0.562
  26 │ ft_cs_1_3                5.5        38.0                   34.91                    33.12                        1.0
  27 │ ft_cs_3_2                5.6       980.0                   40.0                     36.28                        1.0
  28 │ ft_cs_2_3               32.0       980.0                   46.96                    44.18                        0.862
  29 │ ft_cs_3_3                5.7        95.0                   97.47                    86.62                        1.0
  30 │ ft_cs_other_home        18.0       980.0                   18.55                    15.05                        0.388
  31 │ ft_cs_other_draw         6.0         0.0                  856.18                   632.37                        1.0
  32 │ ft_cs_other_away        21.0       980.0                   29.88                    26.52                        0.672
  33 │ ht_1x2_home              3.1         3.55                   3.39                     3.27                        0.606
  34 │ ht_1x2_draw              2.26        2.52                   2.4                      2.39                        0.66
  35 │ ht_1x2_away              3.35        3.9                    3.92                     3.77                        0.712
  36 │ ht_ou_05_under           3.15        3.7                    3.33                     3.21                        0.536
  37 │ ht_ou_05_over            1.37        1.47                   1.47                     1.45                        0.748
  38 │ ht_ou_15_under           1.47        1.59                   1.53                     1.5                         0.564
  39 │ ht_ou_15_over            2.7         3.15                   3.12                     2.98                        0.704
  40 │ ht_ou_25_under           1.12        1.16                   1.15                     1.14                        0.612
  41 │ ht_ou_25_over            7.2         8.8                    8.96                     8.1                         0.658
  42 │ ht_cs_0_0                3.15        3.6                    3.33                     3.21                        0.536
  43 │ ht_cs_1_0                4.9         5.8                    5.56                     5.43                        0.77
  44 │ ht_cs_0_1                5.3         6.4                    6.23                     6.02                        0.784
  45 │ ht_cs_1_1                7.8         9.6                   10.42                    10.22                        0.996
  46 │ ht_cs_2_0                7.2        19.5                   18.45                    16.92                        1.0
  47 │ ht_cs_2_1                8.6      1000.0                   34.71                    32.1                         1.0
  48 │ ht_cs_2_2               10.5      1000.0                  128.38                   113.04                        1.0
  49 │ ht_cs_0_2                7.6      1000.0                   22.85                    21.03                        1.0
  50 │ ht_cs_1_2                8.6      1000.0                   38.35                    35.59                        1.0







#### champ 




Analyzing Match: airdrieonians vs greenock-morton
results: ht: 0-0, ft: 1-2

--- Expected Value (EV) Analysis ---
51×7 DataFrame
 Row │ market            market_back  market_lay  ssm_poiss_mean_ev_pct  ssm_poiss_std_ev_pct  ssm_neg_bin_mean_ev_pct  ssm_neg_bin_std_ev_pct 
     │ String            Float64      Float64     Float64?               Float64?              Float64?                 Float64?               
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ ft_1x2_home              3.0         3.25               13.1897              25.8342                   13.5981                25.3529
   2 │ ft_1x2_draw              3.45        3.6                -7.22923              9.23589                  -7.66388                9.28251
   3 │ ft_1x2_away              2.5         2.6               -11.5502              20.0052                  -11.5759                19.601
   4 │ ft_ou_05_under           9.6        13.5               -14.8725              29.9379                  -13.4422                29.9452
   5 │ ft_ou_05_over            1.08        1.1                -1.57684              3.36802                  -1.73776                3.36883
   6 │ ft_ou_15_under           3.25        3.75               -3.12442             24.2717                   -2.65393               24.086
   7 │ ft_ou_15_over            1.37        1.44               -3.83678             10.2314                   -4.03511               10.1532
   8 │ ft_ou_25_under           1.78        1.93               -2.01548             16.3963                   -2.16051               16.3033
   9 │ ft_ou_25_over            2.08        2.3                -6.49876             19.1597                   -6.32929               19.051
  10 │ ft_ou_35_under           1.34        1.36                1.50763             10.4666                    1.121                 10.5346
  11 │ ft_ou_35_over            3.85        4.5                -6.64505             30.072                    -5.5342                30.2674
  12 │ ft_btts_yes              1.84        1.96               -9.17741             13.2424                   -9.78581               13.1472
  13 │ ft_btts_no               2.04        2.2                 3.30539             14.6818                    3.97992               14.5763
  14 │ ft_cs_0_0               11.5        16.0                 1.97564             35.8631                    3.68908               35.8718
  15 │ ft_cs_1_0               10.0        13.0                 6.06486             24.2174                    6.18016               23.8945
  16 │ ft_cs_0_1                9.2         9.8                -4.92778             25.325                    -5.07274               24.8706
  17 │ ft_cs_1_1                6.8         7.8               -15.8065               7.21789                 -17.2026                 7.24147
  18 │ ft_cs_2_0               16.0        24.0                 6.15042             27.6913                    6.20926               27.0268
  19 │ ft_cs_0_2               15.0        20.0                -6.64272             27.4927                   -7.01198               26.7905
  20 │ ft_cs_2_1               12.0        15.5                -6.93095             14.7551                   -8.49379               14.1785
  21 │ ft_cs_1_2               10.5        12.5               -21.596               12.1377                  -23.0698                11.6679
  22 │ ft_cs_2_2               15.5        18.0               -27.4009              15.3352                  -28.756                 14.7052
  23 │ ft_cs_3_0               32.0       980.0                -7.28847             39.7045                   -5.98069               38.7622
  24 │ ft_cs_0_3               28.0       980.0               -27.6355              30.5306                  -27.0006                29.9174
  25 │ ft_cs_3_1               28.0       980.0                -5.07548             34.7014                   -5.35708               33.4858
  26 │ ft_cs_1_3               22.0       980.0               -31.7181              22.3586                  -32.1356                21.6041
  27 │ ft_cs_3_2               16.5       980.0               -66.1803              13.3366                  -66.3197                12.9979
  28 │ ft_cs_2_3                8.8       980.0               -82.8446               6.26379                 -82.9388                 6.08518
  29 │ ft_cs_3_3                8.2       150.0               -92.9946               3.56873                 -92.9224                 3.54941
  30 │ ft_cs_other_home         8.4        42.0               -63.072               25.7924                  -61.1503                26.7797
  31 │ ft_cs_other_draw         8.4      1000.0               -99.1338               0.804428                -99.0587                 0.877489
  32 │ ft_cs_other_away        19.0       980.0               -29.3683              42.6792                  -25.9959                43.8748
  33 │ ht_1x2_home              1.04        4.9               -75.8218               6.34074                 -75.7754                 6.36732
  34 │ ht_1x2_draw              1.25     1000.0               -50.5198               5.41273                 -50.3112                 5.37496
  35 │ ht_1x2_away              1.25      980.0               -53.5405               9.01676                 -53.8049                 9.00279
  36 │ ht_ou_05_under           1.2       110.0               -66.7949               7.13112                 -65.8299                 6.95033
  37 │ ht_ou_05_over            1.01      110.0               -26.9476               6.00203                 -27.7598                 5.84986
  38 │ ht_ou_15_under           1.44        1.5                -9.89318             11.1937                   -9.51346               10.7383
  39 │ ht_ou_15_over            3.0      1000.0                12.2775              23.3202                   11.4864                22.3714
  40 │ ht_ou_25_under           1.1         1.13               -6.27192              5.86307                  -6.62999                5.70561
  41 │ ht_ou_25_over            1.1      1000.0               -83.7281               5.86307                 -83.37                   5.70561
  42 │ ht_cs_0_0                3.1      1000.0               -14.2201              18.4221                  -11.7273                17.955
  43 │ ht_cs_1_0                1.03     1000.0               -85.185                2.88206                 -85.3151                 2.82295
  44 │ ht_cs_0_1                1.03     1000.0               -78.8646               3.03826                 -79.2912                 2.92869
  45 │ ht_cs_1_1                1.03     1000.0               -88.9883               1.44036                 -89.623                  1.37193
  46 │ ht_cs_2_0                1.03     1000.0               -95.893                1.64501                 -95.8222                 1.64452
  47 │ ht_cs_2_1                1.03     1000.0               -96.9394               1.1502                  -97.0482                 1.07207
  48 │ ht_cs_2_2                1.03     1000.0               -98.8078               0.563926                -98.8385                 0.51751
  49 │ ht_cs_0_2                1.03     1000.0               -91.8066               2.36426                 -91.8495                 2.29533
  50 │ ht_cs_1_2                1.03     1000.0               -95.7224               1.23172                 -95.9182                 1.12848
  51 │ ht_cs_other              1.03     1000.0               -93.2939               3.06545                 -92.6234                 3.17447
--- Distribution Comparison for ssm_neg_bin ---
51×6 DataFrame
 Row │ market            market_back  market_lay  ssm_neg_bin_mean_odds  ssm_neg_bin_median_odds  ssm_neg_bin_market_quantile 
     │ String            Float64      Float64     Float64?               Float64?                 Float64?                    
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ ft_1x2_home              3.0         3.25                   2.78                     2.7                         0.306
   2 │ ft_1x2_draw              3.45        3.6                    3.78                     3.74                        0.794
   3 │ ft_1x2_away              2.5         2.6                    2.98                     2.88                        0.732
   4 │ ft_ou_05_under           9.6        13.5                   12.71                    11.58                        0.702
   5 │ ft_ou_05_over            1.08        1.1                    1.1                      1.09                        0.658
   6 │ ft_ou_15_under           3.25        3.75                   3.58                     3.38                        0.552
   7 │ ft_ou_15_over            1.37        1.44                   1.44                     1.42                        0.628
   8 │ ft_ou_25_under           1.78        1.93                   1.88                     1.81                        0.54
   9 │ ft_ou_25_over            2.08        2.3                    2.32                     2.23                        0.628
  10 │ ft_ou_35_under           1.34        1.36                   1.34                     1.31                        0.43
  11 │ ft_ou_35_over            3.85        4.5                    4.53                     4.2                         0.596
  12 │ ft_btts_yes              1.84        1.96                   2.08                     2.05                        0.768
  13 │ ft_btts_no               2.04        2.2                    2.0                      1.95                        0.402
  14 │ ft_cs_0_0               11.5        16.0                   12.71                    11.58                        0.508
  15 │ ft_cs_1_0               10.0        13.0                    9.98                     9.44                        0.398
  16 │ ft_cs_0_1                9.2         9.8                   10.53                     9.74                        0.576
  17 │ ft_cs_1_1                6.8         7.8                    8.29                     8.03                        1.0
  18 │ ft_cs_2_0               16.0        24.0                   16.2                     15.03                        0.42
  19 │ ft_cs_0_2               15.0        20.0                   17.75                    16.3                         0.632
  20 │ ft_cs_2_1               12.0        15.5                   13.47                    13.06                        0.674
  21 │ ft_cs_1_2               10.5        12.5                   14.0                     13.56                        0.992
  22 │ ft_cs_2_2               15.5        18.0                   22.81                    21.87                        0.978
  23 │ ft_cs_3_0               32.0       980.0                   40.72                    36.46                        0.62
  24 │ ft_cs_0_3               28.0       980.0                   45.71                    41.14                        0.848
  25 │ ft_cs_3_1               28.0       980.0                   33.89                    30.97                        0.612
  26 │ ft_cs_1_3               22.0       980.0                   36.1                     33.46                        0.904
  27 │ ft_cs_3_2               16.5       980.0                   57.45                    50.55                        1.0
  28 │ ft_cs_2_3                8.8       980.0                   58.92                    53.07                        1.0
  29 │ ft_cs_3_3                8.2       150.0                  148.78                   125.79                        1.0
  30 │ ft_cs_other_home         8.4        42.0                   32.98                    25.81                        0.964
  31 │ ft_cs_other_draw         8.4      1000.0                 1725.74                  1187.0                         1.0
  32 │ ft_cs_other_away        19.0       980.0                   35.27                    29.6                         0.804
  33 │ ht_1x2_home              1.04        4.9                    4.61                     4.36                        1.0
  34 │ ht_1x2_draw              1.25     1000.0                    2.55                     2.51                        1.0
  35 │ ht_1x2_away              1.25      980.0                    2.81                     2.74                        1.0
  36 │ ht_ou_05_under           1.2       110.0                    3.68                     3.53                        1.0
  37 │ ht_ou_05_over            1.01      110.0                    1.41                     1.4                         1.0
  38 │ ht_ou_15_under           1.44        1.5                    1.62                     1.58                        0.824
  39 │ ht_ou_15_over            3.0      1000.0                    2.8                      2.72                        0.322
  40 │ ht_ou_25_under           1.1         1.13                   1.18                     1.17                        0.916
  41 │ ht_ou_25_over            1.1      1000.0                    7.44                     6.9                         1.0
  42 │ ht_cs_0_0                3.1      1000.0                    3.68                     3.53                        0.756
  43 │ ht_cs_1_0                1.03     1000.0                    7.31                     7.03                        1.0
  44 │ ht_cs_0_1                1.03     1000.0                    5.08                     4.96                        1.0
  45 │ ht_cs_1_1                1.03     1000.0                   10.12                     9.89                        1.0
  46 │ ht_cs_2_0                1.03     1000.0                   29.08                    25.89                        1.0
  47 │ ht_cs_2_1                1.03     1000.0                   40.16                    36.52                        1.0
  48 │ ht_cs_2_2                1.03     1000.0                  108.58                    95.0                         1.0
  49 │ ht_cs_0_2                1.03     1000.0                   13.75                    12.96                        1.0
  50 │ ht_cs_1_2                1.03     1000.0                   27.36                    26.01                        1.0
  51 │ ht_cs_other              1.03     1000.0                   16.63                    15.03                        1.0


Analyzing Match: dunfermline-athletic vs queens-park-fc
result: ht: 0-0, ft:0-0

--- Expected Value (EV) Analysis ---
51×7 DataFrame
 Row │ market            market_back  market_lay  ssm_poiss_mean_ev_pct  ssm_poiss_std_ev_pct  ssm_neg_bin_mean_ev_pct  ssm_neg_bin_std_ev_pct 
     │ String            Float64      Float64     Float64?               Float64?              Float64?                 Float64?               
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ ft_1x2_home              1.97        2.02             -11.9875               16.6371               -12.3502                    16.3974
   2 │ ft_1x2_draw              3.4         3.6               -4.3735               10.7693                -4.41397                   10.5617
   3 │ ft_1x2_away              4.7         4.9               27.8311               30.7412                28.7519                    30.419
   4 │ ft_ou_05_under           8.2        12.0               -8.33206              28.9885                -6.40478                   28.695
   5 │ ft_ou_05_over            1.09        1.11              -3.18513               3.85335               -3.44132                    3.81434
   6 │ ft_ou_15_under           3.35        3.55              17.6437               25.7421                18.6458                    25.2126
   7 │ ft_ou_15_over            1.39        1.41              -9.81334              10.681                -10.2292                    10.4614
   8 │ ft_ou_25_under           1.74        3.75               6.6256               14.9501                 6.80601                   14.6013
   9 │ ft_ou_25_over            1.37        2.34             -46.9523               11.771                -47.0944                    11.4965
  10 │ ft_ou_35_under           1.31        1.34               5.7634                8.62968                5.64025                    8.46855
  11 │ ft_ou_35_over            4.0         4.3              -22.9417               26.3502               -22.5657                    25.8582
  12 │ ft_btts_yes              1.03     1000.0              -55.3174                6.79038              -55.7613                     6.65561
  13 │ ft_btts_no               1.03     1000.0              -41.6826                6.79038              -41.2387                     6.65561
  14 │ ft_cs_0_0               10.5        12.0               17.3797               37.1194                19.8475                    36.7436
  15 │ ft_cs_1_0                7.8         8.0                7.36881              19.4447                 7.25481                   18.9564
  16 │ ft_cs_0_1               13.5        17.5               37.339                35.7106                38.402                     35.1193
  17 │ ft_cs_1_1                7.2         8.0               -9.67749               6.75076              -10.964                      6.55144
  18 │ ft_cs_2_0               10.0        12.0              -11.6211               19.1274               -12.2966                    18.6936
  19 │ ft_cs_0_2               25.0       980.0               19.4723               37.6617                20.7411                    37.4643
  20 │ ft_cs_2_1                9.8        11.0              -20.9839               11.3193               -22.6463                    11.0097
  21 │ ft_cs_1_2               16.0        21.0               -5.55816              18.7949                -6.69383                   18.212
  22 │ ft_cs_2_2               17.5        21.0              -33.5216               15.9803               -34.8044                    15.3387
  23 │ ft_cs_3_0               20.0       980.0              -21.0827               30.2346               -21.102                     29.4289
  24 │ ft_cs_0_3                9.4       180.0              -85.4771                6.55619              -85.0645                     6.56499
  25 │ ft_cs_3_1               20.0        26.0              -27.9576               24.5855               -28.9714                    23.6683
  26 │ ft_cs_1_3                9.0        85.0              -82.7978                6.44894              -82.7204                     6.25908
  27 │ ft_cs_3_2               17.5        50.0              -70.2727               11.8085               -70.648                     11.3451
  28 │ ft_cs_2_3                9.0       140.0              -88.9144                4.45723              -88.9516                     4.30101
  29 │ ft_cs_3_3                9.6       200.0              -94.7071                2.75646              -94.6896                     2.67734
  30 │ ft_cs_other_home        13.0       980.0              -38.0719               40.2546               -36.2413                    40.3635
  31 │ ft_cs_other_draw        10.0      1000.0              -99.4848                0.448103             -99.4525                     0.45823
  32 │ ft_cs_other_away        10.0       570.0              -83.4474               10.2905               -82.5571                    10.3415
  33 │ ht_1x2_home              2.54        3.3              -33.2433               17.9557               -33.5401                    17.3075
  34 │ ht_1x2_draw              2.14        2.34              -4.08998              10.3179                -3.54488                   10.1824
  35 │ ht_1x2_away              3.7         5.9                6.93028              24.4276                 6.42016                   23.7274
  36 │ ht_ou_05_under           2.8         3.15              -2.80894              18.7354                -0.70587                   18.2071
  37 │ ht_ou_05_over            1.47        1.56              -4.02531               9.83611               -5.12942                    9.55873
  38 │ ht_ou_15_under           1.41        1.46              -0.199626             10.3054                 0.0198641                  9.91377
  39 │ ht_ou_15_over            3.15        3.65              -7.95828              23.0227                -8.44863                   22.1478
  40 │ ht_ou_25_under           1.09        1.12              -1.73563               4.59157               -2.02549                    4.51258
  41 │ ht_ou_25_over            9.6        11.5               -5.44767              40.4395                -2.89474                   39.7438
  42 │ ht_cs_0_0                2.82        3.15              -2.11472              18.8693                 0.00337347                18.3372
  43 │ ht_cs_1_0                4.0         4.5              -30.856                12.655                -31.9106                    12.0692
  44 │ ht_cs_0_1                6.8         8.0               27.7265               22.6093                25.4704                    21.3444
  45 │ ht_cs_1_1                9.2        11.0              -13.9145               14.4767               -18.4742                    13.5661
  46 │ ht_cs_2_0               12.5      1000.0              -41.8545               25.4156               -41.3758                    24.2146
  47 │ ht_cs_2_1                6.0      1000.0              -84.8771                6.57112              -85.3466                     6.02143
  48 │ ht_cs_2_2                6.8      1000.0              -95.1006                2.73829              -95.1404                     2.5584
  49 │ ht_cs_0_2                6.2      1000.0              -66.7579               12.0773               -66.5364                    11.6138
  50 │ ht_cs_1_2                6.4      1000.0              -82.899                 6.15374              -83.4204                     5.72264
  51 │ ht_cs_other              5.9      1000.0              -76.7765               12.2389               -74.2306                    12.8705
--- Distribution Comparison for ssm_neg_bin ---
51×6 DataFrame
 Row │ market            market_back  market_lay  ssm_neg_bin_mean_odds  ssm_neg_bin_median_odds  ssm_neg_bin_market_quantile 
     │ String            Float64      Float64     Float64?               Float64?                 Float64?                    
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ ft_1x2_home              1.97        2.02                   2.33                     2.27                        0.798
   2 │ ft_1x2_draw              3.4         3.6                    3.6                      3.55                        0.648
   3 │ ft_1x2_away              4.7         4.9                    3.88                     3.67                        0.164
   4 │ ft_ou_05_under           8.2        12.0                    9.7                      8.96                        0.61
   5 │ ft_ou_05_over            1.09        1.11                   1.13                     1.13                        0.81
   6 │ ft_ou_15_under           3.35        3.55                   2.97                     2.83                        0.246
   7 │ ft_ou_15_over            1.39        1.41                   1.57                     1.55                        0.83
   8 │ ft_ou_25_under           1.74        3.75                   1.66                     1.61                        0.314
   9 │ ft_ou_25_over            1.37        2.34                   2.72                     2.63                        1.0
  10 │ ft_ou_35_under           1.31        1.34                   1.25                     1.23                        0.242
  11 │ ft_ou_35_over            4.0         4.3                    5.79                     5.41                        0.812
  12 │ ft_btts_yes              1.03     1000.0                    2.38                     2.34                        1.0
  13 │ ft_btts_no               1.03     1000.0                    1.78                     1.75                        1.0
  14 │ ft_cs_0_0               10.5        12.0                    9.7                      8.96                        0.322
  15 │ ft_cs_1_0                7.8         8.0                    7.52                     7.25                        0.338
  16 │ ft_cs_0_1               13.5        17.5                   10.49                     9.92                        0.14
  17 │ ft_cs_1_1                7.2         8.0                    8.14                     7.96                        1.0
  18 │ ft_cs_2_0               10.0        12.0                   11.97                    11.47                        0.76
  19 │ ft_cs_0_2               25.0       980.0                   23.06                    21.22                        0.292
  20 │ ft_cs_2_1                9.8        11.0                   12.96                    12.53                        1.0
  21 │ ft_cs_1_2               16.0        21.0                   17.89                    17.18                        0.64
  22 │ ft_cs_2_2               17.5        21.0                   28.51                    27.06                        0.984
  23 │ ft_cs_3_0               20.0       980.0                   29.37                    26.81                        0.8
  24 │ ft_cs_0_3                9.4       180.0                   77.45                    67.07                        1.0
  25 │ ft_cs_3_1               20.0        26.0                   31.83                    28.68                        0.876
  26 │ ft_cs_1_3                9.0        85.0                   60.03                    54.54                        1.0
  27 │ ft_cs_3_2               17.5        50.0                   70.02                    62.87                        1.0
  28 │ ft_cs_2_3                9.0       140.0                   95.63                    86.06                        1.0
  29 │ ft_cs_3_3                9.6       200.0                  234.73                   199.8                         1.0
  30 │ ft_cs_other_home        13.0       980.0                   29.93                    23.8                         0.846
  31 │ ft_cs_other_draw        10.0      1000.0                 3432.63                  2430.9                         1.0
  32 │ ft_cs_other_away        10.0       570.0                   81.48                    64.21                        1.0
  33 │ ht_1x2_home              2.54        3.3                    4.1                      3.95                        0.952
  34 │ ht_1x2_draw              2.14        2.34                   2.24                     2.23                        0.642
  35 │ ht_1x2_away              3.7         5.9                    3.66                     3.52                        0.42
  36 │ ht_ou_05_under           2.8         3.15                   2.93                     2.83                        0.522
  37 │ ht_ou_05_over            1.47        1.56                   1.57                     1.55                        0.688
  38 │ ht_ou_15_under           1.41        1.46                   1.42                     1.4                         0.482
  39 │ ht_ou_15_over            3.15        3.65                   3.65                     3.49                        0.664
  40 │ ht_ou_25_under           1.09        1.12                   1.12                     1.11                        0.626
  41 │ ht_ou_25_over            9.6        11.5                   11.58                    10.49                        0.582
  42 │ ht_cs_0_0                2.82        3.15                   2.93                     2.83                        0.512
  43 │ ht_cs_1_0                4.0         4.5                    6.08                     5.88                        0.994
  44 │ ht_cs_0_1                6.8         8.0                    5.59                     5.39                        0.124
  45 │ ht_cs_1_1                9.2        11.0                   11.63                    11.17                        0.93
  46 │ ht_cs_2_0               12.5      1000.0                   25.51                    23.03                        0.938
  47 │ ht_cs_2_1                6.0      1000.0                   48.64                    43.17                        1.0
  48 │ ht_cs_2_2                6.8      1000.0                  182.58                   152.0                         1.0
  49 │ ht_cs_0_2                6.2      1000.0                   21.09                    19.18                        1.0
  50 │ ht_cs_1_2                6.4      1000.0                   43.76                    39.67                        1.0
  51 │ ht_cs_other              5.9      1000.0                   28.45                    25.46                        1.0

Analyzing Match: partick-thistle vs arbroath
result: ht: 1-0, ft: 1-1

--- Expected Value (EV) Analysis ---
51×7 DataFrame
 Row │ market            market_back  market_lay  ssm_poiss_mean_ev_pct  ssm_poiss_std_ev_pct  ssm_neg_bin_mean_ev_pct  ssm_neg_bin_std_ev_pct 
     │ String            Float64      Float64     Float64?               Float64?              Float64?                 Float64?               
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ ft_1x2_home              1.53        1.57               0.606811              15.4317                  0.210218                15.2195
   2 │ ft_1x2_draw              4.4         4.8              -17.1854                19.6232                -17.0417                  19.4247
   3 │ ft_1x2_away              6.6         7.6                1.70029               39.6274                  3.13505                 39.1166
   4 │ ft_ou_05_under          15.0        19.0              -31.7975                35.1617                -28.8452                  35.9898
   5 │ ft_ou_05_over            1.06        1.08               1.18036                2.48476                 0.971727                 2.54328
   6 │ ft_ou_15_under           4.2         4.6              -24.3154                29.4543                -22.6686                  29.4949
   7 │ ft_ou_15_over            1.28        1.32               4.93422                8.97655                 4.43235                  8.98894
   8 │ ft_ou_25_under           2.04        2.22             -21.4343                22.2065                -20.8026                  22.0
   9 │ ft_ou_25_over            1.83        1.97              12.522                 19.9205                 11.9553                  19.7353
  10 │ ft_ou_35_under           1.41        1.46             -15.6189                16.5086                -15.5947                  16.348
  11 │ ft_ou_35_over            3.1         3.45              24.4812                36.2955                 24.4281                  35.9425
  12 │ ft_btts_yes              1.94        2.08               4.65986               13.3266                  3.65505                 13.2476
  13 │ ft_btts_no               1.93        2.08             -11.1204                13.2579                -10.1207                  13.1793
  14 │ ft_cs_0_0               15.0        17.5              -31.7975                35.1617                -28.8452                  35.9898
  15 │ ft_cs_1_0                8.0         8.8              -25.9987                23.4706                -25.2811                  23.1911
  16 │ ft_cs_0_1               21.0        28.0              -11.314                 42.1439                 -9.09709                 41.8907
  17 │ ft_cs_1_1                9.4        10.0              -19.1919                21.2256                -19.7241                  20.5479
  18 │ ft_cs_2_0                9.0         9.4              -11.1849                17.1572                -11.8583                  16.6781
  19 │ ft_cs_0_2               46.0        60.0               -6.73548               47.812                  -4.4795                  47.3169
  20 │ ft_cs_2_1                9.2         9.6              -15.6189                 8.01384               -17.554                    8.03418
  21 │ ft_cs_1_2               23.0        27.0               -5.00777               30.7985                 -5.64131                 29.9822
  22 │ ft_cs_2_2               20.0        24.0              -11.8474                18.7159                -13.8555                  18.3551
  23 │ ft_cs_3_0               13.5        14.5               -0.649517              23.9942                 -1.73953                 23.0346
  24 │ ft_cs_0_3               13.0       230.0              -91.2778                 5.39741               -90.9389                   5.38826
  25 │ ft_cs_3_1               14.5        16.0               -0.86927               16.5489                 -3.46396                 15.7917
  26 │ ft_cs_1_3               15.0       110.0              -79.4815                 9.53778               -79.3237                   9.33473
  27 │ ft_cs_3_2               30.0        36.0               -1.45872               24.9475                 -4.03124                 23.9291
  28 │ ft_cs_2_3               14.5       150.0              -78.8242                 8.2617                -79.0078                   8.01314
  29 │ ft_cs_3_3               14.5       140.0              -84.2193                 6.55452               -84.4107                   6.31297
  30 │ ft_cs_other_home         7.4         8.4               47.8899                76.9455                 49.9744                  76.9293
  31 │ ft_cs_other_draw        16.5      1000.0              -96.9954                 2.23178               -96.8952                   2.24273
  32 │ ft_cs_other_away        12.5       130.0              -80.8854                12.3951                -80.0555                  12.3711
  33 │ ht_1x2_home              2.08        2.34             -25.5837                16.8373                -26.2405                  16.7379
  34 │ ht_1x2_draw              2.42        2.76              -4.4254                10.8142                 -3.83412                 11.0872
  35 │ ht_1x2_away              6.4         8.8               58.2675                40.2567                 58.7247                  39.3254
  36 │ ht_ou_05_under           3.25        3.7              -10.9842                19.6882                 -8.06657                 19.8112
  37 │ ht_ou_05_over            1.37        1.45              -0.52359                8.29934                -1.75348                  8.35117
  38 │ ht_ou_15_under           1.49        1.59              -7.34734               11.8126                 -6.80427                 11.6353
  39 │ ht_ou_15_over            2.7         3.05               2.10592               21.4054                  1.12183                 21.0841
  40 │ ht_ou_25_under           1.13        1.16              -4.02365                6.09455                -4.33621                  6.09645
  41 │ ht_ou_25_over            7.2         8.8                8.46927               38.8325                 10.4608                  38.8446
  42 │ ht_cs_0_0                3.25        3.7              -10.9842                19.6882                 -8.06657                 19.8112
  43 │ ht_cs_1_0                3.7         4.1              -27.1548                10.6107                -28.7001                  10.0613
  44 │ ht_cs_0_1                9.0        10.5               35.9507                28.9348                 34.9102                  27.7117
  45 │ ht_cs_1_1                9.4        11.0                1.89302               12.0402                 -4.04161                 11.8096
  46 │ ht_cs_2_0                8.6        10.5              -35.2444                21.462                 -35.9315                  20.7257
  47 │ ht_cs_2_1               19.0        26.0              -21.2611                25.493                 -25.0557                  23.7657
  48 │ ht_cs_2_2               13.5      1000.0              -83.8419                 7.1705                -84.2685                   6.68232
  49 │ ht_cs_0_2               11.5      1000.0              -49.602                 19.4335                -48.7952                  18.8967
  50 │ ht_cs_1_2               12.0      1000.0              -62.3721                12.0781                -63.7423                  11.1781
  51 │ ht_cs_other              8.6        18.5              -43.3392                25.9464                -37.9893                  27.8324
--- Distribution Comparison for ssm_neg_bin ---
51×6 DataFrame
 Row │ market            market_back  market_lay  ssm_neg_bin_mean_odds  ssm_neg_bin_median_odds  ssm_neg_bin_market_quantile 
     │ String            Float64      Float64     Float64?               Float64?                 Float64?                    
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ ft_1x2_home              1.53        1.57                   1.56                     1.52                        0.492
   2 │ ft_1x2_draw              4.4         4.8                    5.69                     5.23                        0.81
   3 │ ft_1x2_away              6.6         7.6                    7.64                     6.64                        0.506
   4 │ ft_ou_05_under          15.0        19.0                   28.51                    23.12                        0.804
   5 │ ft_ou_05_over            1.06        1.08                   1.05                     1.05                        0.28
   6 │ ft_ou_15_under           4.2         4.6                    6.49                     5.67                        0.788
   7 │ ft_ou_15_over            1.28        1.32                   1.24                     1.21                        0.276
   8 │ ft_ou_25_under           2.04        2.22                   2.84                     2.6                         0.822
   9 │ ft_ou_25_over            1.83        1.97                   1.69                     1.63                        0.264
  10 │ ft_ou_35_under           1.41        1.46                   1.75                     1.65                        0.826
  11 │ ft_ou_35_over            3.1         3.45                   2.73                     2.53                        0.258
  12 │ ft_btts_yes              1.94        2.08                   1.9                      1.87                        0.396
  13 │ ft_btts_no               1.93        2.08                   2.2                      2.15                        0.766
  14 │ ft_cs_0_0               15.0        17.5                   28.51                    23.12                        0.804
  15 │ ft_cs_1_0                8.0         8.8                   12.07                    10.78                        0.85
  16 │ ft_cs_0_1               21.0        28.0                   30.28                    24.45                        0.65
  17 │ ft_cs_1_1                9.4        10.0                   12.82                    11.51                        0.826
  18 │ ft_cs_2_0                9.0         9.4                   10.64                    10.15                        0.75
  19 │ ft_cs_0_2               46.0        60.0                   65.59                    51.37                        0.6
  20 │ ft_cs_2_1                9.2         9.6                   11.3                     10.88                        1.0
  21 │ ft_cs_1_2               23.0        27.0                   27.73                    24.67                        0.584
  22 │ ft_cs_2_2               20.0        24.0                   24.43                    23.23                        0.764
  23 │ ft_cs_3_0               13.5        14.5                   14.62                    13.72                        0.528
  24 │ ft_cs_0_3               13.0       230.0                  217.25                   164.51                        1.0
  25 │ ft_cs_3_1               14.5        16.0                   15.54                    14.51                        0.5
  26 │ ft_cs_1_3               15.0       110.0                   91.6                     79.24                        1.0
  27 │ ft_cs_3_2               30.0        36.0                   33.59                    31.53                        0.56
  28 │ ft_cs_2_3               14.5       150.0                   80.58                    73.24                        1.0
  29 │ ft_cs_3_3               14.5       140.0                  110.77                    99.36                        1.0
  30 │ ft_cs_other_home         7.4         8.4                    6.55                     5.48                        0.276
  31 │ ft_cs_other_draw        16.5      1000.0                  875.15                   643.66                        1.0
  32 │ ft_cs_other_away        12.5       130.0                   91.65                    74.05                        1.0
  33 │ ht_1x2_home              2.08        2.34                   2.98                     2.85                        0.93
  34 │ ht_1x2_draw              2.42        2.76                   2.55                     2.52                        0.64
  35 │ ht_1x2_away              6.4         8.8                    4.29                     4.12                        0.046
  36 │ ht_ou_05_under           3.25        3.7                    3.72                     3.55                        0.666
  37 │ ht_ou_05_over            1.37        1.45                   1.4                      1.39                        0.584
  38 │ ht_ou_15_under           1.49        1.59                   1.63                     1.58                        0.696
  39 │ ht_ou_15_over            2.7         3.05                   2.79                     2.71                        0.516
  40 │ ht_ou_25_under           1.13        1.16                   1.19                     1.17                        0.736
  41 │ ht_ou_25_over            7.2         8.8                    7.41                     6.84                        0.448
  42 │ ht_cs_0_0                3.25        3.7                    3.72                     3.55                        0.666
  43 │ ht_cs_1_0                3.7         4.1                    5.3                      5.13                        0.998
  44 │ ht_cs_0_1                9.0        10.5                    6.98                     6.71                        0.104
  45 │ ht_cs_1_1                9.4        11.0                    9.96                     9.72                        0.614
  46 │ ht_cs_2_0                8.6        10.5                   15.14                    13.72                        0.954
  47 │ ht_cs_2_1               19.0        26.0                   28.35                    26.31                        0.834
  48 │ ht_cs_2_2               13.5      1000.0                  104.26                    92.09                        1.0
  49 │ ht_cs_0_2               11.5      1000.0                   25.72                    23.56                        0.982
  50 │ ht_cs_1_2               12.0      1000.0                   36.67                    33.8                         1.0
  51 │ ht_cs_other              8.6        18.5                   16.87                    15.21                        0.892

Analyzing Match: ross-county vs raith-rovers
results: ht: 0-0, ft: 2-0

--- Expected Value (EV) Analysis ---
51×7 DataFrame
 Row │ market            market_back  market_lay  ssm_poiss_mean_ev_pct  ssm_poiss_std_ev_pct  ssm_neg_bin_mean_ev_pct  ssm_neg_bin_std_ev_pct 
     │ String            Float64      Float64     Float64?               Float64?              Float64?                 Float64?               
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ ft_1x2_home              2.9         2.96               2.1997               24.8167                   2.09328                24.5775
   2 │ ft_1x2_draw              3.3         3.45             -11.8249                8.04612                -12.0089                  8.2004
   3 │ ft_1x2_away              2.76        2.82               4.98733              23.4016                   5.24217                23.3289
   4 │ ft_ou_05_under          10.0        12.5              -13.191                29.5322                 -10.5908                 30.2163
   5 │ ft_ou_05_over            1.09        1.12              -0.462178              3.21901                 -0.745607                3.29357
   6 │ ft_ou_15_under           3.2         3.5               -5.91789              22.8078                  -4.65078                23.001
   7 │ ft_ou_15_over            1.4         1.43              -1.16092               9.97841                 -1.71528                10.0629
   8 │ ft_ou_25_under           1.8         1.84              -1.71894              15.9268                  -1.34408                15.9806
   9 │ ft_ou_25_over            2.2         2.24              -0.1213               19.4661                  -0.579455               19.5318
  10 │ ft_ou_35_under           1.29        1.32              -2.70286               9.72347                 -2.77928                 9.79101
  11 │ ft_ou_35_over            4.1         4.5                0.761041             30.9041                   1.00391                31.1187
  12 │ ft_btts_yes              1.87        2.02              -7.11979              13.3184                  -8.23586                13.4267
  13 │ ft_btts_no               1.99        2.16               0.159559             14.1731                   1.34725                14.2883
  14 │ ft_cs_0_0               10.0        13.0              -13.191                29.5322                 -10.5908                 30.2163
  15 │ ft_cs_1_0                9.6        12.0               -3.33814              23.0851                  -2.7565                 23.1907
  16 │ ft_cs_0_1                9.6        11.0                2.24785              26.5608                   2.97129                26.4475
  17 │ ft_cs_1_1                7.2         7.8              -11.2002                7.21209                -12.592                   7.20562
  18 │ ft_cs_2_0               16.0        21.0               -2.28859              28.6649                  -2.40034                28.3055
  19 │ ft_cs_0_2               16.0        21.0                8.15458              31.4413                   8.35571                31.18
  20 │ ft_cs_2_1               11.5        15.0              -14.1152               14.9282                 -16.0186                 14.6034
  21 │ ft_cs_1_2               11.5        14.5              -10.0766               13.084                  -11.9084                 12.7762
  22 │ ft_cs_2_2               16.0        21.0              -24.3195               15.8243                 -26.3076                 15.584
  23 │ ft_cs_3_0               32.0        60.0              -17.4821               37.2243                 -16.872                  36.6393
  24 │ ft_cs_0_3               30.0        60.0              -11.2842               36.1595                 -10.137                  36.2496
  25 │ ft_cs_3_1               25.0        44.0              -21.3031               28.964                  -22.3658                 28.1231
  26 │ ft_cs_1_3               25.0        44.0              -14.5011               27.2253                 -15.3088                 26.6808
  27 │ ft_cs_3_2                6.8        85.0              -86.4594                5.18787                -86.6906                  5.08051
  28 │ ft_cs_2_3                6.2        60.0              -87.1774                4.6033                 -87.3685                  4.53922
  29 │ ft_cs_3_3                6.4       140.0              -94.4304                2.74242                -94.4577                  2.72277
  30 │ ft_cs_other_home        12.5        44.0              -51.5088               32.2233                 -49.8505                 32.6096
  31 │ ft_cs_other_draw         6.8      1000.0              -99.2843                0.632476               -99.2431                  0.654241
  32 │ ft_cs_other_away         6.4        44.0              -72.0725               16.7541                 -70.8656                 17.6368
  33 │ ht_1x2_home              3.45        3.95             -34.0273               21.0103                 -33.153                  20.7765
  34 │ ht_1x2_draw              2.1         2.3               -9.98781               9.92383                 -9.4887                  9.70121
  35 │ ht_1x2_away              3.45        3.9               31.1501               26.8935                  29.4558                 26.1893
  36 │ ht_ou_05_under           2.76        3.1               -8.67812              16.8397                  -6.86329                16.3111
  37 │ ht_ou_05_over            1.48        1.57              -0.969705              9.02998                 -1.94288                 8.74655
  38 │ ht_ou_15_under           1.39        1.45              -3.93534               9.59899                 -3.90458                 9.2824
  39 │ ht_ou_15_over            3.25        3.7                0.388387             22.4437                   0.316472               21.7034
  40 │ ht_ou_25_under           1.09        1.12              -2.69256               4.42393                 -3.12509                 4.43922
  41 │ ht_ou_25_over           10.0        12.0                7.2712               40.5865                  11.2393                 40.7268
  42 │ ht_cs_0_0                2.98        3.1               -1.39884              18.182                    0.560655               17.6113
  43 │ ht_cs_1_0                5.1         5.9              -32.3643               16.3505                 -32.4089                 15.8243
  44 │ ht_cs_0_1                5.0         5.8               13.8086               15.9887                  10.6752                 15.4486
  45 │ ht_cs_1_1                8.6        10.5              -22.1487               13.6573                 -25.8507                 12.6546
  46 │ ht_cs_2_0                8.6      1000.0              -75.1099               12.8418                 -74.072                  12.739
  47 │ ht_cs_2_1               10.0      1000.0              -80.3949                9.16785                -80.5574                  8.64146
  48 │ ht_cs_2_2               12.5      1000.0              -91.3058                4.57029                -91.2477                  4.36314
  49 │ ht_cs_0_2                8.6      1000.0              -29.3515               22.8192                 -30.2903                 21.6164
  50 │ ht_cs_1_2               10.0      1000.0              -67.6135               10.0281                 -68.7063                  9.23631
  51 │ ht_cs_other              9.8      1000.0              -52.6422               23.2102                 -47.5689                 24.6192
--- Distribution Comparison for ssm_neg_bin ---
51×6 DataFrame
 Row │ market            market_back  market_lay  ssm_neg_bin_mean_odds  ssm_neg_bin_median_odds  ssm_neg_bin_market_quantile 
     │ String            Float64      Float64     Float64?               Float64?                 Float64?                    
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ ft_1x2_home              2.9         2.96                   3.02                     2.85                        0.476
   2 │ ft_1x2_draw              3.3         3.45                   3.78                     3.76                        0.922
   3 │ ft_1x2_away              2.76        2.82                   2.77                     2.65                        0.43
   4 │ ft_ou_05_under          10.0        12.5                   12.71                    11.66                        0.684
   5 │ ft_ou_05_over            1.09        1.12                   1.1                      1.09                        0.544
   6 │ ft_ou_15_under           3.2         3.5                    3.58                     3.4                         0.592
   7 │ ft_ou_15_over            1.4         1.43                   1.44                     1.42                        0.548
   8 │ ft_ou_25_under           1.8         1.84                   1.88                     1.82                        0.522
   9 │ ft_ou_25_over            2.2         2.24                   2.3                      2.22                        0.522
  10 │ ft_ou_35_under           1.29        1.32                   1.34                     1.31                        0.58
  11 │ ft_ou_35_over            4.1         4.5                    4.48                     4.18                        0.524
  12 │ ft_btts_yes              1.87        2.02                   2.08                     2.04                        0.746
  13 │ ft_btts_no               1.99        2.16                   2.01                     1.96                        0.468
  14 │ ft_cs_0_0               10.0        13.0                   12.71                    11.66                        0.684
  15 │ ft_cs_1_0                9.6        12.0                   10.53                     9.97                        0.57
  16 │ ft_cs_0_1                9.6        11.0                   10.01                     9.46                        0.478
  17 │ ft_cs_1_1                7.2         7.8                    8.3                      8.1                         1.0
  18 │ ft_cs_2_0               16.0        21.0                   18.0                     16.62                        0.552
  19 │ ft_cs_0_2               16.0        21.0                   16.14                    15.08                        0.41
  20 │ ft_cs_2_1               11.5        15.0                   14.2                     13.56                        0.856
  21 │ ft_cs_1_2               11.5        14.5                   13.38                    12.91                        0.824
  22 │ ft_cs_2_2               16.0        21.0                   22.86                    21.62                        0.954
  23 │ ft_cs_3_0               32.0        60.0                   47.84                    40.96                        0.732
  24 │ ft_cs_0_3               30.0        60.0                   39.94                    35.63                        0.652
  25 │ ft_cs_3_1               25.0        44.0                   37.69                    33.81                        0.79
  26 │ ft_cs_1_3               25.0        44.0                   33.08                    30.01                        0.746
  27 │ ft_cs_3_2                6.8        85.0                   60.57                    53.11                        1.0
  28 │ ft_cs_2_3                6.2        60.0                   56.43                    50.84                        1.0
  29 │ ft_cs_3_3                6.4       140.0                  149.25                   125.03                        1.0
  30 │ ft_cs_other_home        12.5        44.0                   39.39                    29.47                        0.926
  31 │ ft_cs_other_draw         6.8      1000.0                 1740.48                  1176.87                        1.0
  32 │ ft_cs_other_away         6.4        44.0                   31.1                     25.1                         0.99
  33 │ ht_1x2_home              3.45        3.95                   5.72                     5.26                        0.938
  34 │ ht_1x2_draw              2.1         2.3                    2.35                     2.3                         0.85
  35 │ ht_1x2_away              3.45        3.9                    2.78                     2.67                        0.126
  36 │ ht_ou_05_under           2.76        3.1                    3.06                     2.94                        0.65
  37 │ ht_ou_05_over            1.48        1.57                   1.52                     1.51                        0.606
  38 │ ht_ou_15_under           1.39        1.45                   1.46                     1.43                        0.62
  39 │ ht_ou_15_over            3.25        3.7                    3.4                      3.31                        0.538
  40 │ ht_ou_25_under           1.09        1.12                   1.13                     1.12                        0.746
  41 │ ht_ou_25_over           10.0        12.0                   10.24                     9.59                        0.46
  42 │ ht_cs_0_0                2.98        3.1                    3.06                     2.94                        0.478
  43 │ ht_cs_1_0                5.1         5.9                    8.03                     7.6                         0.98
  44 │ ht_cs_0_1                5.0         5.8                    4.61                     4.52                        0.246
  45 │ ht_cs_1_1                8.6        10.5                   11.97                    11.58                        0.986
  46 │ ht_cs_2_0                8.6      1000.0                   43.09                    35.82                        1.0
  47 │ ht_cs_2_1               10.0      1000.0                   63.47                    54.8                         1.0
  48 │ ht_cs_2_2               12.5      1000.0                  183.71                   158.99                        1.0
  49 │ ht_cs_0_2                8.6      1000.0                   13.59                    12.51                        0.924
  50 │ ht_cs_1_2               10.0      1000.0                   34.92                    33.42                        1.0
  51 │ ht_cs_other              9.8      1000.0                   22.62                    21.14                        0.944


Analyzing Match: st-johnstone vs ayr-united
results: ht: 0-0, ft: 0-0

--- Expected Value (EV) Analysis ---
51×7 DataFrame
 Row │ market            market_back  market_lay  ssm_poiss_mean_ev_pct  ssm_poiss_std_ev_pct  ssm_neg_bin_mean_ev_pct  ssm_neg_bin_std_ev_pct 
     │ String            Float64      Float64     Float64?               Float64?              Float64?                 Float64?               
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ ft_1x2_home              1.71        1.72             -27.1284               15.8229                 -27.1513                 15.6458
   2 │ ft_1x2_draw              4.1         4.3                8.42845              11.951                    8.12063                12.0185
   3 │ ft_1x2_away              5.4         5.7               67.0701               42.4252                  67.5466                 41.7517
   4 │ ft_ou_05_under          11.5        18.0               -0.516422             36.9761                   1.88092                37.2721
   5 │ ft_ou_05_over            1.06        1.08              -3.16979               3.40823                 -3.39076                 3.43552
   6 │ ft_ou_15_under           3.8         4.3               11.1069               29.3121                  12.2122                 29.079
   7 │ ft_ou_15_over            1.31        1.34              -7.30263              10.105                   -7.68368                10.0246
   8 │ ft_ou_25_under           2.02        2.16               9.67337              19.3297                   9.88762                19.0325
   9 │ ft_ou_25_over            1.87        1.97             -14.5293               17.8944                 -14.7276                 17.6192
  10 │ ft_ou_35_under           1.38        1.45               3.60903              11.293                    3.44084                11.1262
  11 │ ft_ou_35_over            3.2         3.75             -20.2528               26.1868                 -19.8628                 25.8
  12 │ ft_btts_yes              1.87        2.0               -7.49021              13.8401                  -8.40182                13.6394
  13 │ ft_btts_no               2.0         2.16               1.05905              14.8022                   2.03403                14.5876
  14 │ ft_cs_0_0               14.0        18.0               21.1104               45.0143                  24.029                  45.3747
  15 │ ft_cs_1_0                7.8        30.0              -12.5115               20.3795                 -12.2971                 19.8777
  16 │ ft_cs_0_1               13.5        29.0               26.5143               37.3103                  27.2558                 36.8787
  17 │ ft_cs_1_1                8.2       980.0               -0.249792              9.73921                 -1.74107                 9.41958
  18 │ ft_cs_2_0                9.4        60.0              -28.3313               17.7988                 -28.5375                 17.3974
  19 │ ft_cs_0_2                8.6       980.0              -54.7114               15.3355                 -54.5482                 15.0367
  20 │ ft_cs_2_1                8.8       980.0              -27.145                10.1543                 -28.6233                  9.91589
  21 │ ft_cs_1_2                7.4        27.0              -49.3784                9.42425                -50.2407                  9.12947
  22 │ ft_cs_2_2                7.0       980.0              -67.3758                7.045                  -68.1121                  6.80027
  23 │ ft_cs_3_0                6.6       980.0              -76.0933                9.661                  -75.9407                  9.53342
  24 │ ft_cs_0_3                4.4       170.0              -91.0095                4.17137                -90.8623                  4.10888
  25 │ ft_cs_3_1                6.6       980.0              -74.0116                8.8894                 -74.3177                  8.57652
  26 │ ft_cs_1_3                9.2        80.0              -75.569                 8.65724                -75.6827                  8.38256
  27 │ ft_cs_3_2                8.0       980.0              -82.2442                6.88607                -82.503                   6.57236
  28 │ ft_cs_2_3                9.2        80.0              -83.3383                6.25091                -83.5129                  6.0265
  29 │ ft_cs_3_3                9.8       150.0              -91.5354                4.32612                -91.558                   4.15535
  30 │ ft_cs_other_home         5.8       980.0              -66.8482               22.2529                 -65.6611                 22.6444
  31 │ ft_cs_other_draw         3.45        0.0              -99.6335                0.334885               -99.6135                  0.337275
  32 │ ft_cs_other_away         9.6       110.0              -71.6731               16.8224                 -70.4373                 17.0518
  33 │ ht_1x2_home              2.26        2.52             -41.7337               16.9109                 -41.6612                 16.387
  34 │ ht_1x2_draw              2.34        2.56              -7.66372               9.86861                 -7.16765                10.0839
  35 │ ht_1x2_away              5.5         6.4               91.1717               40.3109                  89.8293                 38.8952
  36 │ ht_ou_05_under           3.15        3.6              -14.1493               19.0646                 -11.4588                 19.0188
  37 │ ht_ou_05_over            1.38        1.46               0.389214              8.35209                 -0.789488                8.33204
  38 │ ht_ou_15_under           1.47        1.57              -8.85754              11.7584                  -8.38931                11.5595
  39 │ ht_ou_15_over            2.78        3.15               5.63534              22.237                    4.74986                21.8608
  40 │ ht_ou_25_under           1.12        1.14              -5.01738               6.16084                 -5.36707                 6.20317
  41 │ ht_ou_25_over            7.6         9.0               15.4751               41.8057                  17.848                  42.0929
  42 │ ht_cs_0_0                3.2         3.6              -12.7866               19.3672                 -10.0534                 19.3207
  43 │ ht_cs_1_0                3.85        4.4              -41.0665               11.4615                 -41.6286                 10.8785
  44 │ ht_cs_0_1                7.6         9.0               47.7446               26.3315                  44.7836                 24.7606
  45 │ ht_cs_1_1                8.8        10.5               -4.07539              12.4288                  -9.57392                11.8594
  46 │ ht_cs_2_0               10.0        12.0              -53.2949               20.4151                 -52.8205                 19.4463
  47 │ ht_cs_2_1                9.0      1000.0              -70.1065               12.2264                 -71.2428                 11.2253
  48 │ ht_cs_2_2               12.0      1000.0              -85.2094                7.21546                -85.5539                  6.75859
  49 │ ht_cs_0_2               10.5      1000.0              -24.0514               23.7178                 -24.6116                 22.4061
  50 │ ht_cs_1_2               10.5      1000.0              -57.4692               12.0356                 -59.3527                 11.176
  51 │ ht_cs_other              8.2      1000.0              -45.9661               23.424                  -40.6645                 25.655
--- Distribution Comparison for ssm_neg_bin ---
51×6 DataFrame
 Row │ market            market_back  market_lay  ssm_neg_bin_mean_odds  ssm_neg_bin_median_odds  ssm_neg_bin_market_quantile 
     │ String            Float64      Float64     Float64?               Float64?                 Float64?                    
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ ft_1x2_home              1.71        1.72                   2.47                     2.35                        0.954
   2 │ ft_1x2_draw              4.1         4.3                    3.84                     3.79                        0.23
   3 │ ft_1x2_away              5.4         5.7                    3.46                     3.26                        0.04
   4 │ ft_ou_05_under          11.5        18.0                   13.01                    11.86                        0.52
   5 │ ft_ou_05_over            1.06        1.08                   1.1                      1.09                        0.834
   6 │ ft_ou_15_under           3.8         4.3                    3.65                     3.44                        0.352
   7 │ ft_ou_15_over            1.31        1.34                   1.44                     1.41                        0.782
   8 │ ft_ou_25_under           2.02        2.16                   1.9                      1.83                        0.296
   9 │ ft_ou_25_over            1.87        1.97                   2.3                      2.21                        0.814
  10 │ ft_ou_35_under           1.38        1.45                   1.35                     1.32                        0.344
  11 │ ft_ou_35_over            3.2         3.75                   4.47                     4.15                        0.804
  12 │ ft_btts_yes              1.87        2.0                    2.09                     2.04                        0.73
  13 │ ft_btts_no               2.0         2.16                   2.0                      1.96                        0.462
  14 │ ft_cs_0_0               14.0        18.0                   13.01                    11.86                        0.326
  15 │ ft_cs_1_0                7.8        30.0                    9.39                     9.03                        0.746
  16 │ ft_cs_0_1               13.5        29.0                   11.68                    10.53                        0.24
  17 │ ft_cs_1_1                8.2       980.0                    8.44                     8.15                        0.468
  18 │ ft_cs_2_0                9.4        60.0                   14.03                    13.31                        0.95
  19 │ ft_cs_0_2                8.6       980.0                   21.4                     19.61                        0.998
  20 │ ft_cs_2_1                8.8       980.0                   12.62                    12.09                        1.0
  21 │ ft_cs_1_2                7.4        27.0                   15.46                    14.77                        1.0
  22 │ ft_cs_2_2                7.0       980.0                   23.15                    21.84                        1.0
  23 │ ft_cs_3_0                6.6       980.0                   32.6                     28.84                        1.0
  24 │ ft_cs_0_3                4.4       170.0                   60.18                    51.59                        1.0
  25 │ ft_cs_3_1                6.6       980.0                   29.33                    27.15                        1.0
  26 │ ft_cs_1_3                9.2        80.0                   43.47                    38.8                         1.0
  27 │ ft_cs_3_2                8.0       980.0                   53.87                    48.58                        1.0
  28 │ ft_cs_2_3                9.2        80.0                   65.15                    58.13                        1.0
  29 │ ft_cs_3_3                9.8       150.0                  151.9                    126.63                        1.0
  30 │ ft_cs_other_home         5.8       980.0                   26.19                    20.5                         0.982
  31 │ ft_cs_other_draw         3.45        0.0                 1802.55                  1150.92                        1.0
  32 │ ft_cs_other_away         9.6       110.0                   46.92                    38.35                        0.998
  33 │ ht_1x2_home              2.26        2.52                   4.2                      4.03                        0.988
  34 │ ht_1x2_draw              2.34        2.56                   2.55                     2.52                        0.754
  35 │ ht_1x2_away              5.5         6.4                    3.03                     2.93                        0.0
  36 │ ht_ou_05_under           3.15        3.6                    3.74                     3.56                        0.712
  37 │ ht_ou_05_over            1.38        1.46                   1.4                      1.39                        0.532
  38 │ ht_ou_15_under           1.47        1.57                   1.63                     1.59                        0.756
  39 │ ht_ou_15_over            2.78        3.15                   2.78                     2.69                        0.44
  40 │ ht_ou_25_under           1.12        1.14                   1.19                     1.17                        0.82
  41 │ ht_ou_25_over            7.6         9.0                    7.33                     6.82                        0.386
  42 │ ht_cs_0_0                3.2         3.6                    3.74                     3.56                        0.694
  43 │ ht_cs_1_0                3.85        4.4                    6.84                     6.57                        1.0
  44 │ ht_cs_0_1                7.6         9.0                    5.42                     5.17                        0.04
  45 │ ht_cs_1_1                8.8        10.5                    9.92                     9.63                        0.78
  46 │ ht_cs_2_0               10.0        12.0                   25.44                    23.09                        0.988
  47 │ ht_cs_2_1                9.0      1000.0                   36.89                    33.78                        1.0
  48 │ ht_cs_2_2               12.0      1000.0                  104.22                    90.25                        1.0
  49 │ ht_cs_0_2               10.5      1000.0                   15.3                     14.4                         0.838
  50 │ ht_cs_1_2               10.5      1000.0                   28.0                     26.47                        1.0
  51 │ ht_cs_other              8.2      1000.0                   16.53                    15.26                        0.94







"""

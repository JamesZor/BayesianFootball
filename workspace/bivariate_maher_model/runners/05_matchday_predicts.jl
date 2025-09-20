using BayesianFootball
using DataFrames
using Dates
using Statistics, StatsBase, StatsPlots, Distributions

include("/home/james/bet_project/models_julia/workspace/bivariate_maher_model/setup.jl")
include("/home/james/bet_project/models_julia/workspace/bivariate_maher_model/prediction.jl")
include("/home/james/bet_project/models_julia/workspace/bivariate_maher_model/analysis_funcs.jl")
include("/home/james/bet_project/models_julia/workspace/bivariate_maher_model/match_day_utils.jl")
using .BivariateMaher
using .BivariatePrediction
using .Analysis
using .MatchDayUtils 




## --- 1. Define Models and Match ---
all_model_paths = Dict(
    "maher_2526" => "/home/james/bet_project/models_julia/experiments/model_comparison/maher_basic_seasons_2526_20250919-200800",
    "bivar_2526" => "/home/james/bet_project/models_julia/experiments/model_comparison/maher_bivariate_seasons_2526_20250919-200835",
    "maher_24_26" => "/home/james/bet_project/models_julia/experiments/model_comparison/maher_basic_seasons_2425_2526_20250919-202508",
    "bivar_24_26" => "/home/james/bet_project/models_julia/experiments/model_comparison/maher_bivariate_seasons_2425_2526_20250919-204350"
)

model_2526_paths = Dict(
    "maher_2526" => "/home/james/bet_project/models_julia/experiments/model_comparison/maher_basic_seasons_2526_20250919-200800",
    "bivar_2526" => "/home/james/bet_project/models_julia/experiments/model_comparison/maher_bivariate_seasons_2526_20250919-200835",
)

model_24_26_paths = Dict(
    "maher_24_26" => "/home/james/bet_project/models_julia/experiments/model_comparison/maher_basic_seasons_2425_2526_20250919-202508",
    "bivar_24_26" => "/home/james/bet_project/models_julia/experiments/model_comparison/maher_bivariate_seasons_2425_2526_20250919-204350"
)

loaded_models_all = load_models_from_paths(all_model_paths)
loaded_models_2526 = load_models_from_paths(model_2526_paths)
loaded_models_24_26 = load_models_from_paths(model_24_26_paths)


#### testing 
# --- Step 1: Setup Paths and Define Market Structure ---
# Path to your python CLI tool
CLI_PATH = "/home/james/bet_project/whatstheodds"

# Define the master list of markets we want to analyze. 
# The order here is important as it defines the structure of our matrices.
# NOTE: You would expand this list to include all markets of interest (e.g., :ft_ou_25_over, :ft_btts_yes, etc.)
MARKET_LIST = [
    :ft_1x2_home,
    :ft_1x2_draw,
    :ft_1x2_away
]

println("✅ Setup complete.")
# --- Step 2: Get Today's Matches ---
println("\nFetching today's matches for England...")
todays_matches = get_todays_matches(["england", "scotland"]; cli_path=CLI_PATH)

# Let's assume the fixture list contains "Liverpool v Everton"
display(todays_matches)
MATCH_OF_INTEREST = "Liverpool v Everton"
MATCH_OF_INTEREST = "Cardiff v Bradford"
LEAGUE_ID = 1 # Assuming Premier League ID for your model

# --- Step 3: Get Live Market Odds for the Selected Match ---
println("\nFetching live odds for: $MATCH_OF_INTEREST")
market_book = get_live_market_odds(MATCH_OF_INTEREST, MARKET_LIST; cli_path=CLI_PATH)

println("MarketBook created:")
println("Markets: ", market_book.markets)
println("Back Odds: ", market_book.back_odds)
println("Lay Odds: ", market_book.lay_odds)

# --- Step 4: Adapt Model Predictions to the PredictionMatrix Format ---
# This helper function bridges your existing prediction logic with our new tensor format.
function generate_prediction_matrix(model, home_team, away_team, league_id, market_list)
    # Generate predictions using your existing function 
    preds_dict = generate_predictions(
        Dict{String, Any}("temp" => model), 
        home_team, 
        away_team, 
        league_id
    )
    preds = preds_dict["temp"]
    
    # Get MCMC sample size from one of the prediction vectors
    num_samples = length(preds.ft.home)
    market_map = Dict(m => i for (i, m) in enumerate(market_list))
    num_markets = length(market_list)
    
    # Initialize the probability matrix
    prob_matrix = Matrix{Float64}(undef, num_samples, num_markets)

    # Populate the matrix based on the market list
    for (market, idx) in market_map
        if market == :ft_1x2_home
            prob_matrix[:, idx] = preds.ft.home
        elseif market == :ft_1x2_draw
            prob_matrix[:, idx] = preds.ft.draw
        elseif market == :ft_1x2_away
            prob_matrix[:, idx] = preds.ft.away
        else
            prob_matrix[:, idx] .= NaN # Mark unsupported markets as NaN
        end
    end
    
    return PredictionMatrix(market_list, market_map, prob_matrix)
end



println("\n✅ PredictionMatrix helper function defined.")

# --- Step 5: Generate Predictions and Calculate EV for a Set of Models ---
println("\nGenerating predictions and EV distributions for models trained on 24/25-25/26 seasons...")

# Use one of your loaded model sets
models_to_run = loaded_models_24_26 

all_ev_dists = Dict{String, EVDistribution}()
all_pred_matrices = Dict{String, PredictionMatrix}()

for (model_name, model) in models_to_run
    println("Processing model: $model_name...")
    
    # Generate the prediction matrix for the match of interest
    pred_matrix = generate_prediction_matrix(
        model, 
        "liverpool", # Assuming model uses 'liverpool'
        "everton",   # Assuming model uses 'everton'
        LEAGUE_ID,
        MARKET_LIST
    )
    all_pred_matrices[model_name] = pred_matrix

    # Calculate the EV distribution against the live market book
    ev_dist = calculate_ev_distributions(pred_matrix, market_book)
    all_ev_dists[model_name] = ev_dist
end

println("\n✅ EV calculation complete for all models.")

# --- Step 6: Visualize the Results ---
println("\nGenerating visualizations...")

# Example 1: Plot a single model's odds distribution vs. the market spread
# Let's inspect the bivariate model
model_to_plot = "bivar_24_26"
p1 = plot_market_distribution_vs_odds(
    all_pred_matrices[model_to_plot], 
    market_book, 
    :ft_1x2_home # We are interested in the home win market
)
title!(p1, "Odds Dist. ($model_to_plot) vs Market for Home Win")
display(p1)

model_to_plot = "maher_24_26"
p1 = plot_market_distribution_vs_odds(
    all_pred_matrices[model_to_plot], 
    market_book, 
    :ft_1x2_home # We are interested in the home win market
)
title!(p1, "Odds Dist. ($model_to_plot) vs Market for Home Win")
display(p1)

# --- Plotting Multiple Odds Distributions on One Graph ---

# 1. Define the models you want to compare on this plot
model1_name = "bivar_24_26"
model2_name = "maher_24_26"
market_to_plot = :ft_1x2_home

# 2. Create the initial plot using the first model
# This sets up the axes, title, and market lines
p1 = plot_market_distribution_vs_odds(
    all_pred_matrices[model1_name], 
    market_book, 
    market_to_plot
)

# We need to manually rename the first model's label for clarity in the legend
p1[1][1][:label] = model1_name # Update the label of the first series

# 3. Extract the odds distribution for the second model
pred_matrix_2 = all_pred_matrices[model2_name]
idx_2 = pred_matrix_2.market_map[market_to_plot]
model2_odds_dist = 1 ./ pred_matrix_2.probabilities[:, idx_2]

# 4. Add the second model's distribution to the existing plot `p1` using `density!`
density!(p1, model2_odds_dist, label=model2_name)

# 5. Finalize and display the combined plot
title!(p1, "Odds Dist. Comparison vs Market for Home Win")
display(p1)

###
# Example 2: Plot and compare the EV distributions for all models
p2 = plot_ev_distributions(
    all_ev_dists, 
    :ft_1x2_home # Compare EV for the home win market across models
)
title!(p2, "EV Comparison for Home Win $MATCH_OF_INTEREST")
display(p2)



###### other match 
#
todays_matches = get_todays_matches(["scotland"]; cli_path=CLI_PATH)
tm = filter(row -> row.time=="14:00", todays_matches)

m1_event = first(tm[1, [:event_name]])

market_book = get_live_market_odds(m1_event, MARKET_LIST; cli_path=CLI_PATH)
println("MarketBook created:")
println("Markets: ", market_book.markets)
println("Back Odds: ", market_book.back_odds)
println("Lay Odds: ", market_book.lay_odds)
"""
julia> println("Markets: ", market_book.markets)
Markets: [:ft_1x2_home, :ft_1x2_draw, :ft_1x2_away]

julia> println("Back Odds: ", market_book.back_odds)
Back Odds: [4.6, 3.95, 1.72]

julia> println("Lay Odds: ", market_book.lay_odds)
Lay Odds: [5.3, 4.5, 1.84]

"""

all_ev_dists = Dict{String, EVDistribution}()
all_pred_matrices = Dict{String, PredictionMatrix}()

for (model_name, model) in models_to_run
    println("Processing model: $model_name...")
    
    # Generate the prediction matrix for the match of interest
    pred_matrix = generate_prediction_matrix(
        model, 
        "liverpool", # Assuming model uses 'liverpool'
        "everton",   # Assuming model uses 'everton'
        LEAGUE_ID,
        MARKET_LIST
    )
    all_pred_matrices[model_name] = pred_matrix

    # Calculate the EV distribution against the live market book
    ev_dist = calculate_ev_distributions(pred_matrix, market_book)
    all_ev_dists[model_name] = ev_dist
end

m1_event

p2 = plot_ev_distributions(
    all_ev_dists, 
    :ft_1x2_home # Compare EV for the home win market across models
)
title!(p2, "EV Comparison for Home Win $MATCH_OF_INTEREST")
display(p2)







# --- Script to Create a Summary EV DataFrame ---

todays_matches = get_todays_matches(["scotland", "england"]; cli_path=CLI_PATH)
tm = filter(row -> row.time=="14:00", todays_matches)

# 1. Select the specific model you want to use for this analysis
MODEL_TO_ANALYZE_NAME = "bivar_24_26"
model_to_run = models_to_run[MODEL_TO_ANALYZE_NAME] # models_to_run is from your previous setup

# 2. Initialize an empty array to store the results for each market
results_list = []

println("Starting analysis for $(nrow(tm)) matches...")

# 3. Loop through each match in the filtered DataFrame `tm`
for row in eachrow(tm)
    event = row.event_name
    home_team_model = row.home_team
    away_team_model = row.away_team

    println("Processing: $event")
    
    # --- Get Live Market Odds ---
    # Note: A try-catch block is good practice in case the market for a match isn't available
    try
        market_book = get_live_market_odds(event, MARKET_LIST; cli_path=CLI_PATH)
        
        # --- Generate Model Predictions ---
        pred_matrix = generate_prediction_matrix(
            model_to_run,
            home_team_model,
            away_team_model,
            # NOTE: Assuming a single league ID. This would need to be dynamic
            # if your DataFrame contains matches from multiple leagues.
            LEAGUE_ID, 
            MARKET_LIST
        )
        
        # --- Calculate Mean Model Odds and Mean EV ---
        # Mean of probabilities across all MCMC samples (dim 1)
        mean_probs = mean(pred_matrix.probabilities, dims=1)
        # Convert mean probabilities to mean odds
        mean_model_odds = 1 ./ mean_probs' # Transpose to make it a column vector
        
        # Calculate the full EV distribution
        ev_dist = calculate_ev_distributions(pred_matrix, market_book)
        # Calculate the mean of the EV distribution
        mean_evs = mean(ev_dist.ev, dims=1)' # Transpose
        
        # --- Append results for each market to our list ---
        for i in 1:length(MARKET_LIST)
            market_name = MARKET_LIST[i]
            
            # Skip if market odds weren't found
            isnan(market_book.back_odds[i]) && continue
            
            push!(results_list, (
                event_name = event,
                market = market_name,
                market_odds = market_book.back_odds[i],
                model_odds = round(mean_model_odds[i], digits=2),
                mean_ev = round(mean_evs[i] * 100, digits=2) # As a percentage
            ))
        end
        
    catch e
        @error "Could not process match: $event. Error: $e"
    end
end

# 4. Convert the list of results into a DataFrame and sort it
summary_df = DataFrame(results_list)
sort!(summary_df, :mean_ev, rev=true)

println("\n✅ Analysis complete.")
display(summary_df)

home = filter(row -> row.market==:ft_1x2_home, summary_df)

""" 
julia> home = filter(row -> row.market==:ft_1x2_home, summary_df)
35×5 DataFrame
 Row │ event_name                     market       market_odds  model_odds  mean_ev 
     │ String                         Symbol       Float64      Float64     Float64 
─────┼──────────────────────────────────────────────────────────────────────────────
   1 │ Blackburn v Ipswich            ft_1x2_home         4.1         1.85   121.14
   2 │ Colchester v Bristol Rovers    ft_1x2_home         3.4         1.87    82.3
   3 │ Burnley v Nottm Forest         ft_1x2_home         3.45        1.92    79.88
   4 │ Queen of South v Inverness CT  ft_1x2_home         4.0         2.3     73.82
   5 │ Hull v Southampton             ft_1x2_home         3.1         1.83    69.23
   6 │ Newport County v Gillingham    ft_1x2_home         4.4         3.05    44.22
   7 │ Notts Co v Crawley Town        ft_1x2_home         2.12        1.57    35.37
   8 │ Airdrieonians v Raith          ft_1x2_home         4.2         3.21    30.94
   9 │ Barrow v Crewe                 ft_1x2_home         3.3         2.58    28.13
  10 │ Cheltenham v Oldham            ft_1x2_home         3.6         2.82    27.56
  11 │ Bromley v Chesterfield         ft_1x2_home         2.84        2.35    20.62
  12 │ Rotherham v Stockport          ft_1x2_home         4.0         3.34    19.63
  13 │ Walsall v Tranmere             ft_1x2_home         2.0         1.73    15.56
  14 │ Wycombe v Northampton          ft_1x2_home         1.97        1.73    14.12
  15 │ Brighton v Tottenham           ft_1x2_home         2.28        2.03    12.13
  16 │ Port Vale v Mansfield          ft_1x2_home         2.18        1.96    11.49
  17 │ Salford City v Swindon         ft_1x2_home         2.42        2.26     6.85
  18 │ Derby v Preston                ft_1x2_home         2.48        2.36     4.96
  19 │ Reading v Leyton Orient        ft_1x2_home         2.52        2.44     3.43
  20 │ Arbroath v Morton              ft_1x2_home         2.6         2.53     2.64
  21 │ West Ham v Crystal Palace      ft_1x2_home         3.2         3.28    -2.54
  22 │ Aberdeen v Motherwell          ft_1x2_home         1.81        1.91    -5.3
  23 │ Huddersfield v Burton Albion   ft_1x2_home         1.75        1.88    -7.14
  24 │ Alloa v Cove Rangers           ft_1x2_home         2.16        2.43   -11.28
  25 │ Dundee v Livingston            ft_1x2_home         3.1         3.57   -13.21
  26 │ Doncaster v AFC Wimbledon      ft_1x2_home         1.99        2.4    -16.97
  27 │ Portsmouth v Sheff Wed         ft_1x2_home         1.7         2.12   -19.85
  28 │ Plymouth v Peterborough        ft_1x2_home         1.97        2.47   -20.16
  29 │ Stevenage v Exeter             ft_1x2_home         1.75        2.31   -24.33
  30 │ Wolves v Leeds                 ft_1x2_home         3.1         4.19   -25.93
  31 │ Sheff Utd v Charlton           ft_1x2_home         1.84        2.61   -29.62
  32 │ Ross Co v Queens Park          ft_1x2_home         1.63        2.32   -29.85
  33 │ MK Dons v Accrington           ft_1x2_home         1.47        2.13   -31.0
  34 │ Norwich v Wrexham              ft_1x2_home         2.04        2.99   -31.75
  35 │ Cardiff v Bradford             ft_1x2_home         1.8         3.25   -44.54
"""




"""
I need a util function in MatchDayUtils to help with the following: 

i have a python package i made to get the odds and matchs from betfair market api service. 
here: 
(webscraper) ⚡➜ whatstheodds (! main) pwd                                             
/home/james/bet_project/whatstheodds

note we need to be in the correct conda envirmoent: webscrape 

need a functions to get the matchs for today via the cli tool i made 
(webscraper) ⚡➜ whatstheodds (! main) python live_odds_cli.py list -f england scotland 
Loaded 99 teams from '/home/james/bet_project/whatstheodds/mappings/england.json'
Loaded 27 teams from '/home/james/bet_project/whatstheodds/mappings/scotland.json'

Finding today's soccer matches...
Found 56 matches:
  - 11:30 | Blackpool v Barnsley
  - 11:30 | Bolton v Wigan
  - 11:30 | Birmingham v Swansea
  - 11:30 | Leicester v Coventry
  - 11:30 | QPR v Stoke
  - 11:30 | Lincoln v Luton
  - 11:30 | Brackley Town v Sutton Utd
  - 11:30 | Cambridge Utd v Fleetwood Town
  - 11:30 | Harrogate Town v Shrewsbury
  - 11:30 | Liverpool v Everton
  - 14:00 | Rotherham v Stockport
  - 14:00 | Huddersfield v Burton Albion
  - 14:00 | Cardiff v Bradford
  - 14:00 | Reading v Leyton Orient
  - 14:00 | Hull v Southampton
  - 14:00 | Plymouth v Peterborough
  - 14:00 | Port Vale v Mansfield

if possible can run this python script via julia functions and create a DataFrames of the results, 
columns: time | event name | home_team | away_team | 

where we need to do a reverse json look up for the home_team, away_team, since our model use a diffeerent name structure, we can use the json files 
at 
Loaded 99 teams from '/home/james/bet_project/whatstheodds/mappings/england.json'
Loaded 27 teams from '/home/james/bet_project/whatstheodds/mappings/scotland.json'
example of the files 
{
  "celtic": "Celtic",
  "rangers": "Rangers", 
  "heart-of-midlothian": "Hearts",
  "falkirk-fc": "Falkirk",
  "hibernian": "Hibernian",
.. 

{
  "accrington-stanley": "Accrington",
  "afc-wimbledon": "AFC Wimbledon",
  "arsenal": "Arsenal",
  "aston-villa": "Aston Villa",
  "barnsley": "Barnsley",
  "barrow": "Barrow",
  "birmingham-city": "Birmingham",
  "blackburn-rovers": "Blackburn",
...


Following this, i want to compare the models to that of the odds, in order to do this we can call the python functions 
to get the live odds in a dict format, thus we need a julia funcitons in MatchDayUtils to call this python 
(webscraper) ⚡➜ whatstheodds (! main) python live_odds_cli.py odds "Rangers v Hibernian" -d 
Searching for event: 'Rangers v Hibernian'
Found Event: Rangers v Hibernian, ID: 34684751
Found 12 markets. Fetching odds individually...
{
  "ft": {
    "Correct Score": {
      "0 - 0": {
        "back": {
          "price": 26.0,
          "size": 29.77
        },
        "lay": {
          "price": 32.0,
          "size": 15.45
        }
      },
      "0 - 1": {
        "back": {
          "price": 21.0,
          "size": 13.06
        },
        "lay": {
          "price": 30.0,
          "size": 10.86
        }
      },
      "0 - 2": {
        "back": {
          "price": 29.0,
          "size": 10.74
        },
        "lay": {
          "price": 36.0,
          "size": 10.0
        }
      },
      "0 - 3": {
        "back": {
          "price": 14.5,
          "size": 12.73
        },
        "lay": {
          "price": 85.0,
          "size": 12.12
        }
      },
      "1 - 0": {
        "back": {
          "price": 14.0,
          "size": 12.09
        },
        "lay": {
          "price": 15.5,
          "size": 13.07
        }
      },
      "1 - 1": {
        "back": {
          "price": 10.5,
          "size": 20.42
        },
        "lay": {
          "price": 12.5,
          "size": 39.92
        }
      },
      "1 - 2": {
        "back": {
          "price": 15.5,
          "size": 20.91
        },
        "lay": {
          "price": 17.0,
          "size": 18.21
        }
      },
      "1 - 3": {
        "back": {
          "price": 20.0,
          "size": 11.99
        },
        "lay": {
          "price": 38.0,
          "size": 33.23
        }
      },
      "2 - 0": {
        "back": {
          "price": 14.5,
          "size": 19.11
        },
        "lay": {
          "price": 16.0,
          "size": 31.86
        }
      },
      "2 - 1": {
        "back": {
          "price": 10.0,
          "size": 60.66
        },
        "lay": {
          "price": 11.0,
          "size": 56.14
        }
      },
      "2 - 2": {
        "back": {
          "price": 14.0,
          "size": 29.89
        },
        "lay": {
          "price": 16.0,
          "size": 59.15
        }
      },
      "2 - 3": {
        "back": {
          "price": 26.0,
          "size": 10.78
        },
        "lay": {
          "price": 38.0,
          "size": 10.04
        }
      },
      "3 - 0": {
        "back": {
          "price": 20.0,
          "size": 15.63
        },
        "lay": {
          "price": 26.0,
          "size": 38.36
        }
      },
      "3 - 1": {
        "back": {
          "price": 14.5,
          "size": 18.89
        },
        "lay": {
          "price": 16.5,
          "size": 17.14
        }
      },
      "3 - 2": {
        "back": {
          "price": 15.0,
          "size": 12.13
        },
        "lay": {
          "price": 23.0,
          "size": 30.44
        }
      },
      "3 - 3": {
        "back": {
          "price": 32.0,
          "size": 13.21
        },
        "lay": {
          "price": 55.0,
          "size": 24.57
        }
      },
      "Any Other Home Win": {
        "back": {
          "price": 6.6,
          "size": 75.86
        },
        "lay": {
          "price": 7.4,
          "size": 24.36
        }
      },
      "Any Other Away Win": {
        "back": {
          "price": 20.0,
          "size": 17.61
        },
        "lay": {
          "price": 24.0,
          "size": 27.59
        }
      },
      "Any Other Draw": {
        "back": {
          "price": 15.0,
          "size": 14.6
        },
        "lay": {
          "price": 1000.0,
          "size": 7.44
        }
      }
    },
    "Over/Under 0.5 Goals": {
      "Under 0.5 Goals": {
        "back": {
          "price": 8.2,
          "size": 12.18
        },
        "lay": {
          "price": 36.0,
          "size": 58.06
        }
      },
      "Over 0.5 Goals": {
        "back": {
          "price": 1.03,
          "size": 1697.29
        },
        "lay": {
          "price": 1.04,
          "size": 61.0
        }
      }
    },
    "Over/Under 1.5 Goals": {
      "Under 1.5 Goals": {
        "back": {
          "price": 6.8,
          "size": 11.33
        },
        "lay": {
          "price": 7.2,
          "size": 24.78
        }
      },
      "Over 1.5 Goals": {
        "back": {
          "price": 1.16,
          "size": 315.84
        },
        "lay": {
          "price": 1.17,
          "size": 31.0
        }
      }
    },
    "Over/Under 2.5 Goals": {
      "Under 2.5 Goals": {
        "back": {
          "price": 2.96,
          "size": 16.14
        },
        "lay": {
          "price": 3.1,
          "size": 14.61
        }
      },
      "Over 2.5 Goals": {
        "back": {
          "price": 1.47,
          "size": 172.93
        },
        "lay": {
          "price": 1.51,
          "size": 23.87
        }
      }
    },
    "Match Odds": {
      "Rangers": {
        "back": {
          "price": 1.86,
          "size": 23.0
        },
        "lay": {
          "price": 1.89,
          "size": 12.76
        }
      },
      "Hibernian": {
        "back": {
          "price": 3.8,
          "size": 10.79
        },
        "lay": {
          "price": 4.1,
          "size": 101.46
        }
      },
      "The Draw": {
        "back": {
          "price": 4.6,
          "size": 12.66
        },
        "lay": {
          "price": 4.8,
          "size": 10.0
        }
      }
    },
    "Both teams to Score?": {
      "Yes": {
        "back": {
          "price": 1.53,
          "size": 90.36
        },
        "lay": {
          "price": 1.61,
          "size": 69.03
        }
      },
      "No": {
        "back": {
          "price": 2.66,
          "size": 41.78
        },
        "lay": {
          "price": 2.86,
          "size": 47.81
        }
      }
    },
    "Over/Under 3.5 Goals": {
      "Under 3.5 Goals": {
        "back": {
          "price": 1.79,
          "size": 13.21
        },
        "lay": {
          "price": 1.85,
          "size": 36.21
        }
      },
      "Over 3.5 Goals": {
        "back": {
          "price": 2.18,
          "size": 18.0
        },
        "lay": {
          "price": 2.28,
          "size": 19.39
        }
      }
    }
  },
  "ht": {
    "First Half Goals 1.5": {
      "Under 1.5 Goals": {
        "back": {
          "price": 1.79,
          "size": 42.91
        },
        "lay": {
          "price": 1.84,
          "size": 20.0
        }
      },
      "Over 1.5 Goals": {
        "back": {
          "price": 2.18,
          "size": 40.88
        },
        "lay": {
          "price": 2.28,
          "size": 55.83
        }
      }
    },
    "First Half Goals 0.5": {
      "Under 0.5 Goals": {
        "back": {
          "price": 4.5,
          "size": 49.82
        },
        "lay": {
          "price": 4.9,
          "size": 29.51
        }
      },
      "Over 0.5 Goals": {
        "back": {
          "price": 1.26,
          "size": 72.0
        },
        "lay": {
          "price": 1.28,
          "size": 111.85
        }
      }
    },
    "First Half Goals 2.5": {
      "Under 2.5 Goals": {
        "back": {
          "price": 1.22,
          "size": 475.59
        },
        "lay": {
          "price": 1.25,
          "size": 154.8
        }
      },
      "Over 2.5 Goals": {
        "back": {
          "price": 5.1,
          "size": 10.0
        },
        "lay": {
          "price": 5.4,
          "size": 37.84
        }
      }
    },
    "Half Time": {
      "Rangers": {
        "back": {
          "price": 2.34,
          "size": 10.0
        },
        "lay": {
          "price": 2.46,
          "size": 17.0
        }
      },
      "Hibernian": {
        "back": {
          "price": 4.2,
          "size": 36.02
        },
        "lay": {
          "price": 4.6,
          "size": 41.21
        }
      },
      "The Draw": {
        "back": {
          "price": 2.74,
          "size": 11.0
        },
        "lay": {
          "price": 2.88,
          "size": 12.0
        }
      }
    },
    "Half Time Score": {
      "0 - 0": {
        "back": {
          "price": 4.4,
          "size": 37.96
        },
        "lay": {
          "price": 4.9,
          "size": 55.32
        }
      },
      "1 - 1": {
        "back": {
          "price": 7.6,
          "size": 14.94
        },
        "lay": {
          "price": 8.6,
          "size": 18.66
        }
      },
      "2 - 2": {
        "back": {
          "price": 13.5,
          "size": 16.31
        },
        "lay": {
          "price": 65.0,
          "size": 11.33
        }
      },
      "1 - 0": {
        "back": {
          "price": 4.7,
          "size": 38.93
        },
        "lay": {
          "price": 5.2,
          "size": 37.4
        }
      },
      "2 - 0": {
        "back": {
          "price": 10.5,
          "size": 16.84
        },
        "lay": {
          "price": 11.5,
          "size": 15.79
        }
      },
      "2 - 1": {
        "back": {
          "price": 17.0,
          "size": 10.0
        },
        "lay": {
          "price": 19.0,
          "size": 12.44
        }
      },
      "0 - 1": {
        "back": {
          "price": 7.4,
          "size": 18.28
        },
        "lay": {
          "price": 8.2,
          "size": 27.34
        }
      },
      "0 - 2": {
        "back": {
          "price": 11.0,
          "size": 10.18
        },
        "lay": {
          "price": 28.0,
          "size": 24.67
        }
      },
      "1 - 2": {
        "back": {
          "price": 25.0,
          "size": 10.83
        },
        "lay": {
          "price": 29.0,
          "size": 10.98
        }
      },
      "Any unquoted": {
        "back": {
          "price": 11.5,
          "size": 15.23
        },
        "lay": {
          "price": 12.0,
          "size": 13.5
        }
      }
    }
  }
}
Logged out.

then it would be great to have a function to generate the models predictions like 
predictions = generate_predictions(loaded_models, team_name_home, team_name_away, match_league_id)
odds_df = create_odds_dataframe(predictions)
but extend this so we can have the model predict displayed to the market odds. 
I assume in a datafram,
and if possible display the EV compared to the models and the market per lines of bets .


alos since we are using mcmc chains for or bayesain model, it would be great to have a plot function of the densisty of the market lines, 
with the market back and lay odds marked with a vertical line ( and leageu mention the quantile of the back and lay odds to the model chain ) 



"""

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


####





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

#=
Im needing to develop a new type of pregame model, as i have reach the limit of correct score model.
I'm looking at implementing a different style pregame model - funnel model, since i have found 
more data. As before the data from the data_store csv are scraped from sofascrape which do not 
give any match Statistics for the lower scottish leagues one and two. 
I have found a better data, which im needing to mix into this, in order to get the data for the funnel model. 

So im looking at developing this model in a development side, so the process can use some of what has been 
developed in the package bayesainfootball, without out too much refactoring, im hoping to be able to create 
some type dispatch functions to allow me to follow the same process. 

this is were i need your help, can you go over the project and help me orginise a plan of how to do this, 

i sort of start with some funcitons to process the data. 

i just need, to beable to generate a feature_set from this new data, as we have done before. 


as the normal process is like 

# ----------------------------------------------
# 1. The set up 
# ----------------------------------------------

using Revise
using BayesianFootball

using DataFrames
using Statistics
using ThreadPinning
using LinearAlgebra
pinthreads(:cores)



using Turing

data_store = BayesianFootball.Data.load_default_datastore()

ds = BayesianFootball.Data.DataStore( 
    Data.add_match_week_column(data_store.matches),
    data_store.odds,
    data_store.incidents
)


# ----------------------------------------------
# 2. Experiment Configs Set up 
# ----------------------------------------------

# --- setup 1 
cv_config = BayesianFootball.Data.CVConfig(
    # tournament_ids = [56,57],
    tournament_ids = [56],
    target_seasons = ["22/23"],
    history_seasons = 0,
    dynamics_col = :match_week,
    # warmup_period = 36,
    warmup_period = 36,
    stop_early = true
)

splits = BayesianFootball.Data.create_data_splits(ds, cv_config)
model = BayesianFootball.Models.PreGame.StaticPoisson() # place holder
vocabulary = BayesianFootball.Features.create_vocabulary(ds, model) 
feature_sets = BayesianFootball.Features.create_features(
    splits, model, cv_config
)
train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=2) 
sampler_conf = Samplers.NUTSConfig(
                250,
                8,
                100,
                0.65,
                10,
  Samplers.UniformInit(-0.05, 0.05),
                :perchain,
)
training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)


# ----------------------------------------------
# 3. GRW models
# ----------------------------------------------

# A: ---  GRW Poisson

grw_poisson_model = Models.PreGame.GRWPoisson()

conf_poisson = Experiments.ExperimentConfig(
                    name = "grw poisson",
                    model = grw_poisson_model,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./data/junk"
)

results_poisson = Experiments.run_experiment(ds, conf_poisson)

describe(results_poisson.training_results[1][1]) 
df_trends_poisson = Models.PreGame.extract_trends(grw_poisson_model, feature_sets[end][1], results_poisson.training_results[end][1])


the new style of model im wanting to test and experiment: 

The Sequential Funnel (Generative DAG)

We model the game as a causal chain. This requires three distinct sets of latent skills per team.

    Creation (λ): How many shots do they create/concede?

    Precision (θ): Given a shot, what is the probability it is on target?

    Finishing (ϕ): Given a shot on target, what is the probability it is a goal?

The Full Generative Model Specification

Let i be the home team and j be the away team for a match k.
1. Layer 1: Shot Generation (Volume)

This is the "engine" of the game. We use your existing Negative Binomial GRW structure here because shots are count data with overdispersion.
Log-RateH​Log-RateA​λH,shots​λA,shots​​=μ+γ+αcreate,i​+βcreate,j​=μ+αcreate,j​+βcreate,i​=exp(Log-RateH​)=exp(Log-RateA​)​

Observation:
SH​SA​​∼NegBinomial(λH,shots​,r)∼NegBinomial(λA,shots​,r)​

    Interpretation: αcreate​ is a team's ability to dominate territory and pull the trigger. βcreate​ is a defense's ability to prevent shots.

2. Layer 2: Shot Precision (Accuracy)

Now we condition on the observed (or generated) number of shots (S). We model the probability that a shot hits the target.
Logit(θH​)Logit(θA​)​=baseθ​+αprec,i​+βprec,j​=baseθ​+αprec,j​+βprec,i​​

Observation (Conditional):
STH​∣SH​STA​∣SA​​∼Binomial(SH​,θH​)∼Binomial(SA​,θA​)​

    Interpretation: αprec​ captures teams that take "high quality" shots (e.g., tap-ins) vs. teams that blast from 30 yards.

3. Layer 3: Conversion (Finishing / Goalkeeping)

Finally, we condition on the Shots on Target (ST). We model the probability that a SOT becomes a goal.
Logit(ϕH​)Logit(ϕA​)​=baseϕ​+αfinish,i​+βsave,j​=baseϕ​+αfinish,j​+βsave,i​​

Observation (Conditional):
GH​∣STH​GA​∣STA​​∼Binomial(STH​,ϕH​)∼Binomial(STA​,ϕA​)​


------


some details regarding the new data score that i been processing, 

Notes for Football Data

All data is in csv format, ready for use within standard spreadsheet applications. Please note that some abbreviations are no longer in use (in particular odds from specific bookmakers no longer used) and refer to data collected in earlier seasons. For a current list of what bookmakers are included in the dataset please visit http://www.football-data.co.uk/matches.php

Key to results data:

Div = League Division
Date = Match Date (dd/mm/yy)
Time = Time of match kick off
HomeTeam = Home Team
AwayTeam = Away Team
FTHG and HG = Full Time Home Team Goals
FTAG and AG = Full Time Away Team Goals
FTR and Res = Full Time Result (H=Home Win, D=Draw, A=Away Win)
HTHG = Half Time Home Team Goals
HTAG = Half Time Away Team Goals
HTR = Half Time Result (H=Home Win, D=Draw, A=Away Win)

Match Statistics (where available)
Attendance = Crowd Attendance
Referee = Match Referee
HS = Home Team Shots
AS = Away Team Shots
HST = Home Team Shots on Target
AST = Away Team Shots on Target
HHW = Home Team Hit Woodwork
AHW = Away Team Hit Woodwork
HC = Home Team Corners
AC = Away Team Corners
HF = Home Team Fouls Committed
AF = Away Team Fouls Committed
HFKC = Home Team Free Kicks Conceded
AFKC = Away Team Free Kicks Conceded
HO = Home Team Offsides
AO = Away Team Offsides
HY = Home Team Yellow Cards
AY = Away Team Yellow Cards
HR = Home Team Red Cards
AR = Away Team Red Cards
HBP = Home Team Bookings Points (10 = yellow, 25 = red)
ABP = Away Team Bookings Points (10 = yellow, 25 = red)

Note that Free Kicks Conceeded includes fouls, offsides and any other offense commmitted and will always be equal to or higher than the number of fouls. Fouls make up the vast majority of Free Kicks Conceded. Free Kicks Conceded are shown when specific data on Fouls are not available (France 2nd, Belgium 1st and Greece 1st divisions).

Note also that English and Scottish yellow cards do not include the initial yellow card when a second is shown to a player converting it into a red, but this is included as a yellow (plus red) for European games.


The following key to betting odds data is described below. These are for pre-closing odds. For the closing odds, as below but with an additional "C" character following the bookmaker abbreviation/Max/Avg (e.g. B365CH = closing Bet365 home win odds).

1XBH = 1XBet home win odds
1XBD = 1XBet draw odds
1XBA = 1XBet away win odds
B365H = Bet365 home win odds
B365D = Bet365 draw odds
B365A = Bet365 away win odds
BFH = Betfair home win odds
BFD = Betfair draw odds
BFA = Betfair away win odds
BFDH = Betfred home win odds
BFDD = Betfred draw odds
BFDA = Betfred away win odds
BMGMH = BetMGM home win odds
BMGMD = BetMGM draw odds
BMGMA = BetMGM away win odds
BVH = Betvictor home win odds
BVD = Betvictor draw odds
BVA = Betvictor away win odds
BSH = Blue Square home win odds
BSD = Blue Square draw odds
BSA = Blue Square away win odds
BWH = Bet&Win home win odds
BWD = Bet&Win draw odds
BWA = Bet&Win away win odds
CLH = Coral home win odds
CLD = Coral draw odds
CLA = Coral away win odds
GBH = Gamebookers home win odds
GBD = Gamebookers draw odds
GBA = Gamebookers away win odds
IWH = Interwetten home win odds
IWD = Interwetten draw odds
IWA = Interwetten away win odds
LBH = Ladbrokes home win odds
LBD = Ladbrokes draw odds
LBA = Ladbrokes away win odds
PSH and PH = Pinnacle home win odds
PSD and PD = Pinnacle draw odds
PSA and PA = Pinnacle away win odds
SOH = Sporting Odds home win odds
SOD = Sporting Odds draw odds
SOA = Sporting Odds away win odds
SBH = Sportingbet home win odds
SBD = Sportingbet draw odds
SBA = Sportingbet away win odds
SJH = Stan James home win odds
SJD = Stan James draw odds
SJA = Stan James away win odds
SYH = Stanleybet home win odds
SYD = Stanleybet draw odds
SYA = Stanleybet away win odds
VCH = VC Bet home win odds (now BetVictor, see above)
VCD = VC Bet draw odds (now BetVictor, see above)
VCA = VC Bet away win odds (now BetVictor, see above)
WHH = William Hill home win odds
WHD = William Hill draw odds
WHA = William Hill away win odds

Bb1X2 = Number of BetBrain bookmakers used to calculate match odds averages and maximums
BbMxH = Betbrain maximum home win odds
BbAvH = Betbrain average home win odds
BbMxD = Betbrain maximum draw odds
BbAvD = Betbrain average draw win odds
BbMxA = Betbrain maximum away win odds
BbAvA = Betbrain average away win odds

MaxH = Market maximum home win odds
MaxD = Market maximum draw win odds
MaxA = Market maximum away win odds
AvgH = Market average home win odds
AvgD = Market average draw win odds
AvgA = Market average away win odds

BFEH = Betfair Exchange home win odds
BFED = Betfair Exchange draw odds
BFEA = Betfair Exchange away win odds



Key to total goals betting odds:

BbOU = Number of BetBrain bookmakers used to calculate over/under 2.5 goals (total goals) averages and maximums
BbMx>2.5 = Betbrain maximum over 2.5 goals
BbAv>2.5 = Betbrain average over 2.5 goals
BbMx<2.5 = Betbrain maximum under 2.5 goals
BbAv<2.5 = Betbrain average under 2.5 goals

GB>2.5 = Gamebookers over 2.5 goals
GB<2.5 = Gamebookers under 2.5 goals
B365>2.5 = Bet365 over 2.5 goals
B365<2.5 = Bet365 under 2.5 goals
P>2.5 = Pinnacle over 2.5 goals
P<2.5 = Pinnacle under 2.5 goals
Max>2.5 = Market maximum over 2.5 goals
Max<2.5 = Market maximum under 2.5 goals
Avg>2.5 = Market average over 2.5 goals
Avg<2.5 = Market average under 2.5 goals



Key to Asian handicap betting odds:

BbAH = Number of BetBrain bookmakers used to Asian handicap averages and maximums
BbAHh = Betbrain size of handicap (home team)
AHh = Market size of handicap (home team) (since 2019/2020)
BbMxAHH = Betbrain maximum Asian handicap home team odds
BbAvAHH = Betbrain average Asian handicap home team odds
BbMxAHA = Betbrain maximum Asian handicap away team odds
BbAvAHA = Betbrain average Asian handicap away team odds

GBAHH = Gamebookers Asian handicap home team odds
GBAHA = Gamebookers Asian handicap away team odds
GBAH = Gamebookers size of handicap (home team)
LBAHH = Ladbrokes Asian handicap home team odds
LBAHA = Ladbrokes Asian handicap away team odds
LBAH = Ladbrokes size of handicap (home team)
B365AHH = Bet365 Asian handicap home team odds
B365AHA = Bet365 Asian handicap away team odds
B365AH = Bet365 size of handicap (home team)
PAHH = Pinnacle Asian handicap home team odds
PAHA = Pinnacle Asian handicap away team odds
MaxAHH = Market maximum Asian handicap home team odds
MaxAHA = Market maximum Asian handicap away team odds	
AvgAHH = Market average Asian handicap home team odds
AvgAHA = Market average Asian handicap away team odds



Football-Data would like to acknowledge the following sources which have been utilised in the compilation of Football-Data's results and odds files.



as you see we are interested in 

Match Statistics (where available)
HS = Home Team Shots
AS = Away Team Shots
HST = Home Team Shots on Target
AST = Away Team Shots on Target
HC = Home Team Corners
AC = Away Team Corners
HF = Home Team Fouls Committed
AF = Away Team Fouls Committed
HY = Home Team Yellow Cards
AY = Away Team Yellow Cards
HR = Home Team Red Cards
AR = Away Team Red Cards


to have a feature set for the model, 








=#




using DataFrames
using CSV
using Dates
using InlineStrings


folder_path = "/home/james/bet_project/football/scotland_l12_extra"
files_list::Vector{String} = readdir(folder_path)


const DATA_COLS_TYPES = Dict(
    :Div => String3,
    :Date => Date,
    :Time => Time,
    :HomeTeam => String31,
    :AwayTeam => String31,
    :FTHG => Float64,
    :FTAG => Float64,
    :FTR => String1,
    :HTHG => Float64,
    :HTAG => Float64,
    :HTR => String1,
    :Referee => String15,
    :HS => Float64,
    :AS => Float64,
    :HST => Float64,
    :AST => Float64,
    :HF => Float64,
    :AF => Float64,
    :HC => Float64,
    :AC => Float64,
    :HY => Float64,
    :AY => Float64,
    :HR => Float64,
    :AR => Float64,
)


# --- Internal Helpers ---
# --- Helper to parse "l1_2122.csv" -> "21/22" ---
function _get_season_from_filename(filename::AbstractString)
    # 1. Extract the "2122" part
    # Split by '_' to get "2122.csv", then by '.' to get "2122"
    raw_digits = split(split(basename(filename), "_")[2], ".")[1]
    
    # 2. Format as "21/22"
    return "$(raw_digits[1:2])/$(raw_digits[3:4])"
end
function _loaded_dataframe(file_path::AbstractString)::AbstractDataFrame
    matches = CSV.read(file_path, DataFrame; 
        types = DATA_COLS_TYPES,
        dateformat = Dict(:Date => dateformat"dd/mm/yyyy")
    )
    
    # Extract season string
    season_str = _get_season_from_filename(file_path)
    
    # Add it as a new column to every row
    # We use :Season (Symbol) to match your other column naming conventions
    insertcols!(matches, :Season => season_str)
    
    return matches 
end

function _loaded_dataframe(file_path::AbstractString)::AbstractDataFrame
    matches = CSV.read(file_path, DataFrame; 
    types= DATA_COLS_TYPES,
    dateformat=Dict(:Date => dateformat"dd/mm/yyyy")
    )
    return matches 
end

file_path_1 = joinpath(folder_path, files_list[1])

d1 = _loaded_dataframe(file_path_1)


file_path_2 = joinpath(folder_path, files_list[2])
d2 = _loaded_dataframe(file_path_2)
append!(d1, d2, promote=true)


function _load_dateframes(folder_dir::AbstractString)::AbstractDataFrame
    # Create a list of all loaded DataFrames
    all_dfs = [ _loaded_dataframe(joinpath(folder_dir, f)) 
                for f in readdir(folder_dir) if endswith(f, ".csv") ]
    
    # Vertically concatenate them all at once
    return vcat(all_dfs..., cols=:union)
end

df = _load_dateframes(folder_path)

###
# name mapping
###

# get all the names

team_names_list = unique( df.HomeTeam)

using BayesianFootball
data_store = BayesianFootball.Data.load_default_datastore()
ds_l12 = subset(data_store.matches, :tournament_id => ByRow(in([56,57])), 
                                    :season => ByRow(in(["21/22", "22/23", "23/24", "24/25"])),
                )

projection_team_names_list = unique(ds_l12.home_team)



const TEAM_NAME_MAPPING = Dict(
    # --- The Mismatches & Abbreviations ---
    "Airdrie Utd"    => "airdrieonians",
    "Albion Rvs"     => "albion-rovers",
    "Alloa"          => "alloa-athletic",
    "Clyde"          => "clyde-fc",
    "Dunfermline"    => "dunfermline-athletic",
    "Elgin"          => "elgin-city",
    "Falkirk"        => "falkirk-fc",
    "Forfar"         => "forfar-athletic",
    "Hamilton"       => "hamilton-academical",
    "Inverness C"    => "inverness-caledonian-thistle",
    "Kelty Hearts"   => "kelty-hearts-fc",
    "Queen of Sth"   => "queen-of-the-south",
    "Queens Park"    => "queens-park-fc",
    "Spartans"       => "the-spartans-fc",
    "Stirling"       => "stirling-albion",

    # --- The Duplicate Case (Rebranding) ---
    "Edinburgh City" => "edinburgh-city-fc",
    "FC Edinburgh"   => "edinburgh-city-fc", 

    # --- The Exact or Near-Exact Matches ---
    "Annan Athletic" => "annan-athletic",
    "Arbroath"       => "arbroath",
    "Bonnyrigg Rose" => "bonnyrigg-rose",
    "Cove Rangers"   => "cove-rangers",
    "Cowdenbeath"    => "cowdenbeath",
    "Dumbarton"      => "dumbarton",
    "East Fife"      => "east-fife",
    "Montrose"       => "montrose",
    "Peterhead"      => "peterhead",
    "Stenhousemuir"  => "stenhousemuir",
    "Stranraer"      => "stranraer"
)

# for the df Div
const TOURNAMENT_MAPPING = Dict(
      "SC2" => 56,
      "SC3" => 57
)


ds_l12
names(ds_l12)

# Apply the mapping to your DataFrame
# This creates a new column :team_id that matches the projection format
df.home_team_id = [get(TEAM_NAME_MAPPING, name, missing) for name in df.HomeTeam]
df.away_team_id = [get(TEAM_NAME_MAPPING, name, missing) for name in df.AwayTeam]



ds = BayesianFootball.Data.DataStore( 
    Data.add_match_week_column(data_store.matches),
    data_store.odds,
    data_store.incidents
)




using DataFrames, Statistics

"""
    join_with_validation(local_df, db_df, team_mapping, tournament_mapping)

Merges local CSV data with the database `ds_l12` to get match_ids, 
but first validates that game counts match per season/division.
"""
function join_with_validation(local_df::DataFrame, db_df::DataFrame, team_map::Dict, tourn_map::Dict)
    
    # --- 1. PRE-PROCESSING LOCAL DATA ---
    # We create a working copy so we don't mutate the original inputs unexpectedly
    df_work = copy(local_df)

    # Apply mappings to match DB format
    df_work.home_team_id = [get(team_map, x, missing) for x in df_work.HomeTeam]
    df_work.away_team_id = [get(team_map, x, missing) for x in df_work.AwayTeam]
    df_work.tournament_id = [get(tourn_map, x, missing) for x in df_work.Div]

    # --- 2. VALIDATION STEP ---
    println("--- Validating Match Counts ---")
    
    # Count games in Local Data (Group by Season + Tournament ID)
    local_counts = combine(
        groupby(df_work, [:Season, :tournament_id]), 
        nrow => :count_local
    )

    # Count games in DB Data (Group by Season + Tournament ID)
    # Note: We filter db_df to only include the tournaments present in our local data
    relevant_ids = unique(skipmissing(df_work.tournament_id))
    db_subset = filter(row -> row.tournament_id in relevant_ids, db_df)
    
    db_counts = combine(
        groupby(db_subset, [:season, :tournament_id]), 
        nrow => :count_db
    )

    # Merge counts to compare
    # We join on Season (String) and tournament_id (Int)
    validation_df = leftjoin(local_counts, db_counts, on = [:Season => :season, :tournament_id])

    # Check for mismatches
    mismatches = filter(row -> row.count_local != row.count_db, validation_df)

    if isempty(mismatches)
        println("✅ Validation Passed: All game counts match exactly.")
    else
        println("⚠️ VALIDATION FAILED: Found mismatches in the following seasons:")
        display(mismatches)
        println("Proceeding with join, but please investigate the missing/extra rows above.")
    end
    println("-------------------------------")

    # --- 3. MERGING STEP ---
    # We join on the mapped IDs and Date
    # selecting only the match_id from the DB
    
    merged_df = leftjoin(
        df_work, 
        select(db_df, :match_id, :season, :tournament_id, :match_date, :home_team, :away_team),
        on = [
            :Season => :season,
            :tournament_id => :tournament_id,
            :Date => :match_date,
            :home_team_id => :home_team,
            :away_team_id => :away_team
        ],
        makeunique=true
    )

    return merged_df
end


final_df = join_with_validation(df, ds_l12, TEAM_NAME_MAPPING, TOURNAMENT_MAPPING)

using DataFrames, RegularExpressions

function extract_data_store(df::DataFrame)
    
    # --- 1. Define the known Bookie Codes ---
    # Based on the text you provided. 
    # We map the CSV code to a Clean Name.
    bookie_map = Dict(
        "B365" => "Bet365",
        "BW"   => "Bet&Win",
        "IW"   => "Interwetten",
        "PS"   => "Pinnacle",
        "VC"   => "VC Bet",
        "WH"   => "William Hill",
        "1X"   => "1XBet",
        "BF"   => "Betfair",
        "BB"   => "BetBrain Avg", # You can exclude this if you don't want averages
    )

    # --- 2. Identify Odds Columns vs Match Columns ---
    
    # We find columns that look like odds (e.g., end in H, D, A or contain >2.5)
    # Exclude standard stats columns that end in H/A like "FTHG" (Goals) or "HF" (Fouls)
    stat_cols = ["FTHG", "FTAG", "HTHG", "HTAG", "HTR", "FTR", 
                 "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", 
                 "HY", "AY", "HR", "AR"]
                 
    all_cols = names(df)
    
    # Regex logic: 
    # 1X2: Starts with a Bookie Code, ends with H, D, or A.
    # O/U: Starts with a Bookie Code, contains >2.5 or <2.5
    odds_cols = String[]
    
    for col in all_cols
        # Check if it matches our known bookie prefixes
        for (code, name) in bookie_map
            # 1X2 Check (e.g. B365H) - Ensure it's not a stat col
            if startswith(col, code) && (endswith(col, "H") || endswith(col, "D") || endswith(col, "A")) && !(col in stat_cols)
                push!(odds_cols, col)
            # Over/Under Check (e.g. B365>2.5)
            elseif startswith(col, code) && (contains(col, ">") || contains(col, "<"))
                push!(odds_cols, col)
            end
        end
    end

    # --- 3. Create Matches Table (Metadata & Stats) ---
    # Select everything that ISN'T an odds column
    matches_df = select(df, Not(odds_cols))


    # --- 4. Create Odds Table (Long Format) ---
    # Stack pivots the chosen columns into two new columns: :variable (name) and :decimal_odds (value)
    long_df = stack(df, odds_cols, [:match_id, :tournament_id, :Season], variable_name=:raw_col, value_name=:decimal_odds)
    
    # Remove missing odds (bookies often have gaps)
    dropmissing!(long_df, :decimal_odds)

    # --- 5. Parse the Column Names ---
    # We need to extract Bookie, Market, and Choice from strings like "B365H" or "B365>2.5"
    
    # Pre-allocate vectors for speed
    n = nrow(long_df)
    bookies = Vector{String}(undef, n)
    markets = Vector{String}(undef, n)
    choices = Vector{String}(undef, n)
    
    raw_cols = long_df.raw_col
    
    for i in 1:n
        s = raw_cols[i]
        
        # --- Parser Logic ---
        # 1. Check for Over/Under 2.5
        if contains(s, ">2.5")
            # e.g., B365>2.5
            code = replace(s, ">2.5" => "")
            bookies[i] = get(bookie_map, code, code)
            markets[i] = "Over/Under 2.5"
            choices[i] = "Over"
            
        elseif contains(s, "<2.5")
            # e.g., B365<2.5
            code = replace(s, "<2.5" => "")
            bookies[i] = get(bookie_map, code, code)
            markets[i] = "Over/Under 2.5"
            choices[i] = "Under"
            
        else 
            # 2. Assume 1X2 (Home/Draw/Away)
            # The last character is the choice (H, D, A)
            choice_char = s[end]
            code = s[1:end-1]
            
            bookies[i] = get(bookie_map, code, code)
            markets[i] = "1X2"
            
            if choice_char == 'H'
                choices[i] = "1" # Home
            elseif choice_char == 'D'
                choices[i] = "X" # Draw
            elseif choice_char == 'A'
                choices[i] = "2" # Away
            else
                choices[i] = "Unknown"
            end
        end
    end
    
    # Assign the new columns
    long_df.bookie = bookies
    long_df.market_name = markets
    long_df.choice_name = choices
    
    # Clean up: Select only relevant columns to match your struct
    # (You can add winning/fractional columns here as 'missing' to match the struct exactly)
    odds_df = select(long_df, 
        :match_id,
        :tournament_id,
        :bookie, 
        :market_name, 
        :choice_name, 
        :decimal_odds
    )

    return matches_df, odds_df
end


# Run the extraction
clean_matches, clean_odds = extract_data_store(final_df)

# Check the results
println("Matches: $(nrow(clean_matches)) rows")
println("Odds: $(nrow(clean_odds)) rows")

# Preview the odds table
first(clean_odds, 5)


### 
# have added to src/data/scotland_extra 
#  export function load_extra_ds

using BayesianFootball


ds = Data.load_extra_ds()

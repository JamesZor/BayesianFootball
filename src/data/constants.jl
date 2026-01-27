# src/data/constants.jl

using Dates
using InlineStrings

export DataPaths, NameType, MATCHES_COLS_TYPES, INCIDENTS_COLS_TYPES, ODDS_COLS_TYPES

# --- Configuration ---

const DataPaths = (
    scotland = "/home/james/bet_project/football/scotland_football_v2",
    uk_all   = "/home/james/bet_project/football/uk_football_data_20_26", 
)

# --- Type Aliases ---

const NameType = String63

# --- Column Mappings ---

const MATCHES_COLS_TYPES = Dict(
    :tournament_id => Int64,
    :season_id => Int64,
    :season => String7,
    :match_id => Int64,
    :tournament_slug => String15,
    :home_team => String31,
    :away_team => String31,
    # Float64 for potential missing values or precision
    :home_score => Float64,
    :away_score => Float64,
    :home_score_ht => Float64,
    :away_score_ht => Float64,
    :winner_code => Float64,
    :match_date => Date,
    :round => Float64,
    :has_xg => Bool,
    :has_stats => Bool,
    :match_hour => Int64,
    :match_dayofweek => Int64,
    :match_month => Int64
)

const INCIDENTS_COLS_TYPES = Dict(
    :tournament_id => Int64,
    :season_id => Int64,
    :match_id => Int64,
    :incident_type => String15,
    :time => Int64,
    :is_home => Bool,
    :is_live => Bool,
    :period_text => String3,
    :player_in_name => NameType,
    :player_out_name => NameType,
    :is_injury => Bool,
    :player_name => NameType,
    :card_type => String15,
    :assist1_name => NameType,
    :assist2_name => NameType,
    # Float64 for safety
    :home_score => Float64,
    :away_score => Float64,
    :added_time => Float64,
    :time_seconds => Float64,
    :reversed_period_time => Float64,
    :reversed_period_time_seconds => Float64,
    :period_time_seconds => Float64,
    :injury_time_length => Float64,
    # Categoricals
    :incident_class => String31,
    :reason => String31,
    :rescinded => Bool,
    :var_confirmed => Bool,
    :var_decision => String31,
    :var_reason => String31,
    :penalty_reason => String31
)

const ODDS_COLS_TYPES = Dict(
    :tournament_id => Int64,
    :season_id => Int64,
    :match_id => Int64,
    :market_id => Int64,
    :market_name => String31,
    :market_group => String31,
    :choice_name => String63, 
    :choice_group => Float64,
    :initial_fractional_value => String7,
    :final_fractional_value => String7,
    :winning => Bool,
    :decimal_odds => Float64
)

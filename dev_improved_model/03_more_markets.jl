
using Revise
using BayesianFootball
using DataFrames

using Statistics
using ThreadPinning
using LinearAlgebra
pinthreads(:cores)


ds = Data.load_extra_ds()
transform!(ds.matches, :match_week => ByRow(w -> cld(w, 4)) => :match_month)


market_data = Data.prepare_market_data(ds)


df = subset( ds.incidents, :incident_type => ByRow(isequal("goal")))


df1 = select(df, :match_id, :incident_type, :is_home, :home_score, :away_score, :player_name, :assist1_name, :assist2_name)
names(df)

julia> names(df)
30-element Vector{String}:
 "tournament_id"
 "season_id"
 "match_id"
 "incident_type"
 "time"
 "is_home"
 "period_text"
 "home_score"
 "away_score"
 "is_live"
 "added_time"
 "time_seconds"
 "reversed_period_time"
 "reversed_period_time_seconds"
 "period_time_seconds"
 "injury_time_length"
 "player_in_name"
 "player_out_name"
 "is_injury"
 "incident_class"
 "player_name"
 "card_type"
 "reason"
 "rescinded"
 "assist1_name"
 "assist2_name"
 "var_confirmed"
 "var_decision"
 "var_reason"
 "penalty_reason"

using BayesianFootball
using DataFrames
using Dates
using Statistics, StatsBase, StatsPlots, Distributions, Plots
using CSV 



const DATA_PATH = "/home/james/bet_project/football/scotland_football"
data_files = DataFiles(DATA_PATH)

path_odd = "/home/james/bet_project/football/scotland_football/football_data_mixed_odds.csv"

odds = CSV.read(path_odd, DataFrame, header=1)

_1x2  = filter(row -> row.market_name=="Full time" && row.market_group=="1X2", odds)

names(_1x2)

_1x2[:, [:tournament_id, :match_id, :choice_name, :decimal_odds]]


filter(row -> row.match_id==13247404, data_store.matches)

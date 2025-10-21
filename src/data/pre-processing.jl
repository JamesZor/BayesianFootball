# src/data/preprocessing.jl

module DataPreprocessing

using DataFrames
using Dates
using ..Data: AbstractDataFrame # Use AbstractDataFrame from Data module if defined there, else use DataFrames.AbstractDataFrame

export add_match_week_column, add_global_round_column # Export user-facing names

"""
Calculates the date of the Sunday for a given date.
Helper for grouping matches by week.
"""
function sunday_of_week(dt::Date)::Date
    day_num = dayofweek(dt)
    return dt + Day(7 - day_num)
end

"""
Adds a ':match_week' column to the DataFrame, grouping matches by
the Sunday of the week they were played. Returns a *new* DataFrame copy.
"""
function add_match_week_column(matches_df::AbstractDataFrame)::DataFrame
    df = copy(matches_df) # Work on a copy
    sort!(df, :match_date)

    df.match_week_date = sunday_of_week.(df.match_date)

    unique_weeks = sort(unique(df.match_week_date))
    week_map = Dict(week_date => i for (i, week_date) in enumerate(unique_weeks))
    df.match_week = [week_map[wd] for wd in df.match_week_date]

    select!(df, Not(:match_week_date)) # Remove intermediate column

    return df
end


"""
Adds a ':global_round' column based on date sorting and team participation.
Returns a *new* DataFrame copy.
"""
function add_global_round_column(matches_df::AbstractDataFrame)::DataFrame
    df = copy(matches_df) # Work on a copy
    sort!(df, :match_date)
    num_matches = nrow(df)
    global_rounds = Vector{Int}(undef, num_matches)
    global_round_counter = 1
    teams_in_current_round = Set{String}()

    for (i, row) in enumerate(eachrow(df))
        home_team, away_team = row.home_team, row.away_team
        if home_team in teams_in_current_round || away_team in teams_in_current_round #
            global_round_counter += 1
            empty!(teams_in_current_round)
        end
        global_rounds[i] = global_round_counter
        push!(teams_in_current_round, home_team, away_team)
    end

    df.global_round = global_rounds
    return df
end


end # module DataPreprocessing

# src/data/preprocessing.jl

module DataPreprocessing

using DataFrames
using Dates
using ..Data: AbstractDataFrame, DataStore # Use AbstractDataFrame from Data module if defined there, else use DataFrames.AbstractDataFrame

export add_match_week_column, add_global_round_column, add_inital_odds_from_fractions! # Export user-facing names
export add_split_col_match_week

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



"""
Section for the odds to be converted, since we have then in fraction for the opening and closing lines.
Hence need to create them in a decimal form - standardise them.
"""


"""
Parses a fractional odds string (e.g., "19/10") into a decimal value (e.g., 2.9).
Returns 0.0 if parsing fails (e.g., for "SP", missing, or "1").
"""
function parse_fractional_to_decimal(s::AbstractString)
    parts = split(s, '/')
    
    # Must be exactly two parts (numerator and denominator)
    if length(parts) != 2
        return 0.0
    end
    
    try
        n = parse(Float64, parts[1])
        d = parse(Float64, parts[2])
        
        # Avoid division by zero
        if d == 0.0
            return 0.0
        end
        
        # Convert from fractional (e.g., 1.9) to decimal (e.g., 2.9)
        return (n / d) + 1.0
    catch e
        # This will catch errors if parts[1] or parts[2] are not valid numbers (e.g., "SP")
        return 0.0
    end
end

function add_inital_odds_from_fractions!(df_odds::DataFrame)
  df_odds.initial_decimal = parse_fractional_to_decimal.(df_odds.initial_fractional_value);
  df_odds.initial_decimal = round.(df_odds.initial_decimal, digits=2);
end

function add_inital_odds_from_fractions!(ds::DataStore)
  add_inital_odds_from_fractions!(ds.odds);
end



"""
basic method to split the data from a certain week of the seaosn.
"""

function add_split_col_match_week(data_store::DataStore, week_number::Int64 )::DataStore
    all_matches_transformed = combine(groupby(data_store.matches, :season)) do sub_df
        df = DataFrame(sub_df)
        df = add_match_week_column(df)
        df.split_col = max.(0, df.match_week .- week_number)
        return df
    end
    
    return DataStore(
    all_matches_transformed,
    data_store.odds,
    data_store.incidents
)
end 



              



end # module DataPreprocessing

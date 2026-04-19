# dev_repl/data_module_sql_update/r01_get_datastore_from_sql.jl 


#=
--- Runner ----
____________________
=#

include("./l01_get_datastore_from_sql.jl")

# Task 1:
#   Create the connection 

db_config = DBConfig("postgresql://admin:supersecretpassword@100.124.38.117:5432/sofascrape_db")

db_conn = connect_to_db(db_config)



# Task 2 
sl = ScottishLower()
tournament_ids(sl)
il = Ireland()
tournament_ids(il)


# Task 3 

# task 3.1 - fetch - matches 

df_matches = fetch_matches(
                  db_conn,
                  ScottishLower())

df_matches = fetch_data(db_conn, Ireland(), MatchesData())
df_inc = fetch_data(db_conn, ScottishLower(), IncidentsData())

df_stats = fetch_data(db_conn, SouthKorea(), StatisticsData())

df_odds = fetch_data(db_conn, ScottishLower(), OddsData())

df_lineup = fetch_data(db_conn, Ireland(), LineUpsData())



data_store = get_datastore(db_conn, ScottishLower())


ds = Data.load_extra_ds()
transform!(ds.matches, :match_week => ByRow(w -> cld(w, 4)) => :match_month)



# --- 1. The Duplicate Hunter (Testing your suspicion)
# 1. Check for exact row-for-row duplicates
exact_dupes = ds.odds[nonunique(ds.odds), :]
println("Exact duplicate rows in old ds.odds: ", nrow(exact_dupes))

# 2. Check for Primary Key duplicates (e.g., same bet recorded twice with different odds)
# Adjust these symbols to match your actual columns if they differ slightly
odds_keys = [:match_id, :market_id, :choice_name]

pk_counts = combine(groupby(ds.odds, odds_keys), nrow => :count)
suspect_odds = subset(pk_counts, :count => ByRow(>(1)))

println("Primary Key duplicates in old ds.odds: ", nrow(suspect_odds))
if nrow(suspect_odds) > 0
    display(suspect_odds) # Look at what got duplicated
end

#=
julia> exact_dupes = ds.odds[nonunique(ds.odds), :]                                                                                                                                                                                                                                                                         
0×12 DataFrame                                                                                                                                                                                                                                                                                                              
 Row │ tournament_id  season_id  match_id  market_id  market_name  market_group  choice_name  choice_group  initial_fractional_value  final_fractional_value  winning  decimal_odds                                                                                                                                         
     │ Int64          Int64      Int64     Int64      String31     String31      String63     Float64?      InlineStrings.String7     InlineStrings.String7   Bool?    Float64                                                                                                                                              
─────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                                        
                                                                                                                                                                                                                                                                                                                            
julia> println("Exact duplicate rows in old ds.odds: ", nrow(exact_dupes))                                                                                                                                                                                                                                                  
Exact duplicate rows in old ds.odds: 0                                                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                                                            
julia>                                                                                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                                                            
julia> odds_keys = [:match_id, :market_id, :choice_name]                                                                                                                                                                                                                                                                    
3-element Vector{Symbol}:                                                                                                                                                                                                                                                                                                   
 :match_id                                                                                                                                                                                                                                                                                                                  
 :market_id                                                                                                                                                                                                                                                                                                                 
 :choice_name                                                                                                                                                                                                                                                                                                               
                                                                                                                                                                                                                                                                                                                            
julia> pk_counts = combine(groupby(ds.odds, odds_keys), nrow => :count)                                                                                                                                                                                                                                                     
84110×4 DataFrame                                                                                                                                                                                                                                                                                                           
   Row │ match_id  market_id  choice_name                  count                                                                                                                                                                                                                                                            
       │ Int64     Int64      String63                     Int64                                                                                                                                                                                                                                                            
───────┼─────────────────────────────────────────────────────────                                                                                                                                                                                                                                                           
     1 │ 10387456          1  1                                1                                                                                                                                                                                                                                                            
     2 │ 10387456          1  X                                1                                                                                                                                                                                                                                                            
     3 │ 10387456          1  2                                1                                                                                                                                                                                                                                                            
     4 │ 10387456          2  1X                               1                                                                                                                                                                                                                                                            
     5 │ 10387456          2  X2                               1                                                                                                                                                                                                                                                            
     6 │ 10387456          2  12                               1                                                                                                                                                                                                                                                            
     7 │ 10387456          3  1                                1                                                                                                                                                                                                                                                            
     8 │ 10387456          3  X                                1                                                                                                                                                                                                                                                            
     9 │ 10387456          3  2                                1                                                                                                                                                                                                                                                            
    10 │ 10387456          4  1                                1                                                                                                                                                                                                                                                            
    11 │ 10387456          4  2                                1                                                                                                                                                                                                                                                            
    12 │ 10387456          5  Yes                              1                                                                                                                                                                                                                                                            
    13 │ 10387456          5  No                               1                                                                                                                                                                                                                                                            
    14 │ 10387456          9  Over                             8                                                                                                                                                                                                                                                            
    15 │ 10387456          9  Under                            8                                                                                                                                                                                                                                                            
    16 │ 10387456         17  (0.25) Motherwell                1                                                                                                                                                                                                                                                            
    17 │ 10387456         17  (-0.25) Heart of Midlothian      1                                                                                                                                                                                                                                                            
    18 │ 10387456         20  Over                             1                                                                                                                                                                                                                                                            
    19 │ 10387456         20  Under                            1                                                                                                                                                                                                                                                            
    20 │ 10387456         21  Over                             1                                                                                                                                                                                                                                                            
    21 │ 10387456         21  Under                            1                                                                                                                                                                                                                                                            
    22 │ 10387456          6  Motherwell                       1                                                                                                                                                                                                                                                            
    23 │ 10387456          6  No goal                          1                                                                                                                                                                                                                                                            
    24 │ 10387456          6  Heart of Midlothian              1                                                                                                                                                                                                                                                            
    25 │ 10387457          1  1                                1                               
    26 │ 10387457          1  X                                1                               
    27 │ 10387457          1  2                                1                               
    28 │ 10387457          2  1X                               1                               
    29 │ 10387457          2  X2                               1                               
    30 │ 10387457          2  12                               1                               
    31 │ 10387457          3  1                                1                               
    32 │ 10387457          3  X                                1                               
    33 │ 10387457          3  2                                1                               
   ⋮   │    ⋮          ⋮                   ⋮                 ⋮                                 
 84079 │ 13343735          4  2                                1                               
 84080 │ 13343735          5  Yes                              1                               
 84081 │ 13343735          5  No                               1                               
 84082 │ 13343735          9  Over                             7                               
 84083 │ 13343735          9  Under                            7                               
 84084 │ 13343735         17  (0) Elgin City                   1                               
 84085 │ 13343735         17  (-0) The Spartans FC             1                               
 84086 │ 13343735          6  Elgin City                       1                               
 84087 │ 13343735          6  No goal                          1                               
 84088 │ 13343735          6  The Spartans FC                  1                               
 84089 │ 13343736          1  1                                1                               
 84090 │ 13343736          1  X                                1                                                                                              
 84091 │ 13343736          1  2                                1                                                                                              
 84092 │ 13343736          2  1X                               1                                                                                              
 84093 │ 13343736          2  X2                               1                                                                                              
 84094 │ 13343736          2  12                               1                                                                                              
 84095 │ 13343736          3  1                                1                                                                                              
 84096 │ 13343736          3  X                                1                                                                                              
 84097 │ 13343736          3  2                                1                                                                                              
 84098 │ 13343736          4  1                                1                                                                                              
 84099 │ 13343736          4  2                                1                                                                                              
 84100 │ 13343736          5  Yes                              1                                                                                              
 84101 │ 13343736          5  No                               1                                                                                              
 84102 │ 13343736          9  Over                             7                                                                                              
 84103 │ 13343736          9  Under                            7                                                                                              
 84104 │ 13343736         17  (-0.25) Bonnyrigg Rose           1                                                                                              
 84105 │ 13343736         17  (0.25) Stranraer                 1                                                                                              
 84106 │ 13343736         21  Over                             1                                                                                              
 84107 │ 13343736         21  Under                            1                                                                                              
 84108 │ 13343736          6  Bonnyrigg Rose                   1                                                                                              
 84109 │ 13343736          6  No goal                          1                                                                                              
 84110 │ 13343736          6  Stranraer                        1                                                                                              
                                               84045 rows omitted                                                                                             
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         21:53 [5/1710]
julia> suspect_odds = subset(pk_counts, :count => ByRow(>(1)))                                                                                                
7574×4 DataFrame                                                                                                                                              
  Row │ match_id  market_id  choice_name  count                                                                                                               
      │ Int64     Int64      String63     Int64                                                                                                               
──────┼─────────────────────────────────────────                                                                                                              
    1 │ 10387456          9  Over             8                                                                                                               
    2 │ 10387456          9  Under            8                                                                                                               
    3 │ 10387457          9  Over             3                                                                                                               
    4 │ 10387457          9  Under            3                                                                                                               
    5 │ 10387461          9  Over             7                                                                                                               
    6 │ 10387461          9  Under            7                                                                                                               
    7 │ 10387462          9  Over             4                                                                                                               
    8 │ 10387462          9  Under            4                                                                                                               
    9 │ 10387463          9  Over             6                                                                                                               
   10 │ 10387463          9  Under            6                                                                                                               
   11 │ 10387464          9  Over             6                                                                                                               
   12 │ 10387464          9  Under            6                                                                                                               
   13 │ 10387466          9  Over             9                                                                                                               
   14 │ 10387466          9  Under            9                                                                                                               
   15 │ 10387467          9  Over             6                                                                                                               
   16 │ 10387467          9  Under            6                                                                                                               
   17 │ 10387468          9  Over             6                                                                                                               
   18 │ 10387468          9  Under            6                                                                                                               
   19 │ 10387469          9  Over             3                                                                                                               
   20 │ 10387469          9  Under            3                                                                                                               
   21 │ 10387470          9  Over             7                                                                                                               
   22 │ 10387470          9  Under            7                                                                                                               
   23 │ 10387471          9  Over             7                                                                                                               
   24 │ 10387471          9  Under            7                                                                                                               
   25 │ 10387473          9  Over             7                                                                                                               
   26 │ 10387473          9  Under            7                                                                                                               
   27 │ 10387474          9  Over             8                                                                                                               
   28 │ 10387474          9  Under            8                                                                                                               
   29 │ 10387475          9  Over             7                                                                                                               
   30 │ 10387475          9  Under            7                                                                                                               
   31 │ 10387476          9  Over             3                                                                                                               
   32 │ 10387476          9  Under            3                                                                                                               
   33 │ 10387477          9  Over             8                                                                                                               
  ⋮   │    ⋮          ⋮           ⋮         ⋮                                                                                                                 
 7543 │ 12476615          9  Over             7                                                                                                               
 7544 │ 12476615          9  Under            7                                                                                                               
 7545 │ 12954858          9  Over             8                                                                                                               
 7546 │ 12954858          9  Under            8                                                                                                               
 7547 │ 12476617          9  Over             7                                                                                                               
 7548 │ 12476617          9  Under            7                                                                                                               
 7549 │ 12476619          9  Over             8                                                                                                               
 7550 │ 12476619          9  Under            8                                                                                                               
 7551 │ 13315347          9  Over             7                                                                                                               
 7552 │ 13315347          9  Under            7                                                                                                               
 7553 │ 13233442          9  Over             7                                                                                                               
 7554 │ 13233442          9  Under            7                                                                                                               
 7555 │ 12811090          9  Over             7                                                                                                               
 7556 │ 12811090          9  Under            7                                                                                                               
 7557 │ 13436762          9  Over             7                                                                                                               
 7558 │ 13436762          9  Under            7                                                                                                               
 7559 │ 13217663          9  Over             7                                                                                                               
 7560 │ 13217663          9  Under            7                                                                                                               
 7561 │ 13255608          9  Over             7                                                                                                               
 7562 │ 13255608          9  Under            7                                                                                                               
 7563 │ 13175771          9  Over             7                                                                                                               
 7564 │ 13175771          9  Under            7                                                                                                               
 7565 │ 13175772          9  Over             7                                                                                                               
 7566 │ 13175772          9  Under            7                                                                                                               
 7567 │ 12476460          9  Over             7                                                                                                               
 7568 │ 12476460          9  Under            7                                                                                                               
 7569 │ 13171688          9  Over             8                                                                                                               
 7570 │ 13171688          9  Under            8                                                                                                               
 7571 │ 13343735          9  Over             7                                                                                                               
 7572 │ 13343735          9  Under            7                                                                                                               
 7573 │ 13343736          9  Over             7                                                                                                               
 7574 │ 13343736          9  Under            7                                                                                                               
                               7509 rows omitted                                    

julia> println("Primary Key duplicates in old ds.odds: ", nrow(suspect_odds))                                                                                 
Primary Key duplicates in old ds.odds: 7574                                                                                                                   

julia> if nrow(suspect_odds) > 0                                                                                                                              
           display(suspect_odds) # Look at what got duplicated                                                                                                
       end                                                                                                                                                    
7574×4 DataFrame                                                                                                                                              
  Row │ match_id  market_id  choice_name  count                                                                                                               
      │ Int64     Int64      String63     Int64                                                                                                               
──────┼─────────────────────────────────────────                                                                                                              
    1 │ 10387456          9  Over             8                                                                                                               
    2 │ 10387456          9  Under            8                                                                                                               
    3 │ 10387457          9  Over             3                                                                                                               
    4 │ 10387457          9  Under            3                                                                                                               
    5 │ 10387461          9  Over             7                                                                                                               
    6 │ 10387461          9  Under            7                                                                                                               
    7 │ 10387462          9  Over             4                                                                                                               
    8 │ 10387462          9  Under            4                                                                                                               
    9 │ 10387463          9  Over             6                                                                                                               
   10 │ 10387463          9  Under            6                                                                                                               
   11 │ 10387464          9  Over             6                                                                                                               
   12 │ 10387464          9  Under            6                                                                                                               
   13 │ 10387466          9  Over             9                                                                                                               
   14 │ 10387466          9  Under            9                                                                                                               
   15 │ 10387467          9  Over             6                                                                                                               
   16 │ 10387467          9  Under            6                                                                                                               
   17 │ 10387468          9  Over             6                                                                                                               
   18 │ 10387468          9  Under            6                                                                                                               
   19 │ 10387469          9  Over             3                                                                                                               
   20 │ 10387469          9  Under            3                                                                                                               
   21 │ 10387470          9  Over             7                                                                                                               
   22 │ 10387470          9  Under            7                                                                                                               
   23 │ 10387471          9  Over             7                                                                                                               
   24 │ 10387471          9  Under            7                                                                                                               
   25 │ 10387473          9  Over             7                                                                                                               
   26 │ 10387473          9  Under            7                                                                                                               
   27 │ 10387474          9  Over             8                                                                                                               
   28 │ 10387474          9  Under            8                                                                                                               
   29 │ 10387475          9  Over             7                                                                                                               
   30 │ 10387475          9  Under            7                                                                                                               
   31 │ 10387476          9  Over             3                                                                                                               
   32 │ 10387476          9  Under            3                                                                                                               
   33 │ 10387477          9  Over             8                                                                                                               
  ⋮   │    ⋮          ⋮           ⋮         ⋮                                                                                                                 
 7543 │ 12476615          9  Over             7                                                                                                               
 7544 │ 12476615          9  Under            7                                                                                                               
 7545 │ 12954858          9  Over             8                                                                                                               
 7546 │ 12954858          9  Under            8                                                                                                               
 7547 │ 12476617          9  Over             7                                                                                                               
 7548 │ 12476617          9  Under            7                                                                                                               
 7549 │ 12476619          9  Over             8                                                                                                               
 7550 │ 12476619          9  Under            8                                                                                                               
 7551 │ 13315347          9  Over             7                                                                                                               
 7552 │ 13315347          9  Under            7                                                                                                               
 7553 │ 13233442          9  Over             7                                                                                                               
 7554 │ 13233442          9  Under            7                                                                                                               
 7555 │ 12811090          9  Over             7                                                                                                               
 7556 │ 12811090          9  Under            7                                                                                                               
 7557 │ 13436762          9  Over             7                                                                                                               
 7558 │ 13436762          9  Under            7                                                                                                               
 7559 │ 13217663          9  Over             7                                                                                                               
 7560 │ 13217663          9  Under            7                                                                                                               
 7561 │ 13255608          9  Over             7                                                                                                               
 7562 │ 13255608          9  Under            7                                                                                                               
 7563 │ 13175771          9  Over             7                                                                                                               
 7564 │ 13175771          9  Under            7                                                                                                               
 7565 │ 13175772          9  Over             7                                                                                                               
 7566 │ 13175772          9  Under            7                                                                                                               
 7567 │ 12476460          9  Over             7                                                                                                               
 7568 │ 12476460          9  Under            7                                                                                                               
 7569 │ 13171688          9  Over             8                                                                                                               
 7570 │ 13171688          9  Under            8                                                                                                               
 7571 │ 13343735          9  Over             7                                                                                                               
 7572 │ 13343735          9  Under            7                                                                                                               
 7573 │ 13343736          9  Over             7                                                                                                               
 7574 │ 13343736          9  Under            7                                                                                                               
                               7509 rows omitted          
=# 


# The TRUE grain of an odds table usually requires the handicap/line
true_odds_keys = [:match_id, :market_id, :choice_name, :choice_group]

# Test the old dataset
old_pk_counts = combine(groupby(ds.odds, true_odds_keys), nrow => :count);
old_suspects = subset(old_pk_counts, :count => ByRow(>(1)));
println("Real PK duplicates in OLD ds: ", nrow(old_suspects))

# Test the new dataset
new_pk_counts = combine(groupby(data_store.odds, true_odds_keys), nrow => :count);
new_suspects = subset(new_pk_counts, :count => ByRow(>(1)));
println("Real PK duplicates in NEW ds: ", nrow(new_suspects))



#=

julia> true_odds_keys = [:match_id, :market_id, :choice_name, :choice_group]
4-element Vector{Symbol}:
 :match_id
 :market_id
 :choice_name
 :choice_group

julia> # Test the old dataset
       old_pk_counts = combine(groupby(ds.odds, true_odds_keys), nrow => :count);

julia> old_suspects = subset(old_pk_counts, :count => ByRow(>(1)));

julia> println("Real PK duplicates in OLD ds: ", nrow(old_suspects))
Real PK duplicates in OLD ds: 42

julia> # Test the new dataset
       new_pk_counts = combine(groupby(data_store.odds, true_odds_keys), nrow => :count);

julia> new_suspects = subset(new_pk_counts, :count => ByRow(>(1)));

julia> println("Real PK duplicates in NEW ds: ", nrow(new_suspects))
Real PK duplicates in NEW ds: 0

=#



# 1. Row counts
println("Old rows: ", nrow(ds.odds), " | New rows: ", nrow(data_store.odds))

# 2. Distinct Matches (Did we drop a whole tournament or season?)
old_matches = unique(ds.odds.match_id);
new_matches = unique(data_store.odds.match_id);

println("Matches uniquely in OLD: ", length(setdiff(old_matches, new_matches)))
println("Matches uniquely in NEW: ", length(setdiff(new_matches, old_matches)))

# 3. Market Coverage (Did our market mapping drop things?)
old_markets = unique(ds.odds.market_id);
new_markets = unique(data_store.odds.market_id);
println("Missing Market IDs in NEW: ", setdiff(old_markets, new_markets))



#= 
julia> println("Old rows: ", nrow(ds.odds), " | New rows: ", nrow(data_store.odds))
Old rows: 132466 | New rows: 66643

julia> old_matches = unique(ds.odds.match_id);

julia> new_matches = unique(data_store.odds.match_id);

julia> println("Matches uniquely in OLD: ", length(setdiff(old_matches, new_matches)))
Matches uniquely in OLD: 1927

julia> println("Matches uniquely in NEW: ", length(setdiff(new_matches, old_matches)))
Matches uniquely in NEW: 17

julia> 

julia> old_markets = unique(ds.odds.market_id);

julia> new_markets = unique(data_store.odds.market_id);

julia> println("Missing Market IDs in NEW: ", setdiff(old_markets, new_markets))
Missing Market IDs in NEW: Union{Missing, Int64}[20]
=#



# Grab only the keys and the value we care about mutating (decimal_odds)
old_subset = select(ds.odds, true_odds_keys..., :decimal_odds => :odds_old);
new_subset = select(data_store.odds, true_odds_keys..., :decimal_odds => :odds_new);

# Outer join aligns them side-by-side
comparison_df = outerjoin(old_subset, new_subset, on=true_odds_keys)

# Categorize the discrepancies
missing_in_new = subset(comparison_df, :odds_new => ByRow(ismissing))
missing_in_old = subset(comparison_df, :odds_old => ByRow(ismissing))

# Find floating point mismatches (using isapprox to avoid 1.50000000001 != 1.5)
mismatched_values = subset(comparison_df, [:odds_old, :odds_new] => ByRow() do o, n
    !ismissing(o) && !ismissing(n) && !isapprox(o, n, atol=1e-4)
end)

println("--- RECONCILIATION SUMMARY ---")
println("Rows exclusively in OLD (Dropped): ", nrow(missing_in_new))
println("Rows exclusively in NEW (Added): ", nrow(missing_in_old))
println("Rows with mismatched odds: ", nrow(mismatched_values))

# Inspect the dropped rows to see if there's a pattern
if nrow(missing_in_new) > 0
    display(combine(groupby(missing_in_new, :market_id), nrow => :dropped_count))
end



#=
julia> old_subset = select(ds.odds, true_odds_keys..., :decimal_odds => :odds_old);

julia> new_subset = select(data_store.odds, true_odds_keys..., :decimal_odds => :odds_new);

julia> comparison_df = outerjoin(old_subset, new_subset, on=true_odds_keys)
ERROR: ArgumentError: Missing values in key columns are not allowed when matchmissing == :error. `missing` found in column :choice_group in left data frame.
Stacktrace:
 [1] DataFrames.DataFrameJoiner(dfl::DataFrame, dfr::DataFrame, on::Vector{Symbol}, matchmissing::Symbol, kind::Symbol)
   @ DataFrames ~/.julia/packages/DataFrames/b4w9K/src/join/composer.jl:85
 [2] _join(df1::DataFrame, df2::DataFrame; on::Vector{Symbol}, kind::Symbol, makeunique::Bool, indicator::Nothing, validate::Tuple{Bool, Bool}, left_rename::Function, right_rename::Function, matchmissing::Symbol, order::Symbol)
   @ DataFrames ~/.julia/packages/DataFrames/b4w9K/src/join/composer.jl:504
 [3] #outerjoin#740
   @ ~/.julia/packages/DataFrames/b4w9K/src/join/composer.jl:1265 [inlined]
 [4] top-level scope
   @ REPL[176]:1


=#


# 1. Isolate the match IDs that exist in the OLD data but were dropped in the NEW data
dropped_match_ids = setdiff(unique(ds.odds.match_id), unique(data_store.odds.match_id));

# 2. Filter your old matches table to ONLY show these dropped matches
dropped_matches_df = subset(ds.matches, :match_id => ByRow(in(dropped_match_ids)));

println("Total dropped matches found in old matches table: ", nrow(dropped_matches_df))

# 3. THE DIAGNOSTICS: Let's group by key columns to see the pattern

# Did the old dataset include Cups or Friendlies? (IDs other than 56 and 57)
if hasproperty(dropped_matches_df, :tournament_id)
    println("\n--- Dropped Matches by Tournament ID ---")
    display(combine(groupby(dropped_matches_df, :tournament_id), nrow => :count))
end

# Did the old dataset include unfinished matches? (Cancelled, Postponed, etc.)
# (Replace :status_type with whatever the status column is called in ds.matches)
if hasproperty(dropped_matches_df, :status_type) || hasproperty(dropped_matches_df, :status)
    status_col = hasproperty(dropped_matches_df, :status_type) ? :status_type : :status
    println("\n--- Dropped Matches by Status ---")
    display(combine(groupby(dropped_matches_df, status_col), nrow => :count))
end

# Did an entire season get dropped?
if hasproperty(dropped_matches_df, :season) || hasproperty(dropped_matches_df, :season_id)
    season_col = hasproperty(dropped_matches_df, :season) ? :season : :season_id
    println("\n--- Dropped Matches by Season ---")
    display(combine(groupby(dropped_matches_df, season_col), nrow => :count))
end


match_footprints = combine(groupby(data_store.odds, :match_id), nrow => :total_odds_rows)
describe(match_footprints)

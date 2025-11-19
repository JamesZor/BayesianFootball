# src/predictions/market-helpers.jl 

#### Get_market is for the closing lines 
function get_market(
    match_id::Int64,
    market::Markets.Market1X2,
    df_odds::AbstractDataFrame
) 
    market_odds = subset(df_odds,
                       :match_id => ByRow(==(match_id)),
                       :market_name => ByRow(==("Full time")),
                       :market_group => ByRow(==("1X2"))
                      )
    odds_map = Dict(market_odds.choice_name .=> market_odds.decimal_odds)

    return (; home=odds_map["1"],
              draw=odds_map["X"],
              away=odds_map["2"])
end

function get_market(
    match_id::Int64,
    market::Markets.MarketOverUnder,
    df_odds::AbstractDataFrame
) 
    market_odds = subset(df_odds,
                       :match_id => ByRow(==(match_id)),
                       :market_name => ByRow(==("Match goals")),
                       :choice_group => ByRow(isequal(market.line))
                      )


    odds_map = Dict(market_odds.choice_name .=> market_odds.decimal_odds)

    line_str = replace(string(market.line), "." => "")
    # e.g., over_key = :over_15
    over_key = Symbol("over_", line_str)
    under_key = Symbol("under_", line_str)
    
  return NamedTuple{(over_key, under_key)}((odds_map["Over"], odds_map["Under"]))

end


function get_market(
    match_id::Int64,
    market::Markets.MarketBTTS,
    df_odds::AbstractDataFrame
) 
    market_odds = subset(df_odds,
                       :match_id => ByRow(==(match_id)),
                       :market_name => ByRow(==("Both teams to score")),
                      )


    odds_map = Dict(market_odds.choice_name .=> market_odds.decimal_odds)

    yes_key = Symbol("btts_yes")
    no_key = Symbol("btts_no")
    
  return NamedTuple{(yes_key, no_key)}((odds_map["Yes"], odds_map["No"]))

end

function get_market(
  match_id::Int64,
  predict_config::PredictionConfig,
  df_odds::AbstractDataFrame 
  )

  market_odds_generator = (
    get_market(match_id, market, df_odds) for market in predict_config.markets
  )

  match_odds = reduce(merge, market_odds_generator; init = (;) );

  return match_odds
end

##################
# opening lines
##################

function get_market_opening_lines(
    match_id::Int64,
    market::Markets.Market1X2,
    df_odds::AbstractDataFrame
) 
    market_odds = subset(df_odds,
                       :match_id => ByRow(==(match_id)),
                       :market_name => ByRow(==("Full time")),
                       :market_group => ByRow(==("1X2"))
                      )
    odds_map = Dict(market_odds.choice_name .=> market_odds.initial_decimal)

    return (; home=odds_map["1"],
              draw=odds_map["X"],
              away=odds_map["2"])
end

function get_market_opening_lines(
    match_id::Int64,
    market::Markets.MarketOverUnder,
    df_odds::AbstractDataFrame
) 
    market_odds = subset(df_odds,
                       :match_id => ByRow(==(match_id)),
                       :market_name => ByRow(==("Match goals")),
                       :choice_group => ByRow(isequal(market.line))
                      )


    odds_map = Dict(market_odds.choice_name .=> market_odds.initial_decimal)

    line_str = replace(string(market.line), "." => "")
    # e.g., over_key = :over_15
    over_key = Symbol("over_", line_str)
    under_key = Symbol("under_", line_str)
    
  return NamedTuple{(over_key, under_key)}((odds_map["Over"], odds_map["Under"]))

end


function get_market_opening_lines(
    match_id::Int64,
    market::Markets.MarketBTTS,
    df_odds::AbstractDataFrame
) 
    market_odds = subset(df_odds,
                       :match_id => ByRow(==(match_id)),
                       :market_name => ByRow(==("Both teams to score")),
                      )


    odds_map = Dict(market_odds.choice_name .=> market_odds.initial_decimal)

    yes_key = Symbol("btts_yes")
    no_key = Symbol("btts_no")
    
  return NamedTuple{(yes_key, no_key)}((odds_map["Yes"], odds_map["No"]))

end

function get_market_opening_lines(
  match_id::Int64,
  predict_config::PredictionConfig,
  df_odds::AbstractDataFrame 
  )

  market_odds_generator = (
    get_market_opening_lines(match_id, market, df_odds) for market in predict_config.markets
  )

  match_odds = reduce(merge, market_odds_generator; init = (;) );

  return match_odds
end


##################
# results helpers
##################
# currently for bet365 bits

function get_market_results(
    match_id::Int64,
    market::Markets.Market1X2,
    df_odds::AbstractDataFrame
) 
    market_odds = subset(df_odds,
                       :match_id => ByRow(==(match_id)),
                       :market_name => ByRow(==("Full time")),
                       :market_group => ByRow(==("1X2"))
                      )
    odds_map = Dict(market_odds.choice_name .=> market_odds.winning)

    return (; home=odds_map["1"],
              draw=odds_map["X"],
              away=odds_map["2"])
end


function get_market_results(
    match_id::Int64,
    market::Markets.MarketOverUnder,
    df_odds::AbstractDataFrame
) 
    market_odds = subset(df_odds,
                       :match_id => ByRow(==(match_id)),
                       :market_name => ByRow(==("Match goals")),
                       :choice_group => ByRow(isequal(market.line))
                      )


    odds_map = Dict(market_odds.choice_name .=> market_odds.winning)

    line_str = replace(string(market.line), "." => "")
    # e.g., over_key = :over_15
    over_key = Symbol("over_", line_str)
    under_key = Symbol("under_", line_str)
    
  return NamedTuple{(over_key, under_key)}((odds_map["Over"], odds_map["Under"]))
end

function get_market_results(
    match_id::Int64,
    market::Markets.MarketBTTS,
    df_odds::AbstractDataFrame
) 
    market_odds = subset(df_odds,
                       :match_id => ByRow(==(match_id)),
                       :market_name => ByRow(==("Both teams to score")),
                      )


    odds_map = Dict(market_odds.choice_name .=> market_odds.winning)

    yes_key = Symbol("btts_yes")
    no_key = Symbol("btts_no")
    
  return NamedTuple{(yes_key, no_key)}((odds_map["Yes"], odds_map["No"]))

end


function get_market_results(
  match_id::Int64,
  predict_config::PredictionConfig,
  df_odds::AbstractDataFrame 
  )

  market_odds_generator = (
    get_market_results(match_id, market, df_odds) for market in predict_config.markets
  )

  match_odds = reduce(merge, market_odds_generator; init = (;) );

  return match_odds
end



# ---------------------------------------------------------
# 1X2 Market
# ---------------------------------------------------------
function get_market_data(
    match_id::Int64,
    market::Markets.Market1X2,
    df_odds::AbstractDataFrame
) 
    # 1. Fetch Keys from centralized logic
    mk = Markets.market_keys(market) # returns (home=:home, draw=:draw, away=:away)

    # 2. Filter Data
    market_odds = subset(df_odds, 
        :match_id => ByRow(==(match_id)), 
        :market_name => ByRow(==("Full time")),
        :market_group => ByRow(==("1X2"))
    )

    row_map = Dict(row.choice_name => row for row in eachrow(market_odds))

    # 3. Map DB choices ("1", "X", "2") to the keys provided by Markets.jl
    # We assume rows exist. If safety is needed, use get(row_map, "1", default_row)
    r1 = row_map["1"]
    rx = row_map["X"]
    r2 = row_map["2"]

    return (
        opening = (; mk.home => r1.initial_decimal, mk.draw => rx.initial_decimal, mk.away => r2.initial_decimal),
        closing = (; mk.home => r1.decimal_odds,    mk.draw => rx.decimal_odds,    mk.away => r2.decimal_odds),
        result  = (; mk.home => r1.winning,         mk.draw => rx.winning,         mk.away => r2.winning)
    )
end

# ---------------------------------------------------------
# Over/Under Market
# ---------------------------------------------------------
function get_market_data(
    match_id::Int64,
    market::Markets.MarketOverUnder,
    df_odds::AbstractDataFrame
) 
    # 1. Fetch Keys (e.g., over=:over_25, under=:under_25)
    mk = Markets.market_keys(market)

    # 2. Filter Data
    market_odds = subset(df_odds, 
        :match_id => ByRow(==(match_id)), 
        :market_name => ByRow(==("Match goals")),
        :choice_group => ByRow(isequal(market.line))
    )

    row_map = Dict(row.choice_name => row for row in eachrow(market_odds))

    # 3. Map DB choices ("Over", "Under") to the dynamic keys
    r_over  = row_map["Over"]
    r_under = row_map["Under"]

    return (
        opening = (; mk.over => r_over.initial_decimal, mk.under => r_under.initial_decimal),
        closing = (; mk.over => r_over.decimal_odds,    mk.under => r_under.decimal_odds),
        result  = (; mk.over => r_over.winning,         mk.under => r_under.winning)
    )
end

# ---------------------------------------------------------
# BTTS Market
# ---------------------------------------------------------
function get_market_data(
    match_id::Int64,
    market::Markets.MarketBTTS,
    df_odds::AbstractDataFrame
) 
    # 1. Fetch Keys (yes=:btts_yes, no=:btts_no)
    mk = Markets.market_keys(market)

    # 2. Filter Data
    market_odds = subset(df_odds, 
        :match_id => ByRow(==(match_id)), 
        :market_name => ByRow(==("Both teams to score")),
    )

    row_map = Dict(row.choice_name => row for row in eachrow(market_odds))

    # 3. Map DB choices ("Yes", "No") to keys
    r_yes = row_map["Yes"]
    r_no  = row_map["No"]

    return (
        opening = (; mk.yes => r_yes.initial_decimal, mk.no => r_no.initial_decimal),
        closing = (; mk.yes => r_yes.decimal_odds,    mk.no => r_no.decimal_odds),
        result  = (; mk.yes => r_yes.winning,         mk.no => r_no.winning)
    )
end

# ---------------------------------------------------------
# Wrapper (Unchanged logic, just context)
# ---------------------------------------------------------
function get_market_data(
    match_id::Int64,
    predict_config::PredictionConfig,
    df_odds::AbstractDataFrame 
)
    market_data_gen = (
        get_market_data(match_id, market, df_odds) for market in predict_config.markets
    )

    combined = reduce(market_data_gen; init = (opening=(;), closing=(;), result=(;))) do acc, next_item
        (
            opening = merge(acc.opening, next_item.opening),
            closing = merge(acc.closing, next_item.closing),
            result  = merge(acc.result,  next_item.result)
        )
    end

    return (combined.opening, combined.closing, combined.result)
end




####

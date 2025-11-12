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










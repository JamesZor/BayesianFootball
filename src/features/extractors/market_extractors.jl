# src/features/extractors/market_extractors.jl

# 1. Market Lambdas (Solves for implied home/away goals using odds)
function add_feature!(F_data::Dict, ::MarketLambdaFeature, ordered_ids, team_map::Dict, ds::Data.DataStore)
    # We rely on market_inverse_utils.jl which is included in the module entry point
    
    id_set = Set(ordered_ids)
    filtered_odds = subset(ds.odds, :match_id => ByRow(in(id_set)))
    odds_by_match = groupby(filtered_odds, :match_id)
    n_matches = length(odds_by_match)
    
    thread_results = Vector{Tuple{Int, Float64, Float64, Float64}}(undef, n_matches)
    
    @threads for i in 1:n_matches
        match_df = odds_by_match[i]
        # fit_market_implied_parameters is defined in market_inverse_utils.jl
        res = fit_market_implied_parameters(match_df)
        thread_results[i] = (res.match_id, res.λ_home, res.λ_away, res.ρ)
    end
    
    market_map = Dict{Int, NTuple{3, Float64}}(
        r[1] => (r[2], r[3], r[4]) for r in thread_results
    )

    F_data[:flat_market_λ_home] = [get(market_map, id, (NaN, NaN, NaN))[1] for id in ordered_ids]
    F_data[:flat_market_λ_away] = [get(market_map, id, (NaN, NaN, NaN))[2] for id in ordered_ids]
    F_data[:flat_market_ρ]      = [get(market_map, id, (NaN, NaN, NaN))[3] for id in ordered_ids]
end

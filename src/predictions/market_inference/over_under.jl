# src/predictions/market_inference/over_under.jl

using ..Data: MarketOverUnder, outcomes

function compute_market_probs(S::ScoreMatrix, market::MarketOverUnder)
    # S.data is [HomeGoals, AwayGoals, Samples]
    (max_h, max_a, n_samples) = size(S.data)
    target_line = market.line
    
    over_prob = zeros(Float64, n_samples)
    under_prob = zeros(Float64, n_samples)
    
    @inbounds for k in 1:n_samples
        mat = view(S.data, :, :, k)
        
        for c in 1:max_a # Away Cols
            for r in 1:max_h # Home Rows
                prob = mat[r, c]
                
                # Map indices to goals (index 1 = 0 goals)
                h_goals = r - 1
                a_goals = c - 1
                total_goals = h_goals + a_goals
                
                if total_goals > target_line
                    over_prob[k] += prob
                elseif total_goals < target_line
                    under_prob[k] += prob
                end
                # Exact equality (integer lines) is usually a push (void), 
                # effectively ignored here or you could split it 50/50.
            end
        end
    end
    
    # Dynamic Keys: outcomes returns (over=:over_25, under=:under_25)
    keys = outcomes(market)
    
    return Dict(
        keys.over => over_prob,
        keys.under => under_prob
    )
end

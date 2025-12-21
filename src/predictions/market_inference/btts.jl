# src/predictions/market_inference/btts.jl

using ..Data: MarketBTTS, outcomes

function compute_market_probs(S::ScoreMatrix, market::MarketBTTS)
    (max_h, max_a, n_samples) = size(S.data)
    
    yes_prob = zeros(Float64, n_samples)
    no_prob = zeros(Float64, n_samples)
    
    @inbounds for k in 1:n_samples
        mat = view(S.data, :, :, k)
        
        for c in 1:max_a
            for r in 1:max_h
                prob = mat[r, c]
                
                h_goals = r - 1
                a_goals = c - 1
                
                # BTTS Condition: Both > 0
                if h_goals > 0 && a_goals > 0
                    yes_prob[k] += prob
                else
                    # Either 0-0, 1-0, 0-1, etc.
                    no_prob[k] += prob
                end
            end
        end
    end
    
    # Dynamic Keys: outcomes returns (yes=:yes, no=:no)
    keys = outcomes(market)
    
    return Dict(
        keys.yes => yes_prob,
        keys.no => no_prob
    )
end

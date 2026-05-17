# src/predictions/market_inference/double_chance.jl

using ..Data: MarketDC, outcomes

function compute_market_probs(S::ScoreMatrix, market::MarketDC)
    # S.data is [HomeGoals, AwayGoals, Samples]
    (max_h, max_a, n_samples) = size(S.data)
    
    prob_1x = zeros(Float64, n_samples)
    prob_x2 = zeros(Float64, n_samples)
    prob_12 = zeros(Float64, n_samples)
    
    @inbounds for k in 1:n_samples
        # Use scalar accumulators for the base outcomes
        ph = 0.0
        pd = 0.0
        pa = 0.0

        # Iterate columns (Away) first for column-major contiguous memory access
        for c in 1:max_a
            # --- Region 1: Away Wins (Home < Away) ---
            limit_away = min(c - 1, max_h)
            for r in 1:limit_away
                pa += S.data[r, c, k]
            end
            
            # --- Region 2: Draw (Home == Away) ---
            if c <= max_h
                pd += S.data[c, c, k]
            end
            
            # --- Region 3: Home Wins (Home > Away) ---
            for r in (c + 1):max_h
                ph += S.data[r, c, k]
            end
        end

        # Combine the base sums for Double Chance outcomes
        prob_1x[k] = ph + pd  # Home or Draw
        prob_x2[k] = pd + pa  # Draw or Away
        prob_12[k] = ph + pa  # Home or Away
    end

    keys = outcomes(market)

    return Dict(
        keys.dc1x => prob_1x,
        keys.dcx2 => prob_x2,
        keys.dc12 => prob_12
    )
end

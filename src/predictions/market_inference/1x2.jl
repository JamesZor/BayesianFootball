# src/predictions/market_inference/1x2.jl

using ..Data: Market1X2

# function compute_market_probs(S::ScoreMatrix, market::Market1x2)
#     # S.data is [HomeGoals, AwayGoals, Samples]
#     (max_h, max_a, n_samples) = size(S.data)
#
#     home_prob = zeros(Float64, n_samples)
#     draw_prob = zeros(Float64, n_samples)
#     away_prob = zeros(Float64, n_samples)
#
#     @inbounds for k in 1:n_samples
#         # Get the matrix for this specific sample chain step
#         mat = view(S.data, :, :, k)
#
#         for c in 1:max_a # Away Cols
#             for r in 1:max_h # Home Rows
#                 prob = mat[r, c]
#
#                 # Map indices to goals: index 1 is 0 goals
#                 h_goals = r - 1
#                 a_goals = c - 1
#
#                 if h_goals > a_goals
#                     home_prob[k] += prob
#                 elseif h_goals == a_goals
#                     draw_prob[k] += prob
#                 else
#                     away_prob[k] += prob
#                 end
#             end
#         end
#     end
#
#     # Return Dict of Outcome Name => Vector of Samples
#     return Dict(
#         "Home" => home_prob,
#         "Draw" => draw_prob,
#         "Away" => away_prob
#     )
# end
#

function compute_market_probs(S::ScoreMatrix, market::Market1X2)
    # S.data is [HomeGoals, AwayGoals, Samples]
    (max_h, max_a, n_samples) = size(S.data)
    
    home_prob = zeros(Float64, n_samples)
    draw_prob = zeros(Float64, n_samples)
    away_prob = zeros(Float64, n_samples)
    
    @inbounds for k in 1:n_samples
        # Use scalar accumulators to avoid repeated array indexing into result vectors
        ph = 0.0
        pd = 0.0
        pa = 0.0

        # We iterate columns (Away) first because Julia is Column-Major.
        # This ensures the inner loops over rows (Home) read contiguous memory.
        for c in 1:max_a
            # --- Region 1: Away Wins (Home < Away) ---
            # Rows 1 to c-1
            # We use min() to ensure we don't exceed matrix bounds if rectangular
            limit_away = min(c - 1, max_h)
            for r in 1:limit_away
                pa += S.data[r, c, k]
            end
            
            # --- Region 2: Draw (Home == Away) ---
            # Row c
            if c <= max_h
                pd += S.data[c, c, k]
            end
            
            # --- Region 3: Home Wins (Home > Away) ---
            # Rows c+1 to End
            for r in (c + 1):max_h
                ph += S.data[r, c, k]
            end
        end

        # Assign final sums to output arrays
        home_prob[k] = ph
        draw_prob[k] = pd
        away_prob[k] = pa
    end
    
    return Dict(
        "Home" => home_prob,
        "Draw" => draw_prob,
        "Away" => away_prob
    )
end

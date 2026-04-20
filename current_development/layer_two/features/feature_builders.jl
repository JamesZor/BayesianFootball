# --- 1. Season Progress Builder ---
Base.@kwdef struct SeasonProgress <: AbstractFeatureBuilder end

function generate_features(::SeasonProgress, matches_df::DataFrame)
    df = select(matches_df, :match_id, :match_date, :season, :home_team, :away_team)
    
    ledger = create_team_ledger(matches_df)
    
    # Count how many games each team has played in the current season up to this date
    transform!(groupby(ledger, [:season, :team]), 
        :match_id => (x -> 1:length(x)) => :games_played
    )
    
    # Pivot back to Match Level
    home_prog = select(ledger, :match_id, :team => :home_team, :games_played => :home_games_played)
    away_prog = select(ledger, :match_id, :team => :away_team, :games_played => :away_games_played)
    
    res = innerjoin(home_prog, away_prog, on=:match_id)
    
    # Progress = Games Played / 38 (assuming roughly 38 games in a season)
    res.season_progress = (res.home_games_played .+ res.away_games_played) ./ (36.0 * 2.0)
    
    return select(res, :match_id, :season_progress)
end

# --- 2. Rolling Window Builder (Goals & Points) ---
Base.@kwdef struct RollingForm <: AbstractFeatureBuilder
    window_size::Int = 5
end

function generate_features(builder::RollingForm, matches_df::DataFrame)
    ledger = create_team_ledger(matches_df)
    w = builder.window_size
    
    # 1. Calculate the rolling sums
    # CRITICAL: We use `ShiftedArrays.lag` to shift the window back by 1. 
    # If we don't do this, we are using the goals scored IN Match 10 to predict Match 10 (Data Leakage!)
    transform!(groupby(ledger, :team),
        :scored   => (x -> lag(rolling_sum(x, w))) => :rolling_scored,
        :conceded => (x -> lag(rolling_sum(x, w))) => :rolling_conceded,
        :points   => (x -> lag(rolling_sum(x, w))) => :rolling_points
    )
    
    # Helper to calculate a simple rolling sum (can use RollingFunctions.jl in production)
    function rolling_sum(v::AbstractVector, window::Int)
        res = Vector{Union{Missing, Float64}}(missing, length(v))
        for i in 1:length(v)
            if i >= window && !any(ismissing, v[i-window+1:i])
                res[i] = sum(v[i-window+1:i])
            end
        end
        return res
    end
    
    # 2. Pivot back to Match Level
    home_form = select(ledger, :match_id, :team => :home_team, 
                       :rolling_scored => :home_scored_last_N,
                       :rolling_points => :home_points_last_N)
                       
    away_form = select(ledger, :match_id, :team => :away_team, 
                       :rolling_scored => :away_scored_last_N,
                       :rolling_points => :away_points_last_N)
                       
    res = innerjoin(home_form, away_form, on=:match_id)
    
    # Optional: Create differential features (Often highly predictive for XGBoost)
    res.form_points_diff = res.home_points_last_N .- res.away_points_last_N
    
    # Rename generic N to specific window size for column clarity
    rename!(res, 
        :home_scored_last_N => Symbol("home_scored_last_$w"),
        :away_scored_last_N => Symbol("away_scored_last_$w")
    )
    
    return res
end

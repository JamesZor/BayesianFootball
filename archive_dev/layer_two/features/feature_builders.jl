# --- 1. Season Progress Builder ---
Base.@kwdef struct SeasonProgress <: AbstractFeatureBuilder end

function generate_features(::SeasonProgress, matches_df::DataFrame)
    ledger = create_team_ledger(matches_df)
    
    transform!(groupby(ledger, [:season, :team]), 
        :match_id => (x -> 1:length(x)) => :games_played
    )
    
    # FIX: Explicitly filter by the is_home flag before selecting!
    home_prog = select(filter(:is_home => ==(true), ledger), 
                       :match_id, :games_played => :home_games_played)
                       
    away_prog = select(filter(:is_home => ==(false), ledger), 
                       :match_id, :games_played => :away_games_played)
    
    res = innerjoin(home_prog, away_prog, on=:match_id)
    res.season_progress = (res.home_games_played .+ res.away_games_played) ./ (38.0 * 2.0)
    
    return select(res, :match_id, :season_progress)
end

# --- 2. Rolling Window Builder (Goals & Points) ---
Base.@kwdef struct RollingForm <: AbstractFeatureBuilder
    window_size::Int = 5
end

function generate_features(builder::RollingForm, matches_df::DataFrame)
    ledger = create_team_ledger(matches_df)
    w = builder.window_size
    
    # FIX: Native Lagged Rolling Sum
    # By looping from (i-window) to (i-1), we guarantee no data leakage
    # without needing the ShiftedArrays package!
    
    # We can now apply it cleanly without the lag() wrapper
    transform!(groupby(ledger, :team),
        :scored   => (x -> rolling_sum_lagged(x, w)) => :rolling_scored,
        :conceded => (x -> rolling_sum_lagged(x, w)) => :rolling_conceded,
        :points   => (x -> rolling_sum_lagged(x, w)) => :rolling_points
    )
    
    home_form = select(filter(:is_home => ==(true), ledger), :match_id, 
                       :rolling_scored => :home_scored_last_N,
                       :rolling_points => :home_points_last_N)
                       
    away_form = select(filter(:is_home => ==(false), ledger), :match_id, 
                       :rolling_scored => :away_scored_last_N,
                       :rolling_points => :away_points_last_N)
                       
    res = innerjoin(home_form, away_form, on=:match_id)
    
    res.form_points_diff = res.home_points_last_N .- res.away_points_last_N
    
    rename!(res, 
        :home_scored_last_N => Symbol("home_scored_last_$w"),
        :away_scored_last_N => Symbol("away_scored_last_$w"),
        :home_points_last_N => Symbol("home_points_last_$w"),
        :away_points_last_N => Symbol("away_points_last_$w"),
        :form_points_diff   => Symbol("form_points_diff_$w")
    )
    
    return res
end

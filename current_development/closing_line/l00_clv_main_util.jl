using JSON
using DataFrames
using Dates

# ====================================================================
# PHASE 1: Order Book Extraction & Time Series Parsing
# ====================================================================

function parse_order_book_timeline(filepath::String, target_market::String="Over/Under 2.5 Goals")::DataFrame
    results = []

    for line in eachline(filepath)
        strip(line) == "" && continue
        
        data = JSON.parse(line)
        ts_str = get(data, "timestamp", missing)
        home_team = get(data, "home_team", missing)
        
        market_data = get(data, "market_data", Dict())
        ft_data = get(market_data, "ft", Dict())
        
        if haskey(ft_data, target_market)
            market_selections = ft_data[target_market]
            
            for (selection_name, book) in market_selections
                back_dict = book["back"] isa AbstractDict ? book["back"] : Dict()
                lay_dict = book["lay"] isa AbstractDict ? book["lay"] : Dict()
                
                # FIX: Default to `nothing` to catch both missing keys and JSON nulls
                b_price = get(back_dict, "price", nothing)
                b_size  = get(back_dict, "size", nothing)
                l_price = get(lay_dict, "price", nothing)
                l_size  = get(lay_dict, "size", nothing)
                
                push!(results, (
                    timestamp = ts_str,
                    home_team = home_team,
                    market = target_market,
                    selection = selection_name, 
                    back_price = isnothing(b_price) ? missing : Float64(b_price),
                    back_size  = isnothing(b_size)  ? missing : Float64(b_size),
                    lay_price  = isnothing(l_price) ? missing : Float64(l_price),
                    lay_size   = isnothing(l_size)  ? missing : Float64(l_size)
                ))
            end
        end
    end
    
    df = DataFrame(results)
    
    if !isempty(df)
        df.timestamp = DateTime.(first.(df.timestamp, 23), "yyyy-mm-ddTHH:MM:SS.sss")
        
        # Static Mapping for Goal Markets to your internal symbols
        mapping = Dict(
            "Over 0.5 Goals" => :over_05, "Under 0.5 Goals" => :under_05,
            "Over 1.5 Goals" => :over_15, "Under 1.5 Goals" => :under_15,
            "Over 2.5 Goals" => :over_25, "Under 2.5 Goals" => :under_25,
            "Over 3.5 Goals" => :over_35, "Under 3.5 Goals" => :under_35
        )
        
        # Create a new column with the clean symbols
        df.selection_sym = [get(mapping, s, Symbol(replace(lowercase(s), " " => "_"))) for s in df.selection]
        
        sort!(df, [:home_team, :selection_sym, :timestamp])
    end
    
    return df
end

# ====================================================================
# PHASE 2: Quantitative Market Metrics
# ====================================================================

"""
    add_market_metrics!(df::DataFrame)

Calculates the Mid-Price and Spread Percentage. 
Only calculates if both sides of the book (back and lay) exist.
"""
function add_market_metrics!(df::DataFrame)
    # 1. Mid Price: The true market implied probability (inverted)
    df.mid_price = map(eachrow(df)) do r
        if ismissing(r.back_price) || ismissing(r.lay_price)
            return missing
        end
        return (r.back_price + r.lay_price) / 2.0
    end
    
    # 2. Spread Percentage: width / mid_price (Measures liquidity/defensiveness)
    df.spread_pct = map(eachrow(df)) do r
        if ismissing(r.back_price) || ismissing(r.lay_price) || ismissing(r.mid_price)
            return missing
        end
        return ((r.lay_price - r.back_price) / r.mid_price) * 100
    end
    
    return df
end

# ====================================================================
# PHASE 3: Edge Trajectory & Stake Simulation
# ====================================================================

using Statistics # Required for the mean() function

# ====================================================================
# PHASE 3: Edge Trajectory & Stake Simulation
# ====================================================================

"""
    calculate_time_series_stakes(ts_df::DataFrame, target_ppd, calib_ppd, todays_matches; min_edge::Float64=0.0)

Takes the time-series order book DataFrame and calculates the Kelly Stake
at every timestamp using both the Raw and Calibrated PPD distributions.
Adds the Model's Fair Odds for direct comparison.
"""
function calculate_time_series_stakes(ts_df::DataFrame, 
                                      target_ppd::BayesianFootball.Predictions.PPD, 
                                      calib_ppd::BayesianFootball.Predictions.PPD, 
                                      todays_matches::DataFrame; 
                                      min_edge::Float64=0.0)::DataFrame
    
    signal = BayesianFootball.Signals.BayesianKelly(min_edge=min_edge)
    ppds_compare = compare_calibrated_odds(target_ppd, calib_ppd, todays_matches)
    
    df_join = copy(ts_df)
    select!(df_join, Not(:selection))
    rename!(df_join, :selection_sym => :selection)
    
    preds_subset = select(ppds_compare, :home_team, :selection, :raw_distribution, :calib_distribution)
    merged_df = innerjoin(df_join, preds_subset, on=[:home_team, :selection])
    
    # NEW: Calculate the Model's Fair Odds
    merged_df.raw_fair_odds = map(r -> 1.0 / mean(r.raw_distribution), eachrow(merged_df))
    merged_df.calib_fair_odds = map(r -> 1.0 / mean(r.calib_distribution), eachrow(merged_df))
    
    merged_df.raw_stake_pct = map(eachrow(merged_df)) do r
        if ismissing(r.back_price)
            return 0.0
        end
        stake = BayesianFootball.Signals.compute_stake(signal, r.raw_distribution, r.back_price)
        return max(0.0, stake) * 100 
    end
    
    merged_df.calib_stake_pct = map(eachrow(merged_df)) do r
        if ismissing(r.back_price)
            return 0.0
        end
        stake = BayesianFootball.Signals.compute_stake(signal, r.calib_distribution, r.back_price)
        return max(0.0, stake) * 100
    end
    
    select!(merged_df, Not([:raw_distribution, :calib_distribution]))
    rename!(merged_df, :selection => :selection_sym)
    sort!(merged_df, [:home_team, :selection_sym, :timestamp])
    
    return merged_df
end




# --- plotting 
using Plots
plotlyjs()
using Statistics
using Dates # Needed for DateTime formatting

"""
    plot_price_discovery(traj_df, target_ppd, todays_matches, team_name, sel_sym, output_dir)

Plots the live market order book (Back, Lay, Mid) against a gradient heatmap 
of the model's Posterior Predictive Distribution (PPD) over time.
"""
function plot_price_discovery(traj_df::DataFrame, 
                              target_ppd::BayesianFootball.Predictions.PPD, 
                              todays_matches::DataFrame, 
                              team_name::String, 
                              sel_sym::Symbol, 
                              label::String,
                              output_dir::String=".")
    
    # 1. Filter trajectory
    df_plot = filter(r -> r.home_team == team_name && r.selection_sym == sel_sym, traj_df)
    if isempty(df_plot)
        @warn "No trajectory data found for $team_name - $sel_sym"
        return
    end
    
    # 2. Get match info
    match_info = filter(r -> r.home_team == team_name, todays_matches)
    m_id = match_info.match_id[1]
    
    # 3. Extract PPD
    ppd_rows = filter(r -> r.match_id == m_id && r.selection == sel_sym, target_ppd.df)
    prob_dist = ppd_rows.distribution[1]
    odds_dist = 1.0 ./ prob_dist
    
    # 4. Lock the X-axis and create clean HH:MM ticks
    min_time = minimum(df_plot.timestamp)
    max_time = maximum(df_plot.timestamp)

    # Generate 6 evenly spaced time markers
    total_ms = Dates.value(max_time - min_time)
    t_ticks = [min_time + Dates.Millisecond(round(Int, i * total_ms / 5)) for i in 0:5]
    # Format them to drop the year/date (e.g., "13:30")
    t_labels = Dates.format.(t_ticks, "HH:MM")
    
    # 5. Initialize Plot with Margins and Custom Ticks
    p = plot(title="Price Discovery vs Model CLV: $(uppercase(team_name)) [$(sel_sym)] - $(label)",
             xlabel="Time (Pre-Match)", ylabel="Implied Odds",
             legend=:topright, size=(1200, 700),
             grid=false,
             xlims=(min_time, max_time),
             xticks=(t_ticks, t_labels),      # <--- Applies the clean time labels
             left_margin=60Plots.px,          # <--- Pushes chart right so title isn't chopped
             bottom_margin=30Plots.px)
             
    # 6. Build Gradient Heatmap
    for i in 1:20
        tail_pct = 0.5 * (i / 20.0) 
        lower_bound = quantile(odds_dist, 0.5 - tail_pct)
        upper_bound = quantile(odds_dist, 0.5 + tail_pct)
        
        lbl = i == 20 ? "Model PPD Density" : ""
        hspan!(p, [lower_bound, upper_bound], color=:dodgerblue, alpha=0.04, label=lbl, linecolor=:transparent)
    end
    
    # 7. Add Model Summary Statistics
    mean_odds = mean(odds_dist)
    p25 = quantile(odds_dist, 0.25)
    p75 = quantile(odds_dist, 0.75)
    
    hline!(p, [mean_odds], color=:navy, linewidth=2, linestyle=:dash, label="Model Mean Fair Odds")
    hline!(p, [p25], color=:navy, linewidth=1.5, linestyle=:dot, label="50% Lower/Upper CI")
    hline!(p, [p75], color=:navy, linewidth=1.5, linestyle=:dot, label="")
    
    # 8. THE FIX: Convert `missing` to `NaN` so Plotly correctly draws the traces
    lay_clean = coalesce.(df_plot.lay_price, NaN)
    back_clean = coalesce.(df_plot.back_price, NaN)
    mid_clean = coalesce.(df_plot.mid_price, NaN)
    
    # 9. Plot Market Action
    plot!(p, df_plot.timestamp, lay_clean, label="Lay Price (Market Ask)", color=:red, linewidth=2.5)
    plot!(p, df_plot.timestamp, back_clean, label="Back Price (Market Bid)", color=:green, linewidth=2.5)
    plot!(p, df_plot.timestamp, mid_clean, label="Mid Price (True Market Belief)", color=:orange, linewidth=2.5, linestyle=:dashdot)
    
    # 10. Dynamically zoom the Y-axis
    max_market = maximum(skipmissing(df_plot.lay_price))
    min_market = minimum(skipmissing(df_plot.back_price))
    max_model = quantile(odds_dist, 0.95)
    min_model = quantile(odds_dist, 0.05)
    
    y_upper = max(max_market, max_model) * 1.05 
    y_lower = min(min_market, min_model) * 0.95
    ylims!(p, (y_lower, y_upper))

    # 11. Save
    mkpath(output_dir)
    filename = joinpath(output_dir, "clv_trajectory_$(team_name)_$(sel_sym)_$(label).html")
    savefig(p, filename)
    println("✅ Plot successfully generated and saved to: ", filename)
    
    return p
end

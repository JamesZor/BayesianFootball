# current_development/match_day_inference/r01_compare_lineups.jl

# ==========================================
# 1. ENVIRONMENT & IMPORTS
# ==========================================
using Pkg
Pkg.activate(".")

using Revise
# Setup CPU thread pinning
using ThreadPinning
pinthreads(:cores)

# Include loader (assumes we are in BayesianFootball workspace)
include("loader.jl")
include("./current_development/match_day_inference/loader.jl")

using DataFrames
using PrettyTables
using Statistics
using Dates

# ==========================================
# 2. LOAD DATASTORE & EXPERIMENT
# ==========================================
println("\n=== 1. Loading Datastore & Experiment ===")
ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.Ireland())

save_dir::String = "./data/ab_test_hierarchical_player/"
saved_files = Experiments.list_experiments(save_dir, data_dir="")
if isempty(saved_files)
    error("No saved experiments found in $save_dir")
end
expr = Experiments.load_experiment(saved_files, 2)

model = expr.config.model
tracker = model.player_ratings_feature.tracker

# Extract current player ratings from historical lineups
player_ratings, global_avg = get_latest_player_ratings(ds, tracker)

# ==========================================
# 3. FETCH TODAY'S MATCHES
# ==========================================
println("\n=== 2. Fetching Today's Fixtures ===")
todays_matches = fetch_todays_matches(ds)
if isempty(todays_matches)
    println("⚠️ No matches scheduled for today!")
    exit(0)
end

println("Fetched $(nrow(todays_matches)) matches for today:")
for row in eachrow(todays_matches)
    println("  - [$(row.match_id)] $(row.home_team) vs $(row.away_team)")
end

# Helper to clean position
function clean_pos(pos::String)
    if pos == "G" || pos == "Goalkeeper" || pos == "GK"
        return "G"
    elseif pos == "D" || pos == "Defender" || pos == "DF"
        return "D"
    elseif pos == "M" || pos == "Midfielder" || pos == "MF"
        return "M"
    elseif pos == "F" || pos == "Forward" || pos == "FW" || pos == "A"
        return "F"
    else
        return "M" # Default to Midfielder
    end
end

# ==========================================
# 4. LINEUP COMPARISON PIPELINE
# ==========================================
println("\n=== 3. Running Lineup Comparison & Analysis ===")

for row in eachrow(todays_matches)
    mid = Int(row.match_id)
    home = String(row.home_team)
    away = String(row.away_team)
    
    println("\n" * "="^80)
    println(" ⚽ MATCH: $home vs $away (ID: $mid)")
    println("="^80)
    
    # 1. Fetch live lineup from Sofascore
    print("  └─ Fetching live lineup from Sofascore API... ")
    live = fetch_lineup_from_sofascore(mid)
    if isnothing(live)
        println("❌ Not available (Match may be too far in the future or lineups not released yet).")
        println("     Comparison is skipped for this fixture.")
        continue
    end
    status_str = live.confirmed ? "✅ CONFIRMED LINEUP" : "⏳ PROVISIONAL LINEUP"
    println("Success! ($status_str)")
    
    # 2. Get fallback lineups from database
    fallback_home = get_most_recent_lineup(ds, home)
    fallback_away = get_most_recent_lineup(ds, away)
    
    # Analyze home and away sides
    for (side, live_players, fallback_players, team_name) in [
        ("Home", live.home, fallback_home, home),
        ("Away", live.away, fallback_away, away)
    ]
        println("\n  --- $side Team: $team_name ---")
        
        # Filter to starters
        live_starters = filter(p -> !p.substitute, live_players)
        fallback_starters = filter(p -> !p.substitute, fallback_players)
        
        if isempty(live_starters)
            println("    ⚠️ Live starters list is empty.")
            continue
        end
        if isempty(fallback_starters)
            println("    ⚠️ Fallback starters list is empty (no history).")
            continue
        end
        
        # Calculate overlap
        live_ids = [p.player_id for p in live_starters]
        fallback_ids = [p.player_id for p in fallback_starters]
        
        overlap_ids = intersect(live_ids, fallback_ids)
        overlap_count = length(overlap_ids)
        overlap_pct = (overlap_count / max(1, length(live_starters))) * 100
        
        # Rotated in (in live starters, but not in fallback starters)
        rotated_in = filter(p -> p.player_id ∉ fallback_ids, live_starters)
        
        # Print general stats
        println("    📊 Starter Overlap: $overlap_count/$(length(live_starters)) ($(round(overlap_pct, digits=1))%)")
        if !isempty(rotated_in)
            println("    🔄 Rotated In / New Players:")
            for p in rotated_in
                is_new = p.player_id ∉ keys(player_ratings)
                debut_tag = is_new ? " (DEBUTANT 🌟 - Fallback rating: $(round(global_avg, digits=2)))" : ""
                println("       - [$(p.player_id)] $(p.player_name) [$(p.position)]$debut_tag")
            end
        else
            println("    🔄 No player rotations/changes compared to last game.")
        end
        
        # Aggregate Positional Ratings
        pos_ratings_live = Dict("G" => 0.0, "D" => 0.0, "M" => 0.0, "F" => 0.0)
        pos_ratings_fallback = Dict("G" => 0.0, "D" => 0.0, "M" => 0.0, "F" => 0.0)
        
        for p in live_starters
            c = clean_pos(p.position)
            rating = get(player_ratings, p.player_id, global_avg)
            pos_ratings_live[c] += rating
        end
        
        for p in fallback_starters
            c = clean_pos(p.position)
            rating = get(player_ratings, p.player_id, global_avg)
            pos_ratings_fallback[c] += rating
        end
        
        # Build comparison table
        comp_data = Matrix{Any}(undef, 4, 4)
        for (i, pos) in enumerate(["G", "D", "M", "F"])
            live_val = pos_ratings_live[pos]
            fall_val = pos_ratings_fallback[pos]
            diff_val = live_val - fall_val
            diff_str = diff_val > 0.05 ? "+$(round(diff_val, digits=2))" : 
                       (diff_val < -0.05 ? "$(round(diff_val, digits=2))" : "0.0")
            comp_data[i, 1] = pos
            comp_data[i, 2] = round(fall_val, digits=2)
            comp_data[i, 3] = round(live_val, digits=2)
            comp_data[i, 4] = diff_str
        end
        
        pretty_table(
            comp_data;
            header = ["Pos", "Fallback Sum", "Live Sum", "Delta (Live-Fallback)"],
            tf = tf_unicode_rounded,
            alignment = [:c, :r, :r, :c]
        )
    end
end
println("\n" * "="^80)
println("🏁 Lineup analysis complete!")
println("="^80)

# dev_repl/match_day_inference/l00_main_utils.jl

using Revise
using BayesianFootball

using LibPQ

using DataFrames
using ThreadPinning
pinthreads(:cores)



# ========================================
#  Stage 1 - Training the model
# ========================================

# --- 1. find the lasts month
#  Find the cv_configs that get the current split 
struct CVParameters 
  target_season::String
  warmup_period::Integer
end


"""
  helper function for the Data.segment type, to get the target_season 
"""
function get_target_seasons_string(segment::Data.DataTournemantSegment) 
    # none - place holder 
    println("Placeholder for the type: $(segment)")
    return 
end  

get_target_seasons_string(::Data.ScottishLower) = ["25/26"]
get_target_seasons_string(::Data.Ireland)       = ["2026"]
get_target_seasons_string(::Data.SouthKorea)    = ["2026"]
get_target_seasons_string(::Data.Norway)        = ["2026"]


"""
  Function to find the current CV split config for the selected DataStore (ds). 
  Return CVParameters type.
"""
function find_current_cv_parameters(ds::Data.DataStore)::CVParameters 
  # Grab the first element [1] to convert Vector{String} -> String
  target_season = get_target_seasons_string(ds.segment)[1] 
  target_tournament = Data.tournament_ids(ds.segment)
  # Notice you actually already do this [1] extraction correctly right here:
  warmup_period = last(unique(subset(ds.matches, :season => ByRow(isequal(get_target_seasons_string(ds.segment)[1]))).match_month))
  return CVParameters(target_season, warmup_period)
end


# --- 2. Running the experiment
struct DSExperimentSettings 
  ds::Data.DataStore
  label::String
  save_dir::String
  cv_params::CVParameters
end

struct ExperimentTask
    ds::Data.DataStore
    config::Experiments.ExperimentConfig
end


"""
  Wrapper verison of the func to allow for the DSExperimentSettings type to 
  be used as a parameter.
"""
function create_experiment_tasks(es::DSExperimentSettings)
    return create_experiment_tasks(es.ds, es.label, es.save_dir, es.cv_params)
end



function create_experiment_tasks(ds::Data.DataStore, label::String, save_dir::String, cv_params::CVParameters)

    # 1. Define the shared parts (CV and Training)
        cv_config = Data.GroupedCVConfig(
        tournament_groups = [Data.tournament_ids(ds.segment)],
        target_seasons = [cv_params.target_season],
        history_seasons = 4,
        dynamics_col = :match_month,
        warmup_period = cv_params.warmup_period,
        stop_early = false
    )

    sampler_conf = Samplers.NUTSConfig(
    600, # Number of samples for each chain
    10,   # Number of chains
    150, # Number of warm up steps 
    0.65,# Accept rate  [0,1]
    10,  # Max tree depth
    Samplers.UniformInit(-1, 1), # Interval for starting a chain 
    :perchain   # show_progress (We use the Global Logger instead)
    # false, # Display progress bar setting
  )
    train_cfg = BayesianFootball.Training.Independent(
    parallel=true,
    max_concurrent_splits=8
  )
    training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)

    # 2. Build the list of Configs
    configs = [
        # Experiments.ExperimentConfig(
        #     name = "$(label)_01_baseline",
        #     model = Models.PreGame.AblationStudy_NB_baseLine(),
        #     splitter = cv_config,
        #     training_config = training_config,
        #     save_dir = save_dir
        # ),
        Experiments.ExperimentConfig(
            name = "monthlyR_$(label)",
            model = Models.PreGame.AblationStudy_NB_baseline_month_r(),
            splitter = cv_config,
            training_config = training_config,
            save_dir = save_dir
        ),
        # Ireland 
        # Experiments.ExperimentConfig(
        #     name = "baseline_HA_R_$(label)", 
        #     model = Models.PreGame.AblationStudy_NB_baseline_HA_r(), 
        #     splitter = cv_config,
        #     training_config = training_config,
        #     save_dir = save_dir
        # ),
    ]

    # 3. THE "SMART" BIT: 
    # Wrap every config with the DS into an ExperimentTask
    # We use Ref(ds) so it doesn't try to "iterate" the DataStore
    return ExperimentTask.(Ref(ds), configs)
end


function run_experiment_task(task::ExperimentTask)
    conf = task.config
    println("Running: $(conf.name)")

    try
        # 2. Execute
        results = Experiments.run_experiment(task.ds, conf)

        # 3. Re-enable logging to save and confirm
        Experiments.save_experiment(results)
        
        return true # Success flag

    catch e
        @error "❌ Failed [$(conf.name)]: $e"
        # If you want to see the stacktrace for debugging:
        # Base.showerror(stdout, e, catch_backtrace())
        return false # Failure flag
    end
end

using Dates

# --- stage 2 
function fetch_todays_matches(segment::Data.DataTournemantSegment)::AbstractDataFrame
    db_config = Data.DBConfig("postgresql://admin:supersecretpassword@100.124.38.117:5432/sofascrape_db")
    db_conn = Data.connect_to_db(db_config)
    
    try
        # Pass the connection down
        data_store = fetch_todays_matches(db_conn, segment)
        return data_store
    finally
        # Always close the connection, even if an error occurs during fetching
        close(db_conn) 
    end
end

function fetch_todays_matches(db_conn::LibPQ.Connection, segment::Data.DataTournemantSegment)::AbstractDataFrame
    # 1. Removed the stray semicolon before the final AND
    query = """
    SELECT 
        match_id,
        home_team,
        away_team,
        round,
        tournament_id,
        season_id
    FROM 
        events
    WHERE 
        status_type = 'notstarted'
        AND start_timestamp >= EXTRACT(EPOCH FROM CURRENT_DATE)
        AND start_timestamp < EXTRACT(EPOCH FROM CURRENT_DATE + INTERVAL '1 day')
        AND tournament_id = ANY(\$1);
    """

    # 2 & 3. Fixed variable name (db_conn) and passed the tournament IDs parameter
    # (Assuming Data.tournament_ids(segment) returns a Vector of IDs)
    t_ids = Data.tournament_ids(segment)
    df = DataFrame(LibPQ.execute(db_conn, query, (t_ids,)))

    # 4. Simplified column assignment using pure broadcasting
    df.match_week .= 999
    df.match_date .= today()

    return df
end

""" 
Wrapper function for the DataStore Type 
"""
function fetch_todays_matches(ds::Data.DataStore)::AbstractDataFrame
  return fetch_todays_matches(ds.segment)
end


using Revise
using BayesianFootball
using DataFrames
using Dates
using GLM
using Statistics
using Printf

# ====================================================================
# UTILITIES: Loading & Models
# ====================================================================

function loaded_experiment_files(folders::Vector{String})
    return [BayesianFootball.Experiments.load_experiment(f) for f in folders]
end

function parse_duration(tags::Vector{String})
    time_idx = findfirst(t -> startswith(t, "time:"), tags)
    isnothing(time_idx) && return 0.0 
    
    time_str = replace(tags[time_idx], "time:" => "")
    
    seconds = 0.0
    m_h = match(r"(\d+)h", time_str) 
    m_m = match(r"(\d+)m", time_str)
    m_s = match(r"(\d+)s", time_str)
    
    if !isnothing(m_h); seconds += parse(Float64, m_h.captures[1]) * 3600; end
    if !isnothing(m_m); seconds += parse(Float64, m_m.captures[1]) * 60; end
    if !isnothing(m_s); seconds += parse(Float64, m_s.captures[1]); end
    
    return seconds
end

function load_same_large_experiment_model(target_exp; dir::String, data_dir::String="")
    saved_folders = BayesianFootball.Experiments.list_experiments(dir; data_dir=data_dir)
    matching_results = []
    
    for folder in saved_folders
        try
            res = BayesianFootball.Experiments.load_experiment(folder)
            if res.config.model == target_exp.config.model
                push!(matching_results, res)
            end 
        catch e
            @warn "Could not load $folder: $e"
        end
    end

    if isempty(matching_results)
        error("No matching experiments found for model: $(target_exp.config.model)")
    end

    best_res = argmax(r -> parse_duration(r.config.tags), matching_results)
    best_time = parse_duration(best_res.config.tags)
    println("Selected Model: $(best_res.config.name) with duration $(best_time)s")
    
    return best_res
end

# ====================================================================
# UTILITIES: PPD Generation
# ====================================================================

function raw_preds_to_df(raw_preds)
    ids = collect(keys(raw_preds))
    cols = Dict{Symbol, Vector{Any}}(:match_id => ids)
    first_entry = raw_preds[ids[1]]
    
    for k in keys(first_entry)
        cols[Symbol(k)] = [raw_preds[i][k] for i in ids]
    end
    
    return DataFrame(cols)
end

function compute_todays_matches_pdds(data_store, experiment, todays_matches)
  feature_collection = BayesianFootball.Features.create_features(
      BayesianFootball.Data.create_data_splits(data_store, experiment.config.splitter),
      experiment.config.model, 
      experiment.config.splitter
  )

  last_split_idx = length(experiment.training_results)
  chain = experiment.training_results[last_split_idx][1]
  feature_set = feature_collection[last_split_idx][1]

  raw_preds = BayesianFootball.Models.PreGame.extract_parameters(
      experiment.config.model, todays_matches, feature_set, chain
  )

  latents = BayesianFootball.Experiments.LatentStates(raw_preds_to_df(raw_preds), experiment.config.model)
  ppd = BayesianFootball.Predictions.model_inference(latents)

  return ppd
end

# ====================================================================
# UTILITIES: Market Shifts & Calibration
# ====================================================================

function calculate_single_shift(ppd_df::DataFrame, ds, target_selection::Symbol)
    println("--- Calibrating: :$target_selection ---")
    
    sub_df = filter(:selection => ==(target_selection), ppd_df)
    calib_df = innerjoin(ds.odds, sub_df, on = [:match_id, :market_name, :market_line, :selection])
    
    if isempty(calib_df)
        @warn "No matching market data found for :$target_selection. Returning 0.0 shift." 
        return 0.0
    end

    calib_df.actual = Float64.(calib_df.is_winner)
    calib_df.mean_prob = [mean(dist) for dist in calib_df.distribution]
    
    eps = 1e-6
    clamped = clamp.(calib_df.mean_prob, eps, 1.0 - eps)
    calib_df.logit_prob = log.(clamped ./ (1.0 .- clamped))
    
    model = glm(@formula(actual ~ 1), calib_df, Binomial(), LogitLink(), offset=calib_df.logit_prob)
    C_shift = coef(model)[1]
    println(">> Calculated Logit Shift: ", round(C_shift, digits=4))
   
    return C_shift
end

function apply_logit_shift(dist_array::Vector{Float64}, C_shift::Float64)
    eps = 1e-6
    clamped = clamp.(dist_array, eps, 1.0 - eps) 
    logits = log.(clamped ./ (1.0 .- clamped))
    shifted_logits = logits .+ C_shift
    return 1.0 ./ (1.0 .+ Base.exp.(.-shifted_logits))
end

function compute_market_shifts(target_model, ds, target_selections::Vector{Symbol})
    # OOS latents extracted directly using the target model
    latents = BayesianFootball.Predictions.extract_oos_predictions(ds, target_model)
    calib_ppd = BayesianFootball.Predictions.model_inference(latents)
    
    shifts = Dict{Symbol, Float64}()
    for sel in target_selections
        shifts[sel] = calculate_single_shift(calib_ppd.df, ds, sel)
    end 
    
    return shifts
end

function apply_market_shifts(ppd::BayesianFootball.Predictions.PPD, shift_dict::Dict{Symbol, Float64})
    new_df = copy(ppd.df) 
    
    new_df.distribution = map(eachrow(new_df)) do row
        if haskey(shift_dict, row.selection)
            shift_val = shift_dict[row.selection]
            return apply_logit_shift(row.distribution, shift_val)
        else
            return row.distribution
        end
    end
    
    println("Successfully applied calibration shifts to new PPD.")
    return BayesianFootball.Predictions.PPD(new_df, ppd.model, ppd.config)
end

# ====================================================================
# UTILITIES: Analysis & Paper Trades
# ====================================================================

function compare_calibrated_odds(raw_ppd::BayesianFootball.Predictions.PPD, calib_ppd::BayesianFootball.Predictions.PPD, matches::DataFrame)
    df_raw = rename(copy(raw_ppd.df), :distribution => :raw_distribution) 
    df_calib = rename(copy(calib_ppd.df), :distribution => :calib_distribution)
    
    compare_df = innerjoin(df_raw, df_calib, on=[:match_id, :market_name, :market_line, :selection])
    compare_view = innerjoin(matches, compare_df, on=:match_id) 
    
    return compare_view 
end

function generate_paper_bets(target_ppd::BayesianFootball.Predictions.PPD, 
                             calibrated_ppd::BayesianFootball.Predictions.PPD, 
                             todays_matches::DataFrame, 
                             live_market_df::DataFrame; 
                             min_edge::Float64=0.0)
    
    signal = BayesianFootball.Signals.BayesianKelly(min_edge=min_edge)
    results = [] 
    ppds_compare = compare_calibrated_odds(target_ppd, calibrated_ppd, todays_matches)
    
    println(repeat("=", 108)) 
    println(" 📝 PAPER TRADING BOARD: EXACT KELLY STAKES (Edge: $(min_edge * 100)%)") 
    println(repeat("=", 108)) 

    function calc_odds_metrics(dist::Vector{Float64}) 
        m_prob = mean(dist) 
        p95 = quantile(dist, 0.95) 
        p05 = quantile(dist, 0.05) 
        return (1.0 / m_prob, 1.0 / p95, 1.0 / p05)  
    end 

    for row in eachrow(live_market_df) 
        team = row.home_team 
        match_preds = filter(:home_team => ==(team), ppds_compare) 
        if isempty(match_preds); continue; end 
        
        match_id = match_preds.match_id[1] 
        printed_header = false 
        
        market_mappings = [ 
            (:over_15, :live_odds_o15), 
            (:over_25, :live_odds_o25) 
        ] 
        
        for (sel, live_col) in market_mappings 
            if !hasproperty(row, live_col) || ismissing(row[live_col]) 
                continue 
            end 
            
            live_odds = row[live_col] 
            sel_row = filter(:selection => ==(sel), match_preds)
            
            if !isempty(sel_row)
                dist_raw = sel_row.raw_distribution[1]
                dist_calib = sel_row.calib_distribution[1]
                
                raw_stake = BayesianFootball.Signals.compute_stake(signal, dist_raw, live_odds)
                calib_stake = BayesianFootball.Signals.compute_stake(signal, dist_calib, live_odds) 
                
                raw_mean, raw_bid, raw_ask = calc_odds_metrics(dist_raw) 
                calib_mean, calib_bid, calib_ask = calc_odds_metrics(dist_calib)
                
                if !printed_header 
                    println("\nMATCH: $(uppercase(team))") 
                    printed_header = true
                end 
                
                @printf("  %-8s (Live: %.2f) | Raw: %.2f [%.2f - %.2f] | Calib: %.2f [%.2f - %.2f] | Raw Stake: %5.2f%% | Calib: %5.2f%%\n", 
                        string(sel), live_odds,  
                        raw_mean, raw_bid, raw_ask,
                        calib_mean, calib_bid, calib_ask,
                        max(0.0, raw_stake) * 100, max(0.0, calib_stake) * 100) 
                
                push!(results, ( 
                    match_id = match_id, 
                    home_team = team, 
                    selection = sel,
                    live_odds = live_odds,
                    raw_mean_odds = round(raw_mean, digits=3),
                    raw_bid_odds = round(raw_bid, digits=3),
                    raw_ask_odds = round(raw_ask, digits=3),
                    calib_mean_odds = round(calib_mean, digits=3),
                    calib_bid_odds = round(calib_bid, digits=3), 
                    calib_ask_odds = round(calib_ask, digits=3),
                    raw_stake_pct = round(max(0.0, raw_stake) * 100, digits=2),
                    calib_stake_pct = round(max(0.0, calib_stake) * 100, digits=2) 
                )) 
            end 
        end 
    end 
    
    println("\n" * repeat("=", 108)) 
    return DataFrame(results) 
end

using JSON
using DataFrames

"""
Reads a .jsonl file generated by the python scraper.
Dynamically extracts the timestamp and ALL available back prices, 
flattening the nested JSON into wide DataFrame columns.
"""
function load_live_market_jsonl(filepath::String)::DataFrame
    all_rows = Dict{Symbol, Any}[]
    
    # Recursive helper to find all "back" prices in the JSON
    function extract_all_back_prices(dict_node, current_path, row_dict)
        if !(dict_node isa AbstractDict)
            return
        end
        
        # If we hit a selection containing a "back" dict
        if haskey(dict_node, "back") && dict_node["back"] isa AbstractDict
            price = get(dict_node["back"], "price", missing)
            
            if price !== nothing && !ismissing(price)
                # Clean up the path to make a valid column name
                # e.g., ["ft", "Match Odds", "Dundalk"] -> :ft_Match_Odds_Dundalk
                clean_path = [replace(p, r"[^a-zA-Z0-9]" => "_") for p in current_path]
                col_name = Symbol(join(clean_path, "_"))
                row_dict[col_name] = Float64(price)
            end
            return
        end
        
        # Otherwise, keep digging deeper into the JSON
        for (key, val) in dict_node
            if val isa AbstractDict
                push!(current_path, key)
                extract_all_back_prices(val, current_path, row_dict)
                pop!(current_path)
            end
        end
    end

    for line in eachline(filepath)
        if strip(line) == ""
            continue
        end
        
        data = JSON.parse(line)
        
        # 1. Base Info (Timestamp & Team)
        row_dict = Dict{Symbol, Any}(
            :timestamp => get(data, "timestamp", missing),
            :home_team => get(data, "home_team", missing)
        )
        
        # 2. Dynamically grab ALL markets
        market_data = get(data, "market_data", Dict())
        extract_all_back_prices(market_data, String[], row_dict)
        
        # 3. Create explicit aliases for your paper_bets runner backwards compatibility
        row_dict[:live_odds_o15] = get(row_dict, Symbol("ft_Over_Under_1_5_Goals_Over_1_5_Goals"), missing)
        row_dict[:live_odds_o25] = get(row_dict, Symbol("ft_Over_Under_2_5_Goals_Over_2_5_Goals"), missing)
        row_dict[:live_odds_o35] = get(row_dict, Symbol("ft_Over_Under_3_5_Goals_Over_3_5_Goals"), missing)
        row_dict[:live_odds_o45] = get(row_dict, Symbol("ft_Over_Under_4_5_Goals_Over_4_5_Goals"), missing)
        
        push!(all_rows, row_dict)
    end
    
    # Safely convert the array of dictionaries into a wide DataFrame.
    # Because some games might have markets that others don't, we union all keys.
    all_keys = unique(vcat([collect(keys(d)) for d in all_rows]...))
    
    df = DataFrame()
    for k in all_keys
        df[!, k] = [get(row, k, missing) for row in all_rows]
    end
    
    # Sort the dataframe by timestamp to ensure chronological order
    sort!(df, :timestamp)
    
    return df
end



using DataFrames

"""
    filter_and_rename_live_markets(live_df::DataFrame, target_selections::Vector{Symbol})

Takes the raw, wide DataFrame from `load_live_market_jsonl` and filters it down to 
only the requested markets. Renames the messy Betfair column strings to your clean 
internal database Symbols (e.g., :over_15, :btts_yes).
"""
function filter_and_rename_live_markets(live_df::DataFrame, target_selections::Vector{Symbol})::DataFrame
    # Master dictionary mapping your clean Symbols to the flattened Betfair strings
    market_map = Dict{Symbol, String}(
        :draw     => "ft_Match_Odds_The_Draw",
        :btts_yes => "ft_Both_teams_to_Score__Yes",
        :btts_no  => "ft_Both_teams_to_Score__No",
        :over_05  => "ft_Over_Under_0_5_Goals_Over_0_5_Goals",
        :under_05 => "ft_Over_Under_0_5_Goals_Under_0_5_Goals",
        :over_15  => "ft_Over_Under_1_5_Goals_Over_1_5_Goals",
        :under_15 => "ft_Over_Under_1_5_Goals_Under_1_5_Goals",
        :over_25  => "ft_Over_Under_2_5_Goals_Over_2_5_Goals",
        :under_25 => "ft_Over_Under_2_5_Goals_Under_2_5_Goals",
        :over_35  => "ft_Over_Under_3_5_Goals_Over_3_5_Goals",
        :under_35 => "ft_Over_Under_3_5_Goals_Under_3_5_Goals",
        :over_45  => "ft_Over_Under_4_5_Goals_Over_4_5_Goals",
        :under_45 => "ft_Over_Under_4_5_Goals_Under_4_5_Goals"
    )

    # We ALWAYS want to keep the base reference columns
    keep_cols = ["timestamp", "home_team"]
    rename_pairs = Pair{String, Symbol}[]

    # Match the requested selections to their Betfair column names
    for sel in target_selections
        if haskey(market_map, sel)
            bf_col = market_map[sel]
            
            # Only attempt to select it if the python scraper actually found it
            if bf_col in names(live_df)
                push!(keep_cols, bf_col)
                push!(rename_pairs, bf_col => sel)
            else
                @warn "Market column '$bf_col' not found in the live DataFrame for selection :$sel"
            end
        else
            # NOTE: :home and :away are tricky because Betfair names them after the actual teams 
            # (e.g. "ft_Match_Odds_Shamrock_Rovers"). They require dynamic string matching.
            @warn "No static Betfair mapping defined for requested selection :$sel"
        end
    end

    # Subset the dataframe to only the columns we want
    clean_df = select(live_df, keep_cols)

    # Rename them to your internal symbols
    rename!(clean_df, rename_pairs)
    
    # -------------------------------------------------------------------------
    # SAFETY NET: Because your `generate_paper_bets` function specifically 
    # looks for `:live_odds_o15` and `:live_odds_o25`, we will duplicate 
    # those columns using your alias so your runner doesn't break!
    # -------------------------------------------------------------------------
    if :over_15 in names(clean_df)
        clean_df.live_odds_o15 = clean_df.over_15
    end
    if :over_25 in names(clean_df)
        clean_df.live_odds_o25 = clean_df.over_25
    end

    if :over_35 in names(clean_df)
        clean_df.live_odds_o35 = clean_df.over_35
    end
    if :over_45 in names(clean_df)
        clean_df.live_odds_o45 = clean_df.over_45
    end


    return clean_df
end


function generate_paper_bets(target_ppd::BayesianFootball.Predictions.PPD, 
                             calibrated_ppd::BayesianFootball.Predictions.PPD, 
                             todays_matches::DataFrame, 
                             live_market_df::DataFrame; 
                             min_edge::Float64=0.0)
    
    signal = BayesianFootball.Signals.BayesianKelly(min_edge=min_edge)
    results = []
    ppds_compare = compare_calibrated_odds(target_ppd, calibrated_ppd, todays_matches)
    
    println(repeat("=", 108))
    println(" 📝 PAPER TRADING BOARD: EXACT KELLY STAKES (Edge: $(min_edge * 100)%)")
    println(repeat("=", 108))

    function calc_odds_metrics(dist::Vector{Float64})
        m_prob = mean(dist)
        p95 = quantile(dist, 0.95)
        p05 = quantile(dist, 0.05)
        return (1.0 / m_prob, 1.0 / p95, 1.0 / p05) 
    end

    # Dynamically find all market columns (everything that isn't metadata)
    metadata_cols = [:timestamp, :home_team, :match_id, :live_odds_o15, :live_odds_o25]
    available_markets = [Symbol(c) for c in names(live_market_df) if Symbol(c) ∉ metadata_cols]

    for row in eachrow(live_market_df)
        team = row.home_team
        match_preds = filter(:home_team => ==(team), ppds_compare)
        if isempty(match_preds); continue; end
        
        match_id = match_preds.match_id[1]
        printed_header = false
        
        # Loop dynamically through whatever markets are in the DataFrame
        for sel in available_markets
            if ismissing(row[sel])
                continue
            end
            
            live_odds = row[sel]
            
            # Find the matching MCMC row for this specific market selection
            sel_row = filter(:selection => ==(sel), match_preds)
            
            if !isempty(sel_row)
                dist_raw = sel_row.raw_distribution[1]
                dist_calib = sel_row.calib_distribution[1]
                
                raw_stake = BayesianFootball.Signals.compute_stake(signal, dist_raw, live_odds)
                calib_stake = BayesianFootball.Signals.compute_stake(signal, dist_calib, live_odds)
                
                raw_mean, raw_bid, raw_ask = calc_odds_metrics(dist_raw)
                calib_mean, calib_bid, calib_ask = calc_odds_metrics(dist_calib)
                
                if !printed_header
                    println("\nMATCH: $(uppercase(team))")
                    printed_header = true
                end
                
                @printf("  %-8s (Live: %.2f) | Raw: %.2f [%.2f - %.2f] | Calib: %.2f [%.2f - %.2f] | Raw Stake: %5.2f%% | Calib: %5.2f%%\n", 
                        string(sel), live_odds, 
                        raw_mean, raw_bid, raw_ask, 
                        calib_mean, calib_bid, calib_ask, 
                        max(0.0, raw_stake) * 100, max(0.0, calib_stake) * 100)
                
                push!(results, (
                    match_id = match_id,
                    home_team = team,
                    selection = sel,
                    live_odds = live_odds,
                    raw_mean_odds = round(raw_mean, digits=3),
                    raw_bid_odds = round(raw_bid, digits=3),
                    raw_ask_odds = round(raw_ask, digits=3),
                    calib_mean_odds = round(calib_mean, digits=3),
                    calib_bid_odds = round(calib_bid, digits=3),
                    calib_ask_odds = round(calib_ask, digits=3),
                    raw_stake_pct = round(max(0.0, raw_stake) * 100, digits=2),
                    calib_stake_pct = round(max(0.0, calib_stake) * 100, digits=2)
                ))
            end
        end
    end
    
    println("\n" * repeat("=", 108))
    return DataFrame(results)
end

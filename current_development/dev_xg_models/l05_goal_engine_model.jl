# current_development/dev_xg_models/r05_goal_engine_model.jl



# --- 2. Running the experiment
struct DSExperimentSettings 
  ds::Data.DataStore
  label::String
  save_dir::String
  target_season::Vector{<:String}
end

struct ExperimentTask
    ds::Data.DataStore
    config::Experiments.ExperimentConfig
end


get_target_seasons_string(::Data.Ireland)       = ["2026"]

function find_current_warmup(ds::Data.DataStore) 
  # Grab the first element [1] to convert Vector{String} -> String
  target_tournament = Data.tournament_ids(ds.segment)
  # Notice you actually already do this [1] extraction correctly right here:
  warmup_period = last(unique(subset(ds.matches, :season => ByRow(isequal(get_target_seasons_string(ds.segment)[1]))).match_month))
  return  warmup_period + 1
end


function create_CVsplit_training_config(ds::Data.DataStore, target_seasons::Vector{<:String})

    # 1. Define the shared parts (CV and Training)
    cv_config = Data.GroupedCVConfig(
        tournament_groups = [Data.tournament_ids(ds.segment)],
        target_seasons = target_seasons,
        history_seasons = 1,
        dynamics_col = :match_month,
        warmup_period = find_current_warmup(ds),
        stop_early = false
    )

    sampler_conf = Samplers.NUTSConfig(
    500, # n steps
    4,    # n chains
    200,  # warm up steps
    0.65, # acceptance rate
    10,   # Max depth
    Samplers.UniformInit(-1, 1), # init step up 
    :perchain # show progress bar
    )

    train_cfg = BayesianFootball.Training.Independent(
        parallel=true, max_concurrent_splits=4
    )
    training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)


    return (; cv_cfg=cv_config, training_cfg=training_config)

end


# ==========================================
#  1: Combine Model + Cfgs into an ExperimentTask
# ==========================================
function build_experiment_task(ds::BayesianFootball.Data.DataStore, model, label, save_dir::String, cfgs::NamedTuple)
    # 1. Define where this specific model will save its chains/metrics
    
    # 2. Build the master config
    exp_config = BayesianFootball.Experiments.ExperimentConfig(
        name = label,
        model = model,
        splitter = cfgs.cv_cfg,
        training_config = cfgs.training_cfg,
        save_dir = save_dir
    )
    
    # 3. Return the task ready for the execution pipeline
    return ExperimentTask(ds, exp_config)
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

###### ------
#

using Dates
using LibPQ



function fetch_todays_matches(ds::Data.DataStore)::AbstractDataFrame
  return fetch_todays_matches(ds.segment)

end

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


function raw_preds_to_df(raw_preds)
    ids = collect(keys(raw_preds))
    cols = Dict{Symbol, Vector{Any}}(:match_id => ids)
    first_entry = raw_preds[ids[1]]
    
    for k in keys(first_entry)
        cols[Symbol(k)] = [raw_preds[i][k] for i in ids]
    end
    
    return DataFrame(cols)
end

function compute_todays_matches_pdds(ds, expr, todays_matches)
boundaries_with_meta = BayesianFootball.Data.create_id_boundaries(ds, expr.config.splitter)
feature_collection = BayesianFootball.Features.create_features(boundaries_with_meta, ds, expr.config.model)

  last_split_idx = length(expr.training_results)
  chain = expr.training_results[last_split_idx][1]
  feature_set = feature_collection[last_split_idx][1]

  raw_preds = BayesianFootball.Models.PreGame.extract_parameters(
      expr.config.model, todays_matches, feature_set, chain
  )

  latents = BayesianFootball.Experiments.LatentStates(raw_preds_to_df(raw_preds), expr.config.model)
  ppd = BayesianFootball.Predictions.model_inference(latents)
  return ppd

end
# ------
function generate_odds_comparison(ds, model_results::AbstractVector{<:Pair}, todays_matches; 
    selections = [:home,:draw, :away, :over_15, :over_25, :under_25, :over_35])
    
    # 1. Fetch matches once
    
    comparison_df = DataFrame()
    odds_columns = Symbol[]

    # 2. Loop through each passed model and process it
    for (i, (model_name, res)) in enumerate(model_results)
        println("Processing model: $model_name...")
        
        # Compute PPD
        ppd = compute_todays_matches_pdds(ds, res, todays_matches)
        
        # Calculate Odds
        odds = calculate_mean_odds(ppd.df, selections)
        
        # Rename the odds column dynamically
        col_name = Symbol("odds_", model_name)
        rename!(odds, :mean_odds => col_name)
        push!(odds_columns, col_name)
        
        # Join to the master dataframe
        if i == 1
            comparison_df = odds
        else
            comparison_df = innerjoin(comparison_df, odds, on = [:match_id, :selection])
        end
    end

    # 3. Add Team Names
    comparison_df = leftjoin(
        comparison_df, 
        select(todays_matches, :match_id, :home_team, :away_team), 
        on = :match_id,
        makeunique = true
    )

    # 4. Reorder columns
    select!(comparison_df, 
        :match_id, :home_team, :away_team, :selection, 
        odds_columns...
    )

    sort!(comparison_df, [:match_id, :selection])
    
    println("Comparison generated successfully!")
    return comparison_df
end

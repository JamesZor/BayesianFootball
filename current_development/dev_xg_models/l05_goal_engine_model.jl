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


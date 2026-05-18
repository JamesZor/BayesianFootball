# current_development/test_dispersion/r01_advanced_dispersion_runner.jl

using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
pinthreads(:cores)

using Dates
using Distributions

const PreGame = BayesianFootball.Models.PreGame

# --- Helper Functions (Copied for runner logic) ---

struct ExperimentTask
    ds::BayesianFootball.Data.DataStore
    config::BayesianFootball.Experiments.ExperimentConfig
end

function create_experiment_tasks(ds::BayesianFootball.Data.DataStore, model, label::String, save_dir::String, target_seasons::Vector{<:String} )
    cv_config = BayesianFootball.Data.GroupedCVConfig(
        tournament_groups = [BayesianFootball.Data.tournament_ids(ds.segment)],
        target_seasons = target_seasons,
        history_seasons = 1,
        dynamics_col = :match_month,
        warmup_period = 0,
        stop_early = true
    )

    sampler_conf = BayesianFootball.Samplers.NUTSConfig(
        500, # Reduced heavily for a quick test run
        2,   
        200,  
        0.65,
        10,  
        BayesianFootball.Samplers.UniformInit(-1, 1),
        false,
    )

    train_cfg = BayesianFootball.Training.Independent(
        parallel=true,
        max_concurrent_splits=4
    )
    training_config = BayesianFootball.Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)

    configs = [
        BayesianFootball.Experiments.ExperimentConfig(
            name = "$(label)_",
            model = model, 
            splitter = cv_config,
            training_config = training_config,
            save_dir = save_dir
        ),
    ]

    return ExperimentTask.(Ref(ds), configs)
end

function run_experiment_task(task::ExperimentTask)
    conf = task.config
    println("\n>>> Running: $(conf.name)")

    try
        results = BayesianFootball.Experiments.run_experiment(task.ds, conf)
        BayesianFootball.Experiments.save_experiment(results)
        println("✅ Success: $(conf.name)")
        return true
    catch e
        @error "❌ Failed [$(conf.name)]: $e"
        return false
    end
end

# --- Runner Logic ---

# 1. Load Data
println("Loading DataStore...")
ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())
save_dir::String = "./data/test_adv_dispersion/"

# 2. Instantiate Model with the NEW AdvancedVolatilityDispersion
println("Initializing Model Configs...")
inter_cfg = PreGame.GlobalInterception()
dyn_cfg   = PreGame.TimeDecayDynamics(days_half_life = 180.0)
disp_cfg  = PreGame.AdvancedVolatilityDispersion() # <--- The newly added dispersion component
disp_cfg  = PreGame.GlobalDispersion() # <--- The newly added dispersion component
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()

model = PreGame.DynamicMarketXGTimeDecayModel(
    interception_config  = inter_cfg,
    dynamics_config      = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
    kappa_config         = kap_cfg,
    market_weight        = 1.0
)

# 3. Quick Feature Pipeline Test
println("\nTesting Feature Pipeline integration for :month...")
cv_config_test = BayesianFootball.Data.GroupedCVConfig(
    tournament_groups = [BayesianFootball.Data.tournament_ids(ds.segment)],
    target_seasons = ["2026"],
    history_seasons = 1,
    dynamics_col = :match_month,
    warmup_period = 0,
    stop_early = true
)
boundaries = BayesianFootball.Data.create_id_boundaries(ds, cv_config_test)
feature_collection = BayesianFootball.Features.create_features(boundaries, ds, model)
if !isempty(feature_collection)
    data = feature_collection[1][1].data
    println("✅ Features extracted successfully.")
    println("Has :flat_months? ", haskey(data, :flat_months))
    if haskey(data, :flat_months)
        println("Sample months: ", data[:flat_months][1:min(5, end)])
    end
else
    println("❌ Feature extraction failed.")
end

# 4. Create and Run Tasks
println("\nCreating Experiment Tasks...")
tasks = create_experiment_tasks(ds, model, "adv_disp_test", save_dir, ["2026"])

println("\nReady! To execute the test run, use:")
println("run_experiment_task.(tasks)")
# Uncomment the line below to run automatically:
run_experiment_task.(tasks)




function loaded_experiment_files(saved_folders::Vector{String})
    loaded_results = Vector{BayesianFootball.Experiments.ExperimentResults}([])
    for folder in saved_folders
        try
            res = Experiments.load_experiment(folder)
            push!(loaded_results, res)
        catch e
            @warn "Could not load $folder: $e"
        end
    end
    return loaded_results
end


saved_folders = Experiments.list_experiments(save_dir; data_dir="")
loaded_results = loaded_experiment_files(saved_folders)



function check_parameter_stability(chains::Vector, target_params::Vector{Symbol})
    # Initialize an empty DataFrame
    df = DataFrame(Fold = Int[])
    
    # FIX: Explicitly tell Julia these columns can contain missing values
    for p in target_params
        df[!, Symbol(string(p), "_mean")] = Union{Missing, Float64}[]
        df[!, Symbol(string(p), "_std")]  = Union{Missing, Float64}[]
    end
    
    # Iterate through each fold's MCMCChain
    for (fold_idx, chain) in enumerate(chains)
        row_dict = Dict{Symbol, Any}(:Fold => fold_idx)
        
        for p in target_params
            # Check if the parameter exists in the chain
            if p in keys(chain)
                samples = vec(chain[p]) 
                row_dict[Symbol(string(p), "_mean")] = mean(samples)
                row_dict[Symbol(string(p), "_std")]  = std(samples)
            else
                row_dict[Symbol(string(p), "_mean")] = missing
                row_dict[Symbol(string(p), "_std")]  = missing
            end
        end
        
        push!(df, row_dict) # This will now safely accept the missing values!
    end
    
    return df
end




using Turing
expr = loaded_results[1]


chain_fold_1 = expr.training_results[1][1]
chain_fold_2 = expr.training_results[2][1]
chain_fold_3 = expr.training_results[3][1]

describe(chain_fold_3)


rqr_data = Evaluation.compute_metric(Evaluation.RQR(), expr, ds)
flat_row = Evaluation.to_dataframe_row(expr, rqr_data)



ll_data = Evaluation.compute_metric(Evaluation.LogLoss(), expr, ds)
flat_row = Evaluation.to_dataframe_row(expr, ll_data)


glm_data = Evaluation.compute_metric(Evaluation.GLMEdge(), expr, ds)
flat_row = Evaluation.to_dataframe_row(expr, glm_data)

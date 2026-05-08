# include("./l00_inverse_problem.jl") # experiment stuff 
# include("./l02_market_featue.jl")


using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
pinthreads(:cores)


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

function create_CVsplit_training_config(ds::Data.DataStore, target_seasons::Vector{<:String})

    # 1. Define the shared parts (CV and Training)
    cv_config = Data.GroupedCVConfig(
        tournament_groups = [Data.tournament_ids(ds.segment)],
        target_seasons = target_seasons,
        history_seasons = 1,
        dynamics_col = :match_biweek,
        warmup_period = 0,
        stop_early = false
    )

    sampler_conf = Samplers.NUTSConfig(
    1000, # n steps
    2,    # n chains
    300,  # warm up steps
    0.65, # acceptance rate
    10,   # Max depth
    Samplers.UniformInit(-1, 1), # init step up 
    :false # show progress bar
    )

    train_cfg = BayesianFootball.Training.Independent(
        parallel=true, max_concurrent_splits=8
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

############
using Distributions

# for running the models
const PreGame = BayesianFootball.Models.PreGame


inter_cfg = PreGame.GlobalInterception()
# Note: Using HomeAwayDispersion based on your previous grid (unless you built a custom TeamDispersion!)
disp_cfg  = PreGame.HomeAwayDispersion() 
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()

dyn_cfg   = PreGame.MultiScaleGRW(
    z₀ = Normal(0, 1), 
      zₛ = Normal(0,1),   # The fat-tailed robust winner
    zₖ = Normal(0, 1)
)
kap_cfg   = PreGame.HierarchicalTeamKappa() 

## models 

model_gm = PreGame.DynamicMarketGoalsModel(
    interception_config  = inter_cfg,
    dynamics_config      = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
)

model_g = PreGame.DynamicGoalsModel(
    interception_config  = inter_cfg,
    dynamics_config      = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
)

model_gxg = PreGame.DynamicXGModel(
    interception_config  = inter_cfg,
    dynamics_config      = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
    kappa_config         = kap_cfg
)


model_gxgm = PreGame.DynamicMarketXGModel(
    interception_config  = inter_cfg,
    dynamics_config      = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
    kappa_config         = kap_cfg
)


save_dir::String = "./data/dev_joint_model/ireland/"


ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())


cfgs = create_CVsplit_training_config(ds,get_target_seasons_string(ds.segment))

task_g = build_experiment_task(ds, model_g, "goals_biweek", save_dir, cfgs)
task_gm = build_experiment_task(ds, model_gm, "goals_market_biweek", save_dir, cfgs)
task_gxg = build_experiment_task(ds, model_gxg, "goals_xg_biweek", save_dir, cfgs)
task_gxgm = build_experiment_task(ds, model_gxgm, "goals_xg_market_biweek", save_dir, cfgs)

all_task = [task_g, task_gm, task_gxg, task_gxgm]

run_experiment_task.(all_task)


# ------
using DataFrames
using Statistics

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



params_to_track_xg = [
    Symbol("inter.μ"), 
    Symbol("σ_market"), # NEW: Variance/spread of team conversion abilities
    Symbol("disp.log_r"), 
    Symbol("ha.γ_global"),
    :ν_xg,       # NEW: xG Gamma shape parameter
    Symbol("kap.κ_base"),
    Symbol("kap.σ_κ"),
    Symbol("dyn.α.σ₀"), 
    Symbol("dyn.α.σₛ"), 
    Symbol("dyn.α.σₖ"),
    Symbol("dyn.β.σ₀"), 
    Symbol("dyn.β.σₛ"), 
    Symbol("dyn.β.σₖ")
]


expr = loaded_results[1]
all_chains = [res[1] for res in expr.training_results] 
# 3. Generate the Stability Report
stability_df_xg = check_parameter_stability(all_chains, params_to_track_xg)






# ------

saved_folders = Experiments.list_experiments(save_dir; data_dir="")
# saved_folders = Experiments.list_experiments("exp/grw_basics_pl_ch"; data_dir="./data")

# Load them all into a list
loaded_results = Vector{BayesianFootball.Experiments.ExperimentResults}([])
for folder in saved_folders
    try
        res = Experiments.load_experiment(folder)
        push!(loaded_results, res)
    catch e
        @warn "Could not load $folder: $e"
    end
end

ledger = BayesianFootball.BackTesting.run_backtest(
    ds, 
    loaded_results, 
  [BayesianFootball.Signals.BayesianKelly()]; 
    market_config = Data.Markets.DEFAULT_MARKET_CONFIG
)

tearsheet = BayesianFootball.BackTesting.generate_tearsheet(ledger)

model_names = unique(tearsheet.selection)

model_names = model_names

for m_name in model_names
    println("\nStats for: $m_name")
    sub = subset(tearsheet, :selection => ByRow(isequal(m_name)))
    show(sub)
end





#=
4×7 DataFrame
 Row │ model                   rqr_all_mean  rqr_all_std  rqr_all_skewness  rqr_all_kurtosis  rqr_all_shapiro_w  rqr_all_shapiro_p 
     │ String                  Float64       Float64      Float64           Float64           Float64            Float64           
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ goals_biweek               0.0741926     0.933369        -0.191402          -0.10208            0.991067           0.485996
   2 │ goals_market_biweek        0.0865336     0.923827         0.0634021         -0.588697           0.991667           0.54871
   3 │ goals_xg_biweek            0.124257      0.904944        -0.0212515         -0.597644           0.989138           0.316398
   4 │ goals_xg_market_biweek     0.0703558     0.98718         -0.221127           0.182947           0.985887           0.142215
=#


#=
julia> # Let's just view the most important columns: The Spread Coef and its P-Value
       display(select(master_glm_df, 
            :model, 
           :glmedge_intercept_coef,
           :glmedge_spread_fair_coef, 
           :glmedge_spread_fair_p_value,
           :glmedge_n_obs
       ))
4×5 DataFrame
 Row │ model                   glmedge_intercept_coef  glmedge_spread_fair_coef  glmedge_spread_fair_p_value  glmedge_n_obs 
     │ String                  Float64                 Float64                   Float64                      Int64         
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ goals_biweek                          -2.52637                 0.0558007                     0.903425           1658
   2 │ goals_market_biweek                   -2.51356                -0.0780719                     0.864705           1658
   3 │ goals_xg_biweek                       -2.54426                 0.241146                      0.588477           1658
   4 │ goals_xg_market_biweek                -2.5167                 -0.0433964                     0.924514           1658
=#



#=
julia> display(select(master_ll_df, 
           :model, 
           :logloss_overall_model_ll, 
           :logloss_overall_market_ll, 
           :logloss_overall_diff_ll
       ))
4×4 DataFrame
 Row │ model                   logloss_overall_model_ll  logloss_overall_market_ll  logloss_overall_diff_ll 
     │ String                  Float64                   Float64                    Float64                 
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ goals_biweek                            0.532836                    18.7905                 -18.2577
   2 │ goals_market_biweek                     0.534614                    18.7905                 -18.2559
   3 │ goals_xg_biweek                         0.530728                    18.7905                 -18.2598
   4 │ goals_xg_market_biweek                  0.533966                    18.7905                 -18.2566

=#




#=
4×4 DataFrame
 Row │ model                   crps_home_mean  crps_away_mean  crps_all_mean 
     │ String                  Float64         Float64         Float64       
─────┼───────────────────────────────────────────────────────────────────────
   1 │ goals_biweek                  0.630039        0.570285       0.600162
   2 │ goals_market_biweek           0.61601         0.579209       0.597609
   3 │ goals_xg_biweek               0.612866        0.578944       0.595905
   4 │ goals_xg_market_biweek        0.615827        0.577542       0.596684
=#


#=


season 2026
Stats for: home                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
4×18 DataFrame                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
 Row │ model_name                selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth                                                                                                                                                                                                                                                                                                                                                                        
     │ String                    Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64                                                                                                                                                                                                                                                                                                                                                                                 
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                                                                                                                                                                                                                                                                     
   1 │ DynamicMarketXGModel      home                  73           24          32.9      0.61    -0.2    -32.84          25.0       -0.101            -0.201                                                                                                                                                                                                                                                                                                                                                                        
   2 │ DynamicXGModel            home                  73           30          41.1      0.91    -0.24   -26.75          26.7       -0.071            -0.245                                                                                                                                                                                                                                                                                                                                                                        
   3 │ DynamicMarketGoalsModel   home                  73           24          32.9      0.61    -0.21   -33.64          25.0       -0.104            -0.206                                                                                                                                                                                                                                                                                                                                                                        
   4 │ DynamicGoalsModel         home                  73           27          37.0      1.51    -0.96   -63.83          22.2       -0.248            -0.962                                                                                                                                                                                                                                                                                                                                                                        
Stats for: draw                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
4×18 DataFrame                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
 Row │ model_name                selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth                                                                                                                                                                                                                                                                                                                                                                        
     │ String                    Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64                                                                                                                                                                                                                                                                                                                                                                                 
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                                                                                                                                                                                                                                                                     
   1 │ DynamicMarketXGModel      draw                  73            3           4.1      0.02    -0.02   -100.0           0.0       -0.163            -0.019                                                                                                                                                                                                                                                                                                                                                                        
   2 │ DynamicXGModel            draw                  73            5           6.8      0.05     0.04     86.7          40.0        0.07              0.041                                                                                                                                                                                                                                                                                                                                                                        
   3 │ DynamicMarketGoalsModel   draw                  73            3           4.1      0.02    -0.02   -100.0           0.0       -0.157            -0.022                                                                                                                                                                                                                                                                                                                                                                        
   4 │ DynamicGoalsModel         draw                  73           11          15.1      0.11     0.04     36.8          27.3        0.052             0.041                                                                                                                                                                                                                                                                                                                                                                        
Stats for: away                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
4×18 DataFrame                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
 Row │ model_name                selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth                                                                                                                                                                                                                                                                                                                                                                        
     │ String                    Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64                                                                                                                                                                                                                                                                                                                                                                                 
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                                                                                                                                                                                                                                                                     
   1 │ DynamicMarketXGModel      away                  73           29          39.7      0.97    -0.57   -58.4           17.2       -0.257            -0.566                                                                                                                                                                                                                                                                                                                                                                        
   2 │ DynamicXGModel            away                  73           26          35.6      0.47    -0.1    -21.5           19.2       -0.05             -0.101                                                                                                                                                                                                                                                                                                                                                                        
   3 │ DynamicMarketGoalsModel   away                  73           31          42.5      1.02    -0.57   -55.91          19.4       -0.241            -0.568                                                                                                                                                                                                                                                                                                                                                                        
   4 │ DynamicGoalsModel         away                  73           35          47.9      1.39    -0.19   -13.6           20.0       -0.033            -0.189                                                                                                                                                                                                                                                                                                                                                                        
Stats for: btts_yes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
4×18 DataFrame                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
 Row │ model_name                selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth                                                                                                                                                                                                                                                                                                                                                                        
     │ String                    Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64                                                                                                                                                                                                                                                                                                                                                                                 
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                                                                                                                                                                                                                                                                     
   1 │ DynamicMarketXGModel      btts_yes              73            3           4.1      0.07     0.1    142.36          66.7        0.129             0.097                                                                                                                                                                                                                                                                                                                                                                        
   2 │ DynamicXGModel            btts_yes              73            8          11.0      0.19    -0.04   -18.4           50.0       -0.04             -0.035                                                                                                                                                                                                                                                                                                                                                                        
   3 │ DynamicMarketGoalsModel   btts_yes              73            3           4.1      0.06     0.09   145.72          66.7        0.127             0.094                                                                                                                                                                                                                                                                                                                                                                        
   4 │ DynamicGoalsModel         btts_yes              73            4           5.5      0.07     0.08   119.74          50.0        0.119             0.083                                                                                                                                                                                                                                                                                                                                                                        
Stats for: btts_no                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
4×18 DataFrame                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
 Row │ model_name                selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth                                                                                                                                                                                                                                                                                                                                                                        
     │ String                    Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64                                                                                                                                                                                                                                                                                                                                                                                 
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                                                                                                                                                                                                                                                                     
   1 │ DynamicMarketXGModel      btts_no               73           22          30.1      0.55    -0.1    -18.91          36.4       -0.067            -0.104                                                                                                                                                                                                                                                                                                                                                                        
   2 │ DynamicXGModel            btts_no               73           34          46.6      1.26    -0.03    -2.22          47.1       -0.012            -0.028                                                                                                                                                                                                                                                                                                                                                                        
   3 │ DynamicMarketGoalsModel   btts_no               73           20          27.4      0.56    -0.11   -19.12          30.0       -0.067            -0.108                                                                                                                                                                                                                                                                                                                                                                        
   4 │ DynamicGoalsModel         btts_no               73           21          28.8      0.33    -0.13   -37.44          47.6       -0.117            -0.125                                                                                                                                                                                                                                                                                                                                                                        
Stats for: DC_1X                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
4×18 DataFrame                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
 Row │ model_name                selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth                                                                                                                                                                                                                                                                                                                                                                        
     │ String                    Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64                                                                                                                                                                                                                                                                                                                                                                                 
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                                                                                                                                                                                                                                                                     
   1 │ DynamicMarketXGModel      DC_1X                 73           13          17.8      0.49     0.17    34.19          61.5        0.094             0.167                                                                                                                                                                                                                                                                                                                                                                        
   2 │ DynamicXGModel            DC_1X                 73           22          30.1      0.84     0.21    25.52          68.2        0.078             0.215                                                                                                                                                                                                                                                                                                                                                                        
   3 │ DynamicMarketGoalsModel   DC_1X                 73           13          17.8      0.51     0.17    32.92          61.5        0.093             0.167                                                                                                                                                                                                                                                                                                                                                                        
   4 │ DynamicGoalsModel         DC_1X                 73           21          28.8      2.04     0.14     7.01          61.9        0.028             0.143                                                                                                                                                                                                                                                                                                                                                                        
Stats for: DC_X2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
4×18 DataFrame                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
 Row │ model_name                selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth                                                                                                                                                                                                                                                                                                                                                                        
     │ String                    Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64                                                                                                                                                                                                                                                                                                                                                                                 
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                                                                                                                                                                                                                                                                     
   1 │ DynamicMarketXGModel      DC_X2                 73           21          28.8      0.81    -0.59   -72.2           14.3       -0.295            -0.586                                                                                                                                                                                                                                                                                                                                                                        
   2 │ DynamicXGModel            DC_X2                 73           16          21.9      0.51    -0.31   -60.9           18.8       -0.168            -0.313                                                                                                                                                                                                                                                                                                                                                                        
   3 │ DynamicMarketGoalsModel   DC_X2                 73           21          28.8      0.86    -0.61   -71.51          19.0       -0.305            -0.614                                                                                                                                                                                                                                                                                                                                                                        
   4 │ DynamicGoalsModel         DC_X2                 73           26          35.6      1.84    -1.13   -61.07          15.4       -0.259            -1.126                                                                                                                                                                                                                                                                                                                                                                        
Stats for: DC_12                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
4×18 DataFrame                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
 Row │ model_name                selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth                                                                                                                                                                                                                                                                                                                                                                        
     │ String                    Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64                                                                                                                                                                                                                                                                                                                                                                                 
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                                                                                                                                                                                                                                                                     
   1 │ DynamicMarketXGModel      DC_12                 73            5           6.8      0.02    -0.02   -100.0           0.0       -0.152            -0.019  
   2 │ DynamicXGModel            DC_12                 73            7           9.6      0.14    -0.14   -100.0           0.0       -0.205            -0.144  
   3 │ DynamicMarketGoalsModel   DC_12                 73            5           6.8      0.02    -0.02   -100.0           0.0       -0.168            -0.017  
   4 │ DynamicGoalsModel         DC_12                 73            6           8.2      0.02    -0.02   -100.0           0.0       -0.2              -0.022  
Stats for: over_05
4×18 DataFrame
 Row │ model_name                selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth   
     │ String                    Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64            
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ DynamicMarketXGModel      over_05               73            2           2.7      0.03     0.0     10.89         100.0        0.122             0.004  
   2 │ DynamicXGModel            over_05               73            2           2.7      0.08     0.01     7.14         100.0        0.119             0.006  
   3 │ DynamicMarketGoalsModel   over_05               73            1           1.4      0.05     0.01    11.11         100.0        0.118             0.006  
   4 │ DynamicGoalsModel         over_05               73            0           0.0      0.0      0.0      0.0            0.0        0.0               0.0    
Stats for: under_05
4×18 DataFrame
 Row │ model_name                selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth   
     │ String                    Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64            
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ DynamicMarketXGModel      under_05              73           10          13.7      0.03    -0.03  -100.0            0.0       -0.294            -0.032  
   2 │ DynamicXGModel            under_05              73           27          37.0      0.12    -0.12   -99.04           7.4       -0.417            -0.12   
   3 │ DynamicMarketGoalsModel   under_05              73           11          15.1      0.04    -0.04  -100.0            0.0       -0.304            -0.038  
   4 │ DynamicGoalsModel         under_05              73           21          28.8      0.07    -0.07  -100.0            0.0       -0.353            -0.073  
Stats for: over_15
4×18 DataFrame
 Row │ model_name                selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth   
     │ String                    Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64            
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ DynamicMarketXGModel      over_15               73            1           1.4      0.06    -0.06  -100.0            0.0       -0.118            -0.06   
   2 │ DynamicXGModel            over_15               73            6           8.2      0.22    -0.16   -72.51          50.0       -0.13             -0.158  
   3 │ DynamicMarketGoalsModel   over_15               73            2           2.7      0.05    -0.05   -95.79          50.0       -0.116            -0.047  
   4 │ DynamicGoalsModel         over_15               73            1           1.4      0.0     -0.0   -100.0            0.0       -0.118            -0.0    
Stats for: under_15
4×18 DataFrame
 Row │ model_name                selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth   
     │ String                    Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64            
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ DynamicMarketXGModel      under_15              73           29          39.7      0.6     -0.02    -3.3           20.7       -0.009            -0.02   
   2 │ DynamicXGModel            under_15              73           36          49.3      0.88    -0.19   -21.39          30.6       -0.088            -0.188  
   3 │ DynamicMarketGoalsModel   under_15              73           29          39.7      0.64    -0.02    -3.69          20.7       -0.01             -0.024  
   4 │ DynamicGoalsModel         under_15              73           36          49.3      0.57    -0.29   -51.57          22.2       -0.183            -0.294  
Stats for: over_25
4×18 DataFrame
 Row │ model_name                selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth   
     │ String                    Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64            
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ DynamicMarketXGModel      over_25               73            5           6.8      0.16    -0.05   -29.64          60.0        0.001            -0.046  
   2 │ DynamicXGModel            over_25               73           10          13.7      0.39    -0.19   -49.08          40.0       -0.133            -0.194  
   3 │ DynamicMarketGoalsModel   over_25               73            5           6.8      0.16    -0.02   -11.74          60.0        0.035            -0.019  
   4 │ DynamicGoalsModel         over_25               73            9          12.3      0.04    -0.02   -51.74          33.3       -0.127            -0.022  
Stats for: under_25
4×18 DataFrame
 Row │ model_name                selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth   
     │ String                    Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64            
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ DynamicMarketXGModel      under_25              73           28          38.4      0.88    -0.22   -24.52          50.0       -0.111            -0.217  
   2 │ DynamicXGModel            under_25              73           35          47.9      1.16    -0.01    -0.66          51.4       -0.004            -0.008  
   3 │ DynamicMarketGoalsModel   under_25              73           30          41.1      0.97    -0.26   -26.89          46.7       -0.128            -0.26   
   4 │ DynamicGoalsModel         under_25              73           30          41.1      0.82    -0.15   -18.53          53.3       -0.083            -0.151  
Stats for: over_35
4×18 DataFrame
 Row │ model_name                selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth   
     │ String                    Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64            
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ DynamicMarketXGModel      over_35               73            8          11.0      0.08    -0.07   -84.05          25.0       -0.135            -0.07   
   2 │ DynamicXGModel            over_35               73           13          17.8      0.29    -0.08   -29.53          38.5       -0.064            -0.085  
   3 │ DynamicMarketGoalsModel   over_35               73            8          11.0      0.09    -0.07   -83.15          25.0       -0.136            -0.071  
   4 │ DynamicGoalsModel         over_35               73           10          13.7      0.04    -0.01   -26.92          10.0       -0.05             -0.011  
Stats for: under_35
4×18 DataFrame
 Row │ model_name                selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth   
     │ String                    Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64            
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ DynamicMarketXGModel      under_35              73           18          24.7      0.59    -0.12   -19.52          66.7       -0.098            -0.115  
   2 │ DynamicXGModel            under_35              73           27          37.0      0.77    -0.18   -22.9           59.3       -0.169            -0.176  
   3 │ DynamicMarketGoalsModel   under_35              73           18          24.7      0.65    -0.12   -19.11          66.7       -0.101            -0.125  
   4 │ DynamicGoalsModel         under_35              73           17          23.3      0.55    -0.22   -39.96          58.8       -0.175            -0.218  
=#

# ========================================
#  Stage 2 - Running inference 
# ========================================

# ---- 1. Load data and model.
saved_folders = BayesianFootball.Experiments.list_experiments(save_dir; data_dir="")
loaded_results = loaded_experiment_files(saved_folders)
expr = loaded_results[1]  # Grabbing the most recent one
exp1 = loaded_results[1]  # Grabbing the most recent one



todays_matches = fetch_todays_matches(ds)

target_ppd = compute_todays_matches_pdds(ds, expr, todays_matches)
target_ppd1 = compute_todays_matches_pdds(ds, exp1, todays_matches)



json_filepath = "/root/BayesianFootball/data/raw_odds_ireland_24_04_26.jsonl"
raw_live_market = load_live_market_jsonl(json_filepath)

selections_to_calibrate = [:over_15, :over_25,:under_25, :over_35, :under_35, :over_45, :under_45]
live_market_closing = filter_and_rename_live_markets(raw_live_market, selections_to_calibrate)
first(live_market_closing, 4)


pppd = Predictions.model_inference(ds, expr)


last(ds.matches, 20)


paper_bets_df = generate_paper_bets(
    target_ppd, 
    target_ppd1, 
    todays_matches, 
    first(live_market_closing, 4), 
    min_edge=0.0
)




# -------

using DataFrames, Statistics, HypothesisTests

function run_scoreboard_audit(matches::DataFrame)
    # 1. Setup Total Goals
    df = copy(matches)
    df.total_goals = df.home_score .+ df.away_score
    
    # 2. Group by Season
    # Note: we only include xG columns if they exist in the dataframe and aren't all missing
    println("\n" * "="^60)
    println(" SEASONAL GOAL & xG AUDIT ")
    println("="^60)
    
    stats = combine(groupby(df, :season), 
        :total_goals => mean => :Mean_Goals,
        :total_goals => std => :Std_Goals,
        :total_goals => median => :Med_Goals,
        :match_id => length => :Match_Count
    )
    
    # Add xG if it's the 2025/2026 period
    # (Assuming columns exist but might be missing for 2024)
    if "home_xg" in names(df)
        xg_stats = combine(groupby(subset(df, :has_xg => ByRow(identity)), :season),
            :home_xg => (x -> mean(skipmissing(x))) => :Mean_xG_H,
            :away_xg => (x -> mean(skipmissing(x))) => :Mean_xG_A
        )
        stats = leftjoin(stats, xg_stats, on=:season)
    end
    
    display(sort(stats, :season))
    
    # 3. Quick Volatility Check: High Scoring Game Frequency
    # Are we seeing a spike in "Outlier" games (e.g. 5+ goals)?
    df.is_over_35 = df.total_goals .> 3.5
    vol_stats = combine(groupby(df, :season), :is_over_35 => mean => :Over35_Freq)
    
    println("\n--- Outlier Frequency (>3.5 Goals) ---")
    display(sort(vol_stats, :season))
end


function run_market_audit(matches::DataFrame, odds::DataFrame)
    # 1. Join Season into Odds
    season_lookup = matches[:, [:match_id, :season]]
    df_odds = innerjoin(odds, season_lookup, on=:match_id)
    
    println("\n" * "="^60)
    println(" MARKET EFFICIENCY & PRICING SHIFT ")
    println("="^60)
    
    # 2. Outcome Distribution (Has the Draw disappeared? Has Home Advantage died?)
    # We use matches for this as it's cleaner
    outcome_dist = combine(groupby(matches, :season), 
        :winner_code => (x -> count(i -> i == 1, x) / length(x)) => :Home_Win_Pct,
        :winner_code => (x -> count(i -> i == 3, x) / length(x)) => :Draw_Win_Pct,
        :winner_code => (x -> count(i -> i == 2, x) / length(x)) => :Away_Win_Pct
    )
    println("--- Outcome Distribution by Season ---")
    display(sort(outcome_dist, :season))

    # 3. Market Pricing Shift (Specifically for Totals if available)
    # We check if the average Closing Implied Probability of 'Over' is rising
    if any(df_odds.market_name .== "overunder")
        totals = subset(df_odds, :market_name => ByRow(==("overunder")), :market_line => ByRow(==(2.5)))
        pricing_shift = combine(groupby(totals, [:season, :selection]),
            :prob_implied_close => mean => :Avg_Implied_Prob,
            :is_winner => mean => :Actual_Win_Rate,
            :overround_close => mean => :Avg_Vig
        )
        println("\n--- Over/Under 2.5 Market Efficiency ---")
        display(sort(pricing_shift, [:season, :selection]))
    end
end


run_scoreboard_audit(ds.matches)




#=
julia> run_scoreboard_audit(ds.matches)

============================================================
 SEASONAL GOAL & xG AUDIT 
============================================================
6×5 DataFrame
 Row │ season   Mean_Goals  Std_Goals  Med_Goals  Match_Count 
     │ String?  Float64     Float64    Float64    Int64       
─────┼────────────────────────────────────────────────────────
   1 │ 2021        2.54444    1.37138        2.0          180
   2 │ 2022        2.50556    1.51528        2.0          180
   3 │ 2023        2.60556    1.6048         2.0          180
   4 │ 2024        2.26667    1.50456        2.0          180
   5 │ 2025        2.42778    1.5428         2.0          180
   6 │ 2026        2.68493    1.76286        2.0           73

--- Outlier Frequency (>3.5 Goals) ---
6×2 DataFrame
 Row │ season   Over35_Freq 
     │ String?  Float64     
─────┼──────────────────────
   1 │ 2021        0.222222
   2 │ 2022        0.238889
   3 │ 2023        0.283333
   4 │ 2024        0.194444
   5 │ 2025        0.205556
   6 │ 2026        0.328767
=#

run_market_audit(ds.matches, ds.odds)


using DataFrames, Statistics

function run_full_market_audit(matches::DataFrame, odds::DataFrame)
    # 1. Join Season into Odds (dropping missing matches to be safe)
    season_lookup = dropmissing(matches[:, [:match_id, :season]])
    df_odds = innerjoin(odds, season_lookup, on=:match_id)
    
    println("\n" * "="^70)
    println(" FULL MARKET EFFICIENCY & PRICING SHIFT ")
    println("="^70)
    
    # 2. Define the core markets we care about for the Audit
    # We will exclude the crazy lines (over_65, under_95, etc.) to keep the output readable
    core_selections = [
        :home, :draw, :away, 
        :btts_yes, :btts_no, 
        :DC_1X, :DC_X2, :DC_12,
        :over_15, :under_15, 
        :over_25, :under_25, 
        :over_35, :under_35
    ]
    
    # 3. Filter down to core selections
    df_core = subset(df_odds, :selection => ByRow(x -> x in core_selections))
    
    # 4. Group by Selection, then Season
    pricing_shift = combine(groupby(df_core, [:selection, :season]),
        :prob_implied_close => mean => :Avg_Implied_Prob,
        :is_winner => (x -> mean(skipmissing(x))) => :Actual_Win_Rate,
        :overround_close => (x -> mean(skipmissing(x))) => :Avg_Vig
    )
    
    # 5. Calculate the Bookmaker's "Hold" or Error (Implied Prob - Actual Win Rate)
    # If this is highly positive, the bookmaker is overcharging. If negative, the market is offering value.
    pricing_shift.Market_Error = pricing_shift.Avg_Implied_Prob .- pricing_shift.Actual_Win_Rate
    
    # 6. Print cleanly grouped by Selection
    for sel in core_selections
        println("\n--- SELECTION: $(sel) ---")
        sel_data = subset(pricing_shift, :selection => ByRow(==(sel)))
        display(sort(sel_data, :season))
    end
end

# Run the function
run_full_market_audit(ds.matches, ds.odds)

#=
julia> run_market_audit(ds.matches, ds.odds)

============================================================
 MARKET EFFICIENCY & PRICING SHIFT 
============================================================
--- Outcome Distribution by Season ---
6×4 DataFrame
 Row │ season   Home_Win_Pct  Draw_Win_Pct  Away_Win_Pct 
     │ String?  Float64       Float64       Float64      
─────┼───────────────────────────────────────────────────
   1 │ 2021         0.422222      0.244444      0.333333
   2 │ 2022         0.438889      0.266667      0.294444
   3 │ 2023         0.45          0.25          0.3
   4 │ 2024         0.416667      0.3           0.283333
   5 │ 2025         0.427778      0.288889      0.283333
   6 │ 2026         0.39726       0.356164      0.246575
=#
# The Fixed Over/Under Audit
df_odds = ds.odds
totals = subset(df_odds, 
    :market_name => ByRow(x -> lowercase(x) == "overunder"), 
    :market_line => ByRow(==(2.5))
)
pricing_shift = combine(groupby(totals, [:season, :selection]),
    :prob_implied_close => mean => :Avg_Implied_Prob,
    :is_winner => mean => :Actual_Win_Rate,
    :overround_close => mean => :Avg_Vig
)
println("\n--- Over/Under 2.5 Market Efficiency ---")
display(sort(pricing_shift, [:season, :selection]))




using DataFrames, Statistics

function run_regime_audit(matches::DataFrame, odds::DataFrame)
    println("\n" * "="^60)
    println(" 1. THE SCOREBOARD AUDIT (MACRO PHYSICS) ")
    println("="^60)
    
    # Create total goals and outlier metrics
    df_m = copy(matches)
    df_m.total_goals = df_m.home_score .+ df_m.away_score
    df_m.is_over_35 = df_m.total_goals .> 3.5

    # Group by season
    goal_stats = combine(groupby(df_m, :season),
        :total_goals => (x -> round(mean(x), digits=3)) => :Avg_Goals,
        :total_goals => (x -> round(std(x), digits=3)) => :Std_Goals,
        :is_over_35 => (x -> round(mean(x) * 100, digits=2)) => :Pct_Over_35,
        :match_id => length => :Sample_Size
    )
    display(sort(goal_stats, :season))


    println("\n" * "="^60)
    println(" 2. THE LEDGER AUDIT (MARKET PRICING SHIFT) ")
    println("="^60)
    
    # Join the season column into the odds dataframe
    df_o = innerjoin(odds, matches[!, [:match_id, :season]], on=:match_id)
    
    # We will specifically audit the Over 2.5 and Away markets as they are the most sensitive
    audit_selections = [:over_25, :away, :draw]
    
    for sel in audit_selections
        println("\n--- Tracking Selection: $(sel) ---")
        sel_df = subset(df_o, :selection => ByRow(==(sel)))
        
        if nrow(sel_df) > 0
            market_stats = combine(groupby(sel_df, :season),
                :prob_implied_close => (x -> round(mean(x) * 100, digits=2)) => :Market_Implied_Pct,
                :is_winner => (x -> round(mean(skipmissing(x)) * 100, digits=2)) => :Actual_Hit_Rate,
                :overround_close => (x -> round(mean(x), digits=4)) => :Avg_Vig
            )
            # Calculate the Bookmaker's Error
            market_stats.Bookie_Error_Pct = round.(market_stats.Market_Implied_Pct .- market_stats.Actual_Hit_Rate, digits=2)
            display(sort(market_stats, :season))
        else
            println("No data found for selection: $sel")
        end
    end
end

# Execute the audit
run_regime_audit(ds.matches, ds.odds)



#=
julia> # Execute the audit
       run_regime_audit(ds.matches, ds.odds)

============================================================
 1. THE SCOREBOARD AUDIT (MACRO PHYSICS) 
============================================================
6×5 DataFrame
 Row │ season   Avg_Goals  Std_Goals  Pct_Over_35  Sample_Size 
     │ String?  Float64    Float64    Float64      Int64       
─────┼─────────────────────────────────────────────────────────
   1 │ 2021         2.544      1.371        22.22          180
   2 │ 2022         2.506      1.515        23.89          180
   3 │ 2023         2.606      1.605        28.33          180
   4 │ 2024         2.267      1.505        19.44          180
   5 │ 2025         2.428      1.543        20.56          180
   6 │ 2026         2.685      1.763        32.88           73

============================================================
 2. THE LEDGER AUDIT (MARKET PRICING SHIFT) 
============================================================

--- Tracking Selection: over_25 ---
6×5 DataFrame
 Row │ season   Market_Implied_Pct  Actual_Hit_Rate  Avg_Vig  Bookie_Error_Pct 
     │ String?  Float64             Float64          Float64  Float64          
─────┼─────────────────────────────────────────────────────────────────────────
   1 │ 2021                  47.41            46.93   1.0616              0.48
   2 │ 2022                  49.67            48.04   1.0601              1.63
   3 │ 2023                  51.32            45.75   1.0604              5.57
   4 │ 2024                  48.43            41.67   1.0604              6.76
   5 │ 2025                  47.94            46.11   1.0602              1.83
   6 │ 2026                  50.09            46.58   1.0594              3.51

--- Tracking Selection: away ---
6×5 DataFrame
 Row │ season   Market_Implied_Pct  Actual_Hit_Rate  Avg_Vig  Bookie_Error_Pct 
     │ String?  Float64             Float64          Float64  Float64          
─────┼─────────────────────────────────────────────────────────────────────────
   1 │ 2021                  35.06            32.96   1.0733              2.1
   2 │ 2022                  35.91            29.44   1.0916              6.47
   3 │ 2023                  35.46            30.0    1.0867              5.46
   4 │ 2024                  33.62            28.33   1.0808              5.29
   5 │ 2025                  33.25            28.33   1.0838              4.92
   6 │ 2026                  32.92            24.66   1.0898              8.26

--- Tracking Selection: draw ---
6×5 DataFrame
 Row │ season   Market_Implied_Pct  Actual_Hit_Rate  Avg_Vig  Bookie_Error_Pct 
     │ String?  Float64             Float64          Float64  Float64          
─────┼─────────────────────────────────────────────────────────────────────────
   1 │ 2021                  27.81            24.58   1.0733              3.23
   2 │ 2022                  27.92            26.67   1.0916              1.25
   3 │ 2023                  27.07            25.0    1.0867              2.07
   4 │ 2024                  28.85            30.0    1.0808             -1.15
   5 │ 2025                  29.22            28.89   1.0838              0.33
   6 │ 2026                  29.14            35.62   1.0898             -6.48
=#


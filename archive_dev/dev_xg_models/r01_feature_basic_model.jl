# current_development/dev_xg_models/r01_basic_models.jl 
#
#=
Here we want to test the updated feature set and data 
work with the training and running of models via the package 
training-module and the experiment-module.
=#


using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
pinthreads(:cores)


include("./l01_feature_basic_model.jl")




ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())

save_dir::String = "./data/dev_xg_models/"

es = DSExperimentSettings(
  ds,
  "test_featureset_v2_",
  save_dir,
  get_target_seasons_string(ds.segment)
)

training_task = create_experiment_tasks(es)

results = run_experiment_task.(training_task)



saved_folders = Experiments.list_experiments(save_dir; data_dir="")
loaded_results = loaded_experiment_files(saved_folders);

exp = loaded_results[1]


using Turing
chain_fold_1 = exp.training_results[1][1]
chain_fold_2 = exp.training_results[2][1]
chain_fold_3 = exp.training_results[3][1]

chain_fold_6 = exp.training_results[6][1]


describe(chain_fold_1)
describe(chain_fold_2)
describe(chain_fold_3)


describe(chain_fold_6)


# 1. Extract the actual MCMCChains from your TrainingResults object
# (Adjust this depending on exactly how your training_results struct stores the chains)
all_chains = [res[1] for res in exp.training_results] 

# 2. Define the exact symbols you want to track
params_to_track = [
    :μ, 
    :γ, 
    :log_r, 
    Symbol("α.σ₀"), 
    Symbol("α.σₛ"), 
    Symbol("α.σₖ"),
    Symbol("β.σ₀"), 
    Symbol("β.σₛ"), 
    Symbol("β.σₖ")
]

# 3. Generate the Stability Report
stability_df = check_parameter_stability(all_chains, params_to_track)

display(stability_df)



### 
# Base.@kwdef struct AblationStudy_NB_home_hierarchy <: AbstractMultiScaledNegBinModel

#=
Experiments.ExperimentConfig(
            name = "$(label)_02_home_hierarchy",
            model = Models.PreGame.AblationStudy_NB_home_hierarchy(
                      μ = Normal(0.21, 0.05),
                      log_r = Normal(2.8, 0.1),
      ),
            splitter = cv_config,
            training_config = training_config,
            save_dir = save_dir
    ),
    ]
=#

save_dir::String = "./data/dev_xg_models/"

es = DSExperimentSettings(
  ds,
  "test_featureset_v2",
  save_dir,
  get_target_seasons_string(ds.segment)
)

training_task = create_experiment_tasks(es)

results = run_experiment_task.(training_task)


saved_folders = Experiments.list_experiments(save_dir; data_dir="")
loaded_results = loaded_experiment_files(saved_folders);




exp = loaded_results[1]

chain_fold_1 = exp.training_results[1][1]
chain_fold_2 = exp.training_results[2][1]
chain_fold_3 = exp.training_results[3][1]
chain_fold_6 = exp.training_results[6][1]


describe(chain_fold_1)
describe(chain_fold_2)
describe(chain_fold_3)



all_chains = [res[1] for res in exp.training_results] 
params_to_track_ha = [
    :μ, 
    :γ_global, 
    Symbol("γ_team.σ"),
    Symbol("γ_team.z[1]"),
    Symbol("γ_team.z[2]"),
    Symbol("γ_team.z[3]"),
    Symbol("γ_team.z[4]"),
    Symbol("γ_team.z[5]"),
    Symbol("γ_team.z[6]"),
    Symbol("γ_team.z[7]"),
    Symbol("γ_team.z[8]"),
    Symbol("γ_team.z[9]"),
    Symbol("γ_team.z[10]"),
    Symbol("γ_team.z[11]")
]

# 3. Generate the Stability Report
stability_df_ha = check_parameter_stability(all_chains, params_to_track_ha)

display(stability_df_ha)




#=
julia> stability_df_ha = check_parameter_stability(all_chains, params_to_track_ha)
9×29 DataFrame
 Row │ Fold   μ_mean    μ_std      γ_global_mean  γ_global_std  γ_team.σ_mean  γ_team.σ_std  γ_team.z[1]_mean  γ_team.z[1]_std  γ_team.z[2]_mean  γ_team.z[2]_std  γ_team.z[3]_mean  γ_team.z[3]_std  γ_team.z[4]_mean  γ_team.z[4]_std  γ_team.z[5]_mean  γ_team.z[5]_std  γ_team.z[6]_mean  γ_team.z[6]_std  γ_team.z[7]_mean  γ_team.z[7]_std  γ_team.z[8]_mean  γ_team.z[8]_std  γ_team.z[9]_mean  γ_team.z[9]_std  γ_team.z[10]_mean  γ_team.z[10]_std  γ_team.z[11]_mean  γ_team.z[11]_std 
     │ Int64  Float64?  Float64?   Float64?       Float64?      Float64?       Float64?      Float64?          Float64?         Float64?          Float64?         Float64?          Float64?         Float64?          Float64?         Float64?          Float64?         Float64?          Float64?         Float64?          Float64?         Float64?          Float64?         Float64?          Float64?         Float64?           Float64?          Float64?           Float64?         
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │     1  0.132019  0.0398623      0.0864228     0.0752806       0.120571     0.070586          -0.346481         0.929926       0.639744            0.901748          0.470669         0.923304         -0.866648         0.972572         -0.370766         0.908654          0.414065         0.894817         -0.170834         0.879021        -0.286873          0.932176          0.330237         0.896735           0.163915          0.919339   missing             missing        
   2 │     2  0.136438  0.0422806      0.108971      0.0740497       0.116699     0.0705154         -0.445853         0.856139       0.11208             0.984766          0.593308         0.937023          0.478342         0.886178         -0.863216         0.972258         -0.302676         0.885423          0.413967         0.863052        -0.138151          0.848501         -0.180015         0.893427           0.429733          0.892665         0.0117186           0.936215
   3 │     3  0.130799  0.0397976      0.0932159     0.0726074       0.115745     0.0704345         -0.306848         0.88273       -0.0556384           0.92905           0.595005         0.921286          0.469483         0.887968         -0.868441         1.01691          -0.426188         0.86741           0.382267         0.898789        -0.133194          0.861339         -0.201632         0.922911           0.416147          0.903939         0.102869            0.882389
   4 │     4  0.13682   0.0381911      0.11907       0.0674271       0.120887     0.0700332         -0.362702         0.909016       0.000336436         0.958467          0.622118         0.86444           0.504395         0.897805         -0.934424         1.00572          -0.302714         0.884451          0.56177          0.912336        -0.173734          0.856811         -0.352077         0.885563           0.531745          0.925216        -0.0374509           0.856752
   5 │     5  0.132987  0.0387477      0.114954      0.0632446       0.123609     0.0697297         -0.381024         0.892989      -0.114937            0.891479          0.664537         0.863456          0.506604         0.883027         -0.951265         0.958298         -0.376461         0.868221          0.625682         0.892714        -0.120685          0.834192         -0.450431         0.888371           0.580592          0.870147        -0.016537            0.919101
   6 │     6  0.120863  0.037697       0.149351      0.0633477       0.122058     0.0674631         -0.28751          0.860683      -0.263536            0.965264          0.879391         0.900385          0.412257         0.853122         -0.976799         1.00764          -0.290233         0.842295          0.622632         0.869818        -0.182579          0.861315         -0.363065         0.853718           0.348096          0.881282         0.0364632           0.866239
   7 │     7  0.120743  0.0354091      0.139176      0.0608065       0.115724     0.0634904         -0.278298         0.86476       -0.1662              0.941169          0.831266         0.912906          0.286912         0.877159         -0.92681          0.99105          -0.245991         0.857987          0.682414         0.874712        -0.107828          0.827156         -0.38966          0.858091           0.417883          0.842607        -0.0574662           0.861665
   8 │     8  0.119393  0.0350601      0.143092      0.0591514       0.110058     0.0641193         -0.12773          0.860056      -0.145462            0.96567           0.737904         0.910618          0.280706         0.914389         -0.900672         1.03788          -0.229942         0.877198          0.678851         0.903053        -0.130786          0.863716         -0.469316         0.866558           0.360744          0.902087        -0.00145922          0.863559
   9 │     9  0.112728  0.0364003      0.157122      0.0584867       0.10348      0.062394          -0.139989         0.832153      -0.112869            0.93296           0.65256          0.908324          0.24615          0.904535         -0.861029         1.01231          -0.241445         0.880996          0.59699          0.917999        -0.0906124         0.886032         -0.446061         0.89029            0.428889          0.888908        -0.0360244           0.908169
=#


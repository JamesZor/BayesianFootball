using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
pinthreads(:cores)




ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())

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
expr = loaded_results[1]




chain_fold_1 = expr.training_results[1][1]
chain_fold_2 = expr.training_results[2][1]
chain_fold_3 = expr.training_results[3][1]

chain_fold_6 = exp.training_results[6][1]


describe(chain_fold_1)
describe(chain_fold_2)


params_to_track_xg = [
    :μ, 
    :γ, 
    :log_r,
    :κ,          # NEW: Goal scaling factor
    :ν_xg,       # NEW: xG Gamma shape parameter
    Symbol("α.σ₀"), 
    Symbol("α.σₛ"), 
    Symbol("α.σₖ"),
    Symbol("β.σ₀"), 
    Symbol("β.σₛ"), 
    Symbol("β.σₖ")
]

all_chains = [res[1] for res in expr.training_results] 
# 3. Generate the Stability Report
stability_df_xg = check_parameter_stability(all_chains, params_to_track_xg)

display(stability_df_xg)


#=
julia> display(stability_df_xg)
9×23 DataFrame
 Row │ Fold   μ_mean    μ_std      γ_mean    γ_std      log_r_mean  log_r_std  κ_mean    κ_std      ν_xg_mean  ν_xg_std  α.σ₀_mean  α.σ₀_std   α.σₛ_mean  α.σₛ_std   α.σₖ_mean  α.σₖ_std   β.σ₀_mean  β.σ₀_std   β.σₛ_mean  β.σₛ_std   β.σₖ_mean  β.σₖ_std  
     │ Int64  Float64?  Float64?   Float64?  Float64?   Float64?    Float64?   Float64?  Float64?   Float64?   Float64?  Float64?   Float64?   Float64?   Float64?   Float64?   Float64?   Float64?   Float64?   Float64?   Float64?   Float64?   Float64?  
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │     1  0.124275  0.192675   0.227173  0.100949      2.81486   0.420345  0.883508  0.182324    10.425    4.63367    0.156404  0.0803867  0.0800811  0.0560835  0.030392   0.0213596   0.215342  0.0864149  0.0788119  0.0572069  0.0305208  0.0231636
   2 │     2  0.153538  0.0952939  0.289451  0.0831265     2.88063   0.414137  0.833608  0.0835335    2.48172  0.49505    0.135601  0.0697949  0.0644452  0.0411537  0.0300132  0.0202667   0.17412   0.0709391  0.0878831  0.0571618  0.0302158  0.0217475
   3 │     3  0.129658  0.0761726  0.283034  0.0720097     2.96151   0.391368  0.855437  0.0697389    2.75293  0.387372   0.119377  0.0630737  0.054582   0.0351659  0.02911    0.0202704   0.213171  0.0750036  0.108193   0.0658634  0.0303579  0.0211584
   4 │     4  0.190049  0.0588143  0.260546  0.0627007     3.05964   0.383846  0.840702  0.0550689    2.96966  0.31745    0.106814  0.0564821  0.0578707  0.0359935  0.0370336  0.0270462   0.201775  0.0690778  0.0907426  0.0552198  0.0276952  0.0195503
   5 │     5  0.197176  0.0534187  0.242074  0.05561       3.08326   0.391778  0.835424  0.051279     3.09783  0.298718   0.117058  0.0623157  0.0556722  0.0357954  0.0346851  0.0230472   0.197417  0.0672085  0.0942795  0.0543073  0.0275758  0.01814
   6 │     6  0.178906  0.0511412  0.274539  0.0555036     3.07642   0.371544  0.837511  0.0461426    3.08574  0.270797   0.110005  0.059124   0.0579961  0.036677   0.0373181  0.0237864   0.206886  0.0670791  0.0952504  0.0544903  0.0280304  0.0183142
   7 │     7  0.172876  0.0476263  0.271594  0.0507423     3.09452   0.38783   0.834902  0.0431688    2.98744  0.247512   0.115322  0.0622279  0.0628165  0.038424   0.032887   0.0212132   0.218799  0.0697947  0.114783   0.0605989  0.0295791  0.0196521
   8 │     8  0.174944  0.0448677  0.260051  0.0498894     3.10425   0.378087  0.843454  0.0432883    3.04903  0.252125   0.116201  0.0611877  0.0614584  0.0387209  0.0305504  0.0194088   0.216462  0.0690233  0.11412    0.0576362  0.0263901  0.0173789
   9 │     9  0.168033  0.0434578  0.273618  0.0468812     3.14798   0.376929  0.839503  0.0410575    3.10498  0.237303   0.115308  0.0591646  0.0607106  0.03705    0.0296024  0.018846    0.2053    0.0624474  0.110048   0.0543838  0.023761   0.0156078
=#


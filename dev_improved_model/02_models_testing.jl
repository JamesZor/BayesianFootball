using Revise
using BayesianFootball
using DataFrames

using Statistics
using ThreadPinning
using LinearAlgebra
pinthreads(:cores)


ds = Data.load_extra_ds()
transform!(ds.matches, :match_week => ByRow(w -> cld(w, 4)) => :match_month)


cv_config = BayesianFootball.Data.GroupedCVConfig(
    tournament_groups = [[56, 57]],
    target_seasons = ["24/25"],
    history_seasons = 3,
    dynamics_col = :match_month,
    warmup_period = 9,
    stop_early = true
)



# -------------------------------------
#  MS kappa
# -------------------------------------

model = BayesianFootball.Models.PreGame.MSNegativeBinomialKappa()

splits = BayesianFootball.Data.create_data_splits(ds, cv_config)

feature_sets = BayesianFootball.Features.create_features(
    splits, model, cv_config
)

splits = BayesianFootball.Data.create_data_splits(ds, cv_config) 
train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=2)  

sampler_conf = Samplers.NUTSConfig(                                                                                                                                                                                                                                     
                       200,                                                                                                                                                                                                                                                    
                       16,                                                                                                                                                                                                                                                     
                       100,                                                                                                                                                                                                                                                    
                       0.65,                                                                                                                                                                                                                                                   
                       10,                                                                                                                                                                                                                                                     
         Samplers.UniformInit(-1, 1),                                                                                                                                                                                                                                      
                       :perchain,                                                                                                                                                                                                                                              
       )

training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false) 

conf_basic = Experiments.ExperimentConfig(                                                                                                                                                                                                                              
                           name = "multi_basic_test",                                                                                                                                                                                                                          
                           model = model,                                                                                                                                                                                                                                      
                           splitter = cv_config,                                                                                                                                                                                                                               
                           training_config = training_config,                                                                                                                                                                                                                  
                           save_dir ="./data/junk"                                                                                                                                                                                                                             
       )  

results_basic = Experiments.run_experiment(ds, conf_basic)    






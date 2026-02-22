
using Revise
using BayesianFootball
using DataFrames


ds = Data.load_extra_ds()
transform!(ds.matches, :match_week => ByRow(w -> cld(w, 4)) => :match_month)

# check to see the months 
# df =subset(ds.matches, :tournament_id => ByRow(isequal(56)), :season => ByRow(isequal("24/25")))
# unique(df.match_month)

cv_config = BayesianFootball.Data.CVConfig(
    # tournament_ids = [56,57],
    tournament_ids = [56],
    target_seasons = ["24/25"],
    history_seasons = 2,
    dynamics_col = :match_month,
    # warmup_period = 36,
    warmup_period = 9,
    stop_early = true
)

model = BayesianFootball.Models.PreGame.MSNegativeBinomial()

splits = BayesianFootball.Data.create_data_splits(ds, cv_config)

feature_sets = BayesianFootball.Features.create_features(
    splits, model, cv_config
)
splits = BayesianFootball.Data.create_data_splits(ds, cv_config) 
train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=2)  
sampler_conf = Samplers.NUTSConfig(                                                                                                                                                                                                                                     
                       500,                                                                                                                                                                                                                                                    
                       16,                                                                                                                                                                                                                                                     
                       200,                                                                                                                                                                                                                                                    
                       0.65,                                                                                                                                                                                                                                                   
                       10,                                                                                                                                                                                                                                                     
         Samplers.UniformInit(-0.5, 0.5),                                                                                                                                                                                                                                      
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
c = results_basic.training_results[1][1]  


sigma_symbols = [                                                                                              
           Symbol("α.σ₀"), Symbol("α.σₛ"), Symbol("α.σₖ"),                                                            
           Symbol("β.σ₀"), Symbol("β.σₛ"), Symbol("β.σₖ")                                                             
       ]          

market_data = Data.prepare_market_data(ds)

latents = BayesianFootball.Experiments.extract_oos_predictions(ds, results_basic)


ppd = BayesianFootball.Predictions.model_inference(latents)



model_features = transform(ppd.df, :distribution => ByRow(mean) => :prob_model)
select!(model_features, :match_id, :market_name, :market_line, :selection, :prob_model)

analysis_df = innerjoin(
    market_data.df,
    model_features,
    on = [:match_id, :market_name, :market_line, :selection]
)

dropmissing!(analysis_df, [:odds_close, :is_winner])


dd =select(analysis_df, :match_id, :selection, :is_winner, :odds_open, :odds_close, :prob_implied_close, :prob_model)



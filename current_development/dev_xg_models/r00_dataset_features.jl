# current_development/dev_xg_models/r00_dataset_features.jl
#
# include("./l01_ireland.jl")


ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())



splitter_config = BayesianFootball.Data.StaticSplit(
    train_seasons = ["2025"], 
)

# 2. Splits
data_splits = Data.create_data_splits(ds, splitter_config)

# 3. Features
_log_step(3, "Building Feature Sets")
feature_sets = Features.create_features(
    data_splits, 
    config.model, 
    config.splitter 
)

# 4. Training
_log_step(4, "Executing Training Strategy")
training_results = Training.train(
    config.model, 
    config.training_config, 
    feature_sets
)


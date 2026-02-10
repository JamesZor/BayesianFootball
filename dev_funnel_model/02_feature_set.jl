# dev_funnel_model/02_feature_set.jl

using Revise
using BayesianFootball

ds = Data.load_extra_ds()





# ----------------------------------------------
# 2. Experiment Configs Set up 
# ----------------------------------------------

# --- setup 1 
cv_config = BayesianFootball.Data.CVConfig(
    # tournament_ids = [56,57],
    tournament_ids = [56],
    target_seasons = ["22/23"],
    history_seasons = 0,
    dynamics_col = :match_week,
    # warmup_period = 36,
    warmup_period = 36,
    stop_early = true
)

splits = BayesianFootball.Data.create_data_splits(ds, cv_config)
model = BayesianFootball.Models.PreGame.StaticPoisson() # place holder
# vocabulary = BayesianFootball.Features.create_vocabulary(ds, model) 
feature_sets = BayesianFootball.Features.create_features(
    splits, model, cv_config
)
feature_sets

###

struct test_model <: BayesianFootball.AbstractFunnelModel end

feature_sets = BayesianFootball.Features.create_features(
  splits, test_model(), cv_config
)
feature_sets[1]



# In your REPL or dev script
funnel_model = BayesianFootball.Models.PreGame.SequentialFunnelModel()

feature_sets = BayesianFootball.Features.create_features(
    splits, funnel_model, cv_config
)

# Compile and Sample (Fast run to check for errors)
# It will be slower than Poisson because it has 6 GRW chains instead of 2!
using Turing
turing_model = BayesianFootball.Models.PreGame.build_turing_model(funnel_model, feature_sets[1][1])
chain = sample(turing_model, NUTS(20, 0.65), 10)

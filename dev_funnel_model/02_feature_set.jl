# dev_funnel_model/02_feature_set.jl

using Revise
using BayesianFootball

pinthreads(:cores)
ds = Data.load_extra_ds()





# ----------------------------------------------
# 2. Experiment Configs Set up 
# ----------------------------------------------

# --- setup 1 
cv_config = BayesianFootball.Data.CVConfig(
    # tournament_ids = [56,57],
    tournament_ids = [56],
    target_seasons = ["24/25"],
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



## --- 

using Revise
using BayesianFootball

using DataFrames
using Statistics
using ThreadPinning
using LinearAlgebra
pinthreads(:cores)
BLAS.set_num_threads(1) 


ds = Data.load_extra_ds()
#=
julia> names(df)
26-element Vector{String}:
 "tournament_id"
 "season_id"
 "season"
 "match_id"
 "tournament_slug"
 "home_team"
 "away_team"
 "home_score"
 "away_score"
 "home_score_ht"
 "away_score_ht"
 "match_date"
 "round"
 "winner_code"
 "has_xg"
 "has_stats"
 "match_hour"
 "match_dayofweek"
 "match_month"
 "match_week"
 "HS"
 "AS"
 "HST"
 "AST"
 "HC"
 "AC"
=#

transform!(ds.matches, :match_week => ByRow(w -> cld(w, 4)) => :match_month)

df = subset(ds.matches, :tournament_id => ByRow(in(56)), :season => ByRow(isequal("24/25")))
df[:, [:home_team, :away_team, :match_date, :match_week, :match_month]] 


cv_config = BayesianFootball.Data.CVConfig(
    # tournament_ids = [56,57],
    tournament_ids = [56],
    target_seasons = ["24/25"],
    history_seasons = 0,
    dynamics_col = :match_month,
    # warmup_period = 36,
    warmup_period = 8,
    stop_early = true
)


splits = BayesianFootball.Data.create_data_splits(ds, cv_config)
train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=1) 
sampler_conf = Samplers.NUTSConfig(
                200,
                16,
                100,
                0.65,
                10,
  Samplers.UniformInit(-0.05, 0.05),
                :perchain,
)

training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)


funnel_model = BayesianFootball.Models.PreGame.SequentialFunnelModel()

conf_funnel = Experiments.ExperimentConfig(
                    name = "grw funnel_model",
                    model = funnel_model,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./dev_data/"
)


results_funnel = Experiments.run_experiment(ds, conf_funnel)


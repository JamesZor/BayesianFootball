
using Revise
using BayesianFootball

using DataFrames
using Statistics
using ThreadPinning
using LinearAlgebra
pinthreads(:cores)


# 1. Setup Data & Splits
# ======================
ds = Data.load_extra_ds()

# default 
# transform!(ds.matches, :match_week => ByRow(w -> cld(w, 4)) => :match_month)
# testing the grw GRWNegativeBinomialMu at every 2 weeks
transform!(ds.matches, :match_week => ByRow(w -> cld(w, 4)) => :match_month)

using DataFramesMeta

df_56 = subset(ds.matches, :tournament_id => ByRow(isequal(56)), :season => ByRow(isequal("25/26")))
unique(df_56.match_month)
@rsubset(df_56, :match_month == 13 )


cv_config_l1 = BayesianFootball.Data.CVConfig(
    tournament_ids = [56],       # Premiership
    target_seasons = ["25/26"],  # Target Season
    history_seasons = 0,
    dynamics_col = :match_month,
    warmup_period = 4,      
    stop_early = false
)
splits_l1 = BayesianFootball.Data.create_data_splits(ds, cv_config_l1)




df_57 =subset(ds.matches, :tournament_id => ByRow(isequal(57)), :season => ByRow(isequal("25/26")))
unique(df_57.match_month)

@rsubset(df_57, :match_month == 7 )

cv_config_l2 = BayesianFootball.Data.CVConfig(
    tournament_ids = [57],       # Premiership
    target_seasons = ["25/26"],  # Target Season
    history_seasons = 0,
    dynamics_col = :match_month,
    warmup_period = 7,      
    stop_early = false
)
splits_l2 = BayesianFootball.Data.create_data_splits(ds, cv_config_l2)


unique( df_56.home_team)

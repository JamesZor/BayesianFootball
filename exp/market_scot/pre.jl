
using Revise
using BayesianFootball

using DataFrames
using Statistics
using ThreadPinning
using LinearAlgebra
pinthreads(:cores)



using Turing

data_store = BayesianFootball.Data.load_default_datastore()

ds = BayesianFootball.Data.DataStore( 
    Data.add_match_week_column(data_store.matches),
    data_store.odds,
    data_store.incidents
)

df =subset(ds.matches, :tournament_id => ByRow(isequal(56)), :season => ByRow(isequal("25/26")))

unique(df.match_month)

df

df =subset(ds.matches, :tournament_id => ByRow(isequal(57)), :season => ByRow(isequal("25/26")))



    cv_config = BayesianFootball.Data.GroupedCVConfig(
        tournament_groups = [[56, 57]],
        target_seasons = ["25/26"],
        history_seasons = 2,
        dynamics_col = :match_month,
        warmup_period = 12,
        stop_early = false
    )


splits_l1 = BayesianFootball.Data.create_data_splits(ds, cv_config)






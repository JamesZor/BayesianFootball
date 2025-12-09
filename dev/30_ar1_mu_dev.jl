using BayesianFootball
using DataFrames


tournament_id = 55 
data_store = BayesianFootball.Data.load_default_datastore()
ds = BayesianFootball.Data.DataStore( 
    Data.add_match_week_column(subset( data_store.matches, 
           :tournament_id => ByRow(isequal(tournament_id)),
                                      :season => ByRow(isequal("24/25")))),
    data_store.odds,
    data_store.incidents
)

model = BayesianFootball.Models.PreGame.AR1Poisson()

required_mapping_keys(model)

struct test_model <: BayesianFootball.TypesInterfaces.AbstractPregameModel end 

model_t = test_model()

mapping_keys = required_mapping_keys(model_t)

function required_mapping_keys(model::test_model)
    # By default, we assume all models need at least a team mapping.
    return [:team_map, :n_teams, :league_map, :n_league]
end

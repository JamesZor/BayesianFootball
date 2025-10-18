using Revise

using BayesianFootball



## 1. Load Data
data_store = BayesianFootball.Data.load_default_datastore()


# 2. Define an Experiment (your "Config")


model =  Models.PreGame.StaticPoisson()
splitter = Experiments.StaticSplit(["24/25"])
sampler_config = Sampling.NUTSMethod(1000, 2, 100)
markets = Predictions.get_standard_markets()
calculations_set = Set{Predictions.Calculations.AbstractCalculation}([
    Predictions.Calculations.CalcProbability() 
    # You could add others here, e.g.:
    # Predictions.Calculations.CalcExpectedValue()
])

prediction_config = Predictions.PredictionConfig(
    markets,
    calculations_set
)


exp1 = Experiments.Experiment(
    name = "StaticPoisson_ExpandingWindow_22-23",
    model = BayesianFootball.Models.PreGame.StaticPoisson(),
    splitter = Experiments.StaticSplit(
    ),
    sampler_config = BayesianFootball.Sampling.NUTSMethod(n_samples=1000, n_chains=2, n_warmup=500),
    prediction_config = PredictionConfig(
        markets = [:1x2, :over_25, :btts],
        calculate_ev = true,
        calculate_kelly = false
    )
)



# 2. Define an Experiment (your "Config")14
exp1 = Experiment(
    name = "StaticPoisson_ExpandingWindow_22-23",
    model = BayesianFootball.Models.PreGame.StaticPoisson(),
    splitter = ExpandingWindowCV(
        base_seasons = ["20/21", "21/22"],
        target_seasons = ["22/23"],
        round_col = :global_round, # or whatever col you use
        ordering = :sequential
    ),
    sampler_config = BayesianFootball.Sampling.NUTSMethod(n_samples=1000, n_chains=2, n_warmup=500),
    prediction_config = PredictionConfig(
        markets = [:1x2, :over_25, :btts],
        calculate_ev = true,
        calculate_kelly = false
    )
)

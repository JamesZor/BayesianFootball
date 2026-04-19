# dev/34_experiment_module_dev.jl


"""

design notes overview:

This module is a wrapper for the configs files for running a model / sampling / experiment. 
It allows for us to define a standrisds config sets in order to run a range of experiments. The 
experiments achieves this by its struct and kwags, and some simple funcitons that aid this. 

The somplest example is to compare models, say we have a set of models M = {m_1 , ... m_n}, 
we can define function/ struct such that: 
  ExperimentConfig(m_i | C) for i in 1:N 
were C denotes the config set which we define as the following elements: 

cvconfig / data splitting -> AbstractSplitter ( src/data/splitter) 
  - this controls how the data is broken up for the sampling. 

traing.independant -> AbstractExecutionStrategy ( src/training/train-module.jl)
- defines how multiple splits are handled. parrellel or sequentual. 

training-config -> overall config for the training process combines the smapler and stratergy. 

sampler-config  -> Abstract sampler config ( src/samplers/samplers-module.jl)
  - Describes train method and config - such as nuts, and its configs etc. 

So in theory with abuse of notation we want a flexiable define the experimentconfig as we could fix 
the model but change an element sampler_config C_sample: 
ExperimentConfig( C_sampler | C' ) 

were C' = (M, C_splitter, C_train, C_indep , Meta) with M being the model, 
C_splitter being the data splitter / cvsplitter config, C_train the training config, C_indep being 
the independant and Meta being some meta data regarding the experiment for house keeping such as 
the name of the experiment etc 

Hence we can have a function 
RunExperiment( Data.DataStore , ExperimentConfig) 
this returns a struct containing the results and the experiment_config 

The struct, ExperimentResults is what is saved and use furhter for the prediction and analysis. 
Since everything can be be more or less synethisis from the configs and the results such as the predictions 

Note the ExperimentConfig thing is a round about way of say that there exist a wrapper function for 
it/api - such that we create an experiment, one defines this function and the set of parameters to 
iterate / models - like in the the model case, 

ExpCong( M_i | C) -> ExpResult_i 

so we can save the expresult etc 



### 

Im struggling to find the correct software engineer/design terms here for how 
i have this to be design and function. Lets start with an example of how I believe the best 
way of using this - though i could be wrong and it be a terrible design pattern so feel, free 
to correct me or discuss changes to improve. 

Consider the case when we want to compare a couple of models, M, in that we want to ensure 
all other variables of the experiment are fixed - which we do through the config. So in this stage 
we to define the fix variables, and some models, so we can run sampling of the inference process, 
- so we can leave and come back to since they will take awhile - so we can save as process and then 
load them another time - or scp out of the server to my laptop to run analysis. 

So when we write an experiment - we want write a function/ constructure, of the ExperimentConfig,
that is conditioned on that experiments define fix variables in the example case is C, but have it so its 
a funciton / constructor that takes an argument of the model, and name : 

MyDefineExperiment_1(M,Name | C ) -> ExperimentConfig( M, name, C_splitter, C_train, C_indep, Meta) 

hence we then in this example can define models, 
model_1 =  Models.Pregame.XXXX() 
model_1_name = "X_model"
model_2 =  Models.Pregame.YYYY() 
model_2_name = "Y_model"


results_1 = Experiments.run_experiment(ds, MyDefineExperiment_1(model_1, model_1_name) ; save_dir="./experiments")
results_2 = Experiments.run_experiment(ds, MyDefineExperiment_1(model_2, model_2_name) ; save_dir="./experiments")

Note we can possible write a loop for this, etc but this showing for the work example. 

But now consider the example that we want to run experiments on the effects of number of samples, then we 
want the flexiable to write a function/ constructor such that in the experiment note files we are working in 

MyDefineExperiment_2(number_samples ,Name | C ) -> ExperimentConfig( M, name, C_splitter, C_train, C_indep, Meta) 

for i in 100:100:1_000
  results_i = Experiments.run_experiment(ds, MyDefineExperiment_2(i, "$i samples") ; save_dir="./experiments/sample_tests")
end




in your proposed strutcure :
struct ExperimentResults
    config::ExperimentConfig
    
    # The raw posterior chains from training
    # Keyed by split ID or name (e.g., "Split_1", "Split_2")
    training_results::Dict{Any, Any} 
    
    # The vocabulary used (critical for encoding/decoding)
    vocabulary::Vocabulary
    
    # Paths to where artifacts were saved (for lazy loading later)
    save_path::String
end


you define -> training_results::Dict{Any, Any}  as a dict, but the results are a vector of tuples the element at [1] is 
the turing trained chain, and element [2] is the split meta data 
defined in the src/data/splitting
struct SplitMetaData <: AbstractSplitMetaData
    tournament_id::Int
    train_season::String       # The primary target season (e.g. "20/21")
    target_season::String      # Usually same as train_season in expanding window
    history_depth::Int         # Number of historical seasons included
    time_step::Int             # The dynamics index (e.g. match_week 5)
    warmup_period::Int         # The starting dynamics index
end



Decision: Do we agree to move the ExecutionStrategy (parallel vs serial) inside the ExperimentConfig? This ensures that "how we run it" is part of the reproducible record.

sort of, maybe use the base.@kwdef to pre define defults but allows use to change it if need. 

Artifact Saving:
        Save chains, vocabulary, and the ExperimentConfig itself (as JSON/JLD2).
hmm, i was think of just saving the ExperimentResults  as it keeps all together -
though we might need in progress saving, so if something goes wrong we can resume. but can add this latter. 


Discussion Point B: In dev/31, you manually create data_splits and then feature_sets.

    Question: Should run_experiment handle the creation of feature_sets internally?
    My Recommendation: Yes. The ExperimentConfig contains the Splitter and Model, which are the only ingredients needed to create features. This removes the biggest chunk of boilerplate.
    yeah, this should help reduce the bolier plate and create a good level of abstraction. 

Question: Should run_experiment automatically run the OOS predictions?
no, we will run that at a later stage, since we need to analysis the chains for convergance etc 
so if we just do a prediction we shall loose information, and or store more information than need, since it is 
cheap to run - i think 

4. Alignment Check: Splitter Structs

There is a mismatch between src/experiments/types.jl and your usage in dev/31.

    types.jl: ExpandingWindowCV has base_seasons, target_seasons...

    dev/31: ExpandingWindowCV has train_seasons, test_seasons, window_col, dynamics_col...

Action Item: We need to update src/experiments/types.jl to match the fields you are actually using in development (dev/31).
possible, we can just use the abstract struct of this - but yeah dev/31 is newer and is the current framework. 


###

Can we not have  a constructor in the types - which is more or less a defult loadout config 

# src/experiments/types.jl
# --- 1. Define the "Factory" (The Fixed Context) ---
function ExperimentConfig_default()
    
    # Define the Fixed Splitter
    splitter = BayesianFootball.Data.ExpandingWindowCV(
        train_seasons = [], 
        test_seasons = ["24/25"], 
        window_col = :split_col,
        method = :sequential,
        dynamics_col = :match_week
    )

    # Define the Fixed Training Strategy
    train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=4)
    sampler_conf = BayesianFootball.Samplers.NUTSConfig(n_samples=samples, n_chains=2, n_warmup=300)
    training_conf = BayesianFootball.Training.TrainingConfig(sampler_conf, train_cfg)

    # Return the Config
    return BayesianFootball.Experiments.ExperimentConfig(
        name = model_name,
        model = model,
        splitter = splitter,
        training_config = training_conf,
        tags = ["benchmark", "dev"],
        description = "Benchmarking $model_name on 24/25 season"
    )
end


then in experiment file we are working in use the

# 1. Define a master constant (The Prototype)
const BENCHMARK_DEFAULTS = ExperimentConfig(
    name = "Default",
    model = StaticPoisson(), # Placeholder
    splitter = ExpandingWindowCV(...),
    training_config = ...
)

but wrap it in funcitons so car pass like the models etc things of interest. 
function create_benchmark_config(model, model_name; samples=500)
# "Take the defaults, but change the model and name"
my_config = @set BENCHMARK_DEFAULTS.model = AR1Poisson()
my_config = @set my_config.name = "AR1 Run"
return 

then possible also use the base.show, 
which will nicely print out all the configs, - and indicates which parameters or fixable maybe 

show(BENCHMARK_DEFAULTS )



"""

using Revise
using BayesianFootball

using DataFrames
using Statistics
using ThreadPinning
pinthreads(:cores)


data_store = BayesianFootball.Data.load_default_datastore()
ds = BayesianFootball.Data.DataStore( 
    Data.add_match_week_column(data_store.matches),
    data_store.odds,
    data_store.incidents
)


model = Models.PreGame.StaticPoisson()


# -- Experiments deconstructed walk-through 
cv_config = BayesianFootball.Data.CVConfig(
    tournament_ids = [55],
    target_seasons = ["21/22"],
    history_seasons = 0, # Will auto-include "23/24" if available
    dynamics_col = :match_week,
    warmup_period = 34,
    stop_early = true  # Splits go 1..5, 1..6, ..., 1..37
    # stop_early = false # Splits go 1..5, ..., 1..38 (The default)
)

splits = Data.create_data_splits(ds, cv_config)



vocabulary = BayesianFootball.Features.create_vocabulary(ds, model) 

# 2. Create Features (NOW AUTOMATICALLY ADAPTS VOCAB)
feature_sets = BayesianFootball.Features.create_features(
    splits, vocabulary, model, cv_config
)

train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=2) 

sampler_conf = BayesianFootball.Samplers.NUTSConfig(n_samples=100, n_chains=1, n_warmup=100) # Use renamed struct

training_config = BayesianFootball.Training.TrainingConfig(sampler_conf, train_cfg)


experimentconfig = BayesianFootball.Experiments.ExperimentConfig(
                  "test config",
                  model,
                  cv_config,
                  training_config,
                  ["benchmark", "test", "dev"],
                  " This is a test example of the experimentconfig to see if it can work. ",
                  "./experiments"
)

 experimentconfig = BayesianFootball.Experiments.ExperimentConfig(
                         name = "test config",
                         model = model,
                         splitter = cv_config,
                         training_config = training_config,
                         tags = ["benchmark", "test", "dev"],
                        description = " 
                               This is a test example of the experimentconfig to see if it can work. 
                             "

       ) 


exp_results = BayesianFootball.Experiments.run_experiment(ds, experimentconfig)



function experiment_config_models( models, names)::Experiments.ExperimentConfig

        cv_config = BayesianFootball.Data.CVConfig(
            tournament_ids = [55],
            target_seasons = ["21/22"],
            history_seasons = 0, # Will auto-include "23/24" if available
            dynamics_col = :match_week,
            warmup_period = 34,
            stop_early = true  # Splits go 1..5, 1..6, ..., 1..37
            # stop_early = false # Splits go 1..5, ..., 1..38 (The default)
        )


        train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=2) 

        sampler_conf = BayesianFootball.Samplers.NUTSConfig(n_samples=100, n_chains=1, n_warmup=100) # Use renamed struct

        training_config = BayesianFootball.Training.TrainingConfig(sampler_conf, train_cfg)

 experimentconfig = BayesianFootball.Experiments.ExperimentConfig(
                         name = names,
                         model = models,
                         splitter = cv_config,
                         training_config = training_config,
                         tags = ["benchmark", "test", "dev"],
                        description = " 
                               This is a test example of the experimentconfig to see if it can work. 
                             "
       ) 

    return experimentconfig

end


using Distributions
model_1 = Models.PreGame.StaticPoisson()
name_1 = "normal"

model_2 = Models.PreGame.StaticPoisson(prior= Cauchy(0))
name_2 = "Cauchy"

model_3 = Models.PreGame.AR1Poisson()
name_3 = "ar1"

model_4 = Models.PreGame.GRWPoisson()
name_4 = "grw"


cfg_1 = Experiments.experiment_config_models(model_1, name_1)
cfg_3 = Experiments.experiment_config_models(model_3, name_3)

cfg_4 = Experiments.experiment_config_models(model_4, name_4)

cfg_2 = experiment_config_models(model_2, name_2)



""""
using the module 

"""
# using Revise

using BayesianFootball

using DataFrames
using Statistics
using ThreadPinning
pinthreads(:cores)

using Distributions

data_store = BayesianFootball.Data.load_default_datastore()
ds = BayesianFootball.Data.DataStore( 
    Data.add_match_week_column(data_store.matches),
    data_store.odds,
    data_store.incidents
)


model_1 = Models.PreGame.StaticPoisson()
name_1 = "normal"

model_2 = Models.PreGame.StaticPoisson(prior= Cauchy(0))
name_2 = "Cauchy"


cfg_1 = Experiments.experiment_config_models(model_1, name_1)

cfg_2 = Experiments.experiment_config_models(model_2, name_2)

# Clean output, no I/O clutter
results1 = Experiments.run_experiment(ds, cfg_1)
results2 = Experiments.run_experiment(ds, cfg_2)

# Explicit saving (Safety)
Experiments.save_experiment(results1)
Experiments.save_experiment(results2)


# 1. List them (and capture the list)
exps = Experiments.list_experiments("experiments")

# 2. Load the one you want using the index
old_results = Experiments.load_experiment(exps, 2)




old_results


"""
 :n_steps
 :is_accept
 :acceptance_rate
 :log_density
 :hamiltonian_energy
 :hamiltonian_energy_error
 :max_hamiltonian_energy_error
 :tree_depth
 :numerical_error
 :step_size
 :nom_step_size
 :lp
 :logprior
 :loglikelihood


"""



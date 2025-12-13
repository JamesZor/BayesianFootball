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


"""

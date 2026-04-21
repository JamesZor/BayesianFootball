# current_development/calibration_module/experiment_utils.jl 

#=
Function that should be added into the Experiment Module of the package.
=#

"""
Function to process an experiment results struct and datastore to get the 
model inference prediction of the matches ( Out of sample predictions). 
returns the predictive posterior distribution (PPD) for the 
specifed markets (has default)
"""
function model_inference(ds::Data.DataStore, exp::Experiments.ExperimentResults)::Predictions.PPD 
  return Predictions.model_inference(
                Experiments.extract_oos_predictions(ds, exp)
         )
end



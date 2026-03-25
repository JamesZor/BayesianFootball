# src/experiments/helpers.jl 


function get_model_name(exp::ExperimentResults)::String 
  return string(exp.config.name)
end

function get_model_type(exp::ExperimentResults)::String 
  return string(type(exp.config.model))
end

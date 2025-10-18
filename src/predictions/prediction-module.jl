# src/predictions/prediction-module.jl


module Predictions 


using ..Models


@enum MarketTypes begin 
  home 
  away
  draw
  u_05
  o_05
  u_15
  o_15
  u_25
  o_25
  btts 
end 




abstract type AbstractPrediction end 

# abstract type AbstractPredictionPreGame <: AbstractPrediction end 
# abstract type AbstractPredictionInGame <: AbstractPrediction end 

struct Probability <: AbstractPrediction 
  markets::Set{MarketTypes}
end 


struct Odds <: AbstractPrediction 
  markets::Set{MarketTypes}
end 


struct EV <: AbstractPrediction 
  markets::Set{MarketTypes}
end 

struct Kelly <: AbstractPrediction 
  markets::Set{MarketTypes}
end 



end

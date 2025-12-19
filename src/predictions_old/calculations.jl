module Calculations

export AbstractCalculation, CalcProbability, CalcExpectedValue

abstract type AbstractCalculation end

struct CalcProbability <: AbstractCalculation end

# Request to calculate the Expected Value (EV)
# This requires the odds, which we'll get from the DataStore.
struct CalcExpectedValue <: AbstractCalculation end

# You could add:
struct CalcKellyFraction <: AbstractCalculation end

end

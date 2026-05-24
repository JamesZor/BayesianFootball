module MyDistributions 

using Distributions
using Random
using LinearAlgebra

# Import specific methods we need to extend
import Distributions: length, eltype, _rand!, logpdf, mean, var, cov, insupport

# dixon - coles
include("./dixoncoles-dist.jl")
export DixonColes 

include("./dixon_coles.jl")
export DixonColesLogGroup
include("./bivariate_poisson-dist.jl")
export BivariateLogPoisson

include("./double_negative_binomial.jl")
export DoubleNegativeBinomial

include("./negative_binomial.jl")
export RobustNegativeBinomial

include("./frank_copula_negbin.jl")
export FrankCopulaNegBin, frank_copula

include("./dixon_coles_negbin.jl")
export DixonColesNegBinLogGroup

include("./weibull_count.jl")
export WeibullCount


end

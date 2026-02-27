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

include("./dixon_coles_negbin.jl")
export DixonColesNegBinLogGroup


end

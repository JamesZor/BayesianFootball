# src/models/pregame/PreGame.jl

module PreGame

# We import the Types to extend them, but we don't need Reexport
using ...TypesInterfaces

using Turing, Distributions, DataFrames
using ..MyDistributions 
using LinearAlgebra
using Statistics


# 1. Load Common Helpers (The Orchestrator)
include("common.jl")
include("./grw_helpers.jl")

# 2. Load Models (The Workers)
# Since these files are included, their 'extract_parameters' overloads 
# merge into the function defined in common.jl
include("./implementations/static_poisson.jl")
include("./implementations/static_hierarchical_poisson.jl")
include("./implementations/static_hierarchical_poisson_NCP.jl")
include("./implementations/bivariate_poisson_ncp.jl")

include("./implementations/static_mvpln.jl")
export StaticMVPLN

include("./implementations/dixon_coles.jl")
export DixonColesNCP

include("./implementations/static_mixture_copula.jl")
export StaticMixtureCopula

include("./implementations/static_dixoncoles.jl")

include("./implementations/grw_poisson.jl")
export GRWPoisson

include("./implementations/grw_dixon_coles.jl")
export GRWDixonColes

include("./implementations/static_double_neg_bin.jl")
export StaticDoubleNegBin
include("./implementations/grw_double_neg_bin.jl")
export GRWNegativeBinomial
include("./implementations/grw_double_neg_bin_phi.jl")
export GRWNegativeBinomialPhi


include("./implementations/grw_double_neg_bin_mu.jl")
export GRWNegativeBinomialMu

include("./implementations/grw_double_neg_bin_mu_phi.jl")
export GRWNegativeBinomialMuPhi

include("./implementations/grw_double_neg_bin_delta.jl")
export GRWNegativeBinomialDelta


include("./implementations/grw_double_neg_bin_full.jl")
export GRWNegativeBinomialFull



include("./implementations/grw_bivariate_poisson.jl")
export GRWBivariatePoisson

include("./implementations/multi_grw_neg_bin.jl")
export MSNegativeBinomial 

include("./implementations/multi_grw_neg_bin_delta.jl")
export MSNegativeBinomialDelta


include("./implementations/sequential_funnel.jl")
export SequentialFunnelModel

# 3. Export
export StaticPoisson, StaticDixonColes, StaticHierarchicalPoisson, StaticHierarchicalPoissonNCP, BivariatePoissonNCP
export build_turing_model, extract_parameters

end # module

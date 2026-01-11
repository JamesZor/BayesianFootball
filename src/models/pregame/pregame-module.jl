# src/models/pregame/PreGame.jl

module PreGame

# We import the Types to extend them, but we don't need Reexport
using ...TypesInterfaces

# 1. Load Common Helpers (The Orchestrator)
include("common.jl")

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

# 3. Export
export StaticPoisson, StaticDixonColes, StaticHierarchicalPoisson, StaticHierarchicalPoissonNCP, BivariatePoissonNCP
export build_turing_model, extract_parameters

end # module

module BayesianFootball

# The order of includes is critical to resolve dependencies.
# 1. Interfaces contains all shared types and contracts. It has no local dependencies.
include("./types-interfaces.jl")
using .TypesInterfaces

# 2. Data is self-contained.
include("data/data-module.jl")

# 3. Models depends only on TypesInterfaces for its contracts.
include("models/models-module.jl")

# 4. Features depends on Data, TypesInterfaces, and the concrete model types from Models.
# This is now safe because Models is loaded first.
include("features/features-module.jl")

# 5. The rest of the modules depend on the core modules above.
include("sampling/sampling-module.jl")
include("./predictions/markets.jl")
include("./predictions/calculations.jl")
include("./predictions/prediction-module.jl")
include("./experiments/experiment-module.jl")

# Export the main modules and key functions/types for users
export Data, Features, Models, Sampling, Experiments, Predictions, Markets, Calculations
export AbstractFootballModel, Vocabulary, FeatureSet, required_mapping_keys

end

# src/experiments/registry.jl

"""
A dictionary that maps descriptive symbols to the actual ModelDefinition structs.
This makes it easy to define experiment batches without instantiating structs manually.
"""
const MODEL_REGISTRY = Dict(
    :maher => Dict(
        :basic     => MaherBasic(),
        :league_ha => MaherLeagueHA()
    )
    # You can add new model families here, e.g., :dixon_coles => Dict(...)
)

"""
    create_experiment_config(name, family, variant, cv_config, sample_config, mapping_funcs)

Constructs a full ExperimentConfig by looking up the model definition in the registry.
"""
function create_experiment_config(
    name::String,
    family::Symbol,
    variant::Symbol,
    cv_config::TimeSeriesSplitsConfig,
    sample_config::ModelSampleConfig,
    mapping_funcs::MappingFunctions
)
    # Check if the requested model exists in our registry
    if !haskey(MODEL_REGISTRY, family) || !haskey(MODEL_REGISTRY[family], variant)
        error("Model variant '$variant' for family '$family' not found in MODEL_REGISTRY.")
    end

    # Look up the model definition struct (e.g., MaherBasic())
    model_def = MODEL_REGISTRY[family][variant]

    # Return the fully constructed config object
    return ExperimentConfig(
        name,
        model_def,
        cv_config,
        sample_config,
        mapping_funcs
    )
end

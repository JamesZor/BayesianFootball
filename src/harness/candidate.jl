"A declarative model × data-scope × sampler recipe consumed by the experiment harness."
Base.@kwdef struct Candidate{M,S}
    name::String
    model::M
    scope::Data.DataScope
    sampler::S = Samplers.QueuedNUTSConfig(
        n_samples = 1_000, n_warmup = 500, n_chains = 4, accept_rate = 0.65)
    role::Symbol = :candidate
    hypothesis::String = ""

    function Candidate(name, model::M, scope, sampler::S, role, hypothesis) where {M,S}
        parsed_role = Symbol(role)
        parsed_role in (:control, :candidate) || throw(ArgumentError(
            "Candidate role must be :control or :candidate; got :$parsed_role"))
        return new{M,S}(String(name), model, scope, sampler, parsed_role, String(hypothesis))
    end
end

"SHA-256 identity of the scientific model and data scope; sampler budgets are excluded."
function recipe_hash(candidate::Candidate)
    canonical = string(candidate.model) * "\u001e" * string(candidate.scope)
    return bytes2hex(SHA.sha256(canonical))
end

function _has_learned_scale(config)
    for field in fieldnames(typeof(config))
        occursin('σ', String(field)) || continue
        getfield(config, field) isa Distributions.Distribution && return true
    end
    return false
end

"MAP validity label: learned hierarchical/random-walk scales make the screen diagnostic-only."
function _screen_validity(candidate::Candidate)
    dynamics = hasproperty(candidate.model, :dynamics) ? candidate.model.dynamics : nothing
    limited = dynamics isa Models.PreGame.MultiScaleGRW ||
              (dynamics !== nothing && _has_learned_scale(dynamics))
    return limited ? "limited" : "ranking_only"
end

function _screen_validity_record(candidate::Candidate, experiment::AbstractString)
    validity = _screen_validity(candidate)
    run_id = uuid5(SCREEN_NAMESPACE_UUID, "$(experiment):$(recipe_hash(candidate)):screen")
    detail = validity == "limited" ?
        "MAP is not comparable to posterior integration for a model with a learned hierarchical/random-walk scale." :
        "MAP is a cheap within-class ranking diagnostic, not a substitute for the NUTS grid."
    return (;
        run_id,
        recipe_hash = recipe_hash(candidate),
        experiment = String(experiment),
        candidate = candidate.name,
        stage = "screen",
        check = "screen_validity",
        severity = "diagnostic",
        status = validity,
        value = (; screen_validity = validity),
        detail,
        git_sha = Training.git_commit_id(),
        at = now(),
    )
end

function _record_screen_validity!(db, candidates, experiment::AbstractString)
    records = [_screen_validity_record(candidate, experiment) for candidate in candidates]
    write_checks!(db, records)
    return records
end

function _smoke_sampler(candidate::Candidate)
    sampler = candidate.sampler
    return Samplers.QueuedNUTSConfig(
        n_samples = 200,
        n_warmup = 200,
        n_chains = 2,
        accept_rate = hasproperty(sampler, :accept_rate) ? sampler.accept_rate : 0.65,
        max_depth = hasproperty(sampler, :max_depth) ? sampler.max_depth : 10,
        initialisation = hasproperty(sampler, :initialisation) ? sampler.initialisation : nothing,
        show_progress = false,
        silence_initial_stepsize = hasproperty(sampler, :silence_initial_stepsize) ?
                                   sampler.silence_initial_stepsize : true,
    )
end

"Build the shared inference recipe for `:screen`, `:smoke`, or `:grid`."
function fit_config(candidate::Candidate; stage::Symbol, experiment::AbstractString)
    stage in (:screen, :smoke, :grid) || throw(ArgumentError(
        "fit_config stage must be :screen, :smoke, or :grid; got :$stage"))
    sampler = stage === :screen ?
        Samplers.MAPConfig(show_progress = false) :
        stage === :smoke ? _smoke_sampler(candidate) : candidate.sampler
    return Training.FitConfig(
        name = candidate.name,
        model = candidate.model,
        splitter = Data.ScopedWalkForwardCV(candidate.scope),
        sampler = sampler,
        execution = Training.QueuedExecution(max_concurrent_tasks = 16),
        tags = ["harness", "stage:$stage", "recipe:" * recipe_hash(candidate)],
        description = candidate.hypothesis,
        save_dir = joinpath("data", "fits", String(experiment), String(stage), candidate.name),
    )
end

# workspace/bivariate_maher_model/ChainAnalysisUtils.jl

module ChainAnalysisUtils

using BayesianFootball
using DataFrames
using MCMCChains
using StatsPlots
using Distributions
using StatsBase

export ChainAnalysisModel, load_chain_length_experiment, get_parameter_samples
export plot_posterior_comparison, tabulate_summary_stats, plot_diagnostic_trends
export tabulate_global_parameter_summary, analyze_kl_divergence

# --- 1. Core Data Structure ---

"""
    ChainAnalysisModel

A struct to hold all relevant information for a single MCMC model run
from the chain length experiment.
"""
struct ChainAnalysisModel
    name::String
    chain_length::Int
    raw_chains::Chains
    summary_stats::DataFrame
    posterior_samples::NamedTuple
    mapping::MappedData
end


# --- 2. Data Loading and Preparation ---

"""
    load_chain_length_experiment(experiment_group_path::String)

Scans an experiment directory, loads each model run, parses metadata,
and returns a dictionary of `ChainAnalysisModel` structs keyed by chain length.
"""
function load_chain_length_experiment(experiment_group_path::String)
    
    analysis_models = Dict{Int, ChainAnalysisModel}()
    
    if !isdir(experiment_group_path)
        error("Experiment directory not found: $experiment_group_path")
        return analysis_models
    end
    
    println("Scanning for models in: $experiment_group_path")

    for dir_name in readdir(experiment_group_path)
        full_path = joinpath(experiment_group_path, dir_name)
        if isdir(full_path)
            try
                # Parse the chain length from the directory name
                m = match(r"_steps_(\d+)", dir_name)
                if isnothing(m)
                    @warn "Could not parse chain length from '$dir_name'. Skipping."
                    continue
                end
                chain_length = parse(Int, m.captures[1])
                
                println(" -> Loading model with $chain_length steps from '$dir_name'")
                
                loaded_model = load_model(full_path)
                
                chains = loaded_model.result.chains_sequence[1].ft
                
                model = ChainAnalysisModel(
                    dir_name,
                    chain_length,
                    chains,
                    summarystats(chains),
                    BayesianFootball.extract_posterior_samples(loaded_model.config.model_def, chains),
                    loaded_model.result.mapping
                )
                
                analysis_models[chain_length] = model
                
            catch e
                @error "Failed to load or process model from '$full_path'. Error: $e"
            end
        end
    end
    
    println("✅ Found and loaded $(length(analysis_models)) models.")
    return analysis_models
end


"""
    get_parameter_samples(model::ChainAnalysisModel, param_symbol::Symbol; team_name::Union{String, Nothing}=nothing)

Extracts the posterior samples for a specific parameter.
"""
function get_parameter_samples(model::ChainAnalysisModel, param_symbol::Symbol; team_name::Union{String, Nothing}=nothing)
    samples_tuple = model.posterior_samples
    
    if param_symbol in (:α, :β)
        isnothing(team_name) && error("`team_name` must be provided for parameters :α or :β.")
        
        team_id = get(model.mapping.team_map, team_name, 0)
        team_id == 0 && error("Team '$team_name' not found in model mapping.")
        
        param_matrix = getfield(samples_tuple, param_symbol)
        return param_matrix[:, team_id]

    elseif param_symbol in (:δ, :γ)
        return getfield(samples_tuple, param_symbol)
    else
        error("Parameter symbol :$param_symbol is not supported for sample extraction.")
    end
end


# --- 3. Analysis and Plotting Functions ---

# ... (plot_posterior_comparison and tabulate_summary_stats remain the same) ...
function plot_posterior_comparison(analysis_models::Dict{Int, ChainAnalysisModel}, param_symbol::Symbol; team_name::Union{String, Nothing}=nothing)
    title_str = isnothing(team_name) ? "Global Parameter: :$param_symbol" : "Parameter: :$param_symbol for $team_name"
    p = plot(title=title_str, xlabel="Value", ylabel="Density", legend=:outertopright)
    sorted_keys = sort(collect(keys(analysis_models)))
    for len in sorted_keys
        model = analysis_models[len]
        samples = get_parameter_samples(model, param_symbol; team_name=team_name)
        density!(p, samples, label="$len steps")
    end
    return p
end

function tabulate_summary_stats(analysis_models::Dict{Int, ChainAnalysisModel}, param_symbol::Symbol; team_name::Union{String, Nothing}=nothing)
    results = DataFrame(chain_length=Int[], mean=Float64[], std=Float64[], rhat=Float64[], ess_bulk=Float64[])
    param_name_str = if param_symbol in (:α, :β)
        team_id = analysis_models[first(keys(analysis_models))].mapping.team_map[team_name]
        param_letter = param_symbol == :α ? "α" : "β"
        "log_$(param_letter)_raw[$team_id]"
    else
        # For global parameters like δ and γ, need to match Turing's naming
        param_symbol == :δ ? "log_δ" : string(param_symbol)
    end
    sorted_keys = sort(collect(keys(analysis_models)))
    for len in sorted_keys
        model = analysis_models[len]
        summary_row = filter(row -> row.parameters == Symbol(param_name_str), model.summary_stats)
        if nrow(summary_row) == 1
            push!(results, (len, summary_row[1, :mean], summary_row[1, :std], summary_row[1, :rhat], summary_row[1, :ess_bulk]))
        else
            @warn "Could not find parameter '$param_name_str' in summary for chain length $len."
        end
    end
    return results
end

"""
    tabulate_global_parameter_summary(analysis_models::Dict{Int, ChainAnalysisModel})

Creates a single summary table for all key global parameters, similar to Table 2
[cite_start]in the Koopman & Lit paper[cite: 652].
"""
function tabulate_global_parameter_summary(analysis_models::Dict{Int, ChainAnalysisModel})
    # These are the global parameters defined in your `setup.jl` model
    global_params = [:log_δ, :γ, :σ_attack, :σ_defense]
    
    df = DataFrame(chain_length = Int[])
    
    sorted_keys = sort(collect(keys(analysis_models)))
    
    for param in global_params
        param_df = DataFrame(chain_length = sorted_keys)
        means = [analysis_models[len].summary_stats[findfirst(==(param), analysis_models[len].summary_stats.parameters), :mean] for len in sorted_keys]
        stds = [analysis_models[len].summary_stats[findfirst(==(param), analysis_models[len].summary_stats.parameters), :std] for len in sorted_keys]
        
        param_df[!, Symbol(param, "_mean")] = means
        param_df[!, Symbol(param, "_std")] = stds
        
        df = outerjoin(df, param_df, on=:chain_length)
    end
    
    return df
end


function plot_diagnostic_trends(summary_df::DataFrame, diagnostic::Symbol)
    title_str = "Trend for $(uppercase(string(diagnostic))) vs. Chain Length"
    p = plot(summary_df.chain_length, summary_df[!, diagnostic], title=title_str, xlabel="Chain Length (Number of Steps)", ylabel=string(diagnostic), legend=false, marker=:circle, xscale=:log10)
    if diagnostic == :rhat
        hline!(p, [1.01], linestyle=:dash, color=:red, label="Convergence Threshold (1.01)")
    end
    return p
end


# --- 4. Advanced Analysis ---

"""
Helper function to compute KL divergence between two sets of samples.
"""
function _calculate_kl_divergence(p_samples::AbstractVector, q_samples::AbstractVector; n_bins=100)
    # 1. Determine common range and create bins
    min_val = min(minimum(p_samples), minimum(q_samples))
    max_val = max(maximum(p_samples), maximum(q_samples))
    edges = range(min_val, stop=max_val, length=n_bins + 1)
    
    # 2. Create normalized histograms (probability distributions)
    p_hist = normalize(fit(Histogram, p_samples, edges), mode=:probability)
    q_hist = normalize(fit(Histogram, q_samples, edges), mode=:probability)
    
    # 3. Add smoothing to avoid log(0)
    epsilon = 1e-10
    p_weights = p_hist.weights .+ epsilon
    q_weights = q_hist.weights .+ epsilon
    p_weights ./= sum(p_weights)
    q_weights ./= sum(q_weights)
    
    # 4. Calculate KL Divergence D_KL(P || Q)
    return sum(p_weights[i] * log(p_weights[i] / q_weights[i]) for i in 1:n_bins if p_weights[i] > 0)
end

"""
    analyze_kl_divergence(
        analysis_models::Dict{Int, ChainAnalysisModel},
        reference_model::ChainAnalysisModel,
        param_symbol::Symbol;
        team_name::Union{String, Nothing}=nothing
    )

Calculates the KL divergence of shorter chains relative to a long reference chain.
"""
function analyze_kl_divergence(analysis_models::Dict{Int, ChainAnalysisModel}, reference_model::ChainAnalysisModel, param_symbol::Symbol; team_name::Union{String, Nothing}=nothing)
    
    ref_samples = get_parameter_samples(reference_model, param_symbol; team_name=team_name)
    
    results = DataFrame(
        chain_length = Int[],
        kl_divergence = Float64[]
    )
    
    sorted_keys = sort(collect(keys(analysis_models)))

    for len in sorted_keys
        # Don't compare the reference model to itself
        if len >= reference_model.chain_length
            continue
        end
        
        model = analysis_models[len]
        target_samples = get_parameter_samples(model, param_symbol; team_name=team_name)
        
        divergence = _calculate_kl_divergence(ref_samples, target_samples)
        
        push!(results, (len, divergence))
    end
    
    return results
end

end # end module

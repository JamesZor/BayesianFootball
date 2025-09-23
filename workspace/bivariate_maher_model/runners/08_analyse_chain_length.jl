# 08_analyze_chain_length.jl
using BayesianFootball
using DataFrames
using StatsPlots

# --- 1. Setup ---
include("/home/james/bet_project/models_julia/workspace/bivariate_maher_model/ChainAnalysisUtils.jl")
using .ChainAnalysisUtils

# --- 2. Configuration ---
const EXPERIMENT_PATH = "/home/james/bet_project/models_julia/experiments/bivar_chain_length"
const TEAM_TO_ANALYZE = "Man City" 
const PARAMETER_TO_ANALYZE = :α 

# **ACTION REQUIRED**: Update this path to your long-run reference model
const REFERENCE_MODEL_PATH = "/home/james/bet_project/models_julia/experiments/bivar_chain_length/maher_bivariate_steps_20000_20250923-154500" # Example path

# --- 3. Main Execution ---

println("--- Starting MCMC Chain Length Analysis ---")

# --- A. Load all models from the experiment ---
all_models = load_chain_length_experiment(EXPERIMENT_PATH);

if isempty(all_models)
    error("No models were loaded. Check the EXPERIMENT_PATH.")
end

# --- B. Visually Compare Posterior Distributions ---
println("\n📊 Generating posterior distribution plot for $PARAMETER_TO_ANALYZE...")
posterior_plot = plot_posterior_comparison(
    all_models,
    PARAMETER_TO_ANALYZE;
    team_name = (PARAMETER_TO_ANALYZE in (:α, :β) ? TEAM_TO_ANALYZE : nothing)
)
display(posterior_plot)
savefig(posterior_plot, "posterior_comparison.png")
println("   -> Plot saved.")

# --- C. Global Parameter Summary ---
println("\n📋 Generating global parameter summary table...")
global_summary_table = tabulate_global_parameter_summary(all_models)
println("\nGlobal Parameter Summary across Chain Lengths:")
show(global_summary_table, allrows=true, allcols=true)


# --- D. Individual Parameter Diagnostics ---
println("\n📈 Generating diagnostic plots for $PARAMETER_TO_ANALYZE ($TEAM_TO_ANALYZE)...")
summary_table = tabulate_summary_stats(
    all_models, PARAMETER_TO_ANALYZE;
    team_name = (PARAMETER_TO_ANALYZE in (:α, :β) ? TEAM_TO_ANALYZE : nothing)
)
rhat_plot = plot_diagnostic_trends(summary_table, :rhat)
ess_plot = plot_diagnostic_trends(summary_table, :ess_bulk)
display(rhat_plot)
display(ess_plot)
savefig(rhat_plot, "rhat_trend.png")
savefig(ess_plot, "ess_trend.png")
println("   -> Diagnostic plots saved.")


# --- E. KL Divergence Analysis ---
println("\n🧠 Performing KL Divergence analysis against reference model...")
if isdir(REFERENCE_MODEL_PATH)
    # Load the reference model and package it into our struct
    ref_loaded_model = load_model(REFERENCE_MODEL_PATH)
    ref_chains = ref_loaded_model.result.chains_sequence[1].ft
    ref_model = ChainAnalysisModel(
        basename(REFERENCE_MODEL_PATH),
        20000, # Manually set chain length
        ref_chains,
        summarystats(ref_chains),
        BayesianFootball.extract_posterior_samples(ref_loaded_model.config.model_def, ref_chains),
        ref_loaded_model.result.mapping
    )

    # Calculate KL divergence
    kl_results = analyze_kl_divergence(all_models, ref_model, PARAMETER_TO_ANALYZE;
        team_name=(PARAMETER_TO_ANALYZE in (:α, :β) ? TEAM_TO_ANALYZE : nothing)
    )

    println("\nKL Divergence from Reference Model:")
    show(kl_results, allrows=true, allcols=true)

    # Plot KL divergence trend
    kl_plot = plot(kl_results.chain_length, kl_results.kl_divergence,
        title="KL Divergence for :$(PARAMETER_TO_ANALYZE) ($(TEAM_TO_ANALYZE))",
        xlabel="Chain Length", ylabel="KL Divergence to Reference",
        legend=false, marker=:circle, xscale=:log10, yscale=:log10
    )
    display(kl_plot)
    savefig(kl_plot, "kl_divergence_trend.png")
    println("\n   -> KL Divergence plot saved.")

else
    @warn """
    Reference model path not found: $REFERENCE_MODEL_PATH
    Skipping KL Divergence analysis. Please train a long-chain model and update the path.
    """
end

println("\n--- Analysis complete! ---")

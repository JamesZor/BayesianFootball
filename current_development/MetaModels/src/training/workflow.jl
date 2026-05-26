# current_development/MetaModels/src/training/workflow.jl

using DataFrames
using Dates
using Turing
using BayesianFootball
import BayesianFootball.Data
import BayesianFootball.Experiments
import BayesianFootball.Samplers

export MetaExperimentTask, MetaExperimentResults, run_meta_experiment

struct MetaExperimentTask
    base_results::Experiments.ExperimentResults
    meta_model::AbstractMetaModel
    sampler_config::Samplers.AbstractNUTSConfig
    splitter::Data.AbstractSplitter 
    target_selection::Symbol # The specific market to model (e.g. :over_15)
end

struct MetaExperimentResults
    task::MetaExperimentTask
    training_results::Any
end

"""
    run_meta_experiment(task::MetaExperimentTask; ds::Data.DataStore)

Executes the Meta Model cross-validation workflow.
"""
function run_meta_experiment(task::MetaExperimentTask; ds::Data.DataStore)
    println("1. Extracting Layer 1 OOS Predictions (Latent States)...")
    latent_states = Experiments.extract_oos_predictions(ds, task.base_results)
    
    println("2. Generating Posterior Predictive Distributions (PPD)...")
    # This generates the actual probabilities for all markets
    ppd = BayesianFootball.Predictions.model_inference(latent_states)
    
    println("3. Filtering for Target Selection: $(task.target_selection)")
    # PPD is a wrapper struct — must access the inner .df field for DataFrames operations
    ppd_filtered = subset(ppd.df, :selection => ByRow(isequal(task.target_selection)))
    odds_filtered = subset(ds.odds, :selection => ByRow(isequal(task.target_selection)))
    
    println("4. Joining with Market Odds and Match Data...")
    joined = innerjoin(ppd_filtered, odds_filtered[!, [:match_id, :prob_fair_close, :odds_close, :is_winner]], on=:match_id)
    joined = innerjoin(joined, ds.matches[!, [:match_id, :home_team, :away_team, :match_date]], on=:match_id)
    
    dropmissing!(joined, [:is_winner, :prob_fair_close, :match_date, :distribution])
    sort!(joined, :match_date)
    
    println("5. Preparing Data structures...")
    start_date = Date(minimum(joined.match_date))
    joined.W = [(Date(d) - start_date).value ÷ 7 + 1 for d in joined.match_date]
    n_weeks = maximum(joined.W)
    
    unique_teams = unique(vcat(joined.home_team, joined.away_team))
    team_map = Dict(t => i for (i, t) in enumerate(unique_teams))
    joined.home_idx = [team_map[t] for t in joined.home_team]
    joined.away_idx = [team_map[t] for t in joined.away_team]
    n_teams = length(unique_teams)
    
    # Calculate the mean probability from the Layer 1 PPD distribution
    joined.p_L1 = [mean(dist) for dist in joined.distribution]
    
    Y = Int.(joined.is_winner)
    p_L1 = Float64.(joined.p_L1)
    m_i = Float64.(joined.prob_fair_close)
    
    meta_data = MetaModelData(
        Y, p_L1, m_i,
        joined.W, joined.home_idx, joined.away_idx,
        n_weeks, n_teams
    )
    
    println("6. Generating Meta CV Splits...")
    println("   (MVP: Running on single unified split to test Turing Engine compilation)")
    
    model = build_turing_meta_model(task.meta_model, meta_data)
    
    println("7. Running Meta Sampler...")
    # For MVP, we run a short 500 sample NUTS chain to ensure it compiles and executes
    chain = sample(model, NUTS(0.65), 500)
    
    return MetaExperimentResults(task, [chain]), joined
end

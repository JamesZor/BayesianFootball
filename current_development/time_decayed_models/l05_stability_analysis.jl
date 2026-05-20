# current_development/time_decayed_models/l05_stability_analysis.jl

using DataFrames
using Statistics
using MCMCChains
using BayesianFootball
using BayesianFootball.Data
using BayesianFootball.Features
using BayesianFootball.Models.PreGame
using BayesianFootball.Experiments

"""
    extract_stability_dataframe(ds::Data.DataStore, exp_results::Experiments.ExperimentResults)

The main entry point for analyzing parameter stability over a walk-forward CV.
Returns a long-format DataFrame with parameter summaries per fold.
"""
function extract_stability_dataframe(ds::Data.DataStore, exp_results::Experiments.ExperimentResults)
    config = exp_results.config
    
    # 1. Reconstruct boundaries and metadata (matches the training logic)
    # This ensures we have the correct temporal context for each split
    boundaries_with_meta = Data.create_id_boundaries(ds, config.splitter)
    
    # 2. Create feature sets (so we get the team_map for each fold)
    # Walk-forward CV often has expanding team lists; we must use the correct map for each split.
    feature_sets = Features.create_features(
        boundaries_with_meta, 
        ds, 
        config.model, 
        config.splitter.dynamics_col
    )
    
    # 3. Access results (Chains and SplitMetaData)
    results_array = exp_results.training_results.items
    n_splits = length(results_array)

    all_fold_dfs = Vector{DataFrame}()

    println("[INFO] Processing $n_splits splits for stability analysis...")

    for i in 1:n_splits
        chain, meta = results_array[i]
        feature_tuple = feature_sets[i]
        
        # Process this fold
        fold_df = _process_parameter_fold(config.model, feature_tuple, chain, meta)
        push!(all_fold_dfs, fold_df)
    end

    return vcat(all_fold_dfs...)
end

"""
    _process_parameter_fold(model, feature_tuple, chain, meta)

Internal helper to extract and summarize parameters for a single split.
"""
function _process_parameter_fold(model, feature_tuple, chain, meta)
    feature_set = feature_tuple[1]
    data = feature_set.data
    
    n_teams   = Int(data[:n_teams])
    n_seasons = Int(data[:n_seasons])
    n_months  = 12 # Hardcoded as per engine
    team_map  = data[:team_map]
    rev_team_map = Dict(v => k for (k, v) in team_map)
    
    # Base fold info for the long dataframe
    fold_info = Dict(
        :fold => meta.time_step,
        :target_season => meta.target_season,
        :week => meta.time_step,
        :train_season => meta.train_season
    )
    
    rows = []

    # --- 1. Global Scalar Parameters ---
    # These are usually sampled at the top level of the engine
    scalar_candidates = [:ν_xg, :σ_market, :kap_κ_global] # kap_κ_global might be prefixed
    for p in scalar_candidates
        if p in keys(chain)
            samples = vec(Array(chain[p]))
            push!(rows, merge(fold_info, Dict(
                :parameter => string(p),
                :entity => "global",
                :mean => mean(samples),
                :std => std(samples)
            )))
        end
    end

    # --- 2. Home Advantage (Hierarchical) ---
    if hasproperty(model, :homeadvantage_config)
        ha_mat = PreGame.extract_home_advantage(chain, model.homeadvantage_config, n_teams)
        for t_idx in 1:n_teams
            team_name = get(rev_team_map, t_idx, "unknown")
            samples = ha_mat[:, t_idx]
            push!(rows, merge(fold_info, Dict(
                :parameter => "home_advantage",
                :entity => team_name,
                :mean => mean(samples),
                :std => std(samples)
            )))
        end
        
        # Also extract ha.γ_base and ha.σ_γ if they exist
        for p in [Symbol("ha.γ_base"), Symbol("ha.σ_γ")]
            if p in keys(chain)
                samples = vec(Array(chain[p]))
                push!(rows, merge(fold_info, Dict(
                    :parameter => string(p),
                    :entity => "global",
                    :mean => mean(samples),
                    :std => std(samples)
                )))
            end
        end
    end

    # --- 3. Kappa (Conversion Rate) ---
    if hasproperty(model, :kappa_config)
        kap_mat = PreGame.extract_kappa(chain, model.kappa_config, n_teams)
        for t_idx in 1:n_teams
            team_name = get(rev_team_map, t_idx, "unknown")
            samples = kap_mat[:, t_idx]
            push!(rows, merge(fold_info, Dict(
                :parameter => "kappa",
                :entity => team_name,
                :mean => mean(samples),
                :std => std(samples)
            )))
        end

        # Base hyperparams
        for p in [Symbol("kap.κ_base"), Symbol("kap.σ_κ")]
            if p in keys(chain)
                samples = vec(Array(chain[p]))
                push!(rows, merge(fold_info, Dict(
                    :parameter => string(p),
                    :entity => "global",
                    :mean => mean(samples),
                    :std => std(samples)
                )))
            end
        end
    end

    # --- 4. Interception (Seasonal) ---
    if hasproperty(model, :interception_config)
        inter_mat = PreGame.extract_interception(chain, model.interception_config, n_seasons)
        for s_idx in 1:n_seasons
            samples = inter_mat[:, s_idx]
            push!(rows, merge(fold_info, Dict(
                :parameter => "interception",
                :entity => "season_$s_idx",
                :mean => mean(samples),
                :std => std(samples)
            )))
        end
    end

    # --- 5. Positional Dynamics (Weights) ---
    if hasproperty(model, :player_dynamics_config)
        # Engines typically use "p_dyn" as the prefix for positional dynamics
        p_dyn_nt = PreGame.extract_dynamics(chain, model.player_dynamics_config, "p_dyn", n_teams)
        for p_name in keys(p_dyn_nt)
            samples = p_dyn_nt[p_name]
            push!(rows, merge(fold_info, Dict(
                :parameter => string(p_name),
                :entity => "global",
                :mean => mean(samples),
                :std => std(samples)
            )))
        end
    end

    # --- 6. Dispersion (r) ---
    if hasproperty(model, :dispersion_config)
        disp_nt = PreGame.extract_dispersion(chain, model.dispersion_config, n_teams, n_months)
        
        if hasproperty(disp_nt, :h) # Simple configs
             push!(rows, merge(fold_info, Dict(
                :parameter => "r_home",
                :entity => "global",
                :mean => mean(disp_nt.h),
                :std => std(disp_nt.h)
            )))
            push!(rows, merge(fold_info, Dict(
                :parameter => "r_away",
                :entity => "global",
                :mean => mean(disp_nt.a),
                :std => std(disp_nt.a)
            )))
        end

        if hasproperty(disp_nt, :team_vol) # AdvancedVolatilityDispersion
            for t_idx in 1:n_teams
                team_name = get(rev_team_map, t_idx, "unknown")
                samples = disp_nt.team_vol[:, t_idx]
                push!(rows, merge(fold_info, Dict(
                    :parameter => "dispersion_vol",
                    :entity => team_name,
                    :mean => mean(samples),
                    :std => std(samples)
                )))
            end
            
            # Global components of dispersion
            for p in [:base, :home_offset]
                samples = disp_nt[p]
                push!(rows, merge(fold_info, Dict(
                    :parameter => "dispersion_" * string(p),
                    :entity => "global",
                    :mean => mean(samples),
                    :std => std(samples)
                )))
            end
        end
    end

    return DataFrame(rows)
end

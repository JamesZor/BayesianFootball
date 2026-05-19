# current_development/player_tracking_systems/src/hyper_optimizer.jl

using Optim
using BayesianFootball.Data

"""
    optimize_bayesian_tracker(ds, boundaries)

Uses Optim.jl (Particle Swarm) to find the optimal continuous parameters for the BayesianTracker.
Returns the Optim.OptimizationResults.
"""
function optimize_bayesian_tracker(ds::Data.DataStore, boundaries)
    
    println("[INFO] Starting Hyperparameter Optimization for BayesianTracker...")
    
    # Objective Function
    function objective(x::Vector{Float64})
        # Unpack parameters
        prior_mean = x[1]
        prior_var = x[2]
        obs_var = x[3]
        process_noise = x[4]
        
        # Instantiate Config
        config = BayesianTracker(prior_mean, prior_var, obs_var, process_noise)
        
        # Evaluate (Threaded across boundaries inside this call)
        metrics = evaluate_tracker_on_boundaries(config, ds, boundaries)
        
        # Return LogLoss (lower is better)
        # If NaN is returned (e.g. from data issues), penalize heavily
        if isnan(metrics.log_loss)
            return 1e6
        end
        return metrics.log_loss
    end
    
    # Define bounds for Particle Swarm to prevent math crashes
    # p1: prior_mean (6.0 to 7.5)
    # p2: prior_var (0.1 to 3.0)
    # p3: obs_var (0.1 to 3.0)
    # p4: process_noise (0.001 to 0.5)
    
    lower_bounds = [6.0, 0.1, 0.1, 0.001]
    upper_bounds = [7.5, 3.0, 3.0, 0.5]
    
    # Run optimization
    opt_result = Optim.optimize(
        objective, 
        lower_bounds, 
        upper_bounds, 
        ParticleSwarm(), 
        Optim.Options(show_trace=true, iterations=100)
    )
    
    return opt_result
end

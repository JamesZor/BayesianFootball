
@model function grw_two_step_component(
        number_teams::Int,         # Tip: Type as Int for matrix dimensions
        number_rounds::Int,
        number_history_steps::Int, # Tip: Type as Int
        number_target_steps::Int,  # Tip: Type as Int
        z₀::Distribution,
        zₛ::Distribution,
        zₖ::Distribution,
        dist_σ₀::Distribution,
        dist_σₛ::Distribution,
        dist_σₖ::Distribution,
      )
    
    # These names (z_init, z_steps) will be prefixed by the caller
    σ₀ ~ dist_σ₀
    σₛ ~ dist_σₛ
    σₖ ~ dist_σₖ

    # 1. Always sample the initial state
    z_init ~ filldist(z₀, number_teams)
    init   = z_init .* σ₀
    
    # 2. Conditionally sample season steps (Handle zero history)
    if number_history_steps > 0
        z_season_steps ~ filldist(zₛ, number_teams, number_history_steps)
        season_steps   = z_season_steps .* σₛ
    else
        # AD-safe empty matrix (0 columns) so hcat still works seamlessly
        season_steps = zeros(eltype(init), number_teams, 0)
    end
    
    # 3. Conditionally sample target steps (Handle single-step targets)
    if number_target_steps > 1
        z_target_steps ~ filldist(zₖ, number_teams, number_target_steps - 1)
        target_steps   = z_target_steps .* σₖ
    else
        # AD-safe empty matrix
        target_steps = zeros(eltype(init), number_teams, 0)
    end
    
    # Deterministic Transformation
    raw = cumsum(
            hcat(
              init,
              season_steps,
              target_steps),
          dims=2)
    
    # Center and Return
    centered = raw .- mean(raw, dims=1)
    return centered
end



"""
    reconstruct_multiscale_submodel(chain, prefix, n_teams, n_history, n_target)

Reconstructs the two-speed GRW states (macro + micro steps) by iterating variables safely,
accounting for edge-case splits where history or target steps might be missing.
"""
function reconstruct_multiscale_submodel(chain, prefix::String, n_teams::Int, n_history::Int, n_target::Int)
    n_samples_per_chain, _, n_chains = size(chain)
    n_total = n_samples_per_chain * n_chains
    
    # 1. Extract the three Variances (Flattened and Reshaped for Broadcasting)
    S_0 = reshape(vec(Array(chain[Symbol("$prefix.σ₀")])), n_total, 1, 1)
    S_s = reshape(vec(Array(chain[Symbol("$prefix.σₛ")])), n_total, 1, 1)
    S_k = reshape(vec(Array(chain[Symbol("$prefix.σₖ")])), n_total, 1, 1)
    
    # Helper to safely grab symbols whether Turing used spaces or not
    function get_sym(base_name, idx...)
        str1 = "$base_name[$(join(idx, ", "))]"
        str2 = "$base_name[$(join(idx, ","))]"
        return Symbol(str1) in names(chain) ? Symbol(str1) : Symbol(str2)
    end
    
    # 2. Extract Initial States [Samples, Teams, 1] (Always present)
    Z_init = zeros(Float64, n_total, n_teams, 1)
    for i in 1:n_teams
        Z_init[:, i, 1] = vec(Array(chain[get_sym("$prefix.z_init", i)]))
    end
    init_scaled = Z_init .* S_0
    
    # 3. Conditionally Extract Macro (Season) Steps
    if n_history > 0
        Z_season = zeros(Float64, n_total, n_teams, n_history)
        for t in 1:n_history
            for i in 1:n_teams
                Z_season[:, i, t] = vec(Array(chain[get_sym("$prefix.z_season_steps", i, t)]))
            end
        end
        season_scaled = Z_season .* S_s
    else
        # AD-safe/concat-safe empty matrix along the Time dimension
        season_scaled = zeros(Float64, n_total, n_teams, 0)
    end
    
    # 4. Conditionally Extract Micro (Target) Steps 
    if n_target > 1
        Z_target = zeros(Float64, n_total, n_teams, n_target - 1)
        for t in 1:(n_target - 1)
            for i in 1:n_teams
                Z_target[:, i, t] = vec(Array(chain[get_sym("$prefix.z_target_steps", i, t)]))
            end
        end
        target_scaled = Z_target .* S_k
    else
        # Concat-safe empty matrix
        target_scaled = zeros(Float64, n_total, n_teams, 0)
    end
    
    # 5. Stitch the timeline together along the Time axis (dim 3)
    raw_steps = cat(init_scaled, season_scaled, target_scaled, dims=3)
    
    # 6. Integrate the walk (Cumulative Sum over Time axis)
    full_raw = cumsum(raw_steps, dims=3)
    
    # 7. Center around 0 (Zero-Sum over Teams axis = 2)
    centered = full_raw .- mean(full_raw, dims=2)
    
    # Return in standard format: [Teams, Time, Samples]
    return permutedims(centered, (2, 3, 1))
end


# src/predictions/score_computation/negative_binomial.jl

using Distributions
using ..Models 
using ..MyDistributions 


function extract_params(model::Models.PreGame.StaticDoubleNegBin, row)
    return (
        λ_h = row.λ_h, # Vector of Home Means (Rates)
        λ_a = row.λ_a, # Vector of Away Means (Rates)
        r   = row.r    # Vector of Shapes
    )
end

function compute_score_matrix(
    model::Models.PreGame.StaticDoubleNegBin, 
    params; 
    max_goals::Int=12
)
    # 1. Unpack Params
    λ_h, λ_a = params.λ_h, params.λ_a
    r_vec = params.r
    n_samples = length(λ_h)

    # 2. Create the Evaluation Grid (Vector of Vectors)
    # We create a Matrix of [home, away] vectors: dimensions (max_goals x max_goals)
    # Note: Your logpdf expects a Vector, so we create vectors, not tuples.
    outcomes_grid = [[h, a] for h in 0:max_goals-1, a in 0:max_goals-1]

    # Output Tensor
    S = zeros(Float64, max_goals, max_goals, n_samples)

    # 3. Compute
    @inbounds for k in 1:n_samples # Instantiate the distribution for this sample
        dist = DoubleNegativeBinomial(λ_h[k], λ_a[k], r_vec[k], r_vec[k])

        # Vectorized PDF Evaluation
        # We broadcast the pdf function over the entire grid of outcomes at once.
        # Ref(dist) treats the distribution as a scalar constant for the broadcast.
        S_k = pdf.(Ref(dist), outcomes_grid)

        # Assign to the slice (permutedims might be needed depending on your dim ordering preference)
        # outcomes_grid varies rows (h) then cols (a), which matches S[h, a, k]
        S[:, :, k] = S_k
    end
    
    return ScoreMatrix(S)
end

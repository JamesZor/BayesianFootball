
# src/predictions/score_computation/negative_binomial.jl

using Distributions
using ..Models 
using ..MyDistributions 



function extract_params(model::Models.PreGame.AbstractNegBinModel, row)
    if hasproperty(row, :r)
        return (
            λ_h = row.λ_h, # Vector of Home Means (Rates)
            λ_a = row.λ_a, # Vector of Away Means (Rates)
            rₕ  = row.r,
            rₐ  = row.r 
        )
    elseif hasproperty(row, :rₕ)
        return (
            λ_h = row.λ_h, # Vector of Home Means (Rates)
            λ_a = row.λ_a, # Vector of Away Means (Rates)
            rₕ  = row.rₕ,
            rₐ  = row.rₐ 
        )
    else
        throw(ArgumentError("Row does not contain expected shape parameters (:r or :rₕ)"))
    end 
end

function compute_score_matrix(
    model::Models.PreGame.AbstractNegBinModel, 
    params; 
    max_goals::Int=12
)
    # 1. Unpack Params
    λ_h, λ_a = params.λ_h, params.λ_a
    rₕ, rₐ = params.rₕ, params.rₐ
    n_samples = length(λ_h)

    # 2. Create the Evaluation Grid (Vector of Vectors)
    # We create a Matrix of [home, away] vectors: dimensions (max_goals x max_goals)
    # Note: Your logpdf expects a Vector, so we create vectors, not tuples.
    outcomes_grid = [[h, a] for h in 0:max_goals-1, a in 0:max_goals-1]

    # Output Tensor
    S = zeros(Float64, max_goals, max_goals, n_samples)

    # 3. Compute
    @inbounds for k in 1:n_samples # Instantiate the distribution for this sample
        dist = DoubleNegativeBinomial(λ_h[k], λ_a[k], rₕ[k], rₐ[k])

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

# Information Theory metrics for model evaluation

## Overview 
To better understand and evaluate our models, we need to go beyond simple accuracy and ROI metrics.
Information theory provides a powerful set of tools for measuring the information content and divergence of probability distributions.

This task involves researching and implementing basic information theory metrics to compare our model's predictive distributions with the actual outcomes and market-implied probabilities.
The initial focus will be on metrics like Kullback-Leibler (KL) divergence and entropy.



## Utilities 

### Vectorized Data Structures

#### 2. Mathematical & Logical Framework

We will define two primary tensors:

-   **`P` (Prediction Tensor):** A 3D tensor to hold the predicted probabilities.
-   **`R` (Result Tensor):** A 2D tensor to hold the actual match outcomes.

#### Tensor Dimensions

-   **Prediction Tensor `P`**: `(L, K, M)`
    -   `L`: Number of matches
    -   `K`: Number of markets (e.g., home win, draw, away win, under 2.5)
    -   `M`: Number of MCMC chains

-   **Result Tensor `R`**: `(L, K)`
    -   `L`: Number of matches
    -   `K`: Number of markets

#### Example: Log-Likelihood Calculation

With this structure, the log-likelihood for a single chain `k` can be calculated as:

$$ \text{LL}_k = \sum_{i=1}^{L} \sum_{j=1}^{K} R_{ij} \log(P_{ijk}) $$

In Julia, this becomes a simple, vectorized operation, avoiding slow loops:

```julia
# First, average predictions across all chains
P_mean = mean(P, dims=3)

# Then, calculate the log-likelihood
log_likelihood = sum(R .* log.(P_mean))
```

#### Proposed Data Structures

``` julia 
struct PredictionTensor
    # Tensor of shape (matches, markets, chains)
    probabilities::Array{Float64, 3}

    # Dimension Mappings
    match_ids::Vector{Int}
    markets::Vector{Symbol}

    # Mappers for quick lookups
    match_map::Dict{Int, Int}
    market_map::Dict{Symbol, Int}
end

struct ResultTensor
    # Tensor of shape (matches, markets) with 0s and 1s
    outcomes::Matrix{Int}

    # Dimension Mappings
    match_ids::Vector{Int}
    markets::Vector{Symbol}

    # Mappers for quick lookups
    match_map::Dict{Int, Int}
    market_map::Dict{Symbol, Int}
end

```



## 

``` bash
⚡➜ models_julia (U! main) git stash
Saved working directory and index state WIP on main: b6fcc76 sampling ht and ft
⚡➜ models_julia (! main) git pull 
remote: Enumerating objects: 27, done.
remote: Counting objects: 100% (27/27), done.
remote: Compressing objects: 100% (15/15), done.
remote: Total 23 (delta 8), reused 19 (delta 5), pack-reused 0 (from 0)
Unpacking objects: 100% (23/23), 18.43 KiB | 1.54 MiB/s, done.
From https://github.com/JamesZor/models_julia
   b6fcc76..9cb608e  main                              -> origin/main
 * [new branch]      3-feat-information-theory-metrics -> origin/3-feat-information-theory-metrics
Updating b6fcc76..9cb608e
Fast-forward
 matchday/01_efl.jl                                    |  512 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 notebooks/09_training_models/batch_test_runner.jl     |   83 +++-------
 notebooks/09_training_models/pipeline_check_runner.jl |    0
 notebooks/09_training_models/pipeline_check_setup.jl  | 1318 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 train/pipeline_train_test.jl                          |    0
 workspace/example_workspace/README.md                 |   48 ++++++
 workspace/example_workspace/runners/runner_one.jl     |    6 +
 workspace/example_workspace/setup.jl                  |   12 ++
 8 files changed, 1920 insertions(+), 59 deletions(-)
 create mode 100644 matchday/01_efl.jl
 create mode 100644 notebooks/09_training_models/pipeline_check_runner.jl
 create mode 100644 notebooks/09_training_models/pipeline_check_setup.jl
 create mode 100644 train/pipeline_train_test.jl
 create mode 100644 workspace/example_workspace/README.md
 create mode 100644 workspace/example_workspace/runners/runner_one.jl
 create mode 100644 workspace/example_workspace/setup.jl
```

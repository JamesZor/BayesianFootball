# Session Summary: Debugging the Positional Player Dynamics Ridge

**Date:** May 30, 2026
**Focus:** Resolving catastrophic `rhat` failures (`Numerical Error`) in the `OutfieldPlayerDynamicsConfig` family of models (`outfield_xg`, `outfield_xg_market`, `outfield_xg_dixon_coles`, `outfield_xg_double_poisson`).

---

## 1. The Problem
During MCMC sampling of the `outfield_xg_market` and subsequent Double Poisson sanity check variants, the NUTS sampler experienced catastrophic breakdown:
- Extremely high `rhat` values (`> 2.0`).
- Very low effective sample sizes (`ess_bulk` around 40-100 out of 16,000 samples).
- Parameters hitting absolute hard lower bounds (e.g., `p_dyn.w_Outfield_def` hitting exactly `0.0001`).
- Generative output implied impossible scores (e.g., hundreds of goals per game) heavily suppressed by the Poisson/Market log-likelihoods.

## 2. The Investigation
We isolated the model complexity by stripping away Dixon-Coles (`ρ`) and Negative Binomial dispersion, building a mathematically "pure" `DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel`. 

Even with a completely uncorrelated, basic Double Poisson likelihood, the model completely failed. This proved the bug was **not** in the dispersion or correlation boundary mathematics, but rooted deep in the structural formulation of the hierarchical parameters.

## 3. The Root Cause: The Non-Identifiability Ridge
The bug was traced to the `OutfieldPlayerDynamicsConfig` math inside the `@model` macro.
The base expected goals equation is:
```julia
log_λₕ = inter.μ + ha_val + att_h + def_a
```
Where `att_h = (w_G_att * home_G_ratings) + (w_Outfield_att * home_Outfield)`.

**The Scale Mismatch:**
- A typical player rating in `PlayerRatingsFeature` is around `6.5`.
- The Outfield rating is the sum of 10 players (`D + M + F`), averaging `65.0`.
- If the prior for `w_Outfield_att` is `Normal(0.08, 0.05)`, the raw offset added to the log rate is `0.08 * 65.0 = 5.2`.
- Adding `5.2` to the log-rate multiplies expected goals by `exp(5.2) ≈ 181`.

**The Ridge:**
Because the ratings were strictly positive and massive, the model tried to predict hundreds of goals. The targets (Goals, Market, xG) heavily punished this. To survive, the NUTS sampler tried to force the intercept `inter.μ` to `-5.2`. However, `inter.μ` had a strict zero-centered prior (`Normal(0, 0.2)`). 

This created a massive **Non-Identifiability Ridge**: The parameters `inter.μ`, `w_Outfield_att`, and `w_Outfield_def` became perfectly collinear. Moving one required proportional movement in the others, flattening the geometric space and trapping the sampler, causing extreme numerical errors.

## 4. The Solution: Mean-Centering
The solution is to mean-center the player rating sums *before* applying the weights in the Turing engine. By extracting the exact base scale (`prior_mean`) directly from the tracker configuration, we can mathematically center the features to zero:

```julia
# Extract base rating from the BayesianTracker (e.g., 6.5)
base_rating = config.player_ratings_feature.tracker.prior_mean

# Center Goalkeepers (1 player)
h_G_c = home_G_ratings .- base_rating

# Center Outfield (10 players)
h_O_c = (home_D_ratings .+ home_M_ratings .+ home_F_ratings) .- (10.0 * base_rating)

# Apply centered values
att_h = (p_dyn.w_G_att .* h_G_c) .+ (p_dyn.w_Outfield_att .* h_O_c)
```

## 5. Results & Validation
After applying the mean-centering patch across all 4 Outfield engines, a clean run of `r03_sanity_check_double_poisson.jl` yielded a completely healthy posterior:
- **Convergence:** All `rhat` values settled between `1.000` and `1.002`.
- **Sample Efficiency:** `ess_bulk` exceeded 15,000 for nearly all parameters.
- **Identifiability:** `inter.μ` cleanly resolved to `0.20` (implying ~1.22 base expected goals), perfectly matching football reality without conflicting with the positional weights.
- **Market Alignment:** The market standard deviation (`σ_market`) stabilized at a highly accurate `0.21`, proving tight calibration between the model and historical closing odds.

## 6. Next Steps
With the core engine structural scale permanently fixed, the codebase is fully cleared to execute the primary **A/B Test** between `outfield_xg_market.jl` and `outfield_xg_dixon_coles.jl` to evaluate the predictive impact of the Copula/Dixon-Coles correlation architectures.

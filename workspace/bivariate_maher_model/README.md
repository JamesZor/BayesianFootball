# Bivariate Maher Model Workspace

This workspace is dedicated to the development, testing, and analysis of an extended Bayesian football prediction model. The core of this work is the implementation of a Bivariate Maher model using the Julia programming language and the `Turing.jl` probabilistic programming framework.

## Overview

The standard Maher model is a well-regarded approach for predicting football match outcomes. However, it relies on a key simplifying assumption: that the number of goals scored by the home and away teams are **independent** events.

This project rectifies that by implementing a **Bivariate Poisson model**, which captures the real-world correlation between the goal-scoring processes of the two teams in a match. This allows for a more nuanced and accurate representation of game dynamics, such as defensive, low-scoring affairs or open, high-scoring matches.

This workspace contains the full workflow: from the statistical model definition and training scripts to the functions for generating match-day predictions, comparing them against live market odds, and calculating Expected Value (EV).

---

## The Core Idea: From Independent to Bivariate Maher

### Limitations of the Standard Maher Model

The classic Maher model estimates goal-scoring rates ($\lambda_{home}$, $\lambda_{away}$) and assumes the goals are drawn from independent Poisson distributions:
$$Y_{home} \sim \text{Poisson}(\lambda_{home})$$
$$Y_{away} \sim \text{Poisson}(\lambda_{away})$$
This assumption fails to capture crucial dependencies. For example, a match played in poor weather or between two highly defensive teams is likely to see fewer goals from *both* sides. The independent model struggles to account for this, often underestimating the probability of low-scoring draws (0-0, 1-1).

### The Bivariate Solution

We replace the independent likelihood with a single observation from a Bivariate Poisson distribution. The probability mass function is given by:

$$
P(X=x, Y=y) = \exp(-(\lambda_x + \lambda_y + \gamma)) \frac{\lambda_x^x}{x!} \frac{\lambda_y^y}{y!} \sum_{k=0}^{\min(x,y)} \binom{x}{k} \binom{y}{k} k! \left(\frac{\gamma}{\lambda_x \lambda_y}\right)^k
$$

The crucial addition is the **covariance parameter `γ` (gamma)**. This term explicitly models the dependency between the home and away goal counts ($Cov(Y_{home}, Y_{away}) = \gamma$). A positive `γ` indicates that if one team scores more, the other is also likely to score more (an open game), while a `γ` near zero suggests the goal counts are nearly independent.

---

## Implementation in Julia & Turing

The model is implemented using `Turing.jl`. Since the Bivariate Poisson is not a built-in Turing distribution, we define a custom log-probability density function and incorporate it into the model.

### Model Definition (`setup.jl`)

The core model logic is defined in `workspace/bivariate_maher_model/setup.jl`.

1.  **`MaherBivariate` struct:** A type to dispatch our specific model functions[cite: 801].
2.  **`logpdf_bivariate_poisson(...)`:** A custom function to calculate the log-probability of a given scoreline[cite: 2].
3.  **Turing `@model`:** The `maher_bivariate_model` function defines the priors for team strengths ($\alpha, \beta$), home advantage ($\delta$), and our new covariance parameter ($\gamma$)[cite: 4]. It uses the custom logpdf function to define the likelihood[cite: 7].

The key line that integrates our custom likelihood is:
```julia
# From setup.jl
Turing.@addlogprob! logpdf_bivariate_poisson(home_goals[k], away_goals[k], λ, μ, γ) [cite: 7]
```

### Prediction (`prediction.jl`)

The prediction logic in `workspace/bivariate_maher_model/prediction.jl` uses the full MCMC chain (posterior samples) to generate predictions[cite: 13]. For each sample, it:
1.  Calculates the goal rates $\lambda$ and $\mu$ and retrieves the sampled `γ`[cite: 15].
2.  Computes the entire score grid probability using `compute_bivariate_xScore`[cite: 11, 12].
3.  Aggregates these probabilities to derive odds for various markets (1x2, O/U, BTTS, Correct Score)[cite: 20, 21].

---

## Initial Findings

Preliminary results from match-day predictions show promising behavior from the bivariate model, particularly in scenarios where goal correlation is expected to be significant.

* **Improved Draw & Low-Score Pricing:** For matches that ended in low-scoring draws, the bivariate model's predictions were often closer to the market odds than the standard independent Maher model.
    * In the **Bournemouth v Newcastle** match (FT: 0-0) [cite: 690], the `bivar_24_26` model priced a 0-0 draw at **13.88** [cite: 276], closer to the market back price of **14.5** [cite: 276] than the `maher_24_26` model's price of **15.42**[cite: 276].
    * Similarly, for the **Sunderland v Aston Villa** match (FT: 1-1) [cite: 697], the `bivar_24_26` model priced the draw at **3.45** [cite: 606] (market: 3.4 [cite: 606]) while the Maher model was at **3.56**[cite: 606].
* **Concordance on Mismatches:** In highly unbalanced matches, such as **Partick v Celtic** (FT: 0-4)[cite: 703], both the bivariate and standard Maher models produced similar odds. This suggests the influence of the `γ` parameter is less pronounced when one team is overwhelmingly dominant. In these cases, the quality and length of the training data proved to be the most critical factor, with models trained on more seasons (`_24_26`) performing significantly better than those with less data (`_2526`)[cite: 351].

These findings support the underlying hypothesis: explicitly modeling goal covariance helps capture the dynamics of tight, defensive games more effectively.

---

## End-to-End Analysis Workflow

The scripts and modules in this workspace provide a complete pipeline for daily match analysis.

1.  **Load Models (`analysis_funcs.jl`):** The `load_models_from_paths` function loads the trained Turing model chains from specified experiment directories[cite: 23, 24].
2.  **Fetch Match Data (`match_day_utils.jl`):** The `get_todays_matches` function runs a Python CLI tool (`whatstheodds`) to scrape today's fixtures and aligns team names using JSON mappings[cite: 750].
3.  **Get Market Odds (`match_day_utils.jl`):** The `fetch_all_market_odds` function calls the same CLI tool to get live back and lay odds from Betfair for all relevant markets and compiles them into a single, wide-format DataFrame[cite: 789, 790].
4.  **Generate & Analyze (`match_day_utils.jl`):** The `generate_match_analysis` function is the main orchestrator[cite: 776]. For a given match, it:
    * Takes the loaded models and the live market odds as input[cite: 774].
    * Generates a `PredictionMatrix` for each model, containing the full posterior probability distribution for every market[cite: 50, 711].
    * Calculates the mean model odds, standard deviation, and the mean Expected Value (EV) in percentage terms for every market against the live back odds[cite: 775].
    * Returns a single, comprehensive DataFrame comparing all models against the market[cite: 775].
5.  **Visualize (`match_day_utils.jl`):** The `plot_multi_model_odds_distribution` function takes the prediction matrices and plots the density of the model's predicted odds for a specific market, overlaying the live back/lay odds as vertical lines[cite: 782, 784]. This provides a powerful visual diagnostic of model confidence vs. market price.

### Example Usage

The following is a condensed example of a complete match-day analysis workflow.

```julia
using .Analysis
using .MatchDayUtils

# 1. Define and load all trained model objects
all_model_paths = Dict(
    "maher_24_26" => "/path/to/maher_model",
    "bivar_24_26" => "/path/to/bivar_model"
)
loaded_models = load_models_from_paths(all_model_paths);

# 2. Get today's matches and fetch all market odds into a single DataFrame
todays_matches = get_todays_matches(["england", "scotland"]; cli_path=CLI_PATH);
odds_df = fetch_all_market_odds(todays_matches, MARKET_LIST; cli_path=CLI_PATH);

# 3. Select a match and run the comprehensive analysis
match_to_analyze = first(todays_matches) # e.g., the first match
(comparison_df, prediction_matrices, market_book) = generate_match_analysis(
    match_to_analyze,
    odds_df,
    loaded_models,
    MARKET_LIST
);

# 4. Display the results DataFrame
println("Analysis for: $(match_to_analyze.event_name)")
show(comparison_df; allcols=true)

# 5. Visualize the odds distribution for a specific market
p = plot_multi_model_odds_distribution(
    prediction_matrices,
    market_book,
    :ft_1x2_home # Plot the home win market
)
display(p)
```

---

## Future Work

While the bivariate model shows promise, several extensions could further improve its predictive power:

* **Time-Varying Parameters:** Team strengths are not static. Future iterations could incorporate a time-series element (e.g., a state-space model or Gaussian random walk) to allow team attack and defense parameters ($\alpha_i, \beta_i$) to evolve over time, better capturing team form.
* **Hierarchical Draw Model:** Draws are often a result of unique game dynamics not fully captured by goal-scoring rates alone. A separate hierarchical model could be implemented specifically for the draw outcome (e.g., a logistic regression model), which would then be combined with the goal-based model for a more robust win/loss prediction.

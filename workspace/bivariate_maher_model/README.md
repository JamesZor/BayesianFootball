# Bivariate Maher Model Extension

This experiment extends the standard Maher football model to a bivariate setting. The key innovation is the use of a bivariate Poisson distribution for the likelihood of the number of goals scored by the home and away teams. This allows us to capture the correlation between the goal-scoring intensity of the two teams in a match.

## Mathematical Formulation

The standard Maher model assumes that the number of goals scored by the home and away teams are independent Poisson distributions. In this extension, we replace this assumption with a bivariate Poisson distribution, which is defined as:
$$
P(X=x, Y=y) = exp(-(λx + λy + γ)) * (λx^x / x!) * (λy^y / y!) * Σ [ (x choose k) * (y choose k) * k! * (γ / (λx * λy))^k ]
$$
where:
- `x` and `y` are the number of goals scored by the home and away teams, respectively.
- `λx` and `λy` are the goal-scoring intensities for the home and away teams.
- `γ` is the covariance term that captures the dependency between the two goal-scoring processes.

This model is implemented in `setup.jl` and can be run using the scripts in the `runners` directory.

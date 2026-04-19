# src/models/pregame/components/copula_densities.jl
using Distributions, LogExpFunctions

# 1. Gaussian Copula Log-Density
# Standard Bivariate Normal correlation density
function log_c_gaussian(u, v, ρ)
    # Map uniform to standard normal
    x = quantile(Normal(0,1), u)
    y = quantile(Normal(0,1), v)
    
    # Quadratic form for correlation
    # log(1 / sqrt(1-rho^2)) - (rho^2(x^2+y^2) - 2rho*x*y) / (2(1-rho^2))
    denom = 1 - ρ^2
    log_det = -0.5 * log(denom)
    quad = (ρ^2 * (x^2 + y^2) - 2 * ρ * x * y) / (2 * denom)
    return log_det - quad
end

# 2. Clayton Copula Log-Density (Lower Tail Dependence)
# c(u,v) = (1+theta) * (u*v)^(-1-theta) * (u^-theta + v^-theta - 1)^(-2 - 1/theta)
function log_c_clayton(u, v, θ)
    # Numerical stability: limit u,v > 0
    u = max(u, 1e-9); v = max(v, 1e-9)
    
    term1 = log(1 + θ)
    term2 = -(1 + θ) * (log(u) + log(v))
    term3 = (-2 - (1/θ)) * log(max(1e-9, u^(-θ) + v^(-θ) - 1))
    
    return term1 + term2 + term3
end

# 3. Frank Copula Log-Density (Negative Dependence / Symmetric)
# Complex formula, good for rho in [-1, 1]
function log_c_frank(u, v, θ)
    # Avoid zero theta (independence)
    if abs(θ) < 1e-6
        return 0.0 
    end
    
    e_theta = exp(-θ) - 1
    num = -θ * (exp(-θ) - 1) * exp(-θ * (u + v))
    den_inner = (exp(-θ) - 1) + (exp(-θ * u) - 1) * (exp(-θ * v) - 1)
    den = den_inner^2
    
    return log(max(1e-9, num)) - log(max(1e-9, den))
end

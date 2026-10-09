# Diagnostic definitions only: distinguish a returned likelihood from an accumulated one.
module Wave2LikelihoodAudit
import BayesianFootball, CSV, DataFrames, Distributions, DynamicPPL, LogDensityProblems
import Random, Turing
const API = BayesianFootball.Models.PreGame.Builder

# Mirror the production engine's observation seam, including its likelihood accumulation.
Turing.@model function accumulated(observation_model)
    ll ~ DynamicPPL.to_submodel(observation_model, false)
    Turing.@addlogprob! ll
end

"Evaluate the same constrained sigma in both coordinate spaces, without changing src/tests."
function audit(output)
    prior = Distributions.truncated(Distributions.Normal(0.0, 0.20), 0.0, Inf)
    table = DataFrames.DataFrame(match_id = [2, 1, 3], log_lambda_h = [0.2, 0.3, NaN],
        log_lambda_a = [-0.1, 0.1, NaN], full_book = [true, true, false])
    feature = BayesianFootball.Features.MarketRatesFeature(table)
    observation = BayesianFootball.MarketRateObservation(feature = feature, sigma_prior = prior)
    h, a, sigma = [0.4, 0.1, 8.0, -4.0], [0.0, -0.2, -3.0, 7.0], 0.12
    design = API.MarketRateDesign([0.3, 0.2, 0.0, 0.0], [0.1, -0.1, 0.0, 0.0],
        [1.0, 0.5, 0.0, 0.0], 1.5)
    hand = sum(design.mask_weights[i] * (-2log(sigma) - log(2pi) -
        ((design.log_h[i] - h[i])^2 + (design.log_a[i] - a[i])^2) / (2sigma^2)) for i in 1:4)
    original_expected = sum(design.mask_weights[i] *
        (Distributions.logpdf(Distributions.Normal(h[i], sigma), design.log_h[i]) +
         Distributions.logpdf(Distributions.Normal(a[i], sigma), design.log_a[i])) for i in 1:4)
    sub = API._observe(observation, h, a, zeros(Int, 4), zeros(Int, 4), ones(4),
        zeros(4), zeros(4), 2, 12, design)
    rows = NamedTuple[]
    for (name, model) in (("returned_only", sub), ("production_seam", accumulated(sub)))
        for linked in (false, true)
            Random.seed!(20261010)
            vi = DynamicPPL.VarInfo(model)
            linked && (vi = DynamicPPL.link!!(vi, model))
            point = linked ? [log(sigma)] : [sigma]
            length(vi[:]) == 1 || error("unexpected diagnostic sites")
            values = map((DynamicPPL.getlogprior, DynamicPPL.getloglikelihood,
                          DynamicPPL.getlogjoint, DynamicPPL.getlogjoint_internal)) do selector
                density = DynamicPPL.LogDensityFunction(model, selector, vi)
                LogDensityProblems.logdensity(density, point)
            end
            row = (; model = name, linked, sigma, coordinate = only(point),
                logprior = values[1], loglikelihood = values[2], model_logjoint = values[3],
                internal_logjoint = values[4], jacobian = values[4] - values[3],
                hand_loglikelihood = hand, original_expected_loglikelihood = original_expected,
                pure_helper_loglikelihood = API._market_rate_ll(h, a, sigma, design),
                hand_logprior = Distributions.logpdf(prior, sigma))
            push!(rows, row)
            println("LIKELIHOOD_AUDIT ", row)
        end
    end
    mkpath(output)
    CSV.write(joinpath(output, "likelihood_audit.csv"), DataFrames.DataFrame(rows))
    return rows
end
end

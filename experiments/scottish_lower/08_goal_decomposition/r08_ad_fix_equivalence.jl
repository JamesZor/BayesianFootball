# ==============================================================================
# r08 — posterior equivalence of the AD fix (TODO 003)
# ==============================================================================
#
# Compares two 16-chain config-B batches of m00 on Fold 1 with the same seeds:
# the production path (`adtype` passed to `sample`, so ForwardDiff runs) and the
# benchmark-only fix (`AutoReverseDiff(compile = true)` in the `NUTS` constructor).
# The draws cannot be bit-identical because the gradients differ in the last bits, so
# the check is distributional. For every parameter it computes the difference of pooled
# posterior means in Monte Carlo standard errors, z = Δμ / √(MCSE₁² + MCSE₂²), and the
# ratio of posterior standard deviations. Equivalent posteriors give z ≈ N(0, 1) across
# parameters and sd ratios near 1.
#
#   julia --project -t 1 r08_ad_fix_equivalence.jl [reference_tag] [candidate_tag]
# ==============================================================================

using BayesianFootball  # the serialized chains carry Turing/AbstractPPL types
using CSV
using DataFrames
using MCMCChains
using Printf
using Serialization
using Statistics

const R08E_DIR = joinpath(@__DIR__, "results", "sampling_budget_fold1")
const R08E_REFERENCE = get(ARGS, 1, "gcA_default_20260910")
const R08E_CANDIDATE = get(ARGS, 2, "gcE_default_adfix_20260910")

"All chains of one run pooled into a single `Chains` (4 fits × 4 chains = 16 chains)."
function r08e_pooled(tag)
    raw = deserialize(joinpath(R08E_DIR, tag, "chains.jls"))
    keys_sorted = sort!(collect(keys(raw)); by = k -> (k[3], k[4]))
    return cat((raw[k] for k in keys_sorted)...; dims = 3)
end

"Posterior mean, sd and MCSE of the mean for every parameter."
function r08e_summary(chain)
    params = MCMCChains.names(chain, :parameters)
    ess = DataFrame(MCMCChains.ess(chain[params]; kind = :bulk))
    rows = map(params) do p
        x = vec(Array(chain[p]))
        n_eff = ess.ess[findfirst(==(p), ess.parameters)]
        (parameter = String(p), mean = mean(x), sd = std(x), mcse = std(x) / sqrt(n_eff))
    end
    return DataFrame(rows)
end

ref = r08e_summary(r08e_pooled(R08E_REFERENCE))
cand = r08e_summary(r08e_pooled(R08E_CANDIDATE))
ref.parameter == cand.parameter || error("parameter sets differ between runs")
comparison = DataFrame(parameter = ref.parameter,
    mean_reference = ref.mean, mean_candidate = cand.mean,
    z = (cand.mean .- ref.mean) ./ sqrt.(ref.mcse .^ 2 .+ cand.mcse .^ 2),
    sd_ratio = cand.sd ./ ref.sd)
CSV.write(joinpath(R08E_DIR, R08E_CANDIDATE, "equivalence_vs_$(R08E_REFERENCE).csv"), comparison)

n = nrow(comparison)
println("\n", "="^92)
println(" AD-FIX POSTERIOR EQUIVALENCE · ", R08E_CANDIDATE, " vs ", R08E_REFERENCE, " · ", n, " parameters")
println("="^92)
@printf("  z = Δmean / MCSE : mean %+.3f · sd %.3f · max |z| %.2f (%s)\n",
    mean(comparison.z), std(comparison.z), maximum(abs, comparison.z), comparison.parameter[argmax(abs.(comparison.z))])
@printf("  |z| > 2 : %d of %d (N(0,1) expects %.1f) · |z| > 3 : %d (expects %.2f)\n",
    count(>(2), abs.(comparison.z)), n, 0.0455 * n, count(>(3), abs.(comparison.z)), 0.0027 * n)
@printf("  sd ratio : median %.3f · range %.3f–%.3f\n",
    median(comparison.sd_ratio), minimum(comparison.sd_ratio), maximum(comparison.sd_ratio))

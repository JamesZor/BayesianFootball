# ==============================================================================
# 08 — Decomposed goal-model deterministic verification helpers
# ==============================================================================
# Include l08_decomposed_models.jl first.  These helpers deliberately do not sample.

import DynamicPPL
import LogDensityProblems
import ReverseDiff
import ForwardDiff
import LinearAlgebra
import Statistics
import Distributions
import Random

"Relative vector error robust at a zero gradient."
l08_relerr(a, b) = LinearAlgebra.norm(a .- b) / max(LinearAlgebra.norm(a), LinearAlgebra.norm(b), 1.0)

"Return the linked NUTS-coordinate log joint (including Jacobians) at a seeded prior draw."
function l08_logdensity_problem(model, fs; seed::Int = 20260909, linked::Bool = true)
    turing_model = GD_PG.build_turing_model(model, fs)
    raw_vi = DynamicPPL.VarInfo(Random.MersenneTwister(seed), turing_model)
    vi = linked ? DynamicPPL.link(raw_vi, turing_model) : raw_vi
    ldf = DynamicPPL.LogDensityFunction(turing_model, DynamicPPL.getlogjoint_internal, vi)
    θ = copy(vi[:])
    f = θx -> LogDensityProblems.logdensity(ldf, θx)
    isfinite(f(θ)) || error("initial model log density is not finite")
    return (; turing_model, vi, raw_vi, θ, f)
end

"Scalar independent density for a fully classified single side, excluding priors."
function l08_reference_classified_side(z, i, η_regular, η_penalty, η_own, conversion; side::Symbol)
    regular = side === :home ? z.regular_h[i] : z.regular_a[i]
    awarded = side === :home ? z.awarded_h[i] : z.awarded_a[i]
    converted = side === :home ? z.converted_h[i] : z.converted_a[i]
    own = side === :home ? z.own_h[i] : z.own_a[i]
    return decomposed_side_loglikelihood(Int(regular), Int(awarded), Int(converted), Int(own),
                                         η_regular, η_penalty, η_own, conversion)
end

"Superposition identity for one side: component Poissons imply the total Poisson rate."
function l08_superposition_rate(λ_regular, λ_awarded, conversion, λ_own)
    return λ_regular .+ conversion .* λ_awarded .+ λ_own
end

"A finite 12×12 independent-Poisson grid whose retained mass is normalized exactly."
function l08_normalized_grid(λh::Real, λa::Real; max_goals::Int = 12)
    ph = [Distributions.pdf(Distributions.Poisson(λh), g) for g in 0:max_goals-1]
    pa = [Distributions.pdf(Distributions.Poisson(λa), g) for g in 0:max_goals-1]
    ph ./= sum(ph)
    pa ./= sum(pa)
    return ph * transpose(pa)
end

"Compiled/fresh ReverseDiff and ForwardDiff agreement, including large static-tape probes."
function l08_gradient_checks(model, fs; perturbation::Float64 = 0.8, probes_count::Int = 40)
    p = l08_logdensity_problem(model, fs)
    raw = ReverseDiff.GradientTape(p.f, p.θ)
    tape = ReverseDiff.compile(raw)
    g_compiled = similar(p.θ)
    ReverseDiff.gradient!(g_compiled, tape, p.θ)
    g_fresh = ReverseDiff.gradient(p.f, p.θ)
    g_forward = ForwardDiff.gradient(p.f, p.θ)
    probes = Float64[]
    forward_probes = Float64[]
    rng = Random.MersenneTwister(20260909)
    for probe in 1:probes_count
        scale = probe <= 3 ? 0.001 * probe : perturbation
        θp = p.θ .+ scale .* Random.randn(rng, length(p.θ))
        isfinite(p.f(θp)) || error("broad linked-coordinate probe $probe has non-finite density")
        gp = similar(θp)
        ReverseDiff.gradient!(gp, tape, θp)
        push!(probes, l08_relerr(ReverseDiff.gradient(p.f, θp), gp))
        push!(forward_probes, l08_relerr(ForwardDiff.gradient(p.f, θp), gp))
    end
    report = (; instructions = length(raw.tape), compiled_fresh = l08_relerr(g_compiled, g_fresh),
              compiled_forward = l08_relerr(g_compiled, g_forward), perturbed = probes,
              perturbed_forward = forward_probes, parameter_space = "linked_NUTS_with_Jacobians",
              tape, θ = p.θ, f = p.f)
    report.compiled_fresh <= 1e-8 || error("compiled/fresh ReverseDiff disagreement $(report.compiled_fresh)")
    report.compiled_forward <= 1e-6 || error("compiled ReverseDiff/ForwardDiff disagreement $(report.compiled_forward)")
    all(<=(1e-8), report.perturbed) || error("compiled-tape perturbation disagreement $(report.perturbed)")
    all(<=(1e-6), report.perturbed_forward) || error("compiled/ForwardDiff broad-probe disagreement")
    return report
end

"Measure warmed compiled-gradient allocations and minimum latency without MCMC."
function l08_gradient_benchmark(check; repetitions::Int = 100)
    g = similar(check.θ)
    for _ in 1:20
        ReverseDiff.gradient!(g, check.tape, check.θ)
    end
    allocations = @allocated ReverseDiff.gradient!(g, check.tape, check.θ)
    seconds = minimum(@elapsed(ReverseDiff.gradient!(g, check.tape, check.θ)) for _ in 1:repetitions)
    return (; allocations, milliseconds = 1e3 * seconds)
end

"Instruction-scaling check: duplicate feature rows; a vectorized model's tape is O(1) in rows."
function l08_instruction_scaling(model, fs, duplicate_fs)
    base = l08_logdensity_problem(model, fs)
    duplicated = l08_logdensity_problem(model, duplicate_fs)
    n_base = length(ReverseDiff.GradientTape(base.f, base.θ).tape)
    n_duplicate = length(ReverseDiff.GradientTape(duplicated.f, duplicated.θ).tape)
    n_base == n_duplicate || error("tape scales with fixture rows: $n_base -> $n_duplicate")
    return (; base_instructions = n_base, duplicate_instructions = n_duplicate,
              ratio = n_duplicate / n_base)
end

l08_parameter_values(vi) = Dict(string(vn) => vi[vn] for vn in keys(vi))

"Independent scalar reconstruction: does not call the engine's rate or prior helpers."
function l08_reference_state(model, values)
    p = model.priors
    scalar(name) = only(values[name])
    center(name, sd) = sd .* (values[name] .- Statistics.mean(values[name]))
    regular_sd = p.regular_team_sd * exp(scalar("sigma_regular_raw"))
    referee_sd = p.referee_sd * exp(scalar("sigma_referee_raw"))
    penalty_att = zeros(length(values["raw_attack"]))
    penalty_def = copy(penalty_att)
    if model isa DecomposedTeamPenaltiesModel
        penalty_att = center("raw_att", p.penalty_team_sd * exp(scalar("sigma_att_raw")))
        penalty_def = center("raw_def", p.penalty_team_sd * exp(scalar("sigma_def_raw")))
    end
    return (; mu_r = scalar("mu_regular"), ha_r = scalar("ha_regular"),
        mu_p = scalar("mu_penalty"), ha_p = scalar("ha_penalty"), mu_o = scalar("mu_own"),
        k = 1 / (1 + exp(-scalar("conversion_raw"))),
        attack = center("raw_attack", regular_sd), defence = center("raw_defence", regular_sd),
        referee = center("raw_referee", referee_sd), penalty_att, penalty_def,
        pressure = model isa DecomposedPressureOwnGoalsModel ? scalar("pressure") : 0.0)
end

function l08_reference_rates(state, hi, ai, referee_shift)
    eta_rh = state.mu_r + state.ha_r + state.attack[hi] + state.defence[ai]
    eta_ra = state.mu_r + state.attack[ai] + state.defence[hi]
    eta_ph = state.mu_p + state.ha_p + referee_shift + state.penalty_att[hi] + state.penalty_def[ai]
    eta_pa = state.mu_p + referee_shift + state.penalty_att[ai] + state.penalty_def[hi]
    eta_oh = state.mu_o + state.pressure * (eta_rh - state.mu_r)
    eta_oa = state.mu_o + state.pressure * (eta_ra - state.mu_r)
    return (; eta_rh, eta_ra, eta_ph, eta_pa, eta_oh, eta_oa,
        lambda_h = exp(eta_rh) + state.k * exp(eta_ph) + exp(eta_oh),
        lambda_a = exp(eta_ra) + state.k * exp(eta_pa) + exp(eta_oa))
end

"Distribution-object log joint, including exact HalfNormal/Beta transformation Jacobians."
function l08_reference_logjoint(model, fs, values)
    p, z = model.priors, gd_design(model, fs)
    scalar(name) = only(values[name])
    normal(name, mu, sd) = Distributions.logpdf(Distributions.Normal(mu, sd), scalar(name))
    halfnormal(name, sd) = begin
        sigma = sd * exp(scalar(name))
        Distributions.logpdf(Distributions.truncated(Distributions.Normal(0, sd), 0, Inf), sigma) + log(sigma)
    end
    lp = normal("mu_regular", p.regular_log_mean, p.regular_log_sd) + normal("ha_regular", 0, p.regular_home_sd) +
         normal("mu_penalty", p.penalty_log_mean, p.penalty_log_sd) + normal("ha_penalty", 0, p.penalty_home_sd) +
         normal("mu_own", p.own_log_mean, p.own_log_sd) + halfnormal("sigma_regular_raw", p.regular_team_sd) +
         halfnormal("sigma_referee_raw", p.referee_sd)
    for name in ("raw_attack", "raw_defence", "raw_referee")
        lp += sum(Distributions.logpdf.(Distributions.Normal(), values[name]))
    end
    state = l08_reference_state(model, values)
    lp += Distributions.logpdf(Distributions.Beta(p.conversion_alpha, p.conversion_beta), state.k) + log(state.k) + log1p(-state.k)
    if model isa DecomposedTeamPenaltiesModel
        for name in ("raw_att", "raw_def")
            lp += sum(Distributions.logpdf.(Distributions.Normal(), values[name]))
        end
        lp += halfnormal("sigma_att_raw", p.penalty_team_sd) + halfnormal("sigma_def_raw", p.penalty_team_sd)
    elseif model isa DecomposedPressureOwnGoalsModel
        lp += normal("pressure", 0, p.own_pressure_sd)
    end
    for i in eachindex(z.h)
        rates = l08_reference_rates(state, z.h[i], z.a[i], z.referee_known[i] * state.referee[z.referee_ids[i]])
        if z.complete[i] == 1.0
            ll = l08_reference_classified_side(z, i, rates.eta_rh, rates.eta_ph, rates.eta_oh, state.k; side = :home) +
                 l08_reference_classified_side(z, i, rates.eta_ra, rates.eta_pa, rates.eta_oa, state.k; side = :away)
        else
            ll = Distributions.logpdf(Distributions.Poisson(rates.lambda_h), Int(z.total_h[i])) +
                 Distributions.logpdf(Distributions.Poisson(rates.lambda_a), Int(z.total_a[i]))
        end
        lp += z.weights[i] * ll
    end
    return lp
end

"Two synthetic parameter draws exercise extraction; these are NOT an MCMC smoke."
function l08_synthetic_chain(model, fs)
    values = [l08_parameter_values(l08_logdensity_problem(model, fs; seed).raw_vi) for seed in (17, 29)]
    names = Symbol[]
    columns = Vector{Float64}[]
    for name in sort!(collect(keys(values[1])))
        for i in eachindex(values[1][name])
            push!(names, Symbol("$name[$i]"))
            push!(columns, [draw[name][i] for draw in values])
        end
    end
    return MCMCChains.Chains(reshape(hcat(columns...), 2, length(names), 1), names), values
end

function l08_duplicate_features(fs)
    n = length(fs.data[:flat_home_ids])
    data = copy(fs.data)
    for (key, value) in data
        value isa AbstractVector && length(value) == n && (data[key] = vcat(value, value))
    end
    return typeof(fs)(data)
end

"Perturb every strictly future component row/referee through the REAL feature adapter."
function l08_filtration_checks(model, fs)
    ids = fs.data[:goal_decomposition_training_ids]
    idset = Set(ids)
    cutoff = maximum(row.match_date for row in eachrow(model.registry.matches) if row.match_id in idset)
    changed = deepcopy(model.registry)
    future = findall(>(cutoff), changed.matches.match_date)
    isempty(future) && error("filtration check requires at least one future row")
    changed.matches[future, :non_penalty_non_own_goal_home] .+= 7
    changed.matches[future, :overall_home] .+= 7
    changed.matches[future, :referee_id] .= "FUTURE_ONLY_REFEREE"
    sha = GoalDecompositionIncidentData.registry_snapshot_hash(changed)
    sha != model.registry_hash || error("filtration perturbation did not change snapshot")
    l08_pre_target_prior_anchor(changed) == l08_pre_target_prior_anchor(model.registry) || error("future counts contaminated prior anchors")
    before, after = Dict{Symbol,Any}(), Dict{Symbol,Any}()
    l08_add_component_features!(before, GoalDecompositionFeature(model.registry, model.registry_hash), ids)
    l08_add_component_features!(after, GoalDecompositionFeature(changed, sha), ids)
    delete!(before, :goal_decomposition_data_hash)
    delete!(after, :goal_decomposition_data_hash)
    isequal(before, after) || error("future rows changed fitted component features or referee vocabulary")
    return length(future)
end

function l08_grid_allocations(latents::GD.CountLatents)
    buffer = GD.Predictions.alloc_score_grid(latents, 12)
    workspace = GD.Predictions.GridWorkspace(12)
    for _ in 1:20
        GD.Predictions.compute_score_grid!(buffer, workspace, latents, 1)
    end
    return @allocated GD.Predictions.compute_score_grid!(buffer, workspace, latents, 1)
end

function l08_extraction_checks(model, fs, fixtures)
    chain, draws = l08_synthetic_chain(model, fs)
    latents = GD.Models.extract_latents(model, chain, fixtures, fs)
    latents isa GD.CountLatents || error("non-count latent family")
    latents.match_ids == Int.(fixtures.match_id) || error("latent fixture ordering changed")
    rows = Dict(Int(row.match_id) => row for row in eachrow(model.registry.matches))
    max_rate_error, max_grid_error, max_tail_mass = 0.0, 0.0, 0.0
    for (j, row) in enumerate(eachrow(fixtures))
        grid = GD.Predictions.compute_score_grid(latents, j)
        size(grid) == (12, 12, 2) || error("unexpected native grid shape")
        all(isfinite, grid) && all(>=(0), grid) || error("invalid native score grid")
        for (k, draw) in enumerate(draws)
            state = l08_reference_state(model, draw)
            ref = get(fs.data[:referee_map], rows[row.match_id].referee_id, 0)
            rates = l08_reference_rates(state, fs.data[:team_map][row.home_team], fs.data[:team_map][row.away_team], ref == 0 ? 0.0 : state.referee[ref])
            max_rate_error = max(max_rate_error, abs(latents.λ_home[j, k] - rates.lambda_h), abs(latents.λ_away[j, k] - rates.lambda_a))
            for h in 0:11, a in 0:11
                expected = Distributions.pdf(Distributions.Poisson(rates.lambda_h), h) * Distributions.pdf(Distributions.Poisson(rates.lambda_a), a)
                max_grid_error = max(max_grid_error, abs(grid[h + 1, a + 1, k] - expected))
            end
            mass = sum(view(grid, :, :, k))
            0 < mass <= 1 + 1e-12 || error("invalid retained score-grid mass")
            max_tail_mass = max(max_tail_mass, 1 - mass)
        end
    end
    max_rate_error < 1e-10 || error("independent per-draw superposition/extraction disagreement")
    max_grid_error < 1e-12 || error("native in-place grid does not match component-summed rates")
    bad = DataFrames.DataFrame(fixtures[1:1, :])
    bad.home_team .= "UNSEEN_TEAM_PROBE"
    refused = try
        GD.Models.extract_latents(model, chain, bad, fs)
        false
    catch err
        occursin("REFUSED unseen-team", sprint(showerror, err)) && occursin(string(bad.match_id[1]), sprint(showerror, err))
    end
    refused || error("unseen team was not refused by fixture ID")
    zero_ref_latents = Any[]
    for ref in ("UNKNOWN", "NEVER_FITTED_REFEREE")
        registry = deepcopy(model.registry)
        registry.matches[findall(==(fixtures.match_id[1]), registry.matches.match_id), :referee_id] .= ref
        sha = GoalDecompositionIncidentData.registry_snapshot_hash(registry)
        replacement = typeof(model)(; priors = model.priors, registry, registry_hash = sha, data_hash = sha, days_half_life = model.days_half_life)
        data = copy(fs.data)
        data[:goal_decomposition_data_hash] = sha
        push!(zero_ref_latents, GD.Models.extract_latents(replacement, chain, fixtures[1:1, :], typeof(fs)(data)))
    end
    zero_ref_latents[1].λ_home == zero_ref_latents[2].λ_home && zero_ref_latents[1].λ_away == zero_ref_latents[2].λ_away || error("missing/unseen referees are not identical")
    for (k, draw) in enumerate(draws)
        row = fixtures[1, :]
        rates = l08_reference_rates(l08_reference_state(model, draw), fs.data[:team_map][row.home_team], fs.data[:team_map][row.away_team], 0.0)
        isapprox(zero_ref_latents[1].λ_home[1, k], rates.lambda_h; rtol = 1e-12) || error("missing referee did not receive exactly zero effect")
        isapprox(zero_ref_latents[1].λ_away[1, k], rates.lambda_a; rtol = 1e-12) || error("missing referee did not receive exactly zero effect")
    end
    grid_allocations = l08_grid_allocations(latents)
    grid_allocations == 0 || error("in-place grid allocated $grid_allocations bytes")
    return (; max_rate_error, max_grid_error, max_tail_mass, grid_allocations, synthetic_draws = 2,
        fixtures = DataFrames.nrow(fixtures), unseen_team_refused = true, missing_referee_zero = true)
end

"Complete no-MCMC real-feature gate; returns only manifest-serializable values."
function l08_deterministic_checks(model, fs, oos; probes_count::Int = 40)
    gradient = l08_gradient_checks(model, fs; probes_count)
    benchmark = l08_gradient_benchmark(gradient)
    benchmark.allocations == 0 || error("compiled gradient allocated $(benchmark.allocations) bytes")
    scaling = l08_instruction_scaling(model, fs, l08_duplicate_features(fs))
    p = l08_logdensity_problem(model, fs)
    errors = Float64[]
    rng = Random.MersenneTwister(733)
    for _ in 1:8
        theta = p.θ .+ 0.5 .* Random.randn(rng, length(p.θ))
        values = l08_parameter_values(DynamicPPL.unflatten(p.vi, theta))
        reference = l08_reference_logjoint(model, fs, values)
        push!(errors, abs(p.f(theta) - reference) / max(abs(reference), 1.0))
    end
    maximum(errors) < 1e-11 || error("independent logjoint/prior Jacobian mismatch: $errors")
    extraction = l08_extraction_checks(model, fs, oos)
    future_rows = l08_filtration_checks(model, fs)
    return Dict{String,Any}(
        "parameters" => length(gradient.θ), "instructions" => gradient.instructions,
        "prior_only_teams" => get(fs.data, :goal_decomposition_prior_only_teams, String[]),
        "duplicate_instructions" => scaling.duplicate_instructions,
        "gradient_allocations" => benchmark.allocations, "gradient_milliseconds" => benchmark.milliseconds,
        "compiled_fresh" => gradient.compiled_fresh, "compiled_forward" => gradient.compiled_forward,
        "max_perturbed_fresh" => maximum(gradient.perturbed), "max_perturbed_forward" => maximum(gradient.perturbed_forward),
        "gradient_probes" => probes_count, "independent_logjoint_relative_error" => maximum(errors),
        "future_rows_perturbed" => future_rows,
        "extraction" => Dict(string(k) => v for (k, v) in pairs(extraction)))
end

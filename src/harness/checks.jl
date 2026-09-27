"A correctness gate failed. Diagnostics use recorded `review` rows instead."
struct HarnessCheckError <: Exception
    check::String
    detail::String
end
Base.showerror(io::IO, error::HarnessCheckError) =
    print(io, "harness check `", error.check, "` failed: ", error.detail)

_check_value(value) = value === nothing ? NamedTuple() : value

function _check_record(base, check, severity, status, value, detail)
    return merge(base, (;
        check = String(check), severity = String(severity), status = String(status),
        value = _check_value(value), detail = String(detail), at = now()))
end

"""
Hard checks a smoke stage must record as passes before `grid` will sample a recipe
(`has_passing_smoke`). A smoke recorded before a check existed does not satisfy it: re-smoke.
"""
const SMOKE_REQUIRED_CHECKS = ("gradient", "tape_allocation", "filtration", "latents",
                               "score_grid_coherence", "fit_parity")

"""
Most heap bytes one compiled-tape gradient may allocate before the `tape_allocation` smoke check
fails. The builder engine allocates 0; the allowance covers runtime bookkeeping only. A tracked
scalar in a fused broadcast costs O(rows) — tens of KB on any real fold — so the gap is wide.
"""
const TAPE_ALLOCATION_LIMIT_BYTES = 1024

"Run a hard check, append its durable row, and throw `HarnessCheckError` on failure."
function _run_hard_check!(records::AbstractVector, base, check::AbstractString, f::Function)
    try
        value = f()
        push!(records, _check_record(base, check, "hard", "pass", value, ""))
        return value
    catch error
        detail = sprint(showerror, error)
        push!(records, _check_record(base, check, "hard", "fail", NamedTuple(), detail))
        throw(HarnessCheckError(String(check), detail))
    end
end
_run_hard_check!(f::Function, records::AbstractVector, base, check::AbstractString) =
    _run_hard_check!(records, base, check, f)

"Run a non-blocking diagnostic. Failure is recorded with its requested severity."
function _run_diagnostic!(records::AbstractVector, base, check::AbstractString,
                          severity::AbstractString, f::Function)
    try
        value = f()
        push!(records, _check_record(base, check, severity, "pass", value, ""))
        return value
    catch error
        detail = sprint(showerror, error)
        push!(records, _check_record(base, check, severity, "fail", NamedTuple(), detail))
        return nothing
    end
end
_run_diagnostic!(f::Function, records::AbstractVector, base, check::AbstractString,
                 severity::AbstractString) =
    _run_diagnostic!(records, base, check, severity, f)

function _match_kickoffs(matches::AbstractDataFrame)
    has_hour = :match_hour in propertynames(matches)
    return Dict(Int(row.match_id) => DateTime(Date(row.match_date)) +
                Hour(has_hour ? Int(coalesce(row.match_hour, 0)) : 0)
                for row in eachrow(matches))
end

"Build scoped folds once so checks, sampling, and latent extraction share identical inputs."
function _fold_inputs(candidate::Candidate, raw_ds; stage::Symbol)
    ds = Data.apply_scope(raw_ds, candidate.scope)
    boundaries = Data.create_id_boundaries(ds, Data.ScopedWalkForwardCV(candidate.scope))
    isempty(boundaries) && error("$(candidate.name): scoped splitter produced no folds")
    indices = stage === :smoke ? unique([1, length(boundaries)]) : collect(eachindex(boundaries))
    selected = boundaries[indices]
    splitter = Data.ScopedWalkForwardCV(candidate.scope)
    feature_sets = Features.create_features(selected, ds, candidate.model, splitter)
    oos = Any[Data.get_next_matches(ds, feature_sets[i], splitter)
              for i in eachindex(feature_sets)]
    return (; ds, boundaries = selected, feature_sets, oos, source_indices = indices)
end

"No train/OOS overlap, a strict kickoff cutoff, and the declared scope is honoured."
function _filtration_check(candidate::Candidate, inputs)
    kickoff = _match_kickoffs(inputs.ds.matches)
    scope = candidate.scope
    cups = scope.cups === :senior_spfl_ties ? Set(Data.SCOTTISH_CUP_TOURNAMENTS) : Set{Int}()
    allowed_train = union(Set(scope.train_tournaments), cups)
    allowed_oos = union(Set(scope.target_tournaments), Set(scope.monitor_tournaments))
    tournament = Dict(Int(row.match_id) => Int(row.tournament_id)
                      for row in eachrow(inputs.ds.matches))
    n_train = 0
    n_oos = 0
    for i in eachindex(inputs.feature_sets)
        boundary = first(inputs.boundaries[i])
        train_ids = Int.(vcat(boundary.history_match_ids, boundary.target_match_ids))
        heldout = inputs.oos[i]
        oos_ids = heldout === nothing ? Int[] : Int.(heldout.match_id)
        isempty(intersect(Set(train_ids), Set(oos_ids))) || error(
            "fold $(inputs.source_indices[i]) has train/OOS overlap")
        all(id -> get(tournament, id, 0) in allowed_train, train_ids) || error(
            "fold $(inputs.source_indices[i]) fits a tournament outside the declared scope")
        all(id -> get(tournament, id, 0) in allowed_oos, oos_ids) || error(
            "fold $(inputs.source_indices[i]) holds out a tournament outside target/monitor")
        monitor_only = setdiff(Set(scope.monitor_tournaments), Set(scope.train_tournaments))
        all(id -> get(tournament, id, 0) ∉ monitor_only, train_ids) || error(
            "fold $(inputs.source_indices[i]) fits a monitor-only tournament")
        if !isempty(oos_ids)
            maximum(kickoff[id] for id in train_ids) < minimum(kickoff[id] for id in oos_ids) ||
                error("fold $(inputs.source_indices[i]) does not have a strict temporal cutoff")
        end
        n_train += length(train_ids)
        n_oos += length(oos_ids)
    end
    return (; folds = length(inputs.boundaries), fitted_rows = n_train, heldout_rows = n_oos)
end

_relative_error(left, right) = norm(left - right) / max(norm(left), norm(right), 1.0)

"Compiled ReverseDiff parity, including three replay points, plus tape telemetry."
function _gradient_audit(model, feature_set; replays::Int = 200, seed::Int = 20260911)
    fs = feature_set isa Tuple ? first(feature_set) : feature_set
    turing_model = Models.PreGame.build_turing_model(model, fs)
    Random.seed!(seed)
    varinfo = DynamicPPL.VarInfo(turing_model)
    turing_model(varinfo)
    theta = copy(varinfo[:])
    density = DynamicPPL.LogDensityFunction(turing_model)
    objective = values -> LogDensityProblems.logdensity(density, values)
    log_density = objective(theta)
    isfinite(log_density) || error("non-finite log density $log_density")

    raw_tape = ReverseDiff.GradientTape(objective, theta)
    tape = ReverseDiff.compile(raw_tape)
    compiled = similar(theta)
    ReverseDiff.gradient!(compiled, tape, theta)
    all(isfinite, compiled) || error("compiled gradient is non-finite")
    fresh = ReverseDiff.gradient(objective, theta)
    forward = ForwardDiff.gradient(objective, theta)
    compiled_fresh_error = _relative_error(compiled, fresh)
    reversediff_forward_error = _relative_error(compiled, forward)
    compiled_fresh_error <= 1.0e-8 || error(
        "compiled/fresh ReverseDiff relative error $compiled_fresh_error > 1e-8")
    reversediff_forward_error <= 1.0e-6 || error(
        "ReverseDiff/ForwardDiff relative error $reversediff_forward_error > 1e-6")

    worst_perturbed_error = 0.0
    coordinates = collect(eachindex(theta))
    for delta in (0.001, -0.002, 0.003)
        perturbed = theta .+ delta .* sin.(coordinates)
        replay = similar(perturbed)
        ReverseDiff.gradient!(replay, tape, perturbed)
        error = _relative_error(ReverseDiff.gradient(objective, perturbed), replay)
        worst_perturbed_error = max(worst_perturbed_error, error)
    end
    worst_perturbed_error <= 1.0e-8 || error(
        "compiled tape replay relative error $worst_perturbed_error > 1e-8")

    for _ in 1:20
        ReverseDiff.gradient!(compiled, tape, theta)
    end
    allocated_bytes = @allocated ReverseDiff.gradient!(compiled, tape, theta)
    best_ns = typemax(UInt64)
    for _ in 1:replays
        started = time_ns()
        ReverseDiff.gradient!(compiled, tape, theta)
        best_ns = min(best_ns, time_ns() - started)
    end
    return (; n_parameters = length(theta), tape_instructions = length(raw_tape.tape),
              tape_bytes = Base.summarysize(raw_tape), gradient_ms = Float64(best_ns) / 1.0e6,
              allocated_bytes, compiled_fresh_error, reversediff_forward_error,
              worst_perturbed_error, log_density)
end

"""
    _tape_allocation_check(model, feature_sets; max_bytes) -> NamedTuple

Bytes per compiled ReverseDiff gradient on every given fold, measured where NUTS evaluates it: at a
linked (unconstrained) prior draw, on the compiled tape, as the minimum over 20 warmed replays.
Fails above `max_bytes`, naming every allocating tape instruction.

Latency alone hides this defect. The W2 `td_base` gradient took 0.24 ms — well inside the AD guide's
latency bar — while allocating 432 KB, and at 16 threads the collector halved sampler throughput.
docs/turing_ad_performance_guide.md §10.5.
"""
function _tape_allocation_check(model, feature_sets;
                                max_bytes::Int = TAPE_ALLOCATION_LIMIT_BYTES,
                                seed::Int = 20260911)
    folds = NamedTuple[]
    failures = String[]
    for (i, entry) in enumerate(feature_sets)
        fs = entry isa Tuple ? first(entry) : entry
        turing_model = Models.PreGame.build_turing_model(model, fs)
        Random.seed!(seed)
        varinfo = DynamicPPL.link!!(DynamicPPL.VarInfo(turing_model), turing_model)
        theta = copy(varinfo[:])
        density = DynamicPPL.LogDensityFunction(
            turing_model, DynamicPPL.getlogjoint_internal, varinfo)
        objective = values -> LogDensityProblems.logdensity(density, values)
        raw_tape = ReverseDiff.GradientTape(objective, theta)
        tape = ReverseDiff.compile(raw_tape)
        bytes = compiled_gradient_bytes(tape, similar(theta), theta)
        push!(folds, (; fold = i, allocated_bytes = bytes,
                        tape_instructions = length(raw_tape.tape),
                        n_parameters = length(theta)))
        if bytes > max_bytes
            listing = join(("#$(row.index) $(row.bytes) B $(row.description)"
                            for row in tape_allocation_profile(raw_tape)), "; ")
            push!(failures, "fold $i allocates $bytes B per compiled gradient " *
                            "(limit $max_bytes B): " * (isempty(listing) ? "no single instruction allocates in isolation" : listing))
        end
    end
    isempty(failures) || error(join(failures, " | ") *
        " — a scalar inside a fused broadcast takes ReverseDiff's allocating " *
        "tracker_∇broadcast; see docs/turing_ad_performance_guide.md §10.5")
    return (; max_allocated_bytes = maximum(f.allocated_bytes for f in folds), max_bytes,
              folds = Tuple(folds))
end

"Finite, positive count latents; MCMC latents must also be non-degenerate."
function _latent_audit(fit; require_variance::Bool = true)
    latents = fit.latents
    latents isa Models.CountLatents || error(
        "latents are $(typeof(latents)); expected CountLatents")
    allunique(latents.match_ids) || error("duplicate held-out match IDs")
    all(isfinite, latents.λ_home) && all(isfinite, latents.λ_away) ||
        error("non-finite posterior rate draws")
    all(>(0.0), latents.λ_home) && all(>(0.0), latents.λ_away) ||
        error("non-positive posterior rate draws")
    means = vcat(vec(mean(latents.λ_home; dims = 2)),
                 vec(mean(latents.λ_away; dims = 2)))
    all(isfinite, means) || error("non-finite posterior rate mean")
    min_sd = NaN
    if require_variance
        variances = vcat(vec(var(latents.λ_home; dims = 2)),
                         vec(var(latents.λ_away; dims = 2)))
        all(isfinite, variances) || error("non-finite posterior rate variance")
        all(>(0.0), variances) || error("a fixture has zero posterior rate variance")
        min_sd = sqrt(minimum(variances))
    end
    return (; n_matches = Models.n_matches(latents), n_draws = Models.n_draws(latents),
              mean_lambda = mean(means), min_sd)
end

function _grid_diagnostics(latents; check_coherence::Bool)
    markets = (Data.Market1X2(), Data.MarketOverUnder(2.5), Data.MarketBTTS())
    tails = Float64[]
    workspace = Predictions.GridWorkspace(12)
    grid = Predictions.alloc_score_grid(latents, 12)
    for fixture in 1:Models.n_matches(latents)
        Predictions.compute_score_grid!(grid, workspace, latents, fixture)
        all(isfinite, grid) || error("fixture $fixture score grid is non-finite")
        prices = check_coherence ?
            [Predictions.price_market(grid, market) for market in markets] : nothing
        for draw in 1:Models.n_draws(latents)
            mass = sum(@view grid[:, :, draw])
            isfinite(mass) && 0.0 < mass <= 1.0 + 1.0e-12 || error(
                "fixture $fixture draw $draw has invalid grid mass $mass")
            push!(tails, max(0.0, 1.0 - mass))
            if check_coherence
                for (market, price) in zip(markets, prices)
                    values_for_draw = (selection[draw] for selection in values(price))
                    abs(sum(values_for_draw) - mass) <= 1.0e-12 || error(
                        "fixture $fixture $(nameof(typeof(market))) partition differs from grid mass")
                end
            end
        end
    end
    isempty(tails) && error("score-grid diagnostic received no draws")
    return (; max_tail_mass = maximum(tails), p99_tail_mass = quantile(tails, 0.99),
              n_fixture_draws = length(tails))
end

"Full target coverage is hard; monitor and Betfair coverage is diagnostic only."
function _target_coverage(candidate::Candidate, fit, inputs;
                          expected_folds::Union{Nothing,Int} = nothing,
                          expected_target::Union{Nothing,Int} = nothing)
    folds_needed = expected_folds === nothing ? length(inputs.boundaries) : expected_folds
    length(fit.folds) == folds_needed || error(
        "$(candidate.name) has $(length(fit.folds)) folds; expected $folds_needed")
    target_tournaments = Set(candidate.scope.target_tournaments)
    expected_ids = Set{Int}()
    for heldout in inputs.oos, row in eachrow(heldout)
        Int(row.tournament_id) in target_tournaments && push!(expected_ids, Int(row.match_id))
    end
    if expected_target !== nothing
        length(expected_ids) == expected_target || error(
            "splitter exposes $(length(expected_ids)) target fixtures; expected $expected_target")
    end
    latent_ids = Set(Int.(fit.latents.match_ids))
    missing_ids = setdiff(expected_ids, latent_ids)
    isempty(missing_ids) || error("latents miss $(length(missing_ids)) target fixtures")
    length(fit.latents.match_ids) == length(unique(fit.latents.match_ids)) ||
        error("latent fixture IDs are not unique")
    return (; folds = length(fit.folds), target_fixtures = length(expected_ids),
              all_latent_fixtures = length(latent_ids))
end

function _monitor_coverage(candidate::Candidate, fit, inputs, raw_ds)
    reference_ids = try
        Set(Int.(_betfair_closing_odds(raw_ds).match_id))
    catch
        Set{Int}()
    end
    latent_ids = Set(Int.(fit.latents.match_ids))
    rows = NamedTuple[]
    for tournament in candidate.scope.monitor_tournaments
        heldout_ids = Set(Int(row.match_id) for frame in inputs.oos for row in eachrow(frame)
                          if Int(row.tournament_id) == tournament)
        push!(rows, (; tournament, heldout = length(heldout_ids),
                      latent = length(intersect(heldout_ids, latent_ids)),
                      betfair_reference = length(intersect(heldout_ids, reference_ids))))
    end
    return (; tournaments = rows)
end

function _structural_equal(left, right)
    typeof(left) === typeof(right) || return false
    left isa Union{Nothing,Missing,Number,AbstractString,Symbol,Char,Date,DateTime} &&
        return isequal(left, right)
    left isa AbstractArray && return size(left) == size(right) &&
        all(_structural_equal(left[i], right[i]) for i in eachindex(left))
    left isa Tuple && return length(left) == length(right) &&
        all(_structural_equal(left[i], right[i]) for i in eachindex(left))
    left isa NamedTuple && return keys(left) == keys(right) &&
        all(_structural_equal(getproperty(left, key), getproperty(right, key)) for key in keys(left))
    fieldcount(typeof(left)) == 0 && return isequal(left, right)
    return all(_structural_equal(getfield(left, field), getfield(right, field))
               for field in fieldnames(typeof(left)))
end

"Exact chain, latent, config and diagnostics parity after a database round trip."
function _fit_parity(original, recovered)
    _structural_equal(original.config, recovered.config) || error("FitConfig changed on reload")
    length(original.folds) == length(recovered.folds) || error("fold count changed on reload")
    for (left, right) in zip(original.folds, recovered.folds)
        left.fold == right.fold || error("fold index changed on reload")
        Array(left.chain) == Array(right.chain) || error("chain changed on fold $(left.fold)")
    end
    _structural_equal(original.diagnostics, recovered.diagnostics) ||
        error("convergence diagnostics changed on reload")
    left = original.latents
    right = recovered.latents
    left.match_ids == right.match_ids || error("latent match IDs changed on reload")
    left.λ_home == right.λ_home || error("home-rate draws changed on reload")
    left.λ_away == right.λ_away || error("away-rate draws changed on reload")
    left.observation_params == right.observation_params ||
        error("observation draws changed on reload")
    return (; folds = length(original.folds), fixtures = Models.n_matches(left),
              draws = Models.n_draws(left))
end

function _convergence_diagnostic(fit)
    summary = Training.audit_convergence(fit;
        thresholds = Training.ConvergenceThresholds(),
        max_depth = hasproperty(fit.config.sampler, :max_depth) ? fit.config.sampler.max_depth : 10)
    value = (; passed = summary.passed, max_rhat = summary.max_rhat,
              min_ess_bulk = summary.min_ess_bulk, min_ess_tail = summary.min_ess_tail,
              divergences = summary.n_divergent, divergence_rate = summary.divergence_rate,
              min_bfmi = summary.min_bfmi, treedepth_rate = summary.treedepth_rate,
              failures = summary.failures, abstained = summary.abstained)
    summary.passed || error(join(summary.failures, "; ") *
                            (isempty(summary.abstained) ? "" :
                             "; abstained: " * join(summary.abstained, ", ")))
    return value
end

function _thin_chain(chain::MCMCChains.Chains, stride::Int)
    return MCMCChains.Chains(
        parent(chain.value)[1:stride:end, :, :],
        names(chain),
        Dict(:parameters => names(chain, :parameters),
             :internals => names(chain, :internals));
        start = 1,
    )
end

function thin_for_persistence(fit::Training.Fit, inputs, stride::Int)
    stride >= 1 || error("persist stride must be ≥ 1; got $stride")
    stride == 1 && return fit
    folds = Training.FoldFit[Training.FoldFit(f.fold, _thin_chain(f.chain, stride), f.meta)
                             for f in fit.folds]
    latents, note = Training.extract_run_latents(fit.config.model, folds, inputs.oos, inputs.feature_sets)
    latents === nothing && error("thinned latent extraction failed: $note")
    Set(latents.match_ids) == Set(fit.latents.match_ids) || error(
        "thinned latents cover a different fixture set than the full-chain latents")
    return Training.Fit(fit.config, folds, latents, fit.diagnostics, fit.metadata, fit.save_path)
end


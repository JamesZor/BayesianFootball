# Version-1 experiment scorecard. The proper-score helpers are a direct graduation of
# current_development/grw_player_hybrid/l02_evaluation.jl.

const SCORECARD_VERSION = "v1"
const SCORE_REFERENCE = "devigged_betfair_twa(-20,0]"
const SCORE_MARKETS = ("all", "1X2", "OU2.5", "BTTS")
const SCORE_METRICS = Evaluation.AbstractScoringRule[
    Evaluation.LogLoss(), Evaluation.CRPS(), Evaluation.PredictionScore()]

"An immutable address for one scored model run."
struct RunRef
    label::String
    experiment::String
    run_id::UUID
    role::Symbol
    function RunRef(label::AbstractString, experiment::AbstractString, run_id,
                    role::Union{Symbol,AbstractString} = :candidate)
        parsed_role = Symbol(role)
        parsed_role in (:control, :candidate) || throw(ArgumentError(
            "RunRef role must be :control or :candidate; got :$parsed_role"))
        return new(String(label), String(experiment), UUID(string(run_id)), parsed_role)
    end
end

function _betfair_closing_odds(ds)
    raw = Data.summarize_odds(ds.betfair_odds, Data.TWAEstimator(); window = (-20.0, 0.0))
    odds = DataFrame(
        match_id = Int.(raw.match_id), market_name = String.(raw.market_name),
        market_line = Float64.(raw.market_line), selection = Symbol.(raw.selection),
        odds_close = Float64.(raw.odds))
    filter!(r -> isfinite(r.odds_close) && r.odds_close > 1.0, odds)
    odds.prob_implied_close = 1.0 ./ odds.odds_close
    transform!(groupby(odds, [:match_id, :market_name, :market_line]),
               :prob_implied_close => (p -> p ./ sum(p)) => :prob_fair_close)
    winners = unique(select(ds.odds, :match_id, :market_name, :market_line,
                            :selection, :is_winner))
    odds = leftjoin(odds, winners;
                    on = [:match_id, :market_name, :market_line, :selection])
    sort!(odds, [:match_id, :market_name, :market_line, :selection])
    return odds
end

function _season_panel(ds, fit, seasons::AbstractVector{<:AbstractString})
    wanted = Set(String.(seasons))
    attrs = Dict(Int(r.match_id) => (String(r.season), Int(r.tournament_id))
                 for r in eachrow(ds.matches))
    return sort!(Int[m for m in Models.latent_match_ids(fit.latents)
                     if (a = get(attrs, Int(m), nothing); a !== nothing &&
                         a[1] in wanted && a[2] in (56, 57))])
end

function _restrict(fit, panel)
    lat = Calibration.restrict_latents(fit.latents, panel)
    Models.n_matches(lat) == length(panel) || error(
        "restricted container holds $(Models.n_matches(lat)) of $(length(panel)) fixtures")
    return Training.Fit(fit.config, fit.folds, lat, fit.diagnostics, fit.metadata, fit.save_path)
end

function _family(market_name::AbstractString, line::Real)
    market = lowercase(market_name)
    market == "1x2" && return "1X2"
    occursin("btts", market) && return "BTTS"
    (occursin("over", market) || occursin("under", market) || occursin("total", market)) &&
        return "OU" * string(line)
    return String(market_name)
end

function _family_selections(odds::AbstractDataFrame)
    out = Dict{String,Vector{Symbol}}()
    for r in eachrow(unique(select(odds, :market_name, :market_line, :selection)))
        family = _family(r.market_name, r.market_line)
        family in ("1X2", "OU2.5", "BTTS") || continue
        push!(get!(out, family, Symbol[]), Symbol(r.selection))
    end
    foreach(v -> sort!(unique!(v)), values(out))
    return out
end

_context(fit, odds, ds) = Evaluation.build_evaluation_context(
    Evaluation.fit_latents(fit), odds, ds.matches, SCORE_METRICS; threaded = true)

"Direct port of `gph_scores`; retained as one path so the r04 reproduction is exact."
function _scores(label::AbstractString, ctx, families)
    rows = NamedTuple[]
    scopes = vcat([("all", nothing)],
                  [(family, families[family]) for family in ("1X2", "OU2.5", "BTTS")
                   if haskey(families, family)])
    for (scope, selections) in scopes
        score = Evaluation.evaluate_predictions(ctx; selections, n_bins = 10)
        push!(rows, (; model = String(label), scope, n_obs = score.model.n_obs,
            logloss = score.model.logloss, market_logloss = score.market.logloss,
            brier = score.model.brier, market_brier = score.market.brier,
            ece = score.model.ece, market_ece = score.market.ece,
            mce = score.model.mce, market_mce = score.market.mce,
            rps = score.model.rps, market_rps = score.market.rps))
    end
    return rows
end

function _observation_frame(label::AbstractString, ctx, odds::AbstractDataFrame)
    rows = Evaluation.evaluation_rows(ctx)
    family_of = Dict((Int(r.match_id), Symbol(r.selection)) =>
                     _family(r.market_name, r.market_line) for r in eachrow(odds))
    frame = DataFrame(
        match_id = Int[r.match_id for r in rows],
        selection = Symbol[r.selection for r in rows],
        p_model = Float64[r.model_prob for r in rows],
        p_market = Float64[r.market_prob for r in rows],
        y = Float64[Evaluation.row_y(r) for r in rows])
    frame.family = [get(family_of, (m, s), "other")
                    for (m, s) in zip(frame.match_id, frame.selection)]
    frame.ll_model = Evaluation.calc_logloss.(frame.p_model, frame.y)
    frame.ll_market = Evaluation.calc_logloss.(frame.p_market, frame.y)
    frame.model .= String(label)
    return frame
end

"Fixture-clustered paired bootstrap, unchanged from `gph_paired_bootstrap`."
function _paired_bootstrap(a::AbstractDataFrame, b; B::Int = 10_000,
                           seed::Int = 20260911,
                           family::Union{Nothing,String} = nothing)
    selected = family === nothing ? a : a[a.family .== family, :]
    joined = if b === :market
        DataFrame(match_id = selected.match_id,
                  d = selected.ll_model .- selected.ll_market)
    else
        comparison = family === nothing ? b : b[b.family .== family, :]
        pairs = innerjoin(select(selected, :match_id, :selection, :ll_model => :ll_a),
                          select(comparison, :match_id, :selection, :ll_model => :ll_b);
                          on = [:match_id, :selection])
        DataFrame(match_id = pairs.match_id, d = pairs.ll_a .- pairs.ll_b)
    end
    nrow(joined) == 0 && return (; n_obs = 0, n_fixtures = 0, delta = NaN,
        lo = NaN, hi = NaN, p_negative = NaN)
    groups = combine(groupby(joined, :match_id), :d => sum => :s, nrow => :n)
    sums = groups.s
    counts = Float64.(groups.n)
    n_groups = length(sums)
    rng = MersenneTwister(seed)
    stats = Vector{Float64}(undef, B)
    indexes = Vector{Int}(undef, n_groups)
    @inbounds for k in 1:B
        rand!(rng, indexes, 1:n_groups)
        total = 0.0
        n = 0.0
        for i in indexes
            total += sums[i]
            n += counts[i]
        end
        stats[k] = total / n
    end
    return (; n_obs = nrow(joined), n_fixtures = n_groups,
        delta = sum(sums) / sum(counts), lo = quantile(stats, 0.025),
        hi = quantile(stats, 0.975), p_negative = mean(stats .< 0.0))
end

function _market_supremacy(odds, ids)
    wanted = Set{Int}(Int.(ids))
    x12 = filter(r -> lowercase(String(r.market_name)) == "1x2" &&
                     Int(r.match_id) in wanted, odds)
    rates = Calibration.invert_market_rates(x12; match_ids = sort!(collect(wanted)))
    return Dict(mid => log(fit.lambda_home) - log(fit.lambda_away)
                for (mid, fit) in rates if fit.accepted)
end

function _model_supremacy(latents)
    ids = Models.latent_match_ids(latents)
    return Dict(Int(ids[i]) => mean(log.(view(latents.λ_home, i, :)) .-
                                    log.(view(latents.λ_away, i, :)))
                for i in eachindex(ids))
end

function _slopes(model_sup, market_sup, ids)
    common = sort!(Int[m for m in ids if haskey(model_sup, Int(m)) &&
                                      haskey(market_sup, Int(m))])
    length(common) >= 2 || return (; compression_slope = NaN,
                                    model_on_market_slope = NaN,
                                    n_fixtures = length(common))
    model = Float64[model_sup[m] for m in common]
    market = Float64[market_sup[m] for m in common]
    vm = var(model)
    vk = var(market)
    return (; compression_slope = iszero(vm) ? NaN : cov(model, market) / vm,
              model_on_market_slope = iszero(vk) ? NaN : cov(market, model) / vk,
              n_fixtures = length(common))
end

"Market-on-model and model-on-market log-rate supremacy slopes."
function compression_slopes(latents, odds, ids)
    return _slopes(_model_supremacy(latents), _market_supremacy(odds, ids), ids)
end

function _rps(frame::AbstractDataFrame, source::Symbol)
    x12 = frame[frame.family .== "1X2", :]
    isempty(x12) && return NaN
    by_match = groupby(x12, :match_id)
    values = Float64[]
    for group in by_match
        length(group.match_id) == 3 || continue
        probs = Float64[]
        ys = Float64[]
        for selection in (:home, :draw, :away)
            index = findfirst(==(selection), group.selection)
            index === nothing && break
            push!(probs, source === :market ? group.p_market[index] : group.p_model[index])
            push!(ys, group.y[index])
        end
        length(probs) == 3 || continue
        push!(values, ((probs[1] - ys[1])^2 +
                       (probs[1] + probs[2] - ys[1] - ys[2])^2) / 2)
    end
    return isempty(values) ? NaN : mean(values)
end

function _observation_scores(frame::AbstractDataFrame, market::String)
    rows = market == "all" ? frame : frame[frame.family .== market, :]
    n = nrow(rows)
    n == 0 && return (; n_obs = 0, n_fixtures = 0,
        logloss = NaN, market_logloss = NaN, brier = NaN, market_brier = NaN,
        ece = NaN, market_ece = NaN, mce = NaN, market_mce = NaN,
        rps = NaN, market_rps = NaN)
    function calibration(prob)
        bins = clamp.(floor.(Int, prob .* 10) .+ 1, 1, 10)
        gaps = Float64[]
        counts = Int[]
        for bin in 1:10
            mask = bins .== bin
            any(mask) || continue
            push!(counts, count(mask))
            push!(gaps, abs(mean(rows.y[mask]) - mean(prob[mask])))
        end
        return sum(counts .* gaps) / sum(counts), maximum(gaps)
    end
    ece, mce = calibration(rows.p_model)
    market_ece, market_mce = calibration(rows.p_market)
    return (; n_obs = n, n_fixtures = length(unique(rows.match_id)),
        logloss = mean(rows.ll_model), market_logloss = mean(rows.ll_market),
        brier = mean((rows.p_model .- rows.y).^2),
        market_brier = mean((rows.p_market .- rows.y).^2),
        ece, market_ece, mce, market_mce,
        rps = _rps(frame, :model), market_rps = _rps(frame, :market))
end

function transition_bias_pp(observations::AbstractDataFrame, ds, clubs)
    isempty(clubs) && return (; value = NaN, n_obs = 0, n_fixtures = 0)
    fixtures = Dict(Int(r.match_id) => (String(r.home_team), String(r.away_team))
                    for r in eachrow(ds.matches))
    deltas = Float64[]
    used = Set{Int}()
    for (mid, transitioned) in clubs
        teams = get(fixtures, mid, nothing)
        teams === nothing && continue
        for club in transitioned
            selection = club == teams[1] ? :home : club == teams[2] ? :away : nothing
            selection === nothing && continue
            row = filter(r -> r.match_id == mid && r.selection === selection &&
                              r.family == "1X2", observations)
            nrow(row) == 1 || continue
            push!(deltas, row.p_model[1] - row.p_market[1])
            push!(used, mid)
        end
    end
    return (; value = isempty(deltas) ? NaN : 100 * mean(deltas),
              n_obs = length(deltas), n_fixtures = length(used))
end

function _subsets(ds, target_ids, odds, tiers)
    tournament = Dict(Int(r.match_id) => Int(r.tournament_id) for r in eachrow(ds.matches))
    subsets = Dict{String,Set{Int}}(
        "target" => Set{Int}(target_ids),
        "t56" => Set{Int}(m for m in target_ids if get(tournament, m, 0) == 56),
        "t57" => Set{Int}(m for m in target_ids if get(tournament, m, 0) == 57),
        "favourites" => intersect(Set{Int}(target_ids), favourites(odds)))
    transitions = Dict{String,Any}()
    for direction in HARNESS_DIRECTIONS, first_n in (10, 20)
        label = "transition_$(direction)_first$(first_n)"
        cohort = transition_cohort(ds, target_ids, tiers; first_n, direction)
        subsets[label] = cohort.match_ids
        transitions[label] = cohort
    end
    return subsets, transitions
end

_metric_pairs(score) = (("logloss", score.logloss),
    ("market_logloss", score.market_logloss), ("brier", score.brier),
    ("market_brier", score.market_brier), ("ece", score.ece),
    ("market_ece", score.market_ece), ("mce", score.mce),
    ("market_mce", score.market_mce), ("rps", score.rps),
    ("market_rps", score.market_rps))

function _score_row(ref, subset, market, metric, value, lo, hi, n_obs, n_fixtures)
    return (; run_id = ref.run_id, model = ref.label, stage = "grid",
        scorecard_version = SCORECARD_VERSION, subset = String(subset),
        market = String(market), metric = String(metric), value = Float64(value),
        lo = Float64(lo), hi = Float64(hi), n_obs = Int(n_obs),
        n_fixtures = Int(n_fixtures), reference = SCORE_REFERENCE)
end

function _score_one(ref::RunRef, fit, ds, tiers; target_seasons, bootstrap_B::Int,
                    control_observations = nothing, expected_panel = nothing)
    panel = _season_panel(ds, fit, target_seasons)
    isempty(panel) && error("$(ref.label): no target fixtures in $(target_seasons)")
    if expected_panel !== nothing && Set(panel) != Set(expected_panel)
        missing_ids = setdiff(Set(expected_panel), Set(panel))
        extra_ids = setdiff(Set(panel), Set(expected_panel))
        error("$(ref.label) covers $(length(panel)) target fixtures, not the control's " *
              "$(length(expected_panel)); missing=$(length(missing_ids)), extra=$(length(extra_ids))")
    end
    restricted = _restrict(fit, panel)
    odds = _betfair_closing_odds(ds)
    families = _family_selections(odds)
    context = _context(restricted, odds, ds)
    observations = _observation_frame(ref.label, context, odds)
    delta_control = control_observations === :self ? observations : control_observations
    subsets, transitions = _subsets(ds, panel, odds, tiers)
    market_sup = _market_supremacy(odds, panel)
    model_sup = _model_supremacy(restricted.latents)

    exact_target = Dict(r.scope => r for r in _scores(ref.label, context, families))
    rows = NamedTuple[]
    for subset in sort!(collect(keys(subsets)))
        ids = subsets[subset]
        subset_obs = observations[in.(observations.match_id, Ref(ids)), :]
        for market in SCORE_MARKETS
            score = subset == "target" && haskey(exact_target, market) ?
                    exact_target[market] : _observation_scores(subset_obs, market)
            n_fixtures = market == "all" ? length(unique(subset_obs.match_id)) :
                length(unique(subset_obs.match_id[subset_obs.family .== market]))
            for (metric, value) in _metric_pairs(score)
                push!(rows, _score_row(ref, subset, market, metric, value, NaN, NaN,
                                       score.n_obs, n_fixtures))
            end
            if delta_control !== nothing
                control_subset = delta_control[
                    in.(delta_control.match_id, Ref(ids)), :]
                family = market == "all" ? nothing : market
                boot = _paired_bootstrap(subset_obs, control_subset;
                                         B = bootstrap_B, family)
                push!(rows, _score_row(ref, subset, market,
                    "delta_logloss_vs_control", boot.delta, boot.lo, boot.hi,
                    boot.n_obs, boot.n_fixtures))
            end
        end
        slopes = _slopes(model_sup, market_sup, ids)
        push!(rows, _score_row(ref, subset, "1X2", "compression_slope",
            slopes.compression_slope, NaN, NaN, slopes.n_fixtures, slopes.n_fixtures))
        push!(rows, _score_row(ref, subset, "1X2", "model_on_market_slope",
            slopes.model_on_market_slope, NaN, NaN, slopes.n_fixtures, slopes.n_fixtures))
        if haskey(transitions, subset)
            bias = transition_bias_pp(subset_obs, ds, transitions[subset].clubs)
            push!(rows, _score_row(ref, subset, "1X2", "transition_bias_pp",
                bias.value, NaN, NaN, bias.n_obs, bias.n_fixtures))
        end
    end
    return (; scores = DataFrame(rows), observations, panel)
end

function _control_ref(refs, control)
    if control isa RunRef
        return control
    elseif control !== nothing
        key = string(control)
        index = findfirst(r -> r.label == key || string(r.run_id) == key, refs)
        index === nothing && error("control '$key' is not in the run list")
        return refs[index]
    end
    controls = filter(r -> r.role === :control, refs)
    length(controls) == 1 || error("expected exactly one role=:control; found $(length(controls))")
    return only(controls)
end

"""Score in-memory fits under scorecard v1."""
function score_fits(fits::AbstractVector{<:Pair}; ds, tiers,
                    control = nothing,
                    target_seasons = ["24/25", "25/26"],
                    bootstrap_B::Int = 10_000)
    refs = RunRef[first(pair) for pair in fits]
    ctl = _control_ref(refs, control)
    lookup = Dict(first(pair).run_id => last(pair) for pair in fits)
    haskey(lookup, ctl.run_id) || error("control fit $(ctl.run_id) was not supplied")
    control_bundle = _score_one(ctl, lookup[ctl.run_id], ds, tiers;
                                target_seasons, bootstrap_B,
                                control_observations = :self)
    frames = DataFrame[control_bundle.scores]
    for ref in refs
        ref.run_id == ctl.run_id && continue
        push!(frames, _score_one(ref, lookup[ref.run_id], ds, tiers;
            target_seasons, bootstrap_B, expected_panel = control_bundle.panel,
            control_observations = control_bundle.observations).scores)
    end
    return vcat(frames...)
end

"""Load and score runs by UUID, releasing each large fit before loading the next."""
function score_runs(refs::AbstractVector{RunRef}; ds, tiers, control = nothing,
                    target_seasons = ["24/25", "25/26"],
                    bootstrap_B::Int = 10_000,
                    failures::Union{Nothing,AbstractVector} = nothing)
    ctl = _control_ref(refs, control)
    control_fit = Training.load_fit(Training.PostgresStorage(ctl.experiment), ctl.run_id)
    control_bundle = _score_one(ctl, control_fit, ds, tiers;
                                target_seasons, bootstrap_B,
                                control_observations = :self)
    control_fit = nothing
    GC.gc()

    frames = DataFrame[control_bundle.scores]
    for ref in refs
        ref.run_id == ctl.run_id && continue
        fit = nothing
        try
            fit = Training.load_fit(Training.PostgresStorage(ref.experiment), ref.run_id)
            bundle = _score_one(ref, fit, ds, tiers; target_seasons, bootstrap_B,
                                expected_panel = control_bundle.panel,
                                control_observations = control_bundle.observations)
            push!(frames, bundle.scores)
        catch error
            failures === nothing && rethrow()
            push!(failures, (; label = ref.label, experiment = ref.experiment,
                              run_id = ref.run_id,
                              reason = sprint(showerror, error)))
        finally
            fit = nothing
            GC.gc()
        end
    end
    return vcat(frames...)
end

function _headline(scores, run_id, subset, market, metric, field = :value)
    rows = filter(r -> string(r.run_id) == string(run_id) && r.subset == subset &&
                       r.market == market && r.metric == metric, scores)
    nrow(rows) == 0 && return NaN
    return Float64(rows[1, field])
end

"""Collapse the long score table into the committed Phase-1 headline leaderboard."""
function leaderboard(scores::AbstractDataFrame)
    rows = NamedTuple[]
    for group in groupby(scores, [:run_id, :model])
        run_id = first(group.run_id)
        base = (;
            run_id, model = first(group.model),
            target_logloss_all = _headline(scores, run_id, "target", "all", "logloss"),
            target_logloss_1x2 = _headline(scores, run_id, "target", "1X2", "logloss"),
            target_ece_all = _headline(scores, run_id, "target", "all", "ece"),
            compression_slope = _headline(scores, run_id, "target", "1X2", "compression_slope"),
            delta_vs_control = _headline(scores, run_id, "target", "all", "delta_logloss_vs_control"),
            delta_lo = _headline(scores, run_id, "target", "all", "delta_logloss_vs_control", :lo),
            delta_hi = _headline(scores, run_id, "target", "all", "delta_logloss_vs_control", :hi))
        extra_names = Symbol[]
        extra_values = Float64[]
        for direction in HARNESS_DIRECTIONS, first_n in (10, 20)
            subset = "transition_$(direction)_first$(first_n)"
            stem = "transition_$(direction)_first$(first_n)"
            push!(extra_names, Symbol(stem * "_logloss"), Symbol(stem * "_bias_pp"))
            push!(extra_values,
                  _headline(scores, run_id, subset, "all", "logloss"),
                  _headline(scores, run_id, subset, "1X2", "transition_bias_pp"))
        end
        extra = NamedTuple{Tuple(extra_names)}(Tuple(extra_values))
        push!(rows, merge(base, extra))
    end
    return sort!(DataFrame(rows), :target_logloss_all)
end

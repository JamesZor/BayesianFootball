# ==============================================================================
# Task 016 loader — arms, per-strike scores, and T011-correct staking
# ==============================================================================
#
# Definitions only. `r04`, `r06`, `r07`, `r08` execute. Nothing here samples: every posterior
# is loaded from `mcmc_experiments` by UUID (smile and spine rungs rebuilt from their chains,
# ticket T010).
#
# WHAT THIS ADDS TO TASK 015'S EVALUATION LAYER, and why each piece exists.
#
# Task 015's `l02_evaluation.jl` and `l04_portfolio_calibration.jl` are included whole and
# used unchanged — the panel, the fixture-clustered bootstrap, the Option B contract, the two
# calibrated-smile definitions, the sweep additions, the compression table. Their helpers are
# untyped in the model, so a `SpineAnchoredCountModel` flows through them. Four additions:
#
# 1. THE SIX-ARM LADDER (§1). Task 015's `gms_arms` looks every rung up in ONE experiment
#    namespace; this task's rungs live in three (Task 013's, Task 015's, this task's), so the
#    arms are built from `GSS_PINNED_RUNS` plus a name lookup for the two spine runs.
#
# 2. THE STRIKE LADDER 0.5 … 4.5 (§2). Task 015 scored O/U 1.5 and 3.5 beside 2.5. The spine's
#    known weakness is at the ENDS: β·(K−2) reproduces Task 015's φ at K = 1…4 within ±0.04 in
#    log φ but misses K = 0 by −0.066 (r02 measured φ₀ = 0.900 against the five-strike 0.843,
#    and φ₄ = 1.111 against 1.069). So O/U 0.5 and 4.5 are scored too, and a per-strike table
#    reports each arm's LogLoss and mean p_model − p_market line by line. A pooled score would
#    average exactly the effect this task needs to see. These scopes are reported separately and
#    never pooled into `all`, which stays Task 013's published basis so the reproduction gate
#    still means something.
#
# 3. T011-CORRECT STAKING (§4). Every portfolio call in Task 015 goes through
#    `Portfolio.build_books_reported`, which prices a smile container's O/U through λ_tot·φ(K)
#    and then sizes every stake off the un-smiled grid — the defect T011 records. `gss_build_books`
#    takes a `route`:
#
#      :grid        Task 015's path, verbatim. Keeps its rows reproducible as a gate.
#      :reweighted  `gss_build_books_reweighted` (l01 §8) — the anti-diagonal reweighted grid,
#                   so the Kelly solve and the BakerMcHale re-solves see the smile.
#
#    A count container is identical under both routes (T011 acceptance, asserted in r06), so the
#    route is a property of the smile arms only.
#
# 4. THE STAKE-SIDE GATE (§5). Task 015's `gms_smile_book_gate` proves the LEDGER'S `p_model` is
#    the smile price. That is the reported-price half of T011 and it passed in Task 015 while the
#    stakes were still off the grid. `gss_book_totals_gate` proves the other half from the books
#    themselves: the distribution the allocator solved on — `book.p_grid` — implies the smile's
#    totals CDF. Both are run on every smile ledger.
# ==============================================================================

if !isdefined(@__MODULE__, :SpineAnchoredCountModel)
    include(joinpath(@__DIR__, "l01_loader.jl"))
end
if !isdefined(@__MODULE__, :gms_option_b)
    include(joinpath(@__DIR__, "..", "grw_market_smile", "l04_portfolio_calibration.jl"))
end

# ==============================================================================
# 1. The six-arm ladder
# ==============================================================================

"""
    gss_arms(c) -> Vector{GMSArm}

The ladder in rung order: the four pinned runs by UUID, then this task's two spine runs looked
up by name in `c.experiment`. `GMSArm` is Task 015's type, so `gms_load_arm` consumes these.

`role` is `"pinned"` or `"candidate"`; r06 persists portfolios for the candidates only, since a
pinned run's portfolio belongs to the task that sampled it.
"""
function gss_arms(c::GMSConfig)
    arms = GMSArm[GMSArm(p.rung, p.experiment, p.run_id, "pinned") for p in GSS_PINNED_RUNS]
    db = PostgresStorage(c.experiment)
    for name in GSS_GRID_MODEL_NAMES
        run_id = gms_run_by_name(db, name)
        run_id === nothing && error("no completed run named $name in $(c.experiment) — run r02")
        push!(arms, GMSArm(name, c.experiment, run_id, "candidate"))
    end
    return arms
end

"The ladder's labels in rung order, for stable table ordering."
gss_arm_order() = String[p.rung for p in GSS_PINNED_RUNS]

"Does this arm carry a smile curve? Read from the loaded container, not from the name."
gss_is_smile(fit) = fit.latents isa SmileLatents

# ==============================================================================
# 2. Scopes and the strike ladder
# ==============================================================================

"""
`all` pools 1X2 + O/U 2.5 + BTTS — Task 013's published basis (2,899 rows on the 710-fixture
panel), which is what makes r04's reproduction gate a real check.
"""
const GSS_PRIMARY_MARKETS = Data.AbstractMarket[
    Data.Market1X2(), Data.MarketOverUnder(2.5), Data.MarketBTTS(),
]

"Every strike the spine parameterises except the pooled 2.5, scored but never pooled into `all`."
const GSS_SECONDARY_MARKETS = Data.AbstractMarket[
    Data.MarketOverUnder(0.5), Data.MarketOverUnder(1.5),
    Data.MarketOverUnder(3.5), Data.MarketOverUnder(4.5),
]

const GSS_SECONDARY_SCOPES = ["OU0.5", "OU1.5", "OU3.5", "OU4.5"]
const GSS_SCOPES = ["1X2", "OU2.5", "BTTS", "OU0.5", "OU1.5", "OU3.5", "OU4.5"]

"The O/U lines in strike order, for the per-strike table: K = 0…4."
const GSS_STRIKE_LINES = [0.5, 1.5, 2.5, 3.5, 4.5]

"""
    GSS_PAIRS

The contrasts, `(candidate, reference)`, fixed before scoring.

* spine − baseline               does the anchored spine help at all?
* spine − five-strike, SAME weight   H3's actual question: is one parameter as good as five?
* spine − supremacy, same weight     what the spine's totals pillar adds over supremacy alone
* spine@0.20 − spine@0.40            the weight grid within this task
"""
const GSS_PAIRS = [
    ("m05_joint_grw_smile_spine_w020", "m05_joint_grw_baseline"),
    ("m05_joint_grw_smile_spine_w040", "m05_joint_grw_baseline"),
    ("m05_joint_grw_smile_spine_w020", "m05_joint_grw_smile_supremacy_w020"),
    ("m05_joint_grw_smile_spine_w040", "m05_joint_grw_smile_supremacy_w040"),
    ("m05_joint_grw_smile_spine_w040", "m05_joint_grw_supremacy_w040"),
    ("m05_joint_grw_smile_spine_w020", "m05_joint_grw_smile_spine_w040"),
]

"Selections present in the book, per scope in `GSS_SCOPES`."
function gss_family_selections(odds::AbstractDataFrame)
    out = Dict{String,Vector{Symbol}}()
    for r in eachrow(unique(select(odds, :market_name, :market_line, :selection)))
        fam = gms_family(r.market_name, r.market_line)
        fam in GSS_SCOPES || continue
        push!(get!(out, fam, Symbol[]), r.selection)
    end
    for (k, v) in out
        out[k] = sort!(unique(v))
    end
    return out
end

"""
    gss_scores(label, primary_ctx, secondary_ctx, families) -> Vector{NamedTuple}

One row per scope. `all` is computed on the PRIMARY context only, so it is Task 013's basis;
each named scope is read from whichever context holds its market.
"""
function gss_scores(label, primary_ctx, secondary_ctx, families)
    rows = NamedTuple[]
    # Explicit element type: a literal first entry would fix the third slot as `Nothing` and
    # refuse the per-family `Vector{Symbol}` pushed after it (Task 015 r04 hit exactly this).
    scopes = Tuple{String,Any,Union{Nothing,Vector{Symbol}}}[("all", primary_ctx, nothing)]
    for f in GSS_SCOPES
        haskey(families, f) || continue
        push!(scopes, (f, f in GSS_SECONDARY_SCOPES ? secondary_ctx : primary_ctx, families[f]))
    end
    for (scope, ctx, sels) in scopes
        s = evaluate_predictions(ctx; selections = sels, n_bins = 10)
        push!(rows, (; model = String(label), scope, n_obs = s.model.n_obs,
                       logloss = s.model.logloss, market_logloss = s.market.logloss,
                       brier = s.model.brier, market_brier = s.market.brier,
                       ece = s.model.ece, market_ece = s.market.ece,
                       rps = s.model.rps, market_rps = s.market.rps))
    end
    return rows
end

"""
    gss_strike_table(frames, arms) -> DataFrame

Per arm and per O/U strike: rows scored, model and market LogLoss, and the mean signed gap
`p_model − p_market` on the UNDER selection.

This is where the spine's one-parameter restriction should show if it costs anything. The
five-strike model sets φ₀ = 0.843 and the spine's line forces 0.900, so on Under 0.5 the spine
prices a LOWER probability of the total staying under — a visible, directional difference that a
pooled totals score cannot show.
"""
function gss_strike_table(frames::AbstractDict, arms::AbstractVector{<:AbstractString})
    rows = NamedTuple[]
    for label in arms
        haskey(frames, label) || continue
        df = frames[label]
        for line in GSS_STRIKE_LINES
            fam = "OU" * string(line)
            # The selection symbol comes from the market type itself (`under_05` … `under_45`),
            # never rebuilt from the line here — one source of truth for the naming.
            under = Data.outcomes(Data.MarketOverUnder(line)).under
            sub = df[(df.family .== fam) .& (df.selection .== under), :]
            nrow(sub) == 0 && continue
            push!(rows, (; model = String(label), strike = line, K = Int(floor(line)),
                           n_obs = nrow(sub),
                           logloss = mean(sub.ll_model), market_logloss = mean(sub.ll_market),
                           delta_logloss = mean(sub.ll_model) - mean(sub.ll_market),
                           mean_p_model = mean(sub.p_model), mean_p_market = mean(sub.p_market),
                           mean_gap = mean(sub.p_model .- sub.p_market),
                           realised_under_rate = mean(sub.y)))
        end
    end
    return sort!(DataFrame(rows), [:strike, :model])
end

# ==============================================================================
# 3. Loading
# ==============================================================================

"""
    gss_load_arms(arms, ds; splitter, latent_dirs) -> Dict{String,Any}

Load every arm, trying each of `latent_dirs` for the file copy a detached smile run was saved
beside (Task 015's runs live under its own results tree, this task's under ours). Task 015's
`gms_load_arm` does the work: it refuses a non-converged or synthetic run, rebuilds a detached
panel from the persisted chains, and requires the rebuilt panel to equal the file copy when one
exists.
"""
function gss_load_arms(arms, ds; splitter, latent_dirs::Vector{String})
    out = Dict{String,Any}()
    for arm in arms
        dir = ""
        for candidate in latent_dirs
            isdir(joinpath(candidate, string(arm.run_id))) || continue
            dir = candidate
            break
        end
        t0 = time()
        out[arm.label] = gms_load_arm(arm, ds; splitter, latent_dir = isempty(dir) ? first(latent_dirs) : dir)
        @printf("  loaded %-36s run %s  %-13s  file copy %-5s  %.0f s\n",
                arm.label, arm.run_id, nameof(typeof(out[arm.label].latents)),
                isempty(dir) ? "none" : "yes", time() - t0)
    end
    return out
end

# ==============================================================================
# 4. Staking routes (ticket T011)
# ==============================================================================

"""
    gss_build_books(spec, source, odds, fixtures, panel; label, route) -> (books, report)

Build one already-restricted source's books over `panel`, refusing any skipped fixture.

`route = :reweighted` stakes a smile container off the anti-diagonal reweighted grid (the T011
fix); `route = :grid` is Task 015's production path verbatim. A `CountLatents` source takes the
production path under either value — reweighting is meaningless without a φ — so a count arm's
ledger is identical across routes by construction rather than by luck.
"""
function gss_build_books(spec, source, odds, fixtures, panel::Vector{Int};
                         label::AbstractString, route::Symbol = :reweighted)
    route in (:reweighted, :grid) || error("unknown staking route $route; use :reweighted or :grid")
    books, report = if route === :reweighted
        gss_build_books_reweighted(spec, source, odds, fixtures; require_converged = false, quiet = true)
    else
        GMS_PORTFOLIO.build_books_reported(spec, source, odds, fixtures;
                                           require_converged = false, quiet = true)
    end
    GMS_PORTFOLIO.n_skipped(report) == 0 || error(
        "$label ($route) skipped $(GMS_PORTFOLIO.n_skipped(report)) panel fixtures")
    length(books) == length(panel) || error(
        "$label ($route) built $(length(books)) books for $(length(panel)) fixtures")
    return books, report
end

"""
    gss_buildable_panel(spec, fits, odds, fixtures, panel; route) -> (keep, dropped_frame)

Task 015's `gms_buildable_panel`, through the chosen staking route: the panel every arm can
build a book for, with each drop named. The route can change which fixtures are buildable — a
reweighting refusal is a container property, not a quote property — so the panel is computed
under the same route the simulation will use.
"""
function gss_buildable_panel(spec, fits::AbstractDict, odds, fixtures, panel::Vector{Int};
                             route::Symbol = :reweighted)
    dropped = Dict{Int,String}()
    for (label, fit) in fits
        source = gms_restrict(fit, panel)
        _, report = if route === :reweighted
            gss_build_books_reweighted(spec, source, odds, fixtures; require_converged = false, quiet = true)
        else
            GMS_PORTFOLIO.build_books_reported(spec, source, odds, fixtures;
                                               require_converged = false, quiet = true)
        end
        for (ids, why) in ((report.skipped_no_fixture, "no fixture row"),
                           (report.skipped_unplayed, "unplayed"),
                           (report.skipped_no_quotes, "no quotes"),
                           (report.skipped_no_selections, "no usable selections"))
            for m in ids
                dropped[Int(m)] = why
            end
        end
        for (m, msg) in report.errored
            dropped[Int(m)] = "error ($label): " * msg
        end
    end
    keep = sort!(collect(setdiff(Set(panel), keys(dropped))))
    frame = DataFrame(match_id = sort!(collect(keys(dropped))))
    frame.reason = [dropped[m] for m in frame.match_id]
    return keep, frame
end

"""
    gss_simulate(spec, policy, fit, odds, fixtures, panel; label, route, B, seed)

Restrict, build through `route`, and stake. Returns `(result, restricted, books)`.
"""
function gss_simulate(spec, policy, fit, odds, fixtures, panel::Vector{Int};
                      label::AbstractString, route::Symbol = :reweighted,
                      B::Int = 4000, seed::Int = 1)
    restricted = gms_restrict(fit, panel)
    books, report = gss_build_books(spec, restricted, odds, fixtures, panel; label, route)
    result = gms_run_policy(policy, books, report; B = B, seed = seed)
    return result, restricted, books
end

# ==============================================================================
# 5. The stake-side gate (the half T011 says Task 015 could not check)
# ==============================================================================

"""
    gss_book_totals_gate(books, latents) -> NamedTuple

For every built book, the distribution the Kelly solve actually read — `book.p_grid`, the
posterior-mean scoreline grid — must imply the smile's totals CDF:

    Σ_{h+a ≤ K} p_grid(h, a)  ==  mean_s cdf(Poisson(λ_tot·φ_K), K),   K = 0 … n_strikes−1

Under the `:grid` route this fails by the size of the smile (that is T011); under `:reweighted`
it must hold to float precision. Reported for both, so the fix is measured rather than asserted.

`p_grid` is normalised by `_finish_book`; a reweighted draw already sums to 1, so the
normalisation is a no-op up to rounding and no tolerance is spent on it.
"""
function gss_book_totals_gate(books, l::SmileLatents)
    row_of = Dict(Int(m) => i for (i, m) in enumerate(l.match_ids))
    nd = n_draws(l)
    worst = 0.0
    worst_fixture = 0
    worst_K = -1
    per_strike = zeros(Float64, length(l.strikes))
    n = 0
    for b in books
        i = get(row_of, b.m_id, 0)
        i == 0 && continue
        M = isqrt(length(b.p_grid))
        M * M == length(b.p_grid) || error("book $(b.m_id) has a non-square p_grid of length $(length(b.p_grid))")
        P = reshape(b.p_grid, M, M)
        for s in eachindex(l.strikes)
            K = s - 1
            implied = 0.0
            for c in 1:M, r in 1:M
                (r - 1) + (c - 1) <= K && (implied += P[r, c])
            end
            reference = mean(cdf(Poisson(l.λ_tot[i, k] * l.φ[i, s, k]), K) for k in 1:nd)
            gap = abs(implied - reference)
            per_strike[s] = max(per_strike[s], gap)
            if gap > worst
                worst = gap
                worst_fixture = b.m_id
                worst_K = K
            end
        end
        n += 1
    end
    return (; n_books = n, max_abs_gap = worst, worst_fixture, worst_K,
              per_strike_gap = join([@sprintf("%.1e", g) for g in per_strike], "/"))
end

"""
    gss_route_contrast(label, grid_result, reweighted_result) -> NamedTuple

What the T011 fix did to one arm's ledger: shared and exclusive bets, and the ROI and bankroll
difference between staking off the plain grid and off the reweighted one. Task 015 r07 measured
zero exclusive bets and ROI equal to 1e-14 between a smile container and its φ-stripped twin;
this row is the same comparison made the other way round, and a non-zero answer here is the
defect being fixed.
"""
function gss_route_contrast(label::AbstractString, grid_result, reweighted_result)
    a = DataFrame(reweighted_result.trajectory.bets)
    b = DataFrame(grid_result.trajectory.bets)
    ka = Set(gms_bet_key(r) for r in eachrow(a))
    kb = Set(gms_bet_key(r) for r in eachrow(b))
    shared = intersect(ka, kb)
    stake_gap = 0.0
    if !isempty(shared)
        sa = Dict(gms_bet_key(r) => r.stake for r in eachrow(a))
        sb = Dict(gms_bet_key(r) => r.stake for r in eachrow(b))
        stake_gap = maximum(abs(sa[k] - sb[k]) for k in shared)
    end
    return (; model = String(label),
              n_bets_grid = nrow(b), n_bets_reweighted = nrow(a),
              n_shared = length(shared),
              n_only_grid = length(setdiff(kb, ka)), n_only_reweighted = length(setdiff(ka, kb)),
              max_shared_stake_gap = stake_gap,
              roi_grid_pct = grid_result.summary.roi, roi_reweighted_pct = reweighted_result.summary.roi,
              delta_roi_pp = reweighted_result.summary.roi - grid_result.summary.roi,
              return_grid_pct = grid_result.summary.total_return_pct,
              return_reweighted_pct = reweighted_result.summary.total_return_pct,
              delta_return_pp = reweighted_result.summary.total_return_pct -
                                grid_result.summary.total_return_pct)
end

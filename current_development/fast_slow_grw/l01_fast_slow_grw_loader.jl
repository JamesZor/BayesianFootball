# ==============================================================================
# TODO 021 loader — Fast & Slow GRW models and posterior draw mixtures
# ==============================================================================
#
# Definitions only. `r01_fast_slow_smoke.jl`, `r02_…` and `r03_…` execute.
#
# THE FOUR ARMS. One observation, one splitter, one sampler; only the walk's
# priors move:
#
#   m01_poisson_grw_tight               MultiScaleGRW() defaults             (the handrail)
#   m02_poisson_grw_loose_var           every σ prior scale × 2.5            (loose A)
#   m03_poisson_grw_loose_tdist         z₀, zₛ ~ TDist(4), σ priors default  (loose B)
#   m04_poisson_grw_loose_fixed_spread  σ₀ pinned at 0.48 (att and def)      (loose C)
#
# WHY m04. The Stage 1 smoke showed the data pins σ₀: a 2.5× wider prior moved the
# posterior attack σ₀ only 0.194 → 0.211. The market-implied spread is ~2.43× the
# tight arm's (0.194 × 2.43 ≈ 0.47, 0.202 × 2.43 ≈ 0.49), so m04 anchors σ₀ at 0.48
# with a truncated Normal(0.48, 0.01) — NUTS cannot sample a fixed site, and a 2%
# prior sd leaves the data no room to pull it back.
#
# COMBINATION = DRAW CONCATENATION. The tight and loose posteriors are combined by
# concatenating posterior draws in ratio ρ (`mixture_latents`), NOT by averaging
# rates element-wise. Every retained draw is one coherent (λ_h, λ_a) score grid, so
# 1X2 / totals / BTTS stay partitions of one grid per draw; at the fixture level
# the mixture prices (1 − ρ) P_tight + ρ P_loose, up to subsampling noise.
#
# `m01` is, component for component, Task 013's `m00_baseline_grw` (run
# `158d2a80-7ea3-4d6c-b3ab-be62bcf1bc11`, 40 folds / 710 OOS), re-sampled here
# under its own name and this package's sampler budget.
#
# REUSE. The fold/feature construction, gradient audit, convergence thresholds,
# save/load round-trip and markdown helpers are Task 013's (`gph_*`), included
# verbatim rather than copied: that loader is definitions-only and its functions
# are the ones the 40-fold GRW grids were validated with.
#
# CAVEAT ON `m03`. TDist(4) has variance 2, so it widens the BODY of the z
# distribution by √2 as well as fattening the tails. A spread gain from `m03` is
# therefore not purely a tail effect; the posterior σ₀ is reported so the two can
# be told apart.
# ==============================================================================

include(joinpath(@__DIR__, "..", "grw_player_hybrid", "l01_loader.jl"))

const FSG_CAL = BayesianFootball.Calibration

# ==============================================================================
# 1. Experiment configuration
# ==============================================================================

Base.@kwdef struct FSGConfig
    experiment::String = "fast_slow_grw_scottish_lower"
    smoke_experiment::String = "smoke_fast_slow_grw"
    save_root::String = joinpath(@__DIR__, "results")

    target_seasons::Vector{String} = ["24/25", "25/26"]
    expected_folds::Int = 40
    expected_oos::Int = 710

    # Spread across the cohort: fold 1 has no target steps (tapes the
    # no-target branch), folds 20 and 40 sit deep in each season.
    smoke_folds::Vector{Int} = [1, 20, 40]
    smoke_samples::Int = 400
    smoke_warmup::Int = 400
    smoke_chains::Int = 4

    samples::Int = 800
    warmup::Int = 800
    chains::Int = 4
    accept_rate::Float64 = 0.80
    max_depth::Int = 10
    max_concurrent_tasks::Int = 16

    gradient_replays::Int = 200
    # 3,200 retained draws per fold; the artefact keeps every 2nd (see GPHConfig).
    persist_stride::Int = 2

    max_rhat::Float64 = 1.05
    min_ess::Float64 = 200.0
    max_divergence_rate::Float64 = 0.001
    min_bfmi::Float64 = 0.30
    max_treedepth_rate::Float64 = 0.05

    # Mixture ratios: share of draws taken from the loose arm.
    rhos::Vector{Float64} = [0.0, 0.25, 0.50, 0.75, 1.0]
end

"The `GPHConfig` view of an `FSGConfig`, for the reused `gph_*` helpers."
fsg_gph_config(c::FSGConfig) = GPHConfig(
    experiment = c.experiment,
    smoke_experiment = c.smoke_experiment,
    save_root = c.save_root,
    target_seasons = c.target_seasons,
    smoke_samples = c.smoke_samples,
    smoke_warmup = c.smoke_warmup,
    smoke_chains = c.smoke_chains,
    samples = c.samples,
    warmup = c.warmup,
    chains = c.chains,
    accept_rate = c.accept_rate,
    max_depth = c.max_depth,
    max_concurrent_tasks = c.max_concurrent_tasks,
    gradient_replays = c.gradient_replays,
    persist_stride = c.persist_stride,
    max_rhat = c.max_rhat,
    min_ess = c.min_ess,
    max_divergence_rate = c.max_divergence_rate,
    min_bfmi = c.min_bfmi,
    max_treedepth_rate = c.max_treedepth_rate,
)

const FSG_TIGHT = "m01_poisson_grw_tight"
const FSG_LOOSE_VAR = "m02_poisson_grw_loose_var"
const FSG_LOOSE_T = "m03_poisson_grw_loose_tdist"
const FSG_LOOSE_FIXED = "m04_poisson_grw_loose_fixed_spread"
const FSG_MODEL_NAMES = [FSG_TIGHT, FSG_LOOSE_VAR, FSG_LOOSE_T, FSG_LOOSE_FIXED]
const FSG_LOOSE_NAMES = [FSG_LOOSE_VAR, FSG_LOOSE_T, FSG_LOOSE_FIXED]

const FSG_DESCRIPTIONS = Dict(
    FSG_TIGHT => "Poisson MultiScaleGRW at graduated default priors (the handrail).",
    FSG_LOOSE_VAR => "Poisson MultiScaleGRW with every σ prior scale widened 2.5x (loose A).",
    FSG_LOOSE_T => "Poisson MultiScaleGRW with TDist(4) level and season innovations (loose B).",
    FSG_LOOSE_FIXED => "Poisson MultiScaleGRW with level scale σ₀ pinned at 0.48, the market-implied 2.43x spread (loose C).",
)

const FSG_TAGS = ["scottish-lower", "24/25", "25/26", "multiscale-grw",
                  "fast-slow", "todo021", "reversediff"]

# ==============================================================================
# 2. Models
# ==============================================================================

fsg_tight_dynamics() = MultiScaleGRW()

"Every scale prior × 2.5 (Gamma(k, θ) → Gamma(k, 2.5θ) scales its mean by 2.5)."
fsg_loose_var_dynamics() = MultiScaleGRW(
    α_σ₀ = Gamma(2, 0.150), α_σₛ = Gamma(2, 0.075), α_σₖ = Gamma(2, 0.0375),
    β_σ₀ = Gamma(2, 0.250), β_σₛ = Gamma(2, 0.1375), β_σₖ = Gamma(2, 0.030),
)

fsg_loose_t_dynamics() = MultiScaleGRW(z₀ = TDist(4.0), zₛ = TDist(4.0))

"The anchored spread: σ₀ at 0.48 ± 0.01 for attack and defence; steps at defaults."
const FSG_FIXED_SIGMA0 = 0.48
fsg_loose_fixed_dynamics() = MultiScaleGRW(
    α_σ₀ = truncated(Normal(FSG_FIXED_SIGMA0, 0.01), 0.0, Inf),
    β_σ₀ = truncated(Normal(FSG_FIXED_SIGMA0, 0.01), 0.0, Inf),
)

function fsg_poisson_grw(name::Symbol, dynamics)
    return CountModelBuilder(name) |>
        add(GlobalInterception()) |>
        add(dynamics) |>
        add(GlobalHomeAdvantage()) |>
        add(PoissonObservation()) |>
        build
end

"All four arms, tight first."
fsg_models() = Tuple{String,Any}[
    (FSG_TIGHT, fsg_poisson_grw(Symbol(FSG_TIGHT), fsg_tight_dynamics())),
    (FSG_LOOSE_VAR, fsg_poisson_grw(Symbol(FSG_LOOSE_VAR), fsg_loose_var_dynamics())),
    (FSG_LOOSE_T, fsg_poisson_grw(Symbol(FSG_LOOSE_T), fsg_loose_t_dynamics())),
    (FSG_LOOSE_FIXED, fsg_poisson_grw(Symbol(FSG_LOOSE_FIXED), fsg_loose_fixed_dynamics())),
]

function fsg_fit_configs(c::FSGConfig, models, splitter, sampler;
                         name_suffix::AbstractString = "")
    g = fsg_gph_config(c)
    return Dict(name => FitConfig(
        name = name * name_suffix,
        model = model,
        splitter = splitter,
        sampler = sampler,
        execution = gph_execution(g),
        tags = copy(FSG_TAGS),
        description = FSG_DESCRIPTIONS[name],
        save_dir = joinpath(c.save_root, name * name_suffix),
    ) for (name, model) in models)
end

# ==============================================================================
# 3. Fold selection
# ==============================================================================

"""
    fsg_fold_inputs(ds, splitter, model; folds) -> (; boundaries, feature_sets, oos, fold_ids)

`gph_fold_inputs` for an arbitrary fold subset (the smoke gate samples 1, 20, 40,
not the first N).
"""
function fsg_fold_inputs(ds, splitter, model; folds::Union{Nothing,Vector{Int}} = nothing)
    boundaries = Data.create_id_boundaries(ds, splitter)
    fold_ids = folds === nothing ? collect(eachindex(boundaries)) :
               filter(i -> i <= length(boundaries), folds)
    selected = boundaries[fold_ids]
    feature_sets = GPH_FEATURES.create_features(selected, ds, model, splitter)
    oos = Any[Data.get_next_matches(ds, feature_sets[i], splitter)
              for i in eachindex(feature_sets)]
    return (; boundaries = selected, feature_sets, oos, fold_ids)
end

# ==============================================================================
# 4. Posterior draw mixtures
# ==============================================================================

"`k` evenly spaced draw indices out of `n`, so every chain contributes."
function fsg_draw_indices(n::Int, k::Int)
    0 <= k <= n || error("cannot take $k of $n draws")
    k == 0 && return Int[]
    return unique!(round.(Int, range(1, n; length = k)))
end

"""
    mixture_latents(tight, loose, ρ; n_total = n_draws(tight)) -> CountLatents

Concatenate posterior draws: `N₂ = round(ρ · n_total)` draws from `loose` and
`N₁ = n_total − N₂` from `tight`, over the fixtures both containers hold:

    λ_combined = hcat(λ_tight[:, idx₁], λ_loose[:, idx₂])

The indices are evenly spaced over each container rather than `1:N`, because draws
are stored chain-major and `1:N` would take whole chains. The draw count is held at
`n_total` for every ρ so ladders compare like with like. ρ = 0 and ρ = 1 return the
pure arms (on the common fixtures).
"""
function mixture_latents(tight::CountLatents, loose::CountLatents, ρ::Real;
                         n_total::Int = n_draws(tight))
    0.0 <= ρ <= 1.0 || error("mixture ratio ρ must be in [0, 1]; got $ρ")
    n2 = round(Int, ρ * n_total)
    n1 = n_total - n2
    n1 <= n_draws(tight) || error("need $n1 tight draws, container has $(n_draws(tight))")
    n2 <= n_draws(loose) || error("need $n2 loose draws, container has $(n_draws(loose))")
    common = sort!(collect(intersect(Set(tight.match_ids), Set(loose.match_ids))))
    it = Dict(m => i for (i, m) in enumerate(tight.match_ids))
    il = Dict(m => i for (i, m) in enumerate(loose.match_ids))
    rt = [it[m] for m in common]
    rl = [il[m] for m in common]
    d1 = fsg_draw_indices(n_draws(tight), n1)
    d2 = fsg_draw_indices(n_draws(loose), n2)
    return CountLatents(common,
        hcat(tight.λ_home[rt, d1], loose.λ_home[rl, d2]),
        hcat(tight.λ_away[rt, d1], loose.λ_away[rl, d2]))
end

# ==============================================================================
# 5. Pricing: 1X2, O/U 2.5, BTTS off one score grid
# ==============================================================================

const FSG_MARKETS = Data.MarketConfig([Data.Market1X2(), Data.MarketOverUnder(2.5), Data.MarketBTTS()])

"""
    fsg_fixture_probs(latents, model) -> (probs, books)

Posterior-mean probability per (match, market, selection), and one row per
(match, market) carrying the per-draw book sum.
"""
function fsg_fixture_probs(latents, model)
    ppd = BayesianFootball.Predictions.model_inference(latents, model; market_config = FSG_MARKETS)
    df = ppd.df
    probs = DataFrame(match_id = Int.(df.match_id),
                      market_name = String.(df.market_name),
                      market_line = Float64.(df.market_line),
                      selection = Symbol.(df.selection),
                      prob = mean.(df.distribution),
                      prob_min = minimum.(df.distribution),
                      prob_max = maximum.(df.distribution))
    books = combine(groupby(df, [:match_id, :market_name, :market_line]),
                    :distribution => (d -> Ref(reduce(+, d))) => :book_sum)
    return probs, books
end

"""
    fsg_grid_audit(latents, model; tol) -> NamedTuple

G7. Every selection probability must lie in [0, 1] on every draw, and every market
book must sum, draw by draw, to the grid's retained mass
`cdf(Poisson(λ_h), G−1) · cdf(Poisson(λ_a), G−1)` within `tol`, where `G` is the
kernel's grid size (12: 0–11 goals). The kernel does not renormalise, so a book
legitimately misses 1 by the truncated tail (~1e-4 on a high-rate draw); what it
may not do is miss by anything else.
"""
function fsg_grid_audit(latents::CountLatents, model; tol::Float64 = 1.0e-10)
    G = BayesianFootball.Predictions.TPL_MAX_GOALS
    probs, books = fsg_fixture_probs(latents, model)
    lo = minimum(probs.prob_min)
    hi = maximum(probs.prob_max)
    (lo >= 0.0 && hi <= 1.0) || error("probability outside [0, 1]: min $lo, max $hi")
    row = Dict(m => i for (i, m) in enumerate(latents.match_ids))
    worst_dev = 0.0
    worst_trunc = 0.0
    for r in eachrow(books)
        i = row[r.match_id]
        kept = cdf.(Poisson.(latents.λ_home[i, :]), G - 1) .* cdf.(Poisson.(latents.λ_away[i, :]), G - 1)
        worst_dev = max(worst_dev, maximum(abs.(r.book_sum .- kept)))
        worst_trunc = max(worst_trunc, maximum(1.0 .- kept))
    end
    worst_dev <= tol || error("a market book deviates from the grid's retained mass by $worst_dev > $tol")
    return (; n_rows = nrow(probs), min_prob = lo, max_prob = hi,
              worst_book_dev = worst_dev, worst_truncation = worst_trunc)
end

# ==============================================================================
# 6. The market: de-vigged Betfair close, inverted to (λ_mkt_h, λ_mkt_a)
# ==============================================================================

"""
    fsg_closing_book(ds) -> DataFrame

Betfair close as a 20-minute time-weighted average before kickoff, de-vigged within
(match, market, line). Same construction as `calibration_generative_eda/l01`.
"""
function fsg_closing_book(ds)
    raw = Data.summarize_odds(ds.betfair_odds, Data.TWAEstimator(); window = (-20.0, 0.0))
    odds = DataFrame(match_id = Int.(raw.match_id),
                     market_name = String.(raw.market_name),
                     market_line = Float64.(raw.market_line),
                     selection = Symbol.(raw.selection),
                     odds_close = Float64.(raw.odds))
    filter!(r -> isfinite(r.odds_close) && r.odds_close > 1.0, odds)
    odds.prob_implied_close = 1.0 ./ odds.odds_close
    transform!(groupby(odds, [:match_id, :market_name, :market_line]),
               :prob_implied_close => (p -> p ./ sum(p)) => :prob_fair_close)
    return odds
end

"Accepted market-rate inversions for `match_ids`, as a frame."
function fsg_market_rates(book, match_ids)
    rates = FSG_CAL.invert_market_rates(book; match_ids = match_ids)
    frame = FSG_CAL.inversion_frame(rates)
    return filter(:accepted => identity, frame)
end

# ==============================================================================
# 7. Supremacy and tail diagnostics
# ==============================================================================

"Posterior-mean log-rate supremacy per fixture: E[log λ_h − log λ_a]."
fsg_supremacy(l::CountLatents) = DataFrame(
    match_id = l.match_ids,
    sup_model = vec(mean(log.(l.λ_home) .- log.(l.λ_away); dims = 2)),
    total_model = vec(mean(l.λ_home .+ l.λ_away; dims = 2)))

"""
    compute_supremacy_slope(sup_model, sup_mkt) -> (; slope, intercept, r2, n)

OLS of model supremacy on market supremacy. Slope 1 is market parity; the
Scottish Lower production models sit at 0.32.
"""
function compute_supremacy_slope(sup_model::AbstractVector{<:Real}, sup_mkt::AbstractVector{<:Real})
    valid = isfinite.(sup_model) .& isfinite.(sup_mkt)
    n = count(valid)
    n < 10 && return (slope = NaN, intercept = NaN, r2 = NaN, n = n)
    x = sup_mkt[valid]
    y = sup_model[valid]
    cxy = sum((x .- mean(x)) .* (y .- mean(y)))
    vx = sum((x .- mean(x)) .^ 2)
    vy = sum((y .- mean(y)) .^ 2)
    slope = cxy / vx
    return (slope = slope, intercept = mean(y) - slope * mean(x),
            r2 = cxy^2 / (vx * vy), n = n)
end

"""
    fsg_supremacy_report(latents, model, market) -> NamedTuple

Slope vs the market, the model's win probability on market favourites
(P_mkt(home or away) ≥ 0.70), and the largest win probability the model issues.
`market` is `fsg_market_rates(...)` joined with the de-vigged 1X2 book.
"""
function fsg_supremacy_report(latents, model, market::AbstractDataFrame)
    sup = fsg_supremacy(latents)
    j = innerjoin(sup, market; on = :match_id)
    s = compute_supremacy_slope(j.sup_model, log.(j.lambda_mkt_h) .- log.(j.lambda_mkt_a))

    probs, _ = fsg_fixture_probs(latents, model)
    x12 = filter(r -> r.market_name == "1X2", probs)
    wide = unstack(select(x12, :match_id, :selection, :prob), :selection, :prob)
    fav = innerjoin(wide, select(market, :match_id, :p_mkt_home, :p_mkt_away); on = :match_id)
    fav_rows = [(r.p_mkt_home >= 0.70 ? (r.home, r.p_mkt_home) :
                 r.p_mkt_away >= 0.70 ? (r.away, r.p_mkt_away) : nothing)
                for r in eachrow(fav)]
    fav_rows = filter(!isnothing, fav_rows)
    return (; slope = s.slope, intercept = s.intercept, r2 = s.r2, n = s.n,
              sup_sd = std(sup.sup_model),
              max_win_prob = maximum(max.(wide.home, wide.away)),
              n_fav70 = length(fav_rows),
              fav70_model = isempty(fav_rows) ? NaN : mean(first.(fav_rows)),
              fav70_market = isempty(fav_rows) ? NaN : mean(last.(fav_rows)))
end

"The market frame `fsg_supremacy_report` consumes: inverted rates + de-vigged 1X2."
function fsg_market_frame(book, match_ids)
    rates = fsg_market_rates(book, match_ids)
    x12 = filter(r -> r.market_name == "1X2" && r.selection in (:home, :away), book)
    nrow(x12) > 0 || error("closing book has no 1X2 :home/:away rows; markets present: " *
                           join(unique(string.(book.market_name, "/", book.selection)), ", "))
    p = unstack(select(x12, :match_id, :selection, :prob_fair_close), :selection, :prob_fair_close)
    rename!(p, :home => :p_mkt_home, :away => :p_mkt_away)
    return innerjoin(select(rates, :match_id, :lambda_mkt_h, :lambda_mkt_a),
                     dropmissing(p); on = :match_id)
end

"Posterior mean of the level scales σ₀ (attack, defence), pooled over folds."
function fsg_sigma0(fit)
    out = Dict{String,Float64}()
    for (key, suffix) in (("α_σ₀", "α.σ₀"), ("β_σ₀", "β.σ₀"))
        vals = Float64[]
        for f in fit.folds
            syms = [s for s in names(f.chain, :parameters) if endswith(string(s), suffix)]
            isempty(syms) || append!(vals, vec(Array(f.chain[first(syms)])))
        end
        out[key] = isempty(vals) ? NaN : mean(vals)
    end
    return out
end

# ==============================================================================
# 8. Config truth for the production grid
# ==============================================================================

"Register every arm, its fit recipe, the splitter and the sampler in `config_registry`."
function fsg_register!(db, models, splitter, sampler, configs, c::FSGConfig)
    model_ids = Dict{String,Int}()
    for (name, model) in models
        model_ids[name] = save_model(db, name, model;
                                     description = FSG_DESCRIPTIONS[name], tags = FSG_TAGS)
        save_config(db, name * "_fit", configs[name];
                    description = FSG_DESCRIPTIONS[name] * " TODO 021 recipe.", tags = FSG_TAGS)
    end
    splitter_id = save_splitter(db, "scottish_lower_fast_slow_grw_40fold", splitter;
        description = "Pooled 56/57, two history seasons, match-biweek walk-forward over 24/25 and 25/26.",
        tags = FSG_TAGS)
    sampler_id = save_sampler(db, "queued_nuts_$(c.chains)x$(c.samples)_w$(c.warmup)", sampler;
        description = "ReverseDiff queued NUTS: $(c.chains) chains, $(c.warmup) warmup, " *
                      "$(c.samples) retained, target acceptance $(c.accept_rate).",
        tags = FSG_TAGS)
    return (; model_ids, splitter_id, sampler_id)
end

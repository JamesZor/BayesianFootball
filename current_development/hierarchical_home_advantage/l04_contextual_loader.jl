# ==============================================================================
# Task 008 Phase 2 loader — contextual home advantage: pitch surface and match timing
# ==============================================================================
#
# Definitions only. `r06_contextual_smoke.jl`, `r07_contextual_production.jl`,
# `r08_contextual_evaluate.jl` and `r09_slate_repricing.jl` execute.
#
# THE MODEL (WORK_PACKAGE_PHASE_2_TURF_TIMING.md §3)
#
#   η_h = base + γ_ij + att_h + def_a + wealth + turf_pace·turf_i
#   η_a = base +        att_a + def_h − wealth + turf_pace·turf_i
#
#   γ_ij = γ_base + u_i
#        + β_turf_asym · turf_i·(1 − turf_j)     grass visitor at a turf ground
#        + β_turf_gen  · turf_i                  any fixture at a turf ground
#        + β_midweek   · midweek                 Tue–Thu, or Friday evening
#        + β_rest      · (rest_i − rest_j)       days, each side capped at 14
#
#   u_i = σ_stadium · ũ_i,  ũ_i ~ N(0, 1),  σ_stadium ~ N⁺(0, 0.05),  γ_base ~ N(0.15, 0.05)
#
# HOW IT IS BUILT. No engine change. γ_base + u_i is `HierarchicalTeamHomeAdvantage`
# with the work package's priors: already non-centred and gathered by `home_ids`. Every
# per-fixture term is a scalar covariate through the builder's existing six-method
# contract, which the builder unrolls at compile time. The only addition is a third
# covariate role, `HomeOnlyRole`, because an HA term shifts η_h alone:
#
#   SupremacyRole  (q, −q)      wealth, distance
#   LevelRole      (q,  q)      turf_pace  (H2: both sides score more on turf)
#   HomeOnlyRole   (q, nothing) turf_asym, turf_gen, midweek, rest_diff
#
# `nothing` is the engine's own structural zero (`_predictor_shift(η, ::Nothing) = η`), so a
# home-only term adds no node on the away side of the tape. §2 below adds the missing
# `_predictor_acc(::Nothing, y)` method so a home-only term may sit anywhere in the tuple.
#
# THE SURFACE REGISTRY. `src/features/data/scottish_stadium_geocodes.csv`, column
# `is_synthetic_pitch` (commit e693fa7d), with dated overrides in `CTX_SURFACE_CHANGES`.
# Spot-checked 2026-09-18 against public sources; see README §Phase 2. It keys the HOME
# CLUB, not the venue. That is exact for the 2022/23-onward panel: every groundshare in it
# (Clyde and Hamilton at New Douglas Park) is turf-to-turf.
#
# FILTRATION. Turf is a function of (club, date). Midweek is a function of the kickoff.
# Rest days read only kickoffs on EARLIER calendar days from the DataStore. All three are
# known before kickoff, so fit-time and prediction-time values come from one function.
#
# KNOWN LIMITS (disclosed in the README, not fixed here)
#   * Rest days are LEAGUE-ONLY: betdb holds tournaments 56/57 and no cup fixtures, so a
#     midweek cup tie is invisible and rest_i − rest_j is almost always 0.
#   * T003 still applies: an unmapped home club gets γ = 0 (not γ_base), but its
#     contextual terms are still priced.
#   * The fitted artefacts carry types defined in this file. Include it before `load_fit`.
# ==============================================================================

using CSV

if !isdefined(@__MODULE__, :HHAConfig)
    include(joinpath(@__DIR__, "l01_loader.jl"))
end

const CTX_BUILDER = BayesianFootball.Models.PreGame.Builder

# ==============================================================================
# 1. Experiment configuration
# ==============================================================================

"""
    CtxConfig

Every number that governs Phase 2. The smoke and production budgets are both the work
package's 4 × (500 warmup + 1000 retained), δ = 0.80.
"""
Base.@kwdef struct CtxConfig
    experiment::String = "scottish_lower_contextual_ha"
    smoke_experiment::String = "smoke_contextual_ha"
    save_root::String = joinpath(@__DIR__, "results", "phase2")

    target_seasons::Vector{String} = ["24/25", "25/26"]
    extension_seasons::Vector{String} = ["24/25", "25/26", "26/27"]
    expected_folds::Int = 40
    expected_oos::Int = 710

    smoke_folds::Int = 2
    smoke_samples::Int = 1000
    smoke_warmup::Int = 500
    smoke_chains::Int = 4

    samples::Int = 1000
    warmup::Int = 500
    chains::Int = 4
    accept_rate::Float64 = 0.80
    max_depth::Int = 10
    max_concurrent_tasks::Int = 16

    persist_stride::Int = 2
    gradient_replays::Int = 200

    max_rhat::Float64 = 1.05
    strict_rhat::Float64 = 1.01
    min_ess::Float64 = 400.0
    max_divergence_rate::Float64 = 0.001
    min_bfmi::Float64 = 0.30
    max_treedepth_rate::Float64 = 0.05
end

"The HHAConfig twin of a CtxConfig, for Phase 1 helpers that take one."
ctx_as_hha(c::CtxConfig) = HHAConfig(
    experiment = c.experiment, smoke_experiment = c.smoke_experiment, save_root = c.save_root,
    target_seasons = c.target_seasons, extension_seasons = c.extension_seasons,
    expected_folds = c.expected_folds, expected_oos = c.expected_oos,
    smoke_folds = c.smoke_folds, smoke_samples = c.smoke_samples,
    smoke_warmup = c.smoke_warmup, smoke_chains = c.smoke_chains,
    samples = c.samples, warmup = c.warmup, chains = c.chains,
    accept_rate = c.accept_rate, max_depth = c.max_depth,
    max_concurrent_tasks = c.max_concurrent_tasks, persist_stride = c.persist_stride,
    gradient_replays = c.gradient_replays, max_rhat = c.max_rhat, strict_rhat = c.strict_rhat,
    min_ess = c.min_ess, max_divergence_rate = c.max_divergence_rate,
    min_bfmi = c.min_bfmi, max_treedepth_rate = c.max_treedepth_rate)

const CTX_MODEL_NAMES = [
    "m05_joint_td_turf_asym",
    "m05_joint_td_turf_dual",
    "m05_joint_td_contextual",
]

const CTX_DESCRIPTIONS = Dict(
    "m05_joint_td_turf_asym" =>
        "Exp 06 m05 (TimeDecay(180) + production wealth + joint) with stadium RE and asymmetric turf HA.",
    "m05_joint_td_turf_dual" =>
        "m05_joint_td_turf_asym + general turf HA shift + turf scoring-pace level term.",
    "m05_joint_td_contextual" =>
        "m05_joint_td_turf_dual + midweek HA shift + league-only rest-day differential.",
    "m12_joint_hybrid_contextual" =>
        "Gen 4 TimeDecay hybrid (lineup + wealth + joint) with the winning Phase 2 contextual HA.",
)

const CTX_TAGS = [
    "scottish-lower", "24/25", "25/26", "contextual-home-advantage",
    "todo008", "phase2", "turf", "reversediff",
]

"The flat-HA control every rung is scored against (Exp 06, persisted)."
const CTX_CONTROL = HHA_CONTROLS["m05_joint_production_wealth_hier_ha"]

# ==============================================================================
# 2. HomeOnlyRole
# ==============================================================================

"η_h += q, η_a unchanged — a home-advantage shift."
struct HomeOnlyRole <: AbstractCovariateRole end

CTX_BUILDER.covariate_sides(::HomeOnlyRole, q) = (q, nothing)

# The engine defines `_predictor_acc(x, ::Nothing)` only, because until now no term could
# return a structural zero on one side. These two make the accumulator total. The second
# resolves the ambiguity the first would create for (nothing, nothing).
CTX_BUILDER._predictor_acc(::Nothing, y) = y
CTX_BUILDER._predictor_acc(::Nothing, ::Nothing) = nothing

# ==============================================================================
# 3. The contextual feature
# ==============================================================================

const CTX_DEFAULT_REGISTRY = normpath(joinpath(@__DIR__, "..", "..", "src", "features",
                                               "data", "scottish_stadium_geocodes.csv"))

"""
Dated surface changes the registry's single flag cannot express: `club => (date, value)`,
meaning the club plays on `value` from `date` onward, and on the registry's opposite value
before it.

* `dumbarton`: The Rock was grass 2000–2026 and turf from the 2026/27 season (Wikipedia,
  "The Rock, Dumbarton"). The registry's 0 is correct for every panel season but not for
  the 2026-09-12 card. The date is the close season.

Checked and NOT overridden: Falkirk's summer 2023 install replaced an older artificial
surface (Falkirk FC, 2023-08-30), so Falkirk is turf throughout the panel.
"""
const CTX_SURFACE_CHANGES = Dict{String,Tuple{Date,Float64}}(
    "dumbarton" => (Date(2026, 6, 1), 1.0),
)

Base.@kwdef struct ContextualMatchFeature <: GPH_FEATURES.AbstractFeatureConfig
    registry_csv::String = CTX_DEFAULT_REGISTRY
    rest_cap_days::Int = 14
    # Friday kickoffs at or after this UTC hour are "evening". A 19:45 BST kickoff is 18:45 UTC,
    # a 15:00 Saturday one 14:00 UTC; 16 separates them in both clock regimes.
    evening_hour_utc::Int = 16
end

"""
Everything prediction-time needs, built once from the registry and the DataStore:
the surface flag per club, the dated overrides, and every club's sorted kickoff dates.
"""
struct CtxBridge
    surface::Dict{String,Float64}
    changes::Dict{String,Tuple{Date,Float64}}
    dates::Dict{String,Vector{Date}}
    rest_cap_days::Int
    evening_hour_utc::Int
end

function ctx_load_registry(path::AbstractString)
    reg = CSV.read(path, DataFrame)
    hasproperty(reg, :is_synthetic_pitch) || error("$path has no is_synthetic_pitch column")
    out = Dict{String,Float64}()
    for r in eachrow(reg)
        v = Float64(r.is_synthetic_pitch)
        v in (0.0, 1.0) || error("$path: is_synthetic_pitch for $(r.team_slug) is $v")
        out[String(r.team_slug)] = v
    end
    return out
end

function ctx_bridge(ds, c::ContextualMatchFeature)
    dates = Dict{String,Vector{Date}}()
    for r in eachrow(ds.matches)
        d = Date(r.match_date)
        push!(get!(dates, String(r.home_team), Date[]), d)
        push!(get!(dates, String(r.away_team), Date[]), d)
    end
    foreach(v -> unique!(sort!(v)), values(dates))
    return CtxBridge(ctx_load_registry(c.registry_csv), CTX_SURFACE_CHANGES, dates,
                     c.rest_cap_days, c.evening_hour_utc)
end

"Surface flag of `club`'s home ground on `date`; `nothing` if the club is not in the registry."
function ctx_surface(b::CtxBridge, club::AbstractString, date::Date)
    base = get(b.surface, String(club), nothing)
    base === nothing && return nothing
    change = get(b.changes, String(club), nothing)
    change === nothing && return base
    since, value = change
    return date >= since ? value : 1.0 - value
end

"Days since `club`'s last kickoff on an earlier calendar day, capped; the cap if none."
function ctx_rest(b::CtxBridge, club::AbstractString, date::Date)
    ds = get(b.dates, String(club), nothing)
    ds === nothing && return Float64(b.rest_cap_days)
    k = searchsortedfirst(ds, date) - 1          # last index strictly before `date`
    k == 0 && return Float64(b.rest_cap_days)
    return Float64(min(Dates.value(date - ds[k]), b.rest_cap_days))
end

"Tuesday–Thursday, or a Friday kickoff at or after the evening hour (UTC)."
function ctx_midweek(b::CtxBridge, date::Date, hour_utc)
    dow = Dates.dayofweek(date)
    dow in (2, 3, 4) && return 1.0
    dow == 5 && hour_utc !== nothing && hour_utc >= b.evening_hour_utc && return 1.0
    return 0.0
end

"""
    ctx_design(b, home, away, date, hour_utc; strict) -> NamedTuple

One fixture's contextual design. `strict = true` (fit time) refuses a club missing from the
registry; `strict = false` (prediction time) treats it as grass and reports it through
`unmapped`.
"""
function ctx_design(b::CtxBridge, home, away, date::Date, hour_utc; strict::Bool)
    th = ctx_surface(b, home, date)
    ta = ctx_surface(b, away, date)
    unmapped = String[]
    th === nothing && push!(unmapped, String(home))
    ta === nothing && push!(unmapped, String(away))
    if !isempty(unmapped)
        strict && error("surface registry has no entry for: $(join(unmapped, ", "))")
        th = something(th, 0.0)
        ta = something(ta, 0.0)
    end
    rest_diff = ctx_rest(b, home, date) - ctx_rest(b, away, date)
    return (; turf_home = th, turf_away = ta, turf_asym = th * (1.0 - ta),
              midweek = ctx_midweek(b, date, hour_utc), rest_diff, unmapped)
end

_ctx_hour(r) = hasproperty(r, :match_hour) && !ismissing(r.match_hour) ? Int(r.match_hour) : nothing

function GPH_FEATURES.add_feature!(F_data::Dict, c::ContextualMatchFeature, ordered_ids,
                                   team_map::Dict, ds::BayesianFootball.Data.DataStore)
    haskey(F_data, :ctx_bridge) && return nothing     # shared by every contextual covariate
    b = ctx_bridge(ds, c)
    rows = Dict(Int(r.match_id) => r for r in eachrow(ds.matches))
    n = length(ordered_ids)
    cols = Dict(k => zeros(n) for k in (:turf_home, :turf_away, :turf_asym, :midweek, :rest_diff))
    for (i, id) in enumerate(ordered_ids)
        r = rows[Int(id)]
        d = ctx_design(b, r.home_team, r.away_team, Date(r.match_date), _ctx_hour(r); strict = true)
        for k in keys(cols)
            cols[k][i] = getproperty(d, k)
        end
    end
    for (k, v) in cols
        F_data[Symbol(:ctx_, k)] = v
    end
    F_data[:ctx_bridge] = b
    return nothing
end

# ==============================================================================
# 4. The contextual covariates
# ==============================================================================

"""
    ContextualCovariate{K}

One scalar contextual term named `K` (`turf_asym`, `turf_gen`, `turf_pace`, `midweek`,
`rest_diff`). The chain site is `K.w`. `column` is the `ctx_*` design key it reads —
`turf_gen` and `turf_pace` both read `turf_home` and differ only in role.
"""
struct ContextualCovariate{K,D<:UnivariateDistribution,R<:AbstractCovariateRole} <: AbstractCovariateConfig
    feature::ContextualMatchFeature
    column::Symbol
    prior::D
    role::R
end

ContextualCovariate{K}(column::Symbol, prior, role;
                       feature = ContextualMatchFeature()) where {K} =
    ContextualCovariate{K,typeof(prior),typeof(role)}(feature, column, prior, role)

CTX_BUILDER.covariate_name(::ContextualCovariate{K}) where {K} = K
CTX_BUILDER.covariate_role(c::ContextualCovariate) = c.role
CTX_BUILDER.covariate_prior(c::ContextualCovariate) = c.prior
CTX_BUILDER.covariate_features(c::ContextualCovariate) = GPH_FEATURES.AbstractFeatureConfig[c.feature]
CTX_BUILDER.covariate_column(c::ContextualCovariate, fs) =
    Vector{Float64}(fs.data[Symbol(:ctx_, c.column)])

function CTX_BUILDER.covariate_oos(c::ContextualCovariate, fs, df)
    b = fs.data[:ctx_bridge]
    out = zeros(nrow(df))
    for (i, r) in enumerate(eachrow(df))
        d = ctx_design(b, r.home_team, r.away_team, Date(r.match_date), _ctx_hour(r); strict = false)
        isempty(d.unmapped) || @warn "contextual HA: unregistered club priced as grass" c.column d.unmapped maxlog = 5
        out[i] = getproperty(d, c.column)
    end
    return out
end

# The default `predictor_oos` hands `covariate_sides` straight to the extractor, which
# accumulates `q.a` into a vector. Give a home-only term an explicit zero away side there.
function CTX_BUILDER.predictor_oos(c::ContextualCovariate, draw, bridge, row)
    q = draw.w .* get(bridge, Int(row.match_id), 0.0)
    h, a = CTX_BUILDER.covariate_sides(c.role, q)
    return (; h, a = a === nothing ? zero(q) : a)
end

# Priors: work package §3A. turf_pace has none there; N(0, 0.05) matches turf_gen.
ctx_turf_asym() = ContextualCovariate{:turf_asym}(:turf_asym, Normal(0.05, 0.05), HomeOnlyRole())
ctx_turf_gen()  = ContextualCovariate{:turf_gen}(:turf_home, Normal(0.0, 0.05), HomeOnlyRole())
ctx_turf_pace() = ContextualCovariate{:turf_pace}(:turf_home, Normal(0.0, 0.05), LevelRole())
ctx_midweek()   = ContextualCovariate{:midweek}(:midweek, Normal(0.05, 0.05), HomeOnlyRole())
ctx_rest_diff() = ContextualCovariate{:rest_diff}(:rest_diff, Normal(0.02, 0.02), HomeOnlyRole())

const CTX_TERMS = (:turf_asym, :turf_gen, :turf_pace, :midweek, :rest_diff)

"γ_base + u_i with the work package's priors."
ctx_home_advantage() = HierarchicalTeamHomeAdvantage(
    γ_base = Normal(0.15, 0.05),
    σ_γ = truncated(Normal(0.0, 0.05), lower = 0.0),
)

# ==============================================================================
# 5. Models
# ==============================================================================

"The contextual terms each rung adds, in ladder order."
const CTX_RUNG_TERMS = Dict(
    "m05_joint_td_turf_asym" => () -> (ctx_turf_asym(),),
    "m05_joint_td_turf_dual" => () -> (ctx_turf_asym(), ctx_turf_gen(), ctx_turf_pace()),
    "m05_joint_td_contextual" => () -> (ctx_turf_asym(), ctx_turf_gen(), ctx_turf_pace(),
                                        ctx_midweek(), ctx_rest_diff()),
)

"""
    ctx_m05(name, terms) -> ComposableCountModel

The Exp 06 m05 recipe (Phase 1's `hha_models` m05, add order unchanged) with the
contextual HA slot and `terms` appended after wealth.
"""
function ctx_m05(name::AbstractString, terms::Tuple)
    b = CountModelBuilder(Symbol(name)) |>
        add(GlobalInterception()) |>
        add(TimeDecayDynamics(days_half_life = 180.0)) |>
        add(ctx_home_advantage()) |>
        add(gph_production_wealth())
    for t in terms
        b = b |> add(t)
    end
    return b |> add(gph_joint_observation()) |> build
end

"The m12 TimeDecay hybrid with the same slot — rung 5, built only if a rung wins."
function ctx_m12(name::AbstractString, terms::Tuple)
    b = CountModelBuilder(Symbol(name)) |>
        add(GlobalInterception()) |>
        add(TimeDecayDynamics(days_half_life = 180.0)) |>
        add(ctx_home_advantage()) |>
        add(gph_shots_pillar()) |>
        add(gph_production_wealth())
    for t in terms
        b = b |> add(t)
    end
    return b |> add(gph_joint_observation()) |> build
end

ctx_models() = Tuple{String,Any}[(n, ctx_m05(n, CTX_RUNG_TERMS[n]())) for n in CTX_MODEL_NAMES]

"The flat twin of every rung, for the gradient audit: the Exp 06 control recipe."
ctx_flat_twin() = first(hha_models(home_advantage = GlobalHomeAdvantage()))[2]

# ==============================================================================
# 6. Sampler, recipes, registration
# ==============================================================================

ctx_smoke_sampler(c::CtxConfig) = QueuedNUTSConfig(
    n_samples = c.smoke_samples, n_warmup = c.smoke_warmup, n_chains = c.smoke_chains,
    accept_rate = c.accept_rate, max_depth = c.max_depth, show_progress = false)

ctx_production_sampler(c::CtxConfig) = QueuedNUTSConfig(
    n_samples = c.samples, n_warmup = c.warmup, n_chains = c.chains,
    accept_rate = c.accept_rate, max_depth = c.max_depth, show_progress = false)

function ctx_fit_configs(c::CtxConfig, models, splitter, sampler; name_suffix::AbstractString = "")
    return Dict(name => FitConfig(
        name = name * name_suffix,
        model = model,
        splitter = splitter,
        sampler = sampler,
        execution = hha_execution(ctx_as_hha(c)),
        tags = copy(CTX_TAGS),
        description = CTX_DESCRIPTIONS[name],
        save_dir = joinpath(c.save_root, name * name_suffix),
    ) for (name, model) in models)
end

function ctx_register!(db, models, splitter, sampler, configs)
    model_ids = Dict{String,Int}()
    for (name, model) in models
        model_ids[name] = save_model(db, name, model; description = CTX_DESCRIPTIONS[name], tags = CTX_TAGS)
        save_config(db, name * "_fit", configs[name];
                    description = CTX_DESCRIPTIONS[name] * " Task 008 Phase 2 recipe.", tags = CTX_TAGS)
    end
    splitter_id = save_splitter(db, "scottish_lower_contextual_ha_40fold", splitter;
        description = "Pooled 56/57, two history seasons, match-biweek walk-forward over 24/25 and 25/26.",
        tags = CTX_TAGS)
    sampler_id = save_sampler(db, "queued_nuts_4x1000_w500_a080", sampler;
        description = "ReverseDiff queued NUTS: 4 chains, 500 warmup, 1000 retained, target acceptance 0.80.",
        tags = CTX_TAGS)
    return (; model_ids, splitter_id, sampler_id)
end

ctx_sample(fit_config::FitConfig, inputs, c::CtxConfig; checkpoint_dir = nothing) =
    hha_sample(fit_config, inputs, ctx_as_hha(c); checkpoint_dir)

# ==============================================================================
# 7. Diagnostics
# ==============================================================================

"Contextual terms present in `model`, as covariate objects."
ctx_terms(model) = [t for t in model.covariates if t isa ContextualCovariate]

"""
    ctx_design_summary(inputs) -> DataFrame

Per fold, how many TRAINING fixtures switch each contextual term on — the effective sample
size behind each coefficient. A term with a handful of non-zero rows is prior-dominated.
"""
function ctx_design_summary(inputs)
    rows = NamedTuple[]
    for (i, fsc) in enumerate(inputs.feature_sets)
        d = first(fsc).data
        haskey(d, :ctx_turf_home) || continue
        push!(rows, (; fold = i, n_train = length(d[:ctx_turf_home]),
                       turf_home = Int(sum(d[:ctx_turf_home])),
                       turf_asym = Int(sum(d[:ctx_turf_asym])),
                       midweek = Int(sum(d[:ctx_midweek])),
                       rest_nonzero = count(!iszero, d[:ctx_rest_diff]),
                       rest_abs_mean = mean(abs, d[:ctx_rest_diff])))
    end
    return DataFrame(rows)
end

"""
    ctx_coefficients(fit, fold) -> DataFrame

Posterior of every contextual weight, γ_base and σ_stadium on one fold, beside its prior.
`contraction = 1 − sd_post / sd_prior`: near 0 means the data said nothing and P(β > 0) is
the prior's (0.84 for a N(0.05, 0.05) prior). H1/H3 are read from both columns together.
"""
function ctx_coefficients(fit, fold::Int)
    model = fit.config.model
    chain = fit.folds[fold].chain
    q(v, p) = quantile(v, p)
    rows = NamedTuple[]
    push_row(site, prior, v) = push!(rows, (; fold, site,
        mean = mean(v), sd = std(v), q05 = q(v, 0.05), q50 = q(v, 0.50), q95 = q(v, 0.95),
        p_positive = mean(v .> 0),
        prior_mean = mean(prior), prior_sd = std(prior),
        prior_p_positive = 1 - cdf(prior, 0.0),
        contraction = 1 - std(v) / std(prior)))
    for t in ctx_terms(model)
        site = "$(covariate_name(t)).w"
        push_row(site, covariate_prior(t), vec(Array(chain[Symbol(site)])))
    end
    ha = model.home_advantage
    if ha isa HierarchicalTeamHomeAdvantage
        push_row("ha.γ_base", ha.γ_base, vec(Array(chain[Symbol("ha.γ_base")])))
        push_row("ha.σ_γ", ha.σ_γ, vec(Array(chain[Symbol("ha.σ_γ")])))
    end
    df = DataFrame(rows)
    if ha isa HierarchicalTeamHomeAdvantage
        sigma = vec(Array(chain[Symbol("ha.σ_γ")]))
        df.p_below_0p02 = [r.site == "ha.σ_γ" ? mean(sigma .< 0.02) : NaN for r in eachrow(df)]
        df.prior_p_below_0p02 = [r.site == "ha.σ_γ" ? cdf(ha.σ_γ, 0.02) : NaN for r in eachrow(df)]
    end
    return df
end

"Worst R̂ / ESS over folds for every `ha.*` and contextual site."
function ctx_site_report(fit)
    names_ctx = Set(string(covariate_name(t), ".w") for t in ctx_terms(fit.config.model))
    worst = Dict{String,NamedTuple}()
    for f in fit.folds
        rh = DataFrame(MCMCChains.rhat(f.chain))
        eb = DataFrame(MCMCChains.ess(f.chain; kind = :bulk))
        et = DataFrame(MCMCChains.ess(f.chain; kind = :tail))
        bulk = Dict(String(p) => v for (p, v) in zip(eb.parameters, eb.ess))
        tail = Dict(String(p) => v for (p, v) in zip(et.parameters, et.ess))
        for (p, r) in zip(rh.parameters, rh.rhat)
            name = String(p)
            (startswith(name, "ha.") || name in names_ctx) || continue
            row = (; rhat = Float64(r), ess_bulk = Float64(bulk[name]),
                     ess_tail = Float64(tail[name]), fold = Int(f.fold))
            prev = get(worst, name, nothing)
            worst[name] = prev === nothing ? row :
                (; rhat = max(prev.rhat, row.rhat), ess_bulk = min(prev.ess_bulk, row.ess_bulk),
                   ess_tail = min(prev.ess_tail, row.ess_tail),
                   fold = row.rhat > prev.rhat ? row.fold : prev.fold)
        end
    end
    ks = sort!(collect(keys(worst)))
    return DataFrame(site = ks, max_rhat = [worst[k].rhat for k in ks],
                     min_ess_bulk = [worst[k].ess_bulk for k in ks],
                     min_ess_tail = [worst[k].ess_tail for k in ks],
                     worst_rhat_fold = [worst[k].fold for k in ks])
end

"""
    ctx_oos_design(inputs) -> DataFrame

The contextual design of every held-out fixture, from the same bridge the extraction uses.
r08 stratifies on it (turf home, midweek) without rebuilding features.
"""
function ctx_oos_design(inputs)
    rows = NamedTuple[]
    for (i, fsc) in enumerate(inputs.feature_sets)
        oos = inputs.oos[i]
        oos === nothing && continue
        b = first(fsc).data[:ctx_bridge]
        for r in eachrow(oos)
            d = ctx_design(b, r.home_team, r.away_team, Date(r.match_date), _ctx_hour(r); strict = false)
            push!(rows, (; fold = i, match_id = Int(r.match_id), home_team = String(r.home_team),
                           away_team = String(r.away_team), match_date = Date(r.match_date),
                           d.turf_home, d.turf_away, d.turf_asym, d.midweek, d.rest_diff,
                           unmapped = join(d.unmapped, ",")))
        end
    end
    return DataFrame(rows)
end

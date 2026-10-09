# Market arm: Phase C C0 (quality/style Gaussian Kalman on inverted Betfair closes),
# refitted per goal-harness fold and forecast pre-week for every held-out fixture.
# Definitions only; runners call `run_fold`. All execution on mcmc-beast.
module QSMarketArm

using BayesianFootball, CSV, DataFrames, Dates, LinearAlgebra, Random, SHA
using Serialization, Statistics, TOML, ThreadPinning

# The Phase C implementation is reused without changes to its prior, filter or sampler.
const STREAM = joinpath(@__DIR__, "..", "..", "..", "current_development", "market_model")
Base.include(@__MODULE__, joinpath(STREAM, "l06_qs_eda.jl"))
const Q = QualityStyleEDA
const MM = Q.MM
const MID = Q.MID
const PM = Q.PM
const FG = Q.FG
const D = BayesianFootball.Data
const SNAPSHOT_SHA = "c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4"
# Phase C C0 10b chain seeds; fold k uses these + 100 * k (frozen before any market fit).
const SEEDS = Int.(TOML.parsefile(joinpath(STREAM, "fullbook_seeds.toml"))["C0"]["10b"])
const PREDICTION_SEED = 3963
const N_THETA = 128
const N_STATE = 4

function checked_datastore()
    path = joinpath(Base.pkgdir(D), ".cache", "datastore_ScottishLower.jls")
    digest = bytes2hex(open(SHA.sha256, path))
    digest == SNAPSHOT_SHA || error("ScottishLower snapshot differs: $digest")
    return D.load_datastore_cached(D.ScottishLower(); max_age_hours = 10^6)
end

"The goal arms' scope (experiment 03 `lower`), so the market folds use identical IDs."
goal_scope() = D.DataScope(name = "lower", train_tournaments = [56, 57],
    cups = :none, target_tournaments = [56, 57], monitor_tournaments = Int[],
    clock_tournaments = [56, 57], target_seasons = ["24/25", "25/26"],
    history_seasons = 2, dynamics_col = :match_biweek)

"Four seasons: goal folds 1–20 train on 22/23+23/24, so 24/25+25/26 alone is not enough."
function panel_config()
    prior = MM.scottish_lower_2425_2526()
    return MM.MarketModelConfig(name = "pooled_56_57_four_seasons",
        segment = D.ScottishLower(), tournaments = [56, 57],
        seasons = ["22/23", "23/24", "24/25", "25/26"],
        honest_train = ["22/23", "23/24", "24/25"], honest_test = ["25/26"],
        excluded_matches = prior.excluded_matches)
end

"B2/C full-book rules: gated TWA(−20,0] close, KL inversion, ≥5 selections (no 1X2-only)."
function prepare_panel(ds, output::AbstractString)
    config = panel_config()
    mkpath(output)
    rates_path = joinpath(output, "inverted_rates.csv")
    manifest = joinpath(output, "inversion_manifest.txt")
    identity = "$(SNAPSHOT_SHA)\n$(repr(config))\n"
    if isfile(rates_path) || isfile(manifest)
        isfile(rates_path) && isfile(manifest) || error("partial market inversion output")
        read(manifest, String) == identity || error("inversion provenance differs")
        rates = CSV.read(rates_path, DataFrame; types = Dict(:season => String))
    else
        book, refusals = MM.gated_close(ds, config)
        rates = MM.invert_panel(ds, book; config)
        CSV.write(rates_path, rates)
        CSV.write(joinpath(output, "market_refusals.csv"), refusals)
        write(manifest, identity)
    end
    excluded = rates.accepted .& (rates.n_selections .< config.min_selections_ladder)
    frame = DataFrame(match_id = Int.(rates.match_id),
        accepted = rates.accepted .& .!excluded,
        lambda_mkt_h = rates.lambda_h, lambda_mkt_a = rates.lambda_a,
        sse = rates.kl, n_targets = rates.n_selections,
        reason = [excluded[i] ? Q.TB.EXCLUSION : coalesce(rates.reason[i], "")
                  for i in 1:nrow(rates)])
    panel = MID.build_market_panel(ds, frame; config)
    # The published B2 counts must reproduce on the 24/25+25/26 subset.
    target = in.(rates.season, Ref(("24/25", "25/26")))
    counts = (panel = count(target), accepted = count(rates.accepted[target]),
              one_x2_only = count(excluded[target]),
              ladder = count(in(Set(rates.match_id[target])), panel.matches.match_id))
    counts == (panel = 710, accepted = 595, one_x2_only = 78, ladder = 517) ||
        error("B2 target counts differ: $counts")
    return (; panel, rates, config, counts)
end

"Immutable fixture IDs and cutoff from the SAME scoped splitter the goal arms use."
function goal_folds(ds)
    scope = goal_scope()
    scoped = D.apply_scope(ds, scope)
    splitter = D.ScopedWalkForwardCV(scope)
    boundaries = D.create_id_boundaries(scoped, splitter)
    length(boundaries) == 40 || error("goal splitter returned $(length(boundaries)) folds")
    match = Dict(Int(r.match_id) => r for r in eachrow(scoped.matches))
    rows = NamedTuple[]
    for (number, (boundary, meta)) in enumerate(boundaries)
        train_ids = sort!(Int.(vcat(boundary.history_match_ids, boundary.target_match_ids)))
        ids = sort!(Int.(D.get_next_matches(scoped, meta, splitter).match_id))
        isempty(ids) && error("empty target fold $number")
        cutoff = minimum(Date(match[id].match_date) for id in ids)
        maximum(Date(match[id].match_date) for id in train_ids) < cutoff ||
            error("market fold $number has a future training match")
        push!(rows, (; number, train_ids, ids, cutoff))
    end
    length(unique(vcat([r.ids for r in rows]...))) == 710 || error("held-out panel is not 710")
    return (; folds = rows, scoped)
end

"""
Fold training panel: the fold's goal-training IDs that have an accepted full book, on a
weekly grid from the first such week to the week BEFORE the first held-out fixture.
"""
function training_panel(full, fold)
    target_week = searchsortedlast(full.week_start, fold.cutoff)
    target_week > 1 || error("no market week before fold $(fold.number)")
    keep = in.(full.obs_match, Ref(Set(fold.train_ids)))
    any(keep) || error("fold $(fold.number) has no quoted training fixture")
    weeks = full.obs_week[keep]
    first_week, last_week = minimum(weeks), target_week - 1
    maximum(weeks) <= last_week || error("same-week close leaks into fold $(fold.number)")
    n = last_week - first_week + 1
    obs_week = weeks .- (first_week - 1)
    ptr = zeros(Int, n + 1)
    ptr[1] = 1
    foreach(w -> ptr[w+1] += 1, obs_week)
    cumsum!(ptr, ptr)
    ids = Set(full.obs_match[keep])
    matches = full.matches[in.(full.matches.match_id, Ref(ids)), :]
    p = MID.MarketPanel(matches, full.refusals[1:0, :], full.teams, n,
        full.week_start[first_week:last_week], obs_week, full.obs_home[keep],
        full.obs_att[keep], full.obs_def[keep], full.obs_y[keep], full.obs_match[keep],
        full.obs_season[keep], ptr, full.obs_X[keep, :], full.feature_names)
    nrow(matches) == MID.n_fixtures(p) || error("fold $(fold.number) fixture metadata mismatch")
    return (; panel = p, first_week, last_week, target_week)
end

"128 equally spaced retained posterior indices, 32 in each of the four chains."
function posterior_indices(fit)
    n, _, chains = size(fit.udraws)
    n == 3000 && chains == 4 || error("C0 budget changed: $(size(fit.udraws))")
    index = [(chain = c, draw = i) for c in 1:chains
             for i in unique(round.(Int, range(1, n; length = N_THETA ÷ chains)))]
    length(index) == N_THETA || error("not $N_THETA unique posterior indices")
    return index
end

"Observation rows (home rate, away rate) of one fixture in the full panel's state layout."
function fixture_design(nteams, home::Int, away::Int)
    B = zeros(2, 2 + 2nteams)
    h = zeros(2 + 2nteams)
    B[1, :] .= MID.obs_row!(h, nteams, 1.0, home, away)
    B[2, :] .= MID.obs_row!(h, nteams, 0.0, away, home)
    return B
end

"""
Pre-week predictive integrated over 128 θ × 4 joint state draws (R03/R07). Every fixture
is forecast from the training filter only; no held-out book enters, and fixtures with no
book are forecast exactly like quoted ones. One state draw per θ/week/rep is shared by all
fixtures of that calendar week. Returns `CountLatents` with 512 draws.
"""
function forecast_fold(fit, training, full, fold, fixtures)
    index = posterior_indices(fit)
    nteams = MID.n_teams(full)
    team = Dict(name => i for (i, name) in enumerate(full.teams))
    isempty(intersect(Set(fold.ids), Set(fold.train_ids))) || error("target/training overlap")
    n = length(fold.ids)
    designs = Matrix{Float64}[]
    weeks = Int[]
    for id in fold.ids
        f = fixtures[id]
        push!(designs, fixture_design(nteams, team[String(f.home_team)], team[String(f.away_team)]))
        week = searchsortedlast(full.week_start, Date(f.match_date))
        week >= training.target_week || error("fixture $id precedes its fold's cutoff week")
        push!(weeks, week)
    end
    groups = Dict(w => findall(==(w), weeks) for w in unique(weeks))
    home, away = zeros(n, N_THETA * N_STATE), zeros(n, N_THETA * N_STATE)
    arm = PM.FullBookRung(:C0)
    state = zeros(2 + 2nteams)
    for (j, at) in enumerate(index)
        θ = vec(fit.udraws[at.draw, :, at.chain])
        filtered = FG.scalar_filter(arm, training.panel, θ; store = true)
        Qstep = PM.fullbook_process(arm, nteams, θ)
        m = filtered.m_filt[:, end]
        V = filtered.P_filt[:, :, end]
        for week in sort!(collect(keys(groups)))
            steps = week - training.last_week
            steps >= 1 || error("predictor would use the target week's book")
            predictive = Symmetric(V + steps * Qstep)
            rng = Random.Xoshiro(PREDICTION_SEED + 1_000_000fold.number + 1000j + week)
            for rep in 1:N_STATE
                MID.draw_psd!(state, rng, m, predictive)
                column = N_STATE * (j - 1) + rep
                for i in groups[week]
                    rates = exp.(designs[i] * state)
                    all(r -> isfinite(r) && r > 0, rates) || error("nonfinite predictive rate")
                    home[i, column], away[i, column] = rates
                end
            end
        end
    end
    return BayesianFootball.Models.CountLatents(fold.ids, home, away)
end

"Posterior-mean 1X2/OU2.5/BTTS probabilities through the production double-Poisson grid."
function probabilities(latents)
    P = BayesianFootball.Predictions
    markets = (D.Market1X2(), D.MarketOverUnder(2.5), D.MarketBTTS())
    rows = NamedTuple[]
    for (i, id) in enumerate(latents.match_ids)
        grid = P.compute_score_grid(latents, i)
        p = merge([Dict(k => mean(v) for (k, v) in P.price_market(grid, m)) for m in markets]...)
        push!(rows, (; match_id = id,
            lambda_home = mean(view(latents.λ_home, i, :)),
            lambda_away = mean(view(latents.λ_away, i, :)),
            log_supremacy = mean(log.(view(latents.λ_home, i, :)) .- log.(view(latents.λ_away, i, :))),
            p_home = p[:home], p_draw = p[:draw], p_away = p[:away],
            p_over_25 = p[:over_25], p_under_25 = p[:under_25],
            p_btts_yes = p[:btts_yes], p_btts_no = p[:btts_no]))
    end
    return DataFrame(rows)
end

"""
One fold at the frozen Phase C C0 budget (4 × (2000 warmup + 3000)). Outputs are
immutable; the convergence gate (R̂ ≤ 1.05, bulk/tail ESS ≥ 200) precedes any price.
"""
function run_fold(full, fold, fixtures, output; warmup = 2000, samples = 3000,
                  seeds = SEEDS .+ 100fold.number)
    work = joinpath(output, "fold_$(lpad(fold.number, 2, '0'))")
    isdir(work) && !isempty(readdir(work)) && error("immutable market fold exists: $work")
    mkpath(work)
    training = training_panel(full, fold)
    arm = PM.FullBookRung(:C0)
    result = PM.fit_fullbook(arm, training.panel; seeds, warmup, samples, progress = false,
        filter_fn = FG.evaluator(arm, training.panel))
    fit = result.fit
    diag = PM.fullbook_diagnostics(fit; protocol = "fold_$(fold.number)", seed = first(seeds))
    CSV.write(joinpath(work, "convergence.csv"), diag)
    Serialization.serialize(joinpath(work, "fit.jls"), fit)
    if !all(diag.gate_pass)
        println("QS_MARKET_FOLD_GATE_FAIL fold=$(fold.number) rhat=$(maximum(diag.rhat)) ",
                "ess_bulk=$(minimum(diag.ess_bulk)) ess_tail=$(minimum(diag.ess_tail))")
        return (; fit, diag, training, latents = nothing, seconds = fit.seconds)
    end
    started = time()
    latents = forecast_fold(fit, training, full, fold, fixtures)
    Serialization.serialize(joinpath(work, "latents.jls"), latents)
    CSV.write(joinpath(work, "probabilities.csv"), probabilities(latents))
    forecast_seconds = time() - started
    println("QS_MARKET_FOLD_DONE fold=$(fold.number) train_fixtures=$(MID.n_fixtures(training.panel)) ",
            "weeks=$(training.panel.n_weeks) heldout=$(length(fold.ids)) fit_s=$(round(fit.seconds; digits = 1)) ",
            "forecast_s=$(round(forecast_seconds; digits = 1)) rhat=$(round(maximum(diag.rhat); digits = 4)) ",
            "ess_bulk=$(round(minimum(diag.ess_bulk); digits = 0)) seeds=$(join(seeds, ';'))")
    flush(stdout)
    return (; fit, diag, training, latents, seconds = fit.seconds + forecast_seconds)
end

"Concatenate fold latents into one 710-fixture container, refusing gaps or overlaps."
function combine_latents(parts)
    ids = vcat([l.match_ids for l in parts]...)
    length(unique(ids)) == length(ids) || error("market fold latents overlap")
    return BayesianFootball.Models.CountLatents(ids,
        vcat([l.λ_home for l in parts]...), vcat([l.λ_away for l in parts]...))
end

end # module
println("QS_MARKET_ARM_LOADED")

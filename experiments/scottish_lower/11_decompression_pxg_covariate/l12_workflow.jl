# Included inside DecompressionPXG. Infrastructure only; equations live in l11.

function source_fingerprint()
    root = normpath(joinpath(@__DIR__, "../../.."))
    local_files = [
        "l11_decompression_loader.jl",
        "l12_workflow.jl",
        "l13_evaluation.jl",
        "r00_preflight.jl",
        "r10_smoke.jl",
        "r20_production_grid.jl",
        "r30_evaluation.jl",
        "test_decompression.jl",
    ]
    paths = [joinpath(@__DIR__, file) for file in local_files]
    append!(paths, [
        joinpath(root, "Project.toml"),
        joinpath(root, "src/features/pxg.jl"),
        joinpath(root, "src/models/pregame/builder/components.jl"),
        joinpath(root, "src/models/pregame/builder/engine.jl"),
        joinpath(root, "current_development/grw_player_hybrid/l01_loader.jl"),
        joinpath(root, "current_development/grw_player_hybrid/l02_evaluation.jl"),
    ])
    return bytes2hex(SHA.sha256(join(bytes2hex(SHA.sha256(read(path))) for path in paths)))
end

function runtime_config(config::DecompressionConfig; smoke::Bool)
    return GPHConfig(
        experiment = config.experiment,
        smoke_experiment = config.smoke_experiment,
        save_root = config.save_root,
        samples = 800,
        warmup = 800,
        chains = 4,
        smoke_samples = 400,
        smoke_warmup = 400,
        smoke_chains = 4,
        accept_rate = 0.90,
        max_rhat = 1.05,
        strict_rhat = 1.01,
        min_ess = 200.0,
        max_divergence_rate = nextfloat(0.0),
        min_bfmi = 0.30,
        max_treedepth_rate = 0.05,
        max_concurrent_tasks = 16,
        persist_stride = 1,
    )
end

function selected_inputs(ds, splitter, model, config::DecompressionConfig; smoke::Bool)
    inputs = gph_fold_inputs(ds, splitter, model)
    length(inputs.boundaries) == config.expected_folds || error(
        "cohort has $(length(inputs.boundaries)) folds; expected $(config.expected_folds)")
    all_ids = reduce(vcat, [Int.(frame.match_id) for frame in inputs.oos])
    length(all_ids) == config.expected_oos && allunique(all_ids) || error(
        "cohort is not $(config.expected_oos) unique OOS fixtures")
    folds = smoke ? config.smoke_folds : collect(1:config.expected_folds)
    selected = (;
        boundaries = inputs.boundaries[folds],
        feature_sets = inputs.feature_sets[folds],
        oos = inputs.oos[folds],
    )
    filtration = gph_filtration_report(ds, selected)
    all(filtration.ordered) || error("training kickoff must precede held-out kickoff")
    filtration.fold = folds
    return selected, filtration
end

function fit_recipe(config::DecompressionConfig, name, model, splitter, runtime; smoke::Bool)
    sampler = smoke ? gph_smoke_sampler(runtime) : gph_production_sampler(runtime)
    suffix = smoke ? "_smoke_f01_20_40" : ""
    description = if name == "m03_negbin_pxg_covariate"
        "Negative binomial goals with direct antisymmetric proxy-xG form; " *
        "w_pxg is the full log-rate-supremacy coefficient"
    elseif name == "m02_joint_gamma_poisson"
        "TimeDecay(180) two-arm Gamma proxy-xG plus Poisson-goals compression control"
    else
        "TimeDecay(180) pure-Poisson control"
    end
    return FitConfig(
        name = name * suffix,
        model = model,
        splitter = splitter,
        sampler = sampler,
        execution = gph_execution(runtime),
        tags = ["todo024", "scottish-lower", "decompression", "proxy-xg", "reversediff"],
        description = description * "; folds=$(folds_label(config, smoke)); source=" * source_fingerprint(),
        save_dir = joinpath(config.save_root, smoke ? "smoke" : "production", name),
    )
end

folds_label(config, smoke) = smoke ? join(config.smoke_folds, ",") : "1:$(config.expected_folds)"

function register_recipe!(db, name, config)
    tags = config.tags
    save_model(db, name, config.model;
        description = "TODO024 decompression arm $name", tags)
    save_splitter(db, "split_2426", config.splitter; tags)
    save_sampler(db, "sampler_" * config.name, config.sampler; tags)
    save_config(db, "fit_" * config.name, config; tags)
    return nothing
end

function feature_audit(fs)
    data = first(fs).data
    haskey(data, :flat_pxg_supremacy) || error("proxy-xG design column is absent")
    raw = Vector{Float64}(data[:flat_pxg_supremacy])
    design = GPH_PG.covariate_column(pxg_covariate(), first(fs))
    design == 0.5 .* raw || error("candidate design is not exactly half the supremacy feature")
    all(isfinite, design) || error("proxy-xG design contains non-finite values")
    availability = Vector{Float64}(data[:flat_pxg_available])
    all(x -> x == 0.0 || x == 1.0, availability) || error("availability is not binary")
    counts = data[:pxg_source_counts]
    return (;
        n_matches = length(design),
        n_available = count(==(1.0), availability),
        n_neutral = count(iszero, design),
        min_design = minimum(design),
        max_design = maximum(design),
        commentary_observations = get(counts, :commentary, 0),
        shot_count_observations = get(counts, :shot_counts, 0),
        goal_observations = get(counts, :goals, 0),
    )
end

function score_grid_audit(fit)
    latents = fit.latents
    ppd = BayesianFootball.Predictions.model_inference(
        latents, fit.config.model; market_config = PXG_MARKETS)
    frame = ppd.df
    positions = Dict(match_id => i for (i, match_id) in enumerate(latents.match_ids))
    max_goals = BayesianFootball.Predictions.TPL_MAX_GOALS
    worst_partition = 0.0
    worst_tail = 0.0
    for block in groupby(frame, [:match_id, :market_name, :market_line])
        index = positions[first(block.match_id)]
        all(distribution -> all(p -> isfinite(p) && 0.0 <= p <= 1.0, distribution),
            block.distribution) || error("invalid score probability")
        total = reduce(+, block.distribution)
        retained = retained_mass(latents, index, max_goals)
        worst_partition = max(worst_partition, maximum(abs.(total .- retained)))
        worst_tail = max(worst_tail, maximum(1.0 .- retained))
    end
    worst_partition <= 1.0e-12 || error(
        "score partition error $worst_partition exceeds 1e-12")
    return (; worst_partition, worst_tail, grid_goals = max_goals)
end

function retained_mass(latents::CountLatents{T,Nothing}, index, max_goals) where {T}
    home = cdf.(Poisson.(latents.λ_home[index, :]), max_goals - 1)
    away = cdf.(Poisson.(latents.λ_away[index, :]), max_goals - 1)
    return home .* away
end

function retained_mass(latents::CountLatents{T,<:NamedTuple}, index, max_goals) where {T}
    params = latents.observation_params
    robust = BayesianFootball.MyDistributions.RobustNegativeBinomial
    home = cdf.(robust.(params.r_h[index, :], latents.λ_home[index, :]), max_goals - 1)
    away = cdf.(robust.(params.r_a[index, :], latents.λ_away[index, :]), max_goals - 1)
    return home .* away
end

function pxg_posterior(fit, fold_ids)
    rows = NamedTuple[]
    symbol = Symbol("pxg_form.w")
    for (fold_fit, fold_id) in zip(fit.folds, fold_ids)
        symbol in names(fold_fit.chain) || continue
        values = vec(Array(fold_fit.chain[symbol]))
        push!(rows, (;
            fold = fold_id,
            mean = mean(values),
            sd = std(values),
            q05 = quantile(values, 0.05),
            q50 = median(values),
            q95 = quantile(values, 0.95),
            p_positive = mean(>(0.0), values),
            mean_in_band = 0.40 <= mean(values) <= 0.80,
        ))
    end
    return rows
end

function pxg_identification_pass(rows)
    length(rows) > 0 || return false
    return all(row.mean_in_band && row.q05 > 0.0 && row.p_positive >= 0.95 for row in rows)
end

function convergence_pass(fit, n_folds)
    diagnostics = fit.diagnostics
    return length(fit.folds) == n_folds && diagnostics.passed &&
        isempty(diagnostics.abstained) && diagnostics.n_applicable == n_folds &&
        diagnostics.n_divergent == 0 && diagnostics.max_rhat <= 1.05 &&
        diagnostics.min_ess_bulk >= 200 && diagnostics.min_ess_tail >= 200
end

function portfolio_specs()
    book = BookSpec(
        markets = Data.MarketConfig([Data.Market1X2(), Data.MarketOverUnder(2.5)]),
        shrink = BakerMcHale(),
    )
    policy = PolicySpec(
        trust = FlatTrust(0.25),
        risk = SlateDrawdown(20.0),
        cap = FixedCap(0.25),
    )
    return book, policy
end

"Common tradeable panel; only missing quotes/selections may be excluded."
function tradeable_panel(fits, odds, ds)
    book, _ = portfolio_specs()
    reference = sort(copy(first(values(fits)).latents.match_ids))
    refusals = NamedTuple[]
    omitted = Set{Int}()
    for (name, fit) in fits
        sort(fit.latents.match_ids) == reference || error("portfolio latent panels differ")
        _, report = GPH_PORTFOLIO.build_books_reported(
            book, fit, odds, ds; require_converged = false, quiet = true)
        isempty(report.errored) || error("portfolio build errors: $(report.errored)")
        isempty(report.skipped_no_fixture) && isempty(report.skipped_unplayed) ||
            error("portfolio has missing fixture identities or unplayed matches")
        for (ids, reason) in ((report.skipped_no_quotes, "no closing quotes"),
                              (report.skipped_no_selections, "no usable selection"))
            for match_id in ids
                push!(omitted, match_id)
                push!(refusals, (; model = name, match_id, reason))
            end
        end
    end
    panel = sort(collect(setdiff(Set(reference), omitted)))
    isempty(panel) && error("no common tradeable fixtures")
    return panel, DataFrame(refusals)
end

function portfolio_roundtrip(db, run_id, fit, ds, odds; panel)
    book, policy = portfolio_specs()
    priced_fit = gph_restrict(fit, panel)
    save_book_spec(db, "main_1x2_ou25", book)
    save_policy_spec(db, "flat025_drawdown20_cap025", policy)
    result, books, report = run_portfolio_simulation(
        book, policy, priced_fit, odds, ds;
        require_converged = true, quiet = true, bootstrap = false)
    GPH_PORTFOLIO.n_skipped(report) == 0 || error(
        "portfolio skipped $(GPH_PORTFOLIO.n_skipped(report)) fixtures")
    length(books) == length(panel) || error("portfolio fixture count differs")
    portfolio_id = save_portfolio_db(result, run_id, db; book_spec = book, policy_spec = policy)
    restored = load_portfolio_db(portfolio_id, db)
    isequal(restored.trajectory.bets, result.trajectory.bets) || error(
        "persisted bet ledger differs")
    reloaded = load_fit(db, run_id)
    rebuilt, _, _ = run_portfolio_simulation(
        book, policy, gph_restrict(reloaded, panel), odds, ds;
        require_converged = true, quiet = true, bootstrap = false)
    isequal(rebuilt.trajectory.bets, result.trajectory.bets) || error(
        "reloaded fit reprices a different ledger")
    return portfolio_id, result
end

# Included inside DecoupledGenerativeXG. Infrastructure only; equations live in l12.

function source_fingerprint()
    root = normpath(joinpath(@__DIR__, "../../.."))
    local_files = [
        "l12_loader.jl",
        "l13_workflow.jl",
        "l14_evaluation.jl",
        "r00_preflight.jl",
        "r10_smoke.jl",
        "r20_production_grid.jl",
        "r30_evaluation.jl",
        "test_decoupled_xg.jl",
    ]
    paths = [joinpath(@__DIR__, file) for file in local_files]
    append!(paths, [
        joinpath(root, "Project.toml"),
        joinpath(root, "src/models/pregame/builder/components.jl"),
        joinpath(root, "src/models/pregame/builder/engine.jl"),
        joinpath(root, "src/models/pregame/builder/equations.jl"),
        joinpath(root, "current_development/grw_player_hybrid/l01_loader.jl"),
        joinpath(root, "current_development/grw_player_hybrid/l02_evaluation.jl"),
    ])
    all(isfile, paths) || error("source fingerprint inventory contains a missing file")
    return bytes2hex(SHA.sha256(join(bytes2hex(SHA.sha256(read(path))) for path in paths)))
end

function runtime_config(config::FunnelConfig)
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

function selected_inputs(ds, splitter, model, config::FunnelConfig; smoke::Bool)
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

folds_label(config, smoke) = smoke ? join(config.smoke_folds, ",") : "1:$(config.expected_folds)"

function fit_recipe(config::FunnelConfig, name, model, splitter, runtime; smoke::Bool)
    sampler = smoke ? gph_smoke_sampler(runtime) : gph_production_sampler(runtime)
    suffix = smoke ? "_smoke_f01_20_40" : ""
    descriptions = Dict(
        "m01_poisson_time_decay" => "TimeDecay(180) pure-Poisson control",
        "m02_joint_gamma_poisson" =>
            "Canonical shared-latent Gamma proxy-xG plus Poisson-goals control",
        "m03_funnel_shared_kappa" =>
            "Shared-kappa xG-to-goals funnel; mathematically identical to m02 by specification",
        "m04_funnel_hierarchical_kappa" =>
            "xG-to-goals funnel with zero-centred partially pooled team finishing factors",
    )
    return FitConfig(
        name = name * suffix,
        model = model,
        splitter = splitter,
        sampler = sampler,
        execution = gph_execution(runtime),
        tags = ["todo025", "scottish-lower", "decoupled-xg", "funnel", "reversediff"],
        description = descriptions[name] * "; folds=$(folds_label(config, smoke)); source=" *
                      source_fingerprint(),
        save_dir = joinpath(config.save_root, smoke ? "smoke" : "production", name),
    )
end

function register_recipe!(db, name, config)
    tags = config.tags
    save_model(db, name, config.model;
        description = "TODO025 decoupled generative xG arm $name", tags)
    save_splitter(db, "split_2426", config.splitter; tags)
    save_sampler(db, "sampler_" * config.name, config.sampler; tags)
    save_config(db, "fit_" * config.name, config; tags)
    return nothing
end

function proxy_feature_audit(feature_sets)
    data = first(feature_sets).data
    for key in (:flat_pxg_home, :flat_pxg_away, :flat_pxg_obs_available)
        haskey(data, key) || error("joint feature design lacks $key")
    end
    mask = Vector{Float64}(data[:flat_pxg_obs_available])
    home = Vector{Float64}(data[:flat_pxg_home])
    away = Vector{Float64}(data[:flat_pxg_away])
    all(x -> x == 0.0 || x == 1.0, mask) || error("proxy availability is not binary")
    all(>(0.0), home) && all(>(0.0), away) || error("proxy observations leave Gamma support")
    return (;
        n_matches = length(mask),
        n_available = count(==(1.0), mask),
        min_home = minimum(home),
        max_home = maximum(home),
        min_away = minimum(away),
        max_away = maximum(away),
    )
end

function score_grid_audit(fit)
    latents = fit.latents
    ppd = BayesianFootball.Predictions.model_inference(
        latents, fit.config.model; market_config = FUNNEL_MARKETS)
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
        home = cdf.(Poisson.(latents.λ_home[index, :]), max_goals - 1)
        away = cdf.(Poisson.(latents.λ_away[index, :]), max_goals - 1)
        retained = home .* away
        worst_partition = max(worst_partition, maximum(abs.(total .- retained)))
        worst_tail = max(worst_tail, maximum(1.0 .- retained))
    end
    worst_partition <= 1.0e-12 || error(
        "score partition error $worst_partition exceeds 1e-12")
    return (; worst_partition, worst_tail, grid_goals = max_goals)
end

function convergence_pass(fit, n_folds)
    diagnostics = fit.diagnostics
    return length(fit.folds) == n_folds && diagnostics.passed &&
        isempty(diagnostics.abstained) && diagnostics.n_applicable == n_folds &&
        diagnostics.n_divergent == 0 && diagnostics.max_rhat <= 1.05 &&
        diagnostics.min_ess_bulk >= 200 && diagnostics.min_ess_tail >= 200
end

function kappa_posterior(fit, fold_ids)
    rows = NamedTuple[]
    model = fit.config.model
    model.observation isa JointGammaPoissonObservation || return rows
    for (fold_fit, fold_id) in zip(fit.folds, fold_ids)
        chain = fold_fit.chain
        kappa = exp.(vec(Array(chain[Symbol("obs.log_κ")])))
        nu = vec(Array(chain[Symbol("obs.ν")]))
        if model.observation.kappa isa HierarchicalKappa
            sigma = vec(Array(chain[Symbol("obs.σ_κ")]))
            push!(rows, (;
                fold = fold_id,
                mode = "hierarchical",
                kappa_mean = mean(kappa),
                kappa_q05 = quantile(kappa, 0.05),
                kappa_q95 = quantile(kappa, 0.95),
                nu_mean = mean(nu),
                sigma_mean = mean(sigma),
                sigma_q05 = quantile(sigma, 0.05),
                sigma_q95 = quantile(sigma, 0.95),
                p_sigma_gt_005 = mean(>(0.05), sigma),
            ))
        else
            push!(rows, (;
                fold = fold_id,
                mode = "shared",
                kappa_mean = mean(kappa),
                kappa_q05 = quantile(kappa, 0.05),
                kappa_q95 = quantile(kappa, 0.95),
                nu_mean = mean(nu),
                sigma_mean = NaN,
                sigma_q05 = NaN,
                sigma_q95 = NaN,
                p_sigma_gt_005 = NaN,
            ))
        end
    end
    return rows
end

function hierarchical_zero_sum_audit(fit)
    model = fit.config.model
    model.observation isa B.HierarchicalKappaJoint || return 0.0
    worst = 0.0
    for fold in fit.folds
        chain = fold.chain
        sigma = vec(Array(chain[Symbol("obs.σ_κ")]))
        n_teams = count(name -> startswith(String(name), "obs.κ_team_raw["), names(chain))
        raw = hcat([vec(Array(chain[Symbol("obs.κ_team_raw[$team]")]))
                    for team in 1:n_teams]...)
        delta = sigma .* (raw .- mean(raw; dims = 2))
        worst = max(worst, maximum(abs.(vec(sum(delta; dims = 2)))))
    end
    worst <= 1.0e-10 || error("hierarchical finishing deltas do not sum to zero: $worst")
    return worst
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

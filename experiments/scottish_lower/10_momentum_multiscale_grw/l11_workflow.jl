# Included inside MomentumGRW. Infrastructure only; equations live in l10.
const MARKETS = Data.MarketConfig([Data.Market1X2(),Data.MarketOverUnder(2.5),Data.MarketBTTS()])

function source_fingerprint()
    root = normpath(joinpath(@__DIR__, "../../.."))
    paths = [joinpath(@__DIR__,p) for p in ("l10_momentum_grw_loader.jl","l11_workflow.jl","r10_momentum_smoke.jl")]
    append!(paths,[joinpath(root,"Project.toml"),
        joinpath(root,"current_development/grw_player_hybrid/l01_loader.jl")])
    return bytes2hex(SHA.sha256(join(bytes2hex(SHA.sha256(read(p))) for p in paths)))
end

function runtime_config(c; smoke)
    return GPHConfig(experiment=c.experiment, smoke_experiment=c.smoke_experiment,
        save_root=c.save_root, samples=800, warmup=800, chains=4,
        smoke_samples=400, smoke_warmup=400, smoke_chains=4,
        # Shared audit uses strict '<': the smallest positive Float64 accepts
        # zero and no positive representable rate. Exact count is also gated.
        accept_rate=0.90, min_ess=200.0, max_divergence_rate=nextfloat(0.0),
        max_concurrent_tasks=16, persist_stride=1)
end

function selected_inputs(ds, splitter, model, c; smoke)
    inputs = gph_fold_inputs(ds,splitter,model)
    length(inputs.boundaries) == c.expected_folds || error("cohort no longer has 40 folds")
    ids = reduce(vcat,[Int.(f.match_id) for f in inputs.oos])
    length(ids) == c.expected_oos && allunique(ids) || error("cohort is not 710 unique fixtures")
    folds = smoke ? c.smoke_folds : collect(1:c.expected_folds)
    selected = (; boundaries=inputs.boundaries[folds],feature_sets=inputs.feature_sets[folds],
                  oos=inputs.oos[folds])
    filtration = gph_filtration_report(ds,selected)
    all(filtration.ordered) || error("training kickoff must precede held-out kickoff")
    filtration.fold = folds
    return selected, filtration
end

function fit_recipe(c, name, model, splitter, runtime; smoke)
    sampler = smoke ? gph_smoke_sampler(runtime) : gph_production_sampler(runtime)
    suffix = smoke ? "_smoke_f01_20_40" : ""
    return FitConfig(name=name*suffix, model=model,splitter=splitter,sampler=sampler,
        execution=gph_execution(runtime), tags=["todo022","poisson","momentum","reversediff"],
        description="Momentum Phase 1 v1; conditional mean; zero boundary velocity; " *
            "folds=$(smoke ? c.smoke_folds : collect(1:40)); source=" * source_fingerprint(),
        save_dir=joinpath(c.save_root,smoke ? "smoke" : "production",name))
end
function register_recipe!(db,name,config)
    tags = config.tags
    save_model(db,name,config.model; description="TODO022 pure Poisson benchmark",tags)
    save_splitter(db,"split_2426",config.splitter;tags)
    save_sampler(db,"sampler_"*config.name,config.sampler;tags)
    save_config(db,"fit_"*config.name,config;tags)
end

function portfolio_specs()
    book = BookSpec(markets=Data.MarketConfig([Data.Market1X2(),Data.MarketOverUnder(2.5)]),
        shrink=BakerMcHale())
    policy = PolicySpec(trust=FlatTrust(0.25),risk=SlateDrawdown(20.0),cap=FixedCap(0.25))
    return book,policy
end

"Every production market partition equals its analytic retained mass, per draw."
function score_grid_audit(fit)
    lat = fit.latents
    ppd = BayesianFootball.Predictions.model_inference(lat,fit.config.model;market_config=MARKETS)
    df = ppd.df
    positions = Dict(m=>i for (i,m) in enumerate(lat.match_ids))
    g = BayesianFootball.Predictions.TPL_MAX_GOALS
    worst_partition = 0.0
    worst_tail = 0.0
    for block in groupby(df,[:match_id,:market_name,:market_line])
        i = positions[first(block.match_id)]
        all(p->all(x->isfinite(x) && 0<=x<=1,p),block.distribution) || error("invalid score probability")
        total = reduce(+,block.distribution)
        retained = cdf.(Poisson.(lat.λ_home[i,:]),g-1) .* cdf.(Poisson.(lat.λ_away[i,:]),g-1)
        worst_partition = max(worst_partition,maximum(abs.(total-retained)))
        worst_tail = max(worst_tail,maximum(1 .- retained))
    end
    worst_partition <= 1e-12 || error("score partition error $worst_partition > 1e-12")
    return (; worst_partition, worst_tail, grid_goals=g)
end

function momentum_posterior(fit, fold_ids)
    rows = NamedTuple[]
    for (fold,label) in zip(fit.folds,fold_ids), side in ("α","β"), param in ("φ","σᵥ")
        symbol = Symbol("dyn.$side.$param")
        symbol in names(fold.chain) || continue
        values = vec(Array(fold.chain[symbol]))
        prior = param == "φ" ? fit.config.model.dynamics.persistence :
            (side == "α" ? fit.config.model.dynamics.attack_velocity : fit.config.model.dynamics.defence_velocity)
        push!(rows,(; fold=label, side, parameter=param, mean=mean(values),sd=std(values),
            q05=quantile(values,0.05),q50=median(values),q95=quantile(values,0.95),
            prior_mean=mean(prior),prior_sd=std(prior)))
    end
    return rows
end

function convergence_pass(fit, n_folds)
    d = fit.diagnostics
    return length(fit.folds)==n_folds && d.passed && isempty(d.abstained) &&
        d.n_applicable==n_folds && d.n_divergent==0 &&
        d.max_rhat<=1.05 && d.min_ess_bulk>=200 && d.min_ess_tail>=200
end

function portfolio_roundtrip(db,run_id,fit,ds,odds)
    book,policy = portfolio_specs()
    save_book_spec(db,"main_1x2_ou25",book)
    save_policy_spec(db,"flat025_drawdown20_cap025",policy)
    result,books,report = run_portfolio_simulation(book,policy,fit,odds,ds;
        require_converged=true,quiet=true,bootstrap=false)
    GPH_PORTFOLIO.n_skipped(report)==0 || error("portfolio skipped fixtures: $(GPH_PORTFOLIO.n_skipped(report))")
    length(books)==n_matches(fit.latents) || error("portfolio fixture count differs")
    portfolio_id = save_portfolio_db(result,run_id,db;book_spec=book,policy_spec=policy)
    restored = load_portfolio_db(portfolio_id,db)
    isequal(restored.trajectory.bets,result.trajectory.bets) || error("persisted bet ledger differs")
    reloaded = load_fit(db,run_id)
    rebuilt,_,_ = run_portfolio_simulation(book,policy,reloaded,odds,ds;
        require_converged=true,quiet=true,bootstrap=false)
    isequal(rebuilt.trajectory.bets,result.trajectory.bets) || error("reloaded fit reprices a different ledger")
    return portfolio_id,result
end

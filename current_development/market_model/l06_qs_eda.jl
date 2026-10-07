module QualityStyleEDA

import BayesianFootball
import DataFrames
import Dates
import Statistics
import Distributions
import CSV
import SHA
import TOML
import Serialization
import LinearAlgebra
import Random

if !isdefined(parentmodule(@__MODULE__), :PooledMarket)
    Base.include(parentmodule(@__MODULE__), joinpath(@__DIR__, "l05_pooled.jl"))
end
if !isdefined(parentmodule(@__MODULE__), :FastGaussianBook)
    Base.include(parentmodule(@__MODULE__), joinpath(@__DIR__, "l05_fast_gaussian.jl"))
end
if !isdefined(parentmodule(@__MODULE__), :FullBookWorkflow)
    Base.include(parentmodule(@__MODULE__), joinpath(@__DIR__, "l05_fullbook_workflow.jl"))
end
if !isdefined(parentmodule(@__MODULE__), :FastFullBookReports)
    Base.include(parentmodule(@__MODULE__), joinpath(@__DIR__, "l05_fast_reports.jl"))
end
const PM = parentmodule(@__MODULE__).PooledMarket
const MM = PM.MM
const MID = PM.MID
const CM = PM.CM
const TB = CM.TB
const FG = parentmodule(@__MODULE__).FastGaussianBook
const FW = parentmodule(@__MODULE__).FullBookWorkflow
const FR = parentmodule(@__MODULE__).FastFullBookReports
const LA = LinearAlgebra
const DF = DataFrames
const ST = Statistics

"Stream-local four-tier England segment; each analysis selects exactly one tournament."
struct MarketModelEnglish <: BayesianFootball.Data.DataTournemantSegment end
BayesianFootball.Data.tournament_ids(::MarketModelEnglish) = [1, 2, 3, 84]

const SNAPSHOT_HASHES = Dict(
    "MarketModelEnglish"=>"2efe200ca6ed2899a75246570e025d5782af5da945ec26100bfa38331c5168bf",
    "ScottishUpper"=>"56207e6354f0e9fbf5a6d1e29a277ae49b50769d2975de7a7e7ccca85c93630a",
    "ScottishLower"=>"e894ac6d59f912aceb01f639c80be37555a73f4eab2fa24271cddd801f13006a",
    "IrelandAll"=>"9cc59b002982fc7d313bda0c1d821a31e98dbdd91827bfcdb2ec30dd209cc257",
    "Veikkausliiga"=>"e4ef1c33939c28b88571b9d0256a3be8d8b83f88af82a7ff83c85f1315d35b1c",
)

const EXCLUSIONS = Dict(
    "sco_championship" => Dict("25/26" => "Betfair closes stop 2025-09-13 (7/175 matches); materially early"),
    "irl_premier" => Dict("2026" => "2026 in progress"),
    "irl_first_division" => Dict("2026" => "2026 in progress"),
    "fin_veikkausliiga" => Dict("2026" => "2026 in progress"),
)

const LEAGUES = (
    ("eng_premier", 1, MarketModelEnglish()),
    ("eng_championship", 2, MarketModelEnglish()),
    ("eng_league_one", 3, MarketModelEnglish()),
    ("eng_league_two", 84, MarketModelEnglish()),
    ("sco_premiership", 54, BayesianFootball.Data.ScottishUpper()),
    ("sco_championship", 55, BayesianFootball.Data.ScottishUpper()),
    ("sco_league_one", 56, BayesianFootball.Data.ScottishLower()),
    ("sco_league_two", 57, BayesianFootball.Data.ScottishLower()),
    ("irl_premier", 79, BayesianFootball.Data.IrelandAll()),
    ("irl_first_division", 718, BayesianFootball.Data.IrelandAll()),
    ("fin_veikkausliiga", 31, BayesianFootball.Data.Veikkausliiga()),
)

"Inventory all seasons and quoted closes before excluding partial seasons from BOTH protocols."
function season_inventory(ds, tournament)
    matches = DF.filter(r -> !ismissing(r.tournament_id) && r.tournament_id == tournament && !ismissing(r.season), ds.matches)
    rows = NamedTuple[]
    for g in DF.groupby(matches, :season)
        push!(rows, (; season=String(first(g.season)), n_matches=DF.nrow(g),
            first_date=minimum(Dates.Date.(g.match_date)), last_date=maximum(Dates.Date.(g.match_date))))
    end
    return DF.sort!(DF.DataFrame(rows), :first_date)
end

"Use observed chronological seasons; caller supplies explicit exclusions with documented reasons."
function league_config(name, tournament, segment, inventory;
                       exclusions=collect(keys(get(EXCLUSIONS,name,Dict{String,String}()))),
                       quoted_seasons=inventory.season)
    seasons = filter(s -> s in quoted_seasons && !(s in exclusions) &&
        !(s in ("26/27","2026")), collect(inventory.season))
    length(seasons) >= 2 || error("$name needs at least two complete seasons")
    return MM.MarketModelConfig(; name, segment, tournaments=[tournament], seasons,
        honest_train=seasons[1:end-1], honest_test=[last(seasons)])
end

"Snapshot provenance without writing or exposing a database URL."
function snapshot_hash(segment)
    path = joinpath(Base.pkgdir(BayesianFootball.Data), ".cache",
        "datastore_$(nameof(typeof(segment))).jls")
    isfile(path) || error("missing pinned snapshot $path")
    return bytes2hex(open(SHA.sha256,path))
end

"Phase A gates and exact inversion, with a per-season reason/line inventory."
function coverage_and_rates(ds, config, raw, book, refusals)
    rates = MM.invert_panel(ds,book; config)
    rows = NamedTuple[]
    panel_ids = Set(rates.match_id)
    valid = DF.filter(r -> r.match_id in panel_ids,book)
    draws = MM.residual_rows(rates,valid)
    draws = DF.filter(r -> r.line == "1X2" && r.selection == "draw",draws)
    draw_map = Dict(r.match_id=>r.residual for r in eachrow(draws))
    for season in config.seasons
        g = DF.filter(:season=>==(season),rates)
        ids = Set(g.match_id)
        season_book = DF.filter(r -> r.match_id in ids,book)
        season_ref = DF.filter(r -> r.match_id in ids,refusals)
        season_raw = DF.filter(r -> r.match_id in ids,raw)
        refused = DF.combine(DF.groupby(season_ref,:reason),DF.nrow=>:n)
        refusal_reasons = join(sort(["$(r.reason)=$(r.n)" for r in eachrow(refused)]),";")
        groups = DF.groupby(season_book,:match_id)
        only1x2 = count(x -> all(==("1X2"),x.market_name),groups)
        totalsonly = count(x -> !any(==("1X2"),x.market_name) &&
            any(==("OverUnder"),x.market_name),groups)
        mix = DF.combine(DF.groupby(season_book,[:market_name,:market_line]),
            :match_id=>(x->length(unique(x)))=>:fixtures)
        mix_text = join(["$(r.market_name):$(r.market_line)=$(r.fixtures)" for r in eachrow(mix)],";")
        overround = [sum(1 ./ Float64.(m.odds_close)) for m in
            DF.groupby(season_raw,[:match_id,:market_name,:market_line])]
        isempty(overround) && error("$(config.name) $season has no quoted closes")
        residuals = [draw_map[id] for id in g.match_id if haskey(draw_map,id)]
        reasons = sort(unique(String.(g.reason)))
        for reason in reasons
            n = count(==(reason),g.reason)
            push!(rows,(; league=config.name,tournament=only(config.tournaments),season,
                snapshot_sha256=snapshot_hash(config.segment),n_fixtures=DF.nrow(g),
                n_raw_quoted=length(unique(season_raw.match_id)),
                n_quoted=length(groups),n_accepted=count(g.accepted),
                n_refused_markets=DF.nrow(season_ref),market_refusal_reasons=refusal_reasons,
                n_1x2_only=only1x2,
                n_totals_only=totalsonly,share_1x2_only=only1x2/length(groups),
                share_totals_only=totalsonly/length(groups),line_mix=mix_text,
                overround_q05=ST.quantile(overround,0.05),
                overround_median=ST.median(overround),
                overround_q95=ST.quantile(overround,0.95),
                draw_grid_minus_book=isempty(residuals) ? NaN : ST.mean(residuals),
                reason=isempty(reason) ? "accepted" : reason,reason_count=n))
        end
    end
    return DF.DataFrame(rows),rates
end

"Prescribed three-rung posterior fits; convergence refuses all downstream inference."
function train_league(panel,config,out)
    mkpath(out)
    seeds = TOML.parsefile(joinpath(@__DIR__,"fullbook_seeds.toml"))
    fits = Dict{Tuple{String,String},MID.ArmFit}()
    for protocol in ("10b","10a")
        train = protocol == "10a" ? panel : MID.restrict_panel(panel,
            in.(panel.obs_season,Ref(config.honest_train)))
        for rung in ("R2","R6","C0")
            start = time()
            if rung == "R2"
                fit = TB.fit_rung(TB.Rung(2),train;
                    seed=3922+200+(protocol == "10b" ? 1 : 0))
            elseif rung == "R6"
                fit = CM.fit_covariance(CM.CovarianceRung(6),train;
                    seed=3922+600+(protocol == "10b" ? 1 : 0))
            else
                fit = FW.train_rung(PM.FullBookRung(:C0),train,nothing,out;
                    protocol,seeds=seeds["C0"][protocol],
                    filter_fn=FG.evaluator(PM.FullBookRung(:C0),train))
            end
            if rung != "C0"
                diag = MID.convergence_table(fit)
                all((diag.rhat .<= 1.05) .& (diag.ess_bulk .>= 200) .&
                    (diag.ess_tail .>= 200)) || error("$rung $protocol convergence failed")
                Serialization.serialize(joinpath(out,"$(rung)_$(protocol).jls"),fit)
                CSV.write(joinpath(out,"convergence_$(rung)_$(protocol).csv"),diag)
            end
            fits[(rung,protocol)] = fit
            println("QS $(config.name) $rung $protocol wall=",time()-start)
            flush(stdout)
        end
    end
    return fits
end

"Conditional Gaussian marginal predictions at median theta, before any same-week updates."
function linear_forecasts(panel,config,fits)
    rows = NamedTuple[]
    test = Set(panel.matches.match_id[in.(panel.matches.season,Ref(config.honest_test))])
    for protocol in ("10b","10a"), rung in ("R2","R6","C0")
        fit = fits[(rung,protocol)]
        theta = MID.median_theta(fit)
        if rung == "R2"
            # TODO 023's filter stores only marginal side variances; B2's own
            # pre-week predictor retains the cross-covariance needed by d and ell.
            pred = TB.preweek_predictions(fit.arm,panel,theta)
            for r in eachrow(pred.rows)
                r.axis in ("supremacy","level") || continue
                protocol == "10b" && !(r.match_id in test) && continue
                push!(rows,(; league=config.name,rung,protocol,match_id=r.match_id,
                    axis=r.axis,observed=r.observed,predicted=r.predicted,
                    variance=r.variance,logpd=r.logpd,cover90=r.cover90))
            end
            continue
        end
        f = rung == "R6" ? CM.covariance_filter(fit.arm,panel,theta; predict=true) :
            PM.fullbook_filter(fit.arm,panel,theta; predict=true)
        for fixture in 1:MID.n_fixtures(panel)
            j = 2fixture-1
            id = panel.obs_match[j]
            protocol == "10b" && !(id in test) && continue
            yh,ya = panel.obs_y[j:j+1]
            ph,pa = f.pair_mean[fixture,:]
            vh,va,cross = f.pair_cov[fixture,1,1],f.pair_cov[fixture,2,2],f.pair_cov[fixture,1,2]
            for (axis,observed,predicted,variance) in
                (("supremacy",yh-ya,ph-pa,vh+va-2cross),
                 ("level",(yh+ya)/2,(ph+pa)/2,(vh+va+2cross)/4))
                variance > 0 || error("$rung $id has nonpositive $axis variance")
                dist = Distributions.Normal(predicted,sqrt(variance))
                cdf = Distributions.cdf(dist,observed)
                push!(rows,(; league=config.name,rung,protocol,match_id=id,axis,
                    observed,predicted,variance,
                    logpd=Distributions.logpdf(dist,observed),
                    cover90=0.05<=cdf<=0.95))
            end
        end
    end
    return DF.DataFrame(rows)
end

"Coverage of the nominal 90% interval on both separately scored axes."
function forecast_coverage(rows)
    return DF.combine(DF.groupby(rows,[:league,:rung,:protocol,:axis]),
        DF.nrow=>:n,:cover90=>ST.mean=>:coverage90,:logpd=>ST.mean=>:mean_logpd)
end

"Same-fixture 10b sum of supremacy/level marginal log densities and fixture SE."
function paired_forecasts(rows)
    selected = DF.filter(:protocol=>==("10b"),rows)
    counts = DF.combine(DF.groupby(selected,[:league,:rung,:match_id]),DF.nrow=>:n)
    all(==(2),counts.n) || error("each fixture needs exactly supremacy and level")
    scored = DF.combine(DF.groupby(selected,
        [:league,:rung,:match_id]),:logpd=>sum=>:score)
    out = NamedTuple[]
    for (candidate,control) in (("C0","R2"),("R6","R2"),("C0","R6"))
        a = DF.filter(:rung=>==(candidate),scored)
        b = DF.select(DF.filter(:rung=>==(control),scored),:league,:match_id,:score=>:control)
        joined = DF.innerjoin(a,b; on=[:league,:match_id],order=:left)
        DF.nrow(joined) == DF.nrow(a) == DF.nrow(b) || error("paired fixtures differ")
        delta = joined.score-joined.control
        push!(out,(; league=only(unique(joined.league)),candidate,control,
            n=length(delta),mean_delta=ST.mean(delta),se=ST.std(delta)/sqrt(length(delta)),
            score="sum of supremacy and level marginal log densities; fixture SE"))
    end
    return DF.DataFrame(out)
end

"90% posterior intervals of physical and per-draw rotated quantities, plus chain diagnostics."
function parameter_rows(config,fits)
    rows = NamedTuple[]
    for protocol in ("10b","10a"), rung in ("R2","R6","C0")
        fit = fits[(rung,protocol)]
        transformed = Pair{String,Array{Float64,2}}[]
        for (j,name) in enumerate(fit.names)
            push!(transformed,name=>fit.draws[:,j,:])
        end
        if rung == "R6"
            ia,ib,ir = (only(findall(==(name),fit.names)) for name in
                ("sigma_att","sigma_def","rho_ab"))
            transforms = [rotated_step(fit.draws[i,ia,c],fit.draws[i,ib,c],fit.draws[i,ir,c])
                for i in axes(fit.draws,1), c in axes(fit.draws,3)]
            for (key,label) in ((:sigma_q,"sigma_q"),(:sigma_s,"sigma_s"),
                (:r,"r"),(:corr_qs,"corr_delta_q_s"))
                push!(transformed,label=>getproperty.(transforms,key))
            end
        elseif rung == "C0"
            iq,is = (only(findall(==(name),fit.names)) for name in ("sigma_q","sigma_s"))
            push!(transformed,"r"=>fit.draws[:,is,:]./fit.draws[:,iq,:])
        end
        for (name,values) in transformed
            ch = MID.MCMCChains.Chains(reshape(values,size(values,1),1,size(values,2)),[name])
            diag = DF.DataFrame(MID.MCMCChains.summarystats(ch))
            q = ST.quantile(vec(values),[0.05,0.5,0.95])
            push!(rows,(; league=config.name,rung,protocol,parameter=name,
                q05=q[1],median=q[2],q95=q[3],rhat=diag.rhat[1],
                ess_bulk=diag.ess_bulk[1],ess_tail=diag.ess_tail[1],
                method="all retained draws; transformed before summarising"))
        end
    end
    return DF.DataFrame(rows)
end

"Exact mixture interval for static C0 gamma; conditional seeded draws supply Rhat/ESS."
function gamma_row(config,fit,p,protocol)
    U = reshape(permutedims(fit.udraws,(2,1,3)),length(fit.names),:)
    n,chains = size(fit.draws,1),size(fit.draws,3)
    means,sds = zeros(n*chains),zeros(n*chains)
    evaluator = FG.evaluator(fit.arm,p)
    Threads.@threads for d in axes(U,2)
        f = evaluator(fit.arm,p,U[:,d]; store=true)
        means[d] = f.m_filt[2,end]
        sds[d] = sqrt(f.P_filt[2,2,end])
    end
    q = FR.normal_mixture_quantiles(means,sds)
    samples = means .+ sds .* Random.randn(Random.Xoshiro(3965),length(means))
    ch = MID.MCMCChains.Chains(reshape(samples,n,1,chains),["gamma"])
    diag = DF.DataFrame(MID.MCMCChains.summarystats(ch))
    return DF.DataFrame([(; league=config.name,rung="C0",protocol,parameter="gamma",
        q05=q[1],median=q[2],q95=q[3],rhat=diag.rhat[1],
        ess_bulk=diag.ess_bulk[1],ess_tail=diag.ess_tail[1],
        method="exact conditional Gaussian mixture over all hyperdraws; seeded conditional gamma draws for MCMC diagnostics")])
end

"B2 season-average RTS levels across active teams, centred per week at median theta."
function level_geometry(panel,config,fits)
    summaries,teams = NamedTuple[],NamedTuple[]
    N = MID.n_teams(panel)
    for protocol in ("10a","10b"), rung in ("R2","R6","C0")
        fit = fits[(rung,protocol)]
        theta = MID.median_theta(fit)
        if rung == "R2"
            f = TB.rung_filter(fit.arm,panel,theta; store=true)
            X = MID.smoothed_mean(fit.arm,panel,TB.rung_schedule(fit.arm,panel,theta),f)
        elseif rung == "R6"
            f = CM.covariance_filter(fit.arm,panel,theta; store=true)
            X = MID.smoothed_mean(CM.conditional_arm(fit.arm,theta),panel,
                CM.covariance_schedule(fit.arm,panel,theta),f)
        else
            f = PM.fullbook_filter(fit.arm,panel,theta; store=true)
            X,_ = PM.fullbook_smoothing(f)
        end
        for season in config.seasons
            matches = DF.filter(:season=>==(season),panel.matches)
            DF.nrow(matches) > 0 || error("$(config.name) $season has no structural fixtures")
            active = sort(unique(vcat(matches.home_team,matches.away_team)))
            indices = [only(findall(==(team),panel.teams)) for team in active]
            ow = panel.obs_week[panel.obs_season .== season]
            weeks = minimum(ow):maximum(ow)
            attack = X[3:2+N,weeks]
            defence = X[3+N:2+2N,weeks]
            attack .-= ST.mean(attack; dims=1)
            defence .-= ST.mean(defence; dims=1)
            a = vec(ST.mean(attack[indices,:]; dims=2))
            b = vec(ST.mean(defence[indices,:]; dims=2))
            q, s = (a-b)/2,(a+b)/2
            rho = ST.cor(a,b)
            for (i,team) in enumerate(active)
                push!(teams,(; league=config.name,rung,protocol,season,team,
                    alpha=a[i],beta=b[i],q=q[i],s=s[i]))
            end
            push!(summaries,(; league=config.name,rung,protocol,season,
                n_teams=length(active),n_weeks=length(weeks),corr_alpha_beta=rho,
                implied_r=sqrt((1+rho)/(1-rho)),corr_q_s=ST.cor(q,s),
                measured_r=ST.std(s)/ST.std(q),
                method="season-average RTS point means; implied r assumes zero q/s correlation"))
        end
    end
    return DF.DataFrame(summaries),DF.DataFrame(teams)
end

"Honest 10b goal check from the *pre-week* C0 state, never smoothed paths."
function goal_fixtures(ds,panel,config,fit)
    theta = MID.median_theta(fit)
    f = PM.fullbook_filter(fit.arm,panel,theta; store=true,predict=true)
    initial,_ = PM.fullbook_initial(fit.arm,MID.n_teams(panel))
    N = MID.n_teams(panel)
    matches = Dict(Int(row.match_id)=>row for row in eachrow(ds.matches))
    test = Set(panel.matches.match_id[in.(panel.matches.season,Ref(config.honest_test))])
    rows = NamedTuple[]
    for fixture in 1:MID.n_fixtures(panel)
        j = 2fixture-1
        id = panel.obs_match[j]
        id in test || continue
        t = panel.obs_week[j]
        state = t == 1 ? initial : f.m_filt[:,t-1]
        h,away = panel.obs_att[j],panel.obs_att[j+1]
        alpha = state[3:2+N] .- ST.mean(state[3:2+N])
        beta = state[3+N:2+2N] .- ST.mean(state[3+N:2+2N])
        q, s = (alpha-beta)/2,(alpha+beta)/2
        arguments = (state[1],state[2],q[h],q[away],s[h],s[away])
        full = goal_rates(arguments...)
        maximum(abs.(collect(full).-exp.(f.pair_mean[fixture,:]))) < 1e-9 ||
            error("C0 goal decomposition disagrees with pre-week filter: $id")
        no_style = goal_rates(arguments...; zero_style=true)
        no_quality = goal_rates(arguments...; zero_quality=true)
        m = matches[id]
        (ismissing(m.home_score) || ismissing(m.away_score)) &&
            error("honest fixture $id has no final score")
        yh,ya = Int(m.home_score),Int(m.away_score)
        full_score = goal_logscore(yh,ya,full)
        push!(rows,(; league=config.name,match_id=id,season=only(config.honest_test),
            goals_home=yh,goals_away=ya,predicted_home=full[1],predicted_away=full[2],
            expected_total=sum(full),full_logpd=full_score,
            no_style_logpd=goal_logscore(yh,ya,no_style),
            no_quality_logpd=goal_logscore(yh,ya,no_quality)))
    end
    return DF.DataFrame(rows)
end

"Paired goal differences and Poisson calibration slope; pooled row is exploratory too."
function goal_summary(fixtures)
    rows = NamedTuple[]
    for league in vcat(sort(unique(fixtures.league)),["all_leagues_pooled"])
        g = league == "all_leagues_pooled" ? fixtures : DF.filter(:league=>==(league),fixtures)
        n = DF.nrow(g)
        n > 1 || error("goal check needs multiple fixtures for $league")
        for (comparison,col) in (("full_minus_no_style",:no_style_logpd),
                                 ("full_minus_no_quality",:no_quality_logpd))
            delta = g.full_logpd-g[!,col]
            push!(rows,(; league,comparison,n,estimate=ST.mean(delta),
                se=ST.std(delta)/sqrt(n),method="paired Poisson goal log score; fixture SE"))
        end
        regression = TB.poisson_glm(g.goals_home+g.goals_away,log.(g.expected_total))
        push!(rows,(; league,comparison="total_goal_calibration_slope",n,
            estimate=regression.coef[2],se=regression.se[2],
            method="Poisson log-link GLM on log predicted expected total; ideal slope 1; model-based SE"))
    end
    return DF.DataFrame(rows)
end

"Posterior draw correlation between style-step scale and Gaussian fixture noise."
function style_noise_correlation(config,fits)
    rows = NamedTuple[]
    for protocol in ("10a","10b")
        fit = fits[("C0",protocol)]
        obs = vec(fit.draws[:,only(findall(==("sigma_obs"),fit.names)),:])
        style = vec(fit.draws[:,only(findall(==("sigma_s"),fit.names)),:])
        push!(rows,(; league=config.name,protocol,n_draws=length(obs),
            posterior_corr_sigma_s_sigma_obs=ST.cor(style,obs)))
    end
    return DF.DataFrame(rows)
end

"Six fixed-budget C0 synthetic refits: 3 generation seeds in each sharp/thin league."
function c0_recovery(panel,templates,config,fit,out)
    config.name in ("eng_premier","sco_league_two") || error("recovery only for prescribed leagues")
    a = PM.FullBookRung(:C0)
    truth = [ST.median(vec(fit.draws[:,j,:])) for j in 1:3]
    seeds = TOML.parsefile(joinpath(@__DIR__,"fullbook_seeds.toml"))["C0"]["10a"]
    rows = NamedTuple[]
    for generation_seed in 3962:3964
        generated = PM.synthetic_fullbook(a,panel,templates,log.(truth); seed=generation_seed)
        protocol = "synthetic_$(generation_seed)"
        recovery = FW.train_rung(a,generated.panel,nothing,out;
            protocol,seeds,filter_fn=FG.evaluator(a,generated.panel))
        for (j,parameter) in enumerate(recovery.names)
            q = ST.quantile(vec(recovery.draws[:,j,:]),[0.05,0.5,0.95])
            push!(rows,(; league=config.name,generation_seed,parameter,
                truth=truth[j],q05=q[1],median=q[2],q95=q[3],
                recovered=q[1]<=truth[j]<=q[3],
                n_fixtures=MID.n_fixtures(panel),n_weeks=panel.n_weeks,
                n_teams=MID.n_teams(panel),
                budget="4 chains x (2000 warmup + 3000 draws), thin 1"))
        end
    end
    return DF.DataFrame(rows)
end

"Per-draw R6 rotation, retaining both scales and the rotated cross-correlation."
function rotated_step(sigma_att, sigma_def, rho)
    abs(rho) < 1 || error("rho must be interior")
    vq = (sigma_att^2+sigma_def^2-2rho*sigma_att*sigma_def)/4
    vs = (sigma_att^2+sigma_def^2+2rho*sigma_att*sigma_def)/4
    sq, ss = sqrt(vq), sqrt(vs)
    return (; sigma_q=sq, sigma_s=ss, r=ss/sq,
        corr_qs=(sigma_att^2-sigma_def^2)/(4sq*ss))
end

"C0 fixture means in three nested goal-model variants; no future book or outcome enters."
function goal_rates(mu, gamma, q_home, q_away, s_home, s_away;
                    zero_quality=false, zero_style=false)
    quality = zero_quality ? 0.0 : q_home-q_away
    style = zero_style ? 0.0 : s_home+s_away
    return (exp(mu+gamma+quality+style), exp(mu-quality+style))
end

"Double-Poisson goal log score for the predicted home/away rate pair."
goal_logscore(goals_home, goals_away, rates) =
    Distributions.logpdf(Distributions.Poisson(rates[1]), goals_home) +
    Distributions.logpdf(Distributions.Poisson(rates[2]), goals_away)

end # module

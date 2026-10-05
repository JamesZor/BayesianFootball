# Included inside TwoStageMarket; definitions only.

function phase_b_figures(P,panel,distribution,calibration,variance,out,fig)
    p = P.plot(layout=(1,3),size=(1200,400))
    for (k,axis) in enumerate(("home","away","total"))
        g = DF.filter(:axis=>==(axis),calibration)
        P.scatter!(p[k],g.mean_lambda,g.mean_goals; yerror=(g.mean_goals.-g.ci_low,g.ci_high.-g.mean_goals),
                   label="fixture-bootstrap 95% CI",xlabel="mean inverted rate",ylabel="mean full-time goals",title=axis)
        limits = extrema(vcat(g.mean_lambda,g.mean_goals))
        P.plot!(p[k],collect(limits),collect(limits); label="calibrated",color=:black)
    end
    P.savefig(p,joinpath(fig,"B_goal_calibration.png"))
    p = P.plot(layout=(1,2),size=(1000,400))
    P.scatter!(p[1],variance.fitted_mean,variance.variance; label="",xlabel="R2 fitted exp(eta)",ylabel="var(inverted rate)")
    P.scatter!(p[2],variance.fitted_mean,variance.cv; label="",xlabel="R2 fitted exp(eta)",ylabel="CV(inverted rate)")
    P.savefig(p,joinpath(fig,"B_mean_variance.png"))
    qq = CSV.read(joinpath(out,"family_qq.csv"),DF.DataFrame)
    p = P.plot(layout=(1,3),size=(1200,400))
    for (k,family) in enumerate(("gamma","lognormal","logt"))
        g = DF.filter(:family=>==(family),qq)
        P.scatter!(p[k],g.theoretical,g.standardised_residual; label="",markersize=2,xlabel="family quantile",ylabel="standardised residual",title=family)
        limits = extrema(g.theoretical)
        P.plot!(p[k],collect(limits),collect(limits); label="reference",color=:black)
    end
    P.savefig(p,joinpath(fig,"B_family_qq.png"))
    p = P.plot(layout=(2,2),size=(1000,700))
    h = panel.matches.lambda_mkt_h
    a = panel.matches.lambda_mkt_a
    for (k,(axis,values)) in enumerate((("lambda_h",h),("lambda_a",a),("lambda_total",h.+a),("log supremacy",log.(h).-log.(a))))
        P.histogram!(p[k],values; bins=25,label="",title=axis,ylabel="fixtures")
    end
    P.savefig(p,joinpath(fig,"B_rate_distributions.png"))
    return nothing
end

"Fixture bootstrap percentile 95% interval, deterministic per table."
function mean_ci(values; reps=2000,seed=3910)
    rng = Random.Xoshiro(seed)
    boot = [ST.mean(values[rand(rng,eachindex(values),length(values))]) for _ in 1:reps]
    return ST.quantile(boot,0.025),ST.quantile(boot,0.975)
end

function rate_eda(panel,exclusions,out)
    m = panel.matches
    rows = NamedTuple[]
    advantage = NamedTuple[]
    correlations = NamedTuple[]
    for scope in ("overall","tournament","season","month","tournament-season")
        group = scope == "overall" ? fill("all",DF.nrow(m)) : scope == "tournament" ? string.(m.tournament_id) :
            scope == "season" ? m.season : scope == "month" ? Dates.format.(m.match_date,"yyyy-mm") :
            string.(m.tournament_id).*"/".*m.season
        for key in sort(unique(group))
            g = m[group.==key,:]
            h = g.lambda_mkt_h
            a = g.lambda_mkt_a
            for (axis,values) in (("lambda_h",h),("lambda_a",a),("lambda_total",h.+a),("supremacy",log.(h).-log.(a)))
                q = ST.quantile(values,[0.05,0.25,0.50,0.75,0.95])
                push!(rows,(; scope,group=key,axis,n=length(values),mean=ST.mean(values),sd=length(values)>1 ? ST.std(values) : NaN,
                            q05=q[1],q25=q[2],q50=q[3],q75=q[4],q95=q[5]))
            end
            if scope in ("overall","tournament")
                push!(correlations,(; scope,group=key,n=DF.nrow(g),correlation=ST.cor(log.(h),log.(a))))
            end
            if scope in ("tournament","season","tournament-season")
                d = log.(h).-log.(a)
                lo,hi = mean_ci(d)
                push!(advantage,(; scope,group=key,n=length(d),mean_d=ST.mean(d),ci_low=lo,ci_high=hi))
            end
        end
    end
    CSV.write(joinpath(out,"rate_distributions.csv"),DF.DataFrame(rows))
    CSV.write(joinpath(out,"home_advantage.csv"),DF.DataFrame(advantage))
    CSV.write(joinpath(out,"rate_correlations.csv"),DF.DataFrame(correlations))
    excluded_rows = NamedTuple[]
    start = first(panel.week_start)
    for r in eachrow(exclusions), team in (r.home,r.away)
        week = div(Dates.value(r.date-start),7)+1
        push!(excluded_rows,(; match_id=r.match_id,week,week_start=panel.week_start[week],team,season=r.season,
                             tournament=r.tournament,reason=EXCLUSION))
    end
    coverage = DF.DataFrame(excluded_rows)
    CSV.write(joinpath(out,"excluded_books.csv"),coverage)
    CSV.write(joinpath(out,"excluded_by_week.csv"),DF.combine(DF.groupby(coverage,[:week,:week_start]),:match_id=>(x->length(unique(x)))=>:n_fixtures))
    CSV.write(joinpath(out,"excluded_by_team.csv"),DF.combine(DF.groupby(coverage,[:team,:season]),DF.nrow=>:n_fixtures))
    return DF.DataFrame(rows)
end

"Equal-count deciles (ties resolved stably by input order), no outcome-based boundaries."
function deciles(values)
    order = sortperm(values; alg=Base.Sort.MergeSort)
    bins = zeros(Int,length(values))
    for (rank,i) in enumerate(order)
        bins[i] = min(10,cld(10rank,length(values)))
    end
    return bins
end
function mean_variance(panel,smooth,out)
    r2 = DF.filter(r -> r.rung == "R2" && r.axis in AXES[1:2],smooth)
    lookup = Dict((r.match_id,r.axis)=>r.predicted for r in eachrow(r2))
    fitted = [exp(lookup[(panel.obs_match[j],panel.obs_home[j] == 1 ? AXES[1] : AXES[2])]) for j in eachindex(panel.obs_y)]
    rates = exp.(panel.obs_y)
    bins = deciles(fitted)
    rows = NamedTuple[]
    for bin in 1:10
        keep = bins.==bin
        values = rates[keep]
        push!(rows,(; bin,n=count(keep),fitted_mean=ST.mean(fitted[keep]),mean_rate=ST.mean(values),
                     variance=ST.var(values),cv=ST.std(values)/ST.mean(values)))
    end
    CSV.write(joinpath(out,"mean_variance.csv"),DF.DataFrame(rows))
    return DF.DataFrame(rows)
end

"Goals from ds.matches.home_score / away_score (full-time); all 517 fixtures required."
function rates_goals(ds,panel,book,out)
    matches = DF.select(ds.matches,:match_id,:home_score,:away_score)
    m = DF.leftjoin(panel.matches,matches; on=:match_id,order=:left)
    DF.nrow(m) == 517 || error("goals join lost/duplicated fixtures")
    any(ismissing,m.home_score) && error("missing full-time home_score")
    any(ismissing,m.away_score) && error("missing full-time away_score")
    glm_rows = NamedTuple[]
    calibration = NamedTuple[]
    for (axis,lambda,y) in (("home",m.lambda_mkt_h,Float64.(m.home_score)),
                           ("away",m.lambda_mkt_a,Float64.(m.away_score)),
                           ("total",m.lambda_mkt_h.+m.lambda_mkt_a,Float64.(m.home_score.+m.away_score)))
        fit = poisson_glm(y,log.(lambda))
        push!(glm_rows,(; axis,n=length(y),intercept=fit.coef[1],intercept_se=fit.se[1],slope=fit.coef[2],slope_se=fit.se[2],
                         goal_columns="ds.matches.home_score / away_score"))
        bins = deciles(lambda)
        for bin in 1:10
            keep = bins.==bin
            lo,hi = mean_ci(y[keep]; seed=3911+bin)
            push!(calibration,(; axis,bin,n=count(keep),mean_lambda=ST.mean(lambda[keep]),mean_goals=ST.mean(y[keep]),ci_low=lo,ci_high=hi))
        end
    end
    CSV.write(joinpath(out,"goal_glm.csv"),DF.DataFrame(glm_rows))
    CSV.write(joinpath(out,"goal_calibration.csv"),DF.DataFrame(calibration))
    byid = Dict(Int(first(g.match_id))=>g for g in DF.groupby(book,:match_id))
    outcomes = NamedTuple[]
    for r in eachrow(m)
        b = byid[r.match_id]
        q = MM.selection_probabilities(log.([r.lambda_mkt_h,r.lambda_mkt_a]))
        fair = Dict(Symbol(x.selection)=>x.prob_fair_close for x in eachrow(b))
        win = r.home_score > r.away_score ? :home : r.home_score < r.away_score ? :away : :draw
        push!(outcomes,(; match_id=r.match_id,market="1X2",selection=String(win),
                         grid_logloss=-log(q[win]),close_logloss=-log(fair[win])))
        if any(x -> x.market_name == "OverUnder" && x.market_line == 2.5,eachrow(b))
            total_selection = r.home_score+r.away_score > 2.5 ? :over_25 : :under_25
            push!(outcomes,(; match_id=r.match_id,market="OU2.5",selection=String(total_selection),
                grid_logloss=-log(q[total_selection]),close_logloss=-log(fair[total_selection])))
        end
    end
    outcomes = DF.DataFrame(outcomes)
    outcomes.difference = outcomes.grid_logloss.-outcomes.close_logloss
    CSV.write(joinpath(out,"rates_vs_goals_fixture.csv"),outcomes)
    losses = NamedTuple[]
    for g in DF.groupby(outcomes,:market)
        lo,hi = mean_ci(g.difference; seed=3912)
        push!(losses,(; market=first(g.market),n=DF.nrow(g),grid_logloss=ST.mean(g.grid_logloss),
                     close_logloss=ST.mean(g.close_logloss),difference=ST.mean(g.difference),ci_low=lo,ci_high=hi))
    end
    CSV.write(joinpath(out,"rates_vs_goals.csv"),DF.DataFrame(losses))
    return DF.DataFrame(calibration)
end

"d residuals are oriented to the named club; level residuals have no sign flip."
function residual_tables(panel,smooth,predictions,out)
    meta = Dict(r.match_id=>r for r in eachrow(panel.matches))
    last_week = Dict(s=>maximum(panel.obs_week[panel.obs_season.==s]) for s in unique(panel.obs_season))
    smooth_lookup = Dict((r.rung,r.match_id,r.axis)=>r.residual for r in eachrow(smooth))
    team_rows = NamedTuple[]
    week_rows = NamedTuple[]
    transitions = Set(["ross-county","airdrieonians","east-kilbride","kelty-hearts"])
    for r in eachrow(predictions)
        r.axis in ("supremacy","level") || continue
        m = meta[r.match_id]
        sm = smooth_lookup[(r.rung,r.match_id,r.axis)]
        os = r.observed-r.predicted
        late = r.week >= last_week[m.season]-3
        push!(week_rows,(; rung=r.rung,protocol=r.protocol,axis=r.axis,week=r.week,season=m.season,
                         smoothed=sm,onestep=os,last_four_weeks=late))
        for (team,orientation) in ((m.home_team,1),(m.away_team,r.axis == "supremacy" ? -1 : 1))
            push!(team_rows,(; rung=r.rung,protocol=r.protocol,axis=r.axis,team,season=m.season,
                smoothed=orientation*sm,onestep=orientation*os,transition_club=team in transitions,last_four_weeks=late))
        end
    end
    teams = DF.combine(DF.groupby(DF.DataFrame(team_rows),[:rung,:protocol,:axis,:team,:season,:transition_club]),
        DF.nrow=>:n,:smoothed=>ST.mean=>:mean_smoothed_residual,:onestep=>ST.mean=>:mean_onestep_residual,
        :last_four_weeks=>sum=>:n_last_four_weeks)
    weeks = DF.combine(DF.groupby(DF.DataFrame(week_rows),[:rung,:protocol,:axis,:week,:season,:last_four_weeks]),
        DF.nrow=>:n,:smoothed=>ST.mean=>:mean_smoothed_residual,:onestep=>ST.mean=>:mean_onestep_residual)
    CSV.write(joinpath(out,"residuals_team.csv"),teams)
    CSV.write(joinpath(out,"residuals_week.csv"),weeks)
    for (name,table) in (("team",teams),("week",weeks))
        worst = DF.DataFrame[]
        for g in DF.groupby(table,[:rung,:protocol,:axis]),metric in (:mean_smoothed_residual,:mean_onestep_residual)
            order = sortperm(abs.(g[!,metric]); rev=true)
            selected = DF.DataFrame(g[order[1:min(15,length(order))],:])
            selected.ranking_metric .= String(metric)
            selected.rank = collect(1:DF.nrow(selected))
            push!(worst,selected)
        end
        CSV.write(joinpath(out,"residuals_$(name)_worst15.csv"),vcat(worst...))
    end
end

function ladder_evaluation(panel,fits,out; particles=20000,seeds=(3906,3907,3908))
    smooth = DF.DataFrame[]
    metrics = DF.DataFrame[]
    predictions = DF.DataFrame[]
    rbpf_weeks = DF.DataFrame[]
    loglik = Dict{Int,Float64}()
    for level in 0:4
        sr,ll = smoothed_rows(fits[(level,"10a")],panel)
        push!(smooth,sr)
        loglik[level] = ll
        for protocol in ("10a","10b")
            fit = fits[(level,protocol)]
            θ = MID.median_theta(fit)
            for seed in (level == 4 ? seeds : seeds[1:1])
                pred = preweek_predictions(Rung(level),panel,θ; particles,seed)
                summary = prediction_summary(panel,pred,protocol,"R$level"; seed)
                push!(metrics,summary)
                if level == 4
                    pred.weeks.protocol .= protocol
                    # Add named fixture identities, including both clubs.
                    fixture_names = Dict(r.match_id=>"$(r.home_team) v $(r.away_team)" for r in eachrow(panel.matches))
                    pred.weeks.fixture_names = [join([fixture_names[parse(Int,id)] for id in split(f,";") if !isempty(id)],";") for f in pred.weeks.fixtures]
                    push!(rbpf_weeks,pred.weeks)
                end
                if seed == first(seeds)
                    pred.rows.rung .= "R$level"
                    pred.rows.protocol .= protocol
                    scored_rows = pred.rows
                    if protocol == "10b"
                        allowed = Set(panel.matches.match_id[panel.matches.season.=="25/26"])
                        scored_rows = DF.filter(r->r.match_id in allowed,pred.rows)
                    end
                    push!(predictions,scored_rows)
                end
            end
        end
        println("R$level predictive evaluation complete")
        flush(stdout)
    end
    sm = vcat(smooth...)
    met = vcat(metrics...)
    raw = vcat(predictions...)
    CSV.write(joinpath(out,"smoothed_fit.csv"),sm)
    CSV.write(joinpath(out,"onestep_metrics.csv"),DF.filter(:seed=>==(first(seeds)),met))
    CSV.write(joinpath(out,"onestep_fixture.csv"),raw)
    stability = DF.filter(:rung=>==("R4"),met)
    CSV.write(joinpath(out,"rbpf_seed_metrics.csv"),stability)
    spread = DF.combine(DF.groupby(stability,[:protocol,:subset,:axis]),
        :rmse=>(x->maximum(x)-minimum(x))=>:rmse_spread,
        :mae=>(x->maximum(x)-minimum(x))=>:mae_spread,
        :mean_logpd=>(x->maximum(x)-minimum(x))=>:mean_logpd_spread,
        :cover90=>(x->maximum(x)-minimum(x))=>:cover90_spread,
        :collapsed_loglik=>(x->maximum(x)-minimum(x))=>:collapsed_loglik_spread)
    CSV.write(joinpath(out,"rbpf_seed_spread.csv"),spread)
    ew = vcat(rbpf_weeks...)
    CSV.write(joinpath(out,"rbpf_week_diagnostics.csv"),ew)
    worst = [DF.sort(DF.DataFrame(g),:ess)[1:min(10,DF.nrow(g)),:] for g in DF.groupby(ew,[:protocol,:seed])]
    CSV.write(joinpath(out,"rbpf_worst10_weeks.csv"),vcat(worst...))
    residual_tables(panel,sm,raw,out)
    ladder = NamedTuple[]
    for level in 0:4
        fit = fits[(level,"10a")]
        marginal = level == 4 ? only(DF.filter(r->r.rung == "R4" && r.protocol == "10a" && r.subset == "all" &&
            r.axis == "log_lambda_h" && r.seed == first(seeds),met).collapsed_loglik) : loglik[level]
        row = Dict{Symbol,Any}(:rung=>"R$level",:collapsed_loglik=>marginal,
            :conditional_smoothed_loglik=>loglik[level],
            :loglik_method=>level == 4 ? "RBPF scales/states integrated at median theta" : "exact Kalman at median theta")
        for name in ("sigma_obs","sigma_att","sigma_def","sigma_break","sigma_gamma","nu")
            j = findfirst(==(name),fit.names)
            q = j === nothing ? fill(NaN,3) : ST.quantile(vec(fit.draws[:,j,:]),[0.05,0.5,0.95])
            for (suffix,value) in zip(("q05","median","q95"),q)
                row[Symbol(name,"_",suffix)] = value
            end
        end
        for axis in AXES
            s = DF.filter(r->r.rung == "R$level" && r.axis == axis,sm)
            row[Symbol("smoothed_r2_",axis)] = 1-sum(s.residual.^2)/sum((s.observed.-ST.mean(s.observed)).^2)
            row[Symbol("smoothed_rmse_",axis)] = sqrt(ST.mean(s.residual.^2))
            for protocol in ("10a","10b")
                g = DF.filter(r->r.rung == "R$level" && r.axis == axis && r.protocol == protocol && r.subset == "all" && r.seed == first(seeds),met)
                row[Symbol("onestep_rmse_",protocol,"_",axis)] = only(g.rmse)
            end
        end
        keys_sorted = sort!(collect(keys(row)); by=String)
        push!(ladder,NamedTuple{Tuple(keys_sorted)}(Tuple(row[k] for k in keys_sorted)))
    end
    CSV.write(joinpath(out,"ladder_summary.csv"),DF.DataFrame(ladder))
    return sm
end

# Included inside CovarianceMarket; addendum post-processing only, never a new fit.

"Posterior innovation rotation; correlation of season-average RTS levels is a separate point statistic."
function quality_style_rotation(panel,fits)
    rows = NamedTuple[]
    N = MID.n_teams(panel)
    for protocol in ("10a","10b"), level in (2,6,7)
        fit = fits[(level,protocol)]
        sa = vec(fit.draws[:,only(findall(==("sigma_att"),fit.names)),:])
        sb = vec(fit.draws[:,only(findall(==("sigma_def"),fit.names)),:])
        jrho = findfirst(==("rho_ab"),fit.names)
        rho = jrho === nothing ? zeros(length(sa)) : vec(fit.draws[:,jrho,:])
        vq = (sa.^2+sb.^2-2rho.*sa.*sb)/4
        vs = (sa.^2+sb.^2+2rho.*sa.*sb)/4
        all(vq .> 0) && all(vs .> 0) || error("quality/style covariance is not positive definite")
        sq,ss = sqrt.(vq),sqrt.(vs)
        correlation = (sa.^2-sb.^2)./(4sq.*ss)
        for (parameter,values) in (("variance_quality",vq),("variance_style",vs),
            ("sigma_quality",sq),("sigma_style",ss),("style_quality_ratio",ss./sq),
            ("correlation_quality_style",correlation))
            q = ST.quantile(values,[0.05,0.5,0.95])
            push!(rows,(; rung="R$level",protocol,scope="innovation",season="all",parameter,
                median=q[2],q05=q[1],q95=q[3],n_draws=length(values),n_teams=N,n_weeks=panel.n_weeks,
                method="posterior draw transform"))
        end
        theta = MID.median_theta(fit)
        if level == 2
            f = TB.rung_filter(fit.arm,panel,theta; store=true)
            X = MID.smoothed_mean(fit.arm,panel,TB.rung_schedule(fit.arm,panel,theta),f)
        else
            f = covariance_filter(fit.arm,panel,theta; store=true)
            X = MID.smoothed_mean(conditional_arm(fit.arm,theta),panel,covariance_schedule(fit.arm,panel,theta),f)
        end
        for season in sort(unique(panel.obs_season); by=s->minimum(panel.obs_week[panel.obs_season .== s]))
            matches = DF.filter(:season=>==(season),panel.matches)
            active = sort(unique(vcat(matches.home_team,matches.away_team)))
            indices = [only(findall(==(team),panel.teams)) for team in active]
            ow = panel.obs_week[panel.obs_season .== season]
            weeks = minimum(ow):maximum(ow)
            attack = X[3:2+N,weeks]
            concessions = X[3+N:2+2N,weeks]
            attack = attack .- ST.mean(attack; dims=1)
            concessions = concessions .- ST.mean(concessions; dims=1)
            avga = vec(ST.mean(attack[indices,:]; dims=2))
            avgb = vec(ST.mean(concessions[indices,:]; dims=2))
            value = ST.cor(avga,avgb)
            push!(rows,(; rung="R$level",protocol,scope="levels",season,
                parameter="correlation_alpha_beta_levels",median=value,q05=NaN,q95=NaN,n_draws=0,
                n_teams=length(active),n_weeks=length(weeks),
                method="season-average RTS means at median theta; active accepted-panel teams; point only"))
        end
    end
    return DF.DataFrame(rows)
end

"Unbalanced one-way random-effects ANOVA ICC; retain signed MOM estimates, including negative ICC."
function week_icc(groups::AbstractVector)
    k = length(groups)
    sizes = length.(groups)
    N = sum(sizes)
    k >= 2 && N > k || error("ICC needs at least two weeks and within-week replication")
    totals = sum.(groups)
    grand = sum(totals)/N
    means = totals./sizes
    ss_between = sum(sizes.*(means.-grand).^2)
    ss_within = sum(sum((g.-mu).^2) for (g,mu) in zip(groups,means))
    ms_between = ss_between/(k-1)
    ms_within = ss_within/(N-k)
    n0 = (N-sum(sizes.^2)/N)/(k-1)
    n0 > 1 || error("ICC replication coefficient must exceed one")
    variance_week = (ms_between-ms_within)/n0
    total_variance = variance_week+ms_within
    total_variance > 0 || error("ICC total variance is not positive")
    return (; icc=variance_week/total_variance,variance_week,variance_within=ms_within,
        ms_between,ms_within,n0,n=N,n_weeks=k,n_multi_fixture_weeks=count(>(1),sizes),
        n_pairs=sum(n*(n-1)÷2 for n in sizes))
end

"Resample complete week clusters; duplicate sampled weeks are separate bootstrap groups."
function league_shock_statistic(weeks,residuals; reps=2000,seed=3928)
    length(weeks) == length(residuals) || error("week/residual lengths differ")
    ids = sort(unique(weeks))
    groups = [Float64.(residuals[weeks .== w]) for w in ids]
    point = week_icc(groups)
    rng = Random.Xoshiro(seed)
    boot = [week_icc(groups[rand(rng,eachindex(groups),length(groups))]).icc for _ in 1:reps]
    return (; point...,ci_low=ST.quantile(boot,0.025),ci_high=ST.quantile(boot,0.975),reps,seed,
        method="unbalanced one-way ANOVA signed MOM; whole-week percentile bootstrap")
end

"All scored level errors; do not drop singleton, opening or adverse weeks."
function league_shock_test(fixtures)
    rows = NamedTuple[]
    for protocol in ("10a","10b"), rung in ("R2","R5","R6","R7")
        g = DF.filter(r -> r.protocol == protocol && r.rung == rung && r.axis == "level",fixtures)
        length(unique(g.match_id)) == DF.nrow(g) || error("level ICC has duplicated fixtures")
        stat = league_shock_statistic(g.week,g.observed-g.predicted)
        push!(rows,(; rung,protocol,scope="all scored fixtures, pooled tournaments",stat...))
    end
    return DF.DataFrame(rows)
end

"Read already-converged prototype fits; this entry point never invokes sampling."
function load_covariance_fits(out)
    convergence = CSV.read(joinpath(out,"convergence_b2.csv"),DF.DataFrame)
    all(convergence.gate_pass) || error("addendum cannot promote unconverged fits")
    return Dict((level,protocol)=>Serialization.deserialize(joinpath(out,"R$(level)_$(protocol).jls"))
        for level in (2,5,6,7) for protocol in ("10a","10b"))
end

function hierarchy_postprocess(panel,fits,out; fixtures=CSV.read(joinpath(out,"onestep_fixture_b2.csv"),DF.DataFrame))
    rotation = quality_style_rotation(panel,fits)
    shock = league_shock_test(fixtures)
    CSV.write(joinpath(out,"rotation_quality_style.csv"),rotation)
    CSV.write(joinpath(out,"league_shock_test.csv"),shock)
    return (; rotation,shock)
end

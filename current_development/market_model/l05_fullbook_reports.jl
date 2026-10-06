# Included inside FullBookWorkflow; definitions only, no database access.

"Recheck recovery from its retained draws, not a done marker or a file's existence."
function require_recovery(out; truth=[0.03,0.01,0.06,1000.0],generation_seed=3962,
                          expected_panel_sha=nothing)
    panel_path = joinpath(out,"synthetic_panel.jls")
    if expected_panel_sha !== nothing
        open(io -> bytes2hex(SHA.sha256(io)),panel_path) == expected_panel_sha ||
            error("wrong frozen recovery panel hash")
    end
    generated = Serialization.deserialize(panel_path)
    generated.seed == generation_seed && generated.truth == exp.(log.(truth)) || error("wrong recovery truth/seed")
    result = Serialization.deserialize(joinpath(out,"C1_recovery.jls"))
    size(result.fit.draws) == (3000,4,4) || error("wrong recovery retained budget")
    result.fit.arm.name == :C1 || error("wrong recovery rung")
    diagnostics = PM.fullbook_diagnostics(result.fit; protocol="synthetic",seed=generation_seed)
    all(diagnostics.gate_pass) || error("recovery convergence failed; no production")
    for j in eachindex(truth)
        lo,hi = ST.quantile(vec(result.fit.draws[:,j,:]),[0.05,0.95])
        lo <= truth[j] <= hi || error("recovery interval missed $(result.fit.names[j]): [$lo,$hi], truth=$(truth[j])")
    end
    return diagnostics
end

"One rung/protocol at frozen 4x(2000+3000) budget; persist diagnostics before promotion."
function train_rung(a,p,markets,out; protocol,seeds)
    path = joinpath(out,"$(a.name)_$(protocol).jls")
    isfile(path) && error("immutable fit exists: $path; use a fresh production directory")
    result = PM.fit_fullbook(a,p; markets=a.name == :C1 ? markets : nothing,seeds,
        accounting_out=out,accounting_run="$(a.name)_$(protocol)")
    Serialization.serialize(path,result)
    diag = PM.fullbook_diagnostics(result.fit; protocol,seed=first(seeds))
    diag.sha .= strip(read(`git rev-parse HEAD`,String))
    diag.chain_seeds .= join(seeds,";")
    CSV.write(joinpath(out,"convergence_$(a.name)_$(protocol).csv"),diag)
    PM.write_newton_accounting(out; run="$(a.name)_$(protocol)")
    CSV.write(joinpath(out,"newton_termination_$(a.name)_$(protocol).csv"),
        CSV.read(joinpath(out,"newton_termination.csv"),DF.DataFrame))
    println("$(a.name) $protocol: $(result.fit.seconds) seconds; convergence $(count(diag.gate_pass))/$(DF.nrow(diag))")
    flush(stdout)
    all(diag.gate_pass) || error("$(a.name) $protocol convergence failed; no inference promotion")
    return result.fit
end

"Summarise all retained physical draws, including style/quality ratio draw by draw."
function parameter_summary(fits)
    rows = NamedTuple[]
    for (rung,protocol) in sort(collect(keys(fits)); by=x->(String(x[1]),x[2]))
        fit = fits[(rung,protocol)]
        for (name,values) in vcat([(name,vec(fit.draws[:,j,:])) for (j,name) in enumerate(fit.names)],
            [("style_quality_ratio",vec(fit.draws[:,findfirst(==("sigma_s"),fit.names),:])./
                vec(fit.draws[:,findfirst(==("sigma_q"),fit.names),:]))])
            lo,median,hi = ST.quantile(values,[0.05,0.5,0.95])
            push!(rows,(; rung=String(rung),protocol,parameter=name,lo,median,hi,
                method="all retained hyperparameter draws"))
        end
    end
    return DF.DataFrame(rows)
end

"Conditional Gaussian q/s paths at median theta; uncertainty excludes hyperparameter mixing."
function team_paths(a,p,means,covs; protocol)
    N = MID.n_teams(p)
    rows = NamedTuple[]
    for team in 1:N, t in 1:p.n_weeks, axis in (:quality,:style)
        h = zeros(size(means,1))
        sign = axis == :quality ? -1.0 : 1.0
        for i in 1:N
            centred = (i == team ? 1.0 : 0.0)-1/N
            h[2+i] = centred/2
            h[2+N+i] = sign*centred/2
        end
        mean = LA.dot(h,means[:,t])
        variance = LA.dot(h,covs[:,:,t]*h)
        variance >= -1e-10 || error("negative smoothed path variance")
        sd = sqrt(max(variance,0.0))
        push!(rows,(; rung=String(a.name),protocol,team=p.teams[team],week=t,
            week_start=p.week_start[t],axis=String(axis),mean,sd,
            lo=mean+DS.quantile(DS.Normal(),0.05)*sd,hi=mean+DS.quantile(DS.Normal(),0.95)*sd,
            method="conditional Gaussian RTS at median theta; C1 approximate"))
    end
    return DF.DataFrame(rows)
end

"Smoothed theta versus isolated target, plus full-book deviation-retention diagnostics."
function smoothed_tables(a,p,f,means,covs; protocol)
    fixtures = fixture_moments(a,p,f,means,covs)
    rows,shrinkage = NamedTuple[],NamedTuple[]
    for row in fixtures
        fixture = findfirst(==(row.match_id),p.obs_match[1:2:end])
        j = 2*fixture-1
        observed = TB.axis_values(p.obs_y[j],p.obs_y[j+1])
        predicted = TB.axis_values(row.theta_mean...)
        for axis in 1:4
            push!(rows,(; rung=String(a.name),protocol,match_id=row.match_id,axis=TB.AXES[axis],
                observed=observed[axis],predicted=predicted[axis],
                residual=observed[axis]-predicted[axis],
                method=a.name == :C1 ? "frozen-factor conditional theta=structure+u RTS" : "structured RTS at median theta"))
        end
        if a.name == :C1
            delta = p.obs_y[j:j+1]-row.structured
            retained = row.theta_mean-row.structured
            push!(shrinkage,(; rung=String(a.name),protocol,book_type="full",match_id=row.match_id,
                isolated_h=p.obs_y[j],isolated_a=p.obs_y[j+1],
                structured_h=row.structured[1],structured_a=row.structured[2],
                pooled_h=row.theta_mean[1],pooled_a=row.theta_mean[2],
                isolated_deviation_squared=LA.dot(delta,delta),
                retained_dot_isolated=LA.dot(retained,delta),
                movement_inf=maximum(abs,row.theta_mean-p.obs_y[j:j+1]),
                method="conditional theta posterior; ratio is descriptive, not a scalar shrinkage operator"))
        end
    end
    return DF.DataFrame(rows),DF.DataFrame(shrinkage)
end

"Local likelihood-only noise at median n/mode; no extra evaluation-book variance."
function book_noise(a,p,theta,f; protocol)
    a.name == :C1 || error("book-noise diagnostic requires C1")
    rows = NamedTuple[]
    for factor in f.factors
        for (axis,h) in (("log_lambda_h",[1.0,0.0]),("log_lambda_a",[0.0,1.0]),
                        ("supremacy",[1.0,-1.0]),("level",[0.5,0.5]))
            sd = sqrt(LA.dot(h,factor.R*h))
            deviation_sd = exp(theta[3])*LA.norm(h)
            push!(rows,(; rung="C1",protocol,match_id=factor.match_id,book_type="full",axis,
                n=exp(theta[4]),book_sd=sd,fixture_sd=deviation_sd,
                book_fixture_sd_ratio=sd/deviation_sd,
                method="local clipped likelihood precision at median theta; not posterior quantiles"))
        end
    end
    return DF.DataFrame(rows)
end

"Evaluate accepted fits only; write partial authorised stages without inventing missing rungs."
function evaluate_fits(p,markets,config,fits,out; prediction_seed=3963)
    metrics,fixtures,smooth,shrinkage,paths,noise = [DF.DataFrame[] for _ in 1:6]
    ladder = NamedTuple[]
    for (rung,protocol) in sort(collect(keys(fits)); by=x->(String(x[1]),x[2]))
        fit = fits[(rung,protocol)]
        all(PM.fullbook_diagnostics(fit; protocol,seed=0).gate_pass) || error("unconverged evaluation input")
        a,theta = fit.arm,MID.median_theta(fit)
        f = PM.fullbook_filter(a,p,theta; markets=a.name == :C1 ? markets : nothing,store=true,predict=true)
        pred = prediction_rows(p,f; seed=prediction_seed)
        push!(metrics,TB.prediction_summary(p,pred,protocol,String(rung); config,seed=prediction_seed))
        pred.rows.rung .= String(rung)
        pred.rows.protocol .= protocol
        allowed = Set(p.matches.match_id[in.(p.matches.season,Ref(config.honest_test))])
        push!(fixtures,protocol == "10a" ? pred.rows : DF.filter(r->r.match_id in allowed,pred.rows))
        means,covs = PM.fullbook_smoothing(f)
        sm,sh = smoothed_tables(a,p,f,means,covs; protocol)
        push!(smooth,sm)
        !isempty(sh) && push!(shrinkage,sh)
        push!(paths,team_paths(a,p,means,covs; protocol))
        a.name == :C1 && push!(noise,book_noise(a,p,theta,f; protocol))
        for axis in unique(sm.axis)
            g = DF.filter(:axis=>==(axis),sm)
            push!(ladder,(; rung=String(rung),protocol,axis,
                smoothed_r2=1-sum(g.residual.^2)/sum((g.observed.-ST.mean(g.observed)).^2),
                smoothed_rmse=sqrt(ST.mean(g.residual.^2)),collapsed_loglik=f.loglik,
                loglik_method=a.name == :C1 ? "approximate sequential Laplace" : "exact Gaussian",
                smoothing_method=a.name == :C1 ? "conditional theta=structure+u frozen-factor RTS" : "structured RTS"))
        end
    end
    raw = vcat(fixtures...)
    CSV.write(joinpath(out,"onestep_metrics_c.csv"),vcat(metrics...))
    CSV.write(joinpath(out,"onestep_fixture_c.csv"),raw)
    CSV.write(joinpath(out,"paired_vs_c0.csv"),paired_scores(raw))
    CSV.write(joinpath(out,"smoothed_fit_c.csv"),vcat(smooth...))
    CSV.write(joinpath(out,"ladder_summary_c.csv"),DF.DataFrame(ladder))
    CSV.write(joinpath(out,"team_paths_c.csv"),vcat(paths...))
    CSV.write(joinpath(out,"parameter_posteriors_c.csv"),parameter_summary(fits))
    if !isempty(shrinkage)
        sh = vcat(shrinkage...)
        CSV.write(joinpath(out,"shrinkage_fixture_c.csv"),sh)
        summary = DF.combine(DF.groupby(sh,[:rung,:protocol,:book_type]),
            DF.nrow=>:fixtures,:movement_inf=>ST.median=>:median_movement_inf,
            [:retained_dot_isolated,:isolated_deviation_squared]=>
                ((r,d)->sum(r)/sum(d))=>:deviation_retained_least_squares)
        CSV.write(joinpath(out,"shrinkage_by_type.csv"),summary)
    end
    !isempty(noise) && CSV.write(joinpath(out,"book_noise_c.csv"),vcat(noise...))
    return raw
end

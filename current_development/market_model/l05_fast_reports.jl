module FastFullBookReports

import CSV
import DataFrames
import Distributions
import LinearAlgebra
import Statistics
import Random
import Plots

const WF = parentmodule(@__MODULE__).FullBookWorkflow
const PM = WF.PM
const DF = DataFrames
const LA = LinearAlgebra
const ST = Statistics
const DS = Distributions

"Quantiles of an equally weighted conditional Normal mixture, integrating state AND hyperparameter uncertainty."
function normal_mixture_quantiles(means,sds; probabilities=(0.05,0.5,0.95))
    length(means) == length(sds) && !isempty(means) || error("invalid mixture")
    all(x -> isfinite(x) && x > 0,sds) || error("invalid mixture SD")
    lower,upper = minimum(means.-12sds),maximum(means.+12sds)
    return [begin
        lo,hi = lower,upper
        for _ in 1:70
            mid = (lo+hi)/2
            cdf = ST.mean(DS.cdf.(DS.Normal.(means,sds),mid))
            if cdf < probability
                lo = mid
            else
                hi = mid
            end
        end
        (lo+hi)/2
    end for probability in probabilities]
end

"""
Static HA coefficients integrate their conditional Gaussian posterior against
ALL 12,000 retained hyperdraws. 10b uses ONLY the training panel here, unlike
full-panel descriptive paths. The static marginal at the last filtered week
already conditions on all training observations; no RTS or extra noise needed.
H1's gamma_att and gamma_def are not separately likelihood-identified from mu:
(mu+c, gamma_att-c, gamma_def+c) leaves both rates unchanged. Their individual
intervals therefore depend on the frozen proper priors. gamma_att+gamma_def is
the identified supremacy lift; (gamma_att-gamma_def)/2 is the level shift under
that prior-dependent decomposition. H2 kappa uses all retained physical draws.
"""
function home_advantage(fits,p,train; filter_builder=nothing)
    rows = NamedTuple[]
    for (rung,protocol) in sort(collect(keys(fits)); by=x->(String(x[1]),x[2]))
        fit = fits[(rung,protocol)]
        panel = protocol == "10a" ? p : train
        if rung == :H2
            q = ST.quantile(vec(fit.draws[:,4,:]),[0.05,0.5,0.95])
            push!(rows,(; rung=String(rung),protocol,parameter="kappa",lo=q[1],median=q[2],hi=q[3],
                hyperdraws=length(fit.draws[:,4,:]),method="all retained hyperparameter draws"))
        end
        axes = rung == :H1 ? ("gamma_att","gamma_def","supremacy_lift","level_shift") : ("gamma",)
        U = reshape(permutedims(fit.udraws,(2,1,3)),length(fit.names),:)
        means,sds = zeros(size(U,2),length(axes)),zeros(size(U,2),length(axes))
        static_filter = filter_builder === nothing ? PM.fullbook_filter : filter_builder(fit.arm,panel)
        # Indexed storage: no shared RNG or reduction depends on task scheduling.
        Threads.@threads for d in axes_of_draws(U)
            f = static_filter(fit.arm,panel,U[:,d]; store=true)
            m,V = f.m_filt[:,end],f.P_filt[:,:,end]
            loadings = rung == :H1 ? ((2=>1.0,),(length(m)=>1.0,),
                (2=>1.0,length(m)=>1.0),(2=>0.5,length(m)=>-0.5)) : ((2=>1.0,),)
            for (j,loading) in enumerate(loadings)
                h = zeros(length(m))
                for (i,value) in loading
                    h[i] = value
                end
                means[d,j] = LA.dot(h,m)
                sds[d,j] = sqrt(LA.dot(h,V*h))
            end
        end
        for (j,name) in enumerate(axes)
            q = normal_mixture_quantiles(means[:,j],sds[:,j])
            push!(rows,(; rung=String(rung),protocol,parameter=name,lo=q[1],median=q[2],hi=q[3],
                hyperdraws=size(U,2),method="conditional static-state Gaussian mixture over ALL retained hyperdraws"))
        end
        println("HA mixture complete: $rung $protocol, $(size(U,2)) hyperdraws")
        flush(stdout)
    end
    return DF.DataFrame(rows)
end
axes_of_draws(U) = axes(U,2)

"Compare newly fitted C0 with the PUBLISHED B2 R6, on exact fixture/axis/protocol keys."
function compare_r6(raw,out; reference=joinpath(@__DIR__,"results","B2","onestep_fixture_b2.csv"))
    r6 = DF.filter(:rung=>==("R6"),CSV.read(reference,DF.DataFrame))
    c0 = DF.filter(:rung=>==("C0"),raw)
    paired = DF.innerjoin(c0,DF.select(r6,:match_id,:protocol,:axis,
        :observed=>:r6_observed,:predicted=>:r6_predicted,:variance=>:r6_variance,
        :logpd=>:r6_logpd,:cover90=>:r6_cover90); on=[:match_id,:protocol,:axis],order=:left)
    DF.nrow(paired) == DF.nrow(c0) || error("C0/R6 fixture join lost rows")
    maximum(abs,paired.observed-paired.r6_observed) == 0 || error("C0/R6 scoring targets differ")
    rows = NamedTuple[]
    for g in DF.groupby(paired,[:protocol,:axis])
        push!(rows,(; protocol=first(g.protocol),axis=first(g.axis),n=DF.nrow(g),
            c0_rmse=sqrt(ST.mean((g.observed-g.predicted).^2)),
            r6_rmse=sqrt(ST.mean((g.observed-g.r6_predicted).^2)),
            mean_logpd_gap=ST.mean(g.logpd-g.r6_logpd),
            c0_cover90=ST.mean(g.cover90),r6_cover90=ST.mean(g.r6_cover90),
            max_mean_gap=maximum(abs,g.predicted-g.r6_predicted)))
    end
    CSV.write(joinpath(out,"c0_vs_r6_metrics.csv"),DF.DataFrame(rows))
    control = copy(r6)
    control.rung .= "R6"
    # Published 10b files contain only scored test fixtures. Restrict explicitly.
    keys = DF.unique(DF.select(c0,:match_id,:protocol,:axis))
    control = DF.semijoin(control,keys; on=[:match_id,:protocol,:axis])
    CSV.write(joinpath(out,"paired_c0_vs_r6.csv"),WF.paired_scores(vcat(c0,control); baseline="R6"))
    return paired
end

"Nonlinear smoothed log-total is a seeded expectation under the structured RTS Gaussian, not log-sum-exp of its mean."
function smoothed_total(fits,p,out; seed=3964,mc_draws=4000)
    rows = NamedTuple[]
    for (rung,protocol) in sort(collect(keys(fits)); by=x->(String(x[1]),x[2]))
        fit = fits[(rung,protocol)]
        f = PM.fullbook_filter(fit.arm,p,PM.MID.median_theta(fit); store=true)
        means,covs = PM.fullbook_smoothing(f)
        rng = Random.Xoshiro(seed)
        for row in WF.fixture_moments(fit.arm,p,f,means,covs)
            h,a = row.theta_mean
            V = row.theta_cov
            sh = sqrt(max(V[1,1],1e-300))
            sa = sqrt(max(V[2,2]-V[1,2]^2/max(V[1,1],1e-300),0))
            draws = [PM.CM.TB.axis_values(h+sh*z[1],a+V[1,2]/sh*z[1]+sa*z[2])[5]
                for z in (randn(rng,2) for _ in 1:mc_draws)]
            fixture = findfirst(==(row.match_id),p.obs_match[1:2:end])
            j = 2*fixture-1
            observed = PM.CM.TB.axis_values(p.obs_y[j],p.obs_y[j+1])[5]
            predicted = ST.mean(draws)
            push!(rows,(; rung=String(rung),protocol,match_id=row.match_id,axis="log_total",
                observed,predicted,residual=observed-predicted,
                method="structured RTS nonlinear MC expectation at median theta"))
        end
    end
    sm = vcat(CSV.read(joinpath(out,"smoothed_fit_c.csv"),DF.DataFrame),DF.DataFrame(rows))
    CSV.write(joinpath(out,"smoothed_fit_c.csv"),sm)
    ladder = CSV.read(joinpath(out,"ladder_summary_c.csv"),DF.DataFrame)
    extra = NamedTuple[]
    for g in DF.groupby(DF.DataFrame(rows),[:rung,:protocol])
        old = DF.filter(r -> r.rung == first(g.rung) && r.protocol == first(g.protocol),ladder)[1,:]
        push!(extra,(; rung=first(g.rung),protocol=first(g.protocol),axis="log_total",
            smoothed_r2=1-sum(g.residual.^2)/sum((g.observed.-ST.mean(g.observed)).^2),
            smoothed_rmse=sqrt(ST.mean(g.residual.^2)),collapsed_loglik=old.collapsed_loglik,
            loglik_method=old.loglik_method,smoothing_method="structured RTS nonlinear MC expectation at median theta"))
    end
    CSV.write(joinpath(out,"ladder_summary_c.csv"),vcat(ladder,DF.DataFrame(extra)))
end

"Figures for the accepted fast rungs only. C1 figures remain pending."
function figures(out)
    metrics = CSV.read(joinpath(out,"onestep_metrics_c.csv"),DF.DataFrame)
    honest = DF.filter(r -> r.protocol == "10b" && r.subset == "all" && r.axis in ("supremacy","level"),metrics)
    coverage = Plots.plot(; ylabel="90% interval coverage",ylim=(0.75,1.0),legend=:bottomright)
    for axis in ("supremacy","level")
        g = DF.filter(:axis=>==(axis),honest)
        Plots.scatter!(coverage,g.rung,g.cover90; label=axis)
    end
    Plots.hline!(coverage,[0.9]; label="nominal",linestyle=:dash)
    Plots.savefig(coverage,joinpath(out,"C_fast_coverage.png"))
    ha = CSV.read(joinpath(out,"home_advantage_rungs.csv"),DF.DataFrame)
    g = DF.filter(r -> r.protocol == "10b",ha)
    labels = g.rung .* ":" .* g.parameter
    fig = Plots.scatter(labels,g.median; yerror=(g.median-g.lo,g.hi-g.median),
        label="10b 90% posterior intervals",xrotation=45,ylabel="log-rate coefficient",size=(1100,650))
    Plots.hline!(fig,[0.0]; label="zero",linestyle=:dash)
    Plots.savefig(fig,joinpath(out,"C_fast_home_advantage.png"))
    paths = CSV.read(joinpath(out,"team_paths_c.csv"),DF.DataFrame)
    panels = []
    for team in ("ross-county","airdrieonians","east-kilbride","kelty-hearts"), axis in ("quality","style")
        fig = Plots.plot(; title="$team $axis",legend=:topleft)
        for rung in ("C0","H1","H2")
            g = DF.filter(r -> r.team == team && r.axis == axis && r.protocol == "10a" && r.rung == rung,paths)
            isempty(g) && error("missing requested transition-club path $team")
            Plots.plot!(fig,g.week_start,g.mean; label=rung)
        end
        push!(panels,fig)
    end
    Plots.savefig(Plots.plot(panels...; layout=(4,2),size=(1200,1200)),joinpath(out,"C_fast_paths.png"))
end

end # module

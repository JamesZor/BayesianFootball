# R11: ordinary A1 bootstrap reproduced exactly; circular score sensitivity only.
function paired_scores(frame,keycol,left,right;source,method,comparison,goal=true)
    keys=[:league,:season,:week,:fixture_id]
    f=DF.filter(keycol=>==(left),frame);a=DF.filter(keycol=>==(right),frame)
    @assert DF.nrow(unique(f,keys))==DF.nrow(f)
    @assert DF.nrow(unique(a,keys))==DF.nrow(a)
    joined=DF.innerjoin(f,a;on=keys,makeunique=true)
    @assert DF.nrow(joined)==DF.nrow(f)==DF.nrow(a) "unexpected fixture loss"
    out=joined[:,keys];out.joint=joined.logp_joint-joined.logp_joint_1
    if goal
        out.total=joined.logp_total-joined.logp_total_1
        out.allocation=joined.logp_allocation-joined.logp_allocation_1
        @assert maximum(abs,out.joint-out.total-out.allocation)<=1e-10
    end
    (;source,method,comparison,goal,frame=out)
end
function circular_indices(frame;L,rng,B=1999)
    strata=[[collect(parentindices(w)[1]) for w in DF.groupby(g,:week)] for g in DF.groupby(frame,:season)]
    out=Vector{Vector{Int}}(undef,B)
    for b in 1:B
        ix=Int[]
        for weeks in strata
            T=length(weeks);chosen=Int[]
            while length(chosen)<T
                start=Random.rand(rng,1:T)
                append!(chosen,[mod1(start+j,T) for j in 0:L-1])
            end
            chosen=chosen[1:T];@assert length(chosen)==T
            for j in chosen;append!(ix,weeks[j]);end
        end
        out[b]=ix
    end
    @assert length(out)==1999
    out
end
function bootstrap_row(source,method,comparison,channel,league,weighting,n,n_weeks,bootstrap,L,B,point,boots,status)
    q05,med,q95=A.quant(boots)
    (;league,weighting,source,method,comparison,channel,n,n_weeks,bootstrap,L,B,point,boot_mean=ST.mean(boots),boot_median=med,boot_se=ST.std(boots),bias=ST.mean(boots)-point,
        percentile_q05=q05,percentile_q95=q95,basic_lo=2point-q95,basic_hi=2point-q05,status)
end
function r11()
    rows,audit,repro=NamedTuple[],NamedTuple[],NamedTuple[]
    fc=loadcsv("forecast_joint.csv";request="R11");jc=loadcsv("joint_comparison.csv";request="R11")
    gf=loadcsv("goal_ablation_fixtures.csv";request="R11");gs=loadcsv("goal_ablation_summary.csv";request="R11")
    tasks=[paired_scores(fc,:rung,"C0","R6";source="R03",method="mixture_128",comparison="C0_minus_R6",goal=false)]
    for (method,alt) in (("mixture_128x4","no_style"),("mixture_128x4","no_quality"),("plugin_median_theta","frozen_style"))
        f=DF.filter(:method=>==(method),gf)
        push!(tasks,paired_scores(f,:variant,"full",alt;source="R07",method,comparison="full_minus_$alt"))
    end
    for task in tasks
        channels=task.goal ? ("joint","total","allocation") : ("joint",)
        for bootstrap in ("ordinary_A1","circular"),L in (4,8,12)
            B=bootstrap=="ordinary_A1" ? 999 : 1999
            perleague=Any[]
            for (li,(league,_,_)) in enumerate(QS.LEAGUES)
                g=DF.filter(:league=>==(league),task.frame)
                # A1 seeds and original row order retained for exact historical reproduction.
                derived=bootstrap=="ordinary_A1" ? (task.source=="R03" ? A.seed(li,2,2,L) : A.seed(li,3,2,700+L)) : SEED+100_000li+10_000L
                ix=bootstrap=="ordinary_A1" ? A.bootstrap_indices(g;block=L,rng=Random.Xoshiro(derived)) : circular_indices(g;L,rng=Random.Xoshiro(derived))
                @assert length(ix)==B
                record!("R11","seed","";detail="$(task.source)/$(task.comparison)/$league/$bootstrap/L$L: $derived; B=$B; circular shared starts across comparisons/channels")
                counts=zeros(Int,DF.nrow(g));for ids in ix;counts[ids].+=1;end
                # Vector indexed assignment has duplicate-index semantics: count each occurrence explicitly.
                fill!(counts,0);for ids in ix,i in ids;counts[i]+=1;end
                den=length.(ix);T=sum(length(unique(s.week)) for s in DF.groupby(g,:season))
                low=any(length(unique(s.week))<2L for s in DF.groupby(g,:season))
                status=low ? "LOW_SUPPORT: stratum T<2L; circular assumes approximate within-season stationarity" : "OK; circular assumes approximate within-season stationarity"
                bs=Dict(c=>[ST.mean(g[ids,Symbol(c)]) for ids in ix] for c in channels)
                if task.goal
                    @assert maximum(abs,bs["joint"]-bs["total"]-bs["allocation"])<=1e-10
                end
                push!(perleague,(;league,n=DF.nrow(g),T,den,bs,point=Dict(c=>ST.mean(g[!,Symbol(c)]) for c in channels),status))
                for c in channels
                    point=ST.mean(g[!,Symbol(c)]);boots=bs[c]
                    push!(rows,bootstrap_row(task.source,task.method,task.comparison,c,league,"fixture",DF.nrow(g),T,bootstrap,L,B,point,boots,status))
                    if bootstrap=="ordinary_A1"
                        expected=task.source=="R03" ? only(eachrow(DF.filter(r->r.league==league&&r.weighting=="fixture"&&r.block_weeks==L&&r.score_kind=="joint",jc))) : only(eachrow(DF.filter(r->r.league==league&&r.method==task.method&&r.comparison==task.comparison&&r.channel==c&&r.block_weeks==L,gs)))
                        ep=task.source=="R03" ? expected.mean_C0_minus_R6 : expected.mean_delta
                        for (quantity,e,a) in (("point",ep,point),("boot_q05",expected.boot_q05,ST.quantile(boots,.05)),("boot_q95",expected.boot_q95,ST.quantile(boots,.95)))
                            err=abs(e-a);@assert err<=1e-10
                            push!(repro,(;source=task.source,comparison=task.comparison,method=task.method,channel=c,league,L,quantity,expected=e,actual=a,abs_error=err,status="PASS"))
                        end
                    end
                end
                # Audit each original week: same inclusion multiplier for all its fixtures.
                for season in unique(g.season)
                    weeks=unique(g.week[g.season.==season]);nt=length(weeks)
                    for (j,w) in enumerate(weeks)
                        ii=findall((g.season.==season).&(g.week.==w))
                        @assert all(counts[ii].==first(counts[ii]))
                        region=j<=L&&j>nt-L ? "first_and_last_L" : j<=L ? "first_L" : j>nt-L ? "last_L" : "interior"
                        for c in channels
                            push!(audit,(;league,season,week=w,bootstrap,L,B,n_fixtures=length(ii),mean_multiplicity=first(counts[ii])/B,region,comparison=task.comparison,source=task.source,method=task.method,channel=c,score_sum=sum(g[ii,Symbol(c)]),status))
                        end
                    end
                end
            end
            for weighting in ("fixture","equal_league"),c in channels
                ns=[x.n for x in perleague];weights=weighting=="fixture" ? ns/sum(ns) : fill(1/length(ns),length(ns))
                point=sum(weights[i]*perleague[i].point[c] for i in eachindex(perleague))
                boots=if bootstrap=="ordinary_A1"||weighting=="equal_league"
                    sum(weights[i]*perleague[i].bs[c] for i in eachindex(perleague))
                else
                    # Correct pooled fixture ratio: variable sampled-fixture denominator in each replicate.
                    sum(perleague[i].bs[c].*perleague[i].den for i in eachindex(perleague))./sum(x.den for x in perleague)
                end
                status=any(startswith(x.status,"LOW_SUPPORT") for x in perleague) ? "LOW_SUPPORT: some strata T<2L" : "OK"
                push!(rows,bootstrap_row(task.source,task.method,task.comparison,c,"ALL",weighting,sum(ns),sum(x.T for x in perleague),bootstrap,L,B,point,boots,status))
                if bootstrap=="ordinary_A1"
                    expected=task.source=="R03" ? only(eachrow(DF.filter(r->r.league=="ALL"&&r.weighting==weighting&&r.block_weeks==L&&r.score_kind=="joint",jc))) : only(eachrow(DF.filter(r->r.league=="ALL"&&r.weighting==weighting&&r.method==task.method&&r.comparison==task.comparison&&r.channel==c&&r.block_weeks==L,gs)))
                    ep=task.source=="R03" ? expected.mean_C0_minus_R6 : expected.mean_delta
                    for (quantity,e,a) in (("point",ep,point),("boot_q05",expected.boot_q05,ST.quantile(boots,.05)),("boot_q95",expected.boot_q95,ST.quantile(boots,.95)))
                        err=abs(e-a);@assert err<=1e-10
                        push!(repro,(;source=task.source,comparison=task.comparison,method=task.method,channel=c,league="ALL/$weighting",L,quantity,expected=e,actual=a,abs_error=err,status="PASS"))
                    end
                end
            end
        end
        println("R11 $(task.comparison) complete");flush(stdout)
    end
    output("R11","score_robustness.csv",rows)
    output("R11","week_weight_audit.csv",audit;bulk=true)
    output("R11","score_reproduction.csv",repro)
    output("R11","partition_score_availability.csv",[(;source="R09",comparison="full_minus_$v",status="NOT_IDENTIFIABLE: manager canceled C0 zero-loading common-style partition; no score rows") for v in ("no_relative_style","frozen_common","neither_motion_nor_relative")])
    lines=["PARTIAL only for canceled R09 scores; R03/R07 complete. Exact A1 points and ordinary999 percentile intervals reproduced≤1e−10. score_robustness.csv, score_reproduction.csv; week_weight_audit.csv beast-only. Ordinary A1 pool=fixed original fixture weights; circular pool=replicate fixture numerator/denominator. B1999 per L4/8/12; paired channels share draws."]
    for comp in ("C0_minus_R6","full_minus_no_style","full_minus_no_quality","full_minus_frozen_style")
        g=filter(r->r.league=="ALL"&&r.weighting=="fixture"&&r.comparison==comp&&r.bootstrap=="circular"&&r.L==8,rows)
        push!(lines,"Circular L8 pooled $comp "*join(["$(r.channel): point=$(r.point), mean=$(r.boot_mean), bias=$(r.bias), pct=[$(r.percentile_q05),$(r.percentile_q95)], basic=[$(r.basic_lo),$(r.basic_hi)]" for r in g],"; ")*".")
    end
    a=only(filter(r->r.league=="ALL"&&r.weighting=="fixture"&&r.comparison=="full_minus_no_style"&&r.channel=="allocation"&&r.bootstrap=="ordinary_A1"&&r.L==8,rows))
    push!(lines,"Original allocation mismatch: point=$(a.point), bootstrap mean=$(a.boot_mean), bias=$(a.bias), SE=$(a.boot_se), median=$(a.boot_median),90%pct=[$(a.percentile_q05),$(a.percentile_q95)]. Noncircular starts/truncation underweight boundaries; no interval forced to contain point.")
    for league in ("sco_premiership","sco_league_two","fin_veikkausliiga")
        g=filter(r->r.league==league&&r.comparison=="full_minus_no_style"&&r.channel=="total"&&r.bootstrap=="circular",rows)
        push!(lines,"$league style total point=$(first(g).point); circular L/pct "*join(["$(r.L) [$(r.percentile_q05),$(r.percentile_q95)]" for r in g],"; ")*". "*first(g).status)
    end
    summary!("R11",lines)
    verify("R11","PASS unique exact paired fixture intersections; original points and999 ordinary percentiles≤1e−10;1999 circular draws/L, exactly T resampled weeks/stratum; joint=total+allocation per fixture and replicate≤1e−10; multiplicities identical within week. Derived seeds recorded. Circular sensitivity assumes stationarity; never used to create lag pairs. R09 comparisons explicitly unavailable; existing integrated MC sensitivity not recomputed or dismissed.")
end

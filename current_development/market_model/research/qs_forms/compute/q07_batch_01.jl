module QSFormsBatch01

import CSV, DataFrames, Dates, Distributions, LinearAlgebra, Random, Serialization, SHA, Statistics
const DF = DataFrames
const ST = Statistics
const LA = LinearAlgebra
const DS = Distributions
if !isdefined(parentmodule(@__MODULE__), :QualityStyleEDA)
    Base.include(parentmodule(@__MODULE__), joinpath(@__DIR__, "../../../l06_qs_eda.jl"))
end
const QS = parentmodule(@__MODULE__).QualityStyleEDA
const MID, TB, CM, PM = QS.MID, QS.TB, QS.CM, QS.PM
const P = rstrip(normpath(joinpath(@__DIR__, "../../..")), '/')
const Q = joinpath(P, "results/QS")
const OUT = normpath(joinpath(@__DIR__, "../answers/batch_01"))
const FITROOT = "/root/BF_runs/market_model_qs/current_development/market_model/results/QS/fits"
const SEED = 610801
const RUNGS = ("R2", "R6", "C0")
const PROTOCOLS = ("10a", "10b")
const SENTINELS = ("eng_premier", "sco_league_two", "sco_premiership", "fin_veikkausliiga")
const MANIFEST = NamedTuple[]
const PANELS = Dict{String,Any}()
const FITS = Dict{Tuple{String,String,String},Any}()

sha(path) = bytes2hex(open(SHA.sha256, path))
function record!(request, kind, name, path; n=0, columns="", detail="")
    push!(MANIFEST, (; request, kind, name, path, sha256=isfile(path) ? sha(path) : "",
        n_rows=n, columns, detail))
end
function flush_manifest!()
    mkpath(OUT)
    path = joinpath(OUT, "manifest.csv")
    old = isfile(path) ? CSV.read(path, DF.DataFrame; stringtype=String) : DF.DataFrame()
    frame=vcat(old,DF.DataFrame(MANIFEST);cols=:union)
    lastrow=Dict{Tuple,Int}()
    for i in 1:DF.nrow(frame)
        r=frame[i,:]
        kind=endswith(r.kind,"output") ? "output" : r.kind
        lastrow[(r.request,kind,r.name,r.path,r.kind=="code" ? r.detail : "")]=i
    end
    CSV.write(path,frame[sort(collect(values(lastrow))),:])
    empty!(MANIFEST)
end
function output(request, name, rows)
    frame = rows isa DF.AbstractDataFrame ? rows : DF.DataFrame(rows)
    path = joinpath(OUT, name)
    CSV.write(path, frame)
    bulk=name in ("geometry_draws.csv","level_geometry_draws.csv","team_season_path_means.csv","null_statistics.csv")
    record!(request, bulk||filesize(path)>5_000_000 ? "beast_only_output" : "committed_output",
        name, path; n=DF.nrow(frame), columns=join(names(frame), ";"), detail="CSV; unavailable values blank")
    return frame
end
function fit(league, rung, protocol; request="shared")
    key = (league, rung, protocol)
    if !haskey(FITS, key)
        path = joinpath(FITROOT, league, "$(rung)_$(protocol).jls")
        saved = Serialization.deserialize(path)
        # C0 is serialized as (fit, widths); R2/R6 store ArmFit directly.
        FITS[key] = saved isa NamedTuple ? saved.fit : saved
        f = FITS[key]
        @assert size(f.draws) == size(f.udraws)
        @assert all(isfinite, f.udraws)
        record!(request, "input_fit", join(key,"/"), path;
            n=size(f.draws,1)*size(f.draws,3), columns=join(f.names,";"),
            detail="draws constrained; udraws linked; all retained post-warmup; size=$(size(f.draws))")
    end
    return FITS[key]
end
"128 deterministic post-warmup indices, evenly spaced within each chain."
function indices(f)
    n, _, nc = size(f.udraws)
    counts = [div(min(128,n*nc),nc)+(c<=rem(min(128,n*nc),nc) ? 1 : 0) for c in 1:nc]
    return [(chain=c, draw=i) for c in 1:nc for i in unique(round.(Int,range(1,n;length=counts[c])))]
end
seed(li, ri, pi, replicate=0) = SEED + 100_000li + 10_000ri + 1_000pi + replicate

"Axial tilt represented in [-90,90); distances to quality axis are abs(delta)."
function geometry(vq, vs, cross, k)
    total = vq+vs
    gap = hypot(vq-vs,2cross)
    lp, lm = (total+gap)/2, (total-gap)/2
    delta = mod(rad2deg(atan(2cross,vq-vs)/2)+90,180)-90
    c = vq>0 && vs>0 ? cross/sqrt(vq*vs) : missing
    @assert ismissing(c) || abs(c)<=1+1e-12
    c = ismissing(c) ? c : clamp(c,-1.0,1.0)
    return (; Vq=vq, Vs=vs, C=cross, r=vq>0 ? sqrt(vs/vq) : missing, c, k,
        phi_deg=mod(45-delta+90,180)-90, delta_deg=delta, l_plus=lp, l_minus=lm,
        f_minor=total>0 ? lm/total : missing, eigengap=total>0 ? gap/total : missing,
        rotation_gain=vs>0 ? 1-lm/vs : missing,
        I=ismissing(c)||abs(c)>=1 ? missing : -0.5log1p(-c^2))
end
function step_geometry(f, rung, i, c)
    value(name) = f.draws[i,only(findall(==(name),f.names)),c]
    if rung == "C0"
        vq,vs = value("sigma_q")^2,value("sigma_s")^2
        return geometry(vq,vs,0.0,1.0)
    end
    sa,sb = value("sigma_att"),value("sigma_def")
    rho = rung=="R6" ? value("rho_ab") : 0.0
    u,v,w = sa^2,sb^2,rho*sa*sb
    g = geometry((u+v-2w)/4,(u+v+2w)/4,(u-v)/4,sa/sb)
    restored = [g.Vq+g.Vs+2g.C g.Vs-g.Vq; g.Vs-g.Vq g.Vq+g.Vs-2g.C]
    @assert maximum(abs,restored-[u w;w v])/maximum(abs,[u w;w v]) <= 1e-10
    rung=="R2" && @assert abs(g.r-1)<=1e-12
    return g
end
function quant(values)
    v = collect(skipmissing(values))
    isempty(v) && return (missing,missing,missing)
    @assert all(isfinite,v)
    return Tuple(ST.quantile(v,[.05,.5,.95]))
end
function verification(request, text)
    open(joinpath(OUT,"VERIFICATION.md"),"a") do io
        println(io,"\n## $request\n",text)
    end
    mkpath(joinpath(OUT,"logs"))
    open(joinpath(OUT,"logs","checks.log"),"a") do io
        println(io,"CHECK $request ",text)
    end
    open("/root/BF_runs/logs/market_model_qsf/checks.log","a") do io
        println(io,"CHECK $request ",text)
    end
    println("CHECK $request ", text)
    flush(stdout)
end
function summary!(request, lines)
    path = joinpath(OUT,"SUMMARY.md")
    text = isfile(path) ? read(path,String) : "# Batch 01 results\n\nInput/output hashes, exact roots, row counts and seeds: [manifest.csv](manifest.csv).\nLarge tables remain on the beast, per manager instruction. Units: natural-log rates, calendar weeks, nats.\n"
    @assert length(lines)<=10
    section="## $request —\n"*join(lines,"\n")*"\n"
    pattern=Regex("(?ms)^## $request —\\n.*?(?=^## |\\z)")
    text=occursin(pattern,text) ? replace(text,pattern=>section*"\n") : text*"\n"*section
    write(path,text)
end
function r01()
    mkpath(OUT)
    started = time()
    summaries = NamedTuple[]
    path = joinpath(OUT,"geometry_draws.csv")
    firstwrite = true
    for (league,_,_) in QS.LEAGUES, rung in RUNGS, protocol in PROTOCOLS
        f = fit(league,rung,protocol;request="R01")
        rows = [(;league,rung,protocol,chain=c,draw=i,step_geometry(f,rung,i,c)...)
            for c in axes(f.draws,3) for i in axes(f.draws,1)]
        frame = DF.DataFrame(rows)
        @assert rung!="C0" || all(iszero,frame.c)
        CSV.write(path,frame;append=!firstwrite)
        firstwrite = false
        probs = (;prob_abs_delta_gt_10deg=ST.mean(abs.(frame.delta_deg).>10),
            prob_abs_c_lt_0p1=ST.mean(abs.(frame.c).<.1),prob_minor_lt_0p05=ST.mean(frame.f_minor.<.05),
            prob_eigengap_lt_0p1=ST.mean(frame.eigengap.<.1))
        for quantity in names(frame)[6:end]
            q05,median,q95=quant(frame[!,quantity])
            push!(summaries,(;league,rung,protocol,quantity,q05,median,q95,probs...))
        end
        println("R01 ",league," ",rung," ",protocol," draws=",length(rows))
    end
    record!("R01","beast_only_output","geometry_draws.csv",path;n=792000,
        columns="league;rung;protocol;chain;draw;Vq;Vs;C;r;c;k;phi_deg;delta_deg;l_plus;l_minus;f_minor;eigengap;rotation_gain;I",
        detail="all 12000 retained draws per 66 fits; angle degrees axial; static states not involved")
    result=output("R01","geometry_summary.csv",summaries)
    lines=["COMPLETE. [geometry_summary.csv](geometry_summary.csv); draw table is beast-only (manifest). Probabilities repeated per quantity; zero denominators are blank, not dropped."]
    for league in SENTINELS
        g=only(eachrow(DF.filter(r->r.league==league && r.rung=="R6" && r.protocol=="10a" && r.quantity=="delta_deg",result)))
        push!(lines,"$league R6 10a delta=$(round(g.median;digits=2))° [$(round(g.q05;digits=2)),$(round(g.q95;digits=2))]; P(|delta|>10°)=$(round(g.prob_abs_delta_gt_10deg;digits=3)).")
    end
    tilts=DF.filter(r->r.rung=="R6" && r.protocol=="10a" && r.quantity=="delta_deg" && r.prob_eigengap_lt_0p1<=.1,result)
    order=sortperm(abs.(tilts.median);rev=true)[1:min(3,DF.nrow(tilts))]
    push!(lines,"Three largest median tilts with ≥90% eigengap>0.1: "*join(["$(r.league) $(round(r.median;digits=2))° (P>10°=$(round(r.prob_abs_delta_gt_10deg;digits=3)))" for r in eachrow(tilts[order,:])],"; ")*".")
    gains=DF.filter(r->r.rung=="R6" && r.quantity=="rotation_gain",result)
    g=gains[argmax(gains.median),:]
    push!(lines,"Largest R6 rotation residual-variance reduction 1−l_minus/Vs: $(g.league) $(g.protocol), median $(round(g.median;digits=4)) [$(round(g.q05;digits=4)),$(round(g.q95;digits=4))]; this is not a forecast gain or refitted R6–C0 comparison.")
    a=DF.filter(r->r.rung=="R6"&&r.quantity=="delta_deg"&&r.protocol=="10a",result)
    b=DF.filter(r->r.rung=="R6"&&r.quantity=="delta_deg"&&r.protocol=="10b",result)
    changes=[abs(only(b.median[b.league.==l])-only(a.median[a.league.==l])) for l in a.league]
    push!(lines,"Largest 10a/10b median tilt change: $(a.league[argmax(changes)]) $(round(maximum(changes);digits=2))°. No R6 10a fit meets P(|delta|>10°)>0.9: $(all(a.prob_abs_delta_gt_10deg.<=.9)).")
    summary!("R01",lines)
    verification("R01","PASS covariance round-trip relative error ≤1e−10; R2 r=1 ≤1e−12; C0 c=0; $(792000) draws. Runtime=$(round(time()-started;digits=2))s. Invocation: Q07.r01() in owned beast REPL; seed=$SEED (algebra deterministic).")
    flush_manifest!()
end

"Rebuild the original panel from the immutable saved rates and pinned metadata; no SQL."
function panel(league; request="shared")
    if !haskey(PANELS,league)
        _,tournament,segment = only(filter(e->e[1]==league,QS.LEAGUES))
        cr = only(eachrow(DF.filter(:league=>==(league),CSV.read(joinpath(Q,"configs.csv"),DF.DataFrame;stringtype=String))))
        config = QS.MM.MarketModelConfig(;name=league,segment,tournaments=[tournament],
            seasons=String.(split(cr.seasons,";")),honest_train=String.(split(cr.honest_train,";")),
            honest_test=[String(cr.honest_test)])
        cache=joinpath(dirname(dirname(P)),".cache","datastore_$(nameof(typeof(segment))).jls")
        @assert sha(cache)==QS.SNAPSHOT_HASHES[string(nameof(typeof(segment)))]
        ds=Serialization.deserialize(cache)
        rp=joinpath(Q,"rates_$(league).csv")
        p=TB.phase_b_panel(ds;config,rates_path=rp).panel
        PANELS[league]=(;p,config)
        record!(request,"input_rates",league,rp;n=length(p.matches.match_id),columns=join(names(CSV.read(rp,DF.DataFrame)),";"),
            detail="reconstructed saved panel via phase_b_panel; 7-day calendar slots; roster includes all structural teams")
        record!(request,"input_config",league,joinpath(Q,"configs.csv");detail="train=$(cr.honest_train); test=$(cr.honest_test)")
    end
    return PANELS[league]
end
function filter_store(f,p,theta,rung)
    if rung=="R2"
        return TB.rung_filter(f.arm,p,theta;store=true)
    elseif rung=="R6"
        return CM.covariance_filter(f.arm,p,theta;store=true)
    end
    return QS.FG.evaluator(f.arm,p)(f.arm,p,theta;store=true)
end
function point_path(f,p,theta,rung,filtered=filter_store(f,p,theta,rung))
    if rung=="R2"
        return MID.smoothed_mean(f.arm,p,TB.rung_schedule(f.arm,p,theta),filtered)
    elseif rung=="R6"
        return MID.smoothed_mean(CM.conditional_arm(f.arm,theta),p,CM.covariance_schedule(f.arm,p,theta),filtered)
    end
    # Means-only RTS: same gain arithmetic as fullbook_smoothing, no unused smoothed covariances.
    X=copy(filtered.m_filt)
    for t in (p.n_weeks-1):-1:1
        Pt=filtered.P_filt[:,:,t]
        Cp=LA.cholesky(LA.Symmetric(Pt+filtered.Q);check=false)
        J=LA.issuccess(Cp) ? (Cp\Pt')' : Pt*LA.pinv(Pt+filtered.Q)
        X[:,t].=filtered.m_filt[:,t]+J*(X[:,t+1]-filtered.m_filt[:,t])
    end
    return X
end
"Exact joint C0 FFBS, preserving static mu/gamma across the sampled path."
function c0_ffbs(p,f,rng)
    X=zeros(size(f.m_filt))
    MID.draw_psd!(view(X,:,p.n_weeks),rng,f.m_filt[:,end],f.P_filt[:,:,end])
    for t in (p.n_weeks-1):-1:1
        Pt=f.P_filt[:,:,t]
        Cp=LA.cholesky(LA.Symmetric(Pt+f.Q);check=false)
        J=LA.issuccess(Cp) ? (Cp\Pt')' : Pt*LA.pinv(Pt+f.Q)
        m=f.m_filt[:,t]+J*(X[:,t+1]-f.m_filt[:,t])
        V=Pt-J*Pt
        MID.draw_psd!(view(X,:,t),rng,m,(V+V')/2)
        X[1:2,t].=X[1:2,t+1]
    end
    return X
end
function sample_path(f,p,theta,rung,rng)
    filtered=filter_store(f,p,theta,rung)
    rung=="C0" && return c0_ffbs(p,filtered,rng)
    a=rung=="R2" ? f.arm : CM.conditional_arm(f.arm,theta)
    sch=rung=="R2" ? TB.rung_schedule(f.arm,p,theta) : CM.covariance_schedule(f.arm,p,theta)
    return MID.ffbs(a,p,sch,filtered,rng)
end
function season_design(p,season)
    m=DF.filter(:season=>==(season),p.matches)
    active=sort(unique(vcat(m.home_team,m.away_team)))
    ix=[only(findall(==(t),p.teams)) for t in active]
    ow=p.obs_week[p.obs_season.==season]
    return (;active,ix,weeks=minimum(ow):maximum(ow))
end
function level_values(p,X,design,gauge)
    N=MID.n_teams(p)
    a,b=copy(X[3:2+N,design.weeks]),copy(X[3+N:2+2N,design.weeks])
    centering=gauge=="full_roster" ? (1:N) : design.ix
    a.-=ST.mean(a[centering,:];dims=1)
    b.-=ST.mean(b[centering,:];dims=1)
    a,b=a[design.ix,:],b[design.ix,:]
    abar,bbar=vec(ST.mean(a;dims=2)),vec(ST.mean(b;dims=2))
    q,s=(a-b)/2,(a+b)/2
    qb,sb=(abar-bbar)/2,(abar+bbar)/2
    rho=ST.cor(abar,bbar)
    vq,vs,cross=ST.var(qb),ST.var(sb),ST.cov(qb,sb)
    geom=geometry(vq,vs,cross,ST.std(abar)/ST.std(bbar))
    quantities = (; r_level=geom.r,c_level=geom.c,k_level=geom.k,rho_ab=rho,
        shortcut_r=rho<1 ? sqrt(max(0.0,(1+rho)/(1-rho))) : missing,delta_level_deg=geom.delta_deg,
        r_weekly_cross_section=sqrt(ST.mean(vec(ST.var(s;dims=1)))/ST.mean(vec(ST.var(q;dims=1)))))
    return quantities, abar, bbar, qb, sb
end
function r02()
    started=time()
    drawrows,meansrows,stats,checks,windows=NamedTuple[],NamedTuple[],NamedTuple[],NamedTuple[],NamedTuple[]
    published=CSV.read(joinpath(Q,"levels_by_league.csv"),DF.DataFrame;stringtype=String)
    team_pub=CSV.read(joinpath(Q,"level_teams.csv"),DF.DataFrame;stringtype=String)
    # Benchmark an actual largest-roster panel, then run the same 128 paths without thinning further.
    for (li,(league,_,_)) in enumerate(QS.LEAGUES)
        p,config=panel(league;request="R02")
        for season in config.seasons
            d=season_design(p,season)
            push!(windows,(;league,season,n_teams=length(d.active),n_weeks=length(d.weeks),
                first_week=first(d.weeks),last_week=last(d.weeks),first_date=minimum(p.matches.match_date[p.matches.season.==season]),
                last_date=maximum(p.matches.match_date[p.matches.season.==season]),
                roster=join(d.active,";"),label="accepted-panel season window; near-complete Scottish lower 25/26 and pre-split Premiership retained as published"))
        end
        for (ri,rung) in enumerate(RUNGS)
            f=fit(league,rung,"10a";request="R02")
            theta=MID.median_theta(f)
            X=point_path(f,p,theta,rung)
            # Gate exact published all-roster centering, season-average indexing and teams.
            for season in config.seasons
                d=season_design(p,season)
                v,a,b,q,s=level_values(p,X,d,"full_roster")
                pub=only(eachrow(DF.filter(r->r.league==league&&r.rung==rung&&r.protocol=="10a"&&r.season==season,published)))
                for (quantity,value,expected) in (("measured_r",v.r_level,pub.measured_r),("implied_r",v.shortcut_r,pub.implied_r),
                    ("corr_alpha_beta",v.rho_ab,pub.corr_alpha_beta),("corr_q_s",v.c_level,pub.corr_q_s))
                    push!(checks,(;league,rung,season,quantity,error=abs(value-expected),pass=abs(value-expected)<=1e-6))
                end
                for (j,team) in enumerate(d.active)
                    pubt=only(eachrow(DF.filter(r->r.league==league&&r.rung==rung&&r.protocol=="10a"&&r.season==season&&r.team==team,team_pub)))
                    err=maximum(abs,[a[j]-pubt.alpha,b[j]-pubt.beta,q[j]-pubt.q,s[j]-pubt.s])
                    push!(checks,(;league,rung,season,quantity="team:"*team,error=err,pass=err<=1e-6))
                end
            end
            if !all(r.pass for r in checks)
                output("R02","reproduction_checks.csv",checks)
                flush_manifest!()
                error("published R02 reproduction failed; see reproduction_checks.csv")
            end
            idx=indices(f)
            paths=Vector{Matrix{Float64}}(undef,length(idx))
            ts=time()
            Threads.@threads for j in eachindex(idx)
                c,i=idx[j].chain,idx[j].draw
                paths[j]=sample_path(f,p,f.udraws[i,:,c],rung,Random.Xoshiro(seed(li,ri,1,j)))
            end
            println("R02 FFBS $league $rung n=$(length(idx)) elapsed=$(time()-ts)s");flush(stdout)
            record!("R02","draw_selection",league*"/"*rung,"";n=length(idx),
                detail="seed=$(seed(li,ri,1)); replicate=index in chain-major list; indices="*join(["$(x.chain):$(x.draw)" for x in idx],";"))
            for season in config.seasons, gauge in ("full_roster","active_cohort")
                d=season_design(p,season)
                vals=NamedTuple[]
                for (j,path) in enumerate(paths)
                    v,a,b,q,s=level_values(p,path,d,gauge)
                    g=step_geometry(f,rung,idx[j].draw,idx[j].chain)
                    push!(vals,(;v...,r_step=g.r,level_step_ratio=v.r_level/g.r))
                    push!(drawrows,(;league,rung,season,draw=j,gauge,vals[end]...))
                    for (t,team) in enumerate(d.active)
                        push!(meansrows,(;league,rung,season,draw=j,gauge,team,alpha=a[t],beta=b[t],q=q[t],s=s[t]))
                    end
                end
                point,_,_,_,_=level_values(p,X,d,gauge)
                gr=geometry_for_theta(f,rung,theta)
                point=(;point...,r_step=gr.r,level_step_ratio=point.r_level/gr.r)
                for method in ("RTS_point","FFBS"), quantity in propertynames(point)
                    qs=method=="RTS_point" ? ntuple(_->getproperty(point,quantity),3) : quant(getproperty.(vals,quantity))
                    n_available=method=="RTS_point" ? Int(!ismissing(getproperty(point,quantity))) : count(!ismissing,getproperty.(vals,quantity))
                    n_missing=(method=="RTS_point" ? 1 : length(vals))-n_available
                    push!(stats,(;league,rung,protocol="10a",season,method,gauge,n_teams=length(d.active),n_weeks=length(d.weeks),quantity=String(quantity),q05=qs[1],median=qs[2],q95=qs[3],n_available,n_missing,reason=n_missing>0 ? "NOT_IDENTIFIABLE: denominator zero or exact rank-one two-team correlation" : "OK"))
                end
            end
            cache=joinpath(OUT,"cache","$(league)_$(rung)_point.jls")
            mkpath(dirname(cache));Serialization.serialize(cache,(;X,theta))
            record!("R02","beast_only_cache",basename(cache),cache;detail="median-theta RTS raw uncentered alpha/beta; reused R04-R06")
        end
    end
    output("R02","reproduction_checks.csv",checks)
    output("R02","season_windows.csv",windows)
    result=output("R02","level_geometry.csv",stats)
    draws=output("R02","level_geometry_draws.csv",drawrows)
    output("R02","team_season_path_means.csv",meansrows)
    lines=["COMPLETE. [level_geometry.csv](level_geometry.csv), [reproduction_checks.csv](reproduction_checks.csv), [season_windows.csv](season_windows.csv); path means/draw table locations and hashes in manifest. Exact joint FFBS, 128 theta draws/fit; fixed gauges never differenced across changing rosters."]
    for league in SENTINELS
        g=DF.filter(r->r.league==league&&r.rung=="C0"&&r.method=="FFBS"&&r.gauge=="full_roster"&&r.quantity=="level_step_ratio",result)
        push!(lines,"$league C0 level/step: "*join(["$(r.season) $(round(r.median;digits=2)) [$(round(r.q05;digits=2)),$(round(r.q95;digits=2))]" for r in eachrow(g)],"; ")*".")
    end
    errors=coalesce.(abs.(draws.shortcut_r-draws.r_level),-Inf)
    g=draws[argmax(errors),:]
    push!(lines,"Largest FFBS shortcut absolute error: $(g.league) $(g.rung) $(g.season), $(round(maximum(errors);digits=4)); shortcut assumes equal alpha/beta variances, not zero q/s correlation.")
    groups=DF.groupby(draws,[:league,:rung,:season,:gauge])
    nominations=["$(first(g.league))/$(first(g.rung))/$(first(g.season))/$(first(g.gauge))" for g in groups if ST.mean((g.level_step_ratio.<.75).|(g.level_step_ratio.>1.25))>.9]
    output("R02","level_step_screen.csv",[(;league=first(g.league),rung=first(g.rung),season=first(g.season),gauge=first(g.gauge),prob_outside_0p75_1p25=ST.mean((g.level_step_ratio.<.75).|(g.level_step_ratio.>1.25))) for g in groups])
    push!(lines,"Level/step screen >0.9 in $(length(nominations)) league/rung/season/gauge cells; [level_step_screen.csv](level_step_screen.csv). Partial-season labels and roster membership are in season_windows; these are latent-population, not independent team replicates.")
    summary!("R02",lines)
    verification("R02","PASS all published 10a point summary/team quantities within 1e−6; max error=$(maximum(r.error for r in checks)). 33 fits ×128 exact joint FFBS paths; static mu/gamma preserved. Runtime=$(round(time()-started;digits=2))s. Invocation: Q07.r02().")
    flush_manifest!()
end
function geometry_for_theta(f,rung,theta)
    # Median theta is not the median of nonlinear derived quantities.
    vals=exp.(theta)
    if rung=="C0"
        return geometry(vals[2]^2,vals[3]^2,0.0,1.0)
    end
    sa,sb=vals[2:3]
    rho=rung=="R6" ? tanh(theta[4]) : 0.0
    return geometry((sa^2+sb^2-2rho*sa*sb)/4,(sa^2+sb^2+2rho*sa*sb)/4,(sa^2-sb^2)/4,sa/sb)
end

"Exact scalar forward filter, explicit pre-week snapshot before all observations of the week."
function forward(f,p,theta,rung; keep_states=false, min_state_week=1)
    N=MID.n_teams(p)
    H=PM.fullbook_design(PM.FullBookRung(:C0),p,theta)
    Q=rung=="C0" ? PM.fullbook_process(f.arm,N,theta) : MID.process_cov(CM.conditional_arm(f.arm,theta),N,CM.covariance_schedule(f.arm,p,theta),2)
    m,V=MID.initial_state(MID.GRW1(),N)
    variance=exp(2theta[1]); k=zeros(length(m))
    means=zeros(MID.n_fixtures(p),2);covs=zeros(MID.n_fixtures(p),2,2)
    states=Dict{Int,Any}(); last_observed=0
    for t in 1:p.n_weeks
        t>=2 && (V.+=Q)
        js=p.week_ptr[t]:(p.week_ptr[t+1]-1)
        @assert last_observed<t
        if !isempty(js)
            keep_states && t>=min_state_week && (states[t]=(;m=copy(m),V=copy(V),last_observed))
            for j in first(js):2:last(js)
                i=cld(j,2);B=H[j:j+1,:]
                means[i,:].=B*m
                covs[i,:,:].=B*V*B'+variance*Matrix(LA.I,2,2)
            end
        end
        for j in js
            h=view(H,j,:);LA.mul!(k,V,h)
            S=LA.dot(h,k)+variance
            @assert S>0 && isfinite(S)
            e=p.obs_y[j]-LA.dot(h,m)
            LA.axpy!(e/S,k,m);LA.BLAS.ger!(-1/S,k,k,V)
            last_observed=t
        end
        V.=(V+V')/2
    end
    return (;means,covs,states)
end
logmeanexp(v) = (a=maximum(v); a+log(ST.mean(exp.(v.-a))))
function mixture_quantile(means,sds,prob)
    lo=minimum(means.-10sds);hi=maximum(means.+10sds)
    for _ in 1:60
        mid=(lo+hi)/2
        c=ST.mean(DS.cdf.(DS.Normal.(means,sds),mid))
        c<prob ? (lo=mid) : (hi=mid)
    end
    return (lo+hi)/2
end
"999 noncircular moving-block resamples of observed weeks within league-season, retaining whole weeks."
function bootstrap_indices(frame;block=8,rng=Random.Xoshiro(SEED))
    groups=[collect(DF.groupby(g,:week)) for g in DF.groupby(frame,:season)]
    # Original row indices are retained, not fixture-resampled.
    groups=[[collect(parentindices(w)[1]) for w in ws] for ws in groups]
    # parentindices of nested SubDataFrames refer to the original parent frame.
    out=Vector{Vector{Int}}(undef,999)
    for b in 1:999
        ix=Int[]
        for weeks in groups
            n=length(weeks);lengthblock=min(block,n);chosen=Int[]
            while length(chosen)<n
                start=Random.rand(rng,1:n-lengthblock+1)
                append!(chosen,start:start+lengthblock-1)
            end
            for j in chosen[1:n]
                append!(ix,weeks[j])
            end
        end
        out[b]=ix
    end
    return out
end
function safe_cor(x,y)
    length(x)<3 && return missing
    (ST.std(x)==0 || ST.std(y)==0) && return missing
    return ST.cor(x,y)
end
function weekly_lag(frame,col,lag;squared=false)
    x,y=Float64[],Float64[]
    for g in DF.groupby(frame,:season)
        weeks=sort(unique(g.week)); vals=Dict(w=>ST.mean(g[g.week.==w,col]) for w in weeks)
        for w in weeks
            haskey(vals,w-lag) || continue
            a,b=vals[w],vals[w-lag]
            push!(x,squared ? a^2 : a);push!(y,squared ? b^2 : b)
        end
    end
    return safe_cor(x,y)
end
function diag_stat(frame,col,kind)
    v=frame[!,col]
    kind=="mean" && return ST.mean(v)
    kind=="mean_square" && return ST.mean(v.^2)
    kind=="cross_correlation" && return safe_cor(frame.u_D,frame.u_M)
    startswith(kind,"weekly_") && return weekly_lag(frame,col,parse(Int,last(split(kind,"_")));squared=occursin("squared",kind))
    error("unknown statistic $kind")
end
function paired_comparison(request,frame;left="C0",right="R6",prefix="logp",outfile="joint_comparison.csv")
    keys=[:league,:season,:week,:fixture_id]
    a=DF.filter(:rung=>==(left),frame);b=DF.filter(:rung=>==(right),frame)
    @assert length(unique(zip(a.league,a.fixture_id)))==DF.nrow(a)
    @assert length(unique(zip(b.league,b.fixture_id)))==DF.nrow(b)
    joined=DF.innerjoin(a,b;on=keys,makeunique=true)
    @assert DF.nrow(joined)==DF.nrow(a)==DF.nrow(b)
    rows=NamedTuple[]
    for block in (4,8,12), channel in ("joint","marginal_sum")
        perleague=NamedTuple[]
        for (li,(league,_,_)) in enumerate(QS.LEAGUES)
            g=DF.filter(:league=>==(league),joined)
            delta=channel=="joint" ? g.logp_joint-g.logp_joint_1 : (g.logp_D+g.logp_M)-(g.logp_D_1+g.logp_M_1)
            indices=bootstrap_indices(g;block,rng=Random.Xoshiro(seed(li,2,2,block)))
            boot=[ST.mean(delta[ix]) for ix in indices]
            ci=quant(boot)
            push!(rows,(;league,weighting="fixture",n=length(delta),score_kind=channel,
                mean_C0_minus_R6=ST.mean(delta),boot_q05=ci[1],boot_q95=ci[3],block_weeks=block))
            push!(perleague,(;n=length(delta),mean=ST.mean(delta),boot))
        end
        for weighting in ("fixture","equal_league")
            weights=weighting=="fixture" ? [g.n for g in perleague] : ones(length(perleague))
            weights=weights/sum(weights)
            boots=sum(weights[i]*perleague[i].boot for i in eachindex(perleague));ci=quant(boots)
            push!(rows,(;league="ALL",weighting,n=sum(g.n for g in perleague),score_kind=channel,
                mean_C0_minus_R6=sum(weights[i]*perleague[i].mean for i in eachindex(perleague)),boot_q05=ci[1],boot_q95=ci[3],block_weeks=block))
        end
    end
    return output(request,outfile,rows)
end
function r03()
    started=time();rows,diagnostics,venues,checks=NamedTuple[],NamedTuple[],NamedTuple[],NamedTuple[]
    R=[1.0 -1.0;.5 .5]
    published=CSV.read(joinpath(Q,"forecast_fixtures.csv"),DF.DataFrame;stringtype=String)
    for (li,(league,_,_)) in enumerate(QS.LEAGUES)
        p,config=panel(league;request="R03")
        testix=findall(in.(p.matches.season,Ref(config.honest_test)))
        trainix=findall(in.(p.matches.season,Ref(config.honest_train)))
        @assert maximum(p.matches.match_date[trainix])<minimum(p.matches.match_date[testix])
        @assert maximum(p.matches.week[trainix])<minimum(p.matches.week[testix])
        record!("R03","split",league,"";n=length(testix),detail="train_first=$(minimum(p.matches.match_date[trainix])); train_end=$(maximum(p.matches.match_date[trainix])); test_first=$(minimum(p.matches.match_date[testix])); test_end=$(maximum(p.matches.match_date[testix])); train/test week disjoint")
        for (ri,rung) in enumerate(("C0","R6"))
            f=fit(league,rung,"10b";request="R03");idx=indices(f)
            predictions=Vector{Any}(undef,length(idx));ts=time()
            Threads.@threads for j in eachindex(idx)
                predictions[j]=forward(f,p,f.udraws[idx[j].draw,:,idx[j].chain],rung;
                    keep_states=rung=="C0",min_state_week=minimum(p.matches.week[testix]))
            end
            if rung=="C0"
                cache=joinpath(OUT,"cache","$(league)_C0_10b_forecasts.jls")
                mkpath(dirname(cache));Serialization.serialize(cache,(;predictions,indices=idx))
                record!("R03","beast_only_cache",basename(cache),cache;n=length(idx),detail="pre-week state mean/covariance, same128 theta indices, test-only snapshots for R07")
            end
            medianpred=forward(f,p,MID.median_theta(f),rung)
            for i in testix
                fixture=p.matches[i,:];y=R*p.obs_y[2i-1:2i]
                mus=[R*v.means[i,:] for v in predictions]
                covs=[R*v.covs[i,:,:]*R' for v in predictions]
                @assert all(LA.isposdef(LA.Symmetric(v)) for v in covs)
                m=ST.mean(mus);S=ST.mean([covs[j]+mus[j]*mus[j]' for j in eachindex(mus)])-m*m'
                @assert LA.isposdef(LA.Symmetric(S))
                lj=logmeanexp([DS.logpdf(DS.MvNormal(mus[j],LA.Symmetric(covs[j])),y) for j in eachindex(mus)])
                ld=logmeanexp([DS.logpdf(DS.Normal(mus[j][1],sqrt(covs[j][1,1])),y[1]) for j in eachindex(mus)])
                lm=logmeanexp([DS.logpdf(DS.Normal(mus[j][2],sqrt(covs[j][2,2])),y[2]) for j in eachindex(mus)])
                @assert all(isfinite,(lj,ld,lm))
                covers=[mixture_quantile(getindex.(mus,k),[sqrt(v[k,k]) for v in covs],.05)<=y[k]<=mixture_quantile(getindex.(mus,k),[sqrt(v[k,k]) for v in covs],.95) for k in 1:2]
                push!(rows,(;league,season=String(fixture.season),week=fixture.week,fixture_id=fixture.match_id,rung,n_draws=length(idx),
                    y_D=y[1],y_M=y[2],mean_D=m[1],mean_M=m[2],var_D=S[1,1],var_M=S[2,2],cov_DM=S[1,2],
                    u_D=(y[1]-m[1])/sqrt(S[1,1]),u_M=(y[2]-m[2])/sqrt(S[2,2]),logp_joint=lj,logp_D=ld,logp_M=lm,
                    covered90_D=covers[1],covered90_M=covers[2],home_team=fixture.home_team,away_team=fixture.away_team))
                mp=R*medianpred.means[i,:];vp=R*medianpred.covs[i,:,:]*R'
                for (k,axis) in enumerate(("supremacy","level"))
                    pub=DF.filter(r->r.league==league&&r.rung==rung&&r.protocol=="10b"&&r.match_id==fixture.match_id&&r.axis==axis,published)
                    @assert DF.nrow(pub)==1
                    err=abs(DS.logpdf(DS.Normal(mp[k],sqrt(vp[k,k])),y[k])-pub.logpd[1])
                    push!(checks,(;league,rung,fixture_id=fixture.match_id,axis,error=err,pass=err<=1e-8))
                end
            end
            @assert all(r.pass for r in checks)
            println("R03 $league $rung 128 forward filters elapsed=$(time()-ts)s");flush(stdout)
        end
    end
    forecasts=output("R03","forecast_joint.csv",rows)
    output("R03","forecast_reproduction_checks.csv",checks)
    comparison=paired_comparison("R03",forecasts)
    for (li,(league,_,_)) in enumerate(QS.LEAGUES), rung in ("C0","R6")
        g=DF.filter(r->r.league==league&&r.rung==rung,forecasts)
        ix=bootstrap_indices(g;rng=Random.Xoshiro(seed(li,3,2)))
        for axis in ("D","M")
            col=Symbol("u_"*axis)
            for kind in ("mean","mean_square","cross_correlation","weekly_signed_lag_1","weekly_signed_lag_2","weekly_signed_lag_4","weekly_squared_lag_1","weekly_squared_lag_2","weekly_squared_lag_4")
                val=diag_stat(g,col,kind)
                # Lag-pair rows are resampled by their later week: no fabricated cross-block neighbors.
                boot=startswith(kind,"weekly_") ? lag_pair_bootstrap(g,col,kind,ix) : [diag_stat(g[ind,:],col,kind) for ind in ix]
                ci=quant(boot)
                push!(diagnostics,(;league,rung,axis,statistic=kind,n=DF.nrow(g),value=val,boot_q05=ci[1],boot_q95=ci[3]))
            end
            coverage=Symbol("covered90_"*axis);boot=[ST.mean(g[ind,coverage]) for ind in ix];ci=quant(boot)
            push!(diagnostics,(;league,rung,axis,statistic="coverage90",n=DF.nrow(g),value=ST.mean(g[!,coverage]),boot_q05=ci[1],boot_q95=ci[3]))
            teams=sort(unique(vcat(g.home_team,g.away_team)))
            lagx,lagy=Float64[],Float64[]
            for team in teams
                nh=count(==(team),g.home_team);na=count(==(team),g.away_team)
                for season in unique(g.season)
                    tg=DF.sort(DF.filter(r->r.season==season&&(r.home_team==team||r.away_team==team),g),[:week,:fixture_id])
                    vals=[axis=="D"&&r.away_team==team ? -r[col] : r[col] for r in eachrow(tg)]
                    length(vals)>1 && (append!(lagx,vals[2:end]);append!(lagy,vals[1:end-1]))
                end
                nh>=8&&na>=8 || continue
                contrast=ST.mean(g[g.home_team.==team,col])-ST.mean((axis=="D" ? -1 : 1)*g[g.away_team.==team,col])
                boots=Float64[]
                for ind in ix
                    h=ind[g.home_team[ind].==team];a=ind[g.away_team[ind].==team]
                    isempty(h)||isempty(a) || push!(boots,ST.mean(g[h,col])-ST.mean((axis=="D" ? -1 : 1)*g[a,col]))
                end
                push!(venues,(;league,rung,team,axis,n_home=nh,n_away=na,contrast,bootstrap_se=ST.std(boots)))
            end
            # Appearance-lag one, within season; lag-pair bootstrap retains whole later weeks.
            ci=quant(team_lag_bootstrap(g,col,axis,ix))
            push!(diagnostics,(;league,rung,axis,statistic="within_team_appearance_lag1_covariance",n=length(lagx),value=length(lagx)>1 ? ST.cov(lagx,lagy) : missing,boot_q05=ci[1],boot_q95=ci[3]))
        end
    end
    ds=output("R03","forecast_diagnostics.csv",diagnostics)
    output("R03","venue_contrasts.csv",venues)
    lines=["COMPLETE. [forecast_joint.csv](forecast_joint.csv), [forecast_diagnostics.csv](forecast_diagnostics.csv), [joint_comparison.csv](joint_comparison.csv), [venue_contrasts.csv](venue_contrasts.csv). D=h−a, M=(h+a)/2; determinant=1, no Jacobian constant. Published median-theta marginal scores reproduced ≤1e−8."]
    for r in eachrow(DF.filter(r->r.league=="ALL"&&r.score_kind=="joint"&&r.block_weeks==8,comparison))
        push!(lines,"Pooled $(r.weighting) joint C0−R6 $(round(r.mean_C0_minus_R6;digits=5)) [$(round(r.boot_q05;digits=5)),$(round(r.boot_q95;digits=5))] nats/fixture; n=$(r.n).")
    end
    for league in SENTINELS
        g=DF.filter(r->r.league==league&&r.rung=="C0"&&r.statistic in ("mean_square","coverage90"),ds)
        push!(lines,"$league C0 "*join(["$(r.axis) $(r.statistic)=$(round(r.value;digits=3))" for r in eachrow(g)],"; ")*".")
    end
    push!(lines,"Serial/team lag uncertainty uses 999 week-block resamples of original lag-pair rows indexed by their later week, without fabricating cross-block neighbors. Team lag means appearance-lag one within season. Venue contrast/SE distribution is unselected with ≥8 appearances per role; tables retain all teams, not selected stars.")
    summary!("R03",lines)
    verification("R03","PASS pre-week last_observed<t in every loop; theta training dates and week labels strictly precede test; all predictive covariances PD, finite scores, unique paired fixture keys; n=$(DF.nrow(forecasts)÷2). Median marginal score reproduction max=$(maximum(r.error for r in checks)). Bootstrap seed deterministic, 999 samples, blocks4/8/12 for comparisons. Runtime=$(round(time()-started;digits=2))s. Invocation: Q07.r03().")
    flush_manifest!()
end

function lag_pair_bootstrap(frame,col,kind,indices)
    lag=parse(Int,last(split(kind,"_")));squared=occursin("squared",kind)
    pairs=NamedTuple[]
    for g in DF.groupby(frame,:season)
        weeks=sort(unique(g.week));values=Dict(w=>ST.mean(g[g.week.==w,col]) for w in weeks)
        for w in weeks
            haskey(values,w-lag)||continue
            a,b=values[w],values[w-lag]
            row=first(findall((frame.season.==first(g.season)).&(frame.week.==w)))
            push!(pairs,(;row,x=squared ? a^2 : a,y=squared ? b^2 : b))
        end
    end
    return [begin
        counts=zeros(Int,DF.nrow(frame));for i in ind;counts[i]+=1;end
        x,y=Float64[],Float64[]
        for p in pairs
            append!(x,fill(p.x,counts[p.row]));append!(y,fill(p.y,counts[p.row]))
        end
        safe_cor(x,y)
    end for ind in indices]
end
function team_lag_bootstrap(frame,col,axis,indices)
    pairs=NamedTuple[]
    for team in unique(vcat(frame.home_team,frame.away_team)),season in unique(frame.season)
        rows=findall((frame.season.==season).&((frame.home_team.==team).|(frame.away_team.==team)))
        sort!(rows;by=i->(frame.week[i],frame.fixture_id[i]))
        val(i)=axis=="D"&&frame.away_team[i]==team ? -frame[i,col] : frame[i,col]
        for j in 2:length(rows)
            push!(pairs,(;row=rows[j],x=val(rows[j]),y=val(rows[j-1])))
        end
    end
    return [begin
        counts=zeros(Int,DF.nrow(frame));for i in ind;counts[i]+=1;end
        x,y=Float64[],Float64[]
        for p in pairs
            append!(x,fill(p.x,counts[p.row]));append!(y,fill(p.y,counts[p.row]))
        end
        length(x)>1 ? ST.cov(x,y) : missing
    end for ind in indices]
end
function r02_benchmark()
    league="eng_premier";p,config=panel(league;request="R02_benchmark")
    rows=NamedTuple[]
    for (ri,rung) in enumerate(RUNGS)
        f=fit(league,rung,"10a";request="R02_benchmark")
        theta=MID.median_theta(f)
        sample_path(f,p,theta,rung,Random.Xoshiro(seed(1,ri,1))) # compilation warmup
        ts=time()
        Threads.@threads for j in 1:16
            sample_path(f,p,theta,rung,Random.Xoshiro(seed(1,ri,1,j)))
        end
        elapsed=time()-ts
        push!(rows,(;league,rung,n_paths=16,seconds=elapsed,projected_11_league_128_path_seconds=elapsed*8*11))
        println("BENCHMARK $rung 16 concurrent paths=$elapsed seconds; 11 league projection=$(elapsed*8*11)")
    end
    output("R02_benchmark","runtime_benchmark.csv",rows);flush_manifest!()
end

include(joinpath(@__DIR__, "q07_batch_01_nulls.jl"))
include(joinpath(@__DIR__, "q07_batch_01_levels.jl"))
include(joinpath(@__DIR__, "q07_batch_01_goals.jl"))
include(joinpath(@__DIR__, "q07_batch_01_structure.jl"))
include(joinpath(@__DIR__, "q07_batch_01_report.jl"))
include(joinpath(@__DIR__, "q07_batch_01_book_audit.jl"))
include(joinpath(@__DIR__, "q07_batch_01_checks.jl"))

end # module

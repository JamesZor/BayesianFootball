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
const P = normpath(joinpath(@__DIR__, "../../.."))
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
    CSV.write(path, unique(vcat(old, DF.DataFrame(MANIFEST); cols=:union)))
    empty!(MANIFEST)
end
function output(request, name, rows)
    frame = rows isa DF.AbstractDataFrame ? rows : DF.DataFrame(rows)
    path = joinpath(OUT, name)
    CSV.write(path, frame)
    record!(request, filesize(path)>5_000_000 ? "beast_only_output" : "committed_output",
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
    return (; Vq=vq, Vs=vs, C=cross, r=vq>0 ? sqrt(vs/vq) : missing, c, k,
        phi_deg=mod(45-delta+90,180)-90, delta_deg=delta, l_plus=lp, l_minus=lm,
        f_minor=total>0 ? lm/total : missing, eigengap=total>0 ? gap/total : missing,
        rotation_gain=vs>0 ? 1-lm/vs : missing,
        I=ismissing(c) ? missing : -0.5log1p(-c^2))
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
    println("CHECK $request ", text)
    flush(stdout)
end
function summary!(request, lines)
    path = joinpath(OUT,"SUMMARY.md")
    text = isfile(path) ? read(path,String) : "# Batch 01 results\n\nInput/output hashes, exact roots, row counts and seeds: [manifest.csv](manifest.csv).\nLarge tables remain on the beast, per manager instruction. Units: natural-log rates, calendar weeks, nats.\n"
    @assert !occursin("## $request —",text)
    open(path,"w") do io
        print(io,text,"\n## $request —\n",join(lines,"\n"),"\n")
    end
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
    return (; r_level=geom.r,c_level=geom.c,k_level=geom.k,rho_ab=rho,
        shortcut_r=sqrt((1+rho)/(1-rho)),delta_level_deg=geom.delta_deg,
        r_weekly_cross_section=sqrt(ST.mean(vec(ST.var(s;dims=1)))/ST.mean(vec(ST.var(q;dims=1)))),
        abar,bbar,qb,sb)
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
            @assert all(r.pass for r in checks) "published R02 reproduction failed; see reproduction_checks.csv"
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
                    push!(stats,(;league,rung,protocol="10a",season,method,gauge,n_teams=length(d.active),n_weeks=length(d.weeks),quantity=String(quantity),q05=qs[1],median=qs[2],q95=qs[3]))
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
    errors=abs.(draws.shortcut_r-draws.r_level)
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

end # module

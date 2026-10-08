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
        FITS[key] = Serialization.deserialize(path)
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

end # module

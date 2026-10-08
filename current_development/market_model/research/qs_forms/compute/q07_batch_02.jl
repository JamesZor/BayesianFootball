module QSFormsBatch02
import CSV, DataFrames, Statistics, LinearAlgebra, Random, Serialization, SHA
if !isdefined(parentmodule(@__MODULE__), :QSFormsBatch01)
    Base.include(parentmodule(@__MODULE__),joinpath(@__DIR__,"q07_batch_01.jl"))
end
const A = parentmodule(@__MODULE__).QSFormsBatch01
const DF,ST,LA = DataFrames,Statistics,LinearAlgebra
const P,QS,MID = A.P,A.QS,A.MID
const A1 = A.OUT
const OUT = normpath(joinpath(@__DIR__,"../answers/batch_02"))
const SEED = 610802
const SENTINELS = A.SENTINELS
const RECORDS = NamedTuple[]
const INPUTS = Dict{String,String}()
sha(path)=A.sha(path)
function record!(request,kind,path;detail="",n=0)
    push!(RECORDS,(;request,kind,name=basename(path),path,sha256=isfile(path) ? sha(path) : "",n_rows=n,detail))
end
function input(path;request="shared")
    if !haskey(INPUTS,path)
        @assert isfile(path) "missing required existing artifact: $path"
        h=sha(path)
        manifest=CSV.read(joinpath(A1,"manifest.csv"),DF.DataFrame;stringtype=String)
        known=manifest.sha256[coalesce.(manifest.path.==path,false).&(.!ismissing.(manifest.sha256))]
        if !isempty(known); @assert h in known "source hash mismatch: $path"; end
        INPUTS[path]=h
        record!(request,"input",path;detail="read-only existing artifact; SHA checked against A1 manifest when listed")
    end
    path
end
function loadcsv(name;request="shared")
    CSV.read(input(joinpath(A1,name);request),DF.DataFrame;stringtype=String)
end
function point(league,rung)
    Serialization.deserialize(input(joinpath(A1,"cache","$(league)_$(rung)_point.jls")))
end
function nulls(league)
    s=Serialization.deserialize(input(joinpath(A1,"cache","$(league)_conditional_nulls.jls")))
    @assert length(s.paths)==99
    s
end
function panel(league)
    p,c=A.panel(league)
    input(joinpath(A.Q,"rates_$league.csv"));input(joinpath(A.Q,"configs.csv"))
    p,c
end
function fit(league,rung,protocol)
    input(joinpath(A.FITROOT,league,"$(rung)_$(protocol).jls"))
    A.fit(league,rung,protocol)
end
function output(request,name,rows;bulk=false)
    frame=rows isa DF.AbstractDataFrame ? rows : DF.DataFrame(rows)
    path=joinpath(OUT,name);mkpath(dirname(path));CSV.write(path,frame)
    record!(request,bulk||filesize(path)>500_000 ? "beast_only_output" : "committed_output",path;n=DF.nrow(frame),detail="CSV; blank=NA; schema="*join(names(frame),";"))
    frame
end
function flush!()
    path=joinpath(OUT,"manifest.csv")
    old=isfile(path) ? CSV.read(path,DF.DataFrame;stringtype=String) : DF.DataFrame()
    CSV.write(path,vcat(old,DF.DataFrame(RECORDS);cols=:union));empty!(RECORDS)
end
function summary!(request,lines)
    @assert length(lines)<=10
    path=joinpath(OUT,"SUMMARY.md")
    text=isfile(path) ? read(path,String) : "# Batch 02 results\n\nSources: batch 01 relay a95ca9ca; code commits, cache/input/output SHA256s and exact beast paths in [manifest.csv](manifest.csv). Seed 610802; rates natural log, energies squared log-rates, scores nats/fixture. Large tables stay on beast. Conditional-null ranks=(1+#null≤observed)/100, not p-values.\n"
    section="## $request\n"*join(lines,"\n")*"\n"
    pattern=Regex("(?ms)^## $request\\n.*?(?=^## |\\z)")
    updated=occursin(pattern,text) ? replace(text,pattern=>section*"\n") : text*"\n"*section
    write(path,rstrip(updated)*"\n")
end
function verify(request,text)
    for (path,h) in INPUTS; @assert sha(path)==h "input changed: $path"; end
    open(joinpath(OUT,"VERIFICATION.md"),"a") do io
        println(io,"\n## $request\nCommand: `ENV[\"QSF2_REQUEST\"]=\"$request\"; include(\"current_development/market_model/research/qs_forms/compute/r07_batch_02.jl\")` in owned beast pane %304, threads16/BLAS1.\n",text)
    end
    println("CHECK $request ",text);flush(stdout);flush!()
end
function envelope(vals,obs)
    v=collect(skipmissing(vals));q=A.quant(v)
    (;null_q05=q[1],null_median=q[2],null_q95=q[3],null_rank=length(v)==99&&!ismissing(obs) ? (1+count(x->x<=obs,v))/100 : missing)
end
meanof(x)=isempty(x) ? missing : ST.mean(x)
function rawaxes(p,X,d)
    N=MID.n_teams(p)
    a,b=X[2 .+ d.ix,d.weeks],X[(2+N) .+ d.ix,d.weeks]
    (;q=(a-b)/2,s=(a+b)/2)
end
include("q07_batch_02_projection.jl")
include("q07_batch_02_horizons.jl")
include("q07_batch_02_scores.jl")
include("q07_batch_02_tiers.jl")
include("q07_batch_02_checks.jl")
end

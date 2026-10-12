# Actual frozen W2 features: provenance/persistence and exact AD/FD/zero-byte tapes.
# No sampling or database writes.
using BayesianFootball, CSV, DataFrames, Dates, DynamicPPL, ForwardDiff, LinearAlgebra
using LogDensityProblems, Random, ReverseDiff, Serialization, ThreadPinning, Test
strip(read(`hostname`,String))=="mcmc-beast" && Threads.nthreads()==16 || error("beast16 only")
pinthreads(:cores)
BLAS.set_num_threads(1)
const R8C = Module(:R8Candidates)
Base.include(R8C,joinpath(@__DIR__,"..","candidates.jl"))
const R8PG = BayesianFootball.Models.PreGame
r8raw=Serialization.deserialize("/root/BF_runs/a76a65df/.cache/datastore_ScottishPyramid.jls")
r8ds=Data.apply_scope(r8raw,R8C.SCOPE)
r8split=Data.ScopedWalkForwardCV(R8C.SCOPE)
r8bounds=Data.create_id_boundaries(r8ds,r8split)
r8rows=NamedTuple[]
r8fd=NamedTuple[]
r8provenance=NamedTuple[]
for name in ("qs_entrant","qs_summer")
    candidate=only(filter(c -> c.name==name,R8C.CANDIDATES))
    pairs=Features.create_features(r8bounds[[1,2,60]],r8ds,candidate.model,r8split)
    if name=="qs_entrant"
        m1=first(pairs[1]).data[:qs_entry_manifest]
        m2=first(pairs[2]).data[:qs_entry_manifest]
        @test !isempty(m1)
        @test Set(r.club for r in m1)==Set(r.club for r in m2)
        for a in m1
            b=only(filter(r -> r.club==a.club,m2))
            @test (a.q_mean,a.q_sd,a.s_mean,a.s_sd,a.source_ids)==(b.q_mean,b.q_sd,b.s_mean,b.s_sd,b.source_ids)
        end
    end
    for (fold,pair) in zip((1,2,60),pairs)
        fs=first(pair)
        if name=="qs_entrant"
            manifests=fs.data[:qs_entry_manifest]
            for r in manifests
                @test r.endpoint_overlap==0
                @test isempty(r.max_source_date) || Date(r.max_source_date)<Date(r.cutoff)
                @test Date(r.cutoff)<=Date(r.entry_node_date)
            end
            push!(r8provenance,(;fold,season=fs.data[:target_season],entrants=length(manifests),
                endpoint_overlap=sum(r.endpoint_overlap for r in manifests;init=0),
                hyper_overlap=sum(r.hyper_source_overlap for r in manifests;init=0),persistence_pass=true))
            CSV.write(joinpath(@__DIR__,"provenance_gates.csv"),DataFrame(r8provenance))
        end
        tm=R8PG.build_turing_model(candidate.model,fs)
        Random.seed!(20261046+fold)
        vi=DynamicPPL.link!!(DynamicPPL.VarInfo(tm),tm)
        theta=copy(vi[:])
        density=DynamicPPL.LogDensityFunction(tm,DynamicPPL.getlogjoint_internal,vi)
        f=x -> LogDensityProblems.logdensity(density,x)
        tape=ReverseDiff.compile(ReverseDiff.GradientTape(f,theta))
        gradient=similar(theta)
        worst=0.0
        for delta in (0.0,0.001,-0.002,0.15,-0.15)
            x=theta.+delta.*sin.(eachindex(theta))
            ReverseDiff.gradient!(gradient,tape,x)
            gf=ForwardDiff.gradient(f,x)
            gr=ReverseDiff.gradient(f,x)
            err=max(norm(gradient-gf)/max(norm(gradient),norm(gf),1.0),
                norm(gradient-gr)/max(norm(gradient),norm(gr),1.0))
            worst=max(worst,err)
            @test err<=1e-10
        end
        bytes=Harness.compiled_gradient_bytes(tape,gradient,theta;warmup=20,reps=20)
        @test bytes==0
        ReverseDiff.gradient!(gradient,tape,theta)
        direction=sin.(Float64.(eachindex(theta)))
        direction ./= norm(direction)
        exact=dot(gradient,direction)
        errors=Float64[]
        for h in (0.01,0.003,0.001,0.0003,0.0001,0.00003,0.00001)
            finite=(f(theta.+h.*direction)-f(theta.-h.*direction))/(2h)
            err=abs(finite-exact)/max(abs(exact),1.0)
            push!(errors,err)
            push!(r8fd,(;arm=name,fold,h,exact,finite,relative_error=err))
        end
        @test minimum(errors)<1e-6
        @test minimum(errors[2:end])<errors[1]
        push!(r8rows,(;arm=name,fold,parameters=length(theta),exact_ad_error=worst,
            linked_allocated_bytes=bytes,fd_minimum_error=minimum(errors),passed=true))
        CSV.write(joinpath(@__DIR__,"real_fold_gates.csv"),DataFrame(r8rows))
        CSV.write(joinpath(@__DIR__,"real_fold_fd.csv"),DataFrame(r8fd))
        println("S2_REAL_GATE arm=$name fold=$fold AD=$worst bytes=$bytes FD=$(minimum(errors))")
    end
end
println("S2_REAL_FEATURE_PROVENANCE_AD_FD_ZERO_ALLOC_PASS")

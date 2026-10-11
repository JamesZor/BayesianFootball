# Source gates: units, exact AD, FD convergence and exactly zero linked-tape allocation.
# %% 1. Runtime / packages
using BayesianFootball, CSV, DataFrames, DynamicPPL, ForwardDiff, LinearAlgebra
using LogDensityProblems, Random, ReverseDiff, ThreadPinning, Test
strip(read(`hostname`,String))=="mcmc-beast" && Threads.nthreads()==16 || error("beast16 only")
pinthreads(:cores)
BLAS.set_num_threads(1)
include(joinpath(@__DIR__,"..","..","..","..","test","test_quality_style_transition.jl"))
include(joinpath(@__DIR__,"..","..","..","..","test","test_quality_style_grw.jl"))

# %% 2. Opt-in models / synthetic opening and observed-entry geometries
s2gate_rows = NamedTuple[]
s2fd_rows = NamedTuple[]
for (arm,config) in (("summer",QualityStyleGRW(target_season_step=true)),
                      ("entrant",QualityStyleGRW(entrant_feature=QTManualEntryFeature())))
    for nt in (0,2)
        fs=qt_features(nt)
        model=qt_model(config)
        tm=QTPG.build_turing_model(model,fs)
        Random.seed!(20261046+nt)
        vi=DynamicPPL.link!!(DynamicPPL.VarInfo(tm),tm)
        θ=copy(vi[:])
        density=DynamicPPL.LogDensityFunction(tm,DynamicPPL.getlogjoint_internal,vi)
        f=x -> LogDensityProblems.logdensity(density,x)
        raw=ReverseDiff.GradientTape(f,θ)
        tape=ReverseDiff.compile(raw)
        gradient=similar(θ)
        worst=0.0
        for δ in (0.0,0.001,-0.002,0.15,-0.15)
            x=θ .+ δ.*sin.(eachindex(θ))
            ReverseDiff.gradient!(gradient,tape,x)
            gf=ForwardDiff.gradient(f,x)
            gr=ReverseDiff.gradient(f,x)
            err=max(norm(gradient-gf)/max(norm(gradient),norm(gf),1.0),
                    norm(gradient-gr)/max(norm(gradient),norm(gr),1.0))
            worst=max(worst,err)
            @test err<=1e-10
        end
        bytes=Harness.compiled_gradient_bytes(tape,gradient,θ;warmup=20,reps=20)
        @test bytes==0
        ReverseDiff.gradient!(gradient,tape,θ)
        direction=sin.(Float64.(eachindex(θ)))
        direction ./= norm(direction)
        exact=dot(gradient,direction)
        errors=Float64[]
        for h in (0.01,0.003,0.001,0.0003,0.0001,0.00003,0.00001)
            finite=(f(θ.+h.*direction)-f(θ.-h.*direction))/(2h)
            err=abs(finite-exact)/max(abs(exact),1.0)
            push!(errors,err)
            push!(s2fd_rows,(;arm,target_steps=nt,h,exact,finite,relative_error=err))
        end
        @test minimum(errors)<1e-6
        @test minimum(errors[2:end])<errors[1]
        push!(s2gate_rows,(;arm,target_steps=nt,parameters=length(θ),exact_ad_error=worst,
            linked_allocated_bytes=bytes,fd_minimum_error=minimum(errors),passed=true))
        println("S2_COMPONENT_GATE arm=$arm target_steps=$nt AD=$worst bytes=$bytes FD=$(minimum(errors))")
    end
end
CSV.write(joinpath(@__DIR__,"component_gates.csv"),DataFrame(s2gate_rows))
CSV.write(joinpath(@__DIR__,"fd_convergence.csv"),DataFrame(s2fd_rows))
println("S2_COMPONENT_UNITS_AD_FD_ZERO_ALLOC_PASS")

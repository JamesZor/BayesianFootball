# Three frozen seeds per opt-in component; independent data generator, no DB writes.
# 90% marginal intervals, >=80% pooled coverage, no parameter misses all three seeds.
using BayesianFootball, CSV, DataFrames, Distributions, LinearAlgebra, MCMCChains
using Random, Statistics, ThreadPinning, Test
strip(read(`hostname`,String))=="mcmc-beast" && Threads.nthreads()==16 || error("beast16 only")
pinthreads(:cores)
BLAS.set_num_threads(1)
const R9C=Module(:R9Candidates)
Base.include(R9C,joinpath(@__DIR__,"..","candidates.jl"))
const R9PG=BayesianFootball.Models.PreGame
const R9API=R9PG.Builder

# Written directly from the q/s walk equations, not model/design/reconstruction calls.
function r9_synthetic(arm,rng)
    n,nh,nt=16,5,4
    inc=arm=="qs_entrant" ? collect(1:14) : collect(1:n)
    entrants=arm=="qs_entrant" ? [15,16] : Int[]
    tau,ratio=[0.16,0.085,0.027],[0.4,0.4,0.3]
    qscale=tau./sqrt.(1 .+ ratio.^2)
    sscale=ratio.*qscale
    moments=(;indices=entrants,q_mean=[0.35,-0.20],q_sd=[0.20,0.20],s_mean=[0.05,-0.02],s_sd=[0.10,0.10])
    truth=Dict("obs.σ_obs"=>0.07,"inter.μ"=>log(1.2),"ha.γ_global"=>0.15)
    for (j,suffix) in enumerate(("₀","ₛ","ₖ"))
        truth["dyn.τ"*suffix]=tau[j]
        truth["dyn.r"*suffix]=ratio[j]
    end
    states=Dict{Symbol,Matrix{Float64}}()
    for (side,scale) in ((:q,qscale),(:s,sscale))
        x=zeros(n,nh+nt)
        initial=scale[1].*randn(rng,length(inc))
        initial .-= mean(initial)
        x[inc,1]=initial
        for t in 2:nh
            delta=scale[2].*randn(rng,length(inc))
            delta .-= mean(delta)
            x[inc,t]=x[inc,t-1]+delta
        end
        for t in (nh+1):(nh+nt)
            delta=scale[3].*randn(rng,length(inc))
            delta .-= mean(delta)
            if arm=="qs_summer" && t==nh+1
                summer=scale[2].*randn(rng,length(inc))
                summer .-= mean(summer)
                delta .+= summer
            end
            x[inc,t]=x[inc,t-1]+delta
        end
        if !isempty(entrants)
            means=side==:q ? moments.q_mean : moments.s_mean
            sds=side==:q ? moments.q_sd : moments.s_sd
            z=randn(rng,length(entrants))
            x[entrants,nh+1]=means+sds.*z
            for t in (nh+2):(nh+nt)
                x[entrants,t]=x[entrants,t-1]+scale[3].*randn(rng,length(entrants))
            end
            for (j,zj) in enumerate(z)
                truth["dyn.$side.z_entry[$j]"]=zj
            end
        end
        truth["state.$side.first_target_incumbent"]=x[1,nh+1]
        truth["state.$side.boundary_change"]=x[1,nh+1]-x[1,nh]
        states[side]=x
    end
    h,a,times=Int[],Int[],Int[]
    for t in 1:(nh+nt)
        clubs=t<=nh ? inc : collect(1:n)
        for i in clubs,j in clubs
            i<j || continue
            # Alternate home orientation; observation HA and q are jointly identifiable.
            home,away=isodd(t+i+j) ? (i,j) : (j,i)
            push!(h,home); push!(a,away); push!(times,t)
        end
    end
    q,s=states[:q],states[:s]
    ih,ia=CartesianIndex.(h,times),CartesianIndex.(a,times)
    eta_h=log(1.2) .+ 0.15 .+ q[ih] .- q[ia] .+ s[ih] .+ s[ia]
    eta_a=log(1.2) .+ q[ia] .- q[ih] .+ s[ih] .+ s[ia]
    m=length(h)
    data=Dict{Symbol,Any}(:flat_home_ids=>h,:flat_away_ids=>a,:time_indices=>times,
        :season_indices=>ones(Int,m),:flat_months=>ones(Int,m),
        :flat_home_goals=>[rand(rng,Poisson(exp(x))) for x in eta_h],
        :flat_away_goals=>[rand(rng,Poisson(exp(x))) for x in eta_a],
        :flat_market_log_h=>eta_h+0.07.*randn(rng,m),
        :flat_market_log_a=>eta_a+0.07.*randn(rng,m),:flat_market_available=>ones(m),
        :dates=>Float64.(1:m),:n_teams=>n,:n_seasons=>1,:n_months=>1,:n_rounds=>nh+nt,
        :n_history_steps=>nh,:n_target_steps=>nt,:grw_oos_seed=>46,
        :team_map=>Dict("t$i"=>i for i in 1:n),:ordered_match_ids=>collect(1:m))
    !isempty(entrants) && (data[:qs_entry_prior]=moments)
    return FeatureSet(data),truth
end

r9intervals=NamedTuple[]
r9diagnostics=NamedTuple[]
for arm in ("qs_entrant","qs_summer")
    candidate=only(filter(c -> c.name==arm,R9C.CANDIDATES))
    for seed in 1:3
        data_seed=202610120+seed
        sampler_seed=202610220+seed
        fs,truth=r9_synthetic(arm,MersenneTwister(data_seed))
        Random.seed!(sampler_seed)
        cfg=Samplers.NUTSConfig(n_samples=400,n_warmup=400,n_chains=2,accept_rate=0.8,
            max_depth=10,show_progress=false,silence_initial_stepsize=true)
        started=time()
        chain=Samplers.run_sampler(R9PG.build_turing_model(candidate.model,fs),cfg)
        seconds=time()-started
        dyn=R9API._cb_extract_dynamics(chain,candidate.model.dynamics,"dyn",fs.data[:n_teams],fs)
        for name in sort!(collect(keys(truth)))
            draws=if startswith(name,"state.")
                side=split(name,'.')[2]=="q" ? dyn.q : dyn.s
                occursin("boundary_change",name) ? vec(side[1,6,:]-side[1,5,:]) : vec(side[1,6,:])
            else
                vec(Array(chain[Symbol(name)]))
            end
            lo,median,hi=quantile(draws,[0.05,0.5,0.95])
            push!(r9intervals,(;arm,seed,data_seed,sampler_seed,parameter=name,truth=truth[name],
                lo,median,hi,covered=lo<=truth[name]<=hi,wall_seconds=seconds))
        end
        CSV.write(joinpath(@__DIR__,"recovery_three_seeds.csv"),DataFrame(r9intervals))
        # Raw diagnostic table retained; recovery is not a production convergence certificate.
        summary=MCMCChains.summarystats(chain)
        CSV.write(joinpath(@__DIR__,"recovery_diagnostics_$(arm)_seed$(seed).csv"),DataFrame(summary))
        println("S2_RECOVERY arm=$arm seed=$seed seconds=$seconds covered=$(count(r -> r.arm==arm && r.seed==seed && r.covered,r9intervals))")
    end
end
r9summary=NamedTuple[]
for arm in ("qs_entrant","qs_summer")
    rows=filter(r -> r.arm==arm,r9intervals)
    misses=String[]
    for parameter in unique(r.parameter for r in rows)
        group=filter(r -> r.parameter==parameter,rows)
        @test length(group)==3
        any(r.covered for r in group) || push!(misses,parameter)
    end
    coverage=count(r.covered for r in rows)/length(rows)
    push!(r9summary,(;arm,intervals=length(rows),covered=count(r.covered for r in rows),
        pooled_coverage=coverage,all_three_misses=join(misses,";"),passed=coverage>=0.80 && isempty(misses)))
end
CSV.write(joinpath(@__DIR__,"recovery_gate.csv"),DataFrame(r9summary))
@test all(r -> r.passed,r9summary)
println("S2_RECOVERY_THREE_SEEDS_PASS ",repr(r9summary))

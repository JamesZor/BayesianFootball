module PooledRecovery
using BayesianFootball, CSV, DataFrames, Distributions, LinearAlgebra, MCMCChains, Random, Statistics, Test
const PG=BayesianFootball.Models.PreGame
const API=PG.Builder

"Independent QS equations, same-tier schedules, Championship goals but no Championship market."
function synthetic(ntiers,rng)
    per_tier=8
    n=per_tier*ntiers; nh=5; nt=4
    # Three-tier panel is Championship/L1/L2; two-tier panel is L1/L2.
    tier=repeat(collect((5-ntiers):4),inner=per_tier)
    anchor=1.0 .- tier
    tau=[0.10,0.055,0.025]; ratio=[0.4,0.4,0.3]
    qscale=tau./sqrt.(1 .+ ratio.^2); sscale=ratio.*qscale
    step=0.55; carry=0.30; jump_alpha=0.06; jump_beta=-0.02
    # Club1 moves down one tier at the target boundary; nonempty event designs
    # exercise EstimatedJump (and avoid Turing's empty filldist Vector{Real}).
    current=repeat(tier,1,nh+nt)
    current[1,(nh+1):end].+=1
    tier_delta=1.0 .- current .- anchor
    truth=Dict("pyramid_tiers.step"=>step,"pyramid_tiers.κ"=>carry,"obs.σ_obs"=>0.07,
        "inter.μ"=>log(1.2),"ha.γ_global"=>0.15)
    for (j,suffix) in enumerate(("₀","ₛ","ₖ"))
        truth["dyn.τ"*suffix]=tau[j]; truth["dyn.r"*suffix]=ratio[j]
    end
    states=Dict{Symbol,Matrix{Float64}}()
    for (side,scale) in ((:q,qscale),(:s,sscale))
        x=zeros(n,nh+nt)
        for t in 1:(nh+nt)
            delta=scale[t==1 ? 1 : t<=nh ? 2 : 3].*randn(rng,n)
            delta .-= mean(delta)
            x[:,t]=t==1 ? delta : x[:,t-1]+delta
        end
        states[side]=x
        truth["state.$side.first_target"]=x[1,nh+1]
        truth["state.$side.boundary_change"]=x[1,nh+1]-x[1,nh]
    end
    h=Int[]; a=Int[]; times=Int[]
    for t in 1:(nh+nt),i in 1:n,j in (i+1):n
        current[i,t]==current[j,t] || continue
        home,away=isodd(t+i+j) ? (i,j) : (j,i)
        push!(h,home); push!(a,away); push!(times,t)
    end
    ih=CartesianIndex.(h,times); ia=CartesianIndex.(a,times)
    q=states[:q]; s=states[:s]
    # Attack = q+s + .48*step*anchor, concession = -q+s - .52*step*anchor.
    eta_h=log(1.2) .+ 0.15 .+ q[ih].-q[ia].+s[ih].+s[ia].+
        step.*(0.48.*anchor[h].-0.52.*anchor[a])
    eta_a=log(1.2) .+ q[ia].-q[ih].+s[ih].+s[ia].+
        step.*(0.48.*anchor[a].-0.52.*anchor[h])
    event_h=Float64.((h.==1).&(times.>nh)); event_a=Float64.((a.==1).&(times.>nh))
    eta_h .+= step.*carry.*(0.48.*tier_delta[ih].-0.52.*tier_delta[ia]).+jump_alpha.*event_h.+jump_beta.*event_a
    eta_a .+= step.*carry.*(0.48.*tier_delta[ia].-0.52.*tier_delta[ih]).+jump_alpha.*event_a.+jump_beta.*event_h
    m=length(h); mask=Float64.(current[ih].>=3)
    d=Dict{Symbol,Any}(:flat_home_ids=>h,:flat_away_ids=>a,:time_indices=>times,
        :season_indices=>ones(Int,m),:flat_months=>ones(Int,m),
        :flat_home_goals=>[rand(rng,Poisson(exp(x))) for x in eta_h],
        :flat_away_goals=>[rand(rng,Poisson(exp(x))) for x in eta_a],
        :flat_market_log_h=>mask.*(eta_h+0.07.*randn(rng,m)),
        :flat_market_log_a=>mask.*(eta_a+0.07.*randn(rng,m)),:flat_market_available=>mask,
        :dates=>Float64.(1:m),:n_teams=>n,:n_seasons=>1,:n_months=>1,:n_rounds=>nh+nt,
        :n_history_steps=>nh,:n_target_steps=>nt,:grw_oos_seed=>46,
        :team_map=>Dict("t$i"=>i for i in 1:n),:ordered_match_ids=>collect(1:m),
        :pyramid_anchor_home=>anchor[h],:pyramid_anchor_away=>anchor[a],
        :pyramid_delta_home=>tier_delta[ih],:pyramid_delta_away=>tier_delta[ia],
        :pyramid_old_firm_home=>zeros(m),:pyramid_old_firm_away=>zeros(m),
        :pyramid_events=>[(;club="t1",season="target",α_mean=0.0,α_sd=0.0,β_mean=0.0,β_sd=0.0)],
        :pyramid_events_home=>reshape(event_h,m,1),:pyramid_events_away=>reshape(event_a,m,1))
    return FeatureSet(d),truth
end

function run(candidate,output)
    rows=NamedTuple[]; gates=NamedTuple[]; centers=NamedTuple[]
    for ntiers in (2,3), seed in 1:3
        data_seed=202610300+10*ntiers+seed; sampler_seed=202610400+10*ntiers+seed
        fs,truth=synthetic(ntiers,MersenneTwister(data_seed))
        Random.seed!(sampler_seed)
        cfg=Samplers.NUTSConfig(n_samples=400,n_warmup=400,n_chains=2,accept_rate=0.8,
            max_depth=10,show_progress=false,silence_initial_stepsize=true)
        started=time()
        chain=Samplers.run_sampler(PG.build_turing_model(candidate.model,fs),cfg)
        seconds=time()-started
        dyn=API._cb_extract_dynamics(chain,candidate.model.dynamics,"dyn",fs.data[:n_teams],fs)
        q_error=maximum(abs,sum(dyn.q;dims=1)); s_error=maximum(abs,sum(dyn.s;dims=1))
        @test q_error<1e-10 && s_error<1e-10
        push!(centers,(;ntiers,seed,q_center_error=q_error,s_center_error=s_error))
        for name in sort!(collect(keys(truth)))
            draws=if startswith(name,"state.")
                side=split(name,'.')[2]=="q" ? dyn.q : dyn.s
                occursin("boundary_change",name) ? vec(side[1,6,:]-side[1,5,:]) : vec(side[1,6,:])
            else
                vec(Array(chain[Symbol(name)]))
            end
            lo,med,hi=quantile(draws,[0.05,0.5,0.95])
            push!(rows,(;ntiers,seed,data_seed,sampler_seed,parameter=name,truth=truth[name],
                lo,median=med,hi,covered=lo<=truth[name]<=hi,wall_seconds=seconds))
        end
        CSV.write(joinpath(output,"recovery_three_seeds.csv"),DataFrame(rows))
        CSV.write(joinpath(output,"recovery_diagnostics_t$(ntiers)_seed$(seed).csv"),DataFrame(MCMCChains.summarystats(chain)))
        println("POOL_RECOVERY tiers=$ntiers seed=$seed seconds=$seconds")
    end
    for ntiers in (2,3)
        group=filter(r->r.ntiers==ntiers,rows)
        misses=[p for p in unique(r.parameter for r in group) if !any(r.covered for r in group if r.parameter==p)]
        step_rows=filter(r->r.parameter=="pyramid_tiers.step",group)
        coverage=count(r.covered for r in group)/length(group)
        passed=coverage>=0.8 && isempty(misses) && count(r.covered for r in step_rows)>=2
        push!(gates,(;ntiers,intervals=length(group),covered=count(r.covered for r in group),coverage,
            tier_step_covered=count(r.covered for r in step_rows),all_three_misses=join(misses,";"),passed))
    end
    CSV.write(joinpath(output,"recovery_centering.csv"),DataFrame(centers))
    CSV.write(joinpath(output,"recovery_gate.csv"),DataFrame(gates))
    @test all(r.passed for r in gates)
    println("POOL_RECOVERY_PASS ",repr(gates))
end
end

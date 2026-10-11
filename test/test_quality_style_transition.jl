using Test, BayesianFootball, Distributions, DynamicPPL, ForwardDiff
using LinearAlgebra, LogDensityProblems, MCMCChains, Random, ReverseDiff, Statistics
const QTPG = BayesianFootball.Models.PreGame
const QTAPI = QTPG.Builder
struct QTManualEntryFeature <: Features.AbstractFeatureConfig end

function qt_features(nt=2)
    nh=2
    n = nt==0 ? 8 : 16
    h = vcat([1,2,3,1,2,3,1,2], [4,1,2,3,4,1,2,3])[1:n]
    a = vcat([2,3,1,2,3,1,2,3], [1,4,3,4,2,3,4,1])[1:n]
    t = nt==0 ? repeat([1,2],4) : vcat(repeat([1,2],4),repeat([3,4],4))
    return BayesianFootball.FeatureSet(Dict{Symbol,Any}(
        :flat_home_ids=>h,:flat_away_ids=>a,:season_indices=>ones(Int,n),
        :time_indices=>t,:flat_months=>ones(Int,n),:flat_home_goals=>mod.(1:n,3),
        :flat_away_goals=>mod.(2:n+1,3),:dates=>Float64.(1:n),
        :n_teams=>4,:n_seasons=>1,:n_months=>1,:n_rounds=>nh+nt,
        :n_history_steps=>nh,:n_target_steps=>nt,:grw_oos_seed=>46,
        :team_map=>Dict("t$i"=>i for i in 1:4),:ordered_match_ids=>collect(1:n),
        :qs_entry_prior=>(;indices=[4],q_mean=[0.35],q_sd=[0.20],s_mean=[0.05],s_sd=[0.10])))
end
qt_model(c) = CountModelBuilder(:qt_test) |> add(GlobalInterception()) |> add(c) |>
    add(GlobalHomeAdvantage()) |> add(PoissonObservation()) |> build

function qt_chain(nh,nt;entry=false,summer=false,draws=5)
    columns = ["dyn.τ₀","dyn.τₛ","dyn.r₀","dyn.rₛ"]
    (nt>0 || entry) && append!(columns,["dyn.τₖ","dyn.rₖ"])
    for side in ("q","s")
        append!(columns,["dyn.$side.z_init[$i]" for i in 1:4])
        append!(columns,["dyn.$side.z_season[$i,$j]" for j in 1:(nh-1+Int(summer && nt>0)) for i in 1:4])
        append!(columns,["dyn.$side.z_target[$i,$j]" for j in 1:nt for i in 1:4])
        entry && push!(columns,"dyn.$side.z_entry[1]")
    end
    values = zeros(draws,length(columns),1)
    for (j,name) in enumerate(columns),k in 1:draws
        values[k,j,1] = occursin("τ",name) ? 0.08+0.01k : occursin(".r",name) ? 0.3+0.03k : 0.01j+0.02k
    end
    return Chains(values,Symbol.(columns))
end

@testset "Opt-in QS transitions" begin
    base = QualityStyleGRW()
    @test QualityStyleGRW(target_season_step=false) === base
    @test fieldnames(typeof(base)) == (:z₀,:zₛ,:zₖ,:τ₀,:τₛ,:τₖ,:r₀,:rₛ,:rₖ)
    summer = QualityStyleGRW(target_season_step=true)
    entrant = QualityStyleGRW(entrant_feature=QTManualEntryFeature())
    @test summer isa QTPG.TargetSeasonQualityStyleGRW
    @test entrant isa QTPG.EntrantQualityStyleGRW
    @test_throws ArgumentError QualityStyleGRW(target_season_step=true,entrant_feature=QTManualEntryFeature())
    @testset "summer map, centring and deterministic opening integration" begin
        for nt in (0,2)
            fs=qt_features(nt)
            d=QTAPI.dynamics_design(summer,fs,length(fs.data[:flat_home_ids]))
            @test size(d.season_accumulator,1)==1+Int(nt>0)
            @test d.n_rounds==2+nt
            ch=qt_chain(2,nt;summer=true)
            draw=QTAPI._cb_extract_dynamics(ch,summer,"dyn",4,fs)
            @test size(draw.q)==(4,2+nt,5)
            @test maximum(abs.(sum(draw.q,dims=1)))<1e-14
            @test maximum(abs.(sum(draw.s,dims=1)))<1e-14
            @test maximum(abs.(sum(draw.oos_q,dims=2)))<1e-14
            @test maximum(abs.(sum(draw.oos_s,dims=2)))<1e-14
            @test draw==QTAPI._cb_extract_dynamics(ch,summer,"dyn",4,fs)
            nt==0 && @test draw.oos_q != QTAPI._cb_extract_dynamics(ch,base,"dyn",4,fs).oos_q
        end
    end
    @testset "entry birth, persistent site and incumbent invariance" begin
        for nt in (0,2)
            fs=qt_features(nt)
            d=QTAPI.dynamics_design(entrant,fs,length(fs.data[:flat_home_ids]))
            @test d.entry.entry_accumulator[1,1:2]==zeros(2)
            nt>0 && @test d.entry.entry_target[1,:]==zeros(2+nt)
            ch=qt_chain(2,nt;entry=true)
            before=QTAPI._cb_extract_dynamics(ch,entrant,"dyn",4,fs)
            changed=deepcopy(fs)
            p=changed.data[:qs_entry_prior]
            changed.data[:qs_entry_prior]=merge(p,(;q_mean=p.q_mean.+1.0,s_mean=p.s_mean.-0.4))
            after=QTAPI._cb_extract_dynamics(ch,entrant,"dyn",4,changed)
            @test before.q[1:3,:,:]==after.q[1:3,:,:]
            @test before.s[1:3,:,:]==after.s[1:3,:,:]
            @test before.oos_q[:,1:3]==after.oos_q[:,1:3]
            @test before.oos_s[:,1:3]==after.oos_s[:,1:3]
            @test before.q[4,1:2,:]==zeros(2,5)
            @test before.s[4,1:2,:]==zeros(2,5)
            @test maximum(abs.(sum(before.q[1:3,:,:],dims=1)))<1e-14
            if nt==0
                @test after.oos_q[:,4]-before.oos_q[:,4] ≈ ones(5)
            else
                @test after.q[4,3:end,:]-before.q[4,3:end,:] ≈ ones(nt,5)
            end
        end
        fs=qt_features()
        p=fs.data[:qs_entry_prior]
        fs.data[:qs_entry_prior]=merge(p,(;indices=[1]))
        @test_throws ErrorException QTAPI.dynamics_design(entrant,fs,16)
        fs=qt_features()
        fs.data[:qs_entry_prior]=merge(fs.data[:qs_entry_prior],(;indices=Int[],q_mean=Float64[],q_sd=Float64[],s_mean=Float64[],s_sd=Float64[]))
        @test QTAPI.dynamics_design(entrant,fs,16) isa QTAPI.GRWDynamicsDesign
    end
    @testset "A2 enters once; q is half the total step" begin
        c=PyramidTiers(anchor=:full_pyramid,carry=EstimatedCarry(),jump=EstimatedJump())
        model=CountModelBuilder(:qt_a2) |> add(GlobalInterception()) |> add(base) |>
            add(GlobalHomeAdvantage()) |> add(c) |> add(PoissonObservation()) |> build
        @test count(x -> x isa PyramidTiers,model.covariates)==1
        b=0.47
        ηh=QTAPI._tier_side(b,0.0,0.0,[c.attack_share],[0.0],[0.0])[1]
        ηa=QTAPI._tier_side(b,0.0,0.0,[-(1-c.attack_share)],[0.0],[0.0])[1]
        @test (ηh-ηa)/2 ≈ 0.235 atol=1e-15
        @test (ηh+ηa)/2 ≈ -0.0094
    end
end

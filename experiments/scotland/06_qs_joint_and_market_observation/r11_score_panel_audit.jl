# Read-only score-population audit. Saved fits/latents and CSVs only; no refits or writes to DB.
using BayesianFootball, CSV, DataFrames, Dates, LinearAlgebra, Statistics, ThreadPinning
pinthreads(:cores); BLAS.set_num_threads(1)
const SP_C = Module(:ScorePanelCandidates)
Base.include(SP_C, joinpath(@__DIR__, "candidates.jl"))
include(joinpath(@__DIR__, "..", "05_quality_style_goal_vs_market", "scoring_adapter.jl"))
import Serialization
const SP_H = BayesianFootball.Harness
const SP_E = BayesianFootball.Evaluation
const SP_OUT = "/root/BF_runs/qs_experiment_w2_out/score_panel_audit"
mkpath(SP_OUT)

# Mirror Evaluation.ranked_probability_score's exact odds traversal and eligibility,
# but retain fixture IDs so its actual denominator is auditable.
function sp_rps_rows(ctx; source)
    slots = Dict{Int,Vector{Float64}}()
    for i in 1:ctx.odds.n
        sel = ctx.odds.selection[i]
        k = sel === :home ? 1 : sel === :draw ? 2 : sel === :away ? 3 : 0
        k == 0 && continue
        p = source === :market ? (ctx.odds.has_fair[i] ? ctx.odds.prob_fair_close[i] : NaN) :
            something(SP_E.prob_mean(ctx.probs, ctx.odds.match_id[i], sel), NaN)
        isfinite(p) || continue
        get!(slots, ctx.odds.match_id[i], fill(NaN,3))[k] = p
    end
    rows = NamedTuple[]
    for (id, p) in slots
        all(isfinite,p) || continue
        sc = SP_E.outcome_of(ctx.outcomes,id)
        sc === nothing && continue
        gh,ga = sc
        yh,yd = Float64(gh>ga),Float64(gh==ga)
        push!(rows,(;match_id=id,rps=((p[1]-yh)^2+(p[1]+p[2]-yh-yd)^2)/2))
    end
    actual,n = SP_E.ranked_probability_score(ctx;source)
    length(rows)==n && abs(mean(r.rps for r in rows)-actual)<1e-14 || error("RPS traversal mismatch")
    return rows,actual
end

let
    ds = Main.QSMarketArm.checked_datastore()
    db = Training.PostgresStorage(SP_C.EXPERIMENT)
    runids = CSV.read(joinpath(@__DIR__,"results","phase3","RUNS.csv"),DataFrame)
    fits = Dict{String,Any}(string(r.label)=>Training.load_fit(db,string(r.run_id)) for r in eachrow(runids))
    w1 = Training.PostgresStorage("scottish_lower_quality_style_2426")
    fits["control_grw"] = Training.load_fit(w1,"a036d22a-ff32-404c-b801-5f928d8a89f4")
    fits["qs_weak_r"] = Training.load_fit(w1,"21f2a9f9-b96f-4034-97de-767704a9d54a")
    market = Serialization.deserialize("/root/BF_runs/qs_experiment_out/market_grid/market_latents.jls")
    fits["market_c0"] = QSScoringAdapter.wrap(fits["control_grw"],market)
    odds = SP_H._betfair_closing_odds(ds)
    lower = Set(Int(r.match_id) for r in eachrow(ds.matches) if Int(r.tournament_id) in (56,57))
    odds = odds[in.(odds.match_id,Ref(lower)),:]
    saved = CSV.read(joinpath(@__DIR__,"results","phase4","observation_frames.csv"),DataFrame)
    headline = CSV.read(joinpath(@__DIR__,"results","phase4","headline.csv"),DataFrame)
    common = filter(r->r.model=="market_close" && r.family=="1X2",saved)
    commonkeys = sort([(Int(r.match_id),String(r.selection)) for r in eachrow(common)])
    commonids = Set(Int.(common.match_id))
    length(commonids)==595 && length(commonkeys)==1785 || error("common panel mismatch")
    summary = NamedTuple[]
    details = NamedTuple[]
    for arm in sort!(vcat(collect(keys(fits)),["market_close"]))
        fit = fits[arm=="market_close" ? "grw_joint" : arm]
        panel = SP_H._season_panel(ds,fit,["24/25","25/26"])
        ctx = SP_H._context(SP_H._restrict(fit,panel),odds,ds)
        original,original_rps = sp_rps_rows(ctx;source=arm=="market_close" ? :market : :model)
        reported = only(filter(r->r.model==arm,headline)).rps
        abs(original_rps-reported)<1e-14 || error("headline RPS mismatch $arm")
        frame = filter(r->r.model==arm && r.family=="1X2",saved)
        sort([(Int(r.match_id),String(r.selection)) for r in eachrow(frame)])==commonkeys || error("selection set differs $arm")
        rp = Float64[]; cats = Float64[]; sums = Float64[]
        for g in groupby(frame,:match_id)
            nrow(g)==3 && Set(String.(g.selection))==Set(["home","draw","away"]) || error("incomplete trio")
            ix = [only(findall(==(s),String.(g.selection))) for s in ("home","draw","away")]
            p,y = g.p_model[ix],g.y[ix]
            sum(y)==1 || error("noncategorical outcome")
            push!(sums,sum(p))
            push!(rp,((p[1]-y[1])^2+(p[1]+p[2]-y[1]-y[2])^2)/2)
            # Standard categorical score uses a unit-sum categorical distribution.
            push!(cats,-log(p[only(findall(==(1.0),y))]/sum(p)))
        end
        ids = Set(r.match_id for r in original)
        common_rps = mean(rp)
        restricted_rps = mean(r.rps for r in original if r.match_id in commonids)
        abs(common_rps-restricted_rps)<1e-14 || error("common RPS mismatch $arm")
        for r in original
            push!(details,(;model=arm,r.match_id,r.rps,in_common=r.match_id in commonids,in_target=r.match_id in Set(panel)))
        end
        push!(summary,(;model=arm,original_rps_fixtures=length(ids),original_rps_selections=3length(ids),
            original_rps,reported_rps=reported,common_fixtures=length(commonids),common_selections=nrow(frame),
            same_common_keys=true,original_extra_fixtures=length(setdiff(ids,commonids)),
            original_missing_common=length(setdiff(commonids,ids)),common_rps,
            binary_logloss=mean(frame.ll_model),categorical_logloss=mean(cats),
            max_probability_sum_error=maximum(abs.(sums.-1))))
    end
    CSV.write(joinpath(SP_OUT,"score_panel_audit.csv"),DataFrame(summary))
    CSV.write(joinpath(SP_OUT,"rps_original_fixture_rows.csv"),DataFrame(details))
    println("SCORE_PANEL_AUDIT_PASS rows=8 common_fixtures=595 common_selections=1785; original RPS populations explicitly recorded")
end

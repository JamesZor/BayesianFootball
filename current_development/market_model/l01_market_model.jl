module MarketModel

import BayesianFootball
import DataFrames
import Optim
import Statistics
import Random
import CSV
import Dates
import LinearAlgebra

if !isdefined(parentmodule(@__MODULE__), :MarketInverseDynamics)
    Base.include(parentmodule(@__MODULE__), joinpath(@__DIR__, "..", "market_inverse_dynamics", "l01_market_inverse_loader.jl"))
end
const MID = parentmodule(@__MODULE__).MarketInverseDynamics
const CAL = BayesianFootball.Calibration
const FEAT = BayesianFootball.Features
const DF = DataFrames
const LINES = CAL.L2_INVERSION_LINES
const STARTS = ([log(1.5), log(1.0)], [log(0.4), log(3.2)], [log(3.2), log(0.4)])

"Return a complete, overround-gated TWA(−20,0] close and one refusal per market group."
function gated_close(ds)
    close = MID.closing_book(ds)
    return gate_book(close)
end

"Gate raw (or already de-vigged) TWA prices BEFORE re-normalising; never trust their fair probabilities."
function gate_book(close::DF.AbstractDataFrame)
    keep = DF.DataFrame[]
    refused = NamedTuple[]
    for g in DF.groupby(close, [:match_id, :market_name, :market_line])
        mid = Int(first(g.match_id))
        name = String(first(g.market_name))
        line = Float64(first(g.market_line))
        name in ("1X2", "BTTS", "OverUnder") || continue
        name == "OverUnder" && !(line in (0.5, 1.5, 2.5, 3.5)) && continue
        want = CAL.expected_selection_count(name, line)
        selections = Set(Symbol.(g.selection))
        expected = name == "1X2" ? Set((:home, :draw, :away)) :
                   name == "BTTS" ? Set((:btts_yes, :btts_no)) :
                   Set((Symbol("over_$(Int(floor(line)))5"), Symbol("under_$(Int(floor(line)))5")))
        implied = 1.0 ./ Float64.(g.odds_close)
        overround = sum(implied)
        reason = if DF.nrow(g) != want || selections != expected
            "incomplete market"
        elseif !(0.90 <= overround <= 1.10)
            "overround outside [0.90, 1.10]"
        else
            ""
        end
        if !isempty(reason)
            push!(refused, (; match_id = mid, market_name = name, market_line = line,
                             n_selections = DF.nrow(g), overround, reason))
            continue
        end
        out = DF.DataFrame(g)
        out.prob_implied_close = implied
        out.prob_fair_close = implied ./ overround
        push!(keep, out)
    end
    book = isempty(keep) ? DF.DataFrame(close[1:0, :]) : reduce(vcat, keep)
    refusals = isempty(refused) ? DF.DataFrame(match_id=Int[], market_name=String[], market_line=Float64[],
                                               n_selections=Int[], overround=Float64[], reason=String[]) : DF.DataFrame(refused)
    return book, refusals
end

"Model probabilities in the same ordering and on the same (unrenormalised) grid as Features."
function selection_probabilities(theta::Vector{Float64})
    P = FEAT.build_probability_matrix(FEAT.DoublePoissonMarketFeature(), theta, 10)
    home = sum(LinearAlgebra.tril(P, -1))
    draw = sum(LinearAlgebra.diag(P))
    away = sum(LinearAlgebra.triu(P, 1))
    yes = sum(@view P[2:end, 2:end])
    q = Dict{Symbol, Float64}(:home => home, :draw => draw, :away => away,
                               :btts_yes => yes, :btts_no => 1 - yes)
    for k in 0:3
        under = sum(P[i+1, j+1] for i in 0:k for j in 0:(k-i))
        q[Symbol("under_$(k)5")] = under
        q[Symbol("over_$(k)5")] = 1 - under
    end
    return q
end

"Sum KL once per quoted market, including every selection of that complete market."
function kl_loss(theta::Vector{Float64}, book::DF.AbstractDataFrame)
    q = selection_probabilities(theta)
    return sum(Float64(r.prob_fair_close) * log(Float64(r.prob_fair_close) / q[Symbol(r.selection)])
               for r in eachrow(book))
end

"Optimise from dispersed starts; return the first fit and the largest inter-start parameter gap."
function solve_kl(book::DF.AbstractDataFrame; starts=STARTS)
    loss = theta -> kl_loss(theta, book)
    fits = [Optim.optimize(loss, copy(start), Optim.BFGS(),
                           Optim.Options(g_tol=1e-10, iterations=1000))
            for start in starts]
    theta = Optim.minimizer(first(fits))
    spread = maximum(maximum(abs.(Optim.minimizer(f) .- theta)) for f in fits)
    return (; theta, kl=Optim.minimum(first(fits)), converged=all(Optim.converged, fits), spread)
end

"One row per fixture, ordered gates (T014 is a separate explicit exclusion)."
function invert_panel(ds, book::DF.AbstractDataFrame; seasons=["24/25", "25/26"], tournaments=[56, 57])
    matches = DF.filter(r -> !ismissing(r.season) && r.season in seasons &&
                       !ismissing(r.tournament_id) && Int(r.tournament_id) in tournaments, ds.matches)
    rows = NamedTuple[]
    bymatch = Dict(Int(first(g.match_id)) => DF.DataFrame(g) for g in DF.groupby(book, :match_id))
    for m in eachrow(matches)
        id = Int(m.match_id)
        b = get(bymatch, id, book[1:0, :])
        nm = DF.nrow(DF.unique(b, [:market_name, :market_line]))
        has1 = any(b.market_name .== "1X2")
        reason = id == 14035501 ? "T014 swapped book" :
                 DF.nrow(b) == 0 ? "no Betfair book" :
                 !has1 ? "no complete 1X2 market" :
                 DF.nrow(b) < 3 ? "fewer than 3 quoted selections" : ""
        lh = NaN; la = NaN; kl = NaN; spread = NaN; converged = false
        if isempty(reason)
            fit = solve_kl(b)
            lh, la = exp.(fit.theta)
            kl = fit.kl
            spread = fit.spread
            converged = fit.converged
            reason = !converged ? "optimiser not converged" :
                     !isfinite(kl) || kl > 0.01 ? "summed KL > 0.01" :
                     !(0.05 <= lh <= 6 && 0.05 <= la <= 6) ? "lambda outside [0.05, 6]" : ""
        end
        lines = join(sort([r.market_name == "OverUnder" ? "OU$(r.market_line)" : r.market_name
                           for r in eachrow(DF.unique(b, [:market_name, :market_line]))]), ";")
        push!(rows, (; match_id=id, date=Dates.Date(m.match_date), season=String(m.season),
                     tournament=Int(m.tournament_id), home=String(m.home_team), away=String(m.away_team),
                     lambda_h=lh, lambda_a=la, kl, n_markets=nm, n_selections=DF.nrow(b),
                     lines_quoted=lines, has_1x2=has1, accepted=isempty(reason), reason,
                     optim_converged=converged, start_spread=spread))
    end
    return DF.sort!(DF.DataFrame(rows), :match_id)
end

"Bridge an external rate frame into TODO 023's *exact* weekly observation contract."
function MID.build_market_panel(ds, frame::DF.AbstractDataFrame;
                                seasons=["24/25", "25/26"], tournaments=[56, 57], step_days::Int=7)
    m = DF.filter(r -> !ismissing(r.season) && r.season in seasons && Int(r.tournament_id) in tournaments, ds.matches)
    meta = DF.DataFrame(match_id=Int.(m.match_id), match_date=Dates.Date.(m.match_date),
                        season=String.(m.season), tournament_id=Int.(m.tournament_id),
                        home_team=String.(m.home_team), away_team=String.(m.away_team))
    full = DF.innerjoin(meta, frame; on=:match_id)
    DF.nrow(full) == DF.nrow(meta) || error("external inversion frame lost fixtures")
    refusals = DF.select(DF.filter(:accepted => !, full), :match_id, :match_date, :season,
                         :home_team, :away_team, :n_targets, :sse, :reason)
    acc = DF.filter(:accepted => identity, full)
    teams = sort!(unique(vcat(meta.home_team, meta.away_team)))
    tix = Dict(t => i for (i,t) in enumerate(teams))
    d0 = minimum(meta.match_date)
    d0 -= Dates.Day(Dates.dayofweek(d0)-1)
    acc.week = [div(Dates.value(d-d0), step_days)+1 for d in acc.match_date]
    DF.sort!(acc, [:week, :match_date, :match_id])
    nw = div(Dates.value(maximum(meta.match_date)-d0), step_days)+1
    ow=Int[]; oh=Float64[]; oa=Int[]; od=Int[]; oy=Float64[]; om=Int[]; os=String[]
    for r in eachrow(acc)
        h=tix[r.home_team]; a=tix[r.away_team]
        append!(ow, (r.week, r.week)); append!(oh, (1.0, 0.0))
        append!(oa, (h,a)); append!(od, (a,h))
        append!(oy, (log(r.lambda_mkt_h),log(r.lambda_mkt_a)))
        append!(om, (r.match_id,r.match_id)); append!(os, (r.season,r.season))
    end
    ptr=zeros(Int,nw+1); ptr[1]=1
    for w in ow
        ptr[w+1]+=1
    end
    cumsum!(ptr,ptr)
    return MID.MarketPanel(DF.select(acc, :match_id,:match_date,:week,:season,:tournament_id,
                                     :home_team,:away_team,:lambda_mkt_h,:lambda_mkt_a,:sse),
                           refusals,teams,nw,[d0+Dates.Day(step_days*(t-1)) for t in 1:nw],
                           ow,oh,oa,od,oy,om,os,ptr,zeros(length(oy),0),String[])
end

"Summarise gate attrition, including the explicit swapped-book exclusion."
function coverage(rates)
    order = ["no Betfair book", "no complete 1X2 market", "fewer than 3 quoted selections",
             "optimiser not converged", "summed KL > 0.01", "lambda outside [0.05, 6]", "T014 swapped book"]
    remaining = DF.nrow(rates)
    rows = [(; gate="panel", refused=0, remaining)]
    for gate in order
        refused=count(==(gate), rates.reason)
        remaining-=refused
        push!(rows, (; gate, refused, remaining))
    end
    push!(rows, (; gate="accepted", refused=0, remaining))
    return DF.DataFrame(rows)
end

"Fixture-cluster bootstrap on mean residual; the same sampled fixtures contribute to both sides."
function residual_summary(residuals; reps=2000, seed=3901)
    rng=Random.MersenneTwister(seed)
    rows=NamedTuple[]
    for g in DF.groupby(residuals, [:scope,:tournament,:n_markets,:line,:selection])
        ids=unique(g.match_id)
        values=Dict(id => collect(g.residual[g.match_id .== id]) for id in ids)
        boot=Vector{Float64}(undef,reps)
        for i in 1:reps
            draw=rand(rng,ids,length(ids))
            boot[i]=Statistics.mean(vcat((values[id] for id in draw)...))
        end
        push!(rows, (; scope=first(g.scope), tournament=first(g.tournament),
                     n_markets=first(g.n_markets), line=first(g.line), selection=first(g.selection),
                     n=length(ids), mean=Statistics.mean(g.residual), sd=length(g.residual)>1 ? Statistics.std(g.residual) : NaN,
                     ci_low=Statistics.quantile(boot,0.025), ci_high=Statistics.quantile(boot,0.975),
                     mean_abs=Statistics.mean(abs.(g.residual))))
    end
    return DF.DataFrame(rows)
end

"Per-selection residuals for a set of fixture books, optionally refitted with only 1X2+OU2.5."
function residual_rows(rates, book; heldout=false)
    bymatch=Dict(Int(first(g.match_id)) => DF.DataFrame(g) for g in DF.groupby(book,:match_id))
    rows=NamedTuple[]
    for r in eachrow(DF.filter(:accepted => identity, rates))
        b=bymatch[r.match_id]
        fitbook=DF.filter(x -> x.market_name == "1X2" ||
                         (x.market_name == "OverUnder" && x.market_line == 2.5), b)
        if heldout
            DF.nrow(fitbook)==5 || continue
            DF.nrow(b)>5 || continue
        end
        theta = heldout ? solve_kl(fitbook; starts=(STARTS[1],)).theta : log.([r.lambda_h,r.lambda_a])
        q=selection_probabilities(theta)
        for x in eachrow(b)
            line=x.market_name == "OverUnder" ? "OU$(x.market_line)" : x.market_name
            heldout && line in ("1X2","OU2.5") && continue
            push!(rows,(; match_id=r.match_id, scope=heldout ? "heldout" : "insample",
                          tournament=r.tournament, n_markets=r.n_markets, line,
                          selection=String(x.selection), residual=q[Symbol(x.selection)]-x.prob_fair_close))
        end
    end
    return DF.DataFrame(rows)
end

end # module

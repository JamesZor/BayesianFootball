module CopulaGridMarket

import CSV
import DataFrames
import Dates
import Distributions
import ForwardDiff
import LinearAlgebra
import Optim
import Random
import Serialization
import Statistics

if !isdefined(parentmodule(@__MODULE__), :CovarianceMarket)
    Base.include(parentmodule(@__MODULE__), joinpath(@__DIR__, "l03_covariance.jl"))
end
const CM = parentmodule(@__MODULE__).CovarianceMarket
const TB = CM.TB
const MM = CM.MM
const MID = CM.MID
const DF = DataFrames
const ST = Statistics
const LA = LinearAlgebra
const DS = Distributions

"One global dependence parameter, never a per-fixture parameter. G2 parameter is lambda3."
struct ScoreGrid
    kind::Int
    parameter::Float64
    function ScoreGrid(kind::Int, parameter::Real=0.0)
        kind in 0:3 || error("grid must be G0–G3")
        kind == 2 && parameter < 0 && error("lambda3 must be nonnegative")
        new(kind, Float64(parameter))
    end
end
grid_name(g::ScoreGrid) = "G$(g.kind)"

"Numerically stable Frank CDF; independence limit fixed before rate optimisation."
function frank_cdf(u, v, kappa)
    abs(kappa) <= 1e-7 && return u*v
    return -log1p(expm1(-kappa*u)*expm1(-kappa*v)/expm1(-kappa))/kappa
end

"Poisson probabilities by recurrence (also differentiable in rate)."
function poisson_vector(rate, cutoff; scaled=false)
    p = fill(zero(rate), cutoff+1)
    p[1] = scaled ? one(rate) : exp(-rate)
    for x in 1:cutoff
        p[x+1] = p[x]*rate/x
    end
    return p
end

"G2 theta contains log idiosyncratic rates; report expected goals, not lambda1/lambda2."
reported_rates(g, theta) = exp.(theta) .+ (g.kind == 2 ? g.parameter : 0.0)

"Local G1–G3 on 11x11 support, renormalised. G0 reuses Phase A exactly."
function score_grid(g::ScoreGrid, theta; cutoff=10)
    if g.kind == 0
        return MM.FEAT.build_probability_matrix(MM.FEAT.DoublePoissonMarketFeature(), Float64.(theta), cutoff)
    end
    a, b = exp.(theta)
    # Common exponential factors cancel on renormalisation in G1/G2; omit them
    # to avoid underflow during line-search excursions (no rate-gate change).
    scaled = g.kind in (1,2)
    ph, pa = poisson_vector(a,cutoff; scaled), poisson_vector(b,cutoff; scaled)
    P = ph*pa'
    if g.kind == 1
        rho = g.parameter
        # Same dixon_coles_tau algebra, generic in theta for ForwardDiff.
        P[1,1] *= 1-a*b*rho
        P[1,2] *= 1+a*rho
        P[2,1] *= 1+b*rho
        P[2,2] *= 1-rho
    elseif g.kind == 2
        shared = poisson_vector(g.parameter,cutoff; scaled=true)
        for y in 0:cutoff, x in 0:cutoff
            P[x+1,y+1] = sum(ph[x-k+1]*pa[y-k+1]*shared[k+1] for k in 0:min(x,y))
        end
    elseif abs(g.parameter) > 1e-7
        fh, fa = vcat(zero(a),cumsum(ph)), vcat(zero(b),cumsum(pa))
        C = [frank_cdf(u,v,g.parameter) for u in fh, v in fa]
        for y in 0:cutoff, x in 0:cutoff
            P[x+1,y+1] = C[x+2,y+2]-C[x+1,y+2]-C[x+2,y+1]+C[x+1,y+1]
        end
        # CDF subtraction may lose a few ulps in cells with mass below machine epsilon.
        minimum(ForwardDiff.value.(P)) >= -2e-14 || error("Frank grid has materially negative mass: $(minimum(P))")
        P = max.(P,zero(a))
    end
    # ForwardDiff orders equal primals by their derivatives; probability validity
    # is a statement about primal mass only, never about the sign of a derivative.
    minimum(ForwardDiff.value.(P)) >= 0 || error("invalid grid mass: grid=$(g.kind), theta=$theta, min=$(minimum(P)), sum=$(sum(P))")
    return P/sum(P)
end

function grid_probabilities(g, theta)
    P = score_grid(g,theta)
    q = Dict(:home=>sum(LA.tril(P,-1)), :draw=>sum(LA.diag(P)), :away=>sum(LA.triu(P,1)),
        :btts_yes=>sum(P[2:end,2:end]))
    q[:btts_no] = 1-q[:btts_yes]
    for k in 0:3
        under = sum(P[x+1,y+1] for x in 0:k for y in 0:(k-x))
        q[Symbol("under_$(k)5")] = under
        q[Symbol("over_$(k)5")] = 1-under
    end
    return q
end

"KL once per complete market; refuse infeasible DC domains instead of clipping tau."
function grid_loss(g, theta, selections, probabilities)
    maximum(abs,theta) > 8 && return oftype(theta[1],Inf)
    if g.kind == 1
        h,a = exp.(theta)
        rho = g.parameter
        min(1-h*a*rho,1+h*rho,1+a*rho,1-rho) <= 0 && return oftype(theta[1],Inf)
    end
    q = grid_probabilities(g,theta)
    return sum(p*log(p/q[s]) for (s,p) in zip(selections,probabilities))
end

"Identical dispersed starts, BFGS tolerance and gates; AD gradients for local grids."
function solve_grid(g, book; starts=MM.STARTS)
    g.kind == 0 && return MM.solve_kl(book; starts)
    selections = Symbol.(book.selection)
    probabilities = Float64.(book.prob_fair_close)
    loss = theta -> grid_loss(g,theta,selections,probabilities)
    fits = [Optim.optimize(loss,copy(start),Optim.BFGS(),
        Optim.Options(g_tol=1e-10,iterations=1000); autodiff=:forward) for start in starts]
    theta = Optim.minimizer(first(fits))
    spread = maximum(maximum(abs.(Optim.minimizer(f)-theta)) for f in fits)
    return (; theta,kl=Optim.minimum(first(fits)),converged=all(Optim.converged,fits),spread)
end

function inversion_gate(g, fit)
    h,a = reported_rates(g,fit.theta)
    return fit.converged && isfinite(fit.kl) && fit.kl <= 0.01 &&
        0.05 <= h <= 6 && 0.05 <= a <= 6 && fit.spread <= 1e-6
end

"Invert exactly Phase A's accepted population; never discard a failed fixture."
function invert_accepted(g, rates, books; enforce=true)
    rows = Vector{NamedTuple}(undef,DF.nrow(rates))
    Threads.@threads for i in 1:DF.nrow(rates)
        r = rates[i,:]
        fit = solve_grid(g,books[r.match_id])
        h,a = reported_rates(g,fit.theta)
        rows[i] = (; grid=grid_name(g),parameter=g.parameter,match_id=r.match_id,
            lambda_h=h,lambda_a=a,kl=fit.kl,converged=fit.converged,
            start_spread=fit.spread,gate_pass=inversion_gate(g,fit))
    end
    out = DF.DataFrame(rows)
    if enforce && !all(out.gate_pass)
        bad = out[.!out.gate_pass,:]
        show(stdout,MIME"text/plain"(),bad; allrows=true,allcols=true)
        error("$(grid_name(g)) inversion gates failed on $(DF.nrow(bad)) fixtures; no subset promotion")
    end
    return out
end

"Profile summed KL over all 595 books, then bounded golden-section refinement."
function profile_grid(kind,rates,books,out; candidates)
    rows = NamedTuple[]
    cache = Dict{Float64,Float64}()
    function evaluate(parameter,stage)
        if haskey(cache,parameter)
            return cache[parameter]
        end
        g = ScoreGrid(kind,parameter)
        frame = invert_accepted(g,rates,books; enforce=false)
        # Keep failed candidates visible: optimisation/multistart failures make a profile unusable;
        # poor KL/rate gates are also recorded, never removed from the sum.
        valid = all(frame.converged) && all(frame.start_spread .<= 1e-6)
        total = valid ? sum(frame.kl) : Inf
        push!(rows,(; grid=grid_name(g),parameter,stage,total_kl=total,
            n=DF.nrow(frame),n_gate_pass=count(frame.gate_pass),
            max_kl=maximum(frame.kl),max_spread=maximum(frame.start_spread)))
        cache[parameter] = total
        println("profile $(grid_name(g)) $parameter: KL $total; gates $(count(frame.gate_pass))/$(DF.nrow(frame))")
        flush(stdout)
        CSV.write(joinpath(out,"profile_G$kind.csv"),DF.DataFrame(rows))
        return total
    end
    values = [evaluate(x,"coarse") for x in candidates]
    k = argmin(values)
    isfinite(values[k]) || error("G$kind has no usable profile candidate")
    lo,hi = candidates[max(1,k-1)],candidates[min(length(candidates),k+1)]
    fit = Optim.optimize(x->evaluate(x,"refine"),lo,hi,Optim.GoldenSection(); abs_tol=1e-5,rel_tol=1e-5)
    options = [(values[k],candidates[k]),(Optim.minimum(fit),Optim.minimizer(fit))]
    _,best = minimum(options)
    g = ScoreGrid(kind,best)
    final = invert_accepted(g,rates,books)
    return g,final,DF.DataFrame(rows)
end

"Fixture bootstrap for signed and absolute error; duplicates remain whole-fixture clusters."
function error_summary(ids, values; reps=2000,seed=3941)
    fixtures = sort(unique(ids))
    groups = Dict(id=>findall(==(id),ids) for id in fixtures)
    rng = Random.Xoshiro(seed)
    boot,boot_abs = zeros(reps),zeros(reps)
    for b in 1:reps
        ix = reduce(vcat,[groups[id] for id in rand(rng,fixtures,length(fixtures))])
        boot[b] = ST.mean(values[ix])
        boot_abs[b] = ST.mean(abs.(values[ix]))
    end
    q = ST.quantile(values,[0.05,0.5,0.95])
    return (; n=length(fixtures),mean=ST.mean(values),mean_abs=ST.mean(abs.(values)),
        ci_low=ST.quantile(boot,0.025),ci_high=ST.quantile(boot,0.975),
        abs_ci_low=ST.quantile(boot_abs,0.025),abs_ci_high=ST.quantile(boot_abs,0.975),
        q05=q[1],q50=q[2],q95=q[3],reps,seed)
end

"Convert reported expected-goal rates to the grid's optimiser coordinates."
rate_theta(g,h,a) = log.([h,a] .- (g.kind == 2 ? g.parameter : 0.0))

function grid_residual_rows(g,frame,rates,books; heldout=false)
    meta = Dict(r.match_id=>r for r in eachrow(rates))
    rows = NamedTuple[]
    for r in eachrow(frame)
        b = books[r.match_id]
        fitbook = DF.filter(x->x.market_name == "1X2" ||
            (x.market_name == "OverUnder" && x.market_line == 2.5),b)
        if heldout
            DF.nrow(fitbook) == 5 && DF.nrow(b) > 5 || continue
            fit = solve_grid(g,fitbook)
            inversion_gate(g,fit) || error("heldout inversion gate $(grid_name(g)), $(r.match_id): $fit")
            theta = fit.theta
        else
            theta = rate_theta(g,r.lambda_h,r.lambda_a)
        end
        q = grid_probabilities(g,theta)
        m = meta[r.match_id]
        for x in eachrow(b)
            line = x.market_name == "OverUnder" ? "OU$(x.market_line)" : x.market_name
            heldout && line in ("1X2","OU2.5") && continue
            push!(rows,(; grid=grid_name(g),match_id=r.match_id,tournament=m.tournament,
                n_markets=m.n_markets,line,selection=String(x.selection),
                residual=q[Symbol(x.selection)]-x.prob_fair_close))
        end
    end
    return DF.DataFrame(rows)
end

"Phase A splits plus pooled heldout loss; all CIs bootstrap fixtures, not selections."
function line_summaries(raw; heldout=false)
    rows = NamedTuple[]
    for g in DF.groupby(raw,:grid), split in (:overall,:tournament,:n_markets)
        data = copy(g)
        split == :tournament || (data.tournament .= 0)
        split == :n_markets || (data.n_markets .= 0)
        for line in DF.groupby(data,[:tournament,:n_markets,:line,:selection])
            push!(rows,(; grid=first(line.grid),scope=heldout ? "heldout" : "insample",
                tournament=first(line.tournament),n_markets=first(line.n_markets),
                line=first(line.line),selection=first(line.selection),
                error_summary(line.match_id,line.residual)...))
        end
    end
    if heldout
        for g in DF.groupby(raw,:grid)
            push!(rows,(; grid=first(g.grid),scope="heldout",tournament=0,n_markets=0,
                line="ALL",selection="ALL",error_summary(g.match_id,g.residual)...))
        end
    end
    return DF.DataFrame(rows)
end

function one_x2_bias(g,frame,rates,books)
    meta = Dict(r.match_id=>r for r in eachrow(rates))
    values,ids = Float64[],Int[]
    for r in eachrow(frame)
        meta[r.match_id].n_selections >= 5 || continue
        fit = solve_grid(g,DF.filter(:market_name=>==("1X2"),books[r.match_id]))
        inversion_gate(g,fit) || error("1X2-only gate $(grid_name(g)) $(r.match_id): $fit")
        h,a = reported_rates(g,fit.theta)
        push!(ids,r.match_id)
        push!(values,log(h+a)-log(r.lambda_h+r.lambda_a))
    end
    return (; grid=grid_name(g),parameter=g.parameter,axis="log_total",error_summary(ids,values)...)
end

function outcome_loss(g,frame,rates,books,ds,baseline)
    fullids = Set(rates.match_id[rates.n_selections .>= 5])
    results = Dict(Int(r.match_id)=>r for r in eachrow(ds.matches))
    rows = NamedTuple[]
    for r in eachrow(frame)
        r.match_id in fullids || continue
        result = results[r.match_id]
        h,a = result.home_score,result.away_score
        (ismissing(h) || ismissing(a)) && error("missing outcome $(r.match_id)")
        win = h > a ? :home : h == a ? :draw : :away
        total = h+a > 2.5 ? :over_25 : :under_25
        q = grid_probabilities(g,rate_theta(g,r.lambda_h,r.lambda_a))
        b = books[r.match_id]
        market = Dict(Symbol(x.selection)=>Float64(x.prob_fair_close) for x in eachrow(b))
        ref = baseline[r.match_id]
        q0 = grid_probabilities(ScoreGrid(0),log.([ref.lambda_h,ref.lambda_a]))
        for (axis,s) in (("1X2",win),("OU2.5",total))
            haskey(market,s) || error("outcome set lacks $axis close $(r.match_id)")
            push!(rows,(; grid=grid_name(g),match_id=r.match_id,axis,
                logloss=-log(q[s]),g0_logloss=-log(q0[s]),close_logloss=-log(market[s])))
        end
    end
    raw = DF.DataFrame(rows)
    summaries = NamedTuple[]
    for data in DF.groupby(raw,:axis), (comparison,values) in
        (("absolute",data.logloss),("minus_G0",data.logloss-data.g0_logloss),
         ("minus_close",data.logloss-data.close_logloss))
        push!(summaries,(; grid=grid_name(g),axis=first(data.axis),comparison,
            error_summary(data.match_id,values)...))
    end
    return DF.DataFrame(summaries),raw
end

function rate_shift(g,frame,baseline)
    rows = NamedTuple[]
    raw = NamedTuple[]
    for r in eachrow(frame)
        b = baseline[r.match_id]
        h,a = log(r.lambda_h),log(r.lambda_a)
        h0,a0 = log(b.lambda_h),log(b.lambda_a)
        for (axis,value) in (("log_lambda_h",h-h0),("log_lambda_a",a-a0),
            ("supremacy",h-a-h0+a0),("level",(h+a-h0-a0)/2),("log_total",log(r.lambda_h+r.lambda_a)-log(b.lambda_h+b.lambda_a)))
            push!(raw,(; grid=grid_name(g),match_id=r.match_id,axis,delta=value))
        end
    end
    table = DF.DataFrame(raw)
    for data in DF.groupby(table,:axis)
        push!(rows,(; grid=grid_name(g),axis=first(data.axis),error_summary(data.match_id,data.delta)...))
    end
    return DF.DataFrame(rows),table
end

"Same panel identities and weeks; only the two measurements change."
function panel_with_rates(panel,frame)
    rates = Dict(r.match_id=>r for r in eachrow(frame))
    y = [log(j % 2 == 1 ? rates[panel.obs_match[j]].lambda_h : rates[panel.obs_match[j]].lambda_a)
        for j in eachindex(panel.obs_y)]
    matches = copy(panel.matches)
    matches.lambda_mkt_h = [rates[id].lambda_h for id in matches.match_id]
    matches.lambda_mkt_a = [rates[id].lambda_a for id in matches.match_id]
    return MID.MarketPanel(matches,panel.refusals,panel.teams,panel.n_weeks,panel.week_start,
        panel.obs_week,panel.obs_home,panel.obs_att,panel.obs_def,y,panel.obs_match,
        panel.obs_season,panel.week_ptr,panel.obs_X,panel.feature_names)
end

include("l04_tail_ppc.jl")
include("l04_grid_diagnostics.jl")
end # module

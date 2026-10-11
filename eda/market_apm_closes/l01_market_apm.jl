module MarketAPMPhaseA
import BayesianFootball, CSV, DataFrames, SHA, LinearAlgebra, SparseArrays, Statistics, Distributions
const LEAGUES = [("sco_league_one",56,"ScottishLower"),("sco_league_two",57,"ScottishLower"),
    ("sco_premiership",54,"ScottishUpper"),("sco_championship",55,"ScottishUpper"),
    ("eng_premier",1,"MarketModelEnglish"),("eng_championship",2,"MarketModelEnglish"),
    ("eng_league_one",3,"MarketModelEnglish"),("eng_league_two",84,"MarketModelEnglish")]
const HASHES = Dict("MarketModelEnglish"=>"2efe200ca6ed2899a75246570e025d5782af5da945ec26100bfa38331c5168bf",
    "ScottishUpper"=>"56207e6354f0e9fbf5a6d1e29a277ae49b50769d2975de7a7e7ccca85c93630a",
    "ScottishLower"=>"e894ac6d59f912aceb01f639c80be37555a73f4eab2fa24271cddd801f13006a")

"Verify before loading; the English serialized segment requires its original module identity."
function pinned_datastores(root)
    segments = Dict("ScottishLower"=>BayesianFootball.Data.ScottishLower(),
        "ScottishUpper"=>BayesianFootball.Data.ScottishUpper(),
        "MarketModelEnglish"=>Main.QualityStyleEDA.MarketModelEnglish())
    for (name, hash) in HASHES
        actual = open(SHA.sha256, joinpath(root, ".cache", "datastore_$(name).jls"))
        bytes2hex(actual) == hash || error("STOP: SHA mismatch for $name")
    end
    Dict(name=>BayesianFootball.Data.load_datastore_cached(segment; max_age_hours=10^6)
        for (name,segment) in segments)
end

"Coverage on the inversion's included seasons, with disjoint accepted-close failure reasons."
function coverage(root, out)
    stores = pinned_datastores(root)
    rows = NamedTuple[]
    failures = NamedTuple[]
    summary = NamedTuple[]
    gates = NamedTuple[]
    for (league, tournament, segment) in LEAGUES
        ds = stores[segment]
        rates = CSV.read(joinpath(root,"current_development/market_model/results/QS/rates_$league.csv"), DataFrames.DataFrame)
        length(unique(rates.match_id)) == DataFrames.nrow(rates) || error("Duplicate rate match IDs: $league")
        fixtures = ds.matches[ds.matches.tournament_id .== tournament,:]
        length(unique(fixtures.match_id)) == DataFrames.nrow(fixtures) || error("Duplicate fixtures: $league")
        fixture_ids = Set(fixtures.match_id)
        starters = ds.lineups[(ds.lineups.tournament_id .== tournament) .& .!ds.lineups.is_substitute,:]
        sides = Dict((g.match_id[1], String(g.team_side[1])) => collect(g.player_id)
            for g in DataFrames.groupby(starters, [:match_id,:team_side]))
        accepted = rates[rates.accepted .== true,:]
        total_joined = 0
        for season in sort(unique(rates.season))
            season_rates = rates[rates.season .== season,:]
            rate_ids = Set(season_rates.match_id)
            no_row = count(id->!(id in rate_ids),fixtures.match_id[fixtures.season .== season])
            no_row>0 && push!(summary,(league=league,season=String(season),stage="before_join",reason="fixture_without_inversion_row",n=no_row))
            rejected = season_rates[season_rates.accepted .== false,:]
            for g in DataFrames.groupby(rejected,:reason)
                reason = ismissing(g.reason[1]) ? "unspecified" : String(g.reason[1])
                push!(summary,(league=league,season=String(season),stage="inversion_rejected",reason=reason,n=DataFrames.nrow(g)))
            end
            r = accepted[accepted.season .== season,:]
            complete = 0
            joined = 0
            for m in DataFrames.eachrow(r)
                h = get(sides,(m.match_id,"home"),Int[])
                a = get(sides,(m.match_id,"away"),Int[])
                valid = length(h)==11 && length(a)==11 && length(unique(h))==11 && length(unique(a))==11 && isempty(intersect(h,a))
                complete += valid
                reason = !(m.match_id in fixture_ids) ? "missing_fixture" :
                    isempty(h) || isempty(a) ? "missing_starting_side" :
                    length(h)!=11 || length(a)!=11 ? "starter_count_not_11" :
                    !valid ? "duplicate_or_shared_player_id" :
                    !isfinite(m.lambda_h) || !isfinite(m.lambda_a) || min(m.lambda_h,m.lambda_a)<=0 ? "invalid_rate" : "ok"
                if reason == "ok"
                    joined += 1
                else
                    push!(failures,(league=league,season=season,match_id=m.match_id,reason=reason,home_starters=length(h),away_starters=length(a)))
                end
            end
            n = DataFrames.nrow(r)
            push!(rows,(league=league,tournament=tournament,season=season,
                fixtures=count(==(season),fixtures.season),accepted_closes=n,
                closes_complete_xi=complete,joined_n=joined,loss_fraction=n==0 ? 0.0 : 1-joined/n))
            total_joined += joined
        end
        n = DataFrames.nrow(accepted)
        loss = 1-total_joined/n
        push!(gates,(league=league,accepted=n,joined_n=total_joined,loss_fraction=loss,blocked=loss>0.15))
        println(league, ": accepted=",n," joined=",total_joined," loss=",round(100loss;digits=2),"%",loss>0.15 ? " STOP" : "")
    end
    CSV.write(joinpath(out,"coverage.csv"),DataFrames.DataFrame(rows))
    failure_df = DataFrames.DataFrame(failures)
    CSV.write(joinpath(out,"join_failures.csv"),failure_df)
    for g in DataFrames.groupby(failure_df,[:league,:season,:reason])
        push!(summary,(league=String(g.league[1]),season=String(g.season[1]),stage="accepted_join",reason=String(g.reason[1]),n=DataFrames.nrow(g)))
    end
    CSV.write(joinpath(out,"failure_summary.csv"),DataFrames.DataFrame(summary))
    CSV.write(joinpath(out,"coverage_gate.csv"),DataFrames.DataFrame(gates))
    return any(r.blocked for r in gates)
end

"Fixture-side histories. Regulars use strictly earlier starts; unknown XIs contribute no starts."
function histories(fixtures, starters)
    sides = Dict((g.match_id[1], String(g.team_side[1])) => collect(Int.(g.player_id))
        for g in DataFrames.groupby(starters, [:match_id,:team_side]))
    records = NamedTuple[]
    for m in DataFrames.eachrow(fixtures), side in ("home","away")
        players = get(sides,(m.match_id,side),Int[])
        other = get(sides,(m.match_id,side=="home" ? "away" : "home"),Int[])
        valid = length(players)==11 && length(unique(players))==11 && isempty(intersect(players,other))
        push!(records,(match_id=Int(m.match_id),season=String(m.season),date=m.match_date,
            team=String(side=="home" ? m.home_team : m.away_team),side=side,
            players=players,valid=valid))
    end
    df = DataFrames.DataFrame(records)
    sort!(df,[:season,:team,:date,:match_id])
    df.ordinal = zeros(Int,DataFrames.nrow(df))
    df.absent = fill(NaN,DataFrames.nrow(df))
    df.overlap = fill(NaN,DataFrames.nrow(df))
    df.prev_match = zeros(Int,DataFrames.nrow(df))
    df.next_match = zeros(Int,DataFrames.nrow(df))
    df.history_complete = falses(DataFrames.nrow(df))
    for g in DataFrames.groupby(df,[:season,:team])
        counts = Dict{Int,Int}()
        all_previous_valid = true
        for i in 1:DataFrames.nrow(g)
            g.ordinal[i] = i
            g.history_complete[i] = all_previous_valid
            regulars = Set(first(sort(collect(keys(counts));by=p->(-counts[p],p)), min(11,length(counts))))
            if g.valid[i]
                g.absent[i] = 11 - length(intersect(regulars,Set(g.players[i])))
            end
            if i>1
                g.prev_match[i] = g.match_id[i-1]
                if g.valid[i] && g.valid[i-1]
                    g.overlap[i] = length(intersect(g.players[i],g.players[i-1]))/11
                end
            end
            i<DataFrames.nrow(g) && (g.next_match[i] = g.match_id[i+1])
            for p in (g.valid[i] ? g.players[i] : Int[])
                counts[p] = get(counts,p,0)+1
            end
            all_previous_valid &= g.valid[i]
        end
    end
    df
end

"Stacked log rates: own team-season attack + opposing defence + home-rate indicator."
function matrices(joined, hdict)
    n = DataFrames.nrow(joined)
    teams = sort(unique(vcat([(String(r.season),String(r.home)) for r in DataFrames.eachrow(joined)],
        [(String(r.season),String(r.away)) for r in DataFrames.eachrow(joined)])))
    team_index = Dict(t=>i for (i,t) in enumerate(teams))
    players = sort(unique(vcat([hdict[(r.match_id,s)].players for r in DataFrames.eachrow(joined) for s in ("home","away")]...)))
    player_index = Dict(p=>i for (i,p) in enumerate(players))
    ti=Int[]; tj=Int[]; pi=Int[]; pj=Int[]
    t=length(teams); p=length(players)
    for (i,r) in enumerate(DataFrames.eachrow(joined))
        h=team_index[(String(r.season),String(r.home))]; a=team_index[(String(r.season),String(r.away))]
        append!(ti,[i,i,i,n+i,n+i]); append!(tj,[h,t+a,2t+1,a,t+h])
        for player in hdict[(r.match_id,"home")].players
            j=player_index[player]; append!(pi,[i,n+i]); append!(pj,[j,p+j])
        end
        for player in hdict[(r.match_id,"away")].players
            j=player_index[player]; append!(pi,[n+i,i]); append!(pj,[j,p+j])
        end
    end
    T=SparseArrays.sparse(ti,tj,ones(length(ti)),2n,2t+1)
    P=SparseArrays.sparse(pi,pj,ones(length(pi)),2n,2p)
    T,P,players
end

"SPQR numerical rank with explicit absolute tolerance on these unit-indicator matrices."
function sparse_rank(A; tol=1e-8)
    F=LinearAlgebra.qr(A;tol=tol)
    count(x->abs(x)>tol,LinearAlgebra.diag(F.R))
end

home_probability(h,a) = sum(Distributions.pdf(Distributions.Poisson(h),k)*
    Distributions.cdf(Distributions.Poisson(a),k-1) for k in 1:80)

"OLS with single-match HC1 and season-block CR1 uncertainty, no causal interpretation."
function slopes(y,x,season; control=nothing)
    X = control===nothing ? hcat(ones(length(y)),x) : hcat(ones(length(y)),x,control)
    n,k=size(X)
    n>k || error("Too few regression observations")
    LinearAlgebra.rank(X)==k || error("Singular absence regression")
    beta=X\y
    e=y-X*beta
    bread=inv(X'X)
    scores=X.*e
    hc=bread*(scores'scores)*bread*n/(n-k)
    groups=unique(season); G=length(groups)
    meat=zeros(k,k)
    for group in groups
        score=vec(sum(scores[season.==group,:];dims=1))
        meat .+= score*score'
    end
    block=bread*meat*bread*(G/(G-1))*((n-1)/(n-k))
    [(se_type="match_HC1",se=sqrt(max(0,hc[2,2])),df=n-k),
     (se_type="season_CR1",se=sqrt(max(0,block[2,2])),df=G-1)], beta[2], n, G
end

"Small, dependency-free SVG of means and 95% descriptive mean intervals."
function figure(path, title, bins)
    xmin=minimum(bins.x)-1; xmax=maximum(bins.x)+1
    ymin=minimum(bins.mean.-1.96bins.se)-0.05
    ymax=maximum(bins.mean.+1.96bins.se)+0.05
    sx(x)=70+640*(x-xmin)/(xmax-xmin)
    sy(y)=350-280*(y-ymin)/(ymax-ymin)
    open(path,"w") do io
        println(io,"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"800\" height=\"450\" viewBox=\"0 0 800 450\"><rect width=\"800\" height=\"450\" fill=\"white\"/>")
        println(io,"<text x=\"70\" y=\"30\" font-family=\"sans-serif\" font-size=\"19\">$title</text>")
        println(io,"<path d=\"M70 65V350H720\" fill=\"none\" stroke=\"black\"/>")
        if ymin<=0<=ymax
            println(io,"<path d=\"M70 $(sy(0))H720\" stroke=\"#aaa\" stroke-dasharray=\"5 5\"/>")
        end
        for b in DataFrames.eachrow(bins)
            x=sx(b.x); y=sy(b.mean)
            println(io,"<path d=\"M$x $(sy(b.mean-1.96b.se))V$(sy(b.mean+1.96b.se))\" stroke=\"#2866a0\"/><circle cx=\"$x\" cy=\"$y\" r=\"5\" fill=\"#2866a0\"/>")
            println(io,"<text x=\"$(x-5)\" y=\"370\" font-family=\"sans-serif\" font-size=\"12\">$(b.x)</text><text x=\"$(x-10)\" y=\"$(y-12)\" font-family=\"sans-serif\" font-size=\"10\">n=$(b.n)</text>")
        end
        for v in range(ymin,ymax;length=5)
            println(io,"<text x=\"10\" y=\"$(sy(v)+4)\" font-family=\"sans-serif\" font-size=\"12\">$(round(v;digits=2))</text>")
        end
        println(io,"<text x=\"180\" y=\"405\" font-family=\"sans-serif\" font-size=\"15\">Missing regulars: home minus away</text><text x=\"70\" y=\"435\" font-family=\"sans-serif\" font-size=\"12\">Team-season residual log supremacy; bars: descriptive 95% mean intervals</text></svg>")
    end
end

function analyze(root,out)
    stores=pinned_datastores(root)
    designs=NamedTuple[]; headrooms=NamedTuple[]; regressions=NamedTuple[]
    team_metrics=NamedTuple[]; bins_all=NamedTuple[]
    joined_all=DataFrames.DataFrame[]; pooled_x=Float64[]; pooled_y=Float64[]
    mkpath(joinpath(out,"figures"))
    for (league,tournament,segment) in LEAGUES
        ds=stores[segment]
        rates=CSV.read(joinpath(root,"current_development/market_model/results/QS/rates_$league.csv"),DataFrames.DataFrame)
        rates=rates[rates.accepted.==true,:]
        seasons=Set(rates.season)
        fixtures=ds.matches[(ds.matches.tournament_id.==tournament).&[!ismissing(s)&&s in seasons for s in ds.matches.season],:]
        starters=ds.lineups[(ds.lineups.tournament_id.==tournament).& .!ds.lineups.is_substitute,:]
        hist=histories(fixtures,starters)
        hd=Dict((r.match_id,r.side)=>r for r in DataFrames.eachrow(hist))
        good=[haskey(hd,(r.match_id,"home")) && haskey(hd,(r.match_id,"away")) &&
            hd[(r.match_id,"home")].valid && hd[(r.match_id,"away")].valid &&
            isempty(intersect(hd[(r.match_id,"home")].players,hd[(r.match_id,"away")].players)) &&
            isfinite(r.lambda_h) && isfinite(r.lambda_a) && min(r.lambda_h,r.lambda_a)>0
            for r in DataFrames.eachrow(rates)]
        joined=sort(rates[good,:],[:date,:match_id])
        n=DataFrames.nrow(joined)
        T,P,players=matrices(joined,hd)
        y=vcat(log.(joined.lambda_h),log.(joined.lambda_a))
        fit=LinearAlgebra.qr(T;tol=1e-8)\y
        e=y-T*fit
        maximum(abs.(T'e))<1e-7 || error("FE orthogonality failure: $league")
        team_rank=sparse_rank(T)
        combined_rank=sparse_rank(hcat(T,P))
        sup=e[1:n]-e[n+1:2n]; total=e[1:n]+e[n+1:2n]
        starts=Dict(p=>0 for p in players); clubs=Dict(p=>Set{String}() for p in players)
        for r in DataFrames.eachrow(joined), side in ("home","away"), p in hd[(r.match_id,side)].players
            starts[p]+=1; push!(clubs[p],String(side=="home" ? r.home : r.away))
        end
        for g in DataFrames.groupby(hist,[:season,:team])
            counts=Dict{Int,Int}()
            for r in DataFrames.eachrow(g), p in (r.valid ? r.players : Int[])
                counts[p]=get(counts,p,0)+1
            end
            values_sorted=sort(collect(values(counts));rev=true)
            share=isempty(values_sorted) ? NaN : sum(first(values_sorted,min(11,length(values_sorted))))/sum(values_sorted)
            ov=filter(isfinite,g.overlap)
            push!(team_metrics,(league=league,season=String(g.season[1]),team=String(g.team[1]),
                fixtures=DataFrames.nrow(g),valid_sides=count(g.valid),top11_start_share=share,
                mean_previous_xi_overlap=isempty(ov) ? NaN : Statistics.mean(ov)))
        end
        metrics=filter(r->r.league==league,team_metrics)
        overlaps=filter(isfinite,hist.overlap)
        push!(designs,(league=league,tournament=tournament,joined_n=n,players_ge1=length(players),
            players_ge5=count(>=(5),values(starts)),players_ge20=count(>=(20),values(starts)),
            median_starts=Statistics.median(collect(values(starts))),within_league_movers=count(s->length(s)>1,values(clubs)),
            mean_top11_start_share=Statistics.mean([r.top11_start_share for r in metrics if isfinite(r.top11_start_share)]),
            mean_previous_xi_overlap=Statistics.mean(overlaps),team_fe_columns=size(T,2),team_fe_rank=team_rank,
            starter_columns=size(P,2),combined_rank=combined_rank,absorbed_starter_rank=combined_rank-team_rank,rank_tol=1e-8))
        push!(headrooms,(league=league,joined_n=n,team_fe_rank=team_rank,rate_residual_df=2n-team_rank,
            residual_sd_supremacy=Statistics.std(sup),residual_sd_total=Statistics.std(total),
            rate_residual_rmse=sqrt(sum(abs2,e)/(2n-team_rank))))
        residual=Dict(joined.match_id[i]=>(home=e[i],away=e[n+i],sup=sup[i],total=total[i]) for i in 1:n)
        rows=NamedTuple[]
        for (i,r) in enumerate(DataFrames.eachrow(joined))
            h=hd[(r.match_id,"home")]; a=hd[(r.match_id,"away")]
            eligible=h.ordinal>5 && a.ordinal>5
            nh=get(hd,(h.next_match,haskey(hd,(h.next_match,"home")) && hd[(h.next_match,"home")].team==h.team ? "home" : "away"),nothing)
            na=get(hd,(a.next_match,haskey(hd,(a.next_match,"home")) && hd[(a.next_match,"home")].team==a.team ? "home" : "away"),nothing)
            ph=get(residual,h.prev_match,nothing); pa=get(residual,a.prev_match,nothing)
            hprev=get(hd,(h.prev_match,"home"),nothing)
            aprev=get(hd,(a.prev_match,"home"),nothing)
            prev_sup=ph===nothing || pa===nothing ? NaN :
                (hprev.team==h.team ? ph.sup : -ph.sup)-(aprev.team==a.team ? pa.sup : -pa.sup)
            prev_total=ph===nothing || pa===nothing ? NaN : ph.total+pa.total
            next_h=nh===nothing ? NaN : nh.absent; next_a=na===nothing ? NaN : na.absent
            push!(rows,(league=league,season=String(r.season),match_id=Int(r.match_id),date=r.date,
                home=String(r.home),away=String(r.away),lambda_h=r.lambda_h,lambda_a=r.lambda_a,
                residual_supremacy=sup[i],residual_total=total[i],absent_home=h.absent,absent_away=a.absent,
                eligible=eligible,home_ordinal=h.ordinal,away_ordinal=a.ordinal,
                history_complete=h.history_complete && a.history_complete,
                next_absent_difference=next_h-next_a,next_absent_sum=next_h+next_a,
                previous_residual_supremacy=prev_sup,previous_residual_total=prev_total))
        end
        matchdf=DataFrames.DataFrame(rows); push!(joined_all,matchdf)
        typical_h=Statistics.median(joined.lambda_h); typical_a=Statistics.median(joined.lambda_a)
        base_prob=home_probability(typical_h,typical_a)
        for response in ("supremacy","total"), test in ("current","placebo_next","form_previous","current_matched_placebo","complete_history")
            outcome=response=="supremacy" ? matchdf.residual_supremacy : matchdf.residual_total
            current=response=="supremacy" ? matchdf.absent_home-matchdf.absent_away : matchdf.absent_home+matchdf.absent_away
            next=response=="supremacy" ? matchdf.next_absent_difference : matchdf.next_absent_sum
            x=test=="placebo_next" ? next : current
            prev=response=="supremacy" ? matchdf.previous_residual_supremacy : matchdf.previous_residual_total
            mask=matchdf.eligible .& isfinite.(x) .& isfinite.(outcome)
            test=="form_previous" && (mask .&= isfinite.(prev))
            test=="current_matched_placebo" && (mask .&= isfinite.(next))
            test=="complete_history" && (mask .&= matchdf.history_complete)
            uncertainty,beta,nreg,G=slopes(outcome[mask],x[mask],matchdf.season[mask];control=test=="form_previous" ? prev[mask] : nothing)
            delta=home_probability(typical_h*exp(beta/2),typical_a*exp((response=="supremacy" ? -beta : beta)/2))-base_prob
            for u in uncertainty
                critical=Distributions.quantile(Distributions.TDist(u.df),0.975)
                push!(regressions,(league=league,response=response,test=test,se_type=u.se_type,n=nreg,season_blocks=G,
                    slope=beta,se=u.se,ci_low=beta-critical*u.se,ci_high=beta+critical*u.se,
                    p_value=2Distributions.ccdf(Distributions.TDist(u.df),abs(beta/u.se)),
                    typical_lambda_h=typical_h,typical_lambda_a=typical_a,typical_home_probability=base_prob,
                    home_probability_delta=delta))
            end
        end
        mask=matchdf.eligible
        x=matchdf.absent_home[mask]-matchdf.absent_away[mask]; yy=matchdf.residual_supremacy[mask]
        append!(pooled_x,x); append!(pooled_y,yy)
        league_bins=bin_rows(league,x,yy)
        append!(bins_all,league_bins)
        figure(joinpath(out,"figures","$league.svg"),league,DataFrames.DataFrame(league_bins))
        println("ANALYZED ",league," n=",n," absorbed rank=",combined_rank-team_rank)
    end
    pooled_bins=bin_rows("pooled",pooled_x,pooled_y); append!(bins_all,pooled_bins)
    figure(joinpath(out,"figures","pooled.svg"),"Pooled (unweighted matches)",DataFrames.DataFrame(pooled_bins))
    CSV.write(joinpath(out,"design.csv"),DataFrames.DataFrame(designs))
    CSV.write(joinpath(out,"headroom.csv"),DataFrames.DataFrame(headrooms))
    CSV.write(joinpath(out,"absent_slopes.csv"),DataFrames.DataFrame(regressions))
    CSV.write(joinpath(out,"team_season_design.csv"),DataFrames.DataFrame(team_metrics))
    CSV.write(joinpath(out,"figure_bins.csv"),DataFrames.DataFrame(bins_all))
    CSV.write(joinpath(out,"joined_matches.csv"),vcat(joined_all...))
end

function bin_rows(league,x,y)
    [(league=league,x=Int(v),n=count(==(v),x),mean=Statistics.mean(y[x.==v]),
      se=count(==(v),x)>1 ? Statistics.std(y[x.==v])/sqrt(count(==(v),x)) : 0.0)
        for v in sort(unique(x))]
end
end

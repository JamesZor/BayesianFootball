# Included inside QSFormsBatch01. Descriptive finite-league posterior algebra; no hierarchy fitting.
const TIERS = [1,2,3,4,1,2,3,4,1,2,1]
const COUNTRIES = ["England","England","England","England","Scotland","Scotland","Scotland","Scotland","Ireland","Ireland","Finland"]
ols_slope(x,y)=length(x)>1&&ST.var(x)>0 ? ST.cov(x,y)/ST.var(x) : missing
function physical_axis(f,rung)
    n,_,nc=size(f.draws)
    r=[step_geometry(f,rung,i,c).r for c in 1:nc for i in 1:n]
    obs=vec(f.draws[:,only(findall(==("sigma_obs"),f.names)),:])
    return log.(r),obs
end
function r08()
    started=time();rows,inventory,confounding,fieldcorr=NamedTuple[],NamedTuple[],NamedTuple[],NamedTuple[]
    structures=Dict{Tuple{String,String},Any}()
    for (pi,protocol) in enumerate(PROTOCOLS), (ri,rung) in enumerate(("C0","R6"))
        R,O=zeros(11,12000),zeros(11,12000)
        for (li,(league,_,_)) in enumerate(QS.LEAGUES)
            f=fit(league,rung,protocol;request="R08");lr,obs=physical_axis(f,rung)
            @assert length(lr)==12000
            perm=Random.randperm(Random.Xoshiro(seed(li,ri,pi,800)),12000)
            R[li,:].=lr[perm];O[li,:].=obs[perm]
            record!("R08","league_permutation",league*"/"*rung*"/"*protocol,"";n=12000,detail="matched log_r/sigma_obs physical draw permutation; seed=$(seed(li,ri,pi,800))")
        end
        structures[(protocol,rung)]=(;R,O)
        points=vec(ST.median(R;dims=2))
        quantities=Dict{String,Any}(
            "SD_log_r"=>[ST.std(R[:,j]) for j in axes(R,2)],
            "OLS_log_r_on_tier"=>[ols_slope(TIERS,R[:,j]) for j in axes(R,2)],
            "topflight_minus_other"=>[ST.mean(R[TIERS.==1,j])-ST.mean(R[TIERS.!=1,j]) for j in axes(R,2)],
            "Spearman_log_r_sigma_obs"=>[spearman(R[:,j],O[:,j]) for j in axes(R,2)])
        for country in ("England","Scotland","Ireland")
            ix=findall(==(country),COUNTRIES)
            quantities[country*"_tier_slope"]=[ols_slope(TIERS[ix],R[ix,j]) for j in axes(R,2)]
        end
        for (quantity,values) in sort(collect(quantities);by=first)
            qs=quant(values);loo=Float64[]
            cohort=startswith(quantity,"England") ? findall(==("England"),COUNTRIES) : startswith(quantity,"Scotland") ? findall(==("Scotland"),COUNTRIES) : startswith(quantity,"Ireland") ? findall(==("Ireland"),COUNTRIES) : collect(1:11)
            for leftout in cohort
                ix=setdiff(cohort,[leftout]);v=missing
                if occursin("slope",quantity)||quantity=="OLS_log_r_on_tier"
                    v=ols_slope(TIERS[ix],points[ix])
                elseif quantity=="SD_log_r"
                    v=ST.std(points[ix])
                elseif quantity=="topflight_minus_other"
                    top=ix[TIERS[ix].==1];other=ix[TIERS[ix].!=1]
                    !isempty(top)&&!isempty(other)&&(v=ST.mean(points[top])-ST.mean(points[other]))
                elseif quantity=="Spearman_log_r_sigma_obs"
                    v=spearman(points[ix],vec(ST.median(O[ix,:];dims=2)))
                end
                ismissing(v)||push!(loo,v)
            end
            push!(rows,(;protocol,rung,quantity,q05=qs[1],median=qs[2],q95=qs[3],n_leagues=length(cohort),leave_one_out_min=isempty(loo) ? missing : minimum(loo),leave_one_out_max=isempty(loo) ? missing : maximum(loo),interpretation="posterior fit uncertainty among these fixed leagues, not population uncertainty; sigma_obs is not liquidity"))
        end
        push!(rows,(;protocol,rung,quantity="Finland_tier_slope",q05=missing,median=missing,q95=missing,n_leagues=1,leave_one_out_min=missing,leave_one_out_max=missing,interpretation="NOT_AVAILABLE: single division"))
    end
    output("R08","league_structure.csv",rows)
    # Inventory only already saved snapshot tables. Match dates are not quote timestamps.
    columnrows=NamedTuple[];seen=Set{String}()
    for (_,_,segment) in QS.LEAGUES
        name=string(nameof(typeof(segment)));name in seen&&continue;push!(seen,name)
        path=joinpath(dirname(dirname(P)),".cache","datastore_$(name).jls")
        ds=Serialization.deserialize(path)
        for field in propertynames(ds)
            table=getproperty(ds,field)
            table isa DF.AbstractDataFrame || continue
            for col in names(table)
                push!(columnrows,(;snapshot=name,table=String(field),column=col,n_rows=DF.nrow(table)))
            end
        end
    end
    output("R08","snapshot_columns.csv",columnrows)
    push!(inventory,(;field="overround",source="Q/coverage_by_league.csv:overround_q05/median/q95",present=true,unit="sum inverse decimal odds",coverage="all11 leagues/selected seasons",limitation="saved per-season quantiles only, not quote-time-matched raw histories"))
    push!(inventory,(;field="inversion_loss",source="Q/rates_<league>.csv:kl",present=true,unit="KL",coverage="all11; saved inversion attempts",limitation="not football truth or measured market liquidity"))
    push!(inventory,(;field="selection_count",source="Q/rates_<league>.csv:n_selections",present=true,unit="count",coverage="all11; saved inversion attempts",limitation="panel requires accepted and >=5; reported accepted-panel summary"))
    push!(inventory,(;field="market_family_availability",source="Q/rates_<league>.csv:lines_quoted,n_markets,has_1x2; coverage:line_mix",present=true,unit="families/lines/count",coverage="all11",limitation="presence only, not age/synchrony/depth"))
    @assert all(any(r->r.snapshot==snapshot&&r.table=="betfair_odds"&&r.column=="timestamp",columnrows) for snapshot in seen)
    archive=[r for r in columnrows if r.table=="betfair_odds"&&r.column=="timestamp"]
    archive_coverage=join([r.snapshot*":n_rows=$(r.n_rows)" for r in archive],";")
    archive_present=any(r->r.n_rows>0,archive)
    push!(inventory,(;field="archive_trade_timestamps",source="pinned DataStore.betfair_odds:timestamp,minutes_to_kickoff",present=archive_present,unit="UTC DateTime/minutes",coverage=archive_coverage,limitation="archived traded-price sample timestamps, not executable quote ages"))
    push!(inventory,(;field="market_family_trade_price_history",source="pinned DataStore.betfair_odds:market_name,market_line,selection,traded_price,timestamp",present=archive_present,unit="decimal traded price by sampled timestamp",coverage=archive_coverage,limitation="sampled trade-price histories can be inventoried; no bid/ask quote-event history or order-book age, so quote-level asynchronous repricing is not identified"))
    teamids=[r for r in columnrows if r.column in ("team_id","home_team_id","away_team_id","sofascore_team_id")]
    idreason=isempty(teamids) ? "no verified stable team-ID columns in cached tables; team slugs not independently verified, no fuzzy joins" : "ID fields exist outside the panel; no validated role-to-stable-ID identity join available in this batch"
    push!(inventory,(;field="stable_team_id",source="all cached tables in snapshot_columns.csv",present=!isempty(teamids),unit="provider identity if present",coverage=isempty(teamids) ? "none" : join(unique([r.snapshot*"/"*r.table*"/"*r.column for r in teamids]),";"),limitation=idreason))
    for field in ("quote_timestamps_or_age","quote_update_history","exchange_depth","exchange_bid_ask_spread")
        push!(inventory,(;field,source="saved rates/configs/coverage and snapshot_columns.csv",present=false,unit="NOT_AVAILABLE",coverage="no verified field with these semantics",limitation="fixture date and inversion start_spread are NOT quote age/bid-ask spread; asynchronous quote repricing NOT_IDENTIFIABLE"))
    end
    output("R08","field_inventory.csv",inventory)
    coverage=CSV.read(joinpath(Q,"coverage_by_league.csv"),DF.DataFrame;stringtype=String)
    goal=CSV.read(joinpath(OUT,"goal_ablation_summary.csv"),DF.DataFrame;stringtype=String)
    fieldvalues=Dict{String,Vector{Float64}}(f=>Float64[] for f in ("overround","inversion_loss","selection_count","market_family_availability"))
    for (li,(league,_,_)) in enumerate(QS.LEAGUES)
        rates=CSV.read(joinpath(Q,"rates_$league.csv"),DF.DataFrame;stringtype=String)
        accepted=DF.filter(r->r.accepted&&r.n_selections>=5,rates)
        cov=unique(DF.filter(:league=>==(league),coverage),:season)
        weights=Float64.(cov.n_raw_quoted);weights./=sum(weights)
        values=Dict("overround"=>sum(weights.*cov.overround_median),"inversion_loss"=>ST.median(accepted.kl),
            "selection_count"=>ST.mean(accepted.n_selections),"market_family_availability"=>ST.mean(accepted.n_markets))
        style=only(eachrow(DF.filter(r->r.league==league&&r.method=="mixture_128x4"&&r.comparison=="full_minus_no_style"&&r.channel=="joint"&&r.block_weeks==8,goal)))
        for field in keys(fieldvalues);push!(fieldvalues[field],values[field]);end
        for protocol in PROTOCOLS
            z=structures[(protocol,"C0")]
            for field in sort(collect(keys(values)))
                push!(confounding,(;league,protocol,r_median=exp(ST.median(z.R[li,:])),sigma_obs_median=ST.median(z.O[li,:]),field,n=DF.nrow(accepted),field_summary=values[field],style_goal_delta=style.mean_delta,
                    status=field=="overround" ? "weighted mean of season medians; NOT_IDENTIFIABLE causally; goals="*style.status : "accepted-panel descriptive summary; NOT_IDENTIFIABLE causally; goals="*style.status))
            end
        end
    end
    for protocol in PROTOCOLS,field in sort(collect(keys(fieldvalues)))
        x=vec(ST.median(structures[(protocol,"C0")].R;dims=2));y=fieldvalues[field]
        loo=[spearman(x[setdiff(1:11,[i])],y[setdiff(1:11,[i])]) for i in 1:11]
        push!(fieldcorr,(;protocol,rung="C0",field,n=11,spearman=spearman(x,y),leave_one_out_min=minimum(loo),leave_one_out_max=maximum(loo),interpretation="descriptive n11; no sharpness index, causal regression or fitted pooling"))
    end
    output("R08","market_confounding.csv",confounding);output("R08","market_field_correlations.csv",fieldcorr)
    # Cannot independently verify names as identities without stable IDs/catalogue in these snapshots.
    movers=DF.DataFrame(country=String[],team_id=String[],from_league=String[],to_league=String[],from_season=String[],to_season=String[],quantity=String[],q05=Float64[],median=Float64[],q95=Float64[],status=String[])
    output("R08","division_movers.csv",movers)
    output("R08","mover_summary.csv",[(;country,n=0,quantity="verified_movers",value=missing,status="NOT_AVAILABLE: "*idreason) for country in ("England","Scotland","Ireland")])
    lines=["PARTIAL. [league_structure.csv](league_structure.csv), [field_inventory.csv](field_inventory.csv), [market_confounding.csv](market_confounding.csv), [market_field_correlations.csv](market_field_correlations.csv), [mover_summary.csv](mover_summary.csv). 12000 independent matched posterior permutations/league; uncertainty is fit uncertainty for these11, not population sampling."]
    for protocol in PROTOCOLS
        g=filter(r->r.rung=="C0"&&r.protocol==protocol&&r.quantity in ("SD_log_r","OLS_log_r_on_tier","England_tier_slope","Scotland_tier_slope","Ireland_tier_slope"),rows)
        push!(lines,"C0/$protocol "*join(["$(r.quantity)=$(round(r.median;digits=3)) [$(round(r.q05;digits=3)),$(round(r.q95;digits=3))]; LOO=$(ismissing(r.leave_one_out_min) ? "NA" : round(r.leave_one_out_min;digits=3)):$(ismissing(r.leave_one_out_max) ? "NA" : round(r.leave_one_out_max;digits=3))" for r in g],"; ")*".")
    end
    push!(lines,"Saved overround quantiles, inversion KL, selection/market counts, family presence AND archived trade-price timestamps/histories are available; overround aggregation is explicitly a weighted mean of season medians. Archive trade samples are not executable quote ages/update events; start_spread is optimizer dispersion, not bid-ask spread. σ_obs is not liquidity.")
    push!(lines,"Quote-level asynchronous repricing is NOT_IDENTIFIABLE without executable quote ages/update histories; archived trade-price timestamps exist but are not a timed/quality-matched quote test. No causal sharpness/football separation. Movers: $idreason; zero independently verified records, NOT_AVAILABLE, not evidence of no movers. No absolute division offset or <5-mover correlation.")
    summary!("R08",lines);verification("R08","PASS all12000 theta draws retained per fit; fixed independent league permutations preserve matched r/σ_obs; available field units/limits documented, no network or SQL. Movers explicitly NOT_AVAILABLE. Runtime=$(time()-started)s; invocation Q07.r08().");flush_manifest!()
end

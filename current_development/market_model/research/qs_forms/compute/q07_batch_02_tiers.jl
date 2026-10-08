# R12: largest-gap partitions, oriented by low/high q, never fitted mixtures.
function tier(q,teams)
    order=sortperm(q);gaps=diff(q[order]);j=argmax(gaps);iqr=ST.quantile(q,.75)-ST.quantile(q,.25)
    upper=Set(String.(teams[order[j+1:end]]));allteams=Set(String.(teams))
    (;gap=iqr>0 ? gaps[j]/iqr : missing,upper,allteams,lower_n=j,upper_n=length(q)-j,named=upper==Set(["celtic","rangers"]))
end
choose2(n)=n*(n-1)/2
function persistence(a,b)
    shared=intersect(a.allteams,b.allteams);n=length(shared)
    if n==0;return (;n,same=missing,oriented_same=missing,jaccard=missing,ARI=missing,status="NOT_IDENTIFIABLE: no shared teams");end
    ua,ub=intersect(a.upper,shared),intersect(b.upper,shared)
    fraction=ST.mean((t in ua)==(t in ub) for t in shared)
    same=max(fraction,1-fraction)
    if isempty(ua)||isempty(ub)||length(ua)==n||length(ub)==n
        return (;n,same,oriented_same=fraction,jaccard=missing,ARI=missing,status="NOT_IDENTIFIABLE: trivial shared partition; no persistence award")
    end
    cross=[count(t->(t in ua)==i&&(t in ub)==j,shared) for i in (false,true),j in (false,true)]
    S=sum(choose2.(cross));AA=sum(choose2.(sum(cross;dims=2)));BB=sum(choose2.(sum(cross;dims=1)));Z=choose2(n)
    den=(AA+BB)/2-AA*BB/Z
    ari=den==0 ? missing : (S-AA*BB/Z)/den
    (;n,same,oriented_same=fraction,jaccard=length(intersect(ua,ub))/length(union(ua,ub)),ARI=ari,status=ismissing(ari) ? "NOT_IDENTIFIABLE: zero ARI denominator" : "OK")
end
function point_tiers(p,X,seasons,window,cutoff)
    out=Dict{String,Any}()
    for season in seasons
        d=A.season_design(p,season);weeks=max(first(d.weeks),window=="null_suffix" ? cutoff+1 : 1):last(d.weeks)
        isempty(weeks)&&continue
        d=(;d.active,d.ix,weeks);z=rawaxes(p,X,d).q
        q=vec(ST.mean(z.-ST.mean(z;dims=1);dims=2))
        out[season]=tier(q,d.active)
    end
    out
end
function joint_tier_count(series,seasons)
    count,eligible=0,0
    for i in 1:length(seasons)-1
        s,t=seasons[i:i+1];haskey(series,s)&&haskey(series,t)||continue
        v=persistence(series[s],series[t]);v.status=="OK"||continue
        eligible+=1
        !ismissing(series[s].gap)&&series[s].gap>=2&&v.jaccard>=.8&&(count+=1)
    end
    (;count,eligible)
end
function r12()
    league="sco_premiership";p,config=panel(league);sim=nulls(league)
    seasons=config.seasons
    # No new paths: this table contains full-season means only, not weekly FFBS arrays.
    means=loadcsv("team_season_path_means.csv";request="R12")
    g=DF.filter(r->r.league==league&&r.rung in ("C0","R6")&&r.gauge=="active_cohort",means)
    old=loadcsv("tier_gaps.csv";request="R12");windows=loadcsv("season_windows.csv";request="R12")
    membership,rows,audit=NamedTuple[],NamedTuple[],NamedTuple[]
    function add(rung,season_or_pair,window,quantity,n_shared,values,nullvalues;status="OK",point=false)
        qs=A.quant(values);nv=length(nullvalues)==99 ? envelope(nullvalues,point ? first(values) : qs[2]) : envelope(Any[],missing)
        push!(rows,(;rung,season_or_pair,window,quantity,n_shared,q05=qs[1],median=qs[2],q95=qs[3],nv...,status=point ? status*"; RTS point, conditional-null rank" : status*"; posterior interval, no null rank"))
    end
    for season in seasons
        w=only(eachrow(DF.filter(r->r.league==league&&r.season==season,windows)))
        matches=DF.filter(:season=>==(season),p.matches)
        push!(audit,(;season,first_date=w.first_date,last_date=w.last_date,n_fixtures=DF.nrow(matches),n_weeks=w.n_weeks,partial_label=w.label,cohort_n=w.n_teams,status="Existing accepted-panel coverage; label does not define an executable complete-season mask"))
    end
    for rung in ("C0","R6")
        posterior=Dict{String,Vector{Any}}()
        for season in seasons
            sg=DF.filter(r->r.rung==rung&&r.season==season,g)
            draws=sort(unique(sg.draw));@assert draws==collect(1:128)
            d=A.season_design(p,season)
            values=Any[]
            for j in draws
                dd=DF.filter(:draw=>==(j),sg);@assert Set(dd.team)==Set(d.active)&&DF.nrow(dd)==length(d.active)
                push!(values,tier(dd.q,dd.team))
            end
            posterior[season]=values
            named=ST.mean(v.named for v in values)
            for team in d.active
                push!(membership,(;rung,season,window="full_season",team,prob_upper_group=ST.mean(team in v.upper for v in values),prob_named_pair_only=named,n_draws=128,status="OK; named Celtic/Rangers contrast post-selected from R06"))
                samewindow=first(d.weeks)>sim.cutoff
                push!(membership,(;rung,season,window="null_suffix",team,prob_upper_group=samewindow ? ST.mean(team in v.upper for v in values) : missing,prob_named_pair_only=samewindow ? named : missing,n_draws=samewindow ? 128 : 0,status=samewindow ? "OK; exact suffix=full-season window; existing128 path means reused" : "NOT_AVAILABLE: first-season suffix cuts saved full-season FFBS means; no weekly paths or resimulation"))
            end
            for quantity in ("gap","lower_n","upper_n","named")
                add(rung,season,"full_season","posterior_"*quantity,length(d.active),[getproperty(v,Symbol(quantity)) for v in values],Any[])
                if first(d.weeks)>sim.cutoff
                    add(rung,season,"null_suffix","posterior_"*quantity,length(d.active),[getproperty(v,Symbol(quantity)) for v in values],Any[];status="OK; exact suffix=full-season mask")
                else
                    add(rung,season,"null_suffix","posterior_"*quantity,length(d.active),[missing],Any[];status="NOT_AVAILABLE: first-season truncated FFBS mean not cached")
                end
            end
        end
        for i in 1:length(seasons)-1
            s,t=seasons[i:i+1];vals=[persistence(posterior[s][j],posterior[t][j]) for j in 1:128]
            nontrivial=count(v->v.status=="OK",vals)
            for quantity in ("same","oriented_same","jaccard","ARI")
                add(rung,"$s->$t","full_season","posterior_"*quantity,first(vals).n,[getproperty(v,Symbol(quantity)) for v in vals],Any[];status="$(nontrivial)/128 nontrivial partitions; trivial partitions excluded from Jaccard/ARI")
                samewindows=first(A.season_design(p,s).weeks)>sim.cutoff&&first(A.season_design(p,t).weeks)>sim.cutoff
                add(rung,"$s->$t","null_suffix","posterior_"*quantity,first(vals).n,samewindows ? [getproperty(v,Symbol(quantity)) for v in vals] : [missing],Any[];status=samewindows ? "OK; both suffix masks equal cached full-season masks" : "NOT_AVAILABLE: first-season posterior suffix mean not cached")
            end
        end
        for window in ("full_season","null_suffix")
            observed=point_tiers(p,point(league,rung).X,seasons,window,sim.cutoff)
            ns=rung=="C0" ? [point_tiers(p,X,seasons,window,sim.cutoff) for X in sim.paths] : Any[]
            for season in seasons
                haskey(observed,season)||continue;v=observed[season]
                ref=only(eachrow(DF.filter(r->r.league==league&&r.rung==rung&&r.season==season&&r.window==window,old)))
                @assert abs(v.gap-ref.gap_over_iqr)<=1e-10
                for quantity in ("gap","lower_n","upper_n","named")
                    add(rung,season,window,"point_"*quantity,length(v.allteams),[getproperty(v,Symbol(quantity))],[getproperty(x[season],Symbol(quantity)) for x in ns];point=true,status="OK; identical largest-gap rule/window/gauge in data/null")
                end
            end
            for i in 1:length(seasons)-1
                s,t=seasons[i:i+1];haskey(observed,s)&&haskey(observed,t)||continue
                v=persistence(observed[s],observed[t]);nv=[persistence(x[s],x[t]) for x in ns]
                ref=only(eachrow(DF.filter(r->r.league==league&&r.rung==rung&&r.season==s&&r.window==window,old)))
                @assert abs(v.same-ref.same_side_fraction)<=1e-10
                for quantity in ("same","oriented_same","jaccard","ARI")
                    add(rung,"$s->$t",window,"point_"*quantity,v.n,[getproperty(v,Symbol(quantity))],[getproperty(x,Symbol(quantity)) for x in nv];point=true,status=v.status*"; null rank blank if fewer than99 defined null values")
                end
            end
            v=joint_tier_count(observed,seasons);nv=[joint_tier_count(x,seasons) for x in ns]
            add(rung,"ALL_WINDOWS",window,"count_gap_ge2_next_Jaccard_ge0p8",v.eligible,[Float64(v.count)],[Float64(x.count) for x in nv];point=true,status="OK; n_shared denotes eligible nontrivial transitions")
            add(rung,"ALL_WINDOWS",window,"eligible_transition_denominator",v.eligible,[Float64(v.eligible)],[Float64(x.eligible) for x in nv];point=true)
            add(rung,"EXISTING_COMPLETE_WINDOWS",window,"count_gap_ge2_next_Jaccard_ge0p8",0,[missing],Any[];status="NOT_AVAILABLE: A1 windows have no complete-season flag/mask; no new favorable subset invented",point=true)
        end
    end
    output("R12","scottish_tier_membership.csv",membership)
    output("R12","scottish_tier_robustness.csv",rows)
    output("R12","tier_window_audit.csv",audit)
    lines=["PARTIAL: scottish_tier_membership.csv, scottish_tier_robustness.csv, tier_window_audit.csv. Exactly128 full-season path means/rung; suffix posterior means reused where suffix=full window (22/23–25/26). Only21/22 suffix posterior unavailable: weekly FFBS paths not saved. Median RTS full/suffix and99 existing C0 nulls matched exactly; old gaps/persistence reproduce≤1e−10."]
    for rung in ("C0","R6")
        g=[only(unique([r.prob_named_pair_only for r in membership if r.rung==rung&&r.season==season&&r.window=="full_season"])) for season in seasons]
        push!(lines,"$rung P(upper group exactly Celtic+Rangers), post-selected: "*join(["$(season) $(prob)" for (season,prob) in zip(seasons,g)],"; ")*"; membership probabilities and group-size/gap quantiles in tables.")
    end
    g=filter(r->r.rung=="C0"&&r.window=="full_season"&&r.quantity in ("posterior_jaccard","posterior_ARI"),rows)
    push!(lines,"C0 shared-team persistence posterior median[5%,95%]: "*join(["$(r.season_or_pair) $(r.quantity) $(r.median)[$(r.q05),$(r.q95)]" for r in g],"; ")*". Same-side retained only as original majority statistic; trivial shared partitions not rewarded.")
    for window in ("full_season","null_suffix")
        r=only(filter(r->r.rung=="C0"&&r.window==window&&r.quantity=="count_gap_ge2_next_Jaccard_ge0p8"&&r.season_or_pair=="ALL_WINDOWS",rows))
        push!(lines,"C0 $window joint gap≥2/next-Jaccard≥.8 count=$(r.median), eligible transitions=$(r.n_shared), null90%=[$(r.null_q05),$(r.null_q95)], rank=$(r.null_rank); dependent seasons not multiplied as independent evidence.")
    end
    a=only(filter(r->r.season=="25/26",audit))
    push!(lines,"25/26 coverage $(a.first_date) to $(a.last_date), fixtures=$(a.n_fixtures), weeks=$(a.n_weeks), roster=$(a.cohort_n); existing label: $(a.partial_label). Gap change cannot be separated from incomplete coverage here.")
    push!(lines,"Existing-complete-window sensitivity NOT_AVAILABLE: no executable complete-season flag in A1 season_windows or tier_gaps; exact all-window masks retained. No fresh EPL clustering, mixture optimization or mover joining; posterior suffix not replaced by fresh paths.")
    push!(lines,"Conditional Gaussian-RW null is not a test against every unimodal heavy-tail population. Local tier/heavy-tail level-prior nomination remains descriptive/post-selected, not a universal mixture, changed axes or demonstrated prospective goal benefit.")
    summary!("R12",lines)
    verify("R12","PASS128 full-season means/draw IDs per C0/R6 season, active-cohort team sets match; original largest gap/IQR and aligned same-side persistence reproduce≤1e−10; low/high orientation fixed, Jaccard/ARI calculated only nontrivial shared partitions; same point/null full and suffix windows;99 C0 panels reused. Later-season suffix posterior reuses identical full-season masks/means; only first-season truncated posterior suffix and complete-window mask unavailable, explicitly reported rather than regenerated.")
end

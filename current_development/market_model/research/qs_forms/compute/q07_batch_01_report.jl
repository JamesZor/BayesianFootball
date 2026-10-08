# Included inside QSFormsBatch01. Compact report amendments from already computed tables.
function insert_summary_lines!(request,lines)
    path=joinpath(OUT,"SUMMARY.md");text=read(path,String)
    marker="## $request —\n";parts=split(text,marker;limit=2);@assert length(parts)==2
    section=split(parts[2],"\n## ";limit=2)
    existing=filter(!isempty,split(strip(section[1]),'\n'))
    @assert length(existing)+length(lines)<=10
    new=parts[1]*marker*join(vcat(existing,lines),"\n")*"\n"
    length(section)==2&&(new*="\n## "*section[2])
    write(path,new)
end
function report_amendments()
    geo=CSV.read(joinpath(OUT,"geometry_summary.csv"),DF.DataFrame;stringtype=String)
    rows=NamedTuple[]
    for (league,_,_) in QS.LEAGUES,protocol in PROTOCOLS
        c0=only(geo.median[(geo.league.==league).&(geo.rung.=="C0").&(geo.protocol.==protocol).&(geo.quantity.=="Vs")])
        r6=only(geo.median[(geo.league.==league).&(geo.rung.=="R6").&(geo.protocol.==protocol).&(geo.quantity.=="l_minus")])
        push!(rows,(;league,protocol,C0_style_variance_median=c0,R6_minor_variance_median=r6,reduction=c0-r6,relative_reduction=1-r6/c0,method="difference/ratio of posterior medians across existing independent fits; not a draw-paired interval or forecast gain"))
    end
    result=output("R01","geometry_residual_comparison.csv",rows);g=result[argmax(result.relative_reduction),:]
    insert_summary_lines!("R01",["Direct R6–C0 residual comparison (medians): largest 1−R6_minor/C0_Vs=$(round(g.relative_reduction;digits=4)) at $(g.league)/$(g.protocol), C0_Vs=$(g.C0_style_variance_median), R6_minor=$(g.R6_minor_variance_median); [geometry_residual_comparison.csv](geometry_residual_comparison.csv). This is distinct from within-R6 rotation gain."])
    levels=CSV.read(joinpath(OUT,"level_geometry.csv"),DF.DataFrame;stringtype=String)
    ldraws=CSV.read(joinpath(OUT,"level_geometry_draws.csv"),DF.DataFrame;stringtype=String)
    # Retain all degenerate rows, but distinguish their unstable shortcut from ordinary roster geometry.
    qualified=DF.innerjoin(ldraws,unique(DF.select(levels,:league,:rung,:season,:n_teams));on=[:league,:rung,:season])
    err=coalesce.(abs.(qualified.shortcut_r-qualified.r_level),-Inf)
    eligible=findall(qualified.n_teams.>=8);i=eligible[argmax(err[eligible])];g=qualified[i,:]
    sensitivity=NamedTuple[]
    for window in eachrow(unique(DF.select(levels,:league,:season,:n_teams)))
        at(rung,method,quantity)=only(levels.median[(levels.league.==window.league).&(levels.season.==window.season).&(levels.rung.==rung).&(levels.method.==method).&(levels.quantity.==quantity).&(levels.gauge.=="full_roster")])
        push!(sensitivity,(;window.league,window.season,window.n_teams,C0_level_step_median=at("C0","FFBS","level_step_ratio"),
            R2_minus_C0_rlevel_median=at("R2","FFBS","r_level")-at("C0","FFBS","r_level"),
            R6_minus_C0_rlevel_median=at("R6","FFBS","r_level")-at("C0","FFBS","r_level"),
            C0_FFBS_minus_RTS_rlevel=at("C0","FFBS","r_level")-at("C0","RTS_point","r_level")))
    end
    sens=output("R02","level_sensitivity.csv",sensitivity)
    a=sens[argmax(abs.(sens.R2_minus_C0_rlevel_median)),:];b=sens[argmax(abs.(sens.C0_FFBS_minus_RTS_rlevel)),:]
    c=sens[argmax(abs.(log.(sens.C0_level_step_median))),:]
    insert_summary_lines!("R02",["The all-row shortcut maximum above comes from Scottish Championship21/22 (2 teams,1 week): near-rank-one rho≈1 makes that shortcut NOT_IDENTIFIABLE as an ellipse scale. No rows were discarded. Largest error among ≥8-team windows: $(g.league)/$(g.rung)/$(g.season) $(round(err[i];digits=4)).",
        "Largest R2−C0 median r_level: $(a.league)/$(a.season) $(round(a.R2_minus_C0_rlevel_median;digits=4)); C0 FFBS−RTS: $(b.league)/$(b.season) $(round(b.C0_FFBS_minus_RTS_rlevel;digits=4)); strongest C0 median log-distance level/step: $(c.league)/$(c.season) ratio=$(round(c.C0_level_step_median;digits=4)); [level_sensitivity.csv](level_sensitivity.csv). These are finite latent-population screens, not calibrated OU half-lives."])
    comp=CSV.read(joinpath(OUT,"joint_comparison.csv"),DF.DataFrame;stringtype=String)
    disagreements=NamedTuple[]
    for league in [String(e[1]) for e in QS.LEAGUES]
        j=only(comp.mean_C0_minus_R6[(comp.league.==league).&(comp.block_weeks.==8).&(comp.score_kind.=="joint")])
        m=only(comp.mean_C0_minus_R6[(comp.league.==league).&(comp.block_weeks.==8).&(comp.score_kind.=="marginal_sum")])
        push!(disagreements,(;league,joint_delta=j,marginal_sum_delta=m,joint_minus_marginal_delta=j-m))
    end
    disag=output("R03","joint_marginal_disagreement.csv",disagreements);g=disag[argmax(abs.(disag.joint_minus_marginal_delta)),:]
    venue=CSV.read(joinpath(OUT,"venue_contrasts.csv"),DF.DataFrame;stringtype=String)
    vrows=NamedTuple[]
    for group in DF.groupby(venue,[:league,:rung,:axis])
        z=[r.contrast/r.bootstrap_se for r in eachrow(group) if r.bootstrap_se>0]
        qs=quant(z)
        push!(vrows,(;league=first(group.league),rung=first(group.rung),axis=first(group.axis),n=length(z),q05=qs[1],median=qs[2],q95=qs[3],method="descriptive standardized contrast distribution, not multiplicity-adjusted significance"))
    end
    output("R03","venue_standardized_distribution.csv",vrows)
    diag=CSV.read(joinpath(OUT,"forecast_diagnostics.csv"),DF.DataFrame;stringtype=String)
    signed=DF.filter(r->r.rung=="C0"&&startswith(r.statistic,"weekly_signed")&&!ismissing(r.value),diag)
    s=signed[argmax(abs.(signed.value)),:]
    insert_summary_lines!("R03",["Largest joint/marginal Δ disagreement: $(g.league), joint $(round(g.joint_delta;digits=5)), marginal-sum $(round(g.marginal_sum_delta;digits=5)); [joint_marginal_disagreement.csv](joint_marginal_disagreement.csv). Both pooled8-week joint intervals lie within ±0.005: practical equivalence at the packet tolerance, not exact equality.",
        "Largest |C0 signed-week lag|: $(s.league)/$(s.axis)/$(s.statistic) $(round(s.value;digits=3)) [$(round(s.boot_q05;digits=3)),$(round(s.boot_q95;digits=3))]; selected maximum, not adjusted evidence. Unselected venue contrast/SE q05/median/q95: [venue_standardized_distribution.csv](venue_standardized_distribution.csv)."])
    flush_manifest!()
end
function additional_report()
    nonlinear_table=CSV.read(joinpath(OUT,"nonlinear_levels.csv"),DF.DataFrame;stringtype=String)
    candidates=DF.filter(r->r.rung=="C0"&&r.method=="FFBS"&&r.quantity=="c",nonlinear_table)
    g=candidates[argmax(abs.(candidates.median)),:]
    at(quantity)=only(eachrow(DF.filter(r->r.league==g.league&&r.rung==g.rung&&r.season==g.season&&r.method==g.method&&r.quantity==quantity,nonlinear_table)))
    a=at("LOTO_quadratic_minus_linear");b=at("LOTO_quadratic_minus_intercept")
    insert_summary_lines!("R06",["Selected largest |C0 FFBS curvature|: $(g.league)/$(g.season), c=$(round(g.median;digits=4)) [$(round(g.q05;digits=4)),$(round(g.q95;digits=4))]; LOTO quadratic−linear=$(round(a.median;digits=5)) [$(round(a.q05;digits=5)),$(round(a.q95;digits=5))], quadratic−horizontal=$(round(b.median;digits=5)) [$(round(b.q05;digits=5)),$(round(b.q95;digits=5))]. Descriptive selected maximum, not model-selection evidence."])
    tiers=CSV.read(joinpath(OUT,"tier_gaps.csv"),DF.DataFrame;stringtype=String)
    ti=DF.filter(r->r.rung=="C0"&&!ismissing(r.null_rank),tiers)
    g=ti[argmax(ti.gap_over_iqr),:]
    insert_summary_lines!("R06",["Largest matched C0 suffix gap/IQR: $(g.league)/$(g.season) $(round(g.gap_over_iqr;digits=3)), gap rank=$(g.null_rank); next-season shared n=$(g.next_season_shared_n), same-side fraction=$(g.same_side_fraction), rank=$(g.null_same_side_rank). Rank includes the same largest-gap search under each null; no Gaussian-mixture fitting."])
    goals=CSV.read(joinpath(OUT,"goal_ablation_summary.csv"),DF.DataFrame;stringtype=String)
    quality=DF.filter(r->r.league=="ALL"&&r.weighting=="fixture"&&r.method=="mixture_128x4"&&r.comparison=="full_minus_no_quality"&&r.block_weeks==8,goals)
    insert_summary_lines!("R07",["Quality control, pooled fixture-weighted mixture4: "*join(["$(r.channel) Δ=$(round(r.mean_delta;digits=5)) [$(round(r.boot_q05;digits=5)),$(round(r.boot_q95;digits=5))] $(r.status)" for r in eachrow(quality)],"; ")*". This is deletion arithmetic, not a refitted quality-only/style-only comparison."])
end

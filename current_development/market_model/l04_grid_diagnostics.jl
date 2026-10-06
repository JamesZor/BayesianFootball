# Included in CopulaGridMarket; readable stages called by r04.

function compare_grids(ds,rates,books,grids,frames,out)
    fitrows = NamedTuple[]
    residuals,heldout,bias,losses,shifts = DF.DataFrame[],DF.DataFrame[],NamedTuple[],DF.DataFrame[],DF.DataFrame[]
    baseline = Dict(r.match_id=>r for r in eachrow(frames[0]))
    for kind in 0:3
        g,frame = grids[kind],frames[kind]
        total = sum(frame.kl)
        push!(fitrows,(; grid=grid_name(g),parameter=g.parameter,n=DF.nrow(frame),total_kl=total,
            mean_kl=ST.mean(frame.kl),kl_q05=ST.quantile(frame.kl,0.05),kl_q50=ST.median(frame.kl),
            kl_q95=ST.quantile(frame.kl,0.95),max_kl=maximum(frame.kl),max_spread=maximum(frame.start_spread),
            n_gate_pass=count(frame.gate_pass)))
        raw = grid_residual_rows(g,frame,rates,books)
        held = grid_residual_rows(g,frame,rates,books; heldout=true)
        push!(residuals,raw)
        push!(heldout,held)
        push!(bias,one_x2_bias(g,frame,rates,books))
        loss,lossraw = outcome_loss(g,frame,rates,books,ds,baseline)
        push!(losses,loss)
        shift,shiftraw = rate_shift(g,frame,baseline)
        push!(shifts,shift)
        CSV.write(joinpath(out,"rates_$(grid_name(g)).csv"),frame)
        CSV.write(joinpath(out,"outcome_fixture_$(grid_name(g)).csv"),lossraw)
        CSV.write(joinpath(out,"rate_shift_fixture_$(grid_name(g)).csv"),shiftraw)
    end
    residual = line_summaries(vcat(residuals...))
    held = line_summaries(vcat(heldout...); heldout=true)
    b = DF.DataFrame(bias)
    CSV.write(joinpath(out,"grid_fit.csv"),DF.DataFrame(fitrows))
    CSV.write(joinpath(out,"grid_selection_residuals.csv"),vcat(residuals...))
    CSV.write(joinpath(out,"grid_heldout_selections.csv"),vcat(heldout...))
    CSV.write(joinpath(out,"grid_line_residuals.csv"),residual)
    CSV.write(joinpath(out,"grid_heldout.csv"),held)
    CSV.write(joinpath(out,"grid_1x2only_bias.csv"),b)
    CSV.write(joinpath(out,"grid_outcome_logloss.csv"),vcat(losses...))
    CSV.write(joinpath(out,"grid_rate_shift.csv"),vcat(shifts...))
    pooled = DF.filter(r->r.line == "ALL",held)
    winner = pooled.grid[argmin(pooled.mean_abs)]
    return (; winner=parse(Int,winner[2:end]),heldout=held,residuals=residual,bias=b)
end

function b3_figures(P,profiles,comparison,tailraw,fig)
    p = P.plot(layout=(1,3),size=(1200,420),bottom_margin=8P.mm,left_margin=8P.mm)
    for (j,kind) in enumerate(1:3)
        g = DF.sort(DF.filter(r->r.grid == "G$kind" && isfinite(r.total_kl),profiles),:parameter)
        P.plot!(p[j],g.parameter,g.total_kl; label="all accepted books",xlabel="global dependence parameter",
            ylabel="summed KL",title="G$kind profile",marker=:circle,markersize=2)
    end
    P.savefig(p,joinpath(fig,"B3_grid_profile.png"))
    p = P.plot(layout=(1,2),size=(1100,440),bottom_margin=8P.mm,left_margin=8P.mm)
    for (j,(line,selection)) in enumerate((("1X2","draw"),("BTTS","btts_yes")))
        g = DF.filter(r->r.tournament == 0 && r.n_markets == 0 && r.line == line && r.selection == selection,comparison.residuals)
        P.scatter!(p[j],g.grid,g.mean; yerror=(g.mean-g.ci_low,g.ci_high-g.mean),label="full-book 95% CI",
            ylabel="grid minus close probability",title="$line $selection")
        if line == "BTTS"
            h = DF.filter(r->r.tournament == 0 && r.n_markets == 0 && r.line == line && r.selection == selection,comparison.heldout)
            P.scatter!(p[j],h.grid,h.mean; yerror=(h.mean-h.ci_low,h.ci_high-h.mean),label="heldout 95% CI")
        end
        P.hline!(p[j],[0.0]; color=:black,label="",linestyle=:dash)
    end
    P.savefig(p,joinpath(fig,"B3_grid_residuals.png"))
    names = ["kurtosis_quality","kurtosis_style","kendall_alpha_beta","joint_improvement_95",
        "joint_collapse_95","lag1_squared_quality"]
    p = P.plot(layout=(2,3),size=(1250,740),bottom_margin=8P.mm,left_margin=8P.mm)
    for (j,name) in enumerate(names)
        g = DF.filter(r->r.protocol == "10a" && r.statistic == name,tailraw)
        P.histogram!(p[j],g.replicated; alpha=0.5,bins=20,label="refiltered replicate",title=name)
        P.histogram!(p[j],g.observed; alpha=0.5,bins=20,label="observed FFBS")
        P.xlabel!(p[j],"statistic")
    end
    P.savefig(p,joinpath(fig,"B3_tail_ppc.png"))
    return nothing
end

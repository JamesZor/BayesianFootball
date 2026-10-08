# Cross-league quality/style EDA: aggregate 11 accepted league runs and plot results.
# No model training, SQL connection, data refetch or post-hoc parameter tuning.
# Run after every fit has passed convergence, in an owned persistent beast REPL.
# The verdict uses 10a full-panel posterior geometry and honest 10b prediction.

# %%
# ===================================================================
# 1. Tables, implementation, deterministic headless figures
# ===================================================================
using CSV, DataFrames, Statistics
ENV["GKSwstype"] = "100"
using Plots
if !isdefined(@__MODULE__, :QualityStyleReport)
    include(joinpath(@__DIR__,"l06_qs_report.jl"))
end
const Q06R = QualityStyleReport
const Q06R_OUT = joinpath(@__DIR__,"results","QS")
const Q06R_FIG = joinpath(Q06R_OUT,"figures")
mkpath(Q06R_FIG)

# %%
# ===================================================================
# 2. Completeness gates, fixed verdicts, goal-pairing and original fixtures
# ===================================================================
q06r_tables = Q06R.collect_results(Q06R_OUT)
for (key,filename) in (("parameters","parameters_by_league.csv"),
    ("levels","levels_by_league.csv"),("level_teams","level_teams.csv"),
    ("paired_scores","paired_scores.csv"),("forecast_coverage","forecast_coverage.csv"),
    ("forecast_fixtures","forecast_fixtures.csv"),
    ("identifiability","identifiability_by_league.csv"),
    ("goal_fixtures","goal_fixtures.csv"))
    CSV.write(joinpath(Q06R_OUT,filename),q06r_tables[key])
end
q06r_verdict = Q06R.verdicts(q06r_tables["parameters"],q06r_tables["paired_scores"])
CSV.write(joinpath(Q06R_OUT,"verdict.csv"),q06r_verdict)
q06r_goals = Q06R.goal_summary(q06r_tables["goal_fixtures"])
CSV.write(joinpath(Q06R_OUT,"goal_check.csv"),q06r_goals)
recovery = DataFrame[]
for league in ("eng_premier","sco_league_two")
    path = joinpath(Q06R_OUT,"fits",league,"c0_recovery.csv")
    isfile(path) || error("missing 3-seed C0 synthetic recovery: $league")
    push!(recovery,CSV.read(path,DataFrame))
end
q06r_recovery = vcat(recovery...)
nrow(q06r_recovery) == 18 || error("expected 2 leagues x 3 seeds x 3 parameters")
CSV.write(joinpath(Q06R_OUT,"c0_recovery.csv"),q06r_recovery)

# %%
# ===================================================================
# 3. Forest plot: full-panel C0 r and R6 innovation rho_ab
# ===================================================================
q06r_names = collect(Q06R.LEAGUES)
q06r_params = q06r_tables["parameters"]
function q06r_parameter(league,rung,name)
    rows = filter(r -> r.league==league && r.rung==rung && r.protocol=="10a" &&
        r.parameter==name,q06r_params)
    nrow(rows)==1 || error("missing plot parameter $league $rung $name")
    return only(eachrow(rows))
end
q06r_plot = plot(layout=(1,2),size=(1450,690),left_margin=24 * Plots.mm,bottom_margin=9 * Plots.mm)
for (axis,rung,parameter,label,baseline) in ((1,"C0","r","style/quality innovation SD ratio",0.3153399879334282),
    (2,"R6","rho_ab","attack/concessions step correlation",-0.8219351600992229))
    stats = [q06r_parameter(name,rung,parameter) for name in q06r_names]
    med = [r.median for r in stats]
    low = [r.q05 for r in stats]
    high = [r.q95 for r in stats]
    scatter!(q06r_plot[axis],med,eachindex(med),
        xerror=(med-low,high-med),yticks=(1:11,q06r_names),
        label="10a median and 90% interval",xlabel=label,legend=:bottomright)
    vline!(q06r_plot[axis],[baseline],color=:black,linestyle=:dash,
        label="Scottish B2 R6 (pooled 56/57)")
end
savefig(q06r_plot,joinpath(Q06R_FIG,"QS_forest_r_rho.png"))

# %%
# ===================================================================
# 4. RTS quality/style season-level scatter; paired predictive forest
# ===================================================================
q06r_team = filter(r -> r.rung=="C0" && r.protocol=="10a",q06r_tables["level_teams"])
q06r_scatter = plot(layout=(3,4),size=(1650,1150),margin=6 * Plots.mm)
for (i,league) in enumerate(q06r_names)
    rows = filter(:league=>==(league),q06r_team)
    scatter!(q06r_scatter[i],rows.q,rows.s,group=rows.season,
        title=league,xlabel="season-mean q",ylabel="season-mean s",legend=false,markersize=3)
    hline!(q06r_scatter[i],[0.0],color=:grey,label="")
    vline!(q06r_scatter[i],[0.0],color=:grey,label="")
end
savefig(q06r_scatter,joinpath(Q06R_FIG,"QS_levels_q_s.png"))
q06r_pairs = q06r_tables["paired_scores"]
q06r_pairplot = plot(layout=(1,3),size=(1900,650),left_margin=20 * Plots.mm)
for (axis,(candidate,control)) in enumerate((("C0","R2"),("R6","R2"),("C0","R6")))
    rows = filter(r -> r.candidate==candidate && r.control==control,q06r_pairs)
    sort!(rows,:league)
    scatter!(q06r_pairplot[axis],rows.mean_delta,1:nrow(rows),
        xerror=2 .* rows.se,yticks=(1:nrow(rows),rows.league),
        xlabel="10b paired mean logpd ± 2 fixture SE",title="$candidate − $control",legend=false)
    vline!(q06r_pairplot[axis],[0.0],color=:black,linestyle=:dash,label="")
end
savefig(q06r_pairplot,joinpath(Q06R_FIG,"QS_paired_scores.png"))

# %%
# ===================================================================
# 5. Goal-score forest (including pooled row), post-processing marker
# ===================================================================
q06r_goalplot = plot(layout=(1,2),size=(1500,680),left_margin=24 * Plots.mm)
for (axis,comparison) in enumerate(("full_minus_no_style","full_minus_no_quality"))
    rows = filter(:comparison=>==(comparison),q06r_goals)
    scatter!(q06r_goalplot[axis],rows.estimate,1:nrow(rows),
        xerror=2 .* rows.se,yticks=(1:nrow(rows),rows.league),
        xlabel="paired Poisson logpd ± 2 fixture SE",title=replace(comparison,"_"=>" "),legend=false)
    vline!(q06r_goalplot[axis],[0.0],color=:black,linestyle=:dash,label="")
end
savefig(q06r_goalplot,joinpath(Q06R_FIG,"QS_goal_check.png"))
println("R06_REPORT_DONE")

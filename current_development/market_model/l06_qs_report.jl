module QualityStyleReport

import CSV
import DataFrames
import Statistics

const DF = DataFrames
const ST = Statistics
const LEAGUES = (
    "eng_premier","eng_championship","eng_league_one","eng_league_two",
    "sco_premiership","sco_championship","sco_league_one","sco_league_two",
    "irl_premier","irl_first_division","fin_veikkausliiga",
)

"Aggregate only complete accepted league runs; fail rather than suppress missing data."
function collect_results(root)
    tables = Dict{String,DF.DataFrame}()
    sources = ("parameters","forecast_coverage","forecast_fixtures","paired_scores",
        "levels","level_teams","goal_fixtures","identifiability")
    for filename in sources
        rows = DF.DataFrame[]
        for league in LEAGUES
            path = joinpath(root,"fits",league,"$filename.csv")
            isfile(path) || error("missing accepted $league: $path")
            frame = CSV.read(path,DF.DataFrame)
            DF.nrow(frame) > 0 || error("empty $league: $filename")
            all(==(league),frame.league) || error("wrong league in $path")
            push!(rows,frame)
        end
        tables[filename] = vcat(rows...)
    end
    return tables
end

"Apply the manager's fixed 10a structural/10b predictive criteria without tuning."
function verdicts(parameters,paired)
    rows = NamedTuple[]
    for league in LEAGUES
        function value(rung,protocol,parameter)
            g = DF.filter(r -> r.league == league && r.rung == rung &&
                r.protocol == protocol && r.parameter == parameter,parameters)
            DF.nrow(g) == 1 || error("not one $league $rung $protocol $parameter")
            return only(eachrow(g))
        end
        function criteria(protocol)
            r6 = value("R6",protocol,"corr_delta_q_s")
            c0 = value("C0",protocol,"r")
            aligned = r6.q05 <= 0 <= r6.q95 || abs(r6.median) < 0.3
            dominated = c0.q95 < 0.6
            clearly_not_dominated = c0.q05 > 0.6
            return (; r6,c0,aligned,dominated,clearly_not_dominated)
        end
        full,train = criteria("10a"),criteria("10b")
        pair = DF.filter(r -> r.league == league && r.candidate == "C0" &&
            r.control == "R6",paired)
        DF.nrow(pair) == 1 || error("missing C0 vs R6: $league")
        p = only(eachrow(pair))
        predictive = p.mean_delta >= -2p.se
        verdict = !full.aligned || full.clearly_not_dominated ? "unsuitable" :
            full.dominated && predictive ? "suitable" : "inconclusive"
        reason = verdict == "suitable" ? "all three criteria pass" :
            !full.aligned ? "quality/style innovation axes not aligned" :
            full.clearly_not_dominated ? "style/quality ratio entirely above 0.6" :
            !full.dominated ? "ratio interval overlaps 0.6" :
            "C0 more than 2 fixture SE below R6"
        flips = String[]
        full.aligned != train.aligned && push!(flips,"alignment")
        full.dominated != train.dominated && push!(flips,"quality dominance")
        push!(rows,(; league,verdict,reason,alignment_pass=full.aligned,
            quality_dominance_pass=full.dominated,prediction_pass=predictive,
            r6_corr_qs_10a=full.r6.median,r6_corr_qs_10a_q05=full.r6.q05,
            r6_corr_qs_10a_q95=full.r6.q95,
            c0_r_10a=full.c0.median,c0_r_10a_q05=full.c0.q05,
            c0_r_10a_q95=full.c0.q95,
            c0_minus_r6_10b=p.mean_delta,fixture_se=p.se,n_test=p.n,
            r6_corr_qs_10b=train.r6.median,r6_corr_qs_10b_q05=train.r6.q05,
            r6_corr_qs_10b_q95=train.r6.q95,
            c0_r_10b=train.c0.median,c0_r_10b_q05=train.c0.q05,
            c0_r_10b_q95=train.c0.q95,
            protocol_sensitive=!isempty(flips),flipped_criteria=join(flips,";")))
    end
    return DF.DataFrame(rows)
end

"Paired goal checks by league and over all 11 leagues, same fixture treatment."
function goal_summary(fixtures)
    rows = NamedTuple[]
    for league in vcat(collect(LEAGUES),["all_leagues_pooled"])
        g = league == "all_leagues_pooled" ? fixtures : DF.filter(:league=>==(league),fixtures)
        n = DF.nrow(g)
        n >= 2 || error("no paired goal fixtures for $league")
        for (comparison,col) in (("full_minus_no_style",:no_style_logpd),
                                 ("full_minus_no_quality",:no_quality_logpd))
            delta = g.full_logpd-g[!,col]
            push!(rows,(; league,comparison,n,estimate=ST.mean(delta),
                se=ST.std(delta)/sqrt(n),
                method="paired double-Poisson goal log score; fixture SE"))
        end
        # IRLS for the Poisson calibration slope of totals with log expected total as covariate.
        x = log.(g.expected_total)
        y = g.goals_home+g.goals_away
        X = hcat(ones(n),x)
        beta = [log(ST.mean(y)),1.0]
        converged = false
        for _ in 1:100
            mu = exp.(X*beta)
            step = (X'*(X.*mu))\(X'*(y-mu))
            beta += step
            if maximum(abs,step) < 1e-10
                converged = true
                break
            end
        end
        converged || error("pooled Poisson calibration IRLS failed for $league")
        mu = exp.(X*beta)
        se = sqrt(inv(X'*(X.*mu))[2,2])
        push!(rows,(; league,comparison="total_goal_calibration_slope",n,
            estimate=beta[2],se,method="Poisson GLM log-link on log predicted total; ideal 1; model-based SE"))
    end
    return DF.DataFrame(rows)
end

end # module

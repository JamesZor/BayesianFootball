# Report figure from frozen CSV results only: no refits, no changed score definitions.
# Native SVG keeps reproduction independent of plotting-package versions.

# %% 1. Packages and frozen posterior-r results
using CSV, DataFrames, Printf
strip(read(`hostname`, String)) == "mcmc-beast" || error("figures only on mcmc-beast")
f5_r = CSV.read(joinpath(@__DIR__, "results", "posterior_r_by_fold.csv"), DataFrame)
all(isfinite, f5_r.r_q05) && all(isfinite, f5_r.r_q95) || error("nonfinite posterior bounds")
all(0.05 .< f5_r.r_q05) && all(f5_r.r_q95 .< 10.0) || error("posterior outside figure limits")

# %% 2. Per-fold medians and 90% intervals, common logarithmic scale
function f5_posterior_svg(rows, path)
    format(x) = @sprintf("%.3f", x)
    open(path, "w") do io
        println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="650" viewBox="0 0 1080 650">
<title>Posterior style-to-quality ratios by fold</title>
<desc>Each point is a posterior median; whiskers are 90 percent intervals. Dashed lines are prior medians. Missing micro-scale sites are not plotted.</desc>
<rect width="1080" height="650" fill="white"/>
<g font-family="sans-serif" fill="#222">
<text x="35" y="30" font-size="20">Posterior r by fold — common logarithmic scale</text>
<text x="35" y="53" font-size="13">Dots: median; whiskers: 90% interval; dashed line: prior median. No rₖ site in folds 1 and 21.</text>""")
        for (a, arm) in enumerate(("qs_market_r", "qs_weak_r")),
            (s, scale) in enumerate(("r₀", "rₛ", "rₖ"))
            left, top, width, height = 65 + 350 * (s - 1), 100 + 270 * (a - 1), 285, 195
            x(fold) = left + (fold - 1) / 39 * width
            y(r) = top + height * (1 - log(r / 0.05) / log(10 / 0.05))
            prior = arm == "qs_weak_r" ? 1.0 : scale == "rₛ" ? 0.4 : 0.3
            color = arm == "qs_market_r" ? "#2266aa" : "#b54b20"
            panel = sort(filter(r -> r.arm == arm && r.scale == scale, rows), :fold)
            nrow(panel) == (scale == "rₖ" ? 38 : 40) || error("incomplete $arm/$scale panel")
            println(io, "<text x=\"$left\" y=\"$(top - 13)\" font-size=\"15\">$arm — $scale</text>")
            for tick in (0.1, 0.3, 1.0, 3.0, 10.0)
                yy = format(y(tick))
                println(io, "<line x1=\"$left\" y1=\"$yy\" x2=\"$(left + width)\" y2=\"$yy\" stroke=\"#ddd\"/>")
                println(io, "<text x=\"$(left - 8)\" y=\"$(y(tick) + 4)\" text-anchor=\"end\" font-size=\"11\">$tick</text>")
            end
            println(io, "<line x1=\"$left\" y1=\"$(y(prior))\" x2=\"$(left + width)\" y2=\"$(y(prior))\" stroke=\"#555\" stroke-dasharray=\"5 4\"/>")
            for r in eachrow(panel)
                xx = format(x(r.fold))
                println(io, "<line x1=\"$xx\" x2=\"$xx\" y1=\"$(format(y(r.r_q05)))\" y2=\"$(format(y(r.r_q95)))\" stroke=\"$color\" stroke-opacity=\"0.5\"/>")
                println(io, "<circle cx=\"$xx\" cy=\"$(format(y(r.r_median)))\" r=\"2.6\" fill=\"$color\"/>")
            end
            for tick in (1, 10, 20, 30, 40)
                println(io, "<text x=\"$(x(tick))\" y=\"$(top + height + 19)\" text-anchor=\"middle\" font-size=\"11\">$tick</text>")
            end
            println(io, "<text x=\"$(left + width / 2)\" y=\"$(top + height + 38)\" text-anchor=\"middle\" font-size=\"12\">Fold</text>")
        end
        println(io, "</g></svg>")
    end
end
f5_posterior_svg(f5_r, joinpath(@__DIR__, "results", "posterior_r_by_fold.svg"))
println("FIGURES_DONE posterior_r_panels=6 posterior_rows=$(nrow(f5_r))")

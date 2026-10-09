# Section 7 of r05_score.jl alone, for a REPL where sections 1–6 already ran (same variables).
# %% 7. Posterior r by fold for both QS arms, and implied step correlation ρ = (r²−1)/(r²+1)
# rₖ is a site only in folds with target steps (the first fold of each season has none).
s5_r_rows = NamedTuple[]
for arm in ("qs_market_r", "qs_weak_r"), (k, fold) in enumerate(s5_fits[arm].folds)
    present = Set(string.(names(fold.chain)))
    for scale in ("r₀", "rₛ", "rₖ")
        "dyn.$scale" in present || continue
        x = vec(Array(fold.chain[Symbol("dyn.$scale")]))
        ρ = (x .^ 2 .- 1) ./ (x .^ 2 .+ 1)
        q = quantile(x, [0.05, 0.5, 0.95])
        push!(s5_r_rows, (; arm, fold = k, scale, r_q05 = q[1], r_median = q[2], r_q95 = q[3],
            rho_median = median(ρ)))
    end
end
CSV.write(joinpath(S5_OUT, "posterior_r_by_fold.csv"), DataFrame(s5_r_rows))
println("PHASE5_SCORE best_goal_1x2=$s5_best")
println("PHASE5_SCORE_DONE")

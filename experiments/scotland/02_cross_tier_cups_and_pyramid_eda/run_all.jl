# Reproduce the whole TODO 029 suite in one warm REPL (≈ 3 minutes on archpc, most of it
# first-call compilation).  Needs BF_DB_URL (env or the git-ignored repo .env) for r01 only;
# r02–r06 read the CSVs r01 writes.
#
#   julia --project=/home/james/bet_project/BayesianFootball -t 8
#   julia> include("experiments/scotland/02_cross_tier_cups_and_pyramid_eda/run_all.jl")

for r in ("r01_extract_scottish_pyramid_dataset.jl", "r02_descriptive_supremacy_eda.jl",
          "r03_econometric_tier_glms.jl", "r04_dixon_coles_network_ratings.jl",
          "r05_market_efficiency_cup_pricing.jl", "r06_prior_calibration_recommendations.jl")
    @info "running $r"
    @time include(joinpath(@__DIR__, r))
end

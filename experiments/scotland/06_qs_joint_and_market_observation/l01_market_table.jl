# Frozen input adapter only; no fitting. Production src never imports this module.
module Wave2MarketTable
import CSV, DataFrames, SHA
using BayesianFootball
include(joinpath(@__DIR__, "..", "05_quality_style_goal_vs_market", "market_arm.jl"))

const SNAPSHOT_SHA = QSMarketArm.SNAPSHOT_SHA
const TABLE_PATH = joinpath(@__DIR__, "results", "market_rates.csv")
const HASH_PATH = joinpath(@__DIR__, "results", "market_rates.sha256")

"Build the injected full-book table using the unchanged wave-1 pinned inversion panel."
function prepare(ds; panel_path = "/root/BF_runs/qs_experiment_out/market_panel")
    prepared = QSMarketArm.prepare_panel(ds, panel_path)
    rates = prepared.rates
    full = rates.accepted .& (rates.n_selections .>= prepared.config.min_selections_ladder)
    table = DataFrames.DataFrame(match_id = Int.(rates.match_id),
        log_lambda_h = [full[i] ? log(rates.lambda_h[i]) : 0.0 for i in 1:DataFrames.nrow(rates)],
        log_lambda_a = [full[i] ? log(rates.lambda_a[i]) : 0.0 for i in 1:DataFrames.nrow(rates)],
        full_book = full)
    DataFrames.sort!(table, :match_id)
    return table
end

"Verify the frozen CSV digest before injecting; immutable snapshot identity is checked separately."
function feature()
    digest = bytes2hex(open(SHA.sha256, TABLE_PATH))
    digest == strip(read(HASH_PATH, String)) || error("wave-2 market-rate table digest differs")
    return Features.MarketRatesFeature(CSV.read(TABLE_PATH, DataFrames.DataFrame))
end
end

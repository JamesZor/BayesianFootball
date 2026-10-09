# Frozen input adapter only; no fitting. Production src never imports this module.
# Manager-authorised original namespace: the saved manifest includes Main.QSMarketArm.
if !isdefined(Main, :QSMarketArm)
    Base.include(Main, joinpath(@__DIR__, "..", "05_quality_style_goal_vs_market", "market_arm.jl"))
end
module Wave2MarketTable
import CSV, DataFrames, SHA
using BayesianFootball
const QSMarketArm = Main.QSMarketArm

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
    check_values(table, prepared)
    return table
end

"Exact value-level parity against wave 1's prepared panel, including uncovered zero dummies."
function check_values(table, prepared)
    prepared.counts == (panel = 710, accepted = 595, one_x2_only = 78, ladder = 517) ||
        error("wave-1 prepared target counts differ")
    ids = Int.(prepared.rates.match_id)
    table.match_id == sort(ids) || error("market table match IDs differ from wave 1")
    length(unique(table.match_id)) == length(ids) || error("market table has duplicate IDs")
    panel = prepared.panel
    panel_ids = Int.(panel.matches.match_id)
    panel.obs_match[1:2:end] == panel_ids && panel.obs_match[2:2:end] == panel_ids ||
        error("wave-1 panel observation pairs are misaligned")
    all(==(1.0), panel.obs_home[1:2:end]) && all(==(0.0), panel.obs_home[2:2:end]) ||
        error("wave-1 panel home/away order differs")
    expected = Dict(id => (panel.obs_y[2i - 1], panel.obs_y[2i])
                    for (i, id) in enumerate(panel_ids))
    for row in DataFrames.eachrow(table)
        full = haskey(expected, row.match_id)
        row.full_book == full || error("full-book flag differs for match $(row.match_id)")
        h, a = get(expected, row.match_id, (0.0, 0.0))
        isequal(row.log_lambda_h, h) && isequal(row.log_lambda_a, a) ||
            error("market log rates differ for match $(row.match_id)")
    end
    target_ids = Set(prepared.rates.match_id[in.(prepared.rates.season, Ref(("24/25", "25/26")))])
    target = in.(table.match_id, Ref(target_ids))
    count(target) == 710 && count(table.full_book[target]) == 517 ||
        error("wave-2 target panel/full-book counts differ")
    return (; rows = DataFrames.nrow(table), full_book = count(table.full_book),
              target_counts = prepared.counts)
end

"Verify the frozen CSV digest before injecting; immutable snapshot identity is checked separately."
function feature()
    digest = bytes2hex(open(SHA.sha256, TABLE_PATH))
    digest == strip(read(HASH_PATH, String)) || error("wave-2 market-rate table digest differs")
    return Features.MarketRatesFeature(CSV.read(TABLE_PATH, DataFrames.DataFrame))
end
end

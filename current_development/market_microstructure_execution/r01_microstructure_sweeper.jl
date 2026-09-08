# r01_microstructure_sweeper.jl
#
# WHAT THIS IS AND IS NOT
# ------------------------
# Empirical archive research on the actual 2026-09-05 Scottish 56/57 live paper slate.
# It compares causal T−25 TouchOnly and 3-level sweeps with a T−25…T−5 staged schedule.
# It does not refit a model, write either ledger, claim confirmed-lineup counterfactuals,
# or validate the production execution timestamp: recorded fills occurred after kick-off.
#
# FILTRATION / COMPARABILITY CONTRACT
# -----------------------------------
# Orders and their p_model/risk/odds are frozen paper_runbook facts.  Archive snapshots are
# selected at or before each decision minute, never after it. The primary staged replay uses
# no-replenishment depletion by (market, runner, side, absolute price); the refreshed-snapshot
# column is an explicitly optimistic upper bound, not an independent fill observation.

# %%
# ===================================================================
# 1. Packages and implementation
# ===================================================================
using Dates
using DataFrames
using Statistics
using Printf
using SHA

include(joinpath(@__DIR__, "l01_microstructure_sweeper.jl"))
include(joinpath(@__DIR__, "l02_archive_research.jl"))

const ME = MicrostructureExecution
const AR = ArchiveMicrostructureResearch

# %%
# ===================================================================
# 2. Configuration
# ===================================================================
const R01_DAY = Date(2026, 9, 5)
const R01_START_MINUTES = 25
const R01_END_MINUTES = 5
const R01_COMMISSION = 0.02
const R01_HURDLE = 0.02
const R01_SWEEP_SLIP = 0.01
const R01_OUTPUT_DIR = joinpath(@__DIR__, "results", "2026-09-05_archive")

# %%
# ===================================================================
# 3. Runtime and output directory
# ===================================================================
# Outputs are replaceable research artefacts. They contain no credentials and no ledger writes.
mkpath(R01_OUTPUT_DIR)
println("Output directory: ", R01_OUTPUT_DIR)
println("Read-only study day: ", R01_DAY)

# %%
# ===================================================================
# 4. Data snapshot and actual ledger provenance
# ===================================================================
conn = AR.connect_readonly()
orders, fixtures, snapshots, whole_slate_snapshots = try
    frozen = AR.extract_live_ledger(conn; day = R01_DAY)
    fixture_rows = AR.extract_fixture_inventory(conn; day = R01_DAY)
    order_rows = AR.extract_order_snapshots(conn, frozen;
                                            start_minutes = R01_START_MINUTES,
                                            end_minutes = R01_END_MINUTES)
    slate_rows = AR.extract_whole_slate_snapshots(conn; day = R01_DAY,
                                                   start_minutes = R01_START_MINUTES,
                                                   end_minutes = R01_END_MINUTES)
    (frozen, fixture_rows, order_rows, slate_rows)
finally
    close(conn)
end

nrow(orders) == 17 || error("expected the known 17-leg slate, got $(nrow(orders))")
all(orders.fill_model .== "touch_only") || error("actual ledger contains a non-TouchOnly fill model")
AR.write_csv(joinpath(R01_OUTPUT_DIR, "frozen_orders.csv"), orders)
AR.write_csv(joinpath(R01_OUTPUT_DIR, "fixture_inventory.csv"), fixtures)
AR.write_csv(joinpath(R01_OUTPUT_DIR, "archive_order_rows.csv"), snapshots)
AR.write_csv(joinpath(R01_OUTPUT_DIR, "whole_slate_archive_rows.csv"), whole_slate_snapshots)

coverage = AR.coverage_report(orders, snapshots;
                              start_minutes = R01_START_MINUTES,
                              end_minutes = R01_END_MINUTES)
AR.write_csv(joinpath(R01_OUTPUT_DIR, "archive_coverage.csv"), coverage)

# Reference quote versus stored-ts archive T−25 quote. This is an implementation-shortfall
# anchor comparison only: a post-kickoff recorded fill does not establish reference quote timing.
reference_mismatch = DataFrame(order_id = String[], side = String[], reference_odds = Float64[],
                               stored_t25_best = Union{Missing,Float64}[], mismatch = Union{Missing,Bool}[])
for row in eachrow(orders)
    snapshot = AR.snapshot_at(snapshots, String(row.order_id), AR._datetime(row.kickoff) - Minute(R01_START_MINUTES))
    best = snapshot === nothing ? missing : (Symbol(row.side) === :back ? snapshot.back[1] : snapshot.lay[1])
    mismatch = ismissing(best) ? missing : !isapprox(Float64(row.venue_odds), best; atol = 1e-9)
    push!(reference_mismatch, (String(row.order_id), String(row.side), Float64(row.venue_odds), best, mismatch))
end
AR.write_csv(joinpath(R01_OUTPUT_DIR, "reference_vs_archive_t25.csv"), reference_mismatch)
reference_mismatch_count = sum(skipmissing(reference_mismatch.mismatch))

# %%
# ===================================================================
# 5. Policies and gates
# ===================================================================
const R01_POLICIES = [
    # Fixed parent reference is an implementation-shortfall anchor, not contemporaneous tick slip.
    (name = "archive_touch_t25", policy = ME.TouchOnly(max_slip = R01_SWEEP_SLIP), refreshed = false),
    (name = "archive_sweep3_t25", policy = ME.MultiLevelSweep(max_slip = R01_SWEEP_SLIP), refreshed = false),
    (name = "staged_t25_t5_no_replenishment", policy = ME.StagedTWAP(max_slip = R01_SWEEP_SLIP,
                                                                        start_minutes = R01_START_MINUTES,
                                                                        end_minutes = R01_END_MINUTES), refreshed = false),
    (name = "staged_t25_t5_refreshed_upper_bound", policy = ME.StagedTWAP(max_slip = R01_SWEEP_SLIP,
                                                                             start_minutes = R01_START_MINUTES,
                                                                             end_minutes = R01_END_MINUTES), refreshed = true),
]

# %%
# ===================================================================
# 6. Archive replay: touch, three-level, and staged execution
# ===================================================================
parent_results = DataFrame()
child_results = DataFrame()
diagnostic_results = DataFrame()
for spec in R01_POLICIES
    parents, children, diagnostics = AR.replay_policy(orders, snapshots, spec.policy, ME;
        start_minutes = R01_START_MINUTES, end_minutes = R01_END_MINUTES,
        commission = R01_COMMISSION, hurdle = R01_HURDLE, refreshed = spec.refreshed)
    parents.policy_name = fill(spec.name, nrow(parents))
    parents.child_count = [sum(children.order_id .== order_id) for order_id in parents.order_id]
    parents.archive_available = parents.child_count .> 0
    append!(parent_results, parents; cols = :union)
    if nrow(children) > 0
        children.policy_name = fill(spec.name, nrow(children))
        append!(child_results, children; cols = :union)
    end
    diagnostics.policy_name = fill(spec.name, nrow(diagnostics))
    append!(diagnostic_results, diagnostics; cols = :union)
end
AR.write_csv(joinpath(R01_OUTPUT_DIR, "simulated_child_fills.csv"), child_results)
AR.write_csv(joinpath(R01_OUTPUT_DIR, "simulation_by_order.csv"), parent_results)
AR.write_csv(joinpath(R01_OUTPUT_DIR, "execution_diagnostics.csv"), diagnostic_results)
reason_detail = stack(diagnostic_results, [:level_1_reason, :level_2_reason, :level_3_reason];
                      variable_name = :level, value_name = :sequential_reason)
reason_summary = combine(groupby(reason_detail, [:policy_name, :status, :sequential_reason]),
                         nrow => :observations)
AR.write_csv(joinpath(R01_OUTPUT_DIR, "execution_reason_summary.csv"), reason_summary)

# Settlement is reported only for realized historical outcomes. Aggregate child gross by
# Betfair market first, then apply commission once to a positive market total. This avoids
# the standalone-win haircut used internally by ExecutionState and is not a future P&L claim.
settlement_columns = select(orders, :order_id, :side, :outcome, :fill_price, :fill_size,
                            :market_group, :match_id)
if nrow(child_results) == 0
    simulated_pnl = DataFrame(policy_name = String[], market_key = String[], gross_pnl = Float64[], net_pnl = Float64[])
else
    child_with_outcome = leftjoin(child_results, settlement_columns; on = :order_id)
    all(uppercase(String(row.outcome)) in ("WIN", "LOSE") for row in eachrow(child_with_outcome)) ||
        error("simulated realized P&L unavailable: a settlement outcome is missing or not WIN/LOSE")
    child_with_outcome.child_gross = [AR.simulated_gross_pnl(Symbol(row.side), String(row.outcome),
                                                               [row.price], [row.venue_stake])
                                      for row in eachrow(child_with_outcome)]
    child_with_outcome.market_key = string.(child_with_outcome.policy_name, "|",
                                            child_with_outcome.match_id, "|", child_with_outcome.market_group)
    simulated_pnl = combine(groupby(child_with_outcome, [:policy_name, :market_key]),
        :child_gross => sum => :gross_pnl,
    )
    simulated_pnl.commission_rate = fill(R01_COMMISSION, nrow(simulated_pnl))
    simulated_pnl.net_pnl = [AR.net_market_pnl(row.gross_pnl, row.commission_rate) for row in eachrow(simulated_pnl)]
end
AR.write_csv(joinpath(R01_OUTPUT_DIR, "simulated_realized_market_pnl.csv"), simulated_pnl)

simulation_summary = combine(groupby(parent_results, :policy_name),
    :target_risk => sum => :target_risk,
    :simulated_risk => sum => :simulated_risk,
    :risk_fill_pct => mean => :mean_order_fill_pct,
    :archive_available => sum => :orders_with_fill,
    :order_id => length => :orders,
)
simulation_summary.portfolio_fill_pct = simulation_summary.simulated_risk ./ simulation_summary.target_risk
AR.write_csv(joinpath(R01_OUTPUT_DIR, "simulation_summary.csv"), simulation_summary)

# %%
# ===================================================================
# 7. Liquidity change and WOM / future-price exploratory association
# ===================================================================
# Target-order sample: one observation per exact mapped frozen runner/order. Future price is
# the best available-to-position quote at T−5 relative to the causal T−25 selection. WOM uses
# all displayed 3-level depth. This is descriptive only: no signal threshold is fitted or selected.
liquidity = DataFrame(order_id = String[], side = String[], t25_available = Union{Missing,Float64}[],
                      t5_available = Union{Missing,Float64}[], liquidity_change = Union{Missing,Float64}[],
                      wom_t25 = Union{Missing,Float64}[], price_t25 = Union{Missing,Float64}[],
                      price_t5 = Union{Missing,Float64}[], future_price_change = Union{Missing,Float64}[])
for row in eachrow(orders)
    kickoff = AR._datetime(row.kickoff)
    t25 = AR.snapshot_at(snapshots, String(row.order_id), kickoff - Minute(R01_START_MINUTES))
    t5 = AR.snapshot_at(snapshots, String(row.order_id), kickoff - Minute(R01_END_MINUTES))
    if t25 === nothing || t5 === nothing
        push!(liquidity, (String(row.order_id), String(row.side), missing, missing, missing, missing, missing, missing, missing))
        continue
    end
    side = Symbol(row.side)
    sizes25 = side === :back ? t25.back_size : t25.lay_size
    sizes5 = side === :back ? t5.back_size : t5.lay_size
    prices25 = side === :back ? t25.back : t25.lay
    prices5 = side === :back ? t5.back : t5.lay
    available25 = sum(sizes25)
    available5 = sum(sizes5)
    push!(liquidity, (String(row.order_id), String(row.side), available25, available5,
                      available5 - available25, ME.wom(AR.execution_ladder(t25, ME)), prices25[1], prices5[1],
                      prices5[1] - prices25[1]))
end
AR.write_csv(joinpath(R01_OUTPUT_DIR, "liquidity_wom_future_price.csv"), liquidity)
valid_liquidity = dropmissing(liquidity, [:liquidity_change, :wom_t25, :future_price_change])
association = DataFrame(metric = ["n_complete_pairs", "cor_wom_t25_future_price_change", "cor_liquidity_change_future_price_change"],
                        value = Union{Missing,Float64}[nrow(valid_liquidity),
                            AR.safe_correlation(valid_liquidity.wom_t25, valid_liquidity.future_price_change),
                            AR.safe_correlation(valid_liquidity.liquidity_change, valid_liquidity.future_price_change)])
AR.write_csv(joinpath(R01_OUTPUT_DIR, "exploratory_association.csv"), association)

# Whole-slate sample is intentionally distinct from the target-order sample above: all archived
# runners in 1X2, O/U 2.5 and BTTS across the ten Scottish fixtures. One paired runner is one
# observation; this runner reports no independent-minute p-values.
function r01_last_whole(rows, at)
    eligible_indices = [i for i in 1:nrow(rows) if !ismissing(rows.ts[i]) && AR._datetime(rows.ts[i]) <= at]
    isempty(eligible_indices) && return nothing
    stamps = [AR._datetime(rows.ts[i]) for i in eligible_indices]
    return rows[eligible_indices[argmax(stamps)], :]
end

whole_pairs = DataFrame(match_id = Int[], market_id = String[], market_type = String[], symbol = String[],
                        liquidity_t25 = Float64[], liquidity_t5 = Float64[], liquidity_change = Float64[],
                        wom_t25 = Union{Missing,Float64}[], best_back_t25 = Float64[], best_back_t5 = Float64[],
                        future_back_change = Float64[])
for group in groupby(whole_slate_snapshots, [:match_id, :market_id, :market_type, :symbol, :kickoff])
    kickoff = AR._datetime(first(group.kickoff))
    first_row = r01_last_whole(group, kickoff - Minute(R01_START_MINUTES))
    last_row = r01_last_whole(group, kickoff - Minute(R01_END_MINUTES))
    (first_row === nothing || last_row === nothing) && continue
    b25 = AR._fixed_tuple(first_row.bid_prices, AR.PRICE_SCALE)
    bs25 = AR._fixed_tuple(first_row.bid_volumes, AR.SIZE_SCALE)
    l25 = AR._fixed_tuple(first_row.ask_prices, AR.PRICE_SCALE)
    ls25 = AR._fixed_tuple(first_row.ask_volumes, AR.SIZE_SCALE)
    b5 = AR._fixed_tuple(last_row.bid_prices, AR.PRICE_SCALE)
    bs5 = AR._fixed_tuple(last_row.bid_volumes, AR.SIZE_SCALE)
    push!(whole_pairs, (Int(first(group.match_id)), String(first(group.market_id)), String(first(group.market_type)),
                        String(first(group.symbol)), sum(bs25), sum(bs5), sum(bs5) - sum(bs25),
                        ME.wom(ME.Ladder(AR._datetime(first_row.ts), b25, bs25, l25, ls25, AR._float(first_row.market_matched))),
                        b25[1], b5[1], b5[1] - b25[1]))
end
AR.write_csv(joinpath(R01_OUTPUT_DIR, "whole_slate_liquidity_wom_pairs.csv"), whole_pairs)
whole_association = DataFrame(metric = ["whole_slate_paired_runner_observations",
                                        "whole_slate_fixture_clusters",
                                        "whole_slate_cor_wom_t25_future_best_back_change",
                                        "whole_slate_cor_liquidity_change_future_best_back_change",
                                        "whole_slate_median_liquidity_change",
                                        "whole_slate_t25_depth_total",
                                        "whole_slate_t5_depth_total",
                                        "whole_slate_depth_ratio_total_t5_over_t25",
                                        "whole_slate_median_runner_depth_ratio_t5_over_t25"],
                              value = Union{Missing,Float64}[nrow(whole_pairs),
                                  length(unique(whole_pairs.match_id)),
                                  AR.safe_correlation(whole_pairs.wom_t25, whole_pairs.future_back_change),
                                  AR.safe_correlation(whole_pairs.liquidity_change, whole_pairs.future_back_change),
                                  isempty(whole_pairs.liquidity_change) ? missing : median(whole_pairs.liquidity_change),
                                  sum(whole_pairs.liquidity_t25), sum(whole_pairs.liquidity_t5),
                                  sum(whole_pairs.liquidity_t25) == 0 ? missing : sum(whole_pairs.liquidity_t5) / sum(whole_pairs.liquidity_t25),
                                  isempty(whole_pairs.liquidity_t25) ? missing : median(whole_pairs.liquidity_t5 ./ whole_pairs.liquidity_t25)])
AR.write_csv(joinpath(R01_OUTPUT_DIR, "whole_slate_exploratory_association.csv"), whole_association)

# Predeclared descriptive WOM bins. A negative available-to-back price change means shortening.
# These are runner observations clustered within ten fixtures, not independent-minute tests.
whole_pairs.wom_bin = [w < 0.35 ? "low_lt_035" : w <= 0.65 ? "neutral_035_to_065" : "high_gt_065"
                       for w in whole_pairs.wom_t25]
wom_bins = combine(groupby(whole_pairs, :wom_bin),
    :future_back_change => length => :runner_observations,
    :future_back_change => mean => :mean_t5_minus_t25_best_back,
    :future_back_change => (x -> mean(x .< 0.0)) => :shortening_fraction,
    :match_id => (x -> length(unique(x))) => :fixture_clusters,
)
AR.write_csv(joinpath(R01_OUTPUT_DIR, "whole_slate_wom_bins.csv"), wom_bins)

# %%
# ===================================================================
# 8. Ledger reconciliation and report
# ===================================================================
# Literal baseline: raw production ledger TouchOnly only. It is not re-priced, re-timed, or
# retroactively passed through this prototype's hurdle. Simulated policies have their own guard.
ledger_reconcile = combine(orders,
    :risk => sum => :frozen_parent_risk,
    :risk_filled => sum => :recorded_fill_risk,
    :venue_stake => sum => :frozen_venue_stake,
    :fill_size => sum => :recorded_fill_venue_stake,
    :net_pnl => sum => :recorded_net_pnl,
    :order_id => length => :orders,
)
ledger_reconcile.recorded_fill_risk_pct = ledger_reconcile.recorded_fill_risk ./ ledger_reconcile.frozen_parent_risk
AR.write_csv(joinpath(R01_OUTPUT_DIR, "ledger_reconciliation.csv"), ledger_reconcile)

# Artefacts are replaceable, because a live archive can change. Pin this run's extraction
# timestamp, inputs, outputs and code state in a manifest rather than calling this path immutable.
function r01_sha256(path)
    return bytes2hex(open(SHA.sha256, path))
end

function r01_git(args...)
    try
        return strip(read(Cmd(["git"; collect(args)]), String))
    catch
        return "unavailable"
    end
end

open(joinpath(R01_OUTPUT_DIR, "run_log.txt"), "w") do io
    println(io, "Archive microstructure runner completed at ", now())
    println(io, "Frozen orders: ", nrow(orders), "; fixture inventory: ", nrow(fixtures))
    println(io, "Mapped runner/order coverage: ", sum(coverage.mapped_runner), "/", nrow(coverage))
    println(io, "Timely T-25 selections (last quote at/before T-25, age <=90s): ", sum(coverage.exact_t25), "/", nrow(coverage))
    println(io, "Recorded ledger fill timestamp range: ", minimum(orders.filled_at), " .. ", maximum(orders.filled_at))
    println(io, "Raw ledger TouchOnly fill fraction: ", only(ledger_reconcile.recorded_fill_risk_pct))
    println(io, "Simulated realized market P&L uses market-net positive-gross commission, not ExecutionState haircut.")
    show(io, MIME("text/plain"), simulation_summary)
    println(io)
    show(io, MIME("text/plain"), association)
    println(io)
end

println("\nG-A ledger facts: ", nrow(orders), " frozen orders; raw TouchOnly baseline retained.")
println("G-B archive coverage: ", sum(coverage.mapped_runner), "/", nrow(coverage), " exact runner mappings; ",
        sum(coverage.exact_t25), " timely T−25 selections.")
println("G-C replay outputs: ", nrow(parent_results), " parent-policy rows and ", nrow(child_results), " child fills.")
manifest_paths = filter(path -> isfile(path) && basename(path) != "manifest.txt", readdir(R01_OUTPUT_DIR; join = true))
open(joinpath(R01_OUTPUT_DIR, "manifest.txt"), "w") do io
    println(io, "extracted_at_utc=", now(UTC))
    println(io, "study_day=", R01_DAY)
    println(io, "slate_id=0fd23606-80bf-4fcd-be44-66d708562fe6")
    println(io, "archive_scale=all exchange-feed integer prices, volumes and market_matched divided by 10000; source=src/MatchDay/implementations/book.jl:66-76")
    println(io, "julia_version=", VERSION)
    println(io, "git_commit=", r01_git("rev-parse", "HEAD"))
    println(io, "git_status_sha256=", bytes2hex(SHA.sha256(r01_git("status", "--porcelain"))))
    println(io, "source_sha256:")
    for path in [joinpath(@__DIR__, "l01_microstructure_sweeper.jl"),
                 joinpath(@__DIR__, "l02_archive_research.jl"),
                 joinpath(@__DIR__, "r01_microstructure_sweeper.jl"),
                 joinpath(@__DIR__, "EMPIRICAL.md"),
                 joinpath(@__DIR__, "test_archive_research.jl"),
                 joinpath(@__DIR__, "test_archive_clock_order.jl")]
        println(io, basename(path), " ", r01_sha256(path))
    end
    println(io, "artifact_sha256:")
    for path in sort(manifest_paths)
        println(io, basename(path), " ", r01_sha256(path))
    end
end
println("G-D report: ", joinpath(R01_OUTPUT_DIR, "run_log.txt"))
println("G-E replaceable-artifact manifest: ", joinpath(R01_OUTPUT_DIR, "manifest.txt"))

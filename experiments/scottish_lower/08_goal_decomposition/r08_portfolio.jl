# ==============================================================================
# r08 — Experiment 08 common-Betfair portfolio benchmark
# ==============================================================================
#
# WHAT THIS IS. A no-MCMC repricing of every persisted baseline/candidate posterior
# on the exact canonical 710-fixture set. Every model receives the same de-vigged
# Betfair TWA[-20,0] 1X2 + O/U 2.5 book and the same commission, shrinkage and
# daily-slate policy. Existing portfolio artefacts are deliberately not reused.
#
# WHAT THIS IS NOT. A prospective return claim. It is a historical simulation on
# an exchange archive and carries the portfolio's stated fill assumptions.
# ===============================================================================

# %%
# ===============================================================================
# 1. Packages and implementation
# ===============================================================================
using BayesianFootball
using CSV
using DataFrames
using Dates
using LinearAlgebra
using Printf
using ThreadPinning

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "l08_workflow.jl"))
include(joinpath(@__DIR__, "l08_evaluation.jl"))

const R08_PORTFOLIO_OUTPUT = joinpath(@__DIR__, "results")

# %%
# ===============================================================================
# 2. Fixed portfolio contract
# ===============================================================================
# Book: de-vigged Betfair TWA[-20,0], 1X2 + O/U 2.5 only, Baker-McHale shrinkage,
# 2% per-bet commission. Policy: 30% FlatTrust, SlateDrawdown(23), FixedCap(20%),
# DailySlate. Every model is recomputed; no stored bookmaker or wider-market ledger
# is a valid benchmark under this contract.
book = l08_book_spec()
policy = l08_policy_spec()

# %%
# ===============================================================================
# 3. Exact baseline/candidate manifests and common book
# ===============================================================================
l08_load_runtime_env!()
ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 10_000)

baseline_pairs = [l08_load_fit_checked(entry) for entry in L08_BASELINE_RUNS]
source_dbs = Dict(entry.name => pair[1] for (entry, pair) in zip(L08_BASELINE_RUNS, baseline_pairs))
fits = Dict(entry.name => pair[2] for (entry, pair) in zip(L08_BASELINE_RUNS, baseline_pairs))
canonical_ids = l08_canonical_match_ids(fits)

candidate_entries = l08_production_candidates(R08_PORTFOLIO_OUTPUT)
mode = candidate_entries === nothing ? "baselinepreflight" : "fullcomparison"
if candidate_entries !== nothing
    for entry in candidate_entries
        source_db, fit = l08_load_fit_checked(entry)
        source_dbs[entry.name] = source_db
        fits[entry.name] = fit
    end
end

for (name, fit) in fits
    l08_require_coverage(name, fit, canonical_ids)
end
odds = l08_betfair_closing_odds(ds)
common_odds = filter(:match_id => in(Set(canonical_ids)), odds)

# %%
# ===============================================================================
# 4. Reprice, simulate, persist, and verify the exact ledger
# ===============================================================================
rows = NamedTuple[]
for (name, fit) in sort!(collect(fits); by = first)
    books, build_report = Portfolio.build_books_reported(
        book, fit, common_odds, ds;
        require_converged = true,
        quiet = true,
    )
    Portfolio.n_skipped(build_report) == 0 || error(
        "$name skipped $(Portfolio.n_skipped(build_report)) canonical fixtures; refuse unequal books")
    length(books) == length(canonical_ids) || error(
        "$name built $(length(books)) books for $(length(canonical_ids)) canonical fixtures")

    result = Portfolio.simulate_portfolio(policy, books, build_report; bootstrap = true)
    run_id = getproperty(first(filter(entry -> entry.name == name,
                                      candidate_entries === nothing ? collect(L08_BASELINE_RUNS) :
                                      vcat(collect(L08_BASELINE_RUNS), candidate_entries))), :run_id)
    portfolio_id = Portfolio.save_portfolio_db(
        result, run_id, source_dbs[name];
        book_spec = book,
        policy_spec = policy,
        metadata = (; runner = "r08_portfolio",
                      mode,
                      odds_source = "Betfair de-vigged TWA[-20,0]",
                      markets = "1X2+OU2.5",
                      commission = 0.02,
                      common_fixture_count = length(canonical_ids),
                      recomputed = true),
    )
    l08_assert_portfolio_roundtrip(result, portfolio_id, source_dbs[name])
    summary = result.summary
    push!(rows, (
        model = name,
        model_run_id = string(run_id),
        portfolio_run_id = string(portfolio_id),
        n_fixtures = length(canonical_ids),
        n_books = length(books),
        n_bets = summary.n_bets,
        total_return_pct = summary.total_return_pct,
        flat_roi_pct = summary.roi,
        roi_1x2_pct = summary.roi_1x2,
        sharpe_ann = summary.sharpe_ann,
        max_drawdown_pct = summary.mdd,
        odds_source = "Betfair de-vigged TWA[-20,0]",
        markets = "1X2+OU2.5",
        commission = 0.02,
    ))
    @printf(" %-38s bets %5d | return %+8.2f%% | ROI %+7.2f%% | Sharpe %6.3f | %s\n",
            name, summary.n_bets, summary.total_return_pct, summary.roi, summary.sharpe_ann,
            portfolio_id)
end

# %%
# ===============================================================================
# 5. Machine-readable result and manifest
# ===============================================================================
summary = DataFrame(rows)
CSV.write(joinpath(R08_PORTFOLIO_OUTPUT, "r08_$(mode)_common_betfair_portfolio.csv"), summary)
registry = l08_registry(ds, PostgresStorage(L08_EXPERIMENT);
    output_dir = R08_PORTFOLIO_OUTPUT,
    source_files = [joinpath(@__DIR__, "l08_evaluation.jl"), @__FILE__])
manifest = l08_write_manifest!(registry; stage = "portfolio_$(mode)",
    extra = Dict(
        "canonical_fixture_count" => length(canonical_ids),
        "odds_source" => "Betfair de-vigged TWA[-20,0]",
        "markets" => "1X2+OU2.5",
        "commission" => 0.02,
        "policy" => "FlatTrust(0.30), SlateDrawdown(23.0), FixedCap(0.20), DailySlate",
        "all_ledgers_roundtripped" => true,
    ))
println("\nMODE: ", mode)
println("Wrote ", joinpath(R08_PORTFOLIO_OUTPUT, "r08_$(mode)_common_betfair_portfolio.csv"))
println("Manifest: ", manifest)

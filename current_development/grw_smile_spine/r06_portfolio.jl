# ==============================================================================
# r06 — Closing-line portfolio under Option B, and the T011 staking fix measured
# ==============================================================================
#
# WHAT THIS IS AND IS NOT
#
# Every arm staked through ONE contract — `MatchDay.option_b_system()`, the audited production
# book and policy — against the de-vigged Betfair TWA(−20, 0] close, over the identical
# buildable subset of the 710-fixture 24/25 + 25/26 walk-forward panel. Only the posterior and
# the STAKING ROUTE differ between rows.
#
# It is a closing-line simulation: prices are the close, which is what the market pillars were
# fitted against, so an anchored arm is partly scored against its own anchor. At n ≈ 630
# fixtures neighbouring arms' bootstrap growth intervals overlap, so a bankroll ranking here is
# descriptive. The paired slate-growth contrasts in r07 are the inferential ones.
#
# THE TWO QUESTIONS
#
#   H4  does the 1-parameter spine keep the five-strike smile's portfolio alpha?
#       Read from the same-weight rows, reweighted route, and the shared/exclusive attribution.
#   T011 what does correct staking actually do? Every smile arm is staked TWICE:
#
#         :grid        Task 015's path — O/U priced through λ_tot·φ(K), stakes solved off the
#                      un-smiled grid. Reproduces Task 015's rows.
#         :reweighted  stakes solved off the anti-diagonal reweighted grid (l01 §8).
#
#       Task 015 r07 measured a smile container and its φ-stripped twin staking IDENTICAL
#       ledgers — φ changed the reported price and nothing else. If that defect matters, these
#       two routes must differ here; if they barely differ, the fix is real but immaterial, and
#       that is the honest finding. A CountLatents arm is identical under both routes by
#       construction and is staked once.
#
# GATES
#
#   P1  reproduction: the pinned baseline, on Task 014's 632-fixture buildable panel, reproduces
#       its published Option B row (+385.8%, ROI 11.68%, 1,247 bets). A different buildable panel
#       is REPORTED as not comparable, never claimed as a reproduction.
#   P2  reported price: every staked totals bet of a smile arm carries p_model equal to
#       mean cdf(Poisson(λ_tot·φ_K), K) to ≤ 1e-9, under BOTH routes (Task 015's gate).
#   P2b stake side (the half P2 cannot see): under `:reweighted` the distribution the Kelly solve
#       read implies the smile's totals CDF to ≤ 1e-9. Under `:grid` the same quantity is
#       MEASURED and expected to fail — that measurement IS ticket T011, so it is recorded in the
#       report rather than thrown.
#   P3  persistence: the two spine portfolios are written to this task's namespace and each
#       reloads with an identical bet ledger. The pinned arms' own portfolios belong to the tasks
#       that sampled them and are not re-written; their reweighted rows are reported here only.
#
# USAGE (mcmc-beast, from /root/BF_grw_smile_spine, after r02)
#
#   julia --project -t 16 current_development/grw_smile_spine/r06_portfolio.jl
# ==============================================================================

# %%
# ===================================================================
# 1. Packages and implementation
# ===================================================================
using ThreadPinning
using LinearAlgebra
pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

using BayesianFootball
using CSV
using DataFrames
using Dates
using Printf
using Statistics

include(joinpath(@__DIR__, "l01_loader.jl"))
include(joinpath(@__DIR__, "l02_evaluation.jl"))

# %%
# ===================================================================
# 2. Configuration
# ===================================================================
const R06_CONFIG = gss_config()
const R06_BOOTSTRAP_B = 4000
const R06_SEED = 1
const R06_GATE_TOL = 1.0e-9
const R06_OUT_DIR = joinpath(R06_CONFIG.save_root, "portfolio")
const R06_LATENT_DIRS = [joinpath(R06_CONFIG.save_root, "latents"),
                         joinpath(@__DIR__, "..", "grw_market_smile", "results", "latents")]
const R06_GIT = try
    readchomp(`git rev-parse --short HEAD`)
catch
    "unknown"
end

# Task 014 README §6 / Task 015 r06 — the same pinned run, same contract, its own panel.
const R06_BASELINE_PUBLISHED = (n_panel = 632, total_return_pct = 385.8, roi_pct = 11.68, n_bets = 1247)

mkpath(R06_OUT_DIR)
println("\n" * "="^96)
println("  r06 CLOSING-LINE PORTFOLIO — Option B; spine vs five-strike smile vs controls")
println("  routes     : :reweighted (T011 fix) for every arm; :grid additionally for smile arms")
println("  git        : ", R06_GIT, "   host ", gethostname(), "   threads ", Threads.nthreads())
println("="^96)

# %%
# ===================================================================
# 3. Data, contract, arms
# ===================================================================
r06_ds = gph_load_data()
r06_splitter = gph_splitter(R06_CONFIG.extension_seasons)
r06_odds = gms_betfair_closing_odds(r06_ds)
r06_book, r06_policy = gms_option_b()
println("  book   : ", r06_book)
println("  policy : ", r06_policy)

r06_arms = gss_arms(R06_CONFIG)
r06_raw = gss_load_arms(r06_arms, r06_ds; splitter = r06_splitter, latent_dirs = R06_LATENT_DIRS)
r06_panel_ids = gms_common_panel(r06_ds, r06_raw, R06_CONFIG.target_seasons)
length(r06_panel_ids) == R06_CONFIG.expected_oos || error(
    "common panel is $(length(r06_panel_ids)) fixtures; expected $(R06_CONFIG.expected_oos)")
r06_fits = Dict(label => gms_restrict(fit, r06_panel_ids) for (label, fit) in r06_raw)
r06_raw = nothing
GC.gc()

# %%
# ===================================================================
# 4. The panel — buildable under BOTH routes
# ===================================================================
# A route can in principle refuse a fixture the other accepts (a reweighting refusal is a
# container property, not a quote property). Comparing routes on different fixture sets would
# confound the comparison, so the panel is the intersection and every drop is named.
r06_quoted = sort!(collect(intersect(Set(r06_panel_ids), Set(Int.(r06_odds.match_id)))))
r06_panel_rw, r06_dropped_rw = gss_buildable_panel(r06_book, r06_fits, r06_odds, r06_ds, r06_quoted;
                                                   route = :reweighted)
r06_panel_grid, r06_dropped_grid = gss_buildable_panel(r06_book, r06_fits, r06_odds, r06_ds, r06_quoted;
                                                       route = :grid)
r06_panel = sort!(collect(intersect(Set(r06_panel_rw), Set(r06_panel_grid))))
r06_dropped = leftjoin(rename(r06_dropped_rw, :reason => :reason_reweighted),
                       rename(r06_dropped_grid, :reason => :reason_grid); on = :match_id)
CSV.write(joinpath(R06_OUT_DIR, "r06_dropped_fixtures.csv"), r06_dropped)
@printf("  panel  : %d walk-forward → %d quoted → %d buildable (reweighted %d, grid %d, %d dropped)\n",
        length(r06_panel_ids), length(r06_quoted), length(r06_panel),
        length(r06_panel_rw), length(r06_panel_grid), nrow(r06_dropped))
length(r06_panel_rw) == length(r06_panel_grid) ||
    println("  NOTE: the two routes differ in buildability; the intersection is used for every row")

# %%
# ===================================================================
# 5. Simulation — one contract, one panel, one or two routes per arm
# ===================================================================
r06_results = Dict{Tuple{String,Symbol},Any}()
r06_rows = NamedTuple[]
r06_breakdown_rows = NamedTuple[]
r06_price_gate_rows = NamedTuple[]
r06_stake_gate_rows = NamedTuple[]

for arm in r06_arms
    is_smile = gss_is_smile(r06_fits[arm.label])
    routes = is_smile ? (:reweighted, :grid) : (:reweighted,)
    for route in routes
        label = "$(arm.label)/$(route)"
        result, restricted, books = gss_simulate(r06_book, r06_policy, r06_fits[arm.label],
                                                 r06_odds, r06_ds, r06_panel;
                                                 label, route, B = R06_BOOTSTRAP_B, seed = R06_SEED)
        r06_results[(arm.label, route)] = result
        bets = result.trajectory.bets
        row = gms_portfolio_row(arm.label, result; n_panel = length(r06_panel))
        edge = gms_edge_summary(bets)
        breakdown = gms_breakdown(arm.label, bets)
        for b in breakdown
            push!(r06_breakdown_rows, (; route = String(route), b...))
        end
        n_1x2 = sum((r.n_bets for r in breakdown if r.level == "market" && r.group == "1X2"); init = 0)
        n_tot = sum((r.n_bets for r in breakdown if r.level == "market" && r.group == "totals"); init = 0)
        push!(r06_rows, (; route = String(route), role = arm.role, row...,
                           n_bets_1x2 = n_1x2, n_bets_totals = n_tot,
                           capture_ratio = edge.capture_ratio, run_id = string(arm.run_id)))

        if restricted.latents isa SmileLatents
            # --- P2: the ledger's reported price is the smile price ---------------------
            g = gms_smile_book_gate(bets, restricted.latents)
            push!(r06_price_gate_rows, (; model = arm.label, route = String(route), g...))
            g.n_totals_bets > 0 || error("P2: $label staked no totals bet to check")
            g.max_abs_vs_smile <= R06_GATE_TOL || error(
                "P2 FAILED: $label staked totals at p_model up to $(g.max_abs_vs_smile) from " *
                "cdf(Poisson(λ_tot·φ)) — the books were not priced through the smile")

            # --- P2b: the distribution the STAKE was solved on --------------------------
            s = gss_book_totals_gate(books, restricted.latents)
            push!(r06_stake_gate_rows, (; model = arm.label, route = String(route), s...,
                                          pass = s.max_abs_gap <= R06_GATE_TOL))
            @printf("  P2/P2b %-34s %-11s price gap %.1e | stake-side totals gap %.2e %s\n",
                    arm.label, route, g.max_abs_vs_smile, s.max_abs_gap,
                    route === :reweighted ?
                        (s.max_abs_gap <= R06_GATE_TOL ? "PASS" : "FAIL") :
                        "(expected non-zero — this is T011)")
            route === :reweighted && s.max_abs_gap > R06_GATE_TOL && error(
                "P2b FAILED: $label reweighted stakes were solved on a grid whose totals differ " *
                "from the smile by $(s.max_abs_gap)")
        end

        @printf("  %-34s %-11s return %+8.2f%%  ROI %+6.2f%%  Sharpe %6.3f  MDD %7.2f%%  bets %4d (1X2 %d / tot %d)  WR %.1f%%\n",
                arm.label, route, row.total_return_pct, row.roi_pct, row.sharpe_ann,
                row.max_drawdown_pct, row.n_bets, n_1x2, n_tot, row.win_rate_pct)
    end
end
r06_summary = DataFrame(r06_rows)
r06_breakdown = DataFrame(r06_breakdown_rows)
r06_price_gates = DataFrame(r06_price_gate_rows)
r06_stake_gates = DataFrame(r06_stake_gate_rows)

# %%
# ===================================================================
# 6. Gate P1 — reproduction of the pinned baseline
# ===================================================================
r06_base = only(filter(r -> r.model == "m05_joint_grw_baseline" && r.route == "reweighted",
                       eachrow(r06_summary)))
r06_repro_basis = length(r06_panel) == R06_BASELINE_PUBLISHED.n_panel
r06_repro_pass = r06_repro_basis &&
    abs(r06_base.total_return_pct - R06_BASELINE_PUBLISHED.total_return_pct) < 0.1 &&
    abs(r06_base.roi_pct - R06_BASELINE_PUBLISHED.roi_pct) < 0.01 &&
    r06_base.n_bets == R06_BASELINE_PUBLISHED.n_bets
@printf("\n  P1 reproduction: baseline %+.2f%% / ROI %.2f%% / %d bets on %d fixtures vs published %+.1f%% / %.2f%% / %d on %d — %s\n",
        r06_base.total_return_pct, r06_base.roi_pct, r06_base.n_bets, length(r06_panel),
        R06_BASELINE_PUBLISHED.total_return_pct, R06_BASELINE_PUBLISHED.roi_pct,
        R06_BASELINE_PUBLISHED.n_bets, R06_BASELINE_PUBLISHED.n_panel,
        r06_repro_pass ? "REPRODUCED" : r06_repro_basis ? "FAILED" : "NOT COMPARABLE (different panel)")
r06_repro_basis && !r06_repro_pass && error(
    "P1 FAILED: on the published panel the pinned baseline does not reproduce its Option B row")

# %%
# ===================================================================
# 7. What the T011 fix did — route contrast per smile arm
# ===================================================================
r06_route_rows = NamedTuple[]
for arm in r06_arms
    haskey(r06_results, (arm.label, :grid)) || continue
    push!(r06_route_rows, gss_route_contrast(arm.label, r06_results[(arm.label, :grid)],
                                             r06_results[(arm.label, :reweighted)]))
end
r06_routes = DataFrame(r06_route_rows)
if nrow(r06_routes) > 0
    println("\n=== T011: staking off the reweighted grid vs the plain grid ===")
    show(stdout, MIME"text/plain"(), r06_routes; allrows = true, allcols = true)
    println()
end

# %%
# ===================================================================
# 8. Attribution — shared vs exclusive bets, reweighted route
# ===================================================================
# Separates "priced the same bets differently" from "took different bets", against the baseline
# and against the five-strike smile at the same weight.
r06_pair_rows = NamedTuple[]
for (cand, ref) in GSS_PAIRS
    haskey(r06_results, (cand, :reweighted)) && haskey(r06_results, (ref, :reweighted)) || continue
    a = r06_results[(cand, :reweighted)]
    b = r06_results[(ref, :reweighted)]
    append!(r06_pair_rows, gms_pair_sets("$cand − $ref", cand, a.trajectory.bets, ref, b.trajectory.bets))
end
r06_pairs = DataFrame(r06_pair_rows)

# %%
# ===================================================================
# 9. Persistence (P3) — this task's two runs only
# ===================================================================
r06_persisted = NamedTuple[]
r06_db = PostgresStorage(R06_CONFIG.experiment)
for arm in r06_arms
    arm.role == "candidate" || continue
    result = r06_results[(arm.label, :reweighted)]
    portfolio_id = save_portfolio_db(result, arm.run_id, r06_db;
        book_spec = r06_book, policy_spec = r06_policy,
        metadata = (; runner = "r06_portfolio", contract = "option_b_system",
                      odds = "betfair_twa_-20_0", staking_route = "anti_diagonal_reweighted",
                      n_panel = length(r06_panel), task = "016"))
    reloaded = load_portfolio_db(portfolio_id, r06_db)
    reloaded.trajectory.bets == result.trajectory.bets || error(
        "P3 FAILED: $(arm.label) reloaded portfolio ledger differs from the simulated one")
    push!(r06_persisted, (; model = arm.label, model_run_id = string(arm.run_id),
                            portfolio_run_id = string(portfolio_id)))
    println("  P3 persisted ", arm.label, " → portfolio ", portfolio_id, " (ledger reloads identically)")
end

# %%
# ===================================================================
# 10. Final report
# ===================================================================
CSV.write(joinpath(R06_OUT_DIR, "r06_portfolio_summary.csv"), r06_summary)
CSV.write(joinpath(R06_OUT_DIR, "r06_market_breakdown.csv"), r06_breakdown)
CSV.write(joinpath(R06_OUT_DIR, "r06_price_gate.csv"), r06_price_gates)
CSV.write(joinpath(R06_OUT_DIR, "r06_stake_gate.csv"), r06_stake_gates)
nrow(r06_routes) > 0 && CSV.write(joinpath(R06_OUT_DIR, "r06_route_contrast.csv"), r06_routes)
CSV.write(joinpath(R06_OUT_DIR, "r06_pair_attribution.csv"), r06_pairs)
CSV.write(joinpath(R06_OUT_DIR, "r06_persisted.csv"), DataFrame(r06_persisted))

r06_f2 = v -> gph_num(v; digits = 2)
r06_f3 = v -> gph_num(v; digits = 3)
open(joinpath(R06_OUT_DIR, "r06_portfolio_report.md"), "w") do io
    println(io, "# r06 closing-line portfolio — Task 016 (1-parameter smile spine)\n")
    println(io, "Generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"), " at `", R06_GIT, "` on ",
            gethostname(), ". Contract: `MatchDay.option_b_system()`. Book: de-vigged Betfair ",
            "TWA(−20, 0] close. Panel: ", length(r06_panel_ids), " walk-forward → ",
            length(r06_quoted), " quoted → ", length(r06_panel),
            " buildable under both staking routes. Bootstrap B = ", R06_BOOTSTRAP_B, ".\n")
    @printf(io, "P1 reproduction — pinned baseline %+.2f%% / ROI %.2f%% / %d bets on %d fixtures; published %+.1f%% / %.2f%% / %d on %d: **%s**.\n\n",
            r06_base.total_return_pct, r06_base.roi_pct, r06_base.n_bets, length(r06_panel),
            R06_BASELINE_PUBLISHED.total_return_pct, R06_BASELINE_PUBLISHED.roi_pct,
            R06_BASELINE_PUBLISHED.n_bets, R06_BASELINE_PUBLISHED.n_panel,
            r06_repro_pass ? "reproduced" : r06_repro_basis ? "failed" : "not comparable (different panel)")
    println(io, "`route = reweighted` solves stakes on the anti-diagonal reweighted grid (ticket ",
            "T011 fixed); `route = grid` is Task 015's path, kept so its rows stay reproducible. ",
            "Count arms are identical under both and are staked once.\n")

    println(io, "## Headline\n")
    print(io, gph_markdown_table(select(sort(r06_summary, [:route, :model]),
        :route, :model, :n_bets, :n_bets_1x2, :n_bets_totals, :total_return_pct, :roi_pct,
        :sharpe_ann, :calmar, :max_drawdown_pct, :win_rate_pct, :growth_lo, :growth_hi,
        :p_roi_positive, :capture_ratio, :mean_edge_pp);
        formats = Dict(:total_return_pct => r06_f2, :roi_pct => r06_f2, :sharpe_ann => r06_f3,
                       :calmar => r06_f3, :max_drawdown_pct => r06_f2, :win_rate_pct => r06_f2,
                       :growth_lo => v -> gph_signed(v; digits = 4),
                       :growth_hi => v -> gph_signed(v; digits = 4),
                       :p_roi_positive => r06_f3, :capture_ratio => r06_f3, :mean_edge_pp => r06_f2)))

    if nrow(r06_routes) > 0
        println(io, "\n## Ticket T011 — what correct staking changed\n")
        println(io, "Task 015 r07 measured a smile container and its φ-stripped twin staking ",
                "identical ledgers. These rows are the same comparison for the fix: the plain-grid ",
                "route against the reweighted one, same posterior, same panel, same contract.\n")
        print(io, gph_markdown_table(r06_routes;
            formats = Dict(:max_shared_stake_gap => v -> @sprintf("%.2e", v),
                           :roi_grid_pct => r06_f2, :roi_reweighted_pct => r06_f2,
                           :delta_roi_pp => v -> gph_signed(v; digits = 2),
                           :return_grid_pct => r06_f2, :return_reweighted_pct => r06_f2,
                           :delta_return_pp => v -> gph_signed(v; digits = 2))))
    end

    println(io, "\n## P2 reported price and P2b stake-side coherence\n")
    println(io, "P2 is the ledger's `p_model`; P2b is the distribution the Kelly solve read. ",
            "A non-zero P2b gap on the `grid` route is ticket T011 measured, not a failure of ",
            "this runner.\n")
    print(io, gph_markdown_table(r06_price_gates;
        formats = Dict(:max_abs_vs_smile => v -> @sprintf("%.1e", v),
                       :min_abs_vs_grid => v -> gph_num(v; digits = 5))))
    println(io)
    print(io, gph_markdown_table(r06_stake_gates;
        formats = Dict(:max_abs_gap => v -> @sprintf("%.2e", v))))

    println(io, "\n## Market breakdown (both routes)\n")
    print(io, gph_markdown_table(select(sort(r06_breakdown, [:level, :group, :route, :model]),
        :route, :level, :group, :model, :n_bets, :win_rate_pct, :roi_pct, :stake_share_pct,
        :edge_mean_pp, :odds_mean, :capture_ratio);
        formats = Dict(:win_rate_pct => r06_f2, :roi_pct => r06_f2, :stake_share_pct => r06_f2,
                       :edge_mean_pp => r06_f2, :odds_mean => r06_f2, :capture_ratio => r06_f3)))

    println(io, "\n## Shared vs exclusive bets (reweighted route)\n")
    print(io, gph_markdown_table(r06_pairs;
        formats = Dict(:win_rate_pct => r06_f2, :roi_pct => r06_f2, :edge_mean_pp => r06_f2,
                       :capture_ratio => r06_f3)))

    println(io, "\n## Persisted portfolios (this task's runs only)\n")
    print(io, gph_markdown_table(DataFrame(r06_persisted)))
end

println("\nR06_DONE report=", joinpath(R06_OUT_DIR, "r06_portfolio_report.md"))

# ==============================================================================
# r07 — Raw vs calibrated portfolio at the tradeable T−25 book, six arms
# ==============================================================================
#
# WHAT THIS ANSWERS
#
# r06 stakes at the CLOSE — the prices the market pillars were fitted against, so an anchored
# arm is partly scored against its own anchor. The trader's questions are whether the spine keeps
# its ROI at a book one could actually trade (T−25), and whether a fit-time spine adds anything
# beyond what the production L2 rate calibrator already extracts from that same book.
#
#   Environment CLOSE — de-vigged Betfair TWA(−20, 0] close
#     raw                   the r06 reweighted rows, reproduced as gate T1
#
#   Environment T25 — tradeable T−25 point-in-time book
#     raw                   no calibration
#     t25_inv               `MatchDay.option_b_calibrator()` on the count arms
#     t25_inv_pooltot       smile arms: calibrated grid, λ_tot rebuilt from it, fitted φ kept
#     t25_inv_grid          smile arms: calibrated grid only, φ dropped
#
# The two smile variants are Task 015's `l04` header's, unchanged and both run: `calibrate_latents`
# REFUSES a `SmileLatents`, so a calibrated smile is not defined in `src` and the definition
# changes the answer. Running one silently would be picking an answer.
#
# STAKING ROUTE. Every smile container is staked through the anti-diagonal reweighted grid
# (ticket T011 fixed). The route ablation belongs to r06, which measured it on one panel and one
# contract; carrying it through this grid would double the runtime for no new question. So the
# rows here are all on the corrected side of T011 and are NOT comparable with Task 015's r07
# rows, which were all on the defective side. T1 pins the comparison that IS valid: r06's own
# reweighted close rows.
#
# THE CONTRASTS, paired slate by slate (bootstrap of per-slate log-growth differences)
#
#   Q1  spine raw − baseline raw                       at T−25: does r06's lead survive?
#   Q2  spine t25_inv_pooltot − baseline t25_inv       beyond L2, smile kept
#   Q3  spine t25_inv_grid − baseline t25_inv          beyond L2, smile removed
#   Q4  spine raw − baseline t25_inv                   does the pillar substitute for L2?
#   H4  spine − five-strike at the SAME weight, raw and pooltot: one parameter vs five, staked
#   within-arm  pooltot − grid: what φ is worth once the rates are calibrated
#
# GATES
#
#   T1  close/raw reproduces `r06_portfolio_summary.csv`'s reweighted rows (return and bet count).
#   T2  t25 baseline reproduces Task 014 §9 (raw +531.78% / 1,124 bets; t25_inv +245.85% /
#       969 bets) when the buildable panel is Task 014's 611 fixtures; otherwise reported as not
#       comparable, never claimed.
#   T3  every staked totals bet of every smile ledger is priced through λ_tot·φ(K) (≤ 1e-9), AND
#       the distribution its stake was solved on implies the same totals CDF (≤ 1e-9).
#
# One panel per environment; bankroll figures are comparable within an environment only.
#
# USAGE (mcmc-beast, from /root/BF_grw_smile_spine, after r06)
#
#   julia --project -t 16 current_development/grw_smile_spine/r07_t25_portfolio.jl
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
const R07_CONFIG = gss_config()
const R07_BOOTSTRAP_B = 4000
const R07_PAIRED_B = 10_000
const R07_SEED = 1
const R07_GATE_TOL = 1.0e-9
const R07_OUT_DIR = joinpath(R07_CONFIG.save_root, "t25_portfolio")
const R07_LATENT_DIRS = [joinpath(R07_CONFIG.save_root, "latents"),
                         joinpath(@__DIR__, "..", "grw_market_smile", "results", "latents")]
const R07_R06_SUMMARY = joinpath(R07_CONFIG.save_root, "portfolio", "r06_portfolio_summary.csv")
const R07_GIT = try
    readchomp(`git rev-parse --short HEAD`)
catch
    "unknown"
end

# Task 014 §9 — the same pinned baseline under the same contract on its 611-fixture T−25 panel.
const R07_TASK014_T25 = (n_panel = 611,
                         raw = (total_return_pct = 531.7811, n_bets = 1124),
                         t25_inv = (total_return_pct = 245.8506, n_bets = 969))

const R07_BASELINE = "m05_joint_grw_baseline"
const R07_SPINES = copy(GSS_GRID_MODEL_NAMES)

mkpath(R07_OUT_DIR)
println("\n" * "="^96)
println("  r07 RAW vs CALIBRATED PORTFOLIO — close and tradeable T−25, six arms")
println("  staking    : anti-diagonal reweighted for every smile container (T011 fixed)")
println("  git        : ", R07_GIT, "   host ", gethostname(), "   threads ", Threads.nthreads())
println("="^96)

# %%
# ===================================================================
# 3. Data, contract, arms
# ===================================================================
r07_ds = gph_load_data()
r07_splitter = gph_splitter(R07_CONFIG.extension_seasons)
r07_book, r07_policy = gms_option_b()
r07_cal = GMS_MD.option_b_calibrator()
println("  book       : ", r07_book)
println("  policy     : ", r07_policy)
println("  calibrator : ", r07_cal.name, "  ", GMS_CAL.law_label(r07_cal.law),
        "  as_of ", r07_cal.book_as_of_minutes)

r07_arms = gss_arms(R07_CONFIG)
r07_raw_fits = gss_load_arms(r07_arms, r07_ds; splitter = r07_splitter, latent_dirs = R07_LATENT_DIRS)
r07_panel_ids = gms_common_panel(r07_ds, r07_raw_fits, R07_CONFIG.target_seasons)
length(r07_panel_ids) == R07_CONFIG.expected_oos || error(
    "common panel is $(length(r07_panel_ids)) fixtures; expected $(R07_CONFIG.expected_oos)")
r07_fits = Dict(label => gms_restrict(fit, r07_panel_ids) for (label, fit) in r07_raw_fits)
r07_raw_fits = nothing
GC.gc()

# %%
# ===================================================================
# 4. The two market environments
# ===================================================================
r07_close_odds = gms_betfair_closing_odds(r07_ds)
r07_t25_odds, r07_t25_refusals = gms_t25_book(r07_ds)
let cov = GMS_CAL.book_coverage(r07_t25_odds, r07_t25_refusals)
    @printf("  t25 book   : %d rows | %d fixtures | staleness med %.0f p90 %.0f min | overround %.4f | %d refused markets\n",
            cov.n_rows, cov.n_fixtures, cov.median_staleness, cov.p90_staleness,
            cov.median_overround, cov.n_refused_markets)
end

const R07_ENVIRONMENTS = [
    (; key = "close", odds = r07_close_odds, calibrate = false,
       label = "de-vigged Betfair TWA(−20, 0] close"),
    (; key = "t25", odds = r07_t25_odds, calibrate = true,
       label = "tradeable T−25 point-in-time book"),
]

# %%
# ===================================================================
# 5. Panels, inversion and the simulation grid
# ===================================================================
r07_rows = NamedTuple[]
r07_breakdown_rows = NamedTuple[]
r07_weight_rows = NamedTuple[]
r07_panel_rows = NamedTuple[]
r07_gate3_rows = NamedTuple[]
r07_results = Dict{Tuple{String,String,String},Any}()   # (env, variant, model)

for env in R07_ENVIRONMENTS
    println("\n" * "-"^96)
    println("  ENVIRONMENT ", uppercase(env.key), " — ", env.label)
    println("-"^96)
    quoted = sort!(collect(intersect(Set(r07_panel_ids), Set(Int.(env.odds.match_id)))))
    env_odds = filter(:match_id => in(Set(quoted)), DataFrame(env.odds))
    panel, dropped = gss_buildable_panel(r07_book, r07_fits, env_odds, r07_ds, quoted;
                                         route = :reweighted)
    CSV.write(joinpath(R07_OUT_DIR, "r07_dropped_$(env.key).csv"), dropped)
    @printf("  panel  : %d walk-forward → %d quoted → %d buildable by every arm (%d dropped)\n",
            length(r07_panel_ids), length(quoted), length(panel), nrow(dropped))
    push!(r07_panel_rows, (; environment = env.key, n_walk_forward = length(r07_panel_ids),
                             n_quoted = length(quoted), n_buildable = length(panel),
                             n_dropped = nrow(dropped)))

    rates = nothing
    if env.calibrate
        rates = GMS_CAL.invert_market_rates(r07_cal, env.odds; match_ids = panel)
        icov = GMS_CAL.inversion_coverage(rates, panel)
        @printf("  invert : %d/%d panel fixtures accepted (%.1f%% of quoted)\n",
                icov.n_accepted, icov.n_fixtures, 100 * icov.coverage_quoted)
        CSV.write(joinpath(R07_OUT_DIR, "r07_inversion_$(env.key).csv"), GMS_CAL.inversion_frame(rates))
    end

    for arm in r07_arms
        fit = gms_restrict(r07_fits[arm.label], panel)
        # `Any[...]`: a literal vector would fix `diagnostics::Nothing` from the raw entry and
        # refuse the calibrated entries appended after it (Task 015 r07 hit exactly this).
        sources = Any[(; variant = "raw", route = gss_is_smile(fit) ? "smile" : "native",
                         source = fit, latents = fit.latents, diagnostics = nothing)]
        env.calibrate && append!(sources, gms_calibrated_sources(r07_cal, fit, env.odds, rates))

        for src in sources
            label = string(arm.label, "/", env.key, "/", src.variant)
            books, report = gss_build_books(r07_book, src.source, env_odds, r07_ds, panel;
                                            label, route = :reweighted)
            result = gms_run_policy(r07_policy, books, report; B = R07_BOOTSTRAP_B, seed = R07_SEED)
            r07_results[(env.key, src.variant, arm.label)] = result
            bets = result.trajectory.bets
            row = gms_portfolio_row(arm.label, result; n_panel = length(panel))
            breakdown = gms_breakdown(arm.label, bets)
            n_1x2 = sum((r.n_bets for r in breakdown if r.level == "market" && r.group == "1X2"); init = 0)
            n_tot = sum((r.n_bets for r in breakdown if r.level == "market" && r.group == "totals"); init = 0)
            push!(r07_rows, (; environment = env.key, variant = src.variant,
                               calibration = gms_calibration_family(src.variant), route = src.route,
                               row..., n_bets_1x2 = n_1x2, n_bets_totals = n_tot,
                               capture_ratio = gms_edge_summary(bets).capture_ratio,
                               run_id = string(arm.run_id)))
            for b in breakdown
                push!(r07_breakdown_rows, (; environment = env.key, variant = src.variant, b...))
            end

            if src.diagnostics !== nothing
                ws = GMS_CAL.weight_summary(src.diagnostics)
                push!(r07_weight_rows, (; environment = env.key, variant = src.variant,
                                          model = arm.label, n_shifted = ws.n_shifted,
                                          w_median = ws.w_median, w_p10 = ws.w_p10, w_p90 = ws.w_p90,
                                          var_retention_median = ws.var_retention_median,
                                          market_share_median = ws.market_share_median))
            end

            # --- T3: reported price AND stake-side coherence --------------------------
            if src.latents isa SmileLatents
                g = gms_smile_book_gate(bets, src.latents)
                s = gss_book_totals_gate(books, src.latents)
                push!(r07_gate3_rows, (; environment = env.key, variant = src.variant,
                                         model = arm.label, g...,
                                         stake_max_abs_gap = s.max_abs_gap,
                                         stake_per_strike_gap = s.per_strike_gap))
                g.n_totals_bets == 0 || g.max_abs_vs_smile <= R07_GATE_TOL || error(
                    "T3 FAILED: $label totals priced up to $(g.max_abs_vs_smile) from λ_tot·φ(K)")
                s.max_abs_gap <= R07_GATE_TOL || error(
                    "T3 FAILED: $label stakes were solved on a grid whose totals differ from the " *
                    "smile by $(s.max_abs_gap)")
            end

            @printf("  %-34s %-16s return %+9.2f%%  ROI %+6.2f%%  Sharpe %6.3f  MDD %7.2f%%  bets %4d (1X2 %d / tot %d)\n",
                    arm.label, src.variant, row.total_return_pct, row.roi_pct, row.sharpe_ann,
                    row.max_drawdown_pct, row.n_bets, n_1x2, n_tot)
        end
    end
end

r07_summary = DataFrame(r07_rows)
r07_breakdown = DataFrame(r07_breakdown_rows)
r07_weights = DataFrame(r07_weight_rows)
r07_panels = DataFrame(r07_panel_rows)
r07_gate3 = DataFrame(r07_gate3_rows)

# %%
# ===================================================================
# 6. Gates T1 and T2
# ===================================================================
r07_gate_rows = NamedTuple[]
if isfile(R07_R06_SUMMARY)
    r06 = filter(:route => ==("reweighted"), CSV.read(R07_R06_SUMMARY, DataFrame))
    for r in eachrow(filter(r -> r.environment == "close" && r.variant == "raw", r07_summary))
        j = findfirst(==(r.model), r06.model)
        j === nothing && continue
        push!(r07_gate_rows, (; gate = "T1 close/raw vs r06", model = r.model,
                                return_pct = r.total_return_pct, reference_pct = r06.total_return_pct[j],
                                delta_pp = r.total_return_pct - r06.total_return_pct[j],
                                n_bets = r.n_bets, reference_bets = r06.n_bets[j]))
    end
    t1 = filter(g -> startswith(g.gate, "T1"), r07_gate_rows)
    worst = isempty(t1) ? NaN : maximum(abs(g.delta_pp) for g in t1)
    bad = count(g -> g.n_bets != g.reference_bets, t1)
    @printf("\n  T1 close/raw vs r06: %d arms, worst |Δ return| %.2e pp, %d bet-count mismatches\n",
            length(t1), worst, bad)
    (length(t1) == length(r07_arms) && worst < 1e-6 && bad == 0) || error(
        "T1 FAILED: the close/raw rows do not reproduce r06's reweighted rows")
else
    println("\n  T1 skipped — ", R07_R06_SUMMARY, " not present (run r06 first)")
end

r07_t25_panel = only(filter(:environment => ==("t25"), r07_panels)).n_buildable
r07_t2_comparable = r07_t25_panel == R07_TASK014_T25.n_panel
for (variant, ref) in (("raw", R07_TASK014_T25.raw), ("t25_inv", R07_TASK014_T25.t25_inv))
    hits = filter(x -> x.environment == "t25" && x.variant == variant && x.model == R07_BASELINE,
                  r07_summary)
    nrow(hits) == 1 || continue
    r = only(hits)
    push!(r07_gate_rows, (; gate = "T2 t25/$variant vs Task 014", model = R07_BASELINE,
                            return_pct = r.total_return_pct, reference_pct = ref.total_return_pct,
                            delta_pp = r.total_return_pct - ref.total_return_pct,
                            n_bets = r.n_bets, reference_bets = ref.n_bets))
    ok = abs(r.total_return_pct - ref.total_return_pct) < 0.01 && r.n_bets == ref.n_bets
    @printf("  T2 t25/%-8s baseline %+.2f%% / %d bets vs Task 014 %+.2f%% / %d on %d fixtures (here %d): %s\n",
            variant, r.total_return_pct, r.n_bets, ref.total_return_pct, ref.n_bets,
            R07_TASK014_T25.n_panel, r07_t25_panel,
            r07_t2_comparable ? (ok ? "REPRODUCED" : "FAILED") : "NOT COMPARABLE (different panel)")
    r07_t2_comparable && !ok && error("T2 FAILED: t25/$variant baseline does not reproduce Task 014")
end
r07_gates = DataFrame(r07_gate_rows)

# %%
# ===================================================================
# 7. Paired contrasts
# ===================================================================
r07_specs = NamedTuple[]
for spine in R07_SPINES
    weight = endswith(spine, "w020") ? "w020" : "w040"
    five = "m05_joint_grw_smile_supremacy_$weight"
    push!(r07_specs,
          (; question = "Q1 raw lead at T−25", env = "t25", a = (spine, "raw"), b = (R07_BASELINE, "raw")),
          (; question = "Q2 beyond L2, smile kept", env = "t25", a = (spine, "t25_inv_pooltot"), b = (R07_BASELINE, "t25_inv")),
          (; question = "Q3 beyond L2, smile dropped", env = "t25", a = (spine, "t25_inv_grid"), b = (R07_BASELINE, "t25_inv")),
          (; question = "Q4 pillar instead of L2", env = "t25", a = (spine, "raw"), b = (R07_BASELINE, "t25_inv")),
          (; question = "H4 spine − five-strike, raw", env = "t25", a = (spine, "raw"), b = (five, "raw")),
          (; question = "H4 spine − five-strike, pooltot", env = "t25", a = (spine, "t25_inv_pooltot"), b = (five, "t25_inv_pooltot")),
          (; question = "within arm: pooltot − grid", env = "t25", a = (spine, "t25_inv_pooltot"), b = (spine, "t25_inv_grid")),
          (; question = "close reference (r06)", env = "close", a = (spine, "raw"), b = (R07_BASELINE, "raw")))
end
push!(r07_specs,
      (; question = "Q1 raw lead at T−25", env = "t25", a = ("m05_joint_grw_supremacy_w040", "raw"), b = (R07_BASELINE, "raw")),
      (; question = "Q2 beyond L2", env = "t25", a = ("m05_joint_grw_supremacy_w040", "t25_inv"), b = (R07_BASELINE, "t25_inv")))

r07_contrast_rows = NamedTuple[]
r07_pairset_rows = NamedTuple[]
for c in r07_specs
    haskey(r07_results, (c.env, c.a[2], c.a[1])) && haskey(r07_results, (c.env, c.b[2], c.b[1])) || continue
    ra = r07_results[(c.env, c.a[2], c.a[1])]
    rb = r07_results[(c.env, c.b[2], c.b[1])]
    p = gms_paired_growth(ra, rb; B = R07_PAIRED_B)
    pair = "$(c.a[1])[$(c.a[2])] − $(c.b[1])[$(c.b[2])]"
    push!(r07_contrast_rows, (; question = c.question, environment = c.env, pair,
                                return_a_pct = ra.summary.total_return_pct,
                                return_b_pct = rb.summary.total_return_pct,
                                roi_a_pct = ra.summary.roi, roi_b_pct = rb.summary.roi,
                                delta_roi_pp = ra.summary.roi - rb.summary.roi, p...))
    append!(r07_pairset_rows, [(; environment = c.env, question = c.question, r...)
                               for r in gms_pair_sets(pair, "a", ra.trajectory.bets, "b", rb.trajectory.bets)])
    @printf("  %-32s %-6s %-76s ΔROI %+6.2f pp  Δlog-growth/slate %+.5f [%+.5f, %+.5f]  p %.3f\n",
            c.question, c.env, pair, ra.summary.roi - rb.summary.roi,
            p.delta_log_growth_per_slate, p.lo, p.hi, p.p_better)
end
r07_contrasts = DataFrame(r07_contrast_rows)
r07_pairsets = DataFrame(r07_pairset_rows)

# %%
# ===================================================================
# 8. Final report
# ===================================================================
sort!(r07_summary, [:environment, :calibration, :model, :variant])
CSV.write(joinpath(R07_OUT_DIR, "r07_portfolio_summary.csv"), r07_summary)
CSV.write(joinpath(R07_OUT_DIR, "r07_market_breakdown.csv"), r07_breakdown)
CSV.write(joinpath(R07_OUT_DIR, "r07_calibration_weights.csv"), r07_weights)
CSV.write(joinpath(R07_OUT_DIR, "r07_panels.csv"), r07_panels)
CSV.write(joinpath(R07_OUT_DIR, "r07_gates.csv"), r07_gates)
CSV.write(joinpath(R07_OUT_DIR, "r07_smile_gate.csv"), r07_gate3)
CSV.write(joinpath(R07_OUT_DIR, "r07_paired_contrasts.csv"), r07_contrasts)
CSV.write(joinpath(R07_OUT_DIR, "r07_pair_sets.csv"), r07_pairsets)

r07_f2 = v -> gph_num(v; digits = 2)
r07_f3 = v -> gph_num(v; digits = 3)
r07_s5 = v -> gph_signed(v; digits = 5)
open(joinpath(R07_OUT_DIR, "r07_t25_portfolio_report.md"), "w") do io
    println(io, "# r07 raw vs calibrated portfolio at T−25 — Task 016\n")
    println(io, "Generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"), " at `", R07_GIT, "` on ",
            gethostname(), ". Contract `MatchDay.option_b_system()` for every row. Calibrator `",
            r07_cal.name, "` (", GMS_CAL.law_label(r07_cal.law), "). Every smile container is ",
            "staked through the anti-diagonal reweighted grid (T011), so these rows are NOT ",
            "comparable with Task 015's r07. Bankroll figures are comparable within an ",
            "environment only.\n")
    print(io, gph_markdown_table(r07_panels))
    println(io, "\n## Gates\n")
    print(io, gph_markdown_table(r07_gates; formats = Dict(:return_pct => r07_f2,
        :reference_pct => r07_f2, :delta_pp => v -> @sprintf("%.2e", v))))
    println(io, "\nT2 comparable: ", r07_t2_comparable, " (T−25 panel ", r07_t25_panel,
            ", Task 014 ", R07_TASK014_T25.n_panel, ").\n")
    if nrow(r07_gate3) > 0
        println(io, "T3 — reported price (`max_abs_vs_smile`) and stake side (`stake_max_abs_gap`):\n")
        print(io, gph_markdown_table(r07_gate3; formats = Dict(
            :max_abs_vs_smile => v -> @sprintf("%.1e", v),
            :min_abs_vs_grid => v -> gph_num(v; digits = 5),
            :stake_max_abs_gap => v -> @sprintf("%.2e", v))))
    end
    println(io, "\n## Headline\n")
    print(io, gph_markdown_table(select(r07_summary,
        :environment, :calibration, :variant, :route, :model, :n_bets, :n_bets_1x2, :n_bets_totals,
        :total_return_pct, :roi_pct, :sharpe_ann, :max_drawdown_pct, :win_rate_pct,
        :growth_lo, :growth_hi, :mean_edge_pp);
        formats = Dict(:total_return_pct => r07_f2, :roi_pct => r07_f2, :sharpe_ann => r07_f3,
                       :max_drawdown_pct => r07_f2, :win_rate_pct => r07_f2,
                       :growth_lo => v -> gph_signed(v; digits = 4),
                       :growth_hi => v -> gph_signed(v; digits = 4), :mean_edge_pp => r07_f2)))
    println(io, "\n## What the calibrator did\n")
    print(io, gph_markdown_table(r07_weights; formats = Dict(:w_median => r07_f3, :w_p10 => r07_f3,
        :w_p90 => r07_f3, :var_retention_median => r07_f3, :market_share_median => r07_f3)))
    println(io, "\n## Paired contrasts (slate-level log-growth bootstrap, B = ", R07_PAIRED_B, ")\n")
    println(io, "Δ is a − b per slate; a slate one arm did not stake counts 0 for it. ",
            "`p_better` is the share of resamples with Δ > 0.\n")
    print(io, gph_markdown_table(select(r07_contrasts,
        :question, :environment, :pair, :return_a_pct, :return_b_pct, :roi_a_pct, :roi_b_pct,
        :delta_roi_pp, :n_slates, :delta_log_growth_per_slate, :lo, :hi, :p_better);
        formats = Dict(:return_a_pct => r07_f2, :return_b_pct => r07_f2, :roi_a_pct => r07_f2,
                       :roi_b_pct => r07_f2, :delta_roi_pp => v -> gph_signed(v; digits = 2),
                       :delta_log_growth_per_slate => r07_s5, :lo => r07_s5, :hi => r07_s5,
                       :p_better => r07_f3)))
    println(io, "\n## Market breakdown\n")
    print(io, gph_markdown_table(select(sort(filter(:level => ==("market"), r07_breakdown),
            [:environment, :group, :variant, :model]),
        :environment, :variant, :group, :model, :n_bets, :win_rate_pct, :roi_pct, :stake_share_pct,
        :edge_mean_pp);
        formats = Dict(:win_rate_pct => r07_f2, :roi_pct => r07_f2, :stake_share_pct => r07_f2,
                       :edge_mean_pp => r07_f2)))
    println(io, "\n## By selection family\n")
    print(io, gph_markdown_table(select(sort(filter(:level => ==("family"), r07_breakdown),
            [:environment, :group, :variant, :model]),
        :environment, :variant, :group, :model, :n_bets, :win_rate_pct, :roi_pct, :stake_share_pct,
        :edge_mean_pp);
        formats = Dict(:win_rate_pct => r07_f2, :roi_pct => r07_f2, :stake_share_pct => r07_f2,
                       :edge_mean_pp => r07_f2)))
    println(io, "\n## Shared vs exclusive bets per contrast\n")
    print(io, gph_markdown_table(r07_pairsets; formats = Dict(:win_rate_pct => r07_f2,
        :roi_pct => r07_f2, :edge_mean_pp => r07_f2, :capture_ratio => r07_f3)))
end

println("\nR07_DONE report=", joinpath(R07_OUT_DIR, "r07_t25_portfolio_report.md"))

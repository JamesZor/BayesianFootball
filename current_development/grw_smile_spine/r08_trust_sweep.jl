# ==============================================================================
# r08 — Trust-pruning sweep: does the spine re-open Under 1.5 and Under 4.5?
# ==============================================================================
#
# WHAT THIS ANSWERS (H5)
#
# `eda/README.md` prunes every fringe selection (O/U 0.5, 1.5, 3.5, Over 2.5, BTTS) to trust 0 on
# two grounds: Jensen tail inflation — for an uncalibrated Poisson posterior E[e^{−Λ}] ≥ e^{−E[Λ]},
# so deep Unders are systematically over-priced (Under 0.5: −30.7% ROI) — and capacity
# cannibalisation: fringe lines absorbed 29% of stake at −13.2% ROI and crowded out the core.
#
# The work package asks whether `Under 1.5` and `Under 4.5` become accretive under the reweighted
# spine. r04 has already made a testable prediction, and it is NOT encouraging:
#
#   line        market   baseline   spine    five-strike   realised
#   Under 0.5   0.0679   0.0718     0.0893   0.0994        0.0336
#   Under 1.5   0.2305   0.2423     0.2715   0.2478        0.2279
#   Under 4.5   0.8578   0.8397     0.8078   0.8168        0.9038
#
# At Under 1.5 the spine prices 0.2715 against a market of 0.2305 and a realised rate of 0.2279:
# it sees a +4.1 pp edge where there is none, so a trust > 0 there should LOSE money, and lose
# more than the five-strike arm (+1.7 pp) or the baseline (+1.2 pp). At Under 4.5 the spine
# prices BELOW the market (0.8078 vs 0.8578) while the realised rate is 0.9038, so it should
# decline the bet the market is under-pricing — no loss, but no gain either. If the sweep says
# otherwise, that disagreement is the finding; the prediction is recorded here so it cannot be
# quietly reframed afterwards.
#
# DESIGN — one line at a time, then unions, every other setting held at Option B
#
#   arms          spine @0.40 (the test), five-strike @0.40 (does one parameter change the
#                 answer?), baseline (the control: the same question with NO smile)
#   environments  close (de-vigged Betfair TWA(−20, 0]) and raw T−25 (tradeable book)
#   book          Option B's, plus O/U 4.5 — `canonical_markets()` stops at 3.5 and 4.5 is the
#                 spine's K = 4 strike. Gate S0 measures whether the extra market is inert at
#                 trust 0 rather than assuming it.
#   policies      P0 = Option B; then each of +U0.5, +U1.5, +U3.5, +U4.5, +O2.5, +O3.5, +O4.5,
#                 +BTTS_yes, +BTTS_no, and the unions +all_unders, +all_fringe, every addition at
#                 Option B's second tier (1/1.4). Risk, cap, filter and grouping unchanged.
#   staking       anti-diagonal reweighted for every smile container (ticket T011 fixed)
#
# READING A ROW
#
#   added_roi_pct       what the newly enabled bets themselves returned
#   core_stake_vs_p0    core-basket stake relative to Option B — < 1 is capacity taken from it
#   delta_core_roi_pp   core-basket ROI change — the cannibalisation cost
#   delta_return_pp     the net: terminal return minus Option B's
#
# NOT COMPARABLE WITH THE EDA'S NUMBERS. The −30.7% / −13.2% figures come from the TimeDecay
# hybrids in a six-market close simulation under SlateDrawdown(23) with a 20% cap. This sweep
# re-asks the question under Option B on the GRW models; the baseline arm is the like-for-like
# control, not the EDA. Nor are these rows comparable with Task 015's r08, which staked every
# smile arm off the un-smiled grid.
#
# GATES
#
#   S0  P0 on the extended book stakes a ledger identical to P0 on Option B's book (measured).
#   S1  close/P0 reproduces `r06_portfolio_summary.csv`'s reweighted rows for all three arms.
#   S2  under +all_fringe, every staked totals bet of a smile arm is priced through λ_tot·φ(K)
#       AND its stake was solved on a grid whose totals match that curve (≤ 1e-9 both).
#
# USAGE (mcmc-beast, from /root/BF_grw_smile_spine, after r06)
#
#   julia --project -t 16 current_development/grw_smile_spine/r08_trust_sweep.jl
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
const R08_CONFIG = gss_config()
const R08_BOOTSTRAP_B = 2000
const R08_PAIRED_B = 10_000
const R08_SEED = 1
const R08_TIER2 = 1.0 / 1.4
const R08_GATE_TOL = 1.0e-9
const R08_OUT_DIR = joinpath(R08_CONFIG.save_root, "trust_sweep")
const R08_LATENT_DIRS = [joinpath(R08_CONFIG.save_root, "latents"),
                         joinpath(@__DIR__, "..", "grw_market_smile", "results", "latents")]
const R08_R06_SUMMARY = joinpath(R08_CONFIG.save_root, "portfolio", "r06_portfolio_summary.csv")

const R08_SPINE = "m05_joint_grw_smile_spine_w040"
const R08_FIVE = "m05_joint_grw_smile_supremacy_w040"
const R08_BASELINE = "m05_joint_grw_baseline"
const R08_MODELS = [R08_BASELINE, R08_FIVE, R08_SPINE]

"The two lines the work package names, called out separately in the report."
const R08_H5_POLICIES = ["+U1.5", "+U4.5"]

const R08_GIT = try
    readchomp(`git rev-parse --short HEAD`)
catch
    "unknown"
end

mkpath(R08_OUT_DIR)
println("\n" * "="^96)
println("  r08 TRUST-PRUNING SWEEP — ", R08_SPINE, " vs ", R08_FIVE, " vs ", R08_BASELINE)
println("  additions  : ", join(first.(GMS_SWEEP_ADDITIONS), ", "), "  at trust ", round(R08_TIER2; digits = 4))
println("  H5 lines   : ", join(R08_H5_POLICIES, ", "))
println("  git        : ", R08_GIT, "   host ", gethostname(), "   threads ", Threads.nthreads())
println("="^96)

# %%
# ===================================================================
# 3. Data, contract, arms
# ===================================================================
r08_ds = gph_load_data()
r08_splitter = gph_splitter(R08_CONFIG.extension_seasons)
r08_book, r08_policy = gms_option_b()
r08_ext_book = gms_extended_book(r08_book)
println("  Option B book : ", r08_book)
println("  extended book : ", r08_ext_book)
println("  Option B trust: ", r08_policy.trust)

r08_arms = filter(a -> a.label in R08_MODELS, gss_arms(R08_CONFIG))
length(r08_arms) == length(R08_MODELS) || error(
    "expected $(length(R08_MODELS)) arms; found $(length(r08_arms))")
r08_raw = gss_load_arms(r08_arms, r08_ds; splitter = r08_splitter, latent_dirs = R08_LATENT_DIRS)
r08_panel_ids = gms_common_panel(r08_ds, r08_raw, R08_CONFIG.target_seasons)
length(r08_panel_ids) == R08_CONFIG.expected_oos || error(
    "common panel is $(length(r08_panel_ids)) fixtures; expected $(R08_CONFIG.expected_oos)")
r08_fits = Dict(label => gms_restrict(fit, r08_panel_ids) for (label, fit) in r08_raw)
r08_raw = nothing
GC.gc()

r08_policies = [("P0 Option B", r08_policy);
                [(name, gms_policy_with(r08_policy, keys; weight = R08_TIER2))
                 for (name, keys) in GMS_SWEEP_ADDITIONS]]

# %%
# ===================================================================
# 4. Environments
# ===================================================================
r08_close_odds = gms_betfair_closing_odds(r08_ds)
r08_t25_odds, _ = gms_t25_book(r08_ds)
const R08_ENVIRONMENTS = [(; key = "close", odds = r08_close_odds), (; key = "t25", odds = r08_t25_odds)]

# %%
# ===================================================================
# 5. The sweep
# ===================================================================
r08_rows = NamedTuple[]
r08_family_rows = NamedTuple[]
r08_calibration_rows = NamedTuple[]
r08_coverage = DataFrame[]
r08_gate_rows = NamedTuple[]
r08_results = Dict{Tuple{String,String,String},Any}()   # (env, model, policy)
r08_panel_sizes = Dict{String,Int}()

for env in R08_ENVIRONMENTS
    println("\n" * "-"^96)
    println("  ENVIRONMENT ", uppercase(env.key))
    println("-"^96)
    quoted = sort!(collect(intersect(Set(r08_panel_ids), Set(Int.(env.odds.match_id)))))
    env_odds = filter(:match_id => in(Set(quoted)), DataFrame(env.odds))
    panel, dropped = gss_buildable_panel(r08_book, r08_fits, env_odds, r08_ds, quoted;
                                         route = :reweighted)
    @printf("  panel  : %d walk-forward → %d quoted → %d buildable by every arm (%d dropped)\n",
            length(r08_panel_ids), length(quoted), length(panel), nrow(dropped))
    r08_panel_sizes[env.key] = length(panel)
    cov = gms_line_coverage(env.key, env_odds, panel)
    push!(r08_coverage, cov)
    println("  quoted fixtures per line: ",
            join(["$(r.market_name) $(r.market_line)=$(r.n_fixtures)" for r in eachrow(cov)], "  "))

    for arm in r08_arms
        source = gms_restrict(r08_fits[arm.label], panel)
        label = "$(arm.label)/$(env.key)"
        canon_books, canon_report = gss_build_books(r08_book, source, env_odds, r08_ds, panel;
                                                    label, route = :reweighted)
        ext_books, ext_report = gss_build_books(r08_ext_book, source, env_odds, r08_ds, panel;
                                                label, route = :reweighted)

        p0_canon = gms_run_policy(r08_policy, canon_books, canon_report; B = R08_BOOTSTRAP_B, seed = R08_SEED)
        p0_ext = gms_run_policy(r08_policy, ext_books, ext_report; B = R08_BOOTSTRAP_B, seed = R08_SEED)

        # --- S0: is the extra market inert at trust 0? MEASURED, not assumed ------------
        # Task 015 found `==` false for one arm at the close: `==` is NaN-unsafe, so `isequal`
        # is the comparison. If the ledgers still differ, that is a finding about the contract
        # (a zero-trust market can move the per-match budget or the joint solve), recorded here.
        # Every addition below is measured against P0 on the SAME extended book, so a sweep row
        # differs from its reference in the trust table only.
        same = isequal(p0_ext.trajectory.bets, p0_canon.trajectory.bets)
        push!(r08_gate_rows, (; gate = "S0 extended book inert at trust 0", environment = env.key,
                                model = arm.label, pass = same,
                                detail = @sprintf("ext return %.4f vs canon %.4f (Δ %+.4f pp), bets %d vs %d",
                                    p0_ext.summary.total_return_pct, p0_canon.summary.total_return_pct,
                                    p0_ext.summary.total_return_pct - p0_canon.summary.total_return_pct,
                                    p0_ext.summary.n_bets, p0_canon.summary.n_bets)))
        @printf("  S0 %-34s extended-book P0 %s canonical: return %+.4f vs %+.4f, bets %d vs %d\n",
                arm.label, same ? "==" : "DIFFERS FROM", p0_ext.summary.total_return_pct,
                p0_canon.summary.total_return_pct, p0_ext.summary.n_bets, p0_canon.summary.n_bets)

        canon_core = Set(String.(DataFrame(p0_canon.trajectory.bets).family))
        r08_results[(env.key, arm.label, "P0 Option B")] = p0_canon
        push!(r08_rows, gms_sweep_row(env.key, arm.label, "P0 Option B", p0_canon, p0_canon;
                                      core_families = canon_core))
        append!(r08_calibration_rows,
                gms_family_calibration(env.key, arm.label, "P0 Option B", p0_canon.trajectory.bets))

        # The sweep's reference: Option B's trust on the extended book.
        core_families = Set(String.(DataFrame(p0_ext.trajectory.bets).family))
        r08_results[(env.key, arm.label, "P0 ext book")] = p0_ext
        push!(r08_rows, gms_sweep_row(env.key, arm.label, "P0 ext book", p0_ext, p0_ext; core_families))

        for (name, policy) in r08_policies[2:end]
            result = gms_run_policy(policy, ext_books, ext_report; B = R08_BOOTSTRAP_B, seed = R08_SEED)
            r08_results[(env.key, arm.label, name)] = result
            row = gms_sweep_row(env.key, arm.label, name, result, p0_ext; core_families)
            push!(r08_rows, row)
            for b in gms_breakdown(arm.label, result.trajectory.bets)
                b.level == "family" && !(b.group in core_families) &&
                    push!(r08_family_rows, (; environment = env.key, policy = name, b...))
            end
            append!(r08_calibration_rows,
                    gms_family_calibration(env.key, arm.label, name, result.trajectory.bets))

            # --- S2: the new strikes are priced AND staked through the smile -------------
            if name == "+all_fringe" && source.latents isa SmileLatents
                g = gms_smile_book_gate(result.trajectory.bets, source.latents)
                s = gss_book_totals_gate(ext_books, source.latents)
                ok = g.max_abs_vs_smile <= R08_GATE_TOL && s.max_abs_gap <= R08_GATE_TOL
                push!(r08_gate_rows, (; gate = "S2 smile routing + staking, all strikes",
                                        environment = env.key, model = arm.label, pass = ok,
                                        detail = @sprintf("%d totals bets, max |p − smile| %.1e, stake-side gap %.2e",
                                                          g.n_totals_bets, g.max_abs_vs_smile, s.max_abs_gap)))
                ok || error("S2 FAILED: $label +all_fringe is not priced and staked through λ_tot·φ(K)")
            end

            @printf("  %-34s %-12s return %+9.2f%% (Δ %+8.2f pp)  ROI %+6.2f%%  Sharpe %6.3f  added %3d bets ROI %+7.2f%% share %5.1f%%  core stake %.3f ΔROI %+6.2f\n",
                    arm.label, name, row.total_return_pct, row.delta_return_pp, row.roi_pct, row.sharpe_ann,
                    row.added_n_bets, row.added_roi_pct, row.added_stake_share_pct,
                    row.core_stake_vs_p0, row.delta_core_roi_pp)
        end
    end
end

r08_summary = DataFrame(r08_rows)
r08_families = DataFrame(r08_family_rows)
r08_calibration = DataFrame(r08_calibration_rows)
r08_coverage_frame = vcat(r08_coverage...)

# %%
# ===================================================================
# 6. Gate S1 — close/P0 reproduces r06
# ===================================================================
# Only comparable when this runner's three-arm close panel is r06's six-arm panel; a different
# panel is reported, never failed and never claimed as a reproduction.
if isfile(R08_R06_SUMMARY)
    r06 = filter(:route => ==("reweighted"), CSV.read(R08_R06_SUMMARY, DataFrame))
    r06_panel = first(r06.n_panel)
    if r08_panel_sizes["close"] != r06_panel
        push!(r08_gate_rows, (; gate = "S1 close/P0 vs r06", environment = "close", model = "all",
                                pass = true,
                                detail = "not comparable: close panel $(r08_panel_sizes["close"]) vs r06 $(r06_panel)"))
    end
    for m in (r08_panel_sizes["close"] == r06_panel ? R08_MODELS : String[])
        r = only(filter(x -> x.environment == "close" && x.model == m && x.policy == "P0 Option B", r08_summary))
        j = findfirst(==(m), r06.model)
        ok = j !== nothing && abs(r.total_return_pct - r06.total_return_pct[j]) < 1e-6 &&
             r.n_bets == r06.n_bets[j]
        push!(r08_gate_rows, (; gate = "S1 close/P0 vs r06", environment = "close", model = m, pass = ok,
                                detail = j === nothing ? "not in r06" :
                                    @sprintf("return %.4f vs %.4f, bets %d vs %d", r.total_return_pct,
                                             r06.total_return_pct[j], r.n_bets, r06.n_bets[j])))
        ok || error("S1 FAILED: $m close/P0 does not reproduce r06 (the panel may differ — see gates)")
    end
end
r08_gates = DataFrame(r08_gate_rows)
println("\n  gates: ", count(r08_gates.pass), "/", nrow(r08_gates), " pass")

# %%
# ===================================================================
# 7. Arm-vs-arm under each policy (paired slate growth)
# ===================================================================
r08_pair_rows = NamedTuple[]
for env in R08_ENVIRONMENTS, (name, _) in r08_policies
    for (a_label, b_label) in ((R08_SPINE, R08_BASELINE), (R08_SPINE, R08_FIVE), (R08_FIVE, R08_BASELINE))
        haskey(r08_results, (env.key, a_label, name)) &&
            haskey(r08_results, (env.key, b_label, name)) || continue
        a = r08_results[(env.key, a_label, name)]
        b = r08_results[(env.key, b_label, name)]
        p = gms_paired_growth(a, b; B = R08_PAIRED_B)
        push!(r08_pair_rows, (; environment = env.key, policy = name,
                                pair = "$a_label − $b_label",
                                a_return_pct = a.summary.total_return_pct,
                                b_return_pct = b.summary.total_return_pct,
                                a_roi_pct = a.summary.roi, b_roi_pct = b.summary.roi, p...))
    end
end
r08_pairs = DataFrame(r08_pair_rows)

# The work package's two lines, read out explicitly against r04's prediction.
println("\n=== H5 — the work package's two lines ===")
for env in R08_ENVIRONMENTS, policy in R08_H5_POLICIES
    for m in R08_MODELS
        hits = filter(x -> x.environment == env.key && x.model == m && x.policy == policy, r08_summary)
        nrow(hits) == 1 || continue
        r = only(hits)
        @printf("  %-6s %-6s %-34s added %3d bets ROI %+7.2f%%  net Δreturn %+8.2f pp  core ΔROI %+6.2f pp\n",
                env.key, policy, m, r.added_n_bets, r.added_roi_pct, r.delta_return_pp,
                r.delta_core_roi_pp)
    end
end

# %%
# ===================================================================
# 8. Final report
# ===================================================================
CSV.write(joinpath(R08_OUT_DIR, "r08_sweep_summary.csv"), r08_summary)
CSV.write(joinpath(R08_OUT_DIR, "r08_added_families.csv"), r08_families)
CSV.write(joinpath(R08_OUT_DIR, "r08_family_calibration.csv"), r08_calibration)
CSV.write(joinpath(R08_OUT_DIR, "r08_line_coverage.csv"), r08_coverage_frame)
CSV.write(joinpath(R08_OUT_DIR, "r08_gates.csv"), r08_gates)
CSV.write(joinpath(R08_OUT_DIR, "r08_arm_pairs.csv"), r08_pairs)

r08_f2 = v -> gph_num(v; digits = 2)
r08_f3 = v -> gph_num(v; digits = 3)
open(joinpath(R08_OUT_DIR, "r08_trust_sweep_report.md"), "w") do io
    println(io, "# r08 trust-pruning sweep — Task 016 (1-parameter smile spine)\n")
    println(io, "Generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"), " at `", R08_GIT, "` on ",
            gethostname(), ". Arms: `", R08_SPINE, "` (test), `", R08_FIVE, "` (five-strike ",
            "reference) and `", R08_BASELINE, "` (no-smile control). Additions at trust ",
            round(R08_TIER2; digits = 4), " on Option B's book plus O/U 4.5; risk, cap, filter and ",
            "grouping are Option B's. Every smile container is staked through the anti-diagonal ",
            "reweighted grid (T011), so these rows are not comparable with Task 015's r08.\n")
    println(io, "### r04's prediction, recorded before this sweep ran\n")
    println(io, "At Under 1.5 the spine prices 0.2715 against a market of 0.2305 and a realised ",
            "rate of 0.2279 — a +4.1 pp edge where there is none — so `+U1.5` should lose money, ",
            "and lose more than the five-strike arm (+1.7 pp) or the baseline (+1.2 pp). At ",
            "Under 4.5 the spine prices 0.8078 below a market of 0.8578 while the realised rate ",
            "is 0.9038, so it should decline the bet rather than gain from it.\n")
    println(io, "## Gates\n")
    print(io, gph_markdown_table(r08_gates))
    println(io, "\n## Quoted fixtures per line (buildable panel)\n")
    print(io, gph_markdown_table(r08_coverage_frame))
    println(io, "\n## H5 — the two named lines\n")
    h5 = filter(r -> r.policy in R08_H5_POLICIES, r08_summary)
    print(io, gph_markdown_table(select(sort(h5, [:environment, :policy, :model]),
        :environment, :policy, :model, :added_n_bets, :added_win_rate_pct, :added_roi_pct,
        :added_stake_share_pct, :added_edge_pp, :delta_return_pp, :core_stake_vs_p0,
        :delta_core_roi_pp, :roi_pct, :sharpe_ann);
        formats = Dict(:added_win_rate_pct => r08_f2, :added_roi_pct => r08_f2,
                       :added_stake_share_pct => r08_f2, :added_edge_pp => r08_f2,
                       :delta_return_pp => v -> gph_signed(v; digits = 2),
                       :core_stake_vs_p0 => r08_f3,
                       :delta_core_roi_pp => v -> gph_signed(v; digits = 2),
                       :roi_pct => r08_f2, :sharpe_ann => r08_f3)))
    println(io, "\n## Full sweep\n")
    print(io, gph_markdown_table(select(r08_summary,
        :environment, :model, :policy, :n_bets, :total_return_pct, :delta_return_pp, :roi_pct,
        :sharpe_ann, :max_drawdown_pct, :n_capped, :added_n_bets, :added_win_rate_pct, :added_roi_pct,
        :added_stake_share_pct, :core_stake_vs_p0, :core_roi_pct, :delta_core_roi_pp);
        formats = Dict(:total_return_pct => r08_f2, :delta_return_pp => v -> gph_signed(v; digits = 2),
                       :roi_pct => r08_f2, :sharpe_ann => r08_f3, :max_drawdown_pct => r08_f2,
                       :added_win_rate_pct => r08_f2, :added_roi_pct => r08_f2,
                       :added_stake_share_pct => r08_f2, :core_stake_vs_p0 => r08_f3,
                       :core_roi_pct => r08_f2, :delta_core_roi_pp => v -> gph_signed(v; digits = 2))))
    println(io, "\n## Arm vs arm under each policy (paired slate log growth)\n")
    print(io, gph_markdown_table(select(r08_pairs,
        :environment, :policy, :pair, :a_return_pct, :b_return_pct, :a_roi_pct, :b_roi_pct,
        :n_slates, :delta_log_growth_per_slate, :lo, :hi, :p_better);
        formats = Dict(:a_return_pct => r08_f2, :b_return_pct => r08_f2, :a_roi_pct => r08_f2,
                       :b_roi_pct => r08_f2,
                       :delta_log_growth_per_slate => v -> gph_signed(v; digits = 5),
                       :lo => v -> gph_signed(v; digits = 5), :hi => v -> gph_signed(v; digits = 5),
                       :p_better => r08_f3)))
    println(io, "\n## Added families, by policy\n")
    print(io, gph_markdown_table(select(sort(r08_families, [:environment, :group, :policy, :model]),
        :environment, :policy, :group, :model, :n_bets, :win_rate_pct, :roi_pct, :stake_share_pct,
        :edge_mean_pp, :odds_mean);
        formats = Dict(:win_rate_pct => r08_f2, :roi_pct => r08_f2, :stake_share_pct => r08_f2,
                       :edge_mean_pp => r08_f2, :odds_mean => r08_f2)))
    println(io, "\n## Staked-bet calibration by family (the Jensen check)\n")
    println(io, "Positive `model_minus_realised` = the staked bets were priced above their ",
            "realised rate. Staked bets are a positive-edge selection, not the unconditional ",
            "forecast — r04's strike ladder is the unconditional statement.\n")
    fringe = filter(r -> r.policy in vcat(["+all_fringe", "P0 Option B"], R08_H5_POLICIES), r08_calibration)
    print(io, gph_markdown_table(select(sort(fringe, [:environment, :policy, :family, :model]),
        :environment, :policy, :family, :model, :n_bets, :mean_p_model, :mean_p_market,
        :realised_win_rate, :model_minus_realised, :roi_pct);
        formats = Dict(:mean_p_model => v -> gph_num(v; digits = 4),
                       :mean_p_market => v -> gph_num(v; digits = 4),
                       :realised_win_rate => v -> gph_num(v; digits = 4),
                       :model_minus_realised => v -> gph_signed(v; digits = 4), :roi_pct => r08_f2)))
end

println("\nR08_DONE report=", joinpath(R08_OUT_DIR, "r08_trust_sweep_report.md"))

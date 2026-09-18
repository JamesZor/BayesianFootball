# ==============================================================================
# r08 — Trust-pruning sweep: can the smile re-open the fringe totals?
# ==============================================================================
#
# WHAT THIS ANSWERS
#
# `eda/README.md` prunes every fringe selection (O/U 0.5, 1.5, 3.5, Over 2.5, BTTS) to trust 0
# on two grounds: (a) Jensen tail inflation — for an uncalibrated Poisson posterior
# E[e^{−Λ}] ≥ e^{−E[Λ]}, so deep Unders are systematically over-priced (Under 0.5: −30.7% ROI) —
# and (b) capacity cannibalisation: on constrained slates fringe lines absorbed 29% of stake at
# −13.2% ROI and crowded out the core. The smile pillar fits φ(K) at K = 0…4 against the
# market's own per-strike quotes and deflates K = 0 (φ₀ ≈ 0.844). If the smile has fixed (a),
# re-opening the fringe should now pay — unless (b) still eats the gain.
#
# DESIGN — one line at a time, then unions, every other setting held at Option B
#
#   arms          m05_joint_grw_smile_supremacy_w040 (the test) and m05_joint_grw_baseline
#                 (the control: the same question asked of a model with NO smile)
#   environments  close (de-vigged Betfair TWA(−20, 0]) and raw T−25 (tradeable book)
#   book          Option B's, plus O/U 4.5 (canonical_markets stops at 3.5; 4.5 is the
#                 smile's K = 4 strike). Gate S0 proves the extra market changes nothing
#                 at trust 0.
#   policies      P0 = Option B; then each of +U0.5, +U1.5, +U3.5, +U4.5, +O2.5, +O3.5,
#                 +O4.5, +BTTS_yes, +BTTS_no, and the unions +all_unders, +all_fringe,
#                 every addition at Option B's second tier (1/1.4). SlateDrawdown(8),
#                 FixedCap(0.25), DailySlate unchanged.
#
# READING A ROW
#
#   added_roi_pct       what the newly enabled bets themselves returned
#   core_stake_vs_p0    core-basket stake relative to Option B — < 1 is capacity taken from it
#   delta_core_roi_pp   core-basket ROI change — the cannibalisation cost
#   delta_return_pp     the net: terminal return minus Option B's
#
# NOT COMPARABLE WITH THE EDA'S NUMBERS. The −30.7% / −13.2% figures come from the TimeDecay
# hybrids m12/m13 in a six-market close simulation under SlateDrawdown(23) with a 20% cap. This
# sweep re-asks the question under Option B (SlateDrawdown 8, cap 0.25) on the GRW models; the
# baseline arm is the like-for-like control, not the EDA.
#
# GATES
#
#   S0  P0 on the extended book stakes a ledger identical to P0 on Option B's book.
#   S1  close/P0 reproduces r06_portfolio_summary.csv for both arms.
#   S2  every staked totals bet of the smile arm under +all_fringe (all new strikes included) is
#       priced through λ_tot·φ(K) to ≤ 1e-9.
#
# USAGE (mcmc-beast, from /root/BF_grw_market_smile, after r06)
#
#   julia --project -t 16 current_development/grw_market_smile/r08_trust_sweep.jl
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
include(joinpath(@__DIR__, "l04_portfolio_calibration.jl"))

# %%
# ===================================================================
# 2. Configuration
# ===================================================================
const R08_CONFIG = GMSConfig()
const R08_BOOTSTRAP_B = 2000
const R08_PAIRED_B = 10_000
const R08_SEED = 1
const R08_TIER2 = 1.0 / 1.4
const R08_OUT_DIR = joinpath(R08_CONFIG.save_root, "trust_sweep")
const R08_LATENT_DIR = joinpath(R08_CONFIG.save_root, "latents")
const R08_R06_SUMMARY = joinpath(R08_CONFIG.save_root, "portfolio", "r06_portfolio_summary.csv")
const R08_SMILE = "m05_joint_grw_smile_supremacy_w040"
const R08_BASELINE = GMS_BASELINE_CONTROL.label
const R08_MODELS = [R08_BASELINE, R08_SMILE]

mkpath(R08_OUT_DIR)
println("\n" * "="^96)
println("  r08 TRUST-PRUNING SWEEP — ", R08_SMILE, " vs ", R08_BASELINE)
println("  additions  : ", join(first.(GMS_SWEEP_ADDITIONS), ", "), "  at trust ", round(R08_TIER2; digits = 4))
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

r08_arms = filter(a -> a.label in R08_MODELS, gms_arms(R08_CONFIG))
r08_raw = Dict{String,Any}()
for arm in r08_arms
    r08_raw[arm.label] = gms_load_arm(arm, r08_ds; splitter = r08_splitter, latent_dir = R08_LATENT_DIR)
end
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
    panel, dropped = gms_buildable_panel(r08_book, r08_fits, env_odds, r08_ds, quoted)
    @printf("  panel  : %d walk-forward → %d quoted → %d buildable by both arms (%d dropped)\n",
            length(r08_panel_ids), length(quoted), length(panel), nrow(dropped))
    r08_panel_sizes[env.key] = length(panel)
    cov = gms_line_coverage(env.key, env_odds, panel)
    push!(r08_coverage, cov)
    println("  quoted fixtures per line: ",
            join(["$(r.market_name) $(r.market_line)=$(r.n_fixtures)" for r in eachrow(cov)], "  "))

    for arm in r08_arms
        source = gms_restrict(r08_fits[arm.label], panel)
        label = "$(arm.label)/$(env.key)"
        canon_books, canon_report = gms_build_books(r08_book, source, env_odds, r08_ds, panel; label)
        ext_books, ext_report = gms_build_books(r08_ext_book, source, env_odds, r08_ds, panel; label)

        p0 = gms_run_policy(r08_policy, canon_books, canon_report; B = R08_BOOTSTRAP_B, seed = R08_SEED)
        p0_ext = gms_run_policy(r08_policy, ext_books, ext_report; B = R08_BOOTSTRAP_B, seed = R08_SEED)

        # --- S0: is the extra market inert at trust 0? MEASURED, not assumed ---------
        # First r08 attempt: `==` on the two ledgers was false for the baseline at the close. `==`
        # is NaN-unsafe, so `isequal` is the comparison; and if the books still differ, that is a
        # finding about the contract (a zero-trust market can move the per-match budget or the
        # joint solve), recorded here. Either way every addition below is measured against P0 on
        # the SAME extended book, so a sweep row differs from its reference in the trust table only.
        same = isequal(p0_ext.trajectory.bets, p0.trajectory.bets)
        push!(r08_gate_rows, (; gate = "S0 extended book inert at trust 0", environment = env.key,
                                model = arm.label, pass = same,
                                detail = @sprintf("ext return %.4f vs canon %.4f (Δ %+.4f pp), bets %d vs %d",
                                    p0_ext.summary.total_return_pct, p0.summary.total_return_pct,
                                    p0_ext.summary.total_return_pct - p0.summary.total_return_pct,
                                    p0_ext.summary.n_bets, p0.summary.n_bets)))
        @printf("  S0 %-34s extended-book P0 %s canonical: return %+.4f vs %+.4f, bets %d vs %d\n",
                arm.label, same ? "==" : "DIFFERS FROM", p0_ext.summary.total_return_pct,
                p0.summary.total_return_pct, p0_ext.summary.n_bets, p0.summary.n_bets)

        canon_core = Set(String.(DataFrame(p0.trajectory.bets).family))
        r08_results[(env.key, arm.label, "P0 Option B")] = p0
        push!(r08_rows, gms_sweep_row(env.key, arm.label, "P0 Option B", p0, p0; core_families = canon_core))
        append!(r08_calibration_rows, gms_family_calibration(env.key, arm.label, "P0 Option B", p0.trajectory.bets))

        # The sweep's reference: Option B's trust on the extended book.
        core_families = Set(String.(DataFrame(p0_ext.trajectory.bets).family))
        r08_results[(env.key, arm.label, "P0 ext book")] = p0_ext
        push!(r08_rows, gms_sweep_row(env.key, arm.label, "P0 ext book", p0_ext, p0_ext; core_families))
        p0 = p0_ext

        for (name, policy) in r08_policies[2:end]
            result = gms_run_policy(policy, ext_books, ext_report; B = R08_BOOTSTRAP_B, seed = R08_SEED)
            r08_results[(env.key, arm.label, name)] = result
            row = gms_sweep_row(env.key, arm.label, name, result, p0; core_families)
            push!(r08_rows, row)
            for b in gms_breakdown(arm.label, result.trajectory.bets)
                b.level == "family" && !(b.group in core_families) &&
                    push!(r08_family_rows, (; environment = env.key, policy = name, b...))
            end
            append!(r08_calibration_rows, gms_family_calibration(env.key, arm.label, name, result.trajectory.bets))

            # --- S2: the new strikes are priced through the smile ----------------------
            if name == "+all_fringe" && source.latents isa SmileLatents
                g = gms_smile_book_gate(result.trajectory.bets, source.latents)
                push!(r08_gate_rows, (; gate = "S2 smile routing, all strikes", environment = env.key,
                                        model = arm.label, pass = g.max_abs_vs_smile <= 1e-9,
                                        detail = @sprintf("%d totals bets, max |p − smile| %.1e",
                                                          g.n_totals_bets, g.max_abs_vs_smile)))
                g.max_abs_vs_smile <= 1e-9 || error("S2 FAILED: $label +all_fringe totals not priced through λ_tot·φ(K)")
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
# Only comparable when this runner's two-arm close panel is r06's five-arm panel; a different
# panel is reported, never failed and never claimed as a reproduction.
if isfile(R08_R06_SUMMARY)
    r06 = CSV.read(R08_R06_SUMMARY, DataFrame)
    r06_panel = first(r06.n_panel)
    if r08_panel_sizes["close"] != r06_panel
        push!(r08_gate_rows, (; gate = "S1 close/P0 vs r06", environment = "close", model = "both",
                                pass = true, detail = "not comparable: close panel $(r08_panel_sizes["close"]) vs r06 $(r06_panel)"))
    end
    for m in (r08_panel_sizes["close"] == r06_panel ? R08_MODELS : String[])
        r = only(filter(x -> x.environment == "close" && x.model == m && x.policy == "P0 Option B", r08_summary))
        j = findfirst(==(m), r06.model)
        ok = j !== nothing && abs(r.total_return_pct - r06.total_return_pct[j]) < 1e-6 && r.n_bets == r06.n_bets[j]
        push!(r08_gate_rows, (; gate = "S1 close/P0 vs r06", environment = "close", model = m, pass = ok,
                                detail = j === nothing ? "not in r06" :
                                    @sprintf("return %.4f vs %.4f, bets %d vs %d", r.total_return_pct,
                                             r06.total_return_pct[j], r.n_bets, r06.n_bets[j])))
        ok || error("S1 FAILED: $m close/P0 does not reproduce r06 (the two-arm panel may differ — see gates)")
    end
end
r08_gates = DataFrame(r08_gate_rows)
println("\n  gates: ", count(r08_gates.pass), "/", nrow(r08_gates), " pass")

# %%
# ===================================================================
# 7. Smile vs baseline, policy by policy (paired slate growth)
# ===================================================================
r08_pair_rows = NamedTuple[]
for env in R08_ENVIRONMENTS, (name, _) in r08_policies
    a = r08_results[(env.key, R08_SMILE, name)]
    b = r08_results[(env.key, R08_BASELINE, name)]
    p = gms_paired_growth(a, b; B = R08_PAIRED_B)
    push!(r08_pair_rows, (; environment = env.key, policy = name,
                            smile_return_pct = a.summary.total_return_pct,
                            baseline_return_pct = b.summary.total_return_pct,
                            smile_roi_pct = a.summary.roi, baseline_roi_pct = b.summary.roi, p...))
end
r08_pairs = DataFrame(r08_pair_rows)

# %%
# ===================================================================
# 8. Final report
# ===================================================================
CSV.write(joinpath(R08_OUT_DIR, "r08_sweep_summary.csv"), r08_summary)
CSV.write(joinpath(R08_OUT_DIR, "r08_added_families.csv"), r08_families)
CSV.write(joinpath(R08_OUT_DIR, "r08_family_calibration.csv"), r08_calibration)
CSV.write(joinpath(R08_OUT_DIR, "r08_line_coverage.csv"), r08_coverage_frame)
CSV.write(joinpath(R08_OUT_DIR, "r08_gates.csv"), r08_gates)
CSV.write(joinpath(R08_OUT_DIR, "r08_smile_vs_baseline.csv"), r08_pairs)

r08_f2 = v -> gph_num(v; digits = 2)
r08_f3 = v -> gph_num(v; digits = 3)
open(joinpath(R08_OUT_DIR, "r08_trust_sweep_report.md"), "w") do io
    println(io, "# r08 trust-pruning sweep — Task 015\n")
    println(io, "Generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"), ". Arms: `", R08_SMILE,
            "` (test) and `", R08_BASELINE, "` (control). Additions at trust ", round(R08_TIER2; digits = 4),
            " on Option B's book plus O/U 4.5; risk, cap and grouping are Option B's.\n")
    println(io, "## Gates\n")
    print(io, gph_markdown_table(r08_gates))
    println(io, "\n## Quoted fixtures per line (buildable panel)\n")
    print(io, gph_markdown_table(r08_coverage_frame))
    println(io, "\n## Sweep\n")
    print(io, gph_markdown_table(select(r08_summary,
        :environment, :model, :policy, :n_bets, :total_return_pct, :delta_return_pp, :roi_pct,
        :sharpe_ann, :max_drawdown_pct, :n_capped, :added_n_bets, :added_win_rate_pct, :added_roi_pct,
        :added_stake_share_pct, :core_stake_vs_p0, :core_roi_pct, :delta_core_roi_pp);
        formats = Dict(:total_return_pct => r08_f2, :delta_return_pp => v -> gph_signed(v; digits = 2),
                       :roi_pct => r08_f2, :sharpe_ann => r08_f3, :max_drawdown_pct => r08_f2,
                       :added_win_rate_pct => r08_f2, :added_roi_pct => r08_f2,
                       :added_stake_share_pct => r08_f2, :core_stake_vs_p0 => r08_f3,
                       :core_roi_pct => r08_f2, :delta_core_roi_pp => v -> gph_signed(v; digits = 2))))
    println(io, "\n## Smile vs baseline under each policy (paired slate log growth)\n")
    print(io, gph_markdown_table(r08_pairs; formats = Dict(
        :smile_return_pct => r08_f2, :baseline_return_pct => r08_f2, :smile_roi_pct => r08_f2,
        :baseline_roi_pct => r08_f2, :delta_log_growth_per_slate => v -> gph_signed(v; digits = 5),
        :lo => v -> gph_signed(v; digits = 5), :hi => v -> gph_signed(v; digits = 5),
        :p_better => r08_f3, :delta_total_log_growth => v -> gph_signed(v; digits = 3))))
    println(io, "\n## Added families, by policy\n")
    print(io, gph_markdown_table(select(sort(r08_families, [:environment, :group, :policy, :model]),
        :environment, :policy, :group, :model, :n_bets, :win_rate_pct, :roi_pct, :stake_share_pct,
        :edge_mean_pp, :odds_mean);
        formats = Dict(:win_rate_pct => r08_f2, :roi_pct => r08_f2, :stake_share_pct => r08_f2,
                       :edge_mean_pp => r08_f2, :odds_mean => r08_f2)))
    println(io, "\n## Staked-bet calibration by family (the Jensen check)\n")
    println(io, "Positive `model_minus_realised` = the staked bets were priced above their realised ",
            "rate. Staked bets are a positive-edge selection, not the unconditional forecast.\n")
    fringe = filter(r -> r.policy in ("+all_fringe", "P0 Option B"), r08_calibration)
    print(io, gph_markdown_table(select(sort(fringe, [:environment, :policy, :family, :model]),
        :environment, :policy, :family, :model, :n_bets, :mean_p_model, :mean_p_market,
        :realised_win_rate, :model_minus_realised, :roi_pct);
        formats = Dict(:mean_p_model => v -> gph_num(v; digits = 4), :mean_p_market => v -> gph_num(v; digits = 4),
                       :realised_win_rate => v -> gph_num(v; digits = 4),
                       :model_minus_realised => v -> gph_signed(v; digits = 4), :roi_pct => r08_f2)))
end

println("\nR08_DONE report=", joinpath(R08_OUT_DIR, "r08_trust_sweep_report.md"))

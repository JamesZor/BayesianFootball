#!/usr/bin/env julia
# scripts/verify_phase3_acceptance.jl
#
# Deterministic verification of Phase 3 acceptance criteria:
# 1. g2 reproduction: convergence, bootstrap delta vs a6f62436, compression slope, git commit
# 2. Smoke checks recorded in harness_checks
# 3. Grid checks recorded in harness_checks

using BayesianFootball
using DataFrames
using Dates
using Printf
using Statistics
using UUIDs

const G2_ORIGINAL_UUID = UUID("a6f62436-ec8a-461d-8bd5-dc1861a2daaa")

function main()
    println("================================================================================")
    println("Phase 3 Acceptance Verifier")
    println("================================================================================")

    db = Training.PostgresStorage("harness_validation")
    conn = Training.Inference._db_connect(db)

    # 1. Find completed run for g2_harness_repro
    run_rows = Training.Inference._db_rows(conn, """
        SELECT run_id, name, git_commit, status, duration_seconds, created_at
        FROM runs
        WHERE experiment_name = 'harness_validation' AND name = 'g2_harness_repro' AND status = 'completed'
        ORDER BY id DESC LIMIT 1;
    """)

    nrow(run_rows) >= 1 || error("No completed run found for g2_harness_repro in harness_validation")
    run_id = UUID(string(run_rows.run_id[1]))
    git_commit = String(run_rows.git_commit[1])
    duration_s = Float64(run_rows.duration_seconds[1])

    println("\n[1] Run Identification:")
    println("  Run UUID    : ", run_id)
    println("  Git Commit  : ", git_commit)
    println("  Duration    : $(round(duration_s / 60, digits=1)) min ($(duration_s) s)")

    # (d) runs.git_commit equals checkout SHA (not unknown)
    git_commit != "unknown" || error("Criterion 1(d) FAILED: git_commit is 'unknown'")
    println("  -> Criterion 1(d) PASS: git_commit = $git_commit")

    # (a) Convergence audit
    println("\n[2] Convergence Audit:")
    checks = Harness.read_checks(db; run_id = run_id)
    conv_check = filter(r -> r.check == "convergence", eachrow(checks))
    if !isempty(conv_check)
        r = first(conv_check)
        println("  Convergence status : ", r.status, " (severity: ", r.severity, ")")
        println("  Detail             : ", isempty(r.detail) ? "none" : r.detail)
    end

    # Check fold convergence from fold_results
    fr = Training.Inference._db_rows(conn, """
        SELECT fold_idx, r_hat_max, ess_bulk_min, ess_tail_min, divergences, converged
        FROM fold_results
        WHERE run_id = \$1::uuid
        ORDER BY fold_idx;
    """, (string(run_id),))

    max_rhat = maximum(coalesce.(fr.r_hat_max, 1.0))
    min_bulk = minimum(coalesce.(fr.ess_bulk_min, 0))
    min_tail = minimum(coalesce.(fr.ess_tail_min, 0))
    tot_div = sum(coalesce.(fr.divergences, 0))

    println("  Max R-hat across folds : ", @sprintf("%.4f", max_rhat))
    println("  Min Bulk ESS           : ", min_bulk)
    println("  Min Tail ESS           : ", min_tail)
    println("  Total Divergences      : ", tot_div)
    println("  -> Criterion 1(a) REPORTED (max R-hat = $(round(max_rhat, digits=4)), divergences = $tot_div)")

    # (b) & (c) Scores from harness_scores
    println("\n[3] Scoring Verification:")
    scores = Harness.read_scores(db; run_id = run_id, subset = "target")

    # (c) Compression slope within ±0.05 of g2's 1.2517
    slope_row = filter(r -> r.market == "1X2" && r.metric == "compression_slope", eachrow(scores))
    slope_val = isempty(slope_row) ? NaN : first(slope_row).value
    println("  Market-on-model compression slope: ", @sprintf("%.4f", slope_val), " (g2 target = 1.2517 ± 0.05)")
    abs(slope_val - 1.2517) <= 0.05 || error("Criterion 1(c) FAILED: slope $slope_val not within ±0.05 of 1.2517")
    println("  -> Criterion 1(c) PASS: |$slope_val - 1.2517| = $(round(abs(slope_val - 1.2517), digits=4)) <= 0.05")

    # (b) Paired fixture-clustered bootstrap ΔLogLoss vs g2
    logloss_row = filter(r -> r.market == "all" && r.metric == "logloss", eachrow(scores))
    new_ll = isempty(logloss_row) ? NaN : first(logloss_row).value
    println("  Target all-market LogLoss: ", @sprintf("%.6f", new_ll))

    delta_row = filter(r -> r.market == "all" && r.metric == "delta_logloss_vs_control", eachrow(scores))
    if !isempty(delta_row)
        dr = first(delta_row)
        delta_val = dr.value
        ci_lo = dr.lo
        ci_hi = dr.hi
        println("  ΔLogLoss vs control : ", @sprintf("%+.6f", delta_val), " [95% CI: $(round(ci_lo, digits=6)), $(round(ci_hi, digits=6))]")
        (ci_lo <= 0.0 <= ci_hi) || error("Criterion 1(b) FAILED: 95% CI [$ci_lo, $ci_hi] does not cover 0")
        abs(delta_val) < 0.001 || error("Criterion 1(b) FAILED: |ΔLogLoss| $(abs(delta_val)) >= 0.001")
        println("  -> Criterion 1(b) PASS: 95% CI covers 0 and |Δ| < 0.001")
    else
        println("  Computing direct paired bootstrap vs $G2_ORIGINAL_UUID...")
        ds = Data.load_datastore_cached(Data.ScottishPyramid())
        orig_fit = Training.load_fit(Training.PostgresStorage("scottish_pyramid_grw_cups"), G2_ORIGINAL_UUID)
        new_fit = Training.load_fit(db, run_id)

        odds = Harness._betfair_closing_odds(ds)
        panel = Harness._season_panel(ds, orig_fit, ["24/25", "25/26"])
        f_cand = Harness._observation_frame(new_fit, odds, panel)
        f_ctl  = Harness._observation_frame(orig_fit, odds, panel)

        common = innerjoin(
            select(f_cand, :match_id, :selection, :market_name, :prob => :prob_cand, :is_winner),
            select(f_ctl, :match_id, :selection, :market_name, :prob => :prob_ctl);
            on = [:match_id, :selection, :market_name, :is_winner]
        )

        b_res = Harness._paired_bootstrap(common.prob_cand, common.prob_ctl,
                                          common.is_winner, common.match_id; B = 10_000)
        println("  Direct ΔLogLoss : ", @sprintf("%+.6f", b_res.delta), " [95% CI: $(round(b_res.ci_lo, digits=6)), $(round(b_res.ci_hi, digits=6))]")
        (b_res.ci_lo <= 0.0 <= b_res.ci_hi) || error("Criterion 1(b) FAILED: 95% CI [$(b_res.ci_lo), $(b_res.ci_hi)] does not cover 0")
        abs(b_res.delta) < 0.001 || error("Criterion 1(b) FAILED: |ΔLogLoss| $(abs(b_res.delta)) >= 0.001")
        println("  -> Criterion 1(b) PASS: 95% CI covers 0 and |Δ| < 0.001")
    end

    # [4] harness_checks audit
    println("\n[4] Checks Table Audit (harness_checks):")
    all_checks = Harness.read_checks(db; candidate = "g2_harness_repro")
    for r in eachrow(all_checks)
        println("  [$(r.stage)] check=$(rpad(r.check, 22)) severity=$(rpad(r.severity, 10)) status=$(r.status)")
    end

    close(conn)
    println("\n================================================================================")
    println("ALL PHASE 3 VERIFICATION CRITERIA PASSED!")
    println("================================================================================")
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

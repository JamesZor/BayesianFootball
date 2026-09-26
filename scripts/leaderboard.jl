#!/usr/bin/env julia

using BayesianFootball
using CSV
using DataFrames
using Printf

const EXPERIMENTS_DIR = joinpath(@__DIR__, "..", "experiments")

number(value; digits = 5) = ismissing(value) || !(value isa Real) || !isfinite(value) ?
    "—" : @sprintf("%.*f", digits, value)

function cohort_number(value, n; digits = 5, suffix = "")
    shown = number(value; digits)
    count = ismissing(n) || !(n isa Real) || !isfinite(n) ? "?" : string(round(Int, n))
    return shown == "—" ? "— (n=$count)" : shown * suffix * " (n=$count)"
end

function leaderboard_markdown(board)
    io = IOBuffer()
    println(io, "# Experiment Harness Leaderboard")
    println(io)
    println(io, "Scorecard `v1.1`; reference: de-vigged Betfair TWA (−20, 0] close. " *
                "Lower LogLoss/ECE is better; compression slope is market-on-model (ideal 1).")
    println(io)
    println(io, "| Model | Target LL | 1X2 LL | ECE | Compression | Δ LL vs control [95% CI] | Run UUID |")
    println(io, "|---|---:|---:|---:|---:|---:|---|")
    for row in eachrow(board)
        delta = number(row.delta_vs_control) * " [" * number(row.delta_lo) * ", " *
                number(row.delta_hi) * "]"
        println(io, "| `$(row.model)` | $(number(row.target_logloss_all)) | " *
                    "$(number(row.target_logloss_1x2)) | $(number(row.target_ece_all)) | " *
                    "$(number(row.compression_slope, digits = 3)) | $delta | `$(row.run_id)` |")
    end
    println(io)
    println(io, "## Transition cohorts")
    println(io)
    for direction in (:relegated_into_L1, :promoted_into_L1, :entered_spfl, :l1_l2, :any)
        println(io, "### `$(direction)`")
        println(io)
        println(io, "| Model | first 10 LL | first 10 bias (pp) | first 20 LL | first 20 bias (pp) |")
        println(io, "|---|---:|---:|---:|---:|")
        for row in eachrow(board)
            stem10 = "transition_$(direction)_first10"
            stem20 = "transition_$(direction)_first20"
            n10 = row[Symbol(stem10 * "_n")]
            n20 = row[Symbol(stem20 * "_n")]
            println(io, "| `$(row.model)` | " *
                "$(cohort_number(row[Symbol(stem10 * "_logloss")], n10)) | " *
                "$(cohort_number(row[Symbol(stem10 * "_bias_pp")], n10; digits = 3, suffix = " pp")) | " *
                "$(cohort_number(row[Symbol(stem20 * "_logloss")], n20)) | " *
                "$(cohort_number(row[Symbol(stem20 * "_bias_pp")], n20; digits = 3, suffix = " pp")) |")
        end
        println(io)
    end
    return String(take!(io))
end

function register_markdown(register)
    io = IOBuffer()
    println(io, "# Experiment Register")
    println(io)
    println(io, "Generated from `harness_experiments`. Historical decisions are records, not " *
                "new claims; use the linked README for evidence and caveats.")
    println(io)
    println(io, "| ID | Date | TODO | Question | Dimension | Status | Decision | Runs | Record |")
    println(io, "|---|---|---|---|---|---|---|---|---|")
    for row in eachrow(register)
        todo = ismissing(row.todo) ? "—" : lpad(string(row.todo), 3, '0')
        record = row.readme == "absent/not run" ? row.readme :
                 "[`record`](../$(row.readme))"
        println(io, "| $(row.id) | $(row.date) | $todo | $(row.question) | $(row.dimension) | " *
                    "$(row.status) | $(row.decision) | $(row.run_ids) | $record |")
    end
    return String(take!(io))
end

function main()
    db = PostgresStorage("harness")
    Harness.ensure_harness_schema!(db)
    scores = Harness.read_scores(db; scorecard_version = "v1.1")
    isempty(scores) && error("harness_scores contains no v1.1 rows; run scripts/score_runs.jl first")
    board = Harness.leaderboard(scores)
    register = Harness.read_experiments(db)
    isempty(register) && error("harness_experiments is empty; run scripts/seed_register.jl first")

    CSV.write(joinpath(EXPERIMENTS_DIR, "LEADERBOARD.csv"), board)
    write(joinpath(EXPERIMENTS_DIR, "LEADERBOARD.md"), leaderboard_markdown(board))
    write(joinpath(EXPERIMENTS_DIR, "REGISTER.md"), register_markdown(register))
    println("Wrote experiments/LEADERBOARD.csv, experiments/LEADERBOARD.md, experiments/REGISTER.md")
end

isempty(ARGS) || error("usage: julia --project scripts/leaderboard.jl")
main()

#!/usr/bin/env julia
# Validate the MAP-only re-screen rehearsal; only reads the disposable test DB.
using BayesianFootball
using BayesianFootball: Harness, Training
using DataFrames
using JSON3

url = get(ENV, "BF_EXPERIMENTS_TEST_DB_URL", "")
isempty(strip(url)) && error("BF_EXPERIMENTS_TEST_DB_URL is required")
db = Training.PostgresStorage(url, "harness")
db.dbname == "mcmc_experiments_test" || error("Refusing non-test DB")
scores = Harness.read_scores(db; scorecard_version = "v1.2")
screen = filter(:stage => ==("screen"), scores)
ids = unique(screen.run_id)
length(ids) == 12 || error("Expected 12 W1 MAP IDs, got $(length(ids))")
expected_panel = "56+57|24/25,25/26|n=710"
all(==(expected_panel), screen.panel) || error("Screen panel changed")
checks = Harness.read_checks(db; stage = "screen", check = "screen_validity")
length(unique(checks.run_id)) == 12 || error("Missing screen_validity checks")
all(r -> r.status in ("limited", "ranking_only"), eachrow(checks)) ||
    error("Unexpected screen_validity status")
board = Harness.leaderboard(screen)
module_for_markdown = Module(:KlmScreenBoard)
Base.include(module_for_markdown, joinpath(@__DIR__, "leaderboard.jl"))
markdown = module_for_markdown.leaderboard_markdown(board[1:0, :], board)
occursin("## MAP screen (diagnostic only) — panel `$expected_panel`", markdown) ||
    error("MAP section/panel missing")
occursin("**Warning:** MAP", markdown) || error("MAP warning missing")
!occursin("## Posterior grid — panel", markdown) || error("Screen leaked into grid section")

original = JSON3.read(read("/root/BF_runs/logs/klm/w1_register_before.json", String))
register = only(eachrow(filter(:id => ==("scottish_dynamics_scope_matrix_w1"),
                                Harness.read_experiments(db))))
for key in (:id, :date, :todo, :question, :dimension, :status, :decision, :readme)
    string(register[key]) == string(getproperty(original, key)) ||
        error("W1 register field $key changed on re-screen")
end
register.status == "completed" || error("W1 register was downgraded")
all(in(split(register.run_ids, ',')), split(String(original.run_ids), ',')) ||
    error("W1 register lost an original run ID")
println("KLM_SCREEN_PASS ids=$(length(ids)) checks=$(length(unique(checks.run_id))) " *
        "score_rows=$(nrow(screen)) panel=$expected_panel register=preserved " *
        "markdown_map_rows=$(count(occursin.(Ref("| `"), split(markdown, '\n'))))")

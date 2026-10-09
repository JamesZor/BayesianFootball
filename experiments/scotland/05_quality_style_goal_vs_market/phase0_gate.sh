#!/usr/bin/env bash
# Run the merged-base test entry points on mcmc-beast only, one fresh tmux Julia REPL per file.
set -euo pipefail
cd /root/BF_runs/qs_experiment
logdir=/root/BF_runs/logs/qs_experiment
mkdir -p "$logdir"
for item in \
  tape:test/tape_allocation_tests.jl \
  builder:test/builder_tests.jl \
  harness:test/harness_runner_tests.jl \
  t01:current_development/market_model/t01_market_model_tests.jl \
  t02:current_development/market_model/t02_two_stage_tests.jl \
  t03:current_development/market_model/t03_covariance_tests.jl \
  t04:current_development/market_model/t04_copula_grid_tests.jl \
  t05_fast:current_development/market_model/t05_fast_gaussian_tests.jl \
  t05_reports:current_development/market_model/t05_fast_reports_tests.jl \
  t05_workflow:current_development/market_model/t05_fullbook_workflow_tests.jl \
  t05_preflight:current_development/market_model/t05_laplace_preflight_tests.jl \
  t05_pooled:current_development/market_model/t05_pooled_tests.jl \
  t06:current_development/market_model/t06_qs_eda_tests.jl; do
  name=${item%%:*}
  file=${item#*:}
  session="pi_qsx_gate_${name}"
  status="$logdir/${name}.status"
  rm -f "$status"
  pane=$(tmux new-session -d -P -F '#{pane_id}' -s "$session" -c /root/BF_runs/qs_experiment)
  printf 'START %s %s pane=%s %s\n' "$name" "$file" "$pane" "$(date -u +%FT%TZ)"
  tmux pipe-pane -o -t "$pane" "cat >> $logdir/${name}.log"
  tmux send-keys -t "$pane" -l -- 'JULIA_PKG_PRECOMPILE_AUTO=0 /root/.juliaup/bin/julia --project -t 16'
  sleep 0.2
  tmux send-keys -t "$pane" Enter
  # Input is sent to the REPL, never as a Julia command-line argument.
  code="using LinearAlgebra; BLAS.set_num_threads(1); try include(\"$file\"); println(\"PHASE0_PASS_$name\"); write(\"$status\", \"PASS\") catch err; showerror(stderr, err, catch_backtrace()); println(stderr); println(\"PHASE0_FAIL_$name\"); write(\"$status\", \"FAIL\") end"
  sleep 8
  tmux send-keys -t "$pane" -l -- "$code"
  sleep 0.2
  tmux send-keys -t "$pane" Enter
  while [[ ! -f "$status" ]]; do
    if ! tmux has-session -t "$session" 2>/dev/null; then
      printf 'PHASE0_BLOCKED: lost session %s; see %s/%s.log\n' "$session" "$logdir" "$name"; exit 1
    fi
    sleep 10
  done
  outcome=$(<"$status")
  printf 'END %s %s %s\n' "$name" "$outcome" "$(date -u +%FT%TZ)"
  tmux kill-session -t "$session"
  if [[ "$outcome" != PASS ]]; then
    printf 'PHASE0_BLOCKED: %s; see %s/%s.log\n' "$name" "$logdir" "$name"
    exit 1
  fi
done
printf 'PHASE0_ALL_PASS\n'

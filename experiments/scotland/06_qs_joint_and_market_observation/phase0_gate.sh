#!/usr/bin/env bash
# Beast only: one fresh persistent Julia REPL per gate; no one-shot Julia.
set -euo pipefail
cd /root/BF_runs/qs_experiment
logdir=/root/BF_runs/logs/qs_experiment_w2
mkdir -p "$logdir"
for item in \
  quality_style:test/test_quality_style_grw.jl \
  multiscale:test/test_multiscale_grw.jl \
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
  t06:current_development/market_model/t06_qs_eda_tests.jl \
  coverage:experiments/scotland/06_qs_joint_and_market_observation/r00_coverage.jl; do
  name=${item%%:*}
  file=${item#*:}
  session="pi_qsx2_gate_${name}"
  status="$logdir/${name}.status"
  [[ ! -e "$status" && ! -e "$logdir/${name}.log" ]] || {
    echo "BLOCKED: existing evidence for $name; do not overwrite"; exit 1;
  }
  # FIFO completion notification blocks the launcher without status polling.
  notify="$logdir/${name}.notify"
  mkfifo "$notify"
  pane=$(tmux new-session -d -P -F '#{pane_id}' -s "$session" -c "$PWD")
  printf 'START %s %s pane=%s %s\n' "$name" "$file" "$pane" "$(date -u +%FT%TZ)"
  tmux pipe-pane -o -t "$pane" "cat >> $logdir/${name}.log"
  tmux send-keys -t "$pane" -l -- 'JULIA_PKG_PRECOMPILE_AUTO=0 /root/.juliaup/bin/julia --project -t 16'
  sleep 0.2
  tmux send-keys -t "$pane" Enter
  code="using LinearAlgebra, ThreadPinning; pinthreads(:cores); BLAS.set_num_threads(1); started = time(); try include(\"$file\"); println(\"PHASE0_INCLUDE_RETURNED_$name wall_s=\", time()-started); write(\"$status\", \"PASS\") catch err; showerror(stderr, err, catch_backtrace()); println(stderr); println(\"PHASE0_FAIL_$name wall_s=\", time()-started); write(\"$status\", \"FAIL\") end; write(\"$notify\", read(\"$status\", String) * \"\\n\")"
  sleep 8
  tmux send-keys -t "$pane" -l -- "$code"
  sleep 0.2
  tmux send-keys -t "$pane" Enter
  IFS= read -r outcome < "$notify"
  rm "$notify"
  if [[ "$name" == t05_pooled ]] && grep -aq 'C2_PENDING_REPORTED: thin-book failures retained' "$logdir/${name}.log"; then
    echo 'C2_PENDING_EXCLUDED: known 29/39, 10 failures; not an acceptance gate'
  fi
  printf 'END %s %s %s\n' "$name" "$outcome" "$(date -u +%FT%TZ)"
  if [[ "$outcome" != PASS ]]; then
    echo "PHASE0_BLOCKED: $name; retain pane=$pane for diagnosis"; exit 1
  fi
  tmux kill-session -t "$session"
done
printf 'PHASE0_ALL_PASS\n'

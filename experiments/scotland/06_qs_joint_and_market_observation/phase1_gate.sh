#!/usr/bin/env bash
# Fresh owned beast REPL per Phase 1 entry point; FIFO completion, no polling.
set -euo pipefail
cd /root/BF_runs/qs_experiment
logdir=/root/BF_runs/logs/qs_experiment_w2
mkdir -p "$logdir"
output=${QSX2_TEST_OUTPUT:-/root/BF_runs/qs_experiment_w2_out/phase1}
rerun_first=${QSX2_RERUN_RECOVERY_FIRST:-false}
if [[ $# == 0 ]]; then
  set -- market:test/test_market_rate_observation.jl \
    tape:test/tape_allocation_tests.jl \
    builder:test/builder_tests.jl \
    harness:test/harness_runner_tests.jl
fi
for item in "$@"; do
  name="phase1_${item%%:*}"
  file=${item#*:}
  session="pi_qsx2_${name}"
  status="$logdir/${name}.status"
  notify="$logdir/${name}.notify"
  [[ ! -e "$status" && ! -e "$logdir/${name}.log" ]] || {
    echo "BLOCKED: existing evidence for $name; do not overwrite"; exit 1;
  }
  mkfifo "$notify"
  pane=$(tmux new-session -d -P -F '#{pane_id}' -s "$session" -c "$PWD")
  printf 'START %s %s pane=%s %s\n' "$name" "$file" "$pane" "$(date -u +%FT%TZ)"
  tmux pipe-pane -o -t "$pane" "cat >> $logdir/${name}.log"
  tmux send-keys -t "$pane" -l -- 'JULIA_PKG_PRECOMPILE_AUTO=0 /root/.juliaup/bin/julia --project -t 16'
  sleep 0.2
  tmux send-keys -t "$pane" Enter
  code="ENV[\"QSX2_TEST_OUTPUT\"] = \"$output\"; ENV[\"QSX2_RERUN_RECOVERY_FIRST\"] = \"$rerun_first\"; using LinearAlgebra, ThreadPinning; pinthreads(:cores); BLAS.set_num_threads(1); started = time(); try include(\"$file\"); println(\"PHASE1_INCLUDE_RETURNED_$name wall_s=\", time()-started); write(\"$status\", \"PASS\") catch err; showerror(stderr, err, catch_backtrace()); println(stderr); println(\"PHASE1_FAIL_$name wall_s=\", time()-started); write(\"$status\", \"FAIL\") end; write(\"$notify\", read(\"$status\", String) * \"\\n\")"
  sleep 8
  tmux send-keys -t "$pane" -l -- "$code"
  sleep 0.2
  tmux send-keys -t "$pane" Enter
  IFS= read -r outcome < "$notify"
  rm "$notify"
  printf 'END %s %s %s\n' "$name" "$outcome" "$(date -u +%FT%TZ)"
  if [[ "$outcome" != PASS ]]; then
    echo "PHASE1_BLOCKED: $name; retain pane=$pane for diagnosis"; exit 1
  fi
  tmux kill-session -t "$session"
done
echo 'PHASE1_REQUESTED_GATES_PASS'

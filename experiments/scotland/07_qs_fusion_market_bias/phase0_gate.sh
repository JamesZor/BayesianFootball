#!/usr/bin/env bash
# Wave 3 Phase 0 (beast only): one fresh persistent Julia REPL per gate, run in sequence.
# Adapted from wave 2's phase0_gate.sh; adds the market-rate-observation test, drops coverage.
set -euo pipefail
cd /root/BF_runs/qs_experiment
logdir=/root/BF_runs/logs/qs_experiment_w3/phase0
out=/root/BF_runs/qs_experiment_w3_out/phase0
mkdir -p "$logdir" "$out"

snapshot=.cache/datastore_ScottishLower.jls
table=experiments/scotland/06_qs_joint_and_market_observation/results/market_rates.csv
want_snapshot=c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4
want_table=680bc2283e9c0fc7a896bf01b4ba209cb3aae06f50d00667c98b02f379872549
got_snapshot=$(sha256sum "$snapshot" | cut -d' ' -f1)
got_table=$(sha256sum "$table" | cut -d' ' -f1)
mtime=$(date -u -r "$snapshot" '+%F %T.%N')
printf 'SNAPSHOT sha=%s mtime=%s UTC\nTABLE sha=%s\n' "$got_snapshot" "$mtime" "$got_table"
[[ $got_snapshot == "$want_snapshot" && $got_table == "$want_table" &&
   $mtime == '2026-09-25 12:57:15.480765468' ]] || { echo 'PHASE0_BLOCKED: pin mismatch'; exit 1; }

for item in \
  quality_style:test/test_quality_style_grw.jl \
  market_rate:test/test_market_rate_observation.jl \
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
  t06:current_development/market_model/t06_qs_eda_tests.jl; do
  name=${item%%:*}
  file=${item#*:}
  session="claude_qsx3_gate_${name}"
  status="$logdir/${name}.status"
  [[ ! -e "$status" && ! -e "$logdir/${name}.log" ]] || {
    echo "BLOCKED: existing evidence for $name; do not overwrite"; exit 1;
  }
  # A FIFO blocks the launcher until the REPL reports, without status polling.
  notify="$logdir/${name}.notify"
  mkfifo "$notify"
  pane=$(tmux new-session -d -P -F '#{pane_id}' -s "$session" -c "$PWD")
  printf 'START %s %s pane=%s %s\n' "$name" "$file" "$pane" "$(date -u +%FT%TZ)"
  tmux pipe-pane -o -t "$pane" "cat >> $logdir/${name}.log"
  tmux send-keys -t "$pane" -l -- 'JULIA_PKG_PRECOMPILE_AUTO=0 /root/.juliaup/bin/julia --project -t 16'
  sleep 0.2
  tmux send-keys -t "$pane" Enter
  code="using LinearAlgebra, ThreadPinning; pinthreads(:cores); BLAS.set_num_threads(1); ENV[\"QSX2_TEST_OUTPUT\"] = \"$out/$name\"; started = time(); try include(\"$file\"); println(\"PHASE0_INCLUDE_RETURNED_$name wall_s=\", time()-started); write(\"$status\", \"PASS\") catch err; showerror(stderr, err, catch_backtrace()); println(stderr); println(\"PHASE0_FAIL_$name wall_s=\", time()-started); write(\"$status\", \"FAIL\") end; write(\"$notify\", read(\"$status\", String) * \"\\n\")"
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

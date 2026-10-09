#!/usr/bin/env bash
# One fresh persistent beast REPL per arm, sequential completion via FIFO; no polling loop.
set -euo pipefail
cd /root/BF_runs/qs_experiment
logdir=/root/BF_runs/logs/qs_experiment_w2
mkdir -p "$logdir"
# Manager: joint smoke passes stand; only the array-backed market recipes changed.
for arm in grw_marketobs qs_marketobs; do
  name="phase2_smoke_arrays_$arm"
  session="pi_qsx2_$name"
  status="$logdir/$name.status"
  notify="$logdir/$name.notify"
  [[ ! -e "$status" && ! -e "$logdir/$name.log" ]] || {
    echo "BLOCKED: existing evidence for $name; do not overwrite"; exit 1;
  }
  mkfifo "$notify"
  pane=$(tmux new-session -d -P -F '#{pane_id}' -s "$session" -c "$PWD")
  printf 'START %s pane=%s %s\n' "$name" "$pane" "$(date -u +%FT%TZ)"
  tmux pipe-pane -o -t "$pane" "cat >> $logdir/$name.log"
  tmux send-keys -t "$pane" -l -- 'JULIA_PKG_PRECOMPILE_AUTO=0 /root/.juliaup/bin/julia --project -t 16'
  sleep 0.2
  tmux send-keys -t "$pane" Enter
  code="ENV[\"QSX2_ONLY\"] = \"$arm\"; ENV[\"QSX2_SMOKE_OUTPUT\"] = \"/root/BF_runs/qs_experiment_w2_out/phase2_arrays\"; started = time(); try include(\"experiments/scotland/06_qs_joint_and_market_observation/r03_smoke.jl\"); println(\"PHASE2_INCLUDE_RETURNED_$arm wall_s=\", time()-started); write(\"$status\", \"PASS\") catch err; showerror(stderr, err, catch_backtrace()); println(stderr); println(\"PHASE2_FAIL_$arm wall_s=\", time()-started); write(\"$status\", \"FAIL\") end; write(\"$notify\", read(\"$status\", String) * \"\\n\")"
  sleep 8
  tmux send-keys -t "$pane" -l -- "$code"
  sleep 0.2
  tmux send-keys -t "$pane" Enter
  IFS= read -r outcome < "$notify"
  rm "$notify"
  printf 'END %s %s %s\n' "$name" "$outcome" "$(date -u +%FT%TZ)"
  if [[ "$outcome" != PASS ]]; then
    echo "PHASE2_BLOCKED: $arm; retain pane=$pane for diagnosis"; exit 1
  fi
  tmux kill-session -t "$session"
done
echo 'PHASE2_SMOKE_HARD_GATES_PASS; STOP — manager approval required before grid'

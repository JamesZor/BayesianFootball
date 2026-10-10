#!/usr/bin/env bash
# Wave 3 Phase 2 (beast only): one fresh persistent Julia REPL per arm, sequential, FIFO
# completion (no polling). Smoke only; the grid needs manager approval.
# Usage: phase2_gate.sh <attempt>; each attempt writes its own evidence directory.
set -euo pipefail
attempt=${1:?usage: phase2_gate.sh <attempt>}
cd /root/BF_runs/qs_experiment
logdir=/root/BF_runs/logs/qs_experiment_w3/phase2/$attempt
out=/root/BF_runs/qs_experiment_w3_out/phase2/$attempt
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
   $mtime == '2026-09-25 12:57:15.480765468' ]] || { echo 'PHASE2_BLOCKED: pin mismatch'; exit 1; }

printf 'COMMIT %s ATTEMPT %s\n' "$(git rev-parse HEAD)" "$attempt"
for arm in fusion_qs_bias fusion_qs_nobias fusion_grw_bias; do
  session="claude_qsx3_p2_${arm}"
  status="$logdir/${arm}.status"
  [[ ! -e "$status" && ! -e "$logdir/${arm}.log" ]] || {
    echo "BLOCKED: existing evidence for $arm; do not overwrite"; exit 1;
  }
  notify="$logdir/${arm}.notify"
  mkfifo "$notify"
  pane=$(tmux new-session -d -P -F '#{pane_id}' -s "$session" -c "$PWD")
  printf 'START %s pane=%s %s\n' "$arm" "$pane" "$(date -u +%FT%TZ)"
  tmux pipe-pane -o -t "$pane" "cat >> $logdir/${arm}.log"
  tmux send-keys -t "$pane" -l -- 'JULIA_PKG_PRECOMPILE_AUTO=0 /root/.juliaup/bin/julia --project -t 16'
  sleep 0.2
  tmux send-keys -t "$pane" Enter
  code="ENV[\"QSX3_ONLY\"] = \"$arm\"; ENV[\"QSX3_SMOKE_OUTPUT\"] = \"$out\"; started = time(); try include(\"experiments/scotland/07_qs_fusion_market_bias/r03_smoke.jl\"); println(\"PHASE2_INCLUDE_RETURNED_$arm wall_s=\", time()-started); write(\"$status\", \"PASS\") catch err; showerror(stderr, err, catch_backtrace()); println(stderr); println(\"PHASE2_FAIL_$arm wall_s=\", time()-started); write(\"$status\", \"FAIL\") end; write(\"$notify\", read(\"$status\", String) * \"\\n\")"
  sleep 8
  tmux send-keys -t "$pane" -l -- "$code"
  sleep 0.2
  tmux send-keys -t "$pane" Enter
  IFS= read -r outcome < "$notify"
  rm "$notify"
  printf 'END %s %s %s\n' "$arm" "$outcome" "$(date -u +%FT%TZ)"
  if [[ "$outcome" != PASS ]]; then
    echo "PHASE2_BLOCKED: $arm; retain pane=$pane for diagnosis"; exit 1
  fi
  tmux kill-session -t "$session"
done
echo 'PHASE2_SMOKE_HARD_GATES_PASS; STOP — manager approval required before grid'

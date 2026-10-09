#!/usr/bin/env bash
# Authorised resume: skip standalone GRW, QS persisted scoring then market arms.
set -euo pipefail
cd /root/BF_runs/qs_experiment
[[ $(git rev-parse --short HEAD) == 081ba5a6 && -z $(git status --porcelain) ]] || exit 1
logdir=/root/BF_runs/logs/qs_experiment_w2
attempt=v2
for arm in qs_joint grw_marketobs qs_marketobs; do
  name="phase3_grid_${attempt}_$arm"
  session="pi_qsx2_$name"
  status="$logdir/$name.status"
  notify="$logdir/$name.notify"
  [[ ! -e "$status" && ! -e "$logdir/$name.log" ]] || {
    echo "BLOCKED: existing evidence for $name"; exit 1;
  }
  mkfifo "$notify"
  pane=$(tmux new-session -d -P -F '#{pane_id}' -s "$session" -c "$PWD")
  printf 'START %s pane=%s %s\n' "$name" "$pane" "$(date -u +%FT%TZ)"
  tmux pipe-pane -o -t "$pane" "cat >> $logdir/$name.log"
  tmux send-keys -t "$pane" -l -- 'JULIA_PKG_PRECOMPILE_AUTO=0 /root/.juliaup/bin/julia --project -t 16'
  sleep 0.2
  tmux send-keys -t "$pane" Enter
  code="ENV[\"QSX2_ONLY\"] = \"$arm\"; started = time(); try include(\"experiments/scotland/06_qs_joint_and_market_observation/r05_grid.jl\"); println(\"PHASE3_INCLUDE_RETURNED_$arm wall_s=\", time()-started); write(\"$status\", \"PASS\") catch err; showerror(stderr, err, catch_backtrace()); println(stderr); println(\"PHASE3_FAIL_$arm wall_s=\", time()-started); write(\"$status\", \"FAIL\") end; write(\"$notify\", read(\"$status\", String) * \"\\n\")"
  sleep 8
  tmux send-keys -t "$pane" -l -- "$code"
  sleep 0.2
  tmux send-keys -t "$pane" Enter
  IFS= read -r outcome < "$notify"
  rm "$notify"
  printf 'END %s %s %s\n' "$name" "$outcome" "$(date -u +%FT%TZ)"
  if [[ "$outcome" != PASS ]]; then
    echo "PHASE3_BLOCKED: $arm; retain pane=$pane for diagnosis"; exit 1
  fi
  tmux kill-session -t "$session"
done
echo 'PHASE3_GRIDS_PASS; next prescribed Phase 4 scoring and Phase 5 reproduction'

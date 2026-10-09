#!/usr/bin/env bash
# Only authorised reproduction fits: fold 1 of each new arm, fresh REPLs,
# frozen attempt-0 per-chain seeds. Stop on the first mismatch; never refit references.
set -euo pipefail
cd /root/BF_runs/qs_experiment
logdir=/root/BF_runs/logs/qs_experiment_w2
[[ $(<"$logdir/phase4_score_v2.status") == PASS &&
   $(<"$logdir/phase4_interval.status") == PASS ]] || { echo 'Phase4 gates incomplete'; exit 1; }
for arm in grw_joint qs_joint grw_marketobs qs_marketobs; do
  name="phase5_repro_$arm"
  session="pi_qsx2_$name"
  status="$logdir/$name.status"
  notify="$logdir/$name.notify"
  [[ ! -e "$status" && ! -e "$logdir/$name.log" ]] || { echo 'Existing reproduction evidence: stop'; exit 1; }
  mkfifo "$notify"
  pane=$(tmux new-session -d -P -F '#{pane_id}' -s "$session" -c "$PWD")
  printf 'START %s pane=%s %s\n' "$name" "$pane" "$(date -u +%FT%TZ)"
  tmux pipe-pane -o -t "$pane" "cat >> $logdir/$name.log"
  tmux send-keys -t "$pane" -l -- 'JULIA_PKG_PRECOMPILE_AUTO=0 /root/.juliaup/bin/julia --project -t 16'
  sleep 0.2
  tmux send-keys -t "$pane" Enter
  sleep 8
  code="ENV[\"QSX2_ONLY\"] = \"$arm\"; try include(\"experiments/scotland/06_qs_joint_and_market_observation/r07_reproduce.jl\"); write(\"$status\", \"PASS\") catch err; showerror(stderr, err, catch_backtrace()); println(stderr); write(\"$status\", \"FAIL\") end; write(\"$notify\", read(\"$status\", String) * \"\\n\")"
  tmux send-keys -t "$pane" -l -- "$code"
  sleep 0.2
  tmux send-keys -t "$pane" Enter
  IFS= read -r outcome < "$notify"
  rm "$notify"
  printf 'END %s %s %s\n' "$name" "$outcome" "$(date -u +%FT%TZ)"
  [[ "$outcome" == PASS ]] || { echo "PHASE5_BLOCKED: retain pane=$pane"; exit 1; }
  tmux kill-session -t "$session"
done
echo PHASE5_REPRODUCTIONS_PASS

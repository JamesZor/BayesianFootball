#!/usr/bin/env bash
# Wave 3 Phase 4 (beast only): pins, then each named step in its own fresh persistent REPL with
# FIFO completion (no polling). Saved fits only: no sampling, refits or score writes, except
# `record`, which writes only this experiment's register row and the canonical boards.
# Usage: phase4_gate.sh <attempt> <step>...; steps: score leakage record.
set -euo pipefail
attempt=${1:?usage: phase4_gate.sh <attempt> <step>...}
shift
(($# > 0)) || { echo 'usage: phase4_gate.sh <attempt> <step>...'; exit 2; }
cd /root/BF_runs/qs_experiment
logdir=/root/BF_runs/logs/qs_experiment_w3/phase4/$attempt
mkdir -p "$logdir" /root/BF_runs/qs_experiment_w3_out/phase4

snapshot=.cache/datastore_ScottishLower.jls
table=experiments/scotland/06_qs_joint_and_market_observation/results/market_rates.csv
want_snapshot=c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4
want_table=680bc2283e9c0fc7a896bf01b4ba209cb3aae06f50d00667c98b02f379872549
got_snapshot=$(sha256sum "$snapshot" | cut -d' ' -f1)
got_table=$(sha256sum "$table" | cut -d' ' -f1)
mtime=$(date -u -r "$snapshot" '+%F %T.%N')
printf 'SNAPSHOT sha=%s mtime=%s UTC\nTABLE sha=%s\n' "$got_snapshot" "$mtime" "$got_table"
[[ $got_snapshot == "$want_snapshot" && $got_table == "$want_table" &&
   $mtime == '2026-09-25 12:57:15.480765468' ]] || { echo 'PHASE4_BLOCKED: pin mismatch'; exit 1; }
printf 'COMMIT %s ATTEMPT %s\n' "$(git rev-parse HEAD)" "$attempt"

# run_step <name> <julia code>: fresh REPL, wait on a FIFO for PASS/FAIL, keep a failed pane.
run_step() {
  local name=$1 body=$2
  local session="claude_qsx3_p4_$name" status="$logdir/$name.status" notify="$logdir/$name.notify"
  [[ ! -e "$status" && ! -e "$logdir/$name.log" ]] || {
    echo "PHASE4_BLOCKED: existing evidence for $name; use a fresh attempt label"; exit 1;
  }
  mkfifo "$notify"
  local pane
  pane=$(tmux new-session -d -P -F '#{pane_id}' -s "$session" -c "$PWD")
  printf 'START %s pane=%s %s\n' "$name" "$pane" "$(date -u +%FT%TZ)"
  tmux pipe-pane -o -t "$pane" "cat >> $logdir/$name.log"
  tmux send-keys -t "$pane" -l -- 'JULIA_PKG_PRECOMPILE_AUTO=0 /root/.juliaup/bin/julia --project -t 16'
  sleep 0.2
  tmux send-keys -t "$pane" Enter
  local code="started = time(); try $body; println(\"PHASE4_INCLUDE_RETURNED_$name wall_s=\", time()-started); write(\"$status\", \"PASS\") catch err; showerror(stderr, err, catch_backtrace()); println(stderr); println(\"PHASE4_FAIL_$name wall_s=\", time()-started); write(\"$status\", \"FAIL\") end; write(\"$notify\", read(\"$status\", String) * \"\\n\")"
  sleep 8
  tmux send-keys -t "$pane" -l -- "$code"
  sleep 0.2
  tmux send-keys -t "$pane" Enter
  local outcome
  IFS= read -r outcome < "$notify"
  rm "$notify"
  printf 'END %s %s %s\n' "$name" "$outcome" "$(date -u +%FT%TZ)"
  if [[ "$outcome" != PASS ]]; then
    echo "PHASE4_BLOCKED: $name; retain pane=$pane for diagnosis"; exit 1
  fi
  tmux kill-session -t "$session"
}

for step in "$@"; do
  case $step in
    score) run_step score 'include("experiments/scotland/07_qs_fusion_market_bias/r06_score.jl")' ;;
    leakage) run_step leakage 'include("experiments/scotland/07_qs_fusion_market_bias/r10_leakage_audit.jl")' ;;
    record) run_step record 'include("experiments/scotland/07_qs_fusion_market_bias/r08_record.jl")' ;;
    *) echo "PHASE4_BLOCKED: unknown step $step"; exit 2 ;;
  esac
done
echo "PHASE4_STEPS_PASS $*"

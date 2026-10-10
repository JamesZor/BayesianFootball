#!/usr/bin/env bash
# Wave 3 Phase 5 (beast only; wave 2's phase5_gate.sh). The only authorised fits: fold 1 of each
# new arm, a fresh REPL per arm, frozen attempt-0 seeds. Stop on the first mismatch; never refit
# references or retry. Then the original grid checkpoints must be byte-unchanged, and the record
# step sets the register status to `completed` in its own fresh REPL.
# Usage: phase5_gate.sh <attempt>
set -euo pipefail
attempt=${1:?usage: phase5_gate.sh <attempt>}
cd /root/BF_runs/qs_experiment
logdir=/root/BF_runs/logs/qs_experiment_w3/phase5/$attempt
out=/root/BF_runs/qs_experiment_w3_out/phase5
[[ ! -e $logdir ]] || { echo "PHASE5_BLOCKED: existing evidence in $logdir"; exit 1; }
[[ ! -e $out ]] || { echo "PHASE5_BLOCKED: existing reproduction output $out"; exit 1; }
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
   $mtime == '2026-09-25 12:57:15.480765468' ]] || { echo 'PHASE5_BLOCKED: pin mismatch'; exit 1; }
printf 'COMMIT %s ATTEMPT %s\n' "$(git rev-parse HEAD)" "$attempt"

grid=data/checkpoints/scottish_lower_qs_wave3_2426
checkpoint_digests() { find "$grid" -type f -print0 | sort -z | xargs -0 sha256sum; }
checkpoint_digests > "$out/grid_checkpoint_sha256_before.txt"
printf 'GRID_CHECKPOINTS files=%s\n' "$(wc -l < "$out/grid_checkpoint_sha256_before.txt")"

# run_step <name> <julia code>: fresh REPL, wait on a FIFO for PASS/FAIL, keep a failed pane.
run_step() {
  local name=$1 body=$2
  local session="claude_qsx3_p5_$name" status="$logdir/$name.status" notify="$logdir/$name.notify"
  mkfifo "$notify"
  local pane
  pane=$(tmux new-session -d -P -F '#{pane_id}' -s "$session" -c "$PWD")
  printf 'START %s pane=%s %s\n' "$name" "$pane" "$(date -u +%FT%TZ)"
  tmux pipe-pane -o -t "$pane" "cat >> $logdir/$name.log"
  tmux send-keys -t "$pane" -l -- 'JULIA_PKG_PRECOMPILE_AUTO=0 /root/.juliaup/bin/julia --project -t 16'
  sleep 0.2
  tmux send-keys -t "$pane" Enter
  local code="started = time(); try $body; println(\"PHASE5_INCLUDE_RETURNED_$name wall_s=\", time()-started); write(\"$status\", \"PASS\") catch err; showerror(stderr, err, catch_backtrace()); println(stderr); println(\"PHASE5_FAIL_$name wall_s=\", time()-started); write(\"$status\", \"FAIL\") end; write(\"$notify\", read(\"$status\", String) * \"\\n\")"
  sleep 8
  tmux send-keys -t "$pane" -l -- "$code"
  sleep 0.2
  tmux send-keys -t "$pane" Enter
  local outcome
  IFS= read -r outcome < "$notify"
  rm "$notify"
  printf 'END %s %s %s\n' "$name" "$outcome" "$(date -u +%FT%TZ)"
  if [[ "$outcome" != PASS ]]; then
    echo "PHASE5_BLOCKED: $name; retain pane=$pane for diagnosis"; exit 1
  fi
  tmux kill-session -t "$session"
}

for arm in fusion_qs_bias fusion_qs_nobias fusion_grw_bias; do
  run_step "repro_$arm" "ENV[\"QSX3_ONLY\"] = \"$arm\"; include(\"experiments/scotland/07_qs_fusion_market_bias/r07_reproduce.jl\")"
done

checkpoint_digests > "$out/grid_checkpoint_sha256_after.txt"
cmp -s "$out/grid_checkpoint_sha256_before.txt" "$out/grid_checkpoint_sha256_after.txt" ||
  { echo 'PHASE5_BLOCKED: original grid checkpoints changed'; exit 1; }
echo 'GRID_CHECKPOINTS_UNCHANGED true'

run_step record 'ENV["QSX3_RECORD_STATUS"] = "completed"; include("experiments/scotland/07_qs_fusion_market_bias/r08_record.jl")'
echo 'PHASE5_REPRODUCTIONS_PASS arms=3 fold=1 record=completed'

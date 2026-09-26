#!/usr/bin/env bash
set -euo pipefail

TARGET_EPOCH=1790283516 # 2026-09-24 21:58:36 BST
SESSION_NAME="agent_pi_scotland_cross_tier"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Scheduler started. Target reset time: $(date -d @${TARGET_EPOCH} '+%Y-%m-%d %H:%M:%S %Z')"

while true; do
    NOW=$(date +%s)
    REMAINING=$((TARGET_EPOCH - NOW))
    if [ "$REMAINING" -le 0 ]; then
        break
    fi
    if [ "$REMAINING" -gt 300 ]; then
        echo "[$(date '+%H:%M:%S')] $((REMAINING / 60)) minutes remaining until reset..."
        sleep 300
    elif [ "$REMAINING" -gt 60 ]; then
        echo "[$(date '+%H:%M:%S')] $((REMAINING / 60)) minutes remaining until reset..."
        sleep 60
    else
        echo "[$(date '+%H:%M:%S')] $REMAINING seconds remaining..."
        sleep 5
    fi
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Quota reset time reached. Waiting 10s for backend token refresh..."
sleep 10

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Refreshing Pi usage..."
tmux send-keys -t "$SESSION_NAME" "/usage --refresh" Enter
sleep 3

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Injecting continuation prompt into tmux session: $SESSION_NAME..."
tmux send-keys -t "$SESSION_NAME" "The 5-hour Codex usage limit has reset to 0%. Please run r01_smoke_test.jl across all five candidates and proceed with launching the 40-fold production grid on mcmc-beast." Enter

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Resume signal dispatched successfully."

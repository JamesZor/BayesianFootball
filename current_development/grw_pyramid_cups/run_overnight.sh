#!/usr/bin/env bash
# Overnight orchestration (TODO 029 follow-up): smoke -> live extensions -> production arms that passed smoke.
set -u
cd /root/BF_grw_pyramid_cups
set -a; source .env; set +a
J=/root/.juliaup/bin/julia
D=current_development/grw_pyramid_cups
L=$D/results/logs; mkdir -p $L
O=$L/orchestrator.log
echo "START $(date -Is) rev $(cat GIT_REV)" >> $O

if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  $J --project -t 16 $D/r01_smoke.jl > $L/r01_smoke.log 2>&1
  echo "SMOKE_EXIT $? $(date -Is)" >> $O
fi

if [ "${SKIP_EXT:-0}" != "1" ]; then
  ( cd /root/BayesianFootball && set -a && source .env && set +a && \
    $J --project -t 16 experiments/scottish_lower/06_joint_player_lineup_fusion/r68_extend_joint_player_2627.jl m12_joint_hybrid_synergy --refresh ) \
    > $L/r68_extend_m12_td.log 2>&1
  echo "EXT_TD_EXIT $? $(date -Is)" >> $O
  $J --project -t 16 $D/r03_extend_m12_grw.jl > $L/r03_extend_m12_grw.log 2>&1
  echo "EXT_GRW_EXIT $? $(date -Is)" >> $O
fi

ARMS=${PCX_ARMS:-$(grep "^R01_ARM .* PASS" $L/r01_smoke.log 2>/dev/null | awk '{print $2}' | paste -sd, -)}
if [ -n "$ARMS" ]; then
  echo "PROD_ARMS $ARMS $(date -Is)" >> $O
  PCX_ARMS=$ARMS $J --project -t 16 $D/r02_overnight.jl > $L/r02_overnight_$(date +%H%M).log 2>&1
  echo "PROD_EXIT $? $(date -Is)" >> $O
else
  echo "PROD_SKIPPED no arm passed smoke $(date -Is)" >> $O
fi
echo "ALL_DONE $(date -Is)" >> $O

#!/usr/bin/env bash
# TODO 003 GC runtime benchmark driver: one Julia process per GC configuration, run
# sequentially so each owns the whole machine. Usage: run_gc_bench.sh TAG CONFIG REFERENCE [julia flags...]
set -uo pipefail
cd /root/BF_goal_decomposition
tag=$1; cfg=$2; ref=$3; shift 3
out=experiments/scottish_lower/08_goal_decomposition/results/sampling_budget_fold1
echo "=== $tag · config $cfg · reference ${ref:-none} · flags: $* · $(date -Is)" | tee "$out/$tag.log"
env JULIA_PKG_PRECOMPILE_AUTO=0 L08_RUN_BENCH=true L08_BENCH_TAG="$tag" \
    L08_BENCH_MODELS=m00_recombined_control L08_BENCH_CONFIGS="$cfg" L08_BENCH_REPS=4 \
    L08_BENCH_GRID_AUDIT=false L08_BENCH_REFERENCE="$ref" L08_BENCH_AD_FIX="${L08_BENCH_AD_FIX:-false}" \
    /root/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia --project -t 16 "$@" \
    experiments/scottish_lower/08_goal_decomposition/r08_sampling_budget_benchmark.jl 2>&1 | tee -a "$out/$tag.log"
echo "=== $tag exit ${PIPESTATUS[0]} · $(date -Is)" | tee -a "$out/$tag.log"

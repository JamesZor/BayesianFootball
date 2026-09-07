# Market Microstructure & Dynamic Staged Execution Engine

## 1. Overview
This research stream addresses the operational execution bottleneck identified during live trading of the Scottish Lower Option B slate on Saturday 2026-09-05.

While the predictive engine (`m12_joint_hybrid_synergy` Fold 43 + confirmed BBC pre-match lineups) demonstrated verified directional alpha, the single-shot `TouchOnly` execution policy at T−25 left ~58% of the calculated Kelly risk volume unmatched due to thin top-of-book depth.

## 2. Active Research Package
Full details, mathematical mandates, and infrastructure context are specified in:
[`experiments/pi_microstructure_astra_prompt.md`](../../experiments/pi_microstructure_astra_prompt.md)

## 3. Work In Progress
- `REPORT.md`: Empirical order book liquidity analysis and mathematical derivation of multi-level sweeping vs WOM order flow.
- `l01_microstructure_sweeper.jl`: Microstructure engine, multi-level reservation price kernels, and execution policy abstractions.
- `r01_microstructure_sweeper.jl`: Historical replay and simulation runner evaluating execution performance across Scottish Lower books.

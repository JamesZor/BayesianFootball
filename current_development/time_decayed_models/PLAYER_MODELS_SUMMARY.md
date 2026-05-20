# Player Models & Time Decay: Stability Analysis Summary

**Date:** May 20, 2026
**Context:** Development and stability testing of `DynamicMarketXGPlayerModel` across walk-forward CV folds, using both Last-Value and Bayesian Tracker player ratings.

## 🛠️ Code Artifacts Created

1. **`l05_stability_analysis.jl` (The Extractor)**
   - Created a robust parameter extraction suite (`extract_stability_dataframe`) to evaluate model parameter drift over time.
   - **Capabilities:** Reconstructs Non-Centered Parameters (NCP) for components like Home Advantage and Kappa, applies necessary transformations (e.g., Softplus), and dynamically maps indices to actual team names per CV fold.

2. **`r06_investigate_chains.jl` (The Diagnoser)**
   - Developed to investigate folds with high variance identified by the stability DataFrame.
   - Evaluates MCMC health using **R-hat ($\hat{R}$)** and **Effective Sample Size (ESS)** for parameters to distinguish between genuine data shifts and sampler convergence failures.

## 📊 Key Findings: MCMC Convergence

- **Symptom:** Certain folds (e.g., Fold 3, 5, 6, 11) exhibited massive spikes in parameter variance (standard deviation jumping from ~0.003 to > 0.40) and erratic mean shifts.
- **Diagnosis:** Running `r06_investigate_chains.jl` confirmed these anomalies were caused by **MCMC Convergence Failure**.
- **Evidence:** R-hat values spiked to > 2.0 (well above the safe threshold of 1.05) and ESS dropped as low as ~2.7. 
- **Cause:** The sampler (NUTS) was failing to mix and explore the complex posterior surface within the limited 200-step warmup and 500 total samples.

## 💡 Architectural Insight: Dimensionality Reduction

**Observation:** In the stable folds (where R-hat < 1.05), the inferred positional weights for **Defenders (`w_D`)**, **Midfielders (`w_M`)**, and **Forwards (`w_F`)** were remarkably similar. 
- *Example (Attacking Weights, Last Value Tracker):*
  - `w_D_att` ≈ 0.034
  - `w_M_att` ≈ 0.035
  - `w_F_att` ≈ 0.038

**Conclusion:** Differentiating between D, M, and F ratings adds unnecessary dimensionality to the posterior, which exacerbates MCMC convergence issues and requires significantly longer warmup phases without adding predictive value. 

**Next Steps / Action Plan:**
Simplify `PositionalPlayerDynamics` from 8 weights to **4 weights**:
- **Goalkeeper (`w_G`)**: Keep separate, as their impact on defending is mechanically unique.
- **Outfield (`w_O`)**: Collapse D, M, and F ratings into a single weighted/average outfield player rating metric for both attacking (`w_O_att`) and defending (`w_O_def`).

*Benefits of this change:* Reduces the parameter space, improves MCMC mixing/stability, and allows models to train faster and converge more reliably on sparse weekly data.

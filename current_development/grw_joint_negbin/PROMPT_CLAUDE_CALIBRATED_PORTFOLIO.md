# Task: Raw vs Calibrated Portfolio Backtesting (Closing Line & T−25) — NegBin vs Poisson Controls

## Context & Motivation

In Task 014, the 40-fold walk-forward grid on Scottish Lower (seasons 24/25 + 25/26, 710 matches) evaluated the Negative Binomial observation ladder against its Task 013 Poisson counterparts under the **raw** Option B portfolio on de-vigged closing odds (`r05_portfolio.jl`).
The raw simulation revealed that while bet overlap was 92–94%, NegBin shifted mass into 0 goals, lowering $P(\text{Over } 1.5)$ and declining 30–40% of Over 1.5 bets that were profitable, leading to a small drag on raw ROI.

The trader has requested a follow-up check:
> "did it run the portfolio backtesting with some of that standard raw and calibration version just to check against the market? can task a fresh / cleared claude agent cli to run this check and report back"

Your mission is to evaluate the **Raw vs Calibrated** versions of all four NegBin rungs and their Poisson controls against both the **closing line** and **T−25 tradeable point-in-time order books**, assessing whether Layer 2 Generative Rate Calibration (`calibrate_fit` / `GenerativeRateCalibrator`) changes the relative standing or restores alpha.

---

## 1. Models & Persisted Runs

All 8 fits are already trained and persisted in `mcmc_experiments`. **Do NOT re-sample or fit any MCMC models.** Simply load the fits via `gjn_load_arm(arm)` / `load_fit_db` from their respective namespaces:

### NegBin Models (Experiment: `scottish_lower_grw_joint_negbin`)
- `m00_baseline_grw_negbin`: `0c0da991-7d4c-4f01-90e2-2af157f27aaa`
- `m05_wealth_grw_negbin`: `019d41d4-9e0e-41eb-bbc1-8984263f0f14`
- `m10_lineup_grw_negbin`: `f7fd8385-fa15-4f6a-ae89-e23447907a80`
- `m12_joint_hybrid_synergy_negbin`: `c3d2aede-ddd5-4590-9c75-5937de4b1bbc`

### Poisson Controls (Experiment: `scottish_lower_grw_player_hybrid`)
- `m00_baseline_grw`: `158d2a80-7ea3-4d6c-b3ab-be62bcf1bc11`
- `m05_wealth_grw`: `b0961bc4-c40c-4dbe-9c05-57df7ae0839e`
- `m10_lineup_grw`: `b13c8fb9-ce34-4210-aa3f-9d2ed493c286`
- `m12_joint_hybrid_synergy_grw`: `3a9a4c7e-378b-45d0-a2d2-c8b69b46786b`

(Note: Restrict all fits to the common 710-fixture panel using `gjn_restrict(fit, panel_ids)` as in `l02_evaluation.jl`).

---

## 2. Experimental Setup

Create `current_development/grw_joint_negbin/r06_calibrated_portfolio.jl`.

Use the production Option B book and policy specifications (`MatchDay.option_b_system()`) and evaluate across two market environments:

### Environment A: De-vigged Betfair Closing Line (Instant = 0.0 min)
- **Book**: `r05_odds` (de-vigged TWA(−20, 0] closing odds).
- **Calibrator**: Production closing calibrator:
  `cal_close = canonical_calibrator(:scot_lower_close_std)`
  (i.e., `GenerativeRateCalibrator("scot_lower_close_std", law = StandardGaussianLaw(w_base = 0.85, sigma = 0.15), book_as_of_minutes = 0.0)`).
- **Runs**:
  - Raw vs Calibrated for all 8 arms.

### Environment B: Tradeable T−25 Point-in-Time Book (Instant = −25.0 min)
- **Book**: `book_t25, refusals = Calibration.point_in_time_book(ds; config = PointInTimeBookConfig(as_of_minutes = -25.0))`.
- **Calibrator**: Production T−25 calibrator:
  `cal_t25 = canonical_calibrator(:scot_lower_t25_inv)`
  (i.e., `GenerativeRateCalibrator("scot_lower_t25_inv", law = InverseGaussianLaw(w_base = 0.25, sigma = 0.35), book_as_of_minutes = -25.0)`).
- **Runs**:
  - Raw vs Calibrated for all 8 arms against `book_t25`.

---

## 3. Core Questions to Answer

1. **Impact of Calibration**: How does pooling log-rate draws with the market affect NegBin vs Poisson?
   - In `calibrate_fit(cal, fit, book)`, rates $\lambda_h, \lambda_a$ are pooled towards market implied rates. Does this reduce the divergence between NegBin and Poisson, or does the dispersion $r$ still drive distinct portfolio behavior?
2. **The Over 1.5 Drag**: In raw Option B, NegBin declined ~30–40% of Over 1.5 bets because of extra mass at zero goals. Does calibration adjust $\lambda$ enough to recover those bets, or does NegBin remain structurally more selective on low totals?
3. **Comparative Performance**:
   - Total Return %, ROI %, Annualized Sharpe, Max Drawdown, Win Rate, Total Bets.
   - Breakdown by market family: 1X2 (Home/Away/Draw), Under 2.5, Over 1.5.

---

## 4. Execution Protocol

- Execute via SSH on `mcmc-beast` inside a tmux session (e.g. `negbin_cal`) or persistent session.
- Output summary tables to `current_development/grw_joint_negbin/results/r06_calibrated_portfolio_summary.csv`.
- Document key findings in `current_development/grw_joint_negbin/README.md` (or an appended section) and summarize results for the user.

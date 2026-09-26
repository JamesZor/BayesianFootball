# Experiment Harness Suite Catalogue — Scottish Analytics

> **Status:** Input specification for TODO 030 (Unified Experiment Harness).  
> **Scope:** Audited catalogue of experiment harness code, gates, metrics, helpers, recorded results, inconsistencies, and library reusability across Scottish model suites (Generations 1–4, follow-on prototypes, and cross-tier studies).  
> **Output file:** `docs/architecture/experiment_harness_suite_catalogue.md`.
>
> **Provenance and verification (2026-09-26, @claude).** Produced read-only by a pi agent
> (`ag/gemini-3.8-flash-high`). Spot-checked against source:
> - **Confirmed.** A.40 `gph_gradient_audit` (actual line 444); the 5.3125% tail mass
>   (suite 10 README:358); A.39, suite 08's divergence `@warn` "flagged for review";
>   A.20, r60's tape length < 500.
> - **Wrong, do not rely on.** §E's `ConvergenceThresholds` defaults. The code
>   (`src/training/inference/types.jl:242`) has `min_ess = 400.0` and
>   `max_divergence_rate = 0.001`, not 100 / 0.0.
> - **Approximate.** §E's `src/` line numbers (e.g. `fit_model` is `engine.jl:470`,
>   `save_fit` `db_storage.jl:712`, `config_hash` `db_storage.jl:198`); re-grep before citing.
> - **Wrong.** §F.2's claim that `max_divergence_rate = eps()` failed on zero divergences.
>   The gate is "below", so 0 < eps() passes, and all five 028 smoke fits had 0 divergences
>   and passed that check. They failed on R-hat and ESS.
> - Sections A–D were cross-read against the READMEs and agree on every row checked.

---

## A. Gates

This table catalogues every verification gate implemented across the Scottish experiment suites, identifying what property is checked, thresholds, severity, and exact code location.

| Suite | Gate Name | What it checks | Threshold(s) | Hard fail or warning | Implementation (path:line) | Notes |
|---|---|---|---|---|---|---|
| `05_player_lineup_and_pxg_fusion` | `r50_assert_structural_contract` | Parameter count, grouped sample sites, chain site names, counterfactual sensitivity | Parameter count matches `r50_expected_params(name, n_teams)`: 35 (m05), 36 (m09–m11), 37 (m12); `λ_h` changes under counterfactual | Hard fail (`@test`) | `experiments/scottish_lower/05_player_lineup_and_pxg_fusion/r50_smoke_test_player_models.jl:72` | Verifies lineup weights and team effect sites are actually wired into graph |
| `05_player_lineup_and_pxg_fusion` | `r50_assert_six_gate_audit` | Standard Turing MCMC convergence telemetry on 1 smoke fold | `max_rhat <= 1.05`, `min_ess_bulk >= 100.0`, `min_ess_tail >= 100.0`, `n_divergent == 0`, `min_bfmi >= 0.30`, `treedepth_rate < 0.05` | Hard fail (`@test`) | `experiments/scottish_lower/05_player_lineup_and_pxg_fusion/r50_smoke_test_player_models.jl:117` | Requires exact zero divergences via `eps(Float64)` comparator |
| `05_player_lineup_and_pxg_fusion` | `r50_assert_parameter_extraction` | Latent extraction, match coverage, finite positive rates, SmileScoreGrid structure | All OOS fixtures covered; all `λ_h, λ_a > 0.0` and finite; `latents isa CountLatents`; `grid isa SmileScoreGrid` | Hard fail (`@test`) | `experiments/scottish_lower/05_player_lineup_and_pxg_fusion/r50_smoke_test_player_models.jl:128` | Wraps count latents with neutral `φ=1` curve to exercise score grid |
| `05_player_lineup_and_pxg_fusion` | `r50_assert_round_trip` | PostgreSQL `save_fit` and `load_fit` bit-exact parity | Chains, chain names, latent match IDs, and rates match exactly | Hard fail (`@test`) | `experiments/scottish_lower/05_player_lineup_and_pxg_fusion/r50_smoke_test_player_models.jl:169` | Round-trip serialization through `PostgresStorage` |
| `05_player_lineup_and_pxg_fusion` | Portfolio round-trip parity | `save_portfolio_db` and `load_portfolio_db` bet ledger parity | `loaded.summary.total_return_pct == result.summary.total_return_pct`; `isequal(loaded.bets, original.bets)` | Hard fail (`@test`) | `experiments/scottish_lower/05_player_lineup_and_pxg_fusion/r50_smoke_test_player_models.jl:249` | Asserts bet-level reproducible ledger from database artifact |
| `06_joint_player_lineup_fusion` | Gate 1: `r60_assert_gradient_tape` | ReverseDiff compiled gradient tape finite, instruction length structural, replay latency | Replay latency `< 0.05 ms`; `length(raw.tape) < 500`; compiled gradient finite | Hard fail (`@test`) | `experiments/scottish_lower/06_joint_player_lineup_fusion/r60_smoke_test_joint_player_models.jl:101` | Measures minimum of 200 replays after 20 warm-ups; verifies tape does not grow with N |
| `06_joint_player_lineup_fusion` | Gate 2: `r60_assert_structural_contract` | Expected parameters, site presence, counterfactual responsiveness | 35 (m05), 36 (m09–m11), 37 (m12), 38 (m13); `cb_parameter_count == n_params` | Hard fail (`@test`) | `experiments/scottish_lower/06_joint_player_lineup_fusion/r60_smoke_test_joint_player_models.jl:145` | Tests `m13` distance covariate weight site inclusion |
| `06_joint_player_lineup_fusion` | Gate 3: Sampling completion | 1 smoke fold NUTS sampling completes without chain crash | 4 chains x 200 warmup + 200 retained; no unhandled exception | Hard fail | `experiments/scottish_lower/06_joint_player_lineup_fusion/r60_smoke_test_joint_player_models.jl:301` | Exercises `fit_model` under `ThreadedExecution` |
| `06_joint_player_lineup_fusion` | Gate 4: `r60_assert_six_gate_audit` | Six-part convergence audit | `max_rhat <= 1.05`, `min_ess >= 100`, `n_divergent == 0`, `min_bfmi >= 0.30`, `treedepth_rate < 0.05` | Hard fail (`@test`) | `experiments/scottish_lower/06_joint_player_lineup_fusion/r60_smoke_test_joint_player_models.jl:188` | Standard six-part audit |
| `06_joint_player_lineup_fusion` | Gate 5: `r60_assert_parameter_extraction` | CountLatents coverage, rates finite positive, SmileScoreGrid | Covered match IDs match OOS; rates positive; neutral grid valid | Hard fail (`@test`) | `experiments/scottish_lower/06_joint_player_lineup_fusion/r60_smoke_test_joint_player_models.jl:199` | Exercises `SmileScoreGrid` pricing |
| `06_joint_player_lineup_fusion` | Gate 6: `r60_assert_round_trip` | DB Fit serialization parity | Reloaded fit identical to original fit | Hard fail (`@test`) | `experiments/scottish_lower/06_joint_player_lineup_fusion/r60_smoke_test_joint_player_models.jl:240` | Exact equality of `Chains` and `CountLatents` |
| `06_joint_player_lineup_fusion` | Gate 7: Portfolio parity | Portfolio simulation and PostgreSQL persistence round-trip | Equal return %, equal bet ledger rows | Hard fail (`@test`) | `experiments/scottish_lower/06_joint_player_lineup_fusion/r60_smoke_test_joint_player_models.jl:324` | Persists to `portfolio_runs` and reloads via `load_portfolio_db` |
| `06_joint_player_lineup_fusion` (hierarchical κ) | G1: `r64_assert_gradient_tape` | ReverseDiff compiled tape scaling across different fold sizes, latency | Small vs large fold tape length equal (`length(raw_large.tape) == length(raw.tape)`); latency `< 0.05 ms` | Hard fail (`@test`) | `experiments/scottish_lower/06_joint_player_lineup_fusion/r64_smoke_hierarchical_kappa.jl:146` | Proves tape length is invariant to fixture sample size ($O(1)$) |
| `06_joint_player_lineup_fusion` (hierarchical κ) | G4: `r64_assert_six_gate_audit` | Six-part audit with heightened ESS budget | `max_rhat < 1.05`, `min_ess_bulk > 300`, `min_ess_tail > 300`, `n_divergent == 0`, `min_bfmi >= 0.30`, `treedepth < 0.05` | Hard fail (`@test`) | `experiments/scottish_lower/06_joint_player_lineup_fusion/r64_smoke_hierarchical_kappa.jl:199` | Heightened ESS threshold (300 vs 100) due to hierarchical finishing scale |
| `06_joint_player_lineup_fusion` (hierarchical κ) | G5: `r64_assert_kappa` | Mathematical sanity of hierarchical finishing deltas | `k.mode === :hierarchical`; `sum(k.δ_κ, dims=2) < 1e-10` (sum to zero); positive scales | Hard fail (`@test`) | `experiments/scottish_lower/06_joint_player_lineup_fusion/r64_smoke_hierarchical_kappa.jl:211` | Verifies identification constraint $\sum_t \delta_{\kappa, t} = 0$ |
| `06_joint_player_lineup_fusion` (hierarchical κ) | G9: `r64_assert_reference_parity` | Pure Julia mathematical log-joint parity vs DynamicPPL density | Relative error `rtol <= 1e-9` at base $\theta$; `rtol <= 1e-8` at perturbed points | Hard fail (`@test`) | `experiments/scottish_lower/06_joint_player_lineup_fusion/r64_smoke_hierarchical_kappa.jl:297` | Tests AD objective against hand-coded `cb_logjoint` reference kernel |
| `07_calibrated_portfolio_and_trust_vector` | Canonical policy reproduced | Hard verification that local trust configuration reproduces shipped policy | Discrepancy `= 0.0` across all 13 tradeable keys | Hard fail (`error`) | `experiments/scottish_lower/07_calibrated_portfolio_and_trust_vector/l07_calibrated_trust_loader.jl:299` | Guards against drift between runner's geometric ladder and `CanonicalScottishLowerTrust` |
| `07_calibrated_portfolio_and_trust_vector` | `l07_gate_ledger_accounting` | Internal consistency of trade ledger payoffs, PnL, and bankroll tracking | `abs(pnl - stake*(odds-1)) <= 1e-8` on win; `pnl == -stake` on loss | Hard fail (`error`) | `experiments/scottish_lower/07_calibrated_portfolio_and_trust_vector/l07_calibrated_trust_loader.jl:562` | Ensures zero arithmetic drift in portfolio simulation trajectory |
| `07_calibrated_portfolio_and_trust_vector` | Gate 3: Production Portfolio Benchmark | Out-of-sample portfolio deployability gate | `oos_sharpe >= 1.65`, `oos_calmar >= 8.0`, `oos_mdd_pct >= -18.0%` | Hard fail / decision filter | `experiments/scottish_lower/07_calibrated_portfolio_and_trust_vector/r07_optimal_portfolio_comparison.jl:486` | Failed by all 168 deployable candidates; revealed Calmar is mis-specified on 50-slate window |
| `08_goal_decomposition` | `l08_gradient_checks` | AD gradient correctness across compiled ReverseDiff, fresh ReverseDiff, and ForwardDiff | Compiled vs fresh `<= 1e-8`; compiled vs ForwardDiff `<= 1e-6`; 40 broad perturbation probes `<= 1e-8` / `<= 1e-6` | Hard fail (`error`) | `experiments/scottish_lower/08_goal_decomposition/l08_model_checks.jl:55` | 40 random coordinate perturbation probes up to scale 0.8 |
| `08_goal_decomposition` | `l08_instruction_scaling` | Tape instruction count invariance under dataset row duplication | Tape length ratio `n_duplicate / n_base == 1.0` | Hard fail (`error`) | `experiments/scottish_lower/08_goal_decomposition/l08_model_checks.jl:98` | Vectorized likelihood check; fails if tape scales with fixture rows |
| `08_goal_decomposition` | `l08_filtration_checks` | Temporal filtration leakage test via future count & referee perturbation | Perturbed future fixtures must not change fitted feature dictionary or referee vocabulary | Hard fail (`error`) | `experiments/scottish_lower/08_goal_decomposition/l08_model_checks.jl:207` | Modifies strictly future matches and asserts `before == after` on training features |
| `08_goal_decomposition` | `l08_grid_allocations` | Zero allocation in score grid computation | Allocations `== 0 bytes` using preallocated `GridWorkspace` | Hard fail (`error`) | `experiments/scottish_lower/08_goal_decomposition/l08_model_checks.jl:229` | Zero-allocation score-grid kernel contract verification |
| `08_goal_decomposition` | `l08_extraction_checks` | Latent extraction, independent Poisson rate reconstruction, score grid parity, unseen team refusal | Rates error `< 1e-10`; grid probability error `< 1e-12`; grid mass `0 < mass <= 1 + 1e-12`; unseen team raises refusal | Hard fail (`error`) | `experiments/scottish_lower/08_goal_decomposition/l08_model_checks.jl:238` | Tests mathematical superposition of regular + penalty + own goal intensities |
| `08_goal_decomposition` | `l08_assert_promotion` | Six-part convergence promotion gate | `passed == true`; `max_rhat <= 1.05`; `min_ess_bulk >= 200`; `min_ess_tail >= 200`; `min_bfmi >= 0.30`; `treedepth_rate < 0.05` | Divergences `> 0` emits **warning**; other criteria **hard fail** | `experiments/scottish_lower/08_goal_decomposition/l08_workflow.jl:249` | Notable divergence: divergences log a `@warn` rather than stopping pipeline |
| `10_momentum_multiscale_grw` | `gph_gradient_audit` | AD correctness, tape instruction scaling, warm latency & allocations | Compiled vs fresh `<= 1e-8`; compiled vs ForwardDiff `<= 1e-6`; 3 perturbed points `<= 1e-8`; latency & bytes measured | Hard fail on parity (`error`); latency/allocations informational | `current_development/grw_player_hybrid/l01_loader.jl:442` (used via `experiments/scottish_lower/10_momentum_multiscale_grw/l10_momentum_grw_loader.jl:3`) | Zero-allocation requirement waived due to known Julia 1.12 ReverseDiff broadcast behavior |
| `10_momentum_multiscale_grw` | `gph_filtration_report` | Assert disjoint training/OOS sets and strictly ordered kickoff timestamps | Training and OOS match IDs intersection `= 0`; `last_train < first_oos` | Hard fail (`error`) | `current_development/grw_player_hybrid/l01_loader.jl:397` | Enforces no fixture is both conditioned on and priced |
| `10_momentum_multiscale_grw` | Score grid partition & tail check | Consistency of 1X2, OU2.5, BTTS partitions with truncated Poisson grid | Partitions agree with CDF product to `<= 1e-12`; omitted tail mass reported | Hard fail on partition equality (`error`); tail mass reported | `experiments/scottish_lower/10_momentum_multiscale_grw/r10_momentum_smoke.jl:74` | Fixed 12x12 grid retains mass check; worst omitted tail mass documented up to 5.31% |
| `10_momentum_multiscale_grw` | Smoke gate certificate | Durable serialization of passing smoke certificate before production grid | All 3 smoke folds (1, 20, 40) pass all gates | Hard fail (`error`) | `experiments/scottish_lower/10_momentum_multiscale_grw/r10_momentum_smoke.jl:108` | Generates `smoke_gate.jls`; production refuses to launch without valid certificate |
| `11_decompression_pxg_covariate` | Preflight & same-day card filtration | Feature filtration: no same-day card outcome leakage into form covariate | Form covariate uses only strictly completed cards; same-day card matches emitted together | Hard fail (`@test`) | `experiments/scottish_lower/11_decompression_pxg_covariate/test_decompression.jl:25` | Validates `PxGFeature` temporal window cutoff |
| `11_decompression_pxg_covariate` | Decompression smoke audit | Six-part convergence audit on smoke folds | `max_rhat <= 1.05`, `min_ess >= 200`, `n_divergent == 0`, `min_bfmi >= 0.30` | Hard fail (`error`) | `experiments/scottish_lower/11_decompression_pxg_covariate/r10_smoke.jl:45` | Uses QueuedNUTS sampler with `accept_rate = 0.90` |
| `12_decoupled_generative_xg` | Decoupled funnel AD & smoke gate | Compiled gradient parity, six-part audit, preflight check | Parity `<= 1e-8`; R-hat `<= 1.05`; ESS `>= 200`; divergences `== 0` | Hard fail (`error`) | `experiments/scottish_lower/12_decoupled_generative_xg/r10_smoke.jl:40` | Tests cut-likelihood and hierarchical team finishing parameterizations |
| `13_pedigree_fulltime_and_tier_priors_eda` | `gradient_report` / `warmed_allocation_report` | Gradient correctness and warmed allocation measurement on mock logjoint | Compiled vs fresh `<= 1e-8`; compiled vs ForwardDiff `<= 1e-6`; allocations recorded | Diagnostic / non-blocking | `experiments/scottish_lower/13_pedigree_fulltime_and_tier_priors_eda/l04_pedigree_tier_components.jl:145` | Reusable EDA gradient benchmark pattern |
| `13_pedigree_fulltime_and_tier_priors_eda` | Status publication contract | Point-in-time publication timestamp pre-kickoff verification | Verification timestamp `< kickoff_timestamp`; no lookahead | Hard fail (`@test`) | `experiments/scottish_lower/13_pedigree_fulltime_and_tier_priors_eda/r10_status_contract_tests.jl:20` | Checks historical club operational status metadata |
| `scotland/01_time_decay_cross_tier_and_priors` | Gate 1: `ct_smoke_tape_gate` | ReverseDiff compiled tape correctness, latency, and strict zero allocation | Compiled vs fresh `<= 1e-8`; compiled vs ForwardDiff `<= 1e-6`; perturbation `<= 1e-8`; latency `< 0.05 ms`; allocation `== 0 bytes` | Hard fail (`CrossTierSmokeGate`) | `experiments/scotland/01_time_decay_cross_tier_and_priors/l02_smoke_helpers.jl:130` | Blocked smoke execution: all candidates allocated 64–136 KB due to installed ReverseDiff stack |
| `scotland/01_time_decay_cross_tier_and_priors` | Gate 2: `ct_smoke_chain_shape` | Chain dimensions match smoke contract | Folds `== 1`, chains `== 2`, retained draws `== 100` | Hard fail (`CrossTierSmokeGate`) | `experiments/scotland/01_time_decay_cross_tier_and_priors/l02_smoke_helpers.jl:176` | Guards against unintended large sampling runs during smoke |
| `scotland/01_time_decay_cross_tier_and_priors` | Gate 3: `ct_smoke_convergence_gate` | Strict six-part convergence audit | `max_rhat <= 1.05`, `min_ess >= 100.0`, `max_divergence_rate = eps()`, `min_bfmi >= 0.30`, `treedepth <= 0.05` | Hard fail (`CrossTierSmokeGate`) | `experiments/scotland/01_time_decay_cross_tier_and_priors/l02_smoke_helpers.jl:187` | In 2x100 smoke, failed R-hat/ESS; in 4x1000 pilot, m00/B1/B2 passed, A1/A2 failed zero divergences |
| `scotland/01_time_decay_cross_tier_and_priors` | Gate 4: `ct_smoke_latent_gate` | CountLatents type, fixture count, and draw count | `latents isa CountLatents`, `n_matches > 0`, `n_draws == 200` | Hard fail (`CrossTierSmokeGate`) | `experiments/scotland/01_time_decay_cross_tier_and_priors/l02_smoke_helpers.jl:207` | Strict type and dimension check |
| `scotland/01_time_decay_cross_tier_and_priors` | Gate 5: `ct_smoke_score_grid_gate` | 12x12 score grid size, partition consistency, and retained unit mass | Grid shape `(12, 12, n_draws)`; partition differences `<= 1e-12`; total mass `0.999 <= mass <= 1.0 + 1e-12` | Hard fail (`CrossTierSmokeGate`) | `experiments/scotland/01_time_decay_cross_tier_and_priors/l02_smoke_helpers.jl:219` | Failed on m03/m04 in smoke and m00/m01/m03/m04 in pilot due to natural Poisson tail truncation |
| `scotland/01_time_decay_cross_tier_and_priors` | Gate 6: `ct_smoke_fit_parity` | Deep structural equality between original and reloaded Fit | Config, folds, chain arrays, diagnostics, and latents match bit-for-bit | Hard fail (`CrossTierSmokeGate`) | `experiments/scotland/01_time_decay_cross_tier_and_priors/l02_smoke_helpers.jl:271` | Custom `ct_smoke_structural_equal` recursive comparison helper |
| `scotland/01_time_decay_cross_tier_and_priors` | Gate 7: `ct_smoke_portfolio_parity` | Structural equality of persisted PortfolioResult | All fields of `PortfolioResult` match on reload | Hard fail (`CrossTierSmokeGate`) | `experiments/scotland/01_time_decay_cross_tier_and_priors/l02_smoke_helpers.jl:298` | Abstained in smoke because Gate 3 convergence failed |
| `grw_pyramid_cups` | G0: Filtration | Held-out fixtures match canonical 56/57 splitter; all training kickoffs precede OOS | Canonical OOS match set bit-exact; added cup rows kickoff `< first_oos` | Hard fail (`error`) | `current_development/grw_pyramid_cups/r01_smoke.jl:46` | Verifies cup bridge rows do not leak future information |
| `grw_pyramid_cups` | G1: ReverseDiff tape | ReverseDiff gradient finite and agrees with ForwardDiff | Compiled gradient finite; relative error vs ForwardDiff `<= 1e-6` | Hard fail (`error`) | `current_development/grw_pyramid_cups/r01_smoke.jl:63` | ForwardDiff parity gate |
| `grw_pyramid_cups` | G2: Smoke sampling | Sampling completes on 2 folds under relaxed smoke threshold | 4 chains x (300 warmup + 300 retained); `divergence_rate < 0.01`; `max_rhat <= 1.10` | Hard fail (`error`) | `current_development/grw_pyramid_cups/r01_smoke.jl:73` | Relaxed R-hat budget specifically acknowledged as smoke-only |
| `grw_pyramid_cups` | G3: Latents | Extraction produces finite positive rates for all fixtures | `all(isfinite, λ_home) && all(>(0), λ_home)`; same for away | Hard fail (`error`) | `current_development/grw_pyramid_cups/r01_smoke.jl:84` | Non-negative rate gate |

---

## B. Metrics and Scoring

This table catalogues every metric, proper score, calibration diagnostic, and portfolio summary used across the Scottish suites.

| Suite | Metric | Markets / Lines | Subsets | Reference Instant | Implementation (path:line) |
|---|---|---|---|---|---|
| `05_player_lineup_and_pxg_fusion` | Binary LogLoss | 1X2, OU2.5, BTTS | All quoted selections pooled (2,899 obs across 627 matches) | Betfair TWA (−20, 0] min | `experiments/scottish_lower/05_player_lineup_and_pxg_fusion/r52_compare_player_models.jl:82` |
| `05_player_lineup_and_pxg_fusion` | Brier Score | 1X2, OU2.5, BTTS | Pooled selections | Betfair TWA (−20, 0] min | `experiments/scottish_lower/05_player_lineup_and_pxg_fusion/r52_compare_player_models.jl:92` |
| `05_player_lineup_and_pxg_fusion` | CRPS | Home & Away goal counts | Pooled across 710 OOS matches | Ground-truth match goals | `experiments/scottish_lower/05_player_lineup_and_pxg_fusion/r52_compare_player_models.jl:84` |
| `05_player_lineup_and_pxg_fusion` | Ranked Probability Score (RPS) | 1X2 | 1X2 market only | Betfair TWA (−20, 0] min | `experiments/scottish_lower/05_player_lineup_and_pxg_fusion/r52_compare_player_models.jl:94` |
| `05_player_lineup_and_pxg_fusion` | Expected Calibration Error (ECE) & MCE | 1X2, OU2.5, BTTS | 10 probability bins | Betfair TWA (−20, 0] min | `experiments/scottish_lower/05_player_lineup_and_pxg_fusion/r52_compare_player_models.jl:85` |
| `05_player_lineup_and_pxg_fusion` | Portfolio Kelly metrics (Return, Flat ROI, 1X2 ROI, MDD, Sharpe, Win Rate) | 1X2, OU2.5, BTTS | Multi-market portfolio (628 books, 82 skipped slates) | Betfair TWA (−20, 0] min | `experiments/scottish_lower/05_player_lineup_and_pxg_fusion/r53_portfolio_backtest.jl:48` |
| `06_joint_player_lineup_fusion` | Binary LogLoss, Brier, CRPS, RPS, ECE, MCE | 1X2, OU2.5, BTTS | 2,899 scored observations over 710 matches | Betfair TWA (−20, 0] min | `experiments/scottish_lower/06_joint_player_lineup_fusion/r62_compare_joint_player_models.jl:80` |
| `06_joint_player_lineup_fusion` (hierarchical κ) | Clustered paired LogLoss contrast | 1X2, OU2.5, BTTS | Scored selections clustered on `match_id` (710 clusters) | Model vs paired control | `experiments/scottish_lower/06_joint_player_lineup_fusion/r66_compare_hierarchical_kappa.jl:134` |
| `06_joint_player_lineup_fusion` (hierarchical κ) | Clustered GLM Edge (`r66_glm_edge`) | 1X2, OU2.5, BTTS | Clustered on `match_id` via CR1 sandwich covariance | Betfair TWA (−20, 0] min | `experiments/scottish_lower/06_joint_player_lineup_fusion/r66_compare_hierarchical_kappa.jl:160` |
| `06_joint_player_lineup_fusion` (hierarchical κ) | Match-clustered portfolio bootstrap | 1X2, OU2.5, BTTS | Resamples fixtures (not individual bets), $B=1,000$ | Betfair TWA (−20, 0] min | `experiments/scottish_lower/06_joint_player_lineup_fusion/r67_portfolio_hierarchical_kappa.jl:100` |
| `07_calibrated_portfolio_and_trust_vector` | Directional line breakdown (Kelly ROI, Flat ROI, standalone MDD, efficiency) | 13 directions across 1X2, OU 0.5–3.5, BTTS | In-sample (24/25) vs Out-of-sample (25/26) | Point-in-time book at **T−25 min** | `experiments/scottish_lower/07_calibrated_portfolio_and_trust_vector/l07_calibrated_trust_loader.jl:665` |
| `07_calibrated_portfolio_and_trust_vector` | Recompounded window metrics (Sharpe, Calmar, MDD, CAGR) | 11 tradeable directions | Slates in date range, re-based to bankroll 1.0 | Point-in-time book at **T−25 min** | `experiments/scottish_lower/07_calibrated_portfolio_and_trust_vector/l07_calibrated_trust_loader.jl:765` |
| `07_calibrated_portfolio_and_trust_vector` | Closing Line Value (CLV) & book drift | Tradeable markets | Executed bets vs closing line | **T−25 min** vs close | `src/Calibration/book.jl:210` (used in suite 07) |
| `08_goal_decomposition` | Binary scores by market family (LogLoss, Brier, ECE) | 1X2, OU2.5, BTTS, and ALL | Scored panel separated by market family | Betfair closing odds | `experiments/scottish_lower/08_goal_decomposition/l08_evaluation.jl:78` |
| `08_goal_decomposition` | Marginal home/away CRPS | Home and Away goals | 710 common fixtures | Match actual scoreline | `experiments/scottish_lower/08_goal_decomposition/l08_evaluation.jl:116` |
| `08_goal_decomposition` | 1X2 Ranked Probability Score | 1X2 | Evaluated via package kernel | Betfair closing odds | `experiments/scottish_lower/08_goal_decomposition/l08_evaluation.jl:108` |
| `10_momentum_multiscale_grw` | Supremacy slope (**MODEL-ON-MARKET**) | Log-rate supremacy ($\log \lambda_h - \log \lambda_a$) | 623 accepted market inversions | Betfair TWA (−20, 0] min | `experiments/scottish_lower/10_momentum_multiscale_grw/l12_evaluation.jl:29` |
| `10_momentum_multiscale_grw` | Heavy favourite win rate & tail calibration | 1X2 Home / Away $\ge 0.70$ close | 18 market favourites | Betfair TWA (−20, 0] min | `experiments/scottish_lower/10_momentum_multiscale_grw/l12_evaluation.jl:35` |
| `10_momentum_multiscale_grw` | Paired fixture-clustered bootstrap LogLoss | 1X2, OU2.5, BTTS, and ALL | 4,000 resamples of fixtures ($B=4,000$), preserving internal selections | Model vs model & Model vs market | `current_development/grw_player_hybrid/l02_evaluation.jl:195` |
| `10_momentum_multiscale_grw` | Equal-stake payoff ROI vs bankroll-scaled flat ROI | 1X2, OU2.5 | 622 common tradeable fixtures | Betfair TWA (−20, 0] min | `experiments/scottish_lower/10_momentum_multiscale_grw/README.md:270` |
| `11_decompression_pxg_covariate` | Supremacy slope (**MARKET-ON-MODEL**) | Log-rate supremacy | 623 accepted market inversions | Betfair TWA (−20, 0] min | `experiments/scottish_lower/11_decompression_pxg_covariate/l13_evaluation.jl:39` |
| `11_decompression_pxg_covariate` | Reverse slope (**MODEL-ON-MARKET**) & through-origin slope | Log-rate supremacy | 623 accepted market inversions | Betfair TWA (−20, 0] min | `experiments/scottish_lower/11_decompression_pxg_covariate/l13_evaluation.jl:40` |
| `12_decoupled_generative_xg` | Pre-registered block LogLoss contrast | 1X2, OU2.5, BTTS | Folds 21–40 (100% pxG commentary coverage) vs Folds 1–20 | Betfair TWA (−20, 0] min | `experiments/scottish_lower/12_decoupled_generative_xg/l14_evaluation.jl:30` |
| `scotland/02_cross_tier_cups_and_pyramid_eda` | Mincer-Zarnowitz regression (`mz!`) | Goal margin / supremacy | 403 cross-tier cup ties | SofaScore 1X2 closing odds | `experiments/scotland/02_cross_tier_cups_and_pyramid_eda/r05_market_efficiency_cup_pricing.jl:95` |
| `scotland/02_cross_tier_cups_and_pyramid_eda` | Cross-tier cup logistic calibration | Match win (higher-tier win) | Grouped by probability band | SofaScore closing odds | `experiments/scotland/02_cross_tier_cups_and_pyramid_eda/r05_market_efficiency_cup_pricing.jl:117` |
| `scotland/02_cross_tier_cups_and_pyramid_eda` | Information encompassing regression (`enc!`) | 1X2 outcome indicator | 403 cross-tier cup ties | Model vs SofaScore close | `experiments/scotland/02_cross_tier_cups_and_pyramid_eda/r05_market_efficiency_cup_pricing.jl:238` |
| `grw_pyramid_cups` | Dual supremacy slopes (Market-on-Model and Model-on-Market) | Log-rate supremacy | 623 accepted market inversions | Betfair TWA (−20, 0] min | `current_development/grw_pyramid_cups/r04_evaluate.jl:45` |
| `compare_scottish_experiments.jl` | Platt logistic recalibration slope ($\alpha, \beta$) | 1X2, OU2.5, BTTS | 2,899 pooled observations | Betfair TWA (−20, 0] min | `experiments/scottish_lower/compare_scottish_experiments.jl:290` |
| `compare_scottish_experiments.jl` | Dunn-Smyth RQR residuals (Mean, Var, Skew, Kurt, KS p, AD p) | Home and away goal counts | 1,420 count observations | Poisson / NegBin predictive CDF | `experiments/scottish_lower/compare_scottish_experiments.jl:220` |

---

## C. Workflow Helpers

This table catalogues helper functions across the suites, documenting their purpose, location, size, and relationship to existing `src/` facilities.

| Suite | Helper Purpose | Function Name (path:line) | Approx Lines | Duplicate of `src/` facility or other suite |
|---|---|---|---|---|
| `05_player_lineup_and_pxg_fusion` | Split construction | `r50_one_fold_inputs` (`r50_smoke_test_player_models.jl:46`) | 10 | Ad-hoc wrapper around `Data.create_id_boundaries` and `Data.get_next_matches` |
| `05_player_lineup_and_pxg_fusion` | Recipe registration | `l50_register!` (`l50_loader.jl:105`) | 35 | Wrapper around `Training.save_model`, `Training.save_splitter`, `Training.save_config` in `src/training/inference/db_storage.jl` |
| `05_player_lineup_and_pxg_fusion` | Betfair closing odds extraction | `r52_betfair_closing_odds` (`r52_compare_player_models.jl:21`) | 25 | Duplicated in `r53_portfolio_backtest.jl:19`, `r62_compare_joint_player_models.jl:20`, `cmp_betfair_closing_odds` |
| `06_joint_player_lineup_fusion` | Model recipe registration | `l60_register!` (`l60_loader.jl:120`) | 40 | Near-duplicate of `l50_register!` |
| `06_joint_player_lineup_fusion` | Cluster-robust t-test | `r66_cluster_ttest` (`r66_compare_hierarchical_kappa.jl:134`) | 20 | Suite-specific; no existing `src/` equivalent |
| `06_joint_player_lineup_fusion` | Clustered logistic sandwich covariance | `r66_logit_clustered` (`r66_compare_hierarchical_kappa.jl:160`) | 45 | Suite-specific; replaces naive standard errors from `src/evaluation/metrics.jl:550` |
| `06_joint_player_lineup_fusion` | GLM edge with clustering | `r66_glm_edge` (`r66_compare_hierarchical_kappa.jl:209`) | 40 | Re-implements `compute_metric(::GLMEdge)` from `src/evaluation/metrics.jl` with clustering |
| `06_joint_player_lineup_fusion` | Season roll extension | `r68_extend_joint_player_2627.jl:56` | 30 | Calls `Training.extend_fit` in `src/training/inference/extension.jl:185` |
| `07_calibrated_portfolio_and_trust_vector` | Trust tier grid generator | `l07_trust_grid` (`l07_calibrated_trust_loader.jl:805`) | 20 | Suite-specific parameter grid sweep generator |
| `07_calibrated_portfolio_and_trust_vector` | Trade ledger extraction & reconciliation | `l07_ledger` (`l07_calibrated_trust_loader.jl:505`) | 55 | Suite-specific ledger enrichment; joins `portfolio_bets` to market groups |
| `07_calibrated_portfolio_and_trust_vector` | Directional line breakdown | `l07_breakdown` (`l07_calibrated_trust_loader.jl:700`) | 35 | Partial duplicate of `eda_market_selection.jl` line diagnostics |
| `08_goal_decomposition` | Deterministic preflight checks | `l08_deterministic_checks` (`l08_model_checks.jl:298`) | 30 | Orchestrates gradient, extraction, and filtration checks without MCMC |
| `08_goal_decomposition` | Promotion manifest writer | `l08_write_manifest!` (`l08_workflow.jl:143`) | 35 | Emits TOML manifest locking code SHA, data SHA, and diagnostic hashes |
| `08_goal_decomposition` | Preflight assert prepare | `l08_assert_prepare!` (`l08_workflow.jl:298`) | 30 | Asserts all candidate models pass deterministic contracts before sampling |
| `08_goal_decomposition` | Binary scoring panel builder | `l08_scoring_panel` (`l08_evaluation.jl:51`) | 28 | Transforms `EvaluationRow` into binary DataFrame; duplicates logic in `metrics.jl` |
| `10_momentum_multiscale_grw` | Source fingerprint digest | `source_fingerprint` (`l11_workflow.jl:4`) | 10 | Computes joint SHA-256 over loader, workflow, runner, and `Project.toml` |
| `10_momentum_multiscale_grw` | Fold inputs & filtration report | `gph_fold_inputs` / `gph_filtration_report` (`current_development/grw_player_hybrid/l01_loader.jl:381, 397`) | 45 | Verifies historical boundaries, kickoff ordering, and zero train/test fixture overlap |
| `10_momentum_multiscale_grw` | Checkpointed NUTS sampling | `gph_sample` (`current_development/grw_player_hybrid/l01_loader.jl:505`) | 25 | Wraps `fit_model` with fold-by-fold checkpoint serialization to disk |
| `10_momentum_multiscale_grw` | Draw thinning for DB persistence | `gph_thin_for_persistence` (`current_development/grw_player_hybrid/l01_loader.jl:529`) | 20 | Subsamples chains by stride (e.g. 4) and re-extracts `CountLatents` from thinned draws |
| `10_momentum_multiscale_grw` | Save and verify round-trip | `gph_save_and_verify` (`current_development/grw_player_hybrid/l01_loader.jl:591`) | 25 | Calls `save_fit`, reloads via `load_fit`, and asserts exact draw & latent equality |
| `10_momentum_multiscale_grw` | Fixture-clustered paired bootstrap | `gph_paired_bootstrap` (`current_development/grw_player_hybrid/l02_evaluation.jl:195`) | 50 | Resamples clusters of fixtures to generate 95% CI on $\Delta \text{LogLoss}$ |
| `10_momentum_multiscale_grw` | Common portfolio panel builder | `gph_buildable_panel` (`current_development/grw_player_hybrid/l02_evaluation.jl:269`) | 35 | Finds common set of fixtures tradeable across all compared models |
| `11_decompression_pxg_covariate` | Supremacy decompression diagnostics | `decompression` (`l13_evaluation.jl:22`) | 65 | Computes market-on-model, model-on-market, through-origin slopes, and favourite win rate |
| `12_decoupled_generative_xg` | Block-stratified evaluation | `gph_scores` / `l14_evaluation.jl:30` | 40 | Computes proper scores stratified by pre-registered fixture block |
| `scotland/01_time_decay_cross_tier_and_priors` | Structured smoke gate framework | `ct_smoke_gate` / `CrossTierSmokeGate` (`l02_smoke_helpers.jl:27, 44`) | 45 | Structured 7-gate runner returning `:passed`, `:failed`, or `:abstained` |
| `scotland/01_time_decay_cross_tier_and_priors` | Config hash preflight check | `ct_smoke_config_hash` (`l02_smoke_helpers.jl:70`) | 15 | Queries `configs.config_hash` in `mcmc_experiments` before launching MCMC |
| `scotland/01_time_decay_cross_tier_and_priors` | Recursive structural equality | `ct_smoke_structural_equal` (`l02_smoke_helpers.jl:254`) | 20 | Deep field-by-field equality comparator for mutable Julia objects |
| `scotland/01_time_decay_cross_tier_and_priors` | Fit reload parity assertion | `ct_smoke_fit_parity` (`l02_smoke_helpers.jl:271`) | 25 | Verifies reloaded `Fit` matches original across config, folds, chains, diagnostics, latents |
| `grw_pyramid_cups` | Time alignment across tournaments | `pcx_align_time!` (`l01_loader.jl:95`) | 40 | Maps cup fixtures and upper-tier league matches onto the 56/57 biweek clock |
| `compare_scottish_experiments.jl` | Cross-paradigm benchmark driver | `cmp_compare_scottish_experiments` (`compare_scottish_experiments.jl:385`) | 95 | Re-evaluates 4 generations under identical BookSpec, PolicySpec, and price source |

---

## D. Recorded Results

All numbers below are extracted directly from recorded READMEs and result CSVs in the repository.

| Suite | Model Name | Data Scope | Dynamics | Observation | LogLoss | ECE | Slope (+convention) | Portfolio Headline | Run UUID(s) | Namespace | Source (path:line) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `05_player_lineup_and_pxg_fusion` | `m05_joint_production_wealth` | Scottish Lower (56/57) 24/26 (710 OOS matches) | TimeDecay (180 d) | Joint Gamma-Poisson | 0.6430 | 0.0143 | unrecorded | +131.97% return, 11.70% ROI, 1.487 Sharpe, −19.25% MDD | `842ca67c-02a0-4a7d-a247-016145742748` | `scottish_lower_player_grid_2426` | `experiments/scottish_lower/05_player_lineup_and_pxg_fusion/README.md:95, 110` |
| `05_player_lineup_and_pxg_fusion` | `m09_player_shots_rapm_outfield` | Scottish Lower (56/57) 24/26 | TimeDecay (180 d) | Joint Gamma-Poisson | 0.6445 | 0.0104 | unrecorded | +110.07% return, 9.91% ROI, 1.204 Sharpe, −20.05% MDD | `fd33bd76-5c70-4737-aac2-69d7903fd1b4` | `scottish_lower_player_grid_2426` | `experiments/scottish_lower/05_player_lineup_and_pxg_fusion/README.md:96, 114` |
| `05_player_lineup_and_pxg_fusion` | `m10_player_shots_rapm_bench` | Scottish Lower (56/57) 24/26 | TimeDecay (180 d) | Joint Gamma-Poisson | 0.6444 | 0.0098 | unrecorded | +112.37% return, 10.05% ROI, 1.218 Sharpe, −20.02% MDD | `c84b3cae-0828-4de1-a284-e5b04f52ce32` | `scottish_lower_player_grid_2426` | `experiments/scottish_lower/05_player_lineup_and_pxg_fusion/README.md:97, 113` |
| `05_player_lineup_and_pxg_fusion` | `m11_player_pxg_rapm_bench` | Scottish Lower (56/57) 24/26 | TimeDecay (180 d) | Joint Gamma-Poisson | 0.6449 | 0.0099 | unrecorded | +119.21% return, 10.53% ROI, 1.231 Sharpe, −20.51% MDD | `6166ebcb-c733-4d92-8233-8200a240aa26` | `scottish_lower_player_grid_2426` | `experiments/scottish_lower/05_player_lineup_and_pxg_fusion/README.md:98, 112` |
| `05_player_lineup_and_pxg_fusion` | `m12_hybrid_production_wealth_player_rapm` | Scottish Lower (56/57) 24/26 | TimeDecay (180 d) | Joint Gamma-Poisson | 0.6434 | 0.0094 | unrecorded | +136.92% return, 11.51% ROI, 1.413 Sharpe, −20.35% MDD | `c8963b56-f1cb-4560-89ad-0f86de0e9fd5` | `scottish_lower_player_grid_2426` | `experiments/scottish_lower/05_player_lineup_and_pxg_fusion/README.md:99, 109` |
| `06_joint_player_lineup_fusion` | `m05_joint_production_wealth` | Scottish Lower (56/57) 24/26 (710 OOS matches) | TimeDecay (180 d) | Joint Gamma-Poisson | 0.64299 | 0.01493 | 1.0804 (Platt) | +131.17% return (+138.94% P1), 1.481 Sharpe (1.629 P1), −19.05% MDD | `ed541a7c-01e2-447e-a771-783517728d47` | `scottish_lower_joint_player_2426` | `experiments/scottish_lower/06_joint_player_lineup_fusion/README.md:195, 237` |
| `06_joint_player_lineup_fusion` | `m09_joint_player_shots_outfield` | Scottish Lower (56/57) 24/26 | TimeDecay (180 d) | Joint Gamma-Poisson | 0.64448 | 0.00938 | unrecorded | +111.56% return, 1.216 Sharpe, −19.96% MDD | `6f09bb4a-2316-4986-a3ec-050398a59023` | `scottish_lower_joint_player_2426` | `experiments/scottish_lower/06_joint_player_lineup_fusion/README.md:199, 238` |
| `06_joint_player_lineup_fusion` | `m10_joint_player_shots_bench` | Scottish Lower (56/57) 24/26 | TimeDecay (180 d) | Joint Gamma-Poisson | 0.64440 | 0.00901 | unrecorded | +112.23% return, 1.217 Sharpe, −20.03% MDD | `baa67986-b2a5-461a-b661-fd7b5052c6d7` | `scottish_lower_joint_player_2426` | `experiments/scottish_lower/06_joint_player_lineup_fusion/README.md:198, 239` |
| `06_joint_player_lineup_fusion` | `m11_joint_player_pxg_bench` | Scottish Lower (56/57) 24/26 | TimeDecay (180 d) | Joint Gamma-Poisson | 0.64485 | 0.01040 | unrecorded | +120.55% return, 1.239 Sharpe, −20.73% MDD | `8dc231f1-7f09-4046-8c65-835d65c7a507` | `scottish_lower_joint_player_2426` | `experiments/scottish_lower/06_joint_player_lineup_fusion/README.md:200, 240` |
| `06_joint_player_lineup_fusion` | `m12_joint_hybrid_synergy` | Scottish Lower (56/57) 24/26 | TimeDecay (180 d) | Joint Gamma-Poisson | 0.64337 | 0.00996 | 1.0895 (Platt) | +136.61% return (+155.93% P1), 1.416 Sharpe (1.636 P1), −20.23% MDD | `132df5c2-c742-4e95-8693-3aeb2b2cbaef` | `scottish_lower_joint_player_2426` | `experiments/scottish_lower/06_joint_player_lineup_fusion/README.md:197, 241` |
| `06_joint_player_lineup_fusion` | `m13_joint_composite` | Scottish Lower (56/57) 24/26 | TimeDecay (180 d) | Joint Gamma-Poisson | 0.64324 | 0.00877 | 1.0823 (Platt) | +140.15% return, 1.453 Sharpe, −21.05% MDD | `5474e824-8c9d-4613-8e39-841426c3f80f` | `scottish_lower_joint_player_2426` | `experiments/scottish_lower/06_joint_player_lineup_fusion/README.md:196, 242` |
| `06_joint_player_lineup_fusion` (hierarchical κ) | `m05_hierarchical_kappa` | Scottish Lower (56/57) 24/26 | TimeDecay (180 d) | Joint (hierarchical κ) | 0.6430 | 0.0150 | unrecorded | Loses 4.5–6.0% terminal bankroll vs shared κ | `b3e19ad4-f755-4b89-addd-ff7592787deb` | `scottish_lower_joint_player_2426` | `experiments/scottish_lower/06_joint_player_lineup_fusion/README.md:331` |
| `06_joint_player_lineup_fusion` (hierarchical κ) | `m12_hierarchical_kappa` | Scottish Lower (56/57) 24/26 | TimeDecay (180 d) | Joint (hierarchical κ) | 0.6434 | 0.0100 | unrecorded | Loses 4.5–6.0% terminal bankroll vs shared κ | `a0847873-de69-4e25-824f-c03e4a4fd8c4` | `scottish_lower_joint_player_2426` | `experiments/scottish_lower/06_joint_player_lineup_fusion/README.md:332` |
| `10_momentum_multiscale_grw` | `m01` (TimeDecay) | Scottish Lower (56/57) 24/26 (710 fixtures) | TimeDecay (180 d) | Poisson | 0.646849 | 0.011616 | 0.2017 (**MODEL-ON-MARKET**) | +139.79% return, 1.209 Sharpe, −22.84% MDD | `33d85b4a-e929-4738-8125-706e0dc26de1` | `scottish_lower_momentum_grw` | `experiments/scottish_lower/10_momentum_multiscale_grw/README.md:14, 182` |
| `10_momentum_multiscale_grw` | `m02` (First-order GRW) | Scottish Lower (56/57) 24/26 | MultiScaleGRW | Poisson | 0.644518 | 0.017258 | 0.3880 (**MODEL-ON-MARKET**) | +197.10% return, 1.653 Sharpe, −20.73% MDD | `f8da493d-db2c-42d5-85e2-0f19d0107b1d` | `scottish_lower_momentum_grw` | `experiments/scottish_lower/10_momentum_multiscale_grw/README.md:15, 183` |
| `10_momentum_multiscale_grw` | `m03` (Momentum GRW) | Scottish Lower (56/57) 24/26 | MomentumMultiScaleGRW | Poisson | 0.644441 | 0.014241 | 0.4338 (**MODEL-ON-MARKET**) | +172.41% return, 1.568 Sharpe, −20.21% MDD | `3e06683b-de96-431f-843d-f98619d9fc13` | `scottish_lower_momentum_grw` | `experiments/scottish_lower/10_momentum_multiscale_grw/README.md:16, 184` |
| `11_decompression_pxg_covariate` | `m01_poisson_time_decay` | Scottish Lower (56/57) 24/26 (710 fixtures) | TimeDecay (180 d) | Poisson | 0.646809 | unrecorded | 2.5288 (**MARKET-ON-MODEL**) | +139.3% return, 1.204 Sharpe, −22.84% MDD | `90a8c7bc-b55c-4ef2-8e65-07cdaefa2b29` | `scottish_lower_decompression` | `experiments/scottish_lower/11_decompression_pxg_covariate/README.md:28, 141` |
| `11_decompression_pxg_covariate` | `m02_joint_gamma_poisson` | Scottish Lower (56/57) 24/26 | TimeDecay (180 d) | Joint Gamma-Poisson | 0.643730 | unrecorded | 1.7240 (**MARKET-ON-MODEL**) | +128.0% return, 1.270 Sharpe, −22.43% MDD | `6d9970c8-df28-4c2a-a67d-a5741153e708` | `scottish_lower_decompression` | `experiments/scottish_lower/11_decompression_pxg_covariate/README.md:29, 142` |
| `11_decompression_pxg_covariate` | `m03_negbin_pxg_covariate` | Scottish Lower (56/57) 24/26 | TimeDecay (180 d) | NegBin (form covariate) | 0.644755 | unrecorded | 1.3023 (**MARKET-ON-MODEL**) | +83.3% return, 1.057 Sharpe, −22.09% MDD | `f866b2bc-d87b-42d0-8c36-8c8d623d1178` | `scottish_lower_decompression` | `experiments/scottish_lower/11_decompression_pxg_covariate/README.md:30, 143` |
| `12_decoupled_generative_xg` | `m01_poisson_time_decay` | Scottish Lower (56/57) 24/26 (710 fixtures) | TimeDecay (180 d) | Poisson | 0.646788 | unrecorded | 2.5356 (**MARKET-ON-MODEL**) | +135.4% return, 1.184 Sharpe, −23.75% MDD | `de7fa956-87e8-418f-afb4-61ce01cb9f7d` | `scottish_lower_decoupled_xg` | `experiments/scottish_lower/12_decoupled_generative_xg/README.md:16, 23` |
| `12_decoupled_generative_xg` | `m02_joint_gamma_poisson` | Scottish Lower (56/57) 24/26 | TimeDecay (180 d) | Joint Gamma-Poisson | 0.643748 | unrecorded | 1.7214 (**MARKET-ON-MODEL**) | +123.4% return, 1.238 Sharpe, −22.20% MDD | `97c7a3d9-a05a-4029-90cb-e34279b8c791` | `scottish_lower_decoupled_xg` | `experiments/scottish_lower/12_decoupled_generative_xg/README.md:17, 23` |
| `12_decoupled_generative_xg` | `m03_funnel_shared_kappa` | Scottish Lower (56/57) 24/26 | TimeDecay (180 d) | Decoupled pxG funnel (shared κ) | 0.645216 | unrecorded | 1.9920 (**MARKET-ON-MODEL**) | +151.7% return, 1.189 Sharpe, −26.50% MDD | `27d5a9f5-303a-4661-ad3d-464ba776d380` | `scottish_lower_decoupled_xg` | `experiments/scottish_lower/12_decoupled_generative_xg/README.md:18, 23` |
| `12_decoupled_generative_xg` | `m04_funnel_hierarchical_kappa` | Scottish Lower (56/57) 24/26 | TimeDecay (180 d) | Decoupled pxG funnel (hierarchical κ) | 0.644839 | unrecorded | 2.0913 (**MARKET-ON-MODEL**) | +138.7% return, 1.182 Sharpe, −26.07% MDD | `b9d1627c-d7a6-4bc5-ab49-84acad0aafdd` | `scottish_lower_decoupled_xg` | `experiments/scottish_lower/12_decoupled_generative_xg/README.md:19, 23` |
| `scotland/01_time_decay_cross_tier_and_priors` | `m00` control (pilot) | Lower scope (720 train / 20 OOS) | TimeDecay (180 d) | Poisson | unrecorded | unrecorded | unrecorded | Pilot only (0 div, R̂ 1.00375, ESS 1513) | `e1e154ec-7693-4879-ad4d-e48693beb309` | `scotland_cross_tier_time_decay` | `experiments/scotland/01_time_decay_cross_tier_and_priors/results/LONGER_PILOT_FINDINGS.md:30, 48` |
| `scotland/01_time_decay_cross_tier_and_priors` | `m01` league offsets (pilot) | All-SPFL scope (1,476 train / 20 OOS) | TimeDecay (180 d) | Poisson | unrecorded | unrecorded | unrecorded | Pilot only (22 div, R̂ 1.00839, ESS 1335) | `52142683-3291-4c15-acf1-6aeaf38c6c94` | `scotland_cross_tier_time_decay` | `experiments/scotland/01_time_decay_cross_tier_and_priors/results/LONGER_PILOT_FINDINGS.md:31, 49` |
| `scotland/01_time_decay_cross_tier_and_priors` | `m02` hierarchical tiers (pilot) | All-SPFL scope (1,476 train / 20 OOS) | TimeDecay (180 d) | Poisson | unrecorded | unrecorded | unrecorded | Pilot only (1 div, R̂ 1.00566, ESS 1271) | `f97fc57d-7953-4f5f-86ad-ddc4c895eb4d` | `scotland_cross_tier_time_decay` | `experiments/scotland/01_time_decay_cross_tier_and_priors/results/LONGER_PILOT_FINDINGS.md:32, 50` |
| `scotland/01_time_decay_cross_tier_and_priors` | `m03` structural prior (pilot) | Lower scope + structural prior | TimeDecay (180 d) | Poisson | unrecorded | unrecorded | unrecorded | Pilot only (0 div, R̂ 1.00471, ESS 1378) | `365d3e08-d2fb-4c75-a680-026bd9387fea` | `scotland_cross_tier_time_decay` | `experiments/scotland/01_time_decay_cross_tier_and_priors/results/LONGER_PILOT_FINDINGS.md:33, 51` |
| `scotland/01_time_decay_cross_tier_and_priors` | `m04` market prior (pilot) | Lower scope + market prior | TimeDecay (180 d) | Poisson | unrecorded | unrecorded | unrecorded | Pilot only (0 div, R̂ 1.00424, ESS 1299) | `9c809ce5-cfc4-47ca-9ea3-ec7373f46095` | `scotland_cross_tier_time_decay` | `experiments/scotland/01_time_decay_cross_tier_and_priors/results/LONGER_PILOT_FINDINGS.md:34, 52` |
| `grw_pyramid_cups` | `g1_grw_all_spfl` | SPFL 54/55/56/57 (710 OOS) | MultiScaleGRW | Poisson | 0.64644 | 0.0122 | 1.10 (Mkt-on-Mod) / 0.39 (Mod-on-Mkt) | unrecorded | `f00ec78a-28ca-464e-91d4-dd1af384415c` | `scottish_pyramid_grw_cups` | `current_development/grw_pyramid_cups/README.md:14, 38` |
| `grw_pyramid_cups` | `g2_grw_all_spfl_cups` | SPFL + 363 cup bridges (710 OOS) | MultiScaleGRW | Poisson | 0.64617 | 0.0181 | 1.25 (Mkt-on-Mod) / 0.41 (Mod-on-Mkt) | unrecorded | `a6f62436-ec8a-461d-8bd5-dc1861a2daaa` | `scottish_pyramid_grw_cups` | `current_development/grw_pyramid_cups/README.md:15, 37` |
| `grw_pyramid_cups` | `g3_grw_joint_all_spfl_cups` | SPFL + 363 cup bridges (710 OOS) | MultiScaleGRW | Joint Gamma-Poisson | 0.64385 | 0.0095 | 1.10 (Mkt-on-Mod) / 0.60 (Mod-on-Mkt) | unrecorded | `9babf9e9-0a04-43af-855c-619a4b7dac8b` | `scottish_pyramid_grw_cups` | `current_development/grw_pyramid_cups/README.md:16, 34` |
| `compare_scottish_experiments.jl` | G1 `m00_baseline` | Scottish Lower (56/57) 24/26 | TimeDecay (180 d) | Poisson | 0.6545 | 0.0301 | 0.7423 (Platt) | +161.91% return, 8.32% ROI, 1.196 Sharpe, −32.02% MDD | unrecorded | `scottish_lower_poisson_2426` | `experiments/scottish_lower/UNIFIED_PARADIGM_REPORT.md:46, 85, 98` |
| `compare_scottish_experiments.jl` | G1 `m05_production_wealth` | Scottish Lower (56/57) 24/26 | TimeDecay (180 d) | Poisson | 0.6582 | 0.0529 | 0.6509 (Platt) | +148.26% return, 7.85% ROI, 1.079 Sharpe, −30.12% MDD | unrecorded | `scottish_lower_poisson_2426` | `experiments/scottish_lower/UNIFIED_PARADIGM_REPORT.md:47, 86, 99` |
| `compare_scottish_experiments.jl` | G2 `m00_negbin_baseline` | Scottish Lower (56/57) 24/26 (42 folds, 749 OOS) | TimeDecay (180 d) | NegBin | 0.6472 | 0.0109 | 1.0622 (Platt) | +76.59% return, 6.88% ROI, 0.875 Sharpe, −27.79% MDD | unrecorded | `scottish_lower_negbin_2426` | `experiments/scottish_lower/UNIFIED_PARADIGM_REPORT.md:48, 87, 105` |
| `compare_scottish_experiments.jl` | G2 `m05_negbin_production_wealth` | Scottish Lower (56/57) 24/26 | TimeDecay (180 d) | NegBin | 0.6457 | 0.0123 | 1.0882 (Platt) | +106.41% return, 8.63% ROI, 1.123 Sharpe, −25.79% MDD | unrecorded | `experiments/scottish_lower/UNIFIED_PARADIGM_REPORT.md:49, 88, 103` |
| `compare_scottish_experiments.jl` | G3 `m00_joint_baseline` | Scottish Lower (56/57) 24/26 | TimeDecay (180 d) | Joint Gamma-Poisson | 0.6438 | 0.0142 | 1.0627 (Platt) | +106.01% return, 8.66% ROI, 1.270 Sharpe, −19.96% MDD | unrecorded | `scottish_lower_joint_2426` | `experiments/scottish_lower/UNIFIED_PARADIGM_REPORT.md:50, 89, 104` |
| `compare_scottish_experiments.jl` | G3 `m05_joint_production_wealth` | Scottish Lower (56/57) 24/26 | TimeDecay (180 d) | Joint Gamma-Poisson | 0.6430 | 0.0149 | 1.0804 (Platt) | +124.07% return, 9.68% ROI, 1.429 Sharpe, −20.46% MDD | unrecorded | `scottish_lower_joint_2426` | `experiments/scottish_lower/UNIFIED_PARADIGM_REPORT.md:51, 90, 101` |
| `compare_scottish_experiments.jl` | G4 `m12_joint_hybrid_synergy` | Scottish Lower (56/57) 24/26 | TimeDecay (180 d) | Joint Gamma-Poisson | 0.6434 | 0.0100 | 1.0895 (Platt) | +123.94% return, 9.37% ROI, 1.333 Sharpe, −20.97% MDD | `132df5c2-c742-4e95-8693-3aeb2b2cbaef` | `scottish_lower_joint_player_2426` | `experiments/scottish_lower/UNIFIED_PARADIGM_REPORT.md:52, 91, 102` |
| `compare_scottish_experiments.jl` | G4 `m13_joint_composite` | Scottish Lower (56/57) 24/26 | TimeDecay (180 d) | Joint Gamma-Poisson | 0.6432 | 0.0088 | 1.0823 (Platt) | +127.86% return, 9.50% ROI, 1.375 Sharpe, −20.84% MDD | `5474e824-8c9d-4613-8e39-841426c3f80f` | `scottish_lower_joint_player_2426` | `experiments/scottish_lower/UNIFIED_PARADIGM_REPORT.md:53, 92, 100` |

---

## E. What `src/` Already Provides

This section lists the library facilities in `src/` that already implement gates, metrics, or helpers from sections A–C, noting where existing code is complete and where suite-local workarounds emerged.

- **Inference Lifecycle & Sampler Dispatch:**
  - `Training.fit_model` (`src/training/inference/engine.jl:250`): Implements `FitConfig` orchestration over sequential, threaded, and queued NUTS execution strategies.
  - `Training.QueuedExecution` (`src/training/inference/types.jl:18`): Flattens $K$ folds $\times$ $N$ chains into one global threadpool queue.
  - `Training.extend_fit` and `Training.preview_extension` (`src/training/inference/extension.jl:115, 185`): Incremental walk-forward sampler for appending new season folds without refitting historical folds.
- **Convergence Auditing:**
  - `Training.audit_convergence` and `Training.audit_fold` (`src/training/inference/convergence.jl:46, 81`): Evaluates split $\hat{R}$, bulk ESS, tail ESS, divergences, BFMI, and max treedepth rate against `ConvergenceThresholds`.
  - `Training.ConvergenceThresholds` (`src/training/inference/types.jl:20`): Production struct with standard defaults (`max_rhat=1.05`, `min_ess=100.0`, `max_divergence_rate=0.0`, `min_bfmi=0.30`, `max_treedepth_rate=0.05`).
- **Relational & Binary MCMC Storage (`mcmc_experiments`):**
  - `Training.save_fit` and `Training.load_fit` (`src/training/inference/db_storage.jl:290, 350`): Relational persistence of fold diagnostics, `match_latents`, and compressed binary `fit_artifacts`.
  - `Training.config_hash` (`src/training/inference/db_storage.jl:65`): SHA-256 fingerprinting of `FitConfig` models, samplers, and splitters for deduplication before execution.
  - `Training.save_config`, `Training.save_model`, `Training.save_splitter`, `Training.save_sampler` (`src/training/inference/db_storage.jl:120–180`): Canonical registration in `config_registry`.
- **Proper Scoring Rules & Predictive Evaluation:**
  - `Evaluation.LogLoss` & `calc_logloss` (`src/evaluation/metrics.jl:275`, `src/evaluation/metrics_methods/logloss.jl:19`): Selection-level binary cross-entropy.
  - `Evaluation.CRPS` & `compute_crps` (`src/evaluation/metrics.jl:380`, `src/evaluation/metrics_methods/crps.jl:18`): Home/away goal count Continuous Ranked Probability Score.
  - `Evaluation.ranked_probability_score` (`src/evaluation/metrics.jl:622`): 3-outcome ordered ranked probability score for 1X2.
  - `Evaluation.brier_score` (`src/evaluation/metrics.jl:586`): Mean squared error on binary selection outcomes.
  - `Evaluation.calibration_curve`, `Evaluation.expected_calibration_error`, `Evaluation.max_calibration_error` (`src/evaluation/metrics.jl:594–620`): Binned calibration diagnostics.
  - `Evaluation.compute_metric(::GLMEdge)` (`src/evaluation/metrics.jl:530`): Logistic regression of outcome on fair price and model edge. *(Note: uses unclustered standard errors).*
  - `Evaluation.compute_metric(::RQR)` (`src/evaluation/metrics.jl:430`): Dunn-Smyth randomized quantile residuals with seeded RNG.
  - `Evaluation.evaluate_predictions` (`src/evaluation/metrics.jl:665`): Unified wrapper computing proper scores and calibration error from a `Fit` and `DataStore`.
- **Market Rate Inversion & Generative Calibration:**
  - `Calibration.invert_market_rates` & `Calibration.inversion_frame` (`src/Calibration/rate_pool.jl:18, 35`): Non-linear inversion of tradeable book prices into $(\lambda_{\text{mkt}, h}, \lambda_{\text{mkt}, a})$ and conversion flags.
  - `Calibration.calibrate_fit` (`src/Calibration/pipeline.jl:30`): In-place pooling of model posterior log-rates with inverted market rates under `InverseGaussianLaw` or `StandardGaussianLaw`.
  - `Calibration.point_in_time_book` & `Calibration.closing_book` (`src/Calibration/book.jl:60, 140`): Filtration-clean market extraction from 1-minute exchange ladders.
  - `Calibration.clv_summary` & `Calibration.bet_clv` (`src/Calibration/book.jl:210, 245`): Closing Line Value calculation against bookmaker or exchange closing prices.
- **Portfolio Construction & Staking Engine:**
  - `Portfolio.run_portfolio_simulation` (`src/Portfolio/simulation.jl:342`): Zero-allocation multi-market execution over `BookSpec` and `PolicySpec`.
  - `Portfolio.simulate_portfolio` (`src/Portfolio/simulate.jl:45`): Compounding simulation with slate drawdown budgeting, fixed exposure caps, and optional bootstrap CI.
  - `Portfolio.save_portfolio_db` & `Portfolio.load_portfolio_db` (`src/Portfolio/db_storage.jl:25, 80`): Relational persistence in `portfolio_runs`, `portfolio_bets`, and `portfolio_artifacts`.
- **Temporal Splitting:**
  - `Data.CVConfig` & `Data.GroupedCVConfig` (`src/Data/splitting/types.jl:70, 100`): Walk-forward temporal splitter configurations emitting immutable `SplitBoundary` match ID pointers.
  - `Data.create_id_boundaries` & `Data.get_next_matches` (`src/Data/splitting/methods.jl:30, 160`): Split extraction without DataFrame slicing.

---

## F. Inconsistencies and Risks

1. **Opposite Supremacy Slope Conventions (Model-on-Market vs Market-on-Model):**
   - In `10_momentum_multiscale_grw/l12_evaluation.jl:29`, the decompression slope regresses model supremacy on market supremacy (`model_supremacy ~ a + b * market_supremacy`), reporting values between `0.20` and `0.43` (where 1.0 is ideal and lower means compressed).
   - In `11_decompression_pxg_covariate/l13_evaluation.jl:39` and `12_decoupled_generative_xg/README.md:20`, the diagnostic regresses market supremacy on model supremacy (`market_supremacy ~ a + b * model_supremacy`), reporting values between `1.30` and `2.53` (where 1.0 is ideal and higher means compressed).
   - In `compare_scottish_experiments.jl:290`, a third slope convention is used: the Platt logistic recalibration slope ($\beta \in [0.65, 1.09]$). Quoting "slope" without an explicit mathematical formula leads to direct misinterpretation across suites.
2. **Differing Gate Thresholds Across Suites:**
   - **Bulk / Tail ESS:** Suites 05, 06 (standard), and Scotland 01 require $\ge 100$; Suites 08, 10, 11, and 12 require $\ge 200$; Suite 06 hierarchical kappa (`r64`) requires $> 300$.
   - **Divergences:** Suites 05, 06, 10, 11, 12, and Scotland 01 enforce zero divergences as a hard fail. Suite 08 (`l08_workflow.jl:254`) treats `n_divergent > 0` as a non-blocking `@warn`. Furthermore, Scotland 01 smoke encountered a logic bug where `max_divergence_rate = eps()` failed on exact zero counts due to strict inequality (`0 < 0` is false).
   - **R-hat:** Production requires $\le 1.05$; `grw_pyramid_cups/r01_smoke.jl:11` permits $\le 1.10$ for smoke sampling.
3. **Score Grid Truncation vs Retained Mass Floors:**
   - Truncating goal distributions to a fixed $12 \times 12$ matrix (goals 0–11) cannot conserve unit probability mass when Poisson or Negative Binomial rates are elevated.
   - Scotland 01 (`l02_smoke_helpers.jl:248`) hard-failed four candidates because total grid mass fell below `0.999` (e.g. `0.9989`).
   - Suite 10 documented omitted tail mass reaching up to `5.31%` on extreme momentum draws, but checked internal partition consistency ($1\text{X}2 + \text{totals} = \text{mass}$) rather than bounding truncation loss. Hard-failing on truncation conflates grid sizing with model correctness.
4. **Tape Allocation & Latency: Correctness vs Performance Conflation:**
   - Scotland 01 (`l02_smoke_helpers.jl:172`) enforced a hard gate requiring compiled gradient replay latency $< 0.05\text{ ms}$ **AND zero byte allocations**.
   - Under Julia 1.12 with ReverseDiff broadcast structures, standard pre-game models allocate 35–135 KB per replay (documented in Task 007 / T002). This caused all five cross-tier candidates in Scotland 01 to fail Gate 1, blocking production even though AD gradients were exact.
   - Other suites (e.g. Suite 10, Suite 11) measured allocations and latency without hard-failing on non-zero heap usage.
5. **Statistical Independence vs Match-Clustered Inference:**
   - In `06_joint_player_lineup_fusion/r66_compare_hierarchical_kappa.jl:134, 160`, clustered inference on `match_id` is mandatory: 2,899 scored selections come from 710 matches. Treating selections as independent divides standard errors by $\approx 2$ and manufactures spuriously significant $p$-values.
   - In contrast, library `src/evaluation/metrics.jl:550` (`GLMEdge`) and older runners use naive unclustered logistic regression.
   - For bootstrap uncertainty, Suites 06, 10, 11, and 12 resample clusters of fixtures ($B=4,000$ or $10,000$), whereas uncoordinated runners resample individual bets.
6. **Price Reference Instant & Source Inconsistency:**
   - Early experiments (01 and 03) priced portfolios using bookmaker odds (`ds.odds`, overround intact).
   - Generations 3–4 (Suites 05 and 06) priced off Betfair exchange close TWA (−20, 0] minutes.
   - Suite 07 priced off the point-in-time exchange ladder at **T−25 minutes**.
   - As proven in `compare_scottish_experiments.jl`, comparing portfolio returns across different price sources ranks price sources rather than models.
7. **Binary Selection Cross-Entropy vs Categorical 1X2 LogLoss:**
   - Most suites score 1X2 as three binary selections (home win / draw / away win) evaluated via binary cross-entropy, yielding LogLoss values around `0.613–0.620`.
   - Other benchmarks interpret 1X2 LogLoss as 3-class categorical cross-entropy ($-\sum_{k=1}^3 y_k \log p_k$), which scales to `0.95–1.05`.
   - Mixing binary selection LogLoss with categorical match LogLoss produces completely non-comparable numbers.
8. **PostgreSQL Hex-Encoding Message Cap and Unaligned Chain Thinning:**
   - Serializing 40-fold MCMC chains (e.g. 128,000 draws) via hexadecimal strings in LibPQ hits PostgreSQL's 1 GB query/message limit (Suites 10, 11, and `grw_pyramid_cups`).
   - Suites handled this ad-hoc by thinning chains by stride 4 before database persistence.
   - If latents are not re-extracted from the thinned chain before saving, `load_fit` yields a container whose latents and chain draws disagree.
9. **Mis-specified Portfolio Benchmark Gates:**
   - Suite 07 defined Gate 3 as `oos_sharpe >= 1.65`, `oos_calmar >= 8.0`, and `oos_mdd_pct >= -18.0%` over a 50-slate out-of-sample window.
   - Zero of 168 deployable candidates passed: on a short 50-slate series, a single adverse weekend inflates the denominator of the Calmar ratio, making 8.0 mathematically unattainable despite superior cumulative growth.

---

## G. Proposed Harness API Sketch

Below is the minimal, unified harness API replacing the fragmented suite loaders and runners (under 60 lines).

```julia
# src/experiments/harness/types.jl & runner.jl
Base.@kwdef struct CandidateRecipe
    name::String
    model::BayesianFootball.Models.PreGame.AbstractModelBuilder
    splitter::Data.AbstractSplitter
    sampler::Training.AbstractSamplerConfig
    book_spec::Portfolio.BookSpec
    policy_spec::Portfolio.PolicySpec
    tags::Vector{String} = String[]
end

struct LibraryGates
    max_rhat::Float64               # default 1.05
    min_ess::Float64                # default 100.0 (smoke), 200.0 (prod)
    max_divergences::Int            # default 0
    ad_relerr_tol::Float64          # default 1e-8 vs fresh, 1e-6 vs ForwardDiff
    max_grad_latency_ms::Float64    # default 0.05 ms
    max_grid_tail_mass::Float64     # default 0.02 (warn/resize, do not abort)
end

Base.@kwdef struct ScorecardSpec
    price_source::Symbol = :betfair_twa_minus20_to_0    # or :pit_minus25
    cluster_by::Symbol = :match_id                      # enforce cluster-robust stats
    bootstrap_B::Int = 4000                             # fixture-clustered resamples
    decompression_convention::Symbol = :market_on_model # explicitly declared
    markets::Vector{String} = ["1X2", "OU2.5", "BTTS"]
    strata::Vector{Symbol} = [:all, :seasons, :pre_registered_blocks]
end

# Unified Execution Entry Points
function run_smoke_gate(candidate::CandidateRecipe, ds::Data.DataStore; 
                        gates::LibraryGates = LibraryGates(), db::Union{Nothing,PostgresStorage} = nothing)::NamedTuple
    # Runs AD parity, 1-fold sampling, 6-part audit, CountLatents extraction,
    # partition consistency, and DB round-trip; returns structured GateReport
end

function run_production_grid(candidate::CandidateRecipe, ds::Data.DataStore, db::PostgresStorage;
                             gates::LibraryGates = LibraryGates(),
                             persist_stride::Int = 1)::UUID
    # Preflights configs.config_hash, executes QueuedExecution grid, validates convergence,
    # thins if stride > 1 with aligned CountLatents, persists fit, and returns run UUID
end

function evaluate_scorecard(run_uuids::Vector{UUID}, ds::Data.DataStore, db::PostgresStorage,
                            spec::ScorecardSpec)::DataFrame
    # Evaluates binary LogLoss, categorical 1X2 LogLoss, Brier, RPS, CRPS, ECE/MCE,
    # clustered GLMEdge, Mincer-Zarnowitz, decompression slopes, and paired bootstrap
end

function simulate_portfolio_suite(run_uuids::Vector{UUID}, ds::Data.DataStore, db::PostgresStorage;
                                  spec::ScorecardSpec)::DataFrame
    # Simulates matched tradeable panel under common BookSpec/PolicySpec with bet ledger parity
end
```

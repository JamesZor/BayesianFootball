# GRW SlateDrawdown lambda risk sweep

Generated `2026-09-18 18:08` at Git `f5809c69` on `mcmc-beast` with 16 Julia threads. No MCMC sampling was performed.

## Contract

- Models: `m05_joint_grw_smile_spine_w040` and `m05_joint_grw_baseline`, loaded by immutable UUID.
- Policy: operational Option B trust, `SlateDrawdown(lambda)`, `FixedCap(0.25)`, daily slates.
- T012: `:excise_pruned`; the priced book contains 1X2, O/U 1.5 (Over active), and O/U 2.5 (Under active). Excision is market-level.
- Smile route: anti-diagonal reweighted score grid. T−25 L2 uses the validated φ-dropped CountLatents control.
- Nominal floor: `D = exp(log(0.01)/lambda)`; overshoot is realised max drawdown divided by `100(1-D)`.

## Data panels

| environment | variant | n_walk_forward | n_quoted | n_buildable | n_excluded |
|---|---|---|---|---|---|
| close | raw | 710 | 635 | 628 | 82 |
| t25 | raw_and_l2_calibrated | 710 | 611 | 594 | 116 |

Environment coverage is reported rather than silently padding absent T−25 quotes. Cross-environment returns are therefore descriptive, not paired fixture-for-fixture unless panel counts match.

## Immutable run inventory

| model | persisted_run_name | experiment | run_id | n_folds | converged_folds | latent_kind |
|---|---|---|---|---|---|---|
| m05_joint_grw_smile_spine_w040 | m05_joint_grw_smile_spine_w040 | scottish_lower_grw_smile_spine | 582035c0-e145-44f7-9f40-89e25388e79a | 43 | 43 | SmileLatents |
| m05_joint_grw_baseline | m05_wealth_grw | scottish_lower_grw_player_hybrid | b0961bc4-c40c-4dbe-9c05-57df7ae0839e | 43 | 43 | CountLatents |

## Verification gates

| gate | pass | detail |
|---|---|---|
| immutable runs completed and converged | true | 86/86 folds |
| latent panel | true | 710/710 fixtures |
| close common buildable panel | true | 628/628 fixtures |
| T012 market-level excision | true | 3 active markets; all-zero markets absent |
| T−25 book instant | true | assert_book_as_of(-25.0) |
| T−25 inversion and identity fallback accounted | true | 573/594 shifted; 21 documented identity fallbacks |
| zero-allocation pricing | true | max 0 bytes |
| lambda=8 published excised reproduction | true | spine close 576.133101990205%, 1188 bets, MDD -42.813337651556% |
| sequential/threaded bit identity | true | 60 canonical result payloads and summary rows |
| mean exposure non-increasing in lambda | true | worst adjacent increase 0.000e+00 |
| grid completeness | true | 60/60 cells |

### Zero-allocation pricing kernels

| case | latent_type | kernel | allocated_bytes | pass |
|---|---|---|---|---|
| baseline/raw | CountLatents | Portfolio.price_fixture! | 0 | true |
| spine/raw-reweighted | SmileLatents | Portfolio.price_fixture! | 0 | true |
| baseline/t25-l2 | CountLatents | Portfolio.price_fixture! | 0 | true |
| spine/t25-l2-phi-dropped | CountLatents | Portfolio.price_fixture! | 0 | true |

### T−25 calibration coverage

| model | n_panel | n_inverted | coverage_pct | book_as_of_minutes | law | phi_pricing | identity_fallback | pass |
|---|---|---|---|---|---|---|---|---|
| m05_joint_grw_smile_spine_w040 | 594 | 573 | 96.46464646464646 | -25.0 | inv_w0.25_s0.35 | dropped | 21 | true |
| m05_joint_grw_baseline | 594 | 573 | 96.46464646464646 | -25.0 | inv_w0.25_s0.35 | dropped | 21 | true |

## First lambda that meets realised max drawdown ≤ 20%

| model | environment | variant | lambda | realised_drawdown_pct | sharpe_ann | terminal_return_pct | mean_slate_exposure |
|---|---|---|---|---|---|---|---|
| m05_joint_grw_baseline | close | raw | 28.0 | 19.6504 | 1.6329 | 92.2617 | 0.0588 |
| m05_joint_grw_smile_spine_w040 | close | raw | 28.0 | 17.0783 | 1.7436 | 109.9190 | 0.0479 |
| m05_joint_grw_baseline | t25 | l2_calibrated | 10.0 | 17.3542 | 2.0002 | 194.8506 | 0.0658 |
| m05_joint_grw_smile_spine_w040 | t25 | l2_calibrated | 8.0 | 16.8599 | 1.7805 | 211.2146 | 0.0633 |
| m05_joint_grw_baseline | t25 | raw | 28.0 | 17.2223 | 1.6574 | 88.9668 | 0.0540 |
| m05_joint_grw_smile_spine_w040 | t25 | raw | 28.0 | 17.8141 | 1.4041 | 84.8877 | 0.0444 |

For every raw book/model cell the first grid point below 20% is `lambda = 28`; `lambda = 23` remains just above the ceiling. The L2-calibrated T−25 paths need materially less risk tightening because calibration already cuts exposure.

## Strict maximum-Sharpe choice under the 20% ceiling

| model | environment | variant | target_drawdown_pct | lambda | realised_drawdown_pct | sharpe_ann | terminal_return_pct | mean_slate_exposure | status |
|---|---|---|---|---|---|---|---|---|---|
| m05_joint_grw_baseline | close | raw | 20.0 | 45.0 | 12.7298 | 1.675 | 52.8227 | 0.0371 | target met; maximum Sharpe among feasible cells |
| m05_joint_grw_smile_spine_w040 | close | raw | 20.0 | 45.0 | 11.0043 | 1.7807 | 61.5426 | 0.0302 | target met; maximum Sharpe among feasible cells |
| m05_joint_grw_baseline | t25 | l2_calibrated | 20.0 | 45.0 | 4.3079 | 2.0909 | 31.3698 | 0.0157 | target met; maximum Sharpe among feasible cells |
| m05_joint_grw_smile_spine_w040 | t25 | l2_calibrated | 20.0 | 45.0 | 3.4511 | 1.8778 | 27.112 | 0.0123 | target met; maximum Sharpe among feasible cells |
| m05_joint_grw_baseline | t25 | raw | 20.0 | 45.0 | 11.1053 | 1.6935 | 50.7996 | 0.034 | target met; maximum Sharpe among feasible cells |
| m05_joint_grw_smile_spine_w040 | t25 | raw | 20.0 | 45.0 | 11.4876 | 1.4459 | 49.1885 | 0.028 | target met; maximum Sharpe among feasible cells |

Sharpe rises as exposure falls across the feasible raw cells, so the literal maximum-Sharpe rule selects the edge of the tested grid, `lambda = 45`. This is not a growth-maximising choice: `lambda = 28` is the less conservative operational knee that first enforces the 20% ceiling while retaining more terminal return. Both are in-sample risk-policy choices on the held-out prediction panel, not fresh predictive-model promotion tests.

## Drawdown–Sharpe Pareto frontier

| model | environment | variant | lambda | terminal_return_pct | sharpe_ann | realised_drawdown_pct | mean_slate_exposure | overshoot_ratio | pareto_all_metrics |
|---|---|---|---|---|---|---|---|---|---|
| m05_joint_grw_baseline | close | raw | 45.0 | 52.8227 | 1.675 | 12.7298 | 0.0371 | 1.3086 | true |
| m05_joint_grw_smile_spine_w040 | close | raw | 45.0 | 61.5426 | 1.7807 | 11.0043 | 0.0302 | 1.1313 | true |
| m05_joint_grw_baseline | t25 | l2_calibrated | 45.0 | 31.3698 | 2.0909 | 4.3079 | 0.0157 | 0.4429 | true |
| m05_joint_grw_smile_spine_w040 | t25 | l2_calibrated | 45.0 | 27.112 | 1.8778 | 3.4511 | 0.0123 | 0.3548 | true |
| m05_joint_grw_baseline | t25 | raw | 45.0 | 50.7996 | 1.6935 | 11.1053 | 0.034 | 1.1416 | true |
| m05_joint_grw_smile_spine_w040 | t25 | raw | 45.0 | 49.1885 | 1.4459 | 11.4876 | 0.028 | 1.1809 | true |

## Overshoot calibration

| lambda | mean_overshoot_ratio | min_overshoot_ratio | max_overshoot_ratio |
|---|---|---|---|
| 8.0 | 0.7896 | 0.3852 | 1.0097 |
| 10.0 | 0.8517 | 0.3777 | 1.1014 |
| 12.0 | 0.8916 | 0.3727 | 1.1547 |
| 15.0 | 0.9079 | 0.3677 | 1.2218 |
| 18.0 | 0.9181 | 0.3645 | 1.2778 |
| 20.0 | 0.9195 | 0.3628 | 1.2827 |
| 23.0 | 0.9211 | 0.3609 | 1.2885 |
| 28.0 | 0.923 | 0.3587 | 1.2957 |
| 35.0 | 0.9249 | 0.3566 | 1.3025 |
| 45.0 | 0.9267 | 0.3548 | 1.3086 |

The 1.15 constant is a historical empirical rule, not a gate. It is a reasonable approximation for the raw smile path and raw T−25 books at tighter lambdas, but it does not transfer uniformly: the close baseline reaches about 1.31 while the L2-calibrated paths remain around 0.35–0.48. Pooling these regimes into one universal correction would therefore be wrong.

## Interpretation guardrails

- The sweep reuses one set of prebuilt posterior books per model/environment; lambda changes staking only.
- Terminal return is path-dependent and should not be maximised without the drawdown and Sharpe columns.
- The close and T−25 books are different price instants. The L2 recipe is applied only to T−25, its declared instant.
- `bit-identical` means every summary value and every canonical `PortfolioResult` field (including ledger columns and daily states) matched exactly between sequential and threaded execution; SHA-256 digests of those canonical payloads also matched.

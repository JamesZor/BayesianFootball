# Phase A summary
- Who starts is associated with closing supremacy after team-season strength: slopes are −0.02593 / −0.03154 per extra home missing regular in Scottish League One / Two, about −0.84 / −1.01 percentage points of home-win probability at typical lines.
- England is similar (−0.02327 to −0.02866); these results do not establish a larger Scottish effect.
- Previous-close form controls attenuate the Scottish slopes to −0.01763 / −0.02473, rather than eliminating them.
- The placebo is **not generally null**: next-match absences predict current supremacy in 6 of 8 leagues under season-block intervals, including League Two. This is not clean player-effect identification.
- Residual supremacy SD is 0.205 / 0.207 in League One / Two; residual total SD is 0.120 / 0.123. Primary total-absence tests are inconclusive.
- Scottish joined samples are 809 / 647 matches; identifiable starter dimensions are 1,456 / 1,206, but the latter saturates the remaining observation dimension. Rank is not reliability.
- **Phase B: conditional exploratory build only, not a validated player signal.** Address persistence and team form first, use strong regularization and mover-aware pooling, and require held-out stability before interpreting ratings.
- Sources for all summary results: `results/A/absent_slopes.csv`, `headroom.csv`, and `design.csv`. Phase B was not implemented.

## Scope and data contract

The response is the existing accepted double-Poisson inversion of Betfair's time-weighted close over (−20, 0] minutes. There is no reinversion, outcome prediction, player ridge, betting test, or database write. Announced starting XIs precede that close; this does **not** establish safety for a pre-announcement forecast.

Pinned cached datastores were SHA256-verified on the beast before loading. England requires the original serialized `QualityStyleEDA.MarketModelEnglish` module identity. `r01_market_apm.jl` restores it without loading the original EDA implementation. Hashes are recorded in `l01_market_apm.jl` and the original `current_development/market_model/results/QS/DATA_SNAPSHOTS.md`.

Coverage is restricted to the seasons present in the existing league rate files. Scottish Championship's excluded season is not silently reintroduced. Starting XI means `is_substitute == false`; minutes are never used. Require exactly eleven distinct player IDs on each side, no shared home/away player ID, a matching fixture, and finite positive rates. Missing or malformed lineups are not repaired.

### Accepted-close join coverage

Source: `results/A/coverage_gate.csv`; fixture counts and season-specific coverage are in `coverage.csv`.

| League | Accepted closes | Joined | Loss |
|---|---:|---:|---:|
| Scottish League One (56) | 835 | 809 | 3.11% |
| Scottish League Two (57) | 693 | 647 | 6.64% |
| Scottish Premiership | 956 | 955 | 0.10% |
| Scottish Championship | 477 | 477 | 0.00% |
| English Premier League | 1,507 | 1,507 | 0.00% |
| English Championship | 2,194 | 2,194 | 0.00% |
| English League One | 2,183 | 2,181 | 0.09% |
| English League Two | 2,173 | 2,173 | 0.00% |

No league or league-season crosses the requested loss gate. `join_failures.csv` lists each failed accepted match, its reason, and both starter counts. `failure_summary.csv` additionally separates fixtures without an inversion row from explicitly rejected inversions and accepted-close join failures; a missing inversion row is not proof of absent market coverage. The largest season loss is Scottish League Two 25/26, 16 of 131 accepted closes (12.21%; `coverage.csv`).

## Design: where player deviations could be identified

Fit stacked home and away log rates with own-team-season attack, opposing-team-season defence, and one home-rate indicator per league. The starter design likewise has separate own-XI attack and opposing-XI defence columns per player. The requested absorbed dimension is

`rank([team-season design, starter design]) − rank(team-season design)`.

This definition matters: it is **not** the rank of a single signed match-supremacy lineup matrix. Columns are unit indicators. Sparse QR uses an explicit absolute tolerance; `rank_audit.csv` confirms stability across tighter/looser tolerances and equality with dense SVD on a small match subset. `design.csv` retains column counts and both constituent ranks.

Sources: `design.csv`. Player counts, starts, and movers below refer to the **joined closes**. Top-eleven usage and previous-XI overlap use all available valid fixture-side XIs in selected seasons; team-season detail and missing-side counts are in `team_season_design.csv`. Usage is an unweighted mean across team-seasons; overlap is an unweighted mean across consecutive valid fixture-side pairs.

| League | Players ≥1 / ≥5 / ≥20 starts | Median starts | Movers | Top-11 share | Previous XI overlap | Team FE rank | Absorbed starter rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| Scottish League One | 760 / 596 / 340 | 16 | 167 | 0.728 | 0.820 | 96 | 1,456 |
| Scottish League Two | 680 / 524 / 266 | 13 | 136 | 0.726 | 0.824 | 88 | 1,206 |
| Scottish Premiership | 871 / 705 / 379 | 16 | 118 | 0.713 | 0.783 | 116 | 1,686 |
| Scottish Championship | 533 / 414 / 200 | 13 | 93 | 0.715 | 0.824 | 62 | 890 |
| English Premier League | 917 / 778 / 518 | 25 | 168 | 0.732 | 0.800 | 157 | 1,778 |
| English Championship | 1,391 / 1,165 / 800 | 25 | 357 | 0.696 | 0.792 | 189 | 2,744 |
| English League One | 1,536 / 1,304 / 838 | 22 | 438 | 0.685 | 0.795 | 189 | 2,994 |
| English League Two | 1,554 / 1,306 / 850 | 22 | 401 | 0.682 | 0.812 | 189 | 3,046 |

A mover means a player seen starting for multiple distinct club slugs within that league's joined sample. It is not necessarily a within-season transfer, and does not by itself demonstrate player identification separately from time-varying club quality.

The Scottish designs are close to saturated: after adding both team and player indicator spans, remaining stacked-rate degrees of freedom are 66 / 0 for League One / Two, versus 1,079–1,455 in England. These are calculated directly as `2*joined_n − combined_rank` from `design.csv`. Consequently, high rank permits in-sample interpolation, not trustworthy unrestricted player ratings. In particular, separate individual attack and defence coefficients need not be uniquely identified even when their fitted span is large.

## Headroom after team strength

Sources: `headroom.csv`. Sample residual SD is descriptive, not an unbiased variance estimate or a forecast error. The loader also stores a rate-level degrees-of-freedom-adjusted RMSE and checks fixed-effect residual orthogonality. The two derived responses are log supremacy and log total intensity **product**; `log λh + log λa` is not `log(λh + λa)`.

| League | Residual supremacy SD | Residual log-total SD |
|---|---:|---:|
| Scottish League One | 0.205 | 0.120 |
| Scottish League Two | 0.207 | 0.123 |
| Scottish Premiership | 0.149 | 0.083 |
| Scottish Championship | 0.214 | 0.120 |
| English Premier League | 0.180 | 0.121 |
| English Championship | 0.163 | 0.104 |
| English League One | 0.171 | 0.103 |
| English League Two | 0.166 | 0.105 |

This residual variation is an upper bound on what a lineup term could explain **in sample**. It also contains evolving team form, market noise and inversion error. More Scottish residual variation is not equivalent to more usable player information.

## Cheap causal-history absence proxy

For each team-season, regulars are the players with the most starts **strictly before** the current fixture. Ties break deterministically by player ID. Use all recorded fixtures to order history, not just accepted closes; no minutes, future starts, or end-of-season roster ranking are used. Drop a match if either team's fixture ordinal is within its first five fixtures. `absent = 11 − regulars present in the XI`.

Malformed or unknown fixture-side XIs contribute no start counts, but still advance fixture ordinal. Thus the primary proxy may be imperfect where prior lineup coverage is poor. `complete_history` excludes matches with any earlier malformed/unknown starting side for either team; cross-side shared IDs are also treated as invalid histories.

Regress the already-fitted residual supremacy on home-minus-away absences, and residual log total on summed absences, with an intercept. As explicitly requested, only the response is team-adjusted: these are **not** full Frisch–Waugh–Lovell partial player coefficients, which would also absorb the predictor. Fixed effects and histories are descriptive within the selected sample, not an out-of-sample estimation protocol.

The CSV reports match-clustered HC1 intervals and season-block CR1 intervals. Tables below prioritize the latter, with small-block Student-t critical values. There are few season blocks, and intervals do not account for every source of generated-response uncertainty. No multiple-testing correction is applied; borderline exclusion of zero should not be treated as decisive evidence.

### Supremacy: current absences

Source: `absent_slopes.csv`, `response=supremacy`, `test=current`, `se_type=season_CR1`.

| League | Eligible n | Δ log supremacy per missing regular | Season-block 95% interval | Δ home-win probability (pp) |
|---|---:|---:|---:|---:|
| Scottish League One | 688 | −0.02593 | [−0.05241, +0.00054] | −0.838 |
| Scottish League Two | 559 | −0.03154 | [−0.06226, −0.00082] | −1.010 |
| Scottish Premiership | 805 | −0.03137 | [−0.04454, −0.01820] | −0.966 |
| Scottish Championship | 429 | −0.03214 | [−0.06840, +0.00413] | −1.008 |
| English Premier League | 1,307 | −0.02812 | [−0.04048, −0.01575] | −0.928 |
| English Championship | 1,953 | −0.02866 | [−0.03949, −0.01782] | −0.895 |
| English League One | 1,939 | −0.02704 | [−0.03047, −0.02360] | −0.854 |
| English League Two | 1,930 | −0.02327 | [−0.03202, −0.01451] | −0.730 |

“Missing regular” here means an increment in home-minus-away absence count: an additional home absence lowers the association, an additional away absence reverses the sign. Probability changes are illustrative double-Poisson changes at the league's median accepted joined home/away rates, holding log total constant: multiply rates by `exp(β/2)` and `exp(−β/2)`. The baseline probability, both typical rates and probability change are stored in the same CSV. They are not observed betting returns or estimated causal replacement effects.

### Supremacy: placebo and form sensitivity

Source: `absent_slopes.csv`, season-block rows. Each placebo uses the **immediately next fixture** for each current team, which can have different dates and different home/away status. If either next XI is unavailable, that current match is excluded; do not jump ahead. Form uses each team's immediately previous accepted complete-close residual, transformed to its team perspective, then home minus away. It excludes a current match if either immediate previous fixture lacks such a close.

| League | Next-absence n | Placebo slope [95% interval] | Form-control n | Current slope with previous form [95% interval] |
|---|---:|---:|---:|---:|
| Scottish League One | 661 | −0.01437 [−0.03927, +0.01054] | 582 | −0.01763 [−0.02897, −0.00629] |
| Scottish League Two | 523 | −0.01635 [−0.03255, −0.00016] | 350 | −0.02473 [−0.04654, −0.00292] |
| Scottish Premiership | 774 | −0.02083 [−0.03059, −0.01107] | 745 | −0.02488 [−0.03338, −0.01638] |
| Scottish Championship | 414 | −0.02135 [−0.05095, +0.00825] | 371 | −0.02361 [−0.05037, +0.00316] |
| English Premier League | 1,267 | −0.01438 [−0.02621, −0.00254] | 1,287 | −0.02557 [−0.03257, −0.01858] |
| English Championship | 1,905 | −0.01910 [−0.02783, −0.01037] | 1,936 | −0.02259 [−0.02980, −0.01538] |
| English League One | 1,901 | −0.01783 [−0.02440, −0.01127] | 1,909 | −0.02159 [−0.02454, −0.01863] |
| English League Two | 1,893 | −0.01415 [−0.02137, −0.00693] | 1,884 | −0.01820 [−0.02587, −0.01053] |

The placebo fails as a clean negative control. Persistent absences, injuries, roster changes, selection responding to form, and persistent team strength can all connect next XIs to today's close. The next-XI result does not prove that future information leaked into the current-XI construction; synthetic tests explicitly change a future XI without changing earlier absence counts. It does show that current-absence slopes cannot simply be labeled player impacts.

Sample changes matter. On exactly the placebo sample, current slopes are −0.02253 / −0.02771 for League One / Two (`current_matched_placebo`), still more negative than their respective placebo estimates. No paired coefficient-difference significance is claimed. Complete-history sensitivities are −0.01876 / −0.02997 with 669 / 359 matches (`complete_history`). Neither sensitivity resolves the persistence confounding. Form-control probability translations are −0.570 / −0.792 pp in 56/57 (`form_previous`), but the form sample differs from the primary sample.

### Log total: no consistent primary effect

Source: `absent_slopes.csv`, `response=total`, `test=current`, season-block rows.

| League | Slope on summed absences | Season-block 95% interval |
|---|---:|---:|
| Scottish League One | +0.00568 | [−0.00159, +0.01296] |
| Scottish League Two | −0.00365 | [−0.01094, +0.00365] |
| Scottish Premiership | −0.00289 | [−0.00956, +0.00377] |
| Scottish Championship | −0.00176 | [−0.01317, +0.00965] |
| English Premier League | +0.00359 | [−0.00548, +0.01265] |
| English Championship | −0.00458 | [−0.01546, +0.00631] |
| English League One | −0.00383 | [−0.01659, +0.00892] |
| English League Two | −0.00443 | [−0.01002, +0.00116] |

All primary, placebo, and previous-form season-block total intervals include zero. The complete-history League One sensitivity is an exception (+0.0087, [0.0036, 0.0137]); it is not a cross-league total effect. The scalar absence count ignores which positions and qualities changed, so a weak total result does not rule out individual attack/defence differences.

## Figures and reproducibility

`results/A/figures/` contains one SVG per league and `pooled.svg`. All bins, means, sample sizes and descriptive mean standard errors are in `figure_bins.csv`. Bins are exact integer absence differences on the primary eligible sample. The pooled figure weights matches equally and mixes leagues with different residual distributions; it is not a pooled causal estimate. Singleton bins receive zero descriptive standard error and should not be interpreted as precisely estimated effects.

- [League One](results/A/figures/sco_league_one.svg) · [League Two](results/A/figures/sco_league_two.svg)
- [Scottish Premiership](results/A/figures/sco_premiership.svg) · [Scottish Championship](results/A/figures/sco_championship.svg)
- [English Premier](results/A/figures/eng_premier.svg) · [English Championship](results/A/figures/eng_championship.svg)
- [English League One](results/A/figures/eng_league_one.svg) · [English League Two](results/A/figures/eng_league_two.svg)
- [Pooled](results/A/figures/pooled.svg)

`joined_matches.csv` makes the response residuals, current/next absence proxies, fixture ordinals, previous residual form covariates and sample flags auditable. Raw starter IDs remain reproducible from the pinned datastores; they are not needed to query the committed CSV summaries.

Run `r01_market_apm.jl`, then `r01_checks.jl` and `r01_rank_checks.jl` in an isolated beast checkout with the verified pinned caches and matching Julia environment. No database credentials are needed. Set BLAS to one thread. Do not reload a differently defined segment struct into a shared REPL. Verification logs are in `results/A/logs/`.

## Phase B decision, not Phase B work

There is enough descriptive signal to justify a **small, strongly regularized research follow-up**, but not enough identification to treat an unrestricted market-APM as warranted or validated. England offers more starts per player and more residual degrees of freedom; Scottish League Two alone is especially fragile. The focus remains 56/57.

Before rating players, compare current-versus-next absences on common samples with explicit persistence/time-varying team controls. Consider shared player identity across Scottish tiers, while keeping league baselines and team effects separate; do not assume movers solve club confounding. Any eventual player model should expose its estimable span, regularization dependence, low-start uncertainty, and stability across temporal holdouts. Assess forecast value only under a later, explicit information-timing contract. None of that work was started here.

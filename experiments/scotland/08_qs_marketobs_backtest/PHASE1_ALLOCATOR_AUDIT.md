# Phase 1 — allocator check

Audited source: `e4b95a19` (Portfolio files unchanged by Phase 0).
**Answer: NO.** Existing KellyLogUtility solves per-match allocations using a
posterior-mean joint score grid. It does not maximise expected log terminal wealth
jointly across the daily slate using shared posterior draws. Phase 2 is needed.

## Evidence chain

| Step | Exact source | What it does |
|---|---|---|
| Draw-level pricing exists temporarily | `src/Portfolio/pricing.jl:71–86,149–154` | Workspace has a per-draw score grid; price_fixture! fills it for one fixture |
| Draws reduced before Kelly | `pricing.jl:352–366` | Mean over axis3, flatten/normalise; construct R and call allocate once for this fixture; save mean grid/raw stakes/scalar shrink |
| Legacy path agrees | `src/Portfolio/book.jl:176–190` | Same mean reduction and per-match allocator call |
| Allocation objective | `src/Portfolio/implementations/allocators.jl:22–54` | Minimise −Σ_w p_w log(1+R_w a) plus a budget log-barrier, with box constraints; threshold tiny stakes afterwards |
| Same-match correlations retained | `src/Portfolio/payoff.jl:38–58` | Every column is graded against the same scoreline row; correlated Home/totals/BTTS legs are NOT independent Bernoulli bets |
| Draw-level uncertainty shrink | `src/Portfolio/implementations/shrinkage.jl:53–86` | Match-ID-seeded draw subset; re-solve each draw's per-match q; select scalar k using posterior-mean p_true, not one joint daily allocation |
| Daily grouping | `src/Portfolio/slates.jl:29–40` | Group already-built MatchBooks by calendar date; no allocation solve |
| Staking | `src/Portfolio/stake.jl:27–43,46–63,65–76` | Copy each book's precomputed a_kelly, multiply trust/shrink, apply risk, cap, filter; no allocate call |
| Slate risk is not slate Kelly | `src/Portfolio/implementations/risk.jl:40–56,73–107` | Default sequential penalty over per-match means; optional joint Monte Carlo samples independent match states from marginal grids, without shared parameter draws; solves one scalar k |
| Actual daily wealth | `src/Portfolio/simulation.jl:128–154` | Sum all realised stake×payoff in the slate, then multiply the one bankroll by 1+sum |
| Information/interface gap | `src/Portfolio/types.jl:123–134,160–164`; `src/Portfolio/interfaces.jl:34–44` | MatchBook retains neither posterior draws nor provenance; SlateContext is index/date/bankroll; allocate receives only p,R,exec |

## Precisely what is and is not missing

For one fixed match's stake vector, posterior averaging is valid integration:

`E_theta E_score|theta log(1 + R(score) a)`
`= sum_score E_theta[p(score|theta)] log(1 + R(score) a)`.

Thus mean-grid Kelly does not inherently lose single-match Bayesian expected log
utility. It is also not merely independent selection-marginal Kelly: coherent
scoreline states preserve correlations between that match's markets. Numerical
truncation/normalisation and the budget barrier/tiny-stake trimming remain caveats.

What is missing is the daily objective

`E_theta E_scores|theta log(1 + sum_matches R_match(score_match) a_match)`.

A collection of separately optimal match allocations is not generally optimal for
one simultaneous bankroll. Shared posterior draws also imply predictive dependence:
`E_theta[product_matches p_match(score|theta)]` need not equal
`product_matches E_theta[p_match(score|theta)]`. Neither MatchBook nor stake_slate
retains the shared draw index needed to represent the former.

Illustrative mathematical counterexample (not a backtest or policy change): two
zero-commission even-money bets, both winning together with probability0.6 and both
losing together with probability0.4 under a shared posterior state. Each isolated
Kelly stake is0.2. The joint objective depends on total stake t and is maximised at
t=0.2, not the isolated total0.4. A downstream common cap/risk multiplier is not a
replacement for jointly choosing the allocation vector in general.

## Phase 2 integration decision required

The existing allocate(p,R,exec) seam can solve a joint scenario matrix **if a caller
constructs it**, but the stock builder calls it before daily grouping, per fixture.
Simply selecting ScenarioKelly in BookSpec would still be a per-match solve and
must not be labelled daily-slate scenario Kelly.

Scope permits src changes only in implementations/allocators.jl. Recommended path:
an opt-in experiment loader builds aligned joint posterior/scoreline scenarios,
invokes the new allocator once per day, distributes its raw stakes into copied
MatchBooks, and delegates unchanged trust/shrink/risk/cap/filter processing to the
existing stake_slate. Original objects and production defaults stay untouched.
Draw alignment must be explicit; separate fold posteriors must not be silently
paired just because their matrices have the same column count. The user confirmed
this experimental-adapter path. Prototype verification precedes src graduation;
Phase 2 implementation/testing is now authorised, not yet claimed complete.

## Coverage carried forward — separate analysis strata

| Snapshot stratum | Staleness bound | Canonical fixtures /710 | Complete1X2 /710 | Status |
|---|---:|---:|---:|---|
| T−25m | 90 min | 611 (86.06%) | 545 (76.76%) | Reference panel |
| T−6h | 90 min | 291 (40.99%) | 182 (25.63%) | Low-coverage subset |
| T−24h primary | 90 min | 53 (7.46%) | 47 (6.62%) | Low-coverage subset |
| T−24h sensitivity | 360 min | 190 (26.76%) | 170 (23.94%) | Separately labelled low-coverage subset |

Any-archive T−6h coverage is295/710; four fixtures have only noncanonical lines.
Source: `results/phase0_historical/phase0_snapshot_overview.csv`. Never pool these
snapshot strata with T−25m, never substitute the360-minute sensitivity for the
90-minute primary. Future metrics/intervals and paired m12 differences must remain
within each labelled snapshot/bound stratum. Availability subsets are not random;
cross-snapshot differences alone are not attributable to timing.

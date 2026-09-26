# Empirical extraction and descriptive notes

**As-of:** 2026-09-23. **Scope:** Scottish league records from 2021-01-01 through the
as-of date, fetched from `betdb` in a read-only transaction. This is descriptive EDA only:
there is no MCMC, causal full-time claim, tier-strength estimate, or extrapolation of the
September live-slate narrative.

## Fixture universe, season and score contract

`data/r01_tournament_inventory.csv` verifies actual source metadata: 54 = *Scottish
Premiership*, 55 = *Championship*, 56 = *League One*, and 57 = *League Two*. The primary
fixture universe is `sofascore.events`, because the narrower `sofascore.matches` enrichment
table is materially incomplete for historical coverage. `events` joins `sofascore.seasons`
on `season_id`, so the season labels are actual source values, not calendar-inferred labels.

The raw fixture file has **4,068** finished league fixtures with normal-time scores: each
score is read from `events.raw_data.homeScore.normaltime` / `awayScore.normaltime`, with
`score_source = events_raw_normaltime` retained per row. This is appropriate to a 90-minute
league-score descriptives contract. It supersedes the preliminary 4,066 `matches`-table
extract. The membership panel uses the union of home and away event sides, rather than
home-only membership, and therefore supports sparse club-season identification.

The current source `tournaments` inventory contains no Scottish Cup, League Cup or Challenge
Cup competition IDs. The retrospective same-season membership candidate query — which labels
cup entrants by their *same-season* league membership, correctly for retrospective EDA but not
for pre-match feature availability — returns no bridge rows. It also retains tournament name,
reserve flags and normal-time score fields for whitelist review if cup records arrive. No
cross-tier match result, adjacent tier step or tier prior has therefore been invented from
within-league goal rates.

## League descriptives (not strength steps)

`results/r02_tier_goal_descriptives.csv` is the primary goals table. Its current pooled
results are Premiership 2.713 goals/match and +0.358 home-minus-away goals; Championship
2.551 and +0.207; League One 2.815 and +0.185; League Two 2.700 and +0.226. These values
combine home advantage, team composition, era and league environment. They are **not**
between-tier talent gaps.

Mean intervals use 2,000-draw club-cluster bootstrap sensitivity (home-club clusters within
tier). It does not resolve schedule dependence, away-team overlap or serial time dependence.
Home-win and draw intervals are **Wilson** match-level intervals, not exact binomial intervals
and not club-cluster uncertainty. `results/r02_tier_season_goal_descriptives.csv` preserves
season × tier values for composition inspection.

## Market, BBC and resource-proxy coverage

`r01_extract_empirical.py` selects the last **coherent** Betfair `MATCH_ODDS` vector at or
before kickoff (all three prices finite and >1) and de-vigs with
`p_k=(1/o_k)/sum_j(1/o_j)`. `results/r02_betfair_1x2_coverage_and_supremacy.csv` reports
coverage and the timing distribution. These are archived last-pre-kickoff proxies, not
verified executable closes: a vector may be minutes or hours before kickoff. Its
`mean_log_fair_p_home_over_away` is a named market supremacy statistic, **not** a Poisson
log-rate difference.

Basic BBC `shotsTotal` coverage is high, but shots are not xG. The table is retained as an
input/coverage diagnostic only. The BBC proxy-xG work is delegated to the dedicated `r06`
extractor path; no claim that the raw `bbc.match_stats` schema contains provider xG is made
here. `match_player_lineups.proposed_market_value` is near-complete in League One/Two but
about 2% in the upper tiers and lacks a scrape-time guarantee. It is therefore not a common,
point-in-time resource or operational-status proxy.

## Membership, status and FT/PT descriptives

`data/r01_club_tier_membership.csv` is the status-worker hand-off: actual season, tier,
provider club slug, first/last observed fixture date, and all league match count. Partial
20/21 and 26/27 seasons are censored and must not be read as complete 42-club panels.

The status join in `r02_empirical_eda.py` is exact on `season`, `tournament_id` and provider
club slug. It never turns Unknown or Hybrid into Part-Time. The strict panel admits only
`evidence_level == Verified`; the continuity sensitivity also admits `Inferred`, and is
labelled as such. Following the final source tranche, the strict analysis has **8 FT-vs-PT**,
**8 FT-vs-FT** and **4 PT-vs-PT** matches. FT-vs-PT, oriented to the FT team, has mean goal
difference **-0.750** and win rate **25.0%** (Wilson95% interval **7.1%–59.1%**). Inferred
sensitivity has 11 FT-vs-PT fixtures, goal difference -0.364 and win rate 36.4%. These sparse
cells are not evidence of a negative full-time effect. Full counts and exclusions for missing slug/panel,
non-accepted evidence, and Hybrid/Unknown are in
`results/r02_league_one_ft_pt_status_descriptives.csv`.

## Transition candidates

`results/r02_transition_home_fixture_candidates.csv` is a superseded home-only diagnostic,
not the final transition analysis. Use the all-side r08 panel and audited r09 saved-draw
pricing below. Complete transition history and same-time execution replay remain unavailable;
posterior probabilities are priced draw-by-draw, never at mean lambda.

## All-fixture transition panel and saved m12 provenance

`r08_transition_descriptives.py` replaces the earlier home-only candidate scaffold. It explodes
each events fixture into both club sides, orders every club's fixtures in an actual new season,
and counts **all** league fixtures before retaining matches 1–5, 6–10 and 11–20. A transition
means a different tier in consecutive *observed* club seasons; the mechanism is not verified as
promotion/relegation. First-observed source seasons are censored, and the table reports whether
the new season has a complete 20-match window.

`results/r08_transition_window_descriptives.csv` gives the full descriptive panel. For downward
tier moves, mean goal difference is +0.264 / +0.403 / +0.263 and win rate .421 / .450 / .454 in
windows 1–5 / 6–10 / 11–20. For upward moves it is -0.136 / -0.143 / -0.037 and .386 / .357 /
.358. Market probability, BBC shots and proxy-xG are reported only with their per-window
coverage denominators; they are not a learning-rate estimate and are not adjusted for opponent,
venue, season or selection into a move.

A read-only relational probe located production `m12_joint_hybrid_synergy`, run
`928dad3b-ccaf-4909-b6b7-4f1a815e1cab` (run 114, `scottish_lower_joint_player_2426`), with 710
latent rows and one portfolio artefact. The latter has 1,328 bets and reported maximum drawdown
-19.629%; its metadata identifies Betfair de-vigged TWA[-20,0], 99 slates and 622 staked
fixtures. The final r09 audit below decodes relational draws after checking saved model configuration
and the BFCL format, then prices the documented native kernel draw-wise. It does not require
loading the whole serialized fit, and is not a Julia fit-roundtrip assertion. No probability
was fabricated from mean lambda. See `results/r08_saved_m12_provenance.csv`.

## Claim boundary and decision gate

Experiment 12's 1.7214 market-on-model slope belongs to `m02_joint_gamma_poisson`, run
`97c7a3d9-a05a-4029-90cb-e34279b8c791`, across 40 folds / 710 held-out fixtures. It is not a
production m12 slope and cannot be apportioned to status or pedigree across another cohort. No
causal “share of slope” is claimed.

The next decision gate is a point-in-time join of dated status/pedigree to one immutable
model-and-market cohort, then an identical-future-fold comparison of initial-state priors,
fixed covariates and resource controls. This EDA is data-contract evidence, not a production
model-change approval.

## Final r09 saved-draw pricing audit

This addendum supersedes the earlier **no posterior prices joined** status above, but not
its same-time-market caveat. `r09_m12_transition_pricing.py` reads the immutable m12 run
`928dad3b-ccaf-4909-b6b7-4f1a815e1cab` and only portfolio
`a7c4c55b-f8d2-416e-ba85-9c7fe9bedc1a`, in a read-only transaction with 30-second statement
and 5-second lock timeouts. Credentials use `BF_EXPERIMENTS_DB_URL` or passwordless
connection parameters with libpq/pgpass; errors suppress connection details.

The saved `configs.model_config` actually declares `PoissonCountModel` and
`JointGammaPoissonObservation()`. BFCL v1 blobs must have **no observation parameters**;
nonzero flags, invalid rates, invalid lengths and trailing bytes are refused. Source audit
of `src/predictions/score_grids/types.jl:2`, `kernels.jl:53–102,507–533`, and legacy
`score_computation/poisson.jl:29–62` confirms independent Poisson goals **0:11**, summed
into 1X2 **without renormalization**, then averaged across posterior draws. Python PMF
recurrence is algebraically equivalent, not a claim of Julia bit parity or a loaded Julia
fit round-trip. Minimum retained mean score mass is 0.9999988299.

Of 1,015 candidate clubside rows, **220 rows / 196 fixtures** have saved draws; all 24
additional transitioning sides in shared fixtures survive. **186 clubside rows** also
have coherent archive prices. Summary model and market means use those *same paired
rows*, not model-only versus market-only cohorts. Downward-move model-minus-market team
win probabilities are +0.0094 / +0.0030 / -0.0105 (paired n=24/31/59); upward-move values
are +0.0772 / +0.0408 / +0.0371 (n=16/19/37), in windows 1–5 / 6–10 / 11–20.

The explicitly tagged **opponent underdog AGAINST a transitioning archive favourite**
analysis has 91 paired rows. Favourite means strictly greatest archive de-vigged 1X2
probability, ties excluded. Downward-transition opponents have mean model-minus-market
+0.0614 / +0.0616 / +0.0772 (n=11/14/31); upward-transition opponents have
-0.0216 / +0.0074 / +0.0225 (n=7/8/20). There are **65 actual opponent bets** in the exact
saved portfolio; these are not inferred from the transitioning-team bet subset. See
`r09_m12_opponent_underdog_summary.csv` for per-window counts, stakes and P&L.

Quarter-Kelly diagnostics assume 2% commission and **independent 100-unit capital per
clubside**, not a sequential or correlated portfolio bankroll. Their sums are exposure
and settlement diagnostics, not strategy returns or drawdowns. Archive-favourite labels
and archive Kelly prices are not necessarily favourites/prices at the saved portfolio's
TWA[-20,0] execution proxy. Saved stakes/P&L are retained unchanged, never re-created from
archive quotes. These small, dependent, selected cohorts do **not** identify causal model
lag, learning rate, a full-time effect, or a profitable prospective rule.

Verification: local lightweight pricing completed without MCMC; the saved config and
coverage audit is `results/r09_m12_pricing_audit.json`. Running
`python experiments/scottish_lower/13_pedigree_fulltime_and_tier_priors_eda/r09_m12_pricing_tests.py`
passed **3 tests** (decoder rejection cases, independent log-PMF kernel agreement to 1e-12,
non-renormalization/draw averaging, complete clubside retention, paired cohorts and opponent
orientation). `./scripts/todo.sh check` passed (27 tasks; AGENTS.md 19,673 bytes).

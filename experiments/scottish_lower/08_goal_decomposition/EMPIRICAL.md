# Phase 1 empirical audit — Scottish Lower goal decomposition

**Status:** regenerated on 2026-09-09 in a repeatable-read, read-only transaction against `betdb.sofascore` and `betdb.bbc.match_officials`. This report covers finished matches in tournaments 56 and 57, all available seasons. It is an EDA/data-contract report, not evidence that a decomposed model improves prediction.

## Frozen registry and model interface

The run extracted **2,019** finished matches. It wrote an immutable serialized registry plus CSV audit trail under `results/`.

- Registry format: `GoalComponentRegistry/v1`
- Snapshot SHA-256: `7571c87570578c63ac9a72c0f24f0d113b082b52b2559b5237485bca47e79951`
- Supersedes pre-referee registry `3095efa97d9d3bd9afc12695e5e6ccc9352a5cab3077779bcc464749c72408eb`; incident counts and quarantine membership are unchanged.
- Component-usable matches: **1,992**
- Quarantined matches: **27**
- Overall-score fallback: present for every finished match in `overall_home`, `overall_away`; it is not replaced by zeros when incident decomposition is unavailable.

The contractual load sequence for a model is:

```julia
include("experiments/scottish_lower/08_goal_decomposition/l08_incident_data.jl")
using .GoalDecompositionIncidentData
registry, snapshot_hash = load_registry("experiments/scottish_lower/08_goal_decomposition/results")
features = model_feature_view(registry, snapshot_hash)
```

A model must accept `registry` and `snapshot_hash` explicitly. `model_feature_view` verifies the hash and produces the following immutable columns:

| field | meaning |
|---|---|
| `flat_non_penalty_non_own_goal_home`, `flat_non_penalty_non_own_goal_away` | SofaScore `goal/regular`: **non-penalty, non-own**, not proven open play. Set pieces can be included. |
| `flat_penalty_goals_home`, `flat_penalty_goals_away` | scored penalties |
| `flat_penalty_awarded_home`, `flat_penalty_awarded_away` | observed valid attempts defined as scored penalty plus `inGamePenalty/missed`; this is not an independent award feed |
| `flat_own_goals_credited_home`, `flat_own_goals_credited_away` | recipient-side own-goal count after cumulative-score audit |
| `component_usable_mask`, `component_quarantine_reason` | mandatory component-likelihood routing; a false mask is never a zero observation |
| `overall_home`, `overall_away` | finished-score fallback retained independently of the component mask |
| `referee_name` (`referee` alias), `referee_id` | BBC name and official ID; `UNKNOWN` for missing entries |

The originally proposed `flat_open_play_*` keys are intentionally **not** emitted: SofaScore’s `regular` label does not establish open-play provenance. Models should call this channel non-penalty/non-own until a genuine set-piece classifier exists.

## Incident semantics and reconciliation

### Own-goal attribution

The prompt’s attribution claim is materially wrong for this operational extract. Cumulative-score deltas resolve 107 unambiguous own goals:

| recipient from embedded score delta | agrees with `is_home` recipient convention | incidents |
|---|---:|---:|
| home | yes | 51 |
| away | yes | 56 |
| ambiguous delta | no decision | 3 |

Thus in these records `is_home` identifies the **receiving/credited side**, not the team that committed the own goal. Three incidents have ambiguous score deltas and their three matches are quarantined. No implicit reversal is permitted.

### Quarantine

Twenty-seven matches fail score-reconciling component construction: 10 include an unclassified goal and 17 have a goal/cumulative/final-score discrepancy; three of the 27 additionally contain an ambiguous own-goal orientation. See `results/eda_quarantine.csv` and `results/eda_incident_audit.csv` for IDs and raw event classification.

The preliminary claim of 5,019 + 428 + 110 + 11 = 5,568 goals cannot be accepted as a denominator without reconciliation: it is inconsistent with the quoted implied denominator and may include duplicate/reversed/cancelled records. The frozen analysis uses final-score reconciliation, not incident-row totals. Rescinded incidents are explicitly excluded from component count construction. The raw incident audit retains all rows so VAR/cancellation, duplicates, shootouts and retakes can be inspected rather than silently normalized.

A 0–0 with no incident record is also **not** promoted to an observed all-zero component vector: it is quarantined as `zero_score_without_incident_feed_evidence`. This guards against treating absent collection as complete coverage.

## Marginal component distribution — usable rows only

These are side-match marginal summaries across 3,984 team observations (1,992 matches), not conditional Poisson tests. Team quality, home advantage, tournament and time are deliberately unadjusted; therefore their VMRs cannot establish conditional overdispersion.

| component | total | mean/side-match | variance | VMR | zero rate |
|---|---:|---:|---:|---:|---:|
| non-penalty/non-own | 4,942 | 1.2405 | 1.3702 | 1.1046 | 30.65% |
| penalty goal | 419 | 0.1052 | 0.1067 | 1.0144 | 90.09% |
| own goal credited | 107 | 0.0269 | 0.0282 | 1.0482 | 97.41% |

The mild marginal VMR elevation for non-penalty/non-own goals is compatible with unmodeled fixture-rate heterogeneity. It is not evidence for a conditional Negative Binomial component likelihood.

## Penalties: generation, concession and conversion

The registry records **555 observable penalty events**: 428 `goal/penalty` and 127 `inGamePenalty/missed`. A targeted raw-event audit is in `results/eda_attempt_quality_summary.csv` and its candidate rows (none detected) in `results/eda_attempt_quality_flags.csv`:

| check | measured result |
|---|---:|
| duplicate database incident IDs | 0 |
| duplicate provider incident IDs | 0 |
| scored penalty rows missing cumulative score | 0 |
| penalty rows missing side or minute | 0 |
| rescinded penalty rows | 0 |
| penalty rows at minute >120 | 0 |
| same match/minute/added-minute/side duplicate candidates | 0 |
| scored+missed same-stamp possible-retake candidates | 0 |

The provider JSON consistently exposes `id`, `incidentType`, `incidentClass`, `isHome`, `time`, and (for score-bearing goals) cumulative scores. It does **not** expose an independently labelled penalty-award event, a shootout marker, a retake link/marker, or an explicit VAR-cancellation marker in this extract. Thus the 555 events have no detected duplicate, same-stamp retake, rescission, or >120-minute evidence, but this is **not independent proof that scored plus missed events enumerate all awards or that no shootout/retake exists**. The model's observation contract is therefore **recorded in-game attempts**, not every initial refereeing award. No detected anomaly currently blocks that explicitly limited observation model; completeness of all awards is not claimed.

The descriptive team table in `results/eda_team_penalty_rates.csv` provides generation and concession exposure/rates plus empirical conversion. Conversion is `missing`, not zero, for a team with zero attempts. Its sparse teams must be shrunk strongly; no team-level generation, concession or conversion effect is claimed from this basic extraction runner. The dedicated statistics runner consumes the frozen registry for adjusted dispersion, generation/concession nulls, conversion heterogeneity, repeatability and pressure inference. On reconciled records there are 545 attempts and 419 conversions (76.88%). With a Beta(1,1) EDA prior the pooled posterior mean is 0.7678, with 95% interval [0.7316, 0.8022], conditional on these recorded attempts.

## Division and own-goal pressure

The two tournament totals (usable matches) are:

| tournament | matches | non-penalty/non-own goals | penalty goals | credited own goals |
|---|---:|---:|---:|---:|
| 56 | 995 | 2,533 | 194 | 57 |
| 57 | 997 | 2,409 | 225 | 50 |

`results/eda_history_only_pressure.csv` supplies expanding, **pre-fixture only** non-penalty/non-own scoring rates for both sides. It is safe as a candidate own-goal-pressure feature, but no association test or predictive claim has been made in Phase 1. Any own-goal pressure model must establish OOS value against a flat rate and avoid contemporaneous-score leakage.

## Referee audit

**Retraction:** the initial report wrongly generalized from an empty SofaScore JSON field to absence of referee information. `raw_data.referee` is null in all 2,019 matches, but **`bbc.match_officials` names 2,009/2,019 referees (99.50%)**. Its match-ID join with `role = 'referee'` has no duplicated assignment rows or conflicting official IDs per fixture. Ten fixtures receive `UNKNOWN`. SofaScore-field coverage is retained separately as `sofascore_referee_present`.

There are **58 distinct named official IDs**; **40** have at least 20 fixtures on the all-finished/raw-attempt cohort. Steven Reid has 3 attempts in 28 matches (**0.1071/match**) and Ross Hardie 19 in 40 (**0.4750/match**): a **4.43×** observed spread. Both endpoint counts survive reconciliation. This motivates partial pooling, not a causal claim.

| Cohort | Minimum matches/referee | Referees | Matches | Attempts | Δ deviance | df | Asymptotic p |
|---|---:|---:|---:|---:|---:|---:|---:|
| All finished, raw incident counts | 1 | 58 | 2,009 | 551 | 70.4721 | 57 | 0.1083 |
| All finished, raw incident counts | 20 | 40 | 1,854 | 501 | 54.8497 | 39 | 0.0475 |
| Reconciled components | 1 | 58 | 1,982 | 541 | 67.1390 | 57 | 0.1685 |
| Reconciled components | 20 | 39 | 1,812 | 486 | 50.7590 | 38 | 0.0807 |

This is an exposure-offset referee-only Poisson comparison: expected count = exposure × pooled rate; deviance = `2Σ[n log(n/e) − (n−e)]`, with zero-count contribution `2e`. G referee rates add G−1 parameters. The supplied **56.63 on 42 df is not reproduced on the stated 40-referee cohort**; 40 groups imply 39 df here. Its original cohort/covariates are needed to reconcile it.

These unadjusted associations use sparse-count chi-square approximations and are sensitive to cohort choice. Referee assignment is not randomized; team, division and season can confound rates. The model uses a hierarchical referee effect common to both sides' attempt intensities. Missing and training-unseen IDs receive zero effect at prediction, per the requested policy—a plug-in mean, not uncertainty marginalization. The historical table does not prove appointment publication times; OOS pricing assumes the eventual named assignment was known pre-kickoff.

## Artifacts

- `results/eda_match_component_registry.jls` — frozen `GoalComponentRegistry/v1`
- `results/eda_registry_snapshot.txt` — hash, timestamp and format
- `results/eda_match_component_registry.csv` — all final-score rows and masks
- `results/eda_incident_audit.csv` — all goal and non-goal incident rows for semantic investigation
- `results/eda_own_goal_orientation.csv`, `eda_own_goal_orientation_summary.csv`
- `results/eda_quarantine.csv`
- `results/eda_component_distribution.csv`
- `results/eda_attempt_quality_summary.csv`, `eda_attempt_quality_flags.csv`
- `results/eda_team_penalty_rates.csv`
- `results/eda_division_rates.csv`, `eda_history_only_pressure.csv`, `eda_own_goal_pressure_rows.csv`
- `results/eda_referee_coverage.csv`, `eda_referee_coverage_by_season.csv`
- `results/eda_referee_rates_raw.csv`, `eda_referee_rates_usable.csv`, `eda_referee_deviance.csv`

## Verification

The extraction/serialization round-trip was rerun after the BBC join. `test08_incident_contract.jl` checks synthetic attribution, missing labels, referee retention, hash tampering, actual component conservation and referee deviance degrees of freedom. These data checks do not establish MCMC convergence.

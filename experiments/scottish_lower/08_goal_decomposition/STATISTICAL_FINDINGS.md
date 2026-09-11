# Phase 1 statistical supplement — goal decomposition

**Status:** superseded pending rerun on the BBC-referee-enriched registry. The results below
were completed full-snapshot, non-MCMC exploratory EDA from the prior registry, which lacked
BBC referee identity. They remain a reproducible **pre-BBC-audit** record only: in particular,
the penalty-drawing p-value must not be interpreted as team box skill independent of referees.
The revised runner now fails loudly on that stale registry and will write referee-adjusted
results only after the incident EDA supplies its new frozen snapshot.

## Data, snapshot and reproducibility

The following pre-BBC-audit calculations consumed the then-frozen `GoalComponentRegistry/v1`
written by the incident EDA:

- snapshot SHA-256: `3095efa97d9d3bd9afc12695e5e6ccc9352a5cab3077779bcc464749c72408eb`;
- finished matches: **2,019**; component-usable matches: **1,992**; quarantined: **27**;
- usable fixture-side observations: **3,984**;
- fixed RNG seed: **20260908**;
- `1,000` refit replicates for non-penalty/non-own conditional Poisson and `2,000`
  replicates each for penalty and conversion nulls.

The runner records all replicate counts and its input hash in
`results/eda_stats_manifest.csv`. It calls `load_registry`, which verifies the serialized
registry against the recorded hash before any test runs. Re-run the supplement if the registry
is regenerated or the hash changes.

## 1. Conditional non-penalty/non-own count dispersion

The marginal VMR is not used as evidence. Instead, a Poisson GLM for each usable fixture-side
count includes receiving team, opponent, home indicator, season and division fixed effects.
The observed Pearson statistic is compared with a **parametric bootstrap that simulates from
that fitted model and refits exactly the same nuisance design in every replicate**.

| quantity | value |
|---|---:|
| observations / residual df | 3,984 / 3,915 |
| observed Pearson X² / dispersion | 4,042.04 / 1.0325 |
| refit-null X² mean (95% simulation interval) | 3,913.30 (3,745.27, 4,088.75) |
| one-sided bootstrap p / MC SE | 0.0779 / 0.0085 |

This is mildly high relative to the fitted conditional-Poisson null, but not conventionally
compelling at this Monte-Carlo precision. It does **not** establish a negative-binomial or
zero-inflated non-penalty/non-own likelihood. The result is a useful sensitivity target for
the later predictive comparison, not a promotion gate.

## 2. Penalty drawing and conceding heterogeneity

Raw team-rate ANOVA was deliberately avoided. For each role, the test uses an
**exposure-weighted, team-aggregated Pearson statistic** after a Poisson mean model containing
home/away, season, division, and a strictly earlier-history non-penalty/non-own scoring-rate
pressure covariate. Thus a team is not declared unusual merely because it played fewer matches
or because an attacking side draws more penalties through general pressure. Same-kickoff rows
read pre-block history and update only after the block; the 31 initial side rows are recorded
as cold starts.

The null draws fixture-side Poisson counts from the fitted adjusted mean, refits the same
nuisance model, then recomputes the team statistic. Drawing and conceding are two views of the
same 545 observed valid attempts, so they are not independent confirmations.

| role | team Pearson X² | refit-null mean (95% interval) | p-value / MC SE |
|---|---:|---:|---:|
| drawing | 55.61 | 28.69 (15.98, 45.39) | 0.0035 / 0.0013 |
| conceding | 36.03 | 28.55 (15.01, 45.65) | 0.1594 / 0.0082 |

Pre-BBC-audit, there was exploratory evidence of residual **penalty-drawing** heterogeneity
beyond the history-pressure adjustment, but no similarly clear evidence for conceding
heterogeneity. That result omitted the now-discovered BBC referee identity and is therefore
**not** evidence of team box skill independent of referees. It is preserved only as a baseline
for the upcoming referee-adjusted refit. `eda_stats_penalty_{drawing,conceding}_effects.csv`
gives each team's observed count, adjusted expected count and SMR for inspection; those effects
are noisy and should not be interpreted in isolation.

## 3. Penalty conversion pooling and team heterogeneity

Attempts are the frozen registry's observed scored penalties plus `inGamePenalty/missed`; they
are not an independently verified complete award process. There are **545** attempts and
**419** conversions. With the explicitly weak `Beta(1, 1)` prior, the pooled conversion
posterior is:

| estimate | value |
|---|---:|
| pooled binomial MLE | 0.7688 |
| Beta posterior mean | 0.7678 |
| 95% equal-tail Beta interval | (0.7316, 0.8022) |

The team heterogeneity statistic conditions on each team's observed attempts and, in each of
2,000 binomial null simulations, refits the pooled conversion probability. Its Pearson X² is
40.33 versus null mean 28.48, p = **0.0685** (MC SE 0.0056). This is suggestive rather than
strong evidence of team conversion heterogeneity. Of 31 teams, **2 have no attempts** and
are retained with conversion `missing`, not an invented zero rate.

## 4. Penalty-rate repeatability (exploratory)

The pre-BBC-audit chronological half split yielded 987 early and 1,005 late usable matches;
across the 21 teams present in both blocks, the unshrunk team drawing-rate correlation was
**0.0828**. It was originally only a descriptive point estimate. The revised helper now adds a
2,000-replicate exposure-aware pooled-Poisson null, its simulation interval and a one-sided
positive-correlation p-value. Interpret only those rerun outputs on the BBC-referee-enriched
registry. Regardless, repeatability is exploratory and any modelled team effect must be
partial pooled and judged OOS.

## 5. Own-goal pressure association uses history only

Own-goal receipts are assigned to the beneficiary side as established by the incident audit.
The pressure candidate is each beneficiary's expanding non-penalty/non-own scoring rate from
strictly earlier kickoffs. Cold starts are excluded rather than encoded as zero; a minimum of
five prior team fixtures leaves **3,830** side observations and **105** own goals.

A Poisson flat model with home, season and division is compared with the same model plus
standardized history-only pressure:

| quantity | value |
|---|---:|
| pressure log-rate coefficient (95% Wald interval) | 0.1264 (-0.0613, 0.3140) |
| rate ratio per pressure SD | 1.1347 |
| likelihood-ratio improvement | 1.7059 |
| flat / pressure log likelihood | -481.550 / -480.697 |

The direction is compatible with pressure, but the interval includes no effect and the
likelihood improvement is small. With only 105 outcomes this is weak-power exploratory
association evidence, not a predictive pressure-link result. Contemporaneous regular goals
were never used as pressure, and no cross-validated/OOS result is claimed.

## 6. Quarantine selection audit

The incident-driven component-usable filter is outcome-associated enough to require disclosure:

| group | matches | mean final goals | final-goals share >= 4 |
|---|---:|---:|---:|
| component usable | 1,992 | 2.7450 | 30.17% |
| quarantined | 27 | 3.3704 | 51.85% |

The quarantined sample is very small, but has a larger high-score share. Component-likelihood
EDA therefore cannot be represented as the distribution of all finished fixtures. The model
contract must retain the documented total-score fallback / component mask for quarantined rows;
it must never convert absence or reconciliation failure into all-zero component observations.

## Artifacts

- `l08_eda_statistics.jl`, `r08_eda_statistics.jl` — non-MCMC implementation and runner.
- `results/eda_stats_conditional_poisson.csv` and `_null.csv` — conditional count test.
- `results/eda_stats_penalty_drawing*.csv`, `eda_stats_penalty_conceding*.csv` — adjusted
  exposure-weighted team tests and effect summaries.
- `results/eda_stats_conversion*.csv` — pooled Beta interval and conditional binomial null.
- `results/eda_stats_repeatability.csv` — explicitly exploratory block statistic.
- `results/eda_stats_own_goal_pressure*.csv` — history-only association design and fit.
- `results/eda_stats_quarantine_outcomes.csv` — outcome-selection audit.

## Referee correction and required rerun

The prior claim that referee identity was absent was a source-selection error: top-level
SofaScore match JSON lacks it, but `bbc.match_officials` supplies a named `role = 'referee'`
record for **2,009 / 2,019 (99.5%)** finished fixtures. The revised statistical helper expects
`referee_name`, reports coverage and rates (including an explicit `UNKNOWN` level), and fits
BBC-referee fixed effects alongside home, season, division and history-only pressure in both
penalty team-heterogeneity tests. It will refuse the older registry rather than silently repeat
the invalid analysis. The incident EDA owns the join and frozen-registry regeneration; once
its new hash exists, rerun `r08_eda_statistics.jl` and replace all pre-BBC-audit penalty claims
with the resulting `eda_stats_referee_*` and adjusted penalty artifacts.

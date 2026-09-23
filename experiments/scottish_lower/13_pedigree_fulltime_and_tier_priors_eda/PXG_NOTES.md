# BBC proxy-xG extraction notes (r06)

## Scope and output contract

`r06_extract_bbc_proxy_xg.py` is a lightweight, read-only extraction for Scottish
SofaScore tournaments **54, 55, 56 and 57**, using r01's finished-event fixture panel
with non-null raw normal-time scores and match dates from **2021-01-01 through
2026-09-23**. Dates are inherited from r01, not independently reconstructed kickoff times. It writes fixture-grain observations to:

- `data/r06_bbc_proxy_xg_match.csv` — one completed fixture per `match_id`, including
  provider-absence flags and `proxy_minus_goals` on each covered side. This is the
  intended input for a later join to the full-time status fixture CSV.
- `results/r06_bbc_proxy_xg_coverage_by_tier_season.csv` and
  `results/r06_bbc_proxy_xg_coverage_by_tier.csv` — coverage and descriptive
  proxy-versus-goals summaries.
- `results/r06_bbc_proxy_xg_conversion_coefficients.csv` — the fitted global and
  zone/body/context conversion values used for this particular extraction.

The script expects `BF_DB_URL` in the environment, starts the connection with
`default_transaction_read_only=on`, issues `SET TRANSACTION READ ONLY`, has a 120-second
statement timeout, and never prints or stores the connection string. It performs no
writes to `betdb` and no MCMC. The conversion table is same-window empirical-Bayes
estimation, not point-in-time predictive inference.

## Measured r06 extraction (executed 2026-09-23)

The final corrected run wrote **4,068** r01 normal-time fixtures and **46,424**
qualifying BBC shot-event rows. This supersedes the initial 4,066/46,592 enrichment-only
extract. Its global conversion inputs were:

| quantity | measured value |
|---|---:|
| Parsed non-penalty attempts | 45,536 |
| Goals in parsed non-penalty attempts | 5,482 |
| Global open-play conversion / fallback xG | 0.12038826 |
| Penalty attempts | 659 |
| Penalty goals | 508 |
| Penalty constant | 0.77086495 |
| Empirical-Bayes pseudo-count `k` | 25.0 |

The complete per-cell counts, raw conversion and shrunk conversion are the reproducible
coefficient provenance in `results/r06_bbc_proxy_xg_conversion_coefficients.csv`; do not
substitute rounded values from this note when joining or recomputing the panel.

### Actual coverage by tier

| Tournament | r01 normal-time fixtures | Both-side commentary proxy | Coverage |
|---:|---:|---:|---:|
| 54 | 1,105 | 605 | 54.75% |
| 55 | 1,008 | 541 | 53.67% |
| 56 | 976 | 552 | 56.56% |
| 57 | 979 | 551 | 56.28% |
| **All tiers** | **4,068** | **2,249** | **55.29%** |

The low all-window percentage is not a within-season failure: BBC live-text coverage is
zero before 2023/24. In the complete 2023/24--2025/26 seasons coverage is typically
99--100%; the partial 2026/27 counts are limited by the requested 2026-09-23 cutoff.
See `results/r06_bbc_proxy_xg_coverage_by_tier_season.csv` for the season-level facts.

### Fixture-universe reconciliation with r01

The final source audit measures **4,058 shared IDs, 10 r01-only IDs, and 8
matches-only IDs**. The difference in totals is two, but the symmetric difference is
**18**, not two. `results/r06_bbc_proxy_xg_fixture_reconciliation.csv` records all IDs,
source statuses and raw normal-time scores.

All eight matches-only rows are `finished` in both sources but lack both raw event
normal-time scores, so they fail r01's score-availability rule; they are **not** excluded
for unfinished status. Their 168 shot events are excluded from the final conversion fit.
The ten r01-only rows are 2026-09-19 finished fixtures in tiers 56/57 absent from the
bounded scored-enrichment query. BBC lookup is attempted directly for every r01 ID,
without requiring `sofascore.matches`; these ten currently have zero shot events.

Both `data/r06_bbc_proxy_xg_match.csv` and
`data/r06_bbc_proxy_xg_joined_fulltime_fixture.csv` now contain exactly the same 4,068
status-join-ready fixtures. Join on `match_id`; status labels themselves belong to r03.
The reconciliation labels live in the separate audit CSV. This denominator is the r01
normal-time-score panel, not every fixture ever played; excluded score-missing fixtures
must not be silently counted as provider failures within its coverage denominator.

Run it after loading the normal local credential environment:

```bash
bash experiments/scottish_lower/13_pedigree_fulltime_and_tier_priors_eda/r06_run_bbc_proxy_xg.sh
```

Then execute the independent project-kernel parity check using the installed project
environment (this is an ordinary parser/table calculation, not MCMC):

```bash
julia --project=/home/james/bet_project/BayesianFootball \
  experiments/scottish_lower/13_pedigree_fulltime_and_tier_priors_eda/r06_verify_bbc_proxy_xg_kernel.jl
```

The final run **passed** row-level parser equality on all **46,424 events**, recorded in
`results/r06_bbc_proxy_xg_parser_parity.csv`, against Python's persisted shot descriptors.
It also writes `data/r06_bbc_proxy_xg_match_julia_kernel.csv` and
`results/r06_bbc_proxy_xg_kernel_parity.csv`. Unique fixture IDs, missingness and all
4,068 availability masks agree; maximum home/away errors are respectively
`4.9970553e-7` and `4.9958318e-7`, below the unchanged `0.5e-6` six-decimal
serialization bound. The verifier executes the project's `parse_shot`, `fit_shot_xg`
and `predict_xg`, with its own aggregation; it does not execute the full DataStore path.
Julia SQL uses a raw string for regex `$` anchors and a real `BEGIN READ ONLY` transaction.

## Proxy definition — project kernel, stricter coverage measurement

This adapts the **commentary rung** of
`Features.pxg_match_observations` in `src/features/pxg.jl`, using the parser and model
semantics in `src/features/plus_minus/shot_parser.jl`:

1. Fetch BBC `live_text` rows whose event type is one of `goal`, `attempt_missed`,
   `attempt_saved`, `attempt_blocked`, `post`, `penalty_missed`, or `penalty_saved`.
2. Attribute each shot to home or away by comparing BBC slugs after only a trailing `-fc`
   normalization. Rows still unresolved are counted but not assigned to either side.
3. Parse each non-penalty attempt into BBC-text `zone × body_part × context` cells.
   The `a free kick` wording is mapped to `outside_box × direct_free_kick`, exactly as in
   the source parser, because that phrasing leaks the realised outcome.
4. Pool all in-window commentary attempts, including unresolved-side rows, for conversion
   estimation (1,144 unresolved-side events are excluded only from side aggregation). For each non-penalty cell `c`,
   with `g_c` goals in `n_c` attempts and global open-play conversion `p`, assign:

   ```text
   proxy_xg_per_attempt(c) = (g_c + 25 p) / (n_c + 25)
   ```

   This is the project's `ShotXGModel` empirical-Bayes cell table with `k = 25`.
   Unknown/unparsed non-penalty attempts receive the global `p`. Penalties receive one
   constant equal to their observed in-window conversion (or the source default 0.76 if
   none are present). The exact fitted values and cell counts are persisted in the
   conversion-coefficients CSV.
5. Sum the values by fixture and home/away side. `proxy_xg_available_* = 1` requires at
   least one **side-resolved** BBC shot event; no commentary is encoded as blank proxy
   values and availability zero, never as 0.0 xG.

This is intentionally a direct measurement, not `PxGFeature`'s later rolling,
pre-match form covariate. It is **not exact full-route equivalence**: the project commentary
rung emits zero for a side with no attributed shots if the other side has any. r06 instead
marks that side unavailable/blank. There are **nine one-side-only fixtures**, and they are
excluded from both-side descriptive summaries. This conservative coverage convention
cannot distinguish genuine zero-shot sides from missing commentary. It also does not use the BBC match-page shot-count fallback or
the goals fallback: the purpose here is to quantify actual BBC commentary coverage and
proxy differences, without filling provider absence with a less direct measurement.

## Interpretation guardrails

- `proxy_minus_goals` is descriptive residual arithmetic, **not** an estimate of luck,
  finishing skill, causal shot quality, team strength, or a betting signal.
- Conversion cells are fit on the same requested extraction window. They support an EDA
  coverage file, not out-of-sample forecasting or causal inference.
- A fixture can be complete in SofaScore and still have absent BBC commentary/enrichment.
  `proxy_xg_available_both = 0` identifies that provider limitation explicitly; it must
  not be read as a scoreless match or zero expected goals.
- The repository's reported `0.817` team-level correlation refers to earlier validation
  against SofaScore xG. This r06 run does not recreate that validation because the task
  asks for all-tier fixture coverage and status-join-ready differences, not an xG-provider
  comparison.

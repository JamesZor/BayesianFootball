# 2026-09-19 Live Slate — Evidence and Provenance Audit

**Status:** PARTIALLY VERIFIED. The read-only, bounded extraction contains 27 fully settled
paper orders and now identifies the fixtures. It confirms the operational probability,
staking and settlement evidence below. It cannot establish immutable m12 saved-run provenance:
the m12 `paper_slates.model_run_id` is `NULL`.

## Scope and evidence lineage

This audit reads `betdb.paper_runbook` and, where an opaque ledger UUID is present, the
`mcmc_experiments` saved-run catalogue. It neither samples nor re-prices a model. The saved
order-time `paper_orders.p_model` is the operational probability; reconstructing a current
model could select a different posterior, fold, data snapshot or revision.

`data/slate_2026-09-19.csv` is the bounded ledger panel and
`results/slate_2026-09-19.csv` is its deterministic arithmetic/provenance panel. All orders
were quoted at **13:34 UTC** on 2026-09-19, one minute before their respective slate records
were created at **13:35 UTC**, for **14:00 UTC** kick-offs.

| Slate / stated run name | Account | Immutable run UUID | Fold | Orders | Requested slate risk | Bankroll | Terminal status |
|---|---|---|---:|---:|---:|---:|---|
| `m12_joint_hybrid_synergy` | `live_scottish_m12_500` | **NULL** | 43 | **13** | £43.53 | £500.00 | `BATCH_SETTLED` |
| `m05_joint_production_wealth_grw` | `live_scottish_grw_500` | `f870dbb7-9df0-4dae-a84a-cf570cf8113e` | 43 | **14** | £41.23 | £500.00 | `BATCH_SETTLED` |

The m05 UUID resolves to completed run ID 126 in experiment
`scottish_lower_multiscale_grw_2426`, configuration hash
`a626db07dba85104d7cd959caed6bda6657dff2bfcc87f61f6d0f3e2644bd8df`, created 2026-09-11.
That provenance belongs to m05 only. The m12 `run_name` is descriptive ledger text, not an
immutable address, so no m12 commit, configuration hash, exact posterior or saved-run UUID may
be claimed from this evidence.

## Transition-fixture evidence

The refreshed join establishes that the actual fixture orientation was **Cove Rangers (home) vs
Ross County (away)**, not the prompt's “Ross County vs Cove Rangers”. Hamilton Academical was
home to Queen of the South. The m12 orders relevant to those fixtures were:

| Fixture | Selection | Odds | \(p_\mathrm{model}\) | \(p_\mathrm{market}\) at 13:34 | Edge | Requested risk | Filled risk | Settlement / net P&L |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Cove Rangers vs Ross County | Cove home | 7.20 | 37.3981% | 13.5734% | +23.8247 pp | £11.06 | £7.00 | LOSE / −£7.00 |
| Cove Rangers vs Ross County | draw | 4.90 | 26.4232% | 19.9446% | +6.4786 pp | £2.96 | £2.96 | LOSE / −£2.96 |
| Hamilton Academical vs Queen of the South | Queen away | 9.00 | 25.3971% | 10.9750% | +14.4221 pp | £4.52 | £4.52 | LOSE / −£4.52 |
| Hamilton Academical vs Queen of the South | draw | 5.50 | 24.5207% | 17.9590% | +6.5617 pp | £1.81 | £1.81 | LOSE / −£1.81 |

The two underdog selections alone requested **£15.58** (35.79% of £43.53 slate risk) and had
**£11.52** filled risk. The prompt's **£20.35** is exactly the four-leg requested-risk sum
\(£11.06 + £2.96 + £4.52 + £1.81\), or **46.75%** of total requested slate risk. It therefore
combines the two underdogs **and the two draws**, rather than describing two underdogs only.
The same four legs had **£16.29** filled risk and settled for **−£16.29** net P&L.

The prompt's stated 23.8 pp and 14.4 pp edge magnitudes are confirmed, at stored six-decimal
precision, for Cove home and Queen away respectively. Its stated 29.7% Cove probability,
25.2% Queen probability, 5.70 Cove odds and 7.40 Queen odds are not the stored values. The
actual values are 37.3981% / 7.20 and 25.3971% / 9.00. The prompt's stated 67.0%/17.5% and
71.0%/13.5% “closing” prices are also not verified: no `clv_audit` field is populated for any
of the 27 orders. `p_market` is an order-time market probability, not demonstrated close.

## m12 settlement, bankroll impact and units

Python/decimal-equivalent summation of the 13 m12 `net_pnl` values is **−£5.09**. Relative to
the reported one-slate £500 bankroll, the settled balance is **£494.91** and the one-slate
change is **−1.018%**. This is a one-slate return versus the reported opening bankroll; calling
it *maximum drawdown* would be unsupported because the extraction does not establish the
account's preceding historical peak or an intra-slate equity path.

All 27 orders have `state = SETTLED`, at least one fill, and a settlement row; 8 are wins and
19 are losses. m12 settlement timestamps are 17:07:53.893 UTC. For the four transition legs,
`gross_return` and `commission` are each £0.00. Commission is a **currency settlement** field,
not an edge unit; for example, other winning m12 orders carry positive commission.

The stored relation is

\[
\mathrm{edge}=p_\mathrm{model}-p_\mathrm{market}.
\]

It is reported above in **percentage points**, not return percentage, commission or Kelly
fraction. All 27 rows reconcile within one stored final decimal unit (absolute difference
≤ \(10^{-6}\)); residual ±\(10^{-6}\) differences are the independent rounding of three
six-decimal database values. Requested `risk`, `risk_filled`, `venue_stake`, realised P&L and
share of bankroll are separate measures and are not interchangeable.

## Extractor safeguards and remaining limitation

`r05_slate_audit.py` now joins teams through `sofascore.events.match_id`, uses parameterised
SQL, enters server-enforced read-only mode, and has **5-second connect** and **5-second
server-side statement** timeouts. It redacts driver exceptions and never prints connection
strings. The current output is sufficient for the operational audit above.

The remaining provenance defect is m12's null UUID. No further lookup by run name/fold should
be treated as exact identity without an independently retained mapping; the ledger only
supports the limited conclusion that its m12 slate was labelled
`m12_joint_hybrid_synergy`, fold 43.

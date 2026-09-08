# Empirical archive findings — 2026-09-05 Scottish live slate

## Scope and provenance

This is a **read-only** extraction and historical archive replay. It does not fit a model,
write to either paper ledger, or establish that an archive-derived fill would have happened on
Betfair. Replaceable artefacts and their SHA-256 hashes are in
[`results/2026-09-05_archive/manifest.txt`](results/2026-09-05_archive/manifest.txt). The
manifest pins extraction time, Git commit, dirty-tree status hash, source-file hashes, output
hashes and the actual runtime (**Julia 1.12.1**), since archive data and output filenames are
mutable/replaceable.

- **Explicit live slate:** `0fd23606-80bf-4fcd-be44-66d708562fe6`, account `live_scottish`.
- **Slate facts:** as-of 2026-09-05 13:35 UTC, `m12_joint_hybrid_synergy`, £2,400 bankroll,
  17 legs, frozen parent-risk sum **£427.07**. `paper_slates.total_risk` displays £427.08 from
  decimal rounding.
- **Fixture universe:** 10 tournament-56/57 Scottish fixtures at 14:00 UTC. The frozen orders
  span 9 fixtures.
- **Frozen-order source:** `paper_runbook.paper_orders`, joined to fills/settlements only for
  literal ledger reconciliation. `frozen_orders.csv` retains IDs, probability, reference odds,
  risk/stake, quote/submission/fill clocks and settlement provenance.
- **Archive mapping:** all 17 frozen `bf_market_id` values are blank. Market identity is therefore
  **inferred**, not persisted venue-market provenance: SofaScore → `betfair.match_meta` event
  crosswalk + market type + exact runner contract. O/U aliases are only
  `under_25 ↔ Under 2.5 Goals` and `over_25 ↔ Over 2.5 Goals`; no fuzzy or opposite-runner
  fallback is allowed.
- **Archive units:** `src/MatchDay/implementations/book.jl:66-76` shows that exchange-feed
  prices, displayed volumes and `market_matched` are all integer values divided by **10,000**.
  Earlier local `/100` capacity figures are invalid and are not findings.
- **Consistent extraction:** the runner opens `REPEATABLE READ READ ONLY`, sets UTC and a
  60-second local statement timeout, then closes without commit.

## Actual ledger reconciliation — literal baseline

The actual ledger contains 17 orders and 17 recorded `touch_only` children: **£427.07** planned
risk, **£144.77** recorded fill risk (**33.90%**), £441.54 planned venue stake, £146.53 recorded
venue stake and settled net P&L **+£19.14**. This is the literal production baseline; it is not
re-priced, re-timed or passed through the prototype hurdle.

All recorded fill stamps are **14:50:21.555 UTC**, after 14:00 kickoff. Frozen reference quotes
are 16 at 13:35 UTC and one at 13:32 UTC. The post-kickoff fill stamp does **not** establish that
the reference quote was post-kickoff; it does mean the ledger cannot validate an archive-T−25
fill or a confirmed-lineup-at-T−25 scenario. Production settlement uses original odds and
per-order commission (T008 semantics), retained separately from prototype price-specific,
market-net accounting.

## Archive coverage and filtration

All **17/17** frozen parents map under the inferred identity contract. At stored-ts T−25,
**16/17** have a quote aged no more than 90 seconds; one O/U lay has no eligible quote and is not
simulated. For all 16 eligible orders, reference odds equal the stored-ts best available-to-position
price (**0/16 mismatches**; see `reference_vs_archive_t25.csv`). This equality is an
implementation-shortfall-anchor comparison, not receipt-time executable-price proof.

The archive is minute-downsampled and may use `ON CONFLICT DO UPDATE`. Until bucket and
receipt-time semantics are verified, `ts <= decision` is causal only in **stored-ts** terms.
No result claims exact executable-at-T−25 proof. A 60-second stored-ts lag sensitivity remains a
required follow-up.

## Fixed-policy archive replay

All simulated policies freeze parent identity, `p_model`, reference odds, parent risk, bankroll,
2% commission and 2% hurdle. TouchOnly and sweep both use the same 1% adverse arithmetic-VWAP
tolerance. The fixed parent reference is an **implementation-shortfall** anchor, not
contemporaneous tick slippage. Guarded TouchOnly is distinct from the literal ledger baseline.

| archive policy | simulated risk | planned-risk fill | parents with fills |
|---|---:|---:|---:|
| guarded TouchOnly at stored-ts T−25 | £130.71 | 30.61% | 13 / 17 |
| guarded three-level sweep at stored-ts T−25 | £199.66 | 46.75% | 13 / 17 |
| staged T−25…T−5, no replenishment | £175.69 | 41.14% | 13 / 17 |
| staged T−25…T−5, refreshed display | £210.16 | 49.21% | 13 / 17 |

The sweep uses reservation-qualified, 1%-VWAP-bounded depth and actual-price lay liability.
The staged policy is a 21-step cumulative catch-up schedule, not selected best-minute execution.
Replay is chronological by decision minute and then deterministic lexicographic UUID order.
No-replenishment depth is shared across all parents by `(market, runner, side, absolute price)`.
The refreshed-display case still shares depletion for parents consuming the **same archive
stamp**; only a later stamp may display fresh depth. Neither is a live fill guarantee or a
probabilistic P&L bound.

`execution_diagnostics.csv` records per-order/tick status and per-level sequential reasons;
`execution_reason_summary.csv` aggregates them. The labels depth, reservation, slip, target,
Kelly and policy-depth identify the sequential binding constraint, including partial fills.
They are **not additive decompositions of unfilled pounds**. On this card evaluated staged levels
reported target/reservation/depth/slip, with no Kelly binding event; unavailable/stale/reused
stored quotes and `not_scheduled` single-shot ticks remain separate from depth.

The sweep is £68.95 higher than guarded TouchOnly on this card, but this is not capacity proof:
the scalar per-parent Kelly/hurdle kernel and minute archive constrain it, and the original
joint portfolio certificate is not preserved. `simulated_realized_market_pnl.csv` applies known
outcomes only after simulated fills freeze, sums child cashflows by market, and takes 2% once
from positive market gross. It is a settlement illustration, not comparable production P&L.

## Liquidity and WOM — descriptive only

Two samples remain separate.

1. **Target-order sample:** 16 valid stored-ts T−25/T−5 parents. Three-level liquidity change
   versus best available-to-position price change has correlation **+0.187**; T−25 WOM versus
   that future change is **−0.187**.
2. **Whole Scottish slate:** 60 paired runners from MATCH_ODDS, OVER_UNDER_25 and
   BOTH_TEAMS_TO_SCORE across 10 fixture clusters. WOM versus future best-back change is
   **−0.117**; liquidity change versus it is **+0.004**. Three-level displayed-depth totals are
   £6,730.73 at T−25 and £6,445.69 at T−5 (ratio **0.958**); median runner depth ratio is
   **1.000**.

Predeclared whole-runner WOM bins (`whole_slate_wom_bins.csv`) use negative best-back change as
shortening: low `<0.35`: 19 runners/8 fixtures, mean +0.0068, 26.3% shortening; neutral
`0.35–0.65`: 15/8, +0.0453, 26.7%; high `>0.65`: 26/10, −0.0627, 34.6%. These are not p-values,
not independent-minute observations, and not a threshold selection exercise. WOM/volume do not
change the replay policy. No market-matched velocity was reported because it must be deduplicated
by market/timestamp rather than summed across runners.

## Blockers and next gates

1. Verify collector bucket/receipt timestamp semantics and run a 60-second stored-ts lag
   sensitivity before asserting executable-at-time causality.
2. Obtain high-frequency traded/fill or live shadow-order data before estimating queue, maker
   fill probability or adverse selection; three resting levels are not traded VWAP.
3. Replace scalar parent Kelly with the score-tensor joint-slate conditional-risk solve before any
   production execution use.
4. Reconcile T008 production settlement semantics separately; do not pool original-odds,
   per-order-commission ledger P&L with prototype market-net P&L.
5. Replicate on independently dated slates before selecting WOM thresholds or claiming a staged
   execution benefit.

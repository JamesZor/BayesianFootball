# T008 — Multi-level fills retain touch liability and settlement prices

- **Status:** open
- **Severity:** high for multi-level replay/CLV; no evidence here of real venue orders
- **Raised:** 2026-09-07, during the market-microstructure prototype
- **Scope:** `src/MatchDay/ledger/fills.jl`, `settle.jl`, associated tests

## Evidence and root cause

Static audit of the current code:

1. `fills.jl:95–116`, `_fill(::LadderSweep, ...)`, checks `(touch-p)/touch`
   for BOTH sides. Worsening back prices decrease; worsening lay prices increase.
   Thus the lay slippage check never limits adverse deeper levels. The docstring
   says VWAP but the implementation checks individual prices.
2. The same method sets `risk_filled = taken / leverage` using the original order
   leverage. For a lay at actual decimal price `p`, liability is `taken*(p-1)`.
   Moving down the lay ladder changes the liability per unit venue stake.
3. `fills.jl:143–158`, `fill_vwap`, returns the harmonic mean
   `sum(size)/sum(size/price)`. For the archive's **backer-stake** size denomination,
   the equivalent fixed-runner payoff price is the arithmetic mean
   `sum(size*price)/sum(size)`. Harmonic averaging describes a different weighting.
4. `settle.jl:60–76`, `settle_order`, uses the ORIGINAL `o.effective_odds` for
   every winning fill, ignoring `Fill.price`. This affects back sweeps too.
   It charges commission per winning position, not on net winnings per market;
   market-netting must be specified explicitly before claiming exchange-equivalence.
5. `settle.jl:157–160`, `clv_for_order`, consumes harmonic `fill_vwap` and converts
   lays through `lay_to_back(d)=d/(d-1)`. Harmonic ≤ arithmetic understates a back's
   achieved odds (conservative), but the decreasing lay conversion turns the same
   error into overstated effective odds: it **flatters synthetic-lay entry CLV**.

## Minimal deterministic reproduction

Use `BookLevels` with back `[3.0, 2.98]`, sizes `[10,10]`, and lay `[3.0,3.1]`,
sizes `[10,10]`. Compare the existing `simulate_fill` and `settle_order` to the
following independently derived expectations (no database required):

- Back £10 at 3.0 plus £10 at 2.98: total risk £20, gross win £39.80,
  equivalent decimal odds 2.99, net standalone win at 2% commission £39.004.
  Original-odds settlement instead reports gross £40.
- Lay £10 at 3.0 plus £10 at 3.1: actual liability £20 + £21 = £41,
  gross favorable-event win £20. Frozen leverage 1/(3−1) records only £40 risk.
- `LadderSweep(max_slippage=0.01)` accepts the deeper lay at 3.1 despite a
  3.33% adverse price move. Explicitly decide per-level vs aggregate VWAP behavior;
  under aggregate VWAP a partial second level may fit but not the entire level.
- With £40 parent liability, level 2 must be capped to £20/2.1 backer stake,
  not £10. Risk and venue size caps are different constraints.

Executed against the loaded production `BayesianFootball.MatchDay` on Julia 1.12.1,
2026-09-07, using exactly the above `BookLevels` and
`simulate_fill(LadderSweep(max_slippage=0.01), book, side, 20.0, leverage, at)`:

```text
lay:  levels = 2, reported_risk = 40.0, cashflow_liability = 41.0
back: reported_vwap = 2.9899665551839467, cashflow_vwap = 2.9899999999999998
```

No database or ledger write was involved. The settlement discrepancy above is a
separate static cashflow derivation, not a claim that this command called settlement.

The independent prototype tests in
`current_development/market_microstructure_execution/test_microstructure_sweeper.jl`
verify price-specific liability and arithmetic payoff equivalence, but do NOT fix
or certify these production methods.

## Blast radius

Multi-level paper/replay capacity, P&L, reserved balance and CLV. Single-touch fills
at unchanged order odds are generally unaffected by price variation, but commission
netting remains a separate convention. Historical persisted rows must not be silently
rewritten; model versions and reconstruction provenance must remain identifiable.

## Proposed fix and trade-offs

1. Pass side and a liability budget into the fill implementation; cap each child at
   remaining risk using its own price. Retain exact price/venue stake per fill.
2. Implement explicit side-aware reservation/slippage rules. Version the fill model,
   rather than changing the meaning of `ladder_sweep_v1` historical records.
3. Settle from fill cashflows, not original instrument odds; settle commission at
   market scope where that is the intended exchange contract.
4. Separate payoff-equivalent arithmetic odds from any probability-space diagnostic.
   Rename/version outputs so CLV charts do not silently change interpretation.

## Acceptance criteria

- Above back/lay examples asserted, including losses and partial last-level fills.
- Arithmetic payoff equivalence asserted for both sides. For lays specifically,
  `d_equiv = 1 + sum(size*(price-1))/sum(size) = sum(size*price)/sum(size)`.
  Pin the resulting back/lay CLV direction against a known closing quote.
- Corrected sweep emits a new persisted fill-model name (e.g. `ladder_sweep_v2`);
  it never changes the interpretation of existing `ladder_sweep_v1` rows.
- Adverse lay movement refused or clipped under the documented limit convention.
- Total actual liability never exceeds reserved parent risk.
- Multi-fill settlement equals sum of exact child cashflows before market commission.
- Opposing positions in one market get the documented net-commission treatment.
- Existing TouchOnly baseline, ledger idempotency, reservation/release and replay
  isolation tests pass; no paper_runbook writes from tests.
- Legacy rows remain reconstructible under the old version and differences are reported.

## Scope guard

Do not fix this inline during the prototype, change Portfolio mathematics, retrain
models, migrate historical balances, or deploy staged live execution as part of T008.
Child-order lifecycle and conditional-slate risk certification are separate work.

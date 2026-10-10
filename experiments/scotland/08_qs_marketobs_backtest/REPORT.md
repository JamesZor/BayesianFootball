# Phase 0 — BLOCKED

The read-only experiment-database session assertion failed on mcmc-beast.
Execution stopped before loading m12, reproducing experiment 07, or building any snapshot.
No wealth, bet counts, coverage, CLV, bootstrap intervals, or new model comparisons were computed.
The production stack inventory is complete; Phase 0 is not complete.
TODO 043 remains BLOCKED; the owned beast pane is closed.

## Verification evidence

| Attempt | Source | Result | Evidence |
|---|---|---|---|
| 1 | `8c4fccec` | Startup `MethodError`, before any data/gate; module-binding world-age warning | [`logs/phase0_inventory_attempt1.log`](logs/phase0_inventory_attempt1.log) |
| 2 | `9249fbcd` | Read-only assertion failed at `l00_phase0_inventory.jl:25`, called from `load_context:37` | [`logs/phase0_inventory_attempt2.log`](logs/phase0_inventory_attempt2.log) |

The runner was launched in persistent tmux pane `%451`, `pi_qsbt_phase0`, with Julia
1.12.4, 16 threads, core pinning and BLAS=1. `PGOPTIONS` was supplied both before Julia
startup and in the runner. The assertion nevertheless did not admit the connection.
The failing statement is `SHOW default_transaction_read_only` followed by an assertion
that its value is `on`. The raw value was not logged; no credential-bearing exception
values were printed. **No `load_fit` call was reached.** No write SQL was issued.
The cached datastore loaded without querying betdb. This failure is not a reproduction
mismatch and gives no evidence for or against any model or staking policy.

## Exact canonical default inventory

| Setting | Canonical `r07_serve_console.jl` construction |
|---|---|
| Markets | 1X2, BTTS, O/U 0.5, 1.5, 2.5, 3.5 |
| Price / allocator | DeArb / KellyLogUtility |
| Shrink | BakerMcHale: 128 draws, grid 0:0.02:1, seed 20260805 |
| Trust | Home / Under 2.5: 0.35; Draw / Away: 0.25; all others: 0 |
| Risk / cap | SlateDrawdown(23.0) / FixedCap(0.25) |
| Group / filter | DailySlate / KeepAll |
| Commission | PerBetCommission(0.02) |
| Per-match budget / per-selection maximum | 0.99 / 0.50 |
| Minimum selection fraction / barrier | 0.0001 / 1e-6 |
| Complete markets | Required |
| Calibration | None supplied to price_slate |

Source references and historical distinctions are in [DECISIONS_LOG.md](DECISIONS_LOG.md).
The historical experiment 07 target is **2.5152 terminal unit wealth / 1127 bets / 99 slates**
(m12/raw/B0_canonical/prod), taken from committed suite 07 evidence, **not newly reproduced**.
It used FractionalKelly(0.30), minimum 0.001 and 11 priced directions, unlike the exact
canonical default. Newer Option B runners also differ in trust and λ. This discrepancy
must not be hidden by calling all three stacks the same production policy.

## Snapshot and model results

| Snapshot | Phase 0 reproduction / coverage |
|---|---|
| T−25m | NOT_RUN — database safety gate failed before fit loading |
| T−6h | NOT_RUN — stop rule |
| T−24h | NOT_RUN — stop rule |

All later phases and the five substantive report questions remain unanswered.
No Phase 3 cell is marked NOT_AVAILABLE: that classification has not been assessed.

## Interpretation limits and follow-ups

1. Manager must authorise a read-only connection setup that actually passes the existing
   assertion; do not remove it or proceed on an unverified writable session.
2. Confirm whether canonical defaults, Option B, or the historical experiment 07 stack is
   intended for Phase 0/3. Exact canonical parity is currently untested, not failed.
3. On authorised resume, load m12 by the frozen UUID, require 710 held-out IDs, reproduce
   the fixed historical control and exact canonical stack, then build snapshot coverage
   only if the required gate passes. Do not refit or tune.
4. Last-traded archive prices are not guaranteed executable quotes or fills. This caveat
   applies even if subsequent numerical backtests succeed.
5. Production policy selection reused these two seasons; future model comparisons carry
   that in-sample policy-selection risk. No uncertainty estimates are available yet.

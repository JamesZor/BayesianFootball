# Market microstructure and staged execution

Research prototype for the Scottish Lower Saturday 2026-09-05 slate. **Not connected
to live execution.** No sampling, ledger writes or production Portfolio changes.

## Read first

- [REPORT.md](REPORT.md): reservation-price and Kelly/VWAP derivations, staged-window
  design, maker/WOM limitations and production integration gates.
- [EMPIRICAL.md](EMPIRICAL.md): measured ledger reconciliation, archived-book policy
  comparison, coverage and liquidity evidence.
- [T008](../../docs/tickets/T008-multilevel-fill-price-accounting.md): existing
  multi-level fill/settlement accounting defects discovered, not fixed here.

The prompt's £451.09 target minus £165.65 fill is £285.44, not £261.43. Do not pool
live and hindsight-confirmed-lineup P&L. A single day's outcome does not prove alpha.

## Files and usage

| File | Purpose |
|---|---|
| `l01_microstructure_sweeper.jl` | Pure `MicrostructureExecution` module: side-aware reservation, price-specific lay liability, arithmetic book VWAP, scalar Kelly cap, policies |
| `l02_archive_research.jl` | Read-only archive/ledger extraction and replay/report helpers |
| `r01_microstructure_sweeper.jl` | Numbered research workflow with explicit configuration and provenance |
| `test_microstructure_sweeper.jl` | Pure regression tests plus independent grid-utility and zero-allocation checks |
| `results/` | Extracted evidence and comparison artifacts; see empirical report for provenance |

```sh
# Pure tests: no database, BayesianFootball load or inference required
julia --project -t 8 current_development/market_microstructure_execution/test_microstructure_sweeper.jl

# Read-only empirical run: BF_DB_URL is required (never print it)
julia --project -t 8 current_development/market_microstructure_execution/r01_microstructure_sweeper.jl
```

Policies are namespaced to avoid confusion with existing MatchDay fill models:
`MicrostructureExecution.TouchOnly()`, `MultiLevelSweep(max_slip=0.01)`, and
`StagedTWAP(start_minutes=25,end_minutes=5,max_slip=0.01)`. The default edge hurdle
is 2% net expected return per unit liability. A staged policy consumes 21 cumulative
tranches, catches up missed volume, and cancels unfillable residual at T−5.

## Recommendation

Use the prototype for **capacity sensitivity**, not a production drawdown guarantee.
The per-position Kelly kernel does not condition the entire correlated slate.
Repeated resting depth is not proven replenishment; passive queue fills cannot be
identified from one-minute snapshots; WOM 0.35/0.65 are hypotheses, not learned
execution thresholds. Book-fill VWAP is not traded VWAP.

Before live adoption: fix/version T008 accounting, implement durable child-order and
cancel/replace reconciliation, certify the full slate after partial fills, preserve
live/replay isolation, enforce venue tick/minimum-size rules, and run multi-day
shadow validation. SQL, reporting and depletion dictionaries stay outside the
preallocated Portfolio/execution kernels.
